//! HIP GQH decode and fused dequant-matvec.
//!
//! Decode is bit-exact against `model_store::gqh`. The fused matvec keeps the
//! same scale product order (`e4m3(d) * tensor_scale`, then `* (ratio/15)`).

#![cfg_attr(not(supersonic_backend_hip), allow(unused_variables, unreachable_code))]

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

/// Per-tensor GQH header looked up by the device pointer of the packed wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisteredHeader {
    pub tensor_scale: f32,
    pub grid_code: u8,
}

fn header_map() -> &'static Mutex<HashMap<usize, RegisteredHeader>> {
    static MAP: OnceLock<Mutex<HashMap<usize, RegisteredHeader>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the GQH header for a packed device buffer. Keyed by `GpuBuffer::as_ptr()`.
#[derive(Clone, Copy, Debug)]
pub struct RegisteredMix {
    pub qtype: i32,
    pub mode: i32,
    pub lut: [f32; 16],
}

fn mix_map() -> &'static Mutex<HashMap<usize, RegisteredMix>> {
    static MAP: OnceLock<Mutex<HashMap<usize, RegisteredMix>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_mix(ptr: *const c_void, qtype: i32, mode: i32, lut: [f32; 16]) {
    if ptr.is_null() {
        return;
    }
    mix_map()
        .lock()
        .expect("mix registry")
        .insert(ptr as usize, RegisteredMix { qtype, mode, lut });
}

pub fn lookup_mix(ptr: *const c_void) -> Option<RegisteredMix> {
    if ptr.is_null() {
        return None;
    }
    mix_map()
        .lock()
        .expect("mix registry")
        .get(&(ptr as usize))
        .copied()
}

pub fn register_header(ptr: *const c_void, tensor_scale: f32, grid_code: u8) {
    if ptr.is_null() {
        return;
    }
    header_map()
        .lock()
        .expect("gqh header registry")
        .insert(
            ptr as usize,
            RegisteredHeader {
                tensor_scale,
                grid_code,
            },
        );
}

pub fn lookup_header(ptr: *const c_void) -> Option<RegisteredHeader> {
    if ptr.is_null() {
        return None;
    }
    header_map()
        .lock()
        .expect("gqh header registry")
        .get(&(ptr as usize))
        .copied()
}

pub const RUNG_GQH3: i32 = 0;
pub const RUNG_GQH2_H: i32 = 1;
pub const RUNG_GQH2_C: i32 = 2;

pub const SUPERBLOCK: usize = 256;

#[cfg(supersonic_backend_hip)]
unsafe extern "C" {
    fn supersonic_gqh_hip_decode(
        device_ordinal: c_int,
        rung: c_int,
        wire: *const c_void,
        tensor_scale: f32,
        grid_code: c_int,
        dst: *mut c_void,
        rows: c_int,
        cols: c_int,
        dst_is_bf16: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn supersonic_gqh_hip_gemm_flush();

    fn supersonic_gqh_hip_dequant_gemm_bf16(
        device_ordinal: c_int,
        rung: c_int,
        wire: *const c_void,
        tensor_scale: f32,
        grid_code: c_int,
        lhs: *const c_void,
        out: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    fn supersonic_gqh_hip_mix_matvec_stream(
        device_ordinal: c_int,
        qtype: c_int,
        wire: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        in_dim: c_int,
        out_dim: c_int,
        ncols: c_int,
        acc: c_int,
        mode: c_int,
        lut: *const f32,
        stream: *mut c_void,
    ) -> c_int;

    fn supersonic_gqh_hip_enable_tight_decode();
    fn supersonic_gqh_hip_ensure_tight(
        device_ordinal: c_int,
        rung: c_int,
        wire: *mut c_void,
        in_dim: c_int,
        out_dim: c_int,
    ) -> c_int;

    fn supersonic_gqh_hip_matvec(
        device_ordinal: c_int,
        rung: c_int,
        wire: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        in_dim: c_int,
        out_dim: c_int,
        ncols: c_int,
        x_col_stride: i64,
        y_col_stride: i64,
        tensor_scale: f32,
        grid_code: c_int,
    ) -> c_int;
}

fn backend_error(backend: Backend, what: &str, status: c_int) -> GpuError {
    GpuError::backend(backend, format!("{what} failed with status {status}"))
}

pub fn rung_from_ggml_type(ty: u32) -> Option<i32> {
    match ty {
        108 => Some(RUNG_GQH3),
        109 => Some(RUNG_GQH2_H),
        110 => Some(RUNG_GQH2_C),
        _ => None,
    }
}

pub fn rung_from_flm_codec(semantic_id: u16) -> Option<i32> {
    match semantic_id {
        13 => Some(RUNG_GQH3),
        14 => Some(RUNG_GQH2_H),
        15 => Some(RUNG_GQH2_C),
        _ => None,
    }
}

/// Dequantize packed GQH superblocks to f32 or bf16. `wire` is the post-header stream.
pub fn decode(
    ordinal: usize,
    rung: i32,
    wire: &GpuBuffer,
    tensor_scale: f32,
    grid_code: u8,
    dst: &mut GpuBuffer,
    rows: usize,
    cols: usize,
) -> Result<(), GpuError> {
    if cols == 0 || cols % SUPERBLOCK != 0 {
        return Err(GpuError::InvalidArg(format!(
            "gqh decode cols {cols} is not a positive multiple of {SUPERBLOCK}"
        )));
    }
    let dst_is_bf16 = match dst.dtype() {
        ScalarType::F32 => 0,
        ScalarType::BF16 => 1,
        other => {
            return Err(GpuError::InvalidArg(format!(
                "gqh decode dst must be f32/bf16, got {other:?}"
            )));
        }
    };
    if dst.elem_count() < rows * cols {
        return Err(GpuError::InvalidArg(format!(
            "gqh decode dst has {} elems, need {}",
            dst.elem_count(),
            rows * cols
        )));
    }
    let backend = dst.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                supersonic_gqh_hip_decode(
                    ordinal as c_int,
                    rung as c_int,
                    wire.as_ptr(),
                    tensor_scale,
                    grid_code as c_int,
                    dst.as_mut_ptr(),
                    rows as c_int,
                    cols as c_int,
                    dst_is_bf16,
                    std::ptr::null_mut(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
            }
        }
        other => {
            return Err(GpuError::Unsupported(format!(
                "gqh decode is HIP-only, got {other:?}"
            )));
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "gqh decode", status));
    }
    Ok(())
}

/// Decode GQH to BF16 and GEMM `out[m,n] = lhs[m,k] @ W[n,k]^T`.
/// Overlaps this dequant with the previous call's GEMM via two streams.
pub fn dequant_gemm_bf16(
    ordinal: usize,
    rung: i32,
    wire: &GpuBuffer,
    tensor_scale: f32,
    grid_code: u8,
    lhs: &GpuBuffer,
    out: &mut GpuBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GpuError> {
    if lhs.dtype() != ScalarType::BF16 || out.dtype() != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "gqh dequant_gemm requires bf16 lhs/out, got {:?}/{:?}",
            lhs.dtype(),
            out.dtype()
        )));
    }
    if m == 0 || n == 0 || k == 0 || k % SUPERBLOCK != 0 {
        return Err(GpuError::InvalidArg(format!(
            "gqh dequant_gemm m={m} n={n} k={k} invalid"
        )));
    }
    if lhs.elem_count() < m * k || out.elem_count() < m * n {
        return Err(GpuError::InvalidArg(
            "gqh dequant_gemm lhs/out too small".into(),
        ));
    }
    let backend = out.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                supersonic_gqh_hip_dequant_gemm_bf16(
                    ordinal as c_int,
                    rung as c_int,
                    wire.as_ptr(),
                    tensor_scale,
                    grid_code as c_int,
                    lhs.as_ptr(),
                    out.as_mut_ptr(),
                    m as c_int,
                    n as c_int,
                    k as c_int,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
            }
        }
        other => {
            return Err(GpuError::Unsupported(format!(
                "gqh dequant_gemm is HIP-only, got {other:?}"
            )));
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "gqh dequant_gemm", status));
    }
    Ok(())
}

pub fn gemm_flush() {
    #[cfg(supersonic_backend_hip)]
    unsafe {
        supersonic_gqh_hip_gemm_flush();
    }
}

pub fn enable_tight_decode() {
    #[cfg(supersonic_backend_hip)]
    unsafe {
        supersonic_gqh_hip_enable_tight_decode();
    }
}

pub fn ensure_tight(
    ordinal: usize,
    rung: i32,
    wire: *mut c_void,
    in_dim: i32,
    out_dim: i32,
) -> Result<(), GpuError> {
    #[cfg(not(supersonic_backend_hip))]
    {
        let _ = (ordinal, rung, wire, in_dim, out_dim);
        return Ok(());
    }
    #[cfg(supersonic_backend_hip)]
    {
        let status = unsafe {
            supersonic_gqh_hip_ensure_tight(
                ordinal as c_int,
                rung,
                wire,
                in_dim,
                out_dim,
            )
        };
        if status != 0 {
            return Err(backend_error(
                Backend::Hip,
                "gqh ensure_tight",
                status,
            ));
        }
        Ok(())
    }
}

/// Fused `y[ncols, out] = W[out, in] @ x[ncols, in]` with inline GQH decode.
pub fn matvec(
    ordinal: usize,
    rung: i32,
    wire: &GpuBuffer,
    x: &GpuBuffer,
    y: &mut GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    ncols: usize,
    x_col_stride: usize,
    y_col_stride: usize,
    tensor_scale: f32,
    grid_code: u8,
) -> Result<(), GpuError> {
    if in_dim == 0 || in_dim % SUPERBLOCK != 0 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matvec in_dim {in_dim} is not a positive multiple of {SUPERBLOCK}"
        )));
    }
    if x.dtype() != ScalarType::F32 || y.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matvec x/y must be f32, got {:?}/{:?}",
            x.dtype(),
            y.dtype()
        )));
    }
    let backend = y.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                supersonic_gqh_hip_matvec(
                    ordinal as c_int,
                    rung as c_int,
                    wire.as_ptr(),
                    x.as_ptr(),
                    y.as_mut_ptr(),
                    in_dim as c_int,
                    out_dim as c_int,
                    ncols as c_int,
                    x_col_stride as i64,
                    y_col_stride as i64,
                    tensor_scale,
                    grid_code as c_int,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
            }
        }
        other => {
            return Err(GpuError::Unsupported(format!(
                "gqh matvec is HIP-only, got {other:?}"
            )));
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "gqh matvec", status));
    }
    Ok(())
}

/// Fused mix (105/106) matvec. `lut` is 16 f32 levels (unused slots ignored).
pub fn mix_matvec(
    ordinal: usize,
    qtype: i32,
    wire: &GpuBuffer,
    x: &GpuBuffer,
    y: &mut GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    ncols: usize,
    acc: bool,
    mode: i32,
    lut: &[f32; 16],
) -> Result<(), GpuError> {
    if in_dim == 0 || in_dim % 32 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "mix matvec in_dim {in_dim} is not a multiple of 32"
        )));
    }
    if x.dtype() != ScalarType::F32 || y.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "mix matvec x/y must be f32, got {:?}/{:?}",
            x.dtype(),
            y.dtype()
        )));
    }
    let backend = y.backend();
    let status = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                supersonic_gqh_hip_mix_matvec_stream(
                    ordinal as c_int,
                    qtype as c_int,
                    wire.as_ptr(),
                    x.as_ptr(),
                    y.as_mut_ptr(),
                    in_dim as c_int,
                    out_dim as c_int,
                    ncols as c_int,
                    if acc { 1 } else { 0 },
                    mode as c_int,
                    lut.as_ptr(),
                    std::ptr::null_mut(),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg("HIP backend not compiled".into()));
            }
        }
        other => {
            return Err(GpuError::Unsupported(format!(
                "mix matvec is HIP-only, got {other:?}"
            )));
        }
    };
    if status != 0 {
        return Err(backend_error(backend, "mix matvec", status));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_hal::set_backend;
    use model_store::gqh::{decode_wire, GqhRung};
    use std::path::PathBuf;

    fn vector_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../model-store/tests/gqh-vectors")
    }

    fn load_case(rung: &str, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let stem = vector_dir().join(format!("{rung}_{rows}x{cols}"));
        let wire = std::fs::read(stem.with_extension("wire.bin")).expect("wire.bin");
        let raw = std::fs::read(stem.with_extension("decode.f32")).expect("decode.f32");
        assert_eq!(raw.len(), rows * cols * 4);
        let mut reference = vec![0.0f32; rows * cols];
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            reference[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        (wire, reference)
    }

    fn split_wire(rung: GqhRung, wire: &[u8]) -> (f32, u8, &[u8]) {
        if rung.has_header() {
            let scale = f32::from_le_bytes(wire[..4].try_into().unwrap());
            (scale, wire[4], &wire[5..])
        } else {
            (0.0, 0, wire)
        }
    }

    fn device_packed(rung: GqhRung, rows: usize, cols: usize, tight: &[u8]) -> Vec<u8> {
        model_store::gqh::planarize(rung, rows, cols, tight).expect("planarize device wire")
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if g.to_bits() != w.to_bits() {
                panic!(
                    "{label} [{i}] got 0x{:08x} ({g}) want 0x{:08x} ({w})",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_f32(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download f32");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn kernel_order_matvec(weights: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        const WARP: usize = 32;
        const PER_LANE: usize = 8;
        let nsb = cols / SUPERBLOCK;
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut lane_acc = [0.0f32; WARP];
            for sb in 0..nsb {
                for lane in 0..WARP {
                    let j0 = lane * PER_LANE;
                    for t in 0..PER_LANE {
                        let j = sb * SUPERBLOCK + j0 + t;
                        lane_acc[lane] += weights[r * cols + j] * x[j];
                    }
                }
            }
            let mut accs = lane_acc;
            let mut off = WARP / 2;
            while off > 0 {
                for i in 0..off {
                    accs[i] += accs[i + off];
                }
                off >>= 1;
            }
            y[r] = accs[0];
        }
        y
    }

    fn require_hip() -> Option<usize> {
        set_backend(Backend::Hip);
        match crate::query_gpu_info(0) {
            Ok(_) => Some(0),
            Err(err) => {
                eprintln!("skip gqh hip test: {err}");
                None
            }
        }
    }

    #[test]
    fn register_and_lookup_header_by_pointer() {
        let ptr = 0x1000 as *const c_void;
        register_header(ptr, 1.25, 3);
        let got = lookup_header(ptr).expect("registered");
        assert_eq!(got.tensor_scale, 1.25);
        assert_eq!(got.grid_code, 3);
        assert!(lookup_header(std::ptr::null()).is_none());
    }

    #[test]
    fn maps_gguf_and_flm_ids() {
        assert_eq!(rung_from_ggml_type(108), Some(RUNG_GQH3));
        assert_eq!(rung_from_ggml_type(109), Some(RUNG_GQH2_H));
        assert_eq!(rung_from_ggml_type(110), Some(RUNG_GQH2_C));
        assert_eq!(rung_from_flm_codec(13), Some(RUNG_GQH3));
        assert_eq!(rung_from_flm_codec(14), Some(RUNG_GQH2_H));
        assert_eq!(rung_from_flm_codec(15), Some(RUNG_GQH2_C));
        assert!(rung_from_ggml_type(107).is_none());
    }

    #[test]
    fn rejects_non_superblock_shapes() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let wire = GpuBuffer::zeros(ordinal, ScalarType::U8, &[1]).expect("wire");
        let x = GpuBuffer::zeros(ordinal, ScalarType::F32, &[255]).expect("x");
        let mut dst = GpuBuffer::zeros(ordinal, ScalarType::F32, &[255]).expect("dst");
        assert!(decode(ordinal, RUNG_GQH3, &wire, 1.0, 0, &mut dst, 1, 255).is_err());
        assert!(matvec(
            ordinal, RUNG_GQH3, &wire, &x, &mut dst, 255, 1, 1, 255, 1, 1.0, 0
        )
        .is_err());
    }

    #[test]
    fn hip_decode_matches_official_vectors_bit_exactly() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let cases = [
            ("gqh3", GqhRung::Gqh3, RUNG_GQH3, 1, 256),
            ("gqh3", GqhRung::Gqh3, RUNG_GQH3, 4, 512),
            ("gqh2_h", GqhRung::Gqh2H, RUNG_GQH2_H, 1, 256),
            ("gqh2_h", GqhRung::Gqh2H, RUNG_GQH2_H, 4, 512),
            ("gqh2_c", GqhRung::Gqh2C, RUNG_GQH2_C, 1, 256),
            ("gqh2_c", GqhRung::Gqh2C, RUNG_GQH2_C, 4, 512),
        ];
        for (name, cpu_rung, rung, rows, cols) in cases {
            let (wire, reference) = load_case(name, rows, cols);
            let cpu = decode_wire(cpu_rung, rows, cols, &wire).expect("cpu decode");
            assert_bits_eq(&cpu, &reference, &format!("{name} cpu {rows}x{cols}"));
            let (scale, grid_code, packed) = split_wire(cpu_rung, &wire);
            let packed = device_packed(cpu_rung, rows, cols, packed);
            let packed_buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[packed.len()],
                &packed,
            )
            .expect("upload packed");
            let mut dst =
                GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows * cols]).expect("alloc dst");
            decode(
                ordinal,
                rung,
                &packed_buf,
                scale,
                grid_code,
                &mut dst,
                rows,
                cols,
            )
            .unwrap_or_else(|e| panic!("{name} decode: {e}"));
            let got = read_f32(&dst);
            assert_bits_eq(&got, &reference, &format!("{name} hip {rows}x{cols}"));
        }
    }

    fn qwen38_gguf_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("SUPERSONIC_GQH_GGUF") {
            let path = PathBuf::from(path);
            return path.is_file().then_some(path);
        }
        let default = PathBuf::from("/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf");
        default.is_file().then_some(default)
    }

    #[test]
    fn hip_qwen38_gguf_gqh_linears_match_cpu() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let Some(path) = qwen38_gguf_path() else {
            eprintln!("skip: Qwen3.8 GQH GGUF not present");
            return;
        };
        let file = model_store::gguf::GgufFile::open(&path).expect("open qwen38 gguf");
        assert_eq!(file.kv("general.architecture"), Some("qwen35"));
        assert_eq!(file.gqh_header_count(), 350);

        let cases = [
            ("blk.0.ffn_gate.weight", GqhRung::Gqh3, RUNG_GQH3, 4),
            ("blk.0.ffn_down.weight", GqhRung::Gqh2H, RUNG_GQH2_H, 4),
            ("output.weight", GqhRung::Gqh2H, RUNG_GQH2_H, 2),
        ];
        for (name, cpu_rung, rung, rows) in cases {
            let tensor = file.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(tensor.tensor_type, cpu_rung.ggml_type());
            let cols = tensor.dims[0];
            let packed_all = file.tensor_bytes(name).expect("tensor bytes");
            let header = if cpu_rung.has_header() {
                Some(
                    file.gqh_header(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("{name} missing GQH header")),
                )
            } else {
                None
            };
            let row_bytes = packed_all.len() / tensor.dims[1];
            let packed = &packed_all[..row_bytes * rows];
            let mut cpu = vec![0.0f32; rows * cols];
            for r in 0..rows {
                model_store::gqh::decode_row(
                    cpu_rung,
                    &packed[r * row_bytes..(r + 1) * row_bytes],
                    cols,
                    header.clone(),
                    &mut cpu[r * cols..(r + 1) * cols],
                )
                .unwrap_or_else(|e| panic!("{name} cpu decode: {e}"));
            }
            let (scale, grid_code) = header
                .as_ref()
                .map(|h| (h.tensor_scale, h.grid_code))
                .unwrap_or((0.0, 0));
            let device = device_packed(cpu_rung, rows, cols, packed);
            let packed_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                    .expect("upload packed");
            let mut dst =
                GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows * cols]).expect("alloc dst");
            decode(
                ordinal,
                rung,
                &packed_buf,
                scale,
                grid_code,
                &mut dst,
                rows,
                cols,
            )
            .unwrap_or_else(|e| panic!("{name} hip decode: {e}"));
            assert_bits_eq(&read_f32(&dst), &cpu, &format!("{name} decode"));

            let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
            let want = kernel_order_matvec(&cpu, &x, rows, cols);
            let x_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &f32_bytes(&x))
                    .expect("upload x");
            let mut y_buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("alloc y");
            matvec(
                ordinal,
                rung,
                &packed_buf,
                &x_buf,
                &mut y_buf,
                cols,
                rows,
                1,
                cols,
                rows,
                scale,
                grid_code,
            )
            .unwrap_or_else(|e| panic!("{name} hip matvec: {e}"));
            assert_bits_eq(&read_f32(&y_buf), &want, &format!("{name} matvec"));
        }
    }

    #[test]
    fn hip_qwen38_gguf_every_gqh_tensor_one_row() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let Some(path) = qwen38_gguf_path() else {
            eprintln!("skip: Qwen3.8 GQH GGUF not present");
            return;
        };
        let file = model_store::gguf::GgufFile::open(&path).expect("open qwen38 gguf");
        let names: Vec<String> = file.tensor_names().map(str::to_owned).collect();
        let mut checked = 0usize;
        for name in &names {
            let Some(rung) = model_store::gqh::GqhRung::from_ggml_type(
                file.tensor(name).expect("tensor").tensor_type,
            ) else {
                continue;
            };
            if name.starts_with("blk.64.") {
                continue;
            }
            let tensor = file.tensor(name).unwrap();
            let cols = tensor.dims[0];
            let packed_all = file.tensor_bytes(name).expect("bytes");
            let row_bytes = packed_all.len() / tensor.dims[1];
            let packed = &packed_all[..row_bytes];
            let header = if rung.has_header() {
                Some(
                    file.gqh_header(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("{name} missing header")),
                )
            } else {
                None
            };
            let mut cpu = vec![0.0f32; cols];
            model_store::gqh::decode_row(rung, packed, cols, header.clone(), &mut cpu)
                .unwrap_or_else(|e| panic!("{name} cpu: {e}"));
            let (scale, grid_code) = header
                .as_ref()
                .map(|h| (h.tensor_scale, h.grid_code))
                .unwrap_or((0.0, 0));
            let device = device_packed(rung, 1, cols, packed);
            let packed_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                    .expect("upload");
            let mut dst = GpuBuffer::zeros(ordinal, ScalarType::F32, &[cols]).expect("dst");
            decode(
                ordinal,
                match rung {
                    GqhRung::Gqh3 => RUNG_GQH3,
                    GqhRung::Gqh2H => RUNG_GQH2_H,
                    GqhRung::Gqh2C => RUNG_GQH2_C,
                },
                &packed_buf,
                scale,
                grid_code,
                &mut dst,
                1,
                cols,
            )
            .unwrap_or_else(|e| panic!("{name} hip: {e}"));
            assert_bits_eq(&read_f32(&dst), &cpu, name);
            checked += 1;
        }
        assert_eq!(checked, 344, "64-layer GQH tensors plus lm_head");
    }

    #[test]
    fn hip_matvec_matches_kernel_order_cpu_dot() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let cases = [
            ("gqh3", GqhRung::Gqh3, RUNG_GQH3, 1, 256),
            ("gqh3", GqhRung::Gqh3, RUNG_GQH3, 4, 512),
            ("gqh2_h", GqhRung::Gqh2H, RUNG_GQH2_H, 1, 256),
            ("gqh2_h", GqhRung::Gqh2H, RUNG_GQH2_H, 4, 512),
            ("gqh2_c", GqhRung::Gqh2C, RUNG_GQH2_C, 1, 256),
            ("gqh2_c", GqhRung::Gqh2C, RUNG_GQH2_C, 4, 512),
        ];
        for (name, cpu_rung, rung, rows, cols) in cases {
            let (wire, weights) = load_case(name, rows, cols);
            let (scale, grid_code, packed) = split_wire(cpu_rung, &wire);
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
                .collect();
            let want = kernel_order_matvec(&weights, &x, rows, cols);

            let device = device_packed(cpu_rung, rows, cols, packed);
            let packed_buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[device.len()],
                &device,
            )
            .expect("upload packed");
            let x_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[cols], &f32_bytes(&x))
                    .expect("upload x");
            let mut y_buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[rows]).expect("alloc y");
            matvec(
                ordinal,
                rung,
                &packed_buf,
                &x_buf,
                &mut y_buf,
                cols,
                rows,
                1,
                cols,
                rows,
                scale,
                grid_code,
            )
            .unwrap_or_else(|e| panic!("{name} matvec: {e}"));
            let got = read_f32(&y_buf);
            assert_bits_eq(&got, &want, &format!("{name} matvec {rows}x{cols}"));
        }
    }
}
