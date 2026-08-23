//! HIP GQH decode and fused dequant-matvec.
//!
//! Decode is bit-exact against `model_store::gqh`. The fused matvec keeps the
//! same scale product order (`e4m3(d) * tensor_scale`, then `* (ratio/15)`).

use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

/// Per-tensor GQH header looked up by the device pointer of the packed wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisteredHeader {
    pub tensor_scale: f32,
    pub grid_code: u8,
}

fn header_map() -> &'static Mutex<HashMap<(usize, usize), RegisteredHeader>> {
    static MAP: OnceLock<Mutex<HashMap<(usize, usize), RegisteredHeader>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the GQH header for a packed device buffer. Keyed by `GpuBuffer::as_ptr()`.
#[derive(Clone, Copy, Debug)]
pub struct RegisteredMix {
    pub qtype: i32,
    pub mode: i32,
    pub lut: [f32; 16],
}

fn mix_map() -> &'static Mutex<HashMap<(usize, usize), RegisteredMix>> {
    static MAP: OnceLock<Mutex<HashMap<(usize, usize), RegisteredMix>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_mix(ordinal: usize, ptr: *const c_void, qtype: i32, mode: i32, lut: [f32; 16]) {
    if ptr.is_null() {
        return;
    }
    mix_map()
        .lock()
        .expect("mix registry")
        .insert((ordinal, ptr as usize), RegisteredMix { qtype, mode, lut });
}

pub fn lookup_mix(ordinal: usize, ptr: *const c_void) -> Option<RegisteredMix> {
    if ptr.is_null() {
        return None;
    }
    mix_map()
        .lock()
        .expect("mix registry")
        .get(&(ordinal, ptr as usize))
        .copied()
}

pub fn register_header(ordinal: usize, ptr: *const c_void, tensor_scale: f32, grid_code: u8) {
    if ptr.is_null() {
        return;
    }
    header_map().lock().expect("gqh header registry").insert(
        (ordinal, ptr as usize),
        RegisteredHeader {
            tensor_scale,
            grid_code,
        },
    );
}

pub fn lookup_header(ordinal: usize, ptr: *const c_void) -> Option<RegisteredHeader> {
    if ptr.is_null() {
        return None;
    }
    header_map()
        .lock()
        .expect("gqh header registry")
        .get(&(ordinal, ptr as usize))
        .copied()
}

/// Remove all process-global metadata associated with a packed device buffer.
///
/// The C++ bridge caches layout conversion state by raw pointer, so this must
/// run before the owning `GpuBuffer` is freed. It is intentionally idempotent:
/// cleanup can run during both normal destruction and error unwinding.
pub fn unregister(ordinal: usize, ptr: *const c_void) {
    if ptr.is_null() {
        return;
    }
    header_map()
        .lock()
        .expect("gqh header registry")
        .remove(&(ordinal, ptr as usize));
    mix_map()
        .lock()
        .expect("mix registry")
        .remove(&(ordinal, ptr as usize));
    unsafe {
        supersonic_gqh_hip_unregister_wire(ordinal as c_int, ptr);
    }
}

fn unregister_many(ordinal: usize, ptrs: &[usize]) {
    if ptrs.is_empty() {
        return;
    }
    {
        let mut headers = header_map().lock().expect("gqh header registry");
        for &ptr in ptrs {
            headers.remove(&(ordinal, ptr));
        }
    }
    {
        let mut mixes = mix_map().lock().expect("mix registry");
        for &ptr in ptrs {
            mixes.remove(&(ordinal, ptr));
        }
    }
    let wires: Vec<*const c_void> = ptrs.iter().map(|&ptr| ptr as *const c_void).collect();
    unsafe {
        supersonic_gqh_hip_unregister_wires(ordinal as c_int, wires.as_ptr(), wires.len());
    }
}

/// Owns one registration for the lifetime of its packed GPU allocation.
/// `Qwen38Weights` keeps these guards so partial loads and normal model drops
/// both invalidate bridge metadata before HIP frees the allocation.
pub struct Registration {
    ordinal: usize,
    ptr: usize,
}

impl Registration {
    pub fn new(ordinal: usize, ptr: *const c_void) -> Self {
        Self {
            ordinal,
            ptr: ptr as usize,
        }
    }

    pub fn unregister(&mut self) {
        if self.ptr == 0 {
            return;
        }
        unregister(self.ordinal, self.ptr as *const c_void);
        self.ptr = 0;
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.unregister();
    }
}

#[derive(Clone, Copy)]
struct PendingRegistration {
    ordinal: usize,
    ptr: usize,
    header: Option<RegisteredHeader>,
    mix: Option<RegisteredMix>,
}

/// Defers publication of process-global metadata until the model owns every
/// packed buffer. Pending entries are inert, so a failed GGUF load cannot
/// publish a pointer whose buffer is subsequently dropped. Once committed,
/// the guards retain normal before-buffer-drop cleanup semantics.
pub struct RegistrationBatch {
    pending: Vec<PendingRegistration>,
    committed: Vec<Registration>,
}

impl RegistrationBatch {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            committed: Vec::new(),
        }
    }

    pub fn stage_header(
        &mut self,
        ordinal: usize,
        ptr: *const c_void,
        tensor_scale: f32,
        grid_code: u8,
    ) {
        self.stage(
            ordinal,
            ptr,
            Some(RegisteredHeader {
                tensor_scale,
                grid_code,
            }),
            None,
        );
    }

    pub fn stage_mix(
        &mut self,
        ordinal: usize,
        ptr: *const c_void,
        qtype: i32,
        mode: i32,
        lut: [f32; 16],
    ) {
        self.stage(ordinal, ptr, None, Some(RegisteredMix { qtype, mode, lut }));
    }

    pub fn stage(
        &mut self,
        ordinal: usize,
        ptr: *const c_void,
        header: Option<RegisteredHeader>,
        mix: Option<RegisteredMix>,
    ) {
        if ptr.is_null() || (header.is_none() && mix.is_none()) {
            return;
        }
        if let Some(existing) = self
            .pending
            .iter_mut()
            .find(|entry| entry.ordinal == ordinal && entry.ptr == ptr as usize)
        {
            existing.header = existing.header.or(header);
            existing.mix = existing.mix.or(mix);
            return;
        }
        self.pending.push(PendingRegistration {
            ordinal,
            ptr: ptr as usize,
            header,
            mix,
        });
    }

    pub fn commit(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        self.committed.reserve(pending.len());
        for entry in pending {
            let ptr = entry.ptr as *const c_void;
            if let Some(header) = entry.header {
                register_header(entry.ordinal, ptr, header.tensor_scale, header.grid_code);
            }
            if let Some(mix) = entry.mix {
                register_mix(entry.ordinal, ptr, mix.qtype, mix.mode, mix.lut);
            }
            self.committed.push(Registration::new(entry.ordinal, ptr));
        }
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for registration in &mut self.committed {
            if registration.ptr != 0 {
                grouped
                    .entry(registration.ordinal)
                    .or_default()
                    .push(registration.ptr);
                registration.ptr = 0;
            }
        }
        for (ordinal, ptrs) in grouped {
            unregister_many(ordinal, &ptrs);
        }
        self.committed.clear();
    }
}

impl Default for RegistrationBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RegistrationBatch {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Invalidate bridge-side decode caches owned by one engine instance.
pub fn invalidate_decode_cache(ordinal: usize, layers: *const c_void, int4: *const c_void) {
    if layers.is_null() && int4.is_null() {
        return;
    }
    unsafe {
        supersonic_qwen35_4b_hip_invalidate_decode_cache(ordinal as c_int, layers, int4);
    }
}

pub const RUNG_GQH3: i32 = 0;
pub const RUNG_GQH2_H: i32 = 1;
pub const RUNG_GQH2_C: i32 = 2;
pub const RUNG_GQH4: i32 = 3;

pub const SUPERBLOCK: usize = 256;

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

    fn supersonic_gqh_hip_gemm_flush(device_ordinal: c_int) -> c_int;

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

    fn supersonic_gqh_hip_unregister_wire(device_ordinal: c_int, wire: *const c_void);
    fn supersonic_gqh_hip_unregister_wires(
        device_ordinal: c_int,
        wires: *const *const c_void,
        count: usize,
    );

    fn supersonic_qwen35_4b_hip_invalidate_decode_cache(
        device_ordinal: c_int,
        layers: *const c_void,
        int4: *const c_void,
    );
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
        111 => Some(RUNG_GQH4),
        _ => None,
    }
}

pub fn rung_from_flm_codec(semantic_id: u16) -> Option<i32> {
    match semantic_id {
        13 => Some(RUNG_GQH3),
        14 => Some(RUNG_GQH2_H),
        15 => Some(RUNG_GQH2_C),
        16 => Some(RUNG_GQH4),
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
    let backend = Backend::Hip;
    let status = unsafe {
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
    let backend = Backend::Hip;
    let status = unsafe {
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
    };
    if status != 0 {
        return Err(backend_error(backend, "gqh dequant_gemm", status));
    }
    Ok(())
}

pub fn gemm_flush(ordinal: usize) -> Result<(), GpuError> {
    let status = unsafe { supersonic_gqh_hip_gemm_flush(ordinal as c_int) };
    if status != 0 {
        return Err(backend_error(Backend::Hip, "gqh gemm flush", status));
    }
    Ok(())
}

pub fn enable_tight_decode() {
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
    let status =
        unsafe { supersonic_gqh_hip_ensure_tight(ordinal as c_int, rung, wire, in_dim, out_dim) };
    if status != 0 {
        return Err(backend_error(Backend::Hip, "gqh ensure_tight", status));
    }
    Ok(())
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
    let backend = Backend::Hip;
    let status = unsafe {
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
    let backend = Backend::Hip;
    let status = unsafe {
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
    };
    if status != 0 {
        return Err(backend_error(backend, "mix matvec", status));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        register_header(0, ptr, 1.25, 3);
        let got = lookup_header(0, ptr).expect("registered");
        assert_eq!(got.tensor_scale, 1.25);
        assert_eq!(got.grid_code, 3);
        assert!(lookup_header(0, std::ptr::null()).is_none());
        unregister(0, ptr);
    }

    #[test]
    fn unregister_is_idempotent_for_header_and_mix_metadata() {
        let ptr = 0x2000 as *const c_void;
        register_header(0, ptr, 2.5, 7);
        register_mix(0, ptr, 105, 3, [0.0; 16]);
        assert!(lookup_header(0, ptr).is_some());
        assert!(lookup_mix(0, ptr).is_some());

        unregister(0, ptr);
        assert!(lookup_header(0, ptr).is_none());
        assert!(lookup_mix(0, ptr).is_none());

        // A buffer can be observed by more than one owner-cleanup path while
        // unwinding a failed load. Repeated cleanup must remain harmless.
        unregister(0, ptr);
    }

    #[test]
    fn registration_guard_cleans_up_on_drop() {
        let ptr = 0x3000 as *const c_void;
        register_header(0, ptr, 3.75, 2);
        {
            let _registration = Registration::new(0, ptr);
            assert!(lookup_header(0, ptr).is_some());
        }
        assert!(lookup_header(0, ptr).is_none());
    }

    #[test]
    fn registration_batch_failure_does_not_publish_uncommitted_metadata() {
        let ptr = 0x4000 as *const c_void;
        {
            let mut batch = RegistrationBatch::new();
            batch.stage_header(0, ptr, 4.5, 5);
            batch.stage_mix(0, ptr, 105, 3, [0.0; 16]);
            assert!(lookup_header(0, ptr).is_none());
            assert!(lookup_mix(0, ptr).is_none());
        }
        assert!(lookup_header(0, ptr).is_none());
        assert!(lookup_mix(0, ptr).is_none());
    }

    #[test]
    fn registration_batch_commit_is_raii_cleanup() {
        let ptr = 0x5000 as *const c_void;
        {
            let mut batch = RegistrationBatch::new();
            batch.stage_header(0, ptr, 5.5, 6);
            batch.commit();
            assert!(lookup_header(0, ptr).is_some());
            batch.commit();
        }
        assert!(lookup_header(0, ptr).is_none());
    }

    #[test]
    fn registration_batch_reuses_pointer_after_owner_drop() {
        let ptr = 0x6000 as *const c_void;
        {
            let mut first = RegistrationBatch::new();
            first.stage_header(0, ptr, 6.5, 1);
            first.commit();
            assert_eq!(lookup_header(0, ptr).unwrap().tensor_scale, 6.5);
        }
        assert!(lookup_header(0, ptr).is_none());

        // A later model may receive the same allocator address. Its metadata
        // must be independent of the dropped model's registration.
        {
            let mut second = RegistrationBatch::new();
            second.stage_header(0, ptr, 7.5, 2);
            second.commit();
            assert_eq!(lookup_header(0, ptr).unwrap().tensor_scale, 7.5);
        }
        assert!(lookup_header(0, ptr).is_none());
    }

    #[test]
    fn registration_registry_isolated_by_device_ordinal() {
        let ptr = 0x7000 as *const c_void;
        register_header(0, ptr, 8.5, 3);
        register_header(1, ptr, 9.5, 4);
        assert_eq!(lookup_header(0, ptr).unwrap().grid_code, 3);
        assert_eq!(lookup_header(1, ptr).unwrap().grid_code, 4);
        unregister(0, ptr);
        assert!(lookup_header(0, ptr).is_none());
        assert!(lookup_header(1, ptr).is_some());
        unregister(1, ptr);
        assert!(lookup_header(1, ptr).is_none());
    }

    #[test]
    fn registration_batches_keep_same_pointer_lifetimes_isolated_by_ordinal() {
        let ptr = 0x7_1000 as *const c_void;
        {
            let mut first = RegistrationBatch::new();
            let mut second = RegistrationBatch::new();
            first.stage_header(0, ptr, 10.5, 1);
            second.stage_header(1, ptr, 11.5, 2);
            first.commit();
            second.commit();
            assert_eq!(lookup_header(0, ptr).unwrap().tensor_scale, 10.5);
            assert_eq!(lookup_header(1, ptr).unwrap().tensor_scale, 11.5);
            first.clear();
            assert!(lookup_header(0, ptr).is_none());
            assert_eq!(lookup_header(1, ptr).unwrap().tensor_scale, 11.5);
        }
        assert!(lookup_header(1, ptr).is_none());
    }

    #[test]
    fn registry_concurrent_register_lookup_unregister_stress() {
        let workers: Vec<_> = (0..8)
            .map(|worker| {
                std::thread::spawn(move || {
                    for iteration in 0..256usize {
                        let ordinal = worker % 2;
                        let ptr =
                            (0x10_0000 + worker * 0x10_000 + iteration * 0x100) as *const c_void;
                        register_header(ordinal, ptr, iteration as f32, worker as u8);
                        register_mix(ordinal, ptr, 105, worker as i32, [worker as f32; 16]);
                        assert!(lookup_header(ordinal, ptr).is_some());
                        assert!(lookup_mix(ordinal, ptr).is_some());
                        unregister(ordinal, ptr);
                        assert!(lookup_header(ordinal, ptr).is_none());
                        assert!(lookup_mix(ordinal, ptr).is_none());
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("registry worker");
        }
    }

    #[test]
    fn maps_gguf_and_flm_ids() {
        assert_eq!(rung_from_ggml_type(108), Some(RUNG_GQH3));
        assert_eq!(rung_from_ggml_type(109), Some(RUNG_GQH2_H));
        assert_eq!(rung_from_ggml_type(110), Some(RUNG_GQH2_C));
        assert_eq!(rung_from_ggml_type(111), Some(RUNG_GQH4));
        assert_eq!(rung_from_flm_codec(13), Some(RUNG_GQH3));
        assert_eq!(rung_from_flm_codec(14), Some(RUNG_GQH2_H));
        assert_eq!(rung_from_flm_codec(15), Some(RUNG_GQH2_C));
        assert_eq!(rung_from_flm_codec(16), Some(RUNG_GQH4));
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
        assert!(
            matvec(ordinal, RUNG_GQH3, &wire, &x, &mut dst, 255, 1, 1, 255, 1, 1.0, 0).is_err()
        );
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
            let packed_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[packed.len()], &packed)
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

    fn require_gqh_artifacts() -> bool {
        std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
    }

    fn qwen38_gguf_path() -> Option<PathBuf> {
        let Some(value) = std::env::var_os("SUPERSONIC_GQH_GGUF") else {
            if require_gqh_artifacts() {
                panic!("SUPERSONIC_GQH_GGUF is required for Qwen3.8 GQH artifact tests");
            }
            return None;
        };
        let path = PathBuf::from(value);
        if path.is_file() {
            Some(path)
        } else if require_gqh_artifacts() {
            panic!(
                "SUPERSONIC_GQH_GGUF points to a missing artifact: {}",
                path.display()
            );
        } else {
            None
        }
    }

    #[test]
    fn hip_qwen38_gguf_gqh_linears_match_cpu() {
        let Some(path) = qwen38_gguf_path() else {
            eprintln!("skip: Qwen3.8 GQH GGUF not present");
            return;
        };
        let Some(ordinal) = require_hip() else {
            return;
        };
        let file = model_store::gguf::GgufFile::open(&path).expect("open qwen38 gguf");
        // Historical GGUF wire ID; Qwen3.8 is the sole public model identity.
        assert_eq!(file.kv("general.architecture"), Some("qwen35"));
        assert_eq!(file.gqh_header_count(), 350);

        let cases = [
            ("blk.0.ffn_gate.weight", GqhRung::Gqh3, RUNG_GQH3, 4),
            ("blk.0.ffn_down.weight", GqhRung::Gqh2H, RUNG_GQH2_H, 4),
            ("output.weight", GqhRung::Gqh2H, RUNG_GQH2_H, 2),
        ];
        for (name, cpu_rung, rung, rows) in cases {
            let tensor = file
                .tensor(name)
                .unwrap_or_else(|| panic!("missing {name}"));
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
    #[ignore = "requires R9700 artifact CI"]
    fn hip_qwen38_gguf_every_gqh_tensor_one_row() {
        let Some(path) = qwen38_gguf_path() else {
            eprintln!("skip: Qwen3.8 GQH GGUF not present");
            return;
        };
        let Some(ordinal) = require_hip() else {
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
                    GqhRung::Gqh4 => RUNG_GQH4,
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
            let x: Vec<f32> = (0..cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
            let want = kernel_order_matvec(&weights, &x, rows, cols);

            let device = device_packed(cpu_rung, rows, cols, packed);
            let packed_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
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
