//! HIP GQH decode and fused dequant-matvec.
//!
//! Decode is bit-exact against `model_store::gqh`. The fused matvec keeps the
//! same scale product order (`e4m3(d) * tensor_scale`, then `* (ratio/15)`).

use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_int, c_void};
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

fn remove_rust_metadata(ordinal: usize, ptrs: &[usize]) {
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
}

fn unregister_many_ffi(ordinal: usize, ptrs: &[usize]) -> Result<(), GpuError> {
    if ptrs.is_empty() {
        return Ok(());
    }
    let wires: Vec<*const c_void> = ptrs.iter().map(|&ptr| ptr as *const c_void).collect();
    let status = unsafe {
        supersonic_gqh_hip_unregister_wires(ordinal as c_int, wires.as_ptr(), wires.len())
    };
    if status == GQH_UNREGISTER_PRESTATE_OOM {
        return Err(backend_error(
            Backend::Hip,
            "gqh unregister pre-state allocation",
            status,
        ));
    }
    if status != 0 {
        unsafe {
            supersonic_gpu_integrity_fail_stop(
                c"gqh unregister returned after tracked-state failure".as_ptr(),
                status,
                ordinal as c_int,
            )
        }
    }
    Ok(())
}

fn abort_registration_cleanup(
    operation: &'static std::ffi::CStr,
    ordinal: usize,
    err: &GpuError,
) -> ! {
    eprintln!(
        "[gpu-integrity] fatal operation={} ordinal={} error={err}",
        operation.to_string_lossy(),
        ordinal
    );
    unsafe {
        supersonic_gpu_integrity_fail_stop(operation.as_ptr(), -1, ordinal as c_int);
    }
}

/// Remove all process-global metadata associated with a packed device buffer.
///
/// The C++ bridge caches layout conversion state by raw pointer, so this must
/// run before the owning `GpuBuffer` is freed. Rust metadata is removed only
/// after the C++ bridge reports success. Safe pre-state failures are returned
/// from [`try_unregister`]; tracked-state HIP failures fail-stop in the bridge.
pub fn unregister(ordinal: usize, ptr: *const c_void) {
    if ptr.is_null() {
        return;
    }
    if let Err(err) = try_unregister(ordinal, ptr) {
        abort_registration_cleanup(c"gqh unregister", ordinal, &err);
    }
}

pub fn try_unregister(ordinal: usize, ptr: *const c_void) -> Result<(), GpuError> {
    if ptr.is_null() {
        return Ok(());
    }
    unregister_many_ffi(ordinal, &[ptr as usize])?;
    remove_rust_metadata(ordinal, &[ptr as usize]);
    Ok(())
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

    pub fn try_unregister(&mut self) -> Result<(), GpuError> {
        if self.ptr == 0 {
            return Ok(());
        }
        try_unregister(self.ordinal, self.ptr as *const c_void)?;
        self.ptr = 0;
        Ok(())
    }

    pub fn unregister(&mut self) {
        if self.ptr == 0 {
            return;
        }
        if let Err(err) = self.try_unregister() {
            abort_registration_cleanup(c"gqh registration drop", self.ordinal, &err);
        }
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

    /// Clear committed registrations. A safe pre-state error leaves the
    /// committed guards intact. Rust maps and guard pointers are changed only
    /// after the C++ unregister callback succeeds; tracked-state HIP failures
    /// terminate in the C++ bridge before this result can cross FFI.
    pub fn try_clear(&mut self) -> Result<(), GpuError> {
        self.pending.clear();
        let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for registration in &self.committed {
            if registration.ptr != 0 {
                grouped
                    .entry(registration.ordinal)
                    .or_default()
                    .push(registration.ptr);
            }
        }
        for (ordinal, ptrs) in grouped {
            unregister_many_ffi(ordinal, &ptrs)?;
            remove_rust_metadata(ordinal, &ptrs);
            for registration in &mut self.committed {
                if registration.ordinal == ordinal && ptrs.contains(&registration.ptr) {
                    registration.ptr = 0;
                }
            }
        }
        self.committed.retain(|registration| registration.ptr != 0);
        Ok(())
    }

    pub fn clear(&mut self) {
        if let Err(err) = self.try_clear() {
            let ordinal = self
                .committed
                .first()
                .map(|registration| registration.ordinal)
                .unwrap_or(0);
            abort_registration_cleanup(c"gqh registration batch clear", ordinal, &err);
        }
    }
}

impl Default for RegistrationBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RegistrationBatch {
    fn drop(&mut self) {
        if let Err(err) = self.try_clear() {
            let ordinal = self
                .committed
                .first()
                .map(|registration| registration.ordinal)
                .unwrap_or(0);
            abort_registration_cleanup(c"gqh registration batch drop", ordinal, &err);
        }
    }
}

/// Invalidate bridge-side decode caches owned by one engine instance.
pub fn invalidate_decode_cache(
    ordinal: usize,
    layers: *const c_void,
    int4: *const c_void,
) -> Result<(), GpuError> {
    if layers.is_null() && int4.is_null() {
        return Ok(());
    }
    let status =
        unsafe { supersonic_qwen35_4b_hip_invalidate_decode_cache(ordinal as c_int, layers, int4) };
    if status != 0 {
        return Err(GpuError::backend_status(
            Backend::Hip,
            "qwen invalidate decode cache",
            status,
        ));
    }
    Ok(())
}

/// Restore any in-place decode weight layouts and invalidate caches owned by
/// one engine before it starts a new session.
pub fn reset_decode_cache(
    ordinal: usize,
    layers: *const c_void,
    int4: *const c_void,
    hidden_dim: usize,
    intermediate_size: usize,
) -> Result<(), GpuError> {
    if layers.is_null() && int4.is_null() {
        return Ok(());
    }
    let status = unsafe {
        supersonic_qwen35_4b_hip_reset_decode_cache(
            ordinal as c_int,
            layers,
            int4,
            hidden_dim as c_int,
            intermediate_size as c_int,
        )
    };
    if status != 0 {
        return Err(GpuError::backend_status(
            Backend::Hip,
            "qwen reset decode cache",
            status,
        ));
    }
    Ok(())
}

pub const RUNG_GQH3: i32 = 0;
pub const RUNG_GQH2_H: i32 = 1;
pub const RUNG_GQH2_C: i32 = 2;
pub const RUNG_GQH4: i32 = 3;

pub const SUPERBLOCK: usize = 256;

// This private bridge status means host allocation failed before any tracked
// state was touched. It is deliberately outside HIP's status range so a
// post-state HIP error can never be mistaken for a recoverable Rust error.
const GQH_UNREGISTER_PRESTATE_OOM: c_int = -0x4751_0001;

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
    fn supersonic_gqh_hip_quant_x_i8(
        device_ordinal: c_int,
        x: *const c_void,
        qx: *mut c_void,
        qxs: *mut c_void,
        ncols: c_int,
        in_dim: c_int,
        x_col_stride: i64,
    ) -> c_int;
    fn supersonic_gqh_hip_bake_q8_lut(
        device_ordinal: c_int,
        out: *mut c_void,
        rung: c_int,
        grid_code: c_int,
        q8n: c_int,
        nlev: c_int,
    ) -> c_int;
    fn supersonic_gqh_hip_matvec_i8(
        device_ordinal: c_int,
        rung: c_int,
        wire: *const c_void,
        qx: *const c_void,
        qxs: *const c_void,
        y: *mut c_void,
        in_dim: c_int,
        out_dim: c_int,
        ncols: c_int,
        q8n: c_int,
        qx_col_stride: i64,
        qxs_col_stride: i64,
        y_col_stride: i64,
        tensor_scale: f32,
        grid_code: c_int,
    ) -> c_int;

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

    fn supersonic_gqh_hip_unregister_wires(
        device_ordinal: c_int,
        wires: *const *const c_void,
        count: usize,
    ) -> c_int;

    fn supersonic_gpu_integrity_fail_stop(
        operation: *const c_char,
        status: c_int,
        device_ordinal: c_int,
    ) -> !;

    fn supersonic_qwen35_4b_hip_invalidate_decode_cache(
        device_ordinal: c_int,
        layers: *const c_void,
        int4: *const c_void,
    ) -> c_int;
    fn supersonic_qwen35_4b_hip_reset_decode_cache(
        device_ordinal: c_int,
        layers: *const c_void,
        int4: *const c_void,
        hidden_dim: c_int,
        intermediate_size: c_int,
    ) -> c_int;
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

/// GQH i8 activation quantization pre-pass (ports PR #35's `gqh_quant_x`).
///
/// Quantizes `ncols * in_dim` f32 activations (column-major, `x_col_stride`)
/// into int8 codes (`qx`, `[ncols * in_dim]`, column-major stride `in_dim`) and
/// per-8-group f32 scales (`qxs`, `[ncols * (in_dim/8)]`, column-major stride
/// `in_dim/8`). The scale folds both `/127`s (`1/(127*127)`) so the matvec's
/// per-row scale chain stays `d_real * ratio * qxs`.
pub fn quant_x_i8(
    ordinal: usize,
    x: &GpuBuffer,
    qx: &mut GpuBuffer,
    qxs: &mut GpuBuffer,
    ncols: usize,
    in_dim: usize,
    x_col_stride: usize,
) -> Result<(), GpuError> {
    if x.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 x must be f32, got {:?}",
            x.dtype()
        )));
    }
    if qx.dtype() != ScalarType::U8 {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 qx must be u8 (int8), got {:?}",
            qx.dtype()
        )));
    }
    if qxs.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 qxs must be f32, got {:?}",
            qxs.dtype()
        )));
    }
    if in_dim == 0 || in_dim % 8 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 in_dim {in_dim} must be a positive multiple of 8"
        )));
    }
    if x_col_stride < in_dim {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 x_col_stride {x_col_stride} < in_dim {in_dim}"
        )));
    }
    if x.elem_count() < ncols * x_col_stride {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 x has {} elems, need {ncols}*{x_col_stride}",
            x.elem_count()
        )));
    }
    if qx.elem_count() < ncols * in_dim {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 qx has {} elems, need {ncols}*{in_dim}",
            qx.elem_count()
        )));
    }
    let ngroups = in_dim / 8;
    if qxs.elem_count() < ncols * ngroups {
        return Err(GpuError::InvalidArg(format!(
            "gqh quant_x_i8 qxs has {} elems, need {ncols}*{ngroups}",
            qxs.elem_count()
        )));
    }
    let status = unsafe {
        supersonic_gqh_hip_quant_x_i8(
            ordinal as c_int,
            x.as_ptr() as *const c_void,
            qx.as_mut_ptr() as *mut c_void,
            qxs.as_mut_ptr() as *mut c_void,
            ncols as c_int,
            in_dim as c_int,
            x_col_stride as i64,
        )
    };
    if status != 0 {
        return Err(backend_error(Backend::Hip, "gqh quant_x_i8", status));
    }
    Ok(())
}

/// GQH i8 weight-LUT bake (device-side q8_round, ports PR #35's gqh_q8_level).
///
/// Bakes a rung's level grid to int8 at denominator `q8n`:
/// `q = rn(level * q8n)` clamped to `[-127, 127]`. Exposed so a harness verifies
/// the device bake matches the host denominator search
/// (`model_store::gqh_q8`) bit-for-bit. `rung`: 0=GQH3, 1=GQH2_H, 3=GQH4.
pub fn bake_q8_lut(
    ordinal: usize,
    out: &mut GpuBuffer,
    rung: i32,
    grid_code: u8,
    q8n: i32,
    nlev: usize,
) -> Result<(), GpuError> {
    if out.dtype() != ScalarType::U8 {
        return Err(GpuError::InvalidArg(format!(
            "gqh bake_q8_lut out must be u8 (int8), got {:?}",
            out.dtype()
        )));
    }
    if nlev == 0 || nlev > 16 {
        return Err(GpuError::InvalidArg(format!(
            "gqh bake_q8_lut nlev {nlev} must be in 1..=16"
        )));
    }
    if out.elem_count() < nlev {
        return Err(GpuError::InvalidArg(format!(
            "gqh bake_q8_lut out has {} elems, need {nlev}",
            out.elem_count()
        )));
    }
    if !(1..=127).contains(&q8n) {
        return Err(GpuError::InvalidArg(format!(
            "gqh bake_q8_lut q8n {q8n} must be in 1..=127"
        )));
    }
    let status = unsafe {
        supersonic_gqh_hip_bake_q8_lut(
            ordinal as c_int,
            out.as_mut_ptr() as *mut c_void,
            rung as c_int,
            grid_code as c_int,
            q8n as c_int,
            nlev as c_int,
        )
    };
    if status != 0 {
        return Err(backend_error(Backend::Hip, "gqh bake_q8_lut", status));
    }
    Ok(())
}

/// GQH i8 multicol matvec (ports PR #35's int8 verify arm). Takes pre-quantized
/// int8 activations (`qx`, `qxs` from [`quant_x_i8`]) and does a dp4a dot with
/// the baked int8 weight LUT (at denominator `q8n`). Not bit-exact; tolerance
/// is bounded by the weight LUT step (`row_max/127`). Supports GQH3/GQH2_H;
/// GQH4 must use the f32 [`matvec`] path.
pub fn matvec_i8(
    ordinal: usize,
    rung: i32,
    wire: &GpuBuffer,
    qx: &GpuBuffer,
    qxs: &GpuBuffer,
    y: &mut GpuBuffer,
    in_dim: usize,
    out_dim: usize,
    ncols: usize,
    q8n: i32,
    qx_col_stride: usize,
    qxs_col_stride: usize,
    y_col_stride: usize,
    tensor_scale: f32,
    grid_code: u8,
) -> Result<(), GpuError> {
    if qx.dtype() != ScalarType::U8 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matvec_i8 qx must be u8 (int8), got {:?}",
            qx.dtype()
        )));
    }
    if qxs.dtype() != ScalarType::F32 || y.dtype() != ScalarType::F32 {
        return Err(GpuError::InvalidArg(format!(
            "gqh matvec_i8 qxs/y must be f32, got {:?}/{:?}",
            qxs.dtype(),
            y.dtype()
        )));
    }
    if !(1..=127).contains(&q8n) {
        return Err(GpuError::InvalidArg(format!(
            "gqh matvec_i8 q8n {q8n} must be in 1..=127"
        )));
    }
    let status = unsafe {
        supersonic_gqh_hip_matvec_i8(
            ordinal as c_int,
            rung as c_int,
            wire.as_ptr() as *const c_void,
            qx.as_ptr() as *const c_void,
            qxs.as_ptr() as *const c_void,
            y.as_mut_ptr() as *mut c_void,
            in_dim as c_int,
            out_dim as c_int,
            ncols as c_int,
            q8n as c_int,
            qx_col_stride as i64,
            qxs_col_stride as i64,
            y_col_stride as i64,
            tensor_scale,
            grid_code as c_int,
        )
    };
    if status != 0 {
        return Err(backend_error(Backend::Hip, "gqh matvec_i8", status));
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
    #[cfg(supersonic_failure_injection)]
    use std::os::unix::process::ExitStatusExt;
    use std::path::PathBuf;
    #[cfg(supersonic_failure_injection)]
    use std::process::Command;

    #[cfg(supersonic_failure_injection)]
    unsafe extern "C" {
        fn supersonic_gqh_test_track_wire(device_ordinal: c_int, wire: *const c_void);
        fn supersonic_gqh_test_inject_unregister_prestate_failure();
        fn supersonic_gqh_test_inject_unregister_sync_failure(status: c_int);
        fn supersonic_gqh_test_trigger_post_enqueue_failure(status: c_int);
        fn supersonic_qwen35_4b_test_trigger_persistent_decode_failure(
            launch_status: c_int,
            sync_status: c_int,
        );
        fn supersonic_qwen35_4b_test_trigger_prepare_only_failure(sync_status: c_int);
    }

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

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_cleanup_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            let wire = 0x7_4000usize as *const c_void;
            unsafe {
                supersonic_gqh_test_track_wire(0, wire);
                supersonic_gqh_test_inject_unregister_sync_failure(709);
                let wires = [wire];
                let _ = supersonic_gqh_hip_unregister_wires(0, wires.as_ptr(), wires.len());
            }
            panic!("injected tracked-wire cleanup returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_cleanup_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn fatal cleanup child");
        // POSIX SIGABRT is signal 6; keep the death test dependency-free.
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_post_enqueue_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            unsafe {
                supersonic_gqh_test_trigger_post_enqueue_failure(811);
            }
            panic!("injected post-enqueue failure returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_post_enqueue_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn post-enqueue child");
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_4b_persistent_decode_launch_failure_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            unsafe {
                supersonic_qwen35_4b_test_trigger_persistent_decode_failure(907, 0);
            }
            panic!("injected 4B persistent launch failure returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_4b_persistent_decode_launch_failure_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn 4B persistent launch child");
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_4b_persistent_decode_sync_failure_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            unsafe {
                supersonic_qwen35_4b_test_trigger_persistent_decode_failure(0, 908);
            }
            panic!("injected 4B persistent synchronize failure returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_4b_persistent_decode_sync_failure_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn 4B persistent synchronize child");
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_4b_prepare_only_graph_sync_failure_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            unsafe {
                supersonic_qwen35_4b_test_trigger_prepare_only_failure(909);
            }
            panic!("injected 4B prepare-only graph synchronize failure returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_4b_prepare_only_graph_sync_failure_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn 4B prepare-only graph child");
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn fatal_4b_prepare_only_eager_sync_failure_aborts_in_child() {
        if std::env::var_os("SUPERSONIC_GPU_FAILURE_CHILD").is_some() {
            unsafe {
                supersonic_qwen35_4b_test_trigger_prepare_only_failure(910);
            }
            panic!("injected 4B prepare-only eager synchronize failure returned");
        }

        let status = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "gqh::tests::fatal_4b_prepare_only_eager_sync_failure_aborts_in_child",
                "--nocapture",
            ])
            .env("SUPERSONIC_GPU_FAILURE_CHILD", "1")
            .status()
            .expect("spawn 4B prepare-only eager child");
        assert_eq!(status.signal(), Some(6));
    }

    #[cfg(supersonic_failure_injection)]
    #[test]
    fn prestate_unregister_error_preserves_metadata_and_guards_for_retry() {
        let registration_ptr = 0x7_5000usize as *const c_void;
        register_header(0, registration_ptr, 12.5, 3);
        let mut registration = Registration::new(0, registration_ptr);
        unsafe { supersonic_gqh_test_inject_unregister_prestate_failure() };
        let err = registration
            .try_unregister()
            .expect_err("pre-state failure must be recoverable");
        assert!(format!("{err}").contains("pre-state"));
        assert!(lookup_header(0, registration_ptr).is_some());
        registration
            .try_unregister()
            .expect("retry after pre-state failure");
        assert!(lookup_header(0, registration_ptr).is_none());

        let batch_ptr = 0x7_6000usize as *const c_void;
        register_header(0, batch_ptr, 13.5, 4);
        let mut batch = RegistrationBatch::new();
        batch.stage_header(0, batch_ptr, 13.5, 4);
        batch.commit();
        unsafe { supersonic_gqh_test_inject_unregister_prestate_failure() };
        batch
            .try_clear()
            .expect_err("batch pre-state failure must be recoverable");
        assert!(lookup_header(0, batch_ptr).is_some());
        batch
            .try_clear()
            .expect("batch retry after pre-state failure");
        assert!(lookup_header(0, batch_ptr).is_none());
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
        assert_eq!(file.gqh_header_count(), 396);

        let cases = [
            ("blk.0.ffn_gate.weight", GqhRung::Gqh3, RUNG_GQH3, 4),
            ("blk.0.ffn_down.weight", GqhRung::Gqh3, RUNG_GQH3, 4),
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
        assert_eq!(checked, 393, "Q3KXL 64-layer GQH tensors plus lm_head");
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

    /// Offset-swept one-hot activation gate (PR #35 correctness finding).
    ///
    /// PR #35 found that the pre-existing fused check drove `x` with one-hot
    /// vectors at position `j < nvec`, so with `cols >= 256` it never placed a
    /// nonzero activation past position 15 -- the k-reduction was only ever
    /// exercised in its first superblock. A superblock-2/3 lane-map bug is
    /// invisible to that check. This gate sweeps a single nonzero activation
    /// across EVERY position j in [0, cols) (one per case), so every superblock
    /// and every lane offset is exercised with a nonzero term. Single-term, so
    /// still bit-exact against the CPU dot.
    #[test]
    fn hip_matvec_multicol_onehot_sweeps_all_positions() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        // gqh3_4x512: cols=512 = 2 superblocks. Sweep one-hot across positions
        // 0, 15, 16, 31, 255, 511 so superblock 0 and 1 and every lane offset
        // each get a nonzero term at ncols 8 and 16.
        let (wire, weights) = load_case("gqh3", 4, 512);
        let (_cpu_rung, rung, rows, cols) = (GqhRung::Gqh3, RUNG_GQH3, 4, 512);
        let (scale, grid_code, packed) = split_wire(_cpu_rung, &wire);
        let device = device_packed(_cpu_rung, rows, cols, packed);
        let packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                .expect("upload packed");
        let positions = [0usize, 15, 16, 31, 128, 255, 256, 511];
        for &ncols in &[8usize, 16] {
            for &j in &positions {
                // All columns share the same one-hot activation at position j.
                let mut x = vec![0.0f32; ncols * cols];
                for c in 0..ncols {
                    x[c * cols + j] = 1.0;
                }
                let mut want = vec![0.0f32; ncols * rows];
                for c in 0..ncols {
                    let col = &x[c * cols..(c + 1) * cols];
                    let y_col = kernel_order_matvec(&weights, col, rows, cols);
                    want[c * rows..(c + 1) * rows].copy_from_slice(&y_col);
                }
                let x_buf = GpuBuffer::from_host_bytes(
                    ordinal,
                    ScalarType::F32,
                    &[ncols * cols],
                    &f32_bytes(&x),
                )
                .expect("upload x");
                let mut y_buf =
                    GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * rows]).expect("alloc y");
                matvec(
                    ordinal,
                    rung,
                    &packed_buf,
                    &x_buf,
                    &mut y_buf,
                    cols,
                    rows,
                    ncols,
                    cols,
                    rows,
                    scale,
                    grid_code,
                )
                .unwrap_or_else(|e| panic!("onehot ncols={ncols} j={j}: {e}"));
                let got = read_f32(&y_buf);
                assert_bits_eq(
                    &got,
                    &want,
                    &format!("onehot ncols={ncols} pos={j} {rows}x{cols}"),
                );
            }
        }
    }

    /// GQH i8 activation quantization pre-pass: the int8 codes and per-8-group
    /// scales must match a CPU reference (amax/127 quantizer, qxs folds both
    /// /127s). This is the activation side of the i8 dot arm (PR #35's
    /// gqh_quant_x), verified standalone before the dot is wired.
    #[test]
    fn hip_quant_x_i8_matches_cpu_reference() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        let ncols = 4;
        let in_dim = 256; // 32 groups of 8
        let x_col_stride = in_dim;
        // Distinct activations per column/position.
        let x: Vec<f32> = (0..ncols * in_dim)
            .map(|idx| {
                let c = idx / in_dim;
                let j = idx % in_dim;
                (((c + j) % 19) as f32 - 9.0) / 7.0
            })
            .collect();
        // CPU reference.
        let ngroups = in_dim / 8;
        let mut want_q = vec![0i8; ncols * in_dim];
        let mut want_s = vec![0.0f32; ncols * ngroups];
        const XSCALE: f32 = 1.0 / (127.0 * 127.0);
        for c in 0..ncols {
            for g in 0..ngroups {
                let mut amax = 0.0f32;
                for t in 0..8 {
                    let v = x[c * x_col_stride + g * 8 + t];
                    amax = amax.max(v.abs());
                }
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                for t in 0..8 {
                    let q = (x[c * x_col_stride + g * 8 + t] * inv).round_ties_even() as i32;
                    let q = q.clamp(-127, 127);
                    want_q[c * in_dim + g * 8 + t] = q as i8;
                }
                want_s[c * ngroups + g] = amax * XSCALE;
            }
        }

        let x_buf = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::F32,
            &[ncols * x_col_stride],
            &f32_bytes(&x),
        )
        .expect("upload x");
        let mut qx_buf =
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[ncols * in_dim]).expect("alloc qx");
        let mut qxs_buf =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * ngroups]).expect("alloc qxs");
        quant_x_i8(
            ordinal,
            &x_buf,
            &mut qx_buf,
            &mut qxs_buf,
            ncols,
            in_dim,
            x_col_stride,
        )
        .expect("quant_x_i8");

        let qx_bytes = qx_buf.to_host_bytes().expect("download qx");
        let got_q: Vec<i8> = qx_bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(got_q.len(), want_q.len(), "qx length");
        for (i, (g, w)) in got_q.iter().zip(want_q.iter()).enumerate() {
            assert_eq!(g, w, "qx[{i}] col={} pos={}", i / in_dim, i % in_dim);
        }
        let got_s = read_f32(&qxs_buf);
        assert_eq!(got_s.len(), want_s.len(), "qxs length");
        for (i, (g, w)) in got_s.iter().zip(want_s.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1e-30),
                "qxs[{i}] got {g} want {w}"
            );
        }
    }

    /// GQH i8 weight-LUT bake: the device `q8_round` (rn(level * q8n) clamped)
    /// must match the CPU reference bit-for-bit, and the extreme-entry invariant
    /// (amax 1.0 -> q8_round(+-1.0, n) == +-n) must hold. Cross-checked against
    /// the derived denominators from model_store::gqh_q8.
    #[test]
    fn hip_bake_q8_lut_matches_cpu_q8_round() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        use model_store::gqh_q8::{q8_denom, GqhRung};
        // (rung code, model_store rung, nlev) for the shipping artifact's grids.
        let cases = [
            (0, GqhRung::Gqh3, 3, 8),
            (0, GqhRung::Gqh3, 4, 8),
            (3, GqhRung::Gqh4, 2, 16),
            (3, GqhRung::Gqh4, 3, 16),
            (3, GqhRung::Gqh4, 4, 16),
        ];
        for (rung, ms_rung, grid_code, nlev) in cases {
            let n = q8_denom(ms_rung, grid_code).unwrap();
            // CPU reference: round_ties_even(level * n) clamped, using the same
            // raw bit-pattern grids the device reads from __constant__ memory.
            let grid_levels = model_store::gqh_q8::grid_levels_f32(ms_rung, grid_code)
                .unwrap_or_else(|| panic!("grid {ms_rung:?} {grid_code}"));
            assert_eq!(grid_levels.len(), nlev);
            let mut want = vec![0i8; nlev];
            for i in 0..nlev {
                let lv = grid_levels[i];
                let q = (lv * (n as f32)).round_ties_even() as i32;
                want[i] = q.clamp(-127, 127) as i8;
            }
            // raw bit patterns for the extreme-entry invariant check below.
            let grid_bits: Vec<u32> = grid_levels.iter().map(|f| f.to_bits()).collect();
            let mut out = GpuBuffer::zeros(ordinal, ScalarType::U8, &[nlev]).expect("alloc");
            bake_q8_lut(ordinal, &mut out, rung, grid_code as u8, n, nlev).expect("bake");
            let bytes = out.to_host_bytes().expect("download");
            let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            assert_eq!(got.len(), want.len(), "rung {rung} code {grid_code} length");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "bake rung={rung} code={grid_code} n={n} level[{i}]");
            }
            // Extreme-entry invariant: the grid's extreme level is +-1.0
            // (0x3f800000 / 0xbf800000), so it bakes to +-n exactly.
            assert!(grid_bits.contains(&0x3f800000));
            assert!(grid_bits.contains(&0xbf800000));
            let pos_ext = grid_bits.iter().position(|&b| b == 0x3f800000).unwrap();
            let neg_ext = grid_bits.iter().position(|&b| b == 0xbf800000).unwrap();
            assert_eq!(got[pos_ext], n as i8, "extreme +1.0 -> +n");
            assert_eq!(got[neg_ext], -(n as i8), "extreme -1.0 -> -n");
        }
    }

    /// GQH i8 multicol matvec: the int8 dp4a dot must match the f32 reference
    /// within PR #35's dense tolerance, normalised by ||w_row|| * ||x_col||
    /// (I8_DENSE_REL = 2e-2). Exercises ncols 8 and 16 (the wide verify arm).
    #[test]
    fn hip_matvec_i8_multicol_matches_f32_within_tolerance() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        use model_store::gqh::GqhRung as MsRung;
        use model_store::gqh_q8::{q8_denom_eff, GqhRung as Q8Rung};
        let (name, cpu_rung, rung, rows, cols) = ("gqh3", MsRung::Gqh3, RUNG_GQH3, 4, 512);
        let (wire, weights) = load_case(name, rows, cols);
        let (scale, grid_code, packed) = split_wire(cpu_rung, &wire);
        let device = device_packed(cpu_rung, rows, cols, packed);
        let packed_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                .expect("upload packed");

        // Dense activations, distinct per column.
        let ncols = 8;
        let x: Vec<f32> = (0..ncols * cols)
            .map(|idx| {
                let c = idx / cols;
                let j = idx % cols;
                (((c + j) % 19) as f32 - 9.0) / 7.0
            })
            .collect();

        // f32 reference: y_ref[n*rows + r] = sum_k weights[r*cols + k] * x[n*cols + k].
        let mut y_ref = vec![0.0f32; ncols * rows];
        let mut w_row_norm = vec![0.0f64; rows];
        let mut x_col_norm = vec![0.0f64; ncols];
        for r in 0..rows {
            let mut sq = 0.0f64;
            for k in 0..cols {
                sq += (weights[r * cols + k] as f64).powi(2);
            }
            w_row_norm[r] = sq.sqrt();
        }
        for n in 0..ncols {
            let mut sq = 0.0f64;
            for k in 0..cols {
                sq += (x[n * cols + k] as f64).powi(2);
            }
            x_col_norm[n] = sq.sqrt();
        }
        for n in 0..ncols {
            for r in 0..rows {
                let mut acc = 0.0f64;
                for k in 0..cols {
                    acc += weights[r * cols + k] as f64 * x[n * cols + k] as f64;
                }
                y_ref[n * rows + r] = acc as f32;
            }
        }

        // Quantize activations to int8 (pre-pass, verified separately).
        let ngroups = cols / 8;
        let x_buf =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::F32, &[ncols * cols], &f32_bytes(&x))
                .expect("upload x");
        let mut qx_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[ncols * cols]).expect("qx");
        let mut qxs_buf =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * ngroups]).expect("qxs");
        quant_x_i8(
            ordinal,
            &x_buf,
            &mut qx_buf,
            &mut qxs_buf,
            ncols,
            cols,
            cols,
        )
        .expect("quant_x_i8");

        // Run the i8 matvec.
        let q8n = q8_denom_eff(Q8Rung::Gqh3, grid_code).unwrap();
        let mut y_buf =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * rows]).expect("alloc y");
        matvec_i8(
            ordinal,
            rung,
            &packed_buf,
            &qx_buf,
            &qxs_buf,
            &mut y_buf,
            cols,
            rows,
            ncols,
            q8n,
            cols,
            ngroups,
            rows,
            scale,
            grid_code,
        )
        .expect("matvec_i8");
        let got = read_f32(&y_buf);

        // PR #35's dense tolerance: |got - ref| <= I8_DENSE_REL * ||w_row|| * ||x_col||.
        const I8_DENSE_REL: f64 = 2e-2;
        assert_eq!(got.len(), y_ref.len(), "i8 matvec length");
        let mut max_rel = 0.0f64;
        for n in 0..ncols {
            for r in 0..rows {
                let idx = n * rows + r;
                let bound = I8_DENSE_REL * w_row_norm[r] * x_col_norm[n];
                let err = (got[idx] as f64 - y_ref[idx] as f64).abs();
                let rel = if bound > 1e-30 { err / bound } else { 0.0 };
                max_rel = max_rel.max(rel);
                assert!(
                    err <= bound,
                    "i8 matvec [{n},{r}] err {err:.6} > bound {bound:.6} (rel {rel:.4})",
                );
            }
        }
        eprintln!("[i8-matvec] max_rel = {max_rel:.4} (tolerance {I8_DENSE_REL})");
    }

    /// The i8 dispatch helper (`try_matmul_gqh_i8_dot`) that
    /// `matmul_rhs_transposed_gqh` calls when `GGML_GQH_I8DOT` is on: it must
    /// run the int8 arm for representable gqh3/gqh2_h grids and match the f32
    /// CPU matvec within PR #35's dense tolerance, and decline (`Ok(false)`)
    /// for rungs with no int8 arm (GQH4). Exercises the full arm -- activation
    /// int8 quantization pre-pass plus the dp4a matvec -- end to end.
    #[test]
    fn hip_matmul_i8_dot_dispatch_matches_f32_within_tolerance() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        use model_store::gqh::GqhRung as MsRung;
        const I8_DENSE_REL: f64 = 2e-2;
        let cases = [
            ("gqh3", MsRung::Gqh3, RUNG_GQH3, 4, 512),
            ("gqh2_h", MsRung::Gqh2H, RUNG_GQH2_H, 4, 512),
        ];
        for (name, cpu_rung, rung, rows, cols) in cases {
            let (wire, weights) = load_case(name, rows, cols);
            let (scale, grid_code, packed) = split_wire(cpu_rung, &wire);
            let device = device_packed(cpu_rung, rows, cols, packed);
            let rhs_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                    .expect("upload packed");
            let ncols = 8;
            let x: Vec<f32> = (0..ncols * cols)
                .map(|idx| {
                    let n = idx / cols;
                    let i = idx % cols;
                    (((n + i) % 19) as f32 - 9.0) / 7.0
                })
                .collect();
            let mut y_ref = vec![0.0f32; ncols * rows];
            let mut w_row_norm = vec![0.0f64; rows];
            let mut x_col_norm = vec![0.0f64; ncols];
            for r in 0..rows {
                let mut sq = 0.0f64;
                for kk in 0..cols {
                    sq += (weights[r * cols + kk] as f64).powi(2);
                }
                w_row_norm[r] = sq.sqrt();
            }
            for n in 0..ncols {
                let mut sq = 0.0f64;
                for kk in 0..cols {
                    sq += (x[n * cols + kk] as f64).powi(2);
                }
                x_col_norm[n] = sq.sqrt();
            }
            for n in 0..ncols {
                let col = &x[n * cols..(n + 1) * cols];
                let y_col = kernel_order_matvec(&weights, col, rows, cols);
                y_ref[n * rows..(n + 1) * rows].copy_from_slice(&y_col);
            }
            let x_buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::F32,
                &[ncols * cols],
                &f32_bytes(&x),
            )
            .expect("upload x");
            let mut y_buf =
                GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * rows]).expect("alloc y");
            let ran = crate::prefill_ffi::try_matmul_gqh_i8_dot(
                ordinal, ncols, rows, cols, &x_buf, &rhs_buf, scale, grid_code, rung, &mut y_buf,
            )
            .expect("try_matmul_gqh_i8_dot");
            assert!(ran, "{name} i8 dot arm should run for grid {grid_code}");
            let got = read_f32(&y_buf);
            assert_eq!(got.len(), y_ref.len(), "{name} i8 dispatch length");
            let mut max_rel = 0.0f64;
            for n in 0..ncols {
                for r in 0..rows {
                    let idx = n * rows + r;
                    let bound = I8_DENSE_REL * w_row_norm[r] * x_col_norm[n];
                    let err = (got[idx] as f64 - y_ref[idx] as f64).abs();
                    let rel = if bound > 1e-30 { err / bound } else { 0.0 };
                    max_rel = max_rel.max(rel);
                    assert!(
                        err <= bound,
                        "{name} i8 dispatch [{n},{r}] err {err:.6} > bound {bound:.6}",
                    );
                }
            }
            eprintln!("[i8-dispatch {name}] max_rel = {max_rel:.4} (tolerance {I8_DENSE_REL})");
        }

        // Decline: GQH4 has no int8 arm; the dispatch returns Ok(false) before
        // touching the weight buffer, so a gqh3 wire paired with the GQH4 rung
        // is a safe decline probe.
        let (wire, _weights) = load_case("gqh3", 4, 512);
        let (scale, grid_code, packed) = split_wire(MsRung::Gqh3, &wire);
        let device = device_packed(MsRung::Gqh3, 4, 512, packed);
        let rhs_buf = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
            .expect("upload packed");
        let x_buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[512]).expect("x");
        let mut y_buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[4]).expect("y");
        let ran = crate::prefill_ffi::try_matmul_gqh_i8_dot(
            ordinal, 1, 4, 512, &x_buf, &rhs_buf, scale, grid_code, RUNG_GQH4, &mut y_buf,
        )
        .expect("try_matmul_gqh_i8_dot decline");
        assert!(!ran, "GQH4 must decline the int8 arm");
    }

    /// Wide multicol matvec (DFlash2 verify blocks 9..16) must be bit-exact
    /// against the per-column CPU dot product. Exercises the kCols=16 wide arm
    /// added to match PR #35's exact-width f32 multicol, including a partial
    /// width (ncols 13, where the `c >= ncol` guard skips inactive columns).
    #[test]
    fn hip_matvec_multicol_wide_matches_cpu_dot() {
        let Some(ordinal) = require_hip() else {
            return;
        };
        // gqh3_4x512 and gqh2_h_4x512 carry rows=4, cols=512 (a SUPERBLOCK
        // multiple); out_dim=4 is small but the wide arm's grid math and the
        // per-column writeback still cover it. ncols 8 is the existing arm
        // (cross-check); 9/13/16 are the new wide arm (13 is partial).
        let cases = [
            ("gqh3", GqhRung::Gqh3, RUNG_GQH3, 4, 512),
            ("gqh2_h", GqhRung::Gqh2H, RUNG_GQH2_H, 4, 512),
        ];
        let ncols_list = [8usize, 9, 13, 16];
        for (name, cpu_rung, rung, rows, cols) in cases {
            let (wire, weights) = load_case(name, rows, cols);
            let (scale, grid_code, packed) = split_wire(cpu_rung, &wire);
            let device = device_packed(cpu_rung, rows, cols, packed);
            let packed_buf =
                GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[device.len()], &device)
                    .expect("upload packed");
            for &ncols in &ncols_list {
                // Distinct activation per column so a mis-mapped column is
                // caught: column n uses value (((n + i) % 19) - 9) / 9.
                let x: Vec<f32> = (0..ncols * cols)
                    .map(|idx| {
                        let n = idx / cols;
                        let i = idx % cols;
                        (((n + i) % 19) as f32 - 9.0) / 9.0
                    })
                    .collect();
                let mut want = vec![0.0f32; ncols * rows];
                for n in 0..ncols {
                    let col = &x[n * cols..(n + 1) * cols];
                    let y_col = kernel_order_matvec(&weights, col, rows, cols);
                    want[n * rows..(n + 1) * rows].copy_from_slice(&y_col);
                }
                let x_buf = GpuBuffer::from_host_bytes(
                    ordinal,
                    ScalarType::F32,
                    &[ncols * cols],
                    &f32_bytes(&x),
                )
                .expect("upload x");
                let mut y_buf =
                    GpuBuffer::zeros(ordinal, ScalarType::F32, &[ncols * rows]).expect("alloc y");
                matvec(
                    ordinal,
                    rung,
                    &packed_buf,
                    &x_buf,
                    &mut y_buf,
                    cols,
                    rows,
                    ncols,
                    cols,
                    rows,
                    scale,
                    grid_code,
                )
                .unwrap_or_else(|e| panic!("{name} multicol ncols={ncols}: {e}"));
                let got = read_f32(&y_buf);
                assert_bits_eq(
                    &got,
                    &want,
                    &format!("{name} multicol ncols={ncols} {rows}x{cols}"),
                );
            }
        }
    }
}
