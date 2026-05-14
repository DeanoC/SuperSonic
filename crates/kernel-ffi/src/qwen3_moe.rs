//! FFI bridge for the Qwen3-MoE HIP kernel family.
//!
//! This is deliberately separate from `qwen36_moe`: Qwen3-30B-A3B is a
//! pure-attention MoE model with different geometry and no Qwen3.6 hybrid
//! linear-attention/shared-expert path.

#![cfg_attr(not(supersonic_backend_hip), allow(unused_imports, unused_variables))]

use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;

use gpu_hal::{Backend, GpuBuffer, GpuError, ScalarType};

const LOWBIT_NATIVE_INT4: i32 = 4;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen3MoeDecodeLayerDesc {
    pub layer_idx: c_int,
    pub input_norm_w: *const c_void,
    pub input_norm_eps: f32,
    pub post_attn_norm_w: *const c_void,
    pub post_attn_norm_eps: f32,

    pub q_proj_w: *const c_void,
    pub k_proj_w: *const c_void,
    pub v_proj_w: *const c_void,
    pub o_proj_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub rope_theta: f32,
    pub head_dim: c_int,
    pub num_heads: c_int,
    pub num_kv_heads: c_int,
    pub kv_cache_k: *mut c_void,
    pub kv_cache_v: *mut c_void,
    pub kv_len: c_int,
    pub kv_max_t: c_int,

    pub router_w: *const c_void,
    pub experts_gate_up_w: *const c_void,
    pub experts_down_w: *const c_void,
    pub num_experts: c_int,
    pub top_k: c_int,
    pub moe_intermediate_size: c_int,
    pub norm_topk_prob: c_int,
}

unsafe impl Send for Qwen3MoeDecodeLayerDesc {}
unsafe impl Sync for Qwen3MoeDecodeLayerDesc {}

impl Default for Qwen3MoeDecodeLayerDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Qwen3MoeInt4ScaleDesc {
    pub q_proj_scale: *const c_void,
    pub q_proj_zero: *const c_void,
    pub k_proj_scale: *const c_void,
    pub k_proj_zero: *const c_void,
    pub v_proj_scale: *const c_void,
    pub v_proj_zero: *const c_void,
    pub o_proj_scale: *const c_void,
    pub o_proj_zero: *const c_void,
    pub experts_gate_up_scale: *const c_void,
    pub experts_gate_up_zero: *const c_void,
    pub experts_down_scale: *const c_void,
    pub experts_down_zero: *const c_void,
    pub group_size: c_int,
}

unsafe impl Send for Qwen3MoeInt4ScaleDesc {}
unsafe impl Sync for Qwen3MoeInt4ScaleDesc {}

impl Default for Qwen3MoeInt4ScaleDesc {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(supersonic_backend_hip)]
extern "C" {
    pub fn qwen3_moe_hip_stub_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: usize,
        layers: *const Qwen3MoeDecodeLayerDesc,
        workspace: *mut f32,
        counters: *mut c_uint,
    ) -> c_int;

    pub fn qwen3_moe_hip_decode_layer_launch(
        dtype: c_int,
        device_ordinal: usize,
        layer: *const Qwen3MoeDecodeLayerDesc,
        int4: *const Qwen3MoeInt4ScaleDesc,
        hidden: c_int,
        position: c_int,
        input_hidden: *const c_void,
        output_hidden: *mut c_void,
        workspace: *mut f32,
    ) -> c_int;

    pub fn qwen3_moe_hip_lm_head_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        logits: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;

    pub fn qwen3_moe_hip_persistent_decode_launch(
        dtype: c_int,
        device_ordinal: usize,
        num_layers: c_int,
        layers: *const Qwen3MoeDecodeLayerDesc,
        int4_descs: *const Qwen3MoeInt4ScaleDesc,
        hidden: c_int,
        position: c_int,
        input_hidden: *const c_void,
        hidden_ping: *mut c_void,
        hidden_pong: *mut c_void,
        workspace: *mut f32,
        sync: *mut c_uint,
        profile: *mut u64,
    ) -> c_int;

    pub fn qwen3_moe_hip_lm_head_int4_launch(
        dtype: c_int,
        device_ordinal: usize,
        hidden: c_int,
        vocab: c_int,
        rms_norm_eps: f32,
        final_hidden: *const c_void,
        final_norm_w: *const c_void,
        lm_head_w: *const c_void,
        lm_head_scale: *const c_void,
        lm_head_zero: *const c_void,
        group_size: c_int,
        logits: *mut c_void,
        counter: *mut c_uint,
    ) -> c_int;
}

pub fn stub_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer_descs_device: &GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    num_layers: usize,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::stub_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = layer_descs_device.backend();
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_stub_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers,
                    layer_descs_device.as_ptr() as *const Qwen3MoeDecodeLayerDesc,
                    workspace.as_mut_ptr() as *mut f32,
                    sync_buf.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::stub_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::stub_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe stub launch failed with status {status}"),
        ));
    }
    Ok(())
}

pub fn decode_layer_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer: &Qwen3MoeDecodeLayerDesc,
    int4: &Qwen3MoeInt4ScaleDesc,
    hidden: i32,
    position: i32,
    input_hidden: &GpuBuffer,
    output_hidden: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = input_hidden.backend();
    if backend == Backend::Metal {
        return decode_layer_launch_metal_host(
            hidden,
            position,
            layer,
            int4,
            input_hidden,
            output_hidden,
            workspace,
        );
    }
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_decode_layer_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    layer as *const Qwen3MoeDecodeLayerDesc,
                    int4 as *const Qwen3MoeInt4ScaleDesc,
                    hidden,
                    position,
                    input_hidden.as_ptr(),
                    output_hidden.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    layer,
                    int4,
                    hidden,
                    position,
                    output_hidden,
                    workspace,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::decode_layer_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::decode_layer_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe decode layer launch failed with status {status}"),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_layer_launch_metal_host(
    hidden: i32,
    position: i32,
    layer: &Qwen3MoeDecodeLayerDesc,
    int4: &Qwen3MoeInt4ScaleDesc,
    input_hidden: &GpuBuffer,
    output_hidden: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if input_hidden.dtype() != ScalarType::BF16
        || output_hidden.dtype() != ScalarType::BF16
        || workspace.dtype() != ScalarType::F32
    {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback expects BF16/BF16/F32 buffers, got {:?}/{:?}/{:?}",
            input_hidden.dtype(),
            output_hidden.dtype(),
            workspace.dtype()
        )));
    }
    let hidden = checked_positive("hidden", hidden)?;
    let head_dim = checked_positive("head_dim", layer.head_dim)?;
    let num_heads = checked_positive("num_heads", layer.num_heads)?;
    let num_kv_heads = checked_positive("num_kv_heads", layer.num_kv_heads)?;
    let experts = checked_positive("num_experts", layer.num_experts)?;
    let top_k = checked_positive("top_k", layer.top_k)?;
    let moe_i = checked_positive("moe_intermediate_size", layer.moe_intermediate_size)?;
    let group_size = checked_positive("group_size", int4.group_size)?;
    if num_heads % num_kv_heads != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback requires num_heads divisible by num_kv_heads, got {num_heads}/{num_kv_heads}"
        )));
    }
    if head_dim % 2 != 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback requires even head_dim, got {head_dim}"
        )));
    }
    if top_k > experts {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback top_k {top_k} exceeds experts {experts}"
        )));
    }
    require_ptrs(
        layer,
        int4,
        &[
            layer.input_norm_w,
            layer.post_attn_norm_w,
            layer.q_proj_w,
            layer.k_proj_w,
            layer.v_proj_w,
            layer.o_proj_w,
            layer.q_norm_w,
            layer.k_norm_w,
            layer.router_w,
            layer.experts_gate_up_w,
            layer.experts_down_w,
            int4.q_proj_scale,
            int4.q_proj_zero,
            int4.k_proj_scale,
            int4.k_proj_zero,
            int4.v_proj_scale,
            int4.v_proj_zero,
            int4.o_proj_scale,
            int4.o_proj_zero,
            int4.experts_gate_up_scale,
            int4.experts_gate_up_zero,
            int4.experts_down_scale,
            int4.experts_down_zero,
        ],
    )?;

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let needed_workspace = hidden
        + q_dim
        + kv_dim
        + kv_dim
        + q_dim
        + hidden
        + experts
        + experts
        + top_k
        + top_k
        + 2 * moe_i
        + moe_i
        + hidden
        + hidden;
    let workspace_elems = workspace.shape().iter().product::<usize>();
    if workspace_elems < needed_workspace {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback workspace too small: need {needed_workspace}, got {workspace_elems}"
        )));
    }
    let input_elems = input_hidden.shape().iter().product::<usize>();
    let output_elems = output_hidden.shape().iter().product::<usize>();
    if input_elems < hidden || output_elems < hidden {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback hidden buffers too small: input {input_elems}, output {output_elems}, hidden {hidden}"
        )));
    }

    let input = unsafe { std::slice::from_raw_parts(input_hidden.as_ptr() as *const u16, hidden) };
    let output =
        unsafe { std::slice::from_raw_parts_mut(output_hidden.as_mut_ptr() as *mut u16, hidden) };

    let mut x = input
        .iter()
        .map(|bits| bf16_bits_to_f32(*bits))
        .collect::<Vec<_>>();
    let mut hnorm = vec![0.0f32; hidden];
    q3_rms_norm_host(
        &x,
        layer.input_norm_w as *const u16,
        layer.input_norm_eps,
        &mut hnorm,
    );

    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    q3_int4_matvec_host(
        layer.q_proj_w as *const u8,
        int4.q_proj_scale as *const u16,
        int4.q_proj_zero as *const u16,
        q_dim,
        hidden,
        group_size,
        &hnorm,
        &mut q,
    );
    q3_int4_matvec_host(
        layer.k_proj_w as *const u8,
        int4.k_proj_scale as *const u16,
        int4.k_proj_zero as *const u16,
        kv_dim,
        hidden,
        group_size,
        &hnorm,
        &mut k,
    );
    q3_int4_matvec_host(
        layer.v_proj_w as *const u8,
        int4.v_proj_scale as *const u16,
        int4.v_proj_zero as *const u16,
        kv_dim,
        hidden,
        group_size,
        &hnorm,
        &mut v,
    );

    q3_head_rms_norm_host(
        &mut q,
        num_heads,
        head_dim,
        layer.q_norm_w as *const u16,
        layer.input_norm_eps,
    );
    q3_head_rms_norm_host(
        &mut k,
        num_kv_heads,
        head_dim,
        layer.k_norm_w as *const u16,
        layer.input_norm_eps,
    );
    q3_rope_half_host(&mut q, num_heads, head_dim, position, layer.rope_theta);
    q3_rope_half_host(&mut k, num_kv_heads, head_dim, position, layer.rope_theta);

    let cache_k_set = !layer.kv_cache_k.is_null();
    let cache_v_set = !layer.kv_cache_v.is_null();
    if cache_k_set != cache_v_set {
        return Err(GpuError::InvalidArg(
            "qwen3_moe::decode_layer_launch: Metal fallback requires paired KV cache pointers"
                .into(),
        ));
    }
    let kv_len = if cache_k_set {
        if layer.kv_len < 0 || layer.kv_len >= layer.kv_max_t {
            return Err(GpuError::InvalidArg(format!(
                "qwen3_moe::decode_layer_launch: Metal fallback invalid kv_len {} for kv_max_t {}",
                layer.kv_len, layer.kv_max_t
            )));
        }
        let base = layer.kv_len as usize * kv_dim;
        let cache_elems = layer.kv_max_t as usize * kv_dim;
        let cache_k =
            unsafe { std::slice::from_raw_parts_mut(layer.kv_cache_k as *mut u16, cache_elems) };
        let cache_v =
            unsafe { std::slice::from_raw_parts_mut(layer.kv_cache_v as *mut u16, cache_elems) };
        for idx in 0..kv_dim {
            cache_k[base + idx] = f32_to_bf16_bits(k[idx]);
            cache_v[base + idx] = f32_to_bf16_bits(v[idx]);
        }
        layer.kv_len as usize + 1
    } else {
        1
    };

    let mut attn = vec![0.0f32; q_dim];
    let rep = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let cache_k = if cache_k_set {
        Some(unsafe {
            std::slice::from_raw_parts(
                layer.kv_cache_k as *const u16,
                layer.kv_max_t as usize * kv_dim,
            )
        })
    } else {
        None
    };
    let cache_v = if cache_v_set {
        Some(unsafe {
            std::slice::from_raw_parts(
                layer.kv_cache_v as *const u16,
                layer.kv_max_t as usize * kv_dim,
            )
        })
    } else {
        None
    };
    for head in 0..num_heads {
        let kv_head = head / rep;
        let mut scores = vec![0.0f32; kv_len];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..kv_len {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                let kval = cache_k
                    .map(|cache| bf16_bits_to_f32(cache[t * kv_dim + kv_head * head_dim + i]))
                    .unwrap_or_else(|| k[kv_head * head_dim + i]);
                dot += q[head * head_dim + i] * kval;
            }
            let score = dot * scale;
            scores[t] = score;
            max_score = max_score.max(score);
        }
        let mut denom = 0.0f32;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        for i in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..kv_len {
                let vv = cache_v
                    .map(|cache| bf16_bits_to_f32(cache[t * kv_dim + kv_head * head_dim + i]))
                    .unwrap_or_else(|| v[kv_head * head_dim + i]);
                acc += (scores[t] / denom) * vv;
            }
            attn[head * head_dim + i] = bf16_round_f32(acc);
        }
    }

    let mut attn_out = vec![0.0f32; hidden];
    q3_int4_matvec_host(
        layer.o_proj_w as *const u8,
        int4.o_proj_scale as *const u16,
        int4.o_proj_zero as *const u16,
        hidden,
        q_dim,
        group_size,
        &attn,
        &mut attn_out,
    );
    for i in 0..hidden {
        x[i] = bf16_round_f32(x[i] + bf16_round_f32(attn_out[i]));
    }

    q3_rms_norm_host(
        &x,
        layer.post_attn_norm_w as *const u16,
        layer.post_attn_norm_eps,
        &mut hnorm,
    );
    let mut router = vec![0.0f32; experts];
    q3_bf16_matvec_host(
        layer.router_w as *const u16,
        experts,
        hidden,
        &hnorm,
        &mut router,
    );
    let (topk_val, topk_idx) = q3_router_topk_host(&router, top_k, layer.norm_topk_prob != 0);

    let mut moe_out = vec![0.0f32; hidden];
    let mut gu = vec![0.0f32; 2 * moe_i];
    let mut mid = vec![0.0f32; moe_i];
    let mut expert_out = vec![0.0f32; hidden];
    for kpos in 0..top_k {
        let expert = topk_idx[kpos];
        q3_expert_int4_matvec_host(
            layer.experts_gate_up_w as *const u8,
            int4.experts_gate_up_scale as *const u16,
            int4.experts_gate_up_zero as *const u16,
            expert,
            2 * moe_i,
            hidden,
            group_size,
            &hnorm,
            &mut gu,
        );
        for i in 0..moe_i {
            let g = gu[i];
            let u = gu[moe_i + i];
            mid[i] = bf16_round_f32((g / (1.0 + (-g).exp())) * u);
        }
        q3_expert_int4_matvec_host(
            layer.experts_down_w as *const u8,
            int4.experts_down_scale as *const u16,
            int4.experts_down_zero as *const u16,
            expert,
            hidden,
            moe_i,
            group_size,
            &mid,
            &mut expert_out,
        );
        for i in 0..hidden {
            moe_out[i] += topk_val[kpos] * expert_out[i];
        }
    }

    for i in 0..hidden {
        output[i] = f32_to_bf16_bits(bf16_round_f32(x[i] + bf16_round_f32(moe_out[i])));
    }
    Ok(())
}

pub fn lm_head_launch(
    ordinal: usize,
    dtype: ScalarType,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    logits: &mut GpuBuffer,
    counter: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::lm_head_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = final_hidden.backend();
    if backend == Backend::Metal {
        let _ = counter;
        return lm_head_launch_metal_bf16(
            ordinal,
            hidden,
            vocab,
            rms_norm_eps,
            final_hidden,
            final_norm_w,
            lm_head_w,
            logits,
        );
    }
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_lm_head_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_hidden.as_ptr(),
                    final_norm_w.as_ptr(),
                    lm_head_w.as_ptr(),
                    logits.as_mut_ptr(),
                    counter.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_norm_w,
                    lm_head_w,
                    logits,
                    counter,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::lm_head_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::lm_head_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe lm_head launch failed with status {status}"),
        ));
    }
    Ok(())
}

pub fn persistent_decode_launch(
    ordinal: usize,
    dtype: ScalarType,
    layer_descs_device: &GpuBuffer,
    int4_descs_device: &GpuBuffer,
    num_layers: usize,
    hidden: i32,
    position: i32,
    input_hidden: &GpuBuffer,
    hidden_ping: &mut GpuBuffer,
    hidden_pong: &mut GpuBuffer,
    workspace: &mut GpuBuffer,
    sync_buf: &mut GpuBuffer,
    profile_buf: Option<&mut GpuBuffer>,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::persistent_decode_launch: only BF16 is wired, got {dtype:?}"
        )));
    }
    let backend = input_hidden.backend();
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_persistent_decode_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    num_layers as c_int,
                    layer_descs_device.as_ptr() as *const Qwen3MoeDecodeLayerDesc,
                    int4_descs_device.as_ptr() as *const Qwen3MoeInt4ScaleDesc,
                    hidden,
                    position,
                    input_hidden.as_ptr(),
                    hidden_ping.as_mut_ptr(),
                    hidden_pong.as_mut_ptr(),
                    workspace.as_mut_ptr() as *mut f32,
                    sync_buf.as_mut_ptr() as *mut c_uint,
                    profile_buf
                        .map(|buf| buf.as_mut_ptr() as *mut u64)
                        .unwrap_or(std::ptr::null_mut()),
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    layer_descs_device,
                    int4_descs_device,
                    num_layers,
                    hidden,
                    position,
                    hidden_ping,
                    hidden_pong,
                    workspace,
                    sync_buf,
                    profile_buf,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::persistent_decode_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::persistent_decode_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe persistent decode launch failed with status {status}"),
        ));
    }
    Ok(())
}

pub fn lm_head_int4_launch(
    ordinal: usize,
    dtype: ScalarType,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    lm_head_scale: &GpuBuffer,
    lm_head_zero: &GpuBuffer,
    group_size: i32,
    logits: &mut GpuBuffer,
    counter: &mut GpuBuffer,
) -> Result<(), GpuError> {
    if dtype != ScalarType::BF16 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::lm_head_int4_launch: only BF16 activations are wired, got {dtype:?}"
        )));
    }
    let backend = final_hidden.backend();
    if backend == Backend::Metal {
        let _ = counter;
        return lm_head_int4_launch_metal_bf16(
            ordinal,
            hidden,
            vocab,
            rms_norm_eps,
            final_hidden,
            final_norm_w,
            lm_head_w,
            lm_head_scale,
            lm_head_zero,
            group_size,
            logits,
        );
    }
    let status: c_int = match backend {
        Backend::Hip => {
            #[cfg(supersonic_backend_hip)]
            unsafe {
                qwen3_moe_hip_lm_head_int4_launch(
                    dtype.kernel_dtype_code(),
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_hidden.as_ptr(),
                    final_norm_w.as_ptr(),
                    lm_head_w.as_ptr(),
                    lm_head_scale.as_ptr(),
                    lm_head_zero.as_ptr(),
                    group_size,
                    logits.as_mut_ptr(),
                    counter.as_mut_ptr() as *mut c_uint,
                )
            }
            #[cfg(not(supersonic_backend_hip))]
            {
                let _ = (
                    ordinal,
                    hidden,
                    vocab,
                    rms_norm_eps,
                    final_norm_w,
                    lm_head_w,
                    lm_head_scale,
                    lm_head_zero,
                    group_size,
                    logits,
                    counter,
                );
                return Err(GpuError::InvalidArg(
                    "qwen3_moe::lm_head_int4_launch: HIP backend not compiled".into(),
                ));
            }
        }
        Backend::Cuda | Backend::Metal => {
            return Err(GpuError::InvalidArg(
                "qwen3_moe::lm_head_int4_launch: Qwen3-MoE is HIP-only".into(),
            ));
        }
    };
    if status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("qwen3_moe INT4 lm_head launch failed with status {status}"),
        ));
    }
    Ok(())
}

fn checked_lm_head_dims(
    hidden: i32,
    vocab: i32,
    group_size: Option<i32>,
) -> Result<(usize, usize, usize), GpuError> {
    if hidden <= 0 || vocab <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe lm_head Metal fallback requires positive hidden/vocab, got {hidden}/{vocab}"
        )));
    }
    let group_size = match group_size {
        Some(g) if g > 0 => g as usize,
        Some(g) => {
            return Err(GpuError::InvalidArg(format!(
                "qwen3_moe INT4 lm_head Metal fallback requires positive group_size, got {g}"
            )));
        }
        None => 0,
    };
    Ok((hidden as usize, vocab as usize, group_size))
}

fn checked_positive(name: &str, value: i32) -> Result<usize, GpuError> {
    if value <= 0 {
        return Err(GpuError::InvalidArg(format!(
            "qwen3_moe::decode_layer_launch: Metal fallback requires positive {name}, got {value}"
        )));
    }
    Ok(value as usize)
}

fn require_ptrs(
    _layer: &Qwen3MoeDecodeLayerDesc,
    _int4: &Qwen3MoeInt4ScaleDesc,
    ptrs: &[*const c_void],
) -> Result<(), GpuError> {
    if ptrs.iter().any(|ptr| ptr.is_null()) {
        return Err(GpuError::InvalidArg(
            "qwen3_moe::decode_layer_launch: Metal fallback requires all layer and INT4 pointers"
                .into(),
        ));
    }
    Ok(())
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let mut bits = x.to_bits();
    let bias = 0x7FFFu32 + ((bits >> 16) & 1);
    bits = bits.wrapping_add(bias) & 0xFFFF_0000;
    (bits >> 16) as u16
}

fn bf16_round_f32(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

fn q3_rms_norm_host(input: &[f32], weight: *const u16, eps: f32, output: &mut [f32]) {
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inv = 1.0f32 / (mean_sq + eps).sqrt();
    let weight = unsafe { std::slice::from_raw_parts(weight, input.len()) };
    for (i, out) in output.iter_mut().enumerate() {
        *out = bf16_round_f32(input[i] * inv * bf16_bits_to_f32(weight[i]));
    }
}

fn q3_head_rms_norm_host(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: *const u16,
    eps: f32,
) {
    let weight = unsafe { std::slice::from_raw_parts(weight, head_dim) };
    for head in 0..heads {
        let base = head * head_dim;
        let mean_sq = x[base..base + head_dim].iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0f32 / (mean_sq + eps).sqrt();
        for i in 0..head_dim {
            x[base + i] = bf16_round_f32(x[base + i] * inv * bf16_bits_to_f32(weight[i]));
        }
    }
}

fn q3_rope_half_host(x: &mut [f32], heads: usize, head_dim: usize, position: i32, theta: f32) {
    let half = head_dim / 2;
    let theta_log = theta.ln();
    for head in 0..heads {
        let base = head * head_dim;
        for i in 0..half {
            let a = x[base + i];
            let b = x[base + half + i];
            let freq = position as f32 * (-(i as f32 / half as f32) * theta_log).exp();
            let c = bf16_round_f32(freq.cos());
            let s = bf16_round_f32(freq.sin());
            x[base + i] = bf16_round_f32(bf16_round_f32(a * c) - bf16_round_f32(b * s));
            x[base + half + i] = bf16_round_f32(bf16_round_f32(b * c) + bf16_round_f32(a * s));
        }
    }
}

fn q3_int4_value(
    packed: &[u8],
    scale: &[u16],
    zero: &[u16],
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> f32 {
    let _ = rows;
    let byte_cols = cols.div_ceil(2);
    let byte = packed[row * byte_cols + col / 2];
    let nibble = if col & 1 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    let scale_cols = cols.div_ceil(group_size);
    let scale_idx = (row / group_size) * scale_cols + col / group_size;
    let s = bf16_bits_to_f32(scale[scale_idx]);
    let z = bf16_bits_to_f32(zero[scale_idx]);
    bf16_round_f32(nibble as f32 * s - z * s)
}

fn q3_int4_matvec_host(
    weight: *const u8,
    scale: *const u16,
    zero: *const u16,
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let packed = unsafe { std::slice::from_raw_parts(weight, rows * cols.div_ceil(2)) };
    let scale_len = rows.div_ceil(group_size) * cols.div_ceil(group_size);
    let scale = unsafe { std::slice::from_raw_parts(scale, scale_len) };
    let zero = unsafe { std::slice::from_raw_parts(zero, scale_len) };
    for row in 0..rows {
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += q3_int4_value(packed, scale, zero, row, col, rows, cols, group_size) * x[col];
        }
        out[row] = bf16_round_f32(acc);
    }
}

fn q3_bf16_matvec_host(weight: *const u16, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let weight = unsafe { std::slice::from_raw_parts(weight, rows * cols) };
    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_base = row * cols;
        for col in 0..cols {
            acc += bf16_bits_to_f32(weight[row_base + col]) * x[col];
        }
        out[row] = bf16_round_f32(acc);
    }
}

fn q3_expert_int4_value(
    packed: &[u8],
    scale: &[u16],
    zero: &[u16],
    expert: usize,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> f32 {
    let byte_cols = cols.div_ceil(2);
    let packed_base = (expert * rows + row) * byte_cols;
    let byte = packed[packed_base + col / 2];
    let nibble = if col & 1 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    };
    let scale_rows = rows.div_ceil(group_size);
    let scale_cols = cols.div_ceil(group_size);
    let scale_idx = (expert * scale_rows + row / group_size) * scale_cols + col / group_size;
    let s = bf16_bits_to_f32(scale[scale_idx]);
    let z = bf16_bits_to_f32(zero[scale_idx]);
    bf16_round_f32(nibble as f32 * s - z * s)
}

fn q3_expert_int4_matvec_host(
    weight: *const u8,
    scale: *const u16,
    zero: *const u16,
    expert: usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let expert_count = expert + 1;
    let packed =
        unsafe { std::slice::from_raw_parts(weight, expert_count * rows * cols.div_ceil(2)) };
    let scale_len = expert_count * rows.div_ceil(group_size) * cols.div_ceil(group_size);
    let scale = unsafe { std::slice::from_raw_parts(scale, scale_len) };
    let zero = unsafe { std::slice::from_raw_parts(zero, scale_len) };
    for row in 0..rows {
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += q3_expert_int4_value(
                packed, scale, zero, expert, row, col, rows, cols, group_size,
            ) * x[col];
        }
        out[row] = bf16_round_f32(acc);
    }
}

fn q3_router_topk_host(
    router: &[f32],
    top_k: usize,
    norm_topk_prob: bool,
) -> (Vec<f32>, Vec<usize>) {
    let experts = router.len();
    let mut probs = vec![0.0f32; experts];
    let mut vals = vec![0.0f32; top_k];
    let mut idxs = vec![0usize; top_k];
    if norm_topk_prob {
        let mut scratch = router.to_vec();
        for kpos in 0..top_k {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (idx, &value) in scratch.iter().enumerate() {
                if value > best_v {
                    best_i = idx;
                    best_v = value;
                }
            }
            idxs[kpos] = best_i;
            vals[kpos] = best_v;
            scratch[best_i] = f32::NEG_INFINITY;
        }
        let maxv = vals[0];
        let mut denom = 0.0f32;
        for expert in 0..experts {
            probs[expert] = (router[expert] - maxv).exp();
            denom += probs[expert];
        }
        for kpos in 0..top_k {
            vals[kpos] = bf16_round_f32(probs[idxs[kpos]] / denom);
        }
        let sum = vals.iter().sum::<f32>();
        for val in vals.iter_mut() {
            *val = bf16_round_f32(*val / sum);
        }
    } else {
        let maxv = router
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
        let mut denom = 0.0f32;
        for expert in 0..experts {
            probs[expert] = (router[expert] - maxv).exp();
            denom += probs[expert];
        }
        for prob in probs.iter_mut() {
            *prob = bf16_round_f32(*prob / denom);
        }
        for kpos in 0..top_k {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (idx, &value) in probs.iter().enumerate() {
                if value > best_v {
                    best_i = idx;
                    best_v = value;
                }
            }
            idxs[kpos] = best_i;
            vals[kpos] = best_v;
            probs[best_i] = f32::NEG_INFINITY;
        }
    }
    (vals, idxs)
}

fn lm_head_launch_metal_bf16(
    ordinal: usize,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    logits: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let (hidden, vocab, _) = checked_lm_head_dims(hidden, vocab, None)?;
    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden])?;
    crate::prefill_ffi::rms_norm_rows_plain(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        rms_norm_eps,
        final_hidden,
        final_norm_w,
        &mut normed,
    )?;
    crate::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::BF16,
        1,
        1,
        vocab,
        hidden,
        &normed,
        lm_head_w,
        logits,
    )
}

#[allow(clippy::too_many_arguments)]
fn lm_head_int4_launch_metal_bf16(
    ordinal: usize,
    hidden: i32,
    vocab: i32,
    rms_norm_eps: f32,
    final_hidden: &GpuBuffer,
    final_norm_w: &GpuBuffer,
    lm_head_w: &GpuBuffer,
    lm_head_scale: &GpuBuffer,
    lm_head_zero: &GpuBuffer,
    group_size: i32,
    logits: &mut GpuBuffer,
) -> Result<(), GpuError> {
    let (hidden, vocab, group_size) = checked_lm_head_dims(hidden, vocab, Some(group_size))?;
    let mut normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden])?;
    crate::prefill_ffi::rms_norm_rows_plain(
        ordinal,
        ScalarType::BF16,
        1,
        hidden,
        rms_norm_eps,
        final_hidden,
        final_norm_w,
        &mut normed,
    )?;
    crate::prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        1,
        vocab,
        hidden,
        &normed,
        lm_head_w,
        lm_head_scale,
        lm_head_zero,
        None,
        group_size,
        LOWBIT_NATIVE_INT4,
        logits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_sizes_match_bridge_static_asserts() {
        assert_eq!(std::mem::size_of::<Qwen3MoeDecodeLayerDesc>(), 168);
        assert_eq!(std::mem::size_of::<Qwen3MoeInt4ScaleDesc>(), 104);
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn read_bf16(buffer: &GpuBuffer) -> Vec<f32> {
        let bytes = buffer.to_host_bytes().expect("download bf16");
        bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_bf16(ordinal: usize, shape: &[usize], values: &[f32]) -> GpuBuffer {
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, shape, &bf16_bytes(values))
            .expect("upload bf16")
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn pack_int4_rows(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            for pair in row.chunks(2) {
                let lo = pair[0] & 0x0f;
                let hi = pair.get(1).copied().unwrap_or(0) & 0x0f;
                out.push(lo | (hi << 4));
            }
        }
        out
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_int4_matrix(
        ordinal: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        nibbles: Vec<Vec<u8>>,
        scale: f32,
    ) -> (GpuBuffer, GpuBuffer, GpuBuffer) {
        let packed = pack_int4_rows(&nibbles);
        let weight =
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[rows, cols.div_ceil(2)], &packed)
                .expect("upload int4 matrix");
        let scale_len = rows.div_ceil(group_size) * cols.div_ceil(group_size);
        let scale_vals = vec![scale; scale_len];
        let zero_vals = vec![0.0; scale_len];
        (
            weight,
            upload_bf16(ordinal, &[scale_len], &scale_vals),
            upload_bf16(ordinal, &[scale_len], &zero_vals),
        )
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn upload_int4_experts(
        ordinal: usize,
        experts: usize,
        rows: usize,
        cols: usize,
        group_size: usize,
        nibbles: Vec<Vec<Vec<u8>>>,
        scale: f32,
    ) -> (GpuBuffer, GpuBuffer, GpuBuffer) {
        let mut flat_rows = Vec::new();
        for expert in nibbles {
            for row in expert {
                flat_rows.push(row);
            }
        }
        let packed = pack_int4_rows(&flat_rows);
        let weight = GpuBuffer::from_host_bytes(
            ordinal,
            ScalarType::U8,
            &[experts, rows, cols.div_ceil(2)],
            &packed,
        )
        .expect("upload int4 experts");
        let scale_len = experts * rows.div_ceil(group_size) * cols.div_ceil(group_size);
        let scale_vals = vec![scale; scale_len];
        let zero_vals = vec![0.0; scale_len];
        (
            weight,
            upload_bf16(ordinal, &[scale_len], &scale_vals),
            upload_bf16(ordinal, &[scale_len], &zero_vals),
        )
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    fn rms_norm_reference(hidden: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / hidden.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        hidden
            .iter()
            .zip(weight.iter())
            .map(|(h, w)| half::bf16::from_f32(h * inv * w).to_f32())
            .collect()
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_lm_head_bf16_fallback_runs() {
        gpu_hal::set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [1.0, 1.0, 1.0, 1.0];
        let lm_vals = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        let hidden = upload_bf16(ordinal, &[4], &hidden_vals);
        let norm = upload_bf16(ordinal, &[4], &norm_vals);
        let lm = upload_bf16(ordinal, &[4, 4], &lm_vals);
        let mut logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("alloc logits");
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");

        lm_head_launch(
            ordinal,
            ScalarType::BF16,
            4,
            4,
            0.0,
            &hidden,
            &norm,
            &lm,
            &mut logits,
            &mut counter,
        )
        .expect("run Qwen3-MoE Metal BF16 lm_head fallback");

        let actual = read_bf16(&logits);
        let expected = rms_norm_reference(&hidden_vals, &norm_vals, 0.0);
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= 0.01, "idx {idx}: expected {e}, got {a}");
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_lm_head_int4_fallback_runs() {
        gpu_hal::set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden_vals = [1.0, 0.5, -1.0, 2.0];
        let norm_vals = [1.0, 1.0, 1.0, 1.0];
        let hidden = upload_bf16(ordinal, &[4], &hidden_vals);
        let norm = upload_bf16(ordinal, &[4], &norm_vals);

        let nibbles: [[u8; 4]; 4] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]];
        let mut rhs_bytes = Vec::with_capacity(8);
        for row in &nibbles {
            rhs_bytes.push(row[0] | (row[1] << 4));
            rhs_bytes.push(row[2] | (row[3] << 4));
        }
        let lm_int4 = GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[1, 4, 2], &rhs_bytes)
            .expect("upload int4 lm_head");
        let scale_vals = [0.5, 0.25, 0.125, 1.0];
        let zero_vals = [2.0, 1.0, 4.0, 0.5];
        let scale = upload_bf16(ordinal, &[2, 2], &scale_vals);
        let zero = upload_bf16(ordinal, &[2, 2], &zero_vals);
        let mut logits = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[4]).expect("alloc logits");
        let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).expect("alloc counter");

        lm_head_int4_launch(
            ordinal,
            ScalarType::BF16,
            4,
            4,
            0.0,
            &hidden,
            &norm,
            &lm_int4,
            &scale,
            &zero,
            2,
            &mut logits,
            &mut counter,
        )
        .expect("run Qwen3-MoE Metal INT4 lm_head fallback");

        let normed = rms_norm_reference(&hidden_vals, &norm_vals, 0.0);
        let bf16_round = |x: f32| -> f32 { half::bf16::from_f32(x).to_f32() };
        let mut expected = [0.0f32; 4];
        for col in 0..4usize {
            let scale_row = col / 2;
            let mut acc = 0.0f32;
            for kk in 0..4usize {
                let si = scale_row * 2 + (kk / 2);
                let w = bf16_round(
                    nibbles[col][kk] as f32 * scale_vals[si] - zero_vals[si] * scale_vals[si],
                );
                acc += normed[kk] * w;
            }
            expected[col] = acc;
        }

        let actual = read_bf16(&logits);
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= 0.05, "idx {idx}: expected {e}, got {a}");
        }
    }

    #[cfg(all(target_os = "macos", supersonic_backend_metal))]
    #[test]
    fn metal_decode_layer_int4_fallback_runs_and_updates_kv() {
        gpu_hal::set_backend(Backend::Metal);
        let ordinal = 0usize;
        let hidden = 4usize;
        let head_dim = 2usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let experts = 2usize;
        let top_k = 1usize;
        let moe_i = 2usize;
        let group_size = 2usize;

        let input = upload_bf16(ordinal, &[hidden], &[1.0, -0.5, 0.25, 0.75]);
        let input_norm = upload_bf16(ordinal, &[hidden], &[1.0, 1.0, 1.0, 1.0]);
        let post_norm = upload_bf16(ordinal, &[hidden], &[1.0, 1.0, 1.0, 1.0]);
        let q_norm = upload_bf16(ordinal, &[head_dim], &[1.0, 1.0]);
        let k_norm = upload_bf16(ordinal, &[head_dim], &[1.0, 1.0]);
        let router = upload_bf16(
            ordinal,
            &[experts, hidden],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );

        let (q_w, q_s, q_z) = upload_int4_matrix(
            ordinal,
            q_dim,
            hidden,
            group_size,
            vec![
                vec![4, 0, 0, 0],
                vec![0, 4, 0, 0],
                vec![0, 0, 4, 0],
                vec![0, 0, 0, 4],
            ],
            0.25,
        );
        let (k_w, k_s, k_z) = upload_int4_matrix(
            ordinal,
            kv_dim,
            hidden,
            group_size,
            vec![vec![4, 0, 0, 0], vec![0, 4, 0, 0]],
            0.25,
        );
        let (v_w, v_s, v_z) = upload_int4_matrix(
            ordinal,
            kv_dim,
            hidden,
            group_size,
            vec![vec![0, 0, 4, 0], vec![0, 0, 0, 4]],
            0.25,
        );
        let (o_w, o_s, o_z) = upload_int4_matrix(
            ordinal,
            hidden,
            q_dim,
            group_size,
            vec![
                vec![4, 0, 0, 0],
                vec![0, 4, 0, 0],
                vec![0, 0, 4, 0],
                vec![0, 0, 0, 4],
            ],
            0.25,
        );
        let (gate_up_w, gate_up_s, gate_up_z) = upload_int4_experts(
            ordinal,
            experts,
            2 * moe_i,
            hidden,
            group_size,
            vec![
                vec![
                    vec![4, 0, 0, 0],
                    vec![0, 4, 0, 0],
                    vec![0, 0, 4, 0],
                    vec![0, 0, 0, 4],
                ],
                vec![
                    vec![0, 4, 0, 0],
                    vec![4, 0, 0, 0],
                    vec![0, 0, 0, 4],
                    vec![0, 0, 4, 0],
                ],
            ],
            0.25,
        );
        let (down_w, down_s, down_z) = upload_int4_experts(
            ordinal,
            experts,
            hidden,
            moe_i,
            group_size,
            vec![
                vec![vec![4, 0], vec![0, 4], vec![4, 0], vec![0, 4]],
                vec![vec![0, 4], vec![4, 0], vec![0, 4], vec![4, 0]],
            ],
            0.25,
        );

        let mut kv_k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, kv_dim]).expect("kv k");
        let mut kv_v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[2, kv_dim]).expect("kv v");
        let mut output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[hidden]).expect("output");
        let workspace_len = hidden
            + q_dim
            + kv_dim
            + kv_dim
            + q_dim
            + hidden
            + experts
            + experts
            + top_k
            + top_k
            + 2 * moe_i
            + moe_i
            + hidden
            + hidden;
        let mut workspace =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_len]).expect("workspace");

        let layer = Qwen3MoeDecodeLayerDesc {
            layer_idx: 0,
            input_norm_w: input_norm.as_ptr(),
            input_norm_eps: 1e-5,
            post_attn_norm_w: post_norm.as_ptr(),
            post_attn_norm_eps: 1e-5,
            q_proj_w: q_w.as_ptr(),
            k_proj_w: k_w.as_ptr(),
            v_proj_w: v_w.as_ptr(),
            o_proj_w: o_w.as_ptr(),
            q_norm_w: q_norm.as_ptr(),
            k_norm_w: k_norm.as_ptr(),
            rope_theta: 10_000.0,
            head_dim: head_dim as i32,
            num_heads: num_heads as i32,
            num_kv_heads: num_kv_heads as i32,
            kv_cache_k: kv_k.as_mut_ptr(),
            kv_cache_v: kv_v.as_mut_ptr(),
            kv_len: 0,
            kv_max_t: 2,
            router_w: router.as_ptr(),
            experts_gate_up_w: gate_up_w.as_ptr(),
            experts_down_w: down_w.as_ptr(),
            num_experts: experts as i32,
            top_k: top_k as i32,
            moe_intermediate_size: moe_i as i32,
            norm_topk_prob: 1,
        };
        let int4 = Qwen3MoeInt4ScaleDesc {
            q_proj_scale: q_s.as_ptr(),
            q_proj_zero: q_z.as_ptr(),
            k_proj_scale: k_s.as_ptr(),
            k_proj_zero: k_z.as_ptr(),
            v_proj_scale: v_s.as_ptr(),
            v_proj_zero: v_z.as_ptr(),
            o_proj_scale: o_s.as_ptr(),
            o_proj_zero: o_z.as_ptr(),
            experts_gate_up_scale: gate_up_s.as_ptr(),
            experts_gate_up_zero: gate_up_z.as_ptr(),
            experts_down_scale: down_s.as_ptr(),
            experts_down_zero: down_z.as_ptr(),
            group_size: group_size as i32,
        };

        decode_layer_launch(
            ordinal,
            ScalarType::BF16,
            &layer,
            &int4,
            hidden as i32,
            0,
            &input,
            &mut output,
            &mut workspace,
        )
        .expect("run Qwen3-MoE Metal decode-layer INT4 fallback");

        let actual = read_bf16(&output);
        assert!(
            actual.iter().all(|v| v.is_finite()) && actual.iter().any(|v| v.abs() > 0.001),
            "decode output should be finite and non-zero: {actual:?}"
        );
        let cached_k = read_bf16(&kv_k);
        let cached_v = read_bf16(&kv_v);
        assert!(
            cached_k.iter().take(kv_dim).any(|v| v.abs() > 0.001)
                && cached_v.iter().take(kv_dim).any(|v| v.abs() > 0.001),
            "KV cache should be updated: k={cached_k:?} v={cached_v:?}"
        );
    }
}
