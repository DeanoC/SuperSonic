use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use base64::Engine as _;
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::desc_builder::{build_layer_descs, build_fp8_scale_descs, build_int4_scale_descs, build_kv_fp8_descs, build_batch_seq_descs};
use qwen35::config::TextConfig;
use qwen35::rotary::RotaryTables;
use qwen35::scratch::{
    PersistentDecodeScratch, PERSISTENT_4B_TIMING_SLOTS_PER_LAYER, PERSISTENT_SYNC_COUNTER_BYTES,
};
use qwen35::state::{kv_fp8_bf16_sidecar_enabled, ModelState};
use qwen35::weights::Qwen35Weights;

use crate::oracle::OracleOutput;
use crate::prefill_engine;
use crate::decode_f32_le;

fn matmul_proj(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    scale: Option<&GpuBuffer>,
    block_size: usize,
    out: &mut GpuBuffer,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<()> {
    if let (Some(sc), Some(zr)) = (int4_scale, int4_zero) {
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
            ordinal, batch, m, n, k, lhs, weight, sc, zr, int4_group_size, out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
    } else {
        match scale {
            Some(s) => kernel_ffi::prefill_ffi::matmul_rhs_transposed_fp8(
                ordinal, batch, m, n, k, lhs, weight, s, block_size, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_fp8: {e}")),
            None => kernel_ffi::prefill_ffi::matmul_rhs_transposed(
                ordinal, ScalarType::BF16, batch, m, n, k, lhs, weight, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul: {e}")),
        }
    }
}

fn residual_add(
    ordinal: usize,
    total_elems: usize,
    dst: &mut GpuBuffer,
    src: &GpuBuffer,
) -> Result<()> {
    let lhs: &GpuBuffer = unsafe { &*(dst as *const GpuBuffer) };
    kernel_ffi::prefill_ffi::element_add(
        ordinal,
        ScalarType::BF16,
        total_elems,
        lhs,
        src,
        dst,
    )
    .map_err(|e| anyhow::anyhow!("residual_add failed: {e}"))?;
    Ok(())
}

fn fp8_e4m3_to_f32_host(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;
    if byte == 0x7F || byte == 0xFF {
        return 0.0;
    }
    let val = if exp == 0 {
        mantissa as f32 / 8.0 * 1.52587890625e-2
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign != 0 { -val } else { val }
}

fn f32_to_bf16_bytes_host(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

pub struct DecodeEngine {
    weights: Qwen35Weights,
    state: ModelState,
    /// Extra model states for batch items 1..batch_size-1.
    extra_states: Vec<ModelState>,
    scratch: PersistentDecodeScratch,
    rotary: RotaryTables,
    hidden_io: GpuBuffer,
    normed_buf: GpuBuffer,
    logits_buf: GpuBuffer,
    argmax_buf: GpuBuffer,
    lm_head_block_best_vals: GpuBuffer,
    lm_head_block_best_idxs: GpuBuffer,
    matvec_counter: GpuBuffer,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
    /// FP8 scale descriptors on GPU (None for BF16 weights).
    fp8_scale_device: Option<GpuBuffer>,
    /// INT4 scale descriptors on GPU (None for non-INT4 weights).
    int4_scale_device: Option<GpuBuffer>,
    /// Prefill chunk size (0 = no chunking).
    prefill_chunk_size: usize,
    /// Use FP8 E4M3 KV cache with dynamic per-head scaling.
    kv_fp8: bool,
    /// Batch size (1 = single-sequence, default).
    batch_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeSamplingMode {
    HostLogits,
    CudaFastGreedy,
    CudaHeroFusedLmHead,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeStageTimings {
    pub persistent_ms: f64,
    pub rms_norm_ms: f64,
    pub lm_head_ms: f64,
    pub logits_d2h_ms: f64,
    pub host_sampling_ms: f64,
    pub gpu_argmax_ms: f64,
    pub token_d2h_ms: f64,
    pub persistent_full_attn_ms: f64,
    pub persistent_full_attn_proj_ms: f64,
    pub persistent_full_attn_core_ms: f64,
    pub persistent_full_attn_out_ms: f64,
    pub persistent_linear_proj_ms: f64,
    pub persistent_linear_core_ms: f64,
    pub persistent_linear_core_conv_ms: f64,
    pub persistent_linear_core_recurrent_ms: f64,
    pub persistent_linear_core_post_ms: f64,
    pub persistent_linear_out_ms: f64,
    pub persistent_mlp_gate_up_ms: f64,
    pub persistent_mlp_down_ms: f64,
}

impl DecodeStageTimings {
    pub fn add_assign(&mut self, rhs: Self) {
        self.persistent_ms += rhs.persistent_ms;
        self.rms_norm_ms += rhs.rms_norm_ms;
        self.lm_head_ms += rhs.lm_head_ms;
        self.logits_d2h_ms += rhs.logits_d2h_ms;
        self.host_sampling_ms += rhs.host_sampling_ms;
        self.gpu_argmax_ms += rhs.gpu_argmax_ms;
        self.token_d2h_ms += rhs.token_d2h_ms;
        self.persistent_full_attn_ms += rhs.persistent_full_attn_ms;
        self.persistent_full_attn_proj_ms += rhs.persistent_full_attn_proj_ms;
        self.persistent_full_attn_core_ms += rhs.persistent_full_attn_core_ms;
        self.persistent_full_attn_out_ms += rhs.persistent_full_attn_out_ms;
        self.persistent_linear_proj_ms += rhs.persistent_linear_proj_ms;
        self.persistent_linear_core_ms += rhs.persistent_linear_core_ms;
        self.persistent_linear_core_conv_ms += rhs.persistent_linear_core_conv_ms;
        self.persistent_linear_core_recurrent_ms += rhs.persistent_linear_core_recurrent_ms;
        self.persistent_linear_core_post_ms += rhs.persistent_linear_core_post_ms;
        self.persistent_linear_out_ms += rhs.persistent_linear_out_ms;
        self.persistent_mlp_gate_up_ms += rhs.persistent_mlp_gate_up_ms;
        self.persistent_mlp_down_ms += rhs.persistent_mlp_down_ms;
    }

    pub fn total_ms(&self) -> f64 {
        self.persistent_ms
            + self.rms_norm_ms
            + self.lm_head_ms
            + self.logits_d2h_ms
            + self.host_sampling_ms
            + self.gpu_argmax_ms
            + self.token_d2h_ms
    }
}

pub struct DecodeStepOutput {
    pub logits: Option<Vec<f32>>,
    pub sampled_token: u32,
    pub timings: DecodeStageTimings,
}

pub struct ComponentLayerTrace {
    pub attn_hidden: Vec<u8>,
    pub post_attn_norm: Vec<u8>,
    pub mlp_swiglu: Vec<u8>,
    pub mlp_out: Vec<u8>,
    pub layer_hidden: Vec<u8>,
}

pub struct ComponentMlpTrace {
    pub swiglu: Vec<u8>,
    pub down: Vec<u8>,
}

pub struct ComponentLinearTrace {
    pub qkv: Vec<u8>,
    pub z: Vec<u8>,
    pub b: Vec<u8>,
    pub a: Vec<u8>,
    pub packed: Vec<u8>,
    pub rec_apply: Vec<u8>,
    pub attn: Vec<u8>,
    pub gated: Vec<u8>,
    pub proj_out: Vec<u8>,
}

pub struct FullAttentionStageTrace {
    pub normed: Vec<u8>,
    pub q_proj: Vec<u8>,
    pub gate_proj: Vec<u8>,
    pub k_proj: Vec<u8>,
    pub v_proj: Vec<u8>,
    pub q_rope: Vec<u8>,
    pub k_rope: Vec<u8>,
}

pub struct FullAttentionLayerOutputTrace {
    pub pre_gate: Vec<u8>,
    pub gated: Vec<u8>,
    pub attn_hidden: Vec<u8>,
}

const PERSISTENT_4B_TIMING_FULL_ATTN: usize = 0;
const PERSISTENT_4B_TIMING_FULL_ATTN_PROJ: usize = 1;
const PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE: usize = 2;
const PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE: usize = 10;
const PERSISTENT_4B_TIMING_LINEAR_PROJ: usize = 18;
const PERSISTENT_4B_TIMING_LINEAR_CORE_BASE: usize = 19;
const PERSISTENT_4B_TIMING_LINEAR_OUT_BASE: usize = 27;
const PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE: usize = 29;
const PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE: usize = 31;
const PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE: usize = 33;
const PERSISTENT_4B_TIMING_MLP_GATE_UP: usize = 35;
const PERSISTENT_4B_TIMING_MLP_DOWN: usize = 36;

fn persistent_4b_clock_cycles_to_ms(cycles: u64, clock_rate_khz: u32) -> f64 {
    if cycles == 0 || clock_rate_khz == 0 {
        0.0
    } else {
        cycles as f64 / clock_rate_khz as f64
    }
}

fn decode_persistent_4b_timing_slots(
    sync_bytes: &[u8],
    num_layers: usize,
    batch_size: usize,
    clock_rate_khz: u32,
) -> DecodeStageTimings {
    let timing_bytes =
        num_layers * PERSISTENT_4B_TIMING_SLOTS_PER_LAYER * std::mem::size_of::<u64>();
    let start = PERSISTENT_SYNC_COUNTER_BYTES;
    let end = start + timing_bytes;
    if sync_bytes.len() < end {
        return DecodeStageTimings::default();
    }

    let load_slot = |idx: usize| -> u64 {
        let byte_start = start + idx * std::mem::size_of::<u64>();
        let byte_end = byte_start + std::mem::size_of::<u64>();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&sync_bytes[byte_start..byte_end]);
        u64::from_le_bytes(raw)
    };

    let mut full_attn_cycles = 0u64;
    let mut full_attn_proj_cycles = 0u64;
    let mut full_attn_core_cycles = 0u64;
    let mut full_attn_out_cycles = 0u64;
    let mut linear_proj_cycles = 0u64;
    let mut linear_core_cycles = 0u64;
    let mut linear_core_conv_cycles = 0u64;
    let mut linear_core_recurrent_cycles = 0u64;
    let mut linear_core_post_cycles = 0u64;
    let mut linear_out_cycles = 0u64;
    let mut mlp_gate_up_cycles = 0u64;
    let mut mlp_down_cycles = 0u64;
    let section_batches = batch_size.min(8);
    let split_batches = batch_size.min(2);
    for layer in 0..num_layers {
        let layer_base = layer * PERSISTENT_4B_TIMING_SLOTS_PER_LAYER;
        full_attn_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN);
        full_attn_proj_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_PROJ);
        linear_proj_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_PROJ);
        mlp_gate_up_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_MLP_GATE_UP);
        mlp_down_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_MLP_DOWN);
        for b in 0..section_batches {
            full_attn_core_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE + b);
            full_attn_out_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE + b);
            linear_core_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_BASE + b);
            linear_out_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_OUT_BASE + b);
        }
        for b in 0..split_batches {
            linear_core_conv_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE + b);
            linear_core_recurrent_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE + b);
            linear_core_post_cycles +=
                load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE + b);
        }
    }

    DecodeStageTimings {
        persistent_full_attn_ms: persistent_4b_clock_cycles_to_ms(
            full_attn_cycles,
            clock_rate_khz,
        ),
        persistent_full_attn_proj_ms: persistent_4b_clock_cycles_to_ms(
            full_attn_proj_cycles,
            clock_rate_khz,
        ),
        persistent_full_attn_core_ms: persistent_4b_clock_cycles_to_ms(
            full_attn_core_cycles,
            clock_rate_khz,
        ),
        persistent_full_attn_out_ms: persistent_4b_clock_cycles_to_ms(
            full_attn_out_cycles,
            clock_rate_khz,
        ),
        persistent_linear_proj_ms: persistent_4b_clock_cycles_to_ms(
            linear_proj_cycles,
            clock_rate_khz,
        ),
        persistent_linear_core_ms: persistent_4b_clock_cycles_to_ms(
            linear_core_cycles,
            clock_rate_khz,
        ),
        persistent_linear_core_conv_ms: persistent_4b_clock_cycles_to_ms(
            linear_core_conv_cycles,
            clock_rate_khz,
        ),
        persistent_linear_core_recurrent_ms: persistent_4b_clock_cycles_to_ms(
            linear_core_recurrent_cycles,
            clock_rate_khz,
        ),
        persistent_linear_core_post_ms: persistent_4b_clock_cycles_to_ms(
            linear_core_post_cycles,
            clock_rate_khz,
        ),
        persistent_linear_out_ms: persistent_4b_clock_cycles_to_ms(
            linear_out_cycles,
            clock_rate_khz,
        ),
        persistent_mlp_gate_up_ms: persistent_4b_clock_cycles_to_ms(
            mlp_gate_up_cycles,
            clock_rate_khz,
        ),
        persistent_mlp_down_ms: persistent_4b_clock_cycles_to_ms(
            mlp_down_cycles,
            clock_rate_khz,
        ),
        ..DecodeStageTimings::default()
    }
}

impl DecodeEngine {
    // Rebuild the BF16 sidecar from the current prefix cache when a KV-FP8 state
    // grows after prefill or is cloned for batched decode.
    fn load_kv_shadow_for_state_static(
        config: &TextConfig,
        ordinal: usize,
        state: &mut ModelState,
    ) -> Result<()> {
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        for layer_idx in 0..state.layers.len() {
            if !config.is_full_attention(layer_idx) {
                continue;
            }
            let should_populate = {
                let ls = &state.layers[layer_idx];
                ls.kv_shadow_k.is_some()
                    && ls.kv_shadow_v.is_some()
                    && ls.kv_shadow_start == ls.kv_filled
            };
            if !should_populate {
                continue;
            }

            let (prefix_k_host, prefix_v_host, prefix_len) =
                Self::assemble_full_attention_prefix_cache_bf16_host_static(config, state, layer_idx)?;
            if prefix_len == 0 {
                state.layers[layer_idx].kv_shadow_start = 0;
                continue;
            }

            let ls = &mut state.layers[layer_idx];
            let shadow_k = ls
                .kv_shadow_k
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K shadow"))?;
            let shadow_v = ls
                .kv_shadow_v
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V shadow"))?;
            let cap = shadow_k.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let contig_stride = prefix_len * head_dim * elem_bytes;

            let tmp_k = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_k_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow K H2D: {e}"))?;
            let tmp_v = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_v_host,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow V H2D: {e}"))?;

            for h in 0..num_kv_heads {
                gpu_hal::copy_d2d(
                    ordinal,
                    shadow_k.offset_ptr(h * cap_stride) as *mut c_void,
                    tmp_k.offset_ptr(h * contig_stride),
                    contig_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow K copy h={h}: {e}"))?;
                gpu_hal::copy_d2d(
                    ordinal,
                    shadow_v.offset_ptr(h * cap_stride) as *mut c_void,
                    tmp_v.offset_ptr(h * contig_stride),
                    contig_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow V copy h={h}: {e}"))?;
            }
            ls.kv_shadow_start = 0;
        }

        Ok(())
    }

    fn assemble_full_attention_prefix_cache_bf16_host_static(
        config: &TextConfig,
        state: &ModelState,
        layer_idx: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        let ls = state
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let prefix_len = ls.kv_filled;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let mut out_k = vec![0u8; num_kv_heads * prefix_len * head_dim * elem_bytes];
        let mut out_v = vec![0u8; num_kv_heads * prefix_len * head_dim * elem_bytes];
        if prefix_len == 0 {
            return Ok((out_k, out_v, prefix_len));
        }

        let cache_k = ls
            .kv_cache_k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
        let cache_v = ls
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
        let cap = cache_k.shape()[2];

        if let (Some(scale_k), Some(scale_v)) = (ls.kv_scale_k.as_ref(), ls.kv_scale_v.as_ref()) {
            let k_bytes = cache_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 K cache D2H: {e}"))?;
            let v_bytes = cache_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 V cache D2H: {e}"))?;
            let k_scales = decode_f32_le(
                &scale_k
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 K scale D2H: {e}"))?,
            );
            let v_scales = decode_f32_le(
                &scale_v
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {layer_idx} fp8 V scale D2H: {e}"))?,
            );

            let mut deq_k = Vec::with_capacity(num_kv_heads * prefix_len * head_dim);
            let mut deq_v = Vec::with_capacity(num_kv_heads * prefix_len * head_dim);
            for h in 0..num_kv_heads {
                for t in 0..prefix_len {
                    let scale_k_val = k_scales[h * cap + t];
                    let scale_v_val = v_scales[h * cap + t];
                    let base = (h * cap + t) * head_dim;
                    for d in 0..head_dim {
                        deq_k.push(fp8_e4m3_to_f32_host(k_bytes[base + d]) * scale_k_val);
                        deq_v.push(fp8_e4m3_to_f32_host(v_bytes[base + d]) * scale_v_val);
                    }
                }
            }
            out_k = f32_to_bf16_bytes_host(deq_k);
            out_v = f32_to_bf16_bytes_host(deq_v);
        } else {
            let k_bytes = cache_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} BF16 K cache D2H: {e}"))?;
            let v_bytes = cache_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} BF16 V cache D2H: {e}"))?;
            let src_head_stride = cap * head_dim * elem_bytes;
            let dst_head_stride = prefix_len * head_dim * elem_bytes;
            let copy_bytes = prefix_len * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                out_k[dst..dst + copy_bytes].copy_from_slice(&k_bytes[src..src + copy_bytes]);
                out_v[dst..dst + copy_bytes].copy_from_slice(&v_bytes[src..src + copy_bytes]);
            }
        }

        Ok((out_k, out_v, prefix_len))
    }

    fn assemble_full_attention_prefix_cache_bf16_host_for_state(
        &self,
        state: &ModelState,
        layer_idx: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        Self::assemble_full_attention_prefix_cache_bf16_host_static(&self.weights.config, state, layer_idx)
    }

    pub fn full_attention_prefix_cache_bf16_host(
        &self,
        layer_idx: usize,
        batch_index: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        let state = self.state_for_batch(batch_index);
        self.assemble_full_attention_prefix_cache_bf16_host_for_state(state, layer_idx)
    }

    pub fn trace_full_attention_stages_from_hidden(
        &self,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionStageTrace> {
        let config = &self.weights.config;
        let fw = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = config.rotary_dim();

        let hidden_buf = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} hidden trace H2D: {e}"))?;
        let mut normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace normed alloc: {e}"))?;
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            hidden_dim,
            config.rms_norm_eps as f32,
            &hidden_buf,
            &self.weights.layers[idx].input_norm_w,
            &mut normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace input rms_norm: {e}"))?;

        let mut q_full = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q_full alloc: {e}"))?;
        let mut query_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace query alloc: {e}"))?;
        let mut gate_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace gate alloc: {e}"))?;
        let mut k_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace k alloc: {e}"))?;
        let mut v_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace v alloc: {e}"))?;
        let mut q_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q_normed alloc: {e}"))?;
        let mut k_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace k_normed alloc: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, q_proj_dim, hidden_dim,
            &normed, &fw.q_proj_w, fw.q_proj_scale.as_ref(), self.weights.fp8_block_size, &mut q_full,
            fw.q_proj_int4_scale.as_ref(), fw.q_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::split_qgate(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            &q_full,
            &mut query_buf,
            &mut gate_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace split qgate: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &normed, &fw.k_proj_w, fw.k_proj_scale.as_ref(), self.weights.fp8_block_size, &mut k_buf,
            fw.k_proj_int4_scale.as_ref(), fw.k_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &normed, &fw.v_proj_w, fw.v_proj_scale.as_ref(), self.weights.fp8_block_size, &mut v_buf,
            fw.v_proj_int4_scale.as_ref(), fw.v_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_q_heads, head_dim, 1e-6,
            &query_buf, &fw.q_norm_w, &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            q_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q norm copy: {e}"))?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_kv_heads, head_dim, 1e-6,
            &k_buf, &fw.k_norm_w, &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace k norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            k_buf.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            kv_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace k norm copy: {e}"))?;

        let q_proj = query_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace q pre-rope D2H: {e}"))?;
        let gate_proj = gate_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace gate pre-rope D2H: {e}"))?;
        let k_proj = k_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace k pre-rope D2H: {e}"))?;
        let v_proj = v_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} trace v D2H: {e}"))?;

        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace k rope: {e}"))?;

        Ok(FullAttentionStageTrace {
            normed: normed
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace normed D2H: {e}"))?,
            q_proj,
            gate_proj,
            k_proj,
            v_proj,
            q_rope: query_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace q rope D2H: {e}"))?,
            k_rope: k_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace k rope D2H: {e}"))?,
        })
    }

    fn trace_full_attention_layer_output_from_hidden_with_state(
        &self,
        state: &ModelState,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionLayerOutputTrace> {
        let config = &self.weights.config;
        let fw = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = config.rotary_dim();
        let kv_len = seqlen_offset + 1;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let hidden_in = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer hidden H2D: {e}"))?;
        let mut normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer normed alloc: {e}"))?;
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            hidden_dim,
            config.rms_norm_eps as f32,
            &hidden_in,
            &self.weights.layers[idx].input_norm_w,
            &mut normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer input rms_norm: {e}"))?;

        let mut q_full = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q_full alloc: {e}"))?;
        let mut query_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer query alloc: {e}"))?;
        let mut gate_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate alloc: {e}"))?;
        let mut k_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k alloc: {e}"))?;
        let mut v_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer v alloc: {e}"))?;
        let mut q_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q_normed alloc: {e}"))?;
        let mut k_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k_normed alloc: {e}"))?;
        let mut attn_q = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_q alloc: {e}"))?;
        let mut step_k = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer step_k alloc: {e}"))?;
        let mut step_v = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer step_v alloc: {e}"))?;
        let mut attn_out_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_out alloc: {e}"))?;
        let mut attn_out_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_out bf16 alloc: {e}"))?;
        let mut attn_flat = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_flat alloc: {e}"))?;
        let mut gated = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gated alloc: {e}"))?;
        let mut proj_out = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer proj_out alloc: {e}"))?;
        let mut hidden_out = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, hidden_dim],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer hidden copy H2D: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, q_proj_dim, hidden_dim,
            &normed, &fw.q_proj_w, fw.q_proj_scale.as_ref(), self.weights.fp8_block_size, &mut q_full,
            fw.q_proj_int4_scale.as_ref(), fw.q_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::split_qgate(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            &q_full,
            &mut query_buf,
            &mut gate_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer split qgate: {e}"))?;
        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &normed, &fw.k_proj_w, fw.k_proj_scale.as_ref(), self.weights.fp8_block_size, &mut k_buf,
            fw.k_proj_int4_scale.as_ref(), fw.k_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &normed, &fw.v_proj_w, fw.v_proj_scale.as_ref(), self.weights.fp8_block_size, &mut v_buf,
            fw.v_proj_int4_scale.as_ref(), fw.v_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_q_heads, head_dim, 1e-6,
            &query_buf, &fw.q_norm_w, &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q norm copy: {e}"))?;
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_kv_heads, head_dim, 1e-6,
            &k_buf, &fw.k_norm_w, &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            k_buf.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            kv_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k norm copy: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k rope: {e}"))?;

        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, head_dim, &query_buf, &mut attn_q,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, &k_buf, &mut step_k,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, &v_buf, &mut step_v,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer v transpose: {e}"))?;

        let (prefix_k_host, prefix_v_host, prefix_len) =
            self.assemble_full_attention_prefix_cache_bf16_host_for_state(state, idx)?;
        anyhow::ensure!(
            prefix_len == seqlen_offset,
            "layer {idx} prefix_len {} != seqlen_offset {}",
            prefix_len,
            seqlen_offset
        );
        let kv_k_contig = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[num_kv_heads, kv_len, head_dim],
            &{
                let mut bytes = vec![0u8; num_kv_heads * kv_len * head_dim * elem_bytes];
                let copy = prefix_k_host.len();
                bytes[..copy].copy_from_slice(&prefix_k_host);
                bytes
            },
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv_k_contig H2D: {e}"))?;
        let kv_v_contig = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[num_kv_heads, kv_len, head_dim],
            &{
                let mut bytes = vec![0u8; num_kv_heads * kv_len * head_dim * elem_bytes];
                let copy = prefix_v_host.len();
                bytes[..copy].copy_from_slice(&prefix_v_host);
                bytes
            },
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv_v_contig H2D: {e}"))?;
        let contig_stride = kv_len * head_dim * elem_bytes;
        let step_stride = head_dim * elem_bytes;
        let dst_offset = seqlen_offset * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                self.ordinal,
                kv_k_contig.offset_ptr(h * contig_stride + dst_offset) as *mut c_void,
                step_k.offset_ptr(h * step_stride),
                step_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv K append h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                self.ordinal,
                kv_v_contig.offset_ptr(h * contig_stride + dst_offset) as *mut c_void,
                step_v.offset_ptr(h * step_stride),
                step_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer kv V append h={h}: {e}"))?;
        }

        kernel_ffi::prefill_ffi::full_attention_prefill(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, num_kv_heads,
            1, kv_len, head_dim, 1.0 / (head_dim as f32).sqrt(), seqlen_offset,
            &attn_q, &kv_k_contig, &kv_v_contig, &mut attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attention: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal, ScalarType::F32, ScalarType::BF16, num_q_heads * head_dim, &attn_out_f32, &mut attn_out_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn cast: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal, ScalarType::BF16, num_q_heads, 1, head_dim, &attn_out_bf16, &mut attn_flat,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn transpose: {e}"))?;
        kernel_ffi::prefill_ffi::sigmoid_mul(
            self.ordinal, ScalarType::BF16, q_dim, &attn_flat, &gate_buf, &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate apply: {e}"))?;
        matmul_proj(
            self.ordinal, 1, 1, hidden_dim, q_dim,
            &gated, &fw.o_proj_w, fw.o_proj_scale.as_ref(), self.weights.fp8_block_size, &mut proj_out,
            fw.o_proj_int4_scale.as_ref(), fw.o_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        residual_add(self.ordinal, hidden_dim, &mut hidden_out, &proj_out)?;
        Ok(FullAttentionLayerOutputTrace {
            pre_gate: attn_flat
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer pre_gate D2H: {e}"))?,
            gated: gated
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gated D2H: {e}"))?,
            attn_hidden: hidden_out
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_hidden D2H: {e}"))?,
        })
    }

    pub fn trace_full_attention_layer_output_from_hidden_current_state(
        &self,
        idx: usize,
        batch_index: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_full_attention_layer_output_from_hidden_with_state(
            self.state_for_batch(batch_index),
            idx,
            hidden_bytes,
            seqlen_offset,
        )
    }

    pub fn trace_full_attention_layer_output_from_hidden_state(
        &self,
        state: &ModelState,
        idx: usize,
        hidden_bytes: &[u8],
        seqlen_offset: usize,
    ) -> Result<FullAttentionLayerOutputTrace> {
        self.trace_full_attention_layer_output_from_hidden_with_state(
            state,
            idx,
            hidden_bytes,
            seqlen_offset,
        )
    }

    pub fn set_hidden_from_bytes(&mut self, hidden_bytes: &[u8]) -> Result<()> {
        let row = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            &[1, self.weights.config.hidden_size],
            hidden_bytes,
        )
        .map_err(|e| anyhow::anyhow!("set hidden row from bytes: {e}"))?;
        let hidden_io = GpuBuffer::zeros(
            self.ordinal,
            ScalarType::BF16,
            &[self.batch_size, 1, self.weights.config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("alloc hidden_io for trace: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            hidden_io.as_ptr() as *mut c_void,
            row.as_ptr(),
            hidden_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("copy hidden trace row into hidden_io: {e}"))?;
        self.hidden_io = hidden_io;
        Ok(())
    }

    pub fn component_trace_linear_layer_from_current_hidden(
        &mut self,
        idx: usize,
    ) -> Result<(ComponentLinearTrace, Vec<u8>, Vec<u8>, Vec<u8>)> {
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            self.weights.config.hidden_size,
            self.weights.config.rms_norm_eps as f32,
            &self.hidden_io,
            &self.weights.layers[idx].input_norm_w,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} component trace input rms_norm: {e}"))?;
        let trace = self
            .component_decode_linear_attention_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear trace output"))?;
        let ls = &self.state.layers[idx];
        let conv = ls
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing conv state after component trace"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component conv D2H: {e}"))?;
        let recurrent = ls
            .recurrent_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing recurrent state after component trace"))?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component recurrent D2H: {e}"))?;
        let hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component hidden D2H: {e}"))?;
        let row_bytes = self.weights.config.hidden_size * ScalarType::BF16.size_in_bytes();
        Ok((trace, conv, recurrent, hidden[..row_bytes].to_vec()))
    }

    pub fn component_trace_full_layer_from_current_hidden(
        &mut self,
        idx: usize,
    ) -> Result<ComponentLayerTrace> {
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            self.weights.config.hidden_size,
            self.weights.config.rms_norm_eps as f32,
            &self.hidden_io,
            &self.weights.layers[idx].input_norm_w,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer input rms_norm: {e}"))?;

        if self.weights.config.is_full_attention(idx) {
            self.component_decode_full_attention_layer(idx, 0)?;
        } else {
            self.component_decode_linear_attention_layer(idx, false)?;
        }
        let row_bytes = self.weights.config.hidden_size * ScalarType::BF16.size_in_bytes();
        let attn_hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer attn hidden D2H: {e}"))?[..row_bytes]
            .to_vec();

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            self.weights.config.hidden_size,
            self.weights.config.rms_norm_eps as f32,
            &self.hidden_io,
            &self.weights.layers[idx].post_attn_norm_w,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer post rms_norm: {e}"))?;
        let post_attn_norm = self
            .normed_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer post norm D2H: {e}"))?[..row_bytes]
            .to_vec();

        let mlp_trace = self
            .component_decode_mlp_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx} component full-layer missing mlp trace"))?;
        let layer_hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component full-layer final hidden D2H: {e}"))?[..row_bytes]
            .to_vec();

        Ok(ComponentLayerTrace {
            attn_hidden,
            post_attn_norm,
            mlp_swiglu: mlp_trace.swiglu,
            mlp_out: mlp_trace.down,
            layer_hidden,
        })
    }

    pub fn full_attention_cache_step_bytes(
        &self,
        layer_idx: usize,
        batch_index: usize,
        seq_pos: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let config = &self.weights.config;
        let ls = self.state_for_batch(batch_index)
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let cache_k = ls.kv_cache_k.as_ref().ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
        let cache_v = ls.kv_cache_v.as_ref().ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let elem_bytes = ScalarType::BF16.size_in_bytes();
        let step_k = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k alloc: {e}"))?;
        let step_v = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v alloc: {e}"))?;

        let cap = cache_k.shape()[2];
        let cap_stride = cap * head_dim * elem_bytes;
        let src_stride = head_dim * elem_bytes;
        let dst_stride = head_dim * elem_bytes;
        let src_offset = seq_pos * head_dim * elem_bytes;
        for h in 0..num_kv_heads {
            gpu_hal::copy_d2d(
                self.ordinal,
                step_k.offset_ptr(h * dst_stride) as *mut c_void,
                cache_k.offset_ptr(h * cap_stride + src_offset),
                src_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k copy h={h}: {e}"))?;
            gpu_hal::copy_d2d(
                self.ordinal,
                step_v.offset_ptr(h * dst_stride) as *mut c_void,
                cache_v.offset_ptr(h * cap_stride + src_offset),
                src_stride,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v copy h={h}: {e}"))?;
        }

        Ok((
            step_k
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_k D2H: {e}"))?,
            step_v
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} trace step_v D2H: {e}"))?,
        ))
    }

    fn component_decode_step_4b_impl(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_input_layer: Option<usize>,
        trace_layer: Option<usize>,
        trace_linear_layer: Option<usize>,
    ) -> Result<(Vec<f32>, Option<Vec<u8>>, Option<ComponentLayerTrace>, Option<ComponentLinearTrace>)> {
        let hidden_dim = self.weights.config.hidden_size;
        let rms_norm_eps = self.weights.config.rms_norm_eps as f32;
        let vocab_size = self.weights.config.vocab_size;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let row_bytes = hidden_dim * elem_bytes;
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        let layer_count = self.state.layers.len();
        let mut traced_hidden = None;
        let mut traced_layer = None;
        let mut traced_linear = None;
        for i in 0..layer_count {
            if trace_input_layer == Some(i) {
                traced_hidden = Some(
                    self.hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} hidden trace D2H: {e}"))?,
                );
            }
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            hidden_dim,
            rms_norm_eps,
            &self.hidden_io,
            &self.weights.layers[i].input_norm_w,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {i} input rms_norm: {e}"))?;

            if self.weights.config.is_full_attention(i) {
                self.component_decode_full_attention_layer(i, seqlen_offset)?;
            } else {
                if let Some(trace) = self.component_decode_linear_attention_layer(
                    i,
                    trace_linear_layer == Some(i),
                )? {
                    traced_linear = Some(trace);
                }
            }

            let mut trace_attn_hidden = None;
            let mut trace_post_attn_norm = None;
            if trace_layer == Some(i) {
                trace_attn_hidden = Some(
                    self.hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} attn hidden trace D2H: {e}"))?,
                );
            }

            kernel_ffi::prefill_ffi::rms_norm_rows(
                self.ordinal,
                ScalarType::BF16,
                1,
                hidden_dim,
                rms_norm_eps,
                &self.hidden_io,
                &self.weights.layers[i].post_attn_norm_w,
                &mut self.normed_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {i} post-attn rms_norm: {e}"))?;

            if trace_layer == Some(i) {
                trace_post_attn_norm = Some(
                    self.normed_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} post-attn norm trace D2H: {e}"))?,
                );
            }

            let maybe_mlp = self.component_decode_mlp_layer(i, trace_layer == Some(i))?;
            if trace_layer == Some(i) {
                let mlp_trace = maybe_mlp
                    .ok_or_else(|| anyhow::anyhow!("missing mlp trace for layer {i}"))?;
                traced_layer = Some(ComponentLayerTrace {
                    attn_hidden: trace_attn_hidden
                        .ok_or_else(|| anyhow::anyhow!("missing attn trace for layer {i}"))?,
                    post_attn_norm: trace_post_attn_norm
                        .ok_or_else(|| anyhow::anyhow!("missing post-attn norm trace for layer {i}"))?,
                    mlp_swiglu: mlp_trace.swiglu,
                    mlp_out: mlp_trace.down,
                    layer_hidden: self
                        .hidden_io
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {i} final hidden trace D2H: {e}"))?,
                });
            }
        }

        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if self.weights.config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            1,
            hidden_dim,
            rms_norm_eps,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm: {e}"))?;

        kernel_ffi::standalone_matvec_4b(
            self.ordinal,
            ScalarType::BF16,
            &mut self.logits_buf,
            &self.normed_buf,
            &*self.weights.lm_head,
            hidden_dim,
            vocab_size,
            &mut self.matvec_counter,
        )
        .map_err(|e| anyhow::anyhow!("lm_head matvec: {e}"))?;

        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
        Ok((
            logits_bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            traced_hidden,
            traced_layer,
            traced_linear,
        ))
    }

    fn component_decode_step_4b(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
        let (logits, _, _, _) =
            self.component_decode_step_4b_impl(token_id, seqlen_offset, None, None, None)?;
        Ok(logits)
    }

    pub fn component_decode_step_4b_traced(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_input_layer: usize,
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        let (logits, trace, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            Some(trace_input_layer),
            None,
            None,
        )?;
        let trace = trace.ok_or_else(|| anyhow::anyhow!("missing trace for layer {trace_input_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLayerTrace)> {
        let (logits, _, trace, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            Some(trace_layer),
            None,
        )?;
        let trace = trace.ok_or_else(|| anyhow::anyhow!("missing stage trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_linear_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLinearTrace)> {
        let (logits, _, _, trace) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            None,
            None,
            Some(trace_layer),
        )?;
        let trace = trace.ok_or_else(|| anyhow::anyhow!("missing linear trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    fn component_decode_full_attention_layer(&mut self, idx: usize, seqlen_offset: usize) -> Result<()> {
        let config = &self.weights.config;
        let fw = self.weights.layers[idx]
            .full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv_heads * head_dim;
        let rotary_dim = config.rotary_dim();
        let kv_len = seqlen_offset + 1;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let mut q_full = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_full alloc: {e}"))?;
        let mut query_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} query alloc: {e}"))?;
        let mut gate_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} gate alloc: {e}"))?;
        let mut k_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k alloc: {e}"))?;
        let mut v_buf = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} v alloc: {e}"))?;
        let mut q_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_normed alloc: {e}"))?;
        let mut k_normed = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_normed alloc: {e}"))?;
        let mut attn_q = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_q alloc: {e}"))?;
        let mut attn_k_step = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_k alloc: {e}"))?;
        let mut attn_v_step = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_v alloc: {e}"))?;
        let mut attn_out_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_out alloc: {e}"))?;
        let mut attn_out_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_out bf16 alloc: {e}"))?;
        let mut attn_flat = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_flat alloc: {e}"))?;
        let mut proj_out = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} proj_out alloc: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, q_proj_dim, hidden_dim,
            &self.normed_buf, &fw.q_proj_w, fw.q_proj_scale.as_ref(), self.weights.fp8_block_size, &mut q_full,
            fw.q_proj_int4_scale.as_ref(), fw.q_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::split_qgate(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            &q_full,
            &mut query_buf,
            &mut gate_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} split qgate: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &self.normed_buf, &fw.k_proj_w, fw.k_proj_scale.as_ref(), self.weights.fp8_block_size, &mut k_buf,
            fw.k_proj_int4_scale.as_ref(), fw.k_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal, 1, 1, kv_dim, hidden_dim,
            &self.normed_buf, &fw.v_proj_w, fw.v_proj_scale.as_ref(), self.weights.fp8_block_size, &mut v_buf,
            fw.v_proj_int4_scale.as_ref(), fw.v_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_q_heads, head_dim, 1e-6,
            &query_buf, &fw.q_norm_w, &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            query_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            q_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q norm copy: {e}"))?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal, ScalarType::BF16, num_kv_heads, head_dim, 1e-6,
            &k_buf, &fw.k_norm_w, &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k norm: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            k_buf.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            kv_dim * elem_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k norm copy: {e}"))?;

        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, rotary_dim,
            &self.rotary.cos, &self.rotary.sin, seqlen_offset, &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k rope: {e}"))?;

        kernel_ffi::prefill_ffi::transpose_shd_hsd(self.ordinal, ScalarType::BF16, 1, num_q_heads, head_dim, &query_buf, &mut attn_q)
            .map_err(|e| anyhow::anyhow!("layer {idx} q transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, &k_buf, &mut attn_k_step)
            .map_err(|e| anyhow::anyhow!("layer {idx} k transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(self.ordinal, ScalarType::BF16, 1, num_kv_heads, head_dim, &v_buf, &mut attn_v_step)
            .map_err(|e| anyhow::anyhow!("layer {idx} v transpose: {e}"))?;

        let ls = &mut self.state.layers[idx];
        ls.ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size, self.kv_fp8)
            .map_err(|e| anyhow::anyhow!("layer {idx} kv alloc: {e}"))?;
        if let Some(ref mut cache_k) = ls.kv_cache_k {
            let cap = cache_k.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = head_dim * elem_bytes;
            let dst_offset = seqlen_offset * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    cache_k.offset_ptr(h * cap_stride + dst_offset) as *mut c_void,
                    attn_k_step.offset_ptr(h * src_stride),
                    src_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cache k write h={h}: {e}"))?;
            }
        }
        if let Some(ref mut cache_v) = ls.kv_cache_v {
            let cap = cache_v.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = head_dim * elem_bytes;
            let dst_offset = seqlen_offset * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    cache_v.offset_ptr(h * cap_stride + dst_offset) as *mut c_void,
                    attn_v_step.offset_ptr(h * src_stride),
                    src_stride,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cache v write h={h}: {e}"))?;
            }
        }

        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
        let cap = cache_k_ref.shape()[2];
        let kv_k_contig;
        let kv_v_contig;
        let attn_k_ref;
        let attn_v_ref;
        if cap == kv_len {
            attn_k_ref = cache_k_ref;
            attn_v_ref = cache_v_ref;
        } else {
            kv_k_contig = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} kv_k_contig alloc: {e}"))?;
            kv_v_contig = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, kv_len, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} kv_v_contig alloc: {e}"))?;
            let cap_stride = cap * head_dim * elem_bytes;
            let contig_stride = kv_len * head_dim * elem_bytes;
            let copy_bytes = kv_len * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    kv_k_contig.offset_ptr(h * contig_stride) as *mut c_void,
                    cache_k_ref.offset_ptr(h * cap_stride),
                    copy_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} kv assemble k h={h}: {e}"))?;
                gpu_hal::copy_d2d(
                    self.ordinal,
                    kv_v_contig.offset_ptr(h * contig_stride) as *mut c_void,
                    cache_v_ref.offset_ptr(h * cap_stride),
                    copy_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} kv assemble v h={h}: {e}"))?;
            }
            attn_k_ref = &kv_k_contig;
            attn_v_ref = &kv_v_contig;
        }

        kernel_ffi::prefill_ffi::full_attention_prefill(
            self.ordinal, ScalarType::BF16, 1, num_q_heads, num_kv_heads,
            1, kv_len, head_dim, 1.0 / (head_dim as f32).sqrt(), seqlen_offset,
            &attn_q, attn_k_ref, attn_v_ref, &mut attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal, ScalarType::F32, ScalarType::BF16, num_q_heads * head_dim, &attn_out_f32, &mut attn_out_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal, ScalarType::BF16, num_q_heads, 1, head_dim, &attn_out_bf16, &mut attn_flat,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn transpose back: {e}"))?;

        let mut gated = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} gated alloc: {e}"))?;
        kernel_ffi::prefill_ffi::sigmoid_mul(
            self.ordinal, ScalarType::BF16, q_dim, &attn_flat, &gate_buf, &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gate apply: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, hidden_dim, q_dim,
            &gated, &fw.o_proj_w, fw.o_proj_scale.as_ref(), self.weights.fp8_block_size, &mut proj_out,
            fw.o_proj_int4_scale.as_ref(), fw.o_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &proj_out)?;
        Ok(())
    }

    fn component_decode_linear_attention_layer(
        &mut self,
        idx: usize,
        trace_output: bool,
    ) -> Result<Option<ComponentLinearTrace>> {
        let config = &self.weights.config;
        let lw = self.weights.layers[idx]
            .linear
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear attention weights"))?;
        let hidden_dim = config.hidden_size;
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;
        let key_dim = nk * khd;
        let val_dim = nv * vhd;
        let qkv_dim = key_dim * 2 + val_dim;
        let head_repeat = nv / nk;

        let mut qkv = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, qkv_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} qkv alloc: {e}"))?;
        let mut z = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} z alloc: {e}"))?;
        let mut a = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("layer {idx} a alloc: {e}"))?;
        let mut b = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("layer {idx} b alloc: {e}"))?;
        let a_beta_raw = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, nv * 2])
            .map_err(|e| anyhow::anyhow!("layer {idx} a_beta alloc: {e}"))?;
        let mut rec_apply = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, val_dim + nv * khd * vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} rec_apply alloc: {e}"))?;
        let mut attn_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} attn_bf16 alloc: {e}"))?;
        let mut norm_w_bf16 = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} norm_w alloc: {e}"))?;
        let mut gated = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("layer {idx} gated alloc: {e}"))?;
        let mut proj_out = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} proj_out alloc: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, qkv_dim, hidden_dim,
            &self.normed_buf, &lw.qkv_proj_w, lw.qkv_proj_scale.as_ref(), self.weights.fp8_block_size, &mut qkv,
            lw.qkv_proj_int4_scale.as_ref(), lw.qkv_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        let qkv_trace = if trace_output {
            Some(qkv.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} qkv trace D2H: {e}"))?)
        } else {
            None
        };
        matmul_proj(
            self.ordinal, 1, 1, val_dim, hidden_dim,
            &self.normed_buf, &lw.z_proj_w, lw.z_proj_scale.as_ref(), self.weights.fp8_block_size, &mut z,
            lw.z_proj_int4_scale.as_ref(), lw.z_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        let z_trace = if trace_output {
            Some(z.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} z trace D2H: {e}"))?)
        } else {
            None
        };
        matmul_proj(
            self.ordinal, 1, 1, nv, hidden_dim,
            &self.normed_buf, &lw.a_proj_w, lw.a_proj_scale.as_ref(), self.weights.fp8_block_size, &mut a,
            None, None, self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal, 1, 1, nv, hidden_dim,
            &self.normed_buf, &lw.b_proj_w, lw.b_proj_scale.as_ref(), self.weights.fp8_block_size, &mut b,
            None, None, self.weights.int4_group_size,
        )?;

        let ab_bytes = nv * ScalarType::BF16.size_in_bytes();
        gpu_hal::copy_d2d(
            self.ordinal,
            a_beta_raw.as_ptr() as *mut c_void,
            a.as_ptr(),
            ab_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} copy A: {e}"))?;
        gpu_hal::copy_d2d(
            self.ordinal,
            a_beta_raw.offset_ptr(ab_bytes) as *mut c_void,
            b.as_ptr(),
            ab_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} copy B: {e}"))?;
        let a_trace = if trace_output {
            Some(a.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} a trace D2H: {e}"))?)
        } else {
            None
        };
        let b_trace = if trace_output {
            Some(b.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} b trace D2H: {e}"))?)
        } else {
            None
        };

        let ls = &mut self.state.layers[idx];
        let conv_state = ls.conv_state.as_ref().ok_or_else(|| anyhow::anyhow!("layer {idx}: missing conv state"))?;
        let recurrent_state = ls.recurrent_state.as_ref().ok_or_else(|| anyhow::anyhow!("layer {idx}: missing recurrent state"))?;

        let mut conv_pack = GpuBuffer::zeros(
            self.ordinal,
            ScalarType::BF16,
            &[1, qkv_dim + nv],
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} conv_pack alloc: {e}"))?;
        kernel_ffi::prefill_ffi::linear_stateful_conv_value_decay_4b(
            self.ordinal,
            ScalarType::BF16,
            1,
            qkv_dim,
            1,
            config.linear_conv_kernel_dim - 1,
            config.linear_conv_kernel_dim,
            nv,
            &qkv,
            conv_state,
            &lw.conv1d_w,
            &a,
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut conv_pack,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} linear conv/value_decay: {e}"))?;

        let mut q_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_linear alloc: {e}"))?;
        let mut k_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_linear alloc: {e}"))?;
        let mut v_linear = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear alloc: {e}"))?;
        kernel_ffi::prefill_ffi::split_qkv(
            self.ordinal,
            ScalarType::BF16,
            1,
            key_dim,
            val_dim,
            &conv_pack,
            &mut q_linear,
            &mut k_linear,
            &mut v_linear,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} split_qkv: {e}"))?;

        let mut q_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_linear_f32 alloc: {e}"))?;
        let mut k_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_linear_f32 alloc: {e}"))?;
        let mut v_linear_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear_f32 alloc: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            key_dim,
            &q_linear,
            &mut q_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q cast: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            key_dim,
            &k_linear,
            &mut k_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k cast: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::BF16,
            ScalarType::F32,
            val_dim,
            &v_linear,
            &mut v_linear_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} v cast: {e}"))?;

        let mut q_normed = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_normed alloc: {e}"))?;
        let mut q_scaled = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} q_scaled alloc: {e}"))?;
        let mut k_normed = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("layer {idx} k_normed alloc: {e}"))?;
        kernel_ffi::prefill_ffi::l2norm(
            self.ordinal,
            ScalarType::F32,
            nk,
            khd,
            1e-6,
            &q_linear_f32,
            &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q l2norm: {e}"))?;
        kernel_ffi::prefill_ffi::mul_scalar(
            self.ordinal,
            ScalarType::F32,
            key_dim,
            (khd as f32).sqrt().recip(),
            &q_normed,
            &mut q_scaled,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q scale: {e}"))?;
        kernel_ffi::prefill_ffi::l2norm(
            self.ordinal,
            ScalarType::F32,
            nk,
            khd,
            1e-6,
            &k_linear_f32,
            &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k l2norm: {e}"))?;

        let q_scaled_host = q_scaled
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} q_scaled D2H: {e}"))?;
        let k_normed_host = k_normed
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} k_normed D2H: {e}"))?;
        let v_linear_host = v_linear_f32
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} v_linear D2H: {e}"))?;
        let a_host = a
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} a D2H: {e}"))?;
        let b_host = b
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} b D2H: {e}"))?;
        let dt_bias_host = lw
            .dt_bias
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} dt_bias D2H: {e}"))?;
        let a_log_exp_host = lw
            .a_log_exp
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} a_log_exp D2H: {e}"))?;
        let q_scaled_f32: Vec<f32> = q_scaled_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let k_normed_f32: Vec<f32> = k_normed_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v_linear_f32_host: Vec<f32> = v_linear_host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let a_bf16: Vec<f32> = a_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let b_bf16: Vec<f32> = b_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let dt_bias_bf16: Vec<f32> = dt_bias_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let a_log_exp_bf16: Vec<f32> = a_log_exp_host
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let packed_width = 2 * khd + vhd + 2;
        let mut packed_host = vec![0f32; nv * packed_width];
        for v_head in 0..nv {
            let k_head = v_head / head_repeat;
            let out_base = v_head * packed_width;
            let q_base = k_head * khd;
            let k_base = k_head * khd;
            let v_base = v_head * vhd;
            for i in 0..khd {
                packed_host[out_base + i] = q_scaled_f32[q_base + i];
                packed_host[out_base + khd + i] = k_normed_f32[k_base + i];
            }
            for i in 0..vhd {
                packed_host[out_base + 2 * khd + i] = v_linear_f32_host[v_base + i];
            }
            packed_host[out_base + 2 * khd + vhd] = 1.0f32 / (1.0f32 + (-b_bf16[v_head]).exp());
            let softplus = (1.0f32 + (a_bf16[v_head] + dt_bias_bf16[v_head]).exp()).ln();
            packed_host[out_base + 2 * khd + vhd + 1] =
                (-softplus * a_log_exp_bf16[v_head]).exp();
        }
        let packed = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::F32,
            &[nv, packed_width],
            &packed_host
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} packed H2D: {e}"))?;

        kernel_ffi::prefill_ffi::linear_decode_apply_4b(
            self.ordinal,
            1,
            nv,
            khd,
            vhd,
            &packed,
            recurrent_state,
            &mut rec_apply,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} linear decode apply: {e}"))?;

        let state_len = config.linear_conv_kernel_dim - 1;
        let state_bytes = ScalarType::BF16.size_in_bytes();
        let new_conv_state = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[qkv_dim, state_len])
            .map_err(|e| anyhow::anyhow!("layer {idx} new_conv_state alloc: {e}"))?;
        for c in 0..qkv_dim {
            let channel_base = c * state_len * state_bytes;
            if state_len > 1 {
                gpu_hal::copy_d2d(
                    self.ordinal,
                    new_conv_state.offset_ptr(channel_base) as *mut c_void,
                    conv_state.offset_ptr(channel_base + state_bytes),
                    (state_len - 1) * state_bytes,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} conv shift c={c}: {e}"))?;
            }
            gpu_hal::copy_d2d(
                self.ordinal,
                new_conv_state.offset_ptr(channel_base + (state_len - 1) * state_bytes) as *mut c_void,
                qkv.offset_ptr(c * state_bytes),
                state_bytes,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} conv append c={c}: {e}"))?;
        }
        gpu_hal::copy_d2d(
            self.ordinal,
            conv_state.as_ptr() as *mut c_void,
            new_conv_state.as_ptr(),
            qkv_dim * state_len * state_bytes,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} conv state update copy: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal, ScalarType::F32, ScalarType::BF16, val_dim, &rec_apply, &mut attn_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
        let attn_trace = if trace_output {
            Some(attn_bf16.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} attn trace D2H: {e}"))?)
        } else {
            None
        };
        let rec_apply_trace = if trace_output {
            Some(
                rec_apply
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} rec_apply trace D2H: {e}"))?,
            )
        } else {
            None
        };
        gpu_hal::copy_d2d(
            self.ordinal,
            recurrent_state.as_ptr() as *mut c_void,
            rec_apply.offset_ptr(val_dim * ScalarType::F32.size_in_bytes()),
            nv * khd * vhd * ScalarType::F32.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} recurrent update copy: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal, ScalarType::F32, ScalarType::BF16, vhd, &lw.norm_w, &mut norm_w_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} norm_w cast: {e}"))?;
        kernel_ffi::prefill_ffi::rms_norm_gated(
            self.ordinal,
            ScalarType::BF16,
            nv,
            vhd,
            config.rms_norm_eps as f32,
            &attn_bf16,
            &z,
            &norm_w_bf16,
            &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;
        let gated_trace = if trace_output {
            Some(gated.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} gated trace D2H: {e}"))?)
        } else {
            None
        };

        matmul_proj(
            self.ordinal, 1, 1, hidden_dim, val_dim,
            &gated, &lw.out_proj_w, lw.out_proj_scale.as_ref(), self.weights.fp8_block_size, &mut proj_out,
            lw.out_proj_int4_scale.as_ref(), lw.out_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        let proj_trace = if trace_output {
            Some(proj_out.to_host_bytes().map_err(|e| anyhow::anyhow!("layer {idx} proj trace D2H: {e}"))?)
        } else {
            None
        };
        let packed_trace = if trace_output {
            Some(
                packed
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} packed trace D2H: {e}"))?,
            )
        } else {
            None
        };
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &proj_out)?;
        Ok(if trace_output {
            Some(ComponentLinearTrace {
                qkv: qkv_trace.unwrap_or_default(),
                z: z_trace.unwrap_or_default(),
                b: b_trace.unwrap_or_default(),
                a: a_trace.unwrap_or_default(),
                packed: packed_trace.unwrap_or_default(),
                rec_apply: rec_apply_trace.unwrap_or_default(),
                attn: attn_trace.unwrap_or_default(),
                gated: gated_trace.unwrap_or_default(),
                proj_out: proj_trace.unwrap_or_default(),
            })
        } else {
            None
        })
    }

    fn component_decode_mlp_layer(&mut self, idx: usize, trace_output: bool) -> Result<Option<ComponentMlpTrace>> {
        let config = &self.weights.config;
        let lw = &self.weights.layers[idx];
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut gate = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("layer {idx} mlp gate alloc: {e}"))?;
        let mut up = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("layer {idx} mlp up alloc: {e}"))?;
        let mut mlp = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("layer {idx} mlp act alloc: {e}"))?;
        let mut down = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("layer {idx} mlp down alloc: {e}"))?;

        matmul_proj(
            self.ordinal, 1, 1, intermediate, hidden_dim,
            &self.normed_buf, &lw.gate_proj_w, lw.gate_proj_scale.as_ref(), self.weights.fp8_block_size, &mut gate,
            lw.gate_proj_int4_scale.as_ref(), lw.gate_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal, 1, 1, intermediate, hidden_dim,
            &self.normed_buf, &lw.up_proj_w, lw.up_proj_scale.as_ref(), self.weights.fp8_block_size, &mut up,
            lw.up_proj_int4_scale.as_ref(), lw.up_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::swiglu_mul(
            self.ordinal,
            ScalarType::BF16,
            intermediate,
            &gate,
            &up,
            &mut mlp,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} swiglu: {e}"))?;
        matmul_proj(
            self.ordinal, 1, 1, hidden_dim, intermediate,
            &mlp, &lw.down_proj_w, lw.down_proj_scale.as_ref(), self.weights.fp8_block_size, &mut down,
            lw.down_proj_int4_scale.as_ref(), lw.down_proj_int4_zero.as_ref(), self.weights.int4_group_size,
        )?;
        let trace = if trace_output {
            Some(ComponentMlpTrace {
                swiglu: mlp
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp swiglu trace D2H: {e}"))?,
                down: down
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} mlp down trace D2H: {e}"))?,
            })
        } else {
            None
        };
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &down)?;
        Ok(trace)
    }

    fn apply_oracle_hidden(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;

        let hidden_b64 = oracle
            .prefill_hidden
            .as_ref()
            .context("oracle output missing prefill_hidden (use --emit-state)")?;
        let hidden_bytes = b64.decode(hidden_b64).context("decode prefill_hidden base64")?;
        let hidden_shape = oracle
            .prefill_hidden_shape
            .as_ref()
            .context("missing prefill_hidden_shape")?;
        // Oracle's tensor_to_b64 may return the full underlying storage (all tokens)
        // instead of just the last token. Take only the last token's worth of bytes.
        let expected_bytes: usize =
            hidden_shape.iter().product::<usize>() * ScalarType::BF16.size_in_bytes();
        let actual_hidden = if hidden_bytes.len() > expected_bytes {
            &hidden_bytes[hidden_bytes.len() - expected_bytes..]
        } else {
            &hidden_bytes
        };
        self.hidden_io = GpuBuffer::from_host_bytes(
            self.ordinal,
            ScalarType::BF16,
            hidden_shape,
            actual_hidden,
        )
        .map_err(|e| anyhow::anyhow!("load prefill hidden: {e}"))?;
        Ok(())
    }

    fn apply_oracle_full_attention_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let kv_caches = oracle
            .kv_caches
            .as_ref()
            .context("oracle output missing kv_caches")?;
        for kv in kv_caches {
            let k_bytes = b64.decode(&kv.k).context("decode KV k base64")?;
            let v_bytes = b64.decode(&kv.v).context("decode KV v base64")?;
            let ls = &mut self.state.layers[kv.layer];
            ls.kv_cache_k = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &kv.k_shape, &k_bytes)
                    .map_err(|e| anyhow::anyhow!("load KV k layer {}: {e}", kv.layer))?,
            );
            ls.kv_cache_v = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &kv.v_shape, &v_bytes)
                    .map_err(|e| anyhow::anyhow!("load KV v layer {}: {e}", kv.layer))?,
            );
            ls.kv_filled = kv.k_shape[2];
        }
        Ok(())
    }

    fn apply_oracle_linear_attention_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        self.apply_oracle_conv_state(oracle)?;
        self.apply_oracle_recurrent_state(oracle)?;
        Ok(())
    }


    fn apply_oracle_conv_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let conv_states = oracle
            .conv_states
            .as_ref()
            .context("oracle output missing conv_states")?;
        for cs in conv_states {
            let bytes = b64.decode(&cs.data).context("decode conv_state base64")?;
            let ls = &mut self.state.layers[cs.layer];
            ls.conv_state = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, &cs.shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("load conv_state layer {}: {e}", cs.layer))?,
            );
        }
        Ok(())
    }

    fn apply_oracle_recurrent_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let rec_states = oracle
            .recurrent_states
            .as_ref()
            .context("oracle output missing recurrent_states")?;
        for rs in rec_states {
            let bytes = b64.decode(&rs.data).context("decode recurrent_state base64")?;
            let ls = &mut self.state.layers[rs.layer];
            ls.recurrent_state = Some(
                GpuBuffer::from_host_bytes(self.ordinal, ScalarType::F32, &rs.shape, &bytes)
                    .map_err(|e| anyhow::anyhow!("load recurrent_state layer {}: {e}", rs.layer))?,
            );
        }
        Ok(())
    }

    pub fn new(
        weights: Qwen35Weights,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
        prefill_chunk_size: usize,
        kv_fp8: bool,
        batch_size: usize,
    ) -> Result<Self> {
        let config = &weights.config;
        let state = ModelState::new(config, ordinal)
            .map_err(|e| anyhow::anyhow!("model state init: {e}"))?;

        // Create extra model states for batch items 1..batch_size
        let mut extra_states = Vec::new();
        for b in 1..batch_size {
            extra_states.push(
                ModelState::new(config, ordinal)
                    .map_err(|e| anyhow::anyhow!("model state init (batch {b}): {e}"))?,
            );
        }

        let scratch = PersistentDecodeScratch::new(
            ordinal,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            proj_buf_floats,
            attn_scratch_floats,
            batch_size,
        )
        .map_err(|e| anyhow::anyhow!("scratch init: {e}"))?;
        let rotary =
            RotaryTables::build(config, ordinal).map_err(|e| anyhow::anyhow!("rotary: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[batch_size, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("hidden_io: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[batch_size, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("normed_buf: {e}"))?;
        let logits_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[batch_size, 1, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("logits_buf: {e}"))?;
        let argmax_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("argmax_buf: {e}"))?;
        let lm_head_block_best_vals = GpuBuffer::zeros(ordinal, ScalarType::F32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_vals: {e}"))?;
        let lm_head_block_best_idxs = GpuBuffer::zeros(ordinal, ScalarType::U32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_idxs: {e}"))?;
        let matvec_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("matvec_counter: {e}"))?;

        let fp8_scale_device = if let Some(fp8_descs) = build_fp8_scale_descs(&weights) {
            let desc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    fp8_descs.as_ptr() as *const u8,
                    fp8_descs.len() * std::mem::size_of::<kernel_ffi::FP8ScaleDesc>(),
                )
            };
            let buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[desc_bytes.len()],
                desc_bytes,
            )
            .map_err(|e| anyhow::anyhow!("upload fp8 scale descs: {e}"))?;
            Some(buf)
        } else {
            None
        };

        let int4_scale_device = if let Some(int4_descs) = build_int4_scale_descs(&weights) {
            let desc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    int4_descs.as_ptr() as *const u8,
                    int4_descs.len() * std::mem::size_of::<kernel_ffi::INT4ScaleDesc>(),
                )
            };
            let buf = GpuBuffer::from_host_bytes(
                ordinal,
                ScalarType::U8,
                &[desc_bytes.len()],
                desc_bytes,
            )
            .map_err(|e| anyhow::anyhow!("upload int4 scale descs: {e}"))?;
            Some(buf)
        } else {
            None
        };

        Ok(Self {
            weights,
            state,
            extra_states,
            scratch,
            rotary,
            hidden_io,
            normed_buf,
            logits_buf,
            argmax_buf,
            lm_head_block_best_vals,
            lm_head_block_best_idxs,
            matvec_counter,
            ordinal,
            kv_chunk_size,
            use_4b_kernel,
            proj_buf_floats,
            attn_scratch_floats,
            fp8_scale_device,
            int4_scale_device,
            prefill_chunk_size,
            kv_fp8,
            batch_size,
        })
    }

    pub fn weights(&self) -> &Qwen35Weights {
        &self.weights
    }

    pub fn kv_fp8_enabled(&self) -> bool {
        self.kv_fp8
    }

    pub fn set_kv_fp8_for_trace(&mut self, enabled: bool) {
        self.kv_fp8 = enabled;
    }

    pub fn rotary(&self) -> &RotaryTables {
        &self.rotary
    }

    pub fn state_for_batch(&self, batch_index: usize) -> &ModelState {
        if batch_index == 0 {
            &self.state
        } else {
            &self.extra_states[batch_index - 1]
        }
    }

    /// Load prefill state from oracle output into GPU buffers.
    pub fn load_prefill_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        self.apply_oracle_hidden(oracle)?;
        self.apply_oracle_full_attention_state(oracle)?;
        self.apply_oracle_linear_attention_state(oracle)?;

        // Convert BF16 KV caches to FP8 if requested
        if self.kv_fp8 {
            prefill_engine::convert_kv_caches_to_fp8(
                &mut self.state, &self.weights.config, self.ordinal,
            )?;
        }

        // Reset sync counters for fresh kernel launch sequence
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync: {e}"))?;

        Ok(())
    }

    /// Run native GPU prefill on the prompt, returning logits for the last token.
    /// Fills KV caches, conv states, and recurrent states for subsequent decode.
    pub fn prefill_native(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        let result = prefill_engine::prefill(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            false,
            None,
        )?;

        // Reset sync counters for the decode kernel
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result.logits)
    }

    /// Rebuild sequence-0 state from scratch by replaying native GPU prefill
    /// over the provided token history. Optionally replicates that state across
    /// extra batch slots for lockstep batch decoding.
    pub fn rebuild_prefill_state(
        &mut self,
        token_ids: &[u32],
        replicate_batch: bool,
    ) -> Result<Vec<f32>> {
        self.state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("rebuild model state init: {e}"))?;
        let logits = self.prefill_native(token_ids)?;
        if replicate_batch && self.batch_size > 1 {
            self.replicate_state_to_batch()?;
        }
        Ok(logits)
    }

    pub fn prefill_native_with_trace(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            true,
            None,
        )?;

        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result)
    }

    fn decode_step_non_4b(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        sampling_mode: DecodeSamplingMode,
    ) -> Result<DecodeStepOutput> {
        let config = &self.weights.config;
        let mut timings = DecodeStageTimings::default();

        // 1. Embedding lookup: copy one row from embed_tokens into hidden_io
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        // 2. Ensure KV capacity for full-attention layers
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size, self.kv_fp8)
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {i}: {e}"))?;
            }
        }
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(&self.weights.config, self.ordinal, &mut self.state)?;
        }

        // 3. Build layer descriptors
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);

        // 4. Upload descriptors to device
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload descs: {e}"))?;

        // 4b. Upload KV FP8 scale descriptors (pointers may change on KV cache growth)
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("clear decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset decode sync: {e}"))?;

        // 5. Launch persistent decode kernel (dispatch by model variant)
        let start = Instant::now();
        let persist_result = if sampling_mode == DecodeSamplingMode::CudaHeroFusedLmHead {
            kernel_ffi::persistent_decode_qwen08_sm86_specialized(
                self.ordinal,
                ScalarType::BF16,
                config.num_hidden_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
            )
        } else {
            kernel_ffi::persistent_decode(
                self.ordinal,
                ScalarType::BF16,
                config.num_hidden_layers,
                config.hidden_size,
                config.intermediate_size,
                seqlen_offset,
                &self.scratch.desc_device,
                &mut self.hidden_io,
                &mut self.scratch.workspace,
                &mut self.scratch.sync_buf,
                &self.rotary.cos,
                &self.rotary.sin,
                self.rotary.rotary_dim,
            )
        };
        persist_result
            .map_err(|e| anyhow::anyhow!("persistent_decode kernel: {e}"))?;
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;

        // 6. Update KV filled counts
        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        // 7. Final RMSNorm
        let start = Instant::now();
        kernel_ffi::rms_norm(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            config.hidden_size,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm: {e}"))?;
        timings.rms_norm_ms = start.elapsed().as_secs_f64() * 1000.0;

        match sampling_mode {
            DecodeSamplingMode::CudaHeroFusedLmHead => {
                let start = Instant::now();
                kernel_ffi::cuda_lm_head_argmax_bf16(
                    self.ordinal,
                    &self.normed_buf,
                    &*self.weights.lm_head,
                    &mut self.lm_head_block_best_vals,
                    &mut self.lm_head_block_best_idxs,
                    &mut self.argmax_buf,
                    config.hidden_size,
                    config.vocab_size,
                )
                .map_err(|e| anyhow::anyhow!("cuda fused lm_head argmax: {e}"))?;
                timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

                let start = Instant::now();
                let token_bytes = self
                    .argmax_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("argmax D2H: {e}"))?;
                timings.token_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
                let sampled_token = u32::from_le_bytes(
                    token_bytes[..4]
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("argmax D2H returned truncated token buffer"))?,
                );

                Ok(DecodeStepOutput {
                    logits: None,
                    sampled_token,
                    timings,
                })
            }
            DecodeSamplingMode::CudaFastGreedy | DecodeSamplingMode::HostLogits => {
                // 8. lm_head projection → logits (work-stealing matvec)
                let start = Instant::now();
                kernel_ffi::standalone_matvec(
                    self.ordinal,
                    ScalarType::BF16,
                    &mut self.logits_buf,
                    &self.normed_buf,
                    &*self.weights.lm_head,
                    config.hidden_size,
                    config.vocab_size,
                    &mut self.matvec_counter,
                )
                .map_err(|e| anyhow::anyhow!("lm_head matvec: {e}"))?;
                timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

                if sampling_mode == DecodeSamplingMode::CudaFastGreedy {
                    let start = Instant::now();
                    kernel_ffi::cuda_argmax_bf16(
                        self.ordinal,
                        &self.logits_buf,
                        &mut self.argmax_buf,
                        config.vocab_size,
                    )
                    .map_err(|e| anyhow::anyhow!("cuda argmax: {e}"))?;
                    timings.gpu_argmax_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let start = Instant::now();
                    let token_bytes = self
                        .argmax_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("argmax D2H: {e}"))?;
                    timings.token_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let sampled_token = u32::from_le_bytes(
                        token_bytes[..4]
                            .try_into()
                            .map_err(|_| anyhow::anyhow!("argmax D2H returned truncated token buffer"))?,
                    );

                    return Ok(DecodeStepOutput {
                        logits: None,
                        sampled_token,
                        timings,
                    });
                }

                // 9. Copy logits to CPU and convert BF16 → F32
                let start = Instant::now();
                let logits_bytes = self
                    .logits_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
                timings.logits_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;

                let start = Instant::now();
                let logits_f32: Vec<f32> = logits_bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                let sampled_token = Self::greedy_sample(&logits_f32);
                timings.host_sampling_ms = start.elapsed().as_secs_f64() * 1000.0;

                Ok(DecodeStepOutput {
                    logits: Some(logits_f32),
                    sampled_token,
                    timings,
                })
            }
        }
    }

    /// Run one decode step and return logits on CPU. Stage timings are only
    /// populated for the non-4B native decode path.
    pub fn decode_step_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        if self.use_4b_kernel {
            let logits = self.component_decode_step_4b(token_id, seqlen_offset)?;
            return Ok((logits, DecodeStageTimings::default()));
        }
        let out = self.decode_step_non_4b(token_id, seqlen_offset, DecodeSamplingMode::HostLogits)?;
        let logits = out
            .logits
            .ok_or_else(|| anyhow::anyhow!("decode_step_with_timings missing logits"))?;
        Ok((logits, out.timings))
    }

    /// CUDA-only fast greedy path for the non-4B single-sequence decode path.
    /// Returns the sampled token without copying full logits to the host.
    pub fn decode_step_cuda_fast_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.use_4b_kernel {
            anyhow::bail!("decode_step_cuda_fast_greedy only supports the non-4B path");
        }
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("decode_step_cuda_fast_greedy requires CUDA backend");
        }
        let out = self.decode_step_non_4b(token_id, seqlen_offset, DecodeSamplingMode::CudaFastGreedy)?;
        Ok((out.sampled_token, out.timings))
    }

    /// CUDA-only sm86/qwen3.5-0.8b hero path for the non-4B single-sequence decode path.
    /// Returns the sampled token without materializing logits on the host.
    pub fn decode_step_cuda_08b_hero(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.use_4b_kernel {
            anyhow::bail!("decode_step_cuda_08b_hero only supports the non-4B path");
        }
        if self.hidden_io.backend() != gpu_hal::Backend::Cuda {
            anyhow::bail!("decode_step_cuda_08b_hero requires CUDA backend");
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::CudaHeroFusedLmHead,
        )?;
        Ok((out.sampled_token, out.timings))
    }

    /// Run one decode step. Returns logits as Vec<f32> on CPU.
    pub fn decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
        let (logits, _) = self.decode_step_with_timings(token_id, seqlen_offset)?;
        Ok(logits)
    }

    /// Forced single-sequence 4B kernel path with native stage timings.
    pub fn decode_step_4b_single_kernel_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        anyhow::ensure!(
            self.use_4b_kernel,
            "decode_step_4b_single_kernel_with_timings requires 4B kernel"
        );
        anyhow::ensure!(
            self.batch_size == 1,
            "decode_step_4b_single_kernel_with_timings requires batch_size == 1"
        );

        let (mut batch_logits, mut timings) =
            self.decode_step_batch_impl(&[token_id], seqlen_offset, true)?;
        let logits = batch_logits
            .pop()
            .ok_or_else(|| anyhow::anyhow!("single-sequence 4B kernel timings missing logits"))?;
        let sampling_start = Instant::now();
        let _ = Self::greedy_sample(&logits);
        timings.host_sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
        Ok((logits, timings))
    }

    /// Greedy argmax over logits.
    pub fn greedy_sample(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Copy prefill state from sequence 0 to all extra batch sequences.
    /// Call after load_prefill_state() or prefill_native() to initialize batch items.
    pub fn replicate_state_to_batch(&mut self) -> Result<()> {
        for b in 0..self.extra_states.len() {
            self.extra_states[b] = self.state
                .clone_gpu()
                .map_err(|e| anyhow::anyhow!("clone state to batch {}: {e}", b + 1))?;
        }
        Ok(())
    }

    /// Run one batched decode step. Returns per-sequence logits.
    /// `token_ids`: one token per batch item.
    /// `seqlen_offset`: shared sequence position (all sequences advance in lockstep).
    pub fn decode_step_batch(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (all_logits, _) = self.decode_step_batch_impl(token_ids, seqlen_offset, false)?;
        Ok(all_logits)
    }

    /// Run one batched decode step and return per-sequence logits plus native
    /// stage timings for the persistent batch path.
    pub fn decode_step_batch_with_timings(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        self.decode_step_batch_impl(token_ids, seqlen_offset, true)
    }

    fn decode_step_batch_impl(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        enable_timing_slots: bool,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        assert_eq!(token_ids.len(), self.batch_size);
        assert!(self.use_4b_kernel, "batched decode requires 4b kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        let mut timings = DecodeStageTimings::default();

        // 1. Embedding lookup: place each sequence's embedding at offset b * hidden_size
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in token_ids.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("embedding lookup batch {bi}: {e}"))?;
        }

        // 2. Ensure KV capacity for all batch items
        let seqlen_offsets: Vec<usize> = vec![seqlen_offset; b];
        for bi in 0..b {
            let st = if bi == 0 { &mut self.state } else { &mut self.extra_states[bi - 1] };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size, self.kv_fp8)
                        .map_err(|e| anyhow::anyhow!("ensure KV capacity batch {bi} layer {i}: {e}"))?;
                }
            }
        }
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(&self.weights.config, self.ordinal, &mut self.state)?;
            for bi in 0..self.extra_states.len() {
                Self::load_kv_shadow_for_state_static(&self.weights.config, self.ordinal, &mut self.extra_states[bi])?;
            }
        }

        // 3. Build layer descriptors (weights only, per-sequence state in batch descs)
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload descs: {e}"))?;

        // 4. Build and upload batch sequence descriptors
        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8) {
            self.scratch
                .upload_batch_seq_descs(&batch_descs)
                .map_err(|e| anyhow::anyhow!("upload batch seq descs: {e}"))?;
        }

        // 4b. Upload KV FP8 scale descriptors
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("clear batched decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset batched decode sync: {e}"))?;

        // 5. Launch batched persistent decode kernel
        let start = Instant::now();
        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            seqlen_offset,
            &self.scratch.desc_device,
            &mut self.hidden_io,
            &mut self.scratch.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            self.fp8_scale_device.as_ref(),
            self.scratch.kv_fp8_desc_device.as_ref(),
            b,
            self.scratch.batch_seq_desc_device.as_ref(),
            self.int4_scale_device.as_ref(),
            enable_timing_slots,
            false,
        )
        .map_err(|e| anyhow::anyhow!("persistent_decode_4b batch kernel: {e}"))?;
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        if enable_timing_slots {
            let clock_rate_khz = gpu_hal::query_device_info(gpu_hal::Backend::Cuda, self.ordinal)
                .map_err(|e| anyhow::anyhow!("query CUDA device clock rate: {e}"))?
                .clock_rate_khz;
            let sync_bytes = self
                .scratch
                .sync_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("persistent timing slots D2H: {e}"))?;
            timings.add_assign(decode_persistent_4b_timing_slots(
                &sync_bytes,
                config.num_hidden_layers,
                b,
                clock_rate_khz,
            ));
        }

        // 6. Update KV filled counts for all batch items
        let filled = seqlen_offset + 1;
        for bi in 0..b {
            let st = if bi == 0 { &mut self.state } else { &mut self.extra_states[bi - 1] };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.set_kv_filled(filled);
                }
            }
        }

        // 7-9. Final multi-row RMSNorm + tiled lm_head matmul, then one D2H.
        let start = Instant::now();
        kernel_ffi::rms_norm_4b_multirow(
            self.ordinal,
            ScalarType::BF16,
            b,
            config.hidden_size,
            config.rms_norm_eps as f32,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm batch rows: {e}"))?;
        timings.rms_norm_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        kernel_ffi::matmul_rhs_transposed_4b(
            self.ordinal,
            ScalarType::BF16,
            1,
            b,
            config.vocab_size,
            config.hidden_size,
            &self.normed_buf,
            &*self.weights.lm_head,
            &mut self.logits_buf,
        )
        .map_err(|e| anyhow::anyhow!("tiled lm_head batch matmul: {e}"))?;
        timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let logits_host = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H batch rows: {e}"))?;
        timings.logits_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
        let row_bytes = config.vocab_size * ScalarType::BF16.size_in_bytes();
        let mut all_logits = Vec::with_capacity(b);
        for bi in 0..b {
            let start = bi * row_bytes;
            let end = start + row_bytes;
            let logits_f32: Vec<f32> = logits_host[start..end]
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect();
            all_logits.push(logits_f32);
        }

        Ok((all_logits, timings))
    }

    /// Debug-only: run the real batched 4B persistent kernel for the first `num_layers`
    /// layers and return one batch row of the resulting hidden state as BF16 bytes.
    /// This mutates the decode state; callers should rebuild state afterwards if they
    /// need to continue from the pre-trace state.
    pub fn decode_step_batch_trace_hidden_after_layers(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        num_layers: usize,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        assert_eq!(token_ids.len(), self.batch_size);
        assert!(self.use_4b_kernel, "persistent trace requires 4b kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        anyhow::ensure!(
            num_layers <= config.num_hidden_layers,
            "trace layer count {} exceeds model layers {}",
            num_layers,
            config.num_hidden_layers
        );
        anyhow::ensure!(
            batch_index < b,
            "trace batch index {} out of range for batch {}",
            batch_index,
            b
        );

        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in token_ids.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("trace embedding lookup batch {bi}: {e}"))?;
        }

        let seqlen_offsets: Vec<usize> = vec![seqlen_offset; b];
        for bi in 0..b {
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (i, ls) in st.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        seqlen_offset,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| anyhow::anyhow!("trace ensure KV capacity batch {bi} layer {i}: {e}"))?;
                }
            }
        }

        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("trace upload descs: {e}"))?;

        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8) {
            self.scratch
                .upload_batch_seq_descs(&batch_descs)
                .map_err(|e| anyhow::anyhow!("trace upload batch seq descs: {e}"))?;
        }

        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("trace upload kv fp8 descs: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("trace clear batched decode workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("trace reset batched decode sync: {e}"))?;

        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            config.hidden_size,
            config.intermediate_size,
            seqlen_offset,
            &self.scratch.desc_device,
            &mut self.hidden_io,
            &mut self.scratch.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            self.fp8_scale_device.as_ref(),
            self.scratch.kv_fp8_desc_device.as_ref(),
            b,
            self.scratch.batch_seq_desc_device.as_ref(),
            self.int4_scale_device.as_ref(),
            false,
            true,
        )
        .map_err(|e| anyhow::anyhow!("trace persistent_decode_4b batch kernel: {e}"))?;

        let hidden = self
            .hidden_io
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("trace hidden D2H: {e}"))?;
        let start = batch_index * row_bytes;
        let end = start + row_bytes;
        Ok(hidden[start..end].to_vec())
    }

    pub fn trace_persistent_linear_proj_buf_after_layers(
        &self,
        batch_index: usize,
        qkv_dim: usize,
        z_dim: usize,
        nv: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let prefix_floats = self.weights.config.hidden_size
            + self.weights.config.hidden_size
            + self.weights.config.intermediate_size * 2
            + self.weights.config.hidden_size
            + self.weights.config.hidden_size;
        let start_floats = prefix_floats * self.batch_size + batch_index * self.proj_buf_floats;
        let total_floats = qkv_dim + z_dim + nv + nv;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = start_floats * ScalarType::F32.size_in_bytes();
        let end = start + total_floats * ScalarType::F32.size_in_bytes();
        anyhow::ensure!(end <= bytes.len(), "persistent projection slice out of bounds");
        let slice = &bytes[start..end];
        let qkv_end = qkv_dim * 4;
        let z_end = qkv_end + z_dim * 4;
        let b_end = z_end + nv * 4;
        Ok((
            slice[..qkv_end].to_vec(),
            slice[qkv_end..z_end].to_vec(),
            slice[z_end..b_end].to_vec(),
            slice[b_end..].to_vec(),
        ))
    }

    pub fn trace_persistent_mlp_stage_after_layers(
        &self,
        batch_index: usize,
        intermediate: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let normed_start = (normed_base + batch_index * hidden) * 4;
        let normed_end = normed_start + hidden * 4;
        let gate_start = (gate_up_base + batch_index * intermediate * 2) * 4;
        let gate_end = gate_start + intermediate * 4;
        let mlp_out_start = (mlp_out_base + batch_index * hidden) * 4;
        let mlp_out_end = mlp_out_start + hidden * 4;
        let token_out_start = (token_out_base + batch_index * hidden) * 4;
        let token_out_end = token_out_start + hidden * 4;
        anyhow::ensure!(token_out_end <= bytes.len(), "persistent MLP slice out of bounds");
        Ok((
            bytes[normed_start..normed_end].to_vec(),
            bytes[gate_start..gate_end].to_vec(),
            bytes[mlp_out_start..mlp_out_end].to_vec(),
            bytes[token_out_start..token_out_end].to_vec(),
        ))
    }

    pub fn trace_persistent_linear_gated_after_layers(
        &self,
        batch_index: usize,
        value_dim: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats) * 4;
        let end = start + value_dim * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent gated slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_gated_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (proj_buf_base + batch_index * self.proj_buf_floats) * 4;
        let end = start + hidden * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent full-attn gated slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_saved_gate_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim =
            self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent full-attn saved_gate slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_q_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim =
            self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent full-attn q slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_pre_gate_after_layers(
        &self,
        batch_index: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim =
            self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim * 2) * 4;
        let end = start + q_dim * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent full-attn pre_gate slice out of bounds");
        Ok(bytes[start..end].to_vec())
    }

    pub fn trace_persistent_full_attention_scores_after_layers(
        &self,
        batch_index: usize,
        kv_len: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            batch_index < self.batch_size,
            "batch index {} out of range for batch {}",
            batch_index,
            self.batch_size
        );
        let q_dim =
            self.weights.config.num_attention_heads * self.weights.config.head_dim;
        let hidden = self.weights.config.hidden_size;
        let intermediate = self.weights.config.intermediate_size;
        let b = self.batch_size;
        let normed_base = b * hidden;
        let gate_up_base = normed_base + b * hidden;
        let mlp_out_base = gate_up_base + b * intermediate * 2;
        let token_out_base = mlp_out_base + b * hidden;
        let proj_buf_base = token_out_base + b * hidden;
        let attn_scratch_base = proj_buf_base + b * self.proj_buf_floats;
        let bytes = self
            .scratch
            .workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("persistent workspace D2H: {e}"))?;
        let start =
            (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim * 3) * 4;
        let end = start + self.weights.config.num_attention_heads * self.kv_chunk_size * 4;
        anyhow::ensure!(end <= bytes.len(), "persistent full-attn scores slice out of bounds");
        let full = &bytes[start..end];
        let mut out = Vec::with_capacity(self.weights.config.num_attention_heads * kv_len * 4);
        let stride = self.kv_chunk_size * 4;
        for h in 0..self.weights.config.num_attention_heads {
            let row = h * stride;
            out.extend_from_slice(&full[row..row + kv_len * 4]);
        }
        Ok(out)
    }
}
