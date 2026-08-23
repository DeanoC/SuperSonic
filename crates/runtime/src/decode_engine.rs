#![allow(dead_code)]

use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use base64::Engine as _;
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::config::TextConfig;
use qwen35::desc_builder::{
    build_batch_seq_descs, build_fp8_scale_descs, build_int4_scale_descs, build_kv_fp8_descs,
    build_layer_descs,
};
use qwen35::rotary::RotaryTables;
use qwen35::scratch::PersistentDecodeScratch;
use qwen35::state::{
    kv_fp8_bf16_sidecar_enabled, kv_fp8_bf16_sidecar_window_tokens, LinearStateSnapshot,
    ModelState, ModelStateDiskSnapshot,
};
use qwen35::weights::Qwen35Weights;
use serde::{Deserialize, Serialize};

use crate::mtp::{
    mtp_decode_step, mtp_forward, prefill_append_verify_cached, restore_linear_prefix,
    restore_linear_state, MtpPrefillAppendCache, MtpVerifyCache, MtpVerifyScratch,
};
use crate::oracle::OracleOutput;
use crate::prefill_engine;

/// Decode a byte slice of little-endian `f32` values into a host `Vec<f32>`.
/// Shared helper used across decode/validate paths.
pub fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_bf16_le_host(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn gqh_state_dump_enabled() -> bool {
    env::var_os("SUPERSONIC_QWEN35_GQH_STATE_DUMP").is_some()
}

fn linear_layer_dump_dir(idx: usize) -> Option<std::path::PathBuf> {
    let want = env::var("SUPERSONIC_QWEN35_DUMP_LINEAR_LAYER")
        .ok()?
        .parse::<usize>()
        .ok()?;
    if want != idx {
        return None;
    }
    env::var_os("SUPERSONIC_QWEN35_DUMP_DECODE_HIDDENS_DIR").map(std::path::PathBuf::from)
}

fn dump_buf_as_f32(dir: &std::path::Path, name: &str, buf: &GpuBuffer, cols: usize) -> Result<()> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("decode dump {name} D2H: {e}"))?;
    let mut f32s = match buf.dtype() {
        ScalarType::F32 => decode_f32_le(&bytes),
        ScalarType::BF16 => decode_bf16_le_host(&bytes),
        other => {
            return Err(anyhow::anyhow!(
                "decode dump {name}: unsupported dtype {other:?}"
            ));
        }
    };
    if f32s.len() > cols {
        f32s.truncate(cols);
    }
    if f32s.len() > 3994 {
        eprintln!("[dump] {name} n={} dim3994={:.6}", f32s.len(), f32s[3994]);
    }
    let out: Vec<u8> = f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32")), out)
        .map_err(|e| anyhow::anyhow!("write decode {name}: {e}"))
}

fn checksum_kv_prefix_and_slot(
    buf: &GpuBuffer,
    nkv: usize,
    cap: usize,
    hd: usize,
    prefix_len: usize,
    slot: usize,
) -> anyhow::Result<(u64, u64, String)> {
    let bytes = buf.to_host_bytes()?;
    let elem = 2usize;
    let head_stride = cap * hd * elem;
    let mut prefix = Vec::with_capacity(nkv * prefix_len.min(cap) * hd * elem);
    for h in 0..nkv {
        let start = h * head_stride;
        let end = start + prefix_len.min(cap) * hd * elem;
        if end <= bytes.len() {
            prefix.extend_from_slice(&bytes[start..end]);
        }
    }
    let mut slot_bytes = Vec::new();
    let mut slot_head = String::from("na");
    if slot < cap {
        for h in 0..nkv {
            let start = h * head_stride + slot * hd * elem;
            let end = start + hd * elem;
            if end <= bytes.len() {
                slot_bytes.extend_from_slice(&bytes[start..end]);
                if h == 0 {
                    slot_head = bytes[start..start + (8 * elem).min(hd * elem)]
                        .chunks_exact(2)
                        .map(|c| format!("{:.4}", half::bf16::from_le_bytes([c[0], c[1]]).to_f32()))
                        .collect::<Vec<_>>()
                        .join(",");
                }
            }
        }
    }
    Ok((
        fnv1a64_bytes(&prefix),
        fnv1a64_bytes(&slot_bytes),
        slot_head,
    ))
}

fn maybe_dump_gqh_decode_state(
    state: &ModelState,
    config: &TextConfig,
    hidden_io: Option<&GpuBuffer>,
    when: &str,
    token_id: u32,
    seqlen_offset: usize,
    path: &str,
) {
    if !gqh_state_dump_enabled() {
        return;
    }
    static DUMPS: AtomicU32 = AtomicU32::new(0);
    if DUMPS.fetch_add(1, Ordering::Relaxed) >= 12 {
        return;
    }
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let mut full_idx = None;
    let mut lin_idx = None;
    for (i, ls) in state.layers.iter().enumerate() {
        if full_idx.is_none() && config.is_full_attention(i) {
            full_idx = Some(i);
        }
        if lin_idx.is_none() && !config.is_full_attention(i) {
            lin_idx = Some(i);
        }
        if full_idx.is_some() && lin_idx.is_some() {
            break;
        }
    }
    let mut msg =
        format!("[gqh-dump] path={path} when={when} token={token_id} seqlen={seqlen_offset}");
    if let Some(idx) = full_idx {
        let ls = &state.layers[idx];
        let cap = ls.kv_capacity();
        msg.push_str(&format!(
            " full_l{idx} kv_filled={} kv_cap={cap}",
            ls.kv_filled
        ));
        if let (Some(k), Some(v)) = (ls.kv_cache_k.as_ref(), ls.kv_cache_v.as_ref()) {
            match (
                checksum_kv_prefix_and_slot(k, nkv, cap, hd, ls.kv_filled, seqlen_offset),
                checksum_kv_prefix_and_slot(v, nkv, cap, hd, ls.kv_filled, seqlen_offset),
            ) {
                (Ok((k_pre, k_slot, k_head)), Ok((v_pre, v_slot, v_head))) => {
                    msg.push_str(&format!(
                        " k_pre={k_pre:016x} k_slot={k_slot:016x} k0=[{k_head}] v_pre={v_pre:016x} v_slot={v_slot:016x} v0=[{v_head}]"
                    ));
                }
                (k_res, v_res) => {
                    msg.push_str(&format!(" kv_err={k_res:?}/{v_res:?}"));
                }
            }
        } else {
            msg.push_str(" kv=missing");
        }
    }
    if let Some(idx) = lin_idx {
        let ls = &state.layers[idx];
        let conv = ls
            .conv_state
            .as_ref()
            .and_then(|b| b.to_host_bytes().ok())
            .map(|b| fnv1a64_bytes(&b));
        let rec = ls
            .recurrent_state
            .as_ref()
            .and_then(|b| b.to_host_bytes().ok())
            .map(|b| fnv1a64_bytes(&b));
        match (conv, rec) {
            (Some(c), Some(r)) => msg.push_str(&format!(" lin_l{idx} conv={c:016x} rec={r:016x}")),
            _ => msg.push_str(&format!(" lin_l{idx} conv={conv:?} rec={rec:?}")),
        }
    }
    if let Some(hidden) = hidden_io {
        match hidden.to_host_bytes() {
            Ok(bytes) => msg.push_str(&format!(" hidden={:016x}", fnv1a64_bytes(&bytes))),
            Err(e) => msg.push_str(&format!(" hidden_err={e}")),
        }
    }
    eprintln!("{msg}");
}

fn lm_head_lowbit(
    ordinal: usize,
    m: usize,
    vocab_size: usize,
    hidden_dim: usize,
    lhs: &GpuBuffer,
    weights: &Qwen35Weights,
    out: &mut GpuBuffer,
    label: &str,
) -> Result<bool> {
    let Some((qtype, scale, zero)) = weights.lm_head_lowbit_params(hidden_dim) else {
        return Ok(false);
    };
    if qwen35::weights::is_gqh_qtype(qtype) {
        qwen35::weights::matmul_gqh(
            ordinal,
            m,
            vocab_size,
            hidden_dim,
            lhs,
            &*weights.lm_head,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} gqh matmul: {e}"))?;
        return Ok(true);
    }
    if qwen35::weights::is_mix_qtype(qtype) {
        qwen35::weights::matmul_mix(
            ordinal,
            m,
            vocab_size,
            hidden_dim,
            lhs,
            &*weights.lm_head,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} mix matmul: {e}"))?;
        return Ok(true);
    }
    kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
        ordinal,
        1,
        m,
        vocab_size,
        hidden_dim,
        lhs,
        &*weights.lm_head,
        scale,
        zero,
        weights.lm_head_awq_inv_scale.as_ref(),
        weights.int4_group_size,
        qtype,
        out,
    )
    .map_err(|e| anyhow::anyhow!("{label} int4 matmul: {e}"))?;
    Ok(true)
}

fn matmul_proj(
    ordinal: usize,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs: &GpuBuffer,
    weight: &GpuBuffer,
    scale: Option<&GpuBuffer>,
    int8_scale: Option<&GpuBuffer>,
    block_size: usize,
    out: &mut GpuBuffer,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    int4_awq_inv_scale: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<()> {
    let qtype = qwen35::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    if qwen35::weights::is_gqh_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("GQH matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen35::weights::matmul_gqh(ordinal, m, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_gqh: {e}"));
    }
    if qwen35::weights::is_mix_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("mix matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen35::weights::matmul_mix(ordinal, m, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_mix: {e}"));
    }
    if qtype != 0 {
        let sc = int4_scale.unwrap_or(weight);
        let zr = int4_zero.unwrap_or(weight);
        kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
            ordinal,
            batch,
            m,
            n,
            k,
            lhs,
            weight,
            sc,
            zr,
            int4_awq_inv_scale,
            int4_group_size,
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
    } else if int8_scale.is_some() {
        anyhow::bail!("integer activation scales are not supported by the HIP Qwen3.8 path")
    } else {
        match scale {
            Some(s) => kernel_ffi::prefill_ffi::matmul_rhs_transposed_fp8(
                ordinal, batch, m, n, k, lhs, weight, s, block_size, out,
            )
            .map_err(|e| anyhow::anyhow!("matmul_fp8: {e}")),
            None => kernel_ffi::prefill_ffi::matmul_rhs_transposed(
                ordinal,
                ScalarType::BF16,
                batch,
                m,
                n,
                k,
                lhs,
                weight,
                out,
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
    kernel_ffi::prefill_ffi::element_add(ordinal, ScalarType::BF16, total_elems, lhs, src, dst)
        .map_err(|e| anyhow::anyhow!("residual_add failed: {e}"))?;
    Ok(())
}

fn rms_norm_rows_model(
    config: &TextConfig,
    ordinal: usize,
    rows: usize,
    cols: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    output: &mut GpuBuffer,
    label: &str,
) -> Result<()> {
    let op = if config.rms_norm_add_unit_offset {
        kernel_ffi::prefill_ffi::rms_norm_rows
    } else {
        kernel_ffi::prefill_ffi::rms_norm_rows_plain
    };
    op(
        ordinal,
        ScalarType::BF16,
        rows,
        cols,
        config.rms_norm_eps as f32,
        input,
        weight,
        output,
    )
    .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    Ok(())
}

fn maybe_attn_rms_norm_rows(
    config: &TextConfig,
    ordinal: usize,
    rows: usize,
    cols: usize,
    input: &GpuBuffer,
    weight: Option<&GpuBuffer>,
    output: &mut GpuBuffer,
    label: &str,
) -> Result<()> {
    if let Some(weight) = weight {
        let op = if config.rms_norm_add_unit_offset {
            kernel_ffi::prefill_ffi::rms_norm_rows
        } else {
            kernel_ffi::prefill_ffi::rms_norm_rows_plain
        };
        op(
            ordinal,
            ScalarType::BF16,
            rows,
            cols,
            1e-6,
            input,
            weight,
            output,
        )
        .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    } else {
        gpu_hal::copy_d2d(
            ordinal,
            output.as_mut_ptr(),
            input.as_ptr(),
            rows * cols * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("{label} copy-through: {e}"))?;
    }
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
    if sign != 0 {
        -val
    } else {
        val
    }
}

fn f32_to_bf16_bytes_host(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
        .collect()
}

fn is_qwen35_4b_shape(config: &TextConfig) -> bool {
    config.hidden_size == 2560
        && config.intermediate_size == 9216
        && config.num_hidden_layers == 32
        && config.num_attention_heads == 16
        && config.num_key_value_heads == 4
}

pub struct MtpSpecRound {
    pub emitted: Vec<u32>,
    pub next_token: u32,
    pub n_drafted: usize,
    pub n_accepted: usize,
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
    /// F32 lm_head scratch for HIP greedy: skip the 248k F32→BF16 store.
    logits_f32_buf: GpuBuffer,
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
    /// Optional total decode context reservation for preallocated KV storage.
    decode_context_limit: Option<usize>,
    /// Batch size (1 = single-sequence, default).
    batch_size: usize,
    /// Cached workspace for the Qwen3.8 MTP fused verifier.
    /// The fused verify path runs the persistent 4B megakernel with
    /// `batch_size = B` while the live engine is constructed with
    /// `batch_size = 1`; the cache owns a B-sized workspace + IO buffers
    /// + batch-seq desc table so the per-round allocation cost is paid
    /// only once per fused-verify call chain. Re-allocated if the block
    /// size changes between calls.
    mtp_verify_cache: Option<MtpVerifyCache>,
    /// Cached scratch for the prefill-append Qwen3.8 MTP verifier. Reusing this
    /// avoids
    /// re-allocating the prefill component scratch every segment.
    mtp_prefill_append_cache: Option<MtpPrefillAppendCache>,
    /// Reusable scratch for the MTP component decode. Lazily allocated on
    /// the first incremental decode step; carries the BF16 inter-chunk linear-attention
    /// buffers across decode steps.
    mtp_verify_scratch: Option<MtpVerifyScratch>,
    /// When true, run the Qwen3.8 NextN head after each greedy token and log
    /// draft vs next greedy. Does not change the emitted token stream.
    mtp_diag: bool,
    /// When true, consume NextN drafts via prefill-append verify (usable spec).
    mtp_spec: bool,
    mtp_k: usize,
    mtp_h: Option<GpuBuffer>,
    mtp_h_tmp: Option<GpuBuffer>,
    mtp_pending_draft: Option<u32>,
    mtp_diag_hits: u32,
    mtp_diag_total: u32,
    mtp_spec_rounds: u32,
    mtp_spec_emitted: u32,
    mtp_force_seq: bool,
    /// After a 0-accept block, finish the request with sequential decode.
    #[allow(dead_code)]
    mtp_seq_rest: bool,
    /// Last fused-verify RMSNorm rows (BF16 `[B, hidden]`), embeddings_nextn.
    fused_last_normed: Option<Vec<u8>>,
    /// Reusable pre-verify linear snapshot. Avoids per-round clone alloc.
    mtp_linear_snap: Option<LinearStateSnapshot>,
}

pub struct DecodeEngineSnapshot {
    state: ModelState,
    pub logits: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct DecodeEngineDiskSnapshot {
    state: ModelStateDiskSnapshot,
    logits: Vec<f32>,
}

impl DecodeEngineSnapshot {
    pub fn resident_bytes(&self) -> usize {
        self.state
            .resident_gpu_bytes()
            .saturating_add(self.logits.len().saturating_mul(std::mem::size_of::<f32>()))
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            state: self
                .state
                .clone_gpu()
                .context("clone Qwen prefix snapshot")?,
            logits: self.logits.clone(),
        })
    }

    pub fn to_disk_bytes(&self) -> Result<Vec<u8>> {
        let disk = DecodeEngineDiskSnapshot {
            state: self
                .state
                .to_disk_snapshot()
                .context("snapshot Qwen state to disk")?,
            logits: self.logits.clone(),
        };
        serde_json::to_vec(&disk).map_err(Into::into)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeSamplingMode {
    HostLogits,
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

impl DecodeEngine {
    pub fn scratch_debug_ptr(&self) -> usize {
        self.scratch.workspace.as_ptr() as usize
    }

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
                Self::assemble_full_attention_prefix_cache_bf16_host_static(
                    config, state, layer_idx,
                )?;
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

            // One-shot scratch: written once via H2D, DMA'd into shadow_k/v
            // exactly once, then dropped at end of this iteration. No
            // re-read, so GPU L2 is irrelevant — Scratch lets gfx1150 skip
            // the H2D driver call (host-mapped pointer = host data target).
            let tmp_k = GpuBuffer::from_host_bytes_with_kind(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_k_host,
                gpu_hal::BufferKind::Scratch,
            )
            .map_err(|e| anyhow::anyhow!("layer {layer_idx} shadow K H2D: {e}"))?;
            let tmp_v = GpuBuffer::from_host_bytes_with_kind(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, prefix_len, head_dim],
                &prefix_v_host,
                gpu_hal::BufferKind::Scratch,
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
            ls.kv_shadow_start = kv_fp8_bf16_sidecar_window_tokens()
                .map(|window| prefix_len.saturating_sub(window))
                .unwrap_or(0);
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

        let cap = ls.kv_capacity();

        if let (Some(scale_k), Some(scale_v)) = (ls.kv_scale_k.as_ref(), ls.kv_scale_v.as_ref()) {
            let cache_k = ls
                .kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing FP8 K cache"))?;
            let cache_v = ls
                .kv_cache_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing FP8 V cache"))?;
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
        } else if ls.has_virtual_kv_cache() {
            let packed_kv = ls
                .virtual_kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing virtual K cache"))?;
            let k_bytes = packed_kv
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {layer_idx} virtual BF16 K cache D2H: {e}"))?;
            let v_bytes = if let Some(v_cache) = ls.virtual_kv_cache_v.as_ref() {
                v_cache.to_host_bytes().map_err(|e| {
                    anyhow::anyhow!("layer {layer_idx} virtual BF16 V cache D2H: {e}")
                })?
            } else {
                k_bytes.clone()
            };
            let v_base = if ls.virtual_kv_cache_v.is_some() {
                0
            } else {
                k_bytes.len() / 2
            };
            let src_head_stride = cap * head_dim * elem_bytes;
            let dst_head_stride = prefix_len * head_dim * elem_bytes;
            let copy_bytes = prefix_len * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                let src = h * src_head_stride;
                let dst = h * dst_head_stride;
                out_k[dst..dst + copy_bytes].copy_from_slice(&k_bytes[src..src + copy_bytes]);
                out_v[dst..dst + copy_bytes]
                    .copy_from_slice(&v_bytes[v_base + src..v_base + src + copy_bytes]);
            }
        } else {
            let cache_k = ls
                .kv_cache_k
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing BF16 K cache"))?;
            let cache_v = ls
                .kv_cache_v
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing BF16 V cache"))?;
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
        Self::assemble_full_attention_prefix_cache_bf16_host_static(
            &self.weights.config,
            state,
            layer_idx,
        )
    }

    pub fn full_attention_prefix_cache_bf16_host(
        &self,
        layer_idx: usize,
        batch_index: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        let state = self.state_for_batch(batch_index);
        self.assemble_full_attention_prefix_cache_bf16_host_for_state(state, layer_idx)
    }

    pub fn full_attention_prefix_cache_snapshots_bf16_host(
        &self,
    ) -> Result<Vec<(usize, Vec<u8>, Vec<u8>, usize)>> {
        let mut snapshots = Vec::new();
        for idx in 0..self.weights.config.num_hidden_layers {
            if self.weights.config.is_full_attention(idx) {
                let (k, v, len) = self.full_attention_prefix_cache_bf16_host(idx, 0)?;
                snapshots.push((idx, k, v, len));
            }
        }
        Ok(snapshots)
    }

    pub fn full_attention_cache_step_bytes(
        &self,
        layer_idx: usize,
        batch_index: usize,
        seq_pos: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let config = self.weights.config.clone();
        let ls = self
            .state_for_batch(batch_index)
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} out of range"))?;
        let cache_k = ls
            .kv_cache_k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing K cache"))?;
        let cache_v = ls
            .kv_cache_v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer_idx} missing V cache"))?;
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

    fn apply_oracle_hidden(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;

        let hidden_b64 = oracle
            .prefill_hidden
            .as_ref()
            .context("oracle output missing prefill_hidden (use --emit-state)")?;
        let hidden_bytes = b64
            .decode(hidden_b64)
            .context("decode prefill_hidden base64")?;
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
        self.hidden_io =
            GpuBuffer::from_host_bytes(self.ordinal, ScalarType::BF16, hidden_shape, actual_hidden)
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
            let bytes = b64
                .decode(&rs.data)
                .context("decode recurrent_state base64")?;
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
        let rotary =
            RotaryTables::build(config, ordinal).map_err(|e| anyhow::anyhow!("rotary: {e}"))?;
        Self::new_with_rotary(
            weights,
            rotary,
            ordinal,
            proj_buf_floats,
            attn_scratch_floats,
            kv_chunk_size,
            use_4b_kernel,
            prefill_chunk_size,
            kv_fp8,
            batch_size,
        )
    }

    pub fn new_with_rotary(
        weights: Qwen35Weights,
        rotary: RotaryTables,
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
        let hidden_io = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("hidden_io: {e}"))?;
        let normed_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.hidden_size],
        )
        .map_err(|e| anyhow::anyhow!("normed_buf: {e}"))?;
        let logits_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[batch_size, 1, config.vocab_size],
        )
        .map_err(|e| anyhow::anyhow!("logits_buf: {e}"))?;
        let logits_f32_buf = GpuBuffer::zeros(
            ordinal,
            ScalarType::F32,
            &[batch_size, 1, config.vocab_size],
        )
        .map_err(|e| anyhow::anyhow!("logits_f32_buf: {e}"))?;
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
            logits_f32_buf,
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
            decode_context_limit: None,
            batch_size,
            mtp_verify_cache: None,
            mtp_prefill_append_cache: None,
            mtp_verify_scratch: None,
            mtp_diag: env::var_os("SUPERSONIC_QWEN38_MTP").is_some(),
            mtp_spec: false,
            mtp_k: env::var("SUPERSONIC_QWEN38_MTP_K")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&k| k > 0)
                .unwrap_or(2),
            mtp_h: None,
            mtp_h_tmp: None,
            mtp_pending_draft: None,
            mtp_diag_hits: 0,
            mtp_diag_total: 0,
            mtp_spec_rounds: 0,
            mtp_spec_emitted: 0,
            mtp_force_seq: false,
            mtp_seq_rest: false,
            fused_last_normed: None,
            mtp_linear_snap: None,
        })
    }

    pub fn weights(&self) -> &Qwen35Weights {
        &self.weights
    }

    pub fn set_mtp_diag(&mut self, on: bool) {
        self.mtp_diag = on;
    }

    pub fn set_mtp_spec(&mut self, on: bool) {
        self.mtp_spec = on;
        if on {
            self.mtp_diag = false;
        }
    }

    pub fn mtp_spec_enabled(&self) -> bool {
        self.mtp_spec && self.weights.mtp.is_some()
    }

    pub fn mtp_diag_summary(&self) -> Option<(u32, u32)> {
        if (self.mtp_diag || self.mtp_spec) && self.mtp_diag_total > 0 {
            Some((self.mtp_diag_hits, self.mtp_diag_total))
        } else {
            None
        }
    }

    pub fn mtp_spec_summary(&self) -> Option<(u32, u32, u32, u32)> {
        if self.mtp_spec && self.mtp_spec_rounds > 0 {
            Some((
                self.mtp_diag_hits,
                self.mtp_diag_total,
                self.mtp_spec_rounds,
                self.mtp_spec_emitted,
            ))
        } else {
            None
        }
    }

    pub fn seed_mtp_h_from_normed(&mut self, bytes: &[u8]) -> Result<()> {
        let hidden = self.weights.config.hidden_size;
        let need = hidden * ScalarType::BF16.size_in_bytes();
        anyhow::ensure!(
            bytes.len() == need,
            "mtp h seed {} bytes, expected {need}",
            bytes.len()
        );
        if self.mtp_h.is_none() {
            self.mtp_h = Some(
                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden])
                    .map_err(|e| anyhow::anyhow!("mtp_h alloc: {e}"))?,
            );
        }
        if self.mtp_h_tmp.is_none() {
            self.mtp_h_tmp = Some(
                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden])
                    .map_err(|e| anyhow::anyhow!("mtp_h_tmp alloc: {e}"))?,
            );
        }
        let dst = self.mtp_h.as_mut().unwrap();
        gpu_hal::copy_h2d(
            self.ordinal,
            dst.as_mut_ptr(),
            bytes.as_ptr() as *const std::ffi::c_void,
            need,
        )
        .map_err(|e| anyhow::anyhow!("mtp h seed h2d: {e}"))?;
        Ok(())
    }

    fn fused_verify_max_batch(&self) -> usize {
        const MAX_INPUT_CACHE_FLOATS: usize = 15872;
        (MAX_INPUT_CACHE_FLOATS / self.weights.config.hidden_size.max(1)).max(1)
    }

    /// One NextN speculative round: draft up to K tokens (capped so the
    /// verify block fits fused-decode LDS), then one B-token fused trunk
    /// verify. Full accept keeps the fused linear/KV writes; partial
    /// restore + sequential replay of the committed prefix.
    pub fn run_mtp_spec_round(
        &mut self,
        first_token: u32,
        pos: usize,
        remaining: usize,
    ) -> Result<MtpSpecRound> {
        anyhow::ensure!(self.mtp_spec_enabled(), "MTP spec is not enabled");
        anyhow::ensure!(remaining > 0, "mtp spec round with remaining=0");
        if self.mtp_h.is_none() {
            anyhow::bail!("mtp spec round missing seeded h_nextn");
        }
        let max_b = self
            .fused_verify_max_batch()
            .min(kernel_ffi::MAX_BATCH_SIZE);
        let k = self
            .mtp_k
            .min(remaining.saturating_sub(1))
            .min(max_b.saturating_sub(1));
        let mtp_kv_start = self.state.mtp.as_ref().map(|ls| ls.kv_filled).unwrap_or(0);

        if self.mtp_force_seq {
            self.mtp_force_seq = false;
            return self.run_mtp_spec_round_sequential(
                first_token,
                pos,
                remaining,
                Vec::new(),
                mtp_kv_start,
                0,
            );
        }
        let drafts = if k == 0 {
            Vec::new()
        } else {
            let t_draft = Instant::now();
            let d = self.mtp_draft_chain(first_token, pos, k)?;
            if env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE").is_some() {
                eprintln!(
                    "[qwen38-mtp-profile] draft k={} {:.1}ms",
                    d.len(),
                    t_draft.elapsed().as_secs_f64() * 1000.0
                );
            }
            d
        };
        if drafts.is_empty() {
            return self.run_mtp_spec_round_sequential(
                first_token,
                pos,
                remaining,
                drafts,
                mtp_kv_start,
                k,
            );
        }

        let mut block = Vec::with_capacity(drafts.len() + 1);
        block.push(first_token);
        block.extend_from_slice(&drafts);
        if block.len() > remaining {
            block.truncate(remaining);
        }

        self.ensure_mtp_linear_snap()?;
        let t0 = Instant::now();
        let use_fused = block.len() <= max_b;
        let (emitted, next_token, n_acc) = if use_fused {
            match self.verify_block_fused_decode_greedy(&block, pos) {
                Ok(greedy) => {
                    let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    self.mtp_accept_fused_or_replay(
                        &block, &drafts, &greedy, remaining, pos, verify_ms,
                    )?
                }
                Err(err) => {
                    let msg = err.to_string();
                    if msg.contains("shared-memory budget exceeded") {
                        return self.run_mtp_spec_round_sequential(
                            first_token,
                            pos,
                            remaining,
                            drafts,
                            mtp_kv_start,
                            k,
                        );
                    }
                    return Err(err);
                }
            }
        } else {
            let result = self.verify_block_prefill_append_impl(&block, pos, true, None)?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let greedy = result
                .target_next
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("mtp append verify missing greedy target_next"))?;
            self.mtp_accept_append_or_replay(
                &block,
                &drafts,
                greedy,
                result.normed_rows.as_deref(),
                remaining,
                pos,
                verify_ms,
            )?
        };

        if k > 0 {
            if let Some(ls) = self.state.mtp.as_mut() {
                ls.set_kv_filled(mtp_kv_start + emitted.len().min(k));
            }
        }
        self.mtp_spec_rounds += 1;
        self.mtp_spec_emitted += emitted.len() as u32;
        self.mtp_diag_total += drafts.len() as u32;
        self.mtp_diag_hits += n_acc as u32;
        Ok(MtpSpecRound {
            emitted,
            next_token,
            n_drafted: drafts.len(),
            n_accepted: n_acc,
        })
    }

    fn ensure_mtp_linear_snap(&mut self) -> Result<()> {
        if self.mtp_linear_snap.is_none() {
            self.mtp_linear_snap = Some(
                self.state
                    .snapshot_linear()
                    .map_err(|e| anyhow::anyhow!("mtp snapshot_linear: {e}"))?,
            );
            return Ok(());
        }
        let ordinal = self.ordinal;
        let snap = self.mtp_linear_snap.as_mut().unwrap();
        self.state
            .snapshot_linear_into(snap, ordinal)
            .map_err(|e| anyhow::anyhow!("mtp snapshot_linear_into: {e}"))
    }

    fn seed_mtp_from_fused_normed(&mut self, commit_len: usize) -> Result<()> {
        let Some(normed) = self.fused_last_normed.clone() else {
            return Ok(());
        };
        let hidden = self.weights.config.hidden_size;
        let row_bytes = hidden * ScalarType::BF16.size_in_bytes();
        let start = commit_len.saturating_sub(1) * row_bytes;
        let end = start + row_bytes;
        if end <= normed.len() {
            self.seed_mtp_h_from_normed(&normed[start..end])?;
        }
        Ok(())
    }

    fn mtp_commit_prefix(
        &mut self,
        block: &[u32],
        greedy: &[u32],
        commit_len: usize,
        n_acc: usize,
        pos: usize,
        fused: bool,
    ) -> Result<(Vec<u32>, u32, usize)> {
        self.commit_fused_kv_filled(pos + commit_len);
        if fused {
            self.seed_mtp_from_fused_normed(commit_len)?;
        }
        Ok((block[..commit_len].to_vec(), greedy[commit_len - 1], n_acc))
    }

    fn mtp_accept_fused_or_replay(
        &mut self,
        block: &[u32],
        drafts: &[u32],
        greedy: &[u32],
        remaining: usize,
        pos: usize,
        verify_ms: f64,
    ) -> Result<(Vec<u32>, u32, usize)> {
        anyhow::ensure!(
            greedy.len() == block.len(),
            "fused verify greedy {} != block {}",
            greedy.len(),
            block.len()
        );
        let mut n_acc = 0usize;
        while n_acc < drafts.len() && n_acc + 1 < greedy.len() && greedy[n_acc] == drafts[n_acc] {
            n_acc += 1;
        }
        let commit_len = (n_acc + 1).min(block.len()).min(remaining);
        if env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE").is_some() {
            eprintln!(
                "[qwen38-mtp-profile] fused B={} verify={:.1}ms commit={} n_acc={}",
                block.len(),
                verify_ms,
                commit_len,
                n_acc
            );
        }
        if commit_len == block.len() {
            return self.mtp_commit_prefix(block, greedy, commit_len, n_acc, pos, true);
        }
        match restore_linear_prefix(commit_len) {
            Ok(true) => {
                if env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE").is_some() {
                    eprintln!(
                        "[qwen38-mtp-profile] prefix-restore commit={} skip-replay",
                        commit_len
                    );
                }
                self.mtp_commit_prefix(block, greedy, commit_len, n_acc, pos, true)
            }
            Ok(false) | Err(_) => self.mtp_replay_committed_prefix(&block[..commit_len], pos),
        }
    }

    fn mtp_accept_append_or_replay(
        &mut self,
        block: &[u32],
        drafts: &[u32],
        greedy: &[u32],
        normed_rows: Option<&[u8]>,
        remaining: usize,
        pos: usize,
        verify_ms: f64,
    ) -> Result<(Vec<u32>, u32, usize)> {
        anyhow::ensure!(
            greedy.len() == block.len(),
            "append verify greedy {} != block {}",
            greedy.len(),
            block.len()
        );
        let mut n_acc = 0usize;
        while n_acc < drafts.len() && n_acc + 1 < greedy.len() && greedy[n_acc] == drafts[n_acc] {
            n_acc += 1;
        }
        let commit_len = (n_acc + 1).min(block.len()).min(remaining);
        if env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE").is_some() {
            eprintln!(
                "[qwen38-mtp-profile] append B={} verify={:.1}ms commit={} n_acc={}",
                block.len(),
                verify_ms,
                commit_len,
                n_acc
            );
        }
        if commit_len == block.len() {
            self.commit_fused_kv_filled(pos + commit_len);
            if let Some(normed) = normed_rows {
                let hidden = self.weights.config.hidden_size;
                let row_bytes = hidden * ScalarType::BF16.size_in_bytes();
                let start = (commit_len - 1) * row_bytes;
                let end = start + row_bytes;
                if end <= normed.len() {
                    self.seed_mtp_h_from_normed(&normed[start..end])?;
                }
            }
            Ok((block.to_vec(), greedy[commit_len - 1], n_acc))
        } else {
            self.mtp_replay_committed_prefix(&block[..commit_len], pos)
        }
    }

    fn commit_fused_kv_filled(&mut self, new_len: usize) {
        let config = &self.weights.config;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(new_len);
            }
        }
    }

    fn mtp_replay_committed_prefix(
        &mut self,
        committed: &[u32],
        pos: usize,
    ) -> Result<(Vec<u32>, u32, usize)> {
        anyhow::ensure!(!committed.is_empty(), "mtp replay empty prefix");
        let ordinal = self.ordinal;
        let snap = self
            .mtp_linear_snap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("mtp replay missing linear snapshot"))?;
        restore_linear_state(&mut self.state, snap, ordinal)?;
        self.rewind_full_kv_filled(pos);

        let mut next_token = committed[0];
        for (i, &token) in committed.iter().enumerate() {
            let (sampled, _) = self.decode_step_hip_fast_greedy(token, pos + i)?;
            next_token = sampled;
        }
        self.store_mtp_h_from_residual()?;
        let n_acc = committed.len().saturating_sub(1);
        Ok((committed.to_vec(), next_token, n_acc))
    }

    fn run_mtp_spec_round_sequential(
        &mut self,
        first_token: u32,
        mut pos: usize,
        remaining: usize,
        drafts: Vec<u32>,
        mtp_kv_start: usize,
        k: usize,
    ) -> Result<MtpSpecRound> {
        let mut emitted = Vec::with_capacity(drafts.len() + 1);
        let mut token = first_token;
        let mut n_acc = 0usize;
        let mut next_token = first_token;
        let verify_steps = drafts.len() + 1;
        for i in 0..verify_steps {
            if emitted.len() >= remaining {
                break;
            }
            let (sampled, _) = self.decode_step_hip_fast_greedy(token, pos)?;
            emitted.push(token);
            self.store_mtp_h_from_residual()?;
            pos += 1;
            next_token = sampled;
            if i == drafts.len() {
                break;
            }
            if sampled == drafts[i] {
                n_acc += 1;
                token = sampled;
            } else {
                break;
            }
        }
        if k > 0 {
            if let Some(ls) = self.state.mtp.as_mut() {
                ls.set_kv_filled(mtp_kv_start + emitted.len().min(k));
            }
        }
        self.mtp_spec_rounds += 1;
        self.mtp_spec_emitted += emitted.len() as u32;
        self.mtp_diag_total += drafts.len() as u32;
        self.mtp_diag_hits += n_acc as u32;
        Ok(MtpSpecRound {
            emitted,
            next_token,
            n_drafted: drafts.len(),
            n_accepted: n_acc,
        })
    }

    fn store_mtp_h_from_residual(&mut self) -> Result<()> {
        let hidden = self.weights.config.hidden_size;
        if self.mtp_h.is_none() {
            self.mtp_h = Some(
                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[1, hidden])
                    .map_err(|e| anyhow::anyhow!("mtp_h alloc: {e}"))?,
            );
        }
        let mtp_h = self.mtp_h.as_mut().unwrap();
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            1,
            hidden,
            &self.hidden_io,
            &self.weights.norm_weight,
            mtp_h,
            "mtp spec output_norm",
        )
    }

    fn mtp_draft_chain(&mut self, mut token: u32, abs_pos: usize, k: usize) -> Result<Vec<u32>> {
        let mut scratch = match self.mtp_verify_scratch.take() {
            Some(scratch) => scratch,
            None => MtpVerifyScratch::new(&self.weights.config, self.ordinal)?,
        };
        let mut src = self
            .mtp_h
            .take()
            .ok_or_else(|| anyhow::anyhow!("mtp chain missing h"))?;
        let mut dst = match self.mtp_h_tmp.take() {
            Some(buf) => buf,
            None => GpuBuffer::zeros(
                self.ordinal,
                ScalarType::BF16,
                &[1, self.weights.config.hidden_size],
            )
            .map_err(|e| anyhow::anyhow!("mtp_h_tmp alloc: {e}"))?,
        };
        let mut drafts = Vec::with_capacity(k);
        let mut err = None;
        for i in 0..k {
            match mtp_forward(
                &self.weights,
                &mut self.state,
                &self.rotary,
                &mut scratch,
                &src,
                true,
                token,
                abs_pos + i,
                &mut dst,
                self.ordinal,
                self.kv_chunk_size,
            ) {
                Ok(d) => {
                    drafts.push(d);
                    token = d;
                    std::mem::swap(&mut src, &mut dst);
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        self.mtp_h = Some(src);
        self.mtp_h_tmp = Some(dst);
        self.mtp_verify_scratch = Some(scratch);
        if let Some(e) = err {
            return Err(e);
        }
        Ok(drafts)
    }

    pub fn set_decode_context_limit(&mut self, context_tokens: usize) {
        let context_tokens = context_tokens.max(1);
        self.decode_context_limit = Some(context_tokens);
        self.maybe_enable_virtual_bf16_kv(context_tokens);
    }

    pub fn kv_fp8_enabled(&self) -> bool {
        self.kv_fp8
    }

    pub fn virtual_kv_memory_stats(&self) -> qwen35::state::VirtualKvMemoryStats {
        self.state.virtual_kv_memory_stats()
    }

    pub fn virtual_kv_memory_stats_by_layer(
        &self,
    ) -> Vec<(usize, qwen35::state::VirtualKvMemoryStats)> {
        self.state.virtual_kv_memory_stats_by_layer()
    }

    pub fn evict_virtual_kv_to_host(&mut self) -> Result<()> {
        if std::env::var_os("SUPERSONIC_VMM_KV_RESTORE_TO_VMM").is_some() {
            self.state
                .evict_virtual_kv_to_host(&self.weights.config)
                .map_err(|e| anyhow::anyhow!("evict virtual KV to host: {e}"))
        } else {
            let snapshots = self.full_attention_prefix_cache_snapshots_bf16_host()?;
            self.state
                .evict_virtual_kv_to_host_from_snapshots(&self.weights.config, snapshots)
                .map_err(|e| anyhow::anyhow!("evict virtual KV to host: {e}"))
        }
    }

    pub fn restore_virtual_kv_from_host(&mut self) -> Result<()> {
        self.state
            .restore_virtual_kv_from_host()
            .map_err(|e| anyhow::anyhow!("restore virtual KV from host: {e}"))
    }

    pub fn restore_virtual_kv_from_host_to_vmm(&mut self) -> Result<()> {
        self.state
            .restore_virtual_kv_from_host_to_vmm()
            .map_err(|e| anyhow::anyhow!("restore virtual KV from host to VMM: {e}"))
    }

    fn maybe_enable_virtual_bf16_kv(&mut self, context_tokens: usize) {
        let env = std::env::var("SUPERSONIC_VMM_KV").ok();
        if env.as_deref() == Some("0") {
            return;
        }
        if self.kv_fp8 || self.batch_size != 1 {
            return;
        }
        let backend = self.hidden_io.backend();
        if self.use_4b_kernel {
            if env.as_deref() == Some("1") {
                eprintln!(
                    "[vmm] requested by SUPERSONIC_VMM_KV=1 but backend={backend} device={} is using the 4B/component decode path, which does not support virtual KV yet; using dense KV allocator",
                    self.ordinal
                );
            }
            return;
        }
        if backend != gpu_hal::Backend::Hip && env.as_deref() != Some("1") {
            return;
        }
        let supported = gpu_hal::vmm_is_supported(backend, self.ordinal);
        if !supported {
            if env.as_deref() == Some("1") {
                eprintln!(
                    "[vmm] requested by SUPERSONIC_VMM_KV=1 but backend={backend} device={} does not support VMM; using dense KV allocator",
                    self.ordinal
                );
            }
            return;
        }
        self.state
            .enable_virtual_bf16_kv(&self.weights.config, context_tokens);
        eprintln!(
            "[vmm] Qwen3.5 BF16 dense KV uses reserved virtual memory for {} tokens",
            context_tokens
        );
    }

    /// Verify the engine's attn_scratch budget covers the current largest
    /// `kv_max_t` across all full-attention layers (of every batch item).
    /// The 4B persistent decode kernel writes `saved_q+gate+pre_gate+scores`
    /// into attn_scratch; `saved_scores` is indexed `[qh * kv_max_b + t]`.
    fn check_attn_scratch_budget(&self) -> Result<()> {
        if !self.use_4b_kernel {
            return Ok(());
        }
        let config = &self.weights.config;
        let nh = config.num_attention_heads;
        let hd = config.head_dim;
        let base = 3 * nh * hd;
        let mut max_kv = 0usize;
        for st in std::iter::once(&self.state).chain(self.extra_states.iter()) {
            for ls in &st.layers {
                max_kv = max_kv.max(ls.kv_capacity());
            }
        }
        let required = base + nh * max_kv;
        if required > self.attn_scratch_floats {
            anyhow::bail!(
                "attn_scratch_floats={} too small for kv_max_t={} \
                 (need {} = 3*{nh}*{hd} + {nh}*{max_kv}). \
                 Pass --context-size to budget the run's max context.",
                self.attn_scratch_floats,
                max_kv,
                required,
            );
        }
        Ok(())
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

    pub fn load_prefill_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        self.apply_oracle_hidden(oracle)?;
        self.apply_oracle_full_attention_state(oracle)?;
        self.apply_oracle_linear_attention_state(oracle)?;

        // Convert BF16 KV caches to FP8 if requested
        if self.kv_fp8 {
            prefill_engine::convert_kv_caches_to_fp8(
                &mut self.state,
                &self.weights.config,
                self.ordinal,
            )?;
        }

        // Reset sync counters for fresh kernel launch sequence
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync: {e}"))?;

        Ok(())
    }

    /// Reset per-session state so the engine is ready for a fresh prompt.
    /// Weights, rotary tables, scratch allocations, and quantization scales are
    /// untouched — only KV caches, conv/recurrent state, and the sync counters
    /// are cleared. Used by the HTTP server between requests.
    pub fn reset(&mut self) -> Result<()> {
        self.state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("reset model state: {e}"))?;
        for es in &mut self.extra_states {
            *es = ModelState::new(&self.weights.config, self.ordinal)
                .map_err(|e| anyhow::anyhow!("reset extra state: {e}"))?;
        }
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync: {e}"))?;
        Ok(())
    }

    pub fn snapshot_prefix(&self, logits: Vec<f32>) -> Result<DecodeEngineSnapshot> {
        Ok(DecodeEngineSnapshot {
            state: self
                .state
                .clone_gpu()
                .context("snapshot Qwen prefix state")?,
            logits,
        })
    }

    pub fn prefix_snapshot_bytes(&self, logits_len: usize) -> usize {
        self.state
            .resident_gpu_bytes()
            .saturating_add(logits_len.saturating_mul(std::mem::size_of::<f32>()))
    }

    pub fn restore_prefix(&mut self, snapshot: &DecodeEngineSnapshot) -> Result<Vec<f32>> {
        self.state = snapshot
            .state
            .clone_gpu()
            .context("restore Qwen prefix state")?;
        self.scratch
            .reset_sync()
            .context("reset sync after prefix restore")?;
        Ok(snapshot.logits.clone())
    }

    pub fn restore_prefix_owned(&mut self, snapshot: DecodeEngineSnapshot) -> Result<Vec<f32>> {
        self.state = snapshot.state;
        self.scratch
            .reset_sync()
            .context("reset sync after prefix restore")?;
        Ok(snapshot.logits)
    }

    pub fn load_prefix_snapshot_bytes(&self, bytes: &[u8]) -> Result<DecodeEngineSnapshot> {
        let disk: DecodeEngineDiskSnapshot = serde_json::from_slice(bytes)?;
        Ok(DecodeEngineSnapshot {
            state: ModelState::from_disk_snapshot(disk.state, &self.weights.config, self.ordinal)
                .context("load Qwen prefix snapshot from disk")?,
            logits: disk.logits,
        })
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

    pub fn prefill_native_with_final_norm(
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
            false,
            None,
        )?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;
        Ok(result)
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

    pub fn rebuild_prefill_state_greedy_token(&mut self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.rebuild_prefill_state(token_ids, false)?;
        Ok(Self::greedy_sample(&logits))
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

    fn sync_stage_if_requested(&self, enabled: bool, stage: &str) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        gpu_hal::sync(self.ordinal).map_err(|e| anyhow::anyhow!("{stage} synchronize: {e}"))
    }

    fn decode_step_non_4b(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        _sampling_mode: DecodeSamplingMode,
        sync_for_timing: bool,
    ) -> Result<DecodeStepOutput> {
        let config = &self.weights.config;
        let mut timings = DecodeStageTimings::default();
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights
                .embed_tokens
                .offset_ptr(token_id as usize * row_bytes),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        for (idx, layer) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(idx) {
                layer
                    .ensure_kv_capacity(
                        seqlen_offset,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {idx}: {e}"))?;
            }
        }
        self.check_attn_scratch_budget()?;
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(
                &self.weights.config,
                self.ordinal,
                &mut self.state,
            )?;
        }
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch.upload_descs(&descs)?;
        if let Some(descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch.upload_kv_fp8_descs(&descs)?;
        }
        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )?;
        self.scratch.reset_sync()?;

        let start = Instant::now();
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
        )?;
        self.sync_stage_if_requested(sync_for_timing, "persistent decode")?;
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;

        let filled = seqlen_offset + 1;
        for (idx, layer) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(idx) {
                layer.set_kv_filled(filled);
            }
        }
        let start = Instant::now();
        kernel_ffi::rms_norm(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            config.hidden_size,
        )?;
        self.sync_stage_if_requested(sync_for_timing, "final rms norm")?;
        timings.rms_norm_ms = start.elapsed().as_secs_f64() * 1000.0;

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
        )?;
        self.sync_stage_if_requested(sync_for_timing, "lm head")?;
        timings.lm_head_ms = start.elapsed().as_secs_f64() * 1000.0;
        let bytes = self.logits_buf.to_host_bytes()?;
        let logits: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let sampled_token = Self::greedy_sample(&logits);
        Ok(DecodeStepOutput {
            logits: Some(logits),
            sampled_token,
            timings,
        })
    }

    /// Run one decode step and return logits on CPU. Stage timings are only
    /// populated for the non-4B native decode path.
    pub fn decode_step_with_timings(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        if self.use_4b_kernel {
            return self.decode_step_4b_single_kernel_with_timings(token_id, seqlen_offset);
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::HostLogits,
            true,
        )?;
        let logits = out
            .logits
            .ok_or_else(|| anyhow::anyhow!("decode_step_with_timings missing logits"))?;
        Ok((logits, out.timings))
    }

    fn gqh_component_decode_enabled(&self) -> bool {
        !self.weights.gqh_headers.is_empty()
            && std::env::var_os("SUPERSONIC_QWEN35_GQH_COMPONENT_DECODE").is_some()
    }

    /// Dedicated GQH/ggml-K matvecs plus the slim prefill attention/linear/MLP
    /// cores. The fat persistent 4B kernel cannot occupy well with GQH walkers
    /// inlined; this path matches the 66 ms standalone proj budget plus cores.
    fn decode_step_gqh_component(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(Vec<f32>, DecodeStageTimings)> {
        let mut scratch = match self.mtp_verify_scratch.take() {
            Some(scratch) => scratch,
            None => MtpVerifyScratch::new(&self.weights.config, self.ordinal)?,
        };
        maybe_dump_gqh_decode_state(
            &self.state,
            &self.weights.config,
            Some(&self.hidden_io),
            "before",
            token_id,
            seqlen_offset,
            "component",
        );
        let start = Instant::now();
        let result = mtp_decode_step(
            &self.weights,
            &mut self.state,
            &self.rotary,
            &mut scratch,
            token_id,
            seqlen_offset,
            self.ordinal,
            self.kv_chunk_size,
        );
        let copy = scratch.copy_last_residual_to(self.ordinal, &mut self.hidden_io);
        self.mtp_verify_scratch = Some(scratch);
        copy?;
        let logits = result?;
        maybe_dump_gqh_decode_state(
            &self.state,
            &self.weights.config,
            Some(&self.hidden_io),
            "after",
            token_id,
            seqlen_offset,
            "component",
        );
        let mut timings = DecodeStageTimings::default();
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((logits, timings))
    }

    pub fn decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
        if self.gqh_component_decode_enabled() {
            return Ok(self.decode_step_gqh_component(token_id, seqlen_offset)?.0);
        }
        if self.use_4b_kernel {
            return Ok(self
                .decode_step_4b_single_kernel_with_timings(token_id, seqlen_offset)?
                .0);
        }
        let out = self.decode_step_non_4b(
            token_id,
            seqlen_offset,
            DecodeSamplingMode::HostLogits,
            false,
        )?;
        out.logits
            .ok_or_else(|| anyhow::anyhow!("decode_step missing logits"))
    }

    /// Pay planar→tight convert and HIP graph capture before decode_ms.
    /// llama.cpp GQH is already compact on disk; this is SuperSonic setup.
    pub fn prepare_hip_gqh_decode(&mut self) -> Result<()> {
        if self.hidden_io.backend() != gpu_hal::Backend::Hip || !self.use_4b_kernel {
            return Ok(());
        }
        kernel_ffi::gqh::enable_tight_decode();
        let hidden = self.weights.config.hidden_size;
        let vocab = self.weights.config.vocab_size;
        if let Some((qtype, _, _)) = self.weights.lm_head_lowbit_params(hidden) {
            if let Some(rung) = kernel_ffi::gqh::rung_from_ggml_type(qtype as u32) {
                kernel_ffi::gqh::ensure_tight(
                    self.ordinal,
                    rung,
                    self.weights.lm_head.as_ptr() as *mut _,
                    hidden as i32,
                    vocab as i32,
                )?;
            }
        }
        let seqlen_offset = 0usize;
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("prepare gqh upload descs: {e}"))?;
        kernel_ffi::set_hip_gqh_prepare_only(true);
        let persist = kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            self.weights.config.num_hidden_layers,
            hidden,
            self.weights.config.intermediate_size,
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
            1,
            self.scratch.batch_seq_desc_device.as_ref(),
            self.int4_scale_device.as_ref(),
            false,
            false,
        );
        kernel_ffi::set_hip_gqh_prepare_only(false);
        persist.map_err(|e| anyhow::anyhow!("prepare hip gqh decode: {e}"))?;
        Ok(())
    }
    pub fn decode_step_hip_fast_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        let (logits, timings) = if self.use_4b_kernel {
            self.decode_step_4b_single_kernel_with_timings(token_id, seqlen_offset)?
        } else {
            let output = self.decode_step_non_4b(
                token_id,
                seqlen_offset,
                DecodeSamplingMode::HostLogits,
                false,
            )?;
            (
                output
                    .logits
                    .ok_or_else(|| anyhow::anyhow!("HIP greedy decode missing logits"))?,
                output.timings,
            )
        };
        Ok((Self::greedy_sample(&logits), timings))
    }

    /// Backend the engine is running on. Used by callers that need to pick
    /// between the incremental decode path and replay-prefill path.
    pub fn backend(&self) -> gpu_hal::Backend {
        self.hidden_io.backend()
    }

    /// Replay-prefill decode: runs prefill from scratch over the full
    /// `token_history` (prompt + everything emitted so far, including the
    /// freshly sampled token whose logits we need next), and returns the
    /// last-position logits. O(N²) per generated token but reuses the
    /// validated prefill pipeline. Non-destructive
    /// to engine state (allocates a throwaway `ModelState`).
    pub fn decode_step_replay(&self, token_history: &[u32]) -> Result<Vec<f32>> {
        prefill_engine::gpu_reference_replay_step(
            &self.weights,
            &self.rotary,
            token_history,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.use_4b_kernel,
        )
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
            self.decode_step_batch_impl(&[token_id], seqlen_offset, true, false)?;
        let logits = batch_logits
            .pop()
            .ok_or_else(|| anyhow::anyhow!("single-sequence 4B kernel timings missing logits"))?;
        let sampling_start = Instant::now();
        let _ = Self::greedy_sample(&logits);
        timings.host_sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
        Ok((logits, timings))
    }

    pub fn state_mut(&mut self) -> &mut ModelState {
        &mut self.state
    }

    /// Device ordinal carried by the engine for MTP state helpers.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Rewind every full-attention layer's `kv_filled` cursor to `new_len`
    /// (no-op if already at or below). The physical K/V beyond the cursor is
    /// untouched and will be harmlessly overwritten by subsequent decodes —
    /// used after a partial-acceptance verify to roll the cache logically back
    /// to the committed length.
    pub fn rewind_full_kv_filled(&mut self, new_len: usize) {
        let config = &self.weights.config;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) && ls.kv_filled > new_len {
                ls.set_kv_filled(new_len);
            }
        }
    }

    fn verify_block_prefill_append_impl(
        &mut self,
        tokens: &[u32],
        pos_offset: usize,
        greedy_only: bool,
        greedy_compare_tokens: Option<&[u32]>,
    ) -> Result<prefill_engine::PrefillAppendVerifyResult> {
        if self.kv_fp8 {
            anyhow::bail!("verify_block_prefill_append does not support kv_fp8");
        }
        if tokens.is_empty() {
            anyhow::bail!("verify_block_prefill_append: tokens must be non-empty");
        }

        let max_pos = pos_offset + tokens.len() - 1;
        {
            let config = &self.weights.config;
            for (idx, layer_state) in self.state.layers.iter_mut().enumerate() {
                if config.is_full_attention(idx) {
                    layer_state
                        .ensure_kv_capacity(
                            max_pos,
                            self.ordinal,
                            config,
                            self.kv_chunk_size,
                            false,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("prefill append ensure KV layer {idx}: {e}")
                        })?;
                }
            }
        }

        let mut cache = match self.mtp_prefill_append_cache.take() {
            Some(cache) => cache,
            None => MtpPrefillAppendCache::new(&self.weights.config, tokens.len(), self.ordinal)?,
        };
        let result = prefill_append_verify_cached(
            &self.weights,
            &mut self.state,
            &self.rotary,
            tokens,
            pos_offset,
            self.ordinal,
            self.kv_chunk_size,
            self.use_4b_kernel,
            greedy_only,
            greedy_compare_tokens,
            &mut cache,
        )?;
        self.mtp_prefill_append_cache = Some(cache);
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill append verify: {e}"))?;
        Ok(result)
    }

    /// Qwen3.8 MTP fused verify: single `persistent_decode_4b` megakernel
    /// launch over all `tokens.len()` consecutive positions starting at
    /// `pos_offset`. Returns per-position logits `[tokens.len()][vocab]`.
    ///
    /// The megakernel's batched path already runs `B` batch elements
    /// sequentially on `blockIdx.x == 0` within a single layer iteration
    /// (see `kernels/full_attention_4b.hip` ~4165). Feeding it a
    /// `BatchSeqDesc` whose slots alias one sequence's KV cache with
    /// `seqlen_offset[b] = pos_offset + b` yields the correct causal
    /// in-sequence verify — each position reads the cache written by
    /// prior positions within the same launch.
    ///
    /// Requirements:
    /// * `use_4b_kernel = true` and `batch_size = 1` (engine construction
    ///   is not mutated; a verify-local B-sized cache is used instead).
    /// * `kv_fp8 = false` — fused verify uses BF16 KV like
    ///   `verify_block_prefill`.
    /// * `tokens.len()` must be in `1..=MAX_BATCH_SIZE` (kernel limit).
    ///
    /// Semantics match `verify_block_prefill`: full-attention K/V is
    /// written at positions `[pos_offset, pos_offset + tokens.len())`
    /// but `kv_filled` is NOT advanced on any layer — the MTP driver
    /// owns rollback via `rewind_full_kv_filled` + `restore_linear`.
    /// Linear-attention `conv_state` / `recurrent_state` are mutated in
    /// place (shared across all B slots via pointer aliasing), so the
    /// caller MUST snapshot linear state before this call and restore
    /// it after the accept decision — same snapshot/restore contract
    /// the existing verify paths already require.
    pub fn verify_block_fused_decode(
        &mut self,
        tokens: &[u32],
        pos_offset: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self
            .verify_block_fused_decode_ex(tokens, pos_offset, false)?
            .0)
    }

    fn verify_block_fused_decode_greedy(
        &mut self,
        tokens: &[u32],
        pos_offset: usize,
    ) -> Result<Vec<u32>> {
        Ok(self
            .verify_block_fused_decode_ex(tokens, pos_offset, true)?
            .1)
    }

    fn verify_block_fused_decode_ex(
        &mut self,
        tokens: &[u32],
        pos_offset: usize,
        greedy_only: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
        if !self.use_4b_kernel {
            anyhow::bail!("verify_block_fused_decode requires use_4b_kernel");
        }
        if self.batch_size != 1 {
            anyhow::bail!("verify_block_fused_decode requires engine batch_size=1");
        }
        if self.kv_fp8 {
            anyhow::bail!("verify_block_fused_decode does not support kv_fp8");
        }
        if tokens.is_empty() {
            anyhow::bail!("verify_block_fused_decode: tokens must be non-empty");
        }
        let b = tokens.len();
        if b > kernel_ffi::MAX_BATCH_SIZE {
            anyhow::bail!(
                "verify_block_fused_decode: block size {b} > MAX_BATCH_SIZE {}",
                kernel_ffi::MAX_BATCH_SIZE,
            );
        }

        // Copy out primitive config values up front so the later
        // `self.state.layers.iter_mut()` borrow doesn't fight with
        // `&self.weights.config` reads.
        let (hidden_dim, intermediate_size, vocab_size, num_layers, _rms_norm_eps) = {
            let c = &self.weights.config;
            (
                c.hidden_size,
                c.intermediate_size,
                c.vocab_size,
                c.num_hidden_layers,
                c.rms_norm_eps as f32,
            )
        };

        // The 4B megakernel's shared-memory footprint per workgroup is
        //   (block_size + max(B * hidden_dim, 2 * hidden_dim) + fp8_lut) * sizeof(f32).
        // gfx1100/gfx115x cap LDS at 64 KiB per workgroup, or 16384 floats.
        // Reserve 512 floats for block_size + fp8_lut, leaving 15872 floats
        // for the input cache.
        // If a user passes a larger MTP block, fail before HIP returns a
        // less helpful launch error.
        const MAX_INPUT_CACHE_FLOATS: usize = 15872;
        let input_cache = (b * hidden_dim).max(2 * hidden_dim);
        if input_cache > MAX_INPUT_CACHE_FLOATS {
            let max_b = (MAX_INPUT_CACHE_FLOATS / hidden_dim.max(1)).max(1);
            anyhow::bail!(
                "verify_block_fused_decode: shared-memory budget exceeded \
                 (B={b} * hidden_dim={hidden_dim} = {}, 2 * hidden_dim = {}; \
                 cap = {MAX_INPUT_CACHE_FLOATS} floats). \
                 Lower the speculative block size to <= {}.",
                b * hidden_dim,
                2 * hidden_dim,
                max_b,
            );
        }

        let max_pos = pos_offset + b - 1;

        // Ensure KV capacity on every full-attention layer for the
        // highest position this launch will write.
        {
            let config = &self.weights.config;
            for (i, ls) in self.state.layers.iter_mut().enumerate() {
                if config.is_full_attention(i) {
                    ls.ensure_kv_capacity(
                        max_pos,
                        self.ordinal,
                        config,
                        self.kv_chunk_size,
                        self.kv_fp8,
                    )
                    .map_err(|e| anyhow::anyhow!("fused verify ensure KV layer {i}: {e}"))?;
                }
            }
        }
        self.check_attn_scratch_budget()?;

        // Take the cached workspace if it matches the current block
        // size, otherwise allocate fresh. Put it back at the end.
        let mut cache = match self.mtp_verify_cache.take() {
            Some(c) if c.block_size == b => c,
            _ => MtpVerifyCache::alloc(
                self.ordinal,
                b,
                hidden_dim,
                intermediate_size,
                vocab_size,
                num_layers,
                self.proj_buf_floats,
                self.attn_scratch_floats,
            )?,
        };

        // Layer descs (state pointers are ignored by the kernel when
        // `batch_descs` is non-null — weights + norm pointers still
        // matter). Reuse `self.scratch.desc_device` to avoid a second
        // device allocation; the scratch is not otherwise touched by
        // this method.
        let descs = build_layer_descs(&self.weights, &self.state, pos_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("fused verify upload layer descs: {e}"))?;

        // Shared-cache batch-seq descriptors: all B slots point at
        // `self.state`'s per-layer buffers; `seqlen_offset[b] =
        // pos_offset + b` gives the kernel the unique per-position
        // offset for RoPE + KV append + causal read.
        let state_refs: Vec<&ModelState> = (0..b).map(|_| &self.state).collect();
        let seqlen_offsets: Vec<usize> = (0..b).map(|bi| pos_offset + bi).collect();
        let batch_descs =
            build_batch_seq_descs(&state_refs, &seqlen_offsets, /* kv_fp8 */ false).ok_or_else(
                || anyhow::anyhow!("fused verify: build_batch_seq_descs returned None for B={b}"),
            )?;
        let desc_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                batch_descs.as_ptr() as *const u8,
                batch_descs.len() * std::mem::size_of::<kernel_ffi::BatchSeqDesc>(),
            )
        };
        gpu_hal::copy_h2d(
            self.ordinal,
            cache.batch_desc_device.as_mut_ptr(),
            desc_bytes.as_ptr() as *const c_void,
            desc_bytes.len(),
        )
        .map_err(|e| anyhow::anyhow!("fused verify upload batch-seq descs: {e}"))?;

        // Embedding lookup: gather each token's row into
        // cache.hidden_io[b, 0, :].
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        for (bi, &tid_val) in tokens.iter().enumerate() {
            let src_offset = tid_val as usize * row_bytes;
            let dst_offset = bi * row_bytes;
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (cache.hidden_io.as_ptr() as *mut u8).add(dst_offset) as *mut c_void },
                self.weights.embed_tokens.offset_ptr(src_offset),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("fused verify embedding slot {bi}: {e}"))?;
        }

        gpu_hal::memset_zeros(
            self.ordinal,
            cache.workspace.as_mut_ptr(),
            cache.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("fused verify clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("fused verify reset sync: {e}"))?;

        let profile_verify = std::env::var_os("SUPERSONIC_QWEN38_MTP_PROFILE_VERIFY").is_some();

        // Launch the fused megakernel. `pos_offset` as the kernel's
        // `seqlen_offset` arg is ignored because `batch_descs` is
        // non-null; pass it through for consistency with the batched
        // call site.
        let persistent_start = Instant::now();
        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            hidden_dim,
            intermediate_size,
            pos_offset,
            &self.scratch.desc_device,
            &mut cache.hidden_io,
            &mut cache.workspace,
            &mut self.scratch.sync_buf,
            &self.rotary.cos,
            &self.rotary.sin,
            self.rotary.rotary_dim,
            self.proj_buf_floats,
            self.attn_scratch_floats,
            self.fp8_scale_device.as_ref(),
            None, // kv_fp8_descs: fused verify disallows kv_fp8
            b,
            Some(&cache.batch_desc_device),
            self.int4_scale_device.as_ref(),
            false, // enable_timing_slots
            false, // enable_attention_trace
        )
        .map_err(|e| anyhow::anyhow!("fused verify persistent_decode_4b: {e}"))?;
        let persistent_ms = persistent_start.elapsed().as_secs_f64() * 1000.0;

        // Deliberately do NOT advance `kv_filled` on any layer. The
        // MTP rolls the K/V cursor back via
        // `rewind_full_kv_filled` and the linear state via
        // `restore_linear` after the accept decision.

        // Final RMSNorm (multirow) + tiled lm_head over all B hiddens.
        let rms_start = Instant::now();
        rms_norm_rows_model(
            &self.weights.config,
            self.ordinal,
            b,
            hidden_dim,
            &cache.hidden_io,
            &self.weights.norm_weight,
            &mut cache.normed_buf,
            "fused verify final rms_norm",
        )?;
        let rms_ms = rms_start.elapsed().as_secs_f64() * 1000.0;

        let lm_head_start = Instant::now();
        if lm_head_lowbit(
            self.ordinal,
            b,
            vocab_size,
            hidden_dim,
            &cache.normed_buf,
            &self.weights,
            &mut cache.logits_buf,
            "fused verify lm_head",
        )? {
        } else {
            kernel_ffi::matmul_rhs_transposed_4b(
                self.ordinal,
                ScalarType::BF16,
                1,
                b,
                vocab_size,
                hidden_dim,
                &cache.normed_buf,
                &*self.weights.lm_head,
                &mut cache.logits_buf,
            )
            .map_err(|e| anyhow::anyhow!("fused verify lm_head matmul: {e}"))?;
        }
        let lm_head_ms = lm_head_start.elapsed().as_secs_f64() * 1000.0;

        let d2h_start = Instant::now();
        let greedy = {
            kernel_ffi::prefill_ffi::argmax_bf16_rows(
                self.ordinal,
                b,
                vocab_size,
                &cache.logits_buf,
                &mut cache.argmax_buf,
            )
            .map_err(|e| anyhow::anyhow!("fused verify gpu argmax: {e}"))?;
            let token_bytes = cache
                .argmax_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("fused verify argmax D2H: {e}"))?;
            anyhow::ensure!(
                token_bytes.len() >= b * 4,
                "fused verify argmax D2H truncated"
            );
            let mut ids = Vec::with_capacity(b);
            for i in 0..b {
                let chunk: [u8; 4] = token_bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("fused verify argmax token truncated"))?;
                ids.push(u32::from_le_bytes(chunk));
            }
            ids
        };
        let mut logits_per_pos = Vec::new();
        let d2h_ms;
        if greedy_only {
            d2h_ms = d2h_start.elapsed().as_secs_f64() * 1000.0;
        } else {
            let logits_host = cache
                .logits_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("fused verify logits D2H: {e}"))?;
            d2h_ms = d2h_start.elapsed().as_secs_f64() * 1000.0;
            let row_stride_bytes = vocab_size * ScalarType::BF16.size_in_bytes();
            logits_per_pos = Vec::with_capacity(b);
            for bi in 0..b {
                let start = bi * row_stride_bytes;
                let end = start + row_stride_bytes;
                let row: Vec<f32> = logits_host[start..end]
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                logits_per_pos.push(row);
            }
        }

        if profile_verify {
            eprintln!(
                "[qwen38-mtp-profile] fused_verify B={b} pos={pos_offset} persistent={persistent_ms:.2}ms rms={rms_ms:.2}ms lm_head={lm_head_ms:.2}ms d2h={d2h_ms:.2}ms greedy_only={greedy_only}"
            );
        }

        let normed_host = cache
            .normed_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("fused verify normed D2H: {e}"))?;
        self.fused_last_normed = Some(normed_host);

        self.mtp_verify_cache = Some(cache);
        Ok((logits_per_pos, greedy))
    }

    /// Greedy argmax over logits.
    pub fn greedy_sample(logits: &[f32]) -> u32 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_idx = idx;
                best_val = val;
            }
        }
        best_idx as u32
    }

    pub fn last_normed_host_f32(&self) -> Result<Vec<f32>> {
        let bytes = self
            .normed_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("normed D2H: {e}"))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }

    /// Copy prefill state from sequence 0 to all extra batch sequences.
    /// Call after load_prefill_state() or prefill_native() to initialize batch items.
    pub fn replicate_state_to_batch(&mut self) -> Result<()> {
        for b in 0..self.extra_states.len() {
            self.extra_states[b] = self
                .state
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
        let (all_logits, _) =
            self.decode_step_batch_impl(token_ids, seqlen_offset, false, false)?;
        Ok(all_logits)
    }

    /// Run one batched decode step and return per-sequence logits plus native
    /// stage timings for the persistent batch path.
    pub fn decode_step_batch_with_timings(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        self.decode_step_batch_impl(token_ids, seqlen_offset, true, false)
    }

    fn decode_step_batch_impl(
        &mut self,
        token_ids: &[u32],
        seqlen_offset: usize,
        enable_timing_slots: bool,
        greedy_argmax: bool,
    ) -> Result<(Vec<Vec<f32>>, DecodeStageTimings)> {
        anyhow::ensure!(
            token_ids.len() == self.batch_size,
            "batch token count must match engine batch size"
        );
        anyhow::ensure!(self.use_4b_kernel, "batched decode requires 4B kernel");
        let config = &self.weights.config;
        let b = self.batch_size;
        let mut timings = DecodeStageTimings::default();
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();

        for (bi, &token_id) in token_ids.iter().enumerate() {
            gpu_hal::copy_d2d(
                self.ordinal,
                unsafe { (self.hidden_io.as_ptr() as *mut u8).add(bi * row_bytes) as *mut c_void },
                self.weights
                    .embed_tokens
                    .offset_ptr(token_id as usize * row_bytes),
                row_bytes,
            )
            .map_err(|e| anyhow::anyhow!("embedding lookup batch {bi}: {e}"))?;
        }

        for bi in 0..b {
            let state = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (layer_idx, layer) in state.layers.iter_mut().enumerate() {
                if config.is_full_attention(layer_idx) {
                    layer
                        .ensure_kv_capacity(
                            seqlen_offset,
                            self.ordinal,
                            config,
                            self.kv_chunk_size,
                            self.kv_fp8,
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("ensure KV batch {bi} layer {layer_idx}: {e}")
                        })?;
                }
            }
        }
        self.check_attn_scratch_budget()?;
        if self.kv_fp8 && kv_fp8_bf16_sidecar_enabled() {
            Self::load_kv_shadow_for_state_static(
                &self.weights.config,
                self.ordinal,
                &mut self.state,
            )?;
            for state in &mut self.extra_states {
                Self::load_kv_shadow_for_state_static(&self.weights.config, self.ordinal, state)?;
            }
        }

        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload decode descriptors: {e}"))?;
        let state_refs: Vec<&ModelState> = std::iter::once(&self.state)
            .chain(self.extra_states.iter())
            .collect();
        let offsets = vec![seqlen_offset; b];
        let batch_descs = build_batch_seq_descs(&state_refs, &offsets, self.kv_fp8)
            .ok_or_else(|| anyhow::anyhow!("build batch sequence descriptors for B={b}"))?;
        self.scratch
            .upload_batch_seq_descs(&batch_descs)
            .map_err(|e| anyhow::anyhow!("upload batch sequence descriptors: {e}"))?;
        if let Some(kv_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_descs)
                .map_err(|e| anyhow::anyhow!("upload KV descriptors: {e}"))?;
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

        let persistent_start = Instant::now();
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
        .map_err(|e| anyhow::anyhow!("persistent decode 4B: {e}"))?;
        timings.persistent_ms = persistent_start.elapsed().as_secs_f64() * 1000.0;

        let filled = seqlen_offset + 1;
        for bi in 0..b {
            let state = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
            for (layer_idx, layer) in state.layers.iter_mut().enumerate() {
                if config.is_full_attention(layer_idx) {
                    layer.set_kv_filled(filled);
                }
            }
        }

        let norm_start = Instant::now();
        rms_norm_rows_model(
            config,
            self.ordinal,
            b,
            config.hidden_size,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
            "final RMSNorm",
        )?;
        timings.rms_norm_ms = norm_start.elapsed().as_secs_f64() * 1000.0;

        if greedy_argmax && b == 1 {
            let lm_start = Instant::now();
            if lm_head_lowbit(
                self.ordinal,
                1,
                config.vocab_size,
                config.hidden_size,
                &self.normed_buf,
                &self.weights,
                &mut self.logits_f32_buf,
                "greedy lm_head",
            )? {
            } else {
                kernel_ffi::matmul_rhs_transposed_4b(
                    self.ordinal,
                    ScalarType::F32,
                    1,
                    1,
                    config.vocab_size,
                    config.hidden_size,
                    &self.normed_buf,
                    &*self.weights.lm_head,
                    &mut self.logits_f32_buf,
                )
                .map_err(|e| anyhow::anyhow!("greedy lm_head: {e}"))?;
            }
            timings.lm_head_ms = lm_start.elapsed().as_secs_f64() * 1000.0;
            let argmax_start = Instant::now();
            kernel_ffi::prefill_ffi::argmax_f32_as_bf16_rows(
                self.ordinal,
                1,
                config.vocab_size,
                &self.logits_f32_buf,
                &mut self.argmax_buf,
            )
            .map_err(|e| anyhow::anyhow!("greedy argmax: {e}"))?;
            let bytes = self.argmax_buf.to_host_bytes()?;
            timings.logits_d2h_ms = argmax_start.elapsed().as_secs_f64() * 1000.0;
            let token = u32::from_le_bytes(
                bytes
                    .get(..4)
                    .ok_or_else(|| anyhow::anyhow!("greedy argmax returned truncated token"))?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("greedy argmax token conversion"))?,
            );
            return Ok((vec![vec![f32::from_bits(token)]], timings));
        }

        let lm_start = Instant::now();
        if lm_head_lowbit(
            self.ordinal,
            b,
            config.vocab_size,
            config.hidden_size,
            &self.normed_buf,
            &self.weights,
            &mut self.logits_buf,
            "lm_head",
        )? {
        } else {
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
            .map_err(|e| anyhow::anyhow!("lm_head: {e}"))?;
        }
        timings.lm_head_ms = lm_start.elapsed().as_secs_f64() * 1000.0;

        let d2h_start = Instant::now();
        let logits_bytes = self.logits_buf.to_host_bytes()?;
        timings.logits_d2h_ms = d2h_start.elapsed().as_secs_f64() * 1000.0;
        let stride = config.vocab_size * ScalarType::BF16.size_in_bytes();
        let all_logits = (0..b)
            .map(|bi| {
                logits_bytes[bi * stride..(bi + 1) * stride]
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect()
            })
            .collect();
        Ok((all_logits, timings))
    }
}

#[cfg(test)]
mod mtp_accept_tests {
    use super::DecodeEngine;

    #[test]
    fn greedy_sample_is_argmax() {
        assert_eq!(DecodeEngine::greedy_sample(&[0.1, 4.0, 3.5, -1.0]), 1);
        assert_eq!(DecodeEngine::greedy_sample(&[-3.0, -2.0, -2.5]), 1);
    }

    #[test]
    fn fused_verify_batch_fits_k2_for_qwen38_hidden() {
        // Qwen3.8-27B hidden=5120. K=2 needs B=3 fused verify.
        const HIDDEN: usize = 5120;
        const MAX_INPUT_CACHE_FLOATS: usize = 15872;
        let max_b = (MAX_INPUT_CACHE_FLOATS / HIDDEN).max(1);
        assert!(max_b >= 3, "fused verify B=3 must fit hidden={HIDDEN}");
    }

    #[test]
    fn mtp_commit_len_is_accepted_plus_one() {
        let n_acc = 1usize;
        let block_len = 3usize;
        let remaining = 32usize;
        let commit_len = (n_acc + 1).min(block_len).min(remaining);
        assert_eq!(commit_len, 2);
        assert_eq!((0usize + 1).min(3).min(32), 1);
        assert_eq!((2usize + 1).min(3).min(32), 3);
    }
}
