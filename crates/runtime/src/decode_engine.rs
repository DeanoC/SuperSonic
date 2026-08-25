#![allow(dead_code)]

use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use qwen38::config::TextConfig;
use qwen38::desc_builder::{
    build_batch_seq_descs, build_fp8_scale_descs, build_int4_scale_descs, build_layer_descs,
};
use qwen38::rotary::RotaryTables;
use qwen38::scratch::PersistentDecodeScratch;
use qwen38::state::{LinearStateSnapshot, ModelState, ModelStateDiskSnapshot};
use qwen38::weights::Qwen38Weights;
use serde::{Deserialize, Serialize};

use crate::mtp::{
    mtp_forward, prefill_append_verify_cached, restore_linear_prefix, restore_linear_state,
    MtpPrefillAppendCache, MtpVerifyCache, MtpVerifyScratch,
};
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
    env::var_os("SUPERSONIC_QWEN38_GQH_STATE_DUMP").is_some()
}

fn linear_layer_dump_dir(idx: usize) -> Option<std::path::PathBuf> {
    let want = env::var("SUPERSONIC_QWEN38_DUMP_LINEAR_LAYER")
        .ok()?
        .parse::<usize>()
        .ok()?;
    if want != idx {
        return None;
    }
    env::var_os("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_DIR").map(std::path::PathBuf::from)
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
    for (i, _ls) in state.layers.iter().enumerate() {
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
    weights: &Qwen38Weights,
    out: &mut GpuBuffer,
    label: &str,
) -> Result<bool> {
    let Some((qtype, scale, zero)) = weights.lm_head_lowbit_params(hidden_dim) else {
        return Ok(false);
    };
    if qwen38::weights::is_gqh_qtype(qtype) {
        qwen38::weights::matmul_gqh(
            ordinal,
            m,
            vocab_size,
            hidden_dim,
            lhs,
            weights.lm_head(),
            qtype,
            out,
        )
        .map_err(|e| anyhow::anyhow!("{label} gqh matmul: {e}"))?;
        return Ok(true);
    }
    if qwen38::weights::is_mix_qtype(qtype) {
        qwen38::weights::matmul_mix(
            ordinal,
            m,
            vocab_size,
            hidden_dim,
            lhs,
            weights.lm_head(),
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
        weights.lm_head(),
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
    let qtype = qwen38::weights::infer_lowbit_type(weight, k, int4_scale.is_some());
    if qwen38::weights::is_gqh_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("GQH matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen38::weights::matmul_gqh(ordinal, m, n, k, lhs, weight, qtype, out)
            .map_err(|e| anyhow::anyhow!("matmul_gqh: {e}"));
    }
    if qwen38::weights::is_mix_qtype(qtype) {
        if batch != 1 {
            anyhow::bail!("mix matmul is batch-1 only (batch={batch} m={m} n={n} k={k})");
        }
        return qwen38::weights::matmul_mix(ordinal, m, n, k, lhs, weight, qtype, out)
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

fn is_qwen38_4b_shape(config: &TextConfig) -> bool {
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
    weights: Qwen38Weights,
    state: ModelState,
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
    /// Optional total decode context reservation for preallocated KV storage.
    decode_context_limit: Option<usize>,
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

impl Drop for DecodeEngine {
    fn drop(&mut self) {
        if !self.use_4b_kernel {
            return;
        }
        let int4 = self
            .int4_scale_device
            .as_ref()
            .map(|buffer| buffer.as_ptr())
            .unwrap_or(std::ptr::null());
        if let Err(err) = kernel_ffi::gqh::invalidate_decode_cache(
            self.ordinal,
            self.scratch.desc_device.as_ptr(),
            int4,
        ) {
            eprintln!("decode cache invalidation returned before ownership was published: {err}");
        }
    }
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
    HipFastGreedy,
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

fn decode_argmax_token(token_bytes: &[u8]) -> Result<u32> {
    let token_bytes: [u8; 4] = token_bytes
        .get(..4)
        .ok_or_else(|| anyhow::anyhow!("argmax D2H returned truncated token buffer"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("argmax D2H token conversion failed"))?;
    Ok(u32::from_le_bytes(token_bytes))
}

fn decode_profile_enabled() -> bool {
    decode_profile_enabled_value(env::var("SUPERSONIC_DECODE_PROF").ok().as_deref())
}

fn decode_profile_enabled_value(value: Option<&str>) -> bool {
    matches!(value, Some(value) if !value.is_empty() && value != "0")
}

fn allocation_layout_fingerprint(addresses: &[usize]) -> u64 {
    let Some(&base) = addresses.first() else {
        return 0;
    };
    addresses.iter().fold(0xcbf29ce484222325, |hash, &address| {
        let relative_pages = address.wrapping_sub(base) >> 12;
        let alignment = address & 0xfff;
        hash.wrapping_mul(0x100000001b3)
            .wrapping_add(relative_pages as u64)
            .wrapping_mul(0x100000001b3)
            .wrapping_add(alignment as u64)
    })
}

fn ensure_hip_fast_greedy_supported(use_4b_kernel: bool) -> Result<()> {
    anyhow::ensure!(
        use_4b_kernel,
        "decode_step_hip_fast_greedy requires the 4B kernel"
    );
    Ok(())
}

fn decode_sampling_result(
    sampling_mode: DecodeSamplingMode,
    logits: Option<Vec<f32>>,
    token_bytes: Option<&[u8]>,
    timings: DecodeStageTimings,
) -> Result<DecodeStepOutput> {
    match sampling_mode {
        DecodeSamplingMode::HostLogits => {
            let logits =
                logits.ok_or_else(|| anyhow::anyhow!("host-logit decode missing logits"))?;
            let sampling_start = Instant::now();
            let sampled_token = DecodeEngine::greedy_sample(&logits);
            let mut timings = timings;
            timings.host_sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
            Ok(DecodeStepOutput {
                logits: Some(logits),
                sampled_token,
                timings,
            })
        }
        DecodeSamplingMode::HipFastGreedy => {
            anyhow::ensure!(
                logits.is_none(),
                "HIP fast greedy must not materialize host logits"
            );
            let token_bytes = token_bytes
                .ok_or_else(|| anyhow::anyhow!("HIP fast greedy missing argmax token"))?;
            let sampled_token = decode_argmax_token(token_bytes)?;
            Ok(DecodeStepOutput {
                logits: None,
                sampled_token,
                timings,
            })
        }
    }
}

impl DecodeEngine {
    pub fn scratch_debug_ptr(&self) -> usize {
        self.scratch.workspace.as_ptr() as usize
    }

    pub fn new(
        weights: Qwen38Weights,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
        prefill_chunk_size: usize,
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
        )
    }

    pub fn new_with_rotary(
        weights: Qwen38Weights,
        rotary: RotaryTables,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
        prefill_chunk_size: usize,
    ) -> Result<Self> {
        let config = &weights.config;
        let state = ModelState::new(config, ordinal)
            .map_err(|e| anyhow::anyhow!("model state init: {e}"))?;

        let scratch = PersistentDecodeScratch::new(
            ordinal,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            proj_buf_floats,
            attn_scratch_floats,
        )
        .map_err(|e| anyhow::anyhow!("scratch init: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("hidden_io: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("normed_buf: {e}"))?;
        let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.vocab_size])
            .map_err(|e| anyhow::anyhow!("logits_buf: {e}"))?;
        let logits_f32_buf = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, 1, config.vocab_size])
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
            decode_context_limit: None,
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

    pub fn weights(&self) -> &Qwen38Weights {
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
        match restore_linear_prefix(self.ordinal, &self.scratch.desc_device, commit_len) {
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
        self.decode_context_limit = Some(context_tokens.max(1));
    }

    /// Verify the engine's attn_scratch budget covers the current largest
    /// `kv_max_t` across all full-attention layers.
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
        for ls in &self.state.layers {
            max_kv = max_kv.max(ls.kv_capacity());
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

    pub fn rotary(&self) -> &RotaryTables {
        &self.rotary
    }

    pub fn state(&self) -> &ModelState {
        &self.state
    }

    /// Reset per-session state so the engine is ready for a fresh prompt.
    /// Weights, rotary tables, scratch allocations, and quantization scales are
    /// untouched — only KV caches, conv/recurrent state, and the sync counters
    /// are cleared. Used by the HTTP server between requests.
    pub fn reset(&mut self) -> Result<()> {
        self.state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("reset model state: {e}"))?;
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
            self.use_4b_kernel,
            false,
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
                .embed_tokens()
                .offset_ptr(token_id as usize * row_bytes),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        for (idx, layer) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(idx) {
                layer
                    .ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size)
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {idx}: {e}"))?;
            }
        }
        self.check_attn_scratch_budget()?;
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch.upload_descs(&descs)?;
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
            self.weights.lm_head(),
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

    pub fn decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
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
                    self.weights.lm_head().as_ptr() as *mut _,
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
            1,
            None,
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
        anyhow::ensure!(
            self.hidden_io.backend() == gpu_hal::Backend::Hip,
            "decode_step_hip_fast_greedy requires HIP"
        );
        ensure_hip_fast_greedy_supported(self.use_4b_kernel)?;
        let output = self.decode_step_single_kernel_impl(
            token_id,
            seqlen_offset,
            true,
            DecodeSamplingMode::HipFastGreedy,
        )?;
        Ok((output.sampled_token, output.timings))
    }

    /// Backend the engine is running on. Used by callers that need to pick
    /// between the incremental decode path and replay-prefill path.
    pub fn backend(&self) -> gpu_hal::Backend {
        self.hidden_io.backend()
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
        let output = self.decode_step_single_kernel_impl(
            token_id,
            seqlen_offset,
            true,
            DecodeSamplingMode::HostLogits,
        )?;
        let logits = output
            .logits
            .ok_or_else(|| anyhow::anyhow!("single-sequence 4B kernel timings missing logits"))?;
        Ok((logits, output.timings))
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
        if tokens.is_empty() {
            anyhow::bail!("verify_block_prefill_append: tokens must be non-empty");
        }

        let max_pos = pos_offset + tokens.len() - 1;
        {
            let config = &self.weights.config;
            for (idx, layer_state) in self.state.layers.iter_mut().enumerate() {
                if config.is_full_attention(idx) {
                    layer_state
                        .ensure_kv_capacity(max_pos, self.ordinal, config, self.kv_chunk_size)
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
    /// * `use_4b_kernel = true` (a verify-local B-slot cache is used).
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
                    ls.ensure_kv_capacity(max_pos, self.ordinal, config, self.kv_chunk_size)
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
        let batch_descs = build_batch_seq_descs(&state_refs, &seqlen_offsets).ok_or_else(|| {
            anyhow::anyhow!("fused verify: build_batch_seq_descs returned None for B={b}")
        })?;
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
                self.weights.embed_tokens().offset_ptr(src_offset),
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
                self.weights.lm_head(),
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

    fn decode_step_single_kernel_impl(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        enable_timing_slots: bool,
        sampling_mode: DecodeSamplingMode,
    ) -> Result<DecodeStepOutput> {
        anyhow::ensure!(self.use_4b_kernel, "single decode requires 4B kernel");
        let config = &self.weights.config;
        let mut timings = DecodeStageTimings::default();
        let row_bytes = config.hidden_size * ScalarType::BF16.size_in_bytes();
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights
                .embed_tokens()
                .offset_ptr(token_id as usize * row_bytes),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("embedding lookup: {e}"))?;

        for (layer_idx, layer) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(layer_idx) {
                layer
                    .ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size)
                    .map_err(|e| anyhow::anyhow!("ensure KV layer {layer_idx}: {e}"))?;
            }
        }
        self.check_attn_scratch_budget()?;

        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload decode descriptors: {e}"))?;
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
        let collect_timing_slots = enable_timing_slots && decode_profile_enabled();
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
            1,
            None,
            self.int4_scale_device.as_ref(),
            collect_timing_slots,
            false,
        )
        .map_err(|e| anyhow::anyhow!("persistent decode 4B: {e}"))?;
        timings.persistent_ms = persistent_start.elapsed().as_secs_f64() * 1000.0;

        if collect_timing_slots {
            let bytes = self
                .scratch
                .sync_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("persistent timing slots D2H: {e}"))?;
            let slot_bytes = kernel_ffi::PERSISTENT_4B_TIMING_SLOT_COUNT * 8;
            let payload = bytes.get(24..).ok_or_else(|| {
                anyhow::anyhow!("persistent timing buffer missing 24-byte header")
            })?;
            anyhow::ensure!(
                payload.len() == config.num_hidden_layers * slot_bytes,
                "persistent timing buffer has {} bytes; expected exactly {}",
                payload.len(),
                config.num_hidden_layers * slot_bytes
            );
            let mut maxima = [0u64; 8];
            for layer_bytes in payload.chunks_exact(slot_bytes) {
                let slots: Vec<u64> = layer_bytes
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                let parsed = kernel_ffi::parse_persistent_4b_timing_slots(&slots)
                    .map_err(|e| anyhow::anyhow!("persistent timing slots: {e}"))?;
                maxima[0] = maxima[0].max(parsed.full_attn);
                maxima[1] = maxima[1].max(parsed.full_attn_proj);
                maxima[2] = maxima[2].max(parsed.full_attn_core.into_iter().max().unwrap_or(0));
                maxima[3] = maxima[3].max(parsed.full_attn_out.into_iter().max().unwrap_or(0));
                maxima[4] = maxima[4].max(parsed.linear_proj);
                maxima[5] = maxima[5].max(parsed.linear_core.into_iter().max().unwrap_or(0));
                maxima[6] = maxima[6].max(parsed.linear_out.into_iter().max().unwrap_or(0));
                maxima[7] = maxima[7].max(parsed.mlp_gate_up.max(parsed.mlp_down));
            }
            let fingerprint = allocation_layout_fingerprint(&[
                self.scratch.workspace.as_ptr() as usize,
                self.scratch.sync_buf.as_ptr() as usize,
                self.scratch.desc_device.as_ptr() as usize,
                self.hidden_io.as_ptr() as usize,
                self.normed_buf.as_ptr() as usize,
                self.logits_buf.as_ptr() as usize,
            ]);
            static LAYOUT_REPORTED: OnceLock<()> = OnceLock::new();
            if LAYOUT_REPORTED.set(()).is_ok() {
                eprintln!(
                    "[decode-profile] persistent_4b_layout layers={} allocation_fingerprint={fingerprint:016x}",
                    config.num_hidden_layers
                );
            }
            eprintln!(
                "[decode-profile] persistent_4b_slots full_attn={} full_attn_proj={} full_attn_core={} full_attn_out={} linear_proj={} linear_core={} linear_out={} mlp={}",
                maxima[0], maxima[1], maxima[2], maxima[3], maxima[4], maxima[5], maxima[6], maxima[7]
            );
        }

        let filled = seqlen_offset + 1;
        for (layer_idx, layer) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(layer_idx) {
                layer.set_kv_filled(filled);
            }
        }

        let fast_greedy = sampling_mode == DecodeSamplingMode::HipFastGreedy;
        let norm_start = Instant::now();
        rms_norm_rows_model(
            config,
            self.ordinal,
            1,
            config.hidden_size,
            &self.hidden_io,
            &self.weights.norm_weight,
            &mut self.normed_buf,
            "final RMSNorm",
        )?;
        if fast_greedy {
            gpu_hal::sync(self.ordinal)
                .map_err(|e| anyhow::anyhow!("HIP fast greedy RMSNorm completion: {e}"))?;
        }
        timings.rms_norm_ms = norm_start.elapsed().as_secs_f64() * 1000.0;

        let lm_start = Instant::now();
        // Preserve the host path's BF16-rounded GQH output and tie semantics.
        // The direct F32 lm-head route is not equivalent on the supported HIP
        // artifact, so only the vocabulary D2H is removed here.
        let (lm_head_out, lm_head_dtype, lm_head_label) =
            (&mut self.logits_buf, ScalarType::BF16, "lm_head");
        if !lm_head_lowbit(
            self.ordinal,
            1,
            config.vocab_size,
            config.hidden_size,
            &self.normed_buf,
            &self.weights,
            lm_head_out,
            lm_head_label,
        )? {
            kernel_ffi::matmul_rhs_transposed_4b(
                self.ordinal,
                lm_head_dtype,
                1,
                1,
                config.vocab_size,
                config.hidden_size,
                &self.normed_buf,
                self.weights.lm_head(),
                lm_head_out,
            )
            .map_err(|e| anyhow::anyhow!("{lm_head_label}: {e}"))?;
        }
        if fast_greedy {
            gpu_hal::sync(self.ordinal)
                .map_err(|e| anyhow::anyhow!("HIP fast greedy lm-head completion: {e}"))?;
        }
        timings.lm_head_ms = lm_start.elapsed().as_secs_f64() * 1000.0;

        if fast_greedy {
            let argmax_start = gpu_hal::GpuEvent::new(self.ordinal)
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax start event: {e}"))?;
            argmax_start
                .record()
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax start record: {e}"))?;
            kernel_ffi::prefill_ffi::argmax_bf16_rows(
                self.ordinal,
                1,
                config.vocab_size,
                &self.logits_buf,
                &mut self.argmax_buf,
            )
            .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax: {e}"))?;
            let argmax_end = gpu_hal::GpuEvent::new(self.ordinal)
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax end event: {e}"))?;
            argmax_end
                .record()
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax end record: {e}"))?;
            argmax_end
                .synchronize()
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax completion: {e}"))?;
            timings.gpu_argmax_ms = f64::from(
                gpu_hal::GpuEvent::elapsed_ms(&argmax_start, &argmax_end)
                    .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax elapsed: {e}"))?,
            );

            let token_d2h_start = Instant::now();
            let token_bytes = self
                .argmax_buf
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("HIP fast greedy argmax D2H: {e}"))?;
            let mut output =
                decode_sampling_result(sampling_mode, None, Some(&token_bytes), timings)?;
            output.timings.token_d2h_ms = token_d2h_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(output);
        }

        let d2h_start = Instant::now();
        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
        timings.logits_d2h_ms = d2h_start.elapsed().as_secs_f64() * 1000.0;
        let logits = logits_bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect();
        decode_sampling_result(sampling_mode, Some(logits), None, timings)
    }
}

#[cfg(test)]
mod hip_fast_greedy_tests {
    use super::{
        decode_sampling_result, ensure_hip_fast_greedy_supported, DecodeSamplingMode,
        DecodeStageTimings,
    };

    #[test]
    fn device_argmax_fast_route_does_not_materialize_host_logits() {
        let token = 0x0102_0304u32;
        let fast = decode_sampling_result(
            DecodeSamplingMode::HipFastGreedy,
            None,
            Some(&token.to_le_bytes()),
            DecodeStageTimings::default(),
        )
        .expect("device argmax token should decode");
        assert_eq!(fast.sampled_token, token);
        assert!(fast.logits.is_none());

        let logits = vec![-1.0, 2.0, 1.0];
        let host = decode_sampling_result(
            DecodeSamplingMode::HostLogits,
            Some(logits.clone()),
            None,
            DecodeStageTimings::default(),
        )
        .expect("host-logit result should remain available");
        assert_eq!(host.logits, Some(logits));
        assert_eq!(host.sampled_token, 1);
    }

    #[test]
    fn fast_greedy_rejects_non_4b_kernel_route() {
        let error = ensure_hip_fast_greedy_supported(false).expect_err("non-4B must reject");
        assert!(error.to_string().contains("requires the 4B kernel"));
        ensure_hip_fast_greedy_supported(true).expect("4B fast route is supported");
    }
}

#[cfg(test)]
mod mtp_accept_tests {
    use super::{DecodeEngine, ModelState};

    #[test]
    fn mtp_verify_keeps_internal_b_slot_descriptors() {
        let first = ModelState {
            layers: Vec::new(),
            mtp: None,
        };
        let second = ModelState {
            layers: Vec::new(),
            mtp: None,
        };
        let states = [&first, &second];

        let descs = qwen38::desc_builder::build_batch_seq_descs(&states, &[0, 0])
            .expect("B>1 must retain internal MTP descriptors");

        assert!(descs.is_empty());
    }

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

#[cfg(test)]
mod decode_profile_tests {
    use super::{allocation_layout_fingerprint, decode_profile_enabled_value};

    #[test]
    fn timing_slot_collection_is_opt_in_to_existing_profile_control() {
        assert!(!decode_profile_enabled_value(None));
        assert!(!decode_profile_enabled_value(Some("0")));
        assert!(!decode_profile_enabled_value(Some("")));
        assert!(decode_profile_enabled_value(Some("1")));
    }

    #[test]
    fn allocation_fingerprint_is_relative_and_address_free() {
        let first = allocation_layout_fingerprint(&[0x1000, 0x5000, 0x9000]);
        let relocated = allocation_layout_fingerprint(&[0x8100_1000, 0x8100_5000, 0x8100_9000]);
        assert_eq!(first, relocated);
        assert_ne!(
            first,
            allocation_layout_fingerprint(&[0x1000, 0x6000, 0x9000])
        );
    }
}
