use std::ffi::c_void;
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
use qwen35::scratch::{
    PersistentDecodeScratch, PERSISTENT_4B_TIMING_SLOTS_PER_LAYER, PERSISTENT_SYNC_COUNTER_BYTES,
};
use qwen35::state::{kv_fp8_bf16_sidecar_enabled, ModelState};
use qwen35::weights::Qwen35Weights;

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

fn flush_metal_batch_for_host_boundary(buffer: &GpuBuffer, label: &str) -> Result<()> {
    if buffer.backend() == gpu_hal::Backend::Metal {
        kernel_ffi::prefill_ffi::flush_metal_batch()
            .map_err(|e| anyhow::anyhow!("{label} Metal flush: {e}"))?;
    }
    Ok(())
}

fn copy_d2d_ordered(
    ordinal: usize,
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    backend_hint: &GpuBuffer,
    label: &str,
) -> Result<()> {
    if backend_hint.backend() == gpu_hal::Backend::Metal {
        kernel_ffi::prefill_ffi::metal_copy_d2d(src, dst, bytes)
            .map_err(|e| anyhow::anyhow!("{label} Metal blit copy: {e}"))?;
    } else {
        gpu_hal::copy_d2d(ordinal, dst, src, bytes).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    }
    Ok(())
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
    block_size: usize,
    out: &mut GpuBuffer,
    int4_scale: Option<&GpuBuffer>,
    int4_zero: Option<&GpuBuffer>,
    int4_group_size: usize,
) -> Result<()> {
    if let (Some(sc), Some(zr)) = (int4_scale, int4_zero) {
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
            int4_group_size,
            out,
        )
        .map_err(|e| anyhow::anyhow!("matmul_int4: {e}"))
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

fn metal_f32_projection_from_f32_input_to_f32(
    ordinal: usize,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    input_f32: &GpuBuffer,
    weight: &GpuBuffer,
    output_f32: &mut GpuBuffer,
) -> Result<()> {
    let mut weight_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[out_dim, in_dim])
        .map_err(|e| anyhow::anyhow!("metal f32 projection weight alloc: {e}"))?;
    kernel_ffi::prefill_ffi::cast(
        ordinal,
        weight.dtype(),
        ScalarType::F32,
        out_dim * in_dim,
        weight,
        &mut weight_f32,
    )
    .map_err(|e| anyhow::anyhow!("metal f32 projection weight cast: {e}"))?;
    kernel_ffi::prefill_ffi::matmul_rhs_transposed(
        ordinal,
        ScalarType::F32,
        1,
        rows,
        out_dim,
        in_dim,
        input_f32,
        &weight_f32,
        output_f32,
    )
    .map_err(|e| anyhow::anyhow!("metal f32 projection matmul: {e}"))?;
    Ok(())
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

fn metal_fused_residual_projection_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_RESIDUAL_PROJ").is_some()
}

fn metal_fused_mlp_enabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_ENABLE_FUSED_MLP").is_some()
}

fn metal_fused_full_projection_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_FULL_PROJ").is_some()
}

fn metal_matmul_residual_add_bf16(
    input_dim: usize,
    output_dim: usize,
    input: &GpuBuffer,
    weight: &GpuBuffer,
    residual_out: &mut GpuBuffer,
) -> Result<()> {
    let residual: &GpuBuffer = unsafe { &*(residual_out as *const GpuBuffer) };
    kernel_ffi::prefill_ffi::metal_matmul_rhs_transposed_residual_bf16(
        1,
        1,
        output_dim,
        input_dim,
        input,
        weight,
        residual,
        residual_out,
    )
    .map_err(|e| anyhow::anyhow!("fused residual projection failed: {e}"))?;
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
    /// Cached DFlash tap scratch: `(tap_layers, workspace [num_taps,
    /// hidden_dim] BF16, layer_ids i32-as-u8 buffer)`. `decode_step_
    /// with_taps_kernel` reuses these across calls with the same
    /// tap_layers list. Avoids a per-call GpuBuffer::zeros + upload of
    /// a small i32 vec — ~100ms savings per call at 9B INT4.
    dflash_tap_cache: Option<(Vec<usize>, GpuBuffer, GpuBuffer)>,
    /// Cached workspace for `verify_block_fused_decode` (DFlash M4.3).
    /// The fused verify path runs the persistent 4B megakernel with
    /// `batch_size = B` while the live engine is constructed with
    /// `batch_size = 1`; the cache owns a B-sized workspace + IO buffers
    /// + batch-seq desc table so the per-round allocation cost is paid
    /// only once per fused-verify call chain. Re-allocated if the block
    /// size changes between calls.
    dflash_fused_verify_cache: Option<DFlashFusedVerifyCache>,
    /// Reused single-token component decode MLP temporaries. Metal component
    /// decode executes many small ops, so avoiding per-layer allocation churn
    /// is currently more important than the tiny retained memory footprint.
    component_mlp_scratch: Option<ComponentMlpScratch>,
    /// Reused single-token component decode linear-attention temporaries.
    component_linear_scratch: Option<ComponentLinearScratch>,
    /// Reused single-token component decode full-attention temporaries.
    component_full_scratch: Option<ComponentFullScratch>,
    /// Per-linear-layer BF16 copies of static norm weights used by Metal
    /// component decode. Avoids recasting the same F32 weight every token.
    component_linear_norm_w_bf16: Vec<Option<GpuBuffer>>,
}

/// Per-call workspace for `DecodeEngine::verify_block_fused_decode`.
///
/// The fused verify path needs a B-sized workspace (F32 projection +
/// attention scratch, multi-row hidden_io / normed_buf / logits_buf) and
/// a `BatchSeqDesc` table, sized independently from `DecodeEngine`'s
/// `batch_size = 1` scratch. The cache is populated lazily on first
/// fused-verify call and reused thereafter via the take/put pattern that
/// `decode_step_with_taps_kernel` uses for `dflash_tap_cache`.
struct DFlashFusedVerifyCache {
    /// Block size the cache is sized for. A change in `--dflash-block`
    /// between calls triggers a full re-allocation.
    block_size: usize,
    /// F32 scratch for projections + MLP + attention. Sized to the same
    /// per-item layout as `PersistentDecodeScratch::workspace` at
    /// `batch_size = block_size`.
    workspace: GpuBuffer,
    /// BF16 hidden I/O, shape `[block_size, 1, hidden_size]`.
    hidden_io: GpuBuffer,
    /// BF16 RMSNorm output, shape `[block_size, 1, hidden_size]`.
    normed_buf: GpuBuffer,
    /// BF16 logits output, shape `[block_size, 1, vocab_size]`.
    logits_buf: GpuBuffer,
    /// Device copy of `Vec<BatchSeqDesc>` (one per layer), re-uploaded
    /// each fused-verify call.
    batch_desc_device: GpuBuffer,
}

impl DFlashFusedVerifyCache {
    #[allow(clippy::too_many_arguments)]
    fn alloc(
        ordinal: usize,
        block_size: usize,
        hidden_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_layers: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
    ) -> Result<Self> {
        // Layout matches `PersistentDecodeScratch::new` — see
        // crates/qwen35/src/scratch.rs. Per-item segments:
        //   [hidden] input/output, [hidden] normed, [inter*2] gate+up,
        //   [hidden] down-proj slab × 2, [proj_buf_floats] proj, and
        //   [attn_scratch_floats] attention saved_q/gate/pre_gate/scores.
        let per_item_floats = hidden_dim
            + hidden_dim
            + intermediate_size * 2
            + hidden_dim
            + hidden_dim
            + proj_buf_floats
            + attn_scratch_floats;
        let workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[per_item_floats * block_size])
            .map_err(|e| anyhow::anyhow!("fused verify workspace alloc: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("fused verify hidden_io alloc: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("fused verify normed_buf alloc: {e}"))?;
        let logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[block_size, 1, vocab_size])
            .map_err(|e| anyhow::anyhow!("fused verify logits_buf alloc: {e}"))?;
        let batch_desc_bytes = num_layers * std::mem::size_of::<kernel_ffi::BatchSeqDesc>();
        let batch_desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[batch_desc_bytes])
            .map_err(|e| anyhow::anyhow!("fused verify batch desc alloc: {e}"))?;
        Ok(Self {
            block_size,
            workspace,
            hidden_io,
            normed_buf,
            logits_buf,
            batch_desc_device,
        })
    }
}

struct ComponentMlpScratch {
    gate: GpuBuffer,
    up: GpuBuffer,
    mlp: GpuBuffer,
    down: GpuBuffer,
}

struct ComponentLinearScratch {
    qkv: GpuBuffer,
    z: GpuBuffer,
    a: GpuBuffer,
    b: GpuBuffer,
    a_beta_raw: GpuBuffer,
    rec_apply: GpuBuffer,
    attn_bf16: GpuBuffer,
    norm_w_bf16: GpuBuffer,
    gated: GpuBuffer,
    proj_out: GpuBuffer,
    conv_pack: GpuBuffer,
    q_linear: GpuBuffer,
    k_linear: GpuBuffer,
    v_linear: GpuBuffer,
    q_linear_f32: GpuBuffer,
    k_linear_f32: GpuBuffer,
    v_linear_f32: GpuBuffer,
    q_normed: GpuBuffer,
    q_scaled: GpuBuffer,
    k_normed: GpuBuffer,
}

struct ComponentFullScratch {
    q_full: GpuBuffer,
    query_buf: GpuBuffer,
    gate_buf: GpuBuffer,
    k_buf: GpuBuffer,
    v_buf: GpuBuffer,
    q_normed: GpuBuffer,
    k_normed: GpuBuffer,
    attn_q: GpuBuffer,
    attn_k_step: GpuBuffer,
    attn_v_step: GpuBuffer,
    attn_out_f32: GpuBuffer,
    attn_flat: GpuBuffer,
    gated: GpuBuffer,
    proj_out: GpuBuffer,
    kv_k_contig: Option<GpuBuffer>,
    kv_v_contig: Option<GpuBuffer>,
    kv_contig_capacity: usize,
}

impl ComponentLinearScratch {
    fn alloc(ordinal: usize, config: &TextConfig) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let khd = config.linear_key_head_dim;
        let vhd = config.linear_value_head_dim;
        let key_dim = nk * khd;
        let val_dim = nv * vhd;
        let qkv_dim = key_dim * 2 + val_dim;
        let rec_apply_dim = val_dim + nv * khd * vhd;
        let qkv = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, qkv_dim])
            .map_err(|e| anyhow::anyhow!("component linear qkv scratch alloc: {e}"))?;
        let z = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("component linear z scratch alloc: {e}"))?;
        let a = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("component linear a scratch alloc: {e}"))?;
        let b = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nv])
            .map_err(|e| anyhow::anyhow!("component linear b scratch alloc: {e}"))?;
        let a_beta_raw = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, nv * 2])
            .map_err(|e| anyhow::anyhow!("component linear a_beta scratch alloc: {e}"))?;
        let rec_apply = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, rec_apply_dim])
            .map_err(|e| anyhow::anyhow!("component linear rec_apply scratch alloc: {e}"))?;
        let attn_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("component linear attn scratch alloc: {e}"))?;
        let norm_w_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[vhd])
            .map_err(|e| anyhow::anyhow!("component linear norm_w scratch alloc: {e}"))?;
        let gated = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, vhd])
            .map_err(|e| anyhow::anyhow!("component linear gated scratch alloc: {e}"))?;
        let proj_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("component linear proj_out scratch alloc: {e}"))?;
        let conv_pack = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, qkv_dim + nv])
            .map_err(|e| anyhow::anyhow!("component linear conv_pack scratch alloc: {e}"))?;
        let q_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("component linear q scratch alloc: {e}"))?;
        let k_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("component linear k scratch alloc: {e}"))?;
        let v_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("component linear v scratch alloc: {e}"))?;
        let q_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("component linear q_f32 scratch alloc: {e}"))?;
        let k_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, key_dim])
            .map_err(|e| anyhow::anyhow!("component linear k_f32 scratch alloc: {e}"))?;
        let v_linear_f32 = GpuBuffer::zeros(ordinal, ScalarType::F32, &[1, val_dim])
            .map_err(|e| anyhow::anyhow!("component linear v_f32 scratch alloc: {e}"))?;
        let q_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("component linear q_norm scratch alloc: {e}"))?;
        let q_scaled = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("component linear q_scale scratch alloc: {e}"))?;
        let k_normed = GpuBuffer::zeros(ordinal, ScalarType::F32, &[nk, khd])
            .map_err(|e| anyhow::anyhow!("component linear k_norm scratch alloc: {e}"))?;
        Ok(Self {
            qkv,
            z,
            a,
            b,
            a_beta_raw,
            rec_apply,
            attn_bf16,
            norm_w_bf16,
            gated,
            proj_out,
            conv_pack,
            q_linear,
            k_linear,
            v_linear,
            q_linear_f32,
            k_linear_f32,
            v_linear_f32,
            q_normed,
            q_scaled,
            k_normed,
        })
    }
}

impl ComponentFullScratch {
    fn alloc(ordinal: usize, config: &TextConfig) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv_heads * head_dim;
        let q_full = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_proj_dim])
            .map_err(|e| anyhow::anyhow!("component full q_full scratch alloc: {e}"))?;
        let query_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("component full query scratch alloc: {e}"))?;
        let gate_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("component full gate scratch alloc: {e}"))?;
        let k_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("component full k scratch alloc: {e}"))?;
        let v_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, kv_dim])
            .map_err(|e| anyhow::anyhow!("component full v scratch alloc: {e}"))?;
        let q_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("component full q_normed scratch alloc: {e}"))?;
        let k_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
            .map_err(|e| anyhow::anyhow!("component full k_normed scratch alloc: {e}"))?;
        let attn_q = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
            .map_err(|e| anyhow::anyhow!("component full attn_q scratch alloc: {e}"))?;
        let attn_k_step =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full attn_k scratch alloc: {e}"))?;
        let attn_v_step =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full attn_v scratch alloc: {e}"))?;
        let attn_out_f32 =
            GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("component full attn_out scratch alloc: {e}"))?;
        let attn_flat = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("component full attn_flat scratch alloc: {e}"))?;
        let gated = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, q_dim])
            .map_err(|e| anyhow::anyhow!("component full gated scratch alloc: {e}"))?;
        let proj_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("component full proj_out scratch alloc: {e}"))?;
        Ok(Self {
            q_full,
            query_buf,
            gate_buf,
            k_buf,
            v_buf,
            q_normed,
            k_normed,
            attn_q,
            attn_k_step,
            attn_v_step,
            attn_out_f32,
            attn_flat,
            gated,
            proj_out,
            kv_k_contig: None,
            kv_v_contig: None,
            kv_contig_capacity: 0,
        })
    }
}

impl ComponentMlpScratch {
    fn alloc(ordinal: usize, hidden_dim: usize, intermediate: usize) -> Result<Self> {
        let gate = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("component mlp gate scratch alloc: {e}"))?;
        let up = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("component mlp up scratch alloc: {e}"))?;
        let mlp = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, intermediate])
            .map_err(|e| anyhow::anyhow!("component mlp act scratch alloc: {e}"))?;
        let down = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
            .map_err(|e| anyhow::anyhow!("component mlp down scratch alloc: {e}"))?;
        Ok(Self {
            gate,
            up,
            mlp,
            down,
        })
    }
}

fn component_scratch_reuse_disabled() -> bool {
    std::env::var_os("SUPERSONIC_METAL_DISABLE_COMPONENT_SCRATCH_REUSE").is_some()
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
    pub normed: Vec<u8>,
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
const PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE: usize = 35;
const PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE: usize = 37;
const PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE: usize = 39;
const PERSISTENT_4B_TIMING_MLP_GATE_UP: usize = 41;
const PERSISTENT_4B_TIMING_MLP_DOWN: usize = 42;

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
            linear_core_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_CORE_BASE + b);
            linear_out_cycles += load_slot(layer_base + PERSISTENT_4B_TIMING_LINEAR_OUT_BASE + b);
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
        persistent_full_attn_ms: persistent_4b_clock_cycles_to_ms(full_attn_cycles, clock_rate_khz),
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
        persistent_mlp_down_ms: persistent_4b_clock_cycles_to_ms(mlp_down_cycles, clock_rate_khz),
        ..DecodeStageTimings::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35::scratch::PERSISTENT_4B_TIMING_SLOTS_PER_LAYER;

    #[test]
    fn persistent_4b_timing_ranges_do_not_overlap() {
        let full_attn_core =
            PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE..PERSISTENT_4B_TIMING_FULL_ATTN_CORE_BASE + 8;
        let full_attn_out =
            PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE..PERSISTENT_4B_TIMING_FULL_ATTN_OUT_BASE + 8;
        let linear_core =
            PERSISTENT_4B_TIMING_LINEAR_CORE_BASE..PERSISTENT_4B_TIMING_LINEAR_CORE_BASE + 8;
        let linear_out =
            PERSISTENT_4B_TIMING_LINEAR_OUT_BASE..PERSISTENT_4B_TIMING_LINEAR_OUT_BASE + 8;
        let linear_core_conv = PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_CONV_BASE + 2;
        let linear_core_recurrent = PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_RECURRENT_BASE + 2;
        let linear_core_post = PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE
            ..PERSISTENT_4B_TIMING_LINEAR_CORE_POST_BASE + 2;
        let singleton_slots = [
            PERSISTENT_4B_TIMING_FULL_ATTN,
            PERSISTENT_4B_TIMING_FULL_ATTN_PROJ,
            PERSISTENT_4B_TIMING_LINEAR_PROJ,
            PERSISTENT_4B_TIMING_MLP_GATE_UP,
            PERSISTENT_4B_TIMING_MLP_DOWN,
        ];

        let mut used = [false; PERSISTENT_4B_TIMING_SLOTS_PER_LAYER];
        for slot in singleton_slots {
            assert!(!used[slot], "slot {slot} overlaps");
            used[slot] = true;
        }
        for range in [
            full_attn_core,
            full_attn_out,
            linear_core,
            linear_out,
            linear_core_conv,
            linear_core_recurrent,
            linear_core_post,
        ] {
            for slot in range {
                assert!(slot < PERSISTENT_4B_TIMING_SLOTS_PER_LAYER);
                assert!(!used[slot], "slot {slot} overlaps");
                used[slot] = true;
            }
        }
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
        let mut q_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace q_normed alloc: {e}"))?;
        let mut k_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace k_normed alloc: {e}"))?;

        matmul_proj(
            self.ordinal,
            1,
            1,
            q_proj_dim,
            hidden_dim,
            &normed,
            &fw.q_proj_w,
            fw.q_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut q_full,
            fw.q_proj_int4_scale.as_ref(),
            fw.q_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
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
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.k_proj_w,
            fw.k_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut k_buf,
            fw.k_proj_int4_scale.as_ref(),
            fw.k_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.v_proj_w,
            fw.v_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut v_buf,
            fw.v_proj_int4_scale.as_ref(),
            fw.v_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            num_q_heads,
            head_dim,
            1e-6,
            &query_buf,
            &fw.q_norm_w,
            &mut q_normed,
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
            self.ordinal,
            ScalarType::BF16,
            num_kv_heads,
            head_dim,
            1e-6,
            &k_buf,
            &fw.k_norm_w,
            &mut k_normed,
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
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut k_buf,
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
        let mut q_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q_normed alloc: {e}"))?;
        let mut k_normed =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k_normed alloc: {e}"))?;
        let mut attn_q =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_q alloc: {e}"))?;
        let mut step_k =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer step_k alloc: {e}"))?;
        let mut step_v =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer step_v alloc: {e}"))?;
        let mut attn_out_f32 =
            GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[num_q_heads, 1, head_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn_out alloc: {e}"))?;
        let mut attn_out_bf16 =
            GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_q_heads, 1, head_dim]).map_err(
                |e| anyhow::anyhow!("layer {idx} trace full layer attn_out bf16 alloc: {e}"),
            )?;
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
            self.ordinal,
            1,
            1,
            q_proj_dim,
            hidden_dim,
            &normed,
            &fw.q_proj_w,
            fw.q_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut q_full,
            fw.q_proj_int4_scale.as_ref(),
            fw.q_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
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
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.k_proj_w,
            fw.k_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut k_buf,
            fw.k_proj_int4_scale.as_ref(),
            fw.k_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            kv_dim,
            hidden_dim,
            &normed,
            &fw.v_proj_w,
            fw.v_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut v_buf,
            fw.v_proj_int4_scale.as_ref(),
            fw.v_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            num_q_heads,
            head_dim,
            1e-6,
            &query_buf,
            &fw.q_norm_w,
            &mut q_normed,
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
            self.ordinal,
            ScalarType::BF16,
            num_kv_heads,
            head_dim,
            1e-6,
            &k_buf,
            &fw.k_norm_w,
            &mut k_normed,
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
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut query_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            &mut k_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k rope: {e}"))?;

        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            &query_buf,
            &mut attn_q,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer q transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            &k_buf,
            &mut step_k,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer k transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            &v_buf,
            &mut step_v,
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
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            num_kv_heads,
            1,
            kv_len,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            seqlen_offset,
            &attn_q,
            &kv_k_contig,
            &kv_v_contig,
            &mut attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attention: {e}"))?;
        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            num_q_heads * head_dim,
            &attn_out_f32,
            &mut attn_out_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn cast: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            num_q_heads,
            1,
            head_dim,
            &attn_out_bf16,
            &mut attn_flat,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer attn transpose: {e}"))?;
        kernel_ffi::prefill_ffi::sigmoid_mul(
            self.ordinal,
            ScalarType::BF16,
            q_dim,
            &attn_flat,
            &gate_buf,
            &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gate apply: {e}"))?;
        matmul_proj(
            self.ordinal,
            1,
            1,
            hidden_dim,
            q_dim,
            &gated,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            &mut proj_out,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        residual_add(self.ordinal, hidden_dim, &mut hidden_out, &proj_out)?;
        Ok(FullAttentionLayerOutputTrace {
            pre_gate: attn_flat
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer pre_gate D2H: {e}"))?,
            gated: gated
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} trace full layer gated D2H: {e}"))?,
            attn_hidden: hidden_out.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} trace full layer attn_hidden D2H: {e}")
            })?,
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
            .ok_or_else(|| {
                anyhow::anyhow!("layer {idx}: missing conv state after component trace")
            })?
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("layer {idx} component conv D2H: {e}"))?;
        let recurrent = ls
            .recurrent_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("layer {idx}: missing recurrent state after component trace")
            })?
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
        let attn_hidden = self.hidden_io.to_host_bytes().map_err(|e| {
            anyhow::anyhow!("layer {idx} component full-layer attn hidden D2H: {e}")
        })?[..row_bytes]
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
        let post_attn_norm =
            self.normed_buf.to_host_bytes().map_err(|e| {
                anyhow::anyhow!("layer {idx} component full-layer post norm D2H: {e}")
            })?[..row_bytes]
                .to_vec();

        let mlp_trace = self
            .component_decode_mlp_layer(idx, true)?
            .ok_or_else(|| anyhow::anyhow!("layer {idx} component full-layer missing mlp trace"))?;
        let layer_hidden = self.hidden_io.to_host_bytes().map_err(|e| {
            anyhow::anyhow!("layer {idx} component full-layer final hidden D2H: {e}")
        })?[..row_bytes]
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

    fn component_decode_step_4b_impl(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        metal_greedy: bool,
        trace_input_layer: Option<usize>,
        trace_layer: Option<usize>,
        trace_linear_layer: Option<usize>,
    ) -> Result<(
        Vec<f32>,
        Option<u32>,
        Option<Vec<u8>>,
        Option<ComponentLayerTrace>,
        Option<ComponentLinearTrace>,
    )> {
        let hidden_dim = self.weights.config.hidden_size;
        let rms_norm_eps = self.weights.config.rms_norm_eps as f32;
        let vocab_size = self.weights.config.vocab_size;
        let elem_bytes = ScalarType::BF16.size_in_bytes();

        let row_bytes = hidden_dim * elem_bytes;
        let src_offset = token_id as usize * row_bytes;
        copy_d2d_ordered(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
            &self.hidden_io,
            "embedding lookup",
        )?;

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
                if let Some(trace) =
                    self.component_decode_linear_attention_layer(i, trace_linear_layer == Some(i))?
                {
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
                let mlp_trace =
                    maybe_mlp.ok_or_else(|| anyhow::anyhow!("missing mlp trace for layer {i}"))?;
                traced_layer = Some(ComponentLayerTrace {
                    attn_hidden: trace_attn_hidden
                        .ok_or_else(|| anyhow::anyhow!("missing attn trace for layer {i}"))?,
                    post_attn_norm: trace_post_attn_norm.ok_or_else(|| {
                        anyhow::anyhow!("missing post-attn norm trace for layer {i}")
                    })?,
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

        let (logits, sampled_token) = if metal_greedy {
            if self.normed_buf.backend() != gpu_hal::Backend::Metal {
                anyhow::bail!("component Metal greedy decode requires Metal buffers");
            }
            let token = kernel_ffi::metal_lm_head_argmax_bf16(
                self.ordinal,
                &self.normed_buf,
                &*self.weights.lm_head,
                hidden_dim,
                vocab_size,
            )
            .map_err(|e| anyhow::anyhow!("lm_head argmax: {e}"))?;
            (Vec::new(), Some(token))
        } else {
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
            let logits = logits_bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect();
            (logits, None)
        };
        Ok((
            logits,
            sampled_token,
            traced_hidden,
            traced_layer,
            traced_linear,
        ))
    }

    fn component_decode_step_4b(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<Vec<f32>> {
        let (logits, _, _, _, _) =
            self.component_decode_step_4b_impl(token_id, seqlen_offset, false, None, None, None)?;
        Ok(logits)
    }

    pub fn component_decode_step_4b_traced(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_input_layer: usize,
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        let (logits, _, trace, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            false,
            Some(trace_input_layer),
            None,
            None,
        )?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing trace for layer {trace_input_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLayerTrace)> {
        let (logits, _, _, trace, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            false,
            None,
            Some(trace_layer),
            None,
        )?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing stage trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    pub fn component_decode_step_4b_trace_linear_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(Vec<f32>, ComponentLinearTrace)> {
        let (logits, _, _, _, trace) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            false,
            None,
            None,
            Some(trace_layer),
        )?;
        let trace =
            trace.ok_or_else(|| anyhow::anyhow!("missing linear trace for layer {trace_layer}"))?;
        Ok((logits, trace))
    }

    fn component_decode_full_attention_layer(
        &mut self,
        idx: usize,
        seqlen_offset: usize,
    ) -> Result<()> {
        if self.hidden_io.backend() == gpu_hal::Backend::Metal && !component_scratch_reuse_disabled()
        {
            let mut scratch = match self.component_full_scratch.take() {
                Some(scratch) => scratch,
                None => ComponentFullScratch::alloc(self.ordinal, &self.weights.config)?,
            };
            let result = self.component_decode_full_attention_layer_with_scratch(
                idx,
                seqlen_offset,
                &mut scratch,
            );
            self.component_full_scratch = Some(scratch);
            return result;
        }

        let mut scratch = ComponentFullScratch::alloc(self.ordinal, &self.weights.config)
            .map_err(|e| anyhow::anyhow!("layer {idx} full scratch alloc: {e}"))?;
        self.component_decode_full_attention_layer_with_scratch(idx, seqlen_offset, &mut scratch)
    }

    fn component_decode_full_attention_layer_with_scratch(
        &mut self,
        idx: usize,
        seqlen_offset: usize,
        scratch: &mut ComponentFullScratch,
    ) -> Result<()> {
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

        let ComponentFullScratch {
            q_full,
            query_buf,
            gate_buf,
            k_buf,
            v_buf,
            q_normed,
            k_normed,
            attn_q,
            attn_k_step,
            attn_v_step,
            attn_out_f32,
            attn_flat,
            gated,
            proj_out,
            kv_k_contig,
            kv_v_contig,
            kv_contig_capacity,
        } = scratch;

        let use_fused_full_proj = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !metal_fused_full_projection_disabled()
            && fw.q_proj_scale.is_none()
            && fw.q_proj_int4_scale.is_none()
            && fw.q_proj_int4_zero.is_none()
            && fw.k_proj_scale.is_none()
            && fw.k_proj_int4_scale.is_none()
            && fw.k_proj_int4_zero.is_none()
            && fw.v_proj_scale.is_none()
            && fw.v_proj_int4_scale.is_none()
            && fw.v_proj_int4_zero.is_none()
            && self.normed_buf.dtype() == ScalarType::BF16
            && fw.q_proj_w.dtype() == ScalarType::BF16
            && fw.k_proj_w.dtype() == ScalarType::BF16
            && fw.v_proj_w.dtype() == ScalarType::BF16;
        if use_fused_full_proj {
            kernel_ffi::prefill_ffi::metal_qwen_full_projections_bf16(
                hidden_dim,
                q_proj_dim,
                kv_dim,
                &self.normed_buf,
                &fw.q_proj_w,
                &fw.k_proj_w,
                &fw.v_proj_w,
                q_full,
                k_buf,
                v_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused full projections: {e}"))?;
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                q_proj_dim,
                hidden_dim,
                &self.normed_buf,
                &fw.q_proj_w,
                fw.q_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                q_full,
                fw.q_proj_int4_scale.as_ref(),
                fw.q_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
        }
        kernel_ffi::prefill_ffi::split_qgate(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            q_full,
            query_buf,
            gate_buf,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} split qgate: {e}"))?;

        if !use_fused_full_proj {
            matmul_proj(
                self.ordinal,
                1,
                1,
                kv_dim,
                hidden_dim,
                &self.normed_buf,
                &fw.k_proj_w,
                fw.k_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                k_buf,
                fw.k_proj_int4_scale.as_ref(),
                fw.k_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
            matmul_proj(
                self.ordinal,
                1,
                1,
                kv_dim,
                hidden_dim,
                &self.normed_buf,
                &fw.v_proj_w,
                fw.v_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                v_buf,
                fw.v_proj_int4_scale.as_ref(),
                fw.v_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
        }

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            num_q_heads,
            head_dim,
            1e-6,
            query_buf,
            &fw.q_norm_w,
            q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q norm: {e}"))?;

        kernel_ffi::prefill_ffi::rms_norm_rows(
            self.ordinal,
            ScalarType::BF16,
            num_kv_heads,
            head_dim,
            1e-6,
            k_buf,
            &fw.k_norm_w,
            k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k norm: {e}"))?;

        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q rope: {e}"))?;
        kernel_ffi::prefill_ffi::apply_rope_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            rotary_dim,
            &self.rotary.cos,
            &self.rotary.sin,
            seqlen_offset,
            k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k rope: {e}"))?;

        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            head_dim,
            q_normed,
            attn_q,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} q transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            k_normed,
            attn_k_step,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} k transpose: {e}"))?;
        kernel_ffi::prefill_ffi::transpose_shd_hsd(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_kv_heads,
            head_dim,
            v_buf,
            attn_v_step,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} v transpose: {e}"))?;

        let ls = &mut self.state.layers[idx];
        ls.ensure_kv_capacity(
            seqlen_offset,
            self.ordinal,
            config,
            self.kv_chunk_size,
            self.kv_fp8,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} kv alloc: {e}"))?;
        if let Some(ref mut cache_k) = ls.kv_cache_k {
            let cap = cache_k.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = head_dim * elem_bytes;
            let dst_offset = seqlen_offset * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                copy_d2d_ordered(
                    self.ordinal,
                    cache_k.offset_ptr(h * cap_stride + dst_offset) as *mut c_void,
                    attn_k_step.offset_ptr(h * src_stride),
                    src_stride,
                    cache_k,
                    &format!("layer {idx} cache k write h={h}"),
                )?;
            }
        }
        if let Some(ref mut cache_v) = ls.kv_cache_v {
            let cap = cache_v.shape()[2];
            let cap_stride = cap * head_dim * elem_bytes;
            let src_stride = head_dim * elem_bytes;
            let dst_offset = seqlen_offset * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                copy_d2d_ordered(
                    self.ordinal,
                    cache_v.offset_ptr(h * cap_stride + dst_offset) as *mut c_void,
                    attn_v_step.offset_ptr(h * src_stride),
                    src_stride,
                    cache_v,
                    &format!("layer {idx} cache v write h={h}"),
                )?;
            }
        }

        let cache_k_ref = ls.kv_cache_k.as_ref().unwrap();
        let cache_v_ref = ls.kv_cache_v.as_ref().unwrap();
        let cap = cache_k_ref.shape()[2];
        let attn_k_ref;
        let attn_v_ref;
        if cap == kv_len {
            attn_k_ref = cache_k_ref;
            attn_v_ref = cache_v_ref;
        } else {
            if *kv_contig_capacity < cap {
                *kv_k_contig = Some(
                    GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, cap, head_dim])
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_k_contig alloc: {e}"))?,
                );
                *kv_v_contig = Some(
                    GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_kv_heads, cap, head_dim])
                        .map_err(|e| anyhow::anyhow!("layer {idx} kv_v_contig alloc: {e}"))?,
                );
                *kv_contig_capacity = cap;
            }
            let kv_k_contig = kv_k_contig
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing cached K contig scratch"))?;
            let kv_v_contig = kv_v_contig
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing cached V contig scratch"))?;
            let cap_stride = cap * head_dim * elem_bytes;
            let contig_stride = kv_len * head_dim * elem_bytes;
            let copy_bytes = kv_len * head_dim * elem_bytes;
            for h in 0..num_kv_heads {
                copy_d2d_ordered(
                    self.ordinal,
                    kv_k_contig.offset_ptr(h * contig_stride) as *mut c_void,
                    cache_k_ref.offset_ptr(h * cap_stride),
                    copy_bytes,
                    &kv_k_contig,
                    &format!("layer {idx} kv assemble k h={h}"),
                )?;
                copy_d2d_ordered(
                    self.ordinal,
                    kv_v_contig.offset_ptr(h * contig_stride) as *mut c_void,
                    cache_v_ref.offset_ptr(h * cap_stride),
                    copy_bytes,
                    &kv_v_contig,
                    &format!("layer {idx} kv assemble v h={h}"),
                )?;
            }
            attn_k_ref = &kv_k_contig;
            attn_v_ref = &kv_v_contig;
        }

        kernel_ffi::prefill_ffi::full_attention_prefill(
            self.ordinal,
            ScalarType::BF16,
            1,
            num_q_heads,
            num_kv_heads,
            1,
            kv_len,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            seqlen_offset,
            attn_q,
            attn_k_ref,
            attn_v_ref,
            attn_out_f32,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            num_q_heads * head_dim,
            attn_out_f32,
            attn_flat,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;

        kernel_ffi::prefill_ffi::sigmoid_mul(
            self.ordinal,
            ScalarType::BF16,
            q_dim,
            attn_flat,
            gate_buf,
            gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gate apply: {e}"))?;

        matmul_proj(
            self.ordinal,
            1,
            1,
            hidden_dim,
            q_dim,
            gated,
            &fw.o_proj_w,
            fw.o_proj_scale.as_ref(),
            self.weights.fp8_block_size,
            proj_out,
            fw.o_proj_int4_scale.as_ref(),
            fw.o_proj_int4_zero.as_ref(),
            self.weights.int4_group_size,
        )?;
        residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, proj_out)?;
        Ok(())
    }

    fn component_decode_linear_attention_layer(
        &mut self,
        idx: usize,
        trace_output: bool,
    ) -> Result<Option<ComponentLinearTrace>> {
        let config = &self.weights.config;
        if self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && !component_scratch_reuse_disabled()
            && std::env::var_os("SUPERSONIC_METAL_ENABLE_COMPONENT_F32_LINEAR_INPUT").is_none()
        {
            let scratch = match self.component_linear_scratch.take() {
                Some(scratch) => scratch,
                None => ComponentLinearScratch::alloc(self.ordinal, config)?,
            };
            let (result, scratch) =
                self.component_decode_linear_attention_layer_with_scratch(idx, trace_output, scratch);
            self.component_linear_scratch = Some(scratch);
            return result;
        }

        let scratch = ComponentLinearScratch::alloc(self.ordinal, config)
            .map_err(|e| anyhow::anyhow!("layer {idx} linear scratch alloc: {e}"))?;
        let (result, _scratch) =
            self.component_decode_linear_attention_layer_with_scratch(idx, trace_output, scratch);
        result
    }

    fn component_decode_linear_attention_layer_with_scratch(
        &mut self,
        idx: usize,
        trace_output: bool,
        scratch: ComponentLinearScratch,
    ) -> (Result<Option<ComponentLinearTrace>>, ComponentLinearScratch) {
        let ComponentLinearScratch {
            mut qkv,
            mut z,
            mut a,
            mut b,
            a_beta_raw,
            mut rec_apply,
            mut attn_bf16,
            mut norm_w_bf16,
            mut gated,
            mut proj_out,
            mut conv_pack,
            mut q_linear,
            mut k_linear,
            mut v_linear,
            mut q_linear_f32,
            mut k_linear_f32,
            mut v_linear_f32,
            mut q_normed,
            mut q_scaled,
            mut k_normed,
        } = scratch;

        let result = (|| -> Result<Option<ComponentLinearTrace>> {
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

        let use_metal_f32_linear_input = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && std::env::var_os("SUPERSONIC_METAL_ENABLE_COMPONENT_F32_LINEAR_INPUT").is_some()
            && lw.qkv_proj_scale.is_none()
            && lw.qkv_proj_int4_scale.is_none()
            && lw.qkv_proj_int4_zero.is_none()
            && lw.z_proj_scale.is_none()
            && lw.z_proj_int4_scale.is_none()
            && lw.z_proj_int4_zero.is_none()
            && lw.b_proj_scale.is_none()
            && lw.a_proj_scale.is_none()
            && matches!(lw.qkv_proj_w.dtype(), ScalarType::BF16 | ScalarType::F32)
            && matches!(lw.z_proj_w.dtype(), ScalarType::BF16 | ScalarType::F32)
            && matches!(lw.b_proj_w.dtype(), ScalarType::BF16 | ScalarType::F32)
            && matches!(lw.a_proj_w.dtype(), ScalarType::BF16 | ScalarType::F32);
        let linear_normed_f32_storage;
        let linear_normed_f32 = if use_metal_f32_linear_input {
            let mut hidden_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, hidden_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} component hidden_f32 alloc: {e}"))?;
            let mut normed_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, hidden_dim])
                .map_err(|e| anyhow::anyhow!("layer {idx} component normed_f32 alloc: {e}"))?;
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::BF16,
                ScalarType::F32,
                hidden_dim,
                &self.hidden_io,
                &mut hidden_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} component hidden cast: {e}"))?;
            kernel_ffi::prefill_ffi::rms_norm_rows(
                self.ordinal,
                ScalarType::F32,
                1,
                hidden_dim,
                config.rms_norm_eps as f32,
                &hidden_f32,
                &self.weights.layers[idx].input_norm_w,
                &mut normed_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} component input norm f32: {e}"))?;
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                hidden_dim,
                &normed_f32,
                &mut self.normed_buf,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} component normed cast: {e}"))?;
            linear_normed_f32_storage = normed_f32;
            Some(&linear_normed_f32_storage)
        } else {
            None
        };

        let use_fused_linear_proj = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && linear_normed_f32.is_none()
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_LINEAR_PROJ").is_none()
            && lw.qkv_proj_scale.is_none()
            && lw.qkv_proj_int4_scale.is_none()
            && lw.qkv_proj_int4_zero.is_none()
            && lw.z_proj_scale.is_none()
            && lw.z_proj_int4_scale.is_none()
            && lw.z_proj_int4_zero.is_none()
            && lw.a_proj_scale.is_none()
            && lw.b_proj_scale.is_none()
            && lw.qkv_proj_w.dtype() == ScalarType::BF16
            && lw.z_proj_w.dtype() == ScalarType::BF16
            && lw.a_proj_w.dtype() == ScalarType::BF16
            && lw.b_proj_w.dtype() == ScalarType::BF16;

        if use_fused_linear_proj {
            kernel_ffi::prefill_ffi::metal_qwen_linear_projections_bf16(
                hidden_dim,
                qkv_dim,
                val_dim,
                nv,
                &self.normed_buf,
                &lw.qkv_proj_w,
                &lw.z_proj_w,
                &lw.a_proj_w,
                &lw.b_proj_w,
                &mut qkv,
                &mut z,
                &mut a,
                &mut b,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused linear projections: {e}"))?;
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                qkv_dim,
                hidden_dim,
                &self.normed_buf,
                &lw.qkv_proj_w,
                lw.qkv_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut qkv,
                lw.qkv_proj_int4_scale.as_ref(),
                lw.qkv_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
        }
        let normed_trace = if trace_output {
            Some(
                self.normed_buf
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} normed trace D2H: {e}"))?,
            )
        } else {
            None
        };
        let qkv_trace = if trace_output {
            Some(
                qkv.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} qkv trace D2H: {e}"))?,
            )
        } else {
            None
        };
        if !use_fused_linear_proj {
            matmul_proj(
                self.ordinal,
                1,
                1,
                val_dim,
                hidden_dim,
                &self.normed_buf,
                &lw.z_proj_w,
                lw.z_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut z,
                lw.z_proj_int4_scale.as_ref(),
                lw.z_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
        }
        let z_trace = if trace_output {
            Some(
                z.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} z trace D2H: {e}"))?,
            )
        } else {
            None
        };
        if !use_fused_linear_proj {
            matmul_proj(
                self.ordinal,
                1,
                1,
                nv,
                hidden_dim,
                &self.normed_buf,
                &lw.a_proj_w,
                lw.a_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut a,
                None,
                None,
                self.weights.int4_group_size,
            )?;
            matmul_proj(
                self.ordinal,
                1,
                1,
                nv,
                hidden_dim,
                &self.normed_buf,
                &lw.b_proj_w,
                lw.b_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut b,
                None,
                None,
                self.weights.int4_group_size,
            )?;
        }

        let mut a_for_beta_f32: Option<Vec<f32>> = None;
        let mut b_for_beta_f32: Option<Vec<f32>> = None;

        let ab_bytes = nv * ScalarType::BF16.size_in_bytes();
        if let Some(normed_f32) = linear_normed_f32 {
            let mut a_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, nv])
                .map_err(|e| anyhow::anyhow!("layer {idx} a_f32 alloc: {e}"))?;
            let mut b_f32 = GpuBuffer::zeros(self.ordinal, ScalarType::F32, &[1, nv])
                .map_err(|e| anyhow::anyhow!("layer {idx} b_f32 alloc: {e}"))?;
            metal_f32_projection_from_f32_input_to_f32(
                self.ordinal,
                1,
                hidden_dim,
                nv,
                normed_f32,
                &lw.a_proj_w,
                &mut a_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} a f32 proj: {e}"))?;
            metal_f32_projection_from_f32_input_to_f32(
                self.ordinal,
                1,
                hidden_dim,
                nv,
                normed_f32,
                &lw.b_proj_w,
                &mut b_f32,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} b f32 proj: {e}"))?;
            a_for_beta_f32 =
                Some(decode_f32_le(&a_f32.to_host_bytes().map_err(|e| {
                    anyhow::anyhow!("layer {idx} a_f32 D2H: {e}")
                })?));
            b_for_beta_f32 =
                Some(decode_f32_le(&b_f32.to_host_bytes().map_err(|e| {
                    anyhow::anyhow!("layer {idx} b_f32 D2H: {e}")
                })?));
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                nv,
                &a_f32,
                &mut a,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} a f32 cast: {e}"))?;
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                nv,
                &b_f32,
                &mut b,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} b f32 cast: {e}"))?;
        }
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
            Some(
                a.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} a trace D2H: {e}"))?,
            )
        } else {
            None
        };
        let b_trace = if trace_output {
            Some(
                b.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} b trace D2H: {e}"))?,
            )
        } else {
            None
        };

        let ls = &mut self.state.layers[idx];
        let conv_state_ref = ls
            .conv_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing conv state"))?;
        let recurrent_state = ls
            .recurrent_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing recurrent state"))?;

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
            conv_state_ref,
            &lw.conv1d_w,
            &a,
            &lw.dt_bias,
            &lw.a_log_exp,
            &mut conv_pack,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} linear conv/value_decay: {e}"))?;

        if self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_FUSED_LINEAR_PREP").is_none()
        {
            kernel_ffi::prefill_ffi::metal_qwen_linear_prep_bf16_f32(
                key_dim,
                val_dim,
                nk,
                khd,
                &conv_pack,
                &mut q_linear,
                &mut k_linear,
                &mut v_linear,
                &mut q_linear_f32,
                &mut k_linear_f32,
                &mut v_linear_f32,
                &mut q_normed,
                &mut q_scaled,
                &mut k_normed,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused linear prep: {e}"))?;
        } else {
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
        }

        let mut packed_trace = None;
        if self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_DIRECT_LINEAR_DECODE").is_none()
            && a_for_beta_f32.is_none()
            && b_for_beta_f32.is_none()
        {
            kernel_ffi::prefill_ffi::metal_linear_decode_apply_parts_f32(
                nv,
                nk,
                khd,
                vhd,
                &q_scaled,
                &k_normed,
                &v_linear_f32,
                &a,
                &b,
                &lw.dt_bias,
                &lw.a_log_exp,
                recurrent_state,
                &mut rec_apply,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} metal direct linear decode apply: {e}"))?;
        } else {
            flush_metal_batch_for_host_boundary(
                &self.hidden_io,
                &format!("layer {idx} linear packed host-read boundary"),
            )?;
            let q_scaled_host = q_scaled
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} q_scaled D2H: {e}"))?;
            let k_normed_host = k_normed
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} k_normed D2H: {e}"))?;
            let v_linear_host = v_linear_f32
                .to_host_bytes()
                .map_err(|e| anyhow::anyhow!("layer {idx} v_linear D2H: {e}"))?;
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
            let a_for_beta = if let Some(values) = a_for_beta_f32 {
                values
            } else {
                a.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} a D2H: {e}"))?
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            };
            let b_for_beta = if let Some(values) = b_for_beta_f32 {
                values
            } else {
                b.to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} b D2H: {e}"))?
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            };
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
                packed_host[out_base + 2 * khd + vhd] =
                    1.0f32 / (1.0f32 + (-b_for_beta[v_head]).exp());
                let softplus = (1.0f32 + (a_for_beta[v_head] + dt_bias_bf16[v_head]).exp()).ln();
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
            packed_trace = if trace_output {
                Some(
                    packed
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} packed trace D2H: {e}"))?,
                )
            } else {
                None
            };
        }

        let state_len = config.linear_conv_kernel_dim - 1;
        let state_bytes = ScalarType::BF16.size_in_bytes();
        let conv_state = ls
            .conv_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing conv state"))?;
        let native_linear_conv_enabled =
            conv_state.backend() == gpu_hal::Backend::Metal
                && std::env::var_os("SUPERSONIC_METAL_DISABLE_NATIVE_LINEAR_CONV_VALUE_DECAY")
                    .is_none();
        if conv_state.backend() == gpu_hal::Backend::Metal
            && (native_linear_conv_enabled
                || std::env::var_os("SUPERSONIC_METAL_ENABLE_DIRECT_CONV_STATE_UPDATE").is_some())
        {
            kernel_ffi::prefill_ffi::metal_conv_state_update_bf16(
                qkv_dim, state_len, &qkv, conv_state,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} metal conv state update: {e}"))?;
        } else {
            let new_conv_state =
                GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[qkv_dim, state_len])
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
                    new_conv_state.offset_ptr(channel_base + (state_len - 1) * state_bytes)
                        as *mut c_void,
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
        }

        kernel_ffi::prefill_ffi::cast(
            self.ordinal,
            ScalarType::F32,
            ScalarType::BF16,
            val_dim,
            &rec_apply,
            &mut attn_bf16,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;
        let attn_trace = if trace_output {
            Some(
                attn_bf16
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} attn trace D2H: {e}"))?,
            )
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
        copy_d2d_ordered(
            self.ordinal,
            recurrent_state.as_ptr() as *mut c_void,
            rec_apply.offset_ptr(val_dim * ScalarType::F32.size_in_bytes()),
            nv * khd * vhd * ScalarType::F32.size_in_bytes(),
            recurrent_state,
            &format!("layer {idx} recurrent update copy"),
        )?;

        let cached_norm_w_bf16 = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && lw.norm_w.dtype() == ScalarType::F32
            && std::env::var_os("SUPERSONIC_METAL_DISABLE_LINEAR_NORM_W_CACHE").is_none();
        let norm_w_bf16_ref = if cached_norm_w_bf16 {
            let slot = self
                .component_linear_norm_w_bf16
                .get_mut(idx)
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing norm_w cache slot"))?;
            if slot.is_none() {
                let mut cached = GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[vhd])
                    .map_err(|e| anyhow::anyhow!("layer {idx} cached norm_w alloc: {e}"))?;
                kernel_ffi::prefill_ffi::cast(
                    self.ordinal,
                    ScalarType::F32,
                    ScalarType::BF16,
                    vhd,
                    &lw.norm_w,
                    &mut cached,
                )
                .map_err(|e| anyhow::anyhow!("layer {idx} cached norm_w cast: {e}"))?;
                *slot = Some(cached);
            }
            slot.as_ref()
                .ok_or_else(|| anyhow::anyhow!("layer {idx} missing cached norm_w"))?
        } else {
            kernel_ffi::prefill_ffi::cast(
                self.ordinal,
                ScalarType::F32,
                ScalarType::BF16,
                vhd,
                &lw.norm_w,
                &mut norm_w_bf16,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} norm_w cast: {e}"))?;
            &norm_w_bf16
        };
        kernel_ffi::prefill_ffi::rms_norm_gated(
            self.ordinal,
            ScalarType::BF16,
            nv,
            vhd,
            config.rms_norm_eps as f32,
            &attn_bf16,
            &z,
            norm_w_bf16_ref,
            &mut gated,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;
        let gated_trace = if trace_output {
            Some(
                gated
                    .to_host_bytes()
                    .map_err(|e| anyhow::anyhow!("layer {idx} gated trace D2H: {e}"))?,
            )
        } else {
            None
        };

        let use_fused_residual_out_proj = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && !metal_fused_residual_projection_disabled()
            && lw.out_proj_scale.is_none()
            && lw.out_proj_int4_scale.is_none()
            && lw.out_proj_int4_zero.is_none()
            && lw.out_proj_w.dtype() == ScalarType::BF16
            && gated.dtype() == ScalarType::BF16
            && self.hidden_io.dtype() == ScalarType::BF16;
        let proj_trace = if use_fused_residual_out_proj {
            metal_matmul_residual_add_bf16(val_dim, hidden_dim, &gated, &lw.out_proj_w, &mut self.hidden_io)
                .map_err(|e| anyhow::anyhow!("layer {idx} fused residual out proj: {e}"))?;
            None
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                hidden_dim,
                val_dim,
                &gated,
                &lw.out_proj_w,
                lw.out_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                &mut proj_out,
                lw.out_proj_int4_scale.as_ref(),
                lw.out_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
            let trace = if trace_output {
                Some(
                    proj_out
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("layer {idx} proj trace D2H: {e}"))?,
                )
            } else {
                None
            };
            residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, &proj_out)?;
            trace
        };
        Ok(if trace_output {
            Some(ComponentLinearTrace {
                normed: normed_trace.unwrap_or_default(),
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
        })();

        let scratch = ComponentLinearScratch {
            qkv,
            z,
            a,
            b,
            a_beta_raw,
            rec_apply,
            attn_bf16,
            norm_w_bf16,
            gated,
            proj_out,
            conv_pack,
            q_linear,
            k_linear,
            v_linear,
            q_linear_f32,
            k_linear_f32,
            v_linear_f32,
            q_normed,
            q_scaled,
            k_normed,
        };
        (result, scratch)
    }

    fn component_decode_mlp_layer(
        &mut self,
        idx: usize,
        trace_output: bool,
    ) -> Result<Option<ComponentMlpTrace>> {
        let config = &self.weights.config;
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;

        if self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && !component_scratch_reuse_disabled()
        {
            let mut scratch = match self.component_mlp_scratch.take() {
                Some(scratch) => scratch,
                None => ComponentMlpScratch::alloc(self.ordinal, hidden_dim, intermediate)?,
            };
            let result = self.component_decode_mlp_layer_with_scratch(idx, trace_output, &mut scratch);
            self.component_mlp_scratch = Some(scratch);
            return result;
        }

        let mut scratch = ComponentMlpScratch::alloc(self.ordinal, hidden_dim, intermediate)
            .map_err(|e| anyhow::anyhow!("layer {idx} mlp scratch alloc: {e}"))?;
        self.component_decode_mlp_layer_with_scratch(idx, trace_output, &mut scratch)
    }

    fn component_decode_mlp_layer_with_scratch(
        &mut self,
        idx: usize,
        trace_output: bool,
        scratch: &mut ComponentMlpScratch,
    ) -> Result<Option<ComponentMlpTrace>> {
        let config = &self.weights.config;
        let lw = &self.weights.layers[idx];
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let gate = &mut scratch.gate;
        let up = &mut scratch.up;
        let mlp = &mut scratch.mlp;
        let down = &mut scratch.down;

        let use_fused_mlp = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && metal_fused_mlp_enabled()
            && lw.gate_proj_scale.is_none()
            && lw.gate_proj_int4_scale.is_none()
            && lw.gate_proj_int4_zero.is_none()
            && lw.up_proj_scale.is_none()
            && lw.up_proj_int4_scale.is_none()
            && lw.up_proj_int4_zero.is_none()
            && lw.down_proj_scale.is_none()
            && lw.down_proj_int4_scale.is_none()
            && lw.down_proj_int4_zero.is_none()
            && self.normed_buf.dtype() == ScalarType::BF16
            && self.hidden_io.dtype() == ScalarType::BF16
            && lw.gate_proj_w.dtype() == ScalarType::BF16
            && lw.up_proj_w.dtype() == ScalarType::BF16
            && lw.down_proj_w.dtype() == ScalarType::BF16;
        if use_fused_mlp {
            kernel_ffi::prefill_ffi::metal_qwen_mlp_gate_up_bf16(
                hidden_dim,
                intermediate,
                &self.normed_buf,
                &lw.gate_proj_w,
                &lw.up_proj_w,
                gate,
                up,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused mlp gate/up: {e}"))?;
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                intermediate,
                hidden_dim,
                &self.normed_buf,
                &lw.gate_proj_w,
                lw.gate_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                gate,
                lw.gate_proj_int4_scale.as_ref(),
                lw.gate_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
            matmul_proj(
                self.ordinal,
                1,
                1,
                intermediate,
                hidden_dim,
                &self.normed_buf,
                &lw.up_proj_w,
                lw.up_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                up,
                lw.up_proj_int4_scale.as_ref(),
                lw.up_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
            )?;
            kernel_ffi::prefill_ffi::swiglu_mul(
                self.ordinal,
                ScalarType::BF16,
                intermediate,
                gate,
                up,
                mlp,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} swiglu: {e}"))?;
        }
        let use_fused_residual_down_proj = self.hidden_io.backend() == gpu_hal::Backend::Metal
            && !trace_output
            && !metal_fused_residual_projection_disabled()
            && lw.down_proj_scale.is_none()
            && lw.down_proj_int4_scale.is_none()
            && lw.down_proj_int4_zero.is_none()
            && lw.down_proj_w.dtype() == ScalarType::BF16
            && mlp.dtype() == ScalarType::BF16
            && self.hidden_io.dtype() == ScalarType::BF16;
        let trace = if use_fused_mlp {
            let residual: &GpuBuffer = unsafe { &*(&self.hidden_io as *const GpuBuffer) };
            kernel_ffi::prefill_ffi::metal_qwen_mlp_down_residual_bf16(
                hidden_dim,
                intermediate,
                gate,
                up,
                &lw.down_proj_w,
                residual,
                &mut self.hidden_io,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} fused mlp down/residual: {e}"))?;
            None
        } else if use_fused_residual_down_proj {
            metal_matmul_residual_add_bf16(intermediate, hidden_dim, mlp, &lw.down_proj_w, &mut self.hidden_io)
                .map_err(|e| anyhow::anyhow!("layer {idx} fused residual down proj: {e}"))?;
            None
        } else {
            matmul_proj(
                self.ordinal,
                1,
                1,
                hidden_dim,
                intermediate,
                &mlp,
                &lw.down_proj_w,
                lw.down_proj_scale.as_ref(),
                self.weights.fp8_block_size,
                down,
                lw.down_proj_int4_scale.as_ref(),
                lw.down_proj_int4_zero.as_ref(),
                self.weights.int4_group_size,
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
            residual_add(self.ordinal, hidden_dim, &mut self.hidden_io, down)?;
            trace
        };
        Ok(trace)
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
        let argmax_buf = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("argmax_buf: {e}"))?;
        let lm_head_block_best_vals = GpuBuffer::zeros(ordinal, ScalarType::F32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_vals: {e}"))?;
        let lm_head_block_best_idxs = GpuBuffer::zeros(ordinal, ScalarType::U32, &[512])
            .map_err(|e| anyhow::anyhow!("lm_head_block_best_idxs: {e}"))?;
        let matvec_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("matvec_counter: {e}"))?;
        let num_hidden_layers = config.num_hidden_layers;

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
            dflash_tap_cache: None,
            dflash_fused_verify_cache: None,
            component_mlp_scratch: None,
            component_linear_scratch: None,
            component_full_scratch: None,
            component_linear_norm_w_bf16: (0..num_hidden_layers).map(|_| None).collect(),
        })
    }

    pub fn weights(&self) -> &Qwen35Weights {
        &self.weights
    }

    pub fn kv_fp8_enabled(&self) -> bool {
        self.kv_fp8
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

    /// Load prefill state from oracle output into GPU buffers.
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
            None,
            None,
        )?;

        // Reset sync counters for the decode kernel
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result.logits)
    }

    /// Run native GPU prefill and return only the greedy next token. Metal uses
    /// this to avoid materializing the full vocabulary logits during generation.
    pub fn prefill_native_greedy_token(&mut self, prompt_ids: &[u32]) -> Result<u32> {
        let token = prefill_engine::prefill_greedy_token(
            &self.weights,
            &mut self.state,
            &self.rotary,
            prompt_ids,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
        )?;

        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after greedy prefill: {e}"))?;

        Ok(token)
    }

    /// Rebuild sequence-0 state from scratch by replaying native GPU prefill
    /// over the provided token history. Optionally replicates that state across
    /// extra batch slots for lockstep batch decoding.
    pub fn rebuild_prefill_state(
        &mut self,
        token_ids: &[u32],
        replicate_batch: bool,
    ) -> Result<Vec<f32>> {
        self.state.reset_for_prefill_reuse();
        let logits = self.prefill_native(token_ids)?;
        if replicate_batch && self.batch_size > 1 {
            self.replicate_state_to_batch()?;
        }
        Ok(logits)
    }

    /// Rebuild sequence-0 state from scratch and return only the greedy token.
    pub fn rebuild_prefill_state_greedy_token(&mut self, token_ids: &[u32]) -> Result<u32> {
        self.state.reset_for_prefill_reuse();
        self.prefill_native_greedy_token(token_ids)
    }

    /// DFlash prefill: runs `prefill_with_taps` against the engine's own
    /// target state + weights, returning the regular PrefillResult with
    /// its `tap_hiddens` populated for the layers in `tap_layers`.
    pub fn prefill_native_with_taps(
        &mut self,
        prompt_ids: &[u32],
        tap_layers: &[usize],
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill_with_taps(
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
            None,
            None,
            tap_layers,
        )?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after dflash prefill: {e}"))?;
        Ok(result)
    }

    pub fn prefill_native_with_trace(
        &mut self,
        prompt_ids: &[u32],
        debug_linear_layer: Option<usize>,
        debug_full_layer: Option<usize>,
        debug_mlp_layer: Option<usize>,
        trace_position: Option<usize>,
    ) -> Result<prefill_engine::PrefillResult> {
        let result = prefill_engine::prefill_with_trace_position(
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
            debug_linear_layer,
            debug_full_layer,
            debug_mlp_layer,
            trace_position,
        )?;

        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result)
    }

    /// Debug-only helper that replays the tail of prefill starting from a
    /// caller-supplied BF16 hidden-state tensor `[seq_len, hidden_dim]`.
    /// Uses a fresh temporary model state so the live decode state is untouched.
    pub fn prefill_tail_from_hidden_with_trace(
        &self,
        hidden_bytes: &[u8],
        start_layer: usize,
        debug_linear_layer: Option<usize>,
        debug_full_layer: Option<usize>,
        debug_mlp_layer: Option<usize>,
        trace_position: Option<usize>,
    ) -> Result<prefill_engine::PrefillResult> {
        let mut state = ModelState::new(&self.weights.config, self.ordinal)
            .map_err(|e| anyhow::anyhow!("tail replay model state init: {e}"))?;
        prefill_engine::prefill_tail_from_hidden_with_trace_position(
            &self.weights,
            &mut state,
            &self.rotary,
            hidden_bytes,
            start_layer,
            self.ordinal,
            self.kv_chunk_size,
            self.prefill_chunk_size,
            self.kv_fp8,
            self.use_4b_kernel,
            true,
            debug_linear_layer,
            debug_full_layer,
            debug_mlp_layer,
            trace_position,
        )
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
                ls.ensure_kv_capacity(
                    seqlen_offset,
                    self.ordinal,
                    config,
                    self.kv_chunk_size,
                    self.kv_fp8,
                )
                .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {i}: {e}"))?;
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
        persist_result.map_err(|e| anyhow::anyhow!("persistent_decode kernel: {e}"))?;
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
                let sampled_token =
                    u32::from_le_bytes(token_bytes[..4].try_into().map_err(|_| {
                        anyhow::anyhow!("argmax D2H returned truncated token buffer")
                    })?);

                Ok(DecodeStepOutput {
                    logits: None,
                    sampled_token,
                    timings,
                })
            }
            DecodeSamplingMode::CudaFastGreedy | DecodeSamplingMode::HostLogits => {
                // 8. lm_head projection → logits (work-stealing matvec)
                let start = Instant::now();
                let metal_host_logits = if sampling_mode == DecodeSamplingMode::HostLogits
                    && self.hidden_io.backend() == gpu_hal::Backend::Metal
                {
                    Some(
                        kernel_ffi::qwen_rms_norm_standalone_matvec_host_f32(
                            self.ordinal,
                            ScalarType::BF16,
                            &self.hidden_io,
                            &self.weights.norm_weight,
                            config.rms_norm_eps as f32,
                            &*self.weights.lm_head,
                            config.hidden_size,
                            config.vocab_size,
                        )
                        .map_err(|e| anyhow::anyhow!("lm_head host-f32 matvec: {e}"))?,
                    )
                } else {
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
                    None
                };
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
                    let sampled_token =
                        u32::from_le_bytes(token_bytes[..4].try_into().map_err(|_| {
                            anyhow::anyhow!("argmax D2H returned truncated token buffer")
                        })?);

                    return Ok(DecodeStepOutput {
                        logits: None,
                        sampled_token,
                        timings,
                    });
                }

                // 9. Copy logits to CPU and convert BF16 → F32
                let logits_f32: Vec<f32> = if let Some(logits) = metal_host_logits {
                    timings.logits_d2h_ms = 0.0;
                    logits
                } else {
                    let start = Instant::now();
                    let logits_bytes = self
                        .logits_buf
                        .to_host_bytes()
                        .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
                    timings.logits_d2h_ms = start.elapsed().as_secs_f64() * 1000.0;
                    logits_bytes
                        .chunks_exact(2)
                        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                        .collect()
                };
                let start = Instant::now();
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
        let out =
            self.decode_step_non_4b(token_id, seqlen_offset, DecodeSamplingMode::HostLogits)?;
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
        let out =
            self.decode_step_non_4b(token_id, seqlen_offset, DecodeSamplingMode::CudaFastGreedy)?;
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

    /// Prototype Metal component decode path for Qwen3.5 0.8B.
    ///
    /// This consumes the ModelState produced by native Metal prefill and
    /// advances it by one token instead of replaying the whole prompt. It is
    /// intentionally opt-in from the CLI while correctness/perf are measured.
    pub fn decode_step_metal_component_greedy(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
    ) -> Result<(u32, DecodeStageTimings)> {
        if self.hidden_io.backend() != gpu_hal::Backend::Metal {
            anyhow::bail!("decode_step_metal_component_greedy requires Metal backend");
        }
        if self.use_4b_kernel {
            anyhow::bail!("decode_step_metal_component_greedy is scoped to the non-4B path");
        }
        let start = Instant::now();
        let metal_batch_guard = if std::env::var_os("SUPERSONIC_METAL_DISABLE_BATCH").is_none() {
            Some(
                kernel_ffi::prefill_ffi::MetalBatchGuard::begin()
                    .map_err(|e| anyhow::anyhow!("begin Metal component decode batch: {e}"))?,
            )
        } else {
            None
        };
        let (_, sampled_token, _, _, _) =
            self.component_decode_step_4b_impl(token_id, seqlen_offset, true, None, None, None)?;
        if let Some(guard) = metal_batch_guard {
            guard
                .finish()
                .map_err(|e| anyhow::anyhow!("finish Metal component decode batch: {e}"))?;
        }
        let mut timings = DecodeStageTimings::default();
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let token =
            sampled_token.ok_or_else(|| anyhow::anyhow!("Metal component decode missing token"))?;
        Ok((token, timings))
    }

    pub fn decode_step_metal_component_greedy_trace_linear_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(u32, DecodeStageTimings, ComponentLinearTrace)> {
        if self.hidden_io.backend() != gpu_hal::Backend::Metal {
            anyhow::bail!(
                "decode_step_metal_component_greedy_trace_linear_layer requires Metal backend"
            );
        }
        if self.use_4b_kernel {
            anyhow::bail!(
                "decode_step_metal_component_greedy_trace_linear_layer is scoped to the non-4B path"
            );
        }
        let start = Instant::now();
        let metal_batch_guard = if std::env::var_os("SUPERSONIC_METAL_TRACE_WITH_BATCH").is_some() {
            Some(
                kernel_ffi::prefill_ffi::MetalBatchGuard::begin().map_err(|e| {
                    anyhow::anyhow!("begin Metal component decode trace batch: {e}")
                })?,
            )
        } else {
            None
        };
        let (_, sampled_token, _, _, trace) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            true,
            None,
            None,
            Some(trace_layer),
        )?;
        if let Some(guard) = metal_batch_guard {
            guard
                .finish()
                .map_err(|e| anyhow::anyhow!("finish Metal component decode trace batch: {e}"))?;
        }
        let mut timings = DecodeStageTimings::default();
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let token =
            sampled_token.ok_or_else(|| anyhow::anyhow!("Metal component decode missing token"))?;
        let trace = trace.ok_or_else(|| {
            anyhow::anyhow!("missing Metal component linear trace for layer {trace_layer}")
        })?;
        Ok((token, timings, trace))
    }

    pub fn decode_step_metal_component_greedy_trace_input_layer(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        trace_layer: usize,
    ) -> Result<(u32, DecodeStageTimings, Vec<u8>)> {
        if self.hidden_io.backend() != gpu_hal::Backend::Metal {
            anyhow::bail!(
                "decode_step_metal_component_greedy_trace_input_layer requires Metal backend"
            );
        }
        if self.use_4b_kernel {
            anyhow::bail!(
                "decode_step_metal_component_greedy_trace_input_layer is scoped to the non-4B path"
            );
        }
        let start = Instant::now();
        let metal_batch_guard = if std::env::var_os("SUPERSONIC_METAL_TRACE_WITH_BATCH").is_some() {
            Some(
                kernel_ffi::prefill_ffi::MetalBatchGuard::begin().map_err(|e| {
                    anyhow::anyhow!("begin Metal component decode input trace batch: {e}")
                })?,
            )
        } else {
            None
        };
        let (_, sampled_token, trace, _, _) = self.component_decode_step_4b_impl(
            token_id,
            seqlen_offset,
            true,
            Some(trace_layer),
            None,
            None,
        )?;
        if let Some(guard) = metal_batch_guard {
            guard.finish().map_err(|e| {
                anyhow::anyhow!("finish Metal component decode input trace batch: {e}")
            })?;
        }
        let mut timings = DecodeStageTimings::default();
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let token =
            sampled_token.ok_or_else(|| anyhow::anyhow!("Metal component decode missing token"))?;
        let trace = trace.ok_or_else(|| {
            anyhow::anyhow!("missing Metal component input trace for layer {trace_layer}")
        })?;
        Ok((token, timings, trace))
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

    /// One decode step via the 4B persistent megakernel, capturing DFlash
    /// hidden-state taps for the specified target layers.
    ///
    /// Returns `(logits_f32, tap_hiddens_bf16_bytes)`:
    /// * `logits_f32` — `[vocab_size]` F32 logits for the next position.
    /// * `tap_hiddens_bf16_bytes` — raw BF16 bytes of shape
    ///   `[num_taps, hidden_dim]`, one row per entry in `tap_layers`.
    ///
    /// Requires `use_4b_kernel=true`, `batch_size=1`, and a non-empty
    /// `tap_layers`. Every element of `tap_layers` must be in
    /// `0..num_hidden_layers`. The tap values match what the persistent
    /// megakernel writes for each listed layer — the post-MLP residual
    /// hidden state, i.e. the same data point captured by
    /// `prefill_with_taps` / `layer_hidden_trace`.
    pub fn decode_step_with_taps_kernel(
        &mut self,
        token_id: u32,
        seqlen_offset: usize,
        tap_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<u8>)> {
        if !self.use_4b_kernel {
            anyhow::bail!("decode_step_with_taps_kernel requires use_4b_kernel");
        }
        if self.batch_size != 1 {
            anyhow::bail!("decode_step_with_taps_kernel requires batch_size=1");
        }
        if tap_layers.is_empty() {
            anyhow::bail!("decode_step_with_taps_kernel requires at least one tap layer");
        }
        let config = &self.weights.config;
        let num_layers = config.num_hidden_layers;
        for &li in tap_layers {
            if li >= num_layers {
                anyhow::bail!("tap layer {li} out of range (num_hidden_layers={num_layers})");
            }
        }

        // 1) Tap workspace + i32-layer-indices: reuse the cache if tap_layers
        //    hasn't changed, otherwise (re)allocate once. DFlash calls this
        //    in a tight loop with a fixed tap_layers list, so the second+
        //    call pays zero allocation / upload cost here.
        //
        // Take the cache out of `self` into locals for the kernel call
        // (split-borrow through `Option::as_mut` conflicts with the many
        // other `&self` / `&mut self.*` borrows persistent_decode_4b needs);
        // put it back after a successful kernel launch. Kernel error paths
        // drop the cache, which is fine — next call re-allocates.
        let hidden_dim = config.hidden_size;
        let num_taps = tap_layers.len();
        let (mut tap_workspace, tap_layers_buf) = match self.dflash_tap_cache.take() {
            Some((cached, ws, lb)) if cached.as_slice() == tap_layers => (ws, lb),
            _ => {
                let workspace =
                    GpuBuffer::zeros(self.ordinal, ScalarType::BF16, &[num_taps, hidden_dim])
                        .map_err(|e| anyhow::anyhow!("alloc tap_workspace: {e}"))?;
                let tap_ints: Vec<i32> = tap_layers.iter().map(|&li| li as i32).collect();
                let tap_ints_bytes: Vec<u8> =
                    tap_ints.iter().flat_map(|v| v.to_le_bytes()).collect();
                let layers_buf = GpuBuffer::from_host_bytes(
                    self.ordinal,
                    ScalarType::U8,
                    &[tap_ints_bytes.len()],
                    &tap_ints_bytes,
                )
                .map_err(|e| anyhow::anyhow!("upload tap_layers: {e}"))?;
                (workspace, layers_buf)
            }
        };

        // 2) Embedding lookup → hidden_io.
        let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
        let src_offset = token_id as usize * row_bytes;
        gpu_hal::copy_d2d(
            self.ordinal,
            self.hidden_io.as_ptr() as *mut c_void,
            self.weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps embedding: {e}"))?;

        // 3) Ensure KV capacity on full-attention layers.
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.ensure_kv_capacity(
                    seqlen_offset,
                    self.ordinal,
                    config,
                    self.kv_chunk_size,
                    self.kv_fp8,
                )
                .map_err(|e| anyhow::anyhow!("dflash-taps ensure KV layer {i}: {e}"))?;
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

        // 4) Build + upload layer descs (pointers + kv_len change each step).
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("dflash-taps upload descs: {e}"))?;
        if let Some(kv_fp8_descs) = build_kv_fp8_descs(&self.state, self.kv_fp8) {
            self.scratch
                .upload_kv_fp8_descs(&kv_fp8_descs)
                .map_err(|e| anyhow::anyhow!("dflash-taps upload KV FP8 descs: {e}"))?;
        }
        gpu_hal::memset_zeros(
            self.ordinal,
            self.scratch.workspace.as_mut_ptr(),
            self.scratch.workspace.len_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps clear workspace: {e}"))?;
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("dflash-taps reset sync: {e}"))?;

        // 5) Launch the 4B megakernel with taps enabled.
        kernel_ffi::persistent_decode_4b(
            self.ordinal,
            ScalarType::BF16,
            num_layers,
            hidden_dim,
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
            1, // batch_size=1
            None,
            self.int4_scale_device.as_ref(),
            false, // enable_timing_slots
            false, // enable_attention_trace
            Some(&mut tap_workspace),
            Some(&tap_layers_buf),
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps persistent_decode_4b: {e}"))?;

        // 6) Advance kv_filled on every full-attention layer.
        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        // 7) Final RMSNorm + lm_head → logits F32.
        kernel_ffi::rms_norm_4b(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            hidden_dim,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps final rms_norm: {e}"))?;
        kernel_ffi::standalone_matvec_4b(
            self.ordinal,
            ScalarType::BF16,
            &mut self.logits_buf,
            &self.normed_buf,
            &*self.weights.lm_head,
            hidden_dim,
            config.vocab_size,
            &mut self.matvec_counter,
        )
        .map_err(|e| anyhow::anyhow!("dflash-taps lm_head matvec: {e}"))?;
        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("dflash-taps logits D2H: {e}"))?;
        let logits_f32: Vec<f32> = logits_bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();

        // 8) D2H the tap workspace, then put the workspace + layer-ids
        // back into the cache so subsequent calls with the same tap_layers
        // avoid the allocation.
        let tap_host = tap_workspace
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("dflash-taps D2H: {e}"))?;
        self.dflash_tap_cache = Some((tap_layers.to_vec(), tap_workspace, tap_layers_buf));

        Ok((logits_f32, tap_host))
    }

    /// Mutable access to the engine's primary `ModelState`. Used by the
    /// DFlash speculative engine to snapshot/restore linear-attention state.
    pub fn state_mut(&mut self) -> &mut ModelState {
        &mut self.state
    }

    /// Device ordinal carried by the engine. Used by the DFlash engine when
    /// invoking free-function helpers (e.g. `ModelState::restore_linear`).
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Rewind every full-attention layer's `kv_filled` cursor to `new_len`
    /// (no-op if already at or below). The physical K/V beyond the cursor is
    /// untouched and will be harmlessly overwritten by subsequent decodes —
    /// used by the DFlash engine after a partial-acceptance verify to roll
    /// the cache logically back to the committed length.
    pub fn rewind_full_kv_filled(&mut self, new_len: usize) {
        let config = &self.weights.config;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) && ls.kv_filled > new_len {
                ls.set_kv_filled(new_len);
            }
        }
    }

    /// DFlash M4.3 fused verify: single `persistent_decode_4b` megakernel
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
    /// but `kv_filled` is NOT advanced on any layer — the DFlash engine
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
        let (hidden_dim, intermediate_size, vocab_size, num_layers, rms_norm_eps) = {
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
        //   (block_size + max(B × hidden_dim, intermediate_size) + fp8_lut) × sizeof(f32)
        // with kernel block_size = 256 and fp8_lut = 256. gfx1150 caps
        // LDS at 64 KiB per workgroup → 16384 floats total. Reserve 2
        // KiB (512 floats) for block_size + fp8_lut, leaving 15872
        // floats for the input cache. 9B (hidden=4096) tops out at B=3;
        // 4B (hidden=2048) tops out at B=7. If a user passes a larger
        // --dflash-block the launch fails with HIP status 254 and a
        // confusing error — fail fast here instead with the math
        // spelled out.
        const MAX_INPUT_CACHE_FLOATS: usize = 15872;
        let input_cache = (b * hidden_dim).max(intermediate_size);
        if input_cache > MAX_INPUT_CACHE_FLOATS {
            anyhow::bail!(
                "verify_block_fused_decode: shared-memory budget exceeded \
                 (B={b} × hidden_dim={hidden_dim} = {}, intermediate={intermediate_size}; \
                 cap = {MAX_INPUT_CACHE_FLOATS} floats). \
                 Lower --dflash-block to ≤ {}.",
                b * hidden_dim,
                MAX_INPUT_CACHE_FLOATS.min(b * hidden_dim) / hidden_dim.max(1),
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
        let mut cache = match self.dflash_fused_verify_cache.take() {
            Some(c) if c.block_size == b => c,
            _ => DFlashFusedVerifyCache::alloc(
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

        // Launch the fused megakernel. `pos_offset` as the kernel's
        // `seqlen_offset` arg is ignored because `batch_descs` is
        // non-null; pass it through for consistency with the batched
        // call site.
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
            None,  // tap_workspace: verify doesn't capture taps — re-decode does
            None,  // tap_layers: ignored when tap_workspace is None
        )
        .map_err(|e| anyhow::anyhow!("fused verify persistent_decode_4b: {e}"))?;

        // Deliberately do NOT advance `kv_filled` on any layer. The
        // DFlash engine rolls the K/V cursor back via
        // `rewind_full_kv_filled` and the linear state via
        // `restore_linear` after the accept decision.

        // Final RMSNorm (multirow) + tiled lm_head over all B hiddens.
        kernel_ffi::rms_norm_4b_multirow(
            self.ordinal,
            ScalarType::BF16,
            b,
            hidden_dim,
            rms_norm_eps,
            &cache.hidden_io,
            &self.weights.norm_weight,
            &mut cache.normed_buf,
        )
        .map_err(|e| anyhow::anyhow!("fused verify final rms_norm: {e}"))?;

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

        let logits_host = cache
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("fused verify logits D2H: {e}"))?;
        let row_stride_bytes = vocab_size * ScalarType::BF16.size_in_bytes();
        let mut logits_per_pos = Vec::with_capacity(b);
        for bi in 0..b {
            let start = bi * row_stride_bytes;
            let end = start + row_stride_bytes;
            let row: Vec<f32> = logits_host[start..end]
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            logits_per_pos.push(row);
        }

        self.dflash_fused_verify_cache = Some(cache);
        Ok(logits_per_pos)
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
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity batch {bi} layer {i}: {e}"))?;
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
            for bi in 0..self.extra_states.len() {
                Self::load_kv_shadow_for_state_static(
                    &self.weights.config,
                    self.ordinal,
                    &mut self.extra_states[bi],
                )?;
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
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8)
        {
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
        // The sm86 hero kernel is tuned and validated only for Qwen3.5 4B geometry.
        // Qwen3.5 2B/9B also use the 4B persistent lane, so we must gate explicitly.
        let is_qwen35_4b_geometry = config.hidden_size == 2560;
        let use_qwen35_4b_cuda_hero = self.hidden_io.backend() == gpu_hal::Backend::Cuda
            && is_qwen35_4b_geometry
            && b == 1
            && self.fp8_scale_device.is_none()
            && self.int4_scale_device.is_none()
            && !self.kv_fp8;
        let persist_result = if use_qwen35_4b_cuda_hero {
            kernel_ffi::persistent_decode_4b_qwen35_sm86_specialized(
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
                None, // tap_workspace: DFlash-only, off in batched decode
                None, // tap_layers: DFlash-only, off in batched decode
            )
        } else {
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
                None, // tap_workspace: DFlash-only, off in batched decode
                None, // tap_layers: DFlash-only, off in batched decode
            )
        };
        persist_result.map_err(|e| anyhow::anyhow!("persistent_decode_4b batch kernel: {e}"))?;
        timings.persistent_ms = start.elapsed().as_secs_f64() * 1000.0;
        if enable_timing_slots {
            let clock_rate_khz = match self.hidden_io.backend() {
                gpu_hal::Backend::Cuda => {
                    gpu_hal::query_device_info(gpu_hal::Backend::Cuda, self.ordinal)
                        .map_err(|e| anyhow::anyhow!("query CUDA device clock rate: {e}"))?
                        .clock_rate_khz
                }
                gpu_hal::Backend::Hip => kernel_ffi::query_hip_device_clock_khz(self.ordinal)
                    .map_err(|e| anyhow::anyhow!("query HIP device clock rate: {e}"))?,
                gpu_hal::Backend::Metal => 0,
            };
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
            let st = if bi == 0 {
                &mut self.state
            } else {
                &mut self.extra_states[bi - 1]
            };
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
                    .map_err(|e| {
                        anyhow::anyhow!("trace ensure KV capacity batch {bi} layer {i}: {e}")
                    })?;
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
        if let Some(batch_descs) = build_batch_seq_descs(&state_refs, &seqlen_offsets, self.kv_fp8)
        {
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
            None, // tap_workspace: DFlash-only, off in trace path
            None, // tap_layers: DFlash-only, off in trace path
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
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent projection slice out of bounds"
        );
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
        anyhow::ensure!(
            token_out_end <= bytes.len(),
            "persistent MLP slice out of bounds"
        );
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
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn gated slice out of bounds"
        );
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
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
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
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn saved_gate slice out of bounds"
        );
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
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
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
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn q slice out of bounds"
        );
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
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
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
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn pre_gate slice out of bounds"
        );
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
        let q_dim = self.weights.config.num_attention_heads * self.weights.config.head_dim;
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
        let start = (attn_scratch_base + batch_index * self.attn_scratch_floats + q_dim * 3) * 4;
        let end = start + self.weights.config.num_attention_heads * self.kv_chunk_size * 4;
        anyhow::ensure!(
            end <= bytes.len(),
            "persistent full-attn scores slice out of bounds"
        );
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
