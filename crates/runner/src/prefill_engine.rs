//! Native HIP prefill engine — replaces the Python oracle.
//!
//! Orchestrates component kernels (embedding, matmul, attention, conv, recurrence,
//! norms, MLP) to process the entire prompt sequence through the model on GPU.

use std::ffi::c_void;

use anyhow::Result;
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::config::TextConfig;
use qwen35::rotary::RotaryTables;
use qwen35::state::ModelState;
use qwen35::weights::Qwen35Weights;

use kernel_ffi::prefill_ffi;

/// In-place residual add: dst += src.
/// Uses unsafe to work around the borrow checker since the GPU kernel
/// reads src[i] and writes dst[i] independently per element.
fn residual_add(
    ordinal: usize,
    total_elems: usize,
    dst: &mut GpuBuffer,
    src: &GpuBuffer,
) -> Result<()> {
    unsafe {
        let status = dotcache_qwen35_hip_element_add(
            ScalarType::BF16.kernel_dtype_code(),
            ordinal,
            total_elems,
            dst.as_ptr(),
            src.as_ptr(),
            dst.as_mut_ptr(),
        );
        if status != 0 {
            anyhow::bail!("residual_add failed: {status}");
        }
    }
    Ok(())
}

unsafe extern "C" {
    fn dotcache_qwen35_hip_element_add(
        dtype: std::ffi::c_int,
        device_ordinal: usize,
        total_elems: usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    ) -> std::ffi::c_int;
}

/// Result of a prefill pass.
pub struct PrefillResult {
    /// Logits for the last token position [vocab_size] as F32 on CPU.
    pub logits: Vec<f32>,
}

/// Scratch buffers for prefill (larger than decode — seq_len > 1).
struct PrefillScratch {
    /// [seq_len, hidden_dim] BF16 — main hidden state
    hidden: GpuBuffer,
    /// [seq_len, hidden_dim] BF16 — normed activations
    normed: GpuBuffer,
    /// [seq_len, max_proj_dim] BF16 — projection output buffer
    proj_buf: GpuBuffer,
    /// [seq_len, max_proj_dim] BF16 — second projection buffer (for gate/up)
    proj_buf2: GpuBuffer,
    /// [seq_len, intermediate_size] BF16 — MLP intermediate
    mlp_buf: GpuBuffer,
    /// [1, vocab_size] BF16 — logits output
    logits_buf: GpuBuffer,
    // Full attention scratch:
    /// [num_q_heads, seq_len, head_dim] BF16 — transposed Q for attention
    attn_q: GpuBuffer,
    /// [num_kv_heads, seq_len, head_dim] BF16 — transposed K
    attn_k: GpuBuffer,
    /// [num_kv_heads, seq_len, head_dim] BF16 — transposed V
    attn_v: GpuBuffer,
    /// [num_q_heads, seq_len, head_dim] F32 — attention output
    attn_out_f32: GpuBuffer,
    // Linear attention scratch:
    /// [qkv_dim, seq_len + kern - 1] BF16 — padded conv input
    conv_input: GpuBuffer,
}

impl PrefillScratch {
    fn new(config: &TextConfig, seq_len: usize, ordinal: usize) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kern = config.linear_conv_kernel_dim;

        // Max projection dim across all layer types
        let max_proj = std::cmp::max(
            // Full attention: q_out + k_out + v_out
            num_q_heads * head_dim + num_kv_heads * head_dim * 2,
            // Linear attention: qkv_out (largest single projection)
            config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim,
        );

        let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_total_len = seq_len + kern - 1;

        Ok(Self {
            hidden: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill hidden: {e}"))?,
            normed: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, hidden_dim])
                .map_err(|e| anyhow::anyhow!("prefill normed: {e}"))?,
            proj_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf: {e}"))?,
            proj_buf2: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, max_proj])
                .map_err(|e| anyhow::anyhow!("prefill proj_buf2: {e}"))?,
            mlp_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, intermediate])
                .map_err(|e| anyhow::anyhow!("prefill mlp_buf: {e}"))?,
            logits_buf: GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("prefill logits: {e}"))?,
            attn_q: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_q_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_q: {e}"))?,
            attn_k: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_k: {e}"))?,
            attn_v: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[num_kv_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_v: {e}"))?,
            attn_out_f32: GpuBuffer::zeros(
                ordinal,
                ScalarType::F32,
                &[num_q_heads, seq_len, head_dim],
            )
            .map_err(|e| anyhow::anyhow!("prefill attn_out_f32: {e}"))?,
            conv_input: GpuBuffer::zeros(
                ordinal,
                ScalarType::BF16,
                &[qkv_dim, conv_total_len],
            )
            .map_err(|e| anyhow::anyhow!("prefill conv_input: {e}"))?,
        })
    }
}

/// Run native prefill on GPU, returning logits and leaving state filled.
pub fn prefill(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    prompt_ids: &[u32],
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<PrefillResult> {
    let config = &weights.config;
    let seq_len = prompt_ids.len();
    let hidden_dim = config.hidden_size;

    // Allocate scratch buffers
    let mut scratch = PrefillScratch::new(config, seq_len, ordinal)?;

    // Upload token IDs to GPU
    let id_bytes: Vec<u8> = prompt_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let token_ids_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::U32, &[seq_len], &id_bytes)
        .map_err(|e| anyhow::anyhow!("upload token IDs: {e}"))?;

    // 1. Embedding lookup: token IDs → hidden [seq_len, hidden_dim]
    prefill_ffi::embedding_lookup(
        ordinal,
        ScalarType::BF16,
        seq_len,
        config.vocab_size,
        hidden_dim,
        &weights.embed_tokens,
        &token_ids_gpu,
        &mut scratch.hidden,
    )?;

    // 2. Layer loop
    for idx in 0..config.num_hidden_layers {
        // Input RMSNorm (multi-row for seq_len > 1)
        prefill_ffi::rms_norm_rows(
            ordinal,
            ScalarType::BF16,
            seq_len,
            hidden_dim,
            config.rms_norm_eps as f32,
            &scratch.hidden,
            &weights.layers[idx].input_norm_w,
            &mut scratch.normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} input norm: {e}"))?;

        if config.is_full_attention(idx) {
            prefill_full_attention_layer(
                weights, state, rotary, &mut scratch, config, idx, seq_len, ordinal, kv_chunk_size,
            )?;
        } else {
            prefill_linear_attention_layer(
                weights, state, &mut scratch, config, idx, seq_len, ordinal,
            )?;
        }

        // Post-attention RMSNorm (multi-row)
        prefill_ffi::rms_norm_rows(
            ordinal,
            ScalarType::BF16,
            seq_len,
            hidden_dim,
            config.rms_norm_eps as f32,
            &scratch.hidden,
            &weights.layers[idx].post_attn_norm_w,
            &mut scratch.normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} post-attn norm: {e}"))?;

        // MLP: gate_proj + up_proj → SwiGLU → down_proj → residual add
        prefill_mlp_layer(weights, &mut scratch, config, idx, seq_len, ordinal)?;
    }

    // 3. Final RMSNorm on last token only
    let last_token_offset = (seq_len - 1) * hidden_dim * ScalarType::BF16.size_in_bytes();
    let last_hidden_ptr = scratch.hidden.offset_ptr(last_token_offset);
    let last_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
        .map_err(|e| anyhow::anyhow!("last_hidden alloc: {e}"))?;
    gpu_hal::copy_d2d(
        ordinal,
        last_hidden.as_ptr() as *mut c_void,
        last_hidden_ptr,
        hidden_dim * ScalarType::BF16.size_in_bytes(),
    )
    .map_err(|e| anyhow::anyhow!("copy last hidden: {e}"))?;

    let mut normed_last = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
        .map_err(|e| anyhow::anyhow!("normed_last alloc: {e}"))?;
    prefill_ffi::rms_norm_rows(
        ordinal,
        ScalarType::BF16,
        1,
        hidden_dim,
        config.rms_norm_eps as f32,
        &last_hidden,
        &weights.norm_weight,
        &mut normed_last,
    )
    .map_err(|e| anyhow::anyhow!("final norm: {e}"))?;

    // 4. lm_head projection → logits [1, vocab_size]
    let mut counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .map_err(|e| anyhow::anyhow!("matvec counter: {e}"))?;
    kernel_ffi::standalone_matvec(
        ordinal,
        ScalarType::BF16,
        &mut scratch.logits_buf,
        &normed_last,
        &*weights.lm_head,
        hidden_dim,
        config.vocab_size,
        &mut counter,
    )
    .map_err(|e| anyhow::anyhow!("lm_head: {e}"))?;

    // 5. Copy logits to CPU, convert BF16 → F32
    let logits_bytes = scratch
        .logits_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
    let logits: Vec<f32> = logits_bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    Ok(PrefillResult { logits })
}

fn prefill_full_attention_layer(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    rotary: &RotaryTables,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    seq_len: usize,
    ordinal: usize,
    kv_chunk_size: usize,
) -> Result<()> {
    let fw = weights.layers[idx]
        .full
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected full attention weights"))?;

    let hidden_dim = config.hidden_size;
    let num_q_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let rotary_dim = config.rotary_dim();

    // 1. Q projection: normed [seq, hidden] × q_proj_w [q_dim, hidden]^T → [seq, q_dim]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        q_dim,
        hidden_dim,
        &scratch.normed,
        &fw.q_proj_w,
        &mut scratch.proj_buf,
    )?;

    // 2. K projection: normed [seq, hidden] × k_proj_w [kv_dim, hidden]^T → [seq, kv_dim]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.k_proj_w,
        &mut scratch.proj_buf2,
    )?;

    // 3. Q normalization: RMSNorm per head [seq*num_q_heads, head_dim]
    //    In-place: proj_buf treated as [seq*num_q_heads, head_dim]
    {
        let mut q_normed =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * num_q_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("q_normed alloc: {e}"))?;
        prefill_ffi::rms_norm_rows(
            ordinal,
            ScalarType::BF16,
            seq_len * num_q_heads,
            head_dim,
            1e-6,
            &scratch.proj_buf,
            &fw.q_norm_w,
            &mut q_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q norm: {e}"))?;
        // Copy back (same size buffer, different logical shape)
        gpu_hal::copy_d2d(
            ordinal,
            scratch.proj_buf.as_ptr() as *mut c_void,
            q_normed.as_ptr(),
            seq_len * q_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} Q norm copy: {e}"))?;
    }

    // 4. K normalization: RMSNorm per head [seq*num_kv_heads, head_dim]
    {
        let mut k_normed =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * num_kv_heads, head_dim])
                .map_err(|e| anyhow::anyhow!("k_normed alloc: {e}"))?;
        prefill_ffi::rms_norm_rows(
            ordinal,
            ScalarType::BF16,
            seq_len * num_kv_heads,
            head_dim,
            1e-6,
            &scratch.proj_buf2,
            &fw.k_norm_w,
            &mut k_normed,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K norm: {e}"))?;
        gpu_hal::copy_d2d(
            ordinal,
            scratch.proj_buf2.as_ptr() as *mut c_void,
            k_normed.as_ptr(),
            seq_len * kv_dim * ScalarType::BF16.size_in_bytes(),
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} K norm copy: {e}"))?;
    }

    // 5. RoPE on Q [seq, num_q_heads, head_dim] and K [seq, num_kv_heads, head_dim]
    prefill_ffi::apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_q_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        &mut scratch.proj_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q RoPE: {e}"))?;

    prefill_ffi::apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_kv_heads,
        head_dim,
        rotary_dim,
        &rotary.cos,
        &rotary.sin,
        &mut scratch.proj_buf2,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} K RoPE: {e}"))?;

    // 6. V projection: normed [seq, hidden] × v_proj_w [kv_dim, hidden]^T → [seq, kv_dim]
    //    We use mlp_buf as temp since it's large enough (intermediate_size > kv_dim)
    let mut v_buf =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, kv_dim])
            .map_err(|e| anyhow::anyhow!("v_buf alloc: {e}"))?;
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        kv_dim,
        hidden_dim,
        &scratch.normed,
        &fw.v_proj_w,
        &mut v_buf,
    )?;

    // 7. Transpose Q [S, H, D] → [H, S, D], K and V same
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_q_heads,
        head_dim,
        &scratch.proj_buf,
        &mut scratch.attn_q,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} Q transpose: {e}"))?;

    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_kv_heads,
        head_dim,
        &scratch.proj_buf2,
        &mut scratch.attn_k,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} K transpose: {e}"))?;

    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_kv_heads,
        head_dim,
        &v_buf,
        &mut scratch.attn_v,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} V transpose: {e}"))?;

    // 8. Causal attention: Q [1, q_heads, seq, hd] × K,V [1, kv_heads, seq, hd]
    //    Output is F32 [1, q_heads, seq, hd]
    let scale = 1.0 / (head_dim as f32).sqrt();
    prefill_ffi::full_attention_prefill(
        ordinal,
        ScalarType::BF16,
        1, // batch_size
        num_q_heads,
        num_kv_heads,
        seq_len,  // q_len
        seq_len,  // kv_len
        head_dim,
        scale,
        0, // seqlen_offset (fresh prefill)
        &scratch.attn_q,
        &scratch.attn_k,
        &scratch.attn_v,
        &mut scratch.attn_out_f32,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} attention: {e}"))?;

    // 9. Cast attention output F32 → BF16 (reuse attn_q buffer)
    prefill_ffi::cast(
        ordinal,
        ScalarType::F32,
        ScalarType::BF16,
        num_q_heads * seq_len * head_dim,
        &scratch.attn_out_f32,
        &mut scratch.attn_q,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} attn cast: {e}"))?;

    // 10. Transpose back [H, S, D] → [S, H, D] = [S, q_dim]
    prefill_ffi::transpose_shd_hsd(
        ordinal,
        ScalarType::BF16,
        num_q_heads,
        seq_len,
        head_dim,
        &scratch.attn_q,
        &mut scratch.proj_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} attn transpose back: {e}"))?;

    // 11. O projection: proj_buf [seq, q_dim] × o_proj_w [hidden, q_dim]^T → [seq, hidden]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        hidden_dim,
        q_dim,
        &scratch.proj_buf,
        &fw.o_proj_w,
        &mut scratch.proj_buf2,
    )?;

    // 12. Residual: hidden += O projection output
    residual_add(ordinal, seq_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf2)
        .map_err(|e| anyhow::anyhow!("layer {idx} attention residual: {e}"))?;

    // 13. Fill KV cache for decode
    //     KV cache layout: [1, num_kv_heads, capacity, head_dim] BF16
    //     K data in attn_k: [num_kv_heads, seq_len, head_dim]
    //     V data in attn_v: [num_kv_heads, seq_len, head_dim]
    let ls = &mut state.layers[idx];
    ls.ensure_kv_capacity(seq_len - 1, ordinal, config, kv_chunk_size)
        .map_err(|e| anyhow::anyhow!("layer {idx} KV alloc: {e}"))?;

    // Copy attn_k → KV cache K (they already have matching [nkv, seq, hd] layout)
    if let Some(ref mut cache_k) = ls.kv_cache_k {
        // Cache is [1, nkv, cap, hd]; data starts at offset 0 with stride cap*hd per head
        // attn_k is [nkv, seq, hd] contiguous
        // For each head, copy seq*hd BF16 values
        let bytes_per_head = seq_len * head_dim * ScalarType::BF16.size_in_bytes();
        let cap = cache_k.shape()[2];
        let cap_stride = cap * head_dim * ScalarType::BF16.size_in_bytes();
        let src_stride = seq_len * head_dim * ScalarType::BF16.size_in_bytes();
        for h in 0..num_kv_heads {
            let dst_off = h * cap_stride;
            let src_off = h * src_stride;
            gpu_hal::copy_d2d(
                ordinal,
                cache_k.offset_ptr(dst_off) as *mut c_void,
                scratch.attn_k.offset_ptr(src_off),
                bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV cache K copy h={h}: {e}"))?;
        }
    }
    if let Some(ref mut cache_v) = ls.kv_cache_v {
        let bytes_per_head = seq_len * head_dim * ScalarType::BF16.size_in_bytes();
        let cap = cache_v.shape()[2];
        let cap_stride = cap * head_dim * ScalarType::BF16.size_in_bytes();
        let src_stride = seq_len * head_dim * ScalarType::BF16.size_in_bytes();
        for h in 0..num_kv_heads {
            let dst_off = h * cap_stride;
            let src_off = h * src_stride;
            gpu_hal::copy_d2d(
                ordinal,
                cache_v.offset_ptr(dst_off) as *mut c_void,
                scratch.attn_v.offset_ptr(src_off),
                bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} KV cache V copy h={h}: {e}"))?;
        }
    }
    ls.set_kv_filled(seq_len);

    Ok(())
}

fn prefill_linear_attention_layer(
    weights: &Qwen35Weights,
    state: &mut ModelState,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    seq_len: usize,
    ordinal: usize,
) -> Result<()> {
    let lw = weights.layers[idx]
        .linear
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {idx}: expected linear attention weights"))?;

    let hidden_dim = config.hidden_size;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let khd = config.linear_key_head_dim;
    let vhd = config.linear_value_head_dim;
    let kern = config.linear_conv_kernel_dim;
    let key_dim = nk * khd;     // Q and K share this dimension
    let val_dim = nv * vhd;
    let qkv_dim = key_dim * 2 + val_dim;
    let z_dim = val_dim;

    // 1. QKV projection: normed [seq, hidden] → [seq, qkv_dim]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        qkv_dim,
        hidden_dim,
        &scratch.normed,
        &lw.qkv_proj_w,
        &mut scratch.proj_buf,
    )?;

    // Save last kern-1 rows of QKV for conv state before conv modifies things
    if let Some(ref mut conv_state) = state.layers[idx].conv_state {
        prefill_ffi::extract_conv_state(
            ordinal,
            ScalarType::BF16,
            seq_len,
            qkv_dim,
            kern - 1,
            &scratch.proj_buf,
            conv_state,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} extract conv state: {e}"))?;
    }

    // 2. Z projection: normed [seq, hidden] → [seq, z_dim]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        z_dim,
        hidden_dim,
        &scratch.normed,
        &lw.z_proj_w,
        &mut scratch.proj_buf2,
    )?;

    // 3. B projection: normed [seq, hidden] → [seq, nv]
    let mut b_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, nv])
        .map_err(|e| anyhow::anyhow!("b_buf alloc: {e}"))?;
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        nv,
        hidden_dim,
        &scratch.normed,
        &lw.b_proj_w,
        &mut b_buf,
    )?;

    // 4. A projection: normed [seq, hidden] → [seq, nv]
    let mut a_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, nv])
        .map_err(|e| anyhow::anyhow!("a_buf alloc: {e}"))?;
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        nv,
        hidden_dim,
        &scratch.normed,
        &lw.a_proj_w,
        &mut a_buf,
    )?;

    // 5. Transpose QKV [S, qkv_dim] → [qkv_dim, pad+S] for conv input
    let pad = kern - 1;
    prefill_ffi::transpose_pad_conv(
        ordinal,
        ScalarType::BF16,
        seq_len,
        qkv_dim,
        pad,
        &scratch.proj_buf,
        &mut scratch.conv_input,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} conv transpose+pad: {e}"))?;

    // 6. Conv1d + SiLU: [qkv_dim, pad+S] → [S, qkv_dim]
    let total_len = seq_len + pad;
    prefill_ffi::linear_prefill_conv_pack(
        ordinal,
        ScalarType::BF16,
        1, // batch_size
        qkv_dim,
        total_len,
        seq_len,
        kern,
        &scratch.conv_input,
        &lw.conv1d_w,
        &mut scratch.proj_buf,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} conv: {e}"))?;

    // 7. Split conv output [S, qkv_dim] into Q [S, key_dim], K [S, key_dim], V [S, val_dim]
    //    Layout within qkv_dim: [Q(key_dim) | K(key_dim) | V(val_dim)]

    // 8. L2-normalize Q and K per head
    //    Q: treat [S, key_dim] as [S*nk, khd], normalize each row
    //    K: treat [S, key_dim] starting at offset key_dim as [S*nk, khd]
    //    The l2norm function normalizes each row independently.
    //
    //    However, the megakernel applies Q_norm = Q / ||Q|| * rsqrt(khd)
    //    and K_norm = K / ||K||. The l2norm kernel does x / ||x|| with eps.
    //    For Q, we need an extra * rsqrt(khd) scaling.
    //
    //    Strategy: normalize both Q and K via l2norm, then scale Q by rsqrt(khd).

    // We need to work with sub-regions of proj_buf.
    // Q is at offset 0, K at offset key_dim, V at offset key_dim*2 in each row.
    // Since l2norm operates on contiguous [n_rows, n_cols], and the Q/K/V data is
    // interleaved across rows, we need to extract them first.

    let mut q_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("q_linear alloc: {e}"))?;
    let k_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("k_linear alloc: {e}"))?;
    let v_linear = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * nv, vhd])
        .map_err(|e| anyhow::anyhow!("v_linear alloc: {e}"))?;

    // Extract Q, K, V from interleaved [S, qkv_dim] rows
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    for s in 0..seq_len {
        let row_base = s * qkv_dim * elem_bytes;
        let q_src = scratch.proj_buf.offset_ptr(row_base);
        let k_src = scratch.proj_buf.offset_ptr(row_base + key_dim * elem_bytes);
        let v_src = scratch.proj_buf.offset_ptr(row_base + key_dim * 2 * elem_bytes);

        let q_dst_off = s * key_dim * elem_bytes;
        let k_dst_off = s * key_dim * elem_bytes;
        let v_dst_off = s * val_dim * elem_bytes;

        gpu_hal::copy_d2d(ordinal, q_linear.offset_ptr(q_dst_off) as *mut c_void, q_src, key_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} Q extract s={s}: {e}"))?;
        gpu_hal::copy_d2d(ordinal, k_linear.offset_ptr(k_dst_off) as *mut c_void, k_src, key_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} K extract s={s}: {e}"))?;
        gpu_hal::copy_d2d(ordinal, v_linear.offset_ptr(v_dst_off) as *mut c_void, v_src, val_dim * elem_bytes)
            .map_err(|e| anyhow::anyhow!("layer {idx} V extract s={s}: {e}"))?;
    }

    // L2-normalize Q [S*nk, khd] and K [S*nk, khd]
    let mut q_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("q_normed alloc: {e}"))?;
    prefill_ffi::l2norm(ordinal, ScalarType::BF16, seq_len * nk, khd, 1e-6, &q_linear, &mut q_normed)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q l2norm: {e}"))?;

    // Scale Q by rsqrt(khd)
    let q_scale = 1.0 / (khd as f32).sqrt();
    prefill_ffi::mul_scalar(ordinal, ScalarType::BF16, seq_len * key_dim, q_scale, &q_normed, &mut q_linear)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q scale: {e}"))?;

    let mut k_normed = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len * nk, khd])
        .map_err(|e| anyhow::anyhow!("k_normed alloc: {e}"))?;
    prefill_ffi::l2norm(ordinal, ScalarType::BF16, seq_len * nk, khd, 1e-6, &k_linear, &mut k_normed)
        .map_err(|e| anyhow::anyhow!("layer {idx} K l2norm: {e}"))?;

    // 9. Compute beta and g on CPU (small: seq_len × nv values)
    //    beta[h, t] = sigmoid(B[t, h])
    //    g[h, t] = -softplus(A[t, h] + dt_bias[h]) * a_log_exp[h]

    let b_host = b_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("B D2H: {e}"))?;
    let a_host = a_buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("A D2H: {e}"))?;
    let dt_bias_host = lw
        .dt_bias
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("dt_bias D2H: {e}"))?;
    let a_log_exp_host = lw
        .a_log_exp
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("a_log_exp D2H: {e}"))?;

    // dt_bias and a_log_exp are BF16 [nv] (after bake-time transform)
    let dt_bias_f32: Vec<f32> = dt_bias_host
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let a_log_exp_f32: Vec<f32> = a_log_exp_host
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    // beta: [nv, seq_len] — transposed from B [seq_len, nv]
    let mut beta_bf16 = Vec::with_capacity(nv * seq_len * 2);
    // g: [nv, seq_len]
    let mut g_bf16 = Vec::with_capacity(nv * seq_len * 2);

    for h in 0..nv {
        for t in 0..seq_len {
            // B[t, h]
            let b_val = {
                let off = (t * nv + h) * 2;
                half::bf16::from_le_bytes([b_host[off], b_host[off + 1]]).to_f32()
            };
            let beta = 1.0 / (1.0 + (-b_val).exp());
            beta_bf16.extend_from_slice(&half::bf16::from_f32(beta).to_le_bytes());

            // A[t, h]
            let a_val = {
                let off = (t * nv + h) * 2;
                half::bf16::from_le_bytes([a_host[off], a_host[off + 1]]).to_f32()
            };
            // g = -softplus(A + dt_bias) * a_log_exp
            let sp = (1.0 + (a_val + dt_bias_f32[h]).exp()).ln();
            let g_val = -sp * a_log_exp_f32[h];
            g_bf16.extend_from_slice(&half::bf16::from_f32(g_val).to_le_bytes());
        }
    }

    let beta_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[nv, seq_len], &beta_bf16)
        .map_err(|e| anyhow::anyhow!("beta upload: {e}"))?;
    let g_gpu = GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[nv, seq_len], &g_bf16)
        .map_err(|e| anyhow::anyhow!("g upload: {e}"))?;

    // 10. Transpose Q [S, nk, khd] → [nk, S, khd] and K, V similarly
    //     q_linear is [S*nk, khd] = [S, nk, khd] contiguous
    let mut q_trans = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nk, seq_len, khd])
        .map_err(|e| anyhow::anyhow!("q_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, seq_len, nk, khd, &q_linear, &mut q_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} Q linear transpose: {e}"))?;

    let mut k_trans = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nk, seq_len, khd])
        .map_err(|e| anyhow::anyhow!("k_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, seq_len, nk, khd, &k_normed, &mut k_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} K linear transpose: {e}"))?;

    let mut v_trans = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, seq_len, vhd])
        .map_err(|e| anyhow::anyhow!("v_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, seq_len, nv, vhd, &v_linear, &mut v_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} V linear transpose: {e}"))?;

    // 11. Delta recurrent prefill
    //     Input: Q [nv, S, khd], K [nv, S, khd], V [nv, S, vhd], beta [nv, S], g [nv, S]
    //     initial_state: [nv, khd, vhd] F32 (zeros on first prefill)
    //     Output: [nv, (S + khd), vhd] BF16 — first S rows are attention, last khd rows are state
    let recurrent_state = state.layers[idx]
        .recurrent_state
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {idx}: missing recurrent state"))?;

    let out_rows = seq_len + khd;
    let mut delta_out =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, out_rows, vhd])
            .map_err(|e| anyhow::anyhow!("delta_out alloc: {e}"))?;

    prefill_ffi::delta_recurrent_prefill(
        ordinal,
        ScalarType::BF16,
        nv,       // batch_heads
        seq_len,
        khd,
        vhd,
        recurrent_state,
        &q_trans,
        &k_trans,
        &v_trans,
        &beta_gpu,
        &g_gpu,
        &mut delta_out,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} delta recurrent: {e}"))?;

    // 12. Extract recurrent state: last khd rows per head → [nv, khd, vhd] F32
    //     delta_out[h, seq_len..seq_len+khd, :] is BF16 — need to cast to F32
    {
        let state_elems = nv * khd * vhd;
        let state_bf16 =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, khd, vhd])
                .map_err(|e| anyhow::anyhow!("state_bf16 alloc: {e}"))?;

        // Copy state portion from delta_out
        let state_bytes_per_head = khd * vhd * elem_bytes;
        let out_stride = out_rows * vhd * elem_bytes;
        let attn_offset = seq_len * vhd * elem_bytes;
        for h in 0..nv {
            let src_off = h * out_stride + attn_offset;
            let dst_off = h * state_bytes_per_head;
            gpu_hal::copy_d2d(
                ordinal,
                state_bf16.offset_ptr(dst_off) as *mut c_void,
                delta_out.offset_ptr(src_off),
                state_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state extract h={h}: {e}"))?;
        }

        // Cast BF16 → F32 into the recurrent state buffer
        if let Some(ref mut rec_state) = state.layers[idx].recurrent_state {
            prefill_ffi::cast(ordinal, ScalarType::BF16, ScalarType::F32, state_elems, &state_bf16, rec_state)
                .map_err(|e| anyhow::anyhow!("layer {idx} recurrent state cast: {e}"))?;
        }
    }

    // 13. Extract attention output: [nv, seq_len, vhd] from delta_out
    //     Transpose [nv, S, vhd] → [S, nv, vhd] = [S, val_dim]
    let attn_output = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, seq_len, vhd])
        .map_err(|e| anyhow::anyhow!("attn_output alloc: {e}"))?;
    // Copy only the first seq_len rows per head
    {
        let attn_bytes_per_head = seq_len * vhd * elem_bytes;
        let out_stride = out_rows * vhd * elem_bytes;
        for h in 0..nv {
            let src_off = h * out_stride;
            let dst_off = h * attn_bytes_per_head;
            gpu_hal::copy_d2d(
                ordinal,
                attn_output.offset_ptr(dst_off) as *mut c_void,
                delta_out.offset_ptr(src_off),
                attn_bytes_per_head,
            )
            .map_err(|e| anyhow::anyhow!("layer {idx} attn output extract h={h}: {e}"))?;
        }
    }

    // 14. Gated RMSNorm: out = rms_norm(attn_output) * norm_w * silu(Z)
    //     attn_output is [nv, S, vhd]; Z (proj_buf2) is [S, val_dim] = [S, nv*vhd]
    //     Need Z in [nv, S, vhd] layout
    let mut z_trans = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv, seq_len, vhd])
        .map_err(|e| anyhow::anyhow!("z_trans alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, seq_len, nv, vhd, &scratch.proj_buf2, &mut z_trans)
        .map_err(|e| anyhow::anyhow!("layer {idx} Z transpose: {e}"))?;

    // The norm weight is F32 [vhd]. rms_norm_gated expects all-same-dtype.
    // Cast norm_w from F32 to BF16 (small, vhd elements).
    let mut norm_w_bf16 = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[vhd])
        .map_err(|e| anyhow::anyhow!("norm_w_bf16 alloc: {e}"))?;
    prefill_ffi::cast(ordinal, ScalarType::F32, ScalarType::BF16, vhd, &lw.norm_w, &mut norm_w_bf16)
        .map_err(|e| anyhow::anyhow!("layer {idx} norm_w cast: {e}"))?;

    // rms_norm_gated: [n_rows, n_cols] with n_rows = nv*seq_len, n_cols = vhd
    let mut gated_out = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[nv * seq_len, vhd])
        .map_err(|e| anyhow::anyhow!("gated_out alloc: {e}"))?;
    prefill_ffi::rms_norm_gated(
        ordinal,
        ScalarType::BF16,
        nv * seq_len,
        vhd,
        config.rms_norm_eps as f32,
        &attn_output,
        &z_trans,
        &norm_w_bf16,
        &mut gated_out,
    )
    .map_err(|e| anyhow::anyhow!("layer {idx} gated norm: {e}"))?;

    // 15. Transpose gated_out [nv, S, vhd] → [S, nv, vhd] = [S, val_dim]
    let mut gated_s_first = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, val_dim])
        .map_err(|e| anyhow::anyhow!("gated_s_first alloc: {e}"))?;
    prefill_ffi::transpose_shd_hsd(ordinal, ScalarType::BF16, nv, seq_len, vhd, &gated_out, &mut gated_s_first)
        .map_err(|e| anyhow::anyhow!("layer {idx} gated transpose: {e}"))?;

    // 16. O projection: [S, val_dim] × out_proj_w [hidden, val_dim]^T → [S, hidden]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        hidden_dim,
        val_dim,
        &gated_s_first,
        &lw.out_proj_w,
        &mut scratch.proj_buf2,
    )?;

    // 17. Residual: hidden += O projection output
    residual_add(ordinal, seq_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf2)
        .map_err(|e| anyhow::anyhow!("layer {idx} linear attn residual: {e}"))?;

    Ok(())
}

fn prefill_mlp_layer(
    weights: &Qwen35Weights,
    scratch: &mut PrefillScratch,
    config: &TextConfig,
    idx: usize,
    seq_len: usize,
    ordinal: usize,
) -> Result<()> {
    let lw = &weights.layers[idx];
    let hidden_dim = config.hidden_size;
    let intermediate = config.intermediate_size;

    // gate_proj: normed [seq, hidden] × gate_w [intermediate, hidden]^T → [seq, intermediate]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1, // batch=1
        seq_len,       // m = seq_len
        intermediate,  // n = intermediate_size
        hidden_dim,    // k = hidden_dim
        &scratch.normed,
        &lw.gate_proj_w,
        &mut scratch.proj_buf,
    )?;

    // up_proj: normed [seq, hidden] × up_w [intermediate, hidden]^T → [seq, intermediate]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        intermediate,
        hidden_dim,
        &scratch.normed,
        &lw.up_proj_w,
        &mut scratch.proj_buf2,
    )?;

    // SwiGLU: out = silu(gate) * up
    prefill_ffi::swiglu_mul(
        ordinal,
        ScalarType::BF16,
        seq_len * intermediate,
        &scratch.proj_buf,
        &scratch.proj_buf2,
        &mut scratch.mlp_buf,
    )?;

    // down_proj: mlp_buf [seq, intermediate] × down_w [hidden, intermediate]^T → [seq, hidden]
    prefill_ffi::batched_matmul(
        ordinal,
        ScalarType::BF16,
        1,
        seq_len,
        hidden_dim,
        intermediate,
        &scratch.mlp_buf,
        &lw.down_proj_w,
        &mut scratch.proj_buf,
    )?;

    // Residual: hidden += down_proj output
    residual_add(ordinal, seq_len * hidden_dim, &mut scratch.hidden, &scratch.proj_buf)
        .map_err(|e| anyhow::anyhow!("layer {idx} MLP residual: {e}"))?;

    Ok(())
}
