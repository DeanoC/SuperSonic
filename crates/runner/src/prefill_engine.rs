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
}

impl PrefillScratch {
    fn new(config: &TextConfig, seq_len: usize, ordinal: usize) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let intermediate = config.intermediate_size;
        // Max projection dim across all layer types
        let max_proj = std::cmp::max(
            // Full attention: q_out + k_out + v_out
            config.num_attention_heads * config.head_dim
                + config.num_key_value_heads * config.head_dim * 2,
            // Linear attention: qkv_out (largest single projection)
            config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim,
        );

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
        // Input RMSNorm
        kernel_ffi::rms_norm(
            ordinal,
            ScalarType::BF16,
            &mut scratch.normed,
            &scratch.hidden,
            &weights.layers[idx].input_norm_w,
            config.rms_norm_eps as f32,
            hidden_dim,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} input norm: {e}"))?;

        // TODO: implement full attention and linear attention prefill per layer
        // For now this is a skeleton — the actual layer computation will be
        // added incrementally with validation at each step.

        if config.is_full_attention(idx) {
            // Full attention prefill: Q/K/V proj → causal attention → O proj
            // This path fills KV caches for decode
            prefill_full_attention_layer(
                weights, state, rotary, &mut scratch, config, idx, seq_len, ordinal, kv_chunk_size,
            )?;
        } else {
            // Linear attention prefill: QKV/Z/B/A proj → conv → recurrence → norm+gate → out_proj
            // This path fills conv_state and recurrent_state for decode
            prefill_linear_attention_layer(
                weights, state, &mut scratch, config, idx, seq_len, ordinal,
            )?;
        }

        // Post-attention RMSNorm
        kernel_ffi::rms_norm(
            ordinal,
            ScalarType::BF16,
            &mut scratch.normed,
            &scratch.hidden,
            &weights.layers[idx].post_attn_norm_w,
            config.rms_norm_eps as f32,
            hidden_dim,
        )
        .map_err(|e| anyhow::anyhow!("layer {idx} post-attn norm: {e}"))?;

        // MLP: gate_proj + up_proj → SwiGLU → down_proj
        prefill_mlp_layer(weights, &mut scratch, config, idx, seq_len, ordinal)?;
    }

    // 3. Final RMSNorm on last token only
    // Extract last token hidden: hidden[seq_len-1, :] → [1, hidden_dim]
    let last_token_offset = (seq_len - 1) * hidden_dim * ScalarType::BF16.size_in_bytes();
    let last_hidden_ptr = scratch.hidden.offset_ptr(last_token_offset);
    let mut last_hidden = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_dim])
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
    kernel_ffi::rms_norm(
        ordinal,
        ScalarType::BF16,
        &mut normed_last,
        &last_hidden,
        &weights.norm_weight,
        config.rms_norm_eps as f32,
        hidden_dim,
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
    _weights: &Qwen35Weights,
    _state: &mut ModelState,
    _rotary: &RotaryTables,
    _scratch: &mut PrefillScratch,
    _config: &TextConfig,
    _idx: usize,
    _seq_len: usize,
    _ordinal: usize,
    _kv_chunk_size: usize,
) -> Result<()> {
    // TODO: implement full attention prefill
    // Q/K/V projection → RoPE → causal attention → O projection → residual
    // Fill KV cache for this layer
    Ok(())
}

fn prefill_linear_attention_layer(
    _weights: &Qwen35Weights,
    _state: &mut ModelState,
    _scratch: &mut PrefillScratch,
    _config: &TextConfig,
    _idx: usize,
    _seq_len: usize,
    _ordinal: usize,
) -> Result<()> {
    // TODO: implement linear attention prefill
    // QKV/Z/B/A projection → conv1d → recurrence → gated norm → out_proj → residual
    // Fill conv_state and recurrent_state for this layer
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
    // Result added to hidden (residual connection)
    // TODO: need an add kernel or fused down_proj+residual
    // For now, write to proj_buf and add manually
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
    // Use the binary_broadcast kernel or a simple add
    // For now, use mul_scalar(1.0) as identity, then we need an add kernel
    // TODO: need a proper add kernel. For now, this is a placeholder.
    // The bridge has dotcache_qwen35_hip_binary_broadcast which could do add.

    Ok(())
}
