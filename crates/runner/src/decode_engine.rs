use std::ffi::c_void;

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use gpu_hal::{GpuBuffer, ScalarType};

use qwen35::config::TextConfig;
use qwen35::desc_builder::build_layer_descs;
use qwen35::rotary::RotaryTables;
use qwen35::scratch::PersistentDecodeScratch;
use qwen35::state::ModelState;
use qwen35::weights::Qwen35Weights;

use crate::oracle::OracleOutput;
use crate::prefill_engine;

pub struct DecodeEngine {
    weights: Qwen35Weights,
    state: ModelState,
    scratch: PersistentDecodeScratch,
    rotary: RotaryTables,
    hidden_io: GpuBuffer,
    normed_buf: GpuBuffer,
    logits_buf: GpuBuffer,
    matvec_counter: GpuBuffer,
    ordinal: usize,
    kv_chunk_size: usize,
    use_4b_kernel: bool,
    proj_buf_floats: usize,
    attn_scratch_floats: usize,
}

impl DecodeEngine {
    pub fn new(
        weights: Qwen35Weights,
        ordinal: usize,
        proj_buf_floats: usize,
        attn_scratch_floats: usize,
        kv_chunk_size: usize,
        use_4b_kernel: bool,
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
        let rotary =
            RotaryTables::build(config, ordinal).map_err(|e| anyhow::anyhow!("rotary: {e}"))?;
        let hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("hidden_io: {e}"))?;
        let normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.hidden_size])
            .map_err(|e| anyhow::anyhow!("normed_buf: {e}"))?;
        let logits_buf =
            GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, 1, config.vocab_size])
                .map_err(|e| anyhow::anyhow!("logits_buf: {e}"))?;
        // Atomic counter for work-stealing matvec (4 bytes)
        let matvec_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
            .map_err(|e| anyhow::anyhow!("matvec_counter: {e}"))?;

        Ok(Self {
            weights,
            state,
            scratch,
            rotary,
            hidden_io,
            normed_buf,
            logits_buf,
            matvec_counter,
            ordinal,
            kv_chunk_size,
            use_4b_kernel,
            proj_buf_floats,
            attn_scratch_floats,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.weights.config
    }

    /// Load prefill state from oracle output into GPU buffers.
    pub fn load_prefill_state(&mut self, oracle: &OracleOutput) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD;

        // Load hidden state
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
        let expected_bytes: usize = hidden_shape.iter().product::<usize>() * ScalarType::BF16.size_in_bytes();
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

        // Load KV caches for full-attention layers
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
            ls.kv_filled = kv.k_shape[2]; // seq dim
        }

        // Load conv states for linear-attention layers
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

        // Load recurrent states for linear-attention layers
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
        )?;

        // Reset sync counters for the decode kernel
        self.scratch
            .reset_sync()
            .map_err(|e| anyhow::anyhow!("reset sync after prefill: {e}"))?;

        Ok(result.logits)
    }

    /// Run one decode step. Returns logits as Vec<f32> on CPU.
    pub fn decode_step(&mut self, token_id: u32, seqlen_offset: usize) -> Result<Vec<f32>> {
        let config = &self.weights.config;

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
                ls.ensure_kv_capacity(seqlen_offset, self.ordinal, config, self.kv_chunk_size)
                    .map_err(|e| anyhow::anyhow!("ensure KV capacity layer {i}: {e}"))?;
            }
        }

        // 3. Build layer descriptors
        let descs = build_layer_descs(&self.weights, &self.state, seqlen_offset);

        // 4. Upload descriptors to device
        self.scratch
            .upload_descs(&descs)
            .map_err(|e| anyhow::anyhow!("upload descs: {e}"))?;

        // 5. Launch persistent decode kernel (dispatch by model variant)
        if self.use_4b_kernel {
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
            )
            .map_err(|e| anyhow::anyhow!("persistent_decode_4b kernel: {e}"))?;
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
            .map_err(|e| anyhow::anyhow!("persistent_decode kernel: {e}"))?;
        }

        // 6. Update KV filled counts
        let filled = seqlen_offset + 1;
        for (i, ls) in self.state.layers.iter_mut().enumerate() {
            if config.is_full_attention(i) {
                ls.set_kv_filled(filled);
            }
        }

        // 7. Final RMSNorm
        let rms_norm_fn = if self.use_4b_kernel { kernel_ffi::rms_norm_4b } else { kernel_ffi::rms_norm };
        rms_norm_fn(
            self.ordinal,
            ScalarType::BF16,
            &mut self.normed_buf,
            &self.hidden_io,
            &self.weights.norm_weight,
            config.rms_norm_eps as f32,
            config.hidden_size,
        )
        .map_err(|e| anyhow::anyhow!("final rms_norm: {e}"))?;

        // 8. lm_head projection → logits (work-stealing matvec)
        let matvec_fn = if self.use_4b_kernel { kernel_ffi::standalone_matvec_4b } else { kernel_ffi::standalone_matvec };
        matvec_fn(
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

        // 9. Copy logits to CPU and convert BF16 → F32
        let logits_bytes = self
            .logits_buf
            .to_host_bytes()
            .map_err(|e| anyhow::anyhow!("logits D2H: {e}"))?;
        let logits_f32: Vec<f32> = logits_bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();

        Ok(logits_f32)
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
}
