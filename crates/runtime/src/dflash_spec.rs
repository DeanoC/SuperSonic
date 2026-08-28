//! DFlash2 draft-model speculative decode loop.
//!
//! Orchestrates the DFlash2 draft forward (via [`DraftEngine`]) against the
//! Qwen3.8 target (via [`DecodeEngine`]). The draft shares the target's
//! `embed_tokens` (for noise construction) and `lm_head` (for projection).
//!
//! Flow per round:
//!   1. Build noise embedding: row 0 = embed(last_tok), rows 1.. = embed(mask).
//!   2. Draft forward → post-norm hidden states [block_size, hidden].
//!   3. Project through target lm_head → selector chain → draft tokens.
//!   4. Target verify (prefill-append with capture) → greedy predictions.
//!   5. Accept longest matching prefix + bonus token.
//!   6. Capture target hidden states at the 5 draft layers during verify.

use std::time::Instant;

use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use crate::decode_engine::DecodeEngine;
use crate::draft_engine::DraftEngine;
use crate::prefill_engine::{self, DflashTargetCapture};

/// Result of one DFlash2 speculative round.
pub struct DflashSpecRound {
    /// All tokens emitted this round (accepted drafts + bonus).
    pub emitted: Vec<u32>,
    /// The target's greedy token for the next round (bonus or last verified).
    pub next_token: u32,
    /// Number of draft tokens proposed (excluding the seed last_tok).
    pub n_drafted: usize,
    /// Number of draft tokens accepted.
    pub n_accepted: usize,
}

/// Telemetry summary.
pub struct DflashSpecSummary {
    pub n_rounds: u32,
    pub n_accepted: u32,
    pub n_drafted: u32,
}

/// The DFlash2 speculative decoder.
pub struct DflashSpecDecoder {
    ordinal: usize,
    draft: DraftEngine,
    capture: DflashTargetCapture,
    mask_token_id: u32,
    block_size: usize,
    /// [block_size, hidden] BF16 — noise embedding scratch.
    noise_embed: GpuBuffer,
    /// Telemetry.
    n_rounds: u32,
    n_accepted: u32,
    n_drafted: u32,
}

impl DflashSpecDecoder {
    /// Build a DFlash2 decoder from uploaded draft weights.
    ///
    /// * `max_ctx` — upper bound on context length (sizes the capture buffer).
    /// * `hidden` — target hidden dim (5120 for Qwen3.8-27B).
    pub fn new(
        draft_weights: model_store::dflash::DraftGpuWeights,
        ordinal: usize,
        max_ctx: usize,
        hidden: usize,
    ) -> Result<Self> {
        let cfg = &draft_weights.config;
        let ntl = cfg.n_target_layers;
        let block_size = cfg.block_size;
        let mask_token_id = cfg.mask_token_id;
        let target_layer_ids = cfg.target_layer_ids.clone();

        // RoPE table sized to max_ctx + block_size (the max position the
        // draft will ever see).
        let max_pos = max_ctx + block_size;
        let draft = DraftEngine::new(draft_weights, ordinal, max_pos)
            .context("build dflash draft engine")?;

        let capture = DflashTargetCapture::new(ordinal, max_ctx, ntl, hidden, target_layer_ids)
            .context("build dflash capture buffer")?;

        let noise_embed = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[block_size, hidden])
            .context("dflash noise_embed alloc")?;

        Ok(Self {
            ordinal,
            draft,
            capture,
            mask_token_id,
            block_size,
            noise_embed,
            n_rounds: 0,
            n_accepted: 0,
            n_drafted: 0,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Raw mutable pointer to the internal capture buffer. Used by the
    /// runner to split the borrow: prefill borrows the capture, while
    /// run_round borrows the whole decoder. The caller must ensure no
    /// aliasing (single-threaded CLI usage).
    pub fn capture_ptr(&mut self) -> *mut DflashTargetCapture {
        &mut self.capture as *mut DflashTargetCapture
    }

    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    pub fn summary(&self) -> DflashSpecSummary {
        DflashSpecSummary {
            n_rounds: self.n_rounds,
            n_accepted: self.n_accepted,
            n_drafted: self.n_drafted,
        }
    }

    /// Build the noise embedding: row 0 = embed(last_tok), rows 1.. =
    /// embed(mask_token_id). The target's embed_tokens is [vocab, hidden] BF16.
    fn build_noise_embed(
        &mut self,
        last_tok: u32,
        embed_tokens: &GpuBuffer,
        hidden: usize,
    ) -> Result<()> {
        let elem = ScalarType::BF16.size_in_bytes();
        let row_bytes = hidden * elem;
        let ordinal = self.ordinal;
        // Row 0: last_tok embedding.
        gpu_hal::copy_d2d(
            ordinal,
            self.noise_embed.as_mut_ptr() as *mut std::ffi::c_void,
            embed_tokens.offset_ptr(last_tok as usize * row_bytes),
            row_bytes,
        )
        .context("dflash noise_embed row0 copy")?;
        // Rows 1..block_size: mask token embedding.
        if self.block_size > 1 {
            let mask_off = self.mask_token_id as usize * row_bytes;
            for i in 1..self.block_size {
                gpu_hal::copy_d2d(
                    ordinal,
                    self.noise_embed.offset_ptr(i * row_bytes) as *mut std::ffi::c_void,
                    embed_tokens.offset_ptr(mask_off),
                    row_bytes,
                )
                .with_context(|| format!("dflash noise_embed row{i} copy"))?;
            }
        }
        Ok(())
    }

    /// Run one DFlash2 speculative round.
    ///
    /// * `engine` — the target decode engine (for verify + capture).
    /// * `last_tok` — the last committed token (seed for the draft).
    /// * `committed` — number of committed positions (context length).
    /// * `remaining` — remaining generation budget.
    pub fn run_round(
        &mut self,
        engine: &mut DecodeEngine,
        last_tok: u32,
        committed: usize,
        remaining: usize,
    ) -> Result<DflashSpecRound> {
        anyhow::ensure!(remaining > 0, "dflash round with remaining=0");
        let hidden = engine.weights().config.hidden_size;
        let block_size = self.block_size;
        let ordinal = self.ordinal;

        // ── 1. Build noise embedding.
        self.build_noise_embed(last_tok, engine.weights().embed_tokens(), hidden)?;

        // ── 2. Draft forward.
        let ctx_len = committed;
        let q_len = block_size;
        let total_k = ctx_len + q_len;
        let positions_q: Vec<usize> = (ctx_len..ctx_len + q_len).collect();
        let positions_k: Vec<usize> = (0..total_k).collect();

        let t_draft = Instant::now();
        let draft_hidden = self
            .draft
            .forward(
                &self.capture.target_hidden,
                &self.noise_embed,
                &positions_q,
                &positions_k,
            )
            .context("dflash draft forward")?;
        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

        // ── 3. Project through target lm_head → logits → argmax → draft tokens.
        let t_proj = Instant::now();
        let logits = prefill_engine::project_normed_through_lm_head(
            &draft_hidden,
            engine.weights(),
            &engine.weights().config,
            q_len,
            ordinal,
        )
        .context("dflash lm_head projection")?;
        let proj_ms = t_proj.elapsed().as_secs_f64() * 1000.0;

        // Draft tokens via the DFlash2 selector chain (bigram-corrected
        // top-K path), falling back to pure argmax when the drafter GGUF
        // ships no selector. The selector traces a greedy path through the
        // lm_head top-K candidates, scoring each with a learned bigram
        // transition (pred_cb/succ_cb codebooks) — this is the path the
        // upstream uses and is essential for high acceptance.
        let draft_tokens = match self
            .draft
            .select_chain(&draft_hidden, &logits, last_tok, q_len)
            .context("dflash selector chain")?
        {
            Some(tokens) => tokens,
            None => {
                let mut tokens = Vec::with_capacity(q_len);
                tokens.push(last_tok);
                for row in 1..q_len {
                    tokens.push(argmax_f32(&logits[row]));
                }
                tokens
            }
        };
        // draft_tokens = [last_tok, tok_1, ..., tok_{q_len-1}]
        // = q_len elements. The verify block is these q_len tokens.

        if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
            eprintln!(
                "[dflash-profile] ctx={ctx_len} draft={q_len} draft_ms={draft_ms:.1} proj_ms={proj_ms:.1} tokens={:?}",
                &draft_tokens[1..]
            );
        }

        // ── 4. Verify: hybrid path.
        // (a) Fused megakernel verify for greedy predictions (fast, single
        //     kernel launch over all 64 layers for all block tokens).
        // (b) After acceptance, capture-only pass for the committed prefix
        //     (prefill-append through 64 layers, capturing at 5 target layers).
        //
        // The fused verify is much faster than the prefill-append path (one
        // megakernel launch vs ~640 individual kernel launches). The capture
        // pass only processes the committed prefix (typically 2-3 tokens),
        // not the full block.
        let max_block = (remaining + 1).min(q_len);
        let block: Vec<u32> = draft_tokens[..max_block].to_vec();

        // Snapshot the linear-attention state before the verify mutates it.
        engine.snapshot_linear_for_spec()?;

        let t_verify = Instant::now();
        // The fused 4B megakernel has a shared-memory budget of ~15872 floats;
        // with hidden=5120, B<=3 fits. For larger blocks, use the prefill-append
        // path (which uses global memory, no LDS constraint).
        let fused_cap = 15872 / engine.weights().config.hidden_size.max(1);
        let use_fused = engine.use_4b_kernel() && block.len() <= fused_cap;
        let greedy = if use_fused {
            engine
                .verify_block_fused_greedy(&block, committed)
                .context("dflash fused verify")?
        } else {
            let result = engine
                .verify_block_dflash(&block, committed, &mut self.capture)
                .context("dflash verify")?;
            result.target_next.unwrap_or_default()
        };
        let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

        // ── 5. Acceptance: longest prefix where draft matches target.
        // greedy[i] = target's prediction for position committed+i+1.
        // drafts[i] = draft's prediction for position committed+i+1.
        let mut n_acc = 0usize;
        let drafts = &block[1..]; // proposed tokens [p1, ..., p_{q_len-1}]
        while n_acc < drafts.len() && n_acc < greedy.len() && drafts[n_acc] == greedy[n_acc] {
            n_acc += 1;
        }
        // commit_len = n_acc + 1 (last_tok + n_acc accepted drafts), capped.
        let commit_len = (n_acc + 1).min(block.len()).min(remaining + 1);
        let bonus_tok = greedy[commit_len.saturating_sub(1)];

        if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
            eprintln!(
                "[dflash-profile] verify_ms={verify_ms:.1} n_acc={n_acc} commit={commit_len} bonus={bonus_tok}"
            );
        }

        // ── 6. Commit / replay + capture.
        let (emitted, next_token) = if commit_len == block.len() {
            // Full accept: commit KV for the entire block.
            engine.commit_fused_kv_filled_public(committed + commit_len);
            let mut emitted = block[1..commit_len].to_vec();
            emitted.push(bonus_tok);
            (emitted, bonus_tok)
        } else {
            // Partial accept: restore linear state, rewind KV, replay the
            // committed prefix (last_tok + accepted drafts) through the fast
            // greedy decode megakernel.
            let replay_block = &block[..commit_len];
            let replay_bonus = engine.replay_committed_prefix(replay_block, committed)?;
            let mut emitted = block[1..commit_len].to_vec();
            emitted.push(replay_bonus);
            (emitted, replay_bonus)
        };

        // ── 7. Capture hidden states for the committed prefix.
        // On the fused path, the prefill-append capture wasn't run during
        // verify, so we run a capture-only pass for the committed positions.
        // On the non-fused path, capture already happened during verify.
        if use_fused && commit_len > 0 {
            let t_cap = Instant::now();
            let capture_block = &block[..commit_len];
            engine
                .capture_block_dflash(capture_block, committed, &mut self.capture)
                .context("dflash capture")?;
            let cap_ms = t_cap.elapsed().as_secs_f64() * 1000.0;
            if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
                eprintln!("[dflash-profile] capture_ms={cap_ms:.1} n_cap={commit_len}");
            }
        }
        self.capture.committed = committed + commit_len;

        let n_drafted = drafts.len().min(block.len() - 1);
        self.n_rounds += 1;
        self.n_accepted += n_acc as u32;
        self.n_drafted += n_drafted as u32;

        Ok(DflashSpecRound {
            emitted,
            next_token,
            n_drafted,
            n_accepted: n_acc,
        })
    }
}

fn argmax_f32(slice: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}
