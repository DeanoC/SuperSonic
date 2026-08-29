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
use crate::prefill_engine::{self, DflashRollbackCapture, DflashTargetCapture};

/// Result of one DFlash2 speculative round.
pub struct DflashSpecRound {
    /// All tokens emitted this round (accepted drafts + bonus).
    pub emitted: Vec<u32>,
    /// The complete block passed through target verification.
    pub verified_block: Vec<u32>,
    /// The target's greedy token for the next round (bonus or last verified).
    pub next_token: u32,
    /// Number of draft tokens proposed (excluding the seed last_tok).
    pub n_drafted: usize,
    /// Number of draft tokens accepted.
    pub n_accepted: usize,
    pub verify_path: DflashVerifyPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashVerifyPath {
    Component,
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
    rollback: DflashRollbackCapture,
    mask_token_id: u32,
    block_size: usize,
    /// [block_size, hidden] BF16 — noise embedding scratch.
    noise_embed: GpuBuffer,
    /// Telemetry.
    n_rounds: u32,
    n_accepted: u32,
    n_drafted: u32,
}

const ACTIVE_BLOCK_SIZE: usize = 16;

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
        target_layer_count: usize,
        target_config: &qwen38::config::TextConfig,
    ) -> Result<Self> {
        let cfg = &draft_weights.config;
        let ntl = cfg.n_target_layers;
        let block_size = ACTIVE_BLOCK_SIZE;
        let mask_token_id = cfg.mask_token_id;
        let target_layer_ids =
            target_capture_layer_ids(target_layer_count, ntl, &cfg.target_layer_ids)?;

        // RoPE table sized to max_ctx + block_size (the max position the
        // draft will ever see).
        let max_pos = max_ctx + block_size;
        let draft = DraftEngine::new(draft_weights, ordinal, max_pos)
            .context("build dflash draft engine")?;

        let capture = DflashTargetCapture::new(ordinal, max_ctx, ntl, hidden, target_layer_ids)
            .context("build dflash capture buffer")?;
        let rollback = DflashRollbackCapture::new(target_config, block_size, ordinal)
            .context("build dflash rollback capture")?;

        let noise_embed = GpuBuffer::alloc(ordinal, ScalarType::BF16, &[block_size, hidden])
            .context("dflash noise_embed alloc")?;

        Ok(Self {
            ordinal,
            draft,
            capture,
            rollback,
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

        let trace_ctx = std::env::var("SUPERSONIC_DFLASH_TRACE_CTX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        if trace_ctx == Some(ctx_len) {
            std::fs::write(
                "/tmp/supersonic-dflash-trace-target.bf16",
                self.capture.target_hidden.to_host_bytes()?,
            )
            .context("dflash trace target write")?;
            std::fs::write(
                "/tmp/supersonic-dflash-trace-noise.bf16",
                self.noise_embed.to_host_bytes()?,
            )
            .context("dflash trace noise write")?;
        }
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
        if trace_ctx == Some(ctx_len) {
            std::fs::write(
                "/tmp/supersonic-dflash-trace-hidden.f32",
                draft_hidden.to_host_bytes()?,
            )
            .context("dflash trace hidden write")?;
        }

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
        let t_select = Instant::now();
        let selected = self
            .draft
            .select_chain(&draft_hidden, &logits, last_tok, q_len)
            .context("dflash selector chain")?;
        let select_ms = t_select.elapsed().as_secs_f64() * 1000.0;
        let draft_tokens = dflash_tokens_from_selector(selected, &logits, last_tok, q_len)?;
        // draft_tokens = [last_tok, tok_1, ..., tok_{q_len-1}]
        // = q_len elements. The verify block is these q_len tokens.

        if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
            let argmax_tokens: Vec<u32> = (1..q_len).map(|row| argmax_f32(&logits[row])).collect();
            eprintln!(
                "[dflash-profile] ctx={ctx_len} draft={q_len} draft_ms={draft_ms:.1} proj_ms={proj_ms:.1} select_ms={select_ms:.1} selector={:?} argmax={:?}",
                &draft_tokens[1..],
                argmax_tokens
            );
        }

        // ── 4. Verify through the batched component path. The integrated
        // capture records target features and per-token rollback intermediates
        // in the same causal B-token forward.
        let max_block = remaining.min(q_len);
        let block: Vec<u32> = draft_tokens[..max_block].to_vec();

        // Snapshot the linear-attention state before the verify mutates it.
        engine.snapshot_linear_for_spec()?;

        let t_verify = Instant::now();
        let verify = engine
            .verify_block_dflash_with_rollback(
                &block,
                committed,
                &mut self.capture,
                &mut self.rollback,
            )
            .context("dflash verify")?;
        let greedy = verify
            .target_next
            .ok_or_else(|| anyhow::anyhow!("dflash component verify missing greedy predictions"))?;
        let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(
            greedy.len() >= block.len(),
            "dflash component verify returned {} greedy predictions for {} tokens",
            greedy.len(),
            block.len()
        );

        // ── 5. Acceptance: longest prefix where draft matches target.
        // greedy[i] = target's prediction for position committed+i+1.
        // drafts[i] = draft's prediction for position committed+i+1.
        let mut n_acc = 0usize;
        let drafts = &block[1..];
        while n_acc < drafts.len() && n_acc < greedy.len() && drafts[n_acc] == greedy[n_acc] {
            n_acc += 1;
        }

        let accepted_len = n_acc + 1;
        let fast_plan = if n_acc < drafts.len() && accepted_len <= remaining {
            Some(dflash_fast_rollback_plan(
                &block, n_acc, &greedy, remaining,
            )?)
        } else {
            None
        };
        let use_fast_rollback = fast_plan.is_some();

        // Fast rollback defers the correction to the next round's seed,
        // matching the upstream chain policy. Full rounds already have the
        // correct final state and need no rollback.
        let mut plan = if let Some(plan) = fast_plan {
            plan
        } else {
            dflash_commit_plan(&block, n_acc, &greedy, remaining)?
        };
        let commit_len = plan.commit_tokens.len();

        if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
            eprintln!(
                "[dflash-profile] verify_ms={verify_ms:.1} root={last_tok} n_acc={n_acc} commit={commit_len} greedy={greedy:?} commit_tokens={:?} next={}",
                plan.commit_tokens, plan.next_token
            );
        }

        // ── 6. Commit / replay + capture. Partial acceptance may need a
        // replay to leave the target state at the committed prefix.
        let needs_replay = !use_fast_rollback && commit_len < block.len();
        let mut replay_ms = 0.0_f64;
        if use_fast_rollback {
            let t_rollback = Instant::now();
            engine.rollback_dflash_prefix(&self.rollback, commit_len)?;
            replay_ms = t_rollback.elapsed().as_secs_f64() * 1000.0;
        } else if needs_replay {
            let t_replay = Instant::now();
            let replay = engine.replay_committed_prefix_dflash(
                &block,
                commit_len,
                committed,
                &mut self.capture,
                &mut self.rollback,
            )?;
            let replay_next = replay
                .target_next
                .as_ref()
                .and_then(|next_tokens| next_tokens.get(commit_len - 1).copied())
                .ok_or_else(|| anyhow::anyhow!("dflash replay missing greedy next token"))?;
            plan.next_token = dflash_next_token(plan.next_token, Some(replay_next));
            replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;
        }
        engine.commit_fused_kv_filled_public(committed + commit_len);

        if std::env::var_os("SUPERSONIC_DFLASH_PROFILE").is_some() {
            eprintln!("[dflash-profile] replay_ms={replay_ms:.1} n_commit={commit_len}");
        }
        self.capture.committed = committed + commit_len;

        let n_drafted = drafts.len().min(block.len() - 1);
        self.n_rounds += 1;
        self.n_accepted += n_acc as u32;
        self.n_drafted += n_drafted as u32;

        Ok(DflashSpecRound {
            emitted: plan.commit_tokens,
            verified_block: block,
            next_token: plan.next_token,
            n_drafted,
            n_accepted: n_acc,
            verify_path: DflashVerifyPath::Component,
        })
    }
}

fn target_capture_layer_ids(
    target_layer_count: usize,
    capture_layer_count: usize,
    artifact_layer_ids: &[usize],
) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(
        target_layer_count >= 2,
        "dflash target capture needs at least 2 target layers, got {target_layer_count}"
    );
    anyhow::ensure!(
        capture_layer_count >= 2,
        "dflash target capture needs at least 2 capture layers, got {capture_layer_count}"
    );
    anyhow::ensure!(
        artifact_layer_ids.len() == capture_layer_count,
        "dflash target capture IDs have {} entries, expected {capture_layer_count}",
        artifact_layer_ids.len()
    );
    for &layer_id in artifact_layer_ids {
        anyhow::ensure!(
            layer_id < target_layer_count,
            "dflash target capture layer {layer_id} must be within the target layer count {target_layer_count}"
        );
    }
    Ok(artifact_layer_ids.to_vec())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DflashCommitPlan {
    commit_tokens: Vec<u32>,
    next_token: u32,
}

fn dflash_commit_plan(
    block: &[u32],
    accepted: usize,
    greedy: &[u32],
    remaining: usize,
) -> anyhow::Result<DflashCommitPlan> {
    anyhow::ensure!(!block.is_empty(), "dflash commit plan: empty block");
    anyhow::ensure!(
        greedy.len() >= block.len(),
        "dflash commit plan: greedy predictions {} are shorter than block {}",
        greedy.len(),
        block.len()
    );
    let accepted = accepted.min(block.len() - 1);

    let mut commit_tokens = if accepted == block.len() - 1 {
        block.to_vec()
    } else {
        let mut tokens = block[..accepted + 1].to_vec();
        tokens.push(greedy[accepted]);
        tokens
    };
    if commit_tokens.len() > remaining {
        commit_tokens.truncate(remaining);
    }
    let next_token = greedy[commit_tokens.len() - 1];

    Ok(DflashCommitPlan {
        commit_tokens,
        next_token,
    })
}

/// Build an upstream-style partial commit that defers the correction token.
///
/// The correction becomes the next round's seed instead of being processed in
/// this round. This keeps the draft feature context at the accepted boundary and
/// allows the target's per-token state capture to replace replay.
fn dflash_fast_rollback_plan(
    block: &[u32],
    accepted: usize,
    greedy: &[u32],
    remaining: usize,
) -> anyhow::Result<DflashCommitPlan> {
    anyhow::ensure!(!block.is_empty(), "dflash fast rollback: empty block");
    anyhow::ensure!(
        greedy.len() >= block.len(),
        "dflash fast rollback: greedy predictions {} are shorter than block {}",
        greedy.len(),
        block.len()
    );
    let accepted = accepted.min(block.len() - 1);
    let commit_len = accepted + 1;
    anyhow::ensure!(
        commit_len <= remaining,
        "dflash fast rollback commit length {commit_len} exceeds remaining {remaining}"
    );

    Ok(DflashCommitPlan {
        commit_tokens: block[..commit_len].to_vec(),
        next_token: greedy[commit_len - 1],
    })
}

fn dflash_tokens_from_selector(
    selected: Option<Vec<u32>>,
    logits: &[Vec<f32>],
    last_tok: u32,
    q_len: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(q_len > 0, "dflash selector needs a non-empty block");
    anyhow::ensure!(
        logits.len() >= q_len,
        "dflash selector logits {} are shorter than block {}",
        logits.len(),
        q_len
    );
    if let Some(tokens) = selected {
        anyhow::ensure!(
            tokens.len() == q_len,
            "dflash selector returned {} tokens for block {q_len}",
            tokens.len()
        );
        let mut tokens = tokens;
        tokens[0] = last_tok;
        return Ok(tokens);
    }

    let mut tokens = Vec::with_capacity(q_len);
    tokens.push(last_tok);
    for row in 1..q_len {
        tokens.push(argmax_f32(&logits[row]));
    }
    Ok(tokens)
}

fn dflash_next_token(verify_next: u32, replay_next: Option<u32>) -> u32 {
    replay_next.unwrap_or(verify_next)
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

#[cfg(test)]
mod dflash_commit_tests {
    use super::{
        dflash_commit_plan, dflash_fast_rollback_plan, dflash_next_token,
        dflash_tokens_from_selector, target_capture_layer_ids,
    };

    #[test]
    fn partial_acceptance_commits_root_accepted_drafts_and_correction() {
        let block = vec![13, 271, 22916, 6970, 279, 37550, 33075, 888];
        let greedy = vec![198, 760, 3841, 13477, 37550, 33075, 888, 279];

        let plan = dflash_commit_plan(&block, 3, &greedy, 100).unwrap();
        assert_eq!(plan.commit_tokens, vec![13, 271, 22916, 6970, 13477]);
        assert_eq!(plan.next_token, 37550);
    }

    #[test]
    fn zero_acceptance_commits_root_and_target_correction() {
        let block = vec![13, 271, 22916, 6970, 279, 37550, 33075, 888];
        let greedy = vec![198, 760, 3841, 13477, 37550, 33075, 888, 279];

        let plan = dflash_commit_plan(&block, 0, &greedy, 100).unwrap();
        assert_eq!(plan.commit_tokens, vec![13, 198]);
        assert_eq!(plan.next_token, 760);
    }

    #[test]
    fn replay_replaces_verify_next_after_partial_correction() {
        // The verify row is conditioned on the rejected draft token; the replay
        // row is conditioned on the committed correction and is authoritative.
        assert_eq!(dflash_next_token(11316, Some(760)), 760);
        assert_eq!(dflash_next_token(760, None), 760);
    }

    #[test]
    fn selector_chain_takes_precedence_over_argmax() {
        let selected = Some(vec![13, 271, 22916, 6970, 279, 37550, 33075, 888]);
        let logits = vec![vec![0.0, 1.0]; 8];

        let tokens = dflash_tokens_from_selector(selected, &logits, 13, 8).unwrap();

        assert_eq!(tokens, vec![13, 271, 22916, 6970, 279, 37550, 33075, 888]);
    }

    #[test]
    fn selector_chain_overwrites_root_with_committed_token() {
        let selected = Some(vec![999, 271, 22916, 6970, 279, 37550, 33075, 888]);
        let logits = vec![vec![0.0, 1.0]; 8];

        let tokens = dflash_tokens_from_selector(selected, &logits, 13, 8).unwrap();

        assert_eq!(tokens, vec![13, 271, 22916, 6970, 279, 37550, 33075, 888]);
    }

    #[test]
    fn argmax_fallback_starts_from_committed_root() {
        let selected = None;
        let logits = vec![vec![0.0, 1.0], vec![3.0, 2.0], vec![1.0, 4.0]];

        let tokens = dflash_tokens_from_selector(selected, &logits, 13, 3).unwrap();

        assert_eq!(tokens, vec![13, 0, 1]);
    }

    #[test]
    fn target_capture_layers_match_geo_lucebox_loader() {
        assert_eq!(
            target_capture_layer_ids(64, 5, &[5, 19, 33, 47, 61]).unwrap(),
            vec![5, 19, 33, 47, 61]
        );
        assert!(target_capture_layer_ids(64, 5, &[5, 19, 33, 47, 64])
            .unwrap_err()
            .to_string()
            .contains("within the target layer count"));
    }

    #[test]
    fn fast_rollback_defers_partial_correction_to_next_seed() {
        let block = vec![13, 271, 22916, 6970, 279, 37550, 33075, 888];
        let greedy = vec![198, 760, 3841, 13477, 37550, 33075, 888, 279];

        let plan = dflash_fast_rollback_plan(&block, 3, &greedy, 100).unwrap();
        assert_eq!(plan.commit_tokens, vec![13, 271, 22916, 6970]);
        assert_eq!(plan.next_token, 13477);
    }

    #[test]
    fn fast_rollback_supports_short_prefix() {
        let block = vec![13, 271, 22916, 6970, 279, 37550, 33075, 888];
        let greedy = vec![198, 760, 3841, 13477, 37550, 33075, 888, 279];

        let plan = dflash_fast_rollback_plan(&block, 1, &greedy, 100).unwrap();
        assert_eq!(plan.commit_tokens, vec![13, 271]);
        assert_eq!(plan.next_token, 760);
    }

    #[test]
    fn full_acceptance_commits_block_without_extra_correction() {
        let block = vec![13, 198, 760, 3841, 13477, 37550, 33075, 888];
        let greedy = vec![198, 760, 3841, 13477, 37550, 33075, 888, 279];

        let plan = dflash_commit_plan(&block, 7, &greedy, 100).unwrap();
        assert_eq!(plan.commit_tokens, block);
        assert_eq!(plan.next_token, 279);
    }
}
