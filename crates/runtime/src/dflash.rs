//! Native DFlash serving support for dense Qwen low-bit targets.
//!
//! The public surface starts small: option normalization that the server
//! builder and tests can share. The GPU-resident session lives in this module
//! too so `supersonic-serve` can keep target and draft weights loaded across
//! OpenAI-compatible requests.

use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Once};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use qwen35::state::LinearStateSnapshot;
use qwen35_dflash as draft;
use supersonic_core::registry::ModelVariant;

use crate::decode_engine::DecodeEngine;
use crate::prefill_engine::PrefillAppendVerifyResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashOptions {
    pub draft_dir: PathBuf,
    pub block: Option<usize>,
    pub tap_layers: Option<String>,
}

impl DFlashOptions {
    pub fn effective_block_size(
        &self,
        model_variant: &ModelVariant,
        draft_block_size: usize,
    ) -> Result<usize> {
        if draft_block_size == 0 {
            bail!("DFlash draft block_size must be greater than 0");
        }
        let block = match self.block {
            Some(block) => block,
            None if matches!(
                model_variant,
                ModelVariant::Qwen3_6_27B | ModelVariant::Qwen3_8_27B
            ) =>
            {
                draft_block_size
            }
            None => 3.min(draft_block_size),
        };
        if block == 0 || block > draft_block_size {
            bail!("--dflash-block must be in 1..={draft_block_size} (got {block})");
        }
        Ok(block)
    }

    pub fn resolve_tap_layers(
        &self,
        draft_layers: &[u32],
        num_target_layers: usize,
    ) -> Result<Vec<usize>> {
        if let Some(raw) = self.tap_layers.as_deref() {
            parse_tap_layers(raw, num_target_layers)
        } else {
            Ok(draft_layers.iter().map(|&layer| layer as usize).collect())
        }
    }
}

pub fn parse_tap_layers(raw: &str, num_target_layers: usize) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let layer: usize = trimmed
            .parse()
            .map_err(|e| anyhow::anyhow!("--dflash-tap-layers: bad integer '{trimmed}': {e}"))?;
        if layer >= num_target_layers {
            bail!("tap layer {layer} out of range (num_target_layers={num_target_layers})");
        }
        out.push(layer);
    }
    if out.is_empty() {
        bail!("--dflash-tap-layers must list at least one integer");
    }
    Ok(out)
}

pub struct DFlashSession {
    target: DecodeEngine,
    draft_config: draft::DFlashConfig,
    draft_weights: draft::DFlashWeights,
    draft_rotary: draft::RotaryTables,
    draft_scratch: draft::DFlashScratch,
    draft_noise_embedding: GpuBuffer,
    draft_state: draft::DFlashState,
    tap_layers: Vec<usize>,
    block_size: usize,
    ordinal: usize,
    tap_history_gpu: GpuBuffer,
    tap_history_capacity: usize,
    per_tap_row_bytes: usize,
    hidden_size: usize,
}

#[derive(Debug, Clone)]
pub struct DFlashGenerateOutput {
    pub token_ids: Vec<u32>,
    pub rounds_run: usize,
    pub accepted_total: usize,
    pub decode_ms: f64,
}

impl DFlashSession {
    pub fn new(
        target: DecodeEngine,
        options: DFlashOptions,
        model_variant: &ModelVariant,
        text_config: &qwen35::config::TextConfig,
        context_tokens: usize,
        ordinal: usize,
    ) -> Result<Self> {
        let target_embed: Arc<GpuBuffer> = Arc::clone(&target.weights().embed_tokens);
        let target_lm_head: Arc<GpuBuffer> = Arc::clone(&target.weights().lm_head);

        let draft_config = draft::load_config(&options.draft_dir)
            .map_err(|e| anyhow!("load draft config.json: {e}"))?;
        if draft_config.num_target_layers != text_config.num_hidden_layers {
            bail!(
                "draft num_target_layers={} != target layers={}",
                draft_config.num_target_layers,
                text_config.num_hidden_layers,
            );
        }
        if draft_config.hidden_size != text_config.hidden_size {
            bail!(
                "draft hidden_size {} != target hidden_size {}",
                draft_config.hidden_size,
                text_config.hidden_size,
            );
        }
        if draft_config.vocab_size != text_config.vocab_size {
            bail!(
                "draft vocab_size {} != target vocab_size {} - the draft shares target embeddings",
                draft_config.vocab_size,
                text_config.vocab_size,
            );
        }
        let mask_id = draft_config.dflash_config.mask_token_id;
        if (mask_id as usize) >= text_config.vocab_size {
            bail!(
                "draft mask_token_id {mask_id} is out of range for target vocab_size {}",
                text_config.vocab_size,
            );
        }

        let tap_layers = options.resolve_tap_layers(
            &draft_config.dflash_config.target_layer_ids,
            draft_config.num_target_layers,
        )?;
        if tap_layers.len() != draft_config.num_taps() {
            bail!(
                "tap layer count {} mismatches draft fuser tap count {}",
                tap_layers.len(),
                draft_config.num_taps(),
            );
        }
        let block_size = options.effective_block_size(model_variant, draft_config.block_size)?;

        let draft_weights = draft::DFlashWeights::load(
            &options.draft_dir,
            &draft_config,
            ordinal,
            target_embed,
            target_lm_head,
        )
        .map_err(|e| anyhow!("load draft weights: {e}"))?;

        let draft_ctx_capacity = context_tokens.max(1);
        let draft_max_ctx = (context_tokens + draft_config.block_size)
            .max(draft_config.block_size * 4)
            .max(1024);
        let draft_rotary = draft::RotaryTables::build(&draft_config, ordinal, draft_max_ctx)
            .map_err(|e| anyhow!("build draft RoPE: {e}"))?;
        let draft_scratch =
            draft::DFlashScratch::new_with_ctx_capacity(ordinal, &draft_config, draft_ctx_capacity)
                .map_err(|e| anyhow!("alloc draft scratch: {e}"))?;
        let draft_noise_embedding = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[1, draft_config.block_size, draft_config.hidden_size],
        )
        .map_err(|e| anyhow!("alloc draft noise embedding scratch: {e}"))?;
        let draft_state = draft::DFlashState::new(ordinal, &draft_config, draft_max_ctx)
            .map_err(|e| anyhow!("alloc draft state: {e}"))?;

        let per_tap_row_bytes =
            tap_layers.len() * text_config.hidden_size * ScalarType::BF16.size_in_bytes();
        let tap_history_capacity = context_tokens + block_size;
        let tap_history_gpu = GpuBuffer::zeros(
            ordinal,
            ScalarType::BF16,
            &[
                tap_history_capacity.max(1),
                tap_layers.len() * text_config.hidden_size,
            ],
        )
        .map_err(|e| anyhow!("alloc GPU tap history: {e}"))?;

        tracing::info!(
            draft_dir = %options.draft_dir.display(),
            block_size,
            taps = ?tap_layers,
            context_tokens,
            "DFlash session ready"
        );

        Ok(Self {
            target,
            draft_config,
            draft_weights,
            draft_rotary,
            draft_scratch,
            draft_noise_embedding,
            draft_state,
            tap_layers,
            block_size,
            ordinal,
            tap_history_gpu,
            tap_history_capacity,
            per_tap_row_bytes,
            hidden_size: text_config.hidden_size,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.target.reset()?;
        self.draft_state.reset();
        Ok(())
    }

    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        eos_ids: &[u32],
    ) -> Result<DFlashGenerateOutput> {
        if max_tokens == 0 {
            return Ok(DFlashGenerateOutput {
                token_ids: Vec::new(),
                rounds_run: 0,
                accepted_total: 0,
                decode_ms: 0.0,
            });
        }
        if prompt_ids.is_empty() {
            bail!("DFlash generation requires a non-empty prompt");
        }
        if prompt_ids.len() + max_tokens + self.block_size > self.tap_history_capacity {
            bail!(
                "DFlash tap history capacity {} is too small for prompt {} + max_tokens {} + block {}",
                self.tap_history_capacity,
                prompt_ids.len(),
                max_tokens,
                self.block_size,
            );
        }

        self.reset()?;
        let prefill_result = self
            .target
            .prefill_native_with_taps(prompt_ids, &self.tap_layers)?;
        let taps = match prefill_result.tap_hiddens_all.as_ref() {
            Some(per_tap) => flatten_tap_history(per_tap, prompt_ids.len(), self.hidden_size)?,
            None => flatten_tap_history(
                &prefill_result.tap_hiddens.unwrap_or_default(),
                1,
                self.hidden_size,
            )?,
        };
        upload_taps_to_gpu_history(&mut self.tap_history_gpu, 0, self.per_tap_row_bytes, &taps)?;
        let mut tap_history_len = if self.per_tap_row_bytes == 0 {
            0
        } else {
            taps.len() / self.per_tap_row_bytes
        };
        if tap_history_len == 0 {
            bail!("DFlash prefill did not produce any target tap history");
        }

        let mut bonus_seed = DecodeEngine::greedy_sample(&prefill_result.logits);
        let mut committed_len = prompt_ids.len();
        let mut generated_ids = Vec::with_capacity(max_tokens);
        let mut rounds_run = 0usize;
        let mut accepted_total = 0usize;
        let decode_start = Instant::now();

        while generated_ids.len() < max_tokens {
            if eos_ids.contains(&bonus_seed) {
                generated_ids.push(bonus_seed);
                break;
            }

            let remaining_budget = max_tokens - generated_ids.len();
            if remaining_budget == 1 {
                generated_ids.push(bonus_seed);
                break;
            }

            rounds_run += 1;
            let l = committed_len;
            let verify_len = dflash_verify_len_for_round(remaining_budget, self.block_size);
            let draft_ctx = tap_history_len.min(self.draft_scratch.ctx_capacity);
            let tap_start_row = tap_history_len - draft_ctx;

            let draft_candidates = draft_forward_and_sample(
                &mut self.draft_state,
                &mut self.draft_scratch,
                &mut self.draft_noise_embedding,
                &self.draft_rotary,
                &self.draft_weights,
                &self.target,
                &self.tap_history_gpu,
                tap_start_row,
                draft_ctx,
                bonus_seed,
                self.block_size,
                self.draft_config.dflash_config.mask_token_id,
                self.ordinal,
            )?;

            let gpu_tap_history = if dflash_gpu_tap_history_enabled() {
                Some((
                    &mut self.tap_history_gpu,
                    tap_history_len,
                    self.per_tap_row_bytes,
                ))
            } else {
                None
            };
            let verify_output = verify_block_for_dflash(
                &mut self.target,
                &draft_candidates[..verify_len],
                l,
                &self.tap_layers,
                gpu_tap_history,
            )?;
            let target_next_ids = verify_output.greedy_ids()?;

            let mut accept_n = 1usize;
            while accept_n < verify_len {
                if accept_n > target_next_ids.len() {
                    bail!(
                        "DFlash verifier returned {} greedy IDs, insufficient for accept_n={accept_n} verify_len={verify_len}",
                        target_next_ids.len()
                    );
                }
                if target_next_ids[accept_n - 1] == draft_candidates[accept_n] {
                    accept_n += 1;
                } else {
                    break;
                }
            }
            let carried_seed = *target_next_ids
                .get(accept_n - 1)
                .ok_or_else(|| anyhow!("target verifier returned no carried seed"))?;

            let accepted_len = accept_n.min(remaining_budget);
            let committed_block = draft_candidates[..accepted_len].to_vec();
            let finish_after_commit = accepted_len >= remaining_budget
                || committed_block.iter().any(|t| eos_ids.contains(t));

            let mut next_bonus_seed = None;
            if !finish_after_commit {
                match verify_output {
                    DFlashVerifyOutput::Captured(result) => {
                        if let Some(per_tap) = result.tap_hiddens_all.as_ref() {
                            let taps_bytes =
                                flatten_tap_history(per_tap, verify_len, self.hidden_size)?;
                            let committed_tap_bytes = accepted_len * self.per_tap_row_bytes;
                            upload_taps_to_gpu_history(
                                &mut self.tap_history_gpu,
                                tap_history_len,
                                self.per_tap_row_bytes,
                                &taps_bytes[..committed_tap_bytes],
                            )?;
                        } else if !dflash_gpu_tap_history_enabled() {
                            bail!("captured prefill verifier did not return tap history");
                        }
                        if accepted_len == verify_len {
                            self.target
                                .commit_prefill_append_full_accept_owned(result)?;
                        } else {
                            self.target
                                .commit_prefill_append_verify_owned(result, accepted_len)?;
                        }
                        tap_history_len += accepted_len;
                        next_bonus_seed = Some(carried_seed);
                    }
                    DFlashVerifyOutput::CapturedWindowed(result) => {
                        let mut remaining = accepted_len;
                        let mut copied_tap_rows = 0usize;
                        let mut committed = false;
                        for segment in &result.segments {
                            if remaining == 0 {
                                break;
                            }
                            let expected_start = accepted_len - remaining;
                            if segment.start != expected_start {
                                bail!(
                                    "windowed prefill segment start {} != expected {}",
                                    segment.start,
                                    expected_start
                                );
                            }
                            let take = remaining.min(segment.len);
                            if let Some(per_tap) = segment.result.tap_hiddens_all.as_ref() {
                                let taps_bytes =
                                    flatten_tap_history(per_tap, segment.len, self.hidden_size)?;
                                let take_bytes = take * self.per_tap_row_bytes;
                                upload_taps_to_gpu_history(
                                    &mut self.tap_history_gpu,
                                    tap_history_len + copied_tap_rows,
                                    self.per_tap_row_bytes,
                                    &taps_bytes[..take_bytes],
                                )?;
                            } else if !dflash_gpu_tap_history_enabled() {
                                bail!("windowed prefill verifier did not return tap history");
                            }
                            copied_tap_rows += take;

                            if remaining <= segment.len {
                                if remaining == segment.len {
                                    self.target
                                        .commit_prefill_append_full_accept(&segment.result)?;
                                } else {
                                    self.target
                                        .commit_prefill_append_verify(&segment.result, remaining)?;
                                }
                                committed = true;
                                remaining = 0;
                                break;
                            }
                            remaining -= segment.len;
                        }
                        if !committed || remaining != 0 || copied_tap_rows != accepted_len {
                            bail!(
                                "windowed prefill commit could not cover accepted_len={} with {} segments",
                                accepted_len,
                                result.segments.len()
                            );
                        }
                        tap_history_len += accepted_len;
                        next_bonus_seed = Some(carried_seed);
                    }
                    DFlashVerifyOutput::Fallback { snap, .. } => {
                        self.target
                            .state_mut()
                            .restore_linear(&snap, self.ordinal)
                            .map_err(|e| anyhow!("restore linear: {e}"))?;
                        self.target.rewind_full_kv_filled(l);
                        let (logits, taps_bytes) = self.target.decode_block_with_taps_kernel(
                            &committed_block,
                            l,
                            &self.tap_layers,
                        )?;
                        let expected_tap_bytes = accepted_len * self.per_tap_row_bytes;
                        if taps_bytes.len() != expected_tap_bytes {
                            bail!(
                                "decode_block_with_taps_kernel returned {} tap bytes, expected {}",
                                taps_bytes.len(),
                                expected_tap_bytes,
                            );
                        }
                        upload_taps_to_gpu_history(
                            &mut self.tap_history_gpu,
                            tap_history_len,
                            self.per_tap_row_bytes,
                            &taps_bytes,
                        )?;
                        tap_history_len += accepted_len;
                        next_bonus_seed = Some(DecodeEngine::greedy_sample(&logits));
                    }
                }
            }

            committed_len = l + accepted_len;
            accepted_total += accepted_len;
            let mut hit_eos = false;
            for &token in &committed_block {
                generated_ids.push(token);
                if eos_ids.contains(&token) {
                    hit_eos = true;
                    break;
                }
                if generated_ids.len() >= max_tokens {
                    break;
                }
            }
            if hit_eos || finish_after_commit {
                break;
            }
            bonus_seed =
                next_bonus_seed.ok_or_else(|| anyhow!("missing DFlash next bonus seed"))?;
            self.draft_state.reset();
        }

        Ok(DFlashGenerateOutput {
            token_ids: generated_ids,
            rounds_run,
            accepted_total,
            decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

enum DFlashVerifyOutput {
    Captured(PrefillAppendVerifyResult),
    CapturedWindowed(CapturedWindowedVerifyOutput),
    Fallback {
        logits: Vec<Vec<f32>>,
        snap: LinearStateSnapshot,
    },
}

struct CapturedWindowedVerifyOutput {
    segments: Vec<CapturedWindowSegment>,
    target_next: Vec<u32>,
}

struct CapturedWindowSegment {
    start: usize,
    len: usize,
    result: PrefillAppendVerifyResult,
}

impl DFlashVerifyOutput {
    fn greedy_ids(&self) -> Result<Vec<u32>> {
        match self {
            Self::Captured(result) => result.target_next.clone().ok_or_else(|| {
                anyhow!("captured prefill verifier did not return greedy target IDs")
            }),
            Self::CapturedWindowed(result) => Ok(result.target_next.clone()),
            Self::Fallback { logits, .. } => Ok(logits
                .iter()
                .map(|row| DecodeEngine::greedy_sample(row))
                .collect()),
        }
    }
}

fn dflash_prefill_window_scan_chunk() -> usize {
    env::var("SUPERSONIC_DFLASH_VERIFY_SCAN_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(16)
}

fn dflash_prefill_window_min_tail() -> usize {
    env::var("SUPERSONIC_DFLASH_VERIFY_MIN_TAIL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

fn dflash_final_verify_min_len(block_size: usize) -> Option<usize> {
    if env::var_os("SUPERSONIC_DFLASH_DISABLE_FINAL_VERIFY_PAD").is_some() {
        return None;
    }
    Some(
        env::var("SUPERSONIC_DFLASH_FINAL_VERIFY_MIN_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(block_size),
    )
}

fn dflash_verify_len_for_round(remaining_budget: usize, block_size: usize) -> usize {
    let clamped = block_size.min(remaining_budget);
    if remaining_budget >= block_size {
        return clamped;
    }
    let Some(min_len) = dflash_final_verify_min_len(block_size) else {
        return clamped;
    };
    clamped.max(min_len.min(block_size))
}

fn dflash_gpu_tap_history_enabled() -> bool {
    env::var_os("SUPERSONIC_DFLASH_DISABLE_GPU_TAP_HISTORY").is_none()
}

fn dflash_window_step(remaining: usize, scan_chunk: usize, min_tail: usize) -> usize {
    let step = scan_chunk.min(remaining);
    let tail = remaining - step;
    if tail > 0 && tail < min_tail {
        remaining
    } else {
        step
    }
}

fn chain_accept_needs_more(target_next: &[u32], tokens: &[u32], verify_len: usize) -> bool {
    let mut accept_n = 1usize;
    while accept_n < verify_len {
        if accept_n > target_next.len() {
            return true;
        }
        if target_next[accept_n - 1] == tokens[accept_n] {
            accept_n += 1;
        } else {
            return false;
        }
    }
    accept_n > target_next.len()
}

fn verify_block_prefill_append_windowed(
    target_engine: &mut DecodeEngine,
    tokens: &[u32],
    pos_offset: usize,
    tap_layers: &[usize],
    scan_chunk: usize,
    mut gpu_tap_history: Option<(&mut GpuBuffer, usize, usize)>,
) -> Result<CapturedWindowedVerifyOutput> {
    if scan_chunk == 0 {
        bail!("prefill append window scan chunk must be > 0");
    }

    let mut segments = Vec::new();
    let mut target_next = Vec::new();
    let min_tail = dflash_prefill_window_min_tail();
    let mut start = 0usize;
    while start < tokens.len() {
        let step = dflash_window_step(tokens.len() - start, scan_chunk, min_tail);
        let result = if let Some((history, base_row, row_bytes)) = gpu_tap_history.as_mut() {
            target_engine.verify_block_prefill_append_captured_lazy_acceptance_gpu_taps(
                &tokens[start..start + step],
                pos_offset + start,
                tap_layers,
                &mut **history,
                *base_row + start,
                *row_bytes,
            )?
        } else {
            target_engine.verify_block_prefill_append_captured_lazy_acceptance(
                &tokens[start..start + step],
                pos_offset + start,
                tap_layers,
            )?
        };
        let ids = result
            .target_next
            .clone()
            .ok_or_else(|| anyhow!("windowed prefill verifier did not return greedy target IDs"))?;
        if ids.is_empty() {
            bail!("windowed prefill verifier returned no greedy target IDs");
        }
        target_next.extend(ids);
        segments.push(CapturedWindowSegment {
            start,
            len: step,
            result,
        });

        if !chain_accept_needs_more(&target_next, tokens, tokens.len()) {
            break;
        }
        start += step;
    }

    Ok(CapturedWindowedVerifyOutput {
        segments,
        target_next,
    })
}

fn verify_block_for_dflash(
    target_engine: &mut DecodeEngine,
    tokens: &[u32],
    pos_offset: usize,
    tap_layers: &[usize],
    mut gpu_tap_history: Option<(&mut GpuBuffer, usize, usize)>,
) -> Result<DFlashVerifyOutput> {
    let force_prefill = env::var_os("SUPERSONIC_DFLASH_PREFILL_VERIFY").is_some();
    let disable_prefill = env::var_os("SUPERSONIC_DFLASH_DISABLE_PREFILL_VERIFY").is_some();
    let chunk_size = dflash_fused_verify_chunk_size(target_engine);
    let config = &target_engine.weights().config;
    let prefer_prefill_append = config.hidden_size == 5120 && config.num_hidden_layers == 64;
    let use_prefill_append = force_prefill
        || (!disable_prefill
            && (prefer_prefill_append || (chunk_size > 0 && tokens.len() > chunk_size)));

    if use_prefill_append {
        static PREFILL_NOTICE: Once = Once::new();
        PREFILL_NOTICE.call_once(|| {
            tracing::info!("DFlash using prefill-append target verifier");
        });
        let scan_chunk = dflash_prefill_window_scan_chunk();
        if scan_chunk < tokens.len()
            && env::var_os("SUPERSONIC_DFLASH_DISABLE_VERIFY_ROW_SCAN").is_none()
        {
            let result = verify_block_prefill_append_windowed(
                target_engine,
                tokens,
                pos_offset,
                tap_layers,
                scan_chunk,
                gpu_tap_history
                    .as_mut()
                    .map(|(history, start_row, row_bytes)| {
                        (&mut **history, *start_row, *row_bytes)
                    }),
            )?;
            return Ok(DFlashVerifyOutput::CapturedWindowed(result));
        }
        let captured_result =
            if let Some((history, start_row, row_bytes)) = gpu_tap_history.as_mut() {
                target_engine.verify_block_prefill_append_captured_lazy_acceptance_gpu_taps(
                    tokens,
                    pos_offset,
                    tap_layers,
                    &mut **history,
                    *start_row,
                    *row_bytes,
                )
            } else {
                target_engine.verify_block_prefill_append_captured_lazy_acceptance(
                    tokens, pos_offset, tap_layers,
                )
            };
        match captured_result {
            Ok(result) => return Ok(DFlashVerifyOutput::Captured(result)),
            Err(err) if force_prefill => {
                return Err(anyhow!("prefill append verify failed: {err}"));
            }
            Err(err) => {
                static PREFILL_FALLBACK_NOTICE: Once = Once::new();
                PREFILL_FALLBACK_NOTICE.call_once(|| {
                    tracing::warn!(
                        "DFlash prefill-append verifier failed ({err}); falling back to persistent verifier"
                    );
                });
            }
        }
    }

    let snap: LinearStateSnapshot = target_engine
        .state_mut()
        .snapshot_linear()
        .map_err(|e| anyhow!("snapshot linear: {e}"))?;

    let needs_sequential = tokens.len() > kernel_ffi::MAX_BATCH_SIZE;
    if !needs_sequential {
        match target_engine.verify_block_fused_decode(tokens, pos_offset) {
            Ok(logits) => return Ok(DFlashVerifyOutput::Fallback { logits, snap }),
            Err(err) => {
                let msg = err.to_string();
                if !msg.contains("shared-memory budget exceeded") {
                    return Err(err);
                }
            }
        }
    }

    if chunk_size > 0 && chunk_size < tokens.len() {
        let mut out = Vec::with_capacity(tokens.len());
        let mut start = 0usize;
        while start < tokens.len() {
            let remaining = tokens.len() - start;
            let step = if remaining > chunk_size && remaining - chunk_size == 1 && chunk_size > 1 {
                chunk_size - 1
            } else {
                remaining.min(chunk_size)
            };
            let end = start + step;
            match target_engine.verify_block_fused_decode(&tokens[start..end], pos_offset + start) {
                Ok(mut logits) => {
                    out.append(&mut logits);
                    start = end;
                }
                Err(err) => {
                    let msg = err.to_string();
                    if start == 0 && msg.contains("shared-memory budget exceeded") {
                        break;
                    }
                    return Err(anyhow!(
                        "chunked fused verify failed at token {start}: {err}"
                    ));
                }
            }
        }
        if out.len() == tokens.len() {
            return Ok(DFlashVerifyOutput::Fallback { logits: out, snap });
        }
    }

    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        tracing::info!(
            "DFlash fused verify does not fit this target shape/block; using sequential target verify fallback"
        );
    });

    let mut logits = Vec::with_capacity(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let (step_logits, _tap_bytes) = target_engine
            .decode_step_with_taps_kernel(tok, pos_offset + i, tap_layers)
            .map_err(|e| anyhow!("sequential verify decode step {i}: {e}"))?;
        logits.push(step_logits);
    }
    Ok(DFlashVerifyOutput::Fallback { logits, snap })
}

fn dflash_fused_verify_chunk_size(target_engine: &DecodeEngine) -> usize {
    const MAX_INPUT_CACHE_FLOATS: usize = 15872;
    let hidden_dim = target_engine.weights().config.hidden_size;
    if hidden_dim == 0 || 2 * hidden_dim > MAX_INPUT_CACHE_FLOATS {
        return 0;
    }
    (MAX_INPUT_CACHE_FLOATS / hidden_dim).min(kernel_ffi::MAX_BATCH_SIZE)
}

fn upload_taps_to_gpu_history(
    tap_history_gpu: &mut GpuBuffer,
    start_row: usize,
    row_bytes: usize,
    taps: &[u8],
) -> Result<()> {
    if taps.is_empty() {
        return Ok(());
    }
    if row_bytes == 0 || taps.len() % row_bytes != 0 {
        bail!(
            "tap upload byte length {} is not a multiple of row_bytes {}",
            taps.len(),
            row_bytes
        );
    }
    let dst_offset = start_row * row_bytes;
    if dst_offset + taps.len() > tap_history_gpu.len_bytes() {
        bail!(
            "GPU tap history write exceeds buffer: offset {} + len {} > {}",
            dst_offset,
            taps.len(),
            tap_history_gpu.len_bytes()
        );
    }
    let dst = unsafe {
        (tap_history_gpu.as_mut_ptr() as *mut u8).add(dst_offset) as *mut std::ffi::c_void
    };
    gpu_hal::copy_h2d(
        tap_history_gpu.device_ordinal(),
        dst,
        taps.as_ptr() as *const std::ffi::c_void,
        taps.len(),
    )
    .map_err(|e| anyhow!("upload tap history: {e}"))
}

fn flatten_tap_history(
    per_tap: &[Vec<u8>],
    num_positions: usize,
    hidden_dim: usize,
) -> Result<Vec<u8>> {
    if per_tap.is_empty() {
        bail!("flatten_tap_history requires at least one tap layer");
    }
    let row_bytes = hidden_dim * ScalarType::BF16.size_in_bytes();
    let expected = num_positions * row_bytes;
    for (idx, tap) in per_tap.iter().enumerate() {
        if tap.len() != expected {
            bail!(
                "tap history {idx} has {} bytes, expected {expected} ({num_positions} positions * {hidden_dim} hidden * 2)",
                tap.len(),
            );
        }
    }
    let mut out = Vec::with_capacity(num_positions * per_tap.len() * row_bytes);
    for pos in 0..num_positions {
        let start = pos * row_bytes;
        for tap in per_tap {
            out.extend_from_slice(&tap[start..start + row_bytes]);
        }
    }
    Ok(out)
}

fn draft_forward_and_sample(
    draft_state: &mut draft::DFlashState,
    draft_scratch: &mut draft::DFlashScratch,
    noise_embedding: &mut GpuBuffer,
    draft_rotary: &draft::RotaryTables,
    draft_weights: &draft::DFlashWeights,
    target_engine: &DecodeEngine,
    tap_history_gpu: &GpuBuffer,
    tap_start_row: usize,
    round_taps_len: usize,
    bonus_seed: u32,
    block_size: usize,
    mask_token_id: u32,
    ordinal: usize,
) -> Result<Vec<u32>> {
    if round_taps_len == 0 {
        bail!("draft_forward: round_taps_len must be > 0");
    }
    let hidden = draft_weights.config.hidden_size;
    let num_taps = draft_weights.config.num_taps();
    let tap_row_bytes = num_taps * hidden * ScalarType::BF16.size_in_bytes();
    let expected_bytes = round_taps_len * tap_row_bytes;
    let src_offset = tap_start_row * tap_row_bytes;
    if src_offset + expected_bytes > tap_history_gpu.len_bytes() {
        bail!(
            "GPU tap history read exceeds buffer: offset {} + len {} > {}",
            src_offset,
            expected_bytes,
            tap_history_gpu.len_bytes()
        );
    }

    let target_embed = &target_engine.weights().embed_tokens;
    let row_bytes = hidden * ScalarType::BF16.size_in_bytes();
    let expected_noise_bytes = block_size * row_bytes;
    if noise_embedding.len_bytes() < expected_noise_bytes {
        bail!(
            "draft noise embedding scratch has {} bytes, need {}",
            noise_embedding.len_bytes(),
            expected_noise_bytes
        );
    }
    for i in 0..block_size {
        let tok = if i == 0 { bonus_seed } else { mask_token_id };
        let src_off = tok as usize * row_bytes;
        let dst_off = i * row_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            unsafe {
                (noise_embedding.as_mut_ptr() as *mut u8).add(dst_off) as *mut std::ffi::c_void
            },
            target_embed.offset_ptr(src_off),
            row_bytes,
        )
        .map_err(|e| anyhow!("noise_embedding gather slot {i}: {e}"))?;
    }

    if expected_bytes > draft_scratch.fuser_input.len_bytes() {
        bail!(
            "draft fuser_input scratch has {} bytes, need {}",
            draft_scratch.fuser_input.len_bytes(),
            expected_bytes
        );
    }
    gpu_hal::copy_d2d(
        ordinal,
        draft_scratch.fuser_input.as_mut_ptr(),
        tap_history_gpu.offset_ptr(src_offset),
        expected_bytes,
    )
    .map_err(|e| anyhow!("copy target_hidden_raw: {e}"))?;

    draft::forward::forward(
        draft_weights,
        draft_state,
        draft_scratch,
        draft_rotary,
        noise_embedding,
        draft::ForwardParams {
            ctx_len: round_taps_len,
            q_len: block_size,
            pos_offset: 0,
        },
    )
    .map_err(|e| anyhow!("draft forward: {e}"))?;

    let target_weights = target_engine.weights();
    let lm_head_buf: &GpuBuffer = target_weights.lm_head.as_ref();
    let vocab = draft_weights.config.vocab_size;
    if draft_scratch.logits.elem_count() < block_size * vocab {
        bail!(
            "draft logits scratch has {} elems, need {}",
            draft_scratch.logits.elem_count(),
            block_size * vocab
        );
    }

    let mut fused_argmax = false;
    if let Some((lm_head_qtype, scale, zero)) = target_weights.lm_head_lowbit_params(hidden) {
        if env::var_os("SUPERSONIC_DFLASH_DISABLE_Q6_K_LM_HEAD_ARGMAX_FUSED").is_none()
            && block_size == 16
            && vocab % 16 == 0
            && hidden % 256 == 0
            && lm_head_qtype == qwen35::weights::LOWBIT_GGML_Q6_K
            && target_weights.lm_head_awq_inv_scale.is_none()
        {
            fused_argmax = kernel_ffi::prefill_ffi::matmul_q6_k_m16_argmax(
                ordinal,
                1,
                block_size,
                vocab,
                hidden,
                &draft_scratch.final_hidden,
                lm_head_buf,
                &mut draft_scratch.lm_head_block_best_vals,
                &mut draft_scratch.lm_head_block_best_indices,
                &mut draft_scratch.argmax_indices,
            )
            .map_err(|e| anyhow!("draft lm_head fused argmax: {e}"))?;
        }
        if !fused_argmax {
            kernel_ffi::prefill_ffi::matmul_rhs_transposed_int4(
                ordinal,
                1,
                block_size,
                vocab,
                hidden,
                &draft_scratch.final_hidden,
                lm_head_buf,
                scale,
                zero,
                target_weights.lm_head_awq_inv_scale.as_ref(),
                target_weights.int4_group_size,
                lm_head_qtype,
                &mut draft_scratch.logits,
            )
            .map_err(|e| anyhow!("draft lm_head low-bit matmul: {e}"))?;
        }
    } else {
        kernel_ffi::matmul_rhs_transposed_4b(
            ordinal,
            ScalarType::BF16,
            1,
            block_size,
            vocab,
            hidden,
            &draft_scratch.final_hidden,
            lm_head_buf,
            &mut draft_scratch.logits,
        )
        .map_err(|e| anyhow!("draft lm_head: {e}"))?;
    }

    if !fused_argmax {
        kernel_ffi::prefill_ffi::argmax_bf16_rows(
            ordinal,
            block_size,
            vocab,
            &draft_scratch.logits,
            &mut draft_scratch.argmax_indices,
        )
        .map_err(|e| anyhow!("draft logits argmax: {e}"))?;
    }
    let argmax_bytes = draft_scratch
        .argmax_indices
        .to_host_bytes()
        .map_err(|e| anyhow!("draft argmax indices D2H: {e}"))?;
    let mut candidates: Vec<u32> = argmax_bytes
        .chunks_exact(4)
        .take(block_size)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if candidates.len() != block_size {
        bail!(
            "draft argmax returned {} candidates, expected {block_size}",
            candidates.len()
        );
    }
    if let Some(first) = candidates.first_mut() {
        *first = bonus_seed;
    }
    Ok(candidates)
}
