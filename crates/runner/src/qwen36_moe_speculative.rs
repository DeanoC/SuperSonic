//! Qwen3.6-MoE self-speculative decoding — Phase 6.3 building blocks.
//!
//! This module hosts the orchestration layer that ties the MTP draft
//! chain (`crate::qwen36_moe_mtp`) to the base-model verifier.
//!
//! Phase 6.3b shipped the pure-logic accept-prefix helpers
//! ([`accept_prefix_greedy`], [`accept_prefix_greedy_partial`]).
//! Phase 6.3c (this module after the next round) adds
//! [`run_speculative_decode_step`], which orchestrates one full
//! speculative iteration: MTP draft chain → sequential base
//! verification with early termination → accept-prefix → emitted
//! tokens. The base stepper is injected as a callback so the
//! orchestration is testable with a mock that doesn't need the
//! full Qwen3.6-MoE model loaded; engine wiring under
//! `--speculative-decode` is Phase 6.3d.
//!
//! ## Performance note
//!
//! Sequential verification has zero amortized speedup over plain
//! greedy decode: each accepted draft still requires one base step
//! (to produce the next prediction), and rejected drafts incur a
//! base step too (for the rejection check). Total base steps per
//! emitted token stays at 1.0 on average. The actual throughput
//! win comes from Phase 6.4's batched verification kernel, which
//! runs all K verify steps in a single base call (multi-query
//! attention). Phase 6.3 builds the protocol and validates
//! correctness; Phase 6.4 makes it fast.
//!
//! Speculative decoding (greedy) at a glance:
//!
//! 1. Base model just sampled token `T_p` at position `p`.
//! 2. MTP produces draft tokens `[d_0, d_1, ..., d_{K-1}]` =
//!    its predictions for what comes after `T_p`.
//! 3. We already have the base model's logits for position `p+1`
//!    (computed when we sampled `T_p`); call its argmax `b_0`.
//!    Compare to `d_0`:
//!      - If `b_0 != d_0`: reject. Emit `b_0` as the new token at
//!        position `p+1`. 1 token committed this step.
//!      - If `b_0 == d_0`: accept `d_0`. Feed it as input at
//!        position `p+1` to advance the base, producing logits for
//!        position `p+2`; argmax = `b_1`. Compare to `d_1`. Etc.
//! 4. If all K drafts agree, the final base step yields a "bonus"
//!    `b_K` at position `p+K+1`. Total tokens this step: K+1.
//! 5. Tokens emitted this step = (accepted prefix) + 1 corrected/bonus
//!    token. Accept rates of 50-70% on similar-architecture models
//!    move our 38 tok/s greedy floor toward 70-100+ tok/s.
//!
//! [`accept_prefix_greedy`] is the pure orchestration function: given
//! the drafts and the base's K+1 predictions, it returns the accept-
//! prefix outcome. The driver in Phase 6.3c is responsible for running
//! the base K+1 times to produce those predictions (with early
//! termination on the first mismatch — this helper assumes all K+1
//! were computed for ease of unit testing).
//!
//! References:
//! - vLLM's `Qwen3NextMultiTokenPredictor` — the Python reference our
//!   MTP path matches byte-for-byte (Phase 6.2c.x parity tests).
//! - `tests/fixtures/qwen36_moe/mtp_vllm_reference.json` — vLLM cross-
//!   check confirms our drafts match `vllm.spec_decode.eagle.propose`.

/// Outcome of greedy-speculative accept-prefix logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptPrefixOutcome {
    /// Drafts accepted in order (length 0..=K). The base verified
    /// `accepted_drafts[i]` produces the same prediction the base
    /// would have made on its own at position `p+1+i`.
    pub accepted_drafts: Vec<u32>,
    /// Token to commit AFTER the accepted prefix. If `accepted_drafts.len()
    /// < drafts.len()`, this is the base's predicted "correction" at
    /// the rejection position (replacing the rejected draft). If all K
    /// drafts were accepted, this is the "bonus" token: the base's
    /// prediction at position `p + K + 1`, computed by feeding the
    /// last accepted draft.
    pub corrected_token: u32,
}

impl AcceptPrefixOutcome {
    /// Total tokens to commit this speculative step:
    ///   `accepted_drafts.len() + 1` (always at least 1, since
    ///   `corrected_token` is always present).
    pub fn n_emit(&self) -> usize {
        self.accepted_drafts.len() + 1
    }

    /// All tokens to commit this speculative step in order.
    pub fn emitted(&self) -> Vec<u32> {
        let mut out = self.accepted_drafts.clone();
        out.push(self.corrected_token);
        out
    }
}

/// Compute the greedy speculative accept-prefix outcome.
///
/// Inputs:
///   - `drafts`: K candidate tokens proposed by the MTP head, where
///     `drafts[i]` is the prediction for position `p + 1 + i`.
///   - `base_predictions`: K+1 greedy-argmax predictions from the base
///     model, where:
///       * `base_predictions[0]` is the base's prediction for
///         position `p + 1` (computed when `T_p` was sampled — we
///         already have it from the previous decode step).
///       * `base_predictions[i]` for `i in 1..=K` is the base's
///         prediction for position `p + 1 + i`, computed by feeding
///         `drafts[i-1]` as input at position `p + i`.
///
/// Output: see [`AcceptPrefixOutcome`].
///
/// # Early-termination caveat
///
/// In a real driver the base predictions past the first mismatch are
/// never computed (we stop the verify loop early to save GPU work).
/// This pure helper accepts them anyway — it just ignores them past
/// the rejection point. The caller can pass shorter slices when early
/// termination has already occurred, as long as `base_predictions.len()
/// >= number_of_predictions_actually_computed`.
///
/// # Panics
///
/// Panics if `base_predictions.len() < drafts.len() + 1` AND no early
/// rejection occurred (i.e., we'd need the bonus prediction but it
/// wasn't supplied). Use [`accept_prefix_greedy_partial`] when the
/// caller has already terminated early.
pub fn accept_prefix_greedy(
    drafts: &[u32],
    base_predictions: &[u32],
) -> AcceptPrefixOutcome {
    let k = drafts.len();
    let mut accepted: Vec<u32> = Vec::with_capacity(k);
    for i in 0..k {
        if i >= base_predictions.len() {
            panic!(
                "accept_prefix_greedy: base_predictions has only {} entries \
                 but {} are needed to verify draft {i}; the caller terminated \
                 the verify loop early without recording a corrected token. \
                 Use `accept_prefix_greedy_partial` for the early-termination \
                 case.",
                base_predictions.len(), i + 1
            );
        }
        let b = base_predictions[i];
        let d = drafts[i];
        if b != d {
            // Rejection at index `i`. Discard d_i and the rest;
            // commit b_i as the correction.
            return AcceptPrefixOutcome {
                accepted_drafts: accepted,
                corrected_token: b,
            };
        }
        accepted.push(d);
    }
    // All K drafts accepted. The bonus token is base_predictions[K],
    // produced by feeding drafts[K-1] (or by feeding T_p if K==0; the
    // K==0 case degenerates to "no speculation, take base_predictions[0]
    // as the next token").
    if base_predictions.len() <= k {
        panic!(
            "accept_prefix_greedy: all {k} drafts accepted but \
             base_predictions has only {} entries — need at least {} \
             (one extra for the bonus token at position p+K+1).",
            base_predictions.len(), k + 1
        );
    }
    AcceptPrefixOutcome {
        accepted_drafts: accepted,
        corrected_token: base_predictions[k],
    }
}

/// Like [`accept_prefix_greedy`] but tolerates a base-predictions
/// slice that ends at the rejection point. Used when the driver
/// terminates the verify loop early (the production fast path —
/// every skipped base step is a kernel launch saved).
///
/// Behaviour:
///   - If the loop walked through all `drafts.len()` indices without
///     rejection, the caller MUST also include the bonus prediction
///     (so `base_predictions.len() >= drafts.len() + 1`) — otherwise
///     this panics, since we cannot fabricate a bonus token.
///   - If rejection occurred at index `i`, `base_predictions.len()`
///     should be exactly `i + 1` (driver stopped after computing the
///     mismatching prediction). Anything ≥ `i + 1` works; extra
///     entries are ignored.
pub fn accept_prefix_greedy_partial(
    drafts: &[u32],
    base_predictions: &[u32],
) -> AcceptPrefixOutcome {
    let n = base_predictions.len().min(drafts.len());
    for i in 0..n {
        let b = base_predictions[i];
        let d = drafts[i];
        if b != d {
            return AcceptPrefixOutcome {
                accepted_drafts: drafts[..i].to_vec(),
                corrected_token: b,
            };
        }
    }
    // No rejection within the supplied slice — full prefix accepted up
    // to whatever the slice covered.
    if n < drafts.len() {
        // Driver terminated before covering all drafts but reported no
        // rejection. That's a contract violation: with no rejection
        // the loop must have gone all the way to the bonus.
        panic!(
            "accept_prefix_greedy_partial: base_predictions stopped at \
             {n} entries (covering drafts[0..{n}]) without a rejection, \
             but drafts has {} entries. The driver must run the verify \
             loop either to first rejection (inclusive) or all the way \
             through to the bonus prediction.",
            drafts.len()
        );
    }
    // n == drafts.len() — all drafts accepted; need the bonus.
    if base_predictions.len() <= drafts.len() {
        panic!(
            "accept_prefix_greedy_partial: all {n} drafts accepted but \
             base_predictions has only {n} entries — need at least \
             {} (one extra for the bonus token at position p+K+1).",
            n + 1
        );
    }
    AcceptPrefixOutcome {
        accepted_drafts: drafts.to_vec(),
        corrected_token: base_predictions[drafts.len()],
    }
}

// ============================================================================
// Phase 6.3c: speculative-decode driver (orchestration only)
// ----------------------------------------------------------------------------
// `run_speculative_decode_step` runs one full speculative iteration:
//
//   1. Upload h_base (last base step's final-hidden) to GPU.
//   2. Run K-step MTP draft chain → K candidate tokens.
//   3. Sequential base verification with early termination on first
//      mismatch. Each verify iter is one base decode step at the
//      corresponding position, fed via the caller-supplied
//      `base_step` closure.
//   4. Apply the accept-prefix logic from Phase 6.3b.
//   5. Return the emitted tokens + n_accepted + final-hidden bytes
//      (so the next speculative step has a fresh `h_base`).
//
// The `base_step` callback abstraction is what makes this testable
// without the full Qwen3.6-MoE model loaded. The real engine wraps
// `lookup_embed_row` + `run_chained_decode_fast` + `lm_head_launch`
// + host argmax in the closure; tests pass a mock that returns canned
// predictions and synthesised final-hidden bytes.
// ============================================================================

use anyhow::{Context, Result};
use gpu_hal::{GpuBuffer, ScalarType};

use crate::qwen36_moe_decode::{MtpLayerBuffers, MultiLayerGeom};
use crate::qwen36_moe_mtp::{
    run_mtp_draft_chain, MtpChainScratch, MtpForwardScratch,
};

/// Result of one speculative-decode step.
#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    /// Tokens to commit this step, in order. Always length 1..=K+1
    /// (always at least one token: the corrected/bonus token).
    pub emitted_tokens: Vec<u32>,
    /// Number of MTP drafts accepted (0..=K). Exposed for stats /
    /// `--emit-stage-timings` / acceptance-rate logging.
    pub n_accepted: usize,
    /// `[hidden]` BF16 little-endian — the final-hidden bytes from
    /// the LAST base decode step that ran during this iteration.
    /// Becomes the next speculative step's `h_base` per the vLLM
    /// recurrent equation.
    pub final_hidden_bytes: Vec<u8>,
}

/// Run one speculative-decode iteration.
///
/// ## Inputs
///
/// - `mtp`, `forward_scratch`, `chain_scratch`, `embed_w_buf`,
///   `lm_head_w_buf`: same as [`run_mtp_draft_chain`] (Phase 6.3a).
/// - `h_base_in`: `[hidden]` BF16 — the LAST base step's final-hidden
///   bytes (the input to the base's lm_head that produced
///   `first_token_id`). Becomes MTP's `h_base` for step 0.
/// - `first_token_id`: the token the base just sampled (= `T_{p+1}`
///   in the docstring above). Becomes MTP's `next_token_id` at
///   step 0 and the base verifier's first input.
/// - `base_position`: absolute base position of `first_token_id` —
///   the cache slot it WILL be written to when the verify loop
///   feeds it. Per the engine convention, this equals
///   `decode_text`'s `position` value AFTER the regular sample
///   that produced `first_token_id`.
/// - `num_drafts` (K): how many MTP drafts to generate.
///
/// ## `base_step` closure contract
///
/// Each call advances the base by one step:
///
///   `(predicted_next_token, final_hidden_bytes) = base_step(position, input_token)`
///
/// where:
///   - `position`: cache slot to write `input_token`'s K/V into.
///     The implementation must run the base chain at `position`
///     and produce logits for `position + 1`.
///   - `input_token`: token id to embed and feed at `position`.
///   - `predicted_next_token`: greedy argmax over the produced
///     logits (= base's prediction for `position + 1`).
///   - `final_hidden_bytes`: `[hidden]` BF16 — the chain output the
///     lm_head consumed.  Used both for the next speculative step's
///     `h_base` and for diagnostics.
///
/// The closure is called K times when all drafts are accepted (the
/// K-th call produces the bonus prediction), or `j+1` times when the
/// j-th draft is rejected (j in 0..K-1).
///
/// ## Output
///
/// `emitted_tokens` is `accepted_drafts ++ [corrected_or_bonus]`,
/// always non-empty. Caller appends to `generated_ids` and advances
/// `position` by `emitted_tokens.len()`.
///
/// ## Performance
///
/// Sequential verification has zero amortized speedup over plain
/// greedy decode: each iter still runs one full base step. The MTP
/// chain is the only "free" compute (~K layers vs the base's
/// 40 layers per step). Phase 6.4's batched verification kernel
/// is what delivers throughput.
#[allow(clippy::too_many_arguments)]
pub fn run_speculative_decode_step<F>(
    ordinal: usize,
    geom: &MultiLayerGeom,
    mtp: &mut MtpLayerBuffers,
    forward_scratch: &mut MtpForwardScratch,
    chain_scratch: &mut MtpChainScratch,
    embed_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    h_base_in: &[u8],
    first_token_id: u32,
    base_position: i32,
    num_drafts: usize,
    mut base_step: F,
) -> Result<SpeculativeStepResult>
where
    F: FnMut(i32, u32) -> Result<(u32, Vec<u8>)>,
{
    let hidden = geom.hidden as usize;
    if h_base_in.len() != hidden * 2 {
        anyhow::bail!(
            "run_speculative_decode_step: h_base_in.len() {} != \
             hidden*2 ({}) BF16 bytes",
            h_base_in.len(),
            hidden * 2
        );
    }
    if num_drafts == 0 {
        // K=0 degenerate: no MTP work, but the contract still requires
        // we emit one token (the function is documented as always
        // returning a non-empty `emitted_tokens` so the caller's
        // `position += emitted.len()` advances). Run a single base
        // step at `base_position` with `first_token_id` — exactly
        // what plain greedy decode would do for this token. This
        // keeps the speculative driver a safe drop-in even when a
        // tunable disables speculation, avoiding the stalled-loop
        // failure mode the `--speculative-decode --num-speculative
        // -tokens=0` knob would otherwise hit.
        let (predicted, fh) = base_step(base_position, first_token_id)
            .context("speculative: K=0 fallback base step")?;
        return Ok(SpeculativeStepResult {
            emitted_tokens: vec![predicted],
            n_accepted: 0,
            final_hidden_bytes: fh,
        });
    }

    // --- 1. Upload h_base for the MTP chain. ---
    let h_base_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[hidden],
        h_base_in,
    )
    .context("speculative: upload h_base")?;

    // --- 2. Generate K MTP drafts. ---
    let drafts_records = run_mtp_draft_chain(
        ordinal,
        geom,
        mtp,
        base_position,
        &h_base_buf,
        first_token_id,
        num_drafts,
        embed_w_buf,
        lm_head_w_buf,
        forward_scratch,
        chain_scratch,
    )
    .context("speculative: MTP draft chain")?;
    let drafts: Vec<u32> = drafts_records.iter().map(|r| r.draft_token_id).collect();

    // --- 3. Sequential verify with early termination. ---
    //
    // Iter k (k in 0..K):
    //   feed `input_k` at position `base_position + k`,
    //   read `predicted_k` = base's prediction for `base_position + k + 1`,
    //   compare to drafts[k].
    //
    // input_0 = first_token_id (= T_{base_position}, just sampled).
    // input_k (k > 0) = drafts[k-1] (the just-accepted token).
    //
    // First mismatch at k: emit drafts[..k] ++ [predicted_k]; n_accepted = k.
    // No mismatch through k = K-1: emit drafts[..K] ++ [bonus]; n_accepted = K.
    //   The bonus is computed by an extra base step at position
    //   `base_position + K` with input = drafts[K-1] — the K+1-th
    //   call to `base_step`.

    let mut emitted: Vec<u32> = Vec::with_capacity(num_drafts + 1);
    let mut last_final_hidden: Vec<u8> = h_base_in.to_vec();
    let mut input = first_token_id;

    for k in 0..num_drafts {
        let pos = base_position + (k as i32);
        let (predicted, fh) = base_step(pos, input)
            .with_context(|| format!("speculative: base verify k={k} pos={pos}"))?;
        last_final_hidden = fh;
        if predicted == drafts[k] {
            // Accept drafts[k]. Carry forward as next iter's input.
            emitted.push(drafts[k]);
            input = drafts[k];
        } else {
            // Reject. predicted replaces drafts[k]; stop here.
            emitted.push(predicted);
            return Ok(SpeculativeStepResult {
                emitted_tokens: emitted,
                n_accepted: k,
                final_hidden_bytes: last_final_hidden,
            });
        }
    }

    // All K drafts accepted. Bonus base step at position p+K with
    // input = drafts[K-1] (= last accepted token).
    let pos = base_position + (num_drafts as i32);
    let (bonus, fh) = base_step(pos, input)
        .with_context(|| format!("speculative: bonus base step pos={pos}"))?;
    emitted.push(bonus);
    Ok(SpeculativeStepResult {
        emitted_tokens: emitted,
        n_accepted: num_drafts,
        final_hidden_bytes: fh,
    })
}

// ============================================================================
// Phase 6.4b: batched-K speculative driver
// ----------------------------------------------------------------------------
// Same protocol as `run_speculative_decode_step` but the verify side runs
// all K+1 base steps via a single closure call instead of K+1 sequential
// closure calls. Lets the closure amortize work across K (notably:
// `qwen36_moe::lm_head_batched_launch` runs ONE batched lm_head over K+1
// inputs instead of K+1 single-M GEMVs, saving ~1.5 ms × K of weight
// reads at vocab=248k).
//
// Trade-off vs the per-step driver:
//   - Per-step: early termination on first mismatch — closure called only
//     j+1 times (j = accept count). Strictly cheaper when the accept
//     rate is low.
//   - Batched: always K+1 closure calls. Wins when accept rate is high
//     enough that the per-step path wouldn't have terminated early
//     anyway, OR when batched lm_head + amortized weight reads
//     dominate.
//
// On the local "quick brown fox" / "Once upon a time" / "def factorial"
// fixtures we measured 100% accept across K=3 — full clean win for
// batched. On lower-accept-rate prompts the win narrows or inverts; the
// engine exposes both paths so a flag-controlled switch is possible.
// ============================================================================

/// Run one speculative-decode iteration with a batched verify
/// closure. See [`run_speculative_decode_step`] for the protocol;
/// the only difference is the closure shape:
///
/// ```ignore
/// F: FnOnce(&[(i32, u32)]) -> Result<Vec<(u32, Vec<u8>)>>
/// ```
///
/// The closure receives K+1 `(position, input_token)` pairs:
///   - `[0]`: `(base_position, first_token_id)` — the just-sampled
///     token's verify slot, predicting `drafts[0]`.
///   - `[i]` for `i in 1..=K`: `(base_position+i, drafts[i-1])` —
///     each accepted draft's slot, predicting the next position.
///
/// And returns K+1 `(predicted_next_token, final_hidden_bytes)` tuples
/// in the same order. The driver applies [`accept_prefix_greedy`]
/// against the K predictions for `drafts` (with the K-th prediction
/// as the bonus when all accepted).
///
/// Implementations should run the K+1 base chains internally and use
/// `qwen36_moe::lm_head_batched_launch` to fold the K+1 lm_head GEMVs
/// into one launch for the actual perf payoff.
#[allow(clippy::too_many_arguments)]
pub fn run_speculative_decode_step_batched<F>(
    ordinal: usize,
    geom: &MultiLayerGeom,
    mtp: &mut MtpLayerBuffers,
    forward_scratch: &mut MtpForwardScratch,
    chain_scratch: &mut MtpChainScratch,
    embed_w_buf: &GpuBuffer,
    lm_head_w_buf: &GpuBuffer,
    h_base_in: &[u8],
    first_token_id: u32,
    base_position: i32,
    num_drafts: usize,
    base_step_batched: F,
) -> Result<SpeculativeStepResult>
where
    F: FnOnce(&[(i32, u32)]) -> Result<Vec<(u32, Vec<u8>)>>,
{
    let hidden = geom.hidden as usize;
    if h_base_in.len() != hidden * 2 {
        anyhow::bail!(
            "run_speculative_decode_step_batched: h_base_in.len() {} != \
             hidden*2 ({}) BF16 bytes",
            h_base_in.len(),
            hidden * 2
        );
    }
    if num_drafts == 0 {
        // K=0 degenerate: single base step. Same fallback as the
        // per-step driver — preserves the "always emit ≥1 token"
        // contract for engine forward progress.
        let outputs = base_step_batched(&[(base_position, first_token_id)])
            .context("speculative_batched: K=0 fallback base step")?;
        if outputs.len() != 1 {
            anyhow::bail!(
                "speculative_batched: K=0 closure returned {} outputs, \
                 expected exactly 1",
                outputs.len()
            );
        }
        let (predicted, fh) = outputs.into_iter().next().unwrap();
        return Ok(SpeculativeStepResult {
            emitted_tokens: vec![predicted],
            n_accepted: 0,
            final_hidden_bytes: fh,
        });
    }

    // --- 1. Upload h_base for the MTP chain. ---
    let h_base_buf = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[hidden],
        h_base_in,
    )
    .context("speculative_batched: upload h_base")?;

    // --- 2. Generate K MTP drafts. ---
    let drafts_records = run_mtp_draft_chain(
        ordinal,
        geom,
        mtp,
        base_position,
        &h_base_buf,
        first_token_id,
        num_drafts,
        embed_w_buf,
        lm_head_w_buf,
        forward_scratch,
        chain_scratch,
    )
    .context("speculative_batched: MTP draft chain")?;
    let drafts: Vec<u32> = drafts_records.iter().map(|r| r.draft_token_id).collect();

    // --- 3. Build K+1 verify (position, input) pairs. ---
    let mut verify_inputs: Vec<(i32, u32)> = Vec::with_capacity(num_drafts + 1);
    verify_inputs.push((base_position, first_token_id));
    for k in 0..num_drafts {
        verify_inputs.push((base_position + (k as i32) + 1, drafts[k]));
    }

    // --- 4. Run the batched closure once. ---
    let predictions = base_step_batched(&verify_inputs)
        .context("speculative_batched: batched verify closure")?;
    if predictions.len() != num_drafts + 1 {
        anyhow::bail!(
            "speculative_batched: closure returned {} predictions for \
             {} verify inputs (expected {})",
            predictions.len(),
            num_drafts + 1,
            num_drafts + 1
        );
    }

    // --- 5. Walk accept-prefix logic over the K+1 predictions. ---
    //
    // predictions[i].0 is the base's predicted-next-token after feeding
    // verify_inputs[i]. predictions[0..K] are compared to drafts[0..K];
    // predictions[K] is the bonus when all accepted.
    let mut emitted: Vec<u32> = Vec::with_capacity(num_drafts + 1);
    let mut n_accepted: usize = 0;
    let mut final_hidden_idx: usize = 0;
    for k in 0..num_drafts {
        let predicted = predictions[k].0;
        if predicted == drafts[k] {
            emitted.push(drafts[k]);
            n_accepted = k + 1;
            final_hidden_idx = k + 1;
        } else {
            // Reject: corrected token is `predicted`. The relevant
            // final-hidden for the recurrent feed is the source
            // hidden of the corrected token — that's
            // `predictions[k].1`, computed when the closure ran the
            // base on (verify_inputs[k]).
            emitted.push(predicted);
            final_hidden_idx = k;
            break;
        }
    }
    // Use `n_accepted == num_drafts` (NOT `emitted.len() == num_drafts`)
    // to detect full-accept — at K=2 with rejection at the last index,
    // both branches exit with `emitted.len() == 2`, but only the
    // full-accept branch has `n_accepted == 2`. Using `len()` would
    // misclassify the last-index rejection as full-accept and
    // append a bogus bonus token.
    if n_accepted == num_drafts {
        // All K drafts accepted; emit the bonus.
        let bonus = predictions[num_drafts].0;
        emitted.push(bonus);
        final_hidden_idx = num_drafts;
    }

    // Take the chosen final_hidden_bytes by destructuring
    // `predictions` to avoid an extra clone.
    let final_hidden_bytes = predictions
        .into_iter()
        .nth(final_hidden_idx)
        .map(|(_, fh)| fh)
        .expect("final_hidden_idx is valid — closure returned K+1 entries");

    Ok(SpeculativeStepResult {
        emitted_tokens: emitted,
        n_accepted,
        final_hidden_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_draft_rejected_emits_only_correction() {
        // Drafts: [42, 7, 9]. Base predicts 100 first → mismatch at 0.
        // No drafts accepted; corrected = 100. Total emit = 1.
        let r = accept_prefix_greedy(
            /* drafts */ &[42, 7, 9],
            /* base_predictions */ &[100, 99, 98, 97],
        );
        assert_eq!(r.accepted_drafts, Vec::<u32>::new());
        assert_eq!(r.corrected_token, 100);
        assert_eq!(r.n_emit(), 1);
        assert_eq!(r.emitted(), vec![100]);
    }

    #[test]
    fn mid_chain_rejection() {
        // Drafts: [12, 689, 12]. Base predicts [12, 689, 999, ...].
        // First two match, third rejects. Accepted = [12, 689];
        // corrected = 999.
        let r = accept_prefix_greedy(
            &[12, 689, 12],
            &[12, 689, 999, 0],
        );
        assert_eq!(r.accepted_drafts, vec![12, 689]);
        assert_eq!(r.corrected_token, 999);
        assert_eq!(r.n_emit(), 3);
        assert_eq!(r.emitted(), vec![12, 689, 999]);
    }

    #[test]
    fn all_drafts_accepted_emits_bonus() {
        // Drafts: [a, b, c]. Base predicts [a, b, c, d] — full
        // agreement, bonus = d. Total emit = K+1 = 4.
        let r = accept_prefix_greedy(
            &[1, 2, 3],
            &[1, 2, 3, 4],
        );
        assert_eq!(r.accepted_drafts, vec![1, 2, 3]);
        assert_eq!(r.corrected_token, 4);
        assert_eq!(r.n_emit(), 4);
        assert_eq!(r.emitted(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn empty_drafts_takes_base_top1() {
        // K=0 (no speculation): we still have base_predictions[0] from
        // the previous step's logits. Outcome degenerates to "emit the
        // base's normal next token".
        let r = accept_prefix_greedy(&[], &[42]);
        assert_eq!(r.accepted_drafts, Vec::<u32>::new());
        assert_eq!(r.corrected_token, 42);
        assert_eq!(r.n_emit(), 1);
        assert_eq!(r.emitted(), vec![42]);
    }

    #[test]
    fn partial_helper_handles_early_termination_at_rejection() {
        // Driver stopped after computing the first mismatching
        // prediction (i=2). base_predictions has exactly 3 entries
        // — drafts[0..2] matched, drafts[2] rejected.
        let r = accept_prefix_greedy_partial(
            &[12, 689, 12],
            &[12, 689, 999],
        );
        assert_eq!(r.accepted_drafts, vec![12, 689]);
        assert_eq!(r.corrected_token, 999);
    }

    #[test]
    fn partial_helper_handles_all_accept_with_bonus() {
        // Driver ran through all K drafts with no rejection; supplied
        // base_predictions of length K+1 includes the bonus.
        let r = accept_prefix_greedy_partial(
            &[1, 2, 3],
            &[1, 2, 3, 99],
        );
        assert_eq!(r.accepted_drafts, vec![1, 2, 3]);
        assert_eq!(r.corrected_token, 99);
    }

    #[test]
    #[should_panic(expected = "without a rejection")]
    fn partial_helper_panics_on_short_no_rejection() {
        // Driver supplied 2 base predictions for 3 drafts but neither
        // mismatched — that's a contract violation (no rejection means
        // the loop must have run to the bonus).
        let _ = accept_prefix_greedy_partial(&[1, 2, 3], &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "need at least")]
    fn partial_helper_panics_on_missing_bonus() {
        // All drafts accepted but bonus not supplied.
        let _ = accept_prefix_greedy_partial(&[1, 2, 3], &[1, 2, 3]);
    }

    #[test]
    fn vllm_fixture_brown_fox_full_accept() {
        // Cross-check using the data we know is correct: SuperSonic's
        // chain produces [15217, 5388, 13] for "The quick brown fox
        // jumps over"; vLLM produces the same; the base model's
        // greedy decode (per `tests/fixtures/qwen36_moe/
        // mtp_vllm_reference.json`'s `outputs[0].text == " the lazy
        // dog.\\n\\nThe quick brown"`) starts with token 279 (" the").
        //
        // If MTP+base agree perfectly, accept-prefix returns
        // accepted=[15217, 5388, 13], corrected=<token after ".">. We
        // simulate "perfect agreement" here — the actual base
        // verification is wired in Phase 6.3c.
        let drafts = [15217u32, 5388, 13];
        let base = [15217u32, 5388, 13, 271]; // 271 = "\n\n" in tokenizer
        let r = accept_prefix_greedy(&drafts, &base);
        assert_eq!(r.accepted_drafts, drafts.to_vec());
        assert_eq!(r.corrected_token, 271);
        assert_eq!(r.n_emit(), 4);
    }
}
