//! Qwen3.6-MoE self-speculative decoding — Phase 6.3 building blocks.
//!
//! This module hosts the orchestration layer that ties the MTP draft
//! chain (`crate::qwen36_moe_mtp`) to the base-model verifier. The
//! current contents are pure-logic helpers — the GPU-bound speculative
//! driver that wires this into `decode_text` lands in Phase 6.3c.
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
