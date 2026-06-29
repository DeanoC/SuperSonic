//! Runtime contracts for Qwen3.6-MoE self-speculative decode.

/// Outcome of greedy-speculative accept-prefix logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptPrefixOutcome {
    /// Drafts accepted in order (length 0..=K).
    pub accepted_drafts: Vec<u32>,
    /// Token to commit after the accepted prefix. This is the base correction
    /// at the rejection point, or the bonus token when all drafts were accepted.
    pub corrected_token: u32,
}

impl AcceptPrefixOutcome {
    /// Total tokens to commit this speculative step.
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
/// `drafts[i]` is the MTP candidate for position `p + 1 + i`, and
/// `base_predictions` contains the base model's greedy predictions for those
/// positions plus one bonus prediction when all drafts are accepted.
pub fn accept_prefix_greedy(drafts: &[u32], base_predictions: &[u32]) -> AcceptPrefixOutcome {
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
                base_predictions.len(),
                i + 1
            );
        }
        let b = base_predictions[i];
        let d = drafts[i];
        if b != d {
            return AcceptPrefixOutcome {
                accepted_drafts: accepted,
                corrected_token: b,
            };
        }
        accepted.push(d);
    }
    if base_predictions.len() <= k {
        panic!(
            "accept_prefix_greedy: all {k} drafts accepted but \
             base_predictions has only {} entries - need at least {} \
             (one extra for the bonus token at position p+K+1).",
            base_predictions.len(),
            k + 1
        );
    }
    AcceptPrefixOutcome {
        accepted_drafts: accepted,
        corrected_token: base_predictions[k],
    }
}

/// Like [`accept_prefix_greedy`] but tolerates a base-predictions slice that
/// ends at the rejection point.
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
    if n < drafts.len() {
        panic!(
            "accept_prefix_greedy_partial: base_predictions stopped at \
             {n} entries (covering drafts[0..{n}]) without a rejection, \
             but drafts has {} entries. The driver must run the verify \
             loop either to first rejection (inclusive) or all the way \
             through to the bonus prediction.",
            drafts.len()
        );
    }
    if base_predictions.len() <= drafts.len() {
        panic!(
            "accept_prefix_greedy_partial: all {n} drafts accepted but \
             base_predictions has only {n} entries - need at least \
             {} (one extra for the bonus token at position p+K+1).",
            n + 1
        );
    }
    AcceptPrefixOutcome {
        accepted_drafts: drafts.to_vec(),
        corrected_token: base_predictions[drafts.len()],
    }
}

/// Result of one speculative-decode step.
#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    /// Tokens to commit this step, in order. Always length 1..=K+1.
    pub emitted_tokens: Vec<u32>,
    /// Number of MTP draft tokens proposed by this step.
    pub n_drafted: usize,
    /// Number of MTP drafts accepted.
    pub n_accepted: usize,
    /// Number of base-model verify chains run by this step.
    pub base_steps: usize,
    /// Extra base-model replay chains after batched partial acceptance.
    pub replay_steps: usize,
    /// `[hidden]` BF16 little-endian from the last base decode step.
    pub final_hidden_bytes: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::{accept_prefix_greedy, accept_prefix_greedy_partial, SpeculativeStepResult};

    #[test]
    fn first_draft_rejected_emits_only_correction() {
        let outcome = accept_prefix_greedy(&[42, 7, 9], &[100, 99, 98, 97]);

        assert_eq!(outcome.accepted_drafts, Vec::<u32>::new());
        assert_eq!(outcome.corrected_token, 100);
        assert_eq!(outcome.n_emit(), 1);
        assert_eq!(outcome.emitted(), vec![100]);
    }

    #[test]
    fn all_drafts_accepted_emits_bonus() {
        let outcome = accept_prefix_greedy(&[1, 2, 3], &[1, 2, 3, 4]);

        assert_eq!(outcome.accepted_drafts, vec![1, 2, 3]);
        assert_eq!(outcome.corrected_token, 4);
        assert_eq!(outcome.n_emit(), 4);
        assert_eq!(outcome.emitted(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn partial_helper_handles_early_rejection() {
        let outcome = accept_prefix_greedy_partial(&[12, 689, 12], &[12, 689, 999]);

        assert_eq!(outcome.accepted_drafts, vec![12, 689]);
        assert_eq!(outcome.corrected_token, 999);
    }

    #[test]
    #[should_panic(expected = "without a rejection")]
    fn partial_helper_panics_on_short_no_rejection() {
        let _ = accept_prefix_greedy_partial(&[1, 2, 3], &[1, 2]);
    }

    #[test]
    fn speculative_step_result_is_runtime_visible() {
        let result = SpeculativeStepResult {
            emitted_tokens: vec![10, 11],
            n_drafted: 3,
            n_accepted: 1,
            base_steps: 4,
            replay_steps: 1,
            final_hidden_bytes: vec![0, 1],
        };

        assert_eq!(result.emitted_tokens, vec![10, 11]);
        assert_eq!(result.n_accepted, 1);
    }
}
