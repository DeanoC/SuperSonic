use crate::qwen36_moe::types::PositionPair;

/// Build replay inputs for speculative verification.
///
/// `base` is the position of `first_token`. Accepted draft tokens advance both
/// the RoPE and cache timelines by one token while preserving split positions.
pub fn speculative_replay_inputs(
    first_token: u32,
    base: PositionPair,
    emitted_tokens: &[u32],
    n_accepted: usize,
) -> Vec<(PositionPair, u32)> {
    let mut replay = Vec::with_capacity(n_accepted + 1);
    replay.push((base, first_token));
    for (i, &tok) in emitted_tokens.iter().take(n_accepted).enumerate() {
        let off = 1 + i as i32;
        replay.push((PositionPair::split(base.rope + off, base.cache + off), tok));
    }
    replay
}

pub fn partial_accept_replay_inputs(
    first_token: u32,
    base: PositionPair,
    emitted_tokens: &[u32],
    n_accepted: usize,
    draft_count: usize,
) -> Option<Vec<(PositionPair, u32)>> {
    (n_accepted < draft_count)
        .then(|| speculative_replay_inputs(first_token, base, emitted_tokens, n_accepted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_replay_inputs_preserve_split_timeline() {
        let replay = speculative_replay_inputs(42, PositionPair::split(11, 3), &[7, 8, 9], 2);

        assert_eq!(
            replay,
            vec![
                (PositionPair::split(11, 3), 42),
                (PositionPair::split(12, 4), 7),
                (PositionPair::split(13, 5), 8),
            ]
        );
    }

    #[test]
    fn partial_accept_replay_inputs_only_for_partial_acceptance() {
        assert!(partial_accept_replay_inputs(42, PositionPair::dense(5), &[7, 8], 2, 2).is_none());

        assert_eq!(
            partial_accept_replay_inputs(42, PositionPair::dense(5), &[7, 8], 1, 2),
            Some(vec![
                (PositionPair::dense(5), 42),
                (PositionPair::dense(6), 7),
            ])
        );
    }
}
