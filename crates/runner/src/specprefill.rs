//! SpecPrefill (arXiv 2502.02789) — host-side token-importance selection.
//!
//! Phase A scaffolding: pure-CPU implementation of the paper's chunked
//! top-K selection rule, kept here so the kernel-side speculator
//! attention export (Phase C) can plug into a validated selection path.
//!
//! ## Algorithm (paper §3.4)
//!
//! Given per-token importance scores `s[0..T]` produced by the speculator's
//! look-ahead attention rows (max over heads + layers, mean over look-ahead
//! tokens — done elsewhere), we:
//!
//!   1. 1-D average-pool `s` with a small odd window (paper uses 5–10) using
//!      "same" boundary handling — smooths score variance across nearby
//!      tokens before chunk selection.
//!   2. Walk the prompt in fixed-size chunks (paper uses 32 or 64). Within
//!      each chunk pick the top `ceil(keep_ratio * chunk_len)` tokens by
//!      smoothed score.
//!   3. Force-keep a fixed prefix (BOS + system) and suffix (final query
//!      tokens) regardless of score. Paper §3.4 calls this out as a
//!      stability fix for instruction-tuned models.
//!   4. Return the union, sorted ascending — these are the "original
//!      position IDs" the target model's prefill receives.
//!
//! ## What this module does NOT do
//!
//! - Run the speculator forward pass (Phase C).
//! - Touch the target's prefill kernel (Phase B/C — needs RoPE-indirect
//!   and sparse-causal-mask kernels first).
//! - Decide the keep ratio at runtime; the caller passes it in.
//!
//! See `docs/research/2026-05-03-specprefill-feasibility.md` for the
//! end-to-end plan and the Python reference implementation in
//! `oracle/specprefill_oracle.py` for ground-truth parity.

/// Selection knobs. All are user-facing CLI surface in Phase C.
#[derive(Debug, Clone, Copy)]
pub struct SelectionConfig {
    /// Fraction of tokens to keep within each chunk. Clamped to [0, 1].
    /// Paper benchmarks use 0.10 / 0.30 / 0.50 / 0.70 / 0.90.
    pub keep_ratio: f32,
    /// Chunk size for top-K selection. Paper: 32 or 64.
    pub chunk_size: usize,
    /// 1-D average-pool window for score smoothing. Must be odd.
    /// Paper: 5–10. We use odd-only for symmetric "same" padding.
    pub pool_window: usize,
    /// Always-keep prefix length (BOS + system prompt). Paper §3.4
    /// argues this stabilises instruction-tuned outputs. 0 disables.
    pub always_keep_prefix: usize,
    /// Always-keep suffix length (final query / lookahead anchors).
    /// 0 disables.
    pub always_keep_suffix: usize,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        // Mid-of-paper defaults. The CLI in Phase C will override these.
        Self {
            keep_ratio: 0.30,
            chunk_size: 32,
            pool_window: 5,
            always_keep_prefix: 4,
            always_keep_suffix: 4,
        }
    }
}

/// 1-D average-pool with "same" boundary handling. Output length equals
/// input length. The window must be odd so the kernel is centred on each
/// position.
///
/// Boundary policy: positions whose window extends past `[0, len)` average
/// only the in-bounds neighbours (i.e., we shrink the divisor at the
/// boundary, equivalent to zero-padding-then-divide-by-actual-count).
/// This matches PyTorch `F.avg_pool1d(..., count_include_pad=False)` with
/// the corresponding "same" padding length, which is what the reference
/// vLLM monkey-patch uses.
pub fn avg_pool_1d_same(scores: &[f32], window: usize) -> Vec<f32> {
    assert!(
        window % 2 == 1,
        "avg_pool_1d_same: window must be odd, got {window}"
    );
    let len = scores.len();
    if window <= 1 || len == 0 {
        return scores.to_vec();
    }
    let half = window / 2;
    let mut out = vec![0.0_f32; len];
    for i in 0..len {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(len);
        let mut sum = 0.0_f32;
        for j in lo..hi {
            sum += scores[j];
        }
        out[i] = sum / (hi - lo) as f32;
    }
    out
}

/// Run the chunked top-K selection on smoothed scores. Returns a sorted
/// vector of kept indices (i.e., original position IDs) — no duplicates,
/// strictly ascending, all values in `[0, scores.len())`.
///
/// The result always includes the forced prefix/suffix bands when the
/// config requests them, even if their scores were below the per-chunk
/// top-K threshold.
pub fn select_kept_positions(scores: &[f32], cfg: &SelectionConfig) -> Vec<u32> {
    let t = scores.len();
    if t == 0 {
        return Vec::new();
    }
    // Clamp keep_ratio defensively. NaN coerces to 0.0.
    let keep = if cfg.keep_ratio.is_nan() {
        0.0_f32
    } else {
        cfg.keep_ratio.clamp(0.0, 1.0)
    };

    let smoothed = avg_pool_1d_same(scores, cfg.pool_window);

    let mut kept = vec![false; t];

    // Force-keep prefix and suffix bands. Clamp lengths to t/2 each so the
    // forced regions can't double-count or overrun.
    let prefix_len = cfg.always_keep_prefix.min(t);
    let suffix_len = cfg.always_keep_suffix.min(t - prefix_len);
    for i in 0..prefix_len {
        kept[i] = true;
    }
    for i in (t - suffix_len)..t {
        kept[i] = true;
    }

    // Chunk-wise top-K. Chunk boundaries are aligned to position 0; the
    // tail chunk may be shorter than `chunk_size`. Within each chunk we
    // pick `ceil(keep_ratio * chunk_len)` highest-scoring positions
    // (ties broken by index ascending — deterministic).
    let chunk_size = cfg.chunk_size.max(1);
    let mut chunk_buf: Vec<(f32, u32)> = Vec::with_capacity(chunk_size);
    let mut start = 0usize;
    while start < t {
        let end = (start + chunk_size).min(t);
        let chunk_len = end - start;
        let target = ((keep * chunk_len as f32).ceil() as usize).min(chunk_len);
        if target == 0 {
            start = end;
            continue;
        }
        chunk_buf.clear();
        for i in start..end {
            chunk_buf.push((smoothed[i], i as u32));
        }
        // Descending by score, ascending by index on tie.
        chunk_buf.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        for &(_, idx) in chunk_buf.iter().take(target) {
            kept[idx as usize] = true;
        }
        start = end;
    }

    let mut out: Vec<u32> = (0..t as u32).filter(|&i| kept[i as usize]).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Convenience: number of positions a given config + length would keep.
/// Useful for VRAM/scratch sizing without running the full selection.
pub fn keep_count(scores_len: usize, cfg: &SelectionConfig) -> usize {
    if scores_len == 0 {
        return 0;
    }
    let keep = cfg.keep_ratio.clamp(0.0, 1.0);
    let chunk_size = cfg.chunk_size.max(1);
    let prefix_len = cfg.always_keep_prefix.min(scores_len);
    let suffix_len = cfg.always_keep_suffix.min(scores_len - prefix_len);
    let mut from_chunks = 0usize;
    let mut start = 0usize;
    while start < scores_len {
        let end = (start + chunk_size).min(scores_len);
        let chunk_len = end - start;
        from_chunks += ((keep * chunk_len as f32).ceil() as usize).min(chunk_len);
        start = end;
    }
    // Upper bound — chunk picks may overlap with forced bands. The exact
    // count requires running selection.
    (from_chunks + prefix_len + suffix_len).min(scores_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_pool_window_1_is_identity() {
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(avg_pool_1d_same(&s, 1), s);
    }

    #[test]
    fn avg_pool_3_centred_with_shrinking_boundary() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = avg_pool_1d_same(&s, 3);
        // i=0: avg(1,2)=1.5  (left edge, divisor=2)
        // i=1: avg(1,2,3)=2.0
        // i=2: avg(2,3,4)=3.0
        // i=3: avg(3,4,5)=4.0
        // i=4: avg(4,5)=4.5  (right edge, divisor=2)
        assert_eq!(out, vec![1.5, 2.0, 3.0, 4.0, 4.5]);
    }

    #[test]
    #[should_panic(expected = "window must be odd")]
    fn avg_pool_rejects_even_window() {
        avg_pool_1d_same(&[1.0, 2.0], 2);
    }

    #[test]
    fn select_returns_sorted_unique_indices() {
        // Decreasing scores; with keep_ratio 0.5, chunk_size 4 we expect
        // the top-2 of each chunk. Pool window 1 (no smoothing).
        let scores: Vec<f32> = (0..8).map(|i| (8 - i) as f32).collect();
        let cfg = SelectionConfig {
            keep_ratio: 0.5,
            chunk_size: 4,
            pool_window: 1,
            always_keep_prefix: 0,
            always_keep_suffix: 0,
        };
        let kept = select_kept_positions(&scores, &cfg);
        // Chunk 1 [0..4]: scores 8,7,6,5 → top-2 = 0,1
        // Chunk 2 [4..8]: scores 4,3,2,1 → top-2 = 4,5
        assert_eq!(kept, vec![0, 1, 4, 5]);
    }

    #[test]
    fn select_force_keeps_prefix_and_suffix() {
        // Uniform scores so chunk top-K is index-tie-broken; the forced
        // bands overlap the tie picks differently though.
        let scores = vec![1.0; 16];
        let cfg = SelectionConfig {
            keep_ratio: 0.5,
            chunk_size: 8,
            pool_window: 1,
            always_keep_prefix: 2,
            always_keep_suffix: 2,
        };
        let kept = select_kept_positions(&scores, &cfg);
        // Tie-break = ascending → first 4 of chunk1 picked: 0,1,2,3.
        // First 4 of chunk2: 8,9,10,11. Plus forced prefix 0,1 (already in)
        // and forced suffix 14,15 (NEW).
        assert_eq!(kept, vec![0, 1, 2, 3, 8, 9, 10, 11, 14, 15]);
    }

    #[test]
    fn select_keep_ratio_zero_keeps_only_forced_bands() {
        let scores = vec![1.0; 10];
        let cfg = SelectionConfig {
            keep_ratio: 0.0,
            chunk_size: 4,
            pool_window: 1,
            always_keep_prefix: 1,
            always_keep_suffix: 1,
        };
        let kept = select_kept_positions(&scores, &cfg);
        assert_eq!(kept, vec![0, 9]);
    }

    #[test]
    fn select_keep_ratio_one_keeps_everything() {
        let scores = vec![1.0; 7];
        let cfg = SelectionConfig {
            keep_ratio: 1.0,
            chunk_size: 4,
            pool_window: 1,
            always_keep_prefix: 0,
            always_keep_suffix: 0,
        };
        let kept = select_kept_positions(&scores, &cfg);
        assert_eq!(kept, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn select_handles_empty_input() {
        let kept = select_kept_positions(&[], &SelectionConfig::default());
        assert!(kept.is_empty());
        assert_eq!(keep_count(0, &SelectionConfig::default()), 0);
    }

    #[test]
    fn select_clamps_out_of_range_keep_ratio() {
        // Above 1.0 should clamp to 1.0; NaN should clamp to 0.0.
        let scores = vec![1.0; 4];
        let mut cfg = SelectionConfig {
            keep_ratio: 999.0,
            chunk_size: 4,
            pool_window: 1,
            always_keep_prefix: 0,
            always_keep_suffix: 0,
        };
        let kept = select_kept_positions(&scores, &cfg);
        assert_eq!(kept, (0..4).collect::<Vec<_>>());

        cfg.keep_ratio = f32::NAN;
        let kept = select_kept_positions(&scores, &cfg);
        assert!(kept.is_empty());
    }

    #[test]
    fn select_smoothing_changes_picks() {
        // Score spike at idx 4 surrounded by low scores. Without smoothing,
        // chunk1 [0..4] (uniform low) ties on idx 0; chunk2 [4..8] (spike
        // at 4) picks 4. With pool_window=3 the spike at 4 also raises
        // smoothed[3] and smoothed[5], pulling those into top picks of
        // their respective chunks.
        let scores = vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];

        let cfg_no_smooth = SelectionConfig {
            keep_ratio: 0.25, // ceil(0.25*4)=1 per chunk
            chunk_size: 4,
            pool_window: 1,
            always_keep_prefix: 0,
            always_keep_suffix: 0,
        };
        let kept = select_kept_positions(&scores, &cfg_no_smooth);
        // Chunk1 uniform → idx 0 (ascending tie). Chunk2 → idx 4 (the spike).
        assert_eq!(kept, vec![0, 4]);

        let cfg_smooth = SelectionConfig {
            pool_window: 3,
            ..cfg_no_smooth
        };
        let kept = select_kept_positions(&scores, &cfg_smooth);
        // After pool_window=3:
        //   smoothed[3] = (0+0+10)/3 ≈ 3.33
        //   smoothed[4] = (0+10+0)/3 ≈ 3.33
        //   smoothed[5] = (10+0+0)/3 ≈ 3.33
        //   chunk1[0..4]: smoothed[3] is the only nonzero → pick idx 3.
        //   chunk2[4..8]: smoothed[4]=smoothed[5]=3.33 (tie) → pick idx 4.
        assert_eq!(kept, vec![3, 4]);
    }

    #[test]
    fn keep_count_upper_bound_matches_or_exceeds_actual() {
        let scores: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let cfg = SelectionConfig {
            keep_ratio: 0.4,
            chunk_size: 16,
            pool_window: 3,
            always_keep_prefix: 2,
            always_keep_suffix: 2,
        };
        let actual = select_kept_positions(&scores, &cfg).len();
        let bound = keep_count(scores.len(), &cfg);
        assert!(
            bound >= actual,
            "keep_count bound {bound} must be >= actual {actual}",
        );
        // And bound is tight when forced bands don't overlap chunk picks
        // (they may, so we only assert >=).
    }
}
