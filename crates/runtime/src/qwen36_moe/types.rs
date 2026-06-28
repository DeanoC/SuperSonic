/// Hybrid pattern: every 4th layer is full attention. Indices 3, 7, 11, ...
/// are full; everything else is linear. Matches Qwen3.6-MoE 35B-A3B.
pub const HYBRID_FULL_ATTN_STRIDE: i32 = 4;

/// `true` when `layer_idx + 1` is a multiple of [`HYBRID_FULL_ATTN_STRIDE`].
pub fn is_full_attn_layer(layer_idx: i32) -> bool {
    (layer_idx + 1) % HYBRID_FULL_ATTN_STRIDE == 0
}

/// Per-step position pair. Decouples the absolute RoPE rotation timeline from
/// the KV cache slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionPair {
    /// Absolute RoPE position.
    pub rope: i32,
    /// KV cache slot index.
    pub cache: i32,
}

impl PositionPair {
    /// Dense-decode shortcut: rope and cache slot agree.
    #[inline]
    pub const fn dense(p: i32) -> Self {
        Self { rope: p, cache: p }
    }

    /// Decoupled SpecPrefill / MTP-style pair.
    #[inline]
    pub const fn split(rope: i32, cache: i32) -> Self {
        Self { rope, cache }
    }

    /// `true` when rope and cache agree.
    #[inline]
    pub const fn is_dense(self) -> bool {
        self.rope == self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_pattern_marks_every_fourth_layer_full() {
        for li in 0..16 {
            let expect_full = matches!(li, 3 | 7 | 11 | 15);
            assert_eq!(is_full_attn_layer(li), expect_full, "layer {li}");
        }
    }

    #[test]
    fn position_pair_helpers_preserve_dense_and_split_shapes() {
        assert_eq!(PositionPair::dense(7), PositionPair { rope: 7, cache: 7 });
        assert!(PositionPair::dense(7).is_dense());
        assert_eq!(
            PositionPair::split(11, 3),
            PositionPair { rope: 11, cache: 3 }
        );
        assert!(!PositionPair::split(11, 3).is_dense());
    }
}
