/// Compute max absolute delta between two f32 slices.
pub fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Argmax of an f32 slice.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Check if the top-k token IDs match between two logit vectors.
/// Returns the number of IDs in common within the top-k.
pub fn top_k_agreement(a: &[f32], b: &[f32], k: usize) -> usize {
    fn top_k_ids(logits: &[f32], k: usize) -> Vec<u32> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| *i as u32).collect()
    }
    let a_top = top_k_ids(a, k);
    let b_top = top_k_ids(b, k);
    a_top.iter().filter(|id| b_top.contains(id)).count()
}

/// Cosine similarity between two f32 slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Full validation report for a logit comparison.
#[derive(Debug)]
pub struct LogitReport {
    pub max_delta: f32,
    pub argmax_match: bool,
    pub top5_agreement: usize,
    pub cosine_sim: f32,
}

impl LogitReport {
    pub fn compare(native: &[f32], oracle: &[f32]) -> Self {
        Self {
            max_delta: max_abs_delta(native, oracle),
            argmax_match: argmax(native) == argmax(oracle),
            top5_agreement: top_k_agreement(native, oracle, 5),
            cosine_sim: cosine_similarity(native, oracle),
        }
    }
}

/// Full validation report for a decode sequence.
#[derive(Debug)]
pub struct DecodeReport {
    pub steps: usize,
    pub max_delta: f32,
    pub token_match_rate: f32,
    pub all_argmax_match: bool,
}

impl DecodeReport {
    pub fn from_steps(
        native_logits: &[Vec<f32>],
        oracle_logits: &[Vec<f32>],
        native_tokens: &[u32],
        oracle_tokens: &[u32],
    ) -> Self {
        let steps = native_logits.len().min(oracle_logits.len());
        let mut max_delta = 0.0f32;
        let mut argmax_matches = 0;

        for i in 0..steps {
            let delta = max_abs_delta(&native_logits[i], &oracle_logits[i]);
            if delta > max_delta {
                max_delta = delta;
            }
            if argmax(&native_logits[i]) == argmax(&oracle_logits[i]) {
                argmax_matches += 1;
            }
        }

        let token_steps = native_tokens.len().min(oracle_tokens.len());
        let token_matches = native_tokens
            .iter()
            .zip(oracle_tokens.iter())
            .filter(|(a, b)| a == b)
            .count();

        Self {
            steps,
            max_delta,
            token_match_rate: if token_steps > 0 {
                token_matches as f32 / token_steps as f32
            } else {
                1.0
            },
            all_argmax_match: argmax_matches == steps,
        }
    }
}
