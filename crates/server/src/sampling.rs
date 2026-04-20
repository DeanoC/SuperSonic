//! CPU-side sampling. Logits already come back from the engine as
//! host-resident `Vec<f32>` (one row at a time for decode), so there's no
//! GPU round-trip cost to doing this on CPU.
//!
//! Supported knobs: `temperature`, `top_p`, optional `seed`.
//! `temperature <= 0.0` collapses to argmax (reproducible, seed-independent).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Deterministic RNG used when a client supplies a `seed`, or freshly
/// generated from entropy when they don't.
pub fn rng_from_seed(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Sample one token id from `logits`.
///
/// - If `temperature <= 0.0`, returns `argmax(logits)`.
/// - Otherwise, applies `logits / temperature`, softmax, top-p truncation,
///   then a weighted pick from the surviving tokens.
///
/// `logits` is mutated in place (scaled, softmaxed). Callers that need the
/// original should clone.
pub fn sample(logits: &mut [f32], temperature: f32, top_p: f32, rng: &mut impl Rng) -> u32 {
    assert!(!logits.is_empty(), "sample: empty logits");

    if temperature <= 0.0 {
        return argmax(logits);
    }

    let inv_t = 1.0 / temperature;
    for x in logits.iter_mut() {
        *x *= inv_t;
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_logit).exp();
        sum += *x;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits);
    }
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }

    let top_p = top_p.clamp(0.0, 1.0);
    if top_p > 0.0 && top_p < 1.0 {
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept: Vec<(u32, f32)> = Vec::new();
        let mut cum = 0.0f32;
        for (idx, p) in indexed {
            kept.push((idx, p));
            cum += p;
            if cum >= top_p {
                break;
            }
        }
        return weighted_pick(&kept, rng);
    }

    let indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();
    weighted_pick(&indexed, rng)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

fn weighted_pick(weights: &[(u32, f32)], rng: &mut impl Rng) -> u32 {
    let total: f32 = weights.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return weights[0].0;
    }
    let r: f32 = rng.gen_range(0.0..total);
    let mut acc = 0.0f32;
    for &(idx, p) in weights {
        acc += p;
        if r < acc {
            return idx;
        }
    }
    weights.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_is_greedy() {
        let mut logits = vec![1.0, 3.0, 2.0, 0.5];
        let mut rng = rng_from_seed(Some(0));
        assert_eq!(sample(&mut logits, 0.0, 1.0, &mut rng), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let base = vec![0.1_f32, 0.5, 0.3, 0.4, 0.2];
        let mut a = base.clone();
        let mut b = base.clone();
        let mut rng_a = rng_from_seed(Some(42));
        let mut rng_b = rng_from_seed(Some(42));
        let tok_a = sample(&mut a, 0.7, 0.9, &mut rng_a);
        let tok_b = sample(&mut b, 0.7, 0.9, &mut rng_b);
        assert_eq!(tok_a, tok_b);
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        let mut logits = vec![10.0, -10.0, -10.0];
        let mut rng = rng_from_seed(Some(1));
        assert_eq!(sample(&mut logits, 1.0, 0.01, &mut rng), 0);
    }
}
