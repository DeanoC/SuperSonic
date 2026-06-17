//! End-to-end parity for cross-family SpecPrefill cosine scoring on
//! Qwen3.6-35B-A3B with a Qwen3.5-0.8B drafter (R1 prototype).
//!
//! Mirrors `specprefill_qwen35_9b_cosine_parity.rs` (same-family) but
//! relaxes the keep=1.00 identity bar from `cossim >= 0.999` to
//! `cossim >= 0.99` and drops the multitoken byte-equality test.
//!
//! **Why the relaxation:** the sparse path forces chained decode (the
//! persistent decode kernel doesn't take a `cache_pos` parameter yet,
//! so we can't decouple KV-slot from RoPE-pos there). Even when
//! `cache_pos == position` for every step (keep=1.00), the chained
//! vs persistent kernels are different fused-op shapes — same math
//! but different reduction order, so the BF16 outputs are close but
//! not bit-identical. Argmax stays robust; cossim ≥ 0.99 is the
//! correct numerical bar. When the persistent kernel learns
//! `cache_pos`, we can tighten this back to ≥ 0.999.
//!
//! **Why keep=0.50 doesn't require argmax match:** at 50% keep on
//! short-to-medium prompts, the dropped tokens can include
//! information that nudges the next-token argmax (we observed
//! "decade" vs "generation" on a 67-token prompt). The paper
//! reports retention on long-form tasks, not 8-token argmax parity.
//! Top-5 overlap ≥ 3 + cossim ≥ 0.65 is the topic-alignment bar.
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - SUPERSONIC_QWEN36_35B_A3B_DIR or SUPERSONIC_QWEN35_0_8B_DIR unset/missing.
//!  - SUPERSONIC_SPECPREFILL_PARITY=0.

use gpu_hal::Backend;
use std::collections::HashSet;
use std::process::Command;

fn run_supersonic_capture_logits(args: &[&str]) -> anyhow::Result<Vec<f32>> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_supersonic"));
    cmd.args(args);
    cmd.arg("--dump-last-logits");
    let out = cmd.output()?;
    if !out.status.success() {
        anyhow::bail!(
            "supersonic exited {}: stderr=\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8(out.stdout)?;
    let line = stdout
        .lines()
        .find(|l| l.starts_with("LAST_LOGITS:"))
        .ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found in stdout"))?;
    let csv = &line["LAST_LOGITS:".len()..];
    csv.trim()
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(Into::into))
        .collect()
}

fn cossim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
            if x > a.1 {
                (i, x)
            } else {
                a
            }
        })
        .0
}

fn top5(v: &[f32]) -> HashSet<usize> {
    let mut idx: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(5).map(|p| p.0).collect()
}

fn check_specprefill_env() -> Option<(String, String)> {
    if !gpu_hal::is_backend_compiled(Backend::Hip) {
        eprintln!("skipped: HIP backend not compiled");
        return None;
    }
    if std::env::var("SUPERSONIC_SPECPREFILL_PARITY").as_deref() == Ok("0") {
        eprintln!("skipped: SUPERSONIC_SPECPREFILL_PARITY=0");
        return None;
    }
    let target = match std::env::var("SUPERSONIC_QWEN36_35B_A3B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN36_35B_A3B_DIR unset/missing");
            return None;
        }
    };
    let draft = match std::env::var("SUPERSONIC_QWEN35_0_8B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_0_8B_DIR unset/missing");
            return None;
        }
    };
    Some((target, draft))
}

fn run_parity_check(
    target: &str,
    draft: &str,
    keep_ratio: &str,
    expected_label: &str,
    cossim_floor: f64,
    require_argmax_match: bool,
    top5_overlap_floor: usize,
) {
    // ~80-token prompt — enough to exercise sparse selection beyond
    // the always_keep_prefix=4 + always_keep_suffix=4 floor (so the
    // 0.50 cell actually drops middle tokens) but short enough that
    // total target-prefill compute stays under ~1 minute per cell.
    // Longer prompts would stress the host's swap/buff-cache state
    // when 4 sequential supersonic invocations run back-to-back, and
    // VMM expert paging starts to thrash with the OS page cache.
    let prompt = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, allowing models to weigh the importance of different parts of the input sequence dynamically. Unlike recurrent networks, transformers can process all tokens in parallel during training, making them highly efficient on modern accelerator hardware. The overall result is";
    let common: Vec<&str> = vec![
        "--backend",
        "hip",
        "--model",
        "qwen3.6-35b-a3b",
        "--model-dir",
        target,
        "--prompt",
        prompt,
        "--max-new-tokens",
        "1",
    ];
    let dense_logits = run_supersonic_capture_logits(&common).expect("dense");
    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir",
        draft,
        "--specprefill-algorithm",
        "cosine",
        "--specprefill-keep-ratio",
        keep_ratio,
    ]);
    let sparse_logits = run_supersonic_capture_logits(&sparse_args).expect("sparse cosine");
    assert_eq!(
        dense_logits.len(),
        sparse_logits.len(),
        "[{expected_label}] logits length mismatch"
    );
    let dense_argmax = argmax(&dense_logits);
    let sparse_argmax = argmax(&sparse_logits);
    let cs = cossim(&dense_logits, &sparse_logits);
    let dense_top5 = top5(&dense_logits);
    let sparse_top5 = top5(&sparse_logits);
    let overlap = dense_top5.intersection(&sparse_top5).count();
    eprintln!(
        "[cross-family cosine parity {expected_label}] cossim={:.6} dense_argmax={} sparse_argmax={} top5_overlap={}/5",
        cs, dense_argmax, sparse_argmax, overlap
    );
    if require_argmax_match {
        assert_eq!(
            dense_argmax, sparse_argmax,
            "[{expected_label}] argmax mismatch (dense={} sparse={})",
            dense_argmax, sparse_argmax
        );
    }
    assert!(
        cs >= cossim_floor,
        "[{expected_label}] cossim {cs} < {cossim_floor}"
    );
    assert!(
        overlap >= top5_overlap_floor,
        "[{expected_label}] top-5 overlap {overlap} < {top5_overlap_floor}"
    );
}

#[test]
fn cosine_qwen36_moe_keep_100_near_identity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // keep=1.00: every prompt position kept, so cache_pos == position
    // for every chain step. Argmax should match dense; cossim is high
    // but not 1.0 because chained vs persistent kernel paths differ
    // in fused-op shape. See module docstring.
    run_parity_check(
        &target,
        &draft,
        "1.00",
        "cosine keep=1.00 near-identity",
        /* cossim_floor */ 0.99,
        /* require_argmax_match */ true,
        /* top5_overlap_floor */ 5,
    );
}

#[test]
fn cosine_qwen36_moe_keep_050_topic_alignment() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // keep=0.50: dropped tokens can shift the argmax. Bar is
    // topic-alignment (top-5 overlap + cossim) rather than exact
    // argmax match.
    run_parity_check(
        &target,
        &draft,
        "0.50",
        "cosine keep=0.50 topic-alignment",
        /* cossim_floor */ 0.65,
        /* require_argmax_match */ false,
        /* top5_overlap_floor */ 3,
    );
}
