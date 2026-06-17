//! Combined-mode parity for Qwen3.6-MoE: `--specprefill-draft-dir`
//! together with `--speculative-decode`. Validates the
//! `PositionPair` plumbing introduced when the engine.rs:165 mutex
//! gate was lifted.
//!
//! Two cells:
//!  - keep=1.00 near-identity: every prompt position kept, so
//!    `position.rope == position.cache` for every step; combined
//!    run should be near-identical to dense+MTP (BF16 floor —
//!    cossim ≥ 0.99). The bar matches `specprefill_qwen36_moe_
//!    cosine_parity::cosine_qwen36_moe_keep_100_near_identity`.
//!  - keep=0.50 topic-alignment: kept tokens land in compact KV
//!    slots while rotating at their original RoPE positions; MTP
//!    drafts at `(rope = abs_pos + k, cache = compact_slot + k)`.
//!    Combined-mode fluency bar (cossim ≥ 0.65, top-5 overlap ≥ 3,
//!    argmax not required).
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - `SUPERSONIC_QWEN36_35B_A3B_DIR` or `SUPERSONIC_QWEN35_0_8B_DIR`
//!    unset/missing.
//!  - `SUPERSONIC_SPECPREFILL_PARITY=0`.

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

fn run_combined_parity_check(
    target: &str,
    draft: &str,
    keep_ratio: &str,
    expected_label: &str,
    cossim_floor: f64,
    require_argmax_match: bool,
    top5_overlap_floor: usize,
) {
    // Same prompt as the SpecPrefill-only parity test: exercises
    // sparse selection beyond the always_keep prefix/suffix floor
    // while keeping per-cell wall time short. MTP adds ~K≤3 draft
    // tokens worth of compute per generation step but we only
    // generate 1 token here, so the comparison stays tight.
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
        "--speculative-decode",
    ];

    // Dense + MTP baseline.
    let dense_logits = run_supersonic_capture_logits(&common).expect("dense+MTP baseline");

    // SpecPrefill + MTP combined.
    let mut combined_args = common.clone();
    combined_args.extend_from_slice(&[
        "--specprefill-draft-dir",
        draft,
        "--specprefill-algorithm",
        "cosine",
        "--specprefill-keep-ratio",
        keep_ratio,
    ]);
    let combined_logits =
        run_supersonic_capture_logits(&combined_args).expect("specprefill+MTP combined");

    assert_eq!(
        dense_logits.len(),
        combined_logits.len(),
        "[{expected_label}] logits length mismatch"
    );

    let dense_argmax = argmax(&dense_logits);
    let combined_argmax = argmax(&combined_logits);
    let cs = cossim(&dense_logits, &combined_logits);
    let dense_top5 = top5(&dense_logits);
    let combined_top5 = top5(&combined_logits);
    let overlap = dense_top5.intersection(&combined_top5).count();

    eprintln!(
        "[mtp+specprefill parity {expected_label}] cossim={:.6} dense_argmax={} combined_argmax={} top5_overlap={}/5",
        cs, dense_argmax, combined_argmax, overlap
    );

    if require_argmax_match {
        assert_eq!(
            dense_argmax, combined_argmax,
            "[{expected_label}] argmax mismatch (dense+MTP={} specprefill+MTP={})",
            dense_argmax, combined_argmax
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
fn cosine_qwen36_moe_mtp_keep_100_near_identity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // keep=1.00: every prompt position kept ⇒
    // PositionPair::dense(p) for every step. The MTP draft chain
    // sees the same base_position whether SpecPrefill is set or
    // not. Combined run should be near bit-equal to dense+MTP at
    // the BF16 floor.
    run_combined_parity_check(
        &target,
        &draft,
        "1.00",
        "mtp+cosine keep=1.00 near-identity",
        /* cossim_floor */ 0.99,
        /* require_argmax_match */ true,
        /* top5_overlap_floor */ 5,
    );
}

#[test]
fn cosine_qwen36_moe_mtp_keep_050_topic_alignment() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // keep=0.50: rope and cache diverge. MTP draft RoPE rotates at
    // base.rope + k = abs_prompt_pos + k while the verify replay
    // writes accepted tokens at base.cache + k = compact_slot + k.
    // Topic-alignment bar matches the SpecPrefill-only keep=0.50
    // test.
    run_combined_parity_check(
        &target,
        &draft,
        "0.50",
        "mtp+cosine keep=0.50 topic-alignment",
        /* cossim_floor */ 0.65,
        /* require_argmax_match */ false,
        /* top5_overlap_floor */ 3,
    );
}
