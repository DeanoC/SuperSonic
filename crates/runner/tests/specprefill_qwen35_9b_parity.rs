//! End-to-end parity: Qwen3.5-9B sparse SpecPrefill (keep=0.50,
//! Qwen3.5-0.8B draft) vs the same prompt's dense prefill. Asserts:
//!  - argmax(last-prefill-step logits) matches (PRIMARY correctness gate
//!    for greedy decoding — what determines the next generated token).
//!  - cossim >= 0.65 (regression backstop). Phase A2 measured cossim on
//!    the 1354-token prompt varying from 0.684 at keep=0.30 to 0.927 at
//!    keep=0.50 against the 9B target. End-to-end through SuperSonic on
//!    the ~225-token test prompt at keep=0.50 we measure ~0.71 — between
//!    those bounds. 0.65 is conservative for any prompt-length ×
//!    keep-ratio combination Phase A2 measured.
//!  - top-5 overlap >= 4/5.
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - SUPERSONIC_QWEN35_9B_DIR or SUPERSONIC_QWEN35_0_8B_DIR not set or path missing.
//!  - SUPERSONIC_SPECPREFILL_PARITY=0.
//!
//! Folds in the identity-parity check from the (deferred) Task 10:
//! at keep_ratio=1.0 the kept-set is [0..T], so the sparse path must
//! produce cossim of 1.000 (the keep=1.00 sub-test additionally asserts
//! cossim >= 0.999 to catch any plumbing drift that wouldn't appear at
//! keep=0.50).

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

/// Run supersonic and return (logits, generated_text). The generated
/// text is the FIRST line after `LAST_LOGITS:` that doesn't start with
/// a known diagnostic prefix — i.e., the line printed by `println!("{text}")`
/// in both the dense and sparse paths. The dense path additionally
/// emits `[tokens] ...` and `[result] ...` diagnostic lines after that
/// which we ignore here.
fn run_supersonic_capture_logits_and_text(
    args: &[&str],
) -> anyhow::Result<(Vec<f32>, String)> {
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

    let mut lines = stdout.lines();
    let mut logits: Option<Vec<f32>> = None;
    let mut text: String = String::new();
    while let Some(line) = lines.next() {
        if let Some(csv) = line.strip_prefix("LAST_LOGITS:") {
            logits = Some(
                csv.trim()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()?,
            );
            // Find the next line that's the actual decoded output (skip
            // empty lines and known diagnostic prefixes).
            for next in lines.by_ref() {
                let trimmed = next.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with('[') {
                    // [tokens] ..., [result] ..., etc. — diagnostics.
                    continue;
                }
                text = next.to_string();
                break;
            }
            break;
        }
    }
    let logits = logits.ok_or_else(|| anyhow::anyhow!("LAST_LOGITS line not found"))?;
    if text.is_empty() {
        anyhow::bail!("generated text not found in stdout (no non-diagnostic line after LAST_LOGITS)");
    }
    Ok((logits, text))
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
    idx.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
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
    let target = match std::env::var("SUPERSONIC_QWEN35_9B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_9B_DIR unset or missing");
            return None;
        }
    };
    let draft = match std::env::var("SUPERSONIC_QWEN35_0_8B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_0_8B_DIR unset or missing");
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
) {
    // Long-prompt regime is where SpecPrefill is meaningful. With a short
    // prompt + forced prefix/suffix bands of 4 each, only a handful of
    // tokens are actually subject to selection — Phase A2 measured top-5
    // overlap = 3/5 on the 222-token short prompt at keep=0.50 (both 4B
    // and 9B targets), versus 5/5 on the 1354-token long prompt. The test
    // bar of >= 4/5 is calibrated for the long-prompt regime, so we use a
    // ~225-token paraphrase of the Phase A2 prompt-1 content.
    let prompt = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, allowing models to weigh the importance of different parts of the input sequence dynamically. Unlike recurrent networks, transformers can process all tokens in parallel during training, making them highly efficient on modern accelerator hardware. The attention computation involves three projections — query, key, and value — followed by a softmax-normalized dot product that produces a weighted combination of value vectors. Multi-head attention extends this by performing several attention operations in parallel across different learned subspaces, then concatenating and projecting the results. Feed-forward networks between attention layers introduce non-linearity. Residual connections and layer normalization stabilize gradients during training. The overall result is";
    let common: Vec<&str> = vec![
        "--backend",
        "hip",
        "--model",
        "qwen3.5-9b",
        "--model-dir",
        target,
        "--prompt",
        prompt,
        "--max-new-tokens",
        "1",
    ];

    let dense_logits = run_supersonic_capture_logits(&common).expect("dense run");

    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir",
        draft,
        "--specprefill-keep-ratio",
        keep_ratio,
    ]);
    let sparse_logits = run_supersonic_capture_logits(&sparse_args).expect("sparse run");

    assert_eq!(
        dense_logits.len(),
        sparse_logits.len(),
        "[{expected_label}] logits length mismatch ({} vs {})",
        dense_logits.len(),
        sparse_logits.len()
    );
    let dense_argmax = argmax(&dense_logits);
    let sparse_argmax = argmax(&sparse_logits);
    let cs = cossim(&dense_logits, &sparse_logits);
    let dense_top5 = top5(&dense_logits);
    let sparse_top5 = top5(&sparse_logits);
    let overlap = dense_top5.intersection(&sparse_top5).count();
    eprintln!(
        "[specprefill parity {expected_label}] cossim={:.6} dense_argmax={} sparse_argmax={} top5_overlap={}/5",
        cs, dense_argmax, sparse_argmax, overlap
    );
    assert_eq!(
        dense_argmax, sparse_argmax,
        "[{expected_label}] argmax mismatch"
    );
    assert!(
        cs >= cossim_floor,
        "[{expected_label}] cossim {cs} < {cossim_floor}"
    );
    assert!(
        overlap >= 4,
        "[{expected_label}] top-5 overlap {overlap} < 4"
    );
}

#[test]
fn specprefill_qwen35_9b_keep_050_parity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // 0.65 cossim floor — see module-level docs for derivation.
    run_parity_check(&target, &draft, "0.50", "keep=0.50", 0.65);
}

#[test]
fn specprefill_qwen35_9b_keep_100_identity() {
    // Folds in the deferred Task 10 identity check -- at keep_ratio=1.0
    // the kept set is [0..T] so the sparse path goes through identical
    // RoPE-indirect + compaction logic with all positions kept. Catches
    // plumbing bugs in Tasks 5-7 that wouldn't surface at keep=0.50.
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    // 0.999 cossim floor for identity — kept_positions = [0..T] should
    // produce numerics identical to dense up to compaction-loop ordering.
    run_parity_check(&target, &draft, "1.00", "keep=1.00 (identity)", 0.999);
}

#[test]
fn specprefill_qwen35_9b_keep_100_multitoken_identity() {
    // Multi-token decode parity at keep_ratio=1.0. The kept set is [0..T]
    // so the sparse path takes the RoPE-indirect kernel with identity
    // positions, which Phase B parity verified is bit-identical to
    // dense. The generated text for max_new_tokens=8 must match exactly.
    //
    // This test catches any regression in the sparse decode loop
    // (e.g., RoPE position vs KV slot conflation) that wouldn't surface
    // on the single-token tests above.
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    let prompt = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, allowing models to weigh the importance of different parts of the input sequence dynamically. Unlike recurrent networks, transformers can process all tokens in parallel during training, making them highly efficient on modern accelerator hardware. The attention computation involves three projections — query, key, and value — followed by a softmax-normalized dot product that produces a weighted combination of value vectors. Multi-head attention extends this by performing several attention operations in parallel across different learned subspaces, then concatenating and projecting the results. Feed-forward networks between attention layers introduce non-linearity. Residual connections and layer normalization stabilize gradients during training. The overall result is";
    let common: Vec<&str> = vec![
        "--backend",
        "hip",
        "--model",
        "qwen3.5-9b",
        "--model-dir",
        &target,
        "--prompt",
        prompt,
        "--max-new-tokens",
        "8",
    ];

    let (_, dense_text) =
        run_supersonic_capture_logits_and_text(&common).expect("dense run");

    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir",
        &draft,
        "--specprefill-keep-ratio",
        "1.00",
    ]);
    let (_, sparse_text) =
        run_supersonic_capture_logits_and_text(&sparse_args).expect("sparse run");

    eprintln!("[multitoken-identity] dense text: {:?}", dense_text);
    eprintln!("[multitoken-identity] sparse text: {:?}", sparse_text);

    assert_eq!(
        dense_text.trim(),
        sparse_text.trim(),
        "[multitoken-identity] dense and sparse generations differ on \
         max_new_tokens=8 with keep_ratio=1.00 (kept=[0..T] so sparse \
         path should be bit-identical to dense)"
    );
}
