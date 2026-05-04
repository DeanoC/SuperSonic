//! End-to-end parity for cosine SpecPrefill + `--kv-fp8` on Qwen3.5-9B.
//!
//! The combo was rejected upfront by `validate_specprefill_flags`
//! before this PR — the gate's stated rationale was the BF16 step-copy
//! fallback in the SpecPrefill *lookahead* decode (PR #177), which
//! never runs for the cosine algorithm. After lifting the gate for
//! cosine only, this test pins:
//!
//! - keep=0.50: cossim ≥ 0.65 vs dense `--kv-fp8` reference, argmax
//!   match, top-5 overlap ≥ 4. Same bars as the non-FP8 cosine parity.
//! - keep=1.00 logit identity: cossim ≥ 0.999, argmax match, top-5
//!   overlap = 5.
//! - keep=1.00 multitoken byte-equal text vs dense `--kv-fp8`.
//!
//! The reference is `dense + --kv-fp8` (NOT plain dense BF16) — both
//! sides quantize their KV cache to FP8, so we measure the SpecPrefill
//! sparse-vs-dense *delta* under FP8, isolating the SpecPrefill effect
//! from the FP8 quantization effect.
//!
//! Skipped silently when:
//!  - HIP backend not compiled.
//!  - SUPERSONIC_QWEN35_9B_DIR or SUPERSONIC_QWEN35_0_8B_DIR unset/missing.
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
            for next in lines.by_ref() {
                let trimmed = next.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with('[') {
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
        anyhow::bail!("generated text not found in stdout");
    }
    Ok((logits, text))
}

fn cossim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
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
    let target = match std::env::var("SUPERSONIC_QWEN35_9B_DIR") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skipped: SUPERSONIC_QWEN35_9B_DIR unset/missing");
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

const PROMPT: &str = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, allowing models to weigh the importance of different parts of the input sequence dynamically. Unlike recurrent networks, transformers can process all tokens in parallel during training, making them highly efficient on modern accelerator hardware. The attention computation involves three projections — query, key, and value — followed by a softmax-normalized dot product that produces a weighted combination of value vectors. Multi-head attention extends this by performing several attention operations in parallel across different learned subspaces, then concatenating and projecting the results. Feed-forward networks between attention layers introduce non-linearity. Residual connections and layer normalization stabilize gradients during training. The overall result is";

fn run_parity_check(
    target: &str,
    draft: &str,
    keep_ratio: &str,
    expected_label: &str,
    cossim_floor: f64,
) {
    let common: Vec<&str> = vec![
        "--backend", "hip",
        "--model", "qwen3.5-9b",
        "--model-dir", target,
        "--prompt", PROMPT,
        "--max-new-tokens", "1",
        "--kv-fp8",
    ];
    let dense_logits = run_supersonic_capture_logits(&common).expect("dense kv-fp8");
    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir", draft,
        "--specprefill-algorithm", "cosine",
        "--specprefill-unload-draft",
        "--specprefill-keep-ratio", keep_ratio,
    ]);
    let sparse_logits =
        run_supersonic_capture_logits(&sparse_args).expect("sparse cosine + kv-fp8");
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
        "[cosine kv-fp8 parity {expected_label}] cossim={:.6} dense_argmax={} sparse_argmax={} top5_overlap={}/5",
        cs, dense_argmax, sparse_argmax, overlap
    );
    assert_eq!(dense_argmax, sparse_argmax, "[{expected_label}] argmax mismatch");
    assert!(cs >= cossim_floor, "[{expected_label}] cossim {cs} < {cossim_floor}");
    assert!(overlap >= 4, "[{expected_label}] top-5 overlap {overlap} < 4");
}

#[test]
fn cosine_kvfp8_qwen35_9b_keep_050_parity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    run_parity_check(&target, &draft, "0.50", "cosine kv-fp8 keep=0.50", 0.65);
}

#[test]
fn cosine_kvfp8_qwen35_9b_keep_100_identity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    run_parity_check(&target, &draft, "1.00", "cosine kv-fp8 keep=1.00 identity", 0.999);
}

#[test]
fn cosine_kvfp8_qwen35_9b_keep_100_multitoken_identity() {
    let (target, draft) = match check_specprefill_env() {
        Some(t) => t,
        None => return,
    };
    let common: Vec<&str> = vec![
        "--backend", "hip",
        "--model", "qwen3.5-9b",
        "--model-dir", &target,
        "--prompt", PROMPT,
        "--max-new-tokens", "8",
        "--kv-fp8",
    ];
    let (_, dense_text) = run_supersonic_capture_logits_and_text(&common).expect("dense kv-fp8");
    let mut sparse_args = common.clone();
    sparse_args.extend_from_slice(&[
        "--specprefill-draft-dir", &draft,
        "--specprefill-algorithm", "cosine",
        "--specprefill-unload-draft",
        "--specprefill-keep-ratio", "1.00",
    ]);
    let (_, sparse_text) =
        run_supersonic_capture_logits_and_text(&sparse_args).expect("sparse cosine + kv-fp8");
    eprintln!("[cosine kv-fp8 multitoken-identity] dense:  {:?}", dense_text);
    eprintln!("[cosine kv-fp8 multitoken-identity] sparse: {:?}", sparse_text);
    assert_eq!(
        dense_text.trim(),
        sparse_text.trim(),
        "[cosine kv-fp8 multitoken-identity] dense and sparse generations differ on \
         max_new_tokens=8 with keep_ratio=1.00 (cosine + --kv-fp8 + kept=[0..T] should be \
         bit-identical to dense + --kv-fp8)"
    );
}
