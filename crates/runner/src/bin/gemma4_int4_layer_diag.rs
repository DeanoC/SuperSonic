//! Per-layer INT4 diagnostic for Gemma 4.
//!
//! Runs the Rust `Gemma4Int4Engine` prefill on a prompt, capturing the
//! post-PLE hidden state at the last prompt-token position after every
//! decoder layer. Reads a matching JSON dump produced by
//! `oracle/gemma4_int4_python_gen.py --emit-hiddens PATH` (faithful Python
//! INT4 reference: bake → dequant → BF16 F.linear via HuggingFace). Reports
//! per-layer cosine similarity + max absolute error so that any layer where
//! the two pipelines diverge shows up as a sharp drop.
//!
//! Intended verdict:
//!   * All 35 layers cos_sim ≥ 0.999 → INT4 pipeline is clean; the BF16/INT4
//!     quality gap is dominated by reduction-order noise (same class
//!     documented in `feedback_gpu_oracle`). Ship current bake quality, look
//!     elsewhere for improvement (calibration, group_size, non-APU bake).
//!   * One or more layers cos_sim drops sharply (e.g. 0.9999 → 0.95) →
//!     Rust-side INT4 pipeline bug localized to that layer.
//!
//! Usage:
//!   python3 oracle/gemma4_int4_python_gen.py --model-dir <ckpt> \
//!     --prompt "Hello" --emit-hiddens /tmp/gemma4_py_int4.json
//!   cargo run --release --bin gemma4_int4_layer_diag -- \
//!     --model-dir <ckpt> --prompt "Hello" \
//!     --python-json /tmp/gemma4_py_int4.json

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use clap::Parser;
use half::bf16;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[path = "../gemma4_int4_engine.rs"]
mod gemma4_int4_engine;
#[path = "../gemma4_engine.rs"]
mod gemma4_engine;
use gemma4_engine::Gemma4Engine;
use gemma4_int4_engine::{int4_bake_ok, Gemma4Int4Engine};

#[derive(Parser, Debug)]
#[command(about = "Compare Rust INT4 per-layer hidden states against a Python INT4 reference")]
struct Cli {
    /// Path to the Gemma 4 checkpoint directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// Prompt — must be the same string passed to the Python emit-hiddens run.
    #[arg(long, default_value = "Hello")]
    prompt: String,
    /// JSON file written by `oracle/gemma4_int4_python_gen.py --emit-hiddens`.
    #[arg(long)]
    python_json: PathBuf,
    /// Weight prefix inside the bake (matches registry default).
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,
    /// Flag layers whose cos_sim drops below this threshold. Default 0.99
    /// matches the "is there a bug" question; drops below 0.999 are worth
    /// eyeballing but are usually reduction-order drift.
    #[arg(long, default_value_t = 0.99)]
    bug_threshold: f32,
    /// Secondary (softer) threshold — layers below this but above bug_threshold
    /// are reported as "drift" rather than "BUG".
    #[arg(long, default_value_t = 0.999)]
    drift_threshold: f32,
    /// Use the BF16 engine instead of the INT4 engine. Pairs with a Python
    /// run of `gemma4_oracle.py` (or `gemma4_int4_python_gen.py` without
    /// applying the bake) to isolate INT4-specific drift from pipeline-wide
    /// noise. When set, `--model-dir` must point at an unquantized
    /// checkpoint; INT4 bake presence is ignored.
    #[arg(long, default_value_t = false)]
    bf16: bool,
    /// If set, also dump Rust's per-layer captures to this JSON path using
    /// the same schema as the Python emit-hiddens output. Lets you do
    /// offline cross-comparisons (e.g. R_INT4 vs R_BF16).
    #[arg(long, default_value = None)]
    dump_rust: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct PythonDump {
    prompt_token_ids: Vec<u32>,
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    prefill_per_layer_hidden: Vec<String>,
    prefill_per_layer_hidden_shape: Vec<usize>,
    final_norm_hidden: String,
    final_norm_hidden_shape: Vec<usize>,
    logits: Vec<f32>,
}

fn decode_bf16_hidden(b64: &str, hidden_size: usize) -> Result<Vec<f32>> {
    let bytes = B64.decode(b64).context("base64 decode hidden")?;
    if bytes.len() != hidden_size * 2 {
        bail!(
            "hidden byte length {} != expected {}",
            bytes.len(),
            hidden_size * 2
        );
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn l2_norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

fn dump_rust_captures(
    out_path: &std::path::Path,
    prompt_token_ids: &[u32],
    hidden_size: usize,
    vocab_size: usize,
    per_layer: &[Vec<f32>],
    final_norm: &[f32],
    logits: &[f32],
) -> Result<()> {
    use serde_json::json;
    fn f32_to_bf16_b64(xs: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(xs.len() * 2);
        for &v in xs {
            bytes.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
        B64.encode(bytes)
    }
    let per_layer_b64: Vec<String> = per_layer.iter().map(|row| f32_to_bf16_b64(row)).collect();
    let payload = json!({
        "prompt_token_ids": prompt_token_ids,
        "hidden_size": hidden_size,
        "num_layers": per_layer.len(),
        "vocab_size": vocab_size,
        "prefill_per_layer_hidden": per_layer_b64,
        "prefill_per_layer_hidden_shape": [1, 1, hidden_size],
        "final_norm_hidden": f32_to_bf16_b64(final_norm),
        "final_norm_hidden_shape": [1, 1, hidden_size],
        "logits": logits,
    });
    std::fs::write(out_path, serde_json::to_vec(&payload)?)
        .with_context(|| format!("write {}", out_path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !cli.bf16 && !int4_bake_ok(&cli.model_dir) {
        bail!(
            "No INT4 bake at {}/.supersonic/. Run:\n  \
             python oracle/bake_int4_gemma4.py --model-dir {}",
            cli.model_dir.display(),
            cli.model_dir.display()
        );
    }

    let py_bytes = fs::read(&cli.python_json)
        .with_context(|| format!("read {}", cli.python_json.display()))?;
    let py: PythonDump = serde_json::from_slice(&py_bytes)
        .with_context(|| format!("parse JSON {}", cli.python_json.display()))?;
    eprintln!(
        "[py] prompt_tokens={} num_layers={} hidden={} vocab={}",
        py.prompt_token_ids.len(),
        py.num_layers,
        py.hidden_size,
        py.vocab_size,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
    let encoded = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize failed: {e}"))?;
    let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();
    if prompt_tokens.is_empty() {
        bail!("tokenizer produced zero tokens for prompt {:?}", cli.prompt);
    }
    eprintln!(
        "[rust] prompt_tokens={} ids={:?}",
        prompt_tokens.len(),
        prompt_tokens
    );

    if prompt_tokens != py.prompt_token_ids {
        bail!(
            "prompt-token mismatch Rust vs Python:\n  rust   = {:?}\n  python = {:?}\n\
             Cannot compare hidden states with different tokenizations.",
            prompt_tokens,
            py.prompt_token_ids,
        );
    }

    let max_t = prompt_tokens.len() + 4;
    let load_start = std::time::Instant::now();
    let (logits, caps) = if cli.bf16 {
        // Gemma4Engine::load wants &'static str for weight_prefix; leak the
        // clap-parsed String so the borrow lifts to 'static. Only called once
        // per run, so the small leak is fine for this diagnostic.
        let static_prefix: &'static str = Box::leak(cli.weight_prefix.clone().into_boxed_str());
        let mut engine = Gemma4Engine::load(&cli.model_dir, static_prefix, max_t, 0)
            .context("load Gemma4Engine (BF16)")?;
        eprintln!(
            "[rust] BF16 engine loaded in {} ms",
            load_start.elapsed().as_millis()
        );
        let prefill_start = std::time::Instant::now();
        let (logits, caps) = engine
            .prefill_with_capture(&prompt_tokens)
            .context("Gemma4Engine::prefill_with_capture")?;
        eprintln!(
            "[rust] BF16 prefill+capture ({} tok) in {} ms",
            prompt_tokens.len(),
            prefill_start.elapsed().as_millis()
        );
        (logits, (caps.per_layer_hidden, caps.final_norm_hidden))
    } else {
        let mut engine = Gemma4Int4Engine::load(&cli.model_dir, &cli.weight_prefix, max_t, 0)
            .context("load Gemma4Int4Engine")?;
        eprintln!(
            "[rust] INT4 engine loaded in {} ms",
            load_start.elapsed().as_millis()
        );
        let prefill_start = std::time::Instant::now();
        let (logits, caps) = engine
            .prefill_with_capture(&prompt_tokens)
            .context("Gemma4Int4Engine::prefill_with_capture")?;
        eprintln!(
            "[rust] INT4 prefill+capture ({} tok) in {} ms",
            prompt_tokens.len(),
            prefill_start.elapsed().as_millis()
        );
        (logits, (caps.per_layer_hidden, caps.final_norm_hidden))
    };
    let (rust_per_layer, rust_final_norm) = caps;

    if rust_per_layer.len() != py.num_layers {
        bail!(
            "layer count mismatch: rust={} python={}",
            rust_per_layer.len(),
            py.num_layers,
        );
    }
    if logits.len() != py.vocab_size {
        bail!(
            "vocab mismatch: rust={} python={}",
            logits.len(),
            py.vocab_size,
        );
    }

    // Decode all Python hidden states up front — easier to reason about errors
    // before we start printing per-layer deltas.
    let py_per_layer: Vec<Vec<f32>> = py
        .prefill_per_layer_hidden
        .iter()
        .enumerate()
        .map(|(i, b64)| {
            decode_bf16_hidden(b64, py.hidden_size)
                .with_context(|| format!("decode python layer {i}"))
        })
        .collect::<Result<_>>()?;
    let py_final_norm = decode_bf16_hidden(&py.final_norm_hidden, py.hidden_size)
        .context("decode python final_norm_hidden")?;

    // Verify layer shape tag matches our capture length.
    let expected_elems: usize = py.prefill_per_layer_hidden_shape.iter().product();
    if expected_elems != py.hidden_size {
        bail!(
            "unexpected python layer shape {:?} (expected product == hidden_size {})",
            py.prefill_per_layer_hidden_shape,
            py.hidden_size,
        );
    }
    let expected_elems_final: usize = py.final_norm_hidden_shape.iter().product();
    if expected_elems_final != py.hidden_size {
        bail!(
            "unexpected python final_norm shape {:?} (expected product == hidden_size {})",
            py.final_norm_hidden_shape,
            py.hidden_size,
        );
    }

    if let Some(out_path) = &cli.dump_rust {
        dump_rust_captures(
            out_path, &prompt_tokens, py.hidden_size, py.vocab_size,
            &rust_per_layer, &rust_final_norm, &logits,
        )?;
        eprintln!("[rust] dumped captures to {}", out_path.display());
    }

    println!();
    println!("[per-layer comparison] Rust INT4 engine vs Python INT4 reference");
    println!(
        "  bug_threshold = {:.4}, drift_threshold = {:.4}",
        cli.bug_threshold, cli.drift_threshold
    );
    println!();
    println!(
        "{:>5} | {:>11} | {:>10} | {:>10} | {:>10} | {}",
        "layer", "cos_sim", "max_abs", "||rust||", "||py||", "verdict"
    );
    println!("{}", "-".repeat(72));

    let mut bug_layers: Vec<usize> = Vec::new();
    let mut drift_layers: Vec<usize> = Vec::new();
    let mut min_cos = f32::INFINITY;
    let mut min_cos_layer = 0usize;

    for (i, (rust_h, py_h)) in rust_per_layer
        .iter()
        .zip(py_per_layer.iter())
        .enumerate()
    {
        let cos = cosine_similarity(rust_h, py_h);
        let max_abs = max_abs_delta(rust_h, py_h);
        let nr = l2_norm(rust_h);
        let np = l2_norm(py_h);
        let verdict = if cos < cli.bug_threshold {
            bug_layers.push(i);
            "BUG"
        } else if cos < cli.drift_threshold {
            drift_layers.push(i);
            "drift"
        } else {
            "ok"
        };
        if cos < min_cos {
            min_cos = cos;
            min_cos_layer = i;
        }
        println!(
            "{:>5} | {:>11.6} | {:>10.4} | {:>10.2} | {:>10.2} | {}",
            i, cos, max_abs, nr, np, verdict
        );
    }

    // Post-final-norm hidden (input to lm-head).
    let cos_fn = cosine_similarity(&rust_final_norm, &py_final_norm);
    let max_abs_fn = max_abs_delta(&rust_final_norm, &py_final_norm);
    println!();
    println!(
        "[final norm hidden] cos_sim={:.6} max_abs={:.4}",
        cos_fn, max_abs_fn
    );

    // Softcapped logits.
    let cos_logits = cosine_similarity(&logits, &py.logits);
    let max_abs_logits = max_abs_delta(&logits, &py.logits);
    let rust_argmax = argmax(&logits);
    let py_argmax = argmax(&py.logits);
    let rust_top5 = top_k_indices(&logits, 5);
    let py_top5 = top_k_indices(&py.logits, 5);
    let top5_overlap = rust_top5.iter().filter(|i| py_top5.contains(i)).count();
    println!(
        "[logits] cos_sim={:.6} max_abs={:.4} argmax: rust={} python={} ({}) top5_overlap={}/5",
        cos_logits,
        max_abs_logits,
        rust_argmax,
        py_argmax,
        if rust_argmax == py_argmax { "MATCH" } else { "MISS" },
        top5_overlap,
    );

    println!();
    println!(
        "[summary] min_cos_sim={:.6} at layer {} | bug_layers={} drift_layers={}",
        min_cos,
        min_cos_layer,
        bug_layers.len(),
        drift_layers.len()
    );
    if !bug_layers.is_empty() {
        println!("[verdict] BUG suspected — layers with cos_sim < {:.4}: {:?}",
            cli.bug_threshold, bug_layers);
        // Still exit 0 — this binary is a diagnostic; the operator reads the
        // table and decides. Non-zero exit would make shell one-liners awkward.
    } else if !drift_layers.is_empty() {
        println!("[verdict] CLEAN with drift — {} layers between {:.4} and {:.4}",
            drift_layers.len(), cli.bug_threshold, cli.drift_threshold);
    } else {
        println!("[verdict] CLEAN — every layer above drift threshold {:.4}",
            cli.drift_threshold);
    }

    Ok(())
}
