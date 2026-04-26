//! Gemma 4 multi-prompt correctness harness.
//!
//! Reads a JSONL prompt corpus and an oracle sidecar JSON (emitted by
//! `oracle/gemma4_oracle.py --prompts-file ... --output-file ...`) and runs
//! each prompt through the same `Gemma4Engine` path that the main
//! `supersonic --model gemma4-e2b` CLI uses. Compares the generated token IDs
//! for an exact greedy match against the oracle's expected IDs.
//!
//! Acceptance: every non-`expect_error` entry matches exactly; every
//! `expect_error` entry triggers a clean engine-side error (no panic).
//!
//! Usage:
//!   python3 oracle/gemma4_oracle.py --model-dir <ckpt> \
//!       --prompts-file tests/gemma4/corpus.jsonl \
//!       --output-file /tmp/gemma4_corpus_expected.json
//!   cargo run --release --bin gemma4_corpus_test -- \
//!       --model-dir <ckpt> \
//!       --corpus-jsonl tests/gemma4/corpus.jsonl \
//!       --expected-json /tmp/gemma4_corpus_expected.json

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde::Deserialize;

#[path = "../gemma4_engine.rs"]
mod gemma4_engine;
#[path = "../gemma4_int4_engine.rs"]
mod gemma4_int4_engine;
use gemma4_engine::Gemma4Engine;
use gemma4_int4_engine::Gemma4Int4Engine;

/// BF16 (default megakernel) or INT4 (GPTQ bake + primitive chain). Mirrors the
/// dispatcher in `crates/runner/src/main.rs` but kept local so this harness
/// doesn't need to pull in the full runner crate.
enum Runtime {
    Bf16(Gemma4Engine),
    Int4(Gemma4Int4Engine),
}

impl Runtime {
    fn prefill(&mut self, prompt_token_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.prefill(prompt_token_ids),
            Self::Int4(e) => e.prefill(prompt_token_ids),
        }
    }

    fn decode_step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.decode_step(token, pos),
            Self::Int4(e) => e.decode_step(token, pos),
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Gemma 4 multi-prompt exact-match corpus regression")]
struct Cli {
    /// Local Gemma 4 snapshot directory (config.json + safetensors + tokenizer.json).
    #[arg(long)]
    model_dir: PathBuf,
    /// JSONL corpus file with one {"name","prompt","max_new_tokens"[,"expect_error"]} per line.
    #[arg(long)]
    corpus_jsonl: PathBuf,
    /// Oracle sidecar JSON produced by `gemma4_oracle.py --prompts-file ...`.
    #[arg(long)]
    expected_json: PathBuf,
    /// Weight prefix inside the checkpoint (typically `model.language_model`).
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,
    /// HIP device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Stop at the first failing entry instead of running the full corpus.
    #[arg(long, default_value_t = false)]
    fail_fast: bool,
    /// Run with INT4 GPTQ bake instead of BF16 weights. Requires a prior
    /// `python oracle/bake_int4_gemma4.py --model-dir <dir>` to have produced
    /// `.supersonic/v1-int4-gptq/`.
    #[arg(long, default_value_t = false)]
    int4: bool,
}

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    max_new_tokens: Option<usize>,
    #[serde(default)]
    expect_error: bool,
}

#[derive(Debug, Deserialize)]
struct ExpectedPayload {
    entries: Vec<ExpectedEntry>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntry {
    name: String,
    #[serde(default)]
    prompt_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    expected_generated_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    expect_error: bool,
    max_new_tokens: usize,
}

fn load_corpus(path: &PathBuf) -> Result<Vec<CorpusEntry>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (lineno, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry: CorpusEntry = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "{}:{} invalid JSON: {}",
                path.display(),
                lineno + 1,
                trimmed
            )
        })?;
        out.push(entry);
    }
    Ok(out)
}

fn entry_name(entry: &CorpusEntry, idx: usize) -> String {
    entry.name.clone().unwrap_or_else(|| format!("entry_{idx}"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let weight_prefix: &'static str = Box::leak(cli.weight_prefix.clone().into_boxed_str());

    let corpus = load_corpus(&cli.corpus_jsonl)?;
    if corpus.is_empty() {
        bail!("corpus {} has no entries", cli.corpus_jsonl.display());
    }

    let expected_raw = std::fs::read_to_string(&cli.expected_json)
        .with_context(|| format!("read {}", cli.expected_json.display()))?;
    let expected: ExpectedPayload = serde_json::from_str(&expected_raw)
        .with_context(|| format!("parse {}", cli.expected_json.display()))?;
    let expected_by_name: HashMap<String, ExpectedEntry> = expected
        .entries
        .into_iter()
        .map(|e| (e.name.clone(), e))
        .collect();

    // Tokenizer from the checkpoint — same path the main CLI uses.
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;

    // Pre-tokenize every live prompt so we can size max_t once.
    struct Prepared {
        name: String,
        prompt: String,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        expect_error: bool,
        expected_gen: Option<Vec<u32>>,
    }
    let mut prepared: Vec<Prepared> = Vec::with_capacity(corpus.len());
    let mut max_needed_t: usize = 8; // minimum floor
    for (idx, entry) in corpus.iter().enumerate() {
        let name = entry_name(entry, idx);
        let prompt = entry.prompt.clone().unwrap_or_default();
        let expected = expected_by_name.get(&name).ok_or_else(|| {
            anyhow!(
                "corpus entry '{name}' has no matching oracle record in {}",
                cli.expected_json.display()
            )
        })?;
        let max_new = entry
            .max_new_tokens
            .unwrap_or(expected.max_new_tokens)
            .max(1);

        let (prompt_ids, expected_gen) = if entry.expect_error {
            (Vec::new(), None)
        } else {
            let encoding = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("tokenize '{name}': {e}"))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            // Cross-check Rust tokenizer vs oracle tokenizer — divergence here
            // means downstream exact-match is impossible, so catch it early.
            if let Some(ref oracle_ids) = expected.prompt_token_ids {
                if &ids != oracle_ids {
                    bail!("tokenizer mismatch on '{name}': rust={ids:?} oracle={oracle_ids:?}");
                }
            }
            let want = expected
                .expected_generated_token_ids
                .clone()
                .ok_or_else(|| {
                    anyhow!("oracle missing expected_generated_token_ids for '{name}'")
                })?;
            (ids, Some(want))
        };

        if !entry.expect_error {
            let total = prompt_ids.len() + max_new;
            if total > max_needed_t {
                max_needed_t = total;
            }
        }

        prepared.push(Prepared {
            name,
            prompt,
            prompt_ids,
            max_new_tokens: max_new,
            expect_error: entry.expect_error,
            expected_gen,
        });
    }

    eprintln!(
        "[corpus] {} entries, max_t={} (model_dir={})",
        prepared.len(),
        max_needed_t,
        cli.model_dir.display()
    );

    // Load the engine once — weight load dominates cost (~5-10s on iGPU).
    let t0 = Instant::now();
    let mut engine: Runtime = if cli.int4 {
        if !gemma4_int4_engine::int4_bake_ok(&cli.model_dir) {
            bail!(
                "No INT4 bake at {}. Run: python oracle/bake_int4_gemma4.py --model-dir {}",
                gemma4_int4_engine::int4_bake_dir(&cli.model_dir).display(),
                cli.model_dir.display()
            );
        }
        eprintln!("[engine] loading INT4 GPTQ bake");
        Runtime::Int4(Gemma4Int4Engine::load(
            &cli.model_dir,
            weight_prefix,
            max_needed_t,
            cli.device,
        )?)
    } else {
        Runtime::Bf16(Gemma4Engine::load(
            &cli.model_dir,
            weight_prefix,
            max_needed_t,
            cli.device,
        )?)
    };
    eprintln!("[engine] loaded in {:.0}ms", t0.elapsed().as_millis());

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut err_pass = 0usize;
    let mut err_fail = 0usize;
    let mut failing_names: Vec<String> = Vec::new();

    println!(
        "{:<28} {:>6} {:>6} {:>10} {:<6} {}",
        "NAME", "P_TOK", "N_NEW", "MS/TOK", "OK", "NOTE"
    );

    for prep in &prepared {
        if prep.expect_error {
            // Run the same tokenize+prefill path the CLI uses; any Err is a
            // clean failure — a panic or exit(1) is what we explicitly guard
            // against by using the `Gemma4Engine::prefill` API.
            let encoding = tokenizer.encode(prep.prompt.as_str(), true);
            let note;
            let ok;
            match encoding {
                Err(e) => {
                    note = format!("tokenize-err: {e}");
                    ok = true;
                }
                Ok(enc) => {
                    let ids: Vec<u32> = enc.get_ids().to_vec();
                    if ids.is_empty() {
                        note = "tokenize-empty".to_string();
                        ok = true;
                    } else {
                        // Non-empty — try the engine and see if it fails.
                        match engine.prefill(&ids) {
                            Err(e) => {
                                note = format!("engine-err: {e}");
                                ok = true;
                            }
                            Ok(_) => {
                                note = format!(
                                    "ran ok (tokenized to {} ids); reclassify entry",
                                    ids.len()
                                );
                                ok = false;
                            }
                        }
                    }
                }
            }
            if ok {
                err_pass += 1;
            } else {
                err_fail += 1;
                failing_names.push(prep.name.clone());
            }
            println!(
                "{:<28} {:>6} {:>6} {:>10} {:<6} {}",
                prep.name,
                "-",
                "-",
                "-",
                if ok { "PASS" } else { "FAIL" },
                note
            );
            if !ok && cli.fail_fast {
                bail!("fail-fast: '{}' failed", prep.name);
            }
            continue;
        }

        let prompt_len = prep.prompt_ids.len();
        let step_start = Instant::now();
        let prefill_logits = engine.prefill(&prep.prompt_ids)?;
        let mut next_token = Gemma4Engine::greedy_sample(&prefill_logits);

        let mut generated: Vec<u32> = Vec::with_capacity(prep.max_new_tokens);
        // Match oracle semantics: exactly max_new_tokens steps, no EOS stop —
        // the oracle generates fixed length so we must too for exact-match.
        for step in 0..prep.max_new_tokens {
            let pos = prompt_len + step;
            let logits = engine.decode_step(next_token, pos)?;
            let sampled = Gemma4Engine::greedy_sample(&logits);
            generated.push(next_token);
            next_token = sampled;
        }
        let elapsed_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        let per_tok = if prep.max_new_tokens == 0 {
            0.0
        } else {
            elapsed_ms / prep.max_new_tokens as f64
        };

        let expected = prep.expected_gen.as_ref().unwrap();
        let ok = &generated == expected;
        let note = if ok {
            String::new()
        } else {
            // First divergence position
            let first_diff = generated
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(generated.len().min(expected.len()));
            format!(
                "MISMATCH @{first_diff}: rust={:?} expected={:?}",
                generated, expected
            )
        };
        if ok {
            pass += 1;
        } else {
            fail += 1;
            failing_names.push(prep.name.clone());
        }
        println!(
            "{:<28} {:>6} {:>6} {:>10.1} {:<6} {}",
            prep.name,
            prompt_len,
            prep.max_new_tokens,
            per_tok,
            if ok { "PASS" } else { "FAIL" },
            note
        );
        if !ok && cli.fail_fast {
            bail!("fail-fast: '{}' failed", prep.name);
        }
    }

    println!();
    println!(
        "[summary] token-match pass={pass} fail={fail} | error-path pass={err_pass} fail={err_fail}"
    );
    if fail + err_fail > 0 {
        bail!(
            "corpus FAIL ({} mismatches): {}",
            fail + err_fail,
            failing_names.join(", ")
        );
    }
    println!("[summary] corpus PASS");
    Ok(())
}
