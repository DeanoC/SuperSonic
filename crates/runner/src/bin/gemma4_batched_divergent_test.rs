//! Divergent-position batched-decode correctness harness.
//!
//! Verifies the batched Gemma 4 megakernels (BF16 `g4::persistent_decode_batch`
//! and INT4 `g4::persistent_decode_batch_int4`) correctly read per-sequence
//! `seqlen_offset[b]` from `Gemma4BatchSeqDesc` by staggering sequences at
//! different positions inside a single kernel launch.
//!
//! Strategy (works identically for BF16 and INT4): after prefill + replication,
//! all B sequences share the same KV cache at position = `prompt_len`. We then
//! advance seq 0 one step ahead using the single-seq `decode_step` API (which
//! only writes to seq 0's cache — both engines' `decode_step` goes through the
//! single-seq megakernel that reads KV pointers from seq 0's descriptor). After
//! that, one `decode_step_batch` launch runs with positions
//!   [len+1, len, len, ..., len]
//! — seq 0 is one token ahead of the others. The batched kernel must read
//! `seqlen_offset[0] = len+1` and `seqlen_offset[b>=1] = len` from batch_descs
//! to produce the right outputs.
//!
//! Reference trajectory is a fresh single-seq engine running the same prompt
//! for N decode steps. We compare greedy-sampled tokens against the reference.
//! Divergent-position correctness holds iff:
//!   - seq 0's t=1 batched output matches reference token at prompt_len+1
//!   - seqs 1..B's t=0 batched outputs match reference token at prompt_len
//!
//! Running multiple batched steps amplifies any bug: by step K, seq 0 has
//! position len+1+K while the others are at len+K.
//!
//! Usage:
//!   cargo run --release --bin gemma4_batched_divergent_test -- \
//!       --model-dir <ckpt> [--int4] [--batch-size 4] [--max-new-tokens 6]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use clap::Parser;

#[path = "../gemma4_engine.rs"]
mod gemma4_engine;
#[path = "../gemma4_int4_engine.rs"]
mod gemma4_int4_engine;
use gemma4_engine::Gemma4Engine;
use gemma4_int4_engine::Gemma4Int4Engine;

#[derive(Parser, Debug)]
#[command(about = "Gemma 4 batched-decode divergent-position correctness test")]
struct Cli {
    /// Local Gemma 4 snapshot directory (config.json + safetensors + tokenizer.json).
    #[arg(long)]
    model_dir: PathBuf,
    /// Weight prefix inside the checkpoint.
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,
    /// HIP device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Run the INT4 GPTQ engine instead of BF16.
    #[arg(long)]
    int4: bool,
    /// Batch size (>= 2).
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    /// Number of decode steps to run after the stagger (exercises per-seq
    /// position arithmetic over multiple steps).
    #[arg(long, default_value_t = 6)]
    max_new_tokens: usize,
    /// Prompt text. Tokenized with the snapshot's tokenizer.json.
    #[arg(long, default_value = "Hello, world")]
    prompt: String,
}

enum Runtime {
    Bf16(Gemma4Engine),
    Int4(Gemma4Int4Engine),
}

impl Runtime {
    fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.prefill(prompt),
            Self::Int4(e) => e.prefill(prompt),
        }
    }
    fn decode_step(&mut self, tok: u32, pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Bf16(e) => e.decode_step(tok, pos),
            Self::Int4(e) => e.decode_step(tok, pos),
        }
    }
    fn decode_step_batch(&mut self, toks: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Bf16(e) => e.decode_step_batch(toks, positions),
            Self::Int4(e) => e.decode_step_batch(toks, positions),
        }
    }
    fn replicate_seq0_kv(&mut self) -> Result<()> {
        match self {
            Self::Bf16(e) => e.replicate_seq0_kv(),
            Self::Int4(e) => e.replicate_seq0_kv(),
        }
    }
    fn batch_size(&self) -> usize {
        match self {
            Self::Bf16(e) => e.batch_size(),
            Self::Int4(e) => e.batch_size(),
        }
    }
}

fn greedy_sample(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best as u32
}

fn load_engine(cli: &Cli, weight_prefix: &'static str, batch_size: usize) -> Result<Runtime> {
    // `context_tokens` must fit prompt + max_new_tokens + stagger step.
    let context_tokens = 128usize;
    if cli.int4 {
        Ok(Runtime::Int4(Gemma4Int4Engine::load_with_batch(
            &cli.model_dir,
            weight_prefix,
            context_tokens,
            cli.device,
            batch_size,
        )?))
    } else {
        Ok(Runtime::Bf16(Gemma4Engine::load_with_batch(
            &cli.model_dir,
            weight_prefix,
            context_tokens,
            cli.device,
            batch_size,
        )?))
    }
}

fn run_reference_trajectory(
    cli: &Cli,
    weight_prefix: &'static str,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<u32>> {
    // Fresh B=1 engine. Prefill + `steps` single-seq decode steps = reference.
    let mut engine = load_engine(cli, weight_prefix, 1)?;
    let mut out = Vec::with_capacity(steps);
    let last = engine.prefill(prompt)?;
    let mut tok = greedy_sample(&last);
    out.push(tok);
    for k in 1..steps {
        let pos = prompt.len() + k - 1;
        let logits = engine.decode_step(tok, pos)?;
        tok = greedy_sample(&logits);
        out.push(tok);
    }
    Ok(out)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.batch_size < 2 {
        bail!("--batch-size must be >= 2 (this test exercises divergent positions)");
    }
    let steps = cli.max_new_tokens;
    if steps < 2 {
        bail!("--max-new-tokens must be >= 2 (stagger needs one reference step)");
    }

    // Engines want a `&'static str` weight prefix; leak the CLI-owned String.
    let weight_prefix: &'static str = Box::leak(cli.weight_prefix.clone().into_boxed_str());

    // Tokenize the prompt exactly as the supersonic CLI does.
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt.len();
    eprintln!(
        "[test] prompt_len={} tokens={:?} int4={} B={}",
        prompt_len, prompt, cli.int4, cli.batch_size
    );

    // --- Single-seq reference trajectory (N tokens from this prompt).
    let t0 = Instant::now();
    let reference = run_reference_trajectory(&cli, weight_prefix, &prompt, steps + 1)?;
    eprintln!(
        "[ref] {} single-seq steps in {:.0}ms → tokens {:?}",
        reference.len(),
        t0.elapsed().as_millis(),
        reference
    );

    // --- Batched run with a stagger: seq 0 is ALWAYS one step ahead of seqs 1..B.
    let mut engine = load_engine(&cli, weight_prefix, cli.batch_size)?;
    assert_eq!(engine.batch_size(), cli.batch_size);
    let _ = engine.prefill(&prompt)?;
    engine.replicate_seq0_kv()?;

    // Stagger: advance seq 0 by one single-seq decode step.
    // seq 0 writes KV at position=prompt_len; all other seqs' caches stay put.
    let logits0 = engine.decode_step(reference[0], prompt_len)?;
    let seq0_first = greedy_sample(&logits0);
    // seq 0's first batched-step input token is reference[1] (we just predicted
    // that from the stagger). If the stagger matches ref, reference[1] ==
    // seq0_first; assert that explicitly.
    if seq0_first != reference[1] {
        bail!(
            "stagger failed: single-seq decode_step produced {} but ref[1]={}",
            seq0_first,
            reference[1]
        );
    }

    // Per-seq rolling state for the batched trajectory.
    // seq 0 is "ahead by 1" throughout — at batched step k, seq 0's position is
    // prompt_len + 1 + k while seqs 1..B are at prompt_len + k.
    let b = cli.batch_size;
    let mut cur_tokens: Vec<u32> = vec![reference[0]; b]; // seqs 1..B start at ref[0]
    cur_tokens[0] = reference[1]; // seq 0 is at step 1

    // Record each seq's greedy trajectory across `steps` batched iterations.
    let mut seq_trajectories: Vec<Vec<u32>> = vec![Vec::with_capacity(steps); b];

    for k in 0..steps {
        let mut positions: Vec<usize> = vec![prompt_len + k; b];
        positions[0] = prompt_len + 1 + k;
        if *positions.iter().max().unwrap() + 1 >= 128 {
            bail!("context overflow; bump context_tokens");
        }
        let outs = engine.decode_step_batch(&cur_tokens, &positions)?;
        for seq in 0..b {
            let tok = greedy_sample(&outs[seq]);
            seq_trajectories[seq].push(tok);
            cur_tokens[seq] = tok;
        }
    }

    // --- Correctness checks.
    // Reference trajectory indices:
    //   seq 0 at batched step k expected: reference[2 + k]  (seq 0 is step-1 ahead)
    //   seq b>=1 at batched step k expected: reference[1 + k]
    let mut failures: Vec<String> = Vec::new();
    for seq in 0..b {
        let offset = if seq == 0 { 2 } else { 1 };
        for k in 0..steps {
            let expected_ix = offset + k;
            if expected_ix >= reference.len() {
                // Ran out of reference tokens; stop comparing this seq.
                break;
            }
            let expected = reference[expected_ix];
            let actual = seq_trajectories[seq][k];
            if actual != expected {
                failures.push(format!(
                    "seq {seq} step {k}: pos={} expected {} got {}",
                    if seq == 0 {
                        prompt_len + 1 + k
                    } else {
                        prompt_len + k
                    },
                    expected,
                    actual
                ));
            }
        }
    }

    eprintln!("[batched] per-seq trajectories:");
    for (seq, traj) in seq_trajectories.iter().enumerate() {
        let pos_start = if seq == 0 { prompt_len + 1 } else { prompt_len };
        eprintln!("  seq {seq} (pos start={pos_start}): {traj:?}");
    }

    if failures.is_empty() {
        eprintln!(
            "[PASS] B={} int4={}: all {} seqs matched single-seq reference across {} batched steps \
             with divergent positions (seq 0 always 1 ahead of seqs 1..{}).",
            cli.batch_size, cli.int4, cli.batch_size, steps, cli.batch_size - 1
        );
        Ok(())
    } else {
        for f in &failures {
            eprintln!("  FAIL: {f}");
        }
        bail!(
            "divergent-position test failed with {} mismatches",
            failures.len()
        );
    }
}
