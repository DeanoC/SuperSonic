use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct OracleOutput {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// Number of prompt tokens. Optional — only the Gemma 4 oracle emits this;
    /// the Phi-4 / Qwen oracles infer prompt length from `prompt_token_ids`.
    #[serde(default)]
    pub prompt_tokens: Option<usize>,
    pub generated_tokens: usize,
    pub prefill_logits: Vec<f32>,
    pub decode_logits: Vec<Vec<f32>>,
    pub generated_token_ids: Vec<u32>,
    // State export (only present with --emit-state)
    pub prefill_hidden: Option<String>,      // base64
    pub prefill_hidden_shape: Option<Vec<usize>>,
    pub layer_attn_residual_states: Option<Vec<String>>, // base64 BF16 [1,1,hidden] per layer
    pub layer_post_attn_norm_states: Option<Vec<String>>, // base64 BF16 [1,1,hidden] per layer
    pub layer_mlp_outputs: Option<Vec<String>>, // base64 BF16 [1,1,hidden] per layer
    pub layer_hidden_states: Option<Vec<String>>, // base64 BF16 [1,1,hidden] per layer, decoder block output
    pub kv_caches: Option<Vec<KvCacheDump>>,
    pub conv_states: Option<Vec<StateDump>>,
    pub recurrent_states: Option<Vec<StateDump>>,
    /// Per-layer post-decoder-block hidden state at the last prompt token.
    /// Each entry is base64-encoded BF16 with shape `prefill_per_layer_hidden_shape`
    /// (typically `[1, 1, hidden]`). Used to validate a single Gemma 4 layer's
    /// kernel output against PyTorch.
    pub prefill_per_layer_hidden: Option<Vec<String>>,
    pub prefill_per_layer_hidden_shape: Option<Vec<usize>>,
    /// Per-layer hidden state snapshot taken just BEFORE the Per-Layer-Embeddings
    /// (PLE) branch and `layer_scalar` multiply — i.e. right after
    /// `residual + post_feedforward_layernorm(mlp_out)`. This is the checkpoint
    /// a Rust kernel can hit without implementing PLE yet.
    pub prefill_per_layer_pre_ple: Option<Vec<String>>,
    /// Prompt token IDs, including any special tokens the tokenizer added.
    /// Emitted by `oracle/gemma4_oracle.py` so Rust single-layer validators
    /// can recover the layer-0 input embedding without re-tokenizing.
    #[serde(default)]
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Per-layer conditioning vector at the last prompt token (Gemma 4 PLE).
    /// Base64-encoded BF16 with shape `[num_hidden_layers, hidden_size_per_layer_input]`
    /// (35x256 for E2B).
    #[serde(default)]
    pub per_layer_inputs: Option<String>,
    #[serde(default)]
    pub per_layer_inputs_shape: Option<Vec<usize>>,
    /// Per decode-step PLE conditioning. Entry k corresponds to the input token
    /// at Rust decode step k: `last_prompt_token` for k==0, else
    /// `generated_token_ids[k-1]`. Each entry is base64-encoded BF16 with shape
    /// `[num_hidden_layers, hidden_size_per_layer_input]`. Lets the Rust decode
    /// validator skip implementing `project_per_layer_inputs` itself.
    #[serde(default)]
    pub per_layer_inputs_by_step: Option<Vec<String>>,
    #[serde(default)]
    pub traced_full_attn_layer: Option<usize>,
    #[serde(default)]
    pub traced_full_attn_input: Option<String>,
    #[serde(default)]
    pub traced_full_attn_normed: Option<String>,
    #[serde(default)]
    pub traced_full_attn_q_proj: Option<String>,
    #[serde(default)]
    pub traced_full_attn_gate_proj: Option<String>,
    #[serde(default)]
    pub traced_full_attn_k_proj: Option<String>,
    #[serde(default)]
    pub traced_full_attn_v_proj: Option<String>,
    #[serde(default)]
    pub traced_full_attn_q_rope: Option<String>,
    #[serde(default)]
    pub traced_full_attn_k_rope: Option<String>,
    #[serde(default)]
    pub traced_full_attn_pre_gate: Option<String>,
    #[serde(default)]
    pub traced_full_attn_gated: Option<String>,
    #[serde(default)]
    pub traced_full_attn_gated_actual: Option<String>,
    #[serde(default)]
    pub traced_mlp_gate: Option<String>,
    #[serde(default)]
    pub traced_mlp_up: Option<String>,
    #[serde(default)]
    pub traced_mlp_swiglu: Option<String>,
    #[serde(default)]
    pub traced_mlp_down: Option<String>,
    #[serde(default)]
    pub traced_mlp_down_ca: Option<String>,
    #[serde(default)]
    pub traced_mlp_down_ca_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub traced_mlp_down_sca: Option<Vec<f32>>,
    #[serde(default)]
    pub traced_mlp_down_outlier_cols: Option<Vec<u32>>,
    #[serde(default)]
    pub traced_mlp_down_outlier_threshold: Option<f32>,
    #[serde(default)]
    pub traced_mlp_down_ca_dense: Option<String>,
    #[serde(default)]
    pub traced_mlp_down_ca_dense_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub traced_mlp_down_sca_dense: Option<Vec<f32>>,
    #[serde(default)]
    pub traced_mlp_down_suba: Option<String>,
    #[serde(default)]
    pub traced_mlp_down_suba_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub traced_mlp_down_subb_t: Option<String>,
    #[serde(default)]
    pub traced_mlp_down_subb_t_shape: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct KvCacheDump {
    pub layer: usize,
    pub k: String,       // base64
    pub k_shape: Vec<usize>,
    pub v: String,       // base64
    pub v_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct StateDump {
    pub layer: usize,
    pub data: String,    // base64
    pub shape: Vec<usize>,
}

/// Run the PyTorch oracle for prefill + decode.
pub fn run_oracle(
    oracle_script: &Path,
    model_id: &str,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    dtype: &str,
    device: &str,
    emit_state: bool,
    load_in_8bit: bool,
    fp8_model_dir: Option<&Path>,
    trace_full_attn_layer: Option<usize>,
) -> Result<OracleOutput> {
    let ids_str = prompt_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let mut cmd = Command::new("python3");
    cmd.arg(oracle_script)
        .arg("--model-id").arg(model_id)
        .arg("--prompt-ids").arg(&ids_str)
        .arg("--max-new-tokens").arg(max_new_tokens.to_string())
        .arg("--dtype").arg(dtype)
        .arg("--device").arg(device);
    if emit_state {
        cmd.arg("--emit-state");
    }
    if load_in_8bit {
        cmd.arg("--load-in-8bit");
    }
    if let Some(dir) = fp8_model_dir {
        cmd.arg("--fp8-model-dir").arg(dir);
    }
    if let Some(layer) = trace_full_attn_layer {
        cmd.arg("--trace-full-attn-layer").arg(layer.to_string());
    }

    let fp8_flag = fp8_model_dir.map(|d| format!(" --fp8-model-dir {}", d.display())).unwrap_or_default();
    let int8_flag = if load_in_8bit { " --load-in-8bit" } else { "" };
    let trace_flag = trace_full_attn_layer
        .map(|layer| format!(" --trace-full-attn-layer {layer}"))
        .unwrap_or_default();
    eprintln!("[oracle] running: python3 {} --model-id {model_id} --prompt-ids {ids_str} --max-new-tokens {max_new_tokens} --dtype {dtype} --device {device}{}{int8_flag}{fp8_flag}{trace_flag}",
        oracle_script.display(),
        if emit_state { " --emit-state" } else { "" }
    );

    let output = cmd.output().context("failed to start oracle process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("oracle process failed (exit {}): {stderr}", output.status);
    }

    let stdout = String::from_utf8(output.stdout).context("oracle stdout not UTF-8")?;
    let oracle: OracleOutput =
        serde_json::from_str(&stdout).context("failed to parse oracle JSON output")?;
    eprintln!(
        "[oracle] done: load={:.0}ms prefill={:.0}ms decode={:.0}ms tokens={}",
        oracle.load_ms, oracle.prefill_ms, oracle.decode_ms, oracle.generated_tokens
    );
    Ok(oracle)
}

/// Run the Phi-4 oracle (`oracle/phi4_oracle.py`) for a single prompt.
/// Mirrors `run_gemma4_oracle`: loads weights from `model_dir`, tokenizes
/// the prompt Python-side. Phi-4 oracle additionally accepts `--device`
/// because it can run on CPU or CUDA depending on availability.
pub fn run_phi4_oracle(
    oracle_script: &Path,
    model_dir: &Path,
    prompt: &str,
    max_new_tokens: usize,
    dtype: &str,
    device: &str,
) -> Result<OracleOutput> {
    let mut cmd = Command::new("python3");
    cmd.arg(oracle_script)
        .arg("--model-dir").arg(model_dir)
        .arg("--prompt").arg(prompt)
        .arg("--max-new-tokens").arg(max_new_tokens.to_string())
        .arg("--dtype").arg(dtype)
        .arg("--device").arg(device);

    eprintln!(
        "[oracle] running: python3 {} --model-dir {} --prompt <...> --max-new-tokens {max_new_tokens} --dtype {dtype} --device {device}",
        oracle_script.display(),
        model_dir.display(),
    );

    let output = cmd.output().context("failed to start phi4 oracle process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("phi4 oracle failed (exit {}): {stderr}", output.status);
    }

    let stdout = String::from_utf8(output.stdout).context("phi4 oracle stdout not UTF-8")?;
    let oracle: OracleOutput = serde_json::from_str(&stdout)
        .context("failed to parse phi4 oracle JSON output")?;
    eprintln!(
        "[oracle] done: load={:.0}ms prefill={:.0}ms decode={:.0}ms tokens={}",
        oracle.load_ms, oracle.prefill_ms, oracle.decode_ms, oracle.generated_tokens
    );
    Ok(oracle)
}

/// Run the Gemma 4 oracle (`oracle/gemma4_oracle.py`) for a single prompt.
/// Unlike the Qwen oracle, this one loads weights directly from `model_dir`
/// and tokenizes the prompt string Python-side. Caller should verify
/// `output.prompt_token_ids` matches its own tokenization.
pub fn run_gemma4_oracle(
    oracle_script: &Path,
    model_dir: &Path,
    prompt: &str,
    max_new_tokens: usize,
    dtype: &str,
) -> Result<OracleOutput> {
    let mut cmd = Command::new("python3");
    cmd.arg(oracle_script)
        .arg("--model-dir").arg(model_dir)
        .arg("--prompt").arg(prompt)
        .arg("--max-new-tokens").arg(max_new_tokens.to_string())
        .arg("--dtype").arg(dtype);

    eprintln!(
        "[oracle] running: python3 {} --model-dir {} --prompt <...> --max-new-tokens {max_new_tokens} --dtype {dtype}",
        oracle_script.display(),
        model_dir.display(),
    );

    let output = cmd.output().context("failed to start gemma4 oracle process")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("gemma4 oracle failed (exit {}): {stderr}", output.status);
    }

    let stdout = String::from_utf8(output.stdout).context("gemma4 oracle stdout not UTF-8")?;
    let oracle: OracleOutput = serde_json::from_str(&stdout)
        .context("failed to parse gemma4 oracle JSON output")?;
    eprintln!(
        "[oracle] done: load={:.0}ms prefill={:.0}ms decode={:.0}ms tokens={}",
        oracle.load_ms, oracle.prefill_ms, oracle.decode_ms, oracle.generated_tokens
    );
    Ok(oracle)
}
