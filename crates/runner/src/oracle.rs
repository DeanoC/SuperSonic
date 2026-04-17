use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct OracleOutput {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_logits: Vec<f32>,
    pub decode_logits: Vec<Vec<f32>>,
    pub generated_token_ids: Vec<u32>,
    // State export (only present with --emit-state)
    pub prefill_hidden: Option<String>,      // base64
    pub prefill_hidden_shape: Option<Vec<usize>>,
    pub kv_caches: Option<Vec<KvCacheDump>>,
    pub conv_states: Option<Vec<StateDump>>,
    pub recurrent_states: Option<Vec<StateDump>>,
    /// Per-layer post-decoder-block hidden state at the last prompt token.
    /// Each entry is base64-encoded BF16 with shape `prefill_per_layer_hidden_shape`
    /// (typically `[1, 1, hidden]`). Used to validate a single Gemma 4 layer's
    /// kernel output against PyTorch.
    pub prefill_per_layer_hidden: Option<Vec<String>>,
    pub prefill_per_layer_hidden_shape: Option<Vec<usize>>,
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
    emit_state: bool,
    fp8_model_dir: Option<&Path>,
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
        .arg("--dtype").arg(dtype);
    if emit_state {
        cmd.arg("--emit-state");
    }
    if let Some(dir) = fp8_model_dir {
        cmd.arg("--fp8-model-dir").arg(dir);
    }

    let fp8_flag = fp8_model_dir.map(|d| format!(" --fp8-model-dir {}", d.display())).unwrap_or_default();
    eprintln!("[oracle] running: python3 {} --model-id {model_id} --prompt-ids {ids_str} --max-new-tokens {max_new_tokens} --dtype {dtype}{}{fp8_flag}",
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
