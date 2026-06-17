use anyhow::Result;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::decode_engine::DecodeEngine;
use crate::model_files::model_dir_has_raw_safetensors;
use crate::prefill_engine::PrefillResult;
use crate::profiling::PrefillProfileScope;
use crate::qwen35_validation::{qwen35_oracle_script_path, Qwen35NativePrefillTrace};
use crate::tensor_bytes::bf16_bytes_to_f32 as decode_bf16_le;
use crate::{oracle, Cli};

pub(crate) struct Qwen35Prefill {
    pub(crate) logits: Vec<f32>,
    pub(crate) native_trace: Option<Qwen35NativePrefillTrace>,
    pub(crate) next_token: u32,
}

pub(crate) struct HostLmHeadRescorer {
    loader: qwen35::loader::WeightLoader,
    tensor_name: String,
}

impl HostLmHeadRescorer {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        if !model_dir_has_raw_safetensors(model_dir) {
            return Ok(None);
        }
        let loader = qwen35::loader::WeightLoader::from_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("open raw lm_head weights: {e}"))?;
        let tensor_name = if loader.contains("lm_head.weight") {
            "lm_head.weight".to_string()
        } else if loader.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight".to_string()
        } else {
            return Ok(None);
        };
        Ok(Some(Self {
            loader,
            tensor_name,
        }))
    }

    fn rescore(&self, normed: &[f32], candidate_ids: &[usize]) -> Result<u32> {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for &candidate in candidate_ids {
            let row = self
                .loader
                .load_bf16_row_f32(&self.tensor_name, candidate)
                .map_err(|e| anyhow::anyhow!("load lm_head row {candidate}: {e}"))?;
            anyhow::ensure!(
                row.len() == normed.len(),
                "lm_head row len {} != normed len {}",
                row.len(),
                normed.len()
            );
            let score = row
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            if score > best_val {
                best_val = score;
                best_idx = candidate;
            }
        }
        Ok(best_idx as u32)
    }
}

pub(crate) fn sample_qwen_logits_with_rescore(
    logits: &[f32],
    normed: Option<&[f32]>,
    rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let greedy = DecodeEngine::greedy_sample(logits);
    let Some(normed) = normed else {
        return Ok(greedy);
    };
    let Some(rescorer) = rescorer else {
        return Ok(greedy);
    };

    const RESCORE_MARGIN: f32 = 0.25;
    const RESCORE_TOP_K: usize = 4;

    let candidates = top_k_candidate_ids(logits, RESCORE_TOP_K);
    if candidates.len() < 2 {
        return Ok(greedy);
    }
    let top0 = logits[candidates[0]];
    let top1 = logits[candidates[1]];
    if top0 - top1 > RESCORE_MARGIN {
        return Ok(greedy);
    }

    let rescored = rescorer.rescore(normed, &candidates)?;
    if rescored != greedy {
        eprintln!(
            "[sample-rescore] token {} -> {} (top_margin={:.4})",
            greedy,
            rescored,
            top0 - top1
        );
    }
    Ok(rescored)
}

fn top_k_candidate_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(usize, f32)> = Vec::new();
    for (idx, &val) in logits.iter().enumerate() {
        let pos = best
            .iter()
            .position(|&(_, best_val)| val > best_val)
            .unwrap_or(best.len());
        if pos < k {
            best.insert(pos, (idx, val));
            if best.len() > k {
                best.pop();
            }
        }
    }
    best.into_iter().map(|(idx, _)| idx).collect()
}

pub(crate) fn run_qwen35_prefill(
    cli: &Cli,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
    oracle_model_id: &str,
    oracle_device: &str,
    fp8_oracle_dir: Option<&Path>,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
    allow_host_lm_head_rescore: bool,
) -> Result<Qwen35Prefill> {
    let prefill_start = Instant::now();
    let profile = PrefillProfileScope::new(
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        "qwen3.5",
        &cli.model,
        &cli.backend,
        prompt_ids.len(),
    );
    if cli.oracle_prefill {
        let oracle_script = qwen35_oracle_script_path();
        let output = oracle::run_oracle(
            &oracle_script,
            oracle_model_id,
            prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            oracle_device,
            true,
            false,
            fp8_oracle_dir,
            None,
        )?;
        engine.load_prefill_state(&output)?;
        let first = output.generated_token_ids.first().copied().ok_or_else(|| {
            anyhow::anyhow!(
                "oracle prefill produced no generated tokens; --oracle-prefill requires --max-new-tokens > 0"
            )
        })?;
        eprintln!(
            "[prefill] oracle prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
        profile.finish()?;
        return Ok(Qwen35Prefill {
            logits: output.prefill_logits,
            native_trace: None,
            next_token: first,
        });
    }

    if std::env::var_os("SUPERSONIC_QWEN35_PREFILL_DECODE_LOOP").is_some() {
        let mut logits = engine.prefill_native(&prompt_ids[..1])?;
        for (pos, &token_id) in prompt_ids.iter().enumerate().skip(1) {
            logits = engine.decode_step_batch(&[token_id], pos)?.remove(0);
        }
        let first = DecodeEngine::greedy_sample(&logits);
        eprintln!(
            "[prefill] sequential decode-loop prefill done in {:.0}ms",
            prefill_start.elapsed().as_millis()
        );
        profile.finish()?;
        return Ok(Qwen35Prefill {
            logits,
            native_trace: None,
            next_token: first,
        });
    }

    let dump_layer_hiddens_dir =
        std::env::var_os("SUPERSONIC_QWEN35_DUMP_LAYER_HIDDENS_DIR").map(PathBuf::from);
    let dump_layer_components_dir =
        std::env::var_os("SUPERSONIC_QWEN35_DUMP_LAYER_COMPONENTS_DIR").map(PathBuf::from);
    let trace_prefill_layers = cli.trace_prefill_layers
        || dump_layer_hiddens_dir.is_some()
        || dump_layer_components_dir.is_some();
    let prefill_result = if trace_prefill_layers {
        engine.prefill_native_with_trace(prompt_ids)?
    } else {
        engine.prefill_native_with_final_norm(prompt_ids)?
    };
    if let (Some(path), Some(final_norm)) = (
        std::env::var_os("SUPERSONIC_QWEN35_DUMP_FINAL_NORM"),
        prefill_result.final_norm_trace.as_ref(),
    ) {
        let path = std::path::PathBuf::from(path);
        std::fs::write(&path, final_norm)
            .map_err(|e| anyhow::anyhow!("write final norm dump {}: {e}", path.display()))?;
        eprintln!("[prefill] dumped final norm to {}", path.display());
    }
    if let (Some(dir), Some(layer_hiddens)) = (
        dump_layer_hiddens_dir.as_ref(),
        prefill_result.layer_hidden_trace.as_ref(),
    ) {
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("create layer hidden dump dir {}: {e}", dir.display()))?;
        for (idx, bytes) in layer_hiddens.iter().enumerate() {
            let path = dir.join(format!("layer_{idx:02}.bf16"));
            std::fs::write(&path, bytes)
                .map_err(|e| anyhow::anyhow!("write layer hidden dump {}: {e}", path.display()))?;
        }
        eprintln!(
            "[prefill] dumped {} layer hidden rows to {}",
            layer_hiddens.len(),
            dir.display()
        );
    }
    if let Some(dir) = dump_layer_components_dir.as_ref() {
        std::fs::create_dir_all(dir).map_err(|e| {
            anyhow::anyhow!("create layer component dump dir {}: {e}", dir.display())
        })?;
        let attn = dump_trace_rows(
            dir,
            "attn",
            prefill_result.layer_attn_trace.as_ref(),
            "attn residual",
        )?;
        let post = dump_trace_rows(
            dir,
            "post",
            prefill_result.layer_post_attn_norm_trace.as_ref(),
            "post-attn norm",
        )?;
        let mlp = dump_trace_rows(
            dir,
            "mlp",
            prefill_result.layer_mlp_out_trace.as_ref(),
            "MLP output",
        )?;
        let hidden = dump_trace_rows(
            dir,
            "hidden",
            prefill_result.layer_hidden_trace.as_ref(),
            "hidden",
        )?;
        eprintln!(
            "[prefill] dumped layer component rows to {} (attn={attn}, post={post}, mlp={mlp}, hidden={hidden})",
            dir.display()
        );
    }
    let first = sample_qwen_prefill_token(
        &prefill_result,
        host_lm_head_rescorer.filter(|_| allow_host_lm_head_rescore),
    )?;
    eprintln!(
        "[prefill] native GPU prefill done in {:.0}ms",
        prefill_start.elapsed().as_millis()
    );
    profile.finish()?;

    Ok(Qwen35Prefill {
        logits: prefill_result.logits,
        native_trace: Some((
            prefill_result.final_norm_trace,
            prefill_result.layer_attn_trace,
            prefill_result.layer_post_attn_norm_trace,
            prefill_result.layer_mlp_out_trace,
            prefill_result.layer_hidden_trace,
        )),
        next_token: first,
    })
}

fn dump_trace_rows(
    dir: &Path,
    prefix: &str,
    rows: Option<&Vec<Vec<u8>>>,
    label: &str,
) -> Result<usize> {
    let rows = rows.ok_or_else(|| anyhow::anyhow!("missing {label} trace rows"))?;
    for (idx, bytes) in rows.iter().enumerate() {
        let path = dir.join(format!("{prefix}_{idx:02}.bf16"));
        std::fs::write(&path, bytes)
            .map_err(|e| anyhow::anyhow!("write {label} dump {}: {e}", path.display()))?;
    }
    Ok(rows.len())
}

fn sample_qwen_prefill_token(
    prefill_result: &PrefillResult,
    host_lm_head_rescorer: Option<&HostLmHeadRescorer>,
) -> Result<u32> {
    let prefill_normed = prefill_result
        .final_norm_trace
        .as_deref()
        .map(decode_bf16_le);
    sample_qwen_logits_with_rescore(
        &prefill_result.logits,
        prefill_normed.as_deref(),
        host_lm_head_rescorer,
    )
}
