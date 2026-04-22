use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use qwen35::config::{Activation, RopeParameters, TextConfig};
use qwen35::loader::WeightLoader;
use qwen35::rotary::RotaryTables;
use qwen35::weights::{FullWeights, LayerKind, LayerWeights, Qwen35Weights};
use serde::Deserialize;

use crate::decode_engine::{DecodeEngine, DecodeStageTimings};
use crate::oracle as oracle_mod;
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
use crate::validate;

#[derive(Debug, Clone, Deserialize)]
struct Llama31RopeScaling {
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Llama31Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    head_dim: Option<usize>,
    rope_theta: f64,
    rope_scaling: Llama31RopeScaling,
}

impl Llama31Config {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn to_text_config(&self) -> TextConfig {
        TextConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            hidden_act: Activation::Silu,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rms_norm_add_unit_offset: false,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id.clone(),
            head_dim: self.head_dim(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            layer_types: vec!["full_attention".to_string(); self.num_hidden_layers],
            rope_parameters: Some(RopeParameters {
                rope_type: "default".to_string(),
                rope_theta: self.rope_theta,
                partial_rotary_factor: 1.0,
            }),
        }
    }
}

fn load_config(model_dir: &Path) -> Result<Llama31Config> {
    let config_path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&config_path).map_err(|e| anyhow!("read config.json: {e}"))?;
    serde_json::from_str(&text).map_err(|e| anyhow!("parse config.json: {e}"))
}

fn build_rotary_tables(config: &Llama31Config, ordinal: usize) -> Result<RotaryTables> {
    let head_dim = config.head_dim();
    let rotary_dim = head_dim;
    let half_dim = rotary_dim / 2;
    let max_pos = config.max_position_embeddings;
    let base = config.rope_theta;
    let factor = config.rope_scaling.factor;
    let low_freq_factor = config.rope_scaling.low_freq_factor;
    let high_freq_factor = config.rope_scaling.high_freq_factor;
    let old_context_len = config.rope_scaling.original_max_position_embeddings as f64;
    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let base_inv = 1.0 / base.powf(2.0 * i as f64 / rotary_dim as f64);
        let wavelen = 2.0 * std::f64::consts::PI / base_inv;
        let scaled = if wavelen > low_freq_wavelen {
            base_inv / factor
        } else if wavelen < high_freq_wavelen {
            base_inv
        } else {
            let smooth_factor =
                (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            let inv_freq_llama = base_inv;
            (1.0 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        };
        inv_freq.push(scaled);
    }

    let mut cos_data = Vec::with_capacity(max_pos * half_dim * 2);
    let mut sin_data = Vec::with_capacity(max_pos * half_dim * 2);
    for pos in 0..max_pos {
        for &freq in &inv_freq {
            let angle = pos as f64 * freq;
            let c = half::bf16::from_f64(angle.cos());
            let s = half::bf16::from_f64(angle.sin());
            cos_data.extend_from_slice(&c.to_le_bytes());
            sin_data.extend_from_slice(&s.to_le_bytes());
        }
    }

    let cos =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half_dim], &cos_data)
            .map_err(|e| anyhow!("upload RoPE cos: {e}"))?;
    let sin =
        GpuBuffer::from_host_bytes(ordinal, ScalarType::BF16, &[max_pos, half_dim], &sin_data)
            .map_err(|e| anyhow!("upload RoPE sin: {e}"))?;
    Ok(RotaryTables {
        cos,
        sin,
        rotary_dim,
    })
}

fn load_weights(
    model_dir: &Path,
    text_config: &TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    let loader = WeightLoader::from_dir(model_dir).map_err(|e| anyhow!("open safetensors: {e}"))?;
    let embed_name = format!("{weight_prefix}.embed_tokens.weight");
    let embed_tokens = Arc::new(
        loader
            .load_to_gpu(&embed_name, ordinal)
            .map_err(|e| anyhow!("load {embed_name}: {e}"))?,
    );

    let lm_head = if loader.contains("lm_head.weight") {
        Arc::new(
            loader
                .load_to_gpu("lm_head.weight", ordinal)
                .map_err(|e| anyhow!("load lm_head.weight: {e}"))?,
        )
    } else {
        embed_tokens.clone()
    };

    let norm_name = format!("{weight_prefix}.norm.weight");
    let norm_weight = loader
        .load_to_gpu(&norm_name, ordinal)
        .map_err(|e| anyhow!("load {norm_name}: {e}"))?;

    let mut layers = Vec::with_capacity(text_config.num_hidden_layers);
    for idx in 0..text_config.num_hidden_layers {
        let lp = format!("{weight_prefix}.layers.{idx}");
        let fa = format!("{lp}.self_attn");
        let mlp = format!("{lp}.mlp");
        layers.push(LayerWeights {
            kind: LayerKind::Full,
            input_norm_w: loader
                .load_to_gpu(&format!("{lp}.input_layernorm.weight"), ordinal)
                .map_err(|e| anyhow!("load {lp}.input_layernorm.weight: {e}"))?,
            post_attn_norm_w: loader
                .load_to_gpu(&format!("{lp}.post_attention_layernorm.weight"), ordinal)
                .map_err(|e| anyhow!("load {lp}.post_attention_layernorm.weight: {e}"))?,
            gate_proj_w: loader
                .load_to_gpu(&format!("{mlp}.gate_proj.weight"), ordinal)
                .map_err(|e| anyhow!("load {mlp}.gate_proj.weight: {e}"))?,
            up_proj_w: loader
                .load_to_gpu(&format!("{mlp}.up_proj.weight"), ordinal)
                .map_err(|e| anyhow!("load {mlp}.up_proj.weight: {e}"))?,
            down_proj_w: loader
                .load_to_gpu(&format!("{mlp}.down_proj.weight"), ordinal)
                .map_err(|e| anyhow!("load {mlp}.down_proj.weight: {e}"))?,
            gate_proj_scale: None,
            up_proj_scale: None,
            down_proj_scale: None,
            gate_proj_int4_scale: None,
            gate_proj_int4_zero: None,
            up_proj_int4_scale: None,
            up_proj_int4_zero: None,
            down_proj_int4_scale: None,
            down_proj_int4_zero: None,
            linear: None,
            full: Some(FullWeights {
                q_proj_w: loader
                    .load_to_gpu(&format!("{fa}.q_proj.weight"), ordinal)
                    .map_err(|e| anyhow!("load {fa}.q_proj.weight: {e}"))?,
                k_proj_w: loader
                    .load_to_gpu(&format!("{fa}.k_proj.weight"), ordinal)
                    .map_err(|e| anyhow!("load {fa}.k_proj.weight: {e}"))?,
                v_proj_w: loader
                    .load_to_gpu(&format!("{fa}.v_proj.weight"), ordinal)
                    .map_err(|e| anyhow!("load {fa}.v_proj.weight: {e}"))?,
                o_proj_w: loader
                    .load_to_gpu(&format!("{fa}.o_proj.weight"), ordinal)
                    .map_err(|e| anyhow!("load {fa}.o_proj.weight: {e}"))?,
                q_norm_w: None,
                k_norm_w: None,
                q_proj_scale: None,
                k_proj_scale: None,
                v_proj_scale: None,
                o_proj_scale: None,
                q_proj_int4_scale: None,
                q_proj_int4_zero: None,
                k_proj_int4_scale: None,
                k_proj_int4_zero: None,
                v_proj_int4_scale: None,
                v_proj_int4_zero: None,
                o_proj_int4_scale: None,
                o_proj_int4_zero: None,
            }),
        });
    }

    Ok(Qwen35Weights {
        config: text_config.clone(),
        embed_tokens,
        lm_head,
        lm_head_scale: None,
        norm_weight,
        layers,
        is_fp8: false,
        fp8_block_size: 0,
        is_int4: false,
        int4_group_size: 0,
    })
}

fn print_stage_timings(timings: DecodeStageTimings, tokens: usize) {
    if tokens == 0 {
        return;
    }
    eprintln!(
        "[stage] tokens={} total_ms={:.1} per_tok_ms={:.1} persistent={:.1} rms_norm={:.1} lm_head={:.1} logits_d2h={:.1} host_sampling={:.1}",
        tokens,
        timings.total_ms(),
        timings.total_ms() / tokens as f64,
        timings.persistent_ms,
        timings.rms_norm_ms,
        timings.lm_head_ms,
        timings.logits_d2h_ms,
        timings.host_sampling_ms,
    );
}

pub fn run_llama31(
    cli: &crate::Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Llama31(p) => p,
        _ => unreachable!("run_llama31 dispatched for non-Llama31 variant {model_variant}"),
    };

    if cli.batch_size != 1 {
        bail!("Llama 3.1 CUDA path is single-sequence at launch (--batch-size must be 1)");
    }
    if cli.oracle_prefill {
        bail!("Llama 3.1 CUDA path has no --oracle-prefill path yet");
    }
    if cli.fp8_runtime || cli.int4 || cli.kv_fp8 {
        bail!("Llama 3.1 CUDA path is BF16 raw-safetensors only at launch");
    }
    if cli.download_bake {
        bail!("Llama 3.1 CUDA path has no release-hosted bake yet; point --model-dir at raw safetensors");
    }
    if !cli.model_dir.join("config.json").exists() {
        bail!(
            "Llama 3.1 model dir {} has no config.json. Populate it with raw Hugging Face files first: `huggingface-cli download {} --local-dir {}`",
            cli.model_dir.display(),
            model_variant.hf_model_id(),
            cli.model_dir.display(),
        );
    }

    let config = load_config(&cli.model_dir)?;
    let text_config = config.to_text_config();
    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    if context_tokens < prompt_ids.len() + cli.max_new_tokens {
        bail!(
            "--context-size {context_tokens} < prompt_tokens {} + max_new_tokens {}",
            prompt_ids.len(),
            cli.max_new_tokens,
        );
    }

    let kv_per_token = text_config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let kv_bytes = kv_per_token * context_tokens as u64;
    let estimated_vram =
        ((entry.vram.fixed_bytes + kv_bytes) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[llama31] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={} max_pos={} rope_theta={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
        text_config.max_position_embeddings,
        config.rope_theta,
    );
    eprintln!(
        "[vram] estimated={:.2}GiB (weights+scratch={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(entry.vram.fixed_bytes),
        gib(kv_bytes),
        context_tokens,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        bail!(
            "Insufficient VRAM for {context_tokens}-token context: need ~{:.2}GiB, GPU has {:.1}GiB.",
            gib(estimated_vram),
            gib(total_vram),
        );
    }

    let oracle_output = if cli.validate {
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow!("could not derive oracle script path from CARGO_MANIFEST_DIR"))?
            .join("oracle/run_oracle.py");
        let oracle_device =
            crate::resolve_oracle_device(&cli.oracle_device, entry.backend, ordinal);
        let model_id = cli.model_dir.to_string_lossy().into_owned();
        let oracle = oracle_mod::run_oracle(
            &oracle_script,
            &model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
            false,
            None,
            None,
        )?;
        if let Some(ref oracle_ids) = oracle.prompt_token_ids {
            if oracle_ids != &prompt_ids {
                bail!(
                    "tokenizer mismatch between Rust and Python oracle: rust={prompt_ids:?} oracle={oracle_ids:?}"
                );
            }
        }
        Some(oracle)
    } else {
        None
    };

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    let t0 = Instant::now();
    eprintln!("[weights] loading raw BF16 safetensors");
    let weights = load_weights(&cli.model_dir, &text_config, ordinal, params.weight_prefix)?;
    let rotary = build_rotary_tables(&config, ordinal)?;
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let proj_buf_floats = text_config
        .intermediate_size
        .max(text_config.num_attention_heads * text_config.head_dim * 2);
    let required_attn_scratch = qwen35::scratch::required_attn_scratch_floats(
        text_config.num_attention_heads,
        text_config.head_dim,
        context_tokens,
        params.kv_chunk_size,
    );

    let mut engine = DecodeEngine::new_with_rotary(
        weights,
        rotary,
        ordinal,
        proj_buf_floats,
        required_attn_scratch,
        params.kv_chunk_size,
        true,
        cli.prefill_chunk_size,
        false,
        1,
    )?;

    let prefill_start = Instant::now();
    let prefill = engine.prefill_native_with_final_norm(&prompt_ids)?;
    let eos_ids = text_config.eos_token_ids();
    let mut generated: Vec<u32> = Vec::with_capacity(cli.max_new_tokens);
    let mut next_token = DecodeEngine::greedy_sample(&prefill.logits);
    let mut max_delta = 0.0f32;
    let mut token_mismatches = 0usize;
    eprintln!(
        "[prefill] {} tokens in {:.0}ms",
        prompt_ids.len(),
        prefill_start.elapsed().as_millis()
    );

    if let Some(ref oracle) = oracle_output {
        let delta = validate::max_abs_delta(&prefill.logits, &oracle.prefill_logits);
        if delta > max_delta {
            max_delta = delta;
        }
        let mismatch = match oracle.generated_token_ids.first().copied() {
            Some(o) if o != next_token => {
                token_mismatches += 1;
                format!(" MISMATCH (oracle_next={o})")
            }
            _ => String::new(),
        };
        eprintln!(
            "[validate] prefill delta={delta:.4} rust_next={next_token}{mismatch}"
        );
    }

    if cli.max_new_tokens > 0 {
        generated.push(next_token);
    }

    let decode_start = Instant::now();
    let mut stage_totals = DecodeStageTimings::default();
    while generated.len() < cli.max_new_tokens && !eos_ids.contains(&next_token) {
        let pos = prompt_ids.len() + generated.len() - 1;
        let (logits, timings) = engine.decode_step_with_timings(next_token, pos)?;
        stage_totals.add_assign(timings);
        next_token = DecodeEngine::greedy_sample(&logits);
        if let Some(ref oracle) = oracle_output {
            let decode_idx = generated.len() - 1;
            if let Some(oracle_logits) = oracle.decode_logits.get(decode_idx) {
                let delta = validate::max_abs_delta(&logits, oracle_logits);
                if delta > max_delta {
                    max_delta = delta;
                }
                let mismatch = match oracle.generated_token_ids.get(decode_idx + 1).copied() {
                    Some(o) if o != next_token => {
                        token_mismatches += 1;
                        format!(" MISMATCH (oracle_next={o})")
                    }
                    _ => String::new(),
                };
                eprintln!(
                    "[validate] step={decode_idx} pos={pos} delta={delta:.4} input_tok={} rust_next={next_token}{mismatch}",
                    generated[decode_idx]
                );
            }
        }
        generated.push(next_token);
    }

    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    println!(
        "[tokens] {}",
        generated
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_step = if generated.is_empty() {
        0.0
    } else {
        decode_ms / generated.len() as f64
    };
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_step={ms_per_step:.1}",
        prompt_ids.len(),
        generated.len(),
    );
    if oracle_output.is_some() {
        eprintln!(
            "[validate] max_delta={max_delta:.4} token_mismatches={token_mismatches}"
        );
    }
    if cli.emit_stage_timings {
        print_stage_timings(stage_totals, generated.len().saturating_sub(1));
    }

    Ok(())
}
