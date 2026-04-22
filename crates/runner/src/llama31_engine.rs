use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use gpu_hal::{GpuBuffer, ScalarType};
use model_store::BakedStore;
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
            gate_proj_int8_scale: None,
            up_proj_int8_scale: None,
            down_proj_int8_scale: None,
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
                q_proj_int8_scale: None,
                k_proj_int8_scale: None,
                v_proj_int8_scale: None,
                o_proj_int8_scale: None,
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
        weight_prefix: weight_prefix.to_string(),
        embed_tokens,
        lm_head,
        lm_head_scale: None,
        norm_weight,
        layers,
        is_fp8: false,
        fp8_block_size: 0,
        is_int4: false,
        int4_group_size: 0,
        is_int8: false,
        int8_baked_store: None,
        int8_outlier_threshold: 0.0,
    })
}

fn scb_name(weight_name: &str) -> String {
    weight_name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.SCB"))
        .unwrap_or_else(|| format!("{weight_name}.SCB"))
}

fn load_baked_weight_raw(store: &BakedStore, name: &str, ordinal: usize) -> Result<GpuBuffer> {
    store
        .load_to_gpu(name, ordinal)
        .map_err(|e| anyhow!("load {name}: {e}"))
}

fn load_baked_scb(store: &BakedStore, weight_name: &str, ordinal: usize) -> Result<Option<GpuBuffer>> {
    let name = scb_name(weight_name);
    if !store.contains(&name) {
        return Ok(None);
    }
    store
        .load_to_gpu(&name, ordinal)
        .map(Some)
        .map_err(|e| anyhow!("load {name}: {e}"))
}

fn load_baked_int8_weights(
    model_dir: &Path,
    store: Arc<BakedStore>,
    text_config: &TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    let raw_loader =
        WeightLoader::from_dir(model_dir).map_err(|e| anyhow!("open safetensors: {e}"))?;
    let load_name = |name: &str| -> Result<GpuBuffer> {
        if store.contains(name) {
            load_baked_weight_raw(&store, name, ordinal)
        } else {
            raw_loader
                .load_to_gpu(name, ordinal)
                .map_err(|e| anyhow!("load {name}: {e}"))
        }
    };

    let embed_name = format!("{weight_prefix}.embed_tokens.weight");
    let embed_tokens = Arc::new(load_name(&embed_name)?);

    let lm_head = if store.contains("lm_head.weight") {
        Arc::new(load_baked_weight_raw(&store, "lm_head.weight", ordinal)?)
    } else {
        Arc::new(
            raw_loader
                .load_to_gpu("lm_head.weight", ordinal)
                .map_err(|e| anyhow!("load lm_head.weight: {e}"))?,
        )
    };

    let norm_name = format!("{weight_prefix}.norm.weight");
    let norm_weight = load_name(&norm_name)?;

    let mut layers = Vec::with_capacity(text_config.num_hidden_layers);
    for idx in 0..text_config.num_hidden_layers {
        let lp = format!("{weight_prefix}.layers.{idx}");
        let fa = format!("{lp}.self_attn");
        let mlp = format!("{lp}.mlp");
        layers.push(LayerWeights {
            kind: LayerKind::Full,
            input_norm_w: load_name(&format!("{lp}.input_layernorm.weight"))?,
            post_attn_norm_w: load_name(&format!("{lp}.post_attention_layernorm.weight"))?,
            gate_proj_w: load_name(&format!("{mlp}.gate_proj.weight"))?,
            up_proj_w: load_name(&format!("{mlp}.up_proj.weight"))?,
            down_proj_w: load_name(&format!("{mlp}.down_proj.weight"))?,
            gate_proj_scale: None,
            up_proj_scale: None,
            down_proj_scale: None,
            gate_proj_int8_scale: load_baked_scb(&store, &format!("{mlp}.gate_proj.weight"), ordinal)?,
            up_proj_int8_scale: load_baked_scb(&store, &format!("{mlp}.up_proj.weight"), ordinal)?,
            down_proj_int8_scale: load_baked_scb(&store, &format!("{mlp}.down_proj.weight"), ordinal)?,
            gate_proj_int4_scale: None,
            gate_proj_int4_zero: None,
            up_proj_int4_scale: None,
            up_proj_int4_zero: None,
            down_proj_int4_scale: None,
            down_proj_int4_zero: None,
            linear: None,
            full: Some(FullWeights {
                q_proj_w: load_name(&format!("{fa}.q_proj.weight"))?,
                k_proj_w: load_name(&format!("{fa}.k_proj.weight"))?,
                v_proj_w: load_name(&format!("{fa}.v_proj.weight"))?,
                o_proj_w: load_name(&format!("{fa}.o_proj.weight"))?,
                q_norm_w: None,
                k_norm_w: None,
                q_proj_scale: None,
                k_proj_scale: None,
                v_proj_scale: None,
                o_proj_scale: None,
                q_proj_int8_scale: load_baked_scb(&store, &format!("{fa}.q_proj.weight"), ordinal)?,
                k_proj_int8_scale: load_baked_scb(&store, &format!("{fa}.k_proj.weight"), ordinal)?,
                v_proj_int8_scale: load_baked_scb(&store, &format!("{fa}.v_proj.weight"), ordinal)?,
                o_proj_int8_scale: load_baked_scb(&store, &format!("{fa}.o_proj.weight"), ordinal)?,
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
        weight_prefix: weight_prefix.to_string(),
        embed_tokens,
        lm_head,
        lm_head_scale: None,
        norm_weight,
        layers,
        is_fp8: false,
        fp8_block_size: 0,
        is_int4: false,
        int4_group_size: 0,
        is_int8: true,
        int8_baked_store: Some(store),
        int8_outlier_threshold: 6.0,
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
        bail!("Llama 3.1 CUDA path does not support --fp8-runtime, --int4, or --kv-fp8");
    }
    if cli.int8 && cli.no_bake {
        bail!("--int8 requires the baked path; drop --no-bake");
    }
    if cli.int8 && cli.download_bake {
        bail!("Llama 3.1 INT8 currently supports only local baking, not --download-bake");
    }
    if !cli.int8 && cli.download_bake {
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

    let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("could not derive oracle script path from CARGO_MANIFEST_DIR"))?
        .join("oracle/run_oracle.py");
    let oracle_device = crate::resolve_oracle_device(&cli.oracle_device, entry.backend, ordinal);
    let model_id = cli.model_dir.to_string_lossy().into_owned();
    let oracle_output = if cli.validate {
        let emit_state = cli.trace_prefill_layers || cli.trace_oracle_prefill_layer.is_some();
        let oracle = oracle_mod::run_oracle(
            &oracle_script,
            &model_id,
            &prompt_ids,
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
            emit_state,
            cli.int8,
            None,
            cli.trace_oracle_prefill_layer,
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
    let weights = if cli.int8 {
        let bake_dir = model_store::bake_dir_int8(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
        if !model_store::version_ok(&bake_dir) {
            let bake_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| anyhow!("could not derive bake script path from CARGO_MANIFEST_DIR"))?
                .join("oracle/bake_int8_llama31.py");
            eprintln!("[weights] baking local INT8 package at {}", bake_dir.display());
            let output = Command::new("python3")
                .arg(&bake_script)
                .arg("--model-dir")
                .arg(&cli.model_dir)
                .arg("--device")
                .arg(format!("cuda:{ordinal}"))
                .output()
                .with_context(|| format!("start {}", bake_script.display()))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                bail!(
                    "INT8 bake failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    stdout.trim(),
                    stderr.trim()
                );
            }
        }
        eprintln!("[weights] loading baked INT8 package {}", bake_dir.display());
        let store = Arc::new(
            BakedStore::open(&bake_dir)
                .map_err(|e| anyhow!("open INT8 bake {}: {e}", bake_dir.display()))?
        );
        load_baked_int8_weights(
            &cli.model_dir,
            store,
            &text_config,
            ordinal,
            params.weight_prefix,
        )?
    } else {
        eprintln!("[weights] loading raw BF16 safetensors");
        load_weights(&cli.model_dir, &text_config, ordinal, params.weight_prefix)?
    };
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
    let prefill = if cli.trace_prefill_layers {
        engine.prefill_native_with_trace(&prompt_ids)?
    } else {
        engine.prefill_native_with_final_norm(&prompt_ids)?
    };
    let eos_ids = text_config.eos_token_ids();
    let mut generated: Vec<u32> = Vec::with_capacity(cli.max_new_tokens);
    let mut next_token = DecodeEngine::greedy_sample(&prefill.logits);
    let mut max_delta = 0.0f32;
    let mut token_mismatches = 0usize;
    let gpu_validate_enabled = cli.gpu_validate;
    let mut gpu_max_delta = 0.0f32;
    let mut gpu_dumped_layer_trace = false;
    eprintln!(
        "[prefill] {} tokens in {:.0}ms",
        prompt_ids.len(),
        prefill_start.elapsed().as_millis()
    );
    if gpu_validate_enabled {
        eprintln!("[gpu-validate] replaying decode steps through GPU prefill reference");
    }

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

        if let (Some(native_final_norm_trace), Some(oracle_final_norm_b64)) = (
            prefill.final_norm_trace.as_deref(),
            oracle.prefill_hidden.as_ref(),
        ) {
            let b64 = base64::engine::general_purpose::STANDARD;
            let oracle_final_norm_bytes = b64
                .decode(oracle_final_norm_b64)
                .map_err(|e| anyhow!("decode oracle prefill_hidden: {e}"))?;
            let final_norm_delta = validate::max_abs_delta(
                &decode_bf16_le(native_final_norm_trace),
                &decode_bf16_le(&oracle_final_norm_bytes),
            );
            eprintln!("[trace-prefill] final_norm_delta={final_norm_delta:.4}");
        }

        if cli.trace_prefill_layers {
            trace_llama31_prefill_layers(&mut engine, &prefill, oracle)?;
        }
        if let Some(trace_layer) = cli.trace_oracle_prefill_layer {
            trace_llama31_oracle_prefill_layer(
                &mut engine,
                trace_layer,
                &prompt_ids,
                &oracle_script,
                &model_id,
                &cli.oracle_dtype,
                &oracle_device,
                oracle,
            )?;
        }
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
        let native_next = DecodeEngine::greedy_sample(&logits);
        if let Some(ref oracle) = oracle_output {
            let decode_idx = generated.len() - 1;
            if let Some(oracle_logits) = oracle.decode_logits.get(decode_idx) {
                let delta = validate::max_abs_delta(&logits, oracle_logits);
                if delta > max_delta {
                    max_delta = delta;
                }
                let mismatch = match oracle.generated_token_ids.get(decode_idx + 1).copied() {
                    Some(o) if o != native_next => {
                        token_mismatches += 1;
                        format!(" MISMATCH (oracle_next={o})")
                    }
                    _ => String::new(),
                };
                eprintln!(
                    "[validate] step={decode_idx} pos={pos} delta={delta:.4} input_tok={} rust_next={native_next}{mismatch}",
                    generated[decode_idx]
                );
            }
        }
        if gpu_validate_enabled {
            let gpu_token_ids: Vec<u32> = prompt_ids
                .iter()
                .copied()
                .chain(generated.iter().copied())
                .collect();
            let gpu_logits = crate::prefill_engine::gpu_reference_replay_step(
                &engine.weights(),
                &engine.rotary(),
                &gpu_token_ids,
                ordinal,
                params.kv_chunk_size,
                cli.prefill_chunk_size,
                true,
            )?;
            let delta = validate::max_abs_delta(&logits, &gpu_logits);
            if delta > gpu_max_delta {
                gpu_max_delta = delta;
            }
            let gpu_next = DecodeEngine::greedy_sample(&gpu_logits);
            let mismatch = if gpu_next == native_next { "" } else { " MISMATCH" };
            eprintln!(
                "[gpu-validate] step={} pos={} input_tok={} delta={delta:.4} native_next={} gpu_next={}{}",
                generated.len() - 1,
                pos,
                generated[generated.len() - 1],
                native_next,
                gpu_next,
                mismatch
            );
            if gpu_next != native_next && !gpu_dumped_layer_trace {
                let full_token_ids = gpu_token_ids.clone();
                let prefix_token_ids = &full_token_ids[..full_token_ids.len().saturating_sub(1)];
                let replay = engine.prefill_native_with_trace(&full_token_ids)?;
                let replay_hidden = replay.layer_hidden_trace.as_ref().ok_or_else(|| {
                    anyhow!("internal: llama31 gpu-validate replay hidden trace missing")
                })?;
                let mut first_bad = None;
                for layer_idx in 0..engine.weights().config.num_hidden_layers {
                    engine.rebuild_prefill_state(prefix_token_ids, false)?;
                    let (_trace_logits, layer_trace) =
                        engine.component_decode_step_4b_trace_layer(next_token, pos, layer_idx)?;
                    let layer_delta = validate::max_abs_delta(
                        &decode_bf16_le(&layer_trace.layer_hidden),
                        &decode_bf16_le(&replay_hidden[layer_idx]),
                    );
                    eprintln!(
                        "[gpu-validate-layer] step={} layer={} hidden_delta={layer_delta:.6}",
                        generated.len() - 1,
                        layer_idx
                    );
                    if first_bad.is_none() && layer_delta > 0.25 {
                        first_bad = Some((layer_idx, layer_delta));
                    }
                }
                engine.rebuild_prefill_state(&full_token_ids, false)?;
                if let Some((layer_idx, layer_delta)) = first_bad {
                    eprintln!(
                        "[gpu-validate-layer] first_bad_layer={} hidden_delta={layer_delta:.6}",
                        layer_idx
                    );
                    let replay_attn = replay
                        .layer_attn_trace
                        .as_ref()
                        .and_then(|layers| layers.get(layer_idx))
                        .ok_or_else(|| anyhow!("internal: missing replay attn trace for layer {layer_idx}"))?;
                    let replay_post = replay
                        .layer_post_attn_norm_trace
                        .as_ref()
                        .and_then(|layers| layers.get(layer_idx))
                        .ok_or_else(|| anyhow!("internal: missing replay post trace for layer {layer_idx}"))?;
                    let replay_mlp = replay
                        .layer_mlp_out_trace
                        .as_ref()
                        .and_then(|layers| layers.get(layer_idx))
                        .ok_or_else(|| anyhow!("internal: missing replay mlp trace for layer {layer_idx}"))?;
                    engine.rebuild_prefill_state(prefix_token_ids, false)?;
                    let (_trace_logits, layer_trace) =
                        engine.component_decode_step_4b_trace_layer(next_token, pos, layer_idx)?;
                    let attn_delta = validate::max_abs_delta(
                        &decode_bf16_le(&layer_trace.attn_hidden),
                        &decode_bf16_le(replay_attn),
                    );
                    let post_delta = validate::max_abs_delta(
                        &decode_bf16_le(&layer_trace.post_attn_norm),
                        &decode_bf16_le(replay_post),
                    );
                    let mlp_delta = validate::max_abs_delta(
                        &decode_bf16_le(&layer_trace.mlp_out),
                        &decode_bf16_le(replay_mlp),
                    );
                    let hidden_delta = validate::max_abs_delta(
                        &decode_bf16_le(&layer_trace.layer_hidden),
                        &decode_bf16_le(&replay_hidden[layer_idx]),
                    );
                    eprintln!(
                        "[gpu-validate-layer] layer={} attn_delta={attn_delta:.6} post_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}",
                        layer_idx
                    );
                    if engine.weights().config.is_full_attention(layer_idx) && layer_idx > 0 {
                        engine.rebuild_prefill_state(prefix_token_ids, false)?;
                        let (_stage_logits, native_input_hidden) =
                            engine.component_decode_step_4b_traced(next_token, pos, layer_idx)?;
                        let replay_input_hidden = &replay_hidden[layer_idx - 1];
                        let input_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_input_hidden),
                            &decode_bf16_le(replay_input_hidden),
                        );
                        engine.rebuild_prefill_state(prefix_token_ids, false)?;
                        let native_stage = engine.trace_full_attention_stages_from_hidden(
                            layer_idx,
                            &native_input_hidden,
                            pos,
                        )?;
                        let replay_stage = engine.trace_full_attention_stages_from_hidden(
                            layer_idx,
                            replay_input_hidden,
                            pos,
                        )?;
                        let normed_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.normed),
                            &decode_bf16_le(&replay_stage.normed),
                        );
                        let q_proj_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.q_proj),
                            &decode_bf16_le(&replay_stage.q_proj),
                        );
                        let gate_proj_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.gate_proj),
                            &decode_bf16_le(&replay_stage.gate_proj),
                        );
                        let k_proj_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.k_proj),
                            &decode_bf16_le(&replay_stage.k_proj),
                        );
                        let v_proj_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.v_proj),
                            &decode_bf16_le(&replay_stage.v_proj),
                        );
                        let q_rope_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.q_rope),
                            &decode_bf16_le(&replay_stage.q_rope),
                        );
                        let k_rope_delta = validate::max_abs_delta(
                            &decode_bf16_le(&native_stage.k_rope),
                            &decode_bf16_le(&replay_stage.k_rope),
                        );
                        eprintln!(
                            "[gpu-validate-full-attn] layer={} input_delta={input_delta:.6} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} k_rope_delta={k_rope_delta:.6}",
                            layer_idx
                        );
                    }
                    engine.rebuild_prefill_state(&full_token_ids, false)?;
                } else {
                    eprintln!("[gpu-validate-layer] no layer exceeded hidden_delta threshold");
                }
                gpu_dumped_layer_trace = true;
            }
        }
        next_token = native_next;
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
    if gpu_validate_enabled {
        eprintln!("[gpu-validate] max_delta={gpu_max_delta:.4}");
    }
    if cli.emit_stage_timings {
        print_stage_timings(stage_totals, generated.len().saturating_sub(1));
    }

    Ok(())
}

fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

fn trace_llama31_prefill_layers(
    engine: &mut DecodeEngine,
    prefill: &crate::prefill_engine::PrefillResult,
    oracle: &oracle_mod::OracleOutput,
) -> Result<()> {
    if let (
        Some(native_attn_trace),
        Some(native_post_norm_trace),
        Some(native_mlp_out_trace),
        Some(native_layer_trace),
        Some(oracle_attn_trace),
        Some(oracle_post_norm_trace),
        Some(oracle_mlp_out_trace),
        Some(oracle_layer_trace),
    ) = (
        prefill.layer_attn_trace.as_ref(),
        prefill.layer_post_attn_norm_trace.as_ref(),
        prefill.layer_mlp_out_trace.as_ref(),
        prefill.layer_hidden_trace.as_ref(),
        oracle.layer_attn_residual_states.as_ref(),
        oracle.layer_post_attn_norm_states.as_ref(),
        oracle.layer_mlp_outputs.as_ref(),
        oracle.layer_hidden_states.as_ref(),
    ) {
        let b64 = base64::engine::general_purpose::STANDARD;
        let oracle_kv = oracle.kv_caches.as_ref();
        let mut first_bad = None;
        for layer in 0..native_layer_trace.len().min(oracle_layer_trace.len()) {
            let oracle_attn_bytes = b64
                .decode(&oracle_attn_trace[layer])
                .map_err(|e| anyhow!("decode oracle layer_attn_residual_states[{layer}]: {e}"))?;
            let oracle_post_norm_bytes = b64
                .decode(&oracle_post_norm_trace[layer])
                .map_err(|e| anyhow!("decode oracle layer_post_attn_norm_states[{layer}]: {e}"))?;
            let oracle_mlp_out_bytes = b64
                .decode(&oracle_mlp_out_trace[layer])
                .map_err(|e| anyhow!("decode oracle layer_mlp_outputs[{layer}]: {e}"))?;
            let oracle_layer_bytes = b64
                .decode(&oracle_layer_trace[layer])
                .map_err(|e| anyhow!("decode oracle layer_hidden_states[{layer}]: {e}"))?;
            let attn_delta = validate::max_abs_delta(
                &decode_bf16_le(&native_attn_trace[layer]),
                &decode_bf16_le(&oracle_attn_bytes),
            );
            let post_norm_delta = validate::max_abs_delta(
                &decode_bf16_le(&native_post_norm_trace[layer]),
                &decode_bf16_le(&oracle_post_norm_bytes),
            );
            let mlp_out_delta = validate::max_abs_delta(
                &decode_bf16_le(&native_mlp_out_trace[layer]),
                &decode_bf16_le(&oracle_mlp_out_bytes),
            );
            let layer_delta = validate::max_abs_delta(
                &decode_bf16_le(&native_layer_trace[layer]),
                &decode_bf16_le(&oracle_layer_bytes),
            );
            let state_delta = match (
                engine.full_attention_prefix_cache_bf16_host(layer, 0),
                oracle_kv.and_then(|caches| caches.iter().find(|kv| kv.layer == layer)),
            ) {
                (Ok((native_k, native_v, _)), Some(oracle_kv)) => {
                    let oracle_k = b64
                        .decode(&oracle_kv.k)
                        .map_err(|e| anyhow!("decode oracle kv k[{layer}]: {e}"))?;
                    let oracle_v = b64
                        .decode(&oracle_kv.v)
                        .map_err(|e| anyhow!("decode oracle kv v[{layer}]: {e}"))?;
                    format!(
                        " kv_k_delta={:.4} kv_v_delta={:.4}",
                        validate::max_abs_delta(&decode_bf16_le(&native_k), &decode_bf16_le(&oracle_k)),
                        validate::max_abs_delta(&decode_bf16_le(&native_v), &decode_bf16_le(&oracle_v)),
                    )
                }
                _ => String::new(),
            };
            if first_bad.is_none() && layer_delta > 0.5 {
                first_bad = Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta));
            }
            eprintln!(
                "[trace-prefill] layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}{state_delta}"
            );
        }
        if let Some((layer, attn_delta, post_norm_delta, mlp_out_delta, layer_delta)) = first_bad {
            eprintln!(
                "[trace-prefill] first_bad_layer={layer} attn_delta={attn_delta:.4} post_norm_delta={post_norm_delta:.4} mlp_out_delta={mlp_out_delta:.4} layer_delta={layer_delta:.4}"
            );
        } else {
            eprintln!("[trace-prefill] no layer exceeded delta threshold");
        }
    } else {
        eprintln!("[trace-prefill] missing native or oracle layer trace data");
    }
    Ok(())
}

fn trace_llama31_oracle_prefill_layer(
    engine: &mut DecodeEngine,
    trace_layer: usize,
    prompt_ids: &[u32],
    oracle_script: &Path,
    model_id: &str,
    oracle_dtype: &str,
    oracle_device: &str,
    oracle_full: &oracle_mod::OracleOutput,
) -> Result<()> {
    let row_bytes = engine.weights().config.hidden_size * 2;
    let prefix_ids = &prompt_ids[..prompt_ids.len() - 1];
    let prefix_oracle = if prefix_ids.is_empty() {
        None
    } else {
        Some(oracle_mod::run_oracle(
            oracle_script,
            model_id,
            prefix_ids,
            1,
            oracle_dtype,
            oracle_device,
            true,
            engine.weights().is_int8,
            None,
            Some(trace_layer),
        )?)
    };

    engine.reset()?;
    if let Some(prefix_oracle) = prefix_oracle.as_ref() {
        engine.load_prefill_state(prefix_oracle)?;
    }

    let b64 = base64::engine::general_purpose::STANDARD;
    let last_row = |bytes: Vec<u8>, label: &str| -> Result<Vec<u8>> {
        anyhow::ensure!(
            bytes.len() % row_bytes == 0,
            "{label} bytes {} not divisible by row_bytes {}",
            bytes.len(),
            row_bytes,
        );
        if bytes.len() == row_bytes {
            return Ok(bytes);
        }
        let start = bytes.len() - row_bytes;
        Ok(bytes[start..].to_vec())
    };
    let oracle_inputs = oracle_full
        .layer_hidden_states
        .as_ref()
        .ok_or_else(|| anyhow!("oracle output missing layer_hidden_states"))?;
    let oracle_attn = oracle_full
        .layer_attn_residual_states
        .as_ref()
        .ok_or_else(|| anyhow!("oracle output missing layer_attn_residual_states"))?;
    let oracle_post = oracle_full
        .layer_post_attn_norm_states
        .as_ref()
        .ok_or_else(|| anyhow!("oracle output missing layer_post_attn_norm_states"))?;
    let oracle_mlp = oracle_full
        .layer_mlp_outputs
        .as_ref()
        .ok_or_else(|| anyhow!("oracle output missing layer_mlp_outputs"))?;

    let oracle_input_bytes = if trace_layer == 0 {
        last_row(
            b64.decode(
                oracle_full
                    .traced_full_attn_input
                    .as_ref()
                    .ok_or_else(|| anyhow!("oracle output missing traced_full_attn_input"))?,
            )
            .map_err(|e| anyhow!("decode oracle input hidden for layer 0: {e}"))?,
            "oracle input hidden",
        )?
    } else {
        last_row(
            b64.decode(
                oracle_inputs
                    .get(trace_layer - 1)
                    .ok_or_else(|| anyhow!("oracle layer_hidden_states missing layer {}", trace_layer - 1))?,
            )
            .map_err(|e| anyhow!("decode oracle input hidden for layer {trace_layer}: {e}"))?,
            "oracle input hidden",
        )?
    };
    let oracle_attn_bytes = last_row(
        b64.decode(
            oracle_attn
                .get(trace_layer)
                .ok_or_else(|| anyhow!("oracle layer_attn_residual_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow!("decode oracle attn for layer {trace_layer}: {e}"))?,
        "oracle attn",
    )?;
    let oracle_post_bytes = last_row(
        b64.decode(
            oracle_post
                .get(trace_layer)
                .ok_or_else(|| anyhow!("oracle layer_post_attn_norm_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow!("decode oracle post-norm for layer {trace_layer}: {e}"))?,
        "oracle post-norm",
    )?;
    let oracle_mlp_bytes = last_row(
        b64.decode(
            oracle_mlp
                .get(trace_layer)
                .ok_or_else(|| anyhow!("oracle layer_mlp_outputs missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow!("decode oracle mlp for layer {trace_layer}: {e}"))?,
        "oracle mlp",
    )?;
    let oracle_hidden_bytes = last_row(
        b64.decode(
            oracle_inputs
                .get(trace_layer)
                .ok_or_else(|| anyhow!("oracle layer_hidden_states missing layer {trace_layer}"))?,
        )
        .map_err(|e| anyhow!("decode oracle hidden for layer {trace_layer}: {e}"))?,
        "oracle hidden",
    )?;

    engine.set_hidden_from_bytes(&oracle_input_bytes)?;
    let trace = engine.component_trace_full_layer_from_current_hidden_with_seqlen(
        trace_layer,
        prefix_ids.len(),
    )?;
    let attn_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.attn_hidden),
        &decode_bf16_le(&oracle_attn_bytes),
    );
    let post_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.post_attn_norm),
        &decode_bf16_le(&oracle_post_bytes),
    );
    let mlp_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.mlp_out),
        &decode_bf16_le(&oracle_mlp_bytes),
    );
    let hidden_delta = validate::max_abs_delta(
        &decode_bf16_le(&trace.layer_hidden),
        &decode_bf16_le(&oracle_hidden_bytes),
    );
    eprintln!(
        "[trace-oracle-prefill-layer] layer={trace_layer} attn_delta={attn_delta:.6} post_norm_delta={post_delta:.6} mlp_delta={mlp_delta:.6} hidden_delta={hidden_delta:.6}"
    );

    if oracle_full.traced_mlp_down.is_some() {
        let decode_opt_bf16 = |field: &Option<String>, label: &str| -> Result<Vec<f32>> {
            let bytes = b64
                .decode(field.as_ref().ok_or_else(|| anyhow!("oracle output missing {label}"))?)
                .map_err(|e| anyhow!("decode oracle {label}: {e}"))?;
            Ok(decode_bf16_le(&bytes))
        };
        let oracle_mlp_gate = decode_opt_bf16(&oracle_full.traced_mlp_gate, "traced_mlp_gate")?;
        let oracle_mlp_up = decode_opt_bf16(&oracle_full.traced_mlp_up, "traced_mlp_up")?;
        let oracle_mlp_swiglu =
            decode_opt_bf16(&oracle_full.traced_mlp_swiglu, "traced_mlp_swiglu")?;
        let oracle_mlp_down = decode_opt_bf16(&oracle_full.traced_mlp_down, "traced_mlp_down")?;
        let mlp_trace = engine.component_trace_mlp_from_post_attn_norm(
            trace_layer,
            &oracle_attn_bytes,
            &oracle_post_bytes,
        )?;
        let gate_delta =
            validate::max_abs_delta(&decode_bf16_le(&mlp_trace.gate), &oracle_mlp_gate);
        let up_delta = validate::max_abs_delta(&decode_bf16_le(&mlp_trace.up), &oracle_mlp_up);
        let swiglu_delta =
            validate::max_abs_delta(&decode_bf16_le(&mlp_trace.swiglu), &oracle_mlp_swiglu);
        let down_delta =
            validate::max_abs_delta(&decode_bf16_le(&mlp_trace.down), &oracle_mlp_down);
        eprintln!(
            "[trace-oracle-mlp] layer={trace_layer} gate_delta={gate_delta:.6} up_delta={up_delta:.6} swiglu_delta={swiglu_delta:.6} down_delta={down_delta:.6}"
        );
    }

    if engine.weights().config.is_full_attention(trace_layer)
        && oracle_full.traced_full_attn_layer == Some(trace_layer)
    {
        let decode_opt_bf16 = |field: &Option<String>, label: &str| -> Result<Vec<f32>> {
            let bytes = b64
                .decode(field.as_ref().ok_or_else(|| anyhow!("oracle output missing {label}"))?)
                .map_err(|e| anyhow!("decode oracle {label}: {e}"))?;
            Ok(decode_bf16_le(&bytes))
        };
        let oracle_normed = decode_opt_bf16(&oracle_full.traced_full_attn_normed, "traced_full_attn_normed")?;
        let oracle_q_proj = decode_opt_bf16(&oracle_full.traced_full_attn_q_proj, "traced_full_attn_q_proj")?;
        let oracle_gate = decode_opt_bf16(&oracle_full.traced_full_attn_gate_proj, "traced_full_attn_gate_proj")?;
        let oracle_k_proj = decode_opt_bf16(&oracle_full.traced_full_attn_k_proj, "traced_full_attn_k_proj")?;
        let oracle_v_proj = decode_opt_bf16(&oracle_full.traced_full_attn_v_proj, "traced_full_attn_v_proj")?;
        let oracle_q_rope = decode_opt_bf16(&oracle_full.traced_full_attn_q_rope, "traced_full_attn_q_rope")?;
        let oracle_k_rope = decode_opt_bf16(&oracle_full.traced_full_attn_k_rope, "traced_full_attn_k_rope")?;
        let oracle_pre_gate = decode_opt_bf16(&oracle_full.traced_full_attn_pre_gate, "traced_full_attn_pre_gate")?;
        let oracle_gated = decode_opt_bf16(&oracle_full.traced_full_attn_gated, "traced_full_attn_gated")?;

        let stage = engine.trace_full_attention_stages_from_hidden(
            trace_layer,
            &oracle_input_bytes,
            prefix_ids.len(),
        )?;
        let stage_out = engine.trace_full_attention_layer_output_from_hidden_current_state(
            trace_layer,
            0,
            &oracle_input_bytes,
            prefix_ids.len(),
        )?;
        let normed_delta = validate::max_abs_delta(&decode_bf16_le(&stage.normed), &oracle_normed);
        let q_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_proj), &oracle_q_proj);
        let gate_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.gate_proj), &oracle_gate);
        let k_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_proj), &oracle_k_proj);
        let v_proj_delta = validate::max_abs_delta(&decode_bf16_le(&stage.v_proj), &oracle_v_proj);
        let q_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.q_rope), &oracle_q_rope);
        let k_rope_delta = validate::max_abs_delta(&decode_bf16_le(&stage.k_rope), &oracle_k_rope);
        let pre_gate_delta =
            validate::max_abs_delta(&decode_bf16_le(&stage_out.pre_gate), &oracle_pre_gate);
        let gated_delta = validate::max_abs_delta(&decode_bf16_le(&stage_out.gated), &oracle_gated);
        eprintln!(
            "[trace-oracle-full-attn] layer={trace_layer} normed_delta={normed_delta:.6} q_proj_delta={q_proj_delta:.6} gate_proj_delta={gate_proj_delta:.6} k_proj_delta={k_proj_delta:.6} v_proj_delta={v_proj_delta:.6} q_rope_delta={q_rope_delta:.6} k_rope_delta={k_rope_delta:.6} pre_gate_delta={pre_gate_delta:.6} gated_delta={gated_delta:.6}"
        );
    }

    Ok(())
}
