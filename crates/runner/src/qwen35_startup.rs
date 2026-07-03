use std::path::Path;

use anyhow::Result;

use crate::bakes::{effective_quant_profile, validate_effective_flm_source_model};
use crate::flm_model_source::{is_flm_model_path, FlmModelSource, FlmModelSourceOptions};
use crate::registry::{Backend, GpuArch, ModelVariant, Qwen35KernelParams};
use crate::Cli;

pub(crate) struct Qwen35Startup {
    pub(crate) text_config: qwen35::config::TextConfig,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) context_tokens: usize,
}

pub(crate) struct Qwen35Policy {
    pub(crate) trace_kv_cache_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenTokenizerSource<'a> {
    Flm(&'a Path),
    TokenizerJson(&'a Path),
}

pub(crate) fn load_qwen35_startup(cli: &Cli) -> Result<Qwen35Startup> {
    let flm_source = open_flm_startup_source(cli)?;
    let config = load_qwen35_config(cli, flm_source.as_ref())?;
    let text_config = config.text_config;
    eprintln!(
        "[config] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={}",
        text_config.hidden_size,
        text_config.num_hidden_layers,
        text_config.vocab_size,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    );

    let tokenizer = load_qwen_tokenizer(cli, flm_source.as_ref())?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        anyhow::bail!("empty prompt after tokenization");
    }

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);

    Ok(Qwen35Startup {
        text_config,
        tokenizer,
        prompt_ids,
        context_tokens,
    })
}

fn open_flm_startup_source(cli: &Cli) -> Result<Option<FlmModelSource>> {
    let Some(path) = flm_config_path(cli) else {
        return Ok(None);
    };
    FlmModelSource::open_with_options(path, flm_startup_open_options(cli)?)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("opening FLM startup source {}: {e}", path.display()))
}

fn flm_startup_open_options(cli: &Cli) -> Result<FlmModelSourceOptions> {
    let profile = effective_quant_profile(cli)?;
    Ok(FlmModelSourceOptions {
        int4_runtime: profile.is_native_int4_runtime(),
        verify_block_hashes: false,
    })
}

fn load_qwen35_config(
    cli: &Cli,
    flm_source: Option<&FlmModelSource>,
) -> Result<qwen35::config::Config> {
    if let Some(source) = flm_source {
        return load_flm_qwen35_config(source);
    }

    qwen35::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))
}

fn flm_config_path(cli: &Cli) -> Option<&Path> {
    cli.flm_file
        .as_deref()
        .or_else(|| is_flm_model_path(&cli.model_dir).then_some(cli.model_dir.as_path()))
}

fn load_flm_qwen35_config(source: &FlmModelSource) -> Result<qwen35::config::Config> {
    eprintln!(
        "[config] loading FLM runtime descriptor at {}",
        source.path.display()
    );
    source
        .qwen_config()
        .map_err(|e| anyhow::anyhow!("loading FLM Qwen config: {e}"))
}

pub(crate) fn qwen_tokenizer_source(cli: &Cli) -> QwenTokenizerSource<'_> {
    if let Some(path) = flm_config_path(cli) {
        QwenTokenizerSource::Flm(path)
    } else {
        QwenTokenizerSource::TokenizerJson(&cli.model_dir)
    }
}

fn load_qwen_tokenizer(
    cli: &Cli,
    flm_source: Option<&FlmModelSource>,
) -> Result<tokenizers::Tokenizer> {
    match qwen_tokenizer_source(cli) {
        QwenTokenizerSource::Flm(path) => {
            eprintln!(
                "[tokenizer] loading FLM tokenizer assets at {}",
                path.display()
            );
            let source = flm_source.ok_or_else(|| {
                anyhow::anyhow!(
                    "internal error: FLM tokenizer source {} was not opened",
                    path.display()
                )
            })?;
            source.qwen_tokenizer()
        }
        QwenTokenizerSource::TokenizerJson(model_dir) => {
            let tokenizer_path = model_dir.join("tokenizer.json");
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use super::{
        flm_config_path, qwen_tokenizer_source, validate_qwen35_startup, QwenTokenizerSource,
    };
    use crate::registry::{Backend, GpuArch, ModelVariant, Qwen35KernelParams};
    use crate::Cli;

    fn cli(model_dir: &str, extra: &[&str]) -> Cli {
        let mut args = vec!["supersonic", "--model-dir", model_dir, "--dry-run"];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    fn qwen35_params() -> Qwen35KernelParams {
        Qwen35KernelParams {
            proj_buf_floats: 0,
            attn_scratch_floats: 0,
            weight_prefix: "",
            kv_chunk_size: 1,
            use_4b_kernel: false,
        }
    }

    #[test]
    fn flm_file_is_authoritative_for_qwen_config_even_with_model_dir_metadata() {
        let cli = cli("/tmp/model-with-config", &["--flm-file", "/tmp/model.flm"]);

        assert_eq!(flm_config_path(&cli), Some(Path::new("/tmp/model.flm")));
    }

    #[test]
    fn flm_model_dir_is_authoritative_for_qwen_config() {
        let cli = cli("/tmp/model.flm", &[]);

        assert_eq!(flm_config_path(&cli), Some(Path::new("/tmp/model.flm")));
    }

    #[test]
    fn effective_flm_source_selects_flm_native_tokenizer_without_tokenizer_json() {
        let flm_model_cli = cli("/tmp/model.flm", &[]);

        assert_eq!(
            qwen_tokenizer_source(&flm_model_cli),
            QwenTokenizerSource::Flm(Path::new("/tmp/model.flm"))
        );

        let flm_file_cli = cli("/tmp/model-dir", &["--flm-file", "/tmp/model.flm"]);
        assert_eq!(
            qwen_tokenizer_source(&flm_file_cli),
            QwenTokenizerSource::Flm(Path::new("/tmp/model.flm"))
        );
    }

    #[test]
    fn flm_file_requires_qwen36_27b_model_variant() {
        let cli = cli("/tmp/model-dir", &["--flm-file", "/tmp/model.flm"]);
        let params = qwen35_params();

        let err = match validate_qwen35_startup(
            &cli,
            &ModelVariant::Qwen3_5_0_8B,
            &params,
            Backend::Hip,
            &GpuArch::Gfx1100,
            false,
        ) {
            Ok(_) => panic!("expected FLM model-variant validation error"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("--flm-file"), "{err}");
        assert!(err.contains("qwen3.6-27b"), "{err}");
        assert!(err.contains("qwen3.5-0.8b"), "{err}");
    }
}

pub(crate) fn validate_qwen35_startup(
    cli: &Cli,
    model_variant: &ModelVariant,
    params: &Qwen35KernelParams,
    backend: Backend,
    registry_arch: &GpuArch,
    q4km_like: bool,
) -> Result<Qwen35Policy> {
    validate_effective_flm_source_model(cli, model_variant)?;

    if cli.trace_prefill_layers && !cli.validate {
        anyhow::bail!("--trace-prefill-layers requires --validate");
    }
    if cli.trace_oracle_prefill_layer.is_some() && !cli.validate {
        anyhow::bail!("--trace-oracle-prefill-layer requires --validate");
    }
    if cli.trace_kv_fp8_cache && !cli.kv_fp8 {
        anyhow::bail!("--trace-kv-fp8-cache requires --kv-fp8");
    }
    let trace_kv_cache_enabled = cli.trace_kv_cache || cli.trace_kv_fp8_cache;
    if cli.force_kernel_decode && cli.force_component_decode {
        anyhow::bail!("Choose at most one of --force-kernel-decode or --force-component-decode");
    }
    if cli.trace_component_input_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-input-layer requires --force-component-decode");
    }
    if cli.trace_component_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-layer requires --force-component-decode");
    }
    if cli.trace_component_linear_state_layer.is_some() && !cli.force_component_decode {
        anyhow::bail!("--trace-component-linear-state-layer requires --force-component-decode");
    }
    if cli.trace_persistent_input_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!("--trace-persistent-input-layer requires the real 4B persistent kernel path");
    }
    if cli.trace_persistent_linear_state_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode || cli.kv_fp8))
    {
        anyhow::bail!(
            "--trace-persistent-linear-state-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_full_attn_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode))
    {
        anyhow::bail!(
            "--trace-persistent-full-attn-layer requires the real 4B persistent kernel path"
        );
    }
    if cli.trace_persistent_linear_layer.is_some()
        && !(params.use_4b_kernel
            && !cli.force_component_decode
            && (cli.batch_size > 1 || cli.force_kernel_decode)
            && !cli.kv_fp8)
    {
        anyhow::bail!(
            "--trace-persistent-linear-layer requires the real 4B persistent BF16 kernel path"
        );
    }

    if backend == Backend::Cuda {
        let qwen35_sm86 = matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B
                | ModelVariant::Qwen3_5_2B
                | ModelVariant::Qwen3_5_4B
                | ModelVariant::Qwen3_5_9B
        ) && *registry_arch == GpuArch::Sm86;
        if cli.int4 && !qwen35_sm86 {
            anyhow::bail!("CUDA --int4 currently supports only Qwen3.5 on sm86");
        }
        if cli.fp8_runtime && !(qwen35_sm86 || *model_variant == ModelVariant::Qwen3_6_27B) {
            anyhow::bail!(
                "CUDA --fp8-runtime currently supports only Qwen3.5 and qwen3.6-27b on sm86"
            );
        }
        if cli.kv_fp8 {
            if !qwen35_sm86 {
                anyhow::bail!("CUDA --kv-fp8 currently supports only Qwen3.5 on sm86");
            }
            if std::env::var_os("SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR").is_none() {
                std::env::set_var("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW", "128");
                eprintln!(
                    "[cuda] KV-FP8 BF16 sidecar window capped to the most recent 128 tokens on CUDA; \
                     set SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR=1 \
                     to restore full-prefix debug sidecar coverage"
                );
            }
        }
    } else if backend == Backend::Metal {
        if !matches!(
            model_variant,
            ModelVariant::Qwen3_5_0_8B
                | ModelVariant::Qwen3_5_2B
                | ModelVariant::Qwen3_5_4B
                | ModelVariant::Qwen3_5_9B
        ) {
            anyhow::bail!(
                "Metal only supports --model qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, or qwen3.5-9b"
            );
        }
        if q4km_like {
            anyhow::bail!("Metal does not support --q4km/--q4km-gptq on Qwen3.5 yet");
        }
        if cli.fp8_runtime {
            anyhow::bail!("Metal does not support --fp8-runtime on Qwen3.5 yet");
        }
        if cli.kv_fp8 {
            anyhow::bail!("Metal does not support --kv-fp8 on Qwen3.5 yet");
        }
        if cli.batch_size != 1 {
            anyhow::bail!("Metal only supports --batch-size 1");
        }
        if cli.force_kernel_decode || cli.force_component_decode {
            anyhow::bail!(
                "Metal does not support --force-kernel-decode or --force-component-decode"
            );
        }
    }

    Ok(Qwen35Policy {
        trace_kv_cache_enabled,
    })
}
