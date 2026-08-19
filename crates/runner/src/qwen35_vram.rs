use anyhow::Result;

use crate::registry::VramBudget;
use crate::Cli;

pub(crate) fn check_qwen35_vram(
    cli: &Cli,
    text_config: &qwen35::config::TextConfig,
    vram: &VramBudget,
    context_tokens: usize,
    kv_chunk_size: usize,
    total_vram: u64,
) -> Result<()> {
    let kv_estimate = qwen35_kv_cache_vram_estimate(
        text_config,
        context_tokens,
        kv_chunk_size,
        cli.kv_fp8,
        qwen35::state::kv_fp8_bf16_sidecar_enabled(),
    );
    let effective_fixed = effective_fixed_vram(
        vram.fixed_bytes,
        crate::bakes::effective_quant_profile(cli)?,
    );
    let kv_bytes = kv_estimate.total_bytes;
    let estimated_vram = ((effective_fixed + kv_bytes) as f64 * vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(effective_fixed),
        gib(kv_bytes),
        context_tokens,
        gib(total_vram),
    );
    eprintln!(
        "[kv-cache] mode={} capacity_tokens={} data={:.2}GiB scales={:.2}GiB sidecar={:.2}GiB total={:.2}GiB",
        kv_estimate.mode,
        kv_estimate.capacity_tokens,
        gib(kv_estimate.data_bytes),
        gib(kv_estimate.scale_bytes),
        gib(kv_estimate.sidecar_bytes),
        gib(kv_estimate.total_bytes),
    );
    if estimated_vram > total_vram {
        anyhow::bail!(
            "Insufficient VRAM for {context_tokens}-token context: \
             need ~{:.2}GiB (weights {:.2}GiB + KV cache {:.2}GiB), \
             GPU has {:.1}GiB. Reduce --context-size or --max-new-tokens.",
            gib(estimated_vram),
            gib(effective_fixed),
            gib(kv_bytes),
            gib(total_vram),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35KvCacheVramEstimate {
    mode: &'static str,
    capacity_tokens: usize,
    data_bytes: u64,
    scale_bytes: u64,
    sidecar_bytes: u64,
    total_bytes: u64,
}

fn round_up_to_chunk(tokens: usize, kv_chunk_size: usize) -> usize {
    let chunk = kv_chunk_size.max(1);
    tokens.max(1).div_ceil(chunk) * chunk
}

fn qwen35_kv_cache_vram_estimate(
    text_config: &qwen35::config::TextConfig,
    context_tokens: usize,
    kv_chunk_size: usize,
    kv_fp8: bool,
    kv_fp8_bf16_sidecar: bool,
) -> Qwen35KvCacheVramEstimate {
    let capacity_tokens = round_up_to_chunk(context_tokens, kv_chunk_size);
    let bf16_kv_bytes = text_config.kv_bytes_per_token(gpu_hal::ScalarType::BF16.size_in_bytes())
        * capacity_tokens as u64;
    if !kv_fp8 {
        return Qwen35KvCacheVramEstimate {
            mode: "bf16",
            capacity_tokens,
            data_bytes: bf16_kv_bytes,
            scale_bytes: 0,
            sidecar_bytes: 0,
            total_bytes: bf16_kv_bytes,
        };
    }

    let data_bytes = text_config.kv_bytes_per_token(1) * capacity_tokens as u64;
    let scale_bytes = (2
        * text_config.num_full_attention_layers()
        * text_config.num_key_value_heads
        * capacity_tokens
        * gpu_hal::ScalarType::F32.size_in_bytes()) as u64;
    let sidecar_bytes = if kv_fp8_bf16_sidecar {
        bf16_kv_bytes
    } else {
        0
    };
    Qwen35KvCacheVramEstimate {
        mode: if kv_fp8_bf16_sidecar {
            "fp8+bf16-sidecar"
        } else {
            "fp8"
        },
        capacity_tokens,
        data_bytes,
        scale_bytes,
        sidecar_bytes,
        total_bytes: data_bytes + scale_bytes + sidecar_bytes,
    }
}

fn effective_fixed_vram(fixed_bytes: u64, profile: model_store::manifest::QuantProfile) -> u64 {
    use model_store::manifest::QuantProfile;
    match profile {
        QuantProfile::Q4Km => (fixed_bytes as f64 * 0.30) as u64,
        profile if profile.is_native_int4_runtime() => {
            // INT4: weights ~= fixed * 0.9, scratch ~= fixed * 0.1.
            // INT4 weights = weights / 4 + ~5% scale/zero overhead.
            (fixed_bytes as f64 * 0.37) as u64
        }
        QuantProfile::Fp8Native => (fixed_bytes as f64 * 0.55) as u64,
        QuantProfile::Higgs4 | QuantProfile::QuipE8 | QuantProfile::QtipTrellis2 => {
            (fixed_bytes as f64 * 0.32) as u64
        }
        _ => fixed_bytes,
    }
}

#[cfg(test)]
mod tests {
    use model_store::manifest::QuantProfile;

    use super::{effective_fixed_vram, qwen35_kv_cache_vram_estimate};

    #[test]
    fn q4km_gptq_uses_int4_vram_estimate() {
        assert_eq!(effective_fixed_vram(100, QuantProfile::Q4KmGptq), 37);
    }

    #[test]
    fn qwen35_kv_vram_rounds_context_tokens_to_chunk_capacity() {
        let config = sample_text_config();

        let estimate = qwen35_kv_cache_vram_estimate(&config, 129, 128, false, false);

        assert_eq!(estimate.mode, "bf16");
        assert_eq!(estimate.capacity_tokens, 256);
        assert_eq!(estimate.data_bytes, config.kv_bytes_per_token(2) * 256);
        assert_eq!(estimate.total_bytes, estimate.data_bytes);
    }

    #[test]
    fn qwen35_kv_vram_does_not_add_an_extra_token_at_chunk_boundary() {
        let config = sample_text_config();

        let estimate = qwen35_kv_cache_vram_estimate(&config, 128, 128, false, false);

        assert_eq!(estimate.capacity_tokens, 128);
        assert_eq!(estimate.total_bytes, config.kv_bytes_per_token(2) * 128);
    }

    #[test]
    fn qwen35_kv_fp8_vram_includes_scales_and_optional_sidecar() {
        let config = sample_text_config();

        let fp8 = qwen35_kv_cache_vram_estimate(&config, 65, 64, true, false);
        let sidecar = qwen35_kv_cache_vram_estimate(&config, 65, 64, true, true);

        assert_eq!(fp8.mode, "fp8");
        assert_eq!(fp8.capacity_tokens, 128);
        assert_eq!(fp8.data_bytes, config.kv_bytes_per_token(1) * 128);
        assert_eq!(
            fp8.scale_bytes,
            (2 * config.num_full_attention_layers() * config.num_key_value_heads * 128 * 4) as u64
        );
        assert_eq!(fp8.sidecar_bytes, 0);
        assert_eq!(
            fp8.total_bytes,
            fp8.data_bytes + fp8.scale_bytes + fp8.sidecar_bytes
        );
        assert_eq!(sidecar.mode, "fp8+bf16-sidecar");
        assert_eq!(sidecar.data_bytes, fp8.data_bytes);
        assert_eq!(sidecar.scale_bytes, fp8.scale_bytes);
        assert_eq!(sidecar.sidecar_bytes, config.kv_bytes_per_token(2) * 128);
        assert_eq!(
            sidecar.total_bytes,
            sidecar.data_bytes + sidecar.scale_bytes + sidecar.sidecar_bytes
        );
    }

    fn sample_text_config() -> qwen35::config::TextConfig {
        qwen35::config::TextConfig {
            vocab_size: 1024,
            hidden_size: 512,
            intermediate_size: 1024,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            hidden_act: qwen35::config::Activation::Silu,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rms_norm_add_unit_offset: true,
            tie_word_embeddings: false,
            eos_token_id: None,
            head_dim: 64,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            linear_num_key_heads: 8,
            linear_num_value_heads: 8,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            layer_types: vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ],
            rope_parameters: None,
        }
    }
}
