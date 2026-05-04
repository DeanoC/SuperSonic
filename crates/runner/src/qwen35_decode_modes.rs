use std::env;

use crate::registry::{Backend, ModelVariant};
use crate::Cli;

pub(crate) struct Qwen35DecodeModes {
    pub(crate) cuda_qwen2b_replay_default: bool,
    pub(crate) metal_v2_incremental: bool,
    pub(crate) replay_decode_enabled: bool,
    pub(crate) replay_kv_fp8_enabled: bool,
    pub(crate) component_single_decode_enabled: bool,
    pub(crate) kernel_single_decode_enabled: bool,
    pub(crate) cuda_fast_greedy_enabled: bool,
    pub(crate) metal_fast_greedy_enabled: bool,
}

pub(crate) fn resolve_qwen35_decode_modes(
    cli: &Cli,
    backend: Backend,
    model_variant: &ModelVariant,
    use_4b_kernel: bool,
    gpu_validate_enabled: bool,
    oracle_output_present: bool,
    cuda_08b_hero_enabled: bool,
) -> Qwen35DecodeModes {
    let cuda_qwen2b_replay_default = backend == Backend::Cuda
        && *model_variant == ModelVariant::Qwen3_5_2B
        && cli.batch_size == 1
        && use_4b_kernel
        && !cli.kv_fp8
        && !cli.force_kernel_decode
        && !cli.force_component_decode;
    let metal_v2_incremental = backend == Backend::Metal && cli.batch_size == 1;
    let replay_decode_enabled = cli.batch_size == 1
        && !cli.force_kernel_decode
        && !cli.force_component_decode
        && !cli.kv_fp8
        && use_4b_kernel
        && (cli.force_replay_decode || cuda_qwen2b_replay_default);
    let replay_kv_fp8_enabled =
        use_4b_kernel && cli.kv_fp8 && cli.batch_size == 1 && !cli.force_kernel_decode;
    let component_single_decode_enabled =
        cli.batch_size == 1 && use_4b_kernel && cli.force_component_decode;
    let kernel_single_decode_enabled = cli.batch_size == 1
        && use_4b_kernel
        && !cli.force_replay_decode
        && !cli.force_component_decode;
    let cuda_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_CUDA_FAST_GREEDY").is_some();
    let cuda_fast_greedy_enabled = backend == Backend::Cuda
        && !use_4b_kernel
        && cli.batch_size == 1
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !cli.kv_fp8
        && !oracle_output_present
        && !cuda_08b_hero_enabled
        && !cuda_fast_greedy_disabled;
    let metal_fast_greedy_disabled = env::var_os("SUPERSONIC_DISABLE_METAL_FAST_GREEDY").is_some();
    let metal_fast_greedy_enabled = backend == Backend::Metal
        && metal_v2_incremental
        && !cli.validate
        && !gpu_validate_enabled
        && !cli.force_component_decode
        && !cli.force_kernel_decode
        && !oracle_output_present
        && !metal_fast_greedy_disabled;

    Qwen35DecodeModes {
        cuda_qwen2b_replay_default,
        metal_v2_incremental,
        replay_decode_enabled,
        replay_kv_fp8_enabled,
        component_single_decode_enabled,
        kernel_single_decode_enabled,
        cuda_fast_greedy_enabled,
        metal_fast_greedy_enabled,
    }
}

pub(crate) fn report_qwen35_decode_modes(
    cli: &Cli,
    modes: &Qwen35DecodeModes,
    use_4b_kernel: bool,
    cuda_08b_hero_enabled: bool,
) {
    if modes.metal_v2_incremental {
        if modes.metal_fast_greedy_enabled {
            eprintln!("[decode] Metal v2 incremental decode (fast-greedy: fused argmax)");
        } else {
            eprintln!("[decode] Metal v2 incremental decode");
        }
    }
    if modes.replay_decode_enabled {
        if modes.cuda_qwen2b_replay_default {
            eprintln!(
                "[decode] single-sequence CUDA qwen3.5-2b uses replayed GPU prefill for correctness"
            );
        } else {
            eprintln!("[decode] single-sequence 4B uses replayed GPU prefill for correctness");
        }
    } else if modes.replay_kv_fp8_enabled && cli.batch_size == 1 {
        eprintln!("[decode] single-sequence KV-FP8 uses replayed GPU prefill for correctness");
    } else if cli.batch_size > 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] batched KV-FP8 uses the persistent kernel path");
    } else if modes.component_single_decode_enabled {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the component decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.force_kernel_decode {
        eprintln!("[decode] WARNING: forcing single-sequence 4B onto the kernel decode path");
    } else if cli.batch_size == 1 && use_4b_kernel && cli.kv_fp8 {
        eprintln!("[decode] WARNING: single-sequence KV-FP8 uses the b=1 kernel path");
    } else if cuda_08b_hero_enabled {
        eprintln!("[decode] CUDA 0.8B sm86 hero path enabled");
    } else if modes.cuda_fast_greedy_enabled {
        eprintln!("[decode] CUDA fast greedy sampling enabled for the non-4B native decode path");
    }
}
