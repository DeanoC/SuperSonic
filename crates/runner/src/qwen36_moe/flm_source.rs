use anyhow::Result;
use supersonic_runtime::qwen36_moe::source::{
    Qwen36MoeDirectProfile, Qwen36MoeSource, Qwen36MoeSourceOpenTimings, Qwen36WeightMode,
};

use crate::bakes::{
    effective_flm_source, flm_source_open_options, validate_effective_flm_source_model,
    validate_flm_weight_source_options,
};
use crate::flm_model_source::FlmModelSource;
use crate::qwen36_moe_cli::layers::Qwen36WeightMode as RunnerQwen36WeightMode;
use crate::registry::ModelVariant;
use crate::Cli;

pub(crate) struct Qwen36MoeFlmSource {
    pub(crate) source: FlmModelSource,
    pub(crate) config: qwen36_moe::config::Config,
    pub(crate) weight_mode: RunnerQwen36WeightMode,
    pub(crate) weight_mode_label: &'static str,
    pub(crate) direct_profile: Qwen36MoeFlmDirectProfile,
    pub(crate) timings: Qwen36MoeFlmSourceOpenTimings,
}

impl Qwen36MoeFlmSource {
    pub(crate) fn load_tokenizer_timed(
        &self,
    ) -> Result<crate::flm_tokenizer::QwenBpeTokenizerLoad> {
        self.source.qwen_tokenizer_timed()
    }
}

pub(crate) type Qwen36MoeFlmDirectProfile = Qwen36MoeDirectProfile;
pub(crate) type Qwen36MoeFlmSourceOpenTimings = Qwen36MoeSourceOpenTimings;

fn runner_weight_mode(mode: Qwen36WeightMode) -> RunnerQwen36WeightMode {
    match mode {
        Qwen36WeightMode::Bf16 => RunnerQwen36WeightMode::Bf16,
        Qwen36WeightMode::Int4 => RunnerQwen36WeightMode::Int4,
    }
}

pub(crate) fn open_qwen36_moe_flm_source(cli: &Cli) -> Result<Option<Qwen36MoeFlmSource>> {
    let Some(path) = effective_flm_source(cli) else {
        return Ok(None);
    };
    validate_effective_flm_source_model(cli, &ModelVariant::Qwen3_6_35B_A3B)?;
    validate_flm_weight_source_options(cli, crate::policy::q4km_like(cli))?;
    let options = flm_source_open_options(cli)?;
    eprintln!(
        "[flm] opening model source at {}{}{}",
        path.display(),
        if options.int4_runtime {
            " (FLM logical INT4 aliases enabled)"
        } else {
            ""
        },
        if options.verify_block_hashes {
            " (BLAKE3 hash verification enabled)"
        } else {
            ""
        }
    );
    eprintln!("[qwen36-moe] loading config from FLM runtime descriptor");
    let runtime_source = Qwen36MoeSource::open(path, options)?;
    eprintln!(
        "[qwen36-moe] FLM weight mode: {}",
        runtime_source.weight_mode.label()
    );
    eprintln!(
        "[qwen36-moe] FLM direct plans: {}",
        runtime_source.direct_profile
    );

    Ok(Some(Qwen36MoeFlmSource {
        source: runtime_source.source,
        config: runtime_source.config,
        weight_mode: runner_weight_mode(runtime_source.weight_mode),
        weight_mode_label: runtime_source.weight_mode.label(),
        direct_profile: runtime_source.direct_profile,
        timings: runtime_source.timings,
    }))
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn cli(extra: &[&str]) -> Cli {
        let mut args = vec![
            "supersonic",
            "--model",
            "qwen3.6-35b-a3b",
            "--model-dir",
            "/tmp/qwen36-moe-missing.flm",
            "--dry-run",
        ];
        args.extend_from_slice(extra);
        Cli::parse_from(args)
    }

    #[test]
    fn rejects_incompatible_quant_flags_before_opening_flm_source() {
        for flag in ["--q4km", "--q4km-gptq", "--int8"] {
            let err = match open_qwen36_moe_flm_source(&cli(&[flag])) {
                Ok(_) => panic!("incompatible FLM quant flag {flag} should fail before file open"),
                Err(err) => err.to_string(),
            };

            assert!(err.contains("FLM"), "{err}");
            assert!(
                err.contains(flag) || err.contains("--q4km/--q4km-gptq"),
                "error should identify {flag}: {err}"
            );
            assert!(
                !err.contains("opening Qwen3.6 MoE FLM source"),
                "validation should run before source open: {err}"
            );
        }
    }
}
