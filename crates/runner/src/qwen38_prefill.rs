use std::time::Instant;

use anyhow::Result;

use crate::decode_engine::DecodeEngine;
use crate::profiling::PrefillProfileScope;
use crate::Cli;
use gpu_hal::Backend;

pub(crate) struct Qwen38Prefill {
    pub(crate) final_norm: Option<Vec<u8>>,
    pub(crate) next_token: u32,
}

pub(crate) fn run_qwen38_prefill(
    cli: &Cli,
    engine: &mut DecodeEngine,
    prompt_ids: &[u32],
) -> Result<Qwen38Prefill> {
    let prefill_start = Instant::now();
    let profile = PrefillProfileScope::new(
        cli.profile_prefill,
        cli.profile_prefill_json.as_deref(),
        "qwen3.8",
        &cli.model,
        backend_label(),
        prompt_ids.len(),
    );
    if !engine.weights().gqh_headers.is_empty() {
        eprintln!(
            "[prefill] GQH GGUF: native batched prefill ({} headers)",
            engine.weights().gqh_headers.len()
        );
    }
    let result = engine.prefill_native_with_final_norm(prompt_ids)?;
    kernel_ffi::gqh::gemm_flush(engine.ordinal())?;
    let next_token = DecodeEngine::greedy_sample(&result.logits);
    eprintln!(
        "[prefill] native {} prefill done in {:.0}ms",
        backend_label(),
        prefill_start.elapsed().as_secs_f64() * 1000.0
    );
    profile.finish()?;

    Ok(Qwen38Prefill {
        final_norm: result.final_norm_trace,
        next_token,
    })
}

fn backend_label() -> &'static str {
    match gpu_hal::current_backend() {
        Backend::Hip => "HIP",
        #[cfg(supersonic_backend_metal)]
        Backend::Metal => "Metal",
    }
}
