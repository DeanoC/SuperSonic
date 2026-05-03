use std::time::Duration;

use crate::qwen36_moe_decode::DecodeOutputs;

/// Bundles the sampling knobs for the multi-token decode loop. `temperature
/// <= 0` means greedy argmax, the deterministic default. At temperature > 0,
/// `top_k`/`top_p` filter the distribution before sampling, then `seed`
/// drives the xorshift RNG.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SamplingParams {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) seed: u64,
}

#[derive(Default)]
pub(crate) struct Qwen36StageTimingTotals {
    gen_steps: usize,
    embed: Duration,
    chain: Duration,
    lm_head: Duration,
    sample: Duration,
    detok: Duration,
    chain_full_attn_us: u64,
    chain_linear_attn_us: u64,
    chain_ffn_us: u64,
}

impl Qwen36StageTimingTotals {
    pub(crate) fn record_generation_step(
        &mut self,
        embed: Duration,
        chain: Duration,
        lm_head: Duration,
        sample: Duration,
        detok: Duration,
        outputs: &DecodeOutputs,
    ) {
        self.count_generation_step();
        self.embed += embed;
        self.chain += chain;
        self.lm_head += lm_head;
        self.sample += sample;
        self.detok += detok;
        self.record_chain_breakdown(outputs);
    }

    pub(crate) fn record_embed(&mut self, elapsed: Duration) {
        self.embed += elapsed;
    }

    pub(crate) fn record_chain(&mut self, elapsed: Duration, outputs: &DecodeOutputs) {
        self.chain += elapsed;
        self.record_chain_breakdown(outputs);
    }

    pub(crate) fn record_lm_head(&mut self, elapsed: Duration) {
        self.lm_head += elapsed;
    }

    pub(crate) fn count_generation_step(&mut self) {
        self.gen_steps += 1;
    }

    fn record_chain_breakdown(&mut self, outputs: &DecodeOutputs) {
        self.chain_full_attn_us += outputs.kernel_full_attn_us;
        self.chain_linear_attn_us += outputs.kernel_linear_attn_us;
        self.chain_ffn_us += outputs.kernel_ffn_us;
    }

    pub(crate) fn print_if_requested(&self, emit_stage_timings: bool) {
        if !emit_stage_timings || self.gen_steps == 0 {
            return;
        }

        let to_ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let chain_ms = to_ms(self.chain);
        let embed_ms = to_ms(self.embed);
        let lm_head_ms = to_ms(self.lm_head);
        let sample_ms = to_ms(self.sample);
        let detok_ms = to_ms(self.detok);
        let total_ms = chain_ms + embed_ms + lm_head_ms + sample_ms + detok_ms;
        let n = self.gen_steps as f64;
        let full_attn_ms = (self.chain_full_attn_us as f64) / 1000.0;
        let linear_attn_ms = (self.chain_linear_attn_us as f64) / 1000.0;
        let ffn_ms = (self.chain_ffn_us as f64) / 1000.0;
        eprintln!(
            "[qwen36-moe stage-timings] gen_steps={} \
             embed_ms_avg={:.3} chain_ms_avg={:.3} lm_head_ms_avg={:.3} \
             sample_ms_avg={:.3} detok_ms_avg={:.3} total_ms_avg={:.3} \
             (chain_total_ms={:.1} lm_head_total_ms={:.1})",
            self.gen_steps,
            embed_ms / n,
            chain_ms / n,
            lm_head_ms / n,
            sample_ms / n,
            detok_ms / n,
            total_ms / n,
            chain_ms,
            lm_head_ms,
        );
        eprintln!(
            "[qwen36-moe chain-breakdown] gen_steps={} \
             full_attn_ms_avg={:.3} linear_attn_ms_avg={:.3} ffn_ms_avg={:.3} \
             (full_attn_total_ms={:.1} linear_attn_total_ms={:.1} ffn_total_ms={:.1})",
            self.gen_steps,
            full_attn_ms / n,
            linear_attn_ms / n,
            ffn_ms / n,
            full_attn_ms,
            linear_attn_ms,
            ffn_ms,
        );
    }
}
