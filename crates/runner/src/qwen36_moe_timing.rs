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
    pub(crate) gen_steps: usize,
    pub(crate) embed: Duration,
    pub(crate) chain: Duration,
    pub(crate) lm_head: Duration,
    pub(crate) sample: Duration,
    pub(crate) detok: Duration,
    pub(crate) chain_full_attn_us: u64,
    pub(crate) chain_linear_attn_us: u64,
    pub(crate) chain_ffn_us: u64,
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
}
