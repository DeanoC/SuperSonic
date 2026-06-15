use std::time::Duration;

use crate::qwen36_moe_cli::host::EmbedLookupTiming;
use crate::qwen36_moe_types::DecodeOutputs;

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
    embed_min: Option<Duration>,
    embed_max: Duration,
    embed_slow_1ms: usize,
    embed_slow_5ms: usize,
    embed_lookup_samples: usize,
    embed_lookup_raw_bytes: Duration,
    embed_lookup_copy: Duration,
    embed_lookup_other: Duration,
    chain_full_attn_us: u64,
    chain_linear_attn_us: u64,
    chain_ffn_us: u64,
    sparse_lookahead_prefetch_us: u64,
    sparse_router_launch_us: u64,
    sparse_route_d2h_us: u64,
    sparse_demand_prefetch_us: u64,
    sparse_ffn_launch_us: u64,
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
        embed_lookup: Option<EmbedLookupTiming>,
    ) {
        self.count_generation_step();
        self.embed += embed;
        self.record_embed_distribution(embed);
        if let Some(embed_lookup) = embed_lookup {
            self.record_embed_lookup_breakdown(embed, embed_lookup);
        }
        self.chain += chain;
        self.lm_head += lm_head;
        self.sample += sample;
        self.detok += detok;
        self.record_chain_breakdown(outputs);
    }

    pub(crate) fn record_embed(&mut self, elapsed: Duration) {
        self.embed += elapsed;
        self.record_embed_distribution(elapsed);
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
        self.sparse_lookahead_prefetch_us += outputs.sparse_lookahead_prefetch_us;
        self.sparse_router_launch_us += outputs.sparse_router_launch_us;
        self.sparse_route_d2h_us += outputs.sparse_route_d2h_us;
        self.sparse_demand_prefetch_us += outputs.sparse_demand_prefetch_us;
        self.sparse_ffn_launch_us += outputs.sparse_ffn_launch_us;
    }

    fn record_embed_distribution(&mut self, elapsed: Duration) {
        self.embed_min = Some(self.embed_min.map_or(elapsed, |min| min.min(elapsed)));
        self.embed_max = self.embed_max.max(elapsed);
        if elapsed >= Duration::from_millis(1) {
            self.embed_slow_1ms += 1;
        }
        if elapsed >= Duration::from_millis(5) {
            self.embed_slow_5ms += 1;
        }
    }

    fn record_embed_lookup_breakdown(&mut self, total: Duration, timing: EmbedLookupTiming) {
        self.embed_lookup_samples += 1;
        self.embed_lookup_raw_bytes += timing.raw_bytes;
        self.embed_lookup_copy += timing.copy;
        self.embed_lookup_other += total.saturating_sub(timing.raw_bytes + timing.copy);
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
        let embed_min_ms = self.embed_min.map(to_ms).unwrap_or(0.0);
        let embed_max_ms = to_ms(self.embed_max);
        let lookup_n = self.embed_lookup_samples.max(1) as f64;
        let raw_bytes_ms = to_ms(self.embed_lookup_raw_bytes);
        let copy_ms = to_ms(self.embed_lookup_copy);
        let other_ms = to_ms(self.embed_lookup_other);
        eprintln!(
            "[qwen36-moe embed-breakdown] gen_steps={} lookup_samples={} \
             embed_min_ms={:.3} embed_max_ms={:.3} embed_slow_1ms={} \
             embed_slow_5ms={} raw_bytes_ms_avg={:.3} copy_ms_avg={:.3} \
             other_ms_avg={:.3} \
             (raw_bytes_total_ms={:.3} copy_total_ms={:.3} other_total_ms={:.3})",
            self.gen_steps,
            self.embed_lookup_samples,
            embed_min_ms,
            embed_max_ms,
            self.embed_slow_1ms,
            self.embed_slow_5ms,
            raw_bytes_ms / lookup_n,
            copy_ms / lookup_n,
            other_ms / lookup_n,
            raw_bytes_ms,
            copy_ms,
            other_ms,
        );
        let sparse_total_us = self
            .sparse_lookahead_prefetch_us
            .saturating_add(self.sparse_router_launch_us)
            .saturating_add(self.sparse_route_d2h_us)
            .saturating_add(self.sparse_demand_prefetch_us)
            .saturating_add(self.sparse_ffn_launch_us);
        if sparse_total_us > 0 {
            let lookahead_ms = (self.sparse_lookahead_prefetch_us as f64) / 1000.0;
            let router_ms = (self.sparse_router_launch_us as f64) / 1000.0;
            let route_d2h_ms = (self.sparse_route_d2h_us as f64) / 1000.0;
            let demand_ms = (self.sparse_demand_prefetch_us as f64) / 1000.0;
            let ffn_launch_ms = (self.sparse_ffn_launch_us as f64) / 1000.0;
            eprintln!(
                "[qwen36-moe sparse-breakdown] gen_steps={} \
                 lookahead_prefetch_ms_avg={:.3} router_launch_ms_avg={:.3} \
                 route_d2h_ms_avg={:.3} demand_prefetch_ms_avg={:.3} \
                 ffn_launch_ms_avg={:.3} \
                 (lookahead_total_ms={:.1} router_total_ms={:.1} \
                 route_d2h_total_ms={:.1} demand_total_ms={:.1} \
                 ffn_launch_total_ms={:.1})",
                self.gen_steps,
                lookahead_ms / n,
                router_ms / n,
                route_d2h_ms / n,
                demand_ms / n,
                ffn_launch_ms / n,
                lookahead_ms,
                router_ms,
                route_d2h_ms,
                demand_ms,
                ffn_launch_ms,
            );
        }
    }
}
