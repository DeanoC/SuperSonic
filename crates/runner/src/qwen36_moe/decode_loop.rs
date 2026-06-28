use std::io::Write as _;

use crate::qwen36_moe_cli::output::print_decoded_token;
use crate::qwen36_moe_speculative::SpeculativeStepResult;
use crate::qwen36_moe_types::PositionPair;
use supersonic_runtime::qwen36_moe::decode_loop::speculative_replay_inputs as runtime_speculative_replay_inputs;

pub struct Qwen36DecodeLoopState {
    pub generated_ids: Vec<u32>,
    pub last_logits_bytes: Vec<u8>,
    pub current_token: u32,
    pub position: i32,
    pub total_steps: usize,
    max_new: usize,
}

impl Qwen36DecodeLoopState {
    pub fn new(prompt_ids: &[u32], max_new: usize) -> Self {
        Self {
            generated_ids: Vec::with_capacity(max_new),
            last_logits_bytes: Vec::new(),
            current_token: prompt_ids[0],
            position: 0,
            total_steps: prompt_ids.len() + max_new - 1,
            max_new,
        }
    }

    pub fn reached_max_new(&self) -> bool {
        self.generated_ids.len() >= self.max_new
    }

    pub fn remaining_generation_slots(&self) -> usize {
        self.max_new.saturating_sub(self.generated_ids.len())
    }

    pub fn speculative_draft_count(&self, max_drafts: usize) -> usize {
        max_drafts.min(self.remaining_generation_slots().saturating_sub(1))
    }

    pub fn speculative_replay_inputs(
        &self,
        first_token: u32,
        base: PositionPair,
        result: &SpeculativeStepResult,
    ) -> Vec<(PositionPair, u32)> {
        runtime_speculative_replay_inputs(
            first_token,
            base,
            &result.emitted_tokens,
            result.n_accepted,
        )
    }

    pub fn partial_accept_replay_inputs(
        &self,
        first_token: u32,
        base: PositionPair,
        result: &SpeculativeStepResult,
        draft_count: usize,
    ) -> Option<Vec<(PositionPair, u32)>> {
        (result.n_accepted < draft_count)
            .then(|| self.speculative_replay_inputs(first_token, base, result))
    }

    pub fn record_last_logits(&mut self, logits: &[u8]) {
        self.last_logits_bytes.clear();
        self.last_logits_bytes.extend_from_slice(logits);
    }

    pub fn append_speculative_emissions(
        &mut self,
        result: &SpeculativeStepResult,
        tokenizer: Option<&tokenizers::Tokenizer>,
        eos_id: Option<u32>,
    ) -> bool {
        let mut hit_stop = false;
        for tok in result.emitted_tokens.iter().copied() {
            if self.reached_max_new() {
                hit_stop = true;
                break;
            }
            self.generated_ids.push(tok);
            print_decoded_token(tokenizer, tok);
            if Some(tok) == eos_id {
                hit_stop = true;
                break;
            }
        }
        std::io::stdout().flush().ok();
        self.position += result.emitted_tokens.len() as i32;
        if !hit_stop {
            self.current_token = *result
                .emitted_tokens
                .last()
                .expect("speculative step must emit at least one token (K=0 fallback ensured)");
        }
        hit_stop
    }
}
