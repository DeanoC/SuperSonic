//! Unified inference session that hides the Qwen3.5 vs. Gemma 4 dispatch
//! from the HTTP handlers. Every v1 call path goes through this enum.
//!
//! The engines are synchronous (blocking HIP calls), so `prefill` and
//! `decode_step` must always be invoked from a `spawn_blocking` context.

use anyhow::Result;

use runner::decode_engine::DecodeEngine;
use runner::gemma4_engine::Gemma4Engine;
use runner::gemma4_int4_engine::Gemma4Int4Engine;

pub enum InferenceSession {
    Qwen(DecodeEngine),
    Gemma4Bf16(Gemma4Engine),
    Gemma4Int4(Gemma4Int4Engine),
}

impl InferenceSession {
    /// Reset per-prompt state (KV caches, conv/recurrent). Weights and
    /// scratch allocations stay resident.
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Qwen(e) => e.reset(),
            Self::Gemma4Bf16(e) => e.reset(),
            Self::Gemma4Int4(e) => e.reset(),
        }
    }

    /// Run prefill over the tokenized prompt and return the logits at the
    /// last position (F32, host-resident).
    pub fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.prefill_native(prompt_ids),
            Self::Gemma4Bf16(e) => e.prefill(prompt_ids),
            Self::Gemma4Int4(e) => e.prefill(prompt_ids),
        }
    }

    /// Decode one token at absolute position `pos`. `pos` must equal the
    /// number of tokens already consumed by prefill + prior decode steps.
    pub fn decode_step(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.decode_step(token_id, pos),
            Self::Gemma4Bf16(e) => e.decode_step(token_id, pos),
            Self::Gemma4Int4(e) => e.decode_step(token_id, pos),
        }
    }
}
