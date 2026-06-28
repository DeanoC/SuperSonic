//! Unified inference session that hides the Qwen3.5 vs. Gemma 4 dispatch
//! from the HTTP handlers. Every v1 call path goes through this enum.
//!
//! The engines are synchronous (blocking HIP calls), so `prefill` and
//! `decode_step` must always be invoked from a `spawn_blocking` context.

use anyhow::{anyhow, Result};

use crate::decode_engine::{DecodeEngine, DecodeEngineSnapshot};
use crate::dflash::{DFlashGenerateOutput, DFlashSession};
use crate::gemma4_engine::{Gemma4Engine, Gemma4EngineSnapshot};
use crate::gemma4_int4_engine::{Gemma4Int4Engine, Gemma4Int4EngineSnapshot};

pub enum InferenceSession {
    Qwen(DecodeEngine),
    QwenDFlash(DFlashSession),
    Gemma4Bf16(Gemma4Engine),
    Gemma4Int4(Gemma4Int4Engine),
}

pub enum SessionSnapshot {
    Qwen(DecodeEngineSnapshot),
    Gemma4Bf16(Gemma4EngineSnapshot),
    Gemma4Int4(Gemma4Int4EngineSnapshot),
}

impl SessionSnapshot {
    pub fn to_disk_bytes(&self) -> Result<Vec<u8>> {
        match self {
            Self::Qwen(s) => s.to_disk_bytes(),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "disk prefix snapshots are currently implemented for Qwen only"
            )),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Qwen(s) => s.resident_bytes(),
            Self::Gemma4Bf16(s) => s.resident_bytes(),
            Self::Gemma4Int4(s) => s.resident_bytes(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Qwen(s) => Ok(Self::Qwen(s.try_clone()?)),
            Self::Gemma4Bf16(s) => Ok(Self::Gemma4Bf16(s.try_clone()?)),
            Self::Gemma4Int4(s) => Ok(Self::Gemma4Int4(s.try_clone()?)),
        }
    }

    pub fn logits(&self) -> &[f32] {
        match self {
            Self::Qwen(s) => &s.logits,
            Self::Gemma4Bf16(s) => &s.logits,
            Self::Gemma4Int4(s) => &s.logits,
        }
    }
}

impl InferenceSession {
    /// Reset per-prompt state (KV caches, conv/recurrent). Weights and
    /// scratch allocations stay resident.
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Qwen(e) => e.reset(),
            Self::QwenDFlash(e) => e.reset(),
            Self::Gemma4Bf16(e) => e.reset(),
            Self::Gemma4Int4(e) => e.reset(),
        }
    }

    /// Run prefill over the tokenized prompt and return the logits at the
    /// last position (F32, host-resident).
    pub fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.prefill_native(prompt_ids),
            Self::QwenDFlash(_) => Err(anyhow!(
                "plain prefill is not exposed for DFlash sessions; use DFlash generation"
            )),
            Self::Gemma4Bf16(e) => e.prefill(prompt_ids),
            Self::Gemma4Int4(e) => e.prefill(prompt_ids),
        }
    }

    /// Decode one token at absolute position `pos`. `pos` must equal the
    /// number of tokens already consumed by prefill + prior decode steps.
    pub fn decode_step(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.decode_step(token_id, pos),
            Self::QwenDFlash(_) => Err(anyhow!(
                "plain decode_step is not exposed for DFlash sessions; use DFlash generation"
            )),
            Self::Gemma4Bf16(e) => e.decode_step(token_id, pos),
            Self::Gemma4Int4(e) => e.decode_step(token_id, pos),
        }
    }

    /// Replay-prefill decode: re-run prefill over the full token history and
    /// return the last-position logits. Kept as a debug helper after Metal v2
    /// landed — incremental decode is the default on every backend now, but
    /// `--gpu-validate` still drives the underlying replay step internally for
    /// per-step correctness comparisons.
    pub fn decode_step_replay(&mut self, token_history: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.decode_step_replay(token_history),
            Self::QwenDFlash(_) => Err(anyhow!(
                "replay-prefill decode is not implemented for DFlash sessions"
            )),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "replay-prefill decode is only implemented for the Qwen3.5 engine"
            )),
        }
    }

    pub fn snapshot_prefix(&self, logits: Vec<f32>) -> Result<SessionSnapshot> {
        match self {
            Self::Qwen(e) => Ok(SessionSnapshot::Qwen(e.snapshot_prefix(logits)?)),
            Self::QwenDFlash(_) => Err(anyhow!(
                "prefix cache snapshots are disabled for DFlash sessions"
            )),
            Self::Gemma4Bf16(e) => Ok(SessionSnapshot::Gemma4Bf16(e.snapshot_prefix(logits)?)),
            Self::Gemma4Int4(e) => Ok(SessionSnapshot::Gemma4Int4(e.snapshot_prefix(logits)?)),
        }
    }

    pub fn prefix_snapshot_bytes(&self, logits_len: usize) -> usize {
        match self {
            Self::Qwen(e) => e.prefix_snapshot_bytes(logits_len),
            Self::QwenDFlash(_) => usize::MAX,
            Self::Gemma4Bf16(e) => e.prefix_snapshot_bytes(logits_len),
            Self::Gemma4Int4(e) => e.prefix_snapshot_bytes(logits_len),
        }
    }

    pub fn restore_prefix(&mut self, snapshot: SessionSnapshot) -> Result<Vec<f32>> {
        match (self, snapshot) {
            (Self::Qwen(e), SessionSnapshot::Qwen(s)) => e.restore_prefix_owned(s),
            (Self::QwenDFlash(_), _) => Err(anyhow!(
                "prefix cache restore is disabled for DFlash sessions"
            )),
            (Self::Gemma4Bf16(e), SessionSnapshot::Gemma4Bf16(s)) => e.restore_prefix(&s),
            (Self::Gemma4Int4(e), SessionSnapshot::Gemma4Int4(s)) => e.restore_prefix(&s),
            _ => Err(anyhow!(
                "prefix cache snapshot does not match loaded model family"
            )),
        }
    }

    pub fn load_disk_prefix(&self, bytes: &[u8]) -> Result<SessionSnapshot> {
        match self {
            Self::Qwen(e) => Ok(SessionSnapshot::Qwen(e.load_prefix_snapshot_bytes(bytes)?)),
            Self::QwenDFlash(_) => Err(anyhow!(
                "disk prefix snapshots are disabled for DFlash sessions"
            )),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "disk prefix snapshots are currently implemented for Qwen only"
            )),
        }
    }

    pub fn is_dflash(&self) -> bool {
        matches!(self, Self::QwenDFlash(_))
    }

    pub fn generate_dflash_greedy(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        eos_ids: &[u32],
    ) -> Result<DFlashGenerateOutput> {
        match self {
            Self::QwenDFlash(session) => session.generate_greedy(prompt_ids, max_tokens, eos_ids),
            _ => Err(anyhow!("loaded session is not a DFlash session")),
        }
    }
}
