//! Unified inference session that hides the Qwen3.5 vs. Gemma 4 dispatch
//! from the HTTP handlers. Every v1 call path goes through this enum.
//!
//! The engines are synchronous (blocking HIP calls), so `prefill` and
//! `decode_step` must always be invoked from a `spawn_blocking` context.

use std::fmt;

use anyhow::{anyhow, Result};

use crate::decode_engine::{DecodeEngine, DecodeEngineSnapshot};
use crate::dflash::{DFlashGenerateOutput, DFlashSession};
use crate::gemma4_engine::{Gemma4Engine, Gemma4EngineSnapshot};
use crate::gemma4_int4_engine::{Gemma4Int4Engine, Gemma4Int4EngineSnapshot};
use crate::qwen36_moe::engine::Qwen36MoeEngine;

#[cfg(test)]
use std::collections::VecDeque;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionFeatures {
    pub plain_prefill_decode: bool,
    pub native_dflash_generate: bool,
    pub prefix_snapshot: bool,
    pub disk_prefix_snapshot: bool,
}

pub const fn qwen36_moe_features() -> SessionFeatures {
    SessionFeatures {
        plain_prefill_decode: true,
        native_dflash_generate: false,
        prefix_snapshot: false,
        disk_prefix_snapshot: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixSnapshotOperation {
    Capture,
    LoadDisk,
    Restore,
}

impl fmt::Display for PrefixSnapshotOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let operation = match self {
            Self::Capture => "capture",
            Self::LoadDisk => "disk load",
            Self::Restore => "restore",
        };
        formatter.write_str(operation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedPrefixSnapshot {
    operation: PrefixSnapshotOperation,
    session: &'static str,
}

impl UnsupportedPrefixSnapshot {
    fn new(operation: PrefixSnapshotOperation, session: &'static str) -> Self {
        Self { operation, session }
    }

    pub fn operation(&self) -> PrefixSnapshotOperation {
        self.operation
    }

    pub fn session(&self) -> &'static str {
        self.session
    }
}

impl fmt::Display for UnsupportedPrefixSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "prefix snapshot {} is unsupported for {} sessions",
            self.operation, self.session
        )
    }
}

impl std::error::Error for UnsupportedPrefixSnapshot {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionFailureClass {
    RequestLocal,
    IntegrityLost,
}

#[derive(Debug)]
struct SessionResetIntegrityFailure {
    cause: String,
}

impl fmt::Display for SessionResetIntegrityFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "inference session reset failed: {}", self.cause)
    }
}

impl std::error::Error for SessionResetIntegrityFailure {}

#[derive(Debug)]
pub struct SessionCancelled;

impl fmt::Display for SessionCancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("generation cancelled by client")
    }
}

impl std::error::Error for SessionCancelled {}

pub fn is_session_cancellation(error: &anyhow::Error) -> bool {
    error.downcast_ref::<SessionCancelled>().is_some()
}

pub fn classify_session_failure(error: &anyhow::Error) -> SessionFailureClass {
    if error
        .downcast_ref::<SessionResetIntegrityFailure>()
        .is_some()
        || error.chain().any(|cause| {
            let message = cause.to_string().to_ascii_lowercase();
            message.contains("device lost")
                || message.contains("device has been lost")
                || message.contains("hiperrordevicelost")
                || message.contains("cuda_error_device_lost")
                || message.contains("resident allocation pointer")
                || message.contains("resident descriptor")
                || message.contains("descriptor pointer")
                || message.contains("mapped virtual addresses changed")
                || message.contains("is not engine-owned")
                || message.contains("integrity failure")
        })
    {
        SessionFailureClass::IntegrityLost
    } else {
        SessionFailureClass::RequestLocal
    }
}

pub fn should_use_dflash_generation(features: SessionFeatures) -> bool {
    features.native_dflash_generate
}

pub enum InferenceSession {
    Qwen(DecodeEngine),
    QwenDFlash(DFlashSession),
    Qwen36Moe(Qwen36MoeEngine),
    Gemma4Bf16(Gemma4Engine),
    Gemma4Int4(Gemma4Int4Engine),
    #[cfg(test)]
    Deterministic(DeterministicSession),
}

pub enum SessionSnapshot {
    Qwen(DecodeEngineSnapshot),
    Gemma4Bf16(Gemma4EngineSnapshot),
    Gemma4Int4(Gemma4Int4EngineSnapshot),
    #[cfg(test)]
    Deterministic {
        logits: Vec<f32>,
    },
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeterministicSessionEvent {
    Reset,
    Prefill(Vec<u32>),
    Decode { token_id: u32, pos: usize },
}

#[cfg(test)]
pub struct DeterministicSession {
    features: SessionFeatures,
    events: Arc<Mutex<Vec<DeterministicSessionEvent>>>,
    reset_failures: VecDeque<Option<String>>,
    prefill_logits: Vec<f32>,
    prefill_failure: Option<String>,
    decode_logits: VecDeque<Vec<f32>>,
    after_prefill: Option<Box<dyn FnOnce() + Send>>,
}

#[cfg(test)]
impl DeterministicSession {
    pub(crate) fn new(
        features: SessionFeatures,
        prefill_logits: Vec<f32>,
        decode_logits: Vec<Vec<f32>>,
    ) -> (Self, Arc<Mutex<Vec<DeterministicSessionEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                features,
                events: events.clone(),
                reset_failures: VecDeque::new(),
                prefill_logits,
                prefill_failure: None,
                decode_logits: decode_logits.into(),
                after_prefill: None,
            },
            events,
        )
    }

    pub(crate) fn with_reset_failures(mut self, failures: Vec<Option<&str>>) -> Self {
        self.reset_failures = failures
            .into_iter()
            .map(|failure| failure.map(ToOwned::to_owned))
            .collect();
        self
    }

    pub(crate) fn with_prefill_failure(mut self, failure: &str) -> Self {
        self.prefill_failure = Some(failure.to_owned());
        self
    }

    pub(crate) fn after_prefill(mut self, action: impl FnOnce() + Send + 'static) -> Self {
        self.after_prefill = Some(Box::new(action));
        self
    }

    fn record(&self, event: DeterministicSessionEvent) {
        self.events
            .lock()
            .expect("deterministic session events")
            .push(event);
    }

    fn reset(&mut self) -> Result<()> {
        self.record(DeterministicSessionEvent::Reset);
        if let Some(Some(failure)) = self.reset_failures.pop_front() {
            return Err(anyhow!(failure));
        }
        Ok(())
    }

    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.record(DeterministicSessionEvent::Prefill(prompt_ids.to_vec()));
        if let Some(action) = self.after_prefill.take() {
            action();
        }
        if let Some(failure) = self.prefill_failure.take() {
            return Err(anyhow!(failure));
        }
        Ok(self.prefill_logits.clone())
    }

    fn decode_step(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        self.record(DeterministicSessionEvent::Decode { token_id, pos });
        self.decode_logits
            .pop_front()
            .ok_or_else(|| anyhow!("deterministic session received an unexpected decode step"))
    }
}

impl SessionSnapshot {
    pub fn to_disk_bytes(&self) -> Result<Vec<u8>> {
        match self {
            Self::Qwen(s) => s.to_disk_bytes(),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "disk prefix snapshots are currently implemented for Qwen only"
            )),
            #[cfg(test)]
            Self::Deterministic { .. } => {
                Err(anyhow!("deterministic snapshot has no disk representation"))
            }
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Qwen(s) => s.resident_bytes(),
            Self::Gemma4Bf16(s) => s.resident_bytes(),
            Self::Gemma4Int4(s) => s.resident_bytes(),
            #[cfg(test)]
            Self::Deterministic { logits } => logits.len() * std::mem::size_of::<f32>(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Qwen(s) => Ok(Self::Qwen(s.try_clone()?)),
            Self::Gemma4Bf16(s) => Ok(Self::Gemma4Bf16(s.try_clone()?)),
            Self::Gemma4Int4(s) => Ok(Self::Gemma4Int4(s.try_clone()?)),
            #[cfg(test)]
            Self::Deterministic { logits } => Ok(Self::Deterministic {
                logits: logits.clone(),
            }),
        }
    }

    pub fn logits(&self) -> &[f32] {
        match self {
            Self::Qwen(s) => &s.logits,
            Self::Gemma4Bf16(s) => &s.logits,
            Self::Gemma4Int4(s) => &s.logits,
            #[cfg(test)]
            Self::Deterministic { logits } => logits,
        }
    }
}

impl InferenceSession {
    pub fn features(&self) -> SessionFeatures {
        match self {
            Self::Qwen(_) => SessionFeatures {
                plain_prefill_decode: true,
                native_dflash_generate: false,
                prefix_snapshot: true,
                disk_prefix_snapshot: true,
            },
            Self::QwenDFlash(_) => SessionFeatures {
                plain_prefill_decode: false,
                native_dflash_generate: true,
                prefix_snapshot: false,
                disk_prefix_snapshot: false,
            },
            Self::Qwen36Moe(_) => qwen36_moe_features(),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => SessionFeatures {
                plain_prefill_decode: true,
                native_dflash_generate: false,
                prefix_snapshot: true,
                disk_prefix_snapshot: false,
            },
            #[cfg(test)]
            Self::Deterministic(session) => session.features,
        }
    }

    /// Reset per-prompt state (KV caches, conv/recurrent). Weights and
    /// scratch allocations stay resident.
    pub fn reset(&mut self) -> Result<()> {
        let result = match self {
            Self::Qwen(e) => e.reset(),
            Self::QwenDFlash(e) => e.reset(),
            Self::Qwen36Moe(e) => e.reset(),
            Self::Gemma4Bf16(e) => e.reset(),
            Self::Gemma4Int4(e) => e.reset(),
            #[cfg(test)]
            Self::Deterministic(e) => e.reset(),
        };
        result.map_err(|error| {
            let context = SessionResetIntegrityFailure {
                cause: error.to_string(),
            };
            error.context(context)
        })
    }

    /// Run prefill over the tokenized prompt and return the logits at the
    /// last position (F32, host-resident).
    pub fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.prefill_native(prompt_ids),
            Self::QwenDFlash(_) => Err(anyhow!(
                "plain prefill is not exposed for DFlash sessions; use DFlash generation"
            )),
            Self::Qwen36Moe(e) => e.prefill(prompt_ids),
            Self::Gemma4Bf16(e) => e.prefill(prompt_ids),
            Self::Gemma4Int4(e) => e.prefill(prompt_ids),
            #[cfg(test)]
            Self::Deterministic(e) => e.prefill(prompt_ids),
        }
    }

    pub fn prefill_cancellable(
        &mut self,
        prompt_ids: &[u32],
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<f32>> {
        if is_cancelled() {
            return Err(SessionCancelled.into());
        }
        let logits = match self {
            Self::Qwen36Moe(engine) => {
                engine
                    .prefill_with_boundaries(prompt_ids, |_| {
                        if is_cancelled() {
                            Err(SessionCancelled.into())
                        } else {
                            Ok(())
                        }
                    })?
                    .logits
            }
            _ => self.prefill(prompt_ids)?,
        };
        if is_cancelled() {
            return Err(SessionCancelled.into());
        }
        Ok(logits)
    }

    /// Decode one token at absolute position `pos`. `pos` must equal the
    /// number of tokens already consumed by prefill + prior decode steps.
    pub fn decode_step(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(e) => e.decode_step(token_id, pos),
            Self::QwenDFlash(_) => Err(anyhow!(
                "plain decode_step is not exposed for DFlash sessions; use DFlash generation"
            )),
            Self::Qwen36Moe(e) => e.decode_step(token_id, pos),
            Self::Gemma4Bf16(e) => e.decode_step(token_id, pos),
            Self::Gemma4Int4(e) => e.decode_step(token_id, pos),
            #[cfg(test)]
            Self::Deterministic(e) => e.decode_step(token_id, pos),
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
            Self::Qwen36Moe(_) => Err(anyhow!(
                "replay-prefill decode is not implemented for Qwen3.6 MoE sessions"
            )),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "replay-prefill decode is only implemented for the Qwen3.5 engine"
            )),
            #[cfg(test)]
            Self::Deterministic(_) => Err(anyhow!(
                "replay-prefill decode is not implemented for deterministic sessions"
            )),
        }
    }

    pub fn snapshot_prefix(&self, logits: Vec<f32>) -> Result<SessionSnapshot> {
        match self {
            Self::Qwen(e) => Ok(SessionSnapshot::Qwen(e.snapshot_prefix(logits)?)),
            Self::QwenDFlash(_) => Err(anyhow!(
                "prefix cache snapshots are disabled for DFlash sessions"
            )),
            Self::Qwen36Moe(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Capture,
                "Qwen3.6 MoE",
            )
            .into()),
            Self::Gemma4Bf16(e) => Ok(SessionSnapshot::Gemma4Bf16(e.snapshot_prefix(logits)?)),
            Self::Gemma4Int4(e) => Ok(SessionSnapshot::Gemma4Int4(e.snapshot_prefix(logits)?)),
            #[cfg(test)]
            Self::Deterministic(session) if !session.features.prefix_snapshot => Err(
                UnsupportedPrefixSnapshot::new(PrefixSnapshotOperation::Capture, "Qwen3.6 MoE")
                    .into(),
            ),
            #[cfg(test)]
            Self::Deterministic(_) => Ok(SessionSnapshot::Deterministic { logits }),
        }
    }

    pub fn prefix_snapshot_bytes(&self, logits_len: usize) -> usize {
        match self {
            Self::Qwen(e) => e.prefix_snapshot_bytes(logits_len),
            Self::QwenDFlash(_) => usize::MAX,
            Self::Qwen36Moe(_) => usize::MAX,
            Self::Gemma4Bf16(e) => e.prefix_snapshot_bytes(logits_len),
            Self::Gemma4Int4(e) => e.prefix_snapshot_bytes(logits_len),
            #[cfg(test)]
            Self::Deterministic(_) => logits_len * std::mem::size_of::<f32>(),
        }
    }

    pub fn restore_prefix(&mut self, snapshot: SessionSnapshot) -> Result<Vec<f32>> {
        match (self, snapshot) {
            (Self::Qwen(e), SessionSnapshot::Qwen(s)) => e.restore_prefix_owned(s),
            (Self::QwenDFlash(_), _) => Err(anyhow!(
                "prefix cache restore is disabled for DFlash sessions"
            )),
            (Self::Qwen36Moe(_), _) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Restore,
                "Qwen3.6 MoE",
            )
            .into()),
            (Self::Gemma4Bf16(e), SessionSnapshot::Gemma4Bf16(s)) => e.restore_prefix(&s),
            (Self::Gemma4Int4(e), SessionSnapshot::Gemma4Int4(s)) => e.restore_prefix(&s),
            #[cfg(test)]
            (Self::Deterministic(session), _other) if !session.features.prefix_snapshot => Err(
                UnsupportedPrefixSnapshot::new(PrefixSnapshotOperation::Restore, "Qwen3.6 MoE")
                    .into(),
            ),
            #[cfg(test)]
            (Self::Deterministic(_), SessionSnapshot::Deterministic { logits }) => Ok(logits),
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
            Self::Qwen36Moe(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::LoadDisk,
                "Qwen3.6 MoE",
            )
            .into()),
            Self::Gemma4Bf16(_) | Self::Gemma4Int4(_) => Err(anyhow!(
                "disk prefix snapshots are currently implemented for Qwen only"
            )),
            #[cfg(test)]
            Self::Deterministic(session) if !session.features.prefix_snapshot => Err(
                UnsupportedPrefixSnapshot::new(PrefixSnapshotOperation::LoadDisk, "Qwen3.6 MoE")
                    .into(),
            ),
            #[cfg(test)]
            Self::Deterministic(_) => Err(anyhow!(
                "disk prefix snapshots are not implemented for deterministic sessions"
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
            #[cfg(test)]
            Self::Deterministic(_) => Err(anyhow!("deterministic session is not a DFlash session")),
            _ => Err(anyhow!("loaded session is not a DFlash session")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_mode_uses_dflash_only_when_session_exposes_it() {
        assert!(should_use_dflash_generation(SessionFeatures {
            plain_prefill_decode: false,
            native_dflash_generate: true,
            prefix_snapshot: false,
            disk_prefix_snapshot: false,
        }));
        assert!(!should_use_dflash_generation(SessionFeatures {
            plain_prefill_decode: true,
            native_dflash_generate: false,
            prefix_snapshot: true,
            disk_prefix_snapshot: true,
        }));
    }

    #[test]
    fn qwen36_moe_variant_and_features_are_first_class() {
        let _: fn(crate::qwen36_moe::engine::Qwen36MoeEngine) -> InferenceSession =
            InferenceSession::Qwen36Moe;

        assert_eq!(
            qwen36_moe_features(),
            SessionFeatures {
                plain_prefill_decode: true,
                native_dflash_generate: false,
                prefix_snapshot: false,
                disk_prefix_snapshot: false,
            }
        );
    }

    #[test]
    fn deterministic_qwen36_boundary_dispatches_reset_prefill_and_decode_in_order() {
        let (backend, events) = DeterministicSession::new(
            qwen36_moe_features(),
            vec![0.0, 2.0],
            vec![vec![0.0, 0.0, 3.0]],
        );
        let mut session = InferenceSession::Deterministic(backend);

        session.reset().unwrap();
        assert_eq!(session.prefill(&[7, 8]).unwrap(), vec![0.0, 2.0]);
        assert_eq!(session.decode_step(1, 2).unwrap(), vec![0.0, 0.0, 3.0]);

        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::Prefill(vec![7, 8]),
                DeterministicSessionEvent::Decode {
                    token_id: 1,
                    pos: 2,
                },
            ]
        );
    }

    #[test]
    fn qwen36_snapshot_operations_return_typed_unsupported_errors() {
        let (backend, _) = DeterministicSession::new(qwen36_moe_features(), vec![1.0], Vec::new());
        let mut session = InferenceSession::Deterministic(backend);

        for (operation, error) in [
            (
                PrefixSnapshotOperation::Capture,
                session
                    .snapshot_prefix(vec![1.0])
                    .err()
                    .expect("snapshot capture must be unsupported"),
            ),
            (
                PrefixSnapshotOperation::LoadDisk,
                session
                    .load_disk_prefix(b"snapshot")
                    .err()
                    .expect("disk snapshot load must be unsupported"),
            ),
            (
                PrefixSnapshotOperation::Restore,
                session
                    .restore_prefix(SessionSnapshot::Deterministic { logits: vec![1.0] })
                    .err()
                    .expect("snapshot restore must be unsupported"),
            ),
        ] {
            let unsupported = error
                .downcast_ref::<UnsupportedPrefixSnapshot>()
                .expect("typed unsupported prefix snapshot error");
            assert_eq!(unsupported.operation(), operation);
            assert_eq!(unsupported.session(), "Qwen3.6 MoE");
        }
    }

    #[test]
    fn integrity_classifier_is_narrow_and_reset_failures_are_typed() {
        let (backend, _) = DeterministicSession::new(qwen36_moe_features(), vec![1.0], Vec::new());
        let mut session = InferenceSession::Deterministic(
            backend.with_reset_failures(vec![Some("clear state failed")]),
        );
        let reset_error = session.reset().unwrap_err();

        for error in [
            reset_error,
            anyhow!("HIP device lost while synchronizing"),
            anyhow!("resident descriptor pointer is not engine-owned"),
        ] {
            assert_eq!(
                classify_session_failure(&error),
                SessionFailureClass::IntegrityLost
            );
        }

        for error in [
            anyhow!("context limit exceeded"),
            anyhow!("sampling rejected empty logits"),
            anyhow!("unsupported protocol field"),
            anyhow!("generation cancelled by client"),
        ] {
            assert_eq!(
                classify_session_failure(&error),
                SessionFailureClass::RequestLocal
            );
        }
    }
}
