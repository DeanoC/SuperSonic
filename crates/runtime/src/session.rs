//! Unified inference session that hides the Qwen3.5 vs. Gemma 4 dispatch
//! from the HTTP handlers. Every v1 call path goes through this enum.
//!
//! The engines are synchronous (blocking HIP calls), so `prefill` and
//! `decode_step` must always be invoked from a `spawn_blocking` context.

use std::fmt;

use anyhow::{anyhow, Result};

use crate::decode_engine::{DecodeEngine, DecodeEngineSnapshot};
use crate::dflash::{DFlashGenerateOutput, DFlashSession};
use crate::qwen36_moe::engine::{Qwen36MoeEngine, Qwen36MoePrefillBoundary};

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

#[derive(Debug)]
pub struct RequestLocalCacheFailure {
    operation: PrefixSnapshotOperation,
    cause: String,
}

impl RequestLocalCacheFailure {
    fn new(operation: PrefixSnapshotOperation, cause: impl Into<String>) -> Self {
        Self {
            operation,
            cause: cause.into(),
        }
    }
}

impl fmt::Display for RequestLocalCacheFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "prefix cache {} failed: {}",
            self.operation, self.cause
        )
    }
}

impl std::error::Error for RequestLocalCacheFailure {}

pub fn is_request_local_cache_failure(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RequestLocalCacheFailure>().is_some()
}

pub(crate) fn cache_operation_error(
    operation: PrefixSnapshotOperation,
    error: anyhow::Error,
) -> anyhow::Error {
    if error.downcast_ref::<gpu_hal::GpuError>().is_some()
        || error
            .downcast_ref::<crate::qwen36_moe::engine::Qwen36MoeIntegrityError>()
            .is_some()
    {
        let context = format!("prefix cache {operation} engine operation failed: {error:#}");
        error.context(context)
    } else {
        let context = RequestLocalCacheFailure::new(operation, error.to_string());
        error.context(context)
    }
}

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
    let integrity_lost = error
        .downcast_ref::<SessionResetIntegrityFailure>()
        .is_some()
        || error
            .downcast_ref::<gpu_hal::GpuError>()
            .is_some_and(gpu_hal::GpuError::is_device_lost)
        || error
            .downcast_ref::<crate::qwen36_moe::engine::Qwen36MoeIntegrityError>()
            .is_some();
    if integrity_lost {
        SessionFailureClass::IntegrityLost
    } else {
        SessionFailureClass::RequestLocal
    }
}

pub fn should_use_dflash_generation(features: SessionFeatures) -> bool {
    features.native_dflash_generate
}

trait Qwen36SessionBackend {
    fn reset_session(&mut self) -> Result<()>;
    fn prefill_session(
        &mut self,
        prompt_ids: &[u32],
        boundary_observer: &mut dyn FnMut(Qwen36MoePrefillBoundary) -> Result<()>,
    ) -> Result<Vec<f32>>;
    fn decode_session(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>>;
}

impl Qwen36SessionBackend for Qwen36MoeEngine {
    fn reset_session(&mut self) -> Result<()> {
        self.reset()
    }

    fn prefill_session(
        &mut self,
        prompt_ids: &[u32],
        boundary_observer: &mut dyn FnMut(Qwen36MoePrefillBoundary) -> Result<()>,
    ) -> Result<Vec<f32>> {
        Ok(self
            .prefill_with_boundaries(prompt_ids, boundary_observer)?
            .logits)
    }

    fn decode_session(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        self.decode_step(token_id, pos)
    }
}

fn qwen36_reset(backend: &mut impl Qwen36SessionBackend) -> Result<()> {
    backend.reset_session()
}

fn qwen36_prefill(backend: &mut impl Qwen36SessionBackend, prompt_ids: &[u32]) -> Result<Vec<f32>> {
    backend.prefill_session(prompt_ids, &mut |_| Ok(()))
}

fn qwen36_prefill_cancellable(
    backend: &mut impl Qwen36SessionBackend,
    prompt_ids: &[u32],
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<Vec<f32>> {
    backend.prefill_session(prompt_ids, &mut |_| {
        if is_cancelled() {
            Err(SessionCancelled.into())
        } else {
            Ok(())
        }
    })
}

fn qwen36_decode(
    backend: &mut impl Qwen36SessionBackend,
    token_id: u32,
    pos: usize,
) -> Result<Vec<f32>> {
    backend.decode_session(token_id, pos)
}

pub enum InferenceSession {
    Qwen(DecodeEngine),
    QwenDFlash(DFlashSession),
    Qwen36Moe(Qwen36MoeEngine),
    #[cfg(test)]
    Qwen36MoeTestAdapter(DeterministicSession),
    #[cfg(test)]
    Deterministic(DeterministicSession),
}

pub enum SessionSnapshot {
    Qwen(DecodeEngineSnapshot),
    #[cfg(test)]
    Deterministic {
        logits: Vec<f32>,
    },
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeterministicSessionEvent {
    Reset,
    PrefillBoundary(Qwen36MoePrefillBoundary),
    Prefill(Vec<u32>),
    Decode { token_id: u32, pos: usize },
}

#[cfg(test)]
pub struct DeterministicSession {
    features: SessionFeatures,
    events: Arc<Mutex<Vec<DeterministicSessionEvent>>>,
    reset_failures: VecDeque<Option<String>>,
    after_reset: Option<Box<dyn FnOnce() + Send>>,
    prefill_logits: Vec<f32>,
    prefill_failure: Option<String>,
    prefill_gpu_status: Option<(gpu_hal::Backend, gpu_hal::BackendApi, i32)>,
    decode_logits: VecDeque<Vec<f32>>,
    after_prefill: Option<Box<dyn FnOnce() + Send>>,
    prefill_panics: bool,
    restore_failure: Option<DeterministicCacheFailure>,
    snapshot_failure: Option<DeterministicCacheFailure>,
    prefill_boundary_action: Option<(Qwen36MoePrefillBoundary, Box<dyn FnOnce() + Send>)>,
    after_decode: Option<Box<dyn FnOnce() + Send>>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
enum DeterministicCacheFailure {
    DeviceLost,
    RequestLocal,
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
                after_reset: None,
                prefill_logits,
                prefill_failure: None,
                prefill_gpu_status: None,
                decode_logits: decode_logits.into(),
                after_prefill: None,
                prefill_panics: false,
                restore_failure: None,
                snapshot_failure: None,
                prefill_boundary_action: None,
                after_decode: None,
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

    pub(crate) fn with_prefill_device_loss(self) -> Self {
        self.with_prefill_gpu_status(gpu_hal::Backend::Hip, 709)
    }

    pub(crate) fn with_prefill_gpu_status(self, backend: gpu_hal::Backend, status: i32) -> Self {
        self.with_prefill_gpu_status_in(backend, gpu_hal::BackendApi::Runtime, status)
    }

    pub(crate) fn with_prefill_gpu_status_in(
        mut self,
        backend: gpu_hal::Backend,
        api: gpu_hal::BackendApi,
        status: i32,
    ) -> Self {
        self.prefill_gpu_status = Some((backend, api, status));
        self
    }

    pub(crate) fn after_first_reset(mut self, action: impl FnOnce() + Send + 'static) -> Self {
        self.after_reset = Some(Box::new(action));
        self
    }

    pub(crate) fn after_prefill(mut self, action: impl FnOnce() + Send + 'static) -> Self {
        self.after_prefill = Some(Box::new(action));
        self
    }

    pub(crate) fn with_prefill_panic(mut self) -> Self {
        self.prefill_panics = true;
        self
    }

    pub(crate) fn with_restore_device_loss(mut self) -> Self {
        self.restore_failure = Some(DeterministicCacheFailure::DeviceLost);
        self
    }

    pub(crate) fn with_restore_request_local_cache_failure(mut self) -> Self {
        self.restore_failure = Some(DeterministicCacheFailure::RequestLocal);
        self
    }

    pub(crate) fn with_snapshot_device_loss(mut self) -> Self {
        self.snapshot_failure = Some(DeterministicCacheFailure::DeviceLost);
        self
    }

    pub(crate) fn close_at_prefill_boundary(
        mut self,
        boundary: Qwen36MoePrefillBoundary,
        action: impl FnOnce() + Send + 'static,
    ) -> Self {
        self.prefill_boundary_action = Some((boundary, Box::new(action)));
        self
    }

    pub(crate) fn after_decode(mut self, action: impl FnOnce() + Send + 'static) -> Self {
        self.after_decode = Some(Box::new(action));
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
        if let Some(action) = self.after_reset.take() {
            action();
        }
        Ok(())
    }

    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.record(DeterministicSessionEvent::Prefill(prompt_ids.to_vec()));
        assert!(!self.prefill_panics, "deterministic prefill panic");
        if let Some(action) = self.after_prefill.take() {
            action();
        }
        if let Some((backend, api, status)) = self.prefill_gpu_status.take() {
            return Err(
                gpu_hal::GpuError::backend_status_in(backend, api, "test prefill", status).into(),
            );
        }
        if let Some(failure) = self.prefill_failure.take() {
            return Err(anyhow!(failure));
        }
        Ok(self.prefill_logits.clone())
    }

    fn cache_failure(
        operation: PrefixSnapshotOperation,
        failure: DeterministicCacheFailure,
    ) -> anyhow::Error {
        let error = match failure {
            DeterministicCacheFailure::DeviceLost => anyhow::Error::new(
                gpu_hal::GpuError::backend_status(gpu_hal::Backend::Hip, "test cache GPU op", 709),
            ),
            DeterministicCacheFailure::RequestLocal => anyhow!("test cache format mismatch"),
        };
        cache_operation_error(operation, error)
    }

    fn decode_step(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        self.record(DeterministicSessionEvent::Decode { token_id, pos });
        self.decode_logits
            .pop_front()
            .ok_or_else(|| anyhow!("deterministic session received an unexpected decode step"))
    }
}

#[cfg(test)]
impl Qwen36SessionBackend for DeterministicSession {
    fn reset_session(&mut self) -> Result<()> {
        self.reset()
    }

    fn prefill_session(
        &mut self,
        prompt_ids: &[u32],
        boundary_observer: &mut dyn FnMut(Qwen36MoePrefillBoundary) -> Result<()>,
    ) -> Result<Vec<f32>> {
        for boundary in [
            Qwen36MoePrefillBoundary::PrefixStarted,
            Qwen36MoePrefillBoundary::FinalProductionStarted,
        ] {
            self.record(DeterministicSessionEvent::PrefillBoundary(boundary));
            if self
                .prefill_boundary_action
                .as_ref()
                .is_some_and(|(target, _)| *target == boundary)
            {
                let (_, action) = self
                    .prefill_boundary_action
                    .take()
                    .expect("prefill boundary action");
                action();
            }
            boundary_observer(boundary)?;
        }
        self.prefill(prompt_ids)
    }

    fn decode_session(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>> {
        let result = self.decode_step(token_id, pos);
        if let Some(action) = self.after_decode.take() {
            action();
        }
        result
    }
}

impl SessionSnapshot {
    pub fn to_disk_bytes(&self) -> Result<Vec<u8>> {
        match self {
            Self::Qwen(s) => s.to_disk_bytes(),
            #[cfg(test)]
            Self::Deterministic { .. } => {
                Err(anyhow!("deterministic snapshot has no disk representation"))
            }
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Qwen(s) => s.resident_bytes(),
            #[cfg(test)]
            Self::Deterministic { logits } => logits.len() * std::mem::size_of::<f32>(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Qwen(s) => Ok(Self::Qwen(s.try_clone()?)),
            #[cfg(test)]
            Self::Deterministic { logits } => Ok(Self::Deterministic {
                logits: logits.clone(),
            }),
        }
    }

    pub fn logits(&self) -> &[f32] {
        match self {
            Self::Qwen(s) => &s.logits,
            #[cfg(test)]
            Self::Deterministic { logits } => logits,
        }
    }
}

impl InferenceSession {
    #[cfg(test)]
    pub(crate) fn test_qwen36_adapter(backend: DeterministicSession) -> Self {
        Self::Qwen36MoeTestAdapter(backend)
    }

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
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(_) => qwen36_moe_features(),
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
            Self::Qwen36Moe(e) => qwen36_reset(e),
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(e) => qwen36_reset(e),
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
            Self::Qwen36Moe(e) => qwen36_prefill(e, prompt_ids),
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(e) => qwen36_prefill(e, prompt_ids),
            #[cfg(test)]
            Self::Deterministic(e) => e.prefill(prompt_ids),
        }
    }

    pub fn prefill_cancellable(
        &mut self,
        prompt_ids: &[u32],
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<f32>> {
        self.prefill_cancellable_with_started(prompt_ids, is_cancelled, || {})
    }

    pub(crate) fn prefill_cancellable_with_started(
        &mut self,
        prompt_ids: &[u32],
        mut is_cancelled: impl FnMut() -> bool,
        mut execution_started: impl FnMut(),
    ) -> Result<Vec<f32>> {
        if is_cancelled() {
            return Err(SessionCancelled.into());
        }
        execution_started();
        let logits = match self {
            Self::Qwen36Moe(engine) => {
                qwen36_prefill_cancellable(engine, prompt_ids, &mut is_cancelled)?
            }
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(engine) => {
                qwen36_prefill_cancellable(engine, prompt_ids, &mut is_cancelled)?
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
            Self::Qwen36Moe(e) => qwen36_decode(e, token_id, pos),
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(e) => qwen36_decode(e, token_id, pos),
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
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(_) => Err(anyhow!(
                "replay-prefill decode is not implemented for Qwen3.6 MoE sessions"
            )),
            #[cfg(test)]
            Self::Deterministic(_) => Err(anyhow!(
                "replay-prefill decode is not implemented for deterministic sessions"
            )),
        }
    }

    pub fn snapshot_prefix(&self, logits: Vec<f32>) -> Result<SessionSnapshot> {
        match self {
            Self::Qwen(e) => e
                .snapshot_prefix(logits)
                .map(SessionSnapshot::Qwen)
                .map_err(|error| cache_operation_error(PrefixSnapshotOperation::Capture, error)),
            Self::QwenDFlash(_) => Err(anyhow!(
                "prefix cache snapshots are disabled for DFlash sessions"
            )),
            Self::Qwen36Moe(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Capture,
                "Qwen3.6 MoE",
            )
            .into()),
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Capture,
                "Qwen3.6 MoE",
            )
            .into()),
            #[cfg(test)]
            Self::Deterministic(session) if !session.features.prefix_snapshot => Err(
                UnsupportedPrefixSnapshot::new(PrefixSnapshotOperation::Capture, "Qwen3.6 MoE")
                    .into(),
            ),
            #[cfg(test)]
            Self::Deterministic(session) => {
                if let Some(failure) = session.snapshot_failure {
                    Err(DeterministicSession::cache_failure(
                        PrefixSnapshotOperation::Capture,
                        failure,
                    ))
                } else {
                    Ok(SessionSnapshot::Deterministic { logits })
                }
            }
        }
    }

    pub fn prefix_snapshot_bytes(&self, logits_len: usize) -> usize {
        match self {
            Self::Qwen(e) => e.prefix_snapshot_bytes(logits_len),
            Self::QwenDFlash(_) => usize::MAX,
            Self::Qwen36Moe(_) => usize::MAX,
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(_) => usize::MAX,
            #[cfg(test)]
            Self::Deterministic(_) => logits_len * std::mem::size_of::<f32>(),
        }
    }

    pub fn restore_prefix(&mut self, snapshot: SessionSnapshot) -> Result<Vec<f32>> {
        match (self, snapshot) {
            (Self::Qwen(e), SessionSnapshot::Qwen(s)) => e
                .restore_prefix_owned(s)
                .map_err(|error| cache_operation_error(PrefixSnapshotOperation::Restore, error)),
            (Self::QwenDFlash(_), _) => Err(anyhow!(
                "prefix cache restore is disabled for DFlash sessions"
            )),
            (Self::Qwen36Moe(_), _) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Restore,
                "Qwen3.6 MoE",
            )
            .into()),
            #[cfg(test)]
            (Self::Qwen36MoeTestAdapter(_), _) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::Restore,
                "Qwen3.6 MoE",
            )
            .into()),
            #[cfg(test)]
            (Self::Deterministic(session), _other) if !session.features.prefix_snapshot => Err(
                UnsupportedPrefixSnapshot::new(PrefixSnapshotOperation::Restore, "Qwen3.6 MoE")
                    .into(),
            ),
            #[cfg(test)]
            (Self::Deterministic(session), SessionSnapshot::Deterministic { logits }) => {
                if let Some(failure) = session.restore_failure {
                    Err(DeterministicSession::cache_failure(
                        PrefixSnapshotOperation::Restore,
                        failure,
                    ))
                } else {
                    Ok(logits)
                }
            }
            _ => Err(RequestLocalCacheFailure::new(
                PrefixSnapshotOperation::Restore,
                "snapshot does not match loaded model family",
            )
            .into()),
        }
    }

    pub fn load_disk_prefix(&self, bytes: &[u8]) -> Result<SessionSnapshot> {
        match self {
            Self::Qwen(e) => e
                .load_prefix_snapshot_bytes(bytes)
                .map(SessionSnapshot::Qwen)
                .map_err(|error| cache_operation_error(PrefixSnapshotOperation::LoadDisk, error)),
            Self::QwenDFlash(_) => Err(anyhow!(
                "disk prefix snapshots are disabled for DFlash sessions"
            )),
            Self::Qwen36Moe(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::LoadDisk,
                "Qwen3.6 MoE",
            )
            .into()),
            #[cfg(test)]
            Self::Qwen36MoeTestAdapter(_) => Err(UnsupportedPrefixSnapshot::new(
                PrefixSnapshotOperation::LoadDisk,
                "Qwen3.6 MoE",
            )
            .into()),
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
    use crate::qwen36_moe::engine::Qwen36MoeIntegrityError;

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
        let mut session = InferenceSession::test_qwen36_adapter(backend);

        session.reset().unwrap();
        assert_eq!(session.prefill(&[7, 8]).unwrap(), vec![0.0, 2.0]);
        assert_eq!(session.decode_step(1, 2).unwrap(), vec![0.0, 0.0, 3.0]);

        assert_eq!(
            *events.lock().unwrap(),
            [
                DeterministicSessionEvent::Reset,
                DeterministicSessionEvent::PrefillBoundary(Qwen36MoePrefillBoundary::PrefixStarted,),
                DeterministicSessionEvent::PrefillBoundary(
                    Qwen36MoePrefillBoundary::FinalProductionStarted,
                ),
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
        let mut session = InferenceSession::test_qwen36_adapter(backend);

        assert_eq!(session.features(), qwen36_moe_features());
        assert_eq!(session.prefix_snapshot_bytes(1), usize::MAX);

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

        assert_eq!(
            classify_session_failure(&reset_error),
            SessionFailureClass::IntegrityLost
        );

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

    #[test]
    fn integrity_classifier_uses_typed_sources_through_context_layers() {
        let device_lost = anyhow::Error::new(gpu_hal::GpuError::backend_status(
            gpu_hal::Backend::Hip,
            "hipMemcpy(D2H)",
            709,
        ))
        .context("download serving logits")
        .context("decode request");
        let resident_errors = [
            Qwen36MoeIntegrityError::ResidentAllocationsChanged,
            Qwen36MoeIntegrityError::MappedVirtualAddressesChanged,
            Qwen36MoeIntegrityError::DescriptorPointerNotOwned {
                label: "experts_gate_up_w".to_string(),
                pointer: 0x4000,
            },
        ]
        .into_iter()
        .map(|error| {
            anyhow::Error::new(error)
                .context("resident identity validation")
                .context("reset request")
        });

        for error in std::iter::once(device_lost).chain(resident_errors) {
            assert_eq!(
                classify_session_failure(&error),
                SessionFailureClass::IntegrityLost
            );
        }

        let ordinary_backend = anyhow::Error::new(gpu_hal::GpuError::backend_status(
            gpu_hal::Backend::Hip,
            "hipMemcpy(D2H)",
            1,
        ))
        .context("decode request");
        assert_eq!(
            classify_session_failure(&ordinary_backend),
            SessionFailureClass::RequestLocal
        );
    }

    #[test]
    fn old_integrity_keywords_do_not_poison_untyped_request_errors() {
        for message in [
            "request mentions device lost",
            "protocol field named descriptor pointer",
            "user text says resident allocation pointer",
            "request-local integrity failure",
            "prompt contains mapped virtual addresses changed",
            "input says is not engine-owned",
        ] {
            assert_eq!(
                classify_session_failure(&anyhow!(message)),
                SessionFailureClass::RequestLocal,
                "{message}"
            );
        }
    }

    #[test]
    fn cache_error_context_marks_only_untyped_local_failures_recoverable() {
        let gpu_error = cache_operation_error(
            PrefixSnapshotOperation::Restore,
            gpu_hal::GpuError::backend_status(gpu_hal::Backend::Hip, "hipMemcpy(D2D)", 2).into(),
        );
        let format_error = cache_operation_error(
            PrefixSnapshotOperation::Restore,
            anyhow!("snapshot family mismatch"),
        );

        assert!(!is_request_local_cache_failure(&gpu_error));
        assert!(is_request_local_cache_failure(&format_error));
    }
}
