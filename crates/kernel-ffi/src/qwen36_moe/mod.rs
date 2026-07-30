//! FFI bridge for the Qwen3.6-MoE persistent decode megakernel.
//!
//! Status: **infrastructure only.** PR 4 (this file) lands the descriptor
//! layout + a stub kernel that validates the launch path and descriptor
//! read-out. Actual transformer compute (attention with `attn_output_gate`,
//! MoE routing + work-stealing dispatch, fused expert mat-vec, shared
//! expert, lm_head) lands in follow-up PRs.
//!
//! Why a stub first: the megakernel will be ~6500 LoC of HIP. Wiring the
//! FFI/bridge/build path before any compute math means a future kernel
//! commit can land focused-on-math code with no orchestration noise. The
//! stub also serves as a smoke test for descriptor field layout — a
//! Rust↔C++ struct-mismatch bug found here saves hours later.

#![cfg_attr(
    not(supersonic_backend_hip),
    allow(unused_variables, unused_mut, unreachable_code)
)]

use std::collections::{HashMap, VecDeque};
use std::ffi::{c_int, c_void};
use std::os::raw::c_uint;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use gpu_hal::{Backend, BufferKind, GpuBuffer, GpuError, ScalarType};
use half::f16;

use crate::layer_desc::MAX_BATCH_SIZE;

fn qwen36_backend_error(backend: Backend, operation: &str, status: i32) -> GpuError {
    GpuError::backend_status(backend, operation, status)
}

include!("profile.rs");
include!("descriptors.rs");
include!("launch.rs");
include!("persistent.rs");
include!("prefill.rs");

#[cfg(test)]
mod typed_status_tests {
    use super::*;

    #[test]
    fn qwen36_launch_status_preserves_device_loss_identity() {
        let error = qwen36_backend_error(Backend::Hip, "qwen36_moe persistent decode launch", 709);
        assert!(matches!(
            error,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                ref operation,
                status: 709,
            } if operation == "qwen36_moe persistent decode launch"
        ));
    }
}
