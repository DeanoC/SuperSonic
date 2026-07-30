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

use gpu_hal::{Backend, BackendApi, BufferKind, GpuBuffer, GpuError, ScalarType};
use half::f16;

use crate::layer_desc::MAX_BATCH_SIZE;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen36BridgeStatus(u64);

impl Qwen36BridgeStatus {
    // The bridge returns the project status in the low word and, for native
    // failures, the unmodified HIP/CUDA runtime status in the high word.
    const fn project_status(self) -> i32 {
        self.0 as u32 as i32
    }

    const fn native_status(self) -> i32 {
        (self.0 >> u32::BITS) as u32 as i32
    }
}

fn qwen36_bridge_result(
    backend: Backend,
    operation: &str,
    status: Qwen36BridgeStatus,
) -> Result<(), GpuError> {
    let native_status = status.native_status();
    if native_status != 0 {
        return Err(GpuError::backend_status_in(
            backend,
            BackendApi::Runtime,
            operation,
            native_status,
        ));
    }
    let project_status = status.project_status();
    if project_status != 0 {
        return Err(GpuError::backend(
            backend,
            format!("{operation} failed with project status {project_status}"),
        ));
    }
    Ok(())
}

include!("profile.rs");
include!("descriptors.rs");
include!("launch.rs");
include!("persistent.rs");
include!("prefill.rs");

#[cfg(test)]
mod typed_status_tests {
    use super::*;

    unsafe extern "C" {
        fn supersonic_qwen36_encode_bridge_status(
            project_status: c_int,
            native_status: c_int,
        ) -> Qwen36BridgeStatus;
    }

    fn encoded_status(project_status: i32, native_status: i32) -> Qwen36BridgeStatus {
        unsafe { supersonic_qwen36_encode_bridge_status(project_status, native_status) }
    }

    #[test]
    fn qwen36_production_status_conversion_preserves_native_and_project_failures() {
        let error = qwen36_bridge_result(
            Backend::Hip,
            "qwen36_moe persistent decode launch",
            encoded_status(254, 709),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpuError::DeviceLost {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 709,
            } if operation == "qwen36_moe persistent decode launch"
        ));

        let ordinary = qwen36_bridge_result(
            Backend::Hip,
            "qwen36_moe attention launch",
            encoded_status(254, 1),
        )
        .unwrap_err();
        assert!(matches!(
            ordinary,
            GpuError::BackendStatus {
                backend: Backend::Hip,
                api: gpu_hal::BackendApi::Runtime,
                ref operation,
                status: 1,
            } if operation == "qwen36_moe attention launch"
        ));

        let validation = qwen36_bridge_result(
            Backend::Hip,
            "qwen36_moe attention launch",
            encoded_status(111, 0),
        )
        .unwrap_err();
        assert!(matches!(
            validation,
            GpuError::Backend {
                backend: Backend::Hip,
                ref message,
            } if message == "qwen36_moe attention launch failed with project status 111"
        ));
    }
}
