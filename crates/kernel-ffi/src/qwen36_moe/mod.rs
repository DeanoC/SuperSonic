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

include!("profile.rs");
include!("descriptors.rs");
include!("launch.rs");
include!("persistent.rs");
include!("prefill.rs");
