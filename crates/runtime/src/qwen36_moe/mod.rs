//! Runtime-facing Qwen3.6 MoE surface.
//!
//! This module starts as a thin home for config and state contracts. Move
//! runner-owned implementation pieces here only when the move has a focused
//! parity or smoke gate.

pub mod config {
    pub use crate::qwen36_moe_config::*;
}

pub mod decode_loop;
pub mod residency;
pub mod route_telemetry;
pub mod source;
pub mod speculative;
pub mod state;
pub mod types;
