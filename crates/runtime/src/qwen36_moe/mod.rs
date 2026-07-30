//! Runtime-facing Qwen3.6 MoE surface.
//!
//! Runtime owns the serving contracts and behavior. Runner modules retain
//! compatibility adapters for CLI translation and process-level telemetry.

pub mod config {
    pub use crate::qwen36_moe_config::*;
}

pub mod decode_loop;
pub mod load_policy;
pub mod prefetch;
pub mod residency;
pub mod residency_pages;
pub mod route_telemetry;
pub mod source;
pub mod speculative;
pub mod state;
pub mod types;
