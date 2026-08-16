//! Shared SuperSonic runtime surface.
//!
//! This crate owns production model loading, session dispatch, token
//! generation, and chat-template helpers so the CLI and HTTP server can share
//! one runtime contract instead of growing separate startup paths.

#[path = "backend.rs"]
pub(crate) mod backend_resolver;
pub(crate) mod bakes;
pub(crate) mod builders;
pub mod chat_template;
pub mod decode_engine;
pub mod dflash;
pub mod flm_model_source;
pub mod flm_tokenizer;
pub mod generate;
pub mod ids;
pub mod oracle;
pub mod prefill_engine;
pub mod prefix_cache;
pub mod qwen36_moe;
pub mod qwen36_moe_config;
pub mod sampling;
pub mod session;
pub mod state;
pub mod tensor_bytes;

pub use state::{
    build, resolve_runtime_policy, LoaderConfig, RuntimeConfig, RuntimeLane, RuntimePolicy,
    ServerState,
};
pub use supersonic_core::{backend, capabilities, registry};
