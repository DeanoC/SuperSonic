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
pub mod generate;
pub mod ids;
pub mod sampling;
pub mod session;
pub mod state;

pub use state::{build, LoaderConfig, ServerState};
pub use supersonic_core::{backend, capabilities, registry};
