//! Shared Qwen3.8 runtime surface.
//!
//! The runner uses the retained decode/prefill engines and chat-template
//! helper directly. Server dispatch, broad family selection, and public FLM
//! loading are intentionally outside this crate's product surface.

pub mod chat_template;
pub mod decode_engine;
pub mod mtp;
pub mod prefill_engine;
pub mod tensor_bytes;
