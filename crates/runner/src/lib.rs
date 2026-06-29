#![recursion_limit = "512"]

//! SuperSonic runner library — exposes the decode engines, prefill engine,
//! model/backend registry, and validation helpers so downstream crates (the
//! `server` crate in particular) can reuse them without duplicating the
//! module tree.
//!
//! The `supersonic` CLI binary lives in `src/main.rs` and historically
//! declared these modules locally via `mod …;`. Those declarations are kept
//! so existing `#[path]`-style bin crates continue to compile; with a lib
//! root present, the same files get compiled once into the library crate
//! (`runner::…`) for external consumers.

#[cfg(feature = "bughunt")]
pub mod backend_runtime;
#[cfg(feature = "bughunt")]
pub mod bughunt;
pub mod decode_engine;
pub mod dflash_ddtree;
pub mod flm_model_source;
pub mod gemma4_engine;
pub mod gemma4_int4_engine;
pub mod oracle;
pub mod prefill_engine;
pub mod profile;
#[path = "qwen36_moe/decode.rs"]
pub mod qwen36_moe_decode;
#[path = "qwen36_moe/logits.rs"]
pub mod qwen36_moe_logits;
#[path = "qwen36_moe/mtp.rs"]
pub mod qwen36_moe_mtp;
#[path = "qwen36_moe/persistent_decode.rs"]
pub mod qwen36_moe_persistent_decode;
#[path = "qwen36_moe/residency.rs"]
pub mod qwen36_moe_residency;
#[path = "qwen36_moe/residency_pages.rs"]
pub mod qwen36_moe_residency_pages;
#[path = "qwen36_moe/residency_types.rs"]
pub mod qwen36_moe_residency_types;
#[path = "qwen36_moe/speculative.rs"]
pub mod qwen36_moe_speculative;
#[path = "qwen36_moe/state.rs"]
pub mod qwen36_moe_state;
#[path = "qwen36_moe/telemetry.rs"]
pub mod qwen36_moe_telemetry;
#[path = "qwen36_moe/types.rs"]
pub mod qwen36_moe_types;
pub mod qwen36_q4km_audit;
pub mod registry;
pub mod specprefill;
pub mod tensor_bytes;
pub mod validate;
