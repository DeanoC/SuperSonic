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
pub mod bughunt;
pub mod decode_engine;
pub mod gemma4_engine;
pub mod gemma4_int4_engine;
pub mod oracle;
pub mod prefill_engine;
pub mod qwen36_moe_decode;
pub mod qwen36_moe_mtp;
pub mod qwen36_moe_speculative;
pub mod qwen36_moe_state;
pub mod registry;
pub mod validate;
