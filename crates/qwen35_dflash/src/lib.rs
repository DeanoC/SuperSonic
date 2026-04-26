//! DFlash speculative-decoding draft model for Qwen3.5-9B.
//!
//! This is the Rust+HIP port of `z-lab/Qwen3.5-9B-DFlash`'s draft network.
//! Spec source-of-truth: `docs/dflash.md` at the repo root, grounded in the
//! canonical `dflash.py` shipped in the HuggingFace checkpoint.
//!
//! Draft shape at a glance: 5 bidirectional full-attention layers, hidden=4096,
//! GQA 32:8, head_dim=128, intermediate=12288, block_size=16, tap layers
//! `[1, 8, 15, 22, 29]` of the Qwen3.5-9B target, fused via
//! `fc: [20480 → 4096] + hidden_norm` once per round.
//!
//! The draft owns neither `embed_tokens` nor `lm_head`; both come in via
//! `Arc<GpuBuffer>` shared with the target model.

pub mod config;
pub mod forward;
pub mod loader;
pub mod rotary;
pub mod state;
pub mod weights;

pub use config::{load_config, DFlashConfig};
pub use forward::{forward, ForwardParams};
pub use loader::LoadError;
pub use rotary::RotaryTables;
pub use state::{DFlashLayerKv, DFlashScratch, DFlashState};
pub use weights::{DFlashLayerWeights, DFlashWeights};
