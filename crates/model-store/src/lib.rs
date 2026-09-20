//! CPU-side model artifact codecs.
//!
//! The product-facing loader is the custom GQH GGUF reader.  FLM remains an
//! internal, CPU-only format foundation until a later product design promotes
//! it through the same artifact and correctness gates.

#[allow(dead_code)]
mod codec;
pub mod dmix2;
// The FLM parser is intentionally dormant background format work.  Keeping it
// private and warning-tolerant makes that boundary explicit until a product
// loader is designed and gated.
pub mod dflash;
pub mod dflash_ref;
#[allow(dead_code)]
mod flm;
pub mod gguf;
pub mod gqh;
pub mod gqh_q8;
pub mod q2k;
pub mod q3k;
pub mod q6_bound;
pub mod q8_0;

/// Errors shared by the retained GGUF/GQH readers and internal FLM codecs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GPU error: {0}")]
    Gpu(#[from] gpu_hal::GpuError),
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Other(String),
}
