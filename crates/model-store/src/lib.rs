//! CPU-side model artifact codecs.
//!
//! The product-facing loader is the custom GQH GGUF reader.  FLM remains an
//! internal, CPU-only format foundation until a later product design promotes
//! it through the same artifact and correctness gates.

#[doc(hidden)]
pub mod codec;
pub mod dmix2;
#[doc(hidden)]
pub mod flm;
pub mod gguf;
pub mod gqh;
pub mod q2k;
pub mod q3k;

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
