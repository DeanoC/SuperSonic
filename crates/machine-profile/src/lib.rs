//! Machine profiling — measure CPU + GPU hardware capabilities.

pub mod catalog;
pub mod fingerprint;
pub mod schema;
pub mod store;

pub use schema::Profile;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend not implemented: {0}")]
    BackendNotImplemented(&'static str),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
