pub mod manifest;
pub mod transforms;
pub mod baker;
pub mod store;

use std::path::{Path, PathBuf};

use manifest::{CONVERTER_VERSION, FORMAT_VERSION, Manifest};

pub use baker::{bake_qwen35, bake_qwen35_int4};
pub use store::BakedStore;

/// Error type for bake and load operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("GPU error: {0}")]
    Gpu(#[from] gpu_hal::GpuError),
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("{0}")]
    Other(String),
}

/// Return the bake directory for a given model directory.
pub fn bake_dir(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}"))
}

/// Return the bake directory for FP8 native mode (runtime dequant on GPU).
/// Uses a separate directory so BF16 and FP8 baked packages coexist.
pub fn bake_dir_fp8(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}-fp8"))
}

/// Check if a valid FP8-native baked package exists.
pub fn version_ok_fp8(bake_dir: &Path) -> bool {
    version_ok(bake_dir)
}

/// Return the bake directory for INT4 quantized mode.
/// Uses a separate directory so BF16/FP8/INT4 baked packages coexist.
pub fn bake_dir_int4(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}-int4"))
}

/// Path to manifest.json within a bake directory.
pub fn manifest_path(bake_dir: &Path) -> PathBuf {
    bake_dir.join("manifest.json")
}

/// Path to weights.bin within a bake directory.
pub fn weights_bin_path(bake_dir: &Path) -> PathBuf {
    bake_dir.join("weights.bin")
}

/// Check if a valid baked package exists at the given bake directory.
/// Returns false on any error (missing, corrupt, wrong version).
pub fn version_ok(bake_dir: &Path) -> bool {
    let mp = manifest_path(bake_dir);
    let Ok(text) = std::fs::read_to_string(&mp) else {
        return false;
    };
    let Ok(m) = serde_json::from_str::<Manifest>(&text) else {
        return false;
    };
    m.format_version == FORMAT_VERSION
        && m.converter_version == CONVERTER_VERSION
        && weights_bin_path(bake_dir).exists()
}
