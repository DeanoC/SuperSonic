pub mod baker;
pub mod fetch;
pub mod flm;
pub mod manifest;
pub mod store;
pub mod transforms;

use std::path::{Path, PathBuf};

use manifest::{Manifest, QuantProfile, CONVERTER_VERSION, FORMAT_VERSION};

pub use baker::{bake_phi4, bake_qwen35};
pub use flm::{
    FlmAsset, FlmQwen36DenseConfig, FlmQwen36MoeConfig, FlmRuntimeDirectory, FlmRuntimeIdentity,
    FlmTokenizerDescriptor,
};
pub use store::{
    read_flm_runtime_identity, BakedStore, FlmLoadOptions, TensorStorageExtent,
    TensorStorageSourceKind,
};

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

/// Return the bake directory for INT4 quantized mode (GPTQ calibration baker).
/// Uses a separate directory so BF16/FP8/INT4 baked packages coexist. The
/// "-gptq" suffix invalidates older naive min/max bakes automatically.
pub fn bake_dir_int4(model_dir: &Path) -> PathBuf {
    bake_dir_for_quant_profile(model_dir, QuantProfile::Int4Gptq)
}

/// Return the bake directory for any named weight quantization profile.
pub fn bake_dir_for_quant_profile(model_dir: &Path, profile: QuantProfile) -> PathBuf {
    match profile {
        QuantProfile::Bf16 => bake_dir(model_dir),
        QuantProfile::Fp8Native => bake_dir_fp8(model_dir),
        QuantProfile::Q4Km => bake_dir_q4km(model_dir),
        QuantProfile::Q4KmGptq => bake_dir_q4km_gptq(model_dir),
        other => model_dir
            .join(".supersonic")
            .join(format!("v{FORMAT_VERSION}-{}", other.as_str())),
    }
}

/// Return the bake directory for GGUF Q4_K_M-compatible raw GGML K-block weights.
pub fn bake_dir_q4km(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}-q4km"))
}

/// Return the bake directory for Q4KM-sourced GPTQ/native INT4 weights.
pub fn bake_dir_q4km_gptq(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}-q4km-gptq"))
}

/// Return the bake directory for INT8 BitsAndBytes-style weights.
/// Uses a separate directory so BF16/FP8/INT4/INT8 baked packages coexist.
pub fn bake_dir_int8(model_dir: &Path) -> PathBuf {
    model_dir
        .join(".supersonic")
        .join(format!("v{FORMAT_VERSION}-int8-bnb"))
}

/// Path to manifest.json within a bake directory.
pub fn manifest_path(bake_dir: &Path) -> PathBuf {
    bake_dir.join("manifest.json")
}

/// Path to weights.bin within a bake directory.
pub fn weights_bin_path(bake_dir: &Path) -> PathBuf {
    bake_dir.join("weights.bin")
}

/// Exclusive filesystem lock held for the duration of a bake or fetch
/// operation. Prevents two concurrent `supersonic` invocations from racing on
/// the same `{model_dir}/.supersonic/` directory. Released on drop.
pub struct BakeLock {
    _file: std::fs::File,
    path: PathBuf,
}

impl BakeLock {
    /// Acquire the lock. Blocks until the existing holder releases it.
    pub fn acquire(model_dir: &Path) -> std::io::Result<Self> {
        use fs2::FileExt;
        let dir = model_dir.join(".supersonic");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(".lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        file.lock_exclusive()?;
        Ok(Self { _file: file, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
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

/// Check if a valid Q4KM GGML K-block baked package exists at the given bake directory.
pub fn version_ok_q4km(bake_dir: &Path) -> bool {
    version_ok_with_quant_profile(bake_dir, "q4km-ggml-v1")
}

/// Check if a valid Q4KM GPTQ baked package exists at the given bake directory.
pub fn version_ok_q4km_gptq(bake_dir: &Path) -> bool {
    version_ok_with_quant_profile(bake_dir, "q4km-gptq-v1")
}

fn version_ok_with_quant_profile(bake_dir: &Path, quant_profile: &str) -> bool {
    let mp = manifest_path(bake_dir);
    let Ok(text) = std::fs::read_to_string(&mp) else {
        return false;
    };
    let Ok(m) = serde_json::from_str::<Manifest>(&text) else {
        return false;
    };
    m.format_version == FORMAT_VERSION
        && m.converter_version == CONVERTER_VERSION
        && m.quant_profile.as_deref() == Some(quant_profile)
        && weights_bin_path(bake_dir).exists()
}

/// Check if a valid baked package exists and declares the expected modern
/// quant profile. BF16 accepts legacy manifests that omitted `quant_profile`.
pub fn version_ok_for_quant_profile(bake_dir: &Path, expected: QuantProfile) -> bool {
    let mp = manifest_path(bake_dir);
    let Ok(text) = std::fs::read_to_string(&mp) else {
        return false;
    };
    let Ok(m) = serde_json::from_str::<Manifest>(&text) else {
        return false;
    };
    if m.format_version != FORMAT_VERSION
        || m.converter_version != CONVERTER_VERSION
        || !weights_bin_path(bake_dir).exists()
    {
        return false;
    }
    match (expected, m.quant_profile.as_deref()) {
        (QuantProfile::Bf16, None | Some("bf16")) => true,
        (_, Some(got)) => got == expected.as_str(),
        _ => false,
    }
}
