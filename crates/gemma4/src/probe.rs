//! Dry-run tensor probe.
//!
//! Opens the safetensors file(s) in a model directory and cross-references
//! actual tensor names / shapes / dtypes against [`weight_spec::all_tensors`].
//! Does NOT allocate GPU memory or copy tensor data — it only consults the
//! safetensors header via a memory map.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::config::TextConfig;
use crate::weight_spec::{all_tensors, TensorSpec};

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no safetensors files found in {0}")]
    NotFound(PathBuf),
}

/// Result of comparing the expected spec to what's actually in the checkpoint.
#[derive(Debug, Default)]
pub struct ProbeReport {
    /// Count of tensors whose name begins with `{prefix}.` in the file.
    pub actual_under_prefix: usize,
    /// Count of tensors expected by the spec.
    pub expected: usize,
    /// Expected but absent from the file.
    pub missing: Vec<String>,
    /// Expected shapes that don't match file shapes.
    pub shape_mismatches: Vec<ShapeMismatch>,
    /// Expected dtype (BF16) that doesn't match file dtype.
    pub dtype_mismatches: Vec<DtypeMismatch>,
    /// Tensors present under the prefix but not in the spec.
    pub extras_under_prefix: Vec<String>,
}

#[derive(Debug)]
pub struct ShapeMismatch {
    pub name: String,
    pub expected: Vec<usize>,
    pub actual: Vec<usize>,
}

#[derive(Debug)]
pub struct DtypeMismatch {
    pub name: String,
    pub actual_dtype: String,
}

impl ProbeReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty()
            && self.shape_mismatches.is_empty()
            && self.dtype_mismatches.is_empty()
            && self.extras_under_prefix.is_empty()
    }
}

/// Locate the safetensors file(s) in a model directory. Supports:
///   - `model.safetensors` (single file)
///   - `model.safetensors.index.json` (sharded)
fn find_shards(dir: &Path) -> Result<Vec<PathBuf>, ProbeError> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let weight_map = raw["weight_map"]
            .as_object()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no weight_map"))?;
        let mut files: BTreeSet<String> = BTreeSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                files.insert(s.to_string());
            }
        }
        Ok(files.into_iter().map(|n| dir.join(n)).collect())
    } else {
        let single = dir.join("model.safetensors");
        if single.exists() {
            Ok(vec![single])
        } else {
            Err(ProbeError::NotFound(dir.to_path_buf()))
        }
    }
}

/// Collect every tensor's name, shape, and dtype from the checkpoint shards.
fn collect_actual(shards: &[PathBuf]) -> Result<BTreeMap<String, (Vec<usize>, Dtype)>, ProbeError> {
    let mut out = BTreeMap::new();
    for path in shards {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        for name in tensors.names() {
            let view = tensors.tensor(name)?;
            out.insert(name.to_string(), (view.shape().to_vec(), view.dtype()));
        }
    }
    Ok(out)
}

pub fn probe(dir: &Path, cfg: &TextConfig, prefix: &str) -> Result<ProbeReport, ProbeError> {
    let shards = find_shards(dir)?;
    let actual = collect_actual(&shards)?;

    let expected: Vec<TensorSpec> = all_tensors(cfg, prefix);
    let expected_names: BTreeSet<&str> = expected.iter().map(|s| s.name.as_str()).collect();
    let prefix_with_dot = format!("{prefix}.");

    let actual_under_prefix_names: BTreeSet<&str> = actual
        .keys()
        .map(String::as_str)
        .filter(|n| n.starts_with(&prefix_with_dot))
        .collect();

    let mut report = ProbeReport {
        actual_under_prefix: actual_under_prefix_names.len(),
        expected: expected.len(),
        ..Default::default()
    };

    for spec in &expected {
        match actual.get(&spec.name) {
            None => report.missing.push(spec.name.clone()),
            Some((shape, dtype)) => {
                if shape != &spec.shape {
                    report.shape_mismatches.push(ShapeMismatch {
                        name: spec.name.clone(),
                        expected: spec.shape.clone(),
                        actual: shape.clone(),
                    });
                }
                if *dtype != Dtype::BF16 {
                    report.dtype_mismatches.push(DtypeMismatch {
                        name: spec.name.clone(),
                        actual_dtype: format!("{:?}", dtype),
                    });
                }
            }
        }
    }

    for name in actual_under_prefix_names {
        if !expected_names.contains(name) {
            report.extras_under_prefix.push(name.to_string());
        }
    }

    Ok(report)
}
