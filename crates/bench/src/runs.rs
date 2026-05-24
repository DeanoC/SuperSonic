use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaJson {
    pub schema_version: u32,
    pub run_id: String,
    pub timestamp_utc: String,
    pub git_sha: String,
    pub hostname: String,
    pub arch: String,
    pub rocminfo: String,
    pub rocm_smi_u: String,
    pub gpu_temp_c_pre: Option<f64>,
    pub gpu_temp_c_post: Option<f64>,
    pub runner_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum PerfStatus {
    Ok {
        ms_per_step: f64,
        ms_per_tok: f64,
        samples: Vec<f64>,
    },
    Skipped {
        reason: String,
    },
    Error {
        stderr_tail: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfCellJson {
    pub schema_version: u32,
    pub model: String,
    pub quant: String,
    pub arch: String,
    pub backend: String,
    pub prompt: String,
    pub max_new_tokens: u32,
    #[serde(flatten)]
    pub status: PerfStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_timings: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_breakdown: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_timings: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_stage_timings: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_chain_breakdown: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_lifecycle_timings: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpp_pilot: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mps_expert_pilot: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen36_pack_cache: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen36_expert_residency: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen36_expert_residency_policies: Option<Vec<BTreeMap<String, f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qwen36_expert_residency_policy_rows: Option<Vec<Qwen36ExpertResidencyPolicyJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal_profile: Option<ProfileJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hal_profile: Option<ProfileJson>,
    pub gpu_temp_c_end: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qwen36ExpertResidencyPolicyJson {
    pub resident_format: String,
    pub scope: String,
    pub miss_policy: String,
    pub capacity: f64,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileJson {
    pub summary: BTreeMap<String, f64>,
    pub entries: Vec<ProfileEntryJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntryJson {
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub calls: u64,
    pub mean_ms: f64,
    pub total_ms: f64,
    pub max_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn meta_path(&self) -> PathBuf {
        self.root.join("meta.json")
    }
    pub fn perf_path(&self, model: &str, quant: &str) -> PathBuf {
        self.root.join("perf").join(format!("{model}_{quant}.json"))
    }
    pub fn quality_path(&self, model: &str, quant: &str, eval: &str) -> PathBuf {
        self.root
            .join("quality")
            .join(format!("{model}_{quant}_{eval}.json"))
    }
    pub fn external_path(&self, engine: &str, model: &str, quant: &str) -> PathBuf {
        self.root
            .join("external")
            .join(engine)
            .join(format!("{model}_{quant}.json"))
    }

    /// Create the directory tree (root, perf/, quality/, external/).
    pub fn create(&self) -> io::Result<()> {
        std::fs::create_dir_all(self.root.join("perf"))?;
        std::fs::create_dir_all(self.root.join("quality"))?;
        std::fs::create_dir_all(self.root.join("external"))?;
        Ok(())
    }

    pub fn write_meta(&self, meta: &MetaJson) -> io::Result<()> {
        let s = serde_json::to_string_pretty(meta)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(self.meta_path(), s)
    }

    pub fn write_perf(&self, cell: &PerfCellJson) -> io::Result<()> {
        let path = self.perf_path(&cell.model, &cell.quant);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(cell)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, s)
    }
}

/// Compute a unique run-dir path under `parent`, dated today, with `-N` suffix on collision.
pub fn allocate_run_dir(parent: &Path, git_sha: &str, today: &str) -> PathBuf {
    let base = format!("{today}-{git_sha}");
    let mut candidate = parent.join(&base);
    let mut n = 2;
    while candidate.exists() {
        candidate = parent.join(format!("{base}-{n}"));
        n += 1;
    }
    candidate
}
