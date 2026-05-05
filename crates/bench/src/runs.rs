use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;

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
    pub prompt: String,
    pub max_new_tokens: u32,
    #[serde(flatten)]
    pub status: PerfStatus,
    pub gpu_temp_c_end: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    pub fn new(root: PathBuf) -> Self { Self { root } }
    pub fn root(&self) -> &Path { &self.root }
    pub fn meta_path(&self) -> PathBuf { self.root.join("meta.json") }
    pub fn perf_path(&self, model: &str, quant: &str) -> PathBuf {
        self.root.join("perf").join(format!("{model}_{quant}.json"))
    }
    pub fn quality_path(&self, model: &str, quant: &str, eval: &str) -> PathBuf {
        self.root.join("quality").join(format!("{model}_{quant}_{eval}.json"))
    }
    pub fn external_path(&self, engine: &str, model: &str, quant: &str) -> PathBuf {
        self.root.join("external").join(engine).join(format!("{model}_{quant}.json"))
    }
}
