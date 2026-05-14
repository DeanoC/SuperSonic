use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use gpu_hal::Backend;
use qwen35::config::{self, TextConfig};
use qwen35::rotary::RotaryTables;
use qwen35::weights::Qwen35Weights;

use super::args::{BackendArg, BughuntMode};
use super::report::RunMetadata;
use crate::backend_runtime;
use crate::decode_engine::DecodeEngine;
use crate::registry::{FamilyParams, ModelVariant};

pub(crate) struct QwenBughuntRuntime {
    pub(crate) backend: Backend,
    pub(crate) ordinal: usize,
    pub(crate) arch_name: String,
    pub(crate) model_dir: PathBuf,
    pub(crate) oracle_device: String,
    pub(crate) model_variant: ModelVariant,
    pub(crate) weights: Qwen35Weights,
    pub(crate) rotary: RotaryTables,
    pub(crate) kv_chunk_size: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) use_4b_kernel: bool,
    pub(crate) proj_buf_floats: usize,
    pub(crate) attn_scratch_floats: usize,
    pub(crate) weight_prefix: String,
    pub(crate) oracle_script: PathBuf,
    pub(crate) qwen35_trace_script: PathBuf,
    pub(crate) commit_ish: Option<String>,
}

impl QwenBughuntRuntime {
    pub(crate) fn new(
        model_dir: &Path,
        backend_choice: BackendArg,
        ordinal: usize,
        oracle_device_spec: &str,
        allow_untested_gpu: Option<&str>,
    ) -> Result<Self> {
        let backend = backend_runtime::resolve_backend(backend_choice.into(), ordinal)?;
        gpu_hal::set_backend(backend);

        let gpu = backend_runtime::query_gpu_info(backend, ordinal)?;
        let model_variant = ModelVariant::Qwen3_5_0_8B;
        let entry = backend_runtime::lookup_registry_entry(
            &model_variant,
            backend,
            &gpu.gpu_arch,
            allow_untested_gpu,
        )?;
        let params = match entry.params {
            FamilyParams::Qwen35(params) => params,
            _ => bail!("bughunt harness only supports Qwen3.5"),
        };

        let loaded = config::load_config(model_dir)
            .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
        let text_config = loaded.text_config;
        let weights = load_qwen35_weights(model_dir, &text_config, ordinal, params.weight_prefix)?;
        let rotary = RotaryTables::build(&text_config, ordinal)
            .map_err(|e| anyhow::anyhow!("rotary: {e}"))?;

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .context("runner crate missing repo root")?
            .to_path_buf();

        Ok(Self {
            backend,
            ordinal,
            arch_name: gpu.arch_name,
            model_dir: model_dir.to_path_buf(),
            oracle_device: backend_runtime::resolve_oracle_device(
                oracle_device_spec,
                backend,
                ordinal,
            ),
            model_variant,
            weights,
            rotary,
            kv_chunk_size: params.kv_chunk_size,
            prefill_chunk_size: 0,
            use_4b_kernel: params.use_4b_kernel,
            proj_buf_floats: params.proj_buf_floats,
            attn_scratch_floats: params.attn_scratch_floats,
            weight_prefix: params.weight_prefix.to_string(),
            oracle_script: repo_root.join("oracle/run_oracle.py"),
            qwen35_trace_script: repo_root.join("oracle/qwen35_oracle.py"),
            commit_ish: git_commit_ish(&repo_root),
        })
    }

    pub(crate) fn metadata(&self, mode: BughuntMode) -> RunMetadata {
        RunMetadata {
            mode: mode.as_str().to_string(),
            model: self.model_variant.to_string(),
            backend: self.backend.to_string(),
            device: self.ordinal,
            arch: self.arch_name.clone(),
            model_dir: self.model_dir.display().to_string(),
            oracle_device: self.oracle_device.clone(),
            commit_ish: self.commit_ish.clone(),
        }
    }

    pub(crate) fn new_component_decode_engine(
        &self,
        context_tokens: usize,
    ) -> Result<DecodeEngine> {
        let attn_scratch_floats = qwen35::scratch::required_attn_scratch_floats(
            self.weights.config.num_attention_heads,
            self.weights.config.head_dim,
            context_tokens,
            self.kv_chunk_size,
        )
        .max(self.attn_scratch_floats);
        let weights = load_qwen35_weights(
            &self.model_dir,
            &self.weights.config,
            self.ordinal,
            &self.weight_prefix,
        )?;
        DecodeEngine::new(
            weights,
            self.ordinal,
            self.proj_buf_floats,
            attn_scratch_floats,
            self.kv_chunk_size,
            self.use_4b_kernel,
            self.prefill_chunk_size,
            false,
            1,
        )
    }
}

fn git_commit_ish(repo_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn load_qwen35_weights(
    model_dir: &Path,
    text_config: &TextConfig,
    ordinal: usize,
    weight_prefix: &str,
) -> Result<Qwen35Weights> {
    let bake_dir = model_store::fetch::BakeVariant::Bf16.bake_dir(model_dir);
    if model_store::version_ok(&bake_dir) {
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow::anyhow!("open baked store: {e}"))?;
        Qwen35Weights::load_baked(&store, text_config, ordinal, weight_prefix)
            .map_err(|e| anyhow::anyhow!("load baked weights: {e}"))
    } else {
        Qwen35Weights::load(model_dir, text_config, ordinal, weight_prefix)
            .map_err(|e| anyhow::anyhow!("load weights: {e}"))
    }
}
