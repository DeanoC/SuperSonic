use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct TestModel {
    pub canonical_subdir: &'static str,
    pub override_env: &'static str,
    pub hf_cache_repos: &'static [&'static str],
}

impl TestModel {
    pub const fn new(
        canonical_subdir: &'static str,
        override_env: &'static str,
        hf_cache_repos: &'static [&'static str],
    ) -> Self {
        Self {
            canonical_subdir,
            override_env,
            hf_cache_repos,
        }
    }
}

pub const QWEN3_30B_A3B: TestModel = TestModel::new(
    "qwen3-30b-a3b",
    "SUPERSONIC_TEST_QWEN3_30B_A3B_MODEL_DIR",
    &["models--Qwen--Qwen3-30B-A3B"],
);

pub const GEMMA4_E2B: TestModel = TestModel::new(
    "gemma4-e2b",
    "SUPERSONIC_TEST_GEMMA4_E2B_MODEL_DIR",
    &["models--google--gemma-4-E2B"],
);

pub const GEMMA4_E4B: TestModel = TestModel::new(
    "gemma4-e4b",
    "SUPERSONIC_TEST_GEMMA4_E4B_MODEL_DIR",
    &["models--google--gemma-4-E4B"],
);

pub const PHI4_MINI: TestModel = TestModel::new(
    "phi4-mini",
    "SUPERSONIC_TEST_PHI4_MINI_MODEL_DIR",
    &["models--microsoft--Phi-4-mini-instruct"],
);

pub fn resolve_model_dir(model: TestModel) -> Option<PathBuf> {
    if let Some(dir) = valid_model_dir_from_env(model.override_env) {
        return Some(dir);
    }

    if let Some(root) = std::env::var_os("SUPERSONIC_TEST_MODEL_ROOT") {
        let dir = PathBuf::from(root).join(model.canonical_subdir);
        if is_model_dir(&dir) {
            return Some(dir);
        }
    }

    discover_hf_cache_snapshot(model)
}

pub fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("runner crate should live under <repo>/crates/runner")
        .to_path_buf()
}

pub fn path_with_repo_venv() -> std::ffi::OsString {
    let venv_bin = repo_root().join(".venv/bin");
    let mut path_entries = Vec::new();
    if venv_bin.exists() {
        path_entries.push(venv_bin);
    }
    path_entries.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    std::env::join_paths(path_entries).expect("join PATH entries")
}

pub fn combined_output(output: &std::process::Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn valid_model_dir_from_env(env_name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env_name).map(PathBuf::from)?;
    if is_model_dir(&dir) {
        Some(dir)
    } else {
        eprintln!("skipping {env_name}: {} is not a model dir", dir.display());
        None
    }
}

fn discover_hf_cache_snapshot(model: TestModel) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    for base in [
        home.join(".cache/huggingface"),
        home.join(".cache/huggingface/hub"),
    ] {
        for repo in model.hf_cache_repos {
            let snapshots = base.join(repo).join("snapshots");
            let Ok(entries) = std::fs::read_dir(&snapshots) else {
                continue;
            };
            let mut candidates: Vec<PathBuf> =
                entries.flatten().map(|entry| entry.path()).collect();
            candidates.sort();
            candidates.reverse();
            for candidate in candidates {
                if is_model_dir(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn is_model_dir(path: &Path) -> bool {
    path.join("config.json").is_file() && path.join("tokenizer.json").is_file()
}
