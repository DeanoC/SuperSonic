pub fn is_flm_model_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("flm"))
        .unwrap_or(false)
}

pub struct FlmModelSource {
    pub path: std::path::PathBuf,
    pub store: model_store::BakedStore,
}

impl FlmModelSource {
    pub fn open(path: &std::path::Path, int4_runtime: bool) -> anyhow::Result<Self> {
        let store = model_store::BakedStore::open_flm_with_options(
            path,
            model_store::FlmLoadOptions {
                flm_int4_logical_aliases: int4_runtime,
                ..Default::default()
            },
        )?;
        Ok(Self {
            path: path.to_path_buf(),
            store,
        })
    }

    pub fn qwen_config(&self) -> anyhow::Result<qwen35::config::Config> {
        let runtime = self.store.flm_runtime().ok_or_else(|| {
            anyhow::anyhow!("FLM {} has no runtime directory", self.path.display())
        })?;
        let cfg = runtime.qwen36_config().ok_or_else(|| {
            anyhow::anyhow!("FLM {} is not Qwen3.6 dense v1", self.path.display())
        })?;
        let config = qwen35::config::Config::try_from_flm_qwen36_dense(cfg).map_err(|e| {
            anyhow::anyhow!(
                "invalid FLM Qwen3.6 dense config in {}: {e}",
                self.path.display()
            )
        })?;
        Ok(config.normalized())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_flm_model_paths_by_extension() {
        assert!(is_flm_model_path(std::path::Path::new("qwen36.flm")));
        assert!(is_flm_model_path(std::path::Path::new("QWEN36.FLM")));
        assert!(!is_flm_model_path(std::path::Path::new("qwen36")));
        assert!(!is_flm_model_path(std::path::Path::new("qwen36.bin")));
    }
}
