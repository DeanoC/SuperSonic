pub fn is_flm_model_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("flm"))
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlmModelSourceOptions {
    pub int4_runtime: bool,
    pub verify_block_hashes: bool,
}

impl FlmModelSourceOptions {
    pub fn to_load_options(self) -> model_store::FlmLoadOptions {
        model_store::FlmLoadOptions {
            flm_int4_logical_aliases: self.int4_runtime,
            verify_block_hashes: self.verify_block_hashes,
        }
    }
}

pub struct FlmModelSource {
    pub path: std::path::PathBuf,
    pub store: model_store::BakedStore,
}

impl FlmModelSource {
    pub fn open_with_options(
        path: &std::path::Path,
        options: FlmModelSourceOptions,
    ) -> anyhow::Result<Self> {
        let store =
            model_store::BakedStore::open_flm_with_options(path, options.to_load_options())?;
        Ok(Self {
            path: path.to_path_buf(),
            store,
        })
    }

    pub fn runtime(&self) -> anyhow::Result<&model_store::FlmRuntimeDirectory> {
        self.store
            .flm_runtime()
            .ok_or_else(|| anyhow::anyhow!("FLM {} has no runtime directory", self.path.display()))
    }

    pub fn qwen_config(&self) -> anyhow::Result<qwen35::config::Config> {
        let runtime = self.runtime()?;
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

    pub fn qwen_moe_config(&self) -> anyhow::Result<qwen36_moe::config::Config> {
        let runtime = self.runtime()?;
        let cfg = runtime
            .qwen36_moe_config()
            .ok_or_else(|| anyhow::anyhow!("FLM {} is not Qwen3.6 MoE v1", self.path.display()))?;
        qwen36_moe::config::Config::try_from_flm_qwen36_moe(cfg).map_err(|e| {
            anyhow::anyhow!(
                "invalid FLM Qwen3.6 MoE config in {}: {e}",
                self.path.display()
            )
        })
    }

    pub fn qwen_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        crate::flm_tokenizer::load_qwen_bpe_from_flm(self.runtime()?)
            .map_err(|e| anyhow::anyhow!("loading FLM Qwen tokenizer: {e}"))
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

    #[test]
    fn open_options_enable_int4_aliases_and_hash_verification() {
        let load = FlmModelSourceOptions {
            int4_runtime: true,
            verify_block_hashes: true,
        }
        .to_load_options();

        assert!(load.flm_int4_logical_aliases);
        assert!(load.verify_block_hashes);
    }

    #[test]
    fn default_open_options_do_not_enable_runtime_conversions_or_hashes() {
        let load = FlmModelSourceOptions::default().to_load_options();

        assert!(!load.flm_int4_logical_aliases);
        assert!(!load.verify_block_hashes);
    }
}
