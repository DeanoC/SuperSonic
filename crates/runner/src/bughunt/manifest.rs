use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

use super::report::PromptManifest;

pub(crate) fn load_prompt_manifest(path: &Path) -> Result<PromptManifest> {
    let manifest_text = fs::read_to_string(path)
        .with_context(|| format!("read prompt manifest {}", path.display()))?;
    parse_prompt_manifest_str(&manifest_text)
}

fn parse_prompt_manifest_str(manifest_text: &str) -> Result<PromptManifest> {
    let manifest: PromptManifest =
        serde_json::from_str(manifest_text).context("parse prompt manifest JSON")?;
    validate_prompt_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_prompt_manifest(manifest: &PromptManifest) -> Result<()> {
    if manifest.prompts.is_empty() {
        bail!("prompt manifest must contain at least one prompt");
    }
    let mut names = HashSet::new();
    for prompt in &manifest.prompts {
        if prompt.name.trim().is_empty() {
            bail!("prompt manifest contains an entry with an empty name");
        }
        if !names.insert(prompt.name.clone()) {
            bail!(
                "prompt manifest contains duplicate prompt name '{}'",
                prompt.name
            );
        }
        if prompt.prompt_ids.is_empty() {
            bail!(
                "prompt '{}' must contain at least one prompt token id",
                prompt.name
            );
        }
        if prompt.positions.is_empty() {
            bail!(
                "prompt '{}' must contain at least one checked position",
                prompt.name
            );
        }
        let mut seen_positions = HashSet::new();
        for &position in &prompt.positions {
            if position >= prompt.prompt_ids.len() {
                bail!(
                    "prompt '{}' position {} is out of range for {} prompt tokens",
                    prompt.name,
                    position,
                    prompt.prompt_ids.len()
                );
            }
            if !seen_positions.insert(position) {
                bail!(
                    "prompt '{}' contains duplicate checked position {}",
                    prompt.name,
                    position
                );
            }
        }
        validate_positive_threshold(
            "prefill_logit_max_abs",
            prompt.thresholds.prefill_logit_max_abs,
            &prompt.name,
        )?;
        validate_positive_threshold(
            "layer_hidden_max_abs",
            prompt.thresholds.layer_hidden_max_abs,
            &prompt.name,
        )?;
        validate_positive_threshold(
            "restart_tail_logit_max_abs",
            prompt.thresholds.restart_tail_logit_max_abs,
            &prompt.name,
        )?;
    }
    Ok(())
}

fn validate_positive_threshold(label: &str, value: f32, prompt_name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!(
            "prompt '{}' threshold '{}' must be a finite positive number",
            prompt_name,
            label
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn manifest_validation_rejects_empty_prompt_ids() {
        let bad = json!({
            "prompts": [{
                "name": "bad",
                "prompt_ids": [],
                "positions": [0],
                "thresholds": {
                    "prefill_logit_max_abs": 0.1,
                    "layer_hidden_max_abs": 0.1,
                    "restart_tail_logit_max_abs": 0.1
                }
            }]
        });
        let err = parse_prompt_manifest_str(&bad.to_string()).unwrap_err();
        assert!(err.to_string().contains("at least one prompt token id"));
    }

    #[test]
    fn manifest_validation_requires_thresholds() {
        let bad = json!({
            "prompts": [{
                "name": "bad",
                "prompt_ids": [1, 2],
                "positions": [0]
            }]
        });
        assert!(parse_prompt_manifest_str(&bad.to_string()).is_err());
    }
}
