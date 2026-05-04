use std::fs;
use std::path::Path;

pub(crate) fn model_dir_has_raw_safetensors(model_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(model_dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
    })
}
