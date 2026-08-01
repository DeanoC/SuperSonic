use std::path::Path;

use supersonic_runtime::flm_model_source::{
    is_flm_model_path, FlmModelSource, FlmModelSourceOptions,
};

#[test]
fn runtime_helper_detects_flm_model_paths_and_runner_re_exports_it() {
    assert!(is_flm_model_path(Path::new("/tmp/model.flm")));
    assert!(is_flm_model_path(Path::new("/tmp/MODEL.FLM")));
    assert!(!is_flm_model_path(Path::new("/tmp/model-dir")));
    assert!(!is_flm_model_path(Path::new("/tmp/model.bin")));

    let _: Option<FlmModelSource> = None;
    let _: Option<FlmModelSourceOptions> = None;
    let _: Option<runner::flm_model_source::FlmModelSource> = None;
    let _: Option<runner::flm_model_source::FlmModelSourceOptions> = None;
    let _: fn(&Path) -> bool = runner::flm_model_source::is_flm_model_path;
    let _ = FlmModelSource::chat_template_source;
}
