use std::path::Path;

#[test]
fn public_helper_detects_flm_model_paths() {
    assert!(runner::flm_model_source::is_flm_model_path(Path::new(
        "/tmp/model.flm"
    )));
    assert!(runner::flm_model_source::is_flm_model_path(Path::new(
        "/tmp/MODEL.FLM"
    )));
    assert!(!runner::flm_model_source::is_flm_model_path(Path::new(
        "/tmp/model-dir"
    )));
    assert!(!runner::flm_model_source::is_flm_model_path(Path::new(
        "/tmp/model.bin"
    )));
}
