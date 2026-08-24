use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    fs::read_to_string(workspace_root().join(relative))
        .unwrap_or_else(|err| panic!("read {relative}: {err}"))
}

#[test]
fn qwen38_runtime_exposes_no_removed_validation_or_batch_routes() {
    let runtime_lib = read("crates/runtime/src/lib.rs");
    let decode = read("crates/runtime/src/decode_engine.rs");

    assert!(!runtime_lib.contains("pub mod oracle;"));
    assert!(!workspace_root()
        .join("crates/runtime/src/oracle.rs")
        .exists());
    for removed in [
        "pub fn load_prefill_state",
        "pub fn replicate_state_to_batch",
        "pub fn decode_step_batch",
        "pub fn decode_step_batch_with_timings",
        "SUPERSONIC_QWEN38_GQH_COMPONENT_DECODE",
        "pub fn kv_fp8_enabled",
        "pub fn virtual_kv_memory_stats",
        "pub fn evict_virtual_kv_to_host",
        "pub fn restore_virtual_kv_from_host",
    ] {
        assert!(
            !decode.contains(removed),
            "removed runtime surface remains: {removed}"
        );
    }
    assert!(!decode.contains("extra_states"));
}

#[test]
fn qwen38_product_drops_kv_fp8_vmm_and_broad_capability_surfaces() {
    let root = workspace_root();
    let core_lib = read("crates/core/src/lib.rs");
    let gpu_hal_lib = read("crates/gpu-hal/src/lib.rs");
    let qwen_state = read("crates/qwen38/src/state.rs");
    let qwen_descs = read("crates/qwen38/src/desc_builder.rs");
    let runtime_scratch = read("crates/qwen38/src/scratch.rs");
    let kernel_lib = read("crates/kernel-ffi/src/lib.rs");

    assert!(!core_lib.contains("pub mod capabilities;"));
    assert!(!root.join("crates/core/src/capabilities.rs").exists());
    assert!(!gpu_hal_lib.contains("mod vmm;"));
    assert!(!gpu_hal_lib.contains("VirtualBuffer"));
    assert!(!root.join("crates/gpu-hal/src/vmm.rs").exists());
    assert!(!qwen_state.contains("VirtualBuffer"));
    assert!(!qwen_state.contains("VirtualKvMemoryStats"));
    assert!(!qwen_state.contains("KvFp8"));
    assert!(!qwen_state.contains("kv_fp8"));
    assert!(!qwen_descs.contains("KVCacheFp8Desc"));
    assert!(!qwen_descs.contains("build_kv_fp8_descs"));
    assert!(!runtime_scratch.contains("KVCacheFp8Desc"));
    assert!(!runtime_scratch.contains("kv_fp8_desc"));
    assert!(!kernel_lib.contains("KVCacheFp8Desc"));
}

#[test]
fn qwen38_retains_internal_mtp_b_slots_and_fp8_weight_descriptors() {
    let mtp = read("crates/runtime/src/decode_engine.rs");
    let qwen_descs = read("crates/qwen38/src/desc_builder.rs");
    let kernel_descs = read("crates/kernel-ffi/src/layer_desc.rs");

    assert!(mtp.contains("verify_block_fused_decode"));
    assert!(mtp.contains("BatchSeqDesc"));
    assert!(qwen_descs.contains("build_batch_seq_descs"));
    assert!(kernel_descs.contains("pub struct FP8ScaleDesc"));
    assert!(qwen_descs.contains("build_fp8_scale_descs"));
}
