use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../kernel-ffi/src/prefill_ffi.rs");
    println!("cargo:rustc-check-cfg=cfg(kernel_lab_has_int4_sparse_outlier_add)");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let prefill_ffi = manifest_dir.join("../kernel-ffi/src/prefill_ffi.rs");
    let Ok(src) = std::fs::read_to_string(prefill_ffi) else {
        return;
    };
    if src.contains("pub fn int4_sparse_outlier_add(") {
        println!("cargo:rustc-cfg=kernel_lab_has_int4_sparse_outlier_add");
    }
}
