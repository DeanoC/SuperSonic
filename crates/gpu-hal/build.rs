use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn detect_hipfile_root() -> Option<PathBuf> {
    for var in ["HIPFILE_ROOT", "ROCM_PATH", "ROCM_HOME"] {
        if let Ok(value) = env::var(var) {
            let root = PathBuf::from(value);
            if root.join("include/hipfile.h").exists()
                && (has_libhipfile(&root.join("lib")) || has_libhipfile(&root.join("lib64")))
            {
                return Some(root);
            }
        }
    }

    [
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr"),
        PathBuf::from("/usr/local"),
    ]
    .into_iter()
    .find(|root| {
        root.join("include/hipfile.h").exists()
            && (has_libhipfile(&root.join("lib")) || has_libhipfile(&root.join("lib64")))
    })
}

fn detect_hipfile_lib_dir(root: &Path) -> Option<PathBuf> {
    [root.join("lib"), root.join("lib64")]
        .into_iter()
        .find(|path| has_libhipfile(path))
}

fn has_libhipfile(dir: &Path) -> bool {
    if dir.join("libhipfile.so").exists() {
        return true;
    }
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("libhipfile.so.")
        })
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/hipfile_bridge.cc");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=HIPFILE_ROOT");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hipfile)");

    assert!(
        command_exists("hipcc"),
        "No HIP toolchain found; install hipcc."
    );
    println!("cargo:rustc-cfg=supersonic_backend_hip");

    let Some(hipfile_root) = detect_hipfile_root() else {
        return;
    };
    let Some(hipfile_lib_dir) = detect_hipfile_lib_dir(&hipfile_root) else {
        return;
    };

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/hipfile_bridge.cc")
        .include(hipfile_root.join("include"))
        .flag("-std=c++17")
        .define("__HIP_PLATFORM_AMD__", None);
    build.compile("gpu_hal_hipfile");
    println!(
        "cargo:rustc-link-search=native={}",
        hipfile_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=hipfile");
    println!("cargo:rustc-cfg=supersonic_backend_hipfile");
}
