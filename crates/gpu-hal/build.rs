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

fn selected_backend() -> &'static str {
    match env::var("SUPERSONIC_BACKEND").ok().as_deref() {
        Some("metal") => "metal",
        Some("hip") | None => "hip",
        Some(other) => panic!("unsupported SUPERSONIC_BACKEND={other}; expected hip or metal"),
    }
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

fn detect_rocm_lib_dir() -> Option<PathBuf> {
    let mut roots = Vec::new();
    for var in ["ROCM_PATH", "HIP_PATH", "ROCM_HOME"] {
        if let Ok(value) = env::var(var) {
            roots.push(PathBuf::from(value));
        }
    }
    roots.extend([
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr"),
        PathBuf::from("/usr/local"),
    ]);
    roots
        .into_iter()
        .flat_map(|root| [root.join("lib"), root.join("lib64")])
        .find(|path| has_libamdhip64(path))
}

fn has_libamdhip64(dir: &Path) -> bool {
    if dir.join("libamdhip64.so").exists() {
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
                .starts_with("libamdhip64.so.")
        })
}

fn build_hip() {
    assert!(
        command_exists("hipcc"),
        "SUPERSONIC_BACKEND=hip requires hipcc in PATH"
    );
    println!("cargo:rustc-cfg=supersonic_backend_hip");
    let rocm_lib_dir = detect_rocm_lib_dir().expect(
        "No ROCm amdhip64 library found under ROCM_PATH, HIP_PATH, ROCM_HOME, or standard roots.",
    );
    println!("cargo:rustc-link-search=native={}", rocm_lib_dir.display());

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

fn build_metal() {
    if !cfg!(target_os = "macos") {
        panic!("SUPERSONIC_BACKEND=metal requires macOS");
    }
    println!("cargo:rustc-cfg=supersonic_backend_metal");
    cc::Build::new()
        .file("src/metal_bridge.mm")
        .cpp(true)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .compile("gpu_hal_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/hipfile_bridge.cc");
    println!("cargo:rerun-if-changed=src/metal_bridge.mm");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKEND");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=HIPFILE_ROOT");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hipfile)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");

    match selected_backend() {
        "hip" => build_hip(),
        "metal" => build_metal(),
        other => panic!("unsupported backend selection: {other}"),
    }
}
