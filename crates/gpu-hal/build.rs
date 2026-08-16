use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn detect_cuda_root() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(value) = env::var(var) {
            let path = PathBuf::from(value);
            if path.join("bin/nvcc").exists() {
                return Some(path);
            }
        }
    }

    let Ok(output) = Command::new("sh")
        .arg("-lc")
        .arg("command -v nvcc")
        .output()
    else {
        return None;
    };
    if !output.status.success() {
        return None;
    }

    let nvcc = fs::canonicalize(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
    .ok()?;
    nvcc.parent()
        .and_then(|bin| bin.parent())
        .map(Path::to_path_buf)
}

fn detect_cuda_lib_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(root) = detect_cuda_root() {
        candidates.extend([
            root.join("lib64"),
            root.join("targets/x86_64-linux/lib"),
            root.join("lib"),
        ]);
    }
    candidates.extend([
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
    ]);
    for candidate in candidates {
        if has_libcudart(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn has_libcudart(dir: &Path) -> bool {
    if dir.join("libcudart.so").exists() {
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
                .starts_with("libcudart.so.")
        })
}

fn detect_hipfile_root() -> Option<PathBuf> {
    for var in ["HIPFILE_ROOT", "ROCM_PATH", "ROCM_HOME"] {
        if let Ok(value) = env::var(var) {
            let path = PathBuf::from(value);
            if path.join("include/hipfile.h").exists() && has_libhipfile(&path.join("lib")) {
                return Some(path);
            }
            if path.join("include/hipfile.h").exists() && has_libhipfile(&path.join("lib64")) {
                return Some(path);
            }
        }
    }

    [
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/usr"),
        PathBuf::from("/usr/local"),
    ]
    .into_iter()
    .find(|path| {
        path.join("include/hipfile.h").exists()
            && (has_libhipfile(&path.join("lib")) || has_libhipfile(&path.join("lib64")))
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
    println!("cargo:rerun-if-changed=src/metal_bridge.mm");
    println!("cargo:rerun-if-changed=src/hipfile_bridge.cc");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=HIPFILE_ROOT");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hipfile)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_cuda)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");

    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "hip".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    let is_auto = normalized == "auto";

    let explicit_hip = !is_auto && normalized.split(',').any(|part| part.trim() == "hip");
    let explicit_cuda = !is_auto && normalized.split(',').any(|part| part.trim() == "cuda");
    let explicit_metal = !is_auto && normalized.split(',').any(|part| part.trim() == "metal");

    let have_hip_toolchain = command_exists("hipcc");

    if explicit_cuda || explicit_metal {
        panic!(
            "SUPERSONIC_BACKENDS={requested} is disabled on the slim HIP/Qwen branch. \
             Build with SUPERSONIC_BACKENDS=hip (the default)."
        );
    }
    if (explicit_hip || is_auto) && !have_hip_toolchain {
        panic!("SUPERSONIC_BACKENDS requested HIP, but hipcc is not available in PATH");
    }

    // Slim branch: HIP is the only compiled backend. `auto` means HIP.
    let (enable_hip, enable_cuda, enable_metal) = (true, false, false);

    assert!(
        enable_hip || enable_cuda || enable_metal,
        "No supported GPU backend toolchain found for SUPERSONIC_BACKENDS={requested}. \
         Install hipcc and/or nvcc, or set SUPERSONIC_BACKENDS to an available backend."
    );

    if enable_hip {
        println!("cargo:rustc-cfg=supersonic_backend_hip");
        if let Some(hipfile_root) = detect_hipfile_root() {
            if let Some(hipfile_lib_dir) = detect_hipfile_lib_dir(&hipfile_root) {
                let mut build = cc::Build::new();
                build
                    .cpp(true)
                    .file("src/hipfile_bridge.cc")
                    .include(hipfile_root.join("include"))
                    .flag("-std=c++17")
                    // Host-side HIP headers (ROCm 7+) refuse to parse unless the
                    // platform is selected. hipcc defines this automatically;
                    // the cc crate compiles this bridge with the host C++ compiler.
                    .define("__HIP_PLATFORM_AMD__", None);
                build.compile("gpu_hal_hipfile");
                println!("cargo:rustc-link-search=native={}", hipfile_lib_dir.display());
                println!("cargo:rustc-link-lib=hipfile");
                println!("cargo:rustc-cfg=supersonic_backend_hipfile");
            }
        }
    }
    if enable_cuda {
        if let Some(cuda_lib_dir) = detect_cuda_lib_dir() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        } else {
            println!(
                "cargo:warning=could not locate libcudart under CUDA_HOME/CUDA_PATH or common system library paths; falling back to linker search path"
            );
        }
        println!("cargo:rustc-cfg=supersonic_backend_cuda");
    }
    if enable_metal {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file("src/metal_bridge.mm")
            .flag("-std=c++17")
            .flag("-fobjc-arc");
        build.compile("gpu_hal_metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-cfg=supersonic_backend_metal");
    }
}
