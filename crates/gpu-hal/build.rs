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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/metal_bridge.mm");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_cuda)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");

    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "auto".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    let is_auto = normalized == "auto";

    let explicit_hip = !is_auto && normalized.split(',').any(|part| part.trim() == "hip");
    let explicit_cuda = !is_auto && normalized.split(',').any(|part| part.trim() == "cuda");
    let explicit_metal = !is_auto && normalized.split(',').any(|part| part.trim() == "metal");

    let have_hip_toolchain = command_exists("hipcc");
    let have_cuda_toolchain = command_exists("nvcc");
    let have_metal_runtime = env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos");

    if explicit_hip && !have_hip_toolchain {
        panic!("SUPERSONIC_BACKENDS requested HIP, but hipcc is not available in PATH");
    }
    if explicit_cuda && !have_cuda_toolchain {
        panic!("SUPERSONIC_BACKENDS requested CUDA, but nvcc is not available in PATH");
    }
    if explicit_metal && !have_metal_runtime {
        panic!("SUPERSONIC_BACKENDS requested Metal, but this target is not macOS");
    }

    // Mirror kernel-ffi: in auto mode, prefer CUDA when both are present; otherwise
    // fall back to whichever toolchain is available. Explicit selection wins.
    let (enable_hip, enable_cuda, enable_metal) = if explicit_hip || explicit_cuda || explicit_metal
    {
        (explicit_hip, explicit_cuda, explicit_metal)
    } else if have_cuda_toolchain {
        (false, true, false)
    } else if have_hip_toolchain {
        (true, false, false)
    } else {
        (false, false, have_metal_runtime)
    };

    assert!(
        enable_hip || enable_cuda || enable_metal,
        "No supported GPU backend toolchain found for SUPERSONIC_BACKENDS={requested}. \
         Install hipcc and/or nvcc, or set SUPERSONIC_BACKENDS to an available backend."
    );

    if enable_hip {
        println!("cargo:rustc-cfg=supersonic_backend_hip");
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
