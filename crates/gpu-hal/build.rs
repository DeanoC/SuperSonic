use std::env;
use std::process::Command;

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_cuda)");

    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "auto".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    let is_auto = normalized == "auto";

    let explicit_hip = !is_auto && normalized.split(',').any(|part| part.trim() == "hip");
    let explicit_cuda = !is_auto && normalized.split(',').any(|part| part.trim() == "cuda");

    let have_hip_toolchain = command_exists("hipcc");
    let have_cuda_toolchain = command_exists("nvcc");

    if explicit_hip && !have_hip_toolchain {
        panic!("SUPERSONIC_BACKENDS requested HIP, but hipcc is not available in PATH");
    }
    if explicit_cuda && !have_cuda_toolchain {
        panic!("SUPERSONIC_BACKENDS requested CUDA, but nvcc is not available in PATH");
    }

    // Mirror kernel-ffi: in auto mode, prefer CUDA when both are present; otherwise
    // fall back to whichever toolchain is available. Explicit selection wins.
    let (enable_hip, enable_cuda) = if explicit_hip || explicit_cuda {
        (explicit_hip, explicit_cuda)
    } else if have_cuda_toolchain {
        (false, true)
    } else {
        (have_hip_toolchain, false)
    };

    assert!(
        enable_hip || enable_cuda,
        "No supported GPU backend toolchain found for SUPERSONIC_BACKENDS={requested}. \
         Install hipcc and/or nvcc, or set SUPERSONIC_BACKENDS to an available backend."
    );

    if enable_hip {
        println!("cargo:rustc-cfg=supersonic_backend_hip");
    }
    if enable_cuda {
        println!("cargo:rustc-cfg=supersonic_backend_cuda");
    }
}
