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

fn detect_hip_archs() -> Vec<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        let list: Vec<String> = arch
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if !list.is_empty() {
            return list;
        }
    }
    let Ok(output) = Command::new("rocminfo").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|token| token.starts_with("gfx"))
        .map(|s| vec![s.to_owned()])
        .unwrap_or_default()
}

fn detect_cuda_archs() -> Vec<String> {
    if let Ok(arch) = env::var("CUDA_ARCH") {
        let list: Vec<String> = arch
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_start_matches("sm").to_owned())
            .collect();
        if !list.is_empty() {
            return list;
        }
    }
    let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output() else {
        return vec!["86".to_string()];
    };
    if !output.status.success() {
        return vec!["86".to_string()];
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| {
            let mut parts = line.trim().split('.');
            Some(format!("{}{}", parts.next()?, parts.next()?))
        })
        .map(|arch| vec![arch])
        .unwrap_or_else(|| vec!["86".to_string()])
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
        .output() else {
        return None;
    };
    if !output.status.success() {
        return None;
    }

    let nvcc = fs::canonicalize(PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
        .ok()?;
    nvcc.parent()
        .and_then(|bin| bin.parent())
        .map(Path::to_path_buf)
}

fn detect_cuda_lib_dir() -> Option<PathBuf> {
    let root = detect_cuda_root()?;
    for candidate in [
        root.join("lib64"),
        root.join("targets/x86_64-linux/lib"),
        root.join("lib"),
    ] {
        if candidate.join("libcudart.so").exists() {
            return Some(candidate);
        }
    }
    None
}

fn run(cmd: &mut Command, context: &str) {
    let status = cmd.status().unwrap_or_else(|err| {
        panic!("{context}: failed to start command {:?}: {err}", cmd);
    });
    assert!(
        status.success(),
        "{context}: command {:?} failed with {status}",
        cmd
    );
}

fn archive(out_dir: &Path, lib_name: &str, objects: &[PathBuf], context: &str) {
    let lib_path = out_dir.join(format!("lib{lib_name}.a"));
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib_path);
    for obj in objects {
        ar.arg(obj);
    }
    run(&mut ar, context);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
}

fn compile_hip(kernel_dir: &Path, out_dir: &Path) {
    let sources = [
        ("full_attention_bridge.cpp", "qwen35_megakernel_hip.o", "building qwen35 megakernel HIP bridge"),
        ("full_attention_bridge_4b.cpp", "qwen35_4b_megakernel_hip.o", "building qwen35-4b megakernel HIP bridge"),
        ("prefill_helpers_bridge.cpp", "qwen35_prefill_helpers_hip.o", "building prefill helpers HIP bridge"),
        ("gemma4_bridge.cpp", "gemma4_hip.o", "building Gemma 4 HIP bridge"),
    ];
    let archs = detect_hip_archs();
    if archs.is_empty() {
        println!("cargo:warning=no HIP arch detected (set HIP_ARCH or install rocminfo); kernel binary may not run on the target GPU");
    } else {
        println!("cargo:warning=building HIP kernels for arch(es): {}", archs.join(", "));
    }

    let mut objects = Vec::new();
    for (src_name, obj_name, context) in sources {
        let mut cmd = Command::new("hipcc");
        let obj_path = out_dir.join(obj_name);
        cmd.arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-I")
            .arg(kernel_dir)
            .arg("-x")
            .arg("hip")
            .arg("-c")
            .arg(kernel_dir.join(src_name))
            .arg("-o")
            .arg(&obj_path);
        for arch in &archs {
            cmd.arg(format!("--offload-arch={arch}"));
        }
        run(&mut cmd, context);
        objects.push(obj_path);
    }

    archive(
        out_dir,
        "qwen35_megakernel_hip",
        &objects,
        "archiving qwen35 megakernel HIP bridges",
    );
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_hip");
}

fn compile_cuda(kernel_dir: &Path, out_dir: &Path) {
    let sources = [
        ("full_attention_bridge_cuda.cu", "qwen35_megakernel_cuda.o", "building qwen35 megakernel CUDA bridge"),
        ("full_attention_bridge_4b_cuda.cu", "qwen35_4b_megakernel_cuda.o", "building qwen35-4b megakernel CUDA bridge"),
        ("prefill_helpers_bridge_cuda.cu", "qwen35_prefill_helpers_cuda.o", "building prefill helpers CUDA bridge"),
    ];
    let archs = detect_cuda_archs();
    println!("cargo:warning=building CUDA kernels for arch(es): {}", archs.join(", "));

    let mut objects = Vec::new();
    for (src_name, obj_name, context) in sources {
        let mut cmd = Command::new("nvcc");
        let obj_path = out_dir.join(obj_name);
        cmd.arg("-std=c++17")
            .arg("-O3")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-I")
            .arg(kernel_dir)
            .arg("-c")
            .arg(kernel_dir.join(src_name))
            .arg("-o")
            .arg(&obj_path);
        for arch in &archs {
            cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        }
        run(&mut cmd, context);
        objects.push(obj_path);
    }

    archive(
        out_dir,
        "qwen35_megakernel_cuda",
        &objects,
        "archiving qwen35 megakernel CUDA bridges",
    );
    let cuda_lib_dir = detect_cuda_lib_dir().unwrap_or_else(|| {
        panic!(
            "could not locate libcudart.so; set CUDA_HOME or CUDA_PATH to a valid CUDA toolkit root"
        )
    });
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_cuda");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_cuda)");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("cannot find workspace root")
        .join("kernels");
    for path in [
        "full_attention.hip",
        "full_attention_4b.hip",
        "prefill_helpers.hip",
        "full_attention_bridge.cpp",
        "full_attention_bridge_4b.cpp",
        "prefill_helpers_bridge.cpp",
        "gemma4.hip",
        "gemma4_bridge.cpp",
        "full_attention_cuda.cuh",
        "full_attention_4b_cuda.cuh",
        "prefill_helpers_cuda.cuh",
        "full_attention_bridge_cuda.cu",
        "full_attention_bridge_4b_cuda.cu",
        "prefill_helpers_bridge_cuda.cu",
    ] {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(path).display());
    }

    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "auto".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    let want_hip = normalized == "auto" || normalized.split(',').any(|part| part.trim() == "hip");
    let want_cuda = normalized == "auto" || normalized.split(',').any(|part| part.trim() == "cuda");
    let have_hip_toolchain = want_hip && command_exists("hipcc");
    let have_cuda_toolchain = want_cuda && command_exists("nvcc");

    assert!(
        have_hip_toolchain || have_cuda_toolchain,
        "No supported kernel toolchain found for SUPERSONIC_BACKENDS={requested}. \
         Install hipcc and/or nvcc, or set SUPERSONIC_BACKENDS to an available backend."
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    if want_hip && want_cuda && have_hip_toolchain && have_cuda_toolchain {
        panic!(
            "SUPERSONIC_BACKENDS={requested} is not supported by kernel-ffi yet: \
             HIP and CUDA bridge archives export the same symbol set. \
             Choose one backend, or build on a machine with only one toolchain available."
        );
    }

    if normalized == "hip" {
        compile_hip(&kernel_dir, &out_dir);
    } else if normalized == "cuda" {
        compile_cuda(&kernel_dir, &out_dir);
    } else if have_cuda_toolchain {
        compile_cuda(&kernel_dir, &out_dir);
    } else {
        compile_hip(&kernel_dir, &out_dir);
    }
}
