use std::env;
use std::path::PathBuf;
use std::process::Command;

fn detect_hip_archs() -> Vec<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        return arch
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
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
        .find(|t| t.starts_with("gfx"))
        .map(|s| vec![s.to_owned()])
        .unwrap_or_default()
}

fn have_hipcc() -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg("command -v hipcc >/dev/null 2>&1")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn have_nvcc() -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg("command -v nvcc >/dev/null 2>&1")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/gpu/metal_bridge.mm");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");

    // Mirror kernel-ffi's SUPERSONIC_BACKENDS gating exactly:
    // - explicit non-HIP selection (e.g. `cuda` or `metal`) must skip the
    //   HIP path even when `hipcc` is in PATH;
    // - in `auto` mode on dual-toolchain hosts (both nvcc and hipcc present)
    //   kernel-ffi and gpu-hal prefer CUDA, so we must NOT compile HIP here
    //   either, otherwise the rest of the workspace would build CUDA while
    //   this crate links amdhip64.
    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "hip".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    if normalized.split(',').any(|p| {
        let p = p.trim();
        p == "cuda" || p == "metal"
    }) {
        panic!(
            "SUPERSONIC_BACKENDS={requested} is disabled on the slim HIP/Qwen branch. \
             Build with SUPERSONIC_BACKENDS=hip (the default)."
        );
    }
    let explicit_hip = normalized == "hip"
        || normalized == "auto"
        || normalized.split(',').any(|p| p.trim() == "hip");
    let explicit_metal = false;
    let auto = normalized == "auto";
    let is_macos = env::var("CARGO_CFG_TARGET_OS")
        .map(|os| os == "macos")
        .unwrap_or(false);
    let want_metal = (explicit_metal || auto) && is_macos && !explicit_hip;
    if want_metal {
        let have_mtl4_mpp = have_mtl4_mpp_sdk();
        let mut metal = cc::Build::new();
        metal
            .cpp(true)
            .file("src/gpu/metal_bridge.mm")
            .flag_if_supported("-std=c++17")
            .flag("-fobjc-arc");
        if have_mtl4_mpp {
            metal.define("SUPERSONIC_HAVE_MTL4_MPP", "1");
        } else {
            println!(
                "cargo:warning=Metal 4 MPP tensor headers not found; MPP profile rows disabled"
            );
        }
        metal.compile("machine_profile_metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        if have_mtl4_mpp {
            println!("cargo:rustc-link-lib=framework=MetalPerformancePrimitives");
        }
        println!("cargo:rustc-cfg=supersonic_backend_metal");
    }

    let want_hip = if explicit_hip {
        true
    } else if auto {
        // Auto mode: defer to CUDA when both toolchains are present.
        !want_metal && !have_nvcc()
    } else {
        false
    };
    if !want_hip {
        return;
    }

    if !have_hipcc() {
        println!("cargo:warning=hipcc not found; machine-profile GPU kernels disabled");
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels = manifest.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archs = detect_hip_archs();

    let sources = [
        ("lds_bandwidth.hip", "lds_bandwidth.o"),
        ("hbm_bandwidth.hip", "hbm_bandwidth.o"),
        ("wmma_peak.hip", "wmma_peak.o"),
        ("profile_bridge.cpp", "profile_bridge.o"),
    ];
    let mut objects = Vec::new();
    for (src, obj) in sources {
        println!("cargo:rerun-if-changed={}", kernels.join(src).display());
        let obj_path = out_dir.join(obj);
        let mut cmd = Command::new("hipcc");
        cmd.args(["-std=c++17", "-O3", "-fPIC", "-x", "hip", "-c"])
            .arg(kernels.join(src))
            .args(["-I"])
            .arg(&kernels)
            .arg("-o")
            .arg(&obj_path);
        for a in &archs {
            cmd.arg(format!("--offload-arch={a}"));
        }
        let status = cmd.status().expect("hipcc failed to start");
        assert!(status.success(), "hipcc failed for {src}");
        objects.push(obj_path);
    }
    let lib = out_dir.join("libmp_profile_hip.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objects {
        ar.arg(o);
    }
    ar.status().expect("ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mp_profile_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_hip");
}

fn have_mtl4_mpp_sdk() -> bool {
    let Ok(output) = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let sdk = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_string());
    sdk.join("System/Library/Frameworks/Metal.framework/Headers/MTL4CommandQueue.h")
        .is_file()
        && sdk
            .join("System/Library/Frameworks/Metal.framework/Headers/MTLTensor.h")
            .is_file()
        && sdk
            .join("System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MetalPerformancePrimitives.h")
            .is_file()
}
