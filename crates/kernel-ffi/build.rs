use std::collections::BTreeSet;
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

fn verbose_build_warnings() -> bool {
    env::var_os("SUPERSONIC_BUILD_VERBOSE").is_some()
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
    let archs: BTreeSet<String> = stdout
        .split_whitespace()
        .filter_map(|token| {
            let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
            let suffix = token.strip_prefix("gfx")?;
            if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_hexdigit()) {
                return None;
            }
            Some(token.to_owned())
        })
        .collect();
    archs.into_iter().collect()
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
        .output()
    else {
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

fn detect_rocm_lib_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(value) = env::var(var) {
            let root = PathBuf::from(value);
            candidates.extend([root.join("lib"), root.join("lib64")]);
        }
    }
    candidates.extend([
        PathBuf::from("/opt/rocm/lib"),
        PathBuf::from("/opt/rocm/lib64"),
        PathBuf::from("/opt/rocm-7.0.0/lib"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib"),
    ]);
    candidates.into_iter().find(|dir| has_libamdhip64(dir))
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

#[derive(Clone, Copy)]
struct KernelBridge {
    group: &'static str,
    src_name: &'static str,
    obj_name: &'static str,
    context: &'static str,
}

// Group ids are an audit scaffold for the future build split. The default build
// still compiles every bridge listed for the selected backend.
const HIP_BRIDGES: &[KernelBridge] = &[
    KernelBridge {
        group: "hip-qwen35",
        src_name: "full_attention_bridge.cpp",
        obj_name: "qwen35_megakernel_hip.o",
        context: "building qwen35 megakernel HIP bridge",
    },
    KernelBridge {
        group: "hip-qwen35",
        src_name: "full_attention_bridge_4b.cpp",
        obj_name: "qwen35_4b_megakernel_hip.o",
        context: "building qwen35-4b megakernel HIP bridge",
    },
    KernelBridge {
        group: "hip-qwen35",
        src_name: "prefill_helpers_bridge.cpp",
        obj_name: "qwen35_prefill_helpers_hip.o",
        context: "building prefill helpers HIP bridge",
    },
    KernelBridge {
        group: "hip-dflash",
        src_name: "dflash_draft_bridge.cpp",
        obj_name: "dflash_draft_hip.o",
        context: "building DFlash draft HIP bridge",
    },
    KernelBridge {
        group: "hip-qwen36-moe",
        src_name: "qwen36_moe_bridge.cpp",
        obj_name: "qwen36_moe_hip.o",
        context: "building Qwen3.6-MoE HIP bridge",
    },
    KernelBridge {
        group: "hip-gqh",
        src_name: "gqh_bridge.cpp",
        obj_name: "gqh_hip.o",
        context: "building GQH decode/matvec HIP bridge",
    },
];

const CUDA_BRIDGES: &[KernelBridge] = &[];

const KERNEL_RERUN_PATHS: &[&str] = &[
    "full_attention.hip",
    "full_attention_4b.hip",
    "prefill_helpers.hip",
    "full_attention_bridge.cpp",
    "full_attention_bridge_4b.cpp",
    "prefill_helpers_bridge.cpp",
    "gemma4.hip",
    "gemma4_bridge.cpp",
    "phi4.hip",
    "phi4_bridge.cpp",
    "dflash_draft.hip",
    "dflash_draft_bridge.cpp",
    "qwen36_moe.hip",
    "qwen36_moe_cuda_prelude.cuh",
    "qwen36_moe_bridge.cpp",
    "qwen36_moe_bridge_cuda.cu",
    "qwen3_moe.hip",
    "qwen3_moe_bridge.cpp",
    "qwen36_moe_persistent/helpers.cuh",
    "qwen36_moe_persistent/full_attn_phase.cuh",
    "qwen36_moe_persistent/linear_attn_phase.cuh",
    "qwen36_moe_persistent/ffn_phase.cuh",
    "qwen36_moe_persistent/lm_head_phase.cuh",
    "qwen36_moe_persistent/persistent_decode.hip",
    "qwen36_moe_persistent/batched_prefill_attn_full.cuh",
    "qwen36_moe_persistent/batched_prefill_grouped_expert.cuh",
    "qwen36_moe_persistent/batched_prefill_router_permute.cuh",
    "qwen36_moe_persistent/batched_prefill_unpermute_combine.cuh",
    "gqh.hip",
    "gqh_bridge.cpp",
    "gqh-tables.h",
    "full_attention_cuda.cuh",
    "full_attention_4b_cuda.cuh",
    "prefill_helpers_cuda.cuh",
    "full_attention_bridge_cuda.cu",
    "full_attention_bridge_4b_cuda.cu",
    "prefill_helpers_bridge_cuda.cu",
    "certified_kv_bridge_cuda.cu",
    "phi4_cuda.cuh",
    "phi4_bridge_cuda.cu",
    "gemma4_cuda.cuh",
    "gemma4_bridge_cuda.cu",
];

fn bridge_group_list(sources: &[KernelBridge]) -> String {
    let groups: BTreeSet<&str> = sources.iter().map(|source| source.group).collect();
    groups.into_iter().collect::<Vec<_>>().join(", ")
}

fn compile_hip(kernel_dir: &Path, out_dir: &Path) {
    let archs = detect_hip_archs();
    if archs.is_empty() {
        println!("cargo:warning=no HIP arch detected (set HIP_ARCH or install rocminfo); kernel binary may not run on the target GPU");
    } else if verbose_build_warnings() {
        println!(
            "cargo:warning=building HIP kernels for arch(es): {}",
            archs.join(", ")
        );
    }
    if verbose_build_warnings() {
        println!(
            "cargo:warning=building HIP bridge groups: {}",
            bridge_group_list(HIP_BRIDGES)
        );
    }

    let mut objects = Vec::new();
    for source in HIP_BRIDGES {
        let mut cmd = Command::new("hipcc");
        let obj_path = out_dir.join(source.obj_name);
        cmd.arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-I")
            .arg(kernel_dir)
            .arg("-x")
            .arg("hip")
            .arg("-c")
            .arg(kernel_dir.join(source.src_name))
            .arg("-o")
            .arg(&obj_path);
        for arch in &archs {
            cmd.arg(format!("--offload-arch={arch}"));
        }
        run(&mut cmd, source.context);
        objects.push(obj_path);
    }

    archive(
        out_dir,
        "qwen35_megakernel_hip",
        &objects,
        "archiving qwen35 megakernel HIP bridges",
    );
    if let Some(rocm_lib_dir) = detect_rocm_lib_dir() {
        println!("cargo:rustc-link-search=native={}", rocm_lib_dir.display());
    } else {
        println!(
            "cargo:warning=could not locate libamdhip64 under ROCM_PATH/HIP_PATH or common system library paths; falling back to linker search path"
        );
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_hip");
}

fn compile_cuda(kernel_dir: &Path, out_dir: &Path) {
    let archs = detect_cuda_archs();
    if verbose_build_warnings() {
        println!(
            "cargo:warning=building CUDA kernels for arch(es): {}",
            archs.join(", ")
        );
        println!(
            "cargo:warning=building CUDA bridge groups: {}",
            bridge_group_list(CUDA_BRIDGES)
        );
    }

    let mut objects = Vec::new();
    for source in CUDA_BRIDGES {
        let mut cmd = Command::new("nvcc");
        let obj_path = out_dir.join(source.obj_name);
        cmd.arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-I")
            .arg(kernel_dir)
            .arg("-c")
            .arg(kernel_dir.join(source.src_name))
            .arg("-o")
            .arg(&obj_path);
        for arch in &archs {
            cmd.arg(format!(
                "-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]"
            ));
        }
        run(&mut cmd, source.context);
        objects.push(obj_path);
    }

    archive(
        out_dir,
        "qwen35_megakernel_cuda",
        &objects,
        "archiving qwen35 megakernel CUDA bridges",
    );
    if let Some(cuda_lib_dir) = detect_cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    } else {
        println!(
            "cargo:warning=could not locate libcudart under CUDA_HOME/CUDA_PATH or common system library paths; falling back to linker search path"
        );
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_cuda");
}

fn compile_metal_stubs(manifest_dir: &Path) {
    if verbose_build_warnings() {
        println!("cargo:warning=building Metal bridge groups: metal-host-stubs");
    }
    cc::Build::new()
        .cpp(true)
        .file(manifest_dir.join("src/metal_link_stubs.cc"))
        .flag_if_supported("-std=c++17")
        .compile("kernel_ffi_metal_stubs");
    let have_mtl4_mpp = have_mtl4_mpp_sdk();
    let mut metal = cc::Build::new();
    metal
        .cpp(true)
        .file(manifest_dir.join("src/metal_native.mm"))
        .flag_if_supported("-std=c++17")
        .flag("-fobjc-arc");
    if have_mtl4_mpp {
        metal.define("SUPERSONIC_HAVE_MTL4_MPP", "1");
    } else {
        println!(
            "cargo:warning=Metal 4 MPP tensor headers not found; M5 MPP attribution pilot disabled"
        );
    }
    metal.compile("kernel_ffi_metal_native");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    if have_mtl4_mpp {
        println!("cargo:rustc-link-lib=framework=MetalPerformancePrimitives");
    }
    println!("cargo:rustc-cfg=supersonic_backend_metal");
}

fn compile_metal_ffi_contract(manifest_dir: &Path) {
    cc::Build::new()
        .cpp(true)
        .include(manifest_dir.join("src"))
        .file(manifest_dir.join("src/metal_native_ffi_contract.cc"))
        .flag_if_supported("-std=c++17")
        .compile("kernel_ffi_metal_ffi_contract");
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKENDS");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_cuda)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("cannot find workspace root")
        .join("kernels");
    for path in KERNEL_RERUN_PATHS {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(path).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/metal_link_stubs.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/metal_native.mm").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/metal_native_ffi.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("src/metal_native_ffi_contract.cc")
            .display()
    );

    let requested = env::var("SUPERSONIC_BACKENDS").unwrap_or_else(|_| "hip".to_string());
    let normalized = requested.trim().to_ascii_lowercase();
    if normalized.split(',').any(|part| {
        let part = part.trim();
        part == "cuda" || part == "metal"
    }) {
        panic!(
            "SUPERSONIC_BACKENDS={requested} is disabled on the slim HIP/Qwen branch. \
             Build with SUPERSONIC_BACKENDS=hip (the default)."
        );
    }
    let want_hip = normalized == "auto"
        || normalized == "hip"
        || normalized.split(',').any(|part| part.trim() == "hip");
    let want_cuda = false;
    let want_metal = false;
    let have_hip_toolchain = want_hip && command_exists("hipcc");
    let have_cuda_toolchain = false;
    let have_metal_backend = false;

    assert!(
        have_hip_toolchain,
        "No HIP toolchain found for SUPERSONIC_BACKENDS={requested}. Install hipcc."
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    if want_hip && want_cuda && have_hip_toolchain && have_cuda_toolchain {
        panic!(
            "SUPERSONIC_BACKENDS={requested} is not supported by kernel-ffi yet: \
             HIP and CUDA bridge archives export the same symbol set. \
             Choose one backend, or build on a machine with only one toolchain available."
        );
    }

    compile_metal_ffi_contract(&manifest_dir);

    if normalized == "hip" {
        compile_hip(&kernel_dir, &out_dir);
    } else if normalized == "cuda" {
        compile_cuda(&kernel_dir, &out_dir);
    } else if normalized == "metal" {
        compile_metal_stubs(&manifest_dir);
    } else if have_cuda_toolchain {
        compile_cuda(&kernel_dir, &out_dir);
    } else if have_hip_toolchain {
        compile_hip(&kernel_dir, &out_dir);
    } else {
        compile_metal_stubs(&manifest_dir);
    }
}
