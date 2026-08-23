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
        .map(|status| status.success())
        .unwrap_or(false)
}

fn verbose_build_warnings() -> bool {
    env::var_os("SUPERSONIC_BUILD_VERBOSE").is_some()
}

fn detect_hip_archs() -> Vec<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        let archs: Vec<_> = arch
            .split(',')
            .map(str::trim)
            .filter(|arch| !arch.is_empty())
            .map(str::to_owned)
            .collect();
        if !archs.is_empty() {
            return archs;
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
            let token = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
            let suffix = token.strip_prefix("gfx")?;
            if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_hexdigit()) {
                return None;
            }
            Some(token.to_owned())
        })
        .collect();
    archs.into_iter().collect()
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

fn run(command: &mut Command, context: &str) {
    let status = command.status().unwrap_or_else(|error| {
        panic!("{context}: failed to start command {:?}: {error}", command)
    });
    assert!(
        status.success(),
        "{context}: command {:?} failed with {status}",
        command
    );
}

fn archive(out_dir: &Path, lib_name: &str, objects: &[PathBuf], context: &str) {
    let lib_path = out_dir.join(format!("lib{lib_name}.a"));
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib_path);
    for object in objects {
        ar.arg(object);
    }
    run(&mut ar, context);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
}

#[derive(Clone, Copy)]
struct KernelBridge {
    src_name: &'static str,
    obj_name: &'static str,
    context: &'static str,
}

const HIP_GROUPS: &[&str] = &["hip-qwen38-dense", "hip-gqh"];

// The product build has one HIP bridge archive. The object names retain the
// historical ABI library identity; exported kernel symbols are unchanged.
const HIP_BRIDGES: &[KernelBridge] = &[
    KernelBridge {
        src_name: "full_attention_bridge.cpp",
        obj_name: "qwen35_megakernel_hip.o",
        context: "building dense attention HIP bridge",
    },
    KernelBridge {
        src_name: "full_attention_bridge_4b.cpp",
        obj_name: "qwen35_4b_megakernel_hip.o",
        context: "building dense 4B HIP bridge",
    },
    KernelBridge {
        src_name: "prefill_helpers_bridge.cpp",
        obj_name: "qwen35_prefill_helpers_hip.o",
        context: "building prefill helpers HIP bridge",
    },
    KernelBridge {
        src_name: "gqh_bridge.cpp",
        obj_name: "gqh_hip.o",
        context: "building GQH HIP bridge",
    },
];

const KERNEL_RERUN_PATHS: &[&str] = &[
    "full_attention.hip",
    "full_attention_4b.hip",
    "prefill_helpers.hip",
    "full_attention_bridge.cpp",
    "full_attention_bridge_4b.cpp",
    "prefill_helpers_bridge.cpp",
    "gqh.hip",
    "gqh_bridge.cpp",
    "gqh-tables.h",
    "gqh-stride.h",
];

fn compile_hip(kernel_dir: &Path, out_dir: &Path) {
    let archs = detect_hip_archs();
    if archs.is_empty() {
        println!(
            "cargo:warning=no HIP arch detected (set HIP_ARCH or install rocminfo); kernel binary may not run on the target GPU"
        );
    } else if verbose_build_warnings() {
        println!(
            "cargo:warning=building HIP kernels for arch(es): {}",
            archs.join(", ")
        );
    }

    println!("cargo:rerun-if-env-changed=GQH_ALLOW_FMA");
    let allow_gqh_fma = env::var_os("GQH_ALLOW_FMA").is_some_and(|value| value != "0");
    let mut objects = Vec::with_capacity(HIP_BRIDGES.len());
    for bridge in HIP_BRIDGES {
        let object = out_dir.join(bridge.obj_name);
        let mut command = Command::new("hipcc");
        command
            .args(["-std=c++17", "-O3", "-fPIC", "-I"])
            .arg(kernel_dir)
            .args(["-x", "hip", "-c"])
            .arg(kernel_dir.join(bridge.src_name))
            .args(["-o"])
            .arg(&object);
        if allow_gqh_fma {
            command.arg("-DGQH_ALLOW_FMA");
        }
        for arch in &archs {
            command.arg(format!("--offload-arch={arch}"));
        }
        run(&mut command, bridge.context);
        objects.push(object);
    }

    archive(
        out_dir,
        "qwen35_megakernel_hip",
        &objects,
        "archiving HIP bridges",
    );
    if let Some(rocm_lib_dir) = detect_rocm_lib_dir() {
        println!("cargo:rustc-link-search=native={}", rocm_lib_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_hip");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|parent| parent.parent())
        .expect("cannot find workspace root")
        .join("kernels");
    for path in KERNEL_RERUN_PATHS {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(path).display());
    }

    assert!(command_exists("hipcc"), "No HIP toolchain found; install hipcc.");
    let _ = HIP_GROUPS;
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    compile_hip(&kernel_dir, &out_dir);
}
