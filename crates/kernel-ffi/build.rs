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

#[derive(Clone, Debug)]
struct KernelBridge {
    src_name: String,
    obj_name: String,
}

fn manifest_value(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_owned())
}

fn kernel_relative_path(path: String) -> String {
    path.strip_prefix("kernels/").unwrap_or(&path).to_owned()
}

/// Read the build inputs from the same manifest used by the source validator.
/// The manifest is deliberately parsed here without another build dependency:
/// it contains only the small string fields needed by this build script.
fn read_kernel_manifest(path: &Path) -> (Vec<KernelBridge>, Vec<String>) {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read kernel manifest {}: {error}", path.display()));
    let mut bridges = Vec::new();
    let mut rerun_paths = Vec::new();
    let mut array = None::<&str>;
    let mut pending_source = None::<String>;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with("kernel_sources") {
            array = Some("kernel_sources");
            continue;
        }
        if array.is_some() {
            if line == "]" {
                array = None;
            } else if let Some(value) = manifest_value(line) {
                rerun_paths.push(kernel_relative_path(value));
            }
            continue;
        }
        if line.starts_with("source") {
            pending_source = manifest_value(line).map(kernel_relative_path);
            continue;
        }
        if line.starts_with("object") {
            let source = pending_source
                .take()
                .unwrap_or_else(|| panic!("kernel manifest object without source: {line}"));
            let object = manifest_value(line).expect("kernel manifest object value");
            rerun_paths.push(source.clone());
            bridges.push(KernelBridge {
                src_name: source,
                obj_name: object,
            });
        }
    }

    assert!(
        !bridges.is_empty(),
        "kernel manifest defines no HIP bridges"
    );
    assert!(
        !rerun_paths.is_empty(),
        "kernel manifest defines no HIP sources"
    );
    rerun_paths.sort();
    rerun_paths.dedup();
    (bridges, rerun_paths)
}

fn compile_hip(kernel_dir: &Path, out_dir: &Path, bridges: &[KernelBridge]) {
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
    let failure_injection = env::var_os("SUPERSONIC_GPU_FAILURE_TESTS").is_some();
    let mut objects = Vec::with_capacity(bridges.len());
    for bridge in bridges {
        let object = out_dir.join(&bridge.obj_name);
        let mut command = Command::new("hipcc");
        command
            .args(["-std=c++17", "-O3", "-fPIC", "-I"])
            .arg(kernel_dir)
            .args(["-x", "hip", "-c"])
            .arg(kernel_dir.join(&bridge.src_name))
            .args(["-o"])
            .arg(&object);
        if allow_gqh_fma {
            command.arg("-DGQH_ALLOW_FMA");
        }
        if failure_injection {
            command.arg("-DSUPERSONIC_FAILURE_INJECTION");
        }
        for arch in &archs {
            command.arg(format!("--offload-arch={arch}"));
        }
        let context = format!("building HIP bridge {}", bridge.src_name);
        run(&mut command, &context);
        objects.push(object);
    }

    // Keep the historical archive name: downstream linkers may refer to this
    // retained HIP bridge artifact by its pre-rename ABI identity.
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
    println!("cargo:rerun-if-env-changed=SUPERSONIC_GPU_FAILURE_TESTS");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_failure_injection)");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|parent| parent.parent())
        .expect("cannot find workspace root")
        .join("kernels");
    let manifest_path = manifest_dir.join("kernel-groups.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let (bridges, rerun_paths) = read_kernel_manifest(&manifest_path);
    for path in rerun_paths {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(path).display());
    }

    assert!(
        command_exists("hipcc"),
        "No HIP toolchain found; install hipcc."
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    compile_hip(&kernel_dir, &out_dir, &bridges);
    if env::var_os("SUPERSONIC_GPU_FAILURE_TESTS").is_some() {
        println!("cargo:rustc-cfg=supersonic_failure_injection");
    }
}
