use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Return one or more GPU archs to target. `HIP_ARCH` may be comma-separated
/// (e.g. `gfx1150,gfx1100`) to produce a multi-arch fat binary. Without the
/// env var, falls back to auto-detection via `rocminfo`.
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    // Kernel sources live at workspace root: ../../kernels/
    let kernel_dir = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("cannot find workspace root")
        .join("kernels");
    let kernel_src = kernel_dir.join("full_attention.hip");
    let bridge_src = kernel_dir.join("full_attention_bridge.cpp");
    let kernel_4b_src = kernel_dir.join("full_attention_4b.hip");
    let bridge_4b_src = kernel_dir.join("full_attention_bridge_4b.cpp");
    let prefill_helpers_src = kernel_dir.join("prefill_helpers.hip");
    let prefill_helpers_bridge_src = kernel_dir.join("prefill_helpers_bridge.cpp");
    let gemma4_src = kernel_dir.join("gemma4.hip");
    let gemma4_bridge_src = kernel_dir.join("gemma4_bridge.cpp");
    for path in [&kernel_src, &bridge_src, &kernel_4b_src, &bridge_4b_src,
                 &prefill_helpers_src, &prefill_helpers_bridge_src,
                 &gemma4_src, &gemma4_bridge_src] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let archs = detect_hip_archs();
    if archs.is_empty() {
        println!("cargo:warning=no HIP arch detected (set HIP_ARCH or install rocminfo); kernel binary may not run on the target GPU");
    } else {
        println!("cargo:warning=building HIP kernels for arch(es): {}", archs.join(", "));
    }

    // Compile 0.8B kernel
    let bridge_obj = out_dir.join("qwen35_megakernel_hip.o");
    let mut hipcc = Command::new("hipcc");
    hipcc
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg(&kernel_dir)
        .arg("-x")
        .arg("hip")
        .arg("-c")
        .arg(&bridge_src)
        .arg("-o")
        .arg(&bridge_obj);
    for arch in &archs {
        hipcc.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc, "building qwen35 megakernel HIP bridge");

    // Compile 4B kernel
    let bridge_4b_obj = out_dir.join("qwen35_4b_megakernel_hip.o");
    let mut hipcc_4b = Command::new("hipcc");
    hipcc_4b
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg(&kernel_dir)
        .arg("-x")
        .arg("hip")
        .arg("-c")
        .arg(&bridge_4b_src)
        .arg("-o")
        .arg(&bridge_4b_obj);
    for arch in &archs {
        hipcc_4b.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc_4b, "building qwen35-4b megakernel HIP bridge");

    // Compile prefill helpers (separate compilation unit)
    let prefill_helpers_obj = out_dir.join("qwen35_prefill_helpers_hip.o");
    let mut hipcc_pfx = Command::new("hipcc");
    hipcc_pfx
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg(&kernel_dir)
        .arg("-x")
        .arg("hip")
        .arg("-c")
        .arg(&prefill_helpers_bridge_src)
        .arg("-o")
        .arg(&prefill_helpers_obj);
    for arch in &archs {
        hipcc_pfx.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc_pfx, "building prefill helpers HIP bridge");

    // Compile Gemma 4 decode primitives (separate compilation unit — hipcc
    // codegen is fragile and Gemma 4 lives alongside but must not touch the
    // Qwen megakernels).
    let gemma4_obj = out_dir.join("gemma4_hip.o");
    let mut hipcc_g4 = Command::new("hipcc");
    hipcc_g4
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg(&kernel_dir)
        .arg("-x")
        .arg("hip")
        .arg("-c")
        .arg(&gemma4_bridge_src)
        .arg("-o")
        .arg(&gemma4_obj);
    for arch in &archs {
        hipcc_g4.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc_g4, "building Gemma 4 HIP bridge");

    // Archive all into a single static library
    let bridge_lib = out_dir.join("libqwen35_megakernel_hip.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&bridge_lib)
        .arg(&bridge_obj).arg(&bridge_4b_obj)
        .arg(&prefill_helpers_obj).arg(&gemma4_obj);
    run(&mut ar, "archiving qwen35 megakernel HIP bridges");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qwen35_megakernel_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
