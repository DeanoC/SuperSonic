use std::env;
use std::path::PathBuf;
use std::process::Command;

fn detect_hip_arch() -> Option<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        let trimmed = arch.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }
    let output = Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|token| token.starts_with("gfx"))
        .map(ToOwned::to_owned)
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
    for path in [&kernel_src, &bridge_src] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let bridge_obj = out_dir.join("qwen35_megakernel_hip.o");
    let bridge_lib = out_dir.join("libqwen35_megakernel_hip.a");

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
    if let Some(arch) = detect_hip_arch() {
        hipcc.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc, "building qwen35 megakernel HIP bridge");

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&bridge_lib).arg(&bridge_obj);
    run(&mut ar, "archiving qwen35 megakernel HIP bridge");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qwen35_megakernel_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
