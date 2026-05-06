use std::env;
use std::path::PathBuf;
use std::process::Command;

fn detect_hip_archs() -> Vec<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        return arch.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    }
    let Ok(output) = Command::new("rocminfo").output() else { return Vec::new() };
    if !output.status.success() { return Vec::new(); }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|t| t.starts_with("gfx"))
        .map(|s| vec![s.to_owned()])
        .unwrap_or_default()
}

fn have_hipcc() -> bool {
    Command::new("sh").arg("-lc").arg("command -v hipcc >/dev/null 2>&1")
        .status().map(|s| s.success()).unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_ARCH");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");

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
        ("profile_bridge.cpp", "profile_bridge.o"),
    ];
    let mut objects = Vec::new();
    for (src, obj) in sources {
        println!("cargo:rerun-if-changed={}", kernels.join(src).display());
        let obj_path = out_dir.join(obj);
        let mut cmd = Command::new("hipcc");
        cmd.args(["-std=c++17", "-O3", "-fPIC", "-x", "hip", "-c"])
            .arg(kernels.join(src))
            .args(["-I"]).arg(&kernels)
            .arg("-o").arg(&obj_path);
        for a in &archs { cmd.arg(format!("--offload-arch={a}")); }
        let status = cmd.status().expect("hipcc failed to start");
        assert!(status.success(), "hipcc failed for {src}");
        objects.push(obj_path);
    }
    let lib = out_dir.join("libmp_profile_hip.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objects { ar.arg(o); }
    ar.status().expect("ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mp_profile_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=supersonic_backend_hip");
}
