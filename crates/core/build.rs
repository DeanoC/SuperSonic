fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SUPERSONIC_BACKEND");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_hip)");
    println!("cargo:rustc-check-cfg=cfg(supersonic_backend_metal)");
    match std::env::var("SUPERSONIC_BACKEND").ok().as_deref() {
        Some("metal") => println!("cargo:rustc-cfg=supersonic_backend_metal"),
        Some("hip") | None => println!("cargo:rustc-cfg=supersonic_backend_hip"),
        Some(other) => panic!("unsupported SUPERSONIC_BACKEND={other}; expected hip or metal"),
    }
}
