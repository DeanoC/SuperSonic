use crate::schema::FingerprintComponents;
use blake3::Hasher;

pub fn compute(components: &FingerprintComponents) -> String {
    let mut h = Hasher::new();
    h.update(components.cpu.as_bytes());
    h.update(b"|");
    for g in &components.gpus {
        h.update(g.as_bytes());
        h.update(b",");
    }
    h.update(b"|");
    h.update(components.driver.as_bytes());
    h.update(b"|");
    for f in &components.isa {
        h.update(f.as_bytes());
        h.update(b",");
    }
    format!("blake3:{}", h.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FingerprintComponents;

    fn comp() -> FingerprintComponents {
        FingerprintComponents {
            cpu: "AMD Ryzen 9 7950X stepping=2 microcode=0xa601206".into(),
            gpus: vec!["HIP:gfx1100:0x744c:GPU-uuid".into()],
            driver: "amdgpu 6.10".into(),
            isa: vec!["AVX2".into(), "AVX-512F".into()],
        }
    }

    #[test]
    fn fingerprint_is_stable_for_same_inputs() {
        let a = compute(&comp());
        let b = compute(&comp());
        assert_eq!(a, b);
        assert!(a.starts_with("blake3:"));
        assert_eq!(a.len(), "blake3:".len() + 64);
    }

    #[test]
    fn fingerprint_changes_when_driver_changes() {
        let a = compute(&comp());
        let mut c = comp();
        c.driver = "amdgpu 6.11".into();
        let b = compute(&c);
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_changes_when_gpu_changes() {
        let a = compute(&comp());
        let mut c = comp();
        c.gpus = vec!["HIP:gfx1150:0x150c:GPU-uuid".into()];
        let b = compute(&c);
        assert_ne!(a, b);
    }
}
