use std::fs;

#[derive(Debug, Clone, Default)]
pub struct CpuId {
    pub vendor: String,
    pub model: String,
    pub stepping: u32,
    pub microcode: Option<String>,
    pub isa: Vec<String>,
}

pub fn detect_cpu_id() -> CpuId {
    let mut id = CpuId::default();
    if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
        parse_proc_cpuinfo(&text, &mut id);
    }
    #[cfg(target_os = "macos")]
    fill_macos_cpu_id(&mut id);
    fill_isa_from_runtime(&mut id);
    id
}

#[cfg(target_os = "macos")]
fn fill_macos_cpu_id(id: &mut CpuId) {
    if id.model.is_empty() {
        id.model = sysctl_string("machdep.cpu.brand_string")
            .or_else(|| sysctl_string("machdep.cpu.brand"))
            .or_else(|| system_profiler_field("chip_type"))
            .unwrap_or_else(|| "Apple Silicon".into());
    }
    if id.vendor.is_empty() {
        id.vendor = "Apple".into();
    }
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(target_os = "macos")]
fn system_profiler_field(field: &str) -> Option<String> {
    let output = std::process::Command::new("/usr/sbin/system_profiler")
        .args(["SPHardwareDataType", "-json"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let root: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    root.get("SPHardwareDataType")?
        .as_array()?
        .first()?
        .get(field)?
        .as_str()
        .map(str::to_string)
}

fn parse_proc_cpuinfo(text: &str, id: &mut CpuId) {
    for line in text.lines() {
        let (key, value) = match line.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        match key {
            "vendor_id" if id.vendor.is_empty() => id.vendor = value.to_string(),
            "model name" if id.model.is_empty() => id.model = value.to_string(),
            "stepping" if id.stepping == 0 => {
                if let Ok(s) = value.parse() {
                    id.stepping = s;
                }
            }
            "microcode" if id.microcode.is_none() => {
                id.microcode = Some(value.to_string());
            }
            "flags" if id.isa.is_empty() => {
                id.isa = value.split_whitespace().map(str::to_string).collect();
            }
            _ => {}
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn fill_isa_from_runtime(id: &mut CpuId) {
    let mut add = |s: &str| {
        if !id.isa.iter().any(|x| x == s) {
            id.isa.push(s.to_string());
        }
    };
    if std::is_x86_feature_detected!("avx2") {
        add("AVX2");
    }
    if std::is_x86_feature_detected!("avx512f") {
        add("AVX-512F");
    }
    if std::is_x86_feature_detected!("avx512bf16") {
        add("AVX-512BF16");
    }
    if std::is_x86_feature_detected!("avxvnni") {
        add("AVX-VNNI");
    }
    #[cfg(feature = "x86_amx_intrinsics")]
    if std::is_x86_feature_detected!("amx-bf16") {
        add("AMX-BF16");
    }
    if std::is_x86_feature_detected!("fma") {
        add("FMA");
    }
}

#[cfg(target_arch = "aarch64")]
fn fill_isa_from_runtime(id: &mut CpuId) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        id.isa.push("NEON".into());
    }
    if std::arch::is_aarch64_feature_detected!("sve") {
        id.isa.push("SVE".into());
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn fill_isa_from_runtime(_: &mut CpuId) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_cpuinfo_parser_extracts_vendor_model_stepping() {
        let sample = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\nstepping\t: 2\nmicrocode\t: 0xa601206\nflags\t\t: fpu vme de avx2 avx512f bmi2\n";
        let mut id = CpuId::default();
        parse_proc_cpuinfo(sample, &mut id);
        assert_eq!(id.vendor, "AuthenticAMD");
        assert!(id.model.contains("7950X"));
        assert_eq!(id.stepping, 2);
        assert_eq!(id.microcode.as_deref(), Some("0xa601206"));
        assert!(id.isa.iter().any(|f| f == "avx2"));
    }
}
