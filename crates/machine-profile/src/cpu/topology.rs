use crate::schema::{CpuTopology, NumaNode};
use std::fs;
use std::path::Path;

pub fn detect() -> CpuTopology {
    #[cfg(target_os = "macos")]
    if let Some(t) = detect_macos() {
        return t;
    }
    detect_from(Path::new("/sys/devices/system"))
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Option<CpuTopology> {
    let cores_total = sysctl_u32("hw.physicalcpu").or_else(|| sysctl_u32("hw.ncpu"))?;
    let (p, e) = macos_core_split().unwrap_or((cores_total, 0));
    Some(CpuTopology {
        sockets: 1,
        cores_total,
        cores_p: p,
        cores_e: e,
        threads_per_core: 1,
        numa_nodes: vec![NumaNode {
            id: 0,
            cpus: (0..cores_total).collect(),
            ram_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
        }],
    })
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Option<u32> {
    sysctl_string(name)?.parse().ok()
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    sysctl_string(name)?.parse().ok()
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
fn macos_core_split() -> Option<(u32, u32)> {
    let output = std::process::Command::new("/usr/sbin/system_profiler")
        .args(["SPHardwareDataType", "-json"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let root: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let number = root
        .get("SPHardwareDataType")?
        .as_array()?
        .first()?
        .get("number_processors")?
        .as_str()?;
    // Example: "proc 18:6:0:12" => total 18, performance 6, efficiency 12.
    let fields: Vec<u32> = number
        .split_whitespace()
        .last()?
        .split(':')
        .filter_map(|s| s.parse().ok())
        .collect();
    if fields.len() >= 4 {
        Some((fields[1], fields[3]))
    } else {
        None
    }
}

pub fn detect_from(sys_root: &Path) -> CpuTopology {
    let cpu_root = sys_root.join("cpu");
    let mut online_cpus = read_cpu_list(&cpu_root.join("online")).unwrap_or_default();
    online_cpus.sort_unstable();

    let mut socket_set = std::collections::BTreeSet::<u32>::new();
    let mut core_set = std::collections::BTreeSet::<(u32, u32)>::new(); // (socket, core_id)
    let mut threads_per_core_count = std::collections::HashMap::<(u32, u32), u32>::new();

    for &cpu in &online_cpus {
        let socket =
            read_u32(&cpu_root.join(format!("cpu{cpu}/topology/physical_package_id"))).unwrap_or(0);
        let core_id = read_u32(&cpu_root.join(format!("cpu{cpu}/topology/core_id"))).unwrap_or(cpu);
        socket_set.insert(socket);
        core_set.insert((socket, core_id));
        *threads_per_core_count.entry((socket, core_id)).or_insert(0) += 1;
    }

    let threads_per_core = threads_per_core_count.values().max().copied().unwrap_or(1);

    let numa_nodes = detect_numa(&sys_root.join("node"));

    CpuTopology {
        sockets: socket_set.len() as u32,
        cores_total: core_set.len() as u32,
        cores_p: core_set.len() as u32,
        cores_e: 0,
        threads_per_core,
        numa_nodes,
    }
}

fn detect_numa(node_root: &Path) -> Vec<NumaNode> {
    let mut nodes = Vec::new();
    let Ok(entries) = fs::read_dir(node_root) else {
        return nodes;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if !s.starts_with("node") {
            continue;
        }
        let Ok(id) = s.trim_start_matches("node").parse::<u32>() else {
            continue;
        };
        let cpus = read_cpu_list(&entry.path().join("cpulist")).unwrap_or_default();
        let ram_bytes = read_meminfo_total(&entry.path().join("meminfo")).unwrap_or(0);
        nodes.push(NumaNode {
            id,
            cpus,
            ram_bytes,
        });
    }
    nodes.sort_by_key(|n| n.id);
    nodes
}

fn read_cpu_list(path: &Path) -> Option<Vec<u32>> {
    let s = fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    for chunk in s.trim().split(',') {
        if chunk.is_empty() {
            continue;
        }
        if let Some((a, b)) = chunk.split_once('-') {
            let a: u32 = a.parse().ok()?;
            let b: u32 = b.parse().ok()?;
            for i in a..=b {
                out.push(i);
            }
        } else if let Ok(v) = chunk.parse::<u32>() {
            out.push(v);
        }
    }
    Some(out)
}

fn read_u32(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_meminfo_total(path: &Path) -> Option<u64> {
    let s = fs::read_to_string(path).ok()?;
    for line in s.lines() {
        if line.contains("MemTotal:") {
            let kib: u64 = line
                .split_whitespace()
                .find_map(|t| t.parse::<u64>().ok())?;
            return Some(kib * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn detects_topology_from_synthetic_sysfs() {
        let tmp = tempdir().unwrap();
        let sys = tmp.path().to_path_buf();
        fs::create_dir_all(sys.join("cpu")).unwrap();
        fs::write(sys.join("cpu/online"), "0-3").unwrap();
        for cpu in 0..4u32 {
            let topo = sys.join(format!("cpu/cpu{cpu}/topology"));
            fs::create_dir_all(&topo).unwrap();
            fs::write(topo.join("physical_package_id"), "0").unwrap();
            fs::write(topo.join("core_id"), format!("{}", cpu / 2)).unwrap();
        }
        let t = detect_from(&sys);
        assert_eq!(t.sockets, 1);
        assert_eq!(t.cores_total, 2);
        assert_eq!(t.threads_per_core, 2);
    }

    #[test]
    fn cpu_list_parses_ranges() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("cpulist");
        fs::write(&path, "0-3,8,10-11").unwrap();
        let v = read_cpu_list(&path).unwrap();
        assert_eq!(v, vec![0, 1, 2, 3, 8, 10, 11]);
    }
}
