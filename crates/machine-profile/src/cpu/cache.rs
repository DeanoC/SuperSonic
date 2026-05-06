use crate::schema::{CacheHierarchy, CacheLevel};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

pub fn detect() -> CacheHierarchy {
    let sys = Path::new("/sys/devices/system/cpu/cpu0/cache");
    let mut h = CacheHierarchy { l1d: None, l2: None, l3: None };
    if let Ok(entries) = fs::read_dir(sys) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("index") { continue; }
            let level: Option<u32> = read_str(&entry.path().join("level"))
                .and_then(|s| s.parse().ok());
            let kind = read_str(&entry.path().join("type")).unwrap_or_default();
            let size = read_size(&entry.path().join("size")).unwrap_or(0);
            let line = read_str(&entry.path().join("coherency_line_size"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(64);
            let ways = read_str(&entry.path().join("ways_of_associativity"))
                .and_then(|s| s.parse().ok());
            let cl = CacheLevel {
                size_bytes: size,
                line_bytes: line,
                ways,
                measured_lat_ns: None,
                measured_bw_gb_s: None,
            };
            match (level, kind.as_str()) {
                (Some(1), "Data") | (Some(1), "Unified") => h.l1d = Some(cl),
                (Some(2), _) => h.l2 = Some(cl),
                (Some(3), _) => h.l3 = Some(cl),
                _ => {}
            }
        }
    }
    populate_measurements(&mut h);
    h
}

fn populate_measurements(h: &mut CacheHierarchy) {
    if let Some(l) = h.l1d.as_mut() {
        let s = l.size_bytes.max(1) as usize;
        let lat = pointer_chase_ns(s / 2);
        l.measured_lat_ns = Some(lat);
        l.measured_bw_gb_s = Some(read_bandwidth_gb_s(s / 2));
    }
    if let Some(l) = h.l2.as_mut() {
        let s = l.size_bytes.max(1) as usize;
        l.measured_lat_ns = Some(pointer_chase_ns(s / 2));
        l.measured_bw_gb_s = Some(read_bandwidth_gb_s(s / 2));
    }
    if let Some(l) = h.l3.as_mut() {
        let s = l.size_bytes.max(1) as usize;
        l.measured_lat_ns = Some(pointer_chase_ns(s / 2));
        l.measured_bw_gb_s = Some(read_bandwidth_gb_s(s / 2));
    }
}

fn pointer_chase_ns(bytes: usize) -> f64 {
    let n = (bytes / 8).max(64);
    let mut buf: Vec<usize> = (0..n).collect();
    let mut idx = 0usize;
    for i in (1..n).rev() {
        let j = ((i.wrapping_mul(2654435761)) % (i + 1)) as usize;
        buf.swap(i, j);
        idx = idx.wrapping_add(1);
    }
    let mut chain: Vec<usize> = vec![0; n];
    let mut prev = 0;
    for i in 0..n {
        chain[prev] = buf[i];
        prev = buf[i];
    }
    let iters = 10_000_000usize;
    let start = Instant::now();
    let mut p = 0usize;
    for _ in 0..iters { p = chain[p]; }
    let elapsed = start.elapsed();
    black_box(p);
    elapsed.as_nanos() as f64 / iters as f64
}

fn read_bandwidth_gb_s(bytes: usize) -> f64 {
    let n = (bytes / 8).max(1024);
    let buf: Vec<u64> = vec![1; n];
    let iters = 32usize;
    let start = Instant::now();
    let mut acc: u64 = 0;
    for _ in 0..iters {
        for &v in &buf { acc = acc.wrapping_add(v); }
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(acc);
    let total_bytes = (n * std::mem::size_of::<u64>() * iters) as f64;
    total_bytes / elapsed / 1e9
}

fn read_str(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn read_size(path: &Path) -> Option<u64> {
    let s = read_str(path)?;
    let (num, mult) = if let Some(stripped) = s.strip_suffix('K') {
        (stripped, 1024u64)
    } else if let Some(stripped) = s.strip_suffix('M') {
        (stripped, 1024 * 1024)
    } else {
        (s.as_str(), 1)
    };
    num.parse::<u64>().ok().map(|n| n * mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_size_with_kilobyte_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("size");
        std::fs::write(&path, "32K").unwrap();
        assert_eq!(read_size(&path), Some(32 * 1024));
    }

    #[test]
    fn pointer_chase_returns_positive_latency() {
        let lat = pointer_chase_ns(64 * 1024);
        assert!(lat > 0.0 && lat < 1000.0);
    }
}
