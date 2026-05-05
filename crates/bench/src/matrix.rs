//! Bench-side mirror of the runner's (model, quant, arch) support matrix.
//!
//! INVARIANT: this table must match `crates/runner/src/registry.rs`'s REGISTRY
//! plus the per-family quant flags. A parity test in
//! `crates/runner/tests/bench_combo_parity.rs` enforces this — if you change
//! the runner registry, you MUST update this table or the test fails.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchArch { Gfx1100, Gfx1150, Sm86, AppleM4 }

impl BenchArch {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "gfx1100" => Self::Gfx1100,
            "gfx1150" => Self::Gfx1150,
            "sm86" => Self::Sm86,
            "apple-m4" => Self::AppleM4,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1150 => "gfx1150",
            Self::Sm86 => "sm86",
            Self::AppleM4 => "apple-m4",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComboDescriptor {
    pub model: &'static str,        // e.g. "qwen3.5-0.8b"
    pub quant: &'static str,        // "bf16" | "int4" | "fp8r" | "kv-fp8" | "int8"
    pub arch: BenchArch,
    pub min_vram_gib: f64,
}

/// Mirrors docs/feature-compatibility.md + docs/performance.md as of 2026-05-05.
pub static SUPPORTED_COMBOS: &[ComboDescriptor] = &[
    // Qwen3.5 — full BF16/INT4/FP8r/KV-FP8 quad on gfx1100.
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 2.0 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 0.7 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 1.2 },
    ComboDescriptor { model: "qwen3.5-0.8b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 2.0 },
    ComboDescriptor { model: "qwen3.5-2b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 5.0 },
    ComboDescriptor { model: "qwen3.5-2b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 1.9 },
    ComboDescriptor { model: "qwen3.5-2b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 3.0 },
    ComboDescriptor { model: "qwen3.5-2b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 5.0 },
    ComboDescriptor { model: "qwen3.5-4b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "qwen3.5-4b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 3.7 },
    ComboDescriptor { model: "qwen3.5-4b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 6.0 },
    ComboDescriptor { model: "qwen3.5-4b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "qwen3.5-9b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 18.0 },
    ComboDescriptor { model: "qwen3.5-9b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 6.7 },
    ComboDescriptor { model: "qwen3.5-9b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 10.8 },
    ComboDescriptor { model: "qwen3.5-9b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 18.0 },
    // Gemma 4 — fp8r and kv-fp8 are wired into the single-batch persistent
    // decode kernel only (require --batch-size=1, cannot combine with --int4).
    // See docs/supported-matrix.md footnote 2.
    ComboDescriptor { model: "gemma4-e2b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 11.0 },
    ComboDescriptor { model: "gemma4-e2b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 4.1 },
    ComboDescriptor { model: "gemma4-e2b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 6.6 },
    ComboDescriptor { model: "gemma4-e2b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 11.0 },
    ComboDescriptor { model: "gemma4-e4b", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    ComboDescriptor { model: "gemma4-e4b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 3.7 },
    ComboDescriptor { model: "gemma4-e4b", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 6.0 },
    ComboDescriptor { model: "gemma4-e4b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 10.0 },
    // Phi-4-mini — full quad
    ComboDescriptor { model: "phi4-mini", quant: "bf16", arch: BenchArch::Gfx1100, min_vram_gib: 8.0 },
    ComboDescriptor { model: "phi4-mini", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 3.0 },
    ComboDescriptor { model: "phi4-mini", quant: "fp8r", arch: BenchArch::Gfx1100, min_vram_gib: 4.8 },
    ComboDescriptor { model: "phi4-mini", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 8.0 },
    // Qwen3.6-MoE — INT4 + KV-FP8 only on gfx1100 (24 GiB cap).
    // KV-FP8 lane requires --int4 simultaneously (only quant lane shipped).
    // See docs/feature-compatibility.md footnote 4.
    ComboDescriptor { model: "qwen3.6-35b-a3b", quant: "int4", arch: BenchArch::Gfx1100, min_vram_gib: 21.0 },
    ComboDescriptor { model: "qwen3.6-35b-a3b", quant: "kv-fp8", arch: BenchArch::Gfx1100, min_vram_gib: 21.0 },
];

pub fn combos_for_arch(arch: BenchArch) -> Vec<&'static ComboDescriptor> {
    SUPPORTED_COMBOS.iter().filter(|c| c.arch == arch).collect()
}
