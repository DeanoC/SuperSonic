use crate::schema::{MeasuredVsTheoretical, VectorPeak};
use std::hint::black_box;
use std::time::Instant;

const TARGET_DURATION_MS: u64 = 200;

pub fn measure(num_p_cores: u32) -> VectorPeak {
    VectorPeak {
        fp32: Some(measure_fp32(num_p_cores)),
        fp16: None,
        bf16: measure_bf16(num_p_cores),
    }
}

fn measure_fp32(num_p_cores: u32) -> MeasuredVsTheoretical {
    // peak_fp32_avx2 is compiled with target_feature = "avx2,fma" and uses
    // _mm256_fmadd_ps. AVX2 does not imply FMA on x86_64 — early Haswell-era
    // and a few low-power lines have one without the other, so gate on both.
    let per_core_gflops = measure_fp32_impl();
    MeasuredVsTheoretical {
        measured_per_unit: Some(per_core_gflops),
        measured_aggregate: per_core_gflops * num_p_cores as f64,
        theoretical_aggregate: None,
        ratio: None,
    }
}

#[cfg(target_arch = "x86_64")]
fn measure_fp32_impl() -> f64 {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { peak_fp32_avx2() }
    } else {
        peak_fp32_scalar()
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn measure_fp32_impl() -> f64 {
    peak_fp32_scalar()
}

// NOTE: On stable Rust 1.95.0, `_mm512_castsi512_bf16` and `_mm512_dpbf16_ps`
// are not exposed (they require `#![feature(stdarch_x86_avx512)]` on nightly).
// `measure_bf16` therefore falls back to `peak_fp32_avx2()` as a conservative
// measured estimate.  The reported BF16 GFLOPS will equal the FP32 result on
// this toolchain — catalog consumers should treat it as a lower bound until the
// AVX-512 BF16 intrinsics land on stable.
fn measure_bf16(num_p_cores: u32) -> Option<MeasuredVsTheoretical> {
    measure_bf16_impl(num_p_cores)
}

#[cfg(target_arch = "x86_64")]
fn measure_bf16_impl(num_p_cores: u32) -> Option<MeasuredVsTheoretical> {
    if std::is_x86_feature_detected!("avx512bf16") {
        let per_core_gflops = unsafe { peak_bf16_avx512() };
        Some(MeasuredVsTheoretical {
            measured_per_unit: Some(per_core_gflops),
            measured_aggregate: per_core_gflops * num_p_cores as f64,
            theoretical_aggregate: None,
            ratio: None,
        })
    } else {
        None
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn measure_bf16_impl(_num_p_cores: u32) -> Option<MeasuredVsTheoretical> {
    None
}

fn peak_fp32_scalar() -> f64 {
    let mut a = 1.0f32;
    let mut b = 1.0f32;
    let mut iters = 0u64;
    let start = Instant::now();
    while start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        for _ in 0..10_000 {
            a = a.mul_add(b, 1.0);
            b = b.mul_add(a, 1.0);
        }
        iters += 20_000;
    }
    black_box(a + b);
    iters as f64 / start.elapsed().as_secs_f64() / 1e9
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn peak_fp32_avx2() -> f64 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut a = [_mm256_set1_ps(1.0); 8];
    let mut flops_per_iter = 0u64;
    let start = Instant::now();
    while start.elapsed().as_millis() < TARGET_DURATION_MS as u128 {
        for _ in 0..10_000 {
            for k in 0..8 {
                a[k] = _mm256_fmadd_ps(a[k], a[(k + 1) % 8], one);
            }
            flops_per_iter += 8 * 8 * 2;
        }
    }
    let mut acc = a[0];
    for k in 1..8 {
        acc = _mm256_add_ps(acc, a[k]);
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    black_box(tmp);
    flops_per_iter as f64 / start.elapsed().as_secs_f64() / 1e9
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn peak_fp32_avx2() -> f64 {
    peak_fp32_scalar()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16")]
unsafe fn peak_bf16_avx512() -> f64 {
    // Stable Rust (1.95.0) doesn't expose `_mm512_castsi512_bf16` or
    // `_mm512_dpbf16_ps` yet — those require the unstable
    // `stdarch_x86_avx512` feature gate.  Fall back to the FP32 AVX2 path
    // as a conservative measured estimate.  Catalog ratios will flag this as
    // below theoretical until AVX-512 BF16 lands on stable.
    peak_fp32_avx2()
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn peak_bf16_avx512() -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_measurement_is_positive() {
        let m = measure(1);
        let fp32 = m.fp32.expect("fp32 should always be reported");
        assert!(fp32.measured_aggregate > 0.0);
    }
}
