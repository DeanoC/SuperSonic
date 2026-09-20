//! GQH int8 weight-LUT denominator selection.
//!
//! Ports geo-lucebox's `ggml_gqh_q8_denom` / `gqh_q8_fit_grid` machinery
//! (server/deps/llama.cpp/ggml/src/gqh.cpp, PR #35 `perf/gqh-wide-ncols`).
//!
//! The int8 arms bake a rung's level grid to int8 once per dispatch. The
//! obvious denominator is 127 (every grid's amax is exactly 1.0, so 127 lands
//! on the extreme level with no clamping), but 127 is only the widest choice,
//! not the most accurate one: the in-between levels land wherever the k/127
//! lattice happens to fall, and for a curved grid that is a poor fit.
//!
//! The objective is occupancy-weighted RMS ABSOLUTE error with UNIFORM level
//! weight, not worst-case relative error. A dot product accumulates
//! `sum_k w_k x_k`, so what reaches the output is the ABSOLUTE perturbation of
//! each weight; a level's relative error never appears in that sum. Minimising
//! relative error instead chases the innermost level of a curved grid, which
//! contributes least to any dot.
//!
//! The reconstruction is `q/N` against a chain that later divides by 127, and
//! the caller folds the compensating `127/N` into the PER-TENSOR WEIGHT SCALE,
//! so a different N costs nothing at runtime. It is one host-side multiply per
//! dispatch and not one extra instruction in any kernel.
//!
//! Bit-exact against the C++ reference: float32 product + round-to-nearest-even
//! (`lrintf` under `FE_TONEAREST`), double accumulation, and ties take the
//! LARGEST N (the search ascends and uses `<=`). A tie (`level*N` exactly
//! `k+0.5`) is exact in float32 for every `|level*N| <= 127`, so the host and
//! the device cannot disagree about which side of a rounding boundary a level
//! sits on.

use crate::gqh::tables::{GQH2H_GRID, GQH3_GRID, GQH4_GRID, GRID_CODES};

/// GQH rung, matching the kernel rung codes.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum GqhRung {
    Gqh3,
    Gqh4,
    Gqh2H,
}

/// Flat default denominator (the widest choice, amax 1.0 -> extreme level with
/// no clamping). Per-grid optimal-s is opt-in via `GGML_GQH_Q8N`.
pub const Q8_DENOM_FLAT_DEFAULT: i32 = 127;

/// Round a level to int8 at denominator `n`: float32 product, round-to-nearest
///-even, clamp to `[-127, 127]`. Matches C++ `gqh_q8_round` (`lrintf` under
/// `FE_TONEAREST`).
fn q8_round(level: f32, n: i32) -> i32 {
    let p = level * (n as f32);
    let mut q = p.round_ties_even() as i32;
    if q > 127 {
        q = 127;
    }
    if q < -127 {
        q = -127;
    }
    q
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Q8Fit {
    pub n: i32,
    pub maxe: f32,
    pub rms: f32,
}

/// Search `N` in `1..=127` for the minimum uniform-weight RMS absolute error of
/// `q/N` against the grid levels. Ties resolve to the largest `N` (loop
/// ascends, `<=`).
fn fit_grid(grid: &[u32], nlev: usize) -> Q8Fit {
    let mut best = Q8Fit {
        n: Q8_DENOM_FLAT_DEFAULT,
        maxe: 0.0,
        rms: 0.0,
    };
    let mut best_rms: f64 = -1.0;
    let levels: Vec<f32> = (0..nlev).map(|i| f32::from_bits(grid[i])).collect();
    for n in 1..=127i32 {
        let mut se: f64 = 0.0;
        let mut mx: f64 = 0.0;
        for lv in &levels {
            let e = (q8_round(*lv, n) as f64) / (n as f64) - (*lv as f64);
            se += e * e;
            if e.abs() > mx {
                mx = e.abs();
            }
        }
        let rms = (se / nlev as f64).sqrt();
        if best_rms < 0.0 || rms <= best_rms {
            best_rms = rms;
            best.n = n;
            best.maxe = mx as f32;
            best.rms = rms as f32;
        }
    }
    best
}

fn grid_levels(rung: GqhRung, grid_code: u8) -> Option<(&'static [u32], usize)> {
    if grid_code as usize >= GRID_CODES {
        return None;
    }
    match rung {
        GqhRung::Gqh3 => Some((&GQH3_GRID[grid_code as usize], 8)),
        GqhRung::Gqh4 => Some((&GQH4_GRID[grid_code as usize], 16)),
        GqhRung::Gqh2H => Some((&GQH2H_GRID[grid_code as usize], 4)),
    }
}

/// Public access to a rung's signed level grid as f32 values (raw bit-pattern
/// decode). Used by harnesses that need to recompute the int8 bake on the host
/// to verify the device bake.
pub fn grid_levels_f32(rung: GqhRung, grid_code: u8) -> Option<Vec<f32>> {
    let (grid, nlev) = grid_levels(rung, grid_code)?;
    Some(grid[..nlev].iter().map(|&b| f32::from_bits(b)).collect())
}

/// Per-grid derived optimal denominator (uniform-weight RMS). Matches
/// `ggml_gqh_q8_denom`.
pub fn q8_denom(rung: GqhRung, grid_code: u8) -> Option<i32> {
    let (grid, nlev) = grid_levels(rung, grid_code)?;
    Some(fit_grid(grid, nlev).n)
}

/// Per-grid derived fit (N, maxe, rms).
pub fn q8_fit(rung: GqhRung, grid_code: u8) -> Option<Q8Fit> {
    let (grid, nlev) = grid_levels(rung, grid_code)?;
    Some(fit_grid(grid, nlev))
}

/// Max absolute level error at an arbitrary denominator (not only the derived
/// optimum). Matches `ggml_gqh_q8_maxe_at`; lets a harness state the bound for
/// the denominator actually in force.
pub fn q8_maxe_at(rung: GqhRung, grid_code: u8, n: i32) -> Option<f32> {
    if !(1..=127).contains(&n) {
        return None;
    }
    let (grid, nlev) = grid_levels(rung, grid_code)?;
    let mut mx: f64 = 0.0;
    for i in 0..nlev {
        let lv = f32::from_bits(grid[i]);
        let e = (q8_round(lv, n) as f64) / (n as f64) - (lv as f64);
        if e.abs() > mx {
            mx = e.abs();
        }
    }
    Some(mx as f32)
}

/// Read the `GGML_GQH_Q8N` policy once: unset/empty -> flat 127; "opt"/"derived"
/// /"auto" -> 0 (per-grid derived); 1..=127 -> that value; else -> derived (0).
fn q8n_env() -> i32 {
    static CACHED: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("GGML_GQH_Q8N") {
        Err(_) => Q8_DENOM_FLAT_DEFAULT,
        Ok(s) if s.is_empty() => Q8_DENOM_FLAT_DEFAULT,
        Ok(s) if matches!(s.as_str(), "opt" | "derived" | "auto") => 0,
        Ok(s) => {
            let v = s.parse::<i32>().unwrap_or(0);
            if (1..=127).contains(&v) {
                v
            } else {
                0
            }
        }
    })
}

/// Int8 representability guard (the "collapse check").
///
/// Ports PR #35's int8 range guard. Int8 cannot represent every GQH grid:
/// when a grid's smallest non-zero level, baked to int8 at denominator `n`,
/// rounds to exactly zero, the int8 arm silently destroys that level -- every
/// weight on it comes back zero. The guard REFUSES such grids so the i8/MMQ
/// arm falls back to the f32/dequant path instead of returning a perturbed
/// (collapsed) result.
///
/// "No non-zero weight may return exactly zero": a level `lv` baked to int8 is
/// `q = rn(lv * n)`; if `lv != 0.0` but `q == 0`, that is a collapse. The
/// numeric bound alone does not catch it, because int8 destroys the innermost
/// levels whose absolute error is tiny -- the collapse check is what catches it
/// (see the PR #35 reviewer notes).
///
/// The shipping artifact uses only GQH3 {3,4} and GQH4 {2,3,4} (worst 36.30:1),
/// so the guard rejects nothing it contains; GQH4 grid codes 8-11 exceed 127:1
/// and codes 10/11 round two levels to zero, so those are refused.
pub fn q8_grid_representable(rung: GqhRung, grid_code: u8, n: i32) -> bool {
    let Some((grid, nlev)) = grid_levels(rung, grid_code) else {
        return false;
    };
    if !(1..=127).contains(&n) {
        return false;
    }
    for i in 0..nlev {
        let lv = f32::from_bits(grid[i]);
        if lv != 0.0 && q8_round(lv, n) == 0 {
            return false;
        }
    }
    true
}

/// The denominator actually in force, honouring the `GGML_GQH_Q8N` policy.
/// Matches `ggml_gqh_q8_denom_eff`.
pub fn q8_denom_eff(rung: GqhRung, grid_code: u8) -> Option<i32> {
    let forced = q8n_env();
    if forced > 0 {
        Some(forced)
    } else {
        q8_denom(rung, grid_code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extreme-entry invariant: every grid's amax is exactly 1.0, so
    /// `q8_round(1.0, N) == N` and `q8_round(-1.0, N) == -N` for `N <= 127`.
    /// This is what makes the int8 LUT safe to bake: the extreme level lands
    /// exactly on `N`, never clamped.
    #[test]
    fn extreme_entries_are_exact() {
        for n in 1..=127i32 {
            assert_eq!(q8_round(1.0_f32, n), n, "q8_round(+1.0, {n})");
            assert_eq!(q8_round(-1.0_f32, n), -n, "q8_round(-1.0, {n})");
        }
        // The grids carry 0x3f800000 (+1.0) / 0xbf800000 (-1.0) as extremes.
        for code in 0..GRID_CODES as u8 {
            let g3 = &GQH3_GRID[code as usize];
            assert!(g3.contains(&0x3f800000) && g3.contains(&0xbf800000));
            let g4 = &GQH4_GRID[code as usize];
            assert!(g4.contains(&0x3f800000) && g4.contains(&0xbf800000));
            let g2h = &GQH2H_GRID[code as usize];
            assert!(g2h.contains(&0x3f800000) && g2h.contains(&0xbf800000));
        }
    }

    /// Derived denominators match the PR #35 reference table for the grids the
    /// shipping artifact uses: GQH3 {3,4}, GQH4 {2,3,4}.
    #[test]
    fn derived_denoms_match_reference() {
        // gqh3 code 3: N=120, gqh3 code 4: N=120
        assert_eq!(q8_denom(GqhRung::Gqh3, 3), Some(120));
        assert_eq!(q8_denom(GqhRung::Gqh3, 4), Some(120));
        // gqh4 code 2: N=114, code 3: N=122, code 4: N=110
        assert_eq!(q8_denom(GqhRung::Gqh4, 2), Some(114));
        assert_eq!(q8_denom(GqhRung::Gqh4, 3), Some(122));
        assert_eq!(q8_denom(GqhRung::Gqh4, 4), Some(110));
    }

    /// The derived optimum beats the flat 127 on RMS for the curved grids the
    /// artifact ships, and the bound at the in-force denominator tracks it.
    #[test]
    fn derived_beats_flat_rms() {
        for &(rung, code) in &[
            (GqhRung::Gqh3, 3),
            (GqhRung::Gqh3, 4),
            (GqhRung::Gqh4, 2),
            (GqhRung::Gqh4, 3),
            (GqhRung::Gqh4, 4),
        ] {
            let fit = q8_fit(rung, code).unwrap();
            let flat_rms = q8_maxe_at(rung, code, 127).unwrap();
            // RMS at derived N is strictly below the flat-127 maxe bound.
            assert!(
                fit.rms < flat_rms,
                "{rung:?} code {code}: rms {} vs flat maxe {}",
                fit.rms,
                flat_rms
            );
            // maxe_at the derived N matches the fit's maxe.
            assert!((q8_maxe_at(rung, code, fit.n).unwrap() - fit.maxe).abs() < 1e-6);
        }
    }

    /// Int8 range guard: the shipping artifact's grids (GQH3 {3,4}, GQH4 {2,3,4})
    /// are representable at their derived denominators -- no non-zero level
    /// collapses to zero. This is what makes the i8 arm safe to enable for the
    /// shipping artifact.
    #[test]
    fn shipping_grids_are_representable() {
        for &(rung, code) in &[
            (GqhRung::Gqh3, 3),
            (GqhRung::Gqh3, 4),
            (GqhRung::Gqh4, 2),
            (GqhRung::Gqh4, 3),
            (GqhRung::Gqh4, 4),
        ] {
            let n = q8_denom(rung, code).unwrap();
            assert!(
                q8_grid_representable(rung, code, n),
                "{rung:?} code {code} at n={n} should be representable"
            );
            // Flat 127 is also representable for these grids.
            assert!(q8_grid_representable(rung, code, 127));
        }
    }

    /// Int8 range guard: grids wide enough to exceed 127:1 collapse a non-zero
    /// level to zero and MUST be refused. GQH4 grid codes 10/11 carry levels
    /// small enough that `rn(lv * n) == 0` for a non-zero `lv`, so the guard
    /// rejects them (matching PR #35's "codes 10/11 round two levels to zero").
    #[test]
    fn wide_grids_are_refused() {
        let mut refused = 0;
        for code in 8..=11u8 {
            // At the flat 127 (the widest lattice), check whether any grid code
            // 8..11 collapses a non-zero level. Codes 10/11 are the documented
            // collapser.
            if !q8_grid_representable(GqhRung::Gqh4, code, 127) {
                refused += 1;
            }
        }
        // At least GQH4 codes 10 and 11 must be refused (the documented pair).
        assert!(
            refused >= 2,
            "expected GQH4 codes 10/11 to be refused, got {refused} of codes 8..11"
        );
    }

    /// The guard rejects a bad denominator (e.g. n=1 collapses nearly every
    /// level) and accepts a good one -- which is the negative-control property
    /// PR #35 relies on: a guard that failed to refuse cannot pass by accident.
    #[test]
    fn guard_is_denominator_sensitive() {
        // GQH3 code 3 at its derived n=120 is representable...
        assert!(q8_grid_representable(GqhRung::Gqh3, 3, 120));
        // ...but at n=1 every non-zero level except exactly +/-1 collapses.
        assert!(!q8_grid_representable(GqhRung::Gqh3, 3, 1));
        // Out-of-range n is refused.
        assert!(!q8_grid_representable(GqhRung::Gqh3, 3, 0));
        assert!(!q8_grid_representable(GqhRung::Gqh3, 3, 200));
    }
}
