use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use model_store::gguf::GgufFile;
use model_store::q6_bound::{
    activation_block_norms, decode_q6_k_block, q8_1_reconstruct, required_exact_tiles,
    summarize_tile_counts, upward_f16_norm, weight_block_norms, ExactTileSelection,
};

const HIDDEN: usize = 5120;
const VOCAB: usize = 248_320;
const Q6_BLOCKS: usize = 20;
const Q6_BYTES: usize = 210;
const TILE_ROWS: usize = 16;
const MAX_EXACT_TILES: usize = 16;

struct State {
    label: String,
    x: Vec<f32>,
    a: Vec<f32>,
    q8_quants: Vec<i8>,
    q8_scales: Vec<f32>,
    activation_norms: Vec<model_store::q6_bound::ActivationBlockNorms>,
    logits: Vec<f32>,
    proposal_real: Vec<f64>,
    proposal_task4_f32: Vec<f64>,
    baseline_real: Vec<f64>,
    q_bound: Vec<f64>,
    full_bound: Vec<f64>,
}

fn read_bf16(path: &Path, expected: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != expected * 2 {
        return Err(format!(
            "{}: expected {} BF16 values, got {} bytes",
            path.display(),
            expected,
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (sorted.len() as f64 * fraction).ceil() as usize;
    sorted[rank.saturating_sub(1)]
}

fn report_bound(
    state: &State,
    name: &str,
    center: &[f64],
    radius: &[f64],
) -> Result<ExactTileSelection, String> {
    let (winner, winner_value) = state.logits.iter().copied().enumerate().fold(
        (0usize, f32::NEG_INFINITY),
        |best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        },
    );
    let runner_up = state
        .logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(row, _)| *row != winner)
        .map(|(_, value)| value)
        .fold(f32::NEG_INFINITY, f32::max);
    let selection = required_exact_tiles(&state.logits, center, radius, TILE_ROWS)?;
    let mut widths: Vec<f64> = radius.iter().map(|value| 2.0 * value).collect();
    widths.sort_by(f64::total_cmp);
    println!("state={} bound={name}", state.label);
    println!("  exact_winner={winner} exact_bf16_logit={winner_value:.9}");
    println!(
        "  exact_runner_up_bf16={runner_up:.9} exact_winner_margin={:.9}",
        winner_value - runner_up
    );
    println!(
        "  interval_width median={:.9} p95={:.9} p99={:.9} max={:.9} winner={:.9}",
        percentile(&widths, 0.5),
        percentile(&widths, 0.95),
        percentile(&widths, 0.99),
        *widths.last().unwrap(),
        2.0 * radius[winner]
    );
    println!(
        "  rows_not_excludable={rows_not_excludable} exact_tiles_required={} fallback={} tile_limit={MAX_EXACT_TILES}",
        selection.exact_tiles_required,
        usize::from(selection.exact_tiles_required > MAX_EXACT_TILES),
        rows_not_excludable = selection.rows_not_excludable,
    );
    Ok(selection)
}

fn q8_metadata(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut quants = vec![0i8; x.len()];
    let mut scales = Vec::with_capacity(x.len() / 32);
    for (block_index, input) in x.chunks_exact(32).enumerate() {
        let amax = input.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let stored_d = f16::from_f32(d).to_f32();
        scales.push(stored_d);
        for (lane, &value) in input.iter().enumerate() {
            quants[block_index * 32 + lane] = if amax == 0.0 {
                0
            } else {
                (value / d).round().clamp(-127.0, 127.0) as i8
            };
        }
    }
    (quants, scales)
}

fn rounded_add(left: f32, right: f32) -> f32 {
    (f64::from(left) + f64::from(right)) as f32
}

fn task4_lane_dot(
    block: &model_store::q6_bound::DecodedQ6Block,
    q8_quants: &[i8],
    q8_scales: &[f32],
    block_index: usize,
    lane: usize,
) -> f32 {
    let bq8_offset = 4 * (lane / 16) + (lane % 16) / 8;
    let within = (lane % 8) * 4;
    let mut sum = 0.0f32;
    for pair in 0..2 {
        let q8_block = block_index * 8 + bq8_offset + 2 * pair;
        let logical = (bq8_offset + 2 * pair) * 32 + within;
        let mut integer_dot = 0i32;
        for coordinate in 0..4 {
            integer_dot += i32::from(block.quants[logical + coordinate])
                * i32::from(q8_quants[q8_block * 32 + within + coordinate]);
        }
        let scaled_integer = integer_dot * i32::from(block.scales[logical]);
        let term = (f64::from(q8_scales[q8_block]) * f64::from(scaled_integer)) as f32;
        sum = rounded_add(sum, term);
    }
    (f64::from(block.d) * f64::from(sum)) as f32
}

fn task4_reduce(mut partials: [[f32; 32]; 8]) -> f32 {
    for warp in 1..8 {
        for lane in 0..32 {
            partials[0][lane] = rounded_add(partials[0][lane], partials[warp][lane]);
        }
    }
    let mut wave = partials[0];
    for offset in [16usize, 8, 4, 2, 1] {
        let before = wave;
        for lane in 0..32 {
            wave[lane] = rounded_add(before[lane], before[lane ^ offset]);
        }
    }
    wave[0]
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || (args.len() - 3) % 3 != 0 {
        return Err(format!(
            "usage: {} <gguf> <sidecar-out> (<label> <hidden.bf16> <logits.bf16>)+",
            args.first().map(String::as_str).unwrap_or("q6_bound_spike")
        ));
    }
    let gguf_path = PathBuf::from(&args[1]);
    let sidecar_path = PathBuf::from(&args[2]);
    let gguf = GgufFile::open(&gguf_path).map_err(|error| error.to_string())?;
    let tensor = gguf
        .tensor("output.weight")
        .ok_or_else(|| "output.weight missing".to_string())?;
    if tensor.dims != [HIDDEN, VOCAB] || tensor.tensor_type != 14 {
        return Err(format!(
            "expected output.weight Q6_K [{HIDDEN}, {VOCAB}], got type={} dims={:?}",
            tensor.tensor_type, tensor.dims
        ));
    }
    let weights = gguf
        .tensor_bytes("output.weight")
        .map_err(|error| error.to_string())?;
    let mut states = Vec::new();
    for triple in args[3..].chunks_exact(3) {
        let x = read_bf16(Path::new(&triple[1]), HIDDEN)?;
        let a = q8_1_reconstruct(&x)?;
        let (q8_quants, q8_scales) = q8_metadata(&x);
        let activation_norms = x
            .chunks_exact(256)
            .zip(a.chunks_exact(256))
            .map(|(x, a)| activation_block_norms(x, a))
            .collect::<Result<Vec<_>, _>>()?;
        states.push(State {
            label: triple[0].clone(),
            x,
            a,
            q8_quants,
            q8_scales,
            activation_norms,
            logits: read_bf16(Path::new(&triple[2]), VOCAB)?,
            proposal_real: vec![0.0; VOCAB],
            proposal_task4_f32: vec![0.0; VOCAB],
            baseline_real: vec![0.0; VOCAB],
            q_bound: vec![0.0; VOCAB],
            full_bound: vec![0.0; VOCAB],
        });
    }

    let sidecar = File::create(&sidecar_path)
        .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
    let mut sidecar = BufWriter::new(sidecar);
    let row_bytes = Q6_BLOCKS * Q6_BYTES;
    let gamma = (10_240.0 * 2f64.powi(-24)) / (1.0 - 10_240.0 * 2f64.powi(-24));
    for row in 0..VOCAB {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut task4_partials = vec![[[0.0f32; 32]; 8]; states.len()];
        for block_index in 0..Q6_BLOCKS {
            let block =
                decode_q6_k_block(&row_data[block_index * Q6_BYTES..(block_index + 1) * Q6_BYTES])?;
            let norms = weight_block_norms(&block);
            let w_up = upward_f16_norm(norms.w_l2)?;
            let d_up = upward_f16_norm(norms.d_l2)?;
            sidecar
                .write_all(&f16::from_f32(w_up).to_bits().to_le_bytes())
                .and_then(|_| sidecar.write_all(&f16::from_f32(d_up).to_bits().to_le_bytes()))
                .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;

            for (state_index, state) in states.iter_mut().enumerate() {
                let act = state.activation_norms[block_index];
                let q = f64::from(w_up) * act.e_l2 + f64::from(d_up) * act.a_l2;
                let sbase = f64::from(w_up) * act.x_l2;
                let sprop = f64::from(w_up + d_up) * act.a_l2;
                state.q_bound[row] += q;
                state.full_bound[row] += q + gamma * (sbase + sprop);
                let offset = block_index * 256;
                for coordinate in 0..256 {
                    state.proposal_real[row] +=
                        f64::from(block.raw[coordinate]) * f64::from(state.a[offset + coordinate]);
                    state.baseline_real[row] += f64::from(block.baseline_bf16[coordinate])
                        * f64::from(state.x[offset + coordinate]);
                }
                let warp = block_index % 8;
                for lane in 0..32 {
                    let contribution = task4_lane_dot(
                        &block,
                        &state.q8_quants,
                        &state.q8_scales,
                        block_index,
                        lane,
                    );
                    task4_partials[state_index][warp][lane] =
                        rounded_add(task4_partials[state_index][warp][lane], contribution);
                }
            }
        }
        for (state, partials) in states.iter_mut().zip(task4_partials) {
            state.proposal_task4_f32[row] = f64::from(task4_reduce(partials));
        }
        if row % 32_768 == 0 {
            eprintln!("analyzed row {row}/{VOCAB}");
        }
    }
    sidecar
        .flush()
        .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;

    println!("artifact={}", gguf_path.display());
    println!("sidecar={}", sidecar_path.display());
    println!("sidecar_encoding=upward-fp16 row-major [row][20][W,D]");
    println!("grouping=20x256 values");
    println!("rounding_model=heuristic gamma(10240) charged to baseline and proposal");
    let mut q_tile_counts = Vec::with_capacity(states.len());
    let mut heuristic_tile_counts = Vec::with_capacity(states.len());
    let mut approximate_mismatches = 0usize;
    let mut q_violation_states = 0usize;
    let mut q_violations_total = 0usize;
    for state in &states {
        let q_violations = (0..VOCAB)
            .filter(|&row| {
                (state.baseline_real[row] - state.proposal_real[row]).abs() > state.q_bound[row]
            })
            .count();
        let real_proposal_winner = state
            .proposal_real
            .iter()
            .copied()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .0;
        let task4_proposal_winner = state
            .proposal_task4_f32
            .iter()
            .copied()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, candidate| {
                if bf16::from_f32(candidate.1 as f32).to_f32()
                    > bf16::from_f32(best.1 as f32).to_f32()
                {
                    candidate
                } else {
                    best
                }
            })
            .0;
        let exact_winner = state
            .logits
            .iter()
            .copied()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .0;
        let approximate_differs = task4_proposal_winner != exact_winner;
        approximate_mismatches += usize::from(approximate_differs);
        q_violation_states += usize::from(q_violations != 0);
        q_violations_total += q_violations;
        println!(
            "state={} activation_norms E_l2_total={:.9} A_l2_total={:.9} X_l2_total={:.9}",
            state.label,
            state
                .activation_norms
                .iter()
                .map(|n| n.e_l2 * n.e_l2)
                .sum::<f64>()
                .sqrt(),
            state
                .activation_norms
                .iter()
                .map(|n| n.a_l2 * n.a_l2)
                .sum::<f64>()
                .sqrt(),
            state
                .activation_norms
                .iter()
                .map(|n| n.x_l2 * n.x_l2)
                .sum::<f64>()
                .sqrt(),
        );
        println!(
            "state={} exact_winner={} task4_emulated_winner={} approximate_differs={} real_dot_q_violations={} cpu_real_proposal_winner={}",
            state.label,
            exact_winner,
            task4_proposal_winner,
            approximate_differs,
            q_violations,
            real_proposal_winner,
        );
        let q_selection = report_bound(
            state,
            "Q-only-arithmetic-independent-real-center",
            &state.proposal_real,
            &state.q_bound,
        )?;
        let heuristic_selection = report_bound(
            state,
            "Q-plus-gamma10240-task4-center-heuristic",
            &state.proposal_task4_f32,
            &state.full_bound,
        )?;
        q_tile_counts.push(q_selection.exact_tiles_required);
        heuristic_tile_counts.push(heuristic_selection.exact_tiles_required);
    }
    let q_summary = summarize_tile_counts(&q_tile_counts, MAX_EXACT_TILES)?;
    let heuristic_summary = summarize_tile_counts(&heuristic_tile_counts, MAX_EXACT_TILES)?;
    println!(
        "aggregate states={} approximate_mismatch_count={} approximate_mismatch_rate={:.6} q_violation_states={} q_violations_total={}",
        states.len(),
        approximate_mismatches,
        approximate_mismatches as f64 / states.len() as f64,
        q_violation_states,
        q_violations_total,
    );
    println!(
        "aggregate bound=Q-only tile_p50={} tile_p95={} tile_p99={} tile_max={} fallback_gt16_count={} fallback_gt16_rate={:.6}",
        q_summary.p50,
        q_summary.p95,
        q_summary.p99,
        q_summary.max,
        q_summary.fallback_count,
        q_summary.fallback_count as f64 / states.len() as f64,
    );
    println!(
        "aggregate bound=Q-plus-gamma10240 tile_p50={} tile_p95={} tile_p99={} tile_max={} fallback_gt16_count={} fallback_gt16_rate={:.6}",
        heuristic_summary.p50,
        heuristic_summary.p95,
        heuristic_summary.p99,
        heuristic_summary.max,
        heuristic_summary.fallback_count,
        heuristic_summary.fallback_count as f64 / states.len() as f64,
    );
    Ok(())
}
