use anyhow::{bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use serde_json::Value;

use super::report::TopDeltaDim;

pub(crate) fn mean_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f32>()
        / len as f32
}

pub(crate) fn mean_square_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len().min(rhs.len());
    if len == 0 {
        return 0.0;
    }
    lhs.iter()
        .copied()
        .zip(rhs.iter().copied())
        .take(len)
        .map(|(lhs, rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

pub(crate) fn mean_square(values: &[f32]) -> f32 {
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    sum_sq / values.len() as f32
}

pub(crate) fn max_abs_delta_details(lhs: &[f32], rhs: &[f32]) -> (usize, f32, f32, f32) {
    let mut best = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (index, (lhs, rhs)) in lhs.iter().copied().zip(rhs.iter().copied()).enumerate() {
        let delta = (lhs - rhs).abs();
        if delta > best.3 {
            best = (index, lhs, rhs, delta);
        }
    }
    best
}

pub(crate) fn top_abs_delta_dims(lhs: &[f32], rhs: &[f32], top_k: usize) -> Vec<TopDeltaDim> {
    let mut dims = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .enumerate()
        .map(|(index, (native, oracle))| TopDeltaDim {
            index,
            native,
            oracle,
            delta: (native - oracle).abs(),
        })
        .collect::<Vec<_>>();
    dims.sort_by(|lhs, rhs| {
        rhs.delta
            .total_cmp(&lhs.delta)
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    dims.truncate(top_k.min(dims.len()));
    dims
}

pub(crate) fn decode_bf16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

pub(crate) fn encode_bf16_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect()
}

fn flatten_json_numbers(value: &Value, out: &mut Vec<f32>) {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_json_numbers(value, out);
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                out.push(value as f32);
            }
        }
        _ => {}
    }
}

pub(crate) fn flatten_bsh(value: &Value) -> Option<Vec<f32>> {
    value.as_array()?;
    let mut out = Vec::new();
    flatten_json_numbers(value, &mut out);
    Some(out)
}

pub(crate) fn flatten_json_vector(value: &Value) -> Option<Vec<f32>> {
    let array = value.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for elem in array {
        out.push(elem.as_f64()? as f32);
    }
    Some(out)
}

pub(crate) fn flatten_token_bsd(value: &Value, position: Option<usize>) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    let token = match position {
        Some(position) => batch.get(position)?,
        None => batch.last()?,
    };
    let mut out = Vec::new();
    flatten_json_numbers(token, &mut out);
    Some(out)
}

pub(crate) fn extract_causal_conv_window_bsd(
    value: &Value,
    position: usize,
    dim: usize,
    kernel_size: usize,
) -> Option<Vec<f32>> {
    let batch = value.as_array()?.first()?.as_array()?;
    if position >= batch.len() {
        return None;
    }
    let pad = kernel_size.saturating_sub(1);
    let mut out = vec![0.0f32; dim * kernel_size];
    for tap in 0..kernel_size {
        let src_pos = position as isize - pad as isize + tap as isize;
        if src_pos < 0 {
            continue;
        }
        let row = flatten_json_vector(batch.get(src_pos as usize)?)?;
        if row.len() != dim {
            return None;
        }
        for channel in 0..dim {
            out[channel * kernel_size + tap] = row[channel];
        }
    }
    Some(out)
}

pub(crate) fn read_buffer_all_f32(buf: &GpuBuffer) -> Result<Vec<f32>> {
    let bytes = buf
        .to_host_bytes()
        .map_err(|e| anyhow::anyhow!("buffer D2H: {e}"))?;
    match buf.dtype() {
        ScalarType::BF16 => Ok(decode_bf16_le(&bytes)),
        ScalarType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        other => bail!("unsupported buffer dtype for debug read: {other:?}"),
    }
}
