//! Runtime-owned Qwen3.6 weight loading and format selection surface.

use anyhow::{anyhow, Context, Result};
use model_store::manifest::LayoutTag;
use model_store::store::{Int4StorageKind, Int4StorageView};
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe::types::MultiLayerGeom;

pub use crate::qwen36_moe::layer_loader::{
    load_to_gpu, resolve_qwen36_store_name, store_contains_qwen36, store_layout_qwen36,
    Qwen36WeightMode, QWEN36_MOE_INT4_GROUP_SIZE, QWEN36_MOE_LOWBIT_GGML_Q4_K,
    QWEN36_MOE_LOWBIT_GGML_Q5_K, QWEN36_MOE_LOWBIT_GGML_Q6_K, QWEN36_MOE_LOWBIT_NATIVE_INT4,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedLmHeadSource {
    TiedBf16,
    StandaloneBf16,
    NativeInt4,
    GgmlKBlock,
}

pub struct PreparedLmHead {
    pub final_norm_bf16: Vec<u8>,
    pub lm_head_bf16: Vec<u8>,
    pub source: PreparedLmHeadSource,
}

pub fn host_load_bytes(store: &BakedStore, name: &str) -> Result<Vec<u8>> {
    store
        .raw_bytes(name)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("missing {name} in bake"))
}

pub fn prepare_lm_head_bf16(
    store: &BakedStore,
    text_config: &TextConfig,
    weight_prefix: &str,
    geom: &MultiLayerGeom,
) -> Result<PreparedLmHead> {
    let final_norm_name = format!("{weight_prefix}.norm.weight");
    let final_norm_bf16 = host_load_bytes(store, &final_norm_name).context("load final norm")?;
    let (lm_name, lm_packed, bf16_source) = if text_config.tie_word_embeddings {
        let name = format!("{weight_prefix}.embed_tokens.weight");
        let bytes = host_load_bytes(store, &name).context("load tied lm_head from embed_tokens")?;
        (name, bytes, PreparedLmHeadSource::TiedBf16)
    } else {
        let name = "lm_head.weight".to_string();
        let bytes = host_load_bytes(store, &name).context("load lm_head")?;
        (name, bytes, PreparedLmHeadSource::StandaloneBf16)
    };
    let scale_name = format!("{lm_name}_int4_scale");
    let (lm_head_bf16, source) = if store.contains(&scale_name) {
        let zero_name = format!("{lm_name}_int4_zero");
        let scale = host_load_bytes(store, &scale_name).context("load lm_head INT4 scale")?;
        let zero = host_load_bytes(store, &zero_name).context("load lm_head INT4 zero")?;
        (
            dequant_int4_to_bf16_bytes(
                &lm_packed,
                &scale,
                &zero,
                geom.vocab as usize,
                geom.hidden as usize,
                QWEN36_MOE_INT4_GROUP_SIZE as usize,
            ),
            PreparedLmHeadSource::NativeInt4,
        )
    } else if let Some(qtype) = match store.layout(&lm_name) {
        Some(LayoutTag::GgmlQ4K) => Some(QWEN36_MOE_LOWBIT_GGML_Q4_K),
        Some(LayoutTag::GgmlQ5K) => Some(QWEN36_MOE_LOWBIT_GGML_Q5_K),
        Some(LayoutTag::GgmlQ6K) => Some(QWEN36_MOE_LOWBIT_GGML_Q6_K),
        _ => None,
    } {
        (
            dequant_ggml_k_to_bf16_bytes(
                &lm_packed,
                qtype,
                geom.vocab as usize,
                geom.hidden as usize,
            ),
            PreparedLmHeadSource::GgmlKBlock,
        )
    } else {
        (lm_packed, bf16_source)
    };
    let expected_norm = geom.hidden as usize * 2;
    let expected_head = geom.vocab as usize * geom.hidden as usize * 2;
    if final_norm_bf16.len() != expected_norm {
        anyhow::bail!(
            "{final_norm_name} has {} bytes, expected {expected_norm}",
            final_norm_bf16.len()
        );
    }
    if lm_head_bf16.len() != expected_head {
        anyhow::bail!(
            "{lm_name} prepared to {} BF16 bytes, expected {expected_head}",
            lm_head_bf16.len()
        );
    }
    Ok(PreparedLmHead {
        final_norm_bf16,
        lm_head_bf16,
        source,
    })
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "BF16 bytes must be even");
    bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

#[derive(Debug, Clone, Copy)]
struct RowGroupInt4Layout {
    experts: usize,
    rows: usize,
    cols: usize,
    minimum_packed_bytes: usize,
    minimum_scale_elements: usize,
}

fn checked_row_group_int4_plane_extent(
    experts: usize,
    rows: usize,
    row_stride: usize,
    expert_stride: usize,
    logical_row_extent: usize,
    rank: usize,
    plane: &str,
) -> Result<usize> {
    let last_row_offset = rows
        .checked_sub(1)
        .and_then(|last_row| last_row.checked_mul(row_stride))
        .ok_or_else(|| anyhow!("row-group INT4 {plane} row index overflows"))?;
    let per_expert_extent = last_row_offset
        .checked_add(logical_row_extent)
        .ok_or_else(|| anyhow!("row-group INT4 {plane} row-plane size overflows"))?;

    if rank == 2 {
        return Ok(per_expert_extent);
    }
    if expert_stride < per_expert_extent {
        anyhow::bail!(
            "row-group INT4 {plane} expert stride {expert_stride} is shorter than the final logical row extent {per_expert_extent}"
        );
    }

    experts
        .checked_sub(1)
        .and_then(|last_expert| last_expert.checked_mul(expert_stride))
        .and_then(|last_expert_offset| last_expert_offset.checked_add(per_expert_extent))
        .ok_or_else(|| anyhow!("row-group INT4 {plane} expert-plane size overflows"))
}

fn checked_row_group_int4_plane_layout(view: &Int4StorageView) -> Result<RowGroupInt4Layout> {
    if view.kind != Int4StorageKind::RowGroupSymmetric {
        anyhow::bail!("row-group INT4 oracle requires RowGroupSymmetric storage view");
    }
    if view.group_size != 32 {
        anyhow::bail!(
            "row-group INT4 oracle requires group size 32, got {}",
            view.group_size
        );
    }
    if view.output_group_size != 1
        || view.implicit_zero_code != Some(8)
        || view.zero_tensor.is_some()
    {
        anyhow::bail!(
            "row-group INT4 oracle requires output group size 1 and implicit zero code 8"
        );
    }

    let (experts, rows, cols) = match view.logical_shape.as_slice() {
        [rows, cols] => (1, *rows, *cols),
        [experts, rows, cols] => (*experts, *rows, *cols),
        shape => anyhow::bail!(
            "row-group INT4 oracle requires rank-2 or rank-3 logical shape, got rank {}",
            shape.len()
        ),
    };
    if experts == 0 || rows == 0 || cols == 0 {
        anyhow::bail!("row-group INT4 oracle requires nonzero logical dimensions");
    }
    if cols % 2 != 0 {
        anyhow::bail!("row-group INT4 oracle requires an even final dimension, got {cols}");
    }
    if cols % view.group_size != 0 {
        anyhow::bail!("row-group INT4 oracle requires final dimension divisible by 32, got {cols}");
    }

    let logical_packed_row_bytes = cols / 2;
    let logical_scale_row_elements = cols / view.group_size;
    if view.packed_row_stride_bytes < logical_packed_row_bytes {
        anyhow::bail!(
            "row-group INT4 packed row stride {} is shorter than logical row bytes {logical_packed_row_bytes}",
            view.packed_row_stride_bytes
        );
    }
    if view.scale_row_stride_elements < logical_scale_row_elements {
        anyhow::bail!(
            "row-group INT4 scale row stride {} is shorter than logical scale elements {logical_scale_row_elements}",
            view.scale_row_stride_elements
        );
    }

    let (minimum_packed_bytes, minimum_scale_elements) = if view.logical_shape.len() == 2 {
        if view.packed_expert_stride_bytes != 0 || view.scale_expert_stride_elements != 0 {
            anyhow::bail!("rank-2 row-group INT4 view must not specify expert strides");
        }
        (
            checked_row_group_int4_plane_extent(
                experts,
                rows,
                view.packed_row_stride_bytes,
                view.packed_expert_stride_bytes,
                logical_packed_row_bytes,
                2,
                "packed",
            )?,
            checked_row_group_int4_plane_extent(
                experts,
                rows,
                view.scale_row_stride_elements,
                view.scale_expert_stride_elements,
                logical_scale_row_elements,
                2,
                "scale",
            )?,
        )
    } else {
        (
            checked_row_group_int4_plane_extent(
                experts,
                rows,
                view.packed_row_stride_bytes,
                view.packed_expert_stride_bytes,
                logical_packed_row_bytes,
                3,
                "packed",
            )?,
            checked_row_group_int4_plane_extent(
                experts,
                rows,
                view.scale_row_stride_elements,
                view.scale_expert_stride_elements,
                logical_scale_row_elements,
                3,
                "scale",
            )?,
        )
    };

    if minimum_packed_bytes == 0 || minimum_scale_elements == 0 {
        anyhow::bail!("row-group INT4 oracle requires nonempty packed and scale planes");
    }
    Ok(RowGroupInt4Layout {
        experts,
        rows,
        cols,
        minimum_packed_bytes,
        minimum_scale_elements,
    })
}

/// Decode a row-group G32 INT4 storage view for fixtures and diagnostics.
///
/// This CPU reference decoder is intentionally not used by the first-class
/// Qwen loader, which keeps the packed plane and sidecars resident for direct
/// GPU consumption.
pub fn dequant_row_group_int4_to_bf16_bytes(
    packed: &[u8],
    scale_bf16: &[u8],
    view: &Int4StorageView,
) -> Result<Vec<u8>> {
    let layout = checked_row_group_int4_plane_layout(view)?;
    let expected_scale_bytes = layout
        .minimum_scale_elements
        .checked_mul(2)
        .ok_or_else(|| anyhow!("row-group INT4 BF16 scale byte size overflows"))?;

    if packed.len() < layout.minimum_packed_bytes {
        anyhow::bail!(
            "row-group INT4 packed plane has {} bytes, needs at least {}",
            packed.len(),
            layout.minimum_packed_bytes
        );
    }
    if scale_bf16.len() % 2 != 0 {
        anyhow::bail!(
            "row-group INT4 scale plane must be BF16-aligned, got {} bytes",
            scale_bf16.len()
        );
    }
    if scale_bf16.len() < expected_scale_bytes {
        anyhow::bail!(
            "row-group INT4 scale plane has {} bytes, needs at least {expected_scale_bytes}",
            scale_bf16.len()
        );
    }

    let output_elements = layout
        .experts
        .checked_mul(layout.rows)
        .and_then(|elements| elements.checked_mul(layout.cols))
        .ok_or_else(|| anyhow!("row-group INT4 output element count overflows"))?;
    let output_bytes = output_elements
        .checked_mul(2)
        .ok_or_else(|| anyhow!("row-group INT4 output byte size overflows"))?;
    let mut output = Vec::with_capacity(output_bytes);

    for expert in 0..layout.experts {
        let packed_expert_base = expert
            .checked_mul(view.packed_expert_stride_bytes)
            .ok_or_else(|| anyhow!("row-group INT4 packed expert index overflows"))?;
        let scale_expert_base = expert
            .checked_mul(view.scale_expert_stride_elements)
            .ok_or_else(|| anyhow!("row-group INT4 scale expert index overflows"))?;
        for row in 0..layout.rows {
            let packed_row_offset = row
                .checked_mul(view.packed_row_stride_bytes)
                .ok_or_else(|| anyhow!("row-group INT4 packed row index overflows"))?;
            let packed_row_base = packed_expert_base
                .checked_add(packed_row_offset)
                .ok_or_else(|| anyhow!("row-group INT4 packed index overflows"))?;
            let scale_row_offset = row
                .checked_mul(view.scale_row_stride_elements)
                .ok_or_else(|| anyhow!("row-group INT4 scale row index overflows"))?;
            let scale_row_base = scale_expert_base
                .checked_add(scale_row_offset)
                .ok_or_else(|| anyhow!("row-group INT4 scale index overflows"))?;

            for col in 0..layout.cols {
                let packed_index = packed_row_base
                    .checked_add(col / 2)
                    .ok_or_else(|| anyhow!("row-group INT4 packed element index overflows"))?;
                let byte = *packed.get(packed_index).ok_or_else(|| {
                    anyhow!("row-group INT4 packed element index exceeds validated plane")
                })?;
                let code = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                if !(1..=15).contains(&code) {
                    anyhow::bail!(
                        "row-group INT4 code {code} at expert={expert} row={row} col={col} is invalid; code 0 is reserved for padding"
                    );
                }

                let scale_index = scale_row_base
                    .checked_add(col / view.group_size)
                    .ok_or_else(|| anyhow!("row-group INT4 scale element index overflows"))?;
                let scale_byte_index = scale_index
                    .checked_mul(2)
                    .ok_or_else(|| anyhow!("row-group INT4 BF16 scale index overflows"))?;
                let scale_second_byte = scale_byte_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("row-group INT4 BF16 scale index overflows"))?;
                let scale = half::bf16::from_bits(u16::from_le_bytes([
                    *scale_bf16.get(scale_byte_index).ok_or_else(|| {
                        anyhow!("row-group INT4 scale index exceeds validated plane")
                    })?,
                    *scale_bf16.get(scale_second_byte).ok_or_else(|| {
                        anyhow!("row-group INT4 scale index exceeds validated plane")
                    })?,
                ]))
                .to_f32();
                let value = (f32::from(code) - 8.0) * scale;
                output.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
            }
        }
    }
    debug_assert_eq!(output.len(), output_bytes);
    Ok(output)
}

pub fn dequant_int4_to_bf16_bytes(
    packed: &[u8],
    scale_bf16: &[u8],
    zero_bf16: &[u8],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<u8> {
    assert_eq!(packed.len(), out_dim * in_dim / 2, "packed size mismatch");
    assert_eq!(
        in_dim % group_size,
        0,
        "in_dim must be divisible by group_size"
    );
    assert_eq!(
        out_dim % group_size,
        0,
        "out_dim must be divisible by group_size"
    );
    let n_row_tiles = out_dim / group_size;
    let n_col_tiles = in_dim / group_size;
    assert_eq!(
        scale_bf16.len(),
        n_row_tiles * n_col_tiles * 2,
        "scale size mismatch"
    );
    assert_eq!(
        zero_bf16.len(),
        n_row_tiles * n_col_tiles * 2,
        "zero size mismatch"
    );

    let scale = bf16_bytes_to_f32(scale_bf16);
    let zero = bf16_bytes_to_f32(zero_bf16);
    let mut out = Vec::with_capacity(out_dim * in_dim * 2);
    let half_in = in_dim / 2;
    for o in 0..out_dim {
        let row_tile = o / group_size;
        let row_base = o * half_in;
        for i in 0..in_dim {
            let col_tile = i / group_size;
            let tile_idx = row_tile * n_col_tiles + col_tile;
            let s = scale[tile_idx];
            let z = zero[tile_idx];
            let byte = packed[row_base + (i / 2)];
            let nib = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let bf = f32_to_bf16_bits(nib as f32 * s - z * s);
            out.extend_from_slice(&bf.to_le_bytes());
        }
    }
    out
}

fn f16_le_to_f32(bytes: &[u8], offset: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn ggml_k_row_bytes(qtype: i32, cols: usize) -> usize {
    assert_eq!(cols % 256, 0, "GGML K-block cols must be divisible by 256");
    let blocks = cols / 256;
    match qtype {
        12 => blocks * 144,
        13 => blocks * 176,
        14 => blocks * 210,
        _ => panic!("unsupported GGML K-block qtype {qtype}"),
    }
}

fn ggml_q4_k_scale_min(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        ((q[j] & 63) as i32, (q[j + 4] & 63) as i32)
    } else {
        (
            ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4)) as i32,
            (((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) & 63) as i32,
        )
    }
}

fn ggml_k_dequant_scalar(packed: &[u8], qtype: i32, row: usize, col: usize, cols: usize) -> f32 {
    let block = col / 256;
    let inb = col - block * 256;
    let row_bytes = ggml_k_row_bytes(qtype, cols);
    let b = row * row_bytes
        + match qtype {
            12 => block * 144,
            13 => block * 176,
            14 => block * 210,
            _ => unreachable!(),
        };
    match qtype {
        12 => {
            let d = f16_le_to_f32(packed, b);
            let dmin = f16_le_to_f32(packed, b + 2);
            let sc = &packed[b + 4..b + 16];
            let qs = &packed[b + 16..b + 144];
            let g = inb / 64;
            let sub = (inb % 64) / 32;
            let (scale, minv) = ggml_q4_k_scale_min(2 * g + sub, sc);
            let qbyte = qs[g * 32 + (inb % 32)];
            let q = if sub != 0 {
                ((qbyte >> 4) & 0x0f) as i32
            } else {
                (qbyte & 0x0f) as i32
            };
            d * scale as f32 * q as f32 - dmin * minv as f32
        }
        13 => {
            let d = f16_le_to_f32(packed, b);
            let dmin = f16_le_to_f32(packed, b + 2);
            let sc = &packed[b + 4..b + 16];
            let qh = &packed[b + 16..b + 48];
            let ql = &packed[b + 48..b + 176];
            let g = inb / 64;
            let sub = (inb % 64) / 32;
            let idx = inb % 32;
            let (scale, minv) = ggml_q4_k_scale_min(2 * g + sub, sc);
            let qbyte = ql[g * 32 + idx];
            let lo = if sub != 0 {
                ((qbyte >> 4) & 0x0f) as i32
            } else {
                (qbyte & 0x0f) as i32
            };
            let high_mask = if sub != 0 {
                2u8 << (2 * g)
            } else {
                1u8 << (2 * g)
            };
            let hi = if qh[idx] & high_mask != 0 { 16 } else { 0 };
            d * scale as f32 * (lo + hi) as f32 - dmin * minv as f32
        }
        14 => {
            let ql = &packed[b..b + 128];
            let qh = &packed[b + 128..b + 192];
            let sc = &packed[b + 192..b + 208];
            let d = f16_le_to_f32(packed, b + 208);
            let half_idx = inb / 128;
            let pos = inb - half_idx * 128;
            let idx = pos % 32;
            let (q, scale_idx) = if pos < 32 {
                (
                    (ql[half_idx * 64 + idx] & 0x0f) as i32
                        | (((qh[half_idx * 32 + idx] >> 0) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16,
                )
            } else if pos < 64 {
                (
                    (ql[half_idx * 64 + 32 + idx] & 0x0f) as i32
                        | (((qh[half_idx * 32 + idx] >> 2) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 2,
                )
            } else if pos < 96 {
                (
                    ((ql[half_idx * 64 + idx] >> 4) & 0x0f) as i32
                        | (((qh[half_idx * 32 + idx] >> 4) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 4,
                )
            } else {
                (
                    ((ql[half_idx * 64 + 32 + idx] >> 4) & 0x0f) as i32
                        | (((qh[half_idx * 32 + idx] >> 6) & 3) as i32) << 4,
                    half_idx * 8 + idx / 16 + 6,
                )
            };
            d * sc[scale_idx] as i8 as f32 * (q - 32) as f32
        }
        _ => unreachable!(),
    }
}

pub fn dequant_ggml_k_to_bf16_bytes(
    packed: &[u8],
    qtype: i32,
    out_dim: usize,
    in_dim: usize,
) -> Vec<u8> {
    let row_bytes = ggml_k_row_bytes(qtype, in_dim);
    assert_eq!(
        packed.len(),
        out_dim * row_bytes,
        "GGML K-block size mismatch"
    );
    let mut out = Vec::with_capacity(out_dim * in_dim * 2);
    for row in 0..out_dim {
        for col in 0..in_dim {
            out.extend_from_slice(
                &f32_to_bf16_bits(ggml_k_dequant_scalar(packed, qtype, row, col, in_dim))
                    .to_le_bytes(),
            );
        }
    }
    out
}
