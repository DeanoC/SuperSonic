use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::manifest::{
    CONVERTER_VERSION, FORMAT_VERSION, LayoutTag, Manifest, TensorMeta,
};
use crate::transforms;
use crate::Error;

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

/// Dtype name for the manifest, matching gpu-hal's ScalarType naming.
fn dtype_name(dt: safetensors::Dtype) -> Result<&'static str, Error> {
    match dt {
        safetensors::Dtype::F16 => Ok("f16"),
        safetensors::Dtype::BF16 => Ok("bf16"),
        safetensors::Dtype::F32 => Ok("f32"),
        safetensors::Dtype::U8 => Ok("u8"),
        safetensors::Dtype::U32 => Ok("u32"),
        safetensors::Dtype::I64 => Ok("i64"),
        safetensors::Dtype::F8_E4M3 => Ok("f8_e4m3"),
        other => Err(Error::UnsupportedDtype(format!("{other:?}"))),
    }
}

/// Determine which transform to apply to a tensor based on its name and shape.
fn classify_tensor(
    name: &str,
    shape: &[usize],
    layer_is_full: &[bool],
    weight_prefix: &str,
) -> LayoutTag {
    // Only transform linear attention layer tensors
    let layer_prefix = format!("{weight_prefix}.layers.");
    if let Some(rest) = name.strip_prefix(&layer_prefix) {
        // Parse layer index: "0.linear_attn.conv1d.weight" → idx=0
        if let Some(dot_pos) = rest.find('.') {
            if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                if idx < layer_is_full.len() && !layer_is_full[idx] {
                    // This is a linear attention layer
                    if name.ends_with(".conv1d.weight") && shape.len() == 3 && shape[1] == 1 {
                        return LayoutTag::DepthwiseConvSqueezed;
                    }
                    if name.ends_with(".dt_bias") && shape.len() == 1 {
                        return LayoutTag::HeadBiasReshaped;
                    }
                    if name.ends_with(".A_log") && shape.len() == 1 {
                        return LayoutTag::HeadExpReshaped;
                    }
                }
            }
        }
    }
    LayoutTag::Raw
}

/// Open all safetensors shards in a model directory.
/// Returns (mmaps, tensor_name → shard_index).
fn open_shards(model_dir: &Path) -> Result<(Vec<Mmap>, BTreeMap<String, usize>), Error> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        open_sharded(model_dir, &index_path)
    } else {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            open_single(&single)
        } else {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no safetensors files found in {}", model_dir.display()),
            )))
        }
    }
}

fn open_single(path: &Path) -> Result<(Vec<Mmap>, BTreeMap<String, usize>), Error> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let mut index = BTreeMap::new();
    for name in tensors.names() {
        index.insert(name.to_string(), 0);
    }
    Ok((vec![mmap], index))
}

fn open_sharded(
    dir: &Path,
    index_path: &Path,
) -> Result<(Vec<Mmap>, BTreeMap<String, usize>), Error> {
    let raw: serde_json::Value = serde_json::from_str(&fs::read_to_string(index_path)?)?;
    let weight_map = raw["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Other("missing weight_map in index.json".into()))?;

    let mut shard_files: Vec<String> = Vec::new();
    let mut shard_idx_map: BTreeMap<String, usize> = BTreeMap::new();
    for filename in weight_map.values() {
        let filename = filename.as_str().unwrap_or("").to_string();
        if !shard_idx_map.contains_key(&filename) {
            shard_idx_map.insert(filename.clone(), shard_files.len());
            shard_files.push(filename);
        }
    }

    let mut shards = Vec::with_capacity(shard_files.len());
    for filename in &shard_files {
        let file = File::open(dir.join(filename))?;
        shards.push(unsafe { Mmap::map(&file)? });
    }

    let mut index = BTreeMap::new();
    for (tensor_name, filename) in weight_map {
        let filename = filename.as_str().unwrap_or("");
        if let Some(&shard_idx) = shard_idx_map.get(filename) {
            index.insert(tensor_name.clone(), shard_idx);
        }
    }

    Ok((shards, index))
}

/// Bake HuggingFace safetensors into a SuperSonic binary package.
///
/// `layer_is_full[i]` should be true if layer i is full-attention.
/// `progress` is called with status messages during baking.
pub fn bake_qwen35(
    model_dir: &Path,
    weight_prefix: &str,
    num_layers: usize,
    layer_is_full: &[bool],
    fp8_native: bool,
    progress: &dyn Fn(&str),
) -> Result<(), Error> {
    assert_eq!(
        layer_is_full.len(),
        num_layers,
        "layer_is_full length must match num_layers"
    );

    let bake_dir = if fp8_native {
        crate::bake_dir_fp8(model_dir)
    } else {
        crate::bake_dir(model_dir)
    };
    fs::create_dir_all(&bake_dir)?;

    let (shards, tensor_index) = open_shards(model_dir)?;
    progress(&format!(
        "[bake] opened {} safetensors shard(s), {} tensors",
        shards.len(),
        tensor_index.len()
    ));

    let weights_path = crate::weights_bin_path(&bake_dir);
    let mut weights_file = File::create(&weights_path)?;
    let mut cursor: u64 = 0;
    let mut entries: Vec<TensorMeta> = Vec::new();

    // Collect and sort tensor names for deterministic output.
    // In BF16-dequant mode, skip _scale_inv tensors (consumed during FP8 dequant).
    // In FP8-native mode, keep them — they're stored as separate entries.
    let mut names: Vec<String> = tensor_index
        .keys()
        .filter(|name| {
            if name.ends_with("_scale_inv") {
                return fp8_native; // keep scale tensors only in FP8-native mode
            }
            name.starts_with(&format!("{weight_prefix}.")) || *name == "lm_head.weight"
        })
        .cloned()
        .collect();
    names.sort();

    progress(&format!("[bake] baking {} tensors...", names.len()));

    for name in &names {
        let &shard_idx = tensor_index.get(name).unwrap();
        let st = SafeTensors::deserialize(&shards[shard_idx])?;
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw_bytes = view.data();
        let raw_dtype = view.dtype();

        // Check if this is an FP8 tensor that needs dequantization (or native storage)
        if raw_dtype == safetensors::Dtype::F8_E4M3 && shape.len() == 2 {
            let scale_name = format!("{name}_scale_inv");
            if let Some(&scale_shard_idx) = tensor_index.get(&scale_name) {
                if fp8_native {
                    // FP8-native mode: store raw FP8 bytes as-is.
                    // The companion _scale_inv tensor will be stored separately
                    // when we encounter it in the sorted name list.
                    let offset = align_up(cursor, 4096);
                    weights_file.seek(SeekFrom::Start(offset))?;
                    weights_file.write_all(raw_bytes)?;
                    let byte_len = raw_bytes.len() as u64;

                    entries.push(TensorMeta {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: "f8_e4m3".to_string(),
                        layout: LayoutTag::Fp8Native,
                        offset,
                        byte_len,
                    });
                    cursor = offset + byte_len;
                    continue;
                }

                // BF16-dequant mode: dequantize on CPU, store as BF16.
                let scale_st = SafeTensors::deserialize(&shards[scale_shard_idx])?;
                let scale_view = scale_st.tensor(&scale_name)?;
                let scale_shape: Vec<usize> = scale_view.shape().to_vec();
                let scale_bytes = scale_view.data();

                let block_size = if scale_shape.len() == 2 && scale_shape[0] > 0 {
                    shape[0] / scale_shape[0]
                } else {
                    128
                };

                let (bytes, final_shape) = transforms::fp8_dequant_to_bf16(
                    raw_bytes, &shape, scale_bytes, &scale_shape, block_size,
                );

                let offset = align_up(cursor, 4096);
                weights_file.seek(SeekFrom::Start(offset))?;
                weights_file.write_all(&bytes)?;
                let byte_len = bytes.len() as u64;

                entries.push(TensorMeta {
                    name: name.clone(),
                    shape: final_shape,
                    dtype: "bf16".to_string(),
                    layout: LayoutTag::Fp8Dequantized,
                    offset,
                    byte_len,
                });
                cursor = offset + byte_len;
                continue;
            }
            // No scale tensor found — fall through to normal handling
        }

        let layout = classify_tensor(name, &shape, layer_is_full, weight_prefix);

        let (bytes, final_shape, final_dtype_name) = match layout {
            LayoutTag::Raw => {
                let dt = dtype_name(raw_dtype)?;
                (raw_bytes.to_vec(), shape, dt)
            }
            LayoutTag::DepthwiseConvSqueezed => {
                let (b, s) = transforms::squeeze_dim1(raw_bytes, &shape);
                (b, s, dtype_name(raw_dtype)?)
            }
            LayoutTag::HeadBiasReshaped => {
                let (b, s) = transforms::head_bias_reshape(raw_bytes, &shape);
                (b, s, dtype_name(raw_dtype)?)
            }
            LayoutTag::HeadExpReshaped => {
                // A_log is F32, output is BF16
                let (b, s) = transforms::a_log_to_exp_bf16(raw_bytes, &shape);
                (b, s, "bf16")
            }
            LayoutTag::Fp8Dequantized | LayoutTag::Fp8Native | LayoutTag::Int4Quantized => {
                unreachable!("FP8/INT4 tensors are handled before this match")
            }
        };

        // Write with 4096-byte alignment
        let offset = align_up(cursor, 4096);
        weights_file.seek(SeekFrom::Start(offset))?;
        weights_file.write_all(&bytes)?;
        let byte_len = bytes.len() as u64;

        entries.push(TensorMeta {
            name: name.clone(),
            shape: final_shape,
            dtype: final_dtype_name.to_string(),
            layout,
            offset,
            byte_len,
        });

        cursor = offset + byte_len;
    }

    weights_file.flush()?;
    drop(weights_file);

    progress(&format!(
        "[bake] wrote {:.1}MiB to weights.bin",
        cursor as f64 / (1024.0 * 1024.0)
    ));

    // Write manifest
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        converter_version: CONVERTER_VERSION,
        model_family: "qwen35".to_string(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(crate::manifest_path(&bake_dir), manifest_json)?;

    progress(&format!("[bake] manifest written to {}", bake_dir.display()));
    Ok(())
}

/// Dtype bytes-per-element lookup for split helpers.
fn dtype_bytes(dt: safetensors::Dtype) -> Result<usize, Error> {
    match dt {
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => Ok(2),
        safetensors::Dtype::F32 => Ok(4),
        safetensors::Dtype::U8 | safetensors::Dtype::F8_E4M3 => Ok(1),
        safetensors::Dtype::U32 => Ok(4),
        safetensors::Dtype::I64 => Ok(8),
        other => Err(Error::UnsupportedDtype(format!("{other:?}"))),
    }
}

/// Bake Phi-4 HuggingFace safetensors into a SuperSonic binary package.
///
/// Unlike Qwen3.5, Phi-4 stores fused `qkv_proj` and `gate_up_proj` tensors.
/// This baker splits them at bake time so the Rust runtime sees canonical
/// q_proj / k_proj / v_proj / gate_proj / up_proj tensors (all `LayoutTag::Raw`).
pub fn bake_phi4(
    model_dir: &Path,
    num_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    progress: &dyn Fn(&str),
) -> Result<(), Error> {
    let bake_dir = crate::bake_dir(model_dir);
    fs::create_dir_all(&bake_dir)?;

    let (shards, tensor_index) = open_shards(model_dir)?;
    progress(&format!(
        "[bake] opened {} safetensors shard(s), {} tensors",
        shards.len(),
        tensor_index.len()
    ));

    let weights_path = crate::weights_bin_path(&bake_dir);
    let mut weights_file = File::create(&weights_path)?;
    let mut cursor: u64 = 0;
    let mut entries: Vec<TensorMeta> = Vec::new();

    let q_rows = num_attention_heads * head_dim;
    let k_rows = num_key_value_heads * head_dim;
    let v_rows = k_rows;

    let mut names: Vec<String> = tensor_index
        .keys()
        .filter(|n| n.starts_with("model.") || *n == "lm_head.weight")
        .cloned()
        .collect();
    names.sort();

    progress(&format!("[bake] baking {} tensors (phi4)...", names.len()));

    for name in &names {
        let &shard_idx = tensor_index.get(name).unwrap();
        let st = SafeTensors::deserialize(&shards[shard_idx])?;
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw_bytes = view.data();
        let raw_dtype = view.dtype();
        let dt_bytes = dtype_bytes(raw_dtype)?;
        let dt_name = dtype_name(raw_dtype)?.to_string();

        if let Some((prefix, _)) = split_phi4_qkv_name(name) {
            let (qb, qs, kb, ks, vb, vs) = transforms::split_qkv_proj(
                raw_bytes, &shape, q_rows, k_rows, v_rows, dt_bytes,
            );
            for (suffix, bytes, shape) in [
                ("q_proj.weight", qb, qs),
                ("k_proj.weight", kb, ks),
                ("v_proj.weight", vb, vs),
            ] {
                let out_name = format!("{prefix}.{suffix}");
                cursor = write_entry(
                    &mut weights_file,
                    cursor,
                    &out_name,
                    &bytes,
                    shape,
                    dt_name.clone(),
                    LayoutTag::Raw,
                    &mut entries,
                )?;
            }
            continue;
        }

        if let Some(prefix) = split_phi4_gate_up_name(name) {
            let (gb, gs, ub, us) = transforms::split_gate_up_proj(
                raw_bytes, &shape, intermediate_size, dt_bytes,
            );
            for (suffix, bytes, shape) in [
                ("gate_proj.weight", gb, gs),
                ("up_proj.weight", ub, us),
            ] {
                let out_name = format!("{prefix}.{suffix}");
                cursor = write_entry(
                    &mut weights_file,
                    cursor,
                    &out_name,
                    &bytes,
                    shape,
                    dt_name.clone(),
                    LayoutTag::Raw,
                    &mut entries,
                )?;
            }
            continue;
        }

        cursor = write_entry(
            &mut weights_file,
            cursor,
            name,
            raw_bytes,
            shape,
            dt_name,
            LayoutTag::Raw,
            &mut entries,
        )?;
    }

    weights_file.flush()?;
    drop(weights_file);

    // Sanity: every layer must have exactly q_proj, k_proj, v_proj, gate_proj, up_proj.
    for layer_idx in 0..num_layers {
        for needed in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        ] {
            let expected = format!("model.layers.{layer_idx}.{needed}");
            if !entries.iter().any(|e| e.name == expected) {
                return Err(Error::Other(format!(
                    "bake_phi4: missing split tensor {expected} after bake"
                )));
            }
        }
    }

    progress(&format!(
        "[bake] wrote {:.1}MiB to weights.bin",
        cursor as f64 / (1024.0 * 1024.0)
    ));

    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        converter_version: CONVERTER_VERSION,
        model_family: "phi4".to_string(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(crate::manifest_path(&bake_dir), manifest_json)?;

    progress(&format!("[bake] manifest written to {}", bake_dir.display()));
    Ok(())
}

fn write_entry(
    file: &mut File,
    cursor: u64,
    name: &str,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: String,
    layout: LayoutTag,
    entries: &mut Vec<TensorMeta>,
) -> Result<u64, Error> {
    let offset = align_up(cursor, 4096);
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    let byte_len = bytes.len() as u64;
    entries.push(TensorMeta {
        name: name.to_string(),
        shape,
        dtype,
        layout,
        offset,
        byte_len,
    });
    Ok(offset + byte_len)
}

/// Match `model.layers.{N}.self_attn.qkv_proj.weight` and return (`model.layers.{N}.self_attn`, N).
fn split_phi4_qkv_name(name: &str) -> Option<(String, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    let tail = &rest[dot + 1..];
    if tail == "self_attn.qkv_proj.weight" {
        Some((format!("model.layers.{idx}.self_attn"), idx))
    } else {
        None
    }
}

/// Match `model.layers.{N}.mlp.gate_up_proj.weight` and return `model.layers.{N}.mlp`.
fn split_phi4_gate_up_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    let tail = &rest[dot + 1..];
    if tail == "mlp.gate_up_proj.weight" {
        Some(format!("model.layers.{idx}.mlp"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_name_match() {
        let (prefix, idx) = split_phi4_qkv_name("model.layers.7.self_attn.qkv_proj.weight").unwrap();
        assert_eq!(prefix, "model.layers.7.self_attn");
        assert_eq!(idx, 7);
        assert!(split_phi4_qkv_name("model.layers.7.self_attn.o_proj.weight").is_none());
        assert!(split_phi4_qkv_name("lm_head.weight").is_none());
    }

    #[test]
    fn gate_up_name_match() {
        let prefix = split_phi4_gate_up_name("model.layers.3.mlp.gate_up_proj.weight").unwrap();
        assert_eq!(prefix, "model.layers.3.mlp");
        assert!(split_phi4_gate_up_name("model.layers.3.mlp.down_proj.weight").is_none());
    }
}

