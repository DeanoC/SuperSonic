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

/// Check if a weight tensor name is a projection weight that should be INT4-quantized.
/// Excludes: norm weights, biases, conv1d, A_log, dt_bias, embed_tokens,
///           b_proj and a_proj (only 16 rows — too small for INT4).
fn is_int4_target(name: &str) -> bool {
    // Must be a 2D weight (projections)
    if !name.ends_with(".weight") {
        return false;
    }
    // Exclude norms, embeddings
    if name.contains("layernorm") || name.contains("norm.weight")
        || name.contains("embed_tokens") || name.contains("conv1d")
    {
        return false;
    }
    // Exclude b_proj and a_proj (only 16 rows)
    if name.contains("in_proj_b.weight") || name.contains("in_proj_a.weight") {
        return false;
    }
    // Include: gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj,
    //          in_proj_qkv, in_proj_z, out_proj
    // Exclude lm_head: used via standalone_matvec which doesn't have INT4 dequant.
    if name.contains("lm_head") {
        return false;
    }
    name.contains("_proj")
}

/// Bake HuggingFace safetensors into INT4-quantized SuperSonic binary package.
/// BF16 projection weights are quantized to INT4 with group_size group quantization.
/// Non-projection weights (norms, biases, small projections) are kept in BF16.
/// FP8 source weights are first dequanted to BF16, then quantized to INT4.
pub fn bake_qwen35_int4(
    model_dir: &Path,
    weight_prefix: &str,
    num_layers: usize,
    layer_is_full: &[bool],
    group_size: usize,
    progress: &dyn Fn(&str),
) -> Result<(), Error> {
    assert_eq!(layer_is_full.len(), num_layers);

    let bake_dir = crate::bake_dir_int4(model_dir);
    fs::create_dir_all(&bake_dir)?;

    let (shards, tensor_index) = open_shards(model_dir)?;
    progress(&format!(
        "[bake-int4] opened {} safetensors shard(s), {} tensors",
        shards.len(), tensor_index.len()
    ));

    let weights_path = crate::weights_bin_path(&bake_dir);
    let mut weights_file = File::create(&weights_path)?;
    let mut cursor: u64 = 0;
    let mut entries: Vec<TensorMeta> = Vec::new();

    // Collect tensor names, skip _scale_inv (we'll consume them inline for FP8→BF16 dequant)
    let mut names: Vec<String> = tensor_index
        .keys()
        .filter(|name| {
            if name.ends_with("_scale_inv") { return false; }
            name.starts_with(&format!("{weight_prefix}.")) || *name == "lm_head.weight"
        })
        .cloned()
        .collect();
    names.sort();

    progress(&format!("[bake-int4] baking {} tensors (group_size={group_size})...", names.len()));

    for name in &names {
        let &shard_idx = tensor_index.get(name).unwrap();
        let st = SafeTensors::deserialize(&shards[shard_idx])?;
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw_bytes = view.data();
        let raw_dtype = view.dtype();

        // Step 1: Get BF16 bytes for this tensor (dequant FP8 if needed)
        let (bf16_bytes, bf16_shape) = if raw_dtype == safetensors::Dtype::F8_E4M3 && shape.len() == 2 {
            let scale_name = format!("{name}_scale_inv");
            if let Some(&scale_shard_idx) = tensor_index.get(&scale_name) {
                let scale_st = SafeTensors::deserialize(&shards[scale_shard_idx])?;
                let scale_view = scale_st.tensor(&scale_name)?;
                let scale_shape: Vec<usize> = scale_view.shape().to_vec();
                let scale_bytes = scale_view.data();
                let block_sz = if scale_shape.len() == 2 && scale_shape[0] > 0 {
                    shape[0] / scale_shape[0]
                } else { 128 };
                transforms::fp8_dequant_to_bf16(raw_bytes, &shape, scale_bytes, &scale_shape, block_sz)
            } else {
                (raw_bytes.to_vec(), shape.clone())
            }
        } else if raw_dtype == safetensors::Dtype::BF16 {
            (raw_bytes.to_vec(), shape.clone())
        } else {
            // Non-BF16/FP8 tensor — store as-is (norms, biases, etc.)
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
                    let (b, s) = transforms::a_log_to_exp_bf16(raw_bytes, &shape);
                    (b, s, "bf16")
                }
                _ => unreachable!(),
            };
            let offset = align_up(cursor, 4096);
            weights_file.seek(SeekFrom::Start(offset))?;
            weights_file.write_all(&bytes)?;
            let byte_len = bytes.len() as u64;
            entries.push(TensorMeta {
                name: name.clone(), shape: final_shape,
                dtype: final_dtype_name.to_string(), layout, offset, byte_len,
            });
            cursor = offset + byte_len;
            continue;
        };

        // Step 2: Check if this tensor should be INT4-quantized
        if is_int4_target(name) && bf16_shape.len() == 2 && bf16_shape[1] % 2 == 0 {
            let (packed, scale_bytes, zero_bytes, packed_shape) =
                transforms::bf16_to_int4(&bf16_bytes, &bf16_shape, group_size);

            // Write packed INT4 weights
            let offset = align_up(cursor, 4096);
            weights_file.seek(SeekFrom::Start(offset))?;
            weights_file.write_all(&packed)?;
            let byte_len = packed.len() as u64;
            entries.push(TensorMeta {
                name: name.clone(),
                shape: packed_shape,
                dtype: "u8".to_string(),
                layout: LayoutTag::Int4Quantized,
                offset,
                byte_len,
            });
            cursor = offset + byte_len;

            // Write scale tensor
            let scale_rows = (bf16_shape[0] + group_size - 1) / group_size;
            let scale_cols = (bf16_shape[1] + group_size - 1) / group_size;
            let scale_name = format!("{name}_int4_scale");
            let offset = align_up(cursor, 4096);
            weights_file.seek(SeekFrom::Start(offset))?;
            weights_file.write_all(&scale_bytes)?;
            let byte_len = scale_bytes.len() as u64;
            entries.push(TensorMeta {
                name: scale_name, shape: vec![scale_rows, scale_cols],
                dtype: "bf16".to_string(), layout: LayoutTag::Raw, offset, byte_len,
            });
            cursor = offset + byte_len;

            // Write zero tensor
            let zero_name = format!("{name}_int4_zero");
            let offset = align_up(cursor, 4096);
            weights_file.seek(SeekFrom::Start(offset))?;
            weights_file.write_all(&zero_bytes)?;
            let byte_len = zero_bytes.len() as u64;
            entries.push(TensorMeta {
                name: zero_name, shape: vec![scale_rows, scale_cols],
                dtype: "bf16".to_string(), layout: LayoutTag::Raw, offset, byte_len,
            });
            cursor = offset + byte_len;
        } else {
            // Store as BF16 (norms, small projections, etc.)
            let layout = classify_tensor(name, &bf16_shape, layer_is_full, weight_prefix);
            let (bytes, final_shape) = match layout {
                LayoutTag::Raw => (bf16_bytes, bf16_shape),
                LayoutTag::DepthwiseConvSqueezed => transforms::squeeze_dim1(&bf16_bytes, &bf16_shape),
                LayoutTag::HeadBiasReshaped => transforms::head_bias_reshape(&bf16_bytes, &bf16_shape),
                LayoutTag::HeadExpReshaped => transforms::a_log_to_exp_bf16(&bf16_bytes, &bf16_shape),
                _ => unreachable!(),
            };
            let offset = align_up(cursor, 4096);
            weights_file.seek(SeekFrom::Start(offset))?;
            weights_file.write_all(&bytes)?;
            let byte_len = bytes.len() as u64;
            entries.push(TensorMeta {
                name: name.clone(), shape: final_shape,
                dtype: "bf16".to_string(), layout, offset, byte_len,
            });
            cursor = offset + byte_len;
        }
    }

    weights_file.flush()?;
    drop(weights_file);

    progress(&format!(
        "[bake-int4] wrote {:.1}MiB to weights.bin",
        cursor as f64 / (1024.0 * 1024.0)
    ));

    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        converter_version: CONVERTER_VERSION,
        model_family: "qwen35".to_string(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(crate::manifest_path(&bake_dir), manifest_json)?;

    progress(&format!("[bake-int4] manifest written to {}", bake_dir.display()));
    Ok(())
}
