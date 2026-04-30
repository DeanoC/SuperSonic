//! Phi-4-mini decode engine.
//!
//! Loads Phi-4-mini weights, builds the per-layer descriptor array, and runs
//! prefill (one decode step per prompt token) + greedy decode through the
//! persistent megakernel in `kernels/phi4.hip`.
//!
//! BF16 only at launch; INT4/FP8 hooks live in [`kernel_ffi::phi4`] for a
//! later flag-flip pass.
use std::ffi::{c_int, c_void};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use gpu_hal::{GpuBuffer, ScalarType};
use kernel_ffi::phi4 as phi4_ffi;

use crate::oracle::OracleOutput;
use crate::registry::{FamilyParams, ModelVariant, RegistryEntry};
use crate::{oracle as oracle_mod, should_fetch_exact_bake, try_download_bake, validate};
use phi4::config::{load_config, Phi4Config};
use phi4::rope::Phi4LongRope;
use phi4::state::Phi4ModelState;
use phi4::weights::Phi4Weights;

/// Run Phi-4-mini as a one-shot CLI decode. Parallel to `run_gemma4` /
/// the inline Qwen path in `main.rs`.
pub fn run_phi4(
    cli: &crate::Cli,
    model_variant: &ModelVariant,
    entry: &RegistryEntry,
    ordinal: usize,
    total_vram: u64,
) -> Result<()> {
    let params = match &entry.params {
        FamilyParams::Phi4(p) => p,
        FamilyParams::Qwen35(_)
        | FamilyParams::Qwen36Moe(_)
        | FamilyParams::Gemma4(_)
        | FamilyParams::Llama31(_) => {
            unreachable!("run_phi4 dispatched for non-Phi4 variant {model_variant}")
        }
    };

    if cli.fp8_runtime && cli.int4 {
        bail!("Phi-4 --fp8-runtime cannot combine with --int4 (pick one weight quantization)");
    }
    if cli.batch_size != 1 {
        bail!("Phi-4 engine is single-sequence at launch (--batch-size must be 1)");
    }
    if cli.oracle_prefill || cli.gpu_validate {
        bail!("Phi-4 engine has no --oracle-prefill / --gpu-validate path yet");
    }
    if cli.prefill_chunk_size != 0 {
        bail!("Phi-4 engine has no chunked prefill yet (--prefill-chunk-size must be 0)");
    }

    if !cli.model_dir.join("config.json").exists() {
        bail!(
            "Phi-4 model dir {} has no config.json. Phi-4 has no release-hosted bake yet — \
             populate the directory by hand: `huggingface-cli download {} --local-dir {}`",
            cli.model_dir.display(),
            model_variant.hf_model_id(),
            cli.model_dir.display(),
        );
    }

    eprintln!(
        "[phi4] variant={model_variant} weight_prefix={} kv_chunk={}",
        params.weight_prefix, params.kv_chunk_size
    );

    let config =
        load_config(&cli.model_dir).map_err(|e| anyhow!("loading Phi-4 config.json: {e}"))?;
    eprintln!(
        "[phi4] hidden={} layers={} vocab={} heads={} kv_heads={} head_dim={} rot_dim={} max_pos={} mscale={:.4} tied_lm_head={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.vocab_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim(),
        config.rotary_dim(),
        config.max_position_embeddings,
        config.mscale(),
        config.tie_word_embeddings,
    );

    let tokenizer_path = cli.model_dir.join("tokenizer.json");
    let tokenizer = crate::load_tokenizer(&tokenizer_path)?;
    let prompt_ids = crate::resolve_prompt_token_ids(cli, &tokenizer)?;

    let context_tokens = cli
        .context_size
        .unwrap_or(prompt_ids.len() + cli.max_new_tokens);
    if context_tokens < prompt_ids.len() + cli.max_new_tokens {
        bail!(
            "--context-size {context_tokens} < prompt_tokens {} + max_new_tokens {}",
            prompt_ids.len(),
            cli.max_new_tokens,
        );
    }

    // KV cache cost per token: under --kv-fp8 the cache stores K/V at 1
    // byte/elem (half BF16's 2) plus an F32 absmax scale per (head, position)
    // for both K and V. Falling back to BF16 sizing under --kv-fp8 was
    // overestimating by ~2× and could spuriously abort runs that would
    // actually fit on a memory-constrained GPU.
    let kv_dtype_bytes = if cli.kv_fp8 {
        ScalarType::U8.size_in_bytes()
    } else {
        ScalarType::BF16.size_in_bytes()
    };
    let mut kv_per_token = config.kv_bytes_per_token(kv_dtype_bytes);
    if cli.kv_fp8 {
        // Two F32 scale buffers (K + V), one entry per (head, position),
        // per layer.
        let scale_bytes_per_layer =
            (2 * config.num_key_value_heads * ScalarType::F32.size_in_bytes()) as u64;
        kv_per_token += scale_bytes_per_layer * config.num_hidden_layers as u64;
    }
    let kv_bytes = kv_per_token * context_tokens as u64;
    // The registry's `fixed_bytes` budget is sized for BF16 weights + scratch.
    // Under --int4 / --fp8-runtime the weight footprint shrinks to ~1/4 / ~1/2
    // (plus a tiny scale/zero overhead), so applying the BF16 budget verbatim
    // would reject valid runs on memory-constrained cards — the exact scenario
    // these flags are meant to unlock. Mirror the 0.37× scaling that
    // qwen35_dflash_engine.rs already uses for INT4 targets, and pick a
    // conservative 0.6× for FP8 (≈ half-bytes weights + small scale_inv +
    // unchanged scratch).
    let quant_fixed_bytes = if cli.int4 {
        (entry.vram.fixed_bytes as f64 * 0.37) as u64
    } else if cli.fp8_runtime {
        (entry.vram.fixed_bytes as f64 * 0.6) as u64
    } else {
        entry.vram.fixed_bytes
    };
    let estimated_vram =
        ((quant_fixed_bytes + kv_bytes) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let weight_label = if cli.int4 {
        "weights+scratch (INT4-scaled)"
    } else if cli.fp8_runtime {
        "weights+scratch (FP8-scaled)"
    } else {
        "weights+scratch"
    };
    eprintln!(
        "[vram] estimated={:.2}GiB ({}={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        weight_label,
        gib(quant_fixed_bytes),
        gib(kv_bytes),
        context_tokens,
        gib(total_vram),
    );
    if estimated_vram > total_vram {
        bail!(
            "Insufficient VRAM for {context_tokens}-token context: need ~{:.2}GiB, GPU has {:.1}GiB. \
             Reduce --context-size or --max-new-tokens.",
            gib(estimated_vram),
            gib(total_vram),
        );
    }

    let oracle_output: Option<OracleOutput> = if cli.validate {
        let oracle_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow!("could not derive oracle script path from CARGO_MANIFEST_DIR"))?
            .join("oracle/phi4_oracle.py");
        let oracle_device =
            crate::resolve_oracle_device(&cli.oracle_device, entry.backend, ordinal);
        let oracle = oracle_mod::run_phi4_oracle(
            &oracle_script,
            &cli.model_dir,
            cli.prompt.as_str(),
            cli.max_new_tokens,
            &cli.oracle_dtype,
            &oracle_device,
        )?;
        if let Some(ref oracle_ids) = oracle.prompt_token_ids {
            if oracle_ids != &prompt_ids {
                bail!(
                    "tokenizer mismatch between Rust and Python oracle: rust={prompt_ids:?} oracle={oracle_ids:?}"
                );
            }
        }
        Some(oracle)
    } else {
        None
    };

    gpu_hal::set_device(ordinal).map_err(|e| anyhow!("set_device: {e}"))?;

    let t0 = Instant::now();
    let weights = if cli.int4 {
        let variant = model_store::fetch::BakeVariant::Int4Gptq;
        let bake_dir = variant.bake_dir(&cli.model_dir);
        // Serialise the version_ok check + try_download_bake call under
        // BakeLock so two concurrent `--int4 phi4-mini` runs against the
        // same --model-dir can't both decide the bake is missing and race
        // while extracting into the same target directory. Mirrors the
        // Qwen / Gemma / Phi-4-BF16 paths.
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
        // `should_fetch_exact_bake` is the same predicate the other engines
        // use: refetch when --download-bake is set OR when the local bake
        // is missing/stale. Honors the documented CLI semantics so users
        // who explicitly pass --download-bake actually get a refresh.
        if !cli.no_download
            && should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir))
        {
            let canonical_model = model_variant.to_string();
            match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => eprintln!(
                    "[fetch] installed Phi-4 INT4 bake at {}",
                    bake_dir.display()
                ),
                Ok(false) => {}
                Err(e) => eprintln!("[fetch] Phi-4 INT4 bake fetch failed: {e}"),
            }
        }
        if !model_store::version_ok(&bake_dir) {
            bail!(
                "Phi-4 INT4 bake not found at {} and download disabled / failed.\n\
                 Run: python3 oracle/bake_int4_phi4.py --model-dir {}",
                bake_dir.display(),
                cli.model_dir.display(),
            );
        }
        eprintln!(
            "[weights] loading INT4 GPTQ bake from {}",
            bake_dir.display()
        );
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow!("open Phi-4 INT4 bake: {e}"))?;
        Phi4Weights::load_baked(&store, &config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load Phi-4 INT4 weights: {e}"))?
    } else if cli.fp8_runtime {
        // FP8-runtime path: load v2-fp8 bake (produced by oracle/bake_fp8_phi4.py).
        // Mirrors the INT4 lock + fetch dance — no auto-bake at runtime, since
        // BF16 → FP8 calibration is a Python-side producer step.
        let variant = model_store::fetch::BakeVariant::Fp8Native;
        let bake_dir = variant.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
        if !cli.no_download
            && should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir))
        {
            let canonical_model = model_variant.to_string();
            match try_download_bake(cli, variant, &canonical_model, &bake_dir) {
                Ok(true) => eprintln!(
                    "[fetch] installed Phi-4 FP8 bake at {}",
                    bake_dir.display()
                ),
                Ok(false) => {}
                Err(e) => eprintln!("[fetch] Phi-4 FP8 bake fetch failed: {e}"),
            }
        }
        if !model_store::version_ok(&bake_dir) {
            bail!(
                "Phi-4 FP8 bake not found at {} and download disabled / failed.\n\
                 Run: python3 oracle/bake_fp8_phi4.py --model-dir {}",
                bake_dir.display(),
                cli.model_dir.display(),
            );
        }
        eprintln!("[weights] loading FP8 baked package from {}", bake_dir.display());
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow!("open Phi-4 FP8 bake: {e}"))?;
        let fp8_weights = Phi4Weights::load_baked(&store, &config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load Phi-4 FP8 weights: {e}"))?;
        // `version_ok` only checks the manifest header — a stale or
        // partial bake could ship FP8 projection bytes without the
        // companion `*_scale_inv` tensors. The loader silently leaves
        // `is_fp8 = false` in that case, after which the kernel would
        // dispatch the BF16 matmul path against FP8-packed bytes
        // (corrupt logits / OOB reads). Refuse to proceed.
        if !fp8_weights.is_fp8 {
            bail!(
                "Phi-4 FP8 bake at {} is missing FP8 scale tensors — refusing to \
                 run --fp8-runtime against a partial bake.\n\
                 Re-bake with: python3 oracle/bake_fp8_phi4.py --model-dir {}",
                bake_dir.display(),
                cli.model_dir.display(),
            );
        }
        fp8_weights
    } else if cli.no_bake {
        eprintln!("[weights] loading from raw safetensors (--no-bake)");
        Phi4Weights::load(&cli.model_dir, &config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load Phi-4 weights: {e}"))?
    } else {
        // BF16 baked path: bake on first run (splits fused qkv/gate_up,
        // 4096-aligns tensors), then mmap+H2D every subsequent run.
        let bake_dir = model_store::fetch::BakeVariant::Bf16.bake_dir(&cli.model_dir);
        let _lock = model_store::BakeLock::acquire(&cli.model_dir)
            .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
        if !model_store::version_ok(&bake_dir) {
            eprintln!("[bake] no Phi-4 baked package found — baking weights (one-time)...");
            let bake_start = Instant::now();
            model_store::bake_phi4(
                &cli.model_dir,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim(),
                config.intermediate_size,
                &|msg| eprintln!("{msg}"),
            )
            .map_err(|e| anyhow!("bake Phi-4 weights: {e}"))?;
            eprintln!("[bake] done in {:.1}s", bake_start.elapsed().as_secs_f64());
        }
        eprintln!(
            "[weights] loading BF16 baked package from {}",
            bake_dir.display()
        );
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow!("open Phi-4 BF16 bake: {e}"))?;
        Phi4Weights::load_baked(&store, &config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load Phi-4 BF16 weights: {e}"))?
    };
    if weights.is_int4 {
        eprintln!(
            "[weights] INT4 runtime dequant active (group_size={})",
            weights.int4_group_size
        );
    }
    if weights.is_fp8 {
        eprintln!(
            "[weights] FP8 runtime dequant active (block_size={})",
            weights.fp8_block_size
        );
    }
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let rope =
        Phi4LongRope::build(&config, ordinal).map_err(|e| anyhow!("build LongRoPE tables: {e}"))?;
    let mut state = Phi4ModelState::new(&config);

    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let num_layers = config.num_hidden_layers;
    let head_dim = config.head_dim();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let rms_eps = config.rms_norm_eps as f32;

    // Workspace sizing. Mirrors phi4.hip workspace layout exactly:
    //   per_batch = 4*hidden + 2*intermediate + proj_buf_floats + attn_scratch_floats.
    // proj_buf max is the largest projection out_dim — gate/up project to
    // intermediate_size, which dominates Q (num_heads*hd) and KV (kv_heads*hd).
    let proj_buf_floats = intermediate_size.max(num_heads * head_dim);
    // attn_scratch holds saved_scores (nh*kv_max_t) only — Phi-4 has no
    // QK-norm and no attention-output gate, so no Q/gate/pre-gate saves.
    // Size against full context capacity rounded up to kv_chunk_size.
    let kv_chunk = params.kv_chunk_size;
    let max_kv_cap = ((context_tokens + kv_chunk - 1) / kv_chunk).max(1) * kv_chunk;
    let attn_scratch_floats = num_heads * max_kv_cap;
    let workspace_floats =
        4 * hidden_size + 2 * intermediate_size + proj_buf_floats + attn_scratch_floats;

    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
        .map_err(|e| anyhow!("alloc workspace: {e}"))?;
    // sync_buf: 32 bytes — work-stealing counter at offset 0, grid-barrier
    // counter+flag at offsets 16/20 (same layout as persistent_decode_4b).
    let mut sync_buf = GpuBuffer::zeros(ordinal, ScalarType::U8, &[32])
        .map_err(|e| anyhow!("alloc sync_buf: {e}"))?;
    let mut matvec_counter = GpuBuffer::zeros(ordinal, ScalarType::U32, &[1])
        .map_err(|e| anyhow!("alloc matvec counter: {e}"))?;

    let mut hidden_io = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_size])
        .map_err(|e| anyhow!("alloc hidden_io: {e}"))?;
    let mut normed_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1, hidden_size])
        .map_err(|e| anyhow!("alloc normed_buf: {e}"))?;
    let mut logits_buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[config.vocab_size])
        .map_err(|e| anyhow!("alloc logits_buf: {e}"))?;

    eprintln!(
        "[phi4] workspace_floats={} (proj_buf={} attn_scratch={} kv_max_cap={})",
        workspace_floats, proj_buf_floats, attn_scratch_floats, max_kv_cap
    );

    // Pre-allocate the per-layer descriptor buffer once. The kernel reads
    // it as opaque bytes; we update kv_len + KV pointers each step via h2d
    // copy rather than freeing+reallocating the GpuBuffer.
    let desc_size_bytes = num_layers * std::mem::size_of::<phi4_ffi::Phi4DecodeLayerDesc>();
    let mut desc_device = GpuBuffer::zeros(ordinal, ScalarType::U8, &[desc_size_bytes])
        .map_err(|e| anyhow!("alloc desc_device: {e}"))?;

    // Pre-allocate the per-layer KV-FP8 scale descriptor buffer when --kv-fp8
    // is set. Rebuilt + h2d'd every step like the main descs because the
    // scale-buffer pointers can move when ensure_kv_capacity grows the cache.
    let mut kv_fp8_desc_device: Option<GpuBuffer> = if cli.kv_fp8 {
        let kv_fp8_desc_bytes =
            num_layers * std::mem::size_of::<phi4_ffi::Phi4KVCacheFp8Desc>();
        Some(
            GpuBuffer::zeros(ordinal, ScalarType::U8, &[kv_fp8_desc_bytes])
                .map_err(|e| anyhow!("alloc kv_fp8 desc_device: {e}"))?,
        )
    } else {
        None
    };

    // INT4 scale/zero descriptor array — built once, uploaded once. Pointers
    // into the (already-resident) weight buffers stay valid for the engine's
    // lifetime. Empty when weights aren't INT4.
    let int4_scale_device: Option<GpuBuffer> = if weights.is_int4 {
        let descs = build_int4_descs(&weights);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                descs.as_ptr() as *const u8,
                descs.len() * std::mem::size_of::<phi4_ffi::Phi4INT4ScaleDesc>(),
            )
        };
        Some(
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[bytes.len()], bytes)
                .map_err(|e| anyhow!("upload phi4 int4 scale descs: {e}"))?,
        )
    } else {
        None
    };

    // FP8 weight scale descriptor array — built and uploaded once. Each entry
    // points at the per-projection `_scale_inv` GPU buffer that
    // `Phi4Weights::load_baked` parked alongside the FP8 weight bytes.
    let fp8_scale_device: Option<GpuBuffer> = if weights.is_fp8 {
        let descs = build_fp8_descs(&weights, weights.fp8_block_size);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                descs.as_ptr() as *const u8,
                descs.len() * std::mem::size_of::<phi4_ffi::Phi4FP8ScaleDesc>(),
            )
        };
        Some(
            GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[bytes.len()], bytes)
                .map_err(|e| anyhow!("upload phi4 fp8 scale descs: {e}"))?,
        )
    } else {
        None
    };

    let total_steps = prompt_ids.len() + cli.max_new_tokens;
    let eos_ids: Vec<u32> = config.eos_token_ids();

    let row_bytes = hidden_size * ScalarType::BF16.size_in_bytes();
    let mut generated: Vec<u32> = Vec::with_capacity(cli.max_new_tokens);
    let mut current_token: u32 = prompt_ids[0];
    let mut last_logits: Option<Vec<f32>> = None;

    // --validate accumulators: max delta across all compared steps and a
    // count of token mismatches vs the oracle's greedy-sampled stream.
    let mut max_delta = 0.0f32;
    let mut token_mismatches: usize = 0;

    let prefill_start = Instant::now();
    let mut decode_start: Option<Instant> = None;

    for step in 0..total_steps {
        let in_prefill = step < prompt_ids.len();
        let pos = step;

        // 1. Embedding gather: copy one row from embed_tokens into hidden_io.
        let src_offset = current_token as usize * row_bytes;
        gpu_hal::copy_d2d(
            ordinal,
            hidden_io.as_mut_ptr(),
            weights.embed_tokens.offset_ptr(src_offset),
            row_bytes,
        )
        .map_err(|e| anyhow!("embedding lookup at step {step}: {e}"))?;

        // 2. Ensure KV capacity for every layer (Phi-4 is pure full-attention).
        for ls in state.layers.iter_mut() {
            ls.ensure_kv_capacity(pos, ordinal, &config, kv_chunk, cli.kv_fp8)
                .map_err(|e| anyhow!("ensure KV capacity at step {step}: {e}"))?;
        }

        // 3. Build per-layer descriptors (KV pointers + kv_len change every step)
        //    and h2d-copy them into the pre-allocated `desc_device` buffer
        //    instead of reallocating a fresh GpuBuffer per step.
        let descs = build_descs(&config, &weights, &state, pos);
        let desc_bytes = descriptor_bytes(&descs);
        gpu_hal::copy_h2d(
            ordinal,
            desc_device.as_mut_ptr(),
            desc_bytes.as_ptr() as *const c_void,
            desc_bytes.len(),
        )
        .map_err(|e| anyhow!("upload layer descs at step {step}: {e}"))?;

        // KV-FP8 scale descs (parallel to layer descs). Pointers can shift
        // whenever ensure_kv_capacity grows the cache, so rebuild every step.
        if cli.kv_fp8 {
            let kv_fp8_descs = build_kv_fp8_descs(&state);
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    kv_fp8_descs.as_ptr() as *const u8,
                    kv_fp8_descs.len() * std::mem::size_of::<phi4_ffi::Phi4KVCacheFp8Desc>(),
                )
            };
            let buf = kv_fp8_desc_device
                .as_mut()
                .expect("kv_fp8_desc_device allocated when --kv-fp8 set");
            gpu_hal::copy_h2d(
                ordinal,
                buf.as_mut_ptr(),
                bytes.as_ptr() as *const c_void,
                bytes.len(),
            )
            .map_err(|e| anyhow!("upload kv_fp8 descs at step {step}: {e}"))?;
        }

        // 4. Reset per-step scratch (workspace + counters/barrier).
        gpu_hal::memset_zeros(ordinal, workspace.as_mut_ptr(), workspace.len_bytes())
            .map_err(|e| anyhow!("clear workspace at step {step}: {e}"))?;
        gpu_hal::memset_zeros(ordinal, sync_buf.as_mut_ptr(), sync_buf.len_bytes())
            .map_err(|e| anyhow!("reset sync_buf at step {step}: {e}"))?;

        // 5. Pick LongRoPE table for this position (short vs long).
        let (cos_table, sin_table) = rope.tables_for_kv_len(pos);

        // 6. Launch persistent decode megakernel.
        phi4_ffi::persistent_decode(
            ordinal,
            ScalarType::BF16,
            num_layers,
            hidden_size,
            intermediate_size,
            pos,
            &desc_device,
            &mut hidden_io,
            &mut workspace,
            &mut sync_buf,
            cos_table,
            sin_table,
            proj_buf_floats,
            attn_scratch_floats,
            fp8_scale_device.as_ref(),
            kv_fp8_desc_device.as_ref(),
            1,
            None,
            int4_scale_device.as_ref(),
        )
        .map_err(|e| anyhow!("phi4 persistent_decode at step {step}: {e}"))?;

        // 7. Mark KV slot filled.
        for ls in state.layers.iter_mut() {
            ls.set_kv_filled(pos + 1);
        }

        // 8. Sample for the next step. Last prefill token also needs sampling
        //    so the first generated token is correct.
        let need_sample = !in_prefill || step == prompt_ids.len() - 1;
        if need_sample {
            phi4_ffi::rms_norm(
                ordinal,
                ScalarType::BF16,
                &mut normed_buf,
                &hidden_io,
                &weights.norm_weight,
                rms_eps,
                hidden_size,
            )
            .map_err(|e| anyhow!("final rms_norm at step {step}: {e}"))?;

            kernel_ffi::standalone_matvec_4b(
                ordinal,
                ScalarType::BF16,
                &mut logits_buf,
                &normed_buf,
                &*weights.lm_head,
                hidden_size,
                config.vocab_size,
                &mut matvec_counter,
            )
            .map_err(|e| anyhow!("lm_head matvec at step {step}: {e}"))?;

            let logits_bytes = logits_buf
                .to_host_bytes()
                .map_err(|e| anyhow!("logits D2H at step {step}: {e}"))?;
            let logits_f32: Vec<f32> = logits_bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();

            let next = greedy_argmax(&logits_f32);

            // --validate: compare against oracle. Prefill comparison happens
            // on the LAST prefill step (the only one we sample); decode
            // comparisons happen per generated step.
            if let Some(ref oracle) = oracle_output {
                if in_prefill {
                    let delta = validate::max_abs_delta(&logits_f32, &oracle.prefill_logits);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    let oracle_first = oracle.generated_token_ids.first().copied();
                    let mismatch = match oracle_first {
                        Some(o) if o != next => {
                            token_mismatches += 1;
                            format!(" MISMATCH (oracle_next={o})")
                        }
                        _ => String::new(),
                    };
                    eprintln!("[validate] prefill delta={delta:.4} rust_next={next}{mismatch}");
                } else {
                    let decode_idx = step - prompt_ids.len();
                    if let Some(oracle_logits) = oracle.decode_logits.get(decode_idx) {
                        let delta = validate::max_abs_delta(&logits_f32, oracle_logits);
                        if delta > max_delta {
                            max_delta = delta;
                        }
                        let oracle_next = oracle.generated_token_ids.get(decode_idx + 1).copied();
                        let mismatch = match oracle_next {
                            Some(o) if o != next => {
                                token_mismatches += 1;
                                format!(" MISMATCH (oracle_next={o})")
                            }
                            _ => String::new(),
                        };
                        eprintln!(
                            "[validate] step={decode_idx} pos={pos} delta={delta:.4} input_tok={current_token} rust_next={next}{mismatch}"
                        );
                    }
                }
            }

            last_logits = Some(logits_f32);

            if in_prefill {
                // End of prefill: hand off to decode loop.
                eprintln!(
                    "[prefill] {} tokens in {:.0}ms",
                    prompt_ids.len(),
                    prefill_start.elapsed().as_millis()
                );
                decode_start = Some(Instant::now());
                current_token = next;
            } else {
                generated.push(current_token);
                if eos_ids.contains(&next) {
                    current_token = next;
                    break;
                }
                current_token = next;
            }
        } else {
            // Still in prefill — feed the next prompt token next iteration.
            current_token = prompt_ids[step + 1];
        }
    }

    // Push the final sampled token (the loop body pushes the *previous* one).
    if last_logits.is_some() && generated.len() < cli.max_new_tokens {
        generated.push(current_token);
    }

    let decode_ms = decode_start
        .map(|s| s.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let all_ids: Vec<u32> = prompt_ids
        .iter()
        .copied()
        .chain(generated.iter().copied())
        .collect();
    let text = tokenizer
        .decode(&all_ids, true)
        .map_err(|e| anyhow!("detokenize: {e}"))?;
    println!("{text}");
    println!(
        "[tokens] {}",
        generated
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let ms_per_step = if generated.is_empty() {
        0.0
    } else {
        decode_ms / generated.len() as f64
    };
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_step={ms_per_step:.1}",
        prompt_ids.len(),
        generated.len(),
    );
    if oracle_output.is_some() {
        eprintln!("[validate] max_delta={max_delta:.4} token_mismatches={token_mismatches}");
    }

    Ok(())
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn build_descs(
    config: &Phi4Config,
    weights: &Phi4Weights,
    state: &Phi4ModelState,
    seqlen_offset: usize,
) -> Vec<phi4_ffi::Phi4DecodeLayerDesc> {
    let head_dim = config.head_dim();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let rms_eps = config.rms_norm_eps as f32;
    let rot_dim = config.rotary_dim();
    let q_out_dim = (num_heads * head_dim) as c_int;
    let k_out_dim = (num_kv_heads * head_dim) as c_int;

    let mut descs = Vec::with_capacity(config.num_hidden_layers);
    for (idx, lw) in weights.layers.iter().enumerate() {
        let ls = &state.layers[idx];
        let (k_ptr, v_ptr, kv_max_t) = match (&ls.kv_cache_k, &ls.kv_cache_v) {
            (Some(k), Some(v)) => (
                k.as_ptr() as *mut c_void,
                v.as_ptr() as *mut c_void,
                k.shape()[2] as c_int,
            ),
            _ => (std::ptr::null_mut(), std::ptr::null_mut(), 0),
        };
        descs.push(phi4_ffi::Phi4DecodeLayerDesc {
            intermediate_size: config.intermediate_size as c_int,
            rot_dim: rot_dim as c_int,
            input_norm_w: lw.input_norm_w.as_ptr(),
            input_norm_eps: rms_eps,
            post_attn_norm_w: lw.post_attn_norm_w.as_ptr(),
            post_attn_norm_eps: rms_eps,
            gate_proj_w: lw.gate_proj_w.as_ptr(),
            up_proj_w: lw.up_proj_w.as_ptr(),
            down_proj_w: lw.down_proj_w.as_ptr(),
            q_proj_w: lw.q_proj_w.as_ptr(),
            q_out_dim,
            k_proj_w: lw.k_proj_w.as_ptr(),
            k_out_dim,
            v_proj_w: lw.v_proj_w.as_ptr(),
            o_proj_w: lw.o_proj_w.as_ptr(),
            attn_head_dim: head_dim as c_int,
            attn_num_heads: num_heads as c_int,
            attn_num_kv_heads: num_kv_heads as c_int,
            kv_cache_k: k_ptr,
            kv_cache_v: v_ptr,
            kv_len: seqlen_offset as c_int,
            kv_max_t,
            kv_shadow_k: std::ptr::null_mut(),
            kv_shadow_v: std::ptr::null_mut(),
            kv_shadow_start: 0,
        });
    }
    descs
}

fn descriptor_bytes(descs: &[phi4_ffi::Phi4DecodeLayerDesc]) -> &[u8] {
    let len_bytes = descs.len() * std::mem::size_of::<phi4_ffi::Phi4DecodeLayerDesc>();
    unsafe { std::slice::from_raw_parts(descs.as_ptr() as *const u8, len_bytes) }
}

/// Build the per-layer parallel-struct array of KV-FP8 scale-buffer pointers.
/// Pointers reference the per-layer scale buffers held by `state`; struct must
/// remain alive only as long as those buffers do (the engine owns both).
/// Caller is responsible for only invoking this when `state` was allocated
/// with `kv_fp8 = true` (otherwise the scale buffers are `None`).
fn build_kv_fp8_descs(state: &Phi4ModelState) -> Vec<phi4_ffi::Phi4KVCacheFp8Desc> {
    use std::ffi::c_void;
    let mut descs = Vec::with_capacity(state.layers.len());
    for ls in &state.layers {
        let mut d = phi4_ffi::Phi4KVCacheFp8Desc::default();
        if let Some(ref sk) = ls.kv_scale_k {
            d.kv_scale_k = sk.as_ptr() as *mut c_void;
        }
        if let Some(ref sv) = ls.kv_scale_v {
            d.kv_scale_v = sv.as_ptr() as *mut c_void;
        }
        descs.push(d);
    }
    descs
}

/// Build the per-layer parallel-struct array of INT4 scale/zero pointers.
/// Pointers reference GPU buffers held by `weights`; struct must remain
/// alive only as long as those buffers do (the engine owns both).
/// Build the per-layer parallel-struct array of FP8 `_scale_inv` pointers.
/// Pointers reference GPU buffers held by `weights`; struct must remain alive
/// only as long as those buffers do (the engine owns both).
fn build_fp8_descs(
    weights: &Phi4Weights,
    block_size: usize,
) -> Vec<phi4_ffi::Phi4FP8ScaleDesc> {
    use std::ffi::c_void;
    let ptr = |opt: &Option<GpuBuffer>| -> *const c_void {
        opt.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null())
    };
    let block_size_c = block_size as std::ffi::c_int;
    let mut descs = Vec::with_capacity(weights.layers.len());
    for lw in &weights.layers {
        descs.push(phi4_ffi::Phi4FP8ScaleDesc {
            gate_proj_scale: ptr(&lw.gate_proj_fp8_scale),
            up_proj_scale: ptr(&lw.up_proj_fp8_scale),
            down_proj_scale: ptr(&lw.down_proj_fp8_scale),
            q_proj_scale: ptr(&lw.q_proj_fp8_scale),
            k_proj_scale: ptr(&lw.k_proj_fp8_scale),
            v_proj_scale: ptr(&lw.v_proj_fp8_scale),
            o_proj_scale: ptr(&lw.o_proj_fp8_scale),
            block_size: block_size_c,
        });
    }
    descs
}

fn build_int4_descs(weights: &Phi4Weights) -> Vec<phi4_ffi::Phi4INT4ScaleDesc> {
    use std::ffi::c_void;
    let ptr = |opt: &Option<GpuBuffer>| -> *const c_void {
        opt.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null())
    };
    let group_size = weights.int4_group_size as std::ffi::c_int;
    let mut descs = Vec::with_capacity(weights.layers.len());
    for lw in &weights.layers {
        descs.push(phi4_ffi::Phi4INT4ScaleDesc {
            gate_proj_scale: ptr(&lw.gate_proj_int4_scale),
            gate_proj_zero: ptr(&lw.gate_proj_int4_zero),
            up_proj_scale: ptr(&lw.up_proj_int4_scale),
            up_proj_zero: ptr(&lw.up_proj_int4_zero),
            down_proj_scale: ptr(&lw.down_proj_int4_scale),
            down_proj_zero: ptr(&lw.down_proj_int4_zero),
            q_proj_scale: ptr(&lw.q_proj_int4_scale),
            q_proj_zero: ptr(&lw.q_proj_int4_zero),
            k_proj_scale: ptr(&lw.k_proj_int4_scale),
            k_proj_zero: ptr(&lw.k_proj_int4_zero),
            v_proj_scale: ptr(&lw.v_proj_int4_scale),
            v_proj_zero: ptr(&lw.v_proj_int4_zero),
            o_proj_scale: ptr(&lw.o_proj_int4_scale),
            o_proj_zero: ptr(&lw.o_proj_int4_zero),
            group_size,
        });
    }
    descs
}
