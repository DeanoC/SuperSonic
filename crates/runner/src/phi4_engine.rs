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
use crate::{oracle as oracle_mod, validate};
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
        FamilyParams::Qwen35(_) | FamilyParams::Gemma4(_) => {
            unreachable!("run_phi4 dispatched for non-Phi4 variant {model_variant}")
        }
    };

    if cli.fp8_runtime || cli.kv_fp8 {
        bail!("Phi-4 has no --fp8-runtime / --kv-fp8 path yet");
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

    let config = load_config(&cli.model_dir)
        .map_err(|e| anyhow!("loading Phi-4 config.json: {e}"))?;
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
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(cli.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[tokenizer] prompt_tokens={}", prompt_ids.len());
    if prompt_ids.is_empty() {
        bail!("empty prompt after tokenization");
    }

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

    let kv_per_token = config.kv_bytes_per_token(ScalarType::BF16.size_in_bytes());
    let kv_bytes = kv_per_token * context_tokens as u64;
    let estimated_vram =
        ((entry.vram.fixed_bytes + kv_bytes) as f64 * entry.vram.overhead_factor) as u64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "[vram] estimated={:.2}GiB (weights+scratch={:.2}GiB + kv_cache={:.2}GiB for {}tok) available={:.1}GiB",
        gib(estimated_vram),
        gib(entry.vram.fixed_bytes),
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
            &cli.prompt,
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
        let bake_dir = model_store::fetch::BakeVariant::Int4Gptq.bake_dir(&cli.model_dir);
        if !model_store::version_ok(&bake_dir) {
            bail!(
                "Phi-4 INT4 bake not found at {}.\n\
                 Run: python3 oracle/bake_int4_phi4.py --model-dir {}",
                bake_dir.display(),
                cli.model_dir.display(),
            );
        }
        eprintln!("[weights] loading INT4 GPTQ bake from {}", bake_dir.display());
        let store = model_store::BakedStore::open(&bake_dir)
            .map_err(|e| anyhow!("open Phi-4 INT4 bake: {e}"))?;
        Phi4Weights::load_baked(&store, &config, ordinal, params.weight_prefix)
            .map_err(|e| anyhow!("load Phi-4 INT4 weights: {e}"))?
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
        eprintln!("[weights] loading BF16 baked package from {}", bake_dir.display());
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
    eprintln!("[weights] loaded in {:.0}ms", t0.elapsed().as_millis());

    let rope = Phi4LongRope::build(&config, ordinal)
        .map_err(|e| anyhow!("build LongRoPE tables: {e}"))?;
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
    let max_kv_cap =
        ((context_tokens + kv_chunk - 1) / kv_chunk).max(1) * kv_chunk;
    let attn_scratch_floats = num_heads * max_kv_cap;
    let workspace_floats =
        4 * hidden_size + 2 * intermediate_size + proj_buf_floats + attn_scratch_floats;

    let mut workspace =
        GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
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
        Some(GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[bytes.len()], bytes)
            .map_err(|e| anyhow!("upload phi4 int4 scale descs: {e}"))?)
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
            ls.ensure_kv_capacity(pos, ordinal, &config, kv_chunk)
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

        // 4. Reset per-step scratch (workspace + counters/barrier).
        gpu_hal::memset_zeros(
            ordinal,
            workspace.as_mut_ptr(),
            workspace.len_bytes(),
        )
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
            None,
            None,
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
                    if delta > max_delta { max_delta = delta; }
                    let oracle_first = oracle.generated_token_ids.first().copied();
                    let mismatch = match oracle_first {
                        Some(o) if o != next => {
                            token_mismatches += 1;
                            format!(" MISMATCH (oracle_next={o})")
                        }
                        _ => String::new(),
                    };
                    eprintln!(
                        "[validate] prefill delta={delta:.4} rust_next={next}{mismatch}"
                    );
                } else {
                    let decode_idx = step - prompt_ids.len();
                    if let Some(oracle_logits) = oracle.decode_logits.get(decode_idx) {
                        let delta = validate::max_abs_delta(&logits_f32, oracle_logits);
                        if delta > max_delta { max_delta = delta; }
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

    let all_ids: Vec<u32> = prompt_ids.iter().copied().chain(generated.iter().copied()).collect();
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
    let ms_per_step = if generated.is_empty() { 0.0 } else { decode_ms / generated.len() as f64 };
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_step={ms_per_step:.1}",
        prompt_ids.len(),
        generated.len(),
    );
    if oracle_output.is_some() {
        eprintln!(
            "[validate] max_delta={max_delta:.4} token_mismatches={token_mismatches}"
        );
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

/// Build the per-layer parallel-struct array of INT4 scale/zero pointers.
/// Pointers reference GPU buffers held by `weights`; struct must remain
/// alive only as long as those buffers do (the engine owns both).
fn build_int4_descs(weights: &Phi4Weights) -> Vec<phi4_ffi::Phi4INT4ScaleDesc> {
    use std::ffi::c_void;
    let ptr = |opt: &Option<GpuBuffer>| -> *const c_void {
        opt.as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null())
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
