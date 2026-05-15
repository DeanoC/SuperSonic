use std::io::Write as _;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::qwen3_moe as q3ffi;
use model_store::BakedStore;
use qwen3_moe::baked::{
    inspect_int4_bake as inspect_qwen3_int4_bake, int4_contract, Int4BakeInspection,
    DEFAULT_INT4_GROUP_SIZE,
};
use qwen3_moe::desc_builder::{build_int4_scale_descs, build_layer_descs};
use qwen3_moe::device_weights::Qwen3MoeInt4DeviceWeights;
use qwen3_moe::loader::{LoadError, ScalarKind, WeightLoader};
use qwen3_moe::state::{StateAccount, StateLayout};
use qwen3_moe::weights::{
    baked_tensor_specs, checkpoint_dtype_acceptable, checkpoint_elems_for, expected_tensor_specs,
    CheckpointAccount, CheckpointDtype, TensorSpec,
};

use crate::qwen36_moe_logits::{sample_bf16_logits, XorshiftRng};
use crate::registry::{FamilyParams, RegistryEntry};
use crate::Cli;

const QWEN3_PROFILE_PHASES: usize = 20;
const QWEN3_PROFILE_PHASE_NAMES: [&str; QWEN3_PROFILE_PHASES] = [
    "input_load",
    "input_norm",
    "qkv",
    "q_norm_rope",
    "k_norm_rope",
    "kv_store",
    "attn_scores",
    "softmax",
    "attn_values",
    "o_proj",
    "attn_residual",
    "post_norm",
    "router",
    "topk",
    "moe_zero",
    "expert_gate_up",
    "expert_act",
    "expert_down",
    "expert_accum",
    "output",
];

pub(crate) fn run(cli: &Cli, entry: &RegistryEntry, total_vram: u64) -> Result<()> {
    validate_policy(cli, entry)?;

    let params = match entry.params {
        FamilyParams::Qwen3Moe(p) => p,
        _ => return Err(anyhow!("registry entry is not Qwen3Moe family")),
    };

    let config = qwen3_moe::config::load_config(&cli.model_dir)
        .map_err(|e| anyhow!("loading Qwen3-MoE config.json: {e}"))?;
    let text = &config.text_config;
    validate_config_matches_registry(text, &params)?;

    let checkpoint = CheckpointAccount::from_config(text);
    let projected_int4 = checkpoint.project_int4_total_bytes(text, 128);
    let context_tokens = cli.context_size.unwrap_or_else(|| {
        cli.prompt
            .chars()
            .count()
            .saturating_add(cli.max_new_tokens.max(1))
    });
    let state = StateAccount::from_config(text, StateLayout::new(context_tokens.max(1), 1, false));

    if cli.dry_run {
        println!("Qwen3-MoE dry run: qwen3-30b-a3b");
        println!("  backend: {:?} {:?}", entry.backend, entry.arch);
        println!("  layers: {}", text.num_hidden_layers);
        println!("  hidden: {}", text.hidden_size);
        println!(
            "  heads/kv/head_dim: {}/{}/{}",
            text.num_attention_heads, text.num_key_value_heads, text.head_dim
        );
        println!(
            "  experts/top_k/moe_hidden: {}/{}/{}",
            text.num_experts,
            text.top_k(),
            text.moe_intermediate_size
        );
        println!(
            "  checkpoint tensors: {}",
            expected_tensor_specs(text, params.weight_prefix).len()
        );
        println!(
            "  baked tensors: {}",
            baked_tensor_specs(text, params.weight_prefix).len()
        );
        println!(
            "  checkpoint bytes: {:.2} GiB",
            checkpoint.total_bytes as f64 / 1024.0_f64.powi(3)
        );
        println!(
            "  projected INT4 bytes: {:.2} GiB",
            projected_int4 as f64 / 1024.0_f64.powi(3)
        );
        println!(
            "  state bytes @ ctx={context_tokens}: {:.2} GiB",
            state.total_bytes as f64 / 1024.0_f64.powi(3)
        );
        println!(
            "  total VRAM: {:.2} GiB",
            total_vram as f64 / 1024.0_f64.powi(3)
        );
        inspect_checkpoint(cli, text, params.weight_prefix)?;
        inspect_int4_bake(cli, text, params.weight_prefix)?;
        return Ok(());
    }

    decode_text(cli, entry, text, params.weight_prefix)
}

fn validate_policy(cli: &Cli, entry: &RegistryEntry) -> Result<()> {
    if !matches!(entry.backend, Backend::Hip | Backend::Metal) {
        anyhow::bail!("Qwen3-30B-A3B v1 is supported only on HIP and Metal");
    }
    if !cli.int4 {
        anyhow::bail!("Qwen3-30B-A3B v1 requires --int4; BF16 weights do not fit gfx1100 24 GB");
    }
    if cli.fp8_runtime || cli.kv_fp8 || cli.q4km || cli.q4km_gptq {
        anyhow::bail!(
            "Qwen3-30B-A3B v1 supports only native --int4; FP8/KV-FP8/q4km are out of scope"
        );
    }
    if cli.batch_size != 1 {
        anyhow::bail!("Qwen3-30B-A3B v1 supports only --batch-size 1");
    }
    Ok(())
}

fn decode_text(
    cli: &Cli,
    entry: &RegistryEntry,
    text: &qwen3_moe::config::TextConfig,
    weight_prefix: &str,
) -> Result<()> {
    ensure_int4_bake(cli, entry)?;
    let bake_dir = model_store::bake_dir_int4(&cli.model_dir);
    let store = BakedStore::open(&bake_dir)
        .with_context(|| format!("open Qwen3-MoE INT4 bake at {}", bake_dir.display()))?;

    let prompt = prepare_prompt(cli, text)?;
    println!(
        "  prompt: {:?} -> {} token{}",
        cli.prompt,
        prompt.prompt_ids.len(),
        if prompt.prompt_ids.len() == 1 {
            ""
        } else {
            "s"
        }
    );

    let ordinal = cli.device;
    let max_new = cli.max_new_tokens.max(1);
    let context = cli
        .context_size
        .unwrap_or(prompt.prompt_ids.len() + max_new)
        .max(prompt.prompt_ids.len() + max_new)
        .max(1);

    println!(
        "  loading Qwen3-30B-A3B INT4 weights from {}",
        bake_dir.display()
    );
    let weights = Qwen3MoeInt4DeviceWeights::load(&store, ordinal, text, weight_prefix)
        .context("load Qwen3-MoE INT4 weights to GPU")?;
    println!(
        "  uploaded Qwen3-MoE weights referenced by manifest ({:.2} GiB)",
        weights.total_bytes as f64 / 1024.0_f64.powi(3)
    );

    let mut kv_caches = Vec::with_capacity(text.num_hidden_layers);
    let kv_dim = text.num_key_value_heads * text.head_dim;
    for _ in 0..text.num_hidden_layers {
        let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[context, kv_dim])
            .context("alloc Qwen3 K cache")?;
        let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[context, kv_dim])
            .context("alloc Qwen3 V cache")?;
        kv_caches.push((k, v));
    }

    let mut layer_ptrs = weights.layer_ptrs();
    for (i, (k, v)) in kv_caches.iter_mut().enumerate() {
        layer_ptrs[i].kv_cache_k = k.as_mut_ptr();
        layer_ptrs[i].kv_cache_v = v.as_mut_ptr();
    }
    let scale_ptrs = weights.int4_scale_ptrs();
    let int4_descs = build_int4_scale_descs(&scale_ptrs, DEFAULT_INT4_GROUP_SIZE);
    let desc_bytes_len =
        text.num_hidden_layers * std::mem::size_of::<q3ffi::Qwen3MoeDecodeLayerDesc>();
    let mut persistent_descs_dev = GpuBuffer::zeros(ordinal, ScalarType::U8, &[desc_bytes_len])
        .context("alloc Qwen3 persistent layer descriptors")?;
    let persistent_int4_descs_dev = {
        let int4_bytes = struct_slice_as_bytes(&int4_descs);
        GpuBuffer::from_host_bytes(ordinal, ScalarType::U8, &[int4_bytes.len()], int4_bytes)
            .context("upload Qwen3 persistent INT4 descriptors")?
    };

    let mut hidden_a = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[text.hidden_size])
        .context("alloc Qwen3 hidden_a")?;
    let mut hidden_b = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[text.hidden_size])
        .context("alloc Qwen3 hidden_b")?;
    let mut hidden_c = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[text.hidden_size])
        .context("alloc Qwen3 hidden_c")?;
    let workspace_floats = qwen3_workspace_floats(text, context);
    let mut workspace = GpuBuffer::zeros(ordinal, ScalarType::F32, &[workspace_floats])
        .context("alloc Qwen3 layer workspace")?;
    let mut persistent_sync = GpuBuffer::zeros(ordinal, ScalarType::U8, &[96])
        .context("alloc Qwen3 persistent sync buffer")?;
    let profile_decode_phases = std::env::var_os("SUPERSONIC_QWEN3_PROFILE_PHASES").is_some();
    let mut decode_profile = if profile_decode_phases {
        Some(
            GpuBuffer::zeros(
                ordinal,
                ScalarType::U8,
                &[text.num_hidden_layers * QWEN3_PROFILE_PHASES * std::mem::size_of::<u64>()],
            )
            .context("alloc Qwen3 decode phase profile buffer")?,
        )
    } else {
        None
    };

    let lm_head = weights
        .lm_head
        .as_ref()
        .ok_or_else(|| anyhow!("Qwen3-MoE INT4 bake is missing lm_head.weight"))?;
    let mut logits =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[text.vocab_size]).context("alloc logits")?;
    let mut lm_counter =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[1]).context("alloc lm_head counter")?;

    print_prompt_prefix(prompt.tokenizer.as_ref(), &prompt.prompt_ids);

    let mut rng = XorshiftRng::new(cli.sampling_seed);
    let mut current = prompt.prompt_ids[0];
    let total_steps = prompt.prompt_ids.len() + max_new - 1;
    let mut generated = Vec::with_capacity(max_new);
    let mut last_logits = Vec::new();
    let start = std::time::Instant::now();
    let mut embed_elapsed = std::time::Duration::ZERO;
    let mut decode_elapsed = std::time::Duration::ZERO;
    let mut lm_head_elapsed = std::time::Duration::ZERO;
    let mut sample_elapsed = std::time::Duration::ZERO;
    let mut gen_steps = 0usize;
    let use_persistent = entry.backend != Backend::Metal && !cli.no_persistent_decode;
    if use_persistent {
        eprintln!(
            "[qwen3-moe] persistent decode enabled; pass --no-persistent-decode for chained A/B"
        );
    } else if entry.backend == Backend::Metal {
        eprintln!(
            "[qwen3-moe] Metal decode uses chained layer fallbacks; persistent decode is HIP-only"
        );
    }

    for step in 0..total_steps {
        let t_embed = std::time::Instant::now();
        let row = lookup_embed_row(&store, weight_prefix, current as usize, text.hidden_size)
            .with_context(|| format!("lookup Qwen3 embedding row for token {current}"))?;
        gpu_hal::copy_h2d(
            ordinal,
            hidden_a.as_mut_ptr(),
            row.as_ptr() as *const _,
            row.len(),
        )
        .context("h2d Qwen3 token embedding")?;
        embed_elapsed += t_embed.elapsed();

        let descs = build_layer_descs(text, &layer_ptrs, step, context)
            .context("build Qwen3 layer descriptors")?;
        let t_decode = std::time::Instant::now();
        let final_in_b = if use_persistent {
            let desc_bytes = struct_slice_as_bytes(&descs);
            gpu_hal::copy_h2d(
                ordinal,
                persistent_descs_dev.as_mut_ptr(),
                desc_bytes.as_ptr() as *const _,
                desc_bytes.len(),
            )
            .context("upload Qwen3 persistent layer descriptors")?;
            q3ffi::persistent_decode_launch(
                ordinal,
                ScalarType::BF16,
                &persistent_descs_dev,
                &persistent_int4_descs_dev,
                descs.len(),
                text.hidden_size as i32,
                step as i32,
                &hidden_a,
                &mut hidden_b,
                &mut hidden_c,
                &mut workspace,
                &mut persistent_sync,
                decode_profile.as_mut(),
            )
            .context("Qwen3 persistent decode")?;
            if step + 1 >= prompt.prompt_ids.len() {
                if let Some(profile) = decode_profile.as_ref() {
                    print_decode_phase_profile(step, text.num_hidden_layers, profile)
                        .context("print Qwen3 decode phase profile")?;
                }
            }
            true
        } else {
            let mut front_a = true;
            for (layer_idx, (desc, int4)) in descs.iter().zip(int4_descs.iter()).enumerate() {
                let (input, output) = if front_a {
                    (&hidden_a, &mut hidden_b)
                } else {
                    (&hidden_b, &mut hidden_a)
                };
                q3ffi::decode_layer_launch(
                    ordinal,
                    ScalarType::BF16,
                    desc,
                    int4,
                    text.hidden_size as i32,
                    step as i32,
                    input,
                    output,
                    &mut workspace,
                )
                .with_context(|| format!("Qwen3 decode layer {layer_idx}"))?;
                front_a = !front_a;
            }
            !front_a
        };
        decode_elapsed += t_decode.elapsed();

        let final_hidden = if final_in_b { &hidden_b } else { &hidden_a };
        if step + 1 < prompt.prompt_ids.len() {
            current = prompt.prompt_ids[step + 1];
            continue;
        }

        let t_lm_head = std::time::Instant::now();
        q3ffi::lm_head_int4_launch(
            ordinal,
            ScalarType::BF16,
            text.hidden_size as i32,
            text.vocab_size as i32,
            text.rms_norm_eps as f32,
            final_hidden,
            &weights.final_norm,
            &lm_head.weight,
            &lm_head.scale,
            &lm_head.zero,
            DEFAULT_INT4_GROUP_SIZE as i32,
            &mut logits,
            &mut lm_counter,
        )
        .context("Qwen3 INT4 lm_head")?;
        last_logits = logits.to_host_bytes().context("d2h Qwen3 logits")?;
        lm_head_elapsed += t_lm_head.elapsed();
        let t_sample = std::time::Instant::now();
        let next = sample_bf16_logits(
            &last_logits,
            cli.temperature,
            cli.top_k,
            cli.top_p,
            &mut rng,
        );
        sample_elapsed += t_sample.elapsed();
        gen_steps += 1;
        generated.push(next);
        print_token(prompt.tokenizer.as_ref(), next);
        std::io::stdout().flush().ok();
        current = next;
        if Some(next) == prompt.eos_id || generated.len() >= max_new {
            break;
        }
    }

    if cli.dump_last_logits && !last_logits.is_empty() {
        let vals = crate::qwen36_moe_logits::bf16_bytes_to_f32(&last_logits);
        println!(
            "\nLAST_LOGITS: {}",
            vals.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    } else {
        println!();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Generated {} token{} in {:.2}s ({:.2} tok/s).",
        generated.len(),
        if generated.len() == 1 { "" } else { "s" },
        elapsed,
        generated.len() as f64 / elapsed.max(1e-9)
    );
    let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
    let ms_per_step = if generated.is_empty() {
        0.0
    } else {
        decode_ms / generated.len() as f64
    };
    eprintln!(
        "[result] prompt_tokens={} generated_tokens={} decode_ms={decode_ms:.0} ms_per_step={ms_per_step:.1}",
        prompt.prompt_ids.len(),
        generated.len(),
    );
    if cli.emit_stage_timings && gen_steps > 0 {
        let n = gen_steps as f64;
        eprintln!(
            "[qwen3-moe stage-timings] gen_steps={} path={} embed_ms_avg={:.3} decode_ms_avg={:.3} lm_head_ms_avg={:.3} sample_ms_avg={:.3}",
            gen_steps,
            if use_persistent { "persistent" } else { "chained" },
            embed_elapsed.as_secs_f64() * 1000.0 / n,
            decode_elapsed.as_secs_f64() * 1000.0 / n,
            lm_head_elapsed.as_secs_f64() * 1000.0 / n,
            sample_elapsed.as_secs_f64() * 1000.0 / n,
        );
    }
    Ok(())
}

fn struct_slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn ensure_int4_bake(cli: &Cli, entry: &RegistryEntry) -> Result<()> {
    let bake_dir = model_store::bake_dir_int4(&cli.model_dir);
    let _lock = model_store::BakeLock::acquire(&cli.model_dir)
        .map_err(|e| anyhow!("acquire bake lock: {e}"))?;
    if !cli.no_download
        && crate::should_fetch_exact_bake(cli.download_bake, model_store::version_ok(&bake_dir))
    {
        let canonical_model = entry.model.to_string();
        match crate::try_download_bake(
            cli,
            model_store::fetch::BakeVariant::Int4Gptq,
            &canonical_model,
            &bake_dir,
        ) {
            Ok(true) => eprintln!(
                "[fetch] installed Qwen3-MoE INT4 bake at {}",
                bake_dir.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!("[fetch] Qwen3-MoE INT4 bake fetch failed: {e}"),
        }
    }
    if !model_store::version_ok(&bake_dir) {
        anyhow::bail!(
            "Qwen3-30B-A3B decode requires an INT4-GPTQ bake at {}",
            bake_dir.display()
        );
    }
    Ok(())
}

struct PromptSetup {
    tokenizer: Option<tokenizers::Tokenizer>,
    prompt_ids: Vec<u32>,
    eos_id: Option<u32>,
}

fn prepare_prompt(cli: &Cli, text: &qwen3_moe::config::TextConfig) -> Result<PromptSetup> {
    let tokenizer = crate::load_tokenizer(&cli.model_dir.join("tokenizer.json")).ok();
    let bos_id = text
        .bos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let eos_id = text
        .eos_token_id
        .as_ref()
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let prompt_ids = match (&tokenizer, cli.prompt.is_empty()) {
        (Some(tok), false) => {
            let enc = tok
                .encode(cli.prompt.as_str(), !cli.prompt_no_special_tokens)
                .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
            let ids = enc.get_ids().to_vec();
            if ids.is_empty() {
                vec![bos_id]
            } else {
                ids
            }
        }
        _ => vec![bos_id],
    };
    Ok(PromptSetup {
        tokenizer,
        prompt_ids,
        eos_id,
    })
}

fn print_prompt_prefix(tokenizer: Option<&tokenizers::Tokenizer>, ids: &[u32]) {
    if let Some(tok) = tokenizer {
        if let Ok(text) = tok.decode(ids, false) {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
    }
}

fn print_token(tokenizer: Option<&tokenizers::Tokenizer>, id: u32) {
    if let Some(tok) = tokenizer {
        if let Ok(text) = tok.decode(&[id], false) {
            print!("{text}");
        } else {
            print!("<{id}>");
        }
    } else {
        print!(" {id}");
    }
}

fn lookup_embed_row(
    store: &BakedStore,
    weight_prefix: &str,
    token_id: usize,
    hidden: usize,
) -> Result<Vec<u8>> {
    let name = format!("{weight_prefix}.embed_tokens.weight");
    let bytes = store
        .raw_bytes(&name)
        .ok_or_else(|| anyhow!("missing {name} in bake"))?;
    let row_bytes = hidden * 2;
    let start = token_id
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow!("embedding row offset overflow for token {token_id}"))?;
    let end = start + row_bytes;
    if end > bytes.len() {
        anyhow::bail!(
            "embed_tokens row {token_id} out of bounds (need {end} bytes, have {})",
            bytes.len()
        );
    }
    Ok(bytes[start..end].to_vec())
}

fn qwen3_workspace_floats(text: &qwen3_moe::config::TextConfig, context: usize) -> usize {
    let hidden = text.hidden_size;
    let q_dim = text.num_attention_heads * text.head_dim;
    let kv_dim = text.num_key_value_heads * text.head_dim;
    let experts = text.num_experts;
    let top_k = text.top_k();
    let i = text.moe_intermediate_size;
    let scores = text.num_attention_heads * context;
    let partials = 128;
    hidden
        + q_dim
        + kv_dim
        + kv_dim
        + q_dim
        + hidden
        + experts
        + experts
        + top_k
        + top_k
        + top_k * 2 * i
        + top_k * i
        + top_k * hidden
        + hidden
        + scores
        + partials
}

fn print_decode_phase_profile(step: usize, num_layers: usize, profile: &GpuBuffer) -> Result<()> {
    let bytes = profile
        .to_host_bytes()
        .context("d2h Qwen3 decode phase profile")?;
    let mut phase_totals = [0u64; QWEN3_PROFILE_PHASES];
    let mut layer_totals = vec![0u64; num_layers];
    for (idx, chunk) in bytes.chunks_exact(std::mem::size_of::<u64>()).enumerate() {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        let cycles = u64::from_ne_bytes(arr);
        let layer = idx / QWEN3_PROFILE_PHASES;
        let phase = idx % QWEN3_PROFILE_PHASES;
        if layer < num_layers {
            phase_totals[phase] = phase_totals[phase].saturating_add(cycles);
            layer_totals[layer] = layer_totals[layer].saturating_add(cycles);
        }
    }
    let total: u64 = phase_totals.iter().copied().sum();
    if total == 0 {
        eprintln!("[qwen3-moe phase-profile] step={step} no samples");
        return Ok(());
    }

    let mut phases: Vec<_> = QWEN3_PROFILE_PHASE_NAMES
        .iter()
        .zip(phase_totals.iter())
        .filter(|(_, cycles)| **cycles > 0)
        .map(|(name, cycles)| (*name, *cycles))
        .collect();
    phases.sort_by_key(|(_, cycles)| std::cmp::Reverse(*cycles));
    eprintln!("[qwen3-moe phase-profile] step={step} total_cycles={total}");
    for (name, cycles) in phases.iter().take(12) {
        eprintln!(
            "[qwen3-moe phase-profile]   {:>16}: {:>14} cycles {:>6.2}%",
            name,
            cycles,
            (*cycles as f64) * 100.0 / (total as f64)
        );
    }

    let mut layers: Vec<_> = layer_totals
        .iter()
        .enumerate()
        .filter(|(_, cycles)| **cycles > 0)
        .map(|(layer, cycles)| (layer, *cycles))
        .collect();
    layers.sort_by_key(|(_, cycles)| std::cmp::Reverse(*cycles));
    let top_layers = layers
        .iter()
        .take(6)
        .map(|(layer, cycles)| {
            format!(
                "{}:{:.2}%",
                layer,
                (*cycles as f64) * 100.0 / (total as f64)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("[qwen3-moe phase-profile]   top_layers: {top_layers}");
    Ok(())
}

fn inspect_checkpoint(cli: &Cli, text: &qwen3_moe::config::TextConfig, prefix: &str) -> Result<()> {
    let specs = expected_tensor_specs(text, prefix);
    match WeightLoader::from_dir(&cli.model_dir) {
        Ok(loader) => {
            let report = inspect_loader_specs(&loader, text, &specs)?;
            println!(
                "  HF safetensors: present ({} tensors)",
                loader.tensor_count()
            );
            println!(
                "    expected tensors present: {}/{}",
                report.present,
                specs.len()
            );
            if report.missing > 0 {
                println!(
                    "    missing expected tensors: {}{}",
                    report.missing,
                    format_examples(&report.examples)
                );
            }
            if report.shape_mismatches > 0 {
                println!(
                    "    shape mismatches: {}{}",
                    report.shape_mismatches,
                    format_examples(&report.shape_examples)
                );
            }
            if report.dtype_mismatches > 0 {
                println!(
                    "    dtype mismatches: {}{}",
                    report.dtype_mismatches,
                    format_examples(&report.dtype_examples)
                );
            }
            if report.missing == 0 && report.shape_mismatches == 0 && report.dtype_mismatches == 0 {
                println!(
                    "    expected tensor bytes: {:.2} GiB",
                    report.bytes as f64 / 1024.0_f64.powi(3)
                );
            }
        }
        Err(LoadError::Malformed(msg)) => {
            println!("  HF safetensors: not found ({msg})");
        }
        Err(err) => {
            println!("  HF safetensors: unreadable ({err})");
        }
    }
    Ok(())
}

fn inspect_int4_bake(cli: &Cli, text: &qwen3_moe::config::TextConfig, prefix: &str) -> Result<()> {
    let specs = int4_contract(text, prefix);
    let inspection = inspect_qwen3_int4_bake(&cli.model_dir, text, prefix).map_err(|e| {
        anyhow!(
            "opening Qwen3-MoE INT4 bake at {}: {e}",
            cli.model_dir.display()
        )
    })?;
    let Int4BakeInspection::Present { bake_dir, report } = inspection else {
        println!(
            "  INT4 bake: missing or outdated at {}",
            inspection.bake_dir().display()
        );
        return Ok(());
    };

    println!("  INT4 bake: present at {}", bake_dir.display());
    println!(
        "    expected tensors present: {}/{}",
        report.present,
        specs.len()
    );
    if report.missing > 0 {
        println!(
            "    missing expected baked tensors: {}{}",
            report.missing,
            format_examples(&report.missing_examples)
        );
    }
    println!(
        "    layouts over expected tensors: int4={} raw={} other={}",
        report.int4_layouts, report.raw_layouts, report.other_layouts
    );
    if report.shape_mismatches > 0 {
        println!(
            "    shape mismatches: {}{}",
            report.shape_mismatches,
            format_examples(&report.shape_examples)
        );
    }
    if report.dtype_mismatches > 0 {
        println!(
            "    dtype mismatches: {}{}",
            report.dtype_mismatches,
            format_examples(&report.dtype_examples)
        );
    }
    if report.layout_mismatches > 0 {
        println!(
            "    layout mismatches: {}{}",
            report.layout_mismatches,
            format_examples(&report.layout_examples)
        );
    }
    println!(
        "    expected tensor bytes in bake: {:.2} GiB",
        report.bytes as f64 / 1024.0_f64.powi(3)
    );
    Ok(())
}

#[derive(Default)]
struct LoaderReport {
    present: usize,
    missing: usize,
    shape_mismatches: usize,
    dtype_mismatches: usize,
    bytes: u64,
    examples: Vec<String>,
    shape_examples: Vec<String>,
    dtype_examples: Vec<String>,
}

fn inspect_loader_specs(
    loader: &WeightLoader,
    text: &qwen3_moe::config::TextConfig,
    specs: &[TensorSpec],
) -> Result<LoaderReport> {
    let mut report = LoaderReport::default();
    for spec in specs {
        let Ok(meta) = loader.meta(&spec.name) else {
            report.missing += 1;
            push_example(&mut report.examples, spec.name.clone());
            continue;
        };
        report.present += 1;
        report.bytes = report.bytes.saturating_add(meta.byte_size());

        let expected_elems = checkpoint_elems_for(text, spec.role);
        if meta.elem_count() != expected_elems {
            report.shape_mismatches += 1;
            push_example(
                &mut report.shape_examples,
                format!(
                    "{} has {:?} ({} elems), expected {} elems",
                    spec.name,
                    meta.shape,
                    meta.elem_count(),
                    expected_elems
                ),
            );
        }

        let Some(dtype) = checkpoint_dtype_from_scalar(meta.dtype) else {
            report.dtype_mismatches += 1;
            push_example(
                &mut report.dtype_examples,
                format!("{} has unsupported dtype {:?}", spec.name, meta.dtype),
            );
            continue;
        };
        if !checkpoint_dtype_acceptable(spec.role, dtype) {
            report.dtype_mismatches += 1;
            push_example(
                &mut report.dtype_examples,
                format!("{} has dtype {:?}", spec.name, meta.dtype),
            );
        }
    }
    Ok(report)
}

fn checkpoint_dtype_from_scalar(dtype: ScalarKind) -> Option<CheckpointDtype> {
    match dtype {
        ScalarKind::Bf16 => Some(CheckpointDtype::Bf16),
        ScalarKind::F32 => Some(CheckpointDtype::F32),
        _ => None,
    }
}

fn push_example(examples: &mut Vec<String>, value: String) {
    if examples.len() < 3 {
        examples.push(value);
    }
}

fn format_examples(examples: &[String]) -> String {
    if examples.is_empty() {
        String::new()
    } else {
        format!(" (examples: {})", examples.join(", "))
    }
}

fn validate_config_matches_registry(
    text: &qwen3_moe::config::TextConfig,
    params: &crate::registry::Qwen3MoeKernelParams,
) -> Result<()> {
    let checks = [
        ("hidden_size", text.hidden_size as u32, params.hidden_size),
        ("vocab_size", text.vocab_size as u32, params.vocab_size),
        (
            "num_hidden_layers",
            text.num_hidden_layers as u32,
            params.num_layers,
        ),
        (
            "num_attention_heads",
            text.num_attention_heads as u32,
            params.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            text.num_key_value_heads as u32,
            params.num_kv_heads,
        ),
        ("head_dim", text.head_dim as u32, params.head_dim),
        ("num_experts", text.num_experts as u32, params.num_experts),
        (
            "num_experts_per_tok",
            text.num_experts_per_tok as u32,
            params.top_k,
        ),
        (
            "moe_intermediate_size",
            text.moe_intermediate_size as u32,
            params.moe_intermediate_size,
        ),
    ];
    for (name, got, expected) in checks {
        if got != expected {
            anyhow::bail!(
                "Qwen3-MoE registry/config mismatch for {name}: config has {got}, registry expects {expected}"
            );
        }
    }
    Ok(())
}
