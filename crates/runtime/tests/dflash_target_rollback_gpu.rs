//! DFlash2 target fast-rollback parity against restore-plus-replay.

use std::path::PathBuf;

use gpu_hal::ScalarType;
use qwen38::gguf_ingest::load_text_config;
use qwen38::scratch::required_attn_scratch_floats;
use qwen38::weights::Qwen38Weights;
use supersonic_runtime::decode_engine::DecodeEngine;
use supersonic_runtime::dflash_spec::{DflashSpecDecoder, DflashVerifyPath};
use supersonic_runtime::prefill_engine::{DflashRollbackCapture, DflashTargetCapture};

fn require_artifacts() -> bool {
    std::env::var("SUPERSONIC_REQUIRE_GQH_ARTIFACTS").as_deref() == Ok("1")
}

fn artifact_path(name: &str) -> Option<PathBuf> {
    let Some(value) = std::env::var_os(name) else {
        if require_artifacts() {
            panic!("{name} is required for the DFlash rollback artifact test");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_file() {
        Some(path)
    } else if require_artifacts() {
        panic!("{name} points to a missing artifact: {}", path.display());
    } else {
        None
    }
}

fn model_dir() -> Option<PathBuf> {
    let Some(value) = std::env::var_os("SUPERSONIC_QWEN38_MODEL_DIR") else {
        if require_artifacts() {
            panic!("SUPERSONIC_QWEN38_MODEL_DIR is required for the DFlash rollback artifact test");
        }
        return None;
    };
    let path = PathBuf::from(value);
    if path.is_dir() {
        Some(path)
    } else if require_artifacts() {
        panic!(
            "SUPERSONIC_QWEN38_MODEL_DIR points to a missing directory: {}",
            path.display()
        );
    } else {
        None
    }
}

fn build_engine() -> Option<(DecodeEngine, qwen38::config::TextConfig)> {
    let gguf = artifact_path("SUPERSONIC_GQH_GGUF")?;
    let model = model_dir()?;
    if kernel_ffi::query_gpu_info(0).is_err() {
        if require_artifacts() {
            panic!("HIP device 0 is required for the DFlash rollback artifact test");
        }
        eprintln!("skip: HIP device 0 unavailable");
        return None;
    }
    let config = load_text_config(&model).expect("load Qwen3.8 config");
    let weights = Qwen38Weights::load_gguf(&gguf, &config, 0).expect("load Qwen3.8 GQH GGUF");
    let scratch =
        required_attn_scratch_floats(config.num_attention_heads, config.head_dim, 256, 256)
            .max(262_144);
    let engine =
        DecodeEngine::new(weights, 0, 16_480, scratch, 256, true, 128).expect("build DecodeEngine");
    Some((engine, config))
}

fn linear_state_bytes(engine: &DecodeEngine) -> Vec<(Vec<u8>, Vec<u8>)> {
    engine
        .state()
        .layers
        .iter()
        .filter(|layer| layer.conv_state.is_some() && layer.recurrent_state.is_some())
        .map(|layer| {
            let conv = layer
                .conv_state
                .as_ref()
                .expect("linear layer conv state")
                .to_host_bytes()
                .expect("download conv state");
            let recurrent = layer
                .recurrent_state
                .as_ref()
                .expect("linear layer recurrent state")
                .to_host_bytes()
                .expect("download recurrent state");
            (conv, recurrent)
        })
        .collect()
}

fn assert_recurrent_state_close(fast: &[u8], replay: &[u8], layer: usize) {
    assert_eq!(fast.len(), replay.len(), "recurrent layer {layer} length");
    for (offset, (fast_chunk, replay_chunk)) in
        fast.chunks_exact(4).zip(replay.chunks_exact(4)).enumerate()
    {
        let fast = f32::from_le_bytes(fast_chunk.try_into().expect("fast f32"));
        let replay = f32::from_le_bytes(replay_chunk.try_into().expect("replay f32"));
        let tolerance = 1e-2 + 1e-2 * replay.abs();
        assert!(
            (fast - replay).abs() <= tolerance,
            "recurrent layer {layer} f32[{offset}] differs beyond F16 precision: fast={fast} replay={replay}"
        );
    }
}

#[test]
fn dflash_fast_rollback_matches_restore_replay_state() {
    let Some((mut engine, config)) = build_engine() else {
        return;
    };
    let prompt = vec![1_u32, 2, 3, 4];
    let block: Vec<u32> = (10..26).collect();
    let commit_len = 3_usize;
    let pos = prompt.len();

    engine.prefill_native(&prompt).expect("prefill prompt");
    engine
        .snapshot_linear_for_spec()
        .expect("snapshot linear state");

    let target_layer_ids = (1..config.num_hidden_layers)
        .step_by((config.num_hidden_layers - 2) / 4)
        .take(5)
        .collect::<Vec<_>>();
    let mut target = DflashTargetCapture::new(
        0,
        256,
        target_layer_ids.len(),
        config.hidden_size,
        target_layer_ids,
    )
    .expect("allocate target capture");
    let mut rollback =
        DflashRollbackCapture::new(&config, block.len(), 0).expect("allocate rollback capture");

    engine
        .verify_block_dflash_with_rollback(&block, pos, &mut target, &mut rollback)
        .expect("verify with rollback capture");
    engine
        .rollback_dflash_prefix(&rollback, commit_len)
        .expect("fast rollback accepted prefix");
    let fast_state = linear_state_bytes(&engine);
    engine
        .replay_committed_prefix_dflash(&block, commit_len, pos, &mut target, &mut rollback)
        .expect("restore and replay accepted prefix");
    let replay_state = linear_state_bytes(&engine);
    assert_eq!(fast_state.len(), replay_state.len());
    for (layer, ((fast_conv, fast_rec), (replay_conv, replay_rec))) in
        fast_state.iter().zip(&replay_state).enumerate()
    {
        let (conv_max_abs, conv_rel_l2) = bf16_state_error(fast_conv, replay_conv);
        let (recurrent_max_abs, recurrent_rel_l2) = recurrent_state_error(fast_rec, replay_rec);
        assert!(
            conv_rel_l2 <= 0.30,
            "rollback conv state differs beyond capture/replay numerics for linear layer {layer}: max_abs={conv_max_abs} rel_l2={conv_rel_l2}"
        );
        assert!(
            recurrent_rel_l2 <= 0.30,
            "rollback recurrent state differs beyond capture/replay numerics for linear layer {layer}: max_abs={recurrent_max_abs} rel_l2={recurrent_rel_l2}"
        );
    }
    assert_eq!(
        rollback.token_count(),
        block.len(),
        "rollback capture must cover every verify token"
    );
    assert_eq!(
        rollback.capture_dtype(),
        ScalarType::F16,
        "recurrent intermediates must match upstream F16 checkpoints"
    );
}

#[test]
fn dflash_short_fast_rollback_matches_restore_replay_state() {
    let Some((mut engine, config)) = build_engine() else {
        return;
    };
    let prompt = vec![1_u32, 2, 3, 4];
    let block: Vec<u32> = (10..26).collect();
    let commit_len = 1_usize;
    let pos = prompt.len();

    engine.prefill_native(&prompt).expect("prefill prompt");
    engine
        .snapshot_linear_for_spec()
        .expect("snapshot linear state");
    let target_layer_ids = (1..config.num_hidden_layers)
        .step_by((config.num_hidden_layers - 2) / 4)
        .take(5)
        .collect::<Vec<_>>();
    let mut target = DflashTargetCapture::new(
        0,
        256,
        target_layer_ids.len(),
        config.hidden_size,
        target_layer_ids,
    )
    .expect("allocate target capture");
    let mut rollback =
        DflashRollbackCapture::new(&config, block.len(), 0).expect("allocate rollback capture");

    engine
        .verify_block_dflash_with_rollback(&block, pos, &mut target, &mut rollback)
        .expect("verify with rollback capture");
    engine
        .rollback_dflash_prefix(&rollback, commit_len)
        .expect("fast rollback one-token prefix");
    let fast_state = linear_state_bytes(&engine);

    engine
        .replay_committed_prefix_dflash(&block, commit_len, pos, &mut target, &mut rollback)
        .expect("restore and replay one-token prefix");
    let replay_state = linear_state_bytes(&engine);

    assert_eq!(fast_state.len(), replay_state.len());
    for (layer, ((fast_conv, fast_rec), (replay_conv, replay_rec))) in
        fast_state.iter().zip(&replay_state).enumerate()
    {
        let (conv_max_abs, conv_rel_l2) = bf16_state_error(fast_conv, replay_conv);
        let (recurrent_max_abs, recurrent_rel_l2) = recurrent_state_error(fast_rec, replay_rec);
        assert!(
            conv_rel_l2 <= 0.30,
            "short rollback conv state differs beyond capture/replay numerics for linear layer {layer}: max_abs={conv_max_abs} rel_l2={conv_rel_l2}"
        );
        assert!(
            recurrent_rel_l2 <= 0.30,
            "short rollback recurrent state differs beyond capture/replay numerics for linear layer {layer}: max_abs={recurrent_max_abs} rel_l2={recurrent_rel_l2}"
        );
    }
}

struct I8DotGuard(Option<String>);

impl Drop for I8DotGuard {
    fn drop(&mut self) {
        match self.0.as_deref() {
            Some("0") => std::env::set_var("GGML_GQH_I8DOT", "0"),
            Some("1") => std::env::set_var("GGML_GQH_I8DOT", "1"),
            Some(value) => std::env::set_var("GGML_GQH_I8DOT", value),
            None => std::env::remove_var("GGML_GQH_I8DOT"),
        }
    }
}

#[test]
fn dflash_component_batch_matches_sequential_tokens() {
    let i8_guard = I8DotGuard(std::env::var("GGML_GQH_I8DOT").ok());
    std::env::set_var("GGML_GQH_I8DOT", "0");
    let Some((mut engine, config)) = build_engine() else {
        return;
    };
    let prompt = vec![1_u32, 2, 3, 4];
    let block = vec![10_u32, 11];
    let pos = prompt.len();

    engine.prefill_native(&prompt).expect("prefill prompt");
    engine
        .snapshot_linear_for_spec()
        .expect("snapshot linear state");

    let target_layer_ids = (1..config.num_hidden_layers)
        .step_by((config.num_hidden_layers - 2) / 4)
        .take(5)
        .collect::<Vec<_>>();
    let mut target = DflashTargetCapture::new(
        0,
        256,
        target_layer_ids.len(),
        config.hidden_size,
        target_layer_ids,
    )
    .expect("allocate target capture");
    let mut rollback =
        DflashRollbackCapture::new(&config, block.len(), 0).expect("allocate rollback capture");

    let batched = engine
        .replay_committed_prefix_dflash(&block, block.len(), pos, &mut target, &mut rollback)
        .expect("batched component reference");
    let batched_greedy = batched.target_next.clone().expect("batched greedy tokens");
    let batched_state = linear_state_bytes(&engine);

    engine
        .replay_committed_prefix_dflash(&block, 1, pos, &mut target, &mut rollback)
        .expect("sequential first token");
    let sequential_second = engine
        .verify_block_dflash_with_rollback(&block[1..], pos + 1, &mut target, &mut rollback)
        .expect("sequential second token");
    let sequential_greedy = sequential_second
        .target_next
        .clone()
        .expect("sequential greedy tokens");
    let sequential_state = linear_state_bytes(&engine);

    let (conv_max_abs, conv_rel_l2) = bf16_state_error(&batched_state[0].0, &sequential_state[0].0);
    let (recurrent_max_abs, recurrent_rel_l2) =
        recurrent_state_error(&batched_state[0].1, &sequential_state[0].1);
    eprintln!(
        "component B=2 vs sequential conv_max_abs={conv_max_abs:.9} conv_rel_l2={conv_rel_l2:.9} recurrent_max_abs={recurrent_max_abs:.9} recurrent_rel_l2={recurrent_rel_l2:.9}"
    );
    assert_eq!(&batched_greedy[1..], &sequential_greedy[..]);
    assert_eq!(batched_state.len(), sequential_state.len());
    for (layer, ((batched_conv, batched_rec), (sequential_conv, sequential_rec))) in
        batched_state.iter().zip(&sequential_state).enumerate()
    {
        let (conv_max_abs, conv_rel_l2) = bf16_state_error(batched_conv, sequential_conv);
        eprintln!(
            "component batch layer {layer} conv_max_abs={conv_max_abs:.9} conv_rel_l2={conv_rel_l2:.9}"
        );
        assert!(
            conv_rel_l2 <= 0.02,
            "component batch conv state differs beyond batch/sequential numerics for linear layer {layer}: max_abs={conv_max_abs} rel_l2={conv_rel_l2}"
        );
        assert_recurrent_state_close(batched_rec, sequential_rec, layer);
    }
    drop(i8_guard);
}

#[test]
fn dflash_component_round_full_width_matches_replay_state() {
    let Some((mut engine, config)) = build_engine() else {
        return;
    };
    let Some(draft_path) = artifact_path("SUPERSONIC_DFLASH_DRAFT_GGUF") else {
        return;
    };
    let draft = model_store::dflash::load_draft(&draft_path)
        .unwrap_or_else(|e| panic!("load DFlash2 drafter: {e}"));
    let draft_gpu = draft
        .upload(0)
        .unwrap_or_else(|e| panic!("upload DFlash2 drafter: {e}"));
    let mut decoder = DflashSpecDecoder::new(
        draft_gpu,
        0,
        256,
        config.hidden_size,
        config.num_hidden_layers,
        &config,
    )
    .unwrap_or_else(|e| panic!("build DFlash2 decoder: {e}"));

    let prompt = vec![1_u32, 2, 3, 4];
    engine.prefill_native(&prompt).expect("prefill prompt");
    let committed = prompt.len();
    let remaining = 16_usize;
    let round = decoder
        .run_round(
            &mut engine,
            *prompt.last().expect("prompt token"),
            committed,
            remaining,
        )
        .expect("DFlash2 generation-limit round");
    assert_eq!(
        round.verify_path,
        DflashVerifyPath::Component,
        "DFlash2 must use the component target verify path"
    );
    assert!(
        !round.emitted.is_empty() && round.emitted.len() <= remaining,
        "DFlash2 round must emit between one and the remaining token budget"
    );

    let fused_state = linear_state_bytes(&engine);
    let target_layer_ids = (1..config.num_hidden_layers)
        .step_by((config.num_hidden_layers - 2) / 4)
        .take(5)
        .collect::<Vec<_>>();
    let mut replay_target = DflashTargetCapture::new(
        0,
        256,
        target_layer_ids.len(),
        config.hidden_size,
        target_layer_ids,
    )
    .expect("allocate full-width replay capture");
    let mut replay_rollback = DflashRollbackCapture::new(&config, round.verified_block.len(), 0)
        .expect("allocate full-width replay rollback");
    let replay = engine
        .replay_committed_prefix_dflash(
            &round.verified_block,
            round.emitted.len(),
            committed,
            &mut replay_target,
            &mut replay_rollback,
        )
        .expect("replay committed prefix at verify width");
    let replay_next = replay
        .target_next
        .as_ref()
        .and_then(|tokens| tokens.get(round.emitted.len() - 1).copied())
        .expect("full-width replay greedy next token");
    let replay_state = linear_state_bytes(&engine);

    assert_eq!(round.next_token, replay_next);
    assert_eq!(fused_state.len(), replay_state.len());
    for (layer, ((fused_conv, fused_rec), (replay_conv, replay_rec))) in
        fused_state.iter().zip(&replay_state).enumerate()
    {
        let (conv_max_abs, conv_rel_l2) = bf16_state_error(fused_conv, replay_conv);
        let (recurrent_max_abs, recurrent_rel_l2) = recurrent_state_error(fused_rec, replay_rec);
        eprintln!(
            "generation-limit layer {layer} conv_max_abs={conv_max_abs:.9} conv_rel_l2={conv_rel_l2:.9} recurrent_max_abs={recurrent_max_abs:.9} recurrent_rel_l2={recurrent_rel_l2:.9}"
        );
        assert!(
            conv_rel_l2 <= 0.02,
            "full-width conv state differs beyond fused/replay numerics for linear layer {layer}: max_abs={conv_max_abs} rel_l2={conv_rel_l2}"
        );
        assert!(
            recurrent_rel_l2 <= 0.02,
            "full-width recurrent state differs beyond fused/replay numerics for linear layer {layer}: max_abs={recurrent_max_abs} rel_l2={recurrent_rel_l2}"
        );
    }
}

fn bf16_state_error(fused: &[u8], replay: &[u8]) -> (f32, f64) {
    let mut max_abs = 0.0_f32;
    let mut sq_error = 0.0_f64;
    let mut sq_replay = 0.0_f64;
    for (fused_chunk, replay_chunk) in fused.chunks_exact(2).zip(replay.chunks_exact(2)) {
        let fused = half::bf16::from_le_bytes(fused_chunk.try_into().expect("fused bf16")).to_f32();
        let replay =
            half::bf16::from_le_bytes(replay_chunk.try_into().expect("replay bf16")).to_f32();
        max_abs = max_abs.max((fused - replay).abs());
        sq_error += f64::from(fused - replay).powi(2);
        sq_replay += f64::from(replay).powi(2);
    }
    (max_abs, (sq_error / sq_replay).sqrt())
}

fn recurrent_state_error(fused: &[u8], replay: &[u8]) -> (f32, f64) {
    let mut max_abs = 0.0_f32;
    let mut sq_error = 0.0_f64;
    let mut sq_replay = 0.0_f64;
    for (fused_chunk, replay_chunk) in fused.chunks_exact(4).zip(replay.chunks_exact(4)) {
        let fused = f32::from_le_bytes(fused_chunk.try_into().expect("fused f32"));
        let replay = f32::from_le_bytes(replay_chunk.try_into().expect("replay f32"));
        max_abs = max_abs.max((fused - replay).abs());
        sq_error += f64::from(fused - replay).powi(2);
        sq_replay += f64::from(replay).powi(2);
    }
    (max_abs, (sq_error / sq_replay).sqrt())
}
