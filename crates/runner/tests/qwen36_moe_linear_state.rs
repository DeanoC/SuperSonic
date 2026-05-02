//! Phase 6.4c.1 GPU integration test for linear-attn state snapshot /
//! refresh / restore.
//!
//! Builds a minimal `LayerBuffers` slice (no real weights — just the
//! state buffers we want to snapshot), seeds the live state with
//! known patterns, takes a snapshot, mutates the live state, then
//! restores and checks the live state matches the original snapshot
//! byte-for-byte.
//!
//! Mirrors the engine's intended use: snapshot before speculative
//! verify chains run, restore on rejection. This test doesn't run any
//! kernels — just exercises the D2D copy paths in
//! `runner::qwen36_moe_state`.
//!
//! Skipped silently when HIP isn't compiled.

use anyhow::Result;
use gpu_hal::{copy_d2h, is_backend_compiled, set_backend, Backend, GpuBuffer, ScalarType};
use runner::qwen36_moe_decode::{
    AttnLayerBuffers, FfnLayerBuffers, FullAttnInt4Sidecars, FullAttnKvCache, LayerBuffers,
};
use runner::qwen36_moe_state::{
    refresh_linear_attn_state, restore_linear_attn_state, save_linear_attn_state,
};

fn stub_bf16(ordinal: usize) -> Result<GpuBuffer> {
    Ok(GpuBuffer::zeros(ordinal, ScalarType::BF16, &[1])?)
}
fn stub_u8(ordinal: usize) -> Result<GpuBuffer> {
    Ok(GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?)
}

/// Build a minimal Linear-attn LayerBuffers using small dummy
/// shapes. Conv state and recurrent state are the only fields the
/// snapshot helpers touch — every other field gets a 1-element
/// stub buffer to satisfy the struct shape.
fn make_linear_layer(
    ordinal: usize,
    conv_bytes: &[u8],
    recurrent_floats: &[f32],
) -> Result<LayerBuffers> {
    let conv_state = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::BF16,
        &[conv_bytes.len() / 2],
        conv_bytes,
    )?;
    let mut rec_bytes: Vec<u8> = Vec::with_capacity(recurrent_floats.len() * 4);
    for &f in recurrent_floats {
        rec_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let recurrent_state = GpuBuffer::from_host_bytes(
        ordinal,
        ScalarType::F32,
        &[recurrent_floats.len()],
        &rec_bytes,
    )?;

    Ok(LayerBuffers {
        attn: AttnLayerBuffers::Linear {
            input_norm_w: stub_bf16(ordinal)?,
            in_proj_qkv_w: stub_u8(ordinal)?,
            in_proj_z_w: stub_u8(ordinal)?,
            in_proj_a_w: stub_bf16(ordinal)?,
            in_proj_b_w: stub_bf16(ordinal)?,
            conv1d_w: stub_bf16(ordinal)?,
            conv1d_bias: None,
            dt_bias: stub_bf16(ordinal)?,
            a_log: stub_bf16(ordinal)?,
            norm_w: stub_bf16(ordinal)?,
            out_proj_w: stub_u8(ordinal)?,
            conv_state,
            recurrent_state,
            int4: None,
        },
        ffn: FfnLayerBuffers {
            post_attn_norm_w: stub_bf16(ordinal)?,
            gate_w: stub_bf16(ordinal)?,
            gate_up_proj_w: stub_u8(ordinal)?,
            down_proj_w: stub_u8(ordinal)?,
            shared_gate_proj_w: stub_u8(ordinal)?,
            shared_up_proj_w: stub_u8(ordinal)?,
            shared_down_proj_w: stub_u8(ordinal)?,
            shared_expert_gate_w: stub_bf16(ordinal)?,
            int4: None,
        },
    })
}

/// Build a minimal Full-attn LayerBuffers (used to verify the
/// snapshot's None-slot handling).
fn make_full_layer(ordinal: usize) -> Result<LayerBuffers> {
    Ok(LayerBuffers {
        attn: AttnLayerBuffers::Full {
            input_norm_w: stub_bf16(ordinal)?,
            q_proj_w: stub_u8(ordinal)?,
            k_proj_w: stub_u8(ordinal)?,
            v_proj_w: stub_u8(ordinal)?,
            q_norm_w: stub_bf16(ordinal)?,
            k_norm_w: stub_bf16(ordinal)?,
            o_proj_w: stub_u8(ordinal)?,
            int4: Some(FullAttnInt4Sidecars {
                group_size: 128,
                q_proj_scale: stub_bf16(ordinal)?,
                q_proj_zero: stub_bf16(ordinal)?,
                k_proj_scale: stub_bf16(ordinal)?,
                k_proj_zero: stub_bf16(ordinal)?,
                v_proj_scale: stub_bf16(ordinal)?,
                v_proj_zero: stub_bf16(ordinal)?,
                o_proj_scale: stub_bf16(ordinal)?,
                o_proj_zero: stub_bf16(ordinal)?,
            }),
            kv_cache: Some(FullAttnKvCache {
                k: stub_bf16(ordinal)?,
                v: stub_bf16(ordinal)?,
                kv_max_t: 1,
            }),
        },
        ffn: FfnLayerBuffers {
            post_attn_norm_w: stub_bf16(ordinal)?,
            gate_w: stub_bf16(ordinal)?,
            gate_up_proj_w: stub_u8(ordinal)?,
            down_proj_w: stub_u8(ordinal)?,
            shared_gate_proj_w: stub_u8(ordinal)?,
            shared_up_proj_w: stub_u8(ordinal)?,
            shared_down_proj_w: stub_u8(ordinal)?,
            shared_expert_gate_w: stub_bf16(ordinal)?,
            int4: None,
        },
    })
}

fn d2h_bytes(buf: &GpuBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buf.len_bytes()];
    copy_d2h(0usize, out.as_mut_ptr() as *mut _, buf.as_ptr(), out.len())
        .expect("d2h");
    out
}

fn live_conv_bytes(layer: &LayerBuffers) -> Vec<u8> {
    match &layer.attn {
        AttnLayerBuffers::Linear { conv_state, .. } => d2h_bytes(conv_state),
        AttnLayerBuffers::Full { .. } => Vec::new(),
    }
}
fn live_rec_bytes(layer: &LayerBuffers) -> Vec<u8> {
    match &layer.attn {
        AttnLayerBuffers::Linear { recurrent_state, .. } => d2h_bytes(recurrent_state),
        AttnLayerBuffers::Full { .. } => Vec::new(),
    }
}

/// Mutate the live state of a Linear layer to a known different
/// pattern. Used between save and restore to confirm restore
/// actually overwrites with the snapshot's contents.
fn mutate_linear_state(layer: &mut LayerBuffers, byte_fill: u8, float_fill: f32) {
    if let AttnLayerBuffers::Linear {
        conv_state,
        recurrent_state,
        ..
    } = &mut layer.attn
    {
        let conv_bytes = vec![byte_fill; conv_state.len_bytes()];
        gpu_hal::copy_h2d(
            0usize,
            conv_state.as_mut_ptr(),
            conv_bytes.as_ptr() as *const _,
            conv_bytes.len(),
        )
        .expect("h2d mutated conv");
        let n_rec = recurrent_state.len_bytes() / 4;
        let mut rec_bytes: Vec<u8> = Vec::with_capacity(n_rec * 4);
        for _ in 0..n_rec {
            rec_bytes.extend_from_slice(&float_fill.to_le_bytes());
        }
        gpu_hal::copy_h2d(
            0usize,
            recurrent_state.as_mut_ptr(),
            rec_bytes.as_ptr() as *const _,
            rec_bytes.len(),
        )
        .expect("h2d mutated rec");
    }
}

#[test]
fn linear_attn_state_save_restore_roundtrip() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    set_backend(Backend::Hip);
    let ordinal = 0usize;

    // Two linear layers with distinct seeded state, plus a Full
    // layer in the middle to verify None-slot handling.
    let conv0: Vec<u8> = (0..128).map(|i| (i as u8) ^ 0xA5).collect();
    let rec0: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125).collect();
    let conv1: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(7)).collect();
    let rec1: Vec<f32> = (0..32).map(|i| (i as f32) * -0.5 + 1.0).collect();

    let l0 = make_linear_layer(ordinal, &conv0, &rec0).expect("linear layer 0");
    let l_full = make_full_layer(ordinal).expect("full layer");
    let l2 = make_linear_layer(ordinal, &conv1, &rec1).expect("linear layer 2");
    let mut layers: Vec<LayerBuffers> = vec![l0, l_full, l2];

    // ---- Initial save ----
    let snap = save_linear_attn_state(ordinal, &layers).expect("save snapshot");
    assert_eq!(
        snap.linear_layer_count(), 2,
        "snapshot should capture exactly 2 Linear layers (full layer is None)"
    );
    assert_eq!(snap.layers.len(), 3, "snapshot has one slot per source layer");
    assert!(snap.layers[0].is_some(), "linear layer 0 captured");
    assert!(snap.layers[1].is_none(), "full layer => None slot");
    assert!(snap.layers[2].is_some(), "linear layer 2 captured");

    // ---- Capture a baseline of live bytes for comparison ----
    let pre_l0_conv = live_conv_bytes(&layers[0]);
    let pre_l0_rec = live_rec_bytes(&layers[0]);
    let pre_l2_conv = live_conv_bytes(&layers[2]);
    let pre_l2_rec = live_rec_bytes(&layers[2]);
    assert_eq!(pre_l0_conv.len(), conv0.len());
    assert_eq!(pre_l0_rec.len(), rec0.len() * 4);

    // ---- Mutate live state away from the snapshot ----
    mutate_linear_state(&mut layers[0], 0xFF, 99.0);
    mutate_linear_state(&mut layers[2], 0x11, -7.5);

    let mid_l0_conv = live_conv_bytes(&layers[0]);
    let mid_l2_rec = live_rec_bytes(&layers[2]);
    assert_ne!(
        mid_l0_conv, pre_l0_conv,
        "mutation should have changed live conv_state"
    );
    assert_ne!(
        mid_l2_rec, pre_l2_rec,
        "mutation should have changed live recurrent_state"
    );

    // ---- Restore from snapshot ----
    restore_linear_attn_state(ordinal, &mut layers, &snap).expect("restore");

    let post_l0_conv = live_conv_bytes(&layers[0]);
    let post_l0_rec = live_rec_bytes(&layers[0]);
    let post_l2_conv = live_conv_bytes(&layers[2]);
    let post_l2_rec = live_rec_bytes(&layers[2]);

    assert_eq!(post_l0_conv, pre_l0_conv, "restored conv0 matches pre-mutate");
    assert_eq!(post_l0_rec, pre_l0_rec, "restored rec0 matches pre-mutate");
    assert_eq!(post_l2_conv, pre_l2_conv, "restored conv2 matches pre-mutate");
    assert_eq!(post_l2_rec, pre_l2_rec, "restored rec2 matches pre-mutate");

    eprintln!(
        "[linear-state] save+restore roundtrip green over {} linear layers (skipping {} Full layer)",
        snap.linear_layer_count(), snap.layers.len() - snap.linear_layer_count()
    );
}

#[test]
fn linear_attn_state_refresh_in_place_reuses_buffers() {
    if !is_backend_compiled(Backend::Hip) {
        eprintln!("skip: HIP backend not compiled");
        return;
    }
    set_backend(Backend::Hip);
    let ordinal = 0usize;

    let conv0: Vec<u8> = (0..64).collect();
    let rec0: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let l0 = make_linear_layer(ordinal, &conv0, &rec0).expect("linear layer");
    let mut layers = vec![l0];

    let mut snap = save_linear_attn_state(ordinal, &layers).expect("save");
    let snap_conv_ptr_before = match &snap.layers[0] {
        Some(s) => s.conv_state.as_ptr() as usize,
        None => panic!("expected snapshot for linear layer 0"),
    };

    // Mutate live state to a NEW pattern, then refresh the snapshot
    // (NOT save_linear_attn_state, which would re-allocate). The
    // snapshot's underlying GpuBuffer pointer should be unchanged.
    mutate_linear_state(&mut layers[0], 0xCC, 42.0);
    let new_pattern_conv = live_conv_bytes(&layers[0]);
    let new_pattern_rec = live_rec_bytes(&layers[0]);

    refresh_linear_attn_state(ordinal, &layers, &mut snap).expect("refresh");

    let snap_conv_ptr_after = match &snap.layers[0] {
        Some(s) => s.conv_state.as_ptr() as usize,
        None => panic!("expected snapshot for linear layer 0 after refresh"),
    };
    assert_eq!(
        snap_conv_ptr_before, snap_conv_ptr_after,
        "refresh must reuse the snapshot's existing GPU allocation \
         (otherwise it'd be equivalent to save and offer no win)"
    );

    // Mutate live again, then restore — snapshot now has the second
    // pattern, NOT the first.
    mutate_linear_state(&mut layers[0], 0x00, 0.0);
    restore_linear_attn_state(ordinal, &mut layers, &snap).expect("restore");

    let restored_conv = live_conv_bytes(&layers[0]);
    let restored_rec = live_rec_bytes(&layers[0]);
    assert_eq!(
        restored_conv, new_pattern_conv,
        "restore should have brought back the refreshed (second) pattern"
    );
    assert_eq!(
        restored_rec, new_pattern_rec,
        "restore should have brought back the refreshed recurrent state"
    );
}
