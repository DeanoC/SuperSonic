use std::sync::Arc;

use kernel_ffi::qwen36_moe::Qwen36MoeLaunchOptions;
use qwen35::state::KvFp8SidecarOptions;
use supersonic_runtime::qwen36_moe::decode::Qwen36ExecutionOptions;
use supersonic_runtime::qwen36_moe::layer_loader::Qwen36LoadOptions;

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn value_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(default)
}

fn parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
}

pub(crate) fn execution_options_from_environment() -> Qwen36ExecutionOptions {
    let force_host_native = flag("SUPERSONIC_METAL_FORCE_HOST_NATIVE");
    let mut options = Qwen36ExecutionOptions::default();

    options.batched_prefill.attention = value_bool("SUPERSONIC_QWEN36_MOE_BATCHED_ATTN", true);
    options.batched_prefill.grouped_ffn = value_bool("SUPERSONIC_QWEN36_MOE_GROUPED_FFN", true);
    options.batched_prefill.metal_split_qgate =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_SPLIT_QGATE", false);
    options.batched_prefill.metal_full_attn_tmajor =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_TMAJOR", false);
    options.batched_prefill.metal_full_attn_vec =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_VEC", true);
    options.batched_prefill.metal_linear_prefill_direct =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_LINEAR_PREFILL_DIRECT", true);
    options.batched_prefill.metal_router_topk =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_ROUTER_TOPK", false);
    options.batched_prefill.metal_shared_expert_batch =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_SHARED_EXPERT_BATCH", true);
    options.batched_prefill.metal_fused_ffn_residual =
        value_bool("SUPERSONIC_QWEN36_MOE_METAL_FUSED_FFN_RESIDUAL", true);

    options.metal.force_host_native = force_host_native;
    options.metal.profile = flag("SUPERSONIC_METAL_PROFILE");
    options.metal.disable_batch = flag("SUPERSONIC_METAL_DISABLE_BATCH");
    options.metal.disable_linear_decode_direct =
        flag("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT");
    options.metal.disable_linear_int4_stage5 =
        flag("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5");
    options.metal.disable_full_attn_decode_direct =
        flag("SUPERSONIC_METAL_DISABLE_QWEN36_FULL_ATTN_DECODE_DIRECT");
    options.metal.disable_decode_batch = flag("SUPERSONIC_METAL_QWEN36_DISABLE_DECODE_BATCH");
    options.metal.profile_ffn_phases = flag("SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES");
    options.metal.decode_batch_profile_phases =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES");
    options.metal.decode_batch_profile_phases_deferred =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED");
    options.metal.decode_batch_sync_phases =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SYNC_PHASES");
    options.metal.decode_batch_ffn_commit_interval =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL");
    options.metal.decode_batch_profile_ffn_phases =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES");
    options.metal.decode_batch_route_snapshot =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT");
    options.metal.decode_batch_shared_stage5_parity_tap =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP")
            || flag("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP");
    options.metal.decode_batch_routed_stage5_parity_tap =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP");
    options.metal.decode_batch_router_stage5_parity_tap =
        flag("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP");
    options.metal.enable_router_stage5_simd =
        flag("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD");
    options.metal.enable_router_fused_exact =
        flag("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT");
    options.metal.disable_router_stage5_exact_simd =
        flag("SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_STAGE5_EXACT_SIMD");
    options.metal.enable_full_attn_native = flag("SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE");
    options.metal.full_attn_native_max_layer =
        parse("SUPERSONIC_METAL_QWEN36_FULL_ATTN_NATIVE_MAX_LAYER");
    options.metal.full_attn_score_tap = flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_SCORE_TAP");
    options.metal.full_attn_prob_tap = flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_PROB_TAP");
    options.metal.full_attn_exp_tap = flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXP_TAP");
    options.metal.full_attn_denom_tap = flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_DENOM_TAP");

    options.diagnostics.trace_norms = value_bool("SUPERSONIC_QWEN36_DEBUG_TRACE_NORMS", false);
    options.diagnostics.layer_output_tap = flag("SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP");
    options.diagnostics.layer_output_delta_tap = flag("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP");
    options.diagnostics.layer_output_delta_tap_position =
        parse("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION");
    options.diagnostics.layer_output_delta_tap_layer =
        parse("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER");
    options.diagnostics.layer_output_delta_tap_phase =
        parse("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE");
    options.diagnostics.full_attn_workspace_tap_words =
        parse("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_WORDS");
    options.diagnostics.full_attn_kv_cache_tap = flag("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP");
    options.diagnostics.full_attn_kv_cache_tap_position =
        parse("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP_POSITION");
    options.diagnostics.full_attn_kv_cache_tap_layer =
        parse("SUPERSONIC_QWEN36_FULL_ATTN_KV_CACHE_TAP_LAYER");
    options.diagnostics.full_attn_workspace_tap = flag("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP");
    options.diagnostics.full_attn_workspace_tap_position =
        parse("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_POSITION");
    options.diagnostics.full_attn_workspace_tap_layer =
        parse("SUPERSONIC_QWEN36_FULL_ATTN_WORKSPACE_TAP_LAYER");
    options.diagnostics.shared_stage5_parity_tap_max_calls =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_MAX_CALLS")
            .or_else(|| parse("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS"));
    options.diagnostics.shared_stage5_parity_tap_position =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_POSITION")
            .or_else(|| parse("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_POSITION"));
    options.diagnostics.shared_stage5_parity_tap_layer =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_LAYER")
            .or_else(|| parse("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_LAYER"));
    options.diagnostics.router_stage5_parity_tap_max_calls =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP_MAX_CALLS");
    options.diagnostics.router_stage5_parity_tap_position =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP_POSITION");
    options.diagnostics.router_stage5_parity_tap_layer =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP_LAYER");
    options.diagnostics.routed_stage5_parity_tap_max_calls =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_MAX_CALLS");
    options.diagnostics.routed_stage5_parity_tap_position =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_POSITION");
    options.diagnostics.routed_stage5_parity_tap_layer =
        parse("SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_LAYER");
    options.diagnostics.route_profile = flag("SUPERSONIC_QWEN36_ROUTE_PROFILE");
    options.diagnostics.ffn_stage_profile = flag("SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES");
    options.diagnostics.linear_stage_profile =
        flag("SUPERSONIC_METAL_PROFILE_QWEN36_LINEAR_PHASES");

    options.kernel_launch = kernel_launch_options_from_environment(force_host_native);
    options.prefill_kernel = kernel_ffi::prefill_ffi::PrefillFfiLaunchOptions {
        force_host_native,
        force_host_rms_norm: flag("SUPERSONIC_METAL_FORCE_HOST_RMS_NORM"),
        force_host_matmul: flag("SUPERSONIC_METAL_FORCE_HOST_MATMUL"),
        force_host_element_add: flag("SUPERSONIC_METAL_FORCE_HOST_ELEMENT_ADD"),
        force_host_cast: flag("SUPERSONIC_METAL_FORCE_HOST_CAST"),
        force_host_transpose_shd_hsd: flag("SUPERSONIC_METAL_FORCE_HOST_TRANSPOSE_SHD_HSD"),
        force_host_split_qgate: flag("SUPERSONIC_METAL_FORCE_HOST_SPLIT_QGATE"),
        disable_gemv_m1: flag("SUPERSONIC_METAL_DISABLE_GEMV_M1"),
        disable_gemv_m1_tiled: flag("SUPERSONIC_METAL_DISABLE_GEMV_M1_TILED"),
        disable_int4_gemv_m1: flag("SUPERSONIC_METAL_DISABLE_INT4_GEMV_M1"),
        disable_int4_gemv_m1_tiled: flag("SUPERSONIC_METAL_DISABLE_INT4_GEMV_M1_TILED"),
        metal_profile: flag("SUPERSONIC_METAL_PROFILE"),
        ffi_profile_shapes: flag("SUPERSONIC_DFLASH_PROFILE_FFI_SHAPES"),
    };
    options.with_diagnostic_observer(Arc::new(|message| eprintln!("{message}")))
}

fn kernel_launch_options_from_environment(force_host_native: bool) -> Qwen36MoeLaunchOptions {
    Qwen36MoeLaunchOptions {
        force_host_native,
        full_attn_online: flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_ONLINE"),
        enable_full_attn_native: flag("SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_NATIVE"),
        full_attn_score_tap: flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_SCORE_TAP"),
        full_attn_prob_tap: flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_PROB_TAP"),
        full_attn_exp_tap: flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_EXP_TAP"),
        full_attn_denom_tap: flag("SUPERSONIC_METAL_QWEN36_FULL_ATTN_DENOM_TAP"),
        disable_linear_int4_stage5: flag("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5"),
        enable_ffn_int4_stage5: flag("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5"),
        enable_ffn_int4_stage5_router: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER",
        ),
        defer_ffn_router_stage5_wait: flag("SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT"),
        sync_ffn_router_stage5_wait: flag("SUPERSONIC_METAL_QWEN36_SYNC_FFN_ROUTER_STAGE5_WAIT"),
        ffn_router_stage5_parity_tap: flag("SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP"),
        ffn_router_stage5_parity_tap_layer: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_LAYER",
        ),
        ffn_router_stage5_parity_tap_max_calls: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_MAX_CALLS",
        ),
        ffn_shared_stage5_parity_tap: flag("SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP"),
        ffn_shared_stage5_parity_tap_max_calls: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS",
        ),
        ffn_stage5_shared_host_correction: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION",
        ),
        ffn_stage5_shared_mid_host_correction: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_MID_HOST_CORRECTION",
        ),
        ffn_stage5_routed_host_correction: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_HOST_CORRECTION",
        ),
        ffn_stage5_routed_down_host_recompute_correction: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_DOWN_HOST_RECOMPUTE_CORRECTION",
        ),
        ffn_stage5_routed_down_host_recompute_correction_layer: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_DOWN_HOST_RECOMPUTE_CORRECTION_LAYER",
        ),
        ffn_stage5_residual_host_snap: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_RESIDUAL_HOST_SNAP",
        ),
        ffn_stage5_residual_host_snap_layer: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_RESIDUAL_HOST_SNAP_LAYER",
        ),
        ffn_stage5_residual_host_snap_row: parse(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_RESIDUAL_HOST_SNAP_ROW",
        ),
        ffn_stage5_routed_gate_up_tap: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_GATE_UP_TAP",
        ),
        ffn_stage5_routed_silu_tap: flag("SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_SILU_TAP"),
        ffn_stage5_routed_finalize_tap: flag(
            "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_FINALIZE_TAP",
        ),
        ffn_stage5_routed_tap_layer: parse("SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_TAP_LAYER"),
        enable_ffn_router_stage5_simd: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD",
        ),
        enable_ffn_router_fused_exact: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT",
        ),
        disable_ffn_router_stage5_exact_simd: flag(
            "SUPERSONIC_METAL_DISABLE_QWEN36_FFN_ROUTER_STAGE5_EXACT_SIMD",
        ),
        enable_ffn_shared_tiled: flag("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"),
        enable_ffn_shared_gate_up_tiled: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_TILED",
        ),
        enable_ffn_shared_gate_up_exp2: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_EXP2",
        ),
        disable_ffn_shared_gate_up_exact_simd: flag(
            "SUPERSONIC_METAL_DISABLE_QWEN36_FFN_SHARED_GATE_UP_EXACT_SIMD",
        ),
        enable_ffn_shared_scalar_simd: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_SIMD",
        ),
        enable_ffn_shared_scalar_exact_simd: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_EXACT_SIMD",
        ),
        enable_ffn_shared_down_tiled: flag("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_TILED"),
        enable_ffn_shared_down_exact_simd: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_EXACT_SIMD",
        ),
        defer_ffn_direct_gather_stage5_wait: flag(
            "SUPERSONIC_METAL_QWEN36_DEFER_FFN_DIRECT_GATHER_STAGE5_WAIT",
        ),
        enable_ffn_expert_gate_up_tiled: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GATE_UP_TILED",
        ),
        enable_ffn_expert_tiled_stage5: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5",
        ),
        enable_ffn_expert_direct_gather_stage5: flag(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5",
        ),
        profile: flag("SUPERSONIC_METAL_PROFILE"),
        route_profile: flag("SUPERSONIC_QWEN36_ROUTE_PROFILE"),
        ..Qwen36MoeLaunchOptions::default()
    }
}

pub(crate) fn load_options_from_environment() -> Qwen36LoadOptions {
    Qwen36LoadOptions::default()
        .with_registered_mmap_upload(value_bool("SUPERSONIC_FLM_REGISTERED_UPLOAD", false))
        .with_kv_fp8_sidecar(KvFp8SidecarOptions {
            enabled: !flag("SUPERSONIC_DEBUG_DISABLE_KV_FP8_BF16_SIDECAR"),
            window_tokens: parse("SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW"),
        })
        .with_diagnostic_observer(Arc::new(|message| eprintln!("{message}")))
}
