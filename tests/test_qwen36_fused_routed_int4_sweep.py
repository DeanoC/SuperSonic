from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_fused_routed_int4.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_fused_routed_int4", SCRIPT)
sweep_qwen36_fused_routed_int4 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_fused_routed_int4
SPEC.loader.exec_module(sweep_qwen36_fused_routed_int4)


def row(
    mode: str,
    *,
    ids: list[int] | None = None,
    headline: float = 100.0,
    ffn: float = 50.0,
    full: float = 10.0,
    linear: float = 20.0,
    lm_head: float = 5.0,
    wait: float | None = 10.0,
    fused: float | None = None,
    fused_gpu: float | None = None,
    decode_batch_gpu: float | None = None,
    decode_batch_linear_gpu: float | None = None,
    decode_batch_ffn_gpu: float | None = None,
    status: str = "ok",
) -> dict:
    profile = None
    if wait is not None:
        entries = [
            {
                "op": "command_buffer_wait",
                "calls": 1,
                "mean_ms": wait,
                "total_ms": wait,
                "max_ms": wait,
            }
        ]
        if fused is not None:
            entries.append(
                {
                    "op": "qwen36_ffn_int4_expert_direct_gather_stage5",
                    "path": "native",
                    "calls": 1,
                    "mean_ms": fused,
                    "total_ms": fused,
                    "max_ms": fused,
                }
            )
        if fused_gpu is not None:
            entries.append(
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_expert_direct_gather_stage5",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": fused_gpu,
                    "total_ms": fused_gpu,
                    "max_ms": fused_gpu,
                }
            )
        if decode_batch_gpu is not None:
            entries.append(
                {
                    "op": "command_buffer_gpu:qwen36_decode_batch",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": decode_batch_gpu,
                    "total_ms": decode_batch_gpu,
                    "max_ms": decode_batch_gpu,
                }
            )
        if decode_batch_linear_gpu is not None:
            entries.append(
                {
                    "op": "command_buffer_gpu:qwen36_decode_batch_linear_attn",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": decode_batch_linear_gpu,
                    "total_ms": decode_batch_linear_gpu,
                    "max_ms": decode_batch_linear_gpu,
                }
            )
        if decode_batch_ffn_gpu is not None:
            entries.append(
                {
                    "op": "command_buffer_gpu:qwen36_decode_batch_ffn",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": decode_batch_ffn_gpu,
                    "total_ms": decode_batch_ffn_gpu,
                    "max_ms": decode_batch_ffn_gpu,
                }
            )
        profile = {"summary": {"calls": len(entries)}, "entries": entries}
    return {
        "prompt_id": "hello",
        "prompt": "Hello",
        "mode": mode,
        "status": status,
        "generated_ids": [11, 353] if ids is None else ids,
        "result": {"ms_per_step": headline, "decode_ms": headline * 2},
        "stage_timings": {"lm_head_ms_avg": lm_head},
        "chain_breakdown": {
            "ffn_ms_avg": ffn,
            "full_attn_ms_avg": full,
            "linear_attn_ms_avg": linear,
        },
        "metal_profile": profile,
        "hal_profile": None,
        "fused_op_ms": fused,
        "wall_seconds": 1.0,
    }


class Qwen36FusedRoutedInt4SweepTests(unittest.TestCase):
    def test_parse_modes_dedupes_aliases(self):
        script = sweep_qwen36_fused_routed_int4

        self.assertEqual(
            script.parse_modes(
                "baseline,direct,direct-defer,defer-direct-wait,gpu-pack,gpack,stage5,native-stage5,router,router-simd,router-batch,batch-router,router-simd-batch,batch-router-simd,router-simd-batch-shared-tiled,batch-router-simd-shared-tiled,router-simd-batch-shared-gate-up-tiled,router-simd-batch-shared-scalar-simd,router-simd-batch-shared-down-tiled,router-simd-batch-routed-gate-up-tap,router-simd-batch-routed-gate-up-host-order-tap,router-simd-batch-routed-gate-up-host-order,router-simd-batch-shared-host-corrected-routed-gate-up-host-order,router-simd-batch-shared-host-corrected,router-simd-batch-shared-routed-host-corrected,router-batch-deferred-phases,batch-router-deferred-phases,router-simd-batch-deferred-phases,batch-router-simd-deferred-phases,router-simd-batch-shared-tiled-deferred-phases,batch-router-simd-shared-tiled-deferred-phases,router-batch-phases,batch-router-phases,router-batch-ffn-phases,batch-router-ffn-phases,router-simd-batch-phases,router-simd-batch-ffn-phases,router-simd-batch-shared-tiled-ffn-phases,batch-router-simd-shared-tiled-ffn-phases,router-defer"
            ),
            [
                "default",
                "direct-gather",
                "direct-defer-wait",
                "gpu-pack",
                "full-stage5",
                "full-stage5-router",
                "full-stage5-router-simd",
                "full-stage5-router-batch",
                "full-stage5-router-simd-batch",
                "full-stage5-router-simd-batch-shared-tiled",
                "full-stage5-router-simd-batch-shared-gate-up-tiled",
                "full-stage5-router-simd-batch-shared-scalar-simd",
                "full-stage5-router-simd-batch-shared-down-tiled",
                "full-stage5-router-simd-batch-routed-gate-up-tap",
                "full-stage5-router-simd-batch-routed-gate-up-host-order-tap",
                "full-stage5-router-simd-batch-routed-gate-up-host-order",
                "full-stage5-router-simd-batch-shared-host-corrected-routed-gate-up-host-order",
                "full-stage5-router-simd-batch-shared-host-corrected",
                "full-stage5-router-simd-batch-shared-routed-host-corrected",
                "full-stage5-router-batch-deferred-phases",
                "full-stage5-router-simd-batch-deferred-phases",
                "full-stage5-router-simd-batch-shared-tiled-deferred-phases",
                "full-stage5-router-batch-phases",
                "full-stage5-router-batch-ffn-phases",
                "full-stage5-router-simd-batch-phases",
                "full-stage5-router-simd-batch-ffn-phases",
                "full-stage5-router-simd-batch-shared-tiled-ffn-phases",
                "router-defer-wait",
            ],
        )

    def test_build_env_overrides_for_fused_modes(self):
        script = sweep_qwen36_fused_routed_int4
        args = Namespace(metal_profile=True, metal_profile_phases=False)

        direct = script.build_env_overrides(args, "direct-gather")
        direct_defer = script.build_env_overrides(args, "direct-defer-wait")
        gpu_pack = script.build_env_overrides(args, "gpu-pack")
        full_stage5 = script.build_env_overrides(args, "full-stage5")
        full_stage5_router = script.build_env_overrides(args, "full-stage5-router")
        full_stage5_router_simd = script.build_env_overrides(args, "full-stage5-router-simd")
        full_stage5_router_batch = script.build_env_overrides(args, "full-stage5-router-batch")
        full_stage5_router_simd_batch = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch",
        )
        full_stage5_router_simd_batch_shared_tiled = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-tiled",
        )
        full_stage5_router_simd_batch_shared_gate_up_tiled = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-gate-up-tiled",
        )
        full_stage5_router_simd_batch_shared_scalar_simd = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-scalar-simd",
        )
        full_stage5_router_simd_batch_shared_down_tiled = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-down-tiled",
        )
        full_stage5_router_simd_batch_routed_gate_up_tap = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-routed-gate-up-tap",
        )
        full_stage5_router_simd_batch_routed_gate_up_host_order_tap = (
            script.build_env_overrides(
                args,
                "full-stage5-router-simd-batch-routed-gate-up-host-order-tap",
            )
        )
        full_stage5_router_simd_batch_routed_gate_up_host_order = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-routed-gate-up-host-order",
        )
        full_stage5_router_simd_batch_shared_host_corrected_routed_gate_up_host_order = (
            script.build_env_overrides(
                args,
                "full-stage5-router-simd-batch-shared-host-corrected-routed-gate-up-host-order",
            )
        )
        full_stage5_router_simd_batch_shared_host_corrected = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-host-corrected",
        )
        full_stage5_router_simd_batch_shared_routed_host_corrected = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-routed-host-corrected",
        )
        full_stage5_router_batch_deferred = script.build_env_overrides(
            args,
            "full-stage5-router-batch-deferred-phases",
        )
        full_stage5_router_simd_batch_deferred = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-deferred-phases",
        )
        full_stage5_router_simd_batch_shared_tiled_deferred = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-tiled-deferred-phases",
        )
        full_stage5_router_batch_phases = script.build_env_overrides(
            args,
            "full-stage5-router-batch-phases",
        )
        full_stage5_router_batch_ffn_phases = script.build_env_overrides(
            args,
            "full-stage5-router-batch-ffn-phases",
        )
        full_stage5_router_simd_batch_phases = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-phases",
        )
        full_stage5_router_simd_batch_ffn_phases = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-ffn-phases",
        )
        full_stage5_router_simd_batch_shared_tiled_ffn_phases = script.build_env_overrides(
            args,
            "full-stage5-router-simd-batch-shared-tiled-ffn-phases",
        )
        router_defer = script.build_env_overrides(args, "router-defer-wait")
        default = script.build_env_overrides(args, "default")

        self.assertEqual(direct["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertNotIn("SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES", direct)
        self.assertEqual(
            direct["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5"],
            "1",
        )
        self.assertNotIn(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5",
            default,
        )
        self.assertEqual(
            direct_defer["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5"],
            "1",
        )
        self.assertEqual(
            direct_defer["SUPERSONIC_METAL_QWEN36_DEFER_FFN_DIRECT_GATHER_STAGE5_WAIT"],
            "1",
        )
        self.assertEqual(
            gpu_pack["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5"],
            "1",
        )
        self.assertEqual(
            gpu_pack["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5"],
            "1",
        )
        self.assertEqual(
            full_stage5["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5"],
            "1",
        )
        self.assertEqual(
            full_stage5_router["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_gate_up_tiled[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_TILED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_scalar_simd[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_down_tiled[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_TILED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_tap[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_tap[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_GATE_UP_TAP"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_tap[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH"
            ],
            "1",
        )
        self.assertNotIn(
            "SUPERSONIC_METAL_QWEN36_FFN_EXPERT_GATE_UP_HOST_ORDER_STAGE5",
            full_stage5_router_simd_batch_routed_gate_up_tap,
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_host_order_tap[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_GATE_UP_TAP"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_host_order_tap[
                "SUPERSONIC_METAL_QWEN36_FFN_EXPERT_GATE_UP_HOST_ORDER_STAGE5"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_host_order[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_host_order[
                "SUPERSONIC_METAL_QWEN36_FFN_EXPERT_GATE_UP_HOST_ORDER_STAGE5"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_routed_gate_up_host_order[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected_routed_gate_up_host_order[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected_routed_gate_up_host_order[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected_routed_gate_up_host_order[
                "SUPERSONIC_METAL_QWEN36_FFN_EXPERT_GATE_UP_HOST_ORDER_STAGE5"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected_routed_gate_up_host_order[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_host_corrected[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_routed_host_corrected[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_routed_host_corrected[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_routed_host_corrected[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_routed_host_corrected[
                "SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_HOST_CORRECTION"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_deferred[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_deferred["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_deferred[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_deferred[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_deferred[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled_deferred[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled_deferred[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_phases[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_phases["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_phases[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_ffn_phases[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_ffn_phases["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_ffn_phases[
                "SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch_ffn_phases[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_phases[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_phases[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_ffn_phases[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_ffn_phases[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled_ffn_phases[
                "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"
            ],
            "1",
        )
        self.assertEqual(
            full_stage5_router_simd_batch_shared_tiled_ffn_phases[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"
            ],
            "1",
        )
        self.assertEqual(
            router_defer["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"],
            "1",
        )
        self.assertEqual(
            router_defer["SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT"],
            "1",
        )

        phase_args = Namespace(metal_profile=True, metal_profile_phases=True)
        phase_default = script.build_env_overrides(phase_args, "default")
        self.assertEqual(phase_default["SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"], "1")

        downstream_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=True,
            layer_output_tap=False,
        )
        downstream_default = script.build_env_overrides(downstream_args, "default")
        self.assertEqual(
            downstream_default["SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP"],
            "1",
        )

        layer_output_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=True,
            layer_output_delta_tap=True,
            layer_output_delta_all=False,
            layer_output_delta_position=0,
            layer_output_delta_layer=0,
            layer_output_delta_phase="ffn",
        )
        layer_output_default = script.build_env_overrides(layer_output_args, "default")
        self.assertEqual(
            layer_output_default["SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP"],
            "1",
        )
        self.assertEqual(
            layer_output_default["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP"],
            "1",
        )
        self.assertEqual(
            layer_output_default["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION"],
            "0",
        )
        self.assertEqual(
            layer_output_default["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER"],
            "0",
        )
        self.assertEqual(
            layer_output_default["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE"],
            "ffn",
        )
        layer_output_all_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=True,
            layer_output_delta_tap=True,
            layer_output_delta_all=True,
            layer_output_delta_position=0,
            layer_output_delta_layer=0,
            layer_output_delta_phase="ffn",
        )
        layer_output_all = script.build_env_overrides(layer_output_all_args, "default")
        self.assertEqual(layer_output_all["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP"], "1")
        self.assertNotIn("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION", layer_output_all)
        self.assertNotIn("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER", layer_output_all)
        self.assertNotIn("SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE", layer_output_all)

        tap_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=False,
            router_parity_tap=True,
            router_parity_tap_max_calls=3,
            shared_parity_tap=True,
            shared_parity_tap_max_calls=5,
            routed_parity_tap=True,
            routed_parity_tap_max_calls=7,
        )
        tap_router = script.build_env_overrides(tap_args, "full-stage5-router")
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP"],
            "1",
        )
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_MAX_CALLS"],
            "3",
        )
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP"],
            "1",
        )
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP"],
            "1",
        )
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS"],
            "5",
        )
        self.assertEqual(
            tap_router[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_MAX_CALLS"
            ],
            "5",
        )
        self.assertEqual(
            tap_router["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP"],
            "1",
        )
        self.assertEqual(
            tap_router[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_MAX_CALLS"
            ],
            "7",
        )
        snapshot_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            decode_batch_route_snapshot=True,
        )
        snapshot_router = script.build_env_overrides(
            snapshot_args, "full-stage5-router-simd-batch-ffn-phases"
        )
        self.assertEqual(
            snapshot_router["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT"],
            "1",
        )

    def test_build_command_omits_stage_timing_for_batch_mode(self):
        script = sweep_qwen36_fused_routed_int4
        args = Namespace(
            binary=Path("target/release/supersonic"),
            model_dir=Path("/models/qwen3.6-35b-a3b"),
            context_size=64,
            max_new_tokens=4,
            seed=7,
        )

        normal = script.build_command(args, "Hello", "full-stage5-router")
        batch = script.build_command(args, "Hello", "full-stage5-router-batch")
        simd_batch = script.build_command(args, "Hello", "full-stage5-router-simd-batch")
        simd_batch_shared_tiled = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-shared-tiled",
        )
        deferred_batch = script.build_command(
            args,
            "Hello",
            "full-stage5-router-batch-deferred-phases",
        )
        deferred_simd_batch = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-deferred-phases",
        )
        deferred_simd_batch_shared_tiled = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-shared-tiled-deferred-phases",
        )
        batch_phases = script.build_command(args, "Hello", "full-stage5-router-batch-phases")
        batch_ffn_phases = script.build_command(
            args,
            "Hello",
            "full-stage5-router-batch-ffn-phases",
        )
        simd_batch_phases = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-phases",
        )
        simd_batch_ffn_phases = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-ffn-phases",
        )
        simd_batch_shared_tiled_ffn_phases = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-shared-tiled-ffn-phases",
        )
        simd_batch_shared_host_corrected = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-shared-host-corrected",
        )
        simd_batch_shared_routed_host_corrected = script.build_command(
            args,
            "Hello",
            "full-stage5-router-simd-batch-shared-routed-host-corrected",
        )

        self.assertIn("--emit-stage-timings", normal)
        self.assertNotIn("--emit-stage-timings", batch)
        self.assertNotIn("--emit-stage-timings", simd_batch)
        self.assertNotIn("--emit-stage-timings", simd_batch_shared_tiled)
        self.assertNotIn("--emit-stage-timings", deferred_batch)
        self.assertNotIn("--emit-stage-timings", deferred_simd_batch)
        self.assertNotIn("--emit-stage-timings", deferred_simd_batch_shared_tiled)
        self.assertNotIn("--emit-stage-timings", batch_phases)
        self.assertNotIn("--emit-stage-timings", batch_ffn_phases)
        self.assertNotIn("--emit-stage-timings", simd_batch_phases)
        self.assertNotIn("--emit-stage-timings", simd_batch_ffn_phases)
        self.assertNotIn("--emit-stage-timings", simd_batch_shared_tiled_ffn_phases)
        self.assertNotIn("--emit-stage-timings", simd_batch_shared_host_corrected)
        self.assertNotIn("--emit-stage-timings", simd_batch_shared_routed_host_corrected)
        self.assertIn("--emit-generated-json", batch)

    def test_promotion_gate_passes_only_real_improvements(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default"),
            row("direct-gather", headline=90.0, ffn=40.0, wait=9.0, fused=12.0),
            row("gpu-pack", headline=120.0, ffn=70.0, wait=30.0, fused=25.0),
            row("full-stage5", headline=300.0, ffn=250.0, wait=80.0, fused=60.0),
            row("full-stage5-router", headline=280.0, ffn=230.0, wait=70.0, fused=55.0),
        ]

        gate = script.build_promotion_gate(
            rows,
            ["default", "direct-gather", "gpu-pack", "full-stage5", "full-stage5-router"],
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["direct-gather"])
        failures = gate["candidates"][1]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", failures)
        self.assertIn("prompt_hello:ffn_not_improved", failures)
        full_failures = gate["candidates"][2]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", full_failures)
        self.assertIn("prompt_hello:ffn_not_improved", full_failures)
        router_failures = gate["candidates"][3]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", router_failures)
        self.assertIn("prompt_hello:ffn_not_improved", router_failures)

    def test_promotion_gate_rejects_id_mismatch_and_missing_profile(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default"),
            row("direct-gather", ids=[99], headline=90.0, ffn=40.0, wait=None),
        ]

        gate = script.build_promotion_gate(rows, ["default", "direct-gather"])

        self.assertFalse(gate["passed"])
        failures = gate["candidates"][0]["failures"]
        self.assertIn("prompt_hello:generated_ids_mismatch", failures)
        self.assertIn("prompt_hello:missing_command_buffer_wait_profile", failures)

    def test_promotion_gate_rejects_layer_output_mismatch_when_tapped(self):
        script = sweep_qwen36_fused_routed_int4
        baseline = row("default", ids=[11, 271])
        candidate = row("direct-gather", ids=[11, 271], headline=90.0, ffn=40.0, wait=9.0)
        baseline["layer_output_taps"] = [
            {"position": 0, "layer": 0, "phase": "attn", "checksum": "same"},
            {"position": 0, "layer": 0, "phase": "ffn", "checksum": "baseline"},
        ]
        candidate["layer_output_taps"] = [
            {"position": 0, "layer": 0, "phase": "attn", "checksum": "same"},
            {"position": 0, "layer": 0, "phase": "ffn", "checksum": "candidate"},
        ]

        gate = script.build_promotion_gate([baseline, candidate], ["default", "direct-gather"])

        self.assertFalse(gate["passed"])
        failures = gate["candidates"][0]["failures"]
        self.assertIn("prompt_hello:layer_output_checksum_mismatch", failures)
        prompt = gate["candidates"][0]["prompts"][0]
        self.assertEqual(prompt["layer_output_compared_rows"], 2)
        self.assertEqual(prompt["layer_output_checksum_mismatches"], 1)
        self.assertEqual(prompt["layer_output_first_mismatch"]["layer"], 0)
        self.assertEqual(prompt["layer_output_first_mismatch"]["phase"], "ffn")

    def test_promotion_gate_can_tolerate_proven_layer_output_delta(self):
        script = sweep_qwen36_fused_routed_int4
        baseline = row("default", ids=[11, 271])
        candidate = row("direct-gather", ids=[11, 271], headline=90.0, ffn=40.0, wait=9.0)
        baseline["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "baseline"},
        ]
        candidate["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "candidate"},
        ]
        baseline["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "chained",
                "checksum": "baseline",
                "bf16": "bdc1",
            }
        ]
        candidate["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "decode_batch",
                "checksum": "candidate",
                "bf16": "bdc2",
            }
        ]

        gate = script.build_promotion_gate(
            [baseline, candidate],
            ["default", "direct-gather"],
            allow_layer_output_tolerance=True,
            layer_output_max_abs_delta=0.001,
            layer_output_max_ulp_delta=1,
            layer_output_max_differing_elems=1,
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["direct-gather"])
        prompt = gate["candidates"][0]["prompts"][0]
        self.assertEqual(prompt["layer_output_checksum_mismatches"], 1)
        self.assertEqual(prompt["layer_output_tolerated_checksum_mismatches"], 1)
        self.assertEqual(prompt["layer_output_untolerated_checksum_mismatches"], 0)
        self.assertNotIn("prompt_hello:layer_output_checksum_mismatch", gate["candidates"][0]["failures"])

    def test_layer_output_tolerance_policy_reports_required_thresholds(self):
        script = sweep_qwen36_fused_routed_int4
        baseline = row("default", ids=[11, 271])
        candidate = row("direct-gather", ids=[11, 271], headline=90.0, ffn=40.0, wait=9.0)
        baseline["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "baseline"},
        ]
        candidate["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "candidate"},
        ]
        baseline["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "chained",
                "checksum": "baseline",
                "bf16": "bdc1",
            }
        ]
        candidate["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "decode_batch",
                "checksum": "candidate",
                "bf16": "bdc2",
            }
        ]

        policy = script.build_layer_output_tolerance_policy(
            [baseline, candidate],
            ["default", "direct-gather"],
        )

        self.assertEqual(
            policy["recommendation"],
            "choose_explicit_layer_output_tolerance_or_fix_kernel",
        )
        self.assertEqual(policy["modes_requiring_tolerance"], ["direct-gather"])
        self.assertEqual(policy["max_required_ulp_delta"], 1)
        self.assertEqual(policy["max_required_differing_elems"], 1)
        self.assertGreater(policy["max_required_abs_delta"], 0.0)
        prompt = policy["prompt_results"][0]
        self.assertEqual(prompt["first_mismatch"]["layer"], 7)
        self.assertEqual(prompt["missing_delta_evidence"], 0)

        covered = script.build_layer_output_tolerance_policy(
            [baseline, candidate],
            ["default", "direct-gather"],
            allow_layer_output_tolerance=True,
            layer_output_max_abs_delta=0.001,
            layer_output_max_ulp_delta=1,
            layer_output_max_differing_elems=1,
        )

        self.assertEqual(
            covered["recommendation"],
            "current_layer_output_tolerance_covers_mismatches",
        )
        self.assertEqual(covered["modes_within_current_tolerance"], ["direct-gather"])
        self.assertEqual(covered["modes_requiring_tolerance"], [])

    def test_render_markdown_includes_layer_output_tolerance_policy(self):
        script = sweep_qwen36_fused_routed_int4
        baseline = row("default", ids=[11, 271])
        candidate = row("direct-gather", ids=[11, 271], headline=90.0, ffn=40.0, wait=9.0)
        baseline["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "baseline"},
        ]
        candidate["layer_output_taps"] = [
            {"position": 0, "layer": 7, "phase": "ffn", "checksum": "candidate"},
        ]
        baseline["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "chained",
                "checksum": "baseline",
                "bf16": "bdc1",
            }
        ]
        candidate["layer_output_delta_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "phase": "ffn",
                "path": "decode_batch",
                "checksum": "candidate",
                "bf16": "bdc2",
            }
        ]
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            metal_profile_phases=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=True,
            promotion_allow_layer_output_tolerance=True,
            promotion_layer_output_max_abs_delta=0.001,
            promotion_layer_output_max_ulp_delta=1,
            promotion_layer_output_max_differing_elems=1,
        )

        report = script.build_report(
            [baseline, candidate],
            args,
            ["default", "direct-gather"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertEqual(
            report["summary"]["layer_output_tolerance_policy"]["recommendation"],
            "current_layer_output_tolerance_covers_mismatches",
        )
        self.assertIn(
            "layer_output_tolerance_recommendation: `current_layer_output_tolerance_covers_mismatches`",
            md,
        )
        self.assertIn("## Layer Output Tolerance Policy", md)
        self.assertIn("| hello | direct-gather | false | ok | 1 | 0 | 0 |", md)

    def test_promotion_gate_rejects_host_correction_diagnostic_modes(self):
        script = sweep_qwen36_fused_routed_int4
        for mode in (
            "full-stage5-router-simd-batch-shared-host-corrected",
            "full-stage5-router-simd-batch-shared-routed-host-corrected",
        ):
            candidate = row(mode, ids=[11, 271], headline=90.0, ffn=40.0, wait=9.0)
            gate = script.build_promotion_gate(
                [row("default", ids=[11, 271]), candidate],
                ["default", mode],
            )

            self.assertFalse(gate["passed"])
            self.assertIn(
                "prompt_hello:diagnostic_mode_not_promotable",
                gate["candidates"][0]["failures"],
            )

    def test_render_markdown_includes_gate_rows(self):
        script = sweep_qwen36_fused_routed_int4
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            metal_profile_phases=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=True,
        )
        report = script.build_report(
            [
                row("default"),
                row(
                    "direct-gather",
                    headline=90.0,
                    ffn=40.0,
                    wait=9.0,
                    fused=12.0,
                    fused_gpu=3.0,
                ),
            ],
            args,
            ["default", "direct-gather"],
            "smoke",
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Fused Routed INT4 Sweep", md)
        self.assertIn("promotion_gate_passed: `True`", md)
        self.assertIn("| direct-gather | true | - |", md)
        self.assertEqual(
            report["summary"]["ffn_residency_gap"]["recommendation"],
            "prototype_ffn_gpu_arithmetic_tiling_path",
        )
        self.assertIn("## FFN Residency Gap", md)

    def test_ffn_residency_gap_classifies_wait_bound_candidates(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default"),
            row(
                "direct-gather",
                headline=90.0,
                ffn=40.0,
                wait=30.0,
                fused=80.0,
                fused_gpu=4.0,
            ),
        ]
        script.annotate_ffn_profile_fields(rows, max_wall_gpu_ratio=4.0, max_wait_gpu_ratio=4.0)

        gap = script.build_ffn_residency_gap(
            rows,
            ["default", "direct-gather"],
            max_wall_gpu_ratio=4.0,
            max_wait_gpu_ratio=4.0,
        )

        direct_prompt = gap["candidates"][0]["prompts"][0]
        self.assertEqual(direct_prompt["fused_wall_ms"], 80.0)
        self.assertEqual(direct_prompt["fused_gpu_ms"], 4.0)
        self.assertEqual(direct_prompt["ffn_attribution_class"], "residency_or_submit_wait")
        self.assertEqual(
            gap["recommendation"],
            "prototype_ffn_residency_or_submit_wait_path",
        )

    def test_phase_profile_sums_full_stage5_gpu_subdispatches(self):
        script = sweep_qwen36_fused_routed_int4
        candidate = row("full-stage5", wait=48.0)
        candidate["metal_profile"]["entries"].extend(
            [
                {
                    "op": "qwen36_ffn_int4_stage5",
                    "path": "native",
                    "calls": 1,
                    "mean_ms": 96.0,
                    "total_ms": 96.0,
                    "max_ms": 96.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": 8.0,
                    "total_ms": 8.0,
                    "max_ms": 8.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_shared_down",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": 4.0,
                    "total_ms": 4.0,
                    "max_ms": 4.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
                    "path": "runtime",
                    "calls": 1,
                    "mean_ms": 12.0,
                    "total_ms": 12.0,
                    "max_ms": 12.0,
                },
            ]
        )
        rows = [row("default"), candidate]

        script.annotate_ffn_profile_fields(rows, max_wall_gpu_ratio=4.0, max_wait_gpu_ratio=4.0)

        self.assertEqual(candidate["fused_wall_ms"], 96.0)
        self.assertEqual(candidate["fused_gpu_ms"], 24.0)
        self.assertEqual(candidate["fused_wall_gpu_ratio"], 4.0)
        self.assertEqual(candidate["wait_gpu_ratio"], 2.0)
        self.assertEqual(candidate["ffn_attribution_class"], "gpu_arithmetic")

    def test_decode_batch_phase_profile_sums_labeled_gpu_chunks(self):
        script = sweep_qwen36_fused_routed_int4
        candidate = row("full-stage5-router-batch-phases", wait=40.0)
        candidate["metal_profile"]["entries"].extend(
            [
                {
                    "op": "command_buffer_gpu:qwen36_decode_batch_linear_attn",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 7.5,
                    "total_ms": 15.0,
                    "max_ms": 8.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_decode_batch_ffn",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 12.5,
                    "total_ms": 25.0,
                    "max_ms": 13.0,
                },
            ]
        )

        script.annotate_ffn_profile_fields(
            [candidate],
            max_wall_gpu_ratio=4.0,
            max_wait_gpu_ratio=4.0,
        )

        self.assertEqual(candidate["decode_batch_linear_gpu_ms"], 15.0)
        self.assertEqual(candidate["decode_batch_ffn_gpu_ms"], 25.0)
        self.assertEqual(candidate["fused_gpu_ms"], 25.0)
        self.assertEqual(candidate["wait_gpu_ratio"], 1.6)

    def test_decode_batch_coarse_comparison_summarizes_serial_vs_simd(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default", ids=[11, 271], headline=180.0),
            row(
                "full-stage5-router-batch",
                ids=[11, 271],
                headline=200.0,
                wait=20.0,
                decode_batch_gpu=100.0,
            ),
            row(
                "full-stage5-router-simd-batch",
                ids=[11, 271],
                headline=150.0,
                wait=24.0,
                decode_batch_gpu=80.0,
            ),
        ]
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )

        report = script.build_report(
            rows,
            args,
            [
                "default",
                "full-stage5-router-batch",
                "full-stage5-router-simd-batch",
            ],
            "smoke",
        )
        md = script.render_markdown(report)
        coarse = report["summary"]["decode_batch_coarse"]

        self.assertTrue(coarse["available"])
        self.assertEqual(coarse["comparison_count"], 1)
        self.assertEqual(coarse["mismatch_count"], 0)
        self.assertEqual(
            coarse["recommendation"],
            "keep_simd_router_enabled_then_target_remaining_gpu_work",
        )
        comparison = coarse["comparisons"][0]
        self.assertEqual(comparison["simd_decode_batch_gpu_ms"], 80.0)
        self.assertAlmostEqual(comparison["decode_ratio"], 0.75)
        self.assertAlmostEqual(comparison["decode_batch_gpu_ratio"], 0.8)
        self.assertIn("decode_batch_coarse_recommendation", md)
        self.assertIn("## Decode Batch Coarse SIMD", md)
        self.assertIn("| hello | true | true | 400.000 | 300.000 | 0.750 |", md)

    def test_decode_batch_deferred_phase_summary_marks_ffn_dominant(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default", ids=[11, 271], headline=100.0),
            row(
                "full-stage5-router-batch-deferred-phases",
                ids=[11, 271],
                wait=80.0,
                decode_batch_linear_gpu=12.0,
                decode_batch_ffn_gpu=88.0,
            ),
            row(
                "full-stage5-router-simd-batch-deferred-phases",
                ids=[11, 271],
                wait=75.0,
                decode_batch_linear_gpu=10.0,
                decode_batch_ffn_gpu=70.0,
            ),
        ]
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )

        report = script.build_report(
            rows,
            args,
            [
                "default",
                "full-stage5-router-batch-deferred-phases",
                "full-stage5-router-simd-batch-deferred-phases",
            ],
            "smoke",
        )
        md = script.render_markdown(report)
        summary = report["summary"]["decode_batch_deferred_phase"]

        self.assertTrue(summary["available"])
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["recommendation"], "target_batched_ffn_gpu_work")
        simd = summary["rows"][1]
        self.assertEqual(simd["linear_gpu_ms"], 10.0)
        self.assertEqual(simd["ffn_gpu_ms"], 70.0)
        self.assertAlmostEqual(simd["ffn_share"], 70.0 / 80.0)
        self.assertIn("decode_batch_deferred_phase_recommendation", md)
        self.assertIn("## Decode Batch Deferred Phase", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-deferred-phases | simd | true |",
            md,
        )

    def test_decode_batch_ffn_subphase_profile_sums_labeled_gpu_chunks(self):
        script = sweep_qwen36_fused_routed_int4
        candidate = row("full-stage5-router-batch-ffn-phases", wait=60.0)
        candidate["metal_profile"]["entries"].extend(
            [
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 1.5,
                    "total_ms": 3.0,
                    "max_ms": 2.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 10.0,
                    "total_ms": 20.0,
                    "max_ms": 11.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 0.5,
                    "total_ms": 1.0,
                    "max_ms": 0.6,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_shared_down",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 6.0,
                    "total_ms": 12.0,
                    "max_ms": 6.5,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 40.0,
                    "total_ms": 80.0,
                    "max_ms": 42.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 25.0,
                    "total_ms": 50.0,
                    "max_ms": 26.0,
                },
            ]
        )

        script.annotate_ffn_profile_fields(
            [candidate],
            max_wall_gpu_ratio=4.0,
            max_wait_gpu_ratio=4.0,
        )

        self.assertEqual(candidate["decode_batch_ffn_router_topk_gpu_ms"], 3.0)
        self.assertEqual(candidate["decode_batch_ffn_shared_gate_up_gpu_ms"], 20.0)
        self.assertEqual(candidate["decode_batch_ffn_shared_scalar_gpu_ms"], 1.0)
        self.assertEqual(candidate["decode_batch_ffn_shared_down_gpu_ms"], 12.0)
        self.assertEqual(candidate["decode_batch_ffn_expert_gate_up_gpu_ms"], 80.0)
        self.assertEqual(candidate["decode_batch_ffn_expert_down_gpu_ms"], 50.0)
        self.assertEqual(candidate["decode_batch_ffn_gpu_ms"], 166.0)
        self.assertEqual(candidate["fused_gpu_ms"], 166.0)
        self.assertAlmostEqual(candidate["wait_gpu_ratio"], 60.0 / 166.0)

    def test_decode_batch_ffn_subphase_profile_accepts_simd_router_label(self):
        script = sweep_qwen36_fused_routed_int4
        candidate = row("full-stage5-router-simd-batch-ffn-phases", wait=30.0)
        candidate["metal_profile"]["entries"].extend(
            [
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 2.5,
                    "total_ms": 5.0,
                    "max_ms": 3.0,
                },
                {
                    "op": "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
                    "path": "runtime",
                    "calls": 2,
                    "mean_ms": 7.5,
                    "total_ms": 15.0,
                    "max_ms": 8.0,
                },
            ]
        )

        script.annotate_ffn_profile_fields(
            [candidate],
            max_wall_gpu_ratio=4.0,
            max_wait_gpu_ratio=4.0,
        )

        self.assertEqual(candidate["decode_batch_ffn_router_topk_gpu_ms"], 5.0)
        self.assertEqual(candidate["decode_batch_ffn_expert_down_gpu_ms"], 15.0)
        self.assertEqual(candidate["decode_batch_ffn_gpu_ms"], 20.0)
        self.assertEqual(candidate["fused_gpu_ms"], 20.0)

    def test_router_parity_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-router-parity] call=0 layer=5 router_path=simd "
            "topk_idx_match=0 workspace_idx_match=0 output_idx_match=0 "
            "topk_first_mismatch=1 workspace_first_idx_mismatch=1 "
            "output_first_idx_mismatch=1 h_norm_max_abs=1.25000000e-01 "
            "h_norm_argmax=17 logits_max_abs=2.50000000e-01 logits_argmax=33 "
            "host_top_logit_idx=2 metal_top_logit_idx=4 "
            "host_top_logit=1.25000000e+00 metal_top_logit=1.37500000e+00 "
            "host_logit_at_metal_top=1.12500000e+00 "
            "metal_logit_at_host_top=1.00000000e+00 "
            "topk_weight_max_abs=3.12500000e-02 topk_weight_argmax=2 "
            "host_idx=1,2,3 workspace_idx=1,4,3 output_idx=1,4,3 "
            "host_w=0.50000000,0.25000000,0.12500000 "
            "metal_w=0.46875000,0.28125000,0.12500000\n"
        )

        taps = script.parse_router_parity_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["layer"], 5)
        self.assertEqual(taps[0]["topk_idx_match"], 0)
        self.assertEqual(taps[0]["router_path"], "simd")
        self.assertEqual(taps[0]["topk_first_mismatch"], 1)
        self.assertEqual(taps[0]["host_idx"], "1,2,3")
        self.assertEqual(taps[0]["metal_top_logit_idx"], 4)
        self.assertAlmostEqual(taps[0]["logits_max_abs"], 0.25)

        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=True,
            router_parity_tap_max_calls=3,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        candidate = row("full-stage5-router", ids=[11, 353])
        candidate["router_parity_taps"] = taps
        report = script.build_report([row("default"), candidate], args, ["default", "full-stage5-router"], "smoke")
        md = script.render_markdown(report)

        self.assertIn("router_parity_tap: `True`", md)
        self.assertIn("router_parity_tap_count: `1`", md)
        self.assertIn("router_parity_mismatches: `1`", md)
        self.assertIn("## Router Parity Tap", md)
        self.assertIn("| hello | full-stage5-router | simd | 5 | false | 1 |", md)
        self.assertEqual(report["summary"]["router_parity"]["paths"], ["simd"])

    def test_shared_parity_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-shared-parity] call=0 layer=5 shared_path=gate_up_tiled "
            "shared_gate_max_abs=1.25000000e-01 shared_gate_argmax=17 "
            "shared_up_max_abs=6.25000000e-02 shared_up_argmax=19 "
            "shared_mid_max_abs=5.00000000e-01 shared_mid_argmax=23 "
            "host_shared_gate_at_mid_argmax=1.00000000e+00 "
            "metal_shared_gate_at_mid_argmax=1.12500000e+00 "
            "host_shared_up_at_mid_argmax=2.00000000e+00 "
            "metal_shared_up_at_mid_argmax=1.75000000e+00 "
            "host_shared_mid_at_argmax=1.50000000e+00 "
            "metal_shared_mid_at_argmax=1.00000000e+00 "
            "shared_scalar_abs=3.12500000e-02 "
            "host_shared_scalar=5.00000000e-01 metal_shared_scalar=5.31250000e-01 "
            "shared_out_max_abs=2.50000000e-01 shared_out_argmax=33 "
            "host_shared_out_at_argmax=1.25000000e+00 "
            "metal_shared_out_at_argmax=1.00000000e+00 "
            "host_shared_down_acc_at_out_argmax=1.00000000e+01 "
            "host_shared_gated_at_out_argmax=5.00000000e+00 "
            "host_shared_out_recomputed_at_argmax=5.00000000e+00 "
            "metal_mid_host_shared_down_acc_at_out_argmax=9.00000000e+00 "
            "metal_mid_host_shared_gated_at_out_argmax=4.50000000e+00 "
            "metal_mid_host_shared_out_at_argmax=4.50000000e+00\n"
        )

        taps = script.parse_shared_parity_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["layer"], 5)
        self.assertEqual(taps[0]["shared_path"], "gate_up_tiled")
        self.assertAlmostEqual(taps[0]["shared_mid_max_abs"], 0.5)
        self.assertAlmostEqual(taps[0]["host_shared_mid_at_argmax"], 1.5)
        self.assertAlmostEqual(taps[0]["metal_shared_mid_at_argmax"], 1.0)
        self.assertEqual(taps[0]["shared_out_argmax"], 33)
        self.assertAlmostEqual(taps[0]["metal_mid_host_shared_out_at_argmax"], 4.5)

        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=True,
            shared_parity_tap_max_calls=3,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        candidate = row("full-stage5-router-simd-batch-shared-gate-up-tiled")
        candidate["shared_parity_taps"] = taps
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-gate-up-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertIn("shared_parity_tap: `True`", md)
        self.assertIn("shared_parity_tap_count: `1`", md)
        self.assertIn("shared_parity_max_out_abs: `0.25000000`", md)
        self.assertIn("## Shared Expert Parity Tap", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-shared-gate-up-tiled | gate_up_tiled | 5 |",
            md,
        )
        self.assertEqual(report["summary"]["shared_parity"]["paths"], ["gate_up_tiled"])

    def test_decode_batch_shared_parity_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-decode-batch-shared-parity] call=0 position=64 cache_pos=64 "
            "layer=5 router_path=simd phase_profile=1 shared_path=all_tiled "
            "shared_gate_max_abs=1.25000000e-01 shared_gate_argmax=17 "
            "shared_up_max_abs=6.25000000e-02 shared_up_argmax=19 "
            "shared_mid_max_abs=5.00000000e-01 shared_mid_argmax=23 "
            "host_shared_gate_at_mid_argmax=1.00000000e+00 "
            "metal_shared_gate_at_mid_argmax=1.12500000e+00 "
            "host_shared_up_at_mid_argmax=2.00000000e+00 "
            "metal_shared_up_at_mid_argmax=1.75000000e+00 "
            "host_shared_mid_at_argmax=1.50000000e+00 "
            "metal_shared_mid_at_argmax=1.00000000e+00 "
            "shared_scalar_abs=3.12500000e-02 "
            "host_shared_scalar=5.00000000e-01 metal_shared_scalar=5.31250000e-01 "
            "shared_out_max_abs=2.50000000e-01 shared_out_argmax=33 "
            "host_shared_out_at_argmax=1.25000000e+00 "
            "metal_shared_out_at_argmax=1.00000000e+00 "
            "host_shared_down_acc_at_out_argmax=1.00000000e+01 "
            "host_shared_gated_at_out_argmax=5.00000000e+00 "
            "host_shared_out_recomputed_at_argmax=5.00000000e+00 "
            "metal_mid_host_shared_down_acc_at_out_argmax=9.00000000e+00 "
            "metal_mid_host_shared_gated_at_out_argmax=4.50000000e+00 "
            "metal_mid_host_shared_out_at_argmax=4.50000000e+00\n"
        )

        taps = script.parse_decode_batch_shared_parity_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["position"], 64)
        self.assertEqual(taps[0]["router_path"], "simd")
        self.assertEqual(taps[0]["phase_profile"], 1)
        self.assertAlmostEqual(taps[0]["shared_out_max_abs"], 0.25)
        self.assertAlmostEqual(taps[0]["host_shared_gate_at_mid_argmax"], 1.0)
        self.assertAlmostEqual(taps[0]["metal_shared_up_at_mid_argmax"], 1.75)
        self.assertAlmostEqual(taps[0]["host_shared_gated_at_out_argmax"], 5.0)

        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=True,
            shared_parity_tap_max_calls=3,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        candidate = row("full-stage5-router-simd-batch-shared-tiled")
        candidate["decode_batch_shared_parity_taps"] = taps
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertIn("decode_batch_shared_parity_tap_count: `1`", md)
        self.assertIn("decode_batch_shared_parity_max_out_abs: `0.25000000`", md)
        self.assertIn("## Decode-Batch Shared Expert Parity Tap", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-shared-tiled | simd | true | all_tiled | 0 | 64 | 5 |",
            md,
        )
        self.assertEqual(
            report["summary"]["decode_batch_shared_parity"]["paths"],
            ["all_tiled"],
        )

    def test_shared_host_correction_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-shared-host-correction] layer=7 shared_path=host_order "
            "hidden=2048 shared_out_max_abs=6.10351562e-05 shared_out_argmax=1621 "
            "host_shared_out_at_argmax=1.23901367e-02 "
            "metal_shared_out_at_argmax=1.23291016e-02 "
            "output_patch_max_abs=4.88281250e-04 output_patch_argmax=1621 "
            "changed_output_elems=1 first_changed_output=1621\n"
        )

        corrections = script.parse_shared_host_corrections(output)

        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["layer"], 7)
        self.assertEqual(corrections[0]["shared_path"], "host_order")
        self.assertEqual(corrections[0]["shared_out_argmax"], 1621)
        self.assertAlmostEqual(corrections[0]["output_patch_max_abs"], 0.00048828125)

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=3,
            routed_parity_tap=False,
            routed_parity_tap_max_calls=3,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        mode = "full-stage5-router-simd-batch-shared-host-corrected"
        candidate = row(mode)
        candidate["shared_host_corrections"] = corrections
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", mode],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertEqual(report["summary"]["shared_host_correction"]["correction_count"], 1)
        self.assertEqual(report["summary"]["shared_host_correction"]["changed_count"], 1)
        self.assertIn("shared_host_correction_count: `1`", md)
        self.assertIn("shared_host_correction_max_output_patch_abs: `0.00048828`", md)
        self.assertIn("## Shared Host Correction", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-shared-host-corrected | host_order | 7 |",
            md,
        )

    def test_routed_host_correction_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-routed-host-correction] layer=33 router_path=simd "
            "hidden=2048 top_k=8 topk_idx_match=1 topk_weight_max_abs=0.00000000e+00 "
            "topk_weight_argmax=0 host_topk_weight_at_argmax=5.00000000e-01 "
            "metal_topk_weight_at_argmax=5.00000000e-01 "
            "expert_mid_max_abs=1.52587891e-05 "
            "expert_mid_argmax=17 expert_mid_group=2 expert_mid_row=1 "
            "host_expert_gate_at_mid_argmax=1.25000000e+00 "
            "metal_expert_gate_at_mid_argmax=1.24998474e+00 "
            "expert_gate_delta_at_mid_argmax=1.52587891e-05 "
            "host_expert_up_at_mid_argmax=1.10000000e+00 "
            "metal_expert_up_at_mid_argmax=1.09999084e+00 "
            "expert_up_delta_at_mid_argmax=9.15527344e-06 "
            "host_expert_silu_at_mid_argmax=9.71445143e-01 "
            "metal_expert_silu_at_mid_argmax=9.71430421e-01 "
            "expert_silu_delta_at_mid_argmax=1.47223473e-05 "
            "host_expert_mid_at_argmax=1.00000000e+00 "
            "metal_expert_mid_at_argmax=9.99984741e-01 "
            "host_expert_mid_recomputed_at_argmax=1.06858969e+00 "
            "metal_expert_mid_recomputed_at_argmax=1.06856441e+00 "
            "expert_mid_recompute_delta_at_argmax=2.52723694e-05 "
            "moe_out_max_abs=6.10351562e-05 moe_out_argmax=8 "
            "host_moe_out_at_argmax=1.25732422e-02 "
            "metal_moe_out_at_argmax=1.25122070e-02 "
            "host_routed_down_acc_at_moe_argmax=1.25740000e-02 "
            "host_routed_moe_out_recomputed_at_argmax=1.25732422e-02 "
            "metal_mid_host_topk_down_acc_at_moe_argmax=1.25120000e-02 "
            "metal_mid_host_topk_moe_out_at_argmax=1.25122070e-02 "
            "metal_mid_metal_topk_down_acc_at_moe_argmax=1.25120000e-02 "
            "metal_mid_metal_topk_moe_out_at_argmax=1.25122070e-02 "
            "output_patch_max_abs=1.22070312e-04 output_patch_argmax=8 "
            "host_final_out_at_argmax=1.26953125e-02 "
            "metal_final_out_at_argmax=1.25732422e-02 "
            "changed_output_elems=1 first_changed_output=8\n"
        )

        corrections = script.parse_routed_host_corrections(output)

        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["layer"], 33)
        self.assertEqual(corrections[0]["router_path"], "simd")
        self.assertEqual(corrections[0]["moe_out_argmax"], 8)
        self.assertEqual(corrections[0]["expert_mid_group"], 2)
        self.assertEqual(corrections[0]["expert_mid_row"], 1)
        self.assertEqual(corrections[0]["topk_idx_match"], 1)
        self.assertAlmostEqual(
            corrections[0]["expert_gate_delta_at_mid_argmax"],
            0.0000152587891,
        )
        self.assertAlmostEqual(
            corrections[0]["expert_mid_recompute_delta_at_argmax"],
            0.0000252723694,
        )
        self.assertAlmostEqual(
            corrections[0]["metal_mid_host_topk_moe_out_at_argmax"],
            0.0125122070,
        )
        self.assertAlmostEqual(corrections[0]["output_patch_max_abs"], 0.000122070312)

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=3,
            routed_parity_tap=False,
            routed_parity_tap_max_calls=3,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        mode = "full-stage5-router-simd-batch-shared-routed-host-corrected"
        candidate = row(mode)
        candidate["routed_host_corrections"] = corrections
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", mode],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertEqual(report["summary"]["routed_host_correction"]["correction_count"], 1)
        self.assertEqual(report["summary"]["routed_host_correction"]["changed_count"], 1)
        self.assertEqual(
            report["summary"]["routed_host_correction"][
                "metal_mid_host_topk_matches_metal_count"
            ],
            1,
        )
        self.assertEqual(
            report["summary"]["routed_host_correction"]["max_topk_weight_abs"],
            0.0,
        )
        self.assertAlmostEqual(
            report["summary"]["routed_host_correction"]["max_expert_gate_abs"],
            0.0000152587891,
        )
        self.assertAlmostEqual(
            report["summary"]["routed_host_correction"][
                "max_expert_mid_recompute_abs"
            ],
            0.0000252723694,
        )
        self.assertIn("routed_host_correction_count: `1`", md)
        self.assertIn(
            "routed_host_correction_metal_mid_host_topk_matches_metal_count: `1`",
            md,
        )
        self.assertIn("routed_host_correction_max_expert_gate_abs: `0.00001526`", md)
        self.assertIn(
            "routed_host_correction_max_expert_mid_recompute_abs: `0.00002527`",
            md,
        )
        self.assertIn("routed_host_correction_max_output_patch_abs: `0.00012207`", md)
        self.assertIn("## Routed Host Correction", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-shared-routed-host-corrected | simd | 33 | true |",
            md,
        )
        self.assertIn("| 17 | 2 | 1 | 1.25000000 | 1.24998474 | 0.00001526 |", md)

    def test_routed_gate_up_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-routed-gate-up-tap] layer=6 router_path=simd "
            "hidden=2048 top_k=8 topk_idx_match=1 topk_weight_max_abs=0.00000000e+00 "
            "topk_weight_argmax=0 host_topk_weight_at_argmax=5.00000000e-01 "
            "metal_topk_weight_at_argmax=5.00000000e-01 "
            "expert_mid_max_abs=1.52587891e-05 "
            "expert_mid_argmax=17 expert_mid_group=2 expert_mid_row=1 "
            "host_expert_gate_at_mid_argmax=1.25000000e+00 "
            "metal_expert_gate_at_mid_argmax=1.24998474e+00 "
            "expert_gate_delta_at_mid_argmax=1.52587891e-05 "
            "host_expert_up_at_mid_argmax=1.10000000e+00 "
            "metal_expert_up_at_mid_argmax=1.09999084e+00 "
            "expert_up_delta_at_mid_argmax=9.15527344e-06 "
            "host_expert_silu_at_mid_argmax=9.71445143e-01 "
            "metal_expert_silu_at_mid_argmax=9.71430421e-01 "
            "expert_silu_delta_at_mid_argmax=1.47223473e-05 "
            "host_expert_mid_at_argmax=1.00000000e+00 "
            "metal_expert_mid_at_argmax=9.99984741e-01 "
            "host_expert_mid_recomputed_at_argmax=1.06858969e+00 "
            "metal_expert_mid_recomputed_at_argmax=1.06856441e+00 "
            "expert_mid_recompute_delta_at_argmax=2.52723694e-05 "
            "moe_out_max_abs=6.10351562e-05 moe_out_argmax=8 "
            "host_moe_out_at_argmax=1.25732422e-02 "
            "metal_moe_out_at_argmax=1.25122070e-02 "
            "host_routed_down_acc_at_moe_argmax=1.25740000e-02 "
            "host_routed_moe_out_recomputed_at_argmax=1.25732422e-02 "
            "metal_mid_host_topk_down_acc_at_moe_argmax=1.25120000e-02 "
            "metal_mid_host_topk_moe_out_at_argmax=1.25122070e-02 "
            "metal_mid_metal_topk_down_acc_at_moe_argmax=1.25120000e-02 "
            "metal_mid_metal_topk_moe_out_at_argmax=1.25122070e-02 "
            "final_out_max_abs=1.22070312e-04 final_out_argmax=8 "
            "host_final_out_at_argmax=1.26953125e-02 "
            "metal_final_out_at_argmax=1.25732422e-02\n"
        )

        taps = script.parse_routed_gate_up_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["layer"], 6)
        self.assertEqual(taps[0]["router_path"], "simd")
        self.assertEqual(taps[0]["moe_out_argmax"], 8)
        self.assertEqual(taps[0]["expert_mid_group"], 2)
        self.assertEqual(taps[0]["expert_mid_row"], 1)
        self.assertEqual(taps[0]["topk_idx_match"], 1)
        self.assertAlmostEqual(
            taps[0]["expert_gate_delta_at_mid_argmax"],
            0.0000152587891,
        )
        self.assertAlmostEqual(
            taps[0]["expert_mid_recompute_delta_at_argmax"],
            0.0000252723694,
        )
        self.assertAlmostEqual(taps[0]["final_out_max_abs"], 0.000122070312)

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=3,
            routed_parity_tap=False,
            routed_parity_tap_max_calls=3,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        mode = "full-stage5-router-simd-batch-routed-gate-up-tap"
        candidate = row(mode)
        candidate["routed_gate_up_taps"] = taps
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", mode],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertEqual(report["summary"]["routed_gate_up_tap"]["tap_count"], 1)
        self.assertEqual(
            report["summary"]["routed_gate_up_tap"][
                "metal_mid_host_topk_matches_metal_count"
            ],
            1,
        )
        self.assertEqual(
            report["summary"]["routed_gate_up_tap"]["max_topk_weight_abs"],
            0.0,
        )
        self.assertAlmostEqual(
            report["summary"]["routed_gate_up_tap"]["max_expert_gate_abs"],
            0.0000152587891,
        )
        self.assertAlmostEqual(
            report["summary"]["routed_gate_up_tap"][
                "max_expert_mid_recompute_abs"
            ],
            0.0000252723694,
        )
        self.assertIn("routed_gate_up_tap_count: `1`", md)
        self.assertIn(
            "routed_gate_up_tap_metal_mid_host_topk_matches_metal_count: `1`",
            md,
        )
        self.assertIn("routed_gate_up_tap_max_expert_gate_abs: `0.00001526`", md)
        self.assertIn(
            "routed_gate_up_tap_max_expert_mid_recompute_abs: `0.00002527`",
            md,
        )
        self.assertIn("routed_gate_up_tap_max_final_out_abs: `0.00012207`", md)
        self.assertIn("## Routed Gate/Up Tap", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-routed-gate-up-tap | simd | 6 | true |",
            md,
        )
        self.assertIn("| 17 | 2 | 1 | 1.25000000 | 1.24998474 | 0.00001526 |", md)

    def test_decode_batch_routed_parity_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-decode-batch-routed-parity] call=0 position=64 cache_pos=64 "
            "layer=5 router_path=simd phase_profile=1 topk_idx_match=1 "
            "workspace_idx_match=1 output_idx_match=1 topk_first_mismatch=-1 "
            "workspace_first_idx_mismatch=-1 output_first_idx_mismatch=-1 "
            "topk_weight_max_abs=1.25000000e-02 topk_weight_argmax=0 "
            "expert_mid_max_abs=5.00000000e-01 expert_mid_argmax=23 "
            "host_expert_mid_at_argmax=1.50000000e+00 "
            "metal_expert_mid_at_argmax=1.00000000e+00 "
            "moe_out_max_abs=2.50000000e-01 moe_out_argmax=33 "
            "host_moe_out_at_argmax=1.25000000e+00 "
            "metal_moe_out_at_argmax=1.00000000e+00 "
            "final_out_max_abs=1.25000000e-01 final_out_argmax=44 "
            "host_final_out_at_argmax=2.00000000e+00 "
            "metal_final_out_at_argmax=1.87500000e+00 "
            "host_idx=1,2 workspace_idx=1,2 output_idx=1,2 "
            "host_w=0.60000002,0.39999998 metal_w=0.61250001,0.38750000\n"
        )

        taps = script.parse_decode_batch_routed_parity_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["position"], 64)
        self.assertEqual(taps[0]["router_path"], "simd")
        self.assertEqual(taps[0]["topk_idx_match"], 1)
        self.assertAlmostEqual(taps[0]["moe_out_max_abs"], 0.25)
        self.assertAlmostEqual(taps[0]["final_out_max_abs"], 0.125)

        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=40,
            routed_parity_tap=True,
            routed_parity_tap_max_calls=3,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        candidate = row("full-stage5-router-simd-batch-shared-tiled")
        candidate["decode_batch_routed_parity_taps"] = taps
        report = script.build_report(
            [row("default"), candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertIn("routed_parity_tap: `True`", md)
        self.assertIn("decode_batch_routed_parity_tap_count: `1`", md)
        self.assertIn("decode_batch_routed_parity_max_moe_out_abs: `0.25000000`", md)
        self.assertIn("## Decode-Batch Routed Expert Parity Tap", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-shared-tiled | simd | true | 0 | 64 | 5 | true |",
            md,
        )
        self.assertEqual(
            report["summary"]["decode_batch_routed_parity"]["paths"],
            ["simd"],
        )

    def test_downstream_parity_taps_are_compared_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = "\n".join(
            [
                "[qwen36-final-hidden-tap] step=0 gen_index=0 position=1 path=chained "
                "lm_head_folded=0 elems=2048 checksum=aaaabbbbccccdddd "
                "l2=1.25000000e+01 max_abs=2.50000000e+00 max_abs_idx=17 "
                "head8=0001,0002",
                "[qwen36-logits-tap] step=0 gen_index=0 position=1 path=chained "
                "lm_head_folded=0 elems=151936 checksum=1111222233334444 "
                "top1_idx=11 top1_val=3.50000000e+00 top5=11:3.5,7:2.0",
            ]
        )
        candidate_output = "\n".join(
            [
                "[qwen36-final-hidden-tap] step=0 gen_index=0 position=1 path=decode_batch "
                "lm_head_folded=0 elems=2048 checksum=ffffbbbbccccdddd "
                "l2=1.26000000e+01 max_abs=2.60000000e+00 max_abs_idx=19 "
                "head8=0001,0003",
                "[qwen36-logits-tap] step=0 gen_index=0 position=1 path=decode_batch "
                "lm_head_folded=0 elems=151936 checksum=9999222233334444 "
                "top1_idx=353 top1_val=3.75000000e+00 top5=353:3.75,11:3.5",
            ]
        )

        default = row("default")
        default["final_hidden_taps"] = script.parse_final_hidden_taps(output)
        default["logits_taps"] = script.parse_logits_taps(output)
        candidate = row("full-stage5-router-simd-batch-shared-tiled")
        candidate["final_hidden_taps"] = script.parse_final_hidden_taps(candidate_output)
        candidate["logits_taps"] = script.parse_logits_taps(candidate_output)

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=True,
            layer_output_tap=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        report = script.build_report(
            [default, candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertIn("downstream_parity_tap: `True`", md)
        self.assertIn("final_hidden_checksum_mismatches: `1`", md)
        self.assertIn("logits_top1_mismatches: `1`", md)
        self.assertIn("## Final Hidden Tap", md)
        self.assertIn("## Logits Tap", md)
        self.assertFalse(
            report["summary"]["final_hidden_tap"]["comparisons"][0]["checksum_match"]
        )
        self.assertFalse(report["summary"]["logits_tap"]["comparisons"][0]["top1_match"])

    def test_layer_output_taps_are_compared_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = "\n".join(
            [
                "[qwen36-layer-output-tap] call=0 position=1 cache_pos=1 path=chained "
                "phase_profile=0 layer=0 phase=attn elems=2048 "
                "checksum=aaaabbbbccccdddd l2=1.25000000e+01 "
                "max_abs=2.50000000e+00 max_abs_idx=17 head8=0001,0002",
                "[qwen36-layer-output-tap] call=1 position=1 cache_pos=1 path=chained "
                "phase_profile=0 layer=0 phase=ffn elems=2048 "
                "checksum=1111222233334444 l2=1.35000000e+01 "
                "max_abs=2.70000000e+00 max_abs_idx=19 head8=0003,0004",
            ]
        )
        candidate_output = "\n".join(
            [
                "[qwen36-layer-output-tap] call=0 position=1 cache_pos=1 path=decode_batch "
                "phase_profile=0 layer=0 phase=attn elems=2048 "
                "checksum=aaaabbbbccccdddd l2=1.25000000e+01 "
                "max_abs=2.50000000e+00 max_abs_idx=17 head8=0001,0002",
                "[qwen36-layer-output-tap] call=1 position=1 cache_pos=1 path=decode_batch "
                "phase_profile=0 layer=0 phase=ffn elems=2048 "
                "checksum=ffff222233334444 l2=1.36000000e+01 "
                "max_abs=2.80000000e+00 max_abs_idx=23 head8=0003,0005",
            ]
        )

        default = row("default")
        default["layer_output_taps"] = script.parse_layer_output_taps(output)
        candidate = row("full-stage5-router-simd-batch-shared-tiled")
        candidate["layer_output_taps"] = script.parse_layer_output_taps(candidate_output)

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=True,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        report = script.build_report(
            [default, candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)

        self.assertIn("layer_output_tap: `True`", md)
        self.assertIn("layer_output_checksum_mismatches: `1`", md)
        self.assertIn("## Layer Output Tap", md)
        comparisons = report["summary"]["layer_output_tap"]["comparisons"]
        self.assertTrue(comparisons[0]["checksum_match"])
        self.assertFalse(comparisons[1]["checksum_match"])
        self.assertEqual(
            report["summary"]["layer_output_tap"]["first_mismatch"]["phase"],
            "ffn",
        )

    def test_layer_output_delta_taps_compute_numeric_delta(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-layer-output-delta-tap] call=0 position=0 cache_pos=0 "
            "path=chained phase_profile=0 layer=0 phase=ffn elems=3 "
            "checksum=aaaabbbbccccdddd bf16=3f80,4000,c000\n"
        )
        candidate_output = (
            "[qwen36-layer-output-delta-tap] call=0 position=0 cache_pos=0 "
            "path=decode_batch phase_profile=0 layer=0 phase=ffn elems=3 "
            "checksum=ffffbbbbccccdddd bf16=3f80,4040,c000\n"
        )

        default = row("default")
        default["layer_output_delta_taps"] = script.parse_layer_output_delta_taps(output)
        candidate = row("full-stage5-router-simd-batch-shared-tiled")
        candidate["layer_output_delta_taps"] = script.parse_layer_output_delta_taps(
            candidate_output
        )

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=False,
            layer_output_delta_tap=True,
            layer_output_delta_position=0,
            layer_output_delta_layer=0,
            layer_output_delta_phase="ffn",
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=40,
            routed_parity_tap=False,
            routed_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        report = script.build_report(
            [default, candidate],
            args,
            ["default", "full-stage5-router-simd-batch-shared-tiled"],
            "smoke",
        )
        md = script.render_markdown(report)
        summary = report["summary"]["layer_output_delta_tap"]
        comparison = summary["comparisons"][0]

        self.assertIn("layer_output_delta_tap: `True`", md)
        self.assertIn("layer_output_delta_tap_count: `2`", md)
        self.assertIn("layer_output_delta_max_abs: `1.00000000`", md)
        self.assertIn("## Layer Output Delta Tap", md)
        self.assertFalse(comparison["checksum_match"])
        self.assertEqual(comparison["max_abs_delta_idx"], 1)
        self.assertAlmostEqual(comparison["max_abs_delta"], 1.0)
        self.assertEqual(comparison["max_ulp_delta"], 64)
        self.assertEqual(comparison["differing_elems"], 1)

    def test_ffn_residual_delta_attribution_links_parity_rows(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-layer-output-delta-tap] call=0 position=0 cache_pos=0 "
            "path=chained phase_profile=0 layer=7 phase=ffn elems=2 "
            "checksum=aaaabbbbccccdddd bf16=3c00,bdc1\n"
        )
        candidate_output = (
            "[qwen36-layer-output-delta-tap] call=0 position=0 cache_pos=0 "
            "path=decode_batch phase_profile=0 layer=7 phase=ffn elems=2 "
            "checksum=ffffbbbbccccdddd bf16=3c00,bdc2\n"
        )

        default = row("default")
        default["layer_output_delta_taps"] = script.parse_layer_output_delta_taps(output)
        candidate = row("full-stage5-router-simd-batch")
        candidate["layer_output_delta_taps"] = script.parse_layer_output_delta_taps(
            candidate_output
        )
        candidate["decode_batch_shared_parity_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "shared_mid_argmax": 23,
                "host_shared_gate_at_mid_argmax": 1.0,
                "metal_shared_gate_at_mid_argmax": 1.125,
                "host_shared_up_at_mid_argmax": 2.0,
                "metal_shared_up_at_mid_argmax": 1.75,
                "host_shared_mid_at_argmax": 1.5,
                "metal_shared_mid_at_argmax": 1.0,
                "shared_out_argmax": 1,
                "shared_out_max_abs": 0.0000610351562,
                "host_shared_out_at_argmax": 0.0123901367,
                "metal_shared_out_at_argmax": 0.0123291016,
                "host_shared_down_acc_at_out_argmax": 0.125,
                "host_shared_gated_at_out_argmax": 0.0123901367,
                "host_shared_out_recomputed_at_argmax": 0.0123901367,
                "metal_mid_host_shared_down_acc_at_out_argmax": 0.125,
                "metal_mid_host_shared_gated_at_out_argmax": 0.0123901367,
                "metal_mid_host_shared_out_at_argmax": 0.0123291016,
            }
        ]
        candidate["decode_batch_routed_parity_taps"] = [
            {
                "position": 0,
                "layer": 7,
                "topk_idx_match": 1,
                "moe_out_argmax": 10,
                "moe_out_max_abs": 0.0000152587891,
                "host_moe_out_at_argmax": -0.00253295898,
                "metal_moe_out_at_argmax": -0.0025177002,
                "final_out_argmax": 1,
                "final_out_max_abs": 0.00048828125,
                "host_final_out_at_argmax": -0.0942382812,
                "metal_final_out_at_argmax": -0.0947265625,
            }
        ]

        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=False,
            metal_profile_phases=False,
            downstream_parity_tap=False,
            layer_output_tap=False,
            layer_output_delta_tap=True,
            layer_output_delta_position=0,
            layer_output_delta_layer=7,
            layer_output_delta_phase="ffn",
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            shared_parity_tap=False,
            shared_parity_tap_max_calls=40,
            routed_parity_tap=True,
            routed_parity_tap_max_calls=40,
            decode_batch_route_snapshot=False,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        report = script.build_report(
            [default, candidate],
            args,
            ["default", "full-stage5-router-simd-batch"],
            "smoke",
        )
        md = script.render_markdown(report)
        summary = report["summary"]["ffn_residual_delta_attribution"]
        first = summary["first"]

        self.assertEqual(summary["item_count"], 1)
        self.assertEqual(first["delta_idx"], 1)
        self.assertTrue(first["shared_out_argmax_matches_delta"])
        self.assertTrue(first["final_out_argmax_matches_delta"])
        self.assertEqual(first["source"], "shared_mid_to_shared_out_bf16_boundary")
        self.assertAlmostEqual(first["shared_out_delta_at_argmax"], -0.0000610351)
        self.assertAlmostEqual(first["shared_gate_delta_at_mid_argmax"], 0.125)
        self.assertAlmostEqual(first["shared_up_delta_at_mid_argmax"], -0.25)
        self.assertAlmostEqual(first["shared_mid_delta_at_argmax"], -0.5)
        self.assertAlmostEqual(first["metal_mid_host_shared_out_delta_vs_host"], -0.0000610351)
        self.assertAlmostEqual(first["metal_mid_host_shared_out_delta_vs_metal"], 0.0)
        self.assertFalse(first["metal_mid_host_shared_out_matches_host"])
        self.assertTrue(first["metal_mid_host_shared_out_matches_metal"])
        self.assertIn("ffn_residual_delta_first_source: `shared_mid_to_shared_out_bf16_boundary`", md)
        self.assertIn("## FFN Residual Delta Attribution", md)

    def test_router_parity_tap_selection_keeps_each_path(self):
        script = sweep_qwen36_fused_routed_int4
        tap_rows = []
        serial_row = {"prompt_id": "hello", "mode": "full-stage5-router"}
        simd_row = {"prompt_id": "hello", "mode": "full-stage5-router-simd"}
        for layer in range(40):
            tap_rows.append(
                (
                    serial_row,
                    {
                        "router_path": "serial",
                        "layer": layer,
                        "topk_idx_match": 1,
                    },
                )
            )
        for layer in range(40):
            tap_rows.append(
                (
                    simd_row,
                    {
                        "router_path": "simd",
                        "layer": layer,
                        "topk_idx_match": 1,
                    },
                )
            )

        selected = script.select_router_parity_tap_rows(tap_rows, limit=40)

        self.assertEqual(len(selected), 40)
        paths = {tap["router_path"] for _, tap in selected}
        self.assertEqual(paths, {"serial", "simd"})
        serial_layers = {tap["layer"] for _, tap in selected if tap["router_path"] == "serial"}
        simd_layers = {tap["layer"] for _, tap in selected if tap["router_path"] == "simd"}
        self.assertIn(39, serial_layers)
        self.assertIn(39, simd_layers)

    def test_decode_batch_route_snapshots_are_compared_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-decode-batch-route-snapshot] call=0 position=64 "
            "cache_pos=-1 router_path=serial phase_profile=1 layers=2 top_k=2 "
            "captured_layers=2 entries=4 first_layer=0 last_layer=1 "
            "checksum=123 routes=1,2;3,4\n"
        )

        snapshots = script.parse_decode_batch_route_snapshots(output)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["router_path"], "serial")
        self.assertEqual(snapshots[0]["checksum"], 123)
        self.assertEqual(snapshots[0]["routes"], "1,2;3,4")

        serial = row("full-stage5-router-batch-ffn-phases")
        serial["decode_batch_route_snapshots"] = snapshots
        simd = row("full-stage5-router-simd-batch-ffn-phases")
        simd["decode_batch_route_snapshots"] = [
            {
                **snapshots[0],
                "router_path": "simd",
                "checksum": 123,
                "routes": "1,2;3,4",
            }
        ]
        args = Namespace(
            max_new_tokens=1,
            context_size=64,
            metal_profile=True,
            metal_profile_phases=False,
            router_parity_tap=False,
            router_parity_tap_max_calls=40,
            decode_batch_route_snapshot=True,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_max_fused_wall_gpu_ratio=4.0,
            promotion_max_wait_gpu_ratio=4.0,
            promotion_require_profile=False,
        )
        report = script.build_report(
            [row("default"), serial, simd],
            args,
            [
                "default",
                "full-stage5-router-batch-ffn-phases",
                "full-stage5-router-simd-batch-ffn-phases",
            ],
            "smoke",
        )
        md = script.render_markdown(report)

        route_summary = report["summary"]["decode_batch_route_snapshot"]
        self.assertEqual(route_summary["snapshot_count"], 2)
        self.assertEqual(route_summary["mismatch_count"], 0)
        self.assertEqual(route_summary["paths"], ["serial", "simd"])
        self.assertIn("decode_batch_route_snapshot: `True`", md)
        self.assertIn("## Decode Batch Route Snapshot", md)
        self.assertIn(
            "| hello | full-stage5-router-simd-batch-ffn-phases | simd | 0 | 64 | 2 | 123 | full-stage5-router-batch-ffn-phases | serial | true |",
            md,
        )

        simd["decode_batch_route_snapshots"][0]["routes"] = "1,2;3,5"
        simd["decode_batch_route_snapshots"][0]["checksum"] = 124
        report = script.build_report(
            [row("default"), serial, simd],
            args,
            [
                "default",
                "full-stage5-router-batch-ffn-phases",
                "full-stage5-router-simd-batch-ffn-phases",
            ],
            "smoke",
        )
        self.assertEqual(
            report["summary"]["decode_batch_route_snapshot"]["mismatch_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
