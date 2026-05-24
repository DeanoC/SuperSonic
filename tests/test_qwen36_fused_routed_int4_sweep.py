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
                "baseline,direct,direct-defer,defer-direct-wait,gpu-pack,gpack,stage5,native-stage5,router,router-batch,batch-router,router-batch-phases,batch-router-phases,router-defer"
            ),
            [
                "default",
                "direct-gather",
                "direct-defer-wait",
                "gpu-pack",
                "full-stage5",
                "full-stage5-router",
                "full-stage5-router-batch",
                "full-stage5-router-batch-phases",
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
        full_stage5_router_batch = script.build_env_overrides(args, "full-stage5-router-batch")
        full_stage5_router_batch_phases = script.build_env_overrides(
            args,
            "full-stage5-router-batch-phases",
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
            full_stage5_router_batch["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"],
            "1",
        )
        self.assertEqual(
            full_stage5_router_batch["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"],
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

        tap_args = Namespace(
            metal_profile=False,
            metal_profile_phases=False,
            router_parity_tap=True,
            router_parity_tap_max_calls=3,
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
        batch_phases = script.build_command(args, "Hello", "full-stage5-router-batch-phases")

        self.assertIn("--emit-stage-timings", normal)
        self.assertNotIn("--emit-stage-timings", batch)
        self.assertNotIn("--emit-stage-timings", batch_phases)
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

    def test_router_parity_tap_rows_are_parsed_and_rendered(self):
        script = sweep_qwen36_fused_routed_int4
        output = (
            "[qwen36-ffn-router-parity] call=0 layer=5 topk_idx_match=0 "
            "workspace_idx_match=0 output_idx_match=0 h_norm_max_abs=1.25000000e-01 "
            "h_norm_argmax=17 logits_max_abs=2.50000000e-01 logits_argmax=33 "
            "topk_weight_max_abs=3.12500000e-02 topk_weight_argmax=2 "
            "host_idx=1,2,3 workspace_idx=1,4,3 output_idx=1,4,3 "
            "host_w=0.50000000,0.25000000,0.12500000 "
            "metal_w=0.46875000,0.28125000,0.12500000\n"
        )

        taps = script.parse_router_parity_taps(output)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0]["layer"], 5)
        self.assertEqual(taps[0]["topk_idx_match"], 0)
        self.assertEqual(taps[0]["host_idx"], "1,2,3")
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
        self.assertIn("## Router Parity Tap", md)
        self.assertIn("| hello | full-stage5-router | 5 | false |", md)


if __name__ == "__main__":
    unittest.main()
