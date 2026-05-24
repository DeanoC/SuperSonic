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
                    "calls": 1,
                    "mean_ms": fused,
                    "total_ms": fused,
                    "max_ms": fused,
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
            script.parse_modes("baseline,direct,gpu-pack,gpack,stage5,native-stage5"),
            ["default", "direct-gather", "gpu-pack", "full-stage5"],
        )

    def test_build_env_overrides_for_fused_modes(self):
        script = sweep_qwen36_fused_routed_int4
        args = Namespace(metal_profile=True)

        direct = script.build_env_overrides(args, "direct-gather")
        gpu_pack = script.build_env_overrides(args, "gpu-pack")
        full_stage5 = script.build_env_overrides(args, "full-stage5")
        default = script.build_env_overrides(args, "default")

        self.assertEqual(direct["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertEqual(
            direct["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5"],
            "1",
        )
        self.assertNotIn(
            "SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5",
            default,
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

    def test_promotion_gate_passes_only_real_improvements(self):
        script = sweep_qwen36_fused_routed_int4
        rows = [
            row("default"),
            row("direct-gather", headline=90.0, ffn=40.0, wait=9.0, fused=12.0),
            row("gpu-pack", headline=120.0, ffn=70.0, wait=30.0, fused=25.0),
            row("full-stage5", headline=300.0, ffn=250.0, wait=80.0, fused=60.0),
        ]

        gate = script.build_promotion_gate(
            rows,
            ["default", "direct-gather", "gpu-pack", "full-stage5"],
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["direct-gather"])
        failures = gate["candidates"][1]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", failures)
        self.assertIn("prompt_hello:ffn_not_improved", failures)
        full_failures = gate["candidates"][2]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", full_failures)
        self.assertIn("prompt_hello:ffn_not_improved", full_failures)

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
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
        )
        report = script.build_report(
            [
                row("default"),
                row("direct-gather", headline=90.0, ffn=40.0, wait=9.0, fused=12.0),
            ],
            args,
            ["default", "direct-gather"],
            "smoke",
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Fused Routed INT4 Sweep", md)
        self.assertIn("promotion_gate_passed: `True`", md)
        self.assertIn("| direct-gather | true | - |", md)


if __name__ == "__main__":
    unittest.main()
