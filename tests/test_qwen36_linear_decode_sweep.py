from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_linear_decode.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_linear_decode", SCRIPT)
sweep_qwen36_linear_decode = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_linear_decode
SPEC.loader.exec_module(sweep_qwen36_linear_decode)


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
    stage5: float | None = 12.0,
    copy_d2d: float | None = 1.0,
    status: str = "ok",
) -> dict:
    metal_profile = None
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
        if stage5 is not None:
            entries.append(
                {
                    "op": "qwen36_linear_int4_stage5",
                    "path": "native",
                    "calls": 1,
                    "mean_ms": stage5,
                    "total_ms": stage5,
                    "max_ms": stage5,
                }
            )
        metal_profile = {"summary": {"calls": len(entries)}, "entries": entries}
    hal_profile = None
    if copy_d2d is not None:
        hal_profile = {
            "summary": {"calls": 1},
            "entries": [
                {
                    "op": "copy_d2d",
                    "calls": 1,
                    "mean_ms": copy_d2d,
                    "total_ms": copy_d2d,
                    "max_ms": copy_d2d,
                    "total_bytes": 4096,
                }
            ],
        }
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
        "metal_profile": metal_profile,
        "hal_profile": hal_profile,
        "linear_stage5_ms": stage5,
        "linear_subdispatch_ms": stage5,
        "linear_host_ms": None,
        "copy_d2d_ms": copy_d2d,
        "command_buffer_wait_ms": wait,
        "wall_seconds": 1.0,
    }


class Qwen36LinearDecodeSweepTests(unittest.TestCase):
    def test_parse_modes_dedupes_aliases(self):
        script = sweep_qwen36_linear_decode

        self.assertEqual(
            script.parse_modes("baseline,no-direct,old-handoff,native-off,host"),
            ["default", "direct-off", "host-linear"],
        )

    def test_build_env_overrides_for_linear_modes(self):
        script = sweep_qwen36_linear_decode
        args = Namespace(metal_profile=True)

        default = script.build_env_overrides(args, "default")
        direct_off = script.build_env_overrides(args, "direct-off")
        host = script.build_env_overrides(args, "host-linear")

        self.assertEqual(default["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertNotIn("SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT", default)
        self.assertEqual(
            direct_off["SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT"],
            "1",
        )
        self.assertEqual(
            host["SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT"],
            "1",
        )
        self.assertEqual(
            host["SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5"],
            "1",
        )

    def test_promotion_gate_passes_only_linear_improvements(self):
        script = sweep_qwen36_linear_decode
        rows = [
            row("default"),
            row("direct-off", headline=90.0, linear=15.0, wait=9.0, stage5=9.0),
            row("host-linear", headline=120.0, linear=30.0, wait=20.0, stage5=None),
        ]

        gate = script.build_promotion_gate(
            rows,
            ["default", "direct-off", "host-linear"],
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["direct-off"])
        failures = gate["candidates"][1]["failures"]
        self.assertIn("prompt_hello:headline_not_improved", failures)
        self.assertIn("prompt_hello:linear_attn_not_improved", failures)

    def test_promotion_gate_rejects_id_mismatch_and_missing_profile(self):
        script = sweep_qwen36_linear_decode
        rows = [
            row("default"),
            row("direct-off", ids=[99], headline=90.0, linear=15.0, wait=None),
        ]

        gate = script.build_promotion_gate(rows, ["default", "direct-off"])

        self.assertFalse(gate["passed"])
        failures = gate["candidates"][0]["failures"]
        self.assertIn("prompt_hello:generated_ids_mismatch", failures)
        self.assertIn("prompt_hello:missing_command_buffer_wait_profile", failures)

    def test_render_markdown_includes_gate_rows(self):
        script = sweep_qwen36_linear_decode
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            promotion_max_headline_ratio=0.999,
            promotion_max_linear_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
            promotion_min_generated_tokens=2,
        )
        report = script.build_report(
            [
                row("default"),
                row("direct-off", headline=90.0, linear=15.0, wait=9.0, stage5=9.0),
            ],
            args,
            ["default", "direct-off"],
            "smoke",
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Linear Decode Sweep", md)
        self.assertIn("promotion_gate_passed: `True`", md)
        self.assertIn("| direct-off | true | - |", md)


if __name__ == "__main__":
    unittest.main()
