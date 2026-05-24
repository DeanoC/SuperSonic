from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_full_decode.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_full_decode", SCRIPT)
sweep_qwen36_full_decode = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_full_decode
SPEC.loader.exec_module(sweep_qwen36_full_decode)


def row(
    mode: str,
    *,
    ids: list[int] | None = None,
    headline: float = 100.0,
    ffn: float = 50.0,
    full: float = 20.0,
    linear: float = 10.0,
    lm_head: float = 5.0,
    wait: float | None = 10.0,
    full_host: float | None = 8.0,
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
        if full_host is not None:
            entries.append(
                {
                    "op": "qwen36_full_attn_input_norm",
                    "path": "host",
                    "calls": 1,
                    "mean_ms": full_host,
                    "total_ms": full_host,
                    "max_ms": full_host,
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
        "full_attn_host_ms": full_host,
        "copy_d2d_ms": copy_d2d,
        "command_buffer_wait_ms": wait,
        "wall_seconds": 1.0,
    }


class Qwen36FullDecodeSweepTests(unittest.TestCase):
    def test_parse_modes_dedupes_aliases(self):
        script = sweep_qwen36_full_decode

        self.assertEqual(
            script.parse_modes("baseline,no-direct,old-handoff,direct,direct-on"),
            ["default", "direct"],
        )

    def test_build_env_overrides_for_full_modes(self):
        script = sweep_qwen36_full_decode
        args = Namespace(metal_profile=True)

        default = script.build_env_overrides(args, "default")
        direct = script.build_env_overrides(args, "direct")

        self.assertEqual(default["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertNotIn("SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_DECODE_DIRECT", default)
        self.assertEqual(
            direct["SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_DECODE_DIRECT"],
            "1",
        )

    def test_promotion_gate_passes_only_full_attention_improvements(self):
        script = sweep_qwen36_full_decode
        rows = [
            row("default"),
            row("direct", headline=90.0, full=15.0, wait=9.0, full_host=7.0),
        ]

        gate = script.build_promotion_gate(rows, ["default", "direct"])

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["direct"])

    def test_promotion_gate_rejects_id_mismatch_and_missing_profile(self):
        script = sweep_qwen36_full_decode
        rows = [
            row("default"),
            row("direct", ids=[99], headline=90.0, full=15.0, wait=None),
        ]

        gate = script.build_promotion_gate(rows, ["default", "direct"])

        self.assertFalse(gate["passed"])
        failures = gate["candidates"][0]["failures"]
        self.assertIn("prompt_hello:generated_ids_mismatch", failures)
        self.assertIn("prompt_hello:missing_command_buffer_wait_profile", failures)

    def test_render_markdown_includes_gate_rows(self):
        script = sweep_qwen36_full_decode
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            promotion_max_headline_ratio=0.999,
            promotion_max_full_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
            promotion_min_generated_tokens=2,
        )
        report = script.build_report(
            [
                row("default"),
                row("direct", headline=90.0, full=15.0, wait=9.0, full_host=7.0),
            ],
            args,
            ["default", "direct"],
            "smoke",
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Full-Attention Decode Sweep", md)
        self.assertIn("promotion_gate_passed: `True`", md)
        self.assertIn("| direct | true | - |", md)


if __name__ == "__main__":
    unittest.main()
