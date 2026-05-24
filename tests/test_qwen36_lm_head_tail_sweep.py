from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_lm_head_tail.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_lm_head_tail", SCRIPT)
sweep_qwen36_lm_head_tail = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_lm_head_tail
SPEC.loader.exec_module(sweep_qwen36_lm_head_tail)


def row(
    mode: str,
    *,
    ids: list[int] | None = None,
    headline: float = 100.0,
    ffn: float = 50.0,
    full: float = 20.0,
    linear: float = 10.0,
    lm_head: float = 5.0,
    sample: float = 0.2,
    wait: float | None = 10.0,
    argmax: float | None = 0.3,
    copy_d2h: float | None = 1.0,
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
        if argmax is not None:
            entries.append(
                {
                    "op": "argmax_bf16",
                    "path": "native",
                    "calls": 1,
                    "mean_ms": argmax,
                    "total_ms": argmax,
                    "max_ms": argmax,
                }
            )
        metal_profile = {"summary": {"calls": len(entries)}, "entries": entries}
    hal_profile = None
    if copy_d2h is not None:
        hal_profile = {
            "summary": {"calls": 1},
            "entries": [
                {
                    "op": "copy_d2h",
                    "calls": 1,
                    "mean_ms": copy_d2h,
                    "total_ms": copy_d2h,
                    "max_ms": copy_d2h,
                    "total_bytes": 4,
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
        "stage_timings": {"lm_head_ms_avg": lm_head, "sample_ms_avg": sample},
        "chain_breakdown": {
            "ffn_ms_avg": ffn,
            "full_attn_ms_avg": full,
            "linear_attn_ms_avg": linear,
        },
        "metal_profile": metal_profile,
        "hal_profile": hal_profile,
        "argmax_bf16_ms": argmax,
        "copy_d2h_ms": copy_d2h,
        "command_buffer_wait_ms": wait,
        "wall_seconds": 1.0,
    }


class Qwen36LmHeadTailSweepTests(unittest.TestCase):
    def test_parse_modes_dedupes_aliases(self):
        script = sweep_qwen36_lm_head_tail

        self.assertEqual(
            script.parse_modes("baseline,full-logits,top1,gpu-argmax"),
            ["default", "gpu-argmax"],
        )

    def test_build_env_overrides_for_gpu_argmax(self):
        script = sweep_qwen36_lm_head_tail
        args = Namespace(metal_profile=True)

        default = script.build_env_overrides(args, "default")
        gpu_argmax = script.build_env_overrides(args, "gpu-argmax")

        self.assertEqual(default["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertNotIn("SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX", default)
        self.assertEqual(
            gpu_argmax["SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX"],
            "1",
        )

    def test_promotion_gate_passes_only_lm_head_improvements(self):
        script = sweep_qwen36_lm_head_tail
        rows = [
            row("default"),
            row("gpu-argmax", headline=90.0, lm_head=4.0, sample=0.01, wait=9.0),
        ]

        gate = script.build_promotion_gate(rows, ["default", "gpu-argmax"])

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["gpu-argmax"])

    def test_promotion_gate_rejects_id_mismatch_and_missing_profile(self):
        script = sweep_qwen36_lm_head_tail
        rows = [
            row("default"),
            row("gpu-argmax", ids=[99], headline=90.0, lm_head=4.0, wait=None),
        ]

        gate = script.build_promotion_gate(rows, ["default", "gpu-argmax"])

        self.assertFalse(gate["passed"])
        failures = gate["candidates"][0]["failures"]
        self.assertIn("prompt_hello:generated_ids_mismatch", failures)
        self.assertIn("prompt_hello:missing_command_buffer_wait_profile", failures)

    def test_render_markdown_includes_gate_rows(self):
        script = sweep_qwen36_lm_head_tail
        args = Namespace(
            max_new_tokens=2,
            context_size=64,
            metal_profile=True,
            promotion_max_headline_ratio=0.999,
            promotion_max_lm_head_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
            promotion_min_generated_tokens=2,
        )
        report = script.build_report(
            [
                row("default"),
                row("gpu-argmax", headline=90.0, lm_head=4.0, sample=0.01, wait=9.0),
            ],
            args,
            ["default", "gpu-argmax"],
            "smoke",
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Lm-Head Tail Sweep", md)
        self.assertIn("promotion_gate_passed: `True`", md)
        self.assertIn("| gpu-argmax | true | - |", md)


if __name__ == "__main__":
    unittest.main()
