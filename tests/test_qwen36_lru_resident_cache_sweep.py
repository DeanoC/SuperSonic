from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_lru_resident_cache.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_lru_resident_cache", SCRIPT)
lru_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = lru_sweep
SPEC.loader.exec_module(lru_sweep)


def row(mode: str, decode: float, ffn: float, wait: float) -> dict:
    return {
        "status": "ok",
        "prompt_id": "hello",
        "mode": mode,
        "generated_ids": [11, 353],
        "result": {"ms_per_step": decode},
        "stage_timings": {"lm_head_ms_avg": 5.0},
        "chain_breakdown": {
            "ffn_ms_avg": ffn,
            "full_attn_ms_avg": 10.0,
            "linear_attn_ms_avg": 20.0,
        },
        "expert_residency": {
            "slot_hit_rate": 0.66 if mode != "default" else None,
            "copied_bytes": 1024,
            "evictions": 1,
        },
        "metal_profile": {
            "entries": [{"op": "command_buffer_wait", "total_ms": wait}]
        },
        "hal_profile": {"summary": {"total_ms": 1.0}},
        "wall_seconds": 1.0,
    }


class Qwen36LruResidentCacheSweepTests(unittest.TestCase):
    def test_parse_capacities_dedupes_and_validates(self):
        self.assertEqual(lru_sweep.parse_capacities("32,64,32"), [32, 64])
        with self.assertRaises(Exception):
            lru_sweep.parse_capacities("0")
        with self.assertRaises(Exception):
            lru_sweep.parse_capacities("nope")

    def test_build_report_exposes_promotion_gate(self):
        args = SimpleNamespace(
            max_new_tokens=2,
            context_size=64,
            promotion_max_headline_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
        )
        rows = [
            row("default", 100.0, 50.0, 100.0),
            row("lru-hotset-64", 90.0, 45.0, 102.0),
        ]

        report = lru_sweep.build_report(rows, args, [64], "smoke")

        self.assertEqual(report["schema"], lru_sweep.SCHEMA)
        self.assertEqual(report["modes"], ["default", "lru-hotset-64"])
        gate = report["summary"]["promotion_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["lru-hotset-64"])
        self.assertIn("thresholds", gate)
        self.assertIn("lru-hotset-64", lru_sweep.render_markdown(report))


if __name__ == "__main__":
    unittest.main()
