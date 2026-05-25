from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_route_residency.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_route_residency", SCRIPT)
sweep_qwen36_route_residency = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_route_residency
SPEC.loader.exec_module(sweep_qwen36_route_residency)


SAMPLE_OUTPUT = """
Generated 4 tokens (1 prompt + 4 new). EOS: no (max_new_tokens hit).
  Generated ids: [11, 353, 599, 264]
[qwen36-route-profile] calls=160 layers=40 assignments=1280 unique_layer_experts=742 adjacent_hits=512 adjacent_total=1280 adjacent_hit_rate=0.400000 dropped_calls=0
[qwen36-route-cache-sim] scope=per_layer_lru capacity=2 hits=6 misses=1274 hit_rate=0.004688
[qwen36-route-cache-sim] scope=per_layer_lru capacity=4 hits=32 misses=1248 hit_rate=0.025000
[qwen36-route-cache-sim] scope=per_layer_lru capacity=8 hits=294 misses=986 hit_rate=0.229688
[qwen36-route-cache-sim] scope=per_layer_lru capacity=16 hits=419 misses=861 hit_rate=0.327344
[qwen36-route-cache-sim] scope=per_layer_lru capacity=64 hits=650 misses=630 hit_rate=0.507812
[qwen36-route-topn] scope=per_layer_oracle_topn capacity=2 covered=246 total=1280 coverage=0.192188
[qwen36-route-topn] scope=per_layer_oracle_topn capacity=4 covered=440 total=1280 coverage=0.343750
[qwen36-route-topn] scope=per_layer_oracle_topn capacity=8 covered=737 total=1280 coverage=0.575781
[qwen36-route-topn] scope=per_layer_oracle_topn capacity=16 covered=1065 total=1280 coverage=0.832031
[qwen36-route-topn] scope=per_layer_oracle_topn capacity=64 covered=1280 total=1280 coverage=1.000000
[metal-profile] calls=2 total_ms=12.000 native_ms=10.000 host_ms=2.000
[metal-profile-op] op=command_buffer_wait path=host calls=1 mean_ms=2.0000 total_ms=2.000 max_ms=2.000
"""


class Qwen36RouteResidencySweepTests(unittest.TestCase):
    def test_parse_route_rows(self):
        script = sweep_qwen36_route_residency

        route = script.parse_route_profile(SAMPLE_OUTPUT)
        cache = script.parse_route_cache_sims(SAMPLE_OUTPUT)
        topn = script.parse_route_topn_sims(SAMPLE_OUTPUT)

        self.assertEqual(script.parse_generated_ids(SAMPLE_OUTPUT), [11, 353, 599, 264])
        self.assertEqual(route["calls"], 160)
        self.assertAlmostEqual(route["adjacent_hit_rate"], 0.4)
        self.assertEqual(cache[-1]["capacity"], 64)
        self.assertAlmostEqual(cache[-1]["hit_rate"], 0.507812)
        self.assertEqual(topn[3]["capacity"], 16)
        self.assertAlmostEqual(topn[3]["coverage"], 0.832031)

    def test_build_summary_aggregates_capacity_rows(self):
        script = sweep_qwen36_route_residency
        rows = [
            {
                "status": "ok",
                "route_profile": script.parse_route_profile(SAMPLE_OUTPUT),
                "cache_sims": script.parse_route_cache_sims(SAMPLE_OUTPUT),
                "topn_sims": script.parse_route_topn_sims(SAMPLE_OUTPUT),
            },
            {
                "status": "ok",
                "route_profile": {
                    "calls": 40,
                    "assignments": 320,
                    "adjacent_hits": 160,
                    "adjacent_total": 320,
                    "dropped_calls": 0,
                    "unique_layer_experts": 160,
                },
                "cache_sims": [
                    {"capacity": 64, "hits": 160, "misses": 160, "hit_rate": 0.5}
                ],
                "topn_sims": [
                    {"capacity": 64, "covered": 320, "total": 320, "coverage": 1.0}
                ],
            },
        ]

        summary = script.build_summary(rows)

        self.assertEqual(summary["measured_count"], 2)
        self.assertAlmostEqual(summary["route_profile"]["adjacent_hit_rate"], 672 / 1600)
        capacity64_lru = [row for row in summary["cache_sims"] if row["capacity"] == 64][0]
        self.assertAlmostEqual(capacity64_lru["hit_rate"], 810 / 1600)
        capacity64_topn = [row for row in summary["topn_sims"] if row["capacity"] == 64][0]
        self.assertAlmostEqual(capacity64_topn["coverage"], 1.0)
        self.assertTrue(summary["decision_gate"]["passed"])
        self.assertEqual(
            summary["decision_gate"]["recommendation"],
            "prototype_larger_lru_resident_cache",
        )

    def test_decision_gate_prefers_static_when_lru_fails(self):
        script = sweep_qwen36_route_residency
        rows = [
            {
                "status": "ok",
                "route_profile": {"calls": 1, "assignments": 8},
                "cache_sims": [{"capacity": 16, "hits": 3, "misses": 5, "hit_rate": 0.375}],
                "topn_sims": [
                    {"capacity": 16, "covered": 7, "total": 8, "coverage": 0.875}
                ],
            }
        ]

        summary = script.build_summary(rows)

        self.assertTrue(summary["decision_gate"]["passed"])
        self.assertEqual(
            summary["decision_gate"]["recommendation"],
            "prototype_static_resident_table",
        )

    def test_decision_gate_prefers_fused_int4_when_residency_is_weak(self):
        script = sweep_qwen36_route_residency
        rows = [
            {
                "status": "ok",
                "route_profile": {"calls": 1, "assignments": 8},
                "cache_sims": [{"capacity": 16, "hits": 1, "misses": 7, "hit_rate": 0.125}],
                "topn_sims": [
                    {"capacity": 16, "covered": 4, "total": 8, "coverage": 0.5}
                ],
            }
        ]

        summary = script.build_summary(rows)

        self.assertFalse(summary["decision_gate"]["passed"])
        self.assertEqual(
            summary["decision_gate"]["recommendation"],
            "prefer_fused_routed_int4",
        )

    def test_row_from_output_marks_missing_profile(self):
        script = sweep_qwen36_route_residency

        row = script.row_from_output(
            "Generated ids: [1]",
            "p",
            "prompt",
            "ok",
            0,
            1.0,
            ["supersonic"],
            {},
        )

        self.assertEqual(row["status"], "missing_route_profile")

    def test_render_markdown_includes_decision_rows(self):
        script = sweep_qwen36_route_residency
        row = script.row_from_output(
            SAMPLE_OUTPUT,
            "profiling",
            "prompt",
            "ok",
            0,
            12.0,
            ["supersonic"],
            {},
        )
        report = script.build_report(
            [row],
            "smoke",
            [2, 4, 8, 16, 64],
            metal_profile=True,
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Route Residency Sweep", md)
        self.assertIn("decision_gate_recommendation", md)
        self.assertIn("| profiling | ok | 11,353,599,264 | 40.0% |", md)
        self.assertIn("| static_topn | 16 | true | 83.2% | - |", md)


if __name__ == "__main__":
    unittest.main()
