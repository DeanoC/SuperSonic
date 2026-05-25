import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "probe_qwen36_mps_resident_table.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen36_mps_resident_table", SCRIPT)
probe_qwen36_mps_resident_table = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = probe_qwen36_mps_resident_table
SPEC.loader.exec_module(probe_qwen36_mps_resident_table)


PILOT_OUTPUT = """
[qwen36-moe mps-expert-pilot] status=ok hidden=2048 moe_intermediate=512 top_k=8 iterations=100 gate_up_ms=0.619 down_ms=0.433 gate_up_tflops=5.423 down_tflops=3.878
"""

STATIC_REPORT = {
    "schema": "qwen36-static-topn-mps-probe-v2",
    "model": "qwen3.6-35b-a3b",
    "layers": 2,
    "hidden": 8,
    "moe_intermediate": 4,
    "rows": [
        {
            "capacity": 2,
            "evaluation_static_topn": {
                "calls": 4,
                "assignments": 8,
                "covered": 2,
                "misses": 6,
                "coverage": 0.25,
                "full_hit_calls": 0,
                "fallback_calls": 4,
                "full_hit_call_rate": 0.0,
            },
            "resident_mps_rhs": {"total_gib": 0.01},
        },
        {
            "capacity": 4,
            "evaluation_static_topn": {
                "calls": 4,
                "assignments": 8,
                "covered": 6,
                "misses": 2,
                "coverage": 0.75,
                "full_hit_calls": 0,
                "fallback_calls": 4,
                "full_hit_call_rate": 0.0,
            },
            "resident_mps_rhs": {"total_gib": 0.02},
        },
    ],
}


class Qwen36MpsResidentTableProbeTests(unittest.TestCase):
    def test_parse_mps_expert_pilot(self):
        script = probe_qwen36_mps_resident_table
        pilot = script.parse_mps_expert_pilot(PILOT_OUTPUT)

        self.assertIsNotNone(pilot)
        self.assertEqual(pilot["status"], "ok")
        self.assertEqual(pilot["top_k"], 8)
        self.assertAlmostEqual(pilot["gate_up_ms"], 0.619)
        self.assertAlmostEqual(pilot["down_tflops"], 3.878)

    def test_cost_model_separates_full_hit_and_partial_hit_estimates(self):
        script = probe_qwen36_mps_resident_table
        pilot = {"status": "ok", "gate_up_ms": 1.0, "down_ms": 1.0}
        evaluation = STATIC_REPORT["rows"][1]["evaluation_static_topn"]

        cost = script.build_cost_model(
            evaluation,
            pilot,
            layers=2,
            top_k=2,
            baseline_ffn_ms=10.0,
        )

        self.assertEqual(cost["status"], "ok")
        self.assertAlmostEqual(cost["all_resident_mps_ms_per_token"], 4.0)
        self.assertAlmostEqual(cost["full_hit_only_ms_per_token_est"], 10.0)
        self.assertAlmostEqual(cost["partial_hit_optimistic_ms_per_token_est"], 5.5)
        self.assertAlmostEqual(cost["partial_hit_optimistic_ratio"], 0.55)

    def test_build_report_picks_partial_hit_capacity(self):
        script = probe_qwen36_mps_resident_table
        pilot = {"source": "manual", "status": "ok", "gate_up_ms": 1.0, "down_ms": 1.0}

        report = script.build_report(
            STATIC_REPORT,
            pilot,
            layers=2,
            hidden=8,
            moe_intermediate=4,
            top_k=2,
            baseline_ffn_ms=10.0,
            baseline_source="unit test",
        )

        self.assertEqual(report["schema"], script.SCHEMA)
        self.assertEqual(report["summary"]["best_partial_capacity"], 4)
        self.assertTrue(report["summary"]["viability_gate"]["passed"])
        self.assertEqual(
            report["summary"]["viability_gate"]["recommendation"],
            "prototype_partial_hit_resident_mps",
        )
        self.assertEqual(
            report["rows"][1]["recommendation"],
            "prototype_partial_hit_resident_mps",
        )
        md = script.render_markdown(report)
        self.assertIn("MPS Resident Table Probe", md)
        self.assertIn("Partial-hit optimistic", md)
        self.assertIn("viability_gate_passed", md)
        self.assertIn("Viability Gate", md)

    def test_viability_gate_rejects_large_or_low_coverage_candidates(self):
        script = probe_qwen36_mps_resident_table
        pilot = {"source": "manual", "status": "ok", "gate_up_ms": 1.0, "down_ms": 1.0}

        report = script.build_report(
            STATIC_REPORT,
            pilot,
            layers=2,
            hidden=8,
            moe_intermediate=4,
            top_k=2,
            baseline_ffn_ms=10.0,
            baseline_source="unit test",
            gate_max_rhs_gib=0.015,
            gate_max_ratio=0.90,
            gate_min_partial_coverage=0.50,
            gate_min_full_hit_call_rate=0.50,
        )

        gate = report["summary"]["viability_gate"]
        self.assertFalse(gate["passed"])
        failures = {
            failure
            for candidate in gate["candidates"]
            for failure in candidate["failures"]
        }
        self.assertIn("resident_rhs_too_large", failures)
        self.assertIn("coverage_below_threshold", failures)

    def test_viability_gate_requires_usable_pilot(self):
        script = probe_qwen36_mps_resident_table

        report = script.build_report(
            STATIC_REPORT,
            pilot=None,
            layers=2,
            hidden=8,
            moe_intermediate=4,
            top_k=2,
            baseline_ffn_ms=10.0,
            baseline_source="unit test",
        )

        gate = report["summary"]["viability_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["recommendation"], "measure_mps_pilot")
        self.assertIn(
            "missing_usable_mps_pilot",
            gate["candidates"][0]["failures"],
        )

    def test_pilot_env_sets_shape_controls(self):
        script = probe_qwen36_mps_resident_table
        args = script.parse_args([])
        env = script.build_pilot_env(args, {})

        self.assertEqual(env["SUPERSONIC_BACKENDS"], "metal")
        self.assertEqual(env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT"], "1")
        self.assertEqual(env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_TOP_K"], "8")
        self.assertEqual(env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_ITERS"], "100")


if __name__ == "__main__":
    unittest.main()
