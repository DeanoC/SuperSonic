import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "probe_qwen36_static_topn.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen36_static_topn", SCRIPT)
probe_qwen36_static_topn = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = probe_qwen36_static_topn
SPEC.loader.exec_module(probe_qwen36_static_topn)


CALIBRATION_OUTPUT = """
[qwen36-route-topn-layer] scope=per_layer_oracle_topn capacity=2 layer=0 experts=1,2 counts=3,2 covered=5 total=6 coverage=0.833333
[qwen36-route-topn-layer] scope=per_layer_oracle_topn capacity=2 layer=1 experts=4,6 counts=2,2 covered=4 total=6 coverage=0.666667
[qwen36-route-topn-layer] scope=per_layer_oracle_topn capacity=4 layer=0 experts=1,2,5 counts=3,2,1 covered=6 total=6 coverage=1.000000
[qwen36-route-topn-layer] scope=per_layer_oracle_topn capacity=4 layer=1 experts=4,6,3,7 counts=2,2,1,1 covered=6 total=6 coverage=1.000000
"""

EVALUATION_OUTPUT = """
[qwen36-route-call] call_idx=0 layer=0 experts=1,2
[qwen36-route-call] call_idx=1 layer=1 experts=6,7
[qwen36-route-call] call_idx=2 layer=0 experts=5,8
"""


class Qwen36StaticTopNProbeTests(unittest.TestCase):
    def test_parse_route_profile_rows(self):
        rows = probe_qwen36_static_topn.parse_topn_layers(CALIBRATION_OUTPUT)
        calls = probe_qwen36_static_topn.parse_route_calls(EVALUATION_OUTPUT)

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["experts"], [1, 2])
        self.assertEqual(rows[0]["counts"], [3, 2])
        self.assertEqual(calls[1]["layer"], 1)
        self.assertEqual(calls[1]["experts"], [6, 7])

    def test_static_table_evaluation_counts_full_hits_and_misses(self):
        rows = probe_qwen36_static_topn.parse_topn_layers(CALIBRATION_OUTPUT)
        calls = probe_qwen36_static_topn.parse_route_calls(EVALUATION_OUTPUT)
        tables = probe_qwen36_static_topn.build_static_tables(rows, [2], layers=2)
        result = probe_qwen36_static_topn.evaluate_static_table(calls, tables[2])

        self.assertEqual(result["assignments"], 6)
        self.assertEqual(result["covered"], 3)
        self.assertEqual(result["misses"], 3)
        self.assertEqual(result["full_hit_calls"], 1)
        self.assertEqual(result["fallback_calls"], 2)
        self.assertAlmostEqual(result["coverage"], 3 / 6)
        self.assertEqual(result["worst_layer"]["layer"], 0)

    def test_resident_mps_rhs_size_uses_gate_up_and_down(self):
        estimate = probe_qwen36_static_topn.estimate_resident_mps_rhs_bytes(
            layers=40,
            capacity=16,
            hidden=2048,
            moe_intermediate=512,
        )

        self.assertEqual(estimate["bytes_per_expert"], 6_291_456)
        self.assertAlmostEqual(estimate["total_gib"], 3.75)

    def test_resident_native_int4_size_includes_sidecars(self):
        estimate = probe_qwen36_static_topn.estimate_resident_native_int4_bytes(
            layers=40,
            capacity=64,
            hidden=2048,
            moe_intermediate=512,
            group_size=128,
        )

        self.assertEqual(estimate["bytes_per_expert"], 1_573_632)
        self.assertAlmostEqual(estimate["total_gib"], 3.7518310546875)

    def test_build_report_summarizes_capacity_rows(self):
        report = probe_qwen36_static_topn.build_report(
            CALIBRATION_OUTPUT,
            EVALUATION_OUTPUT,
            capacities=[2, 4],
            layers=2,
            hidden=8,
            moe_intermediate=4,
        )

        self.assertEqual(report["schema"], probe_qwen36_static_topn.SCHEMA)
        self.assertEqual(report["calibration"]["topn_layer_rows"], 4)
        self.assertEqual(report["evaluation"]["route_calls"], 3)
        self.assertEqual(report["rows"][0]["capacity"], 2)
        self.assertEqual(report["static_tables"]["2"]["layers"][0]["experts"], [1, 2])
        self.assertEqual(report["static_tables"]["4"]["layers"][1]["experts"], [4, 6, 3, 7])
        self.assertGreater(
            report["rows"][1]["evaluation_static_topn"]["coverage"],
            report["rows"][0]["evaluation_static_topn"]["coverage"],
        )
        md = probe_qwen36_static_topn.render_markdown(report)
        self.assertIn("Static Top-N MPS Probe", md)
        self.assertIn("Native INT4 GiB", md)
        self.assertIn("MPS RHS GiB", md)


if __name__ == "__main__":
    unittest.main()
