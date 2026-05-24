from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "summarize_qwen36_sota_gates.py"
SPEC = importlib.util.spec_from_file_location("summarize_qwen36_sota_gates", SCRIPT)
summarize_qwen36_sota_gates = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = summarize_qwen36_sota_gates
SPEC.loader.exec_module(summarize_qwen36_sota_gates)


def write_report(path: Path, schema: str, gate_key: str, gate: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "summary": {
                    gate_key: gate,
                },
            }
        )
        + "\n"
    )
    return path


class Qwen36SotaGateSummaryTests(unittest.TestCase):
    def paths_in(self, root: Path) -> dict[str, Path]:
        return {
            "batched_prefill_variants": root / "prefill.json",
            "static_topn_runtime": root / "static.json",
            "mps_resident_table": root / "mps.json",
            "mtp_acceptance": root / "mtp.json",
        }

    def write_default_reports(
        self,
        root: Path,
        prefill_gate: dict | None = None,
        static_gate: dict | None = None,
        mps_gate: dict | None = None,
        mtp_gate: dict | None = None,
    ) -> dict[str, Path]:
        script = summarize_qwen36_sota_gates
        paths = self.paths_in(root)
        write_report(
            paths["batched_prefill_variants"],
            script.GATE_SPECS[0].expected_schema,
            "promotion_gate",
            prefill_gate
            if prefill_gate is not None
            else {
                "passed": False,
                "candidates": [
                    {"mode": "router-topk", "passed": False, "failures": ["ffn_not_improved"]}
                ],
            },
        )
        write_report(
            paths["static_topn_runtime"],
            script.GATE_SPECS[1].expected_schema,
            "promotion_gate",
            static_gate
            if static_gate is not None
            else {
                "passed": False,
                "candidates": [
                    {"mode": "static", "passed": False, "failures": ["headline_not_improved"]}
                ],
            },
        )
        write_report(
            paths["mps_resident_table"],
            script.GATE_SPECS[2].expected_schema,
            "viability_gate",
            mps_gate
            if mps_gate is not None
            else {
                "passed": False,
                "recommendation": "reject_resident_mps_for_now",
                "candidates": [
                    {
                        "kind": "partial_hit_optimistic",
                        "capacity": 64,
                        "passed": False,
                        "failures": ["estimate_not_fast_enough"],
                    }
                ],
            },
        )
        write_report(
            paths["mtp_acceptance"],
            script.GATE_SPECS[3].expected_schema,
            "promotion_gate",
            mtp_gate
            if mtp_gate is not None
            else {
                "passed": False,
                "failures": ["acceptance_below_threshold"],
            },
        )
        return paths

    def test_missing_reports_are_rows_by_default(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            report = script.build_report(self.paths_in(Path(tmp)))

        self.assertEqual(report["schema"], script.SCHEMA)
        self.assertEqual(report["summary"]["status_counts"], {"missing": 4})
        self.assertFalse(report["summary"]["all_inputs_ok"])
        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "run_or_refresh_gate_reports",
        )
        self.assertEqual(report["rows"][0]["recommendation_action"], "run_harness")

    def test_runtime_promotion_pass_wins_next_action(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(
                Path(tmp),
                prefill_gate={
                    "passed": True,
                    "passed_modes": ["router-topk"],
                    "candidates": [{"mode": "router-topk", "passed": True}],
                },
                mps_gate={
                    "passed": True,
                    "recommendation": "prototype_partial_hit_resident_mps",
                    "candidates": [
                        {
                            "kind": "partial_hit_optimistic",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
            )
            report = script.build_report(paths)

        self.assertTrue(report["summary"]["all_inputs_ok"])
        self.assertIn("batched_prefill_variants", report["summary"]["passed_gate_ids"])
        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "prepare_runtime_promotion:batched_prefill_variants:router-topk",
        )
        prefill = report["rows"][0]
        self.assertEqual(prefill["passed_candidates"], ["router-topk"])
        self.assertEqual(
            prefill["recommendation_action"],
            "prepare_runtime_promotion:router-topk",
        )

    def test_viability_pass_is_next_when_runtime_gates_fail(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(
                Path(tmp),
                mps_gate={
                    "passed": True,
                    "recommendation": "prototype_partial_hit_resident_mps",
                    "candidates": [
                        {
                            "kind": "partial_hit_optimistic",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
            )
            report = script.build_report(paths)

        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "prototype_partial_hit_resident_mps",
        )
        self.assertEqual(report["rows"][2]["passed_candidates"], ["partial_hit_optimistic:64"])

    def test_malformed_schema_mismatch_and_missing_gate_are_reported(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.paths_in(root)
            paths["batched_prefill_variants"].write_text("{nope")
            write_report(
                paths["static_topn_runtime"],
                "old-schema",
                "promotion_gate",
                {"passed": True, "passed_modes": ["static"]},
            )
            paths["mps_resident_table"].write_text(
                json.dumps({"schema": script.GATE_SPECS[2].expected_schema, "summary": {}})
                + "\n"
            )
            write_report(
                paths["mtp_acceptance"],
                script.GATE_SPECS[3].expected_schema,
                "promotion_gate",
                {"passed": False, "failures": ["acceptance_below_threshold"]},
            )
            report = script.build_report(paths)

        statuses = {row["gate_id"]: row["status"] for row in report["rows"]}
        self.assertEqual(statuses["batched_prefill_variants"], "malformed")
        self.assertEqual(statuses["static_topn_runtime"], "schema_mismatch")
        self.assertEqual(statuses["mps_resident_table"], "missing_gate")
        self.assertEqual(statuses["mtp_acceptance"], "ok")
        self.assertEqual(report["summary"]["input_failure_count"], 3)
        self.assertIn("static_topn_runtime", report["summary"]["next_action"]["blocked_reason"])

    def test_render_markdown_and_require_exit(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.write_default_reports(root)
            report = script.build_report(paths)
            md = script.render_markdown(report)

            self.assertIn("Qwen3.6 Metal SOTA Gate Summary", md)
            self.assertIn("| Batched-prefill variants | ok | false |", md)
            self.assertIn("keep_default_lane_and_select_next_measured_bottleneck", md)

            missing = root / "missing.json"
            out_json = root / "summary.json"
            out_md = root / "summary.md"
            rc = script.main(
                [
                    "--prefill-json",
                    str(missing),
                    "--static-runtime-json",
                    str(paths["static_topn_runtime"]),
                    "--mps-json",
                    str(paths["mps_resident_table"]),
                    "--mtp-json",
                    str(paths["mtp_acceptance"]),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--require",
                ]
            )

            self.assertEqual(rc, 1)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())


if __name__ == "__main__":
    unittest.main()
