from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
            "fused_routed_int4": root / "fused.json",
            "mps_resident_table": root / "mps.json",
            "route_residency": root / "route.json",
            "mtp_acceptance": root / "mtp.json",
            "lru_resident_cache": root / "lru.json",
            "linear_decode_variants": root / "linear.json",
        }

    def write_default_reports(
        self,
        root: Path,
        prefill_gate: dict | None = None,
        static_gate: dict | None = None,
        fused_gate: dict | None = None,
        mps_gate: dict | None = None,
        route_gate: dict | None = None,
        mtp_gate: dict | None = None,
        lru_gate: dict | None = None,
        linear_gate: dict | None = None,
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
            paths["fused_routed_int4"],
            script.GATE_SPECS[2].expected_schema,
            "promotion_gate",
            fused_gate
            if fused_gate is not None
            else {
                "passed": False,
                "candidates": [
                    {
                        "mode": "direct-gather",
                        "passed": False,
                        "failures": ["ffn_not_improved"],
                    }
                ],
            },
        )
        write_report(
            paths["mps_resident_table"],
            script.GATE_SPECS[3].expected_schema,
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
            paths["route_residency"],
            script.GATE_SPECS[4].expected_schema,
            "decision_gate",
            route_gate
            if route_gate is not None
            else {
                "passed": False,
                "recommendation": "prefer_fused_routed_int4",
                "candidates": [
                    {
                        "kind": "lru_hotset",
                        "capacity": 16,
                        "passed": False,
                        "failures": ["lru_hit_rate_below_threshold"],
                    }
                ],
            },
        )
        write_report(
            paths["mtp_acceptance"],
            script.GATE_SPECS[5].expected_schema,
            "promotion_gate",
            mtp_gate
            if mtp_gate is not None
            else {
                "passed": False,
                "failures": ["acceptance_below_threshold"],
            },
        )
        write_report(
            paths["lru_resident_cache"],
            script.GATE_SPECS[6].expected_schema,
            "promotion_gate",
            lru_gate
            if lru_gate is not None
            else {
                "passed": False,
                "candidates": [
                    {
                        "mode": "lru-hotset-64",
                        "passed": False,
                        "failures": ["command_buffer_wait_regressed"],
                    }
                ],
            },
        )
        write_report(
            paths["linear_decode_variants"],
            script.GATE_SPECS[7].expected_schema,
            "promotion_gate",
            linear_gate
            if linear_gate is not None
            else {
                "passed": False,
                "candidates": [
                    {
                        "mode": "direct-off",
                        "passed": False,
                        "failures": ["linear_attn_not_improved"],
                    }
                ],
            },
        )
        return paths

    def test_missing_reports_are_rows_by_default(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            report = script.build_report(self.paths_in(Path(tmp)))

        self.assertEqual(report["schema"], script.SCHEMA)
        self.assertEqual(report["summary"]["status_counts"], {"missing": 8})
        self.assertFalse(report["summary"]["all_inputs_ok"])
        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "run_or_refresh_gate_reports",
        )
        self.assertEqual(report["rows"][0]["recommendation_action"], "run_harness")
        self.assertIn(
            "sweep_qwen36_batched_prefill_variants.py",
            report["rows"][0]["refresh_command"],
        )
        self.assertIn(
            "sweep_qwen36_fused_routed_int4.py",
            report["rows"][2]["refresh_command"],
        )
        self.assertIn(
            "sweep_qwen36_linear_decode.py",
            report["rows"][7]["refresh_command"],
        )

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
                route_gate={
                    "passed": True,
                    "recommendation": "prototype_larger_lru_resident_cache",
                    "candidates": [
                        {
                            "kind": "lru_hotset",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
                lru_gate={"passed": False, "candidates": []},
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
                lru_gate={"passed": False, "candidates": []},
            )
            report = script.build_report(paths)

        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "prototype_partial_hit_resident_mps",
        )
        self.assertEqual(report["rows"][3]["passed_candidates"], ["partial_hit_optimistic:64"])

    def test_route_decision_pass_is_next_when_runtime_and_viability_fail(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(
                Path(tmp),
                route_gate={
                    "passed": True,
                    "recommendation": "prototype_larger_lru_resident_cache",
                    "candidates": [
                        {
                            "kind": "lru_hotset",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
                lru_gate={"passed": False, "candidates": []},
            )
            report = script.build_report(paths)

        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "prototype_larger_lru_resident_cache",
        )
        self.assertEqual(report["rows"][4]["passed_candidates"], ["lru_hotset:64"])

    def test_mps_viability_is_superseded_by_failed_runtime_candidate(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(
                Path(tmp),
                static_gate={
                    "passed": False,
                    "candidates": [
                        {
                            "mode": "mps-static-partial",
                            "passed": False,
                            "failures": ["headline_not_improved"],
                        }
                    ],
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
                route_gate={
                    "passed": True,
                    "recommendation": "prototype_larger_lru_resident_cache",
                    "candidates": [
                        {
                            "kind": "lru_hotset",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
                lru_gate={"passed": False, "candidates": []},
            )
            report = script.build_report(paths)

        mps_row = report["rows"][3]
        self.assertEqual(
            mps_row["superseded_by"],
            "static_topn_runtime:mps-static-partial",
        )
        self.assertEqual(
            mps_row["recommendation_action"],
            "keep_disabled_runtime_failed",
        )
        self.assertEqual(report["summary"]["superseded_gate_ids"], ["mps_resident_table"])
        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "prototype_larger_lru_resident_cache",
        )

    def test_route_decision_is_superseded_by_failed_lru_runtime_candidate(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(
                Path(tmp),
                route_gate={
                    "passed": True,
                    "recommendation": "prototype_larger_lru_resident_cache",
                    "candidates": [
                        {
                            "kind": "lru_hotset",
                            "capacity": 64,
                            "passed": True,
                        }
                    ],
                },
                lru_gate={
                    "passed": False,
                    "candidates": [
                        {
                            "mode": "lru-hotset-64",
                            "passed": False,
                            "failures": ["headline_not_improved"],
                        }
                    ],
                },
            )
            report = script.build_report(paths)

        route_row = report["rows"][4]
        self.assertEqual(route_row["superseded_by"], "lru_resident_cache")
        self.assertEqual(
            route_row["recommendation_action"],
            "keep_disabled_runtime_failed",
        )
        self.assertEqual(report["summary"]["superseded_gate_ids"], ["route_residency"])
        self.assertEqual(
            report["summary"]["next_action"]["action"],
            "keep_default_lane_and_select_next_measured_bottleneck",
        )

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
                json.dumps({"schema": script.GATE_SPECS[3].expected_schema, "summary": {}})
                + "\n"
            )
            write_report(
                paths["route_residency"],
                script.GATE_SPECS[4].expected_schema,
                "decision_gate",
                {"passed": False, "recommendation": "prefer_fused_routed_int4"},
            )
            write_report(
                paths["fused_routed_int4"],
                script.GATE_SPECS[2].expected_schema,
                "promotion_gate",
                {"passed": False, "failures": ["ffn_not_improved"]},
            )
            write_report(
                paths["mtp_acceptance"],
                script.GATE_SPECS[5].expected_schema,
                "promotion_gate",
                {"passed": False, "failures": ["acceptance_below_threshold"]},
            )
            write_report(
                paths["lru_resident_cache"],
                script.GATE_SPECS[6].expected_schema,
                "promotion_gate",
                {"passed": False, "failures": ["command_buffer_wait_regressed"]},
            )
            write_report(
                paths["linear_decode_variants"],
                script.GATE_SPECS[7].expected_schema,
                "promotion_gate",
                {"passed": False, "failures": ["linear_attn_not_improved"]},
            )
            report = script.build_report(paths)

        statuses = {row["gate_id"]: row["status"] for row in report["rows"]}
        self.assertEqual(statuses["batched_prefill_variants"], "malformed")
        self.assertEqual(statuses["static_topn_runtime"], "schema_mismatch")
        self.assertEqual(statuses["mps_resident_table"], "missing_gate")
        self.assertEqual(statuses["fused_routed_int4"], "ok")
        self.assertEqual(statuses["mtp_acceptance"], "ok")
        self.assertEqual(statuses["lru_resident_cache"], "ok")
        self.assertEqual(statuses["linear_decode_variants"], "ok")
        self.assertEqual(report["summary"]["input_failure_count"], 3)
        self.assertIn("static_topn_runtime", report["summary"]["next_action"]["blocked_reason"])

    def test_stale_reports_are_input_failures_when_max_age_is_set(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_reports(Path(tmp))
            report = script.build_report(
                paths,
                now=datetime.now(timezone.utc) + timedelta(hours=2),
                max_age_seconds=3600.0,
            )

        statuses = {row["gate_id"]: row["status"] for row in report["rows"]}
        self.assertEqual(set(statuses.values()), {"stale"})
        self.assertFalse(report["summary"]["all_inputs_ok"])
        self.assertEqual(report["summary"]["status_counts"], {"stale": 8})
        self.assertEqual(report["rows"][0]["recommendation_action"], "refresh_harness")
        self.assertIn(
            "sweep_qwen36_batched_prefill_variants.py",
            report["rows"][0]["refresh_command"],
        )
        self.assertIn("report age", report["rows"][0]["error"])
        self.assertIsNotNone(report["rows"][0]["mtime_utc"])
        self.assertGreater(report["rows"][0]["age_seconds"], 3600.0)

    def test_render_markdown_and_require_exit(self):
        script = summarize_qwen36_sota_gates
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.write_default_reports(root)
            report = script.build_report(paths)
            md = script.render_markdown(report)

            self.assertIn("Qwen3.6 Metal SOTA Gate Summary", md)
            self.assertIn("| Batched-prefill variants | ok | false |", md)
            self.assertIn("## Refresh Commands", md)
            self.assertIn("sweep_qwen36_fused_routed_int4.py", md)
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
                    "--fused-json",
                    str(paths["fused_routed_int4"]),
                    "--mps-json",
                    str(paths["mps_resident_table"]),
                    "--route-json",
                    str(paths["route_residency"]),
                    "--mtp-json",
                    str(paths["mtp_acceptance"]),
                    "--lru-json",
                    str(paths["lru_resident_cache"]),
                    "--linear-json",
                    str(paths["linear_decode_variants"]),
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
