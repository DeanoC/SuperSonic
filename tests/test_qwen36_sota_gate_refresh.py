from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).parent
METAL_DIR = TESTS_DIR / "metal"
SUMMARY_SCRIPT = METAL_DIR / "summarize_qwen36_sota_gates.py"
REFRESH_SCRIPT = METAL_DIR / "refresh_qwen36_sota_gates.py"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summary = load_script("summarize_qwen36_sota_gates", SUMMARY_SCRIPT)
refresh = load_script("refresh_qwen36_sota_gates", REFRESH_SCRIPT)


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


class Qwen36SotaGateRefreshTests(unittest.TestCase):
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
            "full_attention_variants": root / "full.json",
        }

    def write_default_reports(self, root: Path) -> dict[str, Path]:
        paths = self.paths_in(root)
        write_report(
            paths["batched_prefill_variants"],
            summary.GATE_SPECS[0].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["prefill_not_improved"]},
        )
        write_report(
            paths["static_topn_runtime"],
            summary.GATE_SPECS[1].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["headline_not_improved"]},
        )
        write_report(
            paths["fused_routed_int4"],
            summary.GATE_SPECS[2].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["ffn_not_improved"]},
        )
        write_report(
            paths["mps_resident_table"],
            summary.GATE_SPECS[3].expected_schema,
            "viability_gate",
            {"passed": False, "recommendation": "reject_resident_mps_for_now"},
        )
        write_report(
            paths["route_residency"],
            summary.GATE_SPECS[4].expected_schema,
            "decision_gate",
            {"passed": False, "recommendation": "prefer_fused_routed_int4"},
        )
        write_report(
            paths["mtp_acceptance"],
            summary.GATE_SPECS[5].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["acceptance_below_threshold"]},
        )
        write_report(
            paths["lru_resident_cache"],
            summary.GATE_SPECS[6].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["command_buffer_wait_regressed"]},
        )
        write_report(
            paths["linear_decode_variants"],
            summary.GATE_SPECS[7].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["linear_attn_not_improved"]},
        )
        write_report(
            paths["full_attention_variants"],
            summary.GATE_SPECS[8].expected_schema,
            "promotion_gate",
            {"passed": False, "failures": ["full_attn_not_improved"]},
        )
        return paths

    def test_missing_reports_are_selected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre_report = summary.build_report(self.paths_in(Path(tmp)))
            rows = refresh.build_plan_rows(pre_report["rows"])
            report = refresh.build_refresh_report(pre_report, rows, dry_run=True)
            md = refresh.render_markdown(report)

        self.assertEqual(report["schema"], refresh.SCHEMA)
        self.assertTrue(report["summary"]["dry_run"])
        self.assertEqual(report["summary"]["selected_count"], 9)
        self.assertEqual(report["summary"]["run_status_counts"], {"planned": 9})
        self.assertEqual(set(row["input_status"] for row in rows), {"missing"})
        self.assertIn("sweep_qwen36_batched_prefill_variants.py", md)
        self.assertIn("Add `--run`", md)

    def test_ok_reports_are_skipped_unless_all_or_only_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre_report = summary.build_report(self.write_default_reports(Path(tmp)))

        default_rows = refresh.build_plan_rows(pre_report["rows"])
        self.assertEqual(sum(1 for row in default_rows if row["selected"]), 0)
        self.assertEqual({row["run_status"] for row in default_rows}, {"skipped"})

        all_rows = refresh.build_plan_rows(pre_report["rows"], refresh_all=True)
        self.assertEqual(sum(1 for row in all_rows if row["selected"]), 9)
        self.assertEqual({row["selection_reason"] for row in all_rows}, {"all"})

        only_rows = refresh.build_plan_rows(
            pre_report["rows"],
            only={"fused_routed_int4"},
        )
        selected = [row for row in only_rows if row["selected"]]
        self.assertEqual([row["gate_id"] for row in selected], ["fused_routed_int4"])
        self.assertEqual(selected[0]["selection_reason"], "requested")

    def test_normalize_only_accepts_repeated_and_comma_values(self):
        only = refresh.normalize_only(
            ["batched_prefill_variants,fused_routed_int4", "mtp_acceptance"]
        )

        self.assertEqual(
            only,
            {"batched_prefill_variants", "fused_routed_int4", "mtp_acceptance"},
        )
        with self.assertRaises(ValueError):
            refresh.normalize_only(["not_a_gate"])

    def test_main_writes_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.paths_in(root)
            out_json = root / "refresh.json"
            out_md = root / "refresh.md"
            rc = refresh.main(
                [
                    "--prefill-json",
                    str(paths["batched_prefill_variants"]),
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
                    "--full-json",
                    str(paths["full_attention_variants"]),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )
            report = json.loads(out_json.read_text())
            md_exists = out_md.exists()

        self.assertEqual(rc, 0)
        self.assertEqual(report["schema"], refresh.SCHEMA)
        self.assertTrue(report["summary"]["dry_run"])
        self.assertEqual(report["summary"]["selected_count"], 9)
        self.assertTrue(md_exists)


if __name__ == "__main__":
    unittest.main()
