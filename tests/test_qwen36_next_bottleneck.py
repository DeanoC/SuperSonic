from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parent / "metal" / "select_qwen36_next_bottleneck.py"
SPEC = importlib.util.spec_from_file_location("select_qwen36_next_bottleneck", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def sota_summary(next_action: str = selector.FALLBACK_ACTION) -> dict:
    return {
        "schema": "qwen36-sota-gate-summary-v10",
        "summary": {
            "next_action": {"action": next_action},
            "failed_gate_ids": [
                "static_topn_runtime",
                "fused_routed_int4",
                "lru_resident_cache",
            ],
            "superseded_gate_ids": [
                "mps_resident_table",
                "route_residency",
            ],
        },
    }


def runtime_report(default_row: dict) -> dict:
    return {
        "schema": "runtime",
        "rows": [default_row],
    }


def fused_report(default_row: dict) -> dict:
    report = runtime_report(default_row)
    report["summary"] = {
        "ffn_residency_gap": {
            "recommendation": "prototype_ffn_residency_or_submit_wait_path",
            "reason": "candidate GPU timestamps are much smaller than native wall or command-buffer wait totals",
            "thresholds": {
                "max_fused_wall_gpu_ratio": 4.0,
                "max_wait_gpu_ratio": 4.0,
            },
            "residency_or_submit_wait_modes": ["full-stage5"],
            "gpu_arithmetic_modes": [],
            "candidates": [
                {
                    "mode": "full-stage5",
                    "classes": ["residency_or_submit_wait"],
                    "prompts": [
                        {
                            "prompt_id": "hello",
                            "generated_ids_match_default": True,
                            "fused_wall_ms": 800.0,
                            "fused_gpu_ms": 80.0,
                            "fused_wall_gpu_ratio": 10.0,
                            "command_buffer_wait_ms": 1000.0,
                            "wait_gpu_ratio": 12.5,
                            "ffn_attribution_class": "residency_or_submit_wait",
                        }
                    ],
                }
            ],
        }
    }
    return report


def default_row(
    report_name: str,
    ffn: float,
    linear: float,
    full: float,
    lm_head: float,
) -> dict:
    return {
        "prompt_id": report_name,
        "mode": "default",
        "status": "ok",
        "chain_breakdown": {
            "ffn_ms_avg": ffn,
            "linear_attn_ms_avg": linear,
            "full_attn_ms_avg": full,
        },
        "stage_timings": {"lm_head_ms_avg": lm_head},
        "metal_profile": {
            "entries": [
                {
                    "op": "command_buffer_wait",
                    "path": "native",
                    "total_ms": 100.0 + ffn,
                    "calls": 40,
                }
            ]
        },
        "hal_profile": {
            "entries": [
                {"op": "copy_h2d", "path": "hal", "total_ms": 50.0, "calls": 4}
            ]
        },
    }


def prefill_report() -> dict:
    return {
        "schema": "prefill",
        "summary": {"promotion_gate": {"passed": False}},
        "rows": [
            {
                "status": "ok",
                "sweep_mode": "baseline",
                "context_tokens_requested": 512,
                "lifecycle": {"prefill_total_ms": 1000.0},
            },
            {
                "status": "ok",
                "sweep_mode": "prototype-default",
                "context_tokens_requested": 512,
                "lifecycle": {"prefill_total_ms": 500.0},
            },
        ],
    }


def bench_perf_report() -> dict:
    return {
        "schema_version": 9,
        "model": selector.MODEL,
        "quant": "int4",
        "arch": "apple-m5-max",
        "backend": selector.BACKEND,
        "status": "ok",
        "ms_per_step": 162.6,
        "samples": [182.7, 162.6, 145.7],
        "stage_timings": {"lm_head_ms_avg": 4.682},
        "chain_breakdown": {
            "ffn_ms_avg": 97.974,
            "linear_attn_ms_avg": 31.335,
            "full_attn_ms_avg": 18.213,
        },
        "profile_stage_timings": {"lm_head_ms_avg": 4.187},
        "profile_chain_breakdown": {
            "ffn_ms_avg": 97.430,
            "linear_attn_ms_avg": 53.165,
            "full_attn_ms_avg": 17.510,
        },
        "metal_profile": {
            "entries": [
                {
                    "op": "qwen36_linear_int4_stage5",
                    "path": "native",
                    "total_ms": 1166.320,
                    "calls": 630,
                }
            ]
        },
        "hal_profile": {
            "entries": [
                {"op": "copy_h2d", "total_ms": 5419.327, "calls": 1389}
            ]
        },
    }


class Qwen36NextBottleneckTests(unittest.TestCase):
    def paths_in(self, root: Path) -> dict[str, Path]:
        return {
            "sota_json": root / "sota.json",
            "static_runtime_json": root / "static.json",
            "fused_json": root / "fused.json",
            "lru_json": root / "lru.json",
            "linear_json": root / "linear.json",
            "full_json": root / "full.json",
            "lm_head_json": root / "lm_head.json",
            "prefill_json": root / "prefill.json",
            "bench_perf_json": None,
            "bench_run_root": root / "empty-bench-runs",
            "repo_root": root / "not-a-git-repo",
            "out_json": root / "next.json",
            "out_md": root / "next.md",
        }

    def write_default_inputs(self, root: Path, next_action: str = selector.FALLBACK_ACTION):
        paths = self.paths_in(root)
        write_json(paths["sota_json"], sota_summary(next_action))
        write_json(
            paths["static_runtime_json"],
            runtime_report(default_row("static", 100.0, 60.0, 20.0, 5.0)),
        )
        write_json(
            paths["fused_json"],
            fused_report(default_row("fused", 90.0, 65.0, 18.0, 6.0)),
        )
        write_json(
            paths["lru_json"],
            runtime_report(default_row("lru", 110.0, 70.0, 22.0, 7.0)),
        )
        write_json(
            paths["linear_json"],
            runtime_report(default_row("linear", 95.0, 55.0, 19.0, 5.5)),
        )
        write_json(
            paths["full_json"],
            runtime_report(default_row("full", 92.0, 56.0, 17.0, 5.1)),
        )
        write_json(
            paths["lm_head_json"],
            runtime_report(default_row("lm_head", 94.0, 57.0, 18.0, 4.9)),
        )
        write_json(paths["prefill_json"], prefill_report())
        return paths

    def args_for(self, paths: dict[str, Path]) -> SimpleNamespace:
        return SimpleNamespace(**paths)

    def test_selects_largest_non_exhausted_bucket_when_ffn_gates_are_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            report = selector.build_report(self.args_for(paths))
            md = selector.render_markdown(report)

        self.assertEqual(report["schema"], selector.SCHEMA)
        rec = report["recommendation"]
        self.assertEqual(rec["status"], "selected")
        self.assertEqual(rec["dominant_bucket"], "ffn_ms_avg")
        self.assertEqual(rec["target_bucket"], "linear_attn_ms_avg")
        self.assertEqual(rec["action"], "prototype_linear_attention_orchestration")
        buckets = {row["bucket"]: row for row in report["decode_bucket_ranking"]}
        self.assertTrue(buckets["ffn_ms_avg"]["exhausted"])
        self.assertFalse(buckets["linear_attn_ms_avg"]["exhausted"])
        self.assertEqual(report["prefill"]["best_mode"], "prototype-default")
        self.assertIn("prototype_linear_attention_orchestration", md)

    def test_includes_schema_v9_bench_perf_as_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            bench_path = Path(tmp) / "bench.json"
            write_json(bench_path, bench_perf_report())
            paths["bench_perf_json"] = bench_path
            report = selector.build_report(self.args_for(paths))
            md = selector.render_markdown(report)

        self.assertEqual(report["schema"], selector.SCHEMA)
        self.assertEqual(report["input_reports"]["bench_perf"]["schema"], 9)
        self.assertEqual(report["bench_perf"]["schema_version"], 9)
        self.assertEqual(report["bench_perf"]["linear_attn_ms_avg"], 31.335)
        self.assertEqual(report["bench_perf"]["profile_linear_attn_ms_avg"], 53.165)
        buckets = {row["bucket"]: row for row in report["decode_bucket_ranking"]}
        self.assertEqual(buckets["ffn_ms_avg"]["sample_count"], 7)
        self.assertTrue(
            any(
                sample["source"] == "bench_perf" and sample["row"] == "bench_perf"
                for sample in buckets["linear_attn_ms_avg"]["samples"]
            )
        )
        self.assertEqual(report["top_metal_profile_ops"][0]["source"], "bench_perf")
        self.assertIn("## Bench Perf", md)
        self.assertIn("profile_linear_attn_ms_avg", md)

    def test_moves_to_full_attention_when_linear_gate_is_also_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            summary = sota_summary()
            summary["summary"]["failed_gate_ids"].append("linear_decode_variants")
            write_json(paths["sota_json"], summary)
            report = selector.build_report(self.args_for(paths))

        rec = report["recommendation"]
        self.assertEqual(rec["status"], "selected")
        self.assertEqual(rec["dominant_bucket"], "ffn_ms_avg")
        self.assertEqual(rec["target_bucket"], "full_attn_ms_avg")
        self.assertEqual(rec["action"], "prototype_full_attention_orchestration")
        buckets = {row["bucket"]: row for row in report["decode_bucket_ranking"]}
        self.assertTrue(buckets["linear_attn_ms_avg"]["exhausted"])

    def test_moves_to_lm_head_when_full_gate_is_also_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            summary = sota_summary()
            summary["summary"]["failed_gate_ids"].extend(
                ["linear_decode_variants", "full_attention_variants"]
            )
            write_json(paths["sota_json"], summary)
            report = selector.build_report(self.args_for(paths))

        rec = report["recommendation"]
        self.assertEqual(rec["status"], "selected")
        self.assertEqual(rec["dominant_bucket"], "ffn_ms_avg")
        self.assertEqual(rec["target_bucket"], "lm_head_ms_avg")
        self.assertEqual(rec["action"], "prototype_lm_head_tail_path")
        buckets = {row["bucket"]: row for row in report["decode_bucket_ranking"]}
        self.assertTrue(buckets["full_attn_ms_avg"]["exhausted"])

    def test_returns_to_dominant_bucket_after_lm_head_gate_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            summary = sota_summary()
            summary["summary"]["failed_gate_ids"].extend(
                [
                    "linear_decode_variants",
                    "full_attention_variants",
                    "lm_head_tail_variants",
                ]
            )
            write_json(paths["sota_json"], summary)
            report = selector.build_report(self.args_for(paths))

        rec = report["recommendation"]
        self.assertEqual(rec["status"], "selected")
        self.assertEqual(rec["dominant_bucket"], "ffn_ms_avg")
        self.assertEqual(rec["target_bucket"], "ffn_ms_avg")
        self.assertEqual(rec["action"], "prototype_new_ffn_residency_or_compute_path")
        self.assertEqual(rec["sub_action"], "prototype_ffn_residency_or_submit_wait_path")
        buckets = {row["bucket"]: row for row in report["decode_bucket_ranking"]}
        self.assertTrue(buckets["lm_head_ms_avg"]["exhausted"])
        self.assertEqual(
            report["ffn_candidate_gap"]["recommendation"],
            "prototype_ffn_residency_or_submit_wait_path",
        )

    def test_auto_discovers_matching_bench_perf_fingerprint_before_newer_stale_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "bench-runs"
            matching = run_root / "2026-05-24-match" / "perf" / "qwen3.6-35b-a3b_int4.json"
            stale = run_root / "2026-05-24-stale" / "perf" / "qwen3.6-35b-a3b_int4.json"
            write_json(matching, bench_perf_report())
            write_json(stale, {**bench_perf_report(), "ms_per_step": 999.0})
            write_json(
                matching.parent.parent / "meta.json",
                {
                    "schema_version": 9,
                    "git_sha": "abc1234",
                    "git_dirty": True,
                    "git_dirty_paths": ["crates/kernel-ffi/src/metal_native.mm"],
                    "git_diff_hash": "current",
                },
            )
            write_json(
                stale.parent.parent / "meta.json",
                {
                    "schema_version": 9,
                    "git_sha": "abc1234",
                    "git_dirty": True,
                    "git_dirty_paths": ["crates/kernel-ffi/src/metal_native.mm"],
                    "git_diff_hash": "old-experiment",
                },
            )

            selected = selector.latest_bench_perf_json(
                run_root,
                {
                    "git_sha": "abc1234",
                    "git_dirty": True,
                    "git_dirty_paths": ["crates/runner/src/main.rs"],
                    "git_diff_hash": "current",
                },
            )

        self.assertEqual(selected, matching)

    def test_auto_discovery_skips_bench_perf_when_current_fingerprint_has_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "bench-runs"
            stale = run_root / "2026-05-24-stale" / "perf" / "qwen3.6-35b-a3b_int4.json"
            write_json(stale, bench_perf_report())
            write_json(
                stale.parent.parent / "meta.json",
                {
                    "schema_version": 9,
                    "git_sha": "abc1234",
                    "git_dirty": True,
                    "git_dirty_paths": ["crates/kernel-ffi/src/metal_native.mm"],
                    "git_diff_hash": "old-experiment",
                },
            )

            selected = selector.latest_bench_perf_json(
                run_root,
                {
                    "git_sha": "abc1234",
                    "git_dirty": True,
                    "git_dirty_paths": ["crates/runner/src/main.rs"],
                    "git_diff_hash": "current",
                },
            )

        self.assertIsNone(selected)

    def test_defers_when_sota_summary_has_specific_next_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(
                Path(tmp),
                next_action="prepare_runtime_promotion:fused_routed_int4",
            )
            report = selector.build_report(self.args_for(paths))

        self.assertEqual(report["recommendation"]["status"], "defer_to_sota_gate_summary")
        self.assertEqual(
            report["recommendation"]["action"],
            "prepare_runtime_promotion:fused_routed_int4",
        )

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_default_inputs(Path(tmp))
            rc = selector.main(
                [
                    "--sota-json",
                    str(paths["sota_json"]),
                    "--static-runtime-json",
                    str(paths["static_runtime_json"]),
                    "--fused-json",
                    str(paths["fused_json"]),
                    "--lru-json",
                    str(paths["lru_json"]),
                    "--linear-json",
                    str(paths["linear_json"]),
                    "--full-json",
                    str(paths["full_json"]),
                    "--lm-head-json",
                    str(paths["lm_head_json"]),
                    "--prefill-json",
                    str(paths["prefill_json"]),
                    "--bench-run-root",
                    str(paths["bench_run_root"]),
                    "--out-json",
                    str(paths["out_json"]),
                    "--out-md",
                    str(paths["out_md"]),
                    "--require-selected",
                ]
            )
            report = json.loads(paths["out_json"].read_text())
            md_exists = paths["out_md"].exists()

        self.assertEqual(rc, 0)
        self.assertEqual(report["schema"], selector.SCHEMA)
        self.assertTrue(md_exists)


if __name__ == "__main__":
    unittest.main()
