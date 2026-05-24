import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


TESTS_DIR = Path(__file__).parent
METAL_DIR = TESTS_DIR / "metal"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


longctx = load_script("qwen36_sota_longctx", METAL_DIR / "bench_qwen36_longctx.py")
prefill_sweep = load_script(
    "qwen36_sota_prefill_sweep",
    METAL_DIR / "sweep_qwen36_batched_prefill_variants.py",
)
static_sweep = load_script(
    "qwen36_sota_static_sweep",
    METAL_DIR / "sweep_qwen36_static_topn_runtime.py",
)
mps_probe = load_script(
    "qwen36_sota_mps_probe",
    METAL_DIR / "probe_qwen36_mps_resident_table.py",
)
route_sweep = load_script(
    "qwen36_sota_route_sweep",
    METAL_DIR / "sweep_qwen36_route_residency.py",
)
mtp_sweep = load_script(
    "qwen36_sota_mtp_sweep",
    METAL_DIR / "sweep_qwen36_mtp_acceptance.py",
)
sota_summary = load_script(
    "qwen36_sota_gate_summary",
    METAL_DIR / "summarize_qwen36_sota_gates.py",
)


class Qwen36SotaContractTests(unittest.TestCase):
    def test_schema_versions_match_current_sota_gate_surface(self):
        self.assertEqual(longctx.SCHEMA, "qwen36-moe-metal-longctx-bench-v5")
        self.assertEqual(
            prefill_sweep.SCHEMA,
            "qwen36-metal-batched-prefill-variant-sweep-v2",
        )
        self.assertEqual(static_sweep.SCHEMA, "qwen36-static-topn-runtime-sweep-v3")
        self.assertEqual(mps_probe.SCHEMA, "qwen36-mps-resident-table-probe-v2")
        self.assertEqual(route_sweep.SCHEMA, "qwen36-route-residency-sweep-v1")
        self.assertEqual(mtp_sweep.SCHEMA, "qwen36-moe-mtp-acceptance-sweep-v2")
        self.assertEqual(sota_summary.SCHEMA, "qwen36-sota-gate-summary-v2")

    def test_sota_summary_tracks_current_gate_reports(self):
        specs = {spec.gate_id: spec for spec in sota_summary.GATE_SPECS}

        self.assertEqual(
            specs["batched_prefill_variants"].expected_schema,
            prefill_sweep.SCHEMA,
        )
        self.assertEqual(
            specs["static_topn_runtime"].expected_schema,
            static_sweep.SCHEMA,
        )
        self.assertEqual(
            specs["mps_resident_table"].expected_schema,
            mps_probe.SCHEMA,
        )
        self.assertEqual(
            specs["route_residency"].expected_schema,
            route_sweep.SCHEMA,
        )
        self.assertEqual(
            specs["mtp_acceptance"].expected_schema,
            mtp_sweep.SCHEMA,
        )
        self.assertEqual(specs["mps_resident_table"].gate_keys, ("viability_gate",))
        self.assertEqual(specs["route_residency"].gate_keys, ("decision_gate",))

    def test_batched_prefill_variants_cover_documented_negative_gates(self):
        expected = {
            "default",
            "linear-direct-off",
            "full-attn-tmajor",
            "split-qgate",
            "router-topk",
            "fused-residual",
        }

        self.assertEqual(set(longctx.BATCHED_PREFILL_VARIANTS), expected)
        for variant in expected - {"default"}:
            self.assertIn(variant, prefill_sweep.MODE_ALIASES)

    def test_batched_prefill_sweep_exposes_promotion_gate(self):
        rows = [
            {
                "context_tokens_requested": 512,
                "sweep_mode": "baseline",
                "status": "ok",
                "generated_ids": [271],
                "lifecycle": {"prefill_total_ms": 12000.0},
                "stage": {"total_ms_avg": 100.0},
                "chain_breakdown": {
                    "ffn_ms_avg": 50.0,
                    "full_attn_ms_avg": 10.0,
                    "linear_attn_ms_avg": 20.0,
                    "lm_head_ms_avg": 5.0,
                },
                "metal_profile": {
                    "entries": [{"op": "command_buffer_wait", "total_ms": 100.0}]
                },
            },
            {
                "context_tokens_requested": 512,
                "sweep_mode": "router-topk",
                "status": "ok",
                "generated_ids": [271],
                "lifecycle": {"prefill_total_ms": 9000.0},
                "stage": {"total_ms_avg": 90.0},
                "chain_breakdown": {
                    "ffn_ms_avg": 45.0,
                    "full_attn_ms_avg": 10.5,
                    "linear_attn_ms_avg": 21.0,
                    "lm_head_ms_avg": 5.1,
                },
                "metal_profile": {
                    "entries": [{"op": "command_buffer_wait", "total_ms": 102.0}]
                },
            },
        ]
        args = SimpleNamespace(
            model_dir=Path("/tmp/model"),
            max_new_tokens=1,
            metal_profile=True,
            batched_prefill_feasibility=False,
            promotion_max_prefill_ratio=0.999,
            promotion_max_decode_ratio=0.999,
            promotion_max_ffn_ratio=0.999,
            promotion_max_component_regression_ratio=1.10,
            promotion_max_command_buffer_wait_ratio=1.05,
            promotion_require_profile=True,
            seed=20260504,
        )

        report = prefill_sweep.build_report(rows, args, [512], ["baseline", "router-topk"])

        gate = report["summary"]["promotion_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["router-topk"])
        self.assertIn("thresholds", gate)
        self.assertIn("candidates", gate)

    def test_static_residency_sweep_exposes_promotion_gate(self):
        rows = [
            {
                "status": "ok",
                "prompt_id": "p",
                "mode": "default",
                "generated_ids": [1, 2],
                "result": {"ms_per_step": 100.0},
                "stage_timings": {"lm_head_ms_avg": 5.0},
                "chain_breakdown": {
                    "ffn_ms_avg": 50.0,
                    "full_attn_ms_avg": 10.0,
                    "linear_attn_ms_avg": 20.0,
                },
                "metal_profile": {
                    "entries": [{"op": "command_buffer_wait", "total_ms": 100.0}]
                },
            },
            {
                "status": "ok",
                "prompt_id": "p",
                "mode": "static",
                "generated_ids": [1, 2],
                "result": {"ms_per_step": 90.0},
                "stage_timings": {"lm_head_ms_avg": 5.1},
                "chain_breakdown": {
                    "ffn_ms_avg": 45.0,
                    "full_attn_ms_avg": 10.5,
                    "linear_attn_ms_avg": 21.0,
                },
                "metal_profile": {
                    "entries": [{"op": "command_buffer_wait", "total_ms": 102.0}]
                },
            },
        ]
        args = static_sweep.parse_args([])

        report = static_sweep.build_report(rows, args, ["default", "static"], "smoke")

        gate = report["summary"]["promotion_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["passed_modes"], ["static"])
        self.assertIn("thresholds", gate)
        self.assertIn("candidates", gate)

    def test_mps_probe_exposes_viability_gate(self):
        static_report = {
            "schema": "qwen36-static-topn-mps-probe-v2",
            "rows": [
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
                }
            ],
        }
        pilot = {"source": "manual", "status": "ok", "gate_up_ms": 1.0, "down_ms": 1.0}

        report = mps_probe.build_report(
            static_report,
            pilot,
            layers=2,
            hidden=8,
            moe_intermediate=4,
            top_k=2,
            baseline_ffn_ms=10.0,
            baseline_source="contract test",
        )

        gate = report["summary"]["viability_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["recommendation"], "prototype_partial_hit_resident_mps")
        self.assertIn("thresholds", gate)
        self.assertIn("candidates", gate)

    def test_mtp_sweep_exposes_promotion_gate(self):
        rows = [
            {
                "status": "measured",
                "acceptance": {
                    "drafted_tokens": 2,
                    "accepted_tokens": 2,
                    "emitted_tokens": 3,
                    "base_steps": 2,
                    "replay_steps": 0,
                    "full_accept_steps": 1,
                    "zero_accept_steps": 0,
                },
            }
        ]

        report = mtp_sweep.build_report(
            rows,
            backend="metal",
            mode="sequential",
            prompt_set="custom",
            env_overrides={"SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT": "1"},
            promotion_min_acceptance=0.60,
            promotion_max_target_steps_per_emitted=0.99,
        )

        gate = report["summary"]["promotion_gate"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])
        self.assertIn("min_acceptance_rate", gate)
        self.assertIn("max_target_steps_per_emitted", gate)


if __name__ == "__main__":
    unittest.main()
