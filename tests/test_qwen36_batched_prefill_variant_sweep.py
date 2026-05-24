import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_batched_prefill_variants.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_batched_prefill_variants", SCRIPT)
sweep_qwen36_batched_prefill_variants = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_batched_prefill_variants
SPEC.loader.exec_module(sweep_qwen36_batched_prefill_variants)


class Qwen36BatchedPrefillVariantSweepTests(unittest.TestCase):
    def test_parse_modes_dedupes_aliases(self):
        parse_modes = sweep_qwen36_batched_prefill_variants.parse_modes

        self.assertEqual(
            parse_modes("baseline,default,prototype,router-topk,router-topk"),
            ["baseline", "prototype-default", "router-topk"],
        )

    def test_mode_maps_to_prototype_and_variant(self):
        script = sweep_qwen36_batched_prefill_variants

        self.assertFalse(script.prototype_enabled("baseline"))
        self.assertTrue(script.prototype_enabled("prototype-default"))
        self.assertTrue(script.prototype_enabled("fused-residual"))
        self.assertEqual(script.variant_for_mode("baseline"), "default")
        self.assertEqual(script.variant_for_mode("prototype-default"), "default")
        self.assertEqual(script.variant_for_mode("fused-residual"), "fused-residual")

    def test_args_for_mode_sets_longctx_flags(self):
        script = sweep_qwen36_batched_prefill_variants
        args = SimpleNamespace(
            batched_prefill_prototype=False,
            batched_prefill_variant="default",
        )

        run_args = script.args_for_mode(args, "router-topk")

        self.assertTrue(run_args.batched_prefill_prototype)
        self.assertEqual(run_args.batched_prefill_variant, "router-topk")
        self.assertFalse(args.batched_prefill_prototype)

    def test_summarize_tracks_parity_and_baseline_ratios(self):
        script = sweep_qwen36_batched_prefill_variants
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
            },
            {
                "context_tokens_requested": 512,
                "sweep_mode": "prototype-default",
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
            },
        ]

        summary = script.summarize(rows, require_profile=False)

        self.assertTrue(summary["generated_ids_match"])
        self.assertEqual(summary["best_prefill_by_context"]["512"]["mode"], "prototype-default")
        comparison = summary["comparisons"][1]
        self.assertEqual(comparison["mode"], "prototype-default")
        self.assertAlmostEqual(comparison["prefill_ratio"], 0.75)
        self.assertAlmostEqual(comparison["decode_total_ratio"], 0.9)
        self.assertAlmostEqual(comparison["ffn_ms_avg_ratio"], 0.9)
        self.assertTrue(summary["promotion_gate"]["passed"])
        self.assertEqual(summary["promotion_gate"]["passed_modes"], ["prototype-default"])

    def test_summarize_reports_generated_id_mismatch(self):
        script = sweep_qwen36_batched_prefill_variants
        summary = script.summarize(
            [
                {
                    "context_tokens_requested": 512,
                    "sweep_mode": "baseline",
                    "status": "ok",
                    "generated_ids": [271],
                },
                {
                    "context_tokens_requested": 512,
                    "sweep_mode": "router-topk",
                    "status": "ok",
                    "generated_ids": [272],
                },
            ],
            require_profile=False,
        )

        self.assertFalse(summary["generated_ids_match"])
        self.assertEqual(summary["generated_id_mismatches"][0]["mode"], "router-topk")
        self.assertIn(
            "context_512:generated_ids_mismatch",
            summary["promotion_gate"]["candidates"][0]["failures"],
        )

    def test_promotion_gate_requires_profile_by_default(self):
        script = sweep_qwen36_batched_prefill_variants
        summary = script.summarize(
            [
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
                },
            ]
        )

        self.assertFalse(summary["promotion_gate"]["passed"])
        self.assertIn(
            "context_512:missing_command_buffer_wait_profile",
            summary["promotion_gate"]["candidates"][0]["failures"],
        )

    def test_promotion_gate_rejects_bucket_regression(self):
        script = sweep_qwen36_batched_prefill_variants
        summary = script.summarize(
            [
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
                        "full_attn_ms_avg": 13.0,
                        "linear_attn_ms_avg": 21.0,
                        "lm_head_ms_avg": 5.1,
                    },
                    "metal_profile": {
                        "entries": [{"op": "command_buffer_wait", "total_ms": 102.0}]
                    },
                },
            ]
        )

        self.assertFalse(summary["promotion_gate"]["passed"])
        self.assertIn(
            "context_512:full_attn_ms_avg_regressed",
            summary["promotion_gate"]["candidates"][0]["failures"],
        )

    def test_render_markdown_includes_profile_and_baseline_delta(self):
        script = sweep_qwen36_batched_prefill_variants
        rows = [
            {
                "context_tokens_requested": 512,
                "sweep_mode": "baseline",
                "status": "ok",
                "generated_ids": [271],
                "niah_contains_expected": True,
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
                "wall_seconds": 13.0,
            },
            {
                "context_tokens_requested": 512,
                "sweep_mode": "router-topk",
                "status": "ok",
                "generated_ids": [271],
                "niah_contains_expected": True,
                "lifecycle": {"prefill_total_ms": 15000.0},
                "stage": {"total_ms_avg": 105.0},
                "chain_breakdown": {
                    "ffn_ms_avg": 55.0,
                    "full_attn_ms_avg": 10.5,
                    "linear_attn_ms_avg": 21.0,
                    "lm_head_ms_avg": 5.1,
                },
                "metal_profile": {
                    "entries": [
                        {"op": "qwen36_router_topk", "total_ms": 150.0},
                        {"op": "command_buffer_wait", "total_ms": 130.0},
                    ]
                },
                "hal_profile": {"summary": {"total_ms": 4.0}},
                "wall_seconds": 16.0,
            },
        ]
        report = script.build_report(
            rows,
            SimpleNamespace(
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
            ),
            [512],
            ["baseline", "router-topk"],
        )

        md = script.render_markdown(report)

        self.assertIn("Qwen3.6 Metal Batched-Prefill Variant Sweep", md)
        self.assertIn("router-topk", md)
        self.assertIn("1.250x", md)
        self.assertIn("qwen36_router_topk", md)
        self.assertIn("promotion_gate_passed", md)
        self.assertIn("prefill_not_improved", md)


if __name__ == "__main__":
    unittest.main()
