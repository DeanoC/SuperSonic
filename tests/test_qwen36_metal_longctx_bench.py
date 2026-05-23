import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parent / "metal" / "bench_qwen36_longctx.py"
SPEC = importlib.util.spec_from_file_location("bench_qwen36_metal_longctx", SCRIPT)
bench_qwen36_metal_longctx = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = bench_qwen36_metal_longctx
SPEC.loader.exec_module(bench_qwen36_metal_longctx)


class Qwen36MetalLongContextBenchTests(unittest.TestCase):
    def test_apply_preset_defaults_uses_metal_comparison_defaults(self):
        args = SimpleNamespace(
            preset="comparison",
            contexts=None,
            max_new_tokens=None,
            timeout=None,
            warmup=None,
        )
        applied = bench_qwen36_metal_longctx.apply_preset_defaults(args)
        self.assertEqual(applied.contexts, "512,2048,8192")
        self.assertEqual(applied.max_new_tokens, 4)
        self.assertEqual(applied.timeout, 21600)
        self.assertTrue(applied.warmup)

    def test_resolve_model_dir_prefers_explicit_then_qwen36_env_then_root(self):
        explicit = Path("/tmp/explicit")
        self.assertEqual(
            bench_qwen36_metal_longctx.resolve_model_dir(
                explicit,
                {
                    "SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR": "/tmp/qwen36",
                    "SUPERSONIC_TEST_MODEL_ROOT": "/tmp/root",
                },
            ),
            explicit,
        )
        self.assertEqual(
            bench_qwen36_metal_longctx.resolve_model_dir(
                None,
                {
                    "SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR": "/tmp/qwen36",
                    "SUPERSONIC_TEST_MODEL_ROOT": "/tmp/root",
                },
            ),
            Path("/tmp/qwen36"),
        )
        self.assertEqual(
            bench_qwen36_metal_longctx.resolve_model_dir(
                None,
                {"SUPERSONIC_TEST_MODEL_ROOT": "/tmp/root"},
            ),
            Path("/tmp/root/qwen3.6-35b-a3b"),
        )

    def test_build_metal_env_sets_supported_lane_and_clears_vmm_knobs(self):
        env = bench_qwen36_metal_longctx.build_metal_env(
            {
                "PATH": os.environ.get("PATH", ""),
                "SUPERSONIC_VMM_KV": "1",
                "SUPERSONIC_VMM_MOE_ISLANDS": "1",
                "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS": "320",
                "SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON": "old.json",
            }
        )
        self.assertEqual(env["SUPERSONIC_BACKENDS"], "metal")
        self.assertEqual(env["SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL"], "0")
        self.assertEqual(env["SUPERSONIC_QWEN36_MOE_BATCHED_ATTN"], "0")
        self.assertEqual(env["SUPERSONIC_QWEN36_MOE_GROUPED_FFN"], "0")
        self.assertEqual(env["SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP"], "1")
        self.assertNotIn("SUPERSONIC_VMM_KV", env)
        self.assertNotIn("SUPERSONIC_VMM_MOE_ISLANDS", env)
        self.assertNotIn("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS", env)
        self.assertNotIn("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON", env)

    def test_parse_profile_extracts_summary_and_entries(self):
        output = """
[metal-profile] calls=2 total_ms=12.000 native_ms=10.000 host_ms=2.000
[metal-profile-op] op=qwen36_ffn_int4_stage5 path=native calls=1 mean_ms=10.0000 total_ms=10.000 max_ms=10.000
"""
        profile = bench_qwen36_metal_longctx.parse_profile(
            output, "[metal-profile]", "[metal-profile-op]"
        )
        self.assertIsNotNone(profile)
        self.assertEqual(profile["summary"]["native_ms"], 10.0)
        self.assertEqual(profile["entries"][0]["op"], "qwen36_ffn_int4_stage5")
        self.assertEqual(profile["entries"][0]["path"], "native")

    def test_parse_batched_prefill_feasibility(self):
        output = """
[qwen36-batched-prefill-feasibility] calls=20440 dropped_calls=0 layers=40 top_k=8 num_experts=256 chunk_size=512 prefill_tokens=511 profiled_tokens=511 chunks=1 assignments=163520 permutation_entries=163520 expert_segments=4127 avg_unique_experts_per_layer_chunk=103.175000 avg_rows_per_segment=39.619578 max_rows_per_segment=90 max_unique_experts_per_layer_chunk=118 wmma16_segments=3880 wmma16_covered_assignments=158000 wmma16_assignment_coverage=0.966243 scalar_tail_segments=247 scalar_tail_assignments=5520 wmma16_padded_assignments=190000 wmma16_padding_overhead=0.161322 metadata_only=1
"""
        profile = bench_qwen36_metal_longctx.parse_batched_prefill_feasibility(output)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["profiled_tokens"], 511)
        self.assertEqual(profile["chunk_size"], 512)
        self.assertAlmostEqual(profile["avg_rows_per_segment"], 39.619578)
        self.assertEqual(profile["scalar_tail_assignments"], 5520)
        self.assertAlmostEqual(profile["wmma16_padding_overhead"], 0.161322)
        self.assertEqual(profile["metadata_only"], 1)

    def test_parse_batched_prefill_plans(self):
        output = """
[qwen36-batched-prefill-plan] calls=20440 dropped_calls=0 layers=40 top_k=8 num_experts=256 chunk_size=128 prefill_tokens=511 profiled_tokens=511 chunks=4 assignments=163520 permutation_entries=163520 expert_segments=13312 avg_unique_experts_per_layer_chunk=83.200000 avg_rows_per_segment=12.283654 max_rows_per_segment=34 max_unique_experts_per_layer_chunk=98 wmma16_segments=3000 wmma16_covered_assignments=72000 wmma16_assignment_coverage=0.440313 scalar_tail_segments=10312 scalar_tail_assignments=91520 wmma16_padded_assignments=250000 wmma16_padding_overhead=0.528867 metadata_only=1
[qwen36-batched-prefill-plan] calls=20440 dropped_calls=0 layers=40 top_k=8 num_experts=256 chunk_size=512 prefill_tokens=511 profiled_tokens=511 chunks=1 assignments=163520 permutation_entries=163520 expert_segments=4127 avg_unique_experts_per_layer_chunk=103.175000 avg_rows_per_segment=39.619578 max_rows_per_segment=90 max_unique_experts_per_layer_chunk=118 wmma16_segments=3880 wmma16_covered_assignments=158000 wmma16_assignment_coverage=0.966243 scalar_tail_segments=247 scalar_tail_assignments=5520 wmma16_padded_assignments=190000 wmma16_padding_overhead=0.161322 metadata_only=1
"""
        plans = bench_qwen36_metal_longctx.parse_batched_prefill_plans(output)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["chunk_size"], 128)
        self.assertEqual(plans[1]["chunks"], 1)
        self.assertAlmostEqual(plans[1]["wmma16_assignment_coverage"], 0.966243)

    def test_append_batched_prefill_feasibility_markdown_adds_table(self):
        md = bench_qwen36_metal_longctx.append_batched_prefill_feasibility_markdown(
            "# report\n",
            [
                {
                    "context_tokens_requested": 512,
                    "batched_prefill_feasibility": {
                        "profiled_tokens": 511,
                        "chunks": 1,
                        "avg_unique_experts_per_layer_chunk": 103.175,
                        "avg_rows_per_segment": 39.619578,
                        "wmma16_assignment_coverage": 0.966243,
                        "dropped_calls": 0,
                    },
                }
            ],
        )
        self.assertIn("Batched-Prefill MoE Feasibility", md)
        self.assertIn("96.6%", md)

    def test_append_batched_prefill_plan_markdown_adds_table(self):
        md = bench_qwen36_metal_longctx.append_batched_prefill_plan_markdown(
            "# report\n",
            [
                {
                    "context_tokens_requested": 512,
                    "batched_prefill_plans": [
                        {
                            "chunk_size": 512,
                            "chunks": 1,
                            "avg_rows_per_segment": 39.619578,
                            "wmma16_assignment_coverage": 0.966243,
                            "scalar_tail_assignments": 5520,
                            "wmma16_padding_overhead": 0.161322,
                        }
                    ],
                }
            ],
        )
        self.assertIn("Batched-Prefill MoE Chunk Plan", md)
        self.assertIn("16.1%", md)

    def test_append_profile_markdown_adds_profile_table(self):
        md = bench_qwen36_metal_longctx.append_profile_markdown(
            "# report\n",
            [
                {
                    "context_tokens_requested": 512,
                    "metal_profile": {
                        "summary": {"total_ms": 12.0},
                        "entries": [{"op": "qwen36_ffn_int4_stage5", "total_ms": 10.0}],
                    },
                    "hal_profile": {"summary": {"total_ms": 1.0}, "entries": []},
                }
            ],
        )
        self.assertIn("Metal/HAL profile", md)
        self.assertIn("qwen36_ffn_int4_stage5", md)


if __name__ == "__main__":
    unittest.main()
