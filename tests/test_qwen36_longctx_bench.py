import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parent / "gfx1100" / "bench_qwen36_longctx.py"
SPEC = importlib.util.spec_from_file_location("bench_qwen36_longctx", SCRIPT)
bench_qwen36_longctx = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = bench_qwen36_longctx
SPEC.loader.exec_module(bench_qwen36_longctx)


class Qwen36LongContextBenchTests(unittest.TestCase):
    def test_parse_int_list_dedupes_and_rejects_empty_or_non_positive(self):
        parse = bench_qwen36_longctx.parse_int_list
        self.assertEqual(parse("8192,16384,8192"), [8192, 16384])
        with self.assertRaises(ValueError):
            parse("")
        with self.assertRaises(ValueError):
            parse("0")

    def test_parse_modes_expands_sparse_aliases(self):
        parse = bench_qwen36_longctx.parse_modes
        self.assertEqual(
            [mode.label for mode in parse("baseline,kv-fp8,sparse,sparse-kv-fp8", [320])],
            ["int4-vmm", "int4-kv-fp8", "cap320", "cap320-kv-fp8"],
        )
        with self.assertRaises(ValueError):
            parse("unknown", [320])

    def test_make_niah_prompt_contains_needle_and_answer_request(self):
        prompt, expected = bench_qwen36_longctx.make_niah_prompt(128, 7)
        self.assertIn(expected, prompt)
        self.assertIn("Question:", prompt)
        self.assertGreater(len(prompt), 128 * bench_qwen36_longctx.TOKEN_CHARS)

    def test_output_parsers_extract_stage_result_tokens_and_generated_json(self):
        output = (
            '[generated_json] "SSB-NEEDLE-00007\\n"\n'
            "[tokens] [1, 2, 3]\n"
            "[result] prompt_tokens=8190 generated_tokens=3 decode_ms=30 ms_per_tok=10\n"
            "[qwen36-moe stage-timings] gen_steps=3 total_ms_avg=10.5 attn_ms_avg=2 (chain_total_ms=30)\n"
            "[qwen36-moe chain-breakdown] gen_steps=3 full_attn_ms_avg=7.0 "
            "linear_attn_ms_avg=1.0 ffn_ms_avg=2.0\n"
            "[qwen36-moe lifecycle-timings] prompt_setup_ms=1.0 bake_open_ms=2.0 "
            "layer_load_ms=3.0 session_ms=4.0 prefill_steps=8190 "
            "prefill_embed_ms=5.0 prefill_chain_ms=6000.0 prefill_total_ms=6005.0 "
            "generation_wall_ms=30.0 total_wall_ms=7000.0\n"
        )
        self.assertEqual(bench_qwen36_longctx.parse_generated_json(output), "SSB-NEEDLE-00007\n")
        self.assertEqual(bench_qwen36_longctx.parse_tokens(output), [1, 2, 3])
        self.assertEqual(bench_qwen36_longctx.parse_result(output)["prompt_tokens"], 8190)
        self.assertEqual(bench_qwen36_longctx.parse_stage_timings(output)["total_ms_avg"], 10.5)
        self.assertEqual(bench_qwen36_longctx.parse_stage_timings(output)["chain_total_ms"], 30)
        chain = bench_qwen36_longctx.parse_chain_breakdown(output)
        self.assertEqual(chain["full_attn_ms_avg"], 7.0)
        self.assertEqual(bench_qwen36_longctx.likely_bottleneck({"chain_breakdown": chain}), "full_attn")
        lifecycle = bench_qwen36_longctx.parse_lifecycle_timings(output)
        self.assertEqual(lifecycle["prefill_steps"], 8190)
        self.assertEqual(lifecycle["prefill_total_ms"], 6005.0)
        self.assertEqual(lifecycle["generation_wall_ms"], 30.0)

    def test_apply_preset_defaults_allows_explicit_overrides(self):
        args = SimpleNamespace(
            preset="comparison",
            contexts=None,
            modes="int4-vmm",
            sparse_caps=None,
            max_new_tokens=None,
            timeout=None,
            warmup=None,
        )
        applied = bench_qwen36_longctx.apply_preset_defaults(args)
        self.assertEqual(applied.contexts, "512,2048,4096,8192")
        self.assertEqual(applied.modes, "int4-vmm")
        self.assertEqual(applied.max_new_tokens, 4)
        self.assertTrue(applied.warmup)

    def test_summarize_ranks_best_mode_and_recommendation(self):
        rows = [
            {
                "context_tokens_requested": 512,
                "mode": "int4-vmm",
                "returncode": 0,
                "stage": {"total_ms_avg": 40.0},
                "chain_breakdown": {"full_attn_ms_avg": 30.0, "ffn_ms_avg": 5.0},
                "lifecycle": {"prefill_total_ms": 1000.0, "generation_wall_ms": 41.0},
            },
            {
                "context_tokens_requested": 512,
                "mode": "int4-kv-fp8",
                "returncode": 0,
                "stage": {"total_ms_avg": 35.0},
                "chain_breakdown": {"full_attn_ms_avg": 27.0, "ffn_ms_avg": 4.0},
                "lifecycle": {"prefill_total_ms": 900.0, "generation_wall_ms": 36.0},
            },
        ]
        summary = bench_qwen36_longctx.summarize(rows)
        self.assertEqual(summary[0]["best_mode"], "int4-kv-fp8")
        self.assertEqual(summary[0]["best_vs_baseline_pct"], -12.5)
        self.assertEqual(summary[0]["likely_bottleneck"], "full_attn")
        self.assertIn("full-attention", bench_qwen36_longctx.recommendation(summary))

    def test_summarize_accepts_metal_int4_as_baseline(self):
        rows = [
            {
                "context_tokens_requested": 512,
                "mode": "int4",
                "returncode": 0,
                "stage": {"total_ms_avg": 42.0},
                "chain_breakdown": {"full_attn_ms_avg": 30.0},
                "lifecycle": {"prefill_total_ms": 1000.0, "generation_wall_ms": 43.0},
            }
        ]
        summary = bench_qwen36_longctx.summarize(rows)
        self.assertEqual(summary[0]["best_mode"], "int4")
        self.assertEqual(summary[0]["baseline_ms_per_tok"], 42.0)
        self.assertEqual(summary[0]["best_vs_baseline_pct"], 0.0)

    def test_parse_result_accepts_qwen36_generated_summary(self):
        output = "Generated 1 token (418 prompt + 1 new). EOS: no (max_new_tokens hit)."
        result = bench_qwen36_longctx.parse_result(output)
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(result["prompt_tokens"], 418)
        self.assertEqual(result["new_tokens"], 1)

    def test_build_run_env_clears_inherited_vmm_when_force_disabled(self):
        base_env = {
            "SUPERSONIC_VMM_MOE_ISLANDS": "1",
            "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS": "320",
            "SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON": "old.json",
        }
        args = SimpleNamespace(backend="hip", force_moe_vmm=False)
        mode = bench_qwen36_longctx.BenchMode("int4-vmm", kv_fp8=False)
        env = bench_qwen36_longctx.build_run_env(base_env, args, mode, Path("new.json"))
        self.assertEqual(env["SUPERSONIC_BACKENDS"], "hip")
        self.assertNotIn("SUPERSONIC_VMM_MOE_ISLANDS", env)
        self.assertNotIn("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS", env)
        self.assertNotIn("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON", env)

    def test_build_run_env_forces_vmm_for_sparse_mode(self):
        args = SimpleNamespace(backend="hip", force_moe_vmm=False)
        mode = bench_qwen36_longctx.BenchMode(
            "cap320",
            kv_fp8=False,
            sparse_cap=320,
            prefetch_mode="transition",
            prefetch_ranks="4",
        )
        env = bench_qwen36_longctx.build_run_env({}, args, mode, Path("telemetry.json"))
        self.assertEqual(env["SUPERSONIC_VMM_MOE_ISLANDS"], "1")
        self.assertEqual(env["SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"], "320")
        self.assertEqual(env["SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON"], "telemetry.json")
        self.assertEqual(env["SUPERSONIC_MOE_ISLAND_PREFETCH"], "transition")
        self.assertEqual(env["SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS"], "4")


if __name__ == "__main__":
    unittest.main()
