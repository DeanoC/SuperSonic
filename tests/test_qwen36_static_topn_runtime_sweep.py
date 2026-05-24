import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_static_topn_runtime.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_static_topn_runtime", SCRIPT)
sweep_qwen36_static_topn_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_static_topn_runtime
SPEC.loader.exec_module(sweep_qwen36_static_topn_runtime)


SAMPLE_OUTPUT = """
Generated 4 tokens (1 prompt + 4 new). EOS: no (max_new_tokens hit).
  Generated ids: [11, 353, 599, 264]
[result] prompt_tokens=1 generated_tokens=4 decode_ms=939 ms_per_step=234.7
[qwen36-moe stage-timings] gen_steps=4 embed_ms_avg=1.000 chain_ms_avg=210.000 lm_head_ms_avg=20.000 sample_ms_avg=0.100 detok_ms_avg=0.001 total_ms_avg=234.700 (chain_total_ms=840 lm_head_total_ms=80)
[qwen36-moe chain-breakdown] gen_steps=4 full_attn_ms_avg=20.000 linear_attn_ms_avg=60.000 ffn_ms_avg=128.730 (full_attn_total_ms=80 linear_attn_total_ms=240 ffn_total_ms=514.9)
[qwen36-moe lifecycle-timings] prompt_setup_ms=10.000 bake_open_ms=1.000 layer_load_ms=100.000 session_ms=1000.000 prefill_steps=0 prefill_embed_ms=0.000 prefill_chain_ms=0.000 prefill_total_ms=0.000 generation_wall_ms=939.000 total_wall_ms=1111.000
[qwen36-expert-residency] calls=160 entries=40 exact_hits=9 route_refills=0 allocations=9 copied_bytes=879660288 exact_hit_rate=0.056250 slot_hits=900 slot_misses=380 slot_hit_rate=0.703125 evictions=0 avg_active_groups=8.000000 max_active_groups=8 avg_copy_bytes=97740032.000
[qwen36-expert-residency-policy] resident_format=native_int4 scope=per_layer miss_policy=static_topn capacity=64 calls=160 exact_hits=9 route_refills=0 allocations=9 copied_bytes=879660288 exact_hit_rate=0.056250 slot_hits=900 slot_misses=380 slot_hit_rate=0.703125 evictions=0 avg_active_groups=8.000000 max_active_groups=8 avg_copy_bytes=97740032.000
[metal-profile] calls=2 total_ms=12.000 native_ms=10.000 host_ms=2.000
[metal-profile-op] op=qwen36_ffn_int4_stage5 path=native calls=1 mean_ms=10.0000 total_ms=10.000 max_ms=10.000
[hal-profile] calls=1 total_ms=3.000 alloc_calls=0 alloc_bytes=0 h2d=0 d2h=0 d2d=0 memset=0 sync_calls=0
[hal-profile-op] op=copy_h2d calls=1 mean_ms=3.0000 total_ms=3.000 max_ms=3.000 total_bytes=1024
"""


class Qwen36StaticTopNRuntimeSweepTests(unittest.TestCase):
    def test_parse_runtime_rows(self):
        parse = sweep_qwen36_static_topn_runtime

        self.assertEqual(parse.parse_generated_ids(SAMPLE_OUTPUT), [11, 353, 599, 264])
        self.assertEqual(parse.parse_result(SAMPLE_OUTPUT)["generated_tokens"], 4)
        self.assertAlmostEqual(
            parse.parse_chain_breakdown(SAMPLE_OUTPUT)["ffn_ms_avg"], 128.730
        )
        residency = parse.parse_expert_residency(SAMPLE_OUTPUT)
        self.assertIsNotNone(residency)
        self.assertEqual(residency["calls"], 160)
        self.assertAlmostEqual(residency["slot_hit_rate"], 0.703125)
        policies = parse.parse_expert_residency_policies(SAMPLE_OUTPUT)
        self.assertEqual(policies[0]["miss_policy"], "static_topn")
        self.assertEqual(policies[0]["capacity"], 64)
        metal = parse.parse_profile(SAMPLE_OUTPUT, "[metal-profile]", "[metal-profile-op]")
        self.assertIsNotNone(metal)
        self.assertEqual(metal["summary"]["native_ms"], 10.0)
        self.assertEqual(metal["entries"][0]["op"], "qwen36_ffn_int4_stage5")
        hal = parse.parse_profile(SAMPLE_OUTPUT, "[hal-profile]", "[hal-profile-op]")
        self.assertIsNotNone(hal)
        self.assertEqual(hal["entries"][0]["total_bytes"], 1024)

    def test_parse_modes_normalizes_aliases(self):
        parse_modes = sweep_qwen36_static_topn_runtime.parse_modes

        self.assertEqual(
            parse_modes("baseline,packed,static-hotset,static-mps-partial,baseline"),
            ["default", "packed", "static-hotset", "mps-static-partial"],
        )
        with self.assertRaises(ValueError):
            parse_modes("unknown")

    def test_mode_env_overrides(self):
        script = sweep_qwen36_static_topn_runtime
        args = script.parse_args([])
        args.static_table_json = Path("table.json")
        args.static_capacity = 64
        args.hotset_capacity = 32
        args.metal_profile = True

        env = script.build_env_overrides(args, "static-hotset")

        self.assertEqual(env["SUPERSONIC_METAL_PROFILE"], "1")
        self.assertEqual(env["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5"], "1")
        self.assertEqual(env["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN"], "1")
        self.assertEqual(env["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY"], "64")
        self.assertEqual(env["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET"], "1")
        self.assertEqual(env["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_HOTSET_CAPACITY"], "32")

        env = script.build_env_overrides(args, "mps-static-partial")
        self.assertEqual(env["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN"], "1")
        self.assertEqual(
            env["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_STATIC_TOPN_PARTIAL"],
            "1",
        )
        self.assertNotIn("SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5", env)

    def test_report_summary_detects_generation_mismatch(self):
        script = sweep_qwen36_static_topn_runtime
        args = script.parse_args([])
        rows = [
            {"status": "ok", "prompt_id": "p", "mode": "default", "generated_ids": [1, 2]},
            {"status": "ok", "prompt_id": "p", "mode": "static", "generated_ids": [1, 3]},
        ]

        report = script.build_report(rows, args, ["default", "static"], "smoke")

        self.assertFalse(report["summary"]["generated_ids_match"])
        md = script.render_markdown(report)
        self.assertIn("Static Top-N Runtime Sweep", md)
        self.assertIn("generated_ids_match", md)

    def test_report_summary_compares_ids_within_each_prompt(self):
        script = sweep_qwen36_static_topn_runtime
        args = script.parse_args([])
        rows = [
            {"status": "ok", "prompt_id": "p1", "mode": "default", "generated_ids": [1, 2]},
            {"status": "ok", "prompt_id": "p1", "mode": "static", "generated_ids": [1, 2]},
            {"status": "ok", "prompt_id": "p2", "mode": "default", "generated_ids": [3, 4]},
            {"status": "ok", "prompt_id": "p2", "mode": "static", "generated_ids": [3, 4]},
        ]

        report = script.build_report(rows, args, ["default", "static"], "comparison")

        self.assertTrue(report["summary"]["generated_ids_match"])
        self.assertEqual(
            report["summary"]["reference_generated_ids_by_prompt"],
            {"p1": [1, 2], "p2": [3, 4]},
        )


if __name__ == "__main__":
    unittest.main()
