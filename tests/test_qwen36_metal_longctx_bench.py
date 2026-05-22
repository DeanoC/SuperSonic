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


if __name__ == "__main__":
    unittest.main()
