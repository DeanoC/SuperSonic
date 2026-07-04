import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parent / "gfx1100" / "bench_qwen36_he_supersonic.py"
SPEC = importlib.util.spec_from_file_location("bench_qwen36_he_supersonic", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class BenchQwen36HeSuperSonicTests(unittest.TestCase):
    def test_render_chatml_wraps_messages_for_lucebox_jsonl(self):
        prompt = bench.render_chatml(
            [{"role": "user", "content": "Complete this function."}]
        )

        self.assertEqual(
            prompt,
            "<|im_start|>user\nComplete this function.<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_render_chatml_can_include_qwen3_no_thinking_prefill(self):
        prompt = bench.render_chatml(
            [{"role": "user", "content": "Complete this function."}],
            no_thinking=True,
        )

        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_load_jsonl_prompts_supports_raw_and_chatml(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as tmp:
            tmp.write(
                json.dumps(
                    {
                        "id": "he_x",
                        "messages": [
                            {"role": "user", "content": "def f():\n    return"}
                        ],
                    }
                )
                + "\n"
            )
            tmp.flush()
            path = Path(tmp.name)

            self.assertEqual(
                bench.load_lucebox_jsonl_prompts(path, "raw"),
                [("he_x", "def f():\n    return")],
            )
            chatml = bench.load_lucebox_jsonl_prompts(path, "chatml")
            no_thinking = bench.load_lucebox_jsonl_prompts(
                path, "chatml-no-thinking"
            )
        self.assertEqual(chatml[0][0], "he_x")
        self.assertIn("<|im_start|>user\n", chatml[0][1])
        self.assertTrue(chatml[0][1].endswith("<|im_start|>assistant\n"))
        self.assertTrue(no_thinking[0][1].endswith("<think>\n\n</think>\n\n"))

    def test_resolve_lucebox_q4_alias_uses_config_dir_and_gguf(self):
        args = types.SimpleNamespace(
            dflash_draft_variant="lucebox-q4-k-m",
            dflash_draft_dir=Path("/custom/config"),
            dflash_draft_gguf=None,
        )

        config_dir, gguf, label = bench.resolve_dflash_draft(args)

        self.assertEqual(label, "lucebox-q4-k-m")
        self.assertEqual(config_dir, bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR)
        self.assertEqual(
            gguf,
            bench.DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q4_k_m.gguf",
        )

    def test_apply_target_profile_sets_qwen36_35b_a3b_defaults(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            model=None,
            model_dir=None,
            quant=None,
            out_json=None,
        )

        bench.apply_target_profile(args)

        self.assertEqual(args.model, "qwen3.6-35b-a3b")
        self.assertEqual(args.model_dir, bench.DEFAULT_35B_A3B_MODEL_DIR)
        self.assertEqual(args.quant, "int4")
        self.assertEqual(args.out_json, bench.DEFAULT_35B_A3B_OUT_JSON)

    def test_apply_target_profile_sets_qwen36_35b_a3b_flm_defaults(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b-flm",
            model=None,
            model_dir=None,
            quant=None,
            out_json=None,
        )

        bench.apply_target_profile(args)

        self.assertIsNone(args.model)
        self.assertEqual(args.model_dir, bench.DEFAULT_35B_A3B_FLM_MODEL_DIR)
        self.assertEqual(args.quant, "none")
        self.assertEqual(args.out_json, bench.DEFAULT_35B_A3B_FLM_OUT_JSON)

    def test_qwen36_35b_a3b_flm_profile_points_at_current_e2e_artifact(self):
        self.assertEqual(
            bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            Path(
                "/mnt/data/tmp/flm-first-class-e2e-20260704/"
                "qwen36-35b-a3b-supersonic-native-int4.flm"
            ),
        )

    def test_apply_target_profile_preserves_explicit_args(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            model="custom-model",
            model_dir=Path("/custom/model"),
            quant="none",
            out_json=Path("/tmp/custom.json"),
        )

        bench.apply_target_profile(args)

        self.assertEqual(args.model, "custom-model")
        self.assertEqual(args.model_dir, Path("/custom/model"))
        self.assertEqual(args.quant, "none")
        self.assertEqual(args.out_json, Path("/tmp/custom.json"))

    def test_lucebox_serving_mode_does_not_enable_dflash_for_35b_a3b(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            prompt_source="script",
            prompt_format="raw",
            ignore_eos=True,
            dflash=False,
            dflash_draft_variant=None,
            context_size=bench.DEFAULT_CONTEXT_SIZE,
        )

        bench.apply_lucebox_serving_mode(args)

        self.assertEqual(args.prompt_source, "jsonl")
        self.assertEqual(args.prompt_format, "chatml-no-thinking")
        self.assertFalse(args.ignore_eos)
        self.assertFalse(args.dflash)
        self.assertIsNone(args.dflash_draft_variant)
        self.assertEqual(args.context_size, bench.LUCEBOX_SERVING_CONTEXT_SIZE)

    def test_build_summary_includes_token_weighted_throughput(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": True,
            },
            {"returncode": 1},
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean_tok_s"], 37.5)
        self.assertEqual(summary["weighted_tok_s"], 37.5)
        self.assertEqual(summary["total_generated_tokens"], 30)
        self.assertEqual(summary["total_decode_ms"], 800.0)
        self.assertEqual(summary["stopped_early_count"], 1)

    def test_build_summary_includes_mean_lifecycle_timings_when_present(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "model_source_ms": 0.010,
                    "layer_load_ms": 6000.0,
                    "generation_wall_ms": 40.0,
                    "total_wall_ms": 6400.0,
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "model_source_ms": 0.030,
                    "layer_load_ms": 5000.0,
                    "generation_wall_ms": 50.0,
                    "total_wall_ms": 5600.0,
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["mean_lifecycle_timings"],
            {
                "model_source_ms": 0.020,
                "layer_load_ms": 5500.0,
                "generation_wall_ms": 45.0,
                "total_wall_ms": 6000.0,
            },
        )

    def test_run_one_sets_gguf_env_and_stop_on_eos_command(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model="qwen3.6-27b",
            model_dir=Path("/models/target"),
            context_size=1024,
            warmup_new_tokens=2,
            n_gen=256,
            seed=1,
            prompt_no_special_tokens=True,
            quant="q4km",
            ignore_eos=False,
            emit_stage_timings=False,
            kv_fp8=False,
            dflash=True,
            dflash_block=0,
            dflash_draft_variant="lucebox-q8-0",
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
        )

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="[result] prompt_tokens=12 generated_tokens=97 decode_ms=970 ms_per_tok=10\n",
        )
        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "prompt")

        cmd = run.call_args.args[0]
        env = run.call_args.kwargs["env"]
        self.assertIn("--dflash", cmd)
        self.assertIn("--prompt-no-special-tokens", cmd)
        self.assertNotIn("--ignore-eos", cmd)
        self.assertEqual(
            env["SUPERSONIC_DFLASH_DRAFT_GGUF"],
            str(bench.DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q8_0.gguf"),
        )
        self.assertEqual(row["generated_tokens"], 97)
        self.assertTrue(row["stopped_early"])
        self.assertEqual(row["dflash_draft_label"], "lucebox-q8-0")

    def test_run_one_flm_profile_omits_int4_and_records_lifecycle_timings(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
        )

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=(
                "[flm] inferred model qwen3.6-35b-a3b from runtime descriptor\n"
                "[result] prompt_tokens=1 generated_tokens=1 decode_ms=41 "
                "ms_per_step=41.0\n"
                "[qwen36-moe lifecycle-timings] prompt_setup_ms=55.351 "
                "model_source_ms=0.009 layer_load_ms=5744.555 session_ms=466.101 "
                "prefill_steps=0 prefill_embed_ms=0.000 prefill_chain_ms=0.000 "
                "prefill_total_ms=0.000 generation_wall_ms=41.006 "
                "total_wall_ms=6307.287\n"
            ),
        )
        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "Hello")

        cmd = run.call_args.args[0]
        self.assertIn("--model-dir", cmd)
        self.assertIn(str(bench.DEFAULT_35B_A3B_FLM_MODEL_DIR), cmd)
        self.assertNotIn("--model", cmd)
        self.assertNotIn("qwen3.6-35b-a3b", cmd)
        self.assertNotIn("--int4", cmd)
        self.assertIn("--emit-stage-timings", cmd)
        self.assertEqual(row["generated_tokens"], 1)
        self.assertEqual(row.get("resolved_model"), "qwen3.6-35b-a3b")
        self.assertEqual(
            row["lifecycle_timings"],
            {
                "prompt_setup_ms": 55.351,
                "model_source_ms": 0.009,
                "layer_load_ms": 5744.555,
                "session_ms": 466.101,
                "prefill_steps": 0.0,
                "prefill_embed_ms": 0.0,
                "prefill_chain_ms": 0.0,
                "prefill_total_ms": 0.0,
                "generation_wall_ms": 41.006,
                "total_wall_ms": 6307.287,
            },
        )


if __name__ == "__main__":
    unittest.main()
