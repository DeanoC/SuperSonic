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


if __name__ == "__main__":
    unittest.main()
