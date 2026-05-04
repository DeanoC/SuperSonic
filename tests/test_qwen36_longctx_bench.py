import importlib.util
import sys
import unittest
from pathlib import Path


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
        )
        self.assertEqual(bench_qwen36_longctx.parse_generated_json(output), "SSB-NEEDLE-00007\n")
        self.assertEqual(bench_qwen36_longctx.parse_tokens(output), [1, 2, 3])
        self.assertEqual(bench_qwen36_longctx.parse_result(output)["prompt_tokens"], 8190)
        self.assertEqual(bench_qwen36_longctx.parse_stage_timings(output)["total_ms_avg"], 10.5)
        self.assertEqual(bench_qwen36_longctx.parse_stage_timings(output)["chain_total_ms"], 30)

    def test_parse_result_accepts_qwen36_generated_summary(self):
        output = "Generated 1 token (418 prompt + 1 new). EOS: no (max_new_tokens hit)."
        result = bench_qwen36_longctx.parse_result(output)
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(result["prompt_tokens"], 418)
        self.assertEqual(result["new_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
