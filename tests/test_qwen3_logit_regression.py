import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "gfx1100" / "qwen3_logit_regression.py"
SPEC = importlib.util.spec_from_file_location("qwen3_logit_regression", SCRIPT)
qwen3_logit_regression = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = qwen3_logit_regression
SPEC.loader.exec_module(qwen3_logit_regression)


class Qwen3LogitRegressionTests(unittest.TestCase):
    def test_parse_int_list_dedupes_and_rejects_invalid_values(self):
        parse = qwen3_logit_regression.parse_int_list
        self.assertEqual(parse("8,64,8"), [8, 64])
        with self.assertRaises(ValueError):
            parse("")
        with self.assertRaises(ValueError):
            parse("0")

    def test_select_cases_rejects_unknown_context_label(self):
        cases = qwen3_logit_regression.select_cases([8, 210])
        self.assertEqual([case.name for case in cases], ["ctx8", "ctx210"])
        self.assertEqual(cases[0].context_size, 8)
        self.assertEqual(cases[1].context_size, 256)
        with self.assertRaises(ValueError):
            qwen3_logit_regression.select_cases([9])

    def test_output_parsers_extract_qwen3_fields(self):
        output = (
            '  prompt: "Hello world" -> 2 tokens\n'
            " text\n"
            "LAST_LOGITS: 1.0,2.5,-3\n"
            "Generated 1 token in 0.50s (2.00 tok/s).\n"
            "[qwen3-moe stage-timings] gen_steps=1 path=persistent "
            "embed_ms_avg=0.010 decode_ms_avg=0.765 lm_head_ms_avg=1.250 sample_ms_avg=0.020\n"
        )
        self.assertEqual(qwen3_logit_regression.parse_prompt_tokens(output), 2)
        self.assertEqual(qwen3_logit_regression.parse_generated_count(output), 1)
        self.assertEqual(qwen3_logit_regression.parse_last_logits(output), [1.0, 2.5, -3.0])
        stage = qwen3_logit_regression.parse_stage_timings(output)
        self.assertEqual(stage["path"], "persistent")
        self.assertEqual(stage["decode_ms_avg"], 0.765)

    def test_compare_logits_reports_top10_argmax_cosine_and_max_abs(self):
        reference = [0.0, 4.0, 1.0, -1.0]
        candidate = [0.0, 3.5, 1.5, -1.0]
        metrics = qwen3_logit_regression.compare_logits(reference, candidate, top_k=3)
        self.assertTrue(metrics["argmax_match"])
        self.assertEqual(metrics["reference_argmax"], 1)
        self.assertEqual(metrics["candidate_argmax"], 1)
        self.assertEqual(metrics["top10_overlap"], 3)
        self.assertAlmostEqual(metrics["max_abs"], 0.5)
        self.assertGreater(metrics["cosine"], 0.98)

    def test_comparison_passes_uses_all_thresholds(self):
        metrics = {
            "argmax_match": True,
            "top10_overlap": 10,
            "cosine": 0.99999,
            "max_abs": 0.1,
        }
        self.assertTrue(
            qwen3_logit_regression.comparison_passes(
                metrics,
                require_argmax=True,
                top10_overlap_floor=10,
                cosine_floor=0.99997,
                max_abs_ceil=0.25,
            )
        )
        metrics["max_abs"] = 0.5
        self.assertFalse(
            qwen3_logit_regression.comparison_passes(
                metrics,
                require_argmax=True,
                top10_overlap_floor=10,
                cosine_floor=0.99997,
                max_abs_ceil=0.25,
            )
        )


if __name__ == "__main__":
    unittest.main()
