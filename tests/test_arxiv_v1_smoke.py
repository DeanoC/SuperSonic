import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from oracle.arxiv_v1_smoke import (
    evaluate_quality_gates,
    load_reference,
    make_niah_single,
    parse_supersonic_generation,
    score_string_match_all,
    stable_seed,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class ArxivV1SmokeTests(unittest.TestCase):
    def test_supersonic_generation_parser_strips_prompt_before_scoring(self):
        prompt = "needle: 12345\nQuestion:"
        output = (
            "[gpu] backend=CUDA\n"
            "[decode] CUDA Llama31 fast greedy lm_head argmax enabled\n"
            "needle: 12345\nQuestion: answer is 67890\n"
            "[tokens] 1 2 3\n"
            "[result] prompt_tokens=4 generated_tokens=3 decode_ms=1 ms_per_step=0.3\n"
        )
        generated = parse_supersonic_generation(output, prompt)
        self.assertEqual(generated, " answer is 67890")
        self.assertEqual(score_string_match_all(generated, ["12345"]), 0.0)
        self.assertEqual(score_string_match_all(generated, ["67890"]), 1.0)

    def test_supersonic_generation_parser_prefers_generated_json(self):
        output = (
            "[decode] CUDA Llama31 fast greedy lm_head argmax enabled\n"
            "needle: 12345\nQuestion: wrong raw segment\n"
            "[generated_json] \" answer is 67890\"\n"
            "[tokens] 1 2 3\n"
        )
        self.assertEqual(
            parse_supersonic_generation(output, "needle: 12345\nQuestion:"),
            " answer is 67890",
        )

    def test_reference_loader_reads_normalized_ruler_smoke_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {
                "benchmark": "ruler",
                "context_length": 4096,
                "config": "certified",
                "quality": {
                    "metric": "accuracy",
                    "value": 0.75,
                    "summary": {
                        "niah_single_4K": {"dense_mean": 1.0, "cert_mean": 0.75, "n": 3}
                    },
                },
            }
            (root / "06_ruler_4K_certified.smoke.json").write_text(json.dumps(payload))
            ref = load_reference(root, "ruler", "certified", 4096, smoke=True)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.metric, "accuracy")
        self.assertEqual(ref.value, 0.75)
        self.assertEqual(ref.subtask_values["niah_single_4K"], 0.75)

    def test_prompt_seed_matches_dotcache_stable_md5_scheme(self):
        seed_a = stable_seed(20260416, "niah_single", 4096, 0)
        seed_b = stable_seed(20260416, "niah_single", 4096, 0)
        seed_c = stable_seed(20260416, "niah_single", 4096, 1)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)

        prompt, refs = make_niah_single(random.Random(seed_a), 512, TinyTokenizer())
        self.assertTrue(prompt.startswith("A special magic number is hidden"))
        self.assertEqual(len(refs), 1)
        self.assertIn(refs[0], prompt)

    def test_quality_gates_detect_reference_and_critical_regressions(self):
        payload = {
            "summary": {
                "certified:niah_single_4K": {
                    "mean_score": 0.0,
                    "reference_score": 1.0,
                }
            },
            "results": [
                {
                    "config": "dense",
                    "context_length": 4096,
                    "subtask": "niah_single",
                    "sample_idx": 0,
                    "score": 1.0,
                },
                {
                    "config": "certified",
                    "context_length": 4096,
                    "subtask": "niah_single",
                    "sample_idx": 0,
                    "score": 0.0,
                },
            ],
        }
        args = Namespace(
            min_score=0.5,
            fail_below_reference=True,
            reference_tolerance=0.0,
            fail_on_critical=True,
        )
        failures = evaluate_quality_gates(payload, args)
        self.assertEqual(payload["critical_failures"], 1)
        self.assertEqual(len(failures), 3)


if __name__ == "__main__":
    unittest.main()
