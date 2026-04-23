import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from oracle.pg19_smoke import (
    aggregate,
    certified_dense_prefix_len,
    evaluate_gates,
    load_reference,
    parse_contexts,
    parse_teacher_forced_json,
    resolve_eval_start_frac,
)


class Pg19SmokeTests(unittest.TestCase):
    def test_parse_contexts_accepts_k_suffix(self):
        self.assertEqual(parse_contexts("512,4K, 8192"), [512, 4096, 8192])

    def test_teacher_forced_json_parser(self):
        output = (
            "[weights] loaded\n"
            "[teacher_forced_json] {\"perplexity\": 6.25, \"scored_tokens\": 10}\n"
        )
        parsed = parse_teacher_forced_json(output)
        self.assertEqual(parsed["perplexity"], 6.25)
        self.assertEqual(parsed["scored_tokens"], 10)

    def test_reference_loader_reads_normalized_pg19(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {
                "benchmark": "pg19",
                "context_length": 4096,
                "config": "certified",
                "quality": {
                    "metric": "perplexity",
                    "value": 6.85,
                    "dense_value": 6.84,
                },
            }
            (root / "04_pg19_4K_certified.smoke.json").write_text(json.dumps(payload))
            ref = load_reference(root, "certified", 4096, smoke=True)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.metric, "perplexity")
        self.assertEqual(ref.value, 6.85)
        self.assertEqual(ref.dense_value, 6.84)

    def test_reference_mode_defaults_to_dotcache_eval_start(self):
        self.assertEqual(resolve_eval_start_frac(None, reference_mode=True), 0.5)
        self.assertEqual(resolve_eval_start_frac(None, reference_mode=False), 0.0)
        self.assertEqual(certified_dense_prefix_len("dense", 4096, 0.5), None)
        self.assertEqual(certified_dense_prefix_len("certified", 4096, 0.5), 2048)

    def test_aggregate_and_gates_detect_certified_delta(self):
        payload = {
            "summary": [
                {"config": "dense", "context_length": 512, "perplexity": 5.0},
                {"config": "certified", "context_length": 512, "perplexity": 5.2},
            ]
        }
        args = Namespace(
            fail_above_reference=False,
            reference_tolerance=0.0,
            max_certified_delta=0.1,
        )
        failures = evaluate_gates(payload, args)
        self.assertEqual(len(failures), 1)
        self.assertAlmostEqual(payload["summary"][1]["delta_vs_dense"], 0.2)

        agg = aggregate([
            {"total_nll": 10.0, "scored_tokens": 5, "total_ms": 50.0},
            {"total_nll": 8.0, "scored_tokens": 5, "total_ms": 40.0},
        ])
        self.assertEqual(agg["total_tokens"], 10)
        self.assertAlmostEqual(agg["ms_per_token"], 9.0)


if __name__ == "__main__":
    unittest.main()
