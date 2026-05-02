import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from oracle.qwen36_verify_suite import (
    Case,
    build_ruler_cases,
    evaluate_failures,
    parse_contexts,
    parse_generated_ids,
    run_once,
    stable_seed,
    summarize_case,
)
import oracle.qwen36_verify_suite as verify_suite


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return text.split()

    def encode_plain(self, text):
        return text.split()

    def decode_plain(self, token_ids):
        return " ".join(token_ids)


class Qwen36VerifySuiteTests(unittest.TestCase):
    def test_parse_contexts_accepts_k_suffix(self):
        self.assertEqual(parse_contexts("512,2K,8192"), [512, 2048, 8192])

    def test_stable_seed_changes_by_case(self):
        self.assertEqual(stable_seed(7, "ruler", 512, 0), stable_seed(7, "ruler", 512, 0))
        self.assertNotEqual(stable_seed(7, "ruler", 512, 0), stable_seed(7, "ruler", 512, 1))

    def test_ruler_cases_are_repeatable_and_contain_reference(self):
        args = Namespace(seed=123, ruler_samples=2, max_new_tokens=1)
        a = build_ruler_cases(args, TinyTokenizer(), [128])
        b = build_ruler_cases(args, TinyTokenizer(), [128])
        self.assertEqual([x.prompt for x in a], [x.prompt for x in b])
        self.assertEqual(len(a), 2)
        self.assertEqual(len(a[0].references), 1)
        self.assertIn(a[0].references[0], a[0].prompt)

    def test_generated_ids_parser(self):
        self.assertEqual(parse_generated_ids("Generated ids: [12, 34]\n"), [12, 34])
        self.assertEqual(parse_generated_ids("no ids"), [])

    def test_summary_detects_nondeterministic_hashes(self):
        case = Case("c0", "pg19", 512, "prompt", [], 511)
        runs = [
            {
                "mode": "fp8",
                "repeat_idx": 0,
                "returncode": 0,
                "generated_ids": [1],
                "logits_sha256": "a",
                "hidden_sha256": "h",
                "logits_stats": {"all_zero": False},
                "wall_seconds": 1.0,
            },
            {
                "mode": "fp8",
                "repeat_idx": 1,
                "returncode": 0,
                "generated_ids": [2],
                "logits_sha256": "b",
                "hidden_sha256": "h",
                "logits_stats": {"all_zero": False},
                "wall_seconds": 1.0,
            },
        ]
        summary = summarize_case(case, runs)
        self.assertFalse(summary["modes"]["fp8"]["deterministic_logits"])
        self.assertTrue(summary["modes"]["fp8"]["deterministic_hidden"])
        payload = {"summary": [summary]}
        failures = evaluate_failures(payload, Namespace(fail_on_nondeterminism=True))
        self.assertIn("c0:fp8 logits are nondeterministic", failures)

    def test_failures_detect_all_zero_logits(self):
        payload = {
            "summary": [
                {
                    "case_id": "c1",
                    "modes": {
                        "int4": {
                            "runs": 1,
                            "ok_runs": 1,
                            "deterministic_logits": True,
                            "deterministic_hidden": True,
                            "deterministic_generated_ids": True,
                            "any_all_zero_logits": True,
                        }
                    },
                }
            ]
        }
        failures = evaluate_failures(payload, Namespace(fail_on_nondeterminism=True))
        self.assertEqual(failures, ["c1:int4 produced all-zero logits"])

    def test_run_once_records_timeout_as_failed_row(self):
        def fake_run(*args, **kwargs):
            del args, kwargs
            raise subprocess.TimeoutExpired(
                cmd=["supersonic"],
                timeout=3,
                output=b"Generated ids: [7]\n",
                stderr=b"[qwen36-moe stage-timings] gen_steps=1 total_ms_avg=1\n",
            )

        old_run = verify_suite.subprocess.run
        verify_suite.subprocess.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as td:
                args = Namespace(
                    binary=Path("target/release/supersonic"),
                    backend="hip",
                    model_dir=Path("/tmp/model"),
                    max_new_tokens=1,
                    seed=42,
                    timeout=3,
                    emit_stage_timings=True,
                )
                case = Case("c_timeout", "pg19", 8, "hello", [], 7)
                row = run_once(args, case, "fp8", 0, Path(td))
        finally:
            verify_suite.subprocess.run = old_run

        self.assertTrue(row["timed_out"])
        self.assertEqual(row["returncode"], None)
        self.assertEqual(row["generated_ids"], [7])
        self.assertIn("timed out", row["error"])


if __name__ == "__main__":
    unittest.main()
