import copy
import importlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "benchmark_fixtures" / "valid-result-v1.json"


def load_compare_module():
    try:
        return importlib.import_module("tools.benchmark.compare")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.compare is absent") from exc


class BenchmarkCompareTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.compare = load_compare_module()
        self.locked_warm = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.uncontrolled_cold = copy.deepcopy(self.locked_warm)
        self.uncontrolled_cold["workload"]["cache_state"] = "cold-load"
        self.uncontrolled_cold["environment"]["cache_state"] = "cold-load"
        self.uncontrolled_cold["workload"]["warmups"] = 0
        self.uncontrolled_cold["environment"]["clock_policy"] = "uncontrolled-clocks"
        self.uncontrolled_cold["hardware"]["clock_policy"] = "uncontrolled-clocks"
        self.uncontrolled_cold["environment"]["headline_eligible"] = False
        self.uncontrolled_cold["environment"]["verification_errors"] = []

    def test_statistics_retain_raw_distribution(self):
        summary = self.compare.summarize_samples([3.0, 1.0, 2.0])

        self.assertEqual((summary.minimum, summary.median, summary.maximum), (1.0, 2.0, 3.0))
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.values, (3.0, 1.0, 2.0))
        self.assertIsNotNone(summary.mad)

    def test_cache_and_clock_mismatch_forbid_speedup(self):
        result = self.compare.compare_records(self.locked_warm, self.uncontrolled_cold)

        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("cache_state", result.reasons)
        self.assertIn("clock_policy", result.reasons)

    def test_comparable_records_compute_ratio_from_decode_medians(self):
        faster = copy.deepcopy(self.locked_warm)
        faster["samples"] = [
            {"decode_ms": 15.0, "tokens_per_second": 2133.3333333333},
            {"decode_ms": 14.0, "tokens_per_second": 2285.7142857143},
            {"decode_ms": 16.0, "tokens_per_second": 2000.0},
        ]

        result = self.compare.compare_records(self.locked_warm, faster)

        self.assertTrue(result.comparable)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.left.median, 30.0)
        self.assertEqual(result.right.median, 15.0)
        self.assertEqual(result.speedup, 2.0)

    def test_semantic_mismatches_are_validator_owned(self):
        mutations = {
            "hardware.identity": ("hardware", "identity", "Different GPU"),
            "hardware.architecture": ("hardware", "architecture", "gfx1100"),
            "artifact.semantic_id": ("artifact", "semantic_id", "different-artifact"),
            "artifact.quantization": ("artifact", "quantization", "Different-Q4"),
            "artifact.tokenizer_sha256": (
                "artifact",
                "tokenizer_sha256",
                "1111111111111111111111111111111111111111111111111111111111111111",
            ),
            "artifact.chat_template_sha256": (
                "artifact",
                "chat_template_sha256",
                "2222222222222222222222222222222222222222222222222222222222222222",
            ),
            "workload.case_id": ("workload", "case_id", "quick-long-warm-ordinary"),
            "workload.prompt_sha256": (
                "workload",
                "prompt_sha256",
                "3333333333333333333333333333333333333333333333333333333333333333",
            ),
            "workload.context_limit": ("workload", "context_limit", 4096),
            "workload.max_new_tokens": ("workload", "max_new_tokens", 64),
            "workload.mode": ("workload", "mode", "mtp"),
            "workload.stop_policy": ("workload", "stop_policy", "honor-eos"),
            "workload.warmups": ("workload", "warmups", 2),
            "workload.measurement_boundary": ("workload", "measurement_boundary", "end-to-end"),
            "environment.power_cap_watts": ("environment", "requested", {"power_cap_watts": 280}),
        }

        for reason, mutation in mutations.items():
            with self.subTest(reason=reason):
                right = copy.deepcopy(self.locked_warm)
                if len(mutation) == 3:
                    section, key, value = mutation
                    right[section][key] = value
                else:
                    raise AssertionError("unexpected mutation fixture")
                if reason == "environment.power_cap_watts":
                    right["environment"]["requested"]["power_cap_watts"] = 280
                result = self.compare.compare_records(self.locked_warm, right)
                self.assertFalse(result.comparable)
                self.assertIsNone(result.speedup)
                self.assertIn(reason.rsplit(".", 1)[-1], result.reasons)

    def test_series_key_splits_all_comparable_identity_fields(self):
        base_key = self.compare.series_key(self.locked_warm)
        for path, value in (
            (("workload", "cache_state"), "cold-load"),
            (("environment", "clock_policy"), "uncontrolled-clocks"),
            (("environment", "requested", "power_cap_watts"), 280),
            (("artifact", "semantic_id"), "other-artifact"),
            (("artifact", "tokenizer_sha256"), "4444444444444444444444444444444444444444444444444444444444444444"),
            (("artifact", "chat_template_sha256"), "5555555555555555555555555555555555555555555555555555555555555555"),
            (("workload", "case_id"), "quick-long-warm-ordinary"),
            (("workload", "measurement_boundary"), "end-to-end"),
        ):
            with self.subTest(path=path):
                record = copy.deepcopy(self.locked_warm)
                target = record
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertNotEqual(base_key, self.compare.series_key(record))


if __name__ == "__main__":
    unittest.main()
