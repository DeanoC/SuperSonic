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
        self.uncontrolled_cold["workload"]["cache_state"] = "warm-resident"
        self.uncontrolled_cold["environment"]["cache_state"] = "warm-resident"
        self.uncontrolled_cold["workload"]["warmups"] = 1
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

    def test_forged_uncontrolled_records_never_compute_speedup(self):
        left = copy.deepcopy(self.locked_warm)
        right = copy.deepcopy(self.locked_warm)
        for record in (left, right):
            record["environment"]["clock_policy"] = "uncontrolled-clocks"
            record["hardware"]["clock_policy"] = "uncontrolled-clocks"
            record["environment"]["headline_eligible"] = True
            record["environment"]["verification_errors"] = []
        right["samples"] = [
            {"decode_ms": 15.0, "tokens_per_second": 2133.3333333333},
            {"decode_ms": 14.0, "tokens_per_second": 2285.7142857143},
            {"decode_ms": 16.0, "tokens_per_second": 2000.0},
        ]

        result = self.compare.compare_records(left, right)

        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("headline_eligible", result.reasons)

    def test_different_artifact_digest_never_computes_speedup(self):
        left = copy.deepcopy(self.locked_warm)
        right = copy.deepcopy(self.locked_warm)
        right["artifact"]["sha256"] = "f" * 64
        right["artifact"]["semantic_id"] = "llama-cpp-artifact-sha256-" + ("f" * 64)
        result = self.compare.compare_records(left, right)
        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("sha256", result.reasons)

    def test_different_artifact_source_revision_never_computes_speedup(self):
        left = copy.deepcopy(self.locked_warm)
        right = copy.deepcopy(self.locked_warm)
        right["artifact"]["source_revision"] = "2" * 40

        result = self.compare.compare_records(left, right)

        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("source_revision", result.reasons)

    def test_cross_physical_gpu_records_never_compute_speedup(self):
        left = copy.deepcopy(self.locked_warm)
        right = copy.deepcopy(self.locked_warm)
        right["hardware"]["physical_gpu"] = "2"
        right["environment"]["physical_gpu"] = "2"
        result = self.compare.compare_records(left, right)
        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("physical_gpu", result.reasons)

    def test_different_physical_bdf_records_never_compute_speedup(self):
        left = copy.deepcopy(self.locked_warm)
        right = copy.deepcopy(self.locked_warm)
        right["hardware"]["identity"] = "0000:66:00.0"
        right["hardware"]["identity_fields"]["identity"] = "0000:66:00.0"
        result = self.compare.compare_records(left, right)
        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("identity", result.reasons)

    def test_drifted_locked_record_never_computes_speedup(self):
        drifted = copy.deepcopy(self.locked_warm)
        drift_sample = copy.deepcopy(drifted["environment"]["telemetry_samples"][0])
        drift_sample["gpu_clock_mhz"] = 2300
        drifted["environment"]["telemetry_samples"] = [copy.deepcopy(drift_sample) for _ in range(3)]
        drifted["environment"]["headline_eligible"] = True
        drifted["environment"]["verification_errors"] = []
        drifted["samples"] = [
            {"decode_ms": 15.0, "tokens_per_second": 2133.3333333333},
            {"decode_ms": 14.0, "tokens_per_second": 2285.7142857143},
            {"decode_ms": 16.0, "tokens_per_second": 2000.0},
        ]

        result = self.compare.compare_records(self.locked_warm, drifted)

        self.assertFalse(result.comparable)
        self.assertIsNone(result.speedup)
        self.assertIn("headline_eligible", result.reasons)

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
            "environment.requested.power_cap_watts": ("environment", "requested", {"power_cap_watts": 280}),
        }

        for reason, mutation in mutations.items():
            with self.subTest(reason=reason):
                right = copy.deepcopy(self.locked_warm)
                if reason == "environment.requested.power_cap_watts":
                    right["environment"]["requested"]["power_cap_watts"] = 280
                elif len(mutation) == 3:
                    section, key, value = mutation
                    right[section][key] = value
                else:
                    raise AssertionError("unexpected mutation fixture")
                result = self.compare.compare_records(self.locked_warm, right)
                self.assertFalse(result.comparable)
                self.assertIsNone(result.speedup)
                self.assertIn(reason.rsplit(".", 1)[-1], result.reasons)

    def test_series_key_splits_all_comparable_identity_fields(self):
        base_key = self.compare.series_key(self.locked_warm)
        for path, value in (
            (("workload", "cache_state"), "warm-resident"),
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

    def test_requested_clock_power_level_cache_process_and_boundary_mismatches_forbid_speedup(self):
        mutations = (
            (("environment", "rocm_version"), "ROCm 7.0.0", "rocm_version"),
            (("environment", "hip_version"), "HIP 7.0.0", "hip_version"),
            (("environment", "requested", "gpu_clock_mhz"), 2500, "gpu_clock_mhz"),
            (("environment", "requested", "memory_clock_mhz"), 1300, "memory_clock_mhz"),
            (("environment", "requested", "power_cap_watts"), 280, "power_cap_watts"),
            (("environment", "requested", "performance_level"), "auto", "performance_level"),
            (("environment", "requested", "temperature_limit_celsius"), 80.0, "temperature_limit_celsius"),
            (("workload", "cache_state"), "warm-resident", "cache_state"),
            (("environment", "process_reuse"), True, "process_reuse"),
            (("workload", "measurement_boundary"), "end-to-end", "measurement_boundary"),
        )
        for path, value, reason in mutations:
            with self.subTest(path=path):
                right = copy.deepcopy(self.locked_warm)
                target = right
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                result = self.compare.compare_records(self.locked_warm, right)
                self.assertFalse(result.comparable)
                self.assertIsNone(result.speedup)
                self.assertIn(reason, result.reasons)
                self.assertNotEqual(
                    self.compare.series_key(self.locked_warm),
                    self.compare.series_key(right),
                )


if __name__ == "__main__":
    unittest.main()
