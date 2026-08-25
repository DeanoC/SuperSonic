import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "benchmark_fixtures" / "valid-result-v1.json"


def load_validation_module():
    try:
        return importlib.import_module("tools.benchmark.validation")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.validation is absent") from exc


class BenchmarkValidationTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.validation = load_validation_module()
        self.valid_record = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        (self.bundle / "one-result.json").write_text(
            json.dumps(self.valid_record, sort_keys=True),
            encoding="utf-8",
        )

    def assert_record_invalid(self, record, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.validation.validate_record(record)

    def assert_bundle_invalid(self, record, pattern):
        path = self.bundle / "mutated-result.json"
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, pattern):
            self.validation.validate_bundle(path, require_complete=True)

    def test_valid_fixture_passes(self):
        self.validation.validate_record(self.valid_record)

    def test_artifact_source_provenance_is_closed_and_required(self):
        record = copy.deepcopy(self.valid_record)
        provenance = {
            "source_repository": "Geometric-AI/Qwen3.8-27B-GQH-Q3KXL-GGUF",
            "source_revision": "91bc7e33c1912856dcd8d2ca4499dd8ccad13ac4",
            "filename": "Qwen3.8-27B-GQH-Q3KXL.gguf",
            "size_bytes": 13440110432,
        }
        record["artifact"].update(provenance)

        self.validation.validate_record(record)
        for field in provenance:
            missing = copy.deepcopy(record)
            missing["artifact"].pop(field)
            self.assert_record_invalid(missing, "missing required fields")

    def test_structured_toolchain_versions_are_required_and_safe(self):
        self.assertRegex(self.valid_record["environment"]["rocm_version"], r"^ROCm ")
        self.assertRegex(self.valid_record["environment"]["hip_version"], r"^HIP ")
        self.assertNotEqual(self.valid_record["engine"]["version"].lower(), "unknown")

        for key in ("rocm_version", "hip_version"):
            record = copy.deepcopy(self.valid_record)
            record["environment"][key] = "unknown"
            self.assert_record_invalid(record, "version|unknown")

            record = copy.deepcopy(self.valid_record)
            record["environment"][key] = record["environment"][key] + " unknown"
            self.assert_record_invalid(record, "version|unknown")

        record = copy.deepcopy(self.valid_record)
        record["engine"]["version"] = "unknown"
        self.assert_record_invalid(record, "engine.*version|unknown")

    def test_version_fields_never_contain_source_paths(self):
        serialized = json.dumps(self.valid_record)
        self.assertNotIn("version-file", serialized)
        self.assertNotRegex(serialized, r"(?:^|[\" ])/(?:home|tmp|workspace)/")

    def test_path_and_non_finite_sample_fail(self):
        record = copy.deepcopy(self.valid_record)
        record["run"]["command"] = ["/home/private/supersonic"]
        record["samples"][0]["decode_ms"] = float("nan")

        with self.assertRaises(ValueError):
            self.validation.validate_record(record)

    def test_incomplete_bundle_is_not_publishable(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.validation.validate_bundle(self.bundle, require_complete=True)

    def test_duration_bundle_requires_sufficient_elapsed_and_balanced_round_evidence(self):
        quick = self.validation.load_suite("quick")
        suite = replace(
            quick,
            name="duration-test",
            minimum_duration_seconds=5,
            performance_cases=(quick.performance_cases[0],),
        )
        record = copy.deepcopy(self.valid_record)
        record["run"]["suite"] = "duration-test"
        (self.bundle / "one-result.json").write_text(json.dumps(record), encoding="utf-8")
        manifest = {
            "run_id": record["run"]["run_id"],
            "suite": {
                "name": "duration-test",
                "budget_seconds": 600,
                "minimum_duration_seconds": 5,
            },
            "status": {
                "state": "complete",
                "records": ["one-result.json"],
                "elapsed_seconds": 4.9,
                "performance_elapsed_seconds": 4.9,
                "completed_rounds": 3,
            },
        }
        (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with mock.patch.object(self.validation, "load_suite", return_value=suite):
            with self.assertRaisesRegex(ValueError, "minimum duration|elapsed"):
                self.validation.validate_bundle(self.bundle, require_complete=True)

            manifest["status"]["elapsed_seconds"] = 5.0
            manifest["status"]["performance_elapsed_seconds"] = 5.0
            manifest["status"]["completed_rounds"] = 2
            (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "balanced|round"):
                self.validation.validate_bundle(self.bundle, require_complete=True)

            manifest["status"]["completed_rounds"] = 3
            (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.validation.validate_bundle(self.bundle, require_complete=True)

    def test_schema_is_recursive_closed_and_strict_about_json_types(self):
        record = copy.deepcopy(self.valid_record)
        record["engine"]["unexpected"] = "field"
        self.assert_record_invalid(record, "unexpected|additional")

        record = copy.deepcopy(self.valid_record)
        record["engine"]["adapter_version"] = True
        self.assert_record_invalid(record, "integer")

        record = copy.deepcopy(self.valid_record)
        record["artifact"]["sha256"] = "not-a-digest"
        self.assert_record_invalid(record, "pattern")

    def test_record_requires_exact_samples_and_quality_summary_consistency(self):
        record = copy.deepcopy(self.valid_record)
        record["samples"].pop()
        self.assert_record_invalid(record, "sample count")

        record = copy.deepcopy(self.valid_record)
        record["quality"]["passed"] = 7
        self.assert_record_invalid(record, "quality")

        record = copy.deepcopy(self.valid_record)
        record["quality"]["cases"][0]["id"] = "instruction-following-2"
        self.assert_record_invalid(record, "quality")

    def test_passed_mtp_equality_requires_matching_evidence_hashes(self):
        record = copy.deepcopy(self.valid_record)
        mtp_case = next(
            case
            for case in record["quality"]["cases"]
            if case["category"] == "ordinary-vs-mtp-token-equality"
        )
        mtp_case["actual_hash"] = "f" * 64

        self.assert_record_invalid(record, "MTP equality.*evidence hashes")

    def test_publishable_bundle_rejects_dirty_errors_failed_quality_and_unverified_headlines(self):
        record = copy.deepcopy(self.valid_record)
        record["run"]["dirty"] = True
        self.assert_bundle_invalid(record, "dirty")

        record = copy.deepcopy(self.valid_record)
        record["status"]["state"] = "failed"
        record["errors"] = [{"code": "runtime-failed", "message": "configured run failed"}]
        self.assert_bundle_invalid(record, "configured|errors|failed")

        record = copy.deepcopy(self.valid_record)
        record["quality"]["failed"] = 1
        record["quality"]["passed"] = 7
        record["quality"]["cases"][0]["passed"] = False
        record["quality"]["cases"][0]["failure"] = "token mismatch"
        self.assert_bundle_invalid(record, "quality")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["headline_eligible"] = False
        record["environment"]["verification_errors"] = ["clock drift"]
        self.assert_bundle_invalid(record, "headline|verification")

    def test_headline_eligibility_is_derived_not_trusted(self):
        record = copy.deepcopy(self.valid_record)
        record["environment"]["clock_policy"] = "uncontrolled-clocks"
        record["hardware"]["clock_policy"] = "uncontrolled-clocks"
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_record_invalid(record, "headline|locked|uncontrolled")

        record = copy.deepcopy(self.valid_record)
        drift_sample = copy.deepcopy(record["environment"]["telemetry_samples"][0])
        drift_sample["gpu_clock_mhz"] = 2300
        record["environment"]["telemetry_samples"] = [copy.deepcopy(drift_sample) for _ in range(3)]
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_record_invalid(record, "headline|clock drift")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["telemetry_samples"][0]["power_cap_watts"] = None
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_record_invalid(record, "headline|power cap")

    def test_publishability_rejects_uncontrolled_missing_and_drifted_headline_records(self):
        record = copy.deepcopy(self.valid_record)
        record["environment"]["clock_policy"] = "uncontrolled-clocks"
        record["hardware"]["clock_policy"] = "uncontrolled-clocks"
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_bundle_invalid(record, "headline|locked|uncontrolled")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["observed_after"]["power_cap_watts"] = None
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_bundle_invalid(record, "headline|power cap")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["telemetry_samples"][0]["memory_clock_mhz"] = 1200
        record["environment"]["headline_eligible"] = True
        record["environment"]["verification_errors"] = []
        self.assert_bundle_invalid(record, "headline|memory clock drift")

    def test_configured_missing_inputs_and_secret_like_values_fail_closed(self):
        record = copy.deepcopy(self.valid_record)
        record["errors"] = [{"code": "missing-configured-input", "message": "artifact missing"}]
        self.assert_bundle_invalid(record, "configured|missing")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["allowlisted_environment"]["RUSTFLAGS"] = "Bearer secret-token"
        self.assert_record_invalid(record, "secret")

        record = copy.deepcopy(self.valid_record)
        record["environment"]["allowlisted_environment"]["OPENAI_API_KEY"] = "redacted"
        self.assert_record_invalid(record, "secret")

    def test_all_absolute_paths_are_rejected_but_ordinary_slashes_are_allowed(self):
        record = copy.deepcopy(self.valid_record)
        record["run"]["command"] = ["/usr/local/bin/supersonic"]
        self.assert_record_invalid(record, "absolute path")

        record = copy.deepcopy(self.valid_record)
        record["run"]["command"] = ["C:\\Users\\deano\\supersonic.exe"]
        self.assert_record_invalid(record, "absolute path")

        record = copy.deepcopy(self.valid_record)
        record["quality"]["cases"][0]["actual_value"] = "The prompt asks about A/B/C ratios."
        self.validation.validate_record(record)


if __name__ == "__main__":
    unittest.main()
