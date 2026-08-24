import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest


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

    def test_path_and_non_finite_sample_fail(self):
        record = copy.deepcopy(self.valid_record)
        record["run"]["command"] = ["/home/private/supersonic"]
        record["samples"][0]["decode_ms"] = float("nan")

        with self.assertRaises(ValueError):
            self.validation.validate_record(record)

    def test_incomplete_bundle_is_not_publishable(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
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


if __name__ == "__main__":
    unittest.main()
