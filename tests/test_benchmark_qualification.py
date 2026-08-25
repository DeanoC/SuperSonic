from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.benchmark import qualification


class QualificationTests(unittest.TestCase):
    def test_series_is_derived_from_validated_scalar_record(self):
        fixture = Path(__file__).parent / "benchmark_fixtures" / "valid-result-v1.json"
        record = json.loads(fixture.read_text(encoding="utf-8"))
        case_id = "scalar-qualification-short-cold-ordinary"
        prompt = "Emit a single sentence describing cold-load benchmark startup."
        record["run"]["suite"] = "full-scalar-qualification"
        record["run"]["case_id"] = case_id
        record["run"]["run_id"] = "scalar-baseline-1"
        record["engine"]["name"] = "supersonic-scalar-lab"
        record["engine"]["version"] = "scalar-head-lab-v1"
        record["workload"]["measurement_boundary"] = "decode"
        record["workload"]["case_id"] = case_id
        record["workload"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        record["workload"]["max_new_tokens"] = 32
        record["workload"]["warmups"] = 0
        record["environment"]["cache_evidence"]["filesystem_flush"] = "unavailable"
        telemetry = {
            "offset_seconds": 0.1,
            "gpu_clock_mhz": 2400,
            "memory_clock_mhz": 1249,
            "power_cap_watts": 295,
            "power_watts": 200.0,
            "temperature_celsius": 60.0,
            "gpu_utilization_percent": 100.0,
            "memory_utilization_percent": 50.0,
            "performance_level": "manual",
            "throttle_status": 1,
            "indep_throttle_status": 1,
            "throttle_label": "THROTTLED",
            "raw_amd_smi_json": "{\"gpu\":0}",
        }
        record["environment"]["telemetry_samples"] = [dict(telemetry) for _ in range(7)]
        record["samples"] = [
            {
                "decode_ms": 1000.0,
                "tokens_per_second": 32.0,
                "lm_head_ms": 310.0,
                "timed_decode_steps": 31,
                "telemetry_start_index": index,
                "telemetry_sample_count": 1,
                "loaded_clock_minimum_mhz": 2400,
                "loaded_clock_median_mhz": 2400,
                "loaded_clock_maximum_mhz": 2400,
            }
            for index in range(7)
        ]
        with mock.patch.object(qualification.validation, "validate_record") as validate:
            series = qualification.series_from_record(
                record,
                compiler_version="hipcc 7.1.0",
                scalar_instruction_sha256="2" * 64,
            )
        validate.assert_called_once_with(record)
        self.assertEqual(series["binding"]["measurement_boundary"], "lm_head_ms/timed_decode_steps")
        self.assertEqual(series["median_ms_per_token"], 10.0)
        qualification.validate_series(series)

        record["samples"][1]["telemetry_start_index"] = 0
        with self.assertRaisesRegex(ValueError, "overlap"):
            qualification.series_from_record(
                record,
                compiler_version="hipcc 7.1.0",
                scalar_instruction_sha256="2" * 64,
            )

    def test_directory_digest_binds_sorted_paths_names_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z.txt").write_bytes(b"z")
            (root / "a.txt").write_bytes(b"a")
            expected = hashlib.sha256()
            for name, payload in (("a.txt", b"a"), ("z.txt", b"z")):
                expected.update(name.encode("utf-8"))
                expected.update(b"\0")
                expected.update(hashlib.sha256(payload).digest())
                expected.update(b"\0")
            self.assertEqual(qualification.directory_digest(root), expected.hexdigest())

    def test_qualification_accepts_seven_stable_samples_within_five_percent(self):
        baseline = self._series("baseline-1", [10.0] * 7)
        candidate = self._series("candidate-1", [10.4] * 7)
        result = qualification.qualify_series(
            baseline,
            candidate,
            baseline_bundle_sha256="a" * 64,
        )
        self.assertTrue(result["qualified"])
        self.assertAlmostEqual(result["percent_regression"], 4.0)
        qualification.validate_qualification(result)
        result["qualified"] = False
        with self.assertRaisesRegex(ValueError, "decision"):
            qualification.validate_qualification(result)

    def test_qualification_rejects_regression_dispersion_and_binding_mismatch(self):
        baseline = self._series("baseline-1", [10.0] * 7)
        slow = self._series("candidate-1", [10.6] * 7)
        rejected = qualification.qualify_series(baseline, slow, baseline_bundle_sha256="a" * 64)
        self.assertFalse(rejected["qualified"])
        qualification.validate_qualification(rejected)
        noisy = self._series("candidate-1", [8.0, 8.0, 8.0, 10.0, 12.0, 12.0, 12.0])
        with self.assertRaisesRegex(ValueError, "MAD"):
            qualification.qualify_series(baseline, noisy, baseline_bundle_sha256="a" * 64)
        mismatched = self._series("candidate-1", [10.0] * 7)
        mismatched["binding"]["artifact_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "artifact_sha256"):
            qualification.qualify_series(baseline, mismatched, baseline_bundle_sha256="a" * 64)

    def test_baseline_is_closed_and_uses_lm_head_boundary(self):
        baseline = self._series("baseline-1", [10.0] * 7)
        qualification.validate_series(baseline)
        baseline["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "additional"):
            qualification.validate_series(baseline)

    @staticmethod
    def _series(run_id: str, values: list[float]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "binding": {
                "commit": "1" * 40,
                "rocm_version": "ROCm 7.1.0",
                "hip_version": "HIP 7.1.0",
                "compiler_version": "hipcc 7.1.0",
                "scalar_instruction_sha256": "2" * 64,
                "artifact_semantic_id": "qwen3.8-27b-gqh-q3kxl-hf-91bc7e33",
                "artifact_quantization": "GQH-Q3KXL",
                "artifact_sha256": "3" * 64,
                "artifact_source_repository": "Geometric-AI/Qwen3.8-27B-GQH-Q3KXL-GGUF",
                "artifact_source_revision": "5" * 40,
                "artifact_filename": "Qwen3.8-27B-GQH-Q3KXL.gguf",
                "artifact_size_bytes": 13440110432,
                "tokenizer_sha256": "6" * 64,
                "chat_template_sha256": "7" * 64,
                "prompt_sha256": "4" * 64,
                "measurement_boundary": "lm_head_ms/timed_decode_steps",
                "gpu_identity": "0000:03:00.0",
                "pci_bdf": "0000:03:00.0",
                "gpu_clock_mhz": 2400,
                "gpu_clock_tolerance_mhz": 20,
                "memory_clock_mhz": 1249,
                "power_cap_watts": 295,
                "performance_level": "manual",
                "temperature_limit_celsius": 85.0,
                "cache_state": "cold-load",
                "process_reuse": False,
                "filesystem_flush": "unavailable",
            },
            "samples": [
                {"sample_id": f"{run_id}-{index}", "lm_head_ms": value * 31, "timed_decode_steps": 31}
                for index, value in enumerate(values, 1)
            ],
            "median_ms_per_token": sorted(values)[3],
        }


if __name__ == "__main__":
    unittest.main()
