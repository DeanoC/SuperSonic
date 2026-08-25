import importlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
SCHEMA = BENCHMARKS / "schema" / "result-v1.schema.json"
APPROVED_CATEGORIES = {
    "instruction-following",
    "structured-extraction",
    "arithmetic-and-reasoning",
    "code-completion",
    "long-context-retrieval",
    "chat-template-behavior",
    "repeated-run-determinism",
    "ordinary-vs-mtp-token-equality",
}
CACHE_STATES = {
    "cold-load",
}
SUPPORTED_CACHE_STATES = {
    "cold-load",
    "warm-resident",
    "prefix-cache-empty",
    "prefix-cache-populated",
    "prefix-cache-reset",
}


def load_manifest_module():
    try:
        return importlib.import_module("tools.benchmark.manifest")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.manifest is absent") from exc


class BenchmarkManifestTests(unittest.TestCase):
    maxDiff = None

    def test_budgets_and_case_sets(self):
        manifest = load_manifest_module()

        quick = manifest.load_suite("quick")
        full = manifest.load_suite("full")
        quality_cases = manifest.load_quality(full.quality_version)

        self.assertEqual(quick.version, 1)
        self.assertEqual(full.version, 1)
        self.assertEqual(quick.quality_version, "v2")
        self.assertEqual(full.quality_version, "v2")
        self.assertEqual(quick.decoding_policy, "greedy")
        self.assertEqual(full.decoding_policy, "greedy")
        self.assertEqual(quick.budget_seconds, 600)
        self.assertEqual(full.budget_seconds, 21600)
        self.assertEqual(quick.minimum_duration_seconds, 0)
        self.assertEqual(full.minimum_duration_seconds, 20700)
        self.assertLess(set(quick.quality_case_ids), set(full.quality_case_ids))
        self.assertEqual(set(full.quality_case_ids), {case.id for case in quality_cases})
        self.assertEqual(tuple(quick.engines), ("supersonic",))
        self.assertEqual(tuple(full.engines), ("supersonic", "llama-cpp"))

    def test_scalar_qualification_suite_balances_three_engines(self):
        manifest = load_manifest_module()
        suite = manifest.load_suite("full-scalar-qualification")

        self.assertEqual(suite.budget_seconds, 21600)
        self.assertEqual(suite.minimum_duration_seconds, 20700)
        self.assertEqual(
            suite.engines,
            ("supersonic-wmma", "supersonic-scalar-lab", "llama-cpp"),
        )
        self.assertTrue(all(case.repetitions == 7 for case in suite.performance_cases))
        ordinary = [case for case in suite.performance_cases if case.mode == "ordinary"]
        mtp = [case for case in suite.performance_cases if case.mode == "mtp"]
        self.assertTrue(ordinary)
        self.assertTrue(mtp)
        self.assertTrue(all(case.engines == suite.engines for case in ordinary))
        self.assertTrue(
            all(
                case.engines == ("supersonic-wmma", "supersonic-scalar-lab")
                for case in mtp
            )
        )

    def test_suite_cases_are_positive_unique_and_reference_supported_modes(self):
        manifest = load_manifest_module()

        full = manifest.load_suite("full")

        case_ids = [case.id for case in full.performance_cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(any("short" in case_id for case_id in case_ids))
        self.assertTrue(any("long" in case_id for case_id in case_ids))
        self.assertTrue(any("cold" in case_id for case_id in case_ids))
        self.assertFalse(any("warm" in case_id for case_id in case_ids))
        self.assertTrue(any("ordinary" in case_id for case_id in case_ids))
        self.assertTrue(any("mtp" in case_id for case_id in case_ids))

        engines = {engine.name: engine for engine in map(manifest.load_engine, full.engines)}
        suite_modes = {case.mode for case in full.performance_cases}
        self.assertEqual(suite_modes, {"ordinary", "mtp"})
        active_cache_states = {case.cache_state for case in full.performance_cases}
        self.assertTrue(CACHE_STATES.issubset(active_cache_states))
        self.assertFalse(active_cache_states & (SUPPORTED_CACHE_STATES - CACHE_STATES))
        for case in full.performance_cases:
            self.assertGreaterEqual(case.warmups, 0)
            self.assertGreater(case.repetitions, 0)
            self.assertGreater(case.timeout_seconds, 0)
            self.assertEqual(case.decoding_policy, "greedy")
            self.assertIn(case.cache_state, SUPPORTED_CACHE_STATES)
            self.assertTrue(case.engines, msg=f"{case.id} must scope at least one engine")
            self.assertLessEqual(set(case.engines), set(full.engines))
            for engine_name in case.engines:
                self.assertIn(
                    case.mode,
                    engines[engine_name].supported_modes,
                    msg=f"{engine_name} does not support {case.mode} for {case.id}",
                )
            if case.mode == "mtp":
                self.assertEqual(case.engines, ("supersonic",))
            self.assertEqual(case.timeout_seconds, 60)

        quick = manifest.load_suite("quick")
        self.assertTrue(all(case.warmups == 0 for case in quick.performance_cases))
        self.assertTrue(all(case.repetitions == 3 for case in quick.performance_cases))
        self.assertTrue(all(case.mode == "ordinary" for case in quick.performance_cases))
        self.assertTrue(all(case.engines == ("supersonic",) for case in quick.performance_cases))

    def test_scoped_engine_case_matrix_rejects_unsupported_combinations(self):
        manifest = load_manifest_module()

        with tempfile.TemporaryDirectory() as temporary:
            bad_root = Path(temporary)
            bad_manifest = bad_root / "bad-suite.toml"
            bad_manifest.write_text(
                """
version = 1
name = "bad"
budget_seconds = 600
minimum_duration_seconds = 0
quality_version = "v1"
quality_case_ids = ["instruction-following-1"]
engines = ["supersonic", "llama-cpp"]
decoding_policy = "greedy"

[[performance_cases]]
id = "bad-mtp-case"
prompt = "hello"
max_new_tokens = 8
warmups = 1
repetitions = 3
mode = "mtp"
cache_state = "warm-resident"
timeout_seconds = 30
decoding_policy = "greedy"
engines = ["llama-cpp"]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "llama-cpp|supports|unsupported"):
                manifest.load_suite_path(bad_manifest)

    def test_minimum_duration_cannot_exceed_hard_budget(self):
        manifest = load_manifest_module()

        with tempfile.TemporaryDirectory() as temporary:
            bad_manifest = Path(temporary) / "bad-duration.toml"
            bad_manifest.write_text(
                """
version = 1
name = "bad-duration"
budget_seconds = 600
minimum_duration_seconds = 601
quality_version = "v1"
quality_case_ids = ["instruction-following-1"]
engines = ["supersonic"]
decoding_policy = "greedy"

[[performance_cases]]
id = "case"
prompt = "hello"
max_new_tokens = 8
warmups = 0
repetitions = 1
mode = "ordinary"
cache_state = "cold-load"
timeout_seconds = 30
decoding_policy = "greedy"
engines = ["supersonic"]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "minimum_duration_seconds.*budget_seconds"):
                manifest.load_suite_path(bad_manifest)

    def test_quality_corpus_has_required_categories_and_unique_ids(self):
        manifest = load_manifest_module()

        quality_cases = manifest.load_quality("v2")

        case_ids = [case.id for case in quality_cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        category_counts = Counter(case.category for case in quality_cases)
        self.assertEqual(set(category_counts), APPROVED_CATEGORIES)
        for category in APPROVED_CATEGORIES:
            self.assertGreaterEqual(category_counts[category], 2, msg=category)
        for case in quality_cases:
            self.assertEqual(case.decoding_policy, "greedy")
            self.assertGreater(case.max_new_tokens, 0)
            self.assertIn(case.scorer, {"exact_text", "exact_tokens", "structured_json"})

    def test_json_loader_rejects_duplicate_keys_and_non_finite_numbers(self):
        manifest = load_manifest_module()

        with tempfile.TemporaryDirectory() as temporary:
            duplicate_path = Path(temporary) / "duplicate.json"
            duplicate_path.write_text('{"name":"first","name":"second"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                manifest._load_json(duplicate_path)

            non_finite_path = Path(temporary) / "non-finite.json"
            non_finite_path.write_text('{"value":1e309}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                manifest._load_json(non_finite_path)

    def test_engine_manifests_resolve_pins_and_exact_keys(self):
        manifest = load_manifest_module()

        supersonic = manifest.load_engine("supersonic")
        llama_cpp = manifest.load_engine("llama-cpp")

        self.assertEqual(supersonic.version, 1)
        self.assertEqual(llama_cpp.version, 1)
        self.assertEqual(tuple(supersonic.supported_modes), ("ordinary", "mtp"))

        wmma = manifest.load_engine("supersonic-wmma")
        scalar = manifest.load_engine("supersonic-scalar-lab")
        self.assertEqual(wmma.binary, "./target/release/supersonic")
        self.assertEqual(scalar.binary, "tools/supersonic-scalar-lab.py")
        self.assertEqual(wmma.supported_modes, ("ordinary", "mtp"))
        self.assertEqual(scalar.supported_modes, ("ordinary", "mtp"))
        self.assertEqual(tuple(llama_cpp.supported_modes), ("ordinary",))
        self.assertIsNone(supersonic.version_pin_file)
        self.assertEqual(llama_cpp.version_pin_file, "tools/external/llama-cpp-version.txt")
        self.assertEqual(llama_cpp.pinned_version, "version: 5 (f8dd7c3)")
        self.assertEqual(llama_cpp.binary, "tools/llama-cpp-peer.py")
        self.assertEqual(llama_cpp.version_command, ("llama-server", "--version"))

        supersonic_raw = tomllib.loads(
            (BENCHMARKS / "engines" / "supersonic.toml").read_text(encoding="utf-8")
        )
        llama_raw = tomllib.loads(
            (BENCHMARKS / "engines" / "llama-cpp.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(supersonic_raw),
            {"version", "name", "binary", "version_command", "supported_modes"},
        )
        self.assertEqual(
            set(llama_raw),
            {
                "version",
                "name",
                "binary",
                "version_command",
                "version_pin_file",
                "supported_modes",
            },
        )

    def test_result_schema_contract_is_versioned_and_closed(self):
        text = SCHEMA.read_text(encoding="utf-8")
        schema = json.loads(text)

        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {
                "run",
                "engine",
                "hardware",
                "artifact",
                "workload",
                "environment",
                "samples",
                "quality",
                "status",
                "errors",
            },
        )
        self.assertEqual(set(schema["properties"]), set(schema["required"]))
        self.assertEqual(schema["properties"]["run"]["properties"]["schema_version"]["const"], 1)
        self.assertEqual(
            schema["properties"]["workload"]["properties"]["cache_state"]["enum"],
            [
                "cold-load",
                "warm-resident",
                "prefix-cache-empty",
                "prefix-cache-populated",
                "prefix-cache-reset",
            ],
        )
        self.assertEqual(
            set(schema["properties"]["environment"]["required"]),
            {
                "clock_policy",
                "rocm_version",
                "hip_version",
                "requested",
                "requested_at",
                "observed_before",
                "observed_before_at",
                "observed_after",
                "observed_after_at",
                "telemetry_samples",
                "headline_eligible",
                "physical_gpu",
                "logical_gpu",
                "cpu_governor",
                "allowlisted_environment",
                "cache_state",
                "cache_evidence",
                "process_reuse",
                "verification_errors",
                "evidence_notes",
            },
        )
        self.assertEqual(
            set(schema["properties"]["environment"]["properties"]["requested"]["required"]),
            {
                "gpu_clock_mhz",
                "clock_tolerance_mhz",
                "memory_clock_mhz",
                "power_cap_watts",
                "performance_level",
            },
        )
        observed_required = {
            "gpu_clock_mhz",
            "memory_clock_mhz",
            "power_cap_watts",
            "power_watts",
            "temperature_celsius",
            "gpu_utilization_percent",
            "memory_utilization_percent",
            "performance_level",
            "throttle_status",
            "indep_throttle_status",
            "throttle_label",
        }
        self.assertEqual(
            set(schema["properties"]["environment"]["properties"]["observed_before"]["required"]),
            observed_required,
        )
        self.assertEqual(
            set(schema["properties"]["environment"]["properties"]["observed_after"]["required"]),
            observed_required,
        )
        sample_required = {
            "offset_seconds",
            "gpu_clock_mhz",
            "memory_clock_mhz",
            "power_cap_watts",
            "power_watts",
            "temperature_celsius",
            "gpu_utilization_percent",
            "memory_utilization_percent",
            "performance_level",
            "throttle_status",
            "indep_throttle_status",
            "throttle_label",
        }
        self.assertEqual(
            set(
                schema["properties"]["environment"]["properties"]["telemetry_samples"]["items"][
                    "required"
                ]
            ),
            sample_required,
        )

    def test_canonical_json_is_sorted_and_compact(self):
        manifest = load_manifest_module()

        value = {"b": 1, "a": {"d": 2, "c": [3, {"b": 1, "a": 0}]}}

        self.assertEqual(
            manifest.canonical_json(value),
            '{"a":{"c":[3,{"a":0,"b":1}],"d":2},"b":1}',
        )

    def test_unknown_cache_state_and_key_fail(self):
        manifest = load_manifest_module()

        with tempfile.TemporaryDirectory() as temporary:
            bad_root = Path(temporary)
            bad_manifest = bad_root / "bad-suite.toml"
            bad_manifest.write_text(
                """
version = 1
name = "bad"
budget_seconds = 600
minimum_duration_seconds = 0
quality_version = "v1"
quality_case_ids = ["instruction-following-1"]
engines = ["supersonic"]
decoding_policy = "greedy"
unknown_key = true

[[performance_cases]]
id = "bad-case"
prompt = "hello"
max_new_tokens = 8
warmups = 1
repetitions = 3
mode = "ordinary"
cache_state = "unknown"
timeout_seconds = 30
decoding_policy = "greedy"
engines = ["supersonic"]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cache_state|unknown_key|unknown"):
                manifest.load_suite_path(bad_manifest)

    def test_manifest_files_have_exact_allowed_keys(self):
        quick = tomllib.loads((BENCHMARKS / "suites" / "quick.toml").read_text(encoding="utf-8"))
        full = tomllib.loads((BENCHMARKS / "suites" / "full.toml").read_text(encoding="utf-8"))
        quality = json.loads((BENCHMARKS / "quality" / "v1.json").read_text(encoding="utf-8"))

        expected_suite_keys = {
            "version",
            "name",
            "budget_seconds",
            "minimum_duration_seconds",
            "quality_version",
            "quality_case_ids",
            "engines",
            "decoding_policy",
            "performance_cases",
        }
        expected_case_keys = {
            "id",
            "prompt",
            "max_new_tokens",
            "warmups",
            "repetitions",
            "mode",
            "cache_state",
            "timeout_seconds",
            "decoding_policy",
            "engines",
        }

        for raw_suite in (quick, full):
            self.assertEqual(set(raw_suite), expected_suite_keys)
            for case in raw_suite["performance_cases"]:
                self.assertEqual(set(case), expected_case_keys)

        self.assertEqual(set(quality), {"version", "categories", "cases"})
        for case in quality["cases"]:
            self.assertEqual(
                set(case),
                {
                    "id",
                    "category",
                    "prompt",
                    "max_new_tokens",
                    "scorer",
                    "expected",
                    "decoding_policy",
                },
            )


if __name__ == "__main__":
    unittest.main()
