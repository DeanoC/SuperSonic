import importlib
import hashlib
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
    "dflash-quality",
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
        self.assertIn("dflash-quality-1", quick.quality_case_ids)
        self.assertIn("dflash-quality-2", full.quality_case_ids)
        self.assertEqual(set(full.quality_case_ids), {case.id for case in quality_cases})
        self.assertEqual(tuple(quick.engines), ("supersonic",))
        self.assertEqual(tuple(full.engines), ("supersonic", "llama-cpp"))

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
        quick = manifest.load_suite("quick")
        self.assertTrue(any(case.mode == "dflash" for case in quick.performance_cases))

        engines = {engine.name: engine for engine in map(manifest.load_engine, full.engines)}
        suite_modes = {case.mode for case in full.performance_cases}
        self.assertEqual(suite_modes, {"ordinary", "mtp", "dflash"})
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

        self.assertTrue(all(case.warmups == 0 for case in quick.performance_cases))
        self.assertTrue(all(
            case.repetitions == (1 if case.id.startswith("quick-geo-humaneval-") else 3)
            for case in quick.performance_cases
        ))
        self.assertTrue(all(case.mode in {"ordinary", "dflash"} for case in quick.performance_cases))
        self.assertTrue(all(case.engines == ("supersonic",) for case in quick.performance_cases))
        self.assertEqual({case.stop_policy for case in full.performance_cases}, {"ignore-eos"})

    def test_quick_performance_owns_only_geo_comparable_workload(self):
        manifest = load_manifest_module()
        quick = manifest.load_suite("quick")
        case_ids = [case.id for case in quick.performance_cases]

        self.assertEqual(len(case_ids), 20)
        self.assertTrue(all(case_id.startswith("quick-geo-humaneval-") for case_id in case_ids))

    def test_quick_includes_geo_comparable_humaneval_workload(self):
        manifest = load_manifest_module()
        quick = manifest.load_suite("quick")
        cases = {case.id: case for case in quick.performance_cases}

        prompt_names = [
            "has-close-elements",
            "separate-paren-groups",
            "truncate-number",
            "below-zero",
            "mean-absolute-deviation",
            "intersperse",
            "parse-nested-parens",
            "filter-by-substring",
            "sum-product",
            "rolling-max",
        ]
        prompt_hashes = {
            "has-close-elements": "eb8ed1bcb5e9bee3d4e0359c703d1dfb6dba7c665432908cf89af91095a74997",
            "separate-paren-groups": "22d37f12dea840dc7fcf35de40d35ea50ba0cd15c9df5d79763518ec1cefe8e7",
            "truncate-number": "4d6228ef4422d00142bf9d4f7e5aad2877b7a7b6f8b2e22aac2ffbeb765e3ffe",
            "below-zero": "0dd6eb42233302e5c785f182b4d88c903b3ea8766dc93c2463081e67e466f919",
            "mean-absolute-deviation": "9545160985c29e1e37d88a1316c1aaaced440fca1074d0c3014d0b4d54198131",
            "intersperse": "9cd0d5c9209ebdae02589e78c0c424188b36e557c48bb120c877afa8bcc3115a",
            "parse-nested-parens": "de6daaf9efa2d062e8b3c330d60b923b1a23a7f5233deef0d111c4a24e1399c7",
            "filter-by-substring": "2fd38d40d559643fa64a73ad1b564a5cad18bef34fde8c47902237aca409ca0e",
            "sum-product": "221fe1e9b5b64f7989fffd79a31bcea1baa9b2b1a2388f9f95565fa808c437ea",
            "rolling-max": "641a22ee5177fd0ad5a53c761b0360b6a537d78dc2f8cd2487627d2542976fa6",
        }

        for prompt_name in prompt_names:
            for mode in ("ordinary", "dflash"):
                case_id = f"quick-geo-humaneval-{prompt_name}-{mode}"
                self.assertIn(case_id, cases)
                case = cases[case_id]
                self.assertEqual(case.max_new_tokens, 256)
                self.assertEqual(case.warmups, 0)
                self.assertEqual(case.repetitions, 1)
                self.assertEqual(case.mode, mode)
                self.assertEqual(case.cache_state, "cold-load")
                self.assertEqual(case.decoding_policy, "greedy")
                self.assertEqual(case.engines, ("supersonic",))
                self.assertEqual(case.timeout_seconds, 60)
                self.assertEqual(case.stop_policy, "honor-eos")
                self.assertEqual(hashlib.sha256(case.prompt.encode()).hexdigest(), prompt_hashes[prompt_name])

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
quality_version = "v2"
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
stop_policy = "ignore-eos"
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
quality_version = "v2"
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
        self.assertEqual(
            category_counts["dflash-quality"],
            2,
        )
        for case in quality_cases:
            self.assertEqual(case.decoding_policy, "greedy")
            self.assertGreater(case.max_new_tokens, 0)
            self.assertIn(
                case.scorer,
                {"exact_text", "semantic_text", "exact_tokens", "structured_json"},
            )

    def test_quality_loader_accepts_current_version_only(self):
        manifest = load_manifest_module()

        self.assertEqual(manifest.CURRENT_QUALITY_VERSION, "v2")
        with self.assertRaisesRegex(ValueError, "quality version must be 'v2'"):
            manifest.load_quality("v1")

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
        self.assertEqual(tuple(supersonic.supported_modes), ("ordinary", "mtp", "dflash"))
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
        self.assertEqual(set(schema["properties"]), set(schema["required"]) | {"draft_artifact"})
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
            schema["properties"]["workload"]["properties"]["mode"]["enum"],
            ["ordinary", "mtp", "dflash"],
        )
        self.assertIn(
            "dflash-quality",
            schema["properties"]["quality"]["properties"]["categories"]["properties"],
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
quality_version = "v2"
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
        quality = json.loads((BENCHMARKS / "quality" / "v2.json").read_text(encoding="utf-8"))

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
            "stop_policy",
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
