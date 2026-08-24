from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "benchmark_fixtures"


def load_execution_module():
    try:
        return importlib.import_module("tools.benchmark.execution")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.execution is absent") from exc


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.last = 0.0

    def __call__(self) -> float:
        try:
            self.last = next(self.values)
        except StopIteration:
            self.last += 1.0
        return self.last


class FakeRunner:
    def __init__(self, output: str | None = None) -> None:
        self.started_case_ids: list[str] = []
        self.calls: list[tuple[tuple[str, ...], int | None]] = []
        self.output = output or (
            '[generated_json] "ok"\n'
            "[tokens] 1\n"
            "[result] prompt_tokens=2 generated_tokens=1 decode_ms=1.0 ms_per_tok=1.0\n"
        )

    def __call__(self, argv, timeout=None, case_id=None, **_kwargs):
        self.calls.append((tuple(argv), timeout))
        if case_id is not None:
            self.started_case_ids.append(case_id)
        elif "--prompt" in argv:
            self.started_case_ids.append(str(argv[argv.index("--prompt") + 1]))
        return {
            "returncode": 0,
            "stdout": self.output,
            "stderr": "raw stderr",
        }


class CorruptRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__(output="not an adapter record")


class BenchmarkExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.execution = load_execution_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.model_dir / name).write_text("{}", encoding="utf-8")
        self.artifact = self.root / "qwen38.gqh.gguf"
        self.artifact.write_bytes(b"artifact")
        self.peer_artifact = self.root / "peer.gguf"
        self.peer_artifact.write_bytes(b"peer")
        self.static_json = self.root / "amd-smi-static.json"
        self.static_json.write_text(
            json.dumps(
                {
                    "gpu_data": [
                        {
                            "gpu": 0,
                            "logical_gpu": 0,
                            "asic": {
                                "market_name": "AMD Radeon AI PRO R9700",
                                "target_graphics_version": "gfx1201",
                                "pci_bdf": "0000:65:00.0",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.rocm_version_file = self.root / "rocm-version.txt"
        self.rocm_version_file.write_text("ROCm 6.4.2\n", encoding="utf-8")
        self.hip_version_file = self.root / "hip-version.txt"
        self.hip_version_file.write_text("HIP 6.4.2\n", encoding="utf-8")
        self.binary = self.root / "supersonic"
        self.binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.binary.chmod(0o755)
        self.peer_binary = self.root / "llama-cli"
        self.peer_binary.write_text(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'version: 5 (f8dd7c3)'; fi\nexit 0\n",
            encoding="utf-8",
        )
        self.peer_binary.chmod(0o755)

    def config(self, *, suite="quick", include_peer=False, run_quality=False, output=None):
        return self.execution.RunConfig(
            suite=suite,
            model_dir=self.model_dir,
            artifact=self.artifact,
            peer_artifact=self.peer_artifact if include_peer else None,
            physical_gpu="0",
            gpu_arch="gfx1201",
            gpu_static_json=self.static_json,
            rocm_version_file=self.rocm_version_file,
            hip_version_file=self.hip_version_file,
            logical_gpu="0",
            output_dir=Path(output or self.root / "candidate"),
            engine_binaries={
                "supersonic": self.binary,
                "llama-cpp": self.peer_binary,
            },
            run_quality=run_quality,
            clock_policy="uncontrolled-clocks",
        )

    def test_configured_peer_missing_fails_preflight(self):
        config = self.config(suite="full", include_peer=True)
        config = self.execution.replace_config(config, peer_artifact=self.root / "missing-peer.gguf")
        with self.assertRaisesRegex(ValueError, "llama-cpp.*unavailable|peer.*unavailable"):
            self.execution.preflight(config)

    def test_preflight_requires_explicit_model_artifact_and_gpu(self):
        for field in (
            "model_dir",
            "artifact",
            "physical_gpu",
            "gpu_arch",
            "gpu_static_json",
            "rocm_version_file",
            "hip_version_file",
        ):
            config = self.config()
            config = self.execution.replace_config(config, **{field: None})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field.replace("_", ".*")):
                self.execution.preflight(config)

    def test_case_engine_scope_is_honored(self):
        manifest = self.execution.preflight(self.config(suite="full", include_peer=True))
        scheduled = {(case.id, engine.name) for case, engine in self.execution.ordered_cases(manifest)}
        self.assertIn(("full-short-cold-ordinary", "llama-cpp"), scheduled)
        self.assertNotIn(("full-short-cold-mtp", "llama-cpp"), scheduled)

    def test_budget_stops_scheduling_and_marks_incomplete(self):
        config = self.config(run_quality=False)
        runner = FakeRunner()
        status = self.execution.run_suite(config, FakeClock([0, 599, 599.5, 601]), runner)
        self.assertEqual(status.state, "incomplete")
        self.assertIn("budget_exhausted", status.errors)
        self.assertEqual(set(runner.started_case_ids), {"quick-short-warm-ordinary"})

    def test_case_timeout_is_bounded_and_preserves_incomplete_evidence(self):
        config = self.config(run_quality=False)

        class TimeoutRunner(FakeRunner):
            def __call__(self, argv, timeout=None, **kwargs):
                self.calls.append((tuple(argv), timeout))
                raise TimeoutError("timed out")

        runner = TimeoutRunner()
        status = self.execution.run_suite(config, FakeClock([0, 1, 2, 3]), runner)
        self.assertIn(status.state, {"failed", "incomplete"})
        self.assertIn("case_timeout", status.errors)
        self.assertTrue(all(timeout is not None and timeout > 0 for _, timeout in runner.calls))

    def test_sigint_marks_suite_incomplete(self):
        config = self.config(run_quality=False)

        class InterruptRunner(FakeRunner):
            def __call__(self, argv, timeout=None, **kwargs):
                raise KeyboardInterrupt

        status = self.execution.run_suite(config, FakeClock([0, 1, 2]), InterruptRunner())
        self.assertEqual(status.state, "incomplete")
        self.assertIn("interrupted", status.errors)

    def test_invalid_record_is_never_atomically_promoted(self):
        config = self.config(run_quality=False)
        status = self.execution.run_suite(config, FakeClock([0, 1, 2, 3, 4, 5]), CorruptRunner())
        self.assertFalse((status.bundle / "records" / "quick-short-warm-ordinary-supersonic.json").exists())
        self.assertFalse(any(path.name.endswith(".tmp") for path in (status.bundle / "records").glob("*")))

    def test_raw_streams_are_captured_locally_and_measured_processes_are_fresh(self):
        config = self.config(run_quality=False)
        runner = FakeRunner()
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 30))), runner)
        logs = list((status.bundle / "logs").glob("*.stdout.log"))
        self.assertTrue(logs)
        self.assertTrue(all("raw stderr" in path.with_name(path.name.replace("stdout", "stderr")).read_text() for path in logs))
        self.assertTrue(all(call[1] is not None for call in runner.calls))

    def test_warm_resident_evidence_does_not_claim_process_reuse(self):
        config = self.config(run_quality=False)
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 30))), FakeRunner())
        records = list((status.bundle / "records").glob("*.json"))
        self.assertTrue(records)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertFalse(record["environment"]["process_reuse"])
        self.assertEqual(record["environment"]["cache_evidence"]["process_state"], "fresh-process")
        self.assertEqual(record["environment"]["rocm_version"], "ROCm 6.4.2")
        self.assertEqual(record["environment"]["hip_version"], "HIP 6.4.2")
        self.assertTrue(record["engine"]["version"].startswith("source-"))
        self.assertEqual(record["engine"]["adapter_version"], 2)
        self.assertNotIn(str(self.rocm_version_file), json.dumps(record))
        self.assertNotIn(str(self.hip_version_file), json.dumps(record))

    def test_version_file_parser_rejects_empty_unknown_and_oversized_captures(self):
        parser = self.execution._read_version_file
        empty = self.root / "empty-version.txt"
        empty.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            parser(empty, "HIP")
        unknown = self.root / "unknown-version.txt"
        unknown.write_text("HIP unknown\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parseable|unknown"):
            parser(unknown, "HIP")
        oversized = self.root / "oversized-version.txt"
        oversized.write_text("HIP 6.4.2\n" + ("x" * 4096), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "4096"):
            parser(oversized, "HIP")

    def test_hip_version_parser_accepts_unknown_in_standard_target_triple(self):
        capture = self.root / "hipcc-version.txt"
        capture.write_text(
            "HIP version: 7.14.60850-0000000\n"
            "AMD clang version 23.0.0git\n"
            "Target: x86_64-unknown-linux-gnu\n",
            encoding="utf-8",
        )

        self.assertEqual(
            self.execution._read_version_file(capture, "HIP"),
            "HIP 7.14.60850-0000000",
        )

    def test_pinned_peer_version_accepts_real_two_line_stderr_shape(self):
        self.peer_binary.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'version: 5 (f8dd7c3)' >&2\n"
            "  echo 'built with GNU 14.2.0 for Linux x86_64' >&2\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )

        run_manifest = self.execution.preflight(self.config(suite="full", include_peer=True))

        peer = next(engine for engine in run_manifest.engines if engine.name == "llama-cpp")
        self.assertEqual(peer.pinned_version, "version: 5 (f8dd7c3)")

    def test_performance_failures_are_report_only_but_quality_failures_block(self):
        config = self.config(run_quality=False)
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 30))), FakeRunner())
        self.assertNotIn("performance_failed", status.errors)

        config = self.execution.replace_config(config, run_quality=True)
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 1000))), CorruptRunner())
        self.assertIn("quality_failed", status.errors)

    def test_seeded_interleaving_is_reproducible(self):
        first = self.execution.ordered_cases(self.execution.preflight(self.config(suite="full", include_peer=True)), seed=7)
        second = self.execution.ordered_cases(self.execution.preflight(self.config(suite="full", include_peer=True)), seed=7)
        third = self.execution.ordered_cases(self.execution.preflight(self.config(suite="full", include_peer=True)), seed=11)
        self.assertEqual([(c.id, e.name) for c, e in first], [(c.id, e.name) for c, e in second])
        self.assertNotEqual([(c.id, e.name) for c, e in first], [(c.id, e.name) for c, e in third])

    def test_mtp_quality_integration_rejects_non_manifest_token_cases(self):
        quality_cases = self.execution.manifest.load_quality("v1")
        non_mtp = next(case for case in quality_cases if case.scorer != "exact_tokens")
        with self.assertRaisesRegex(ValueError, "MTP|exact_tokens|manifest"):
            self.execution._validate_mtp_quality_case(non_mtp)

    def test_quality_timeout_allows_bounded_fresh_process_model_load(self):
        quality_case = self.execution.manifest.load_quality("v1")[0]

        timeout = self.execution._quality_timeout(quality_case, 600.0, FakeClock([0.0]))

        self.assertEqual(timeout, 180.0)

    def test_peer_artifact_identity_is_not_copied_from_supersonic(self):
        config = self.config(suite="full", include_peer=True, run_quality=False)
        llama_log = (FIXTURES / "llama-cpp-peer-run.log").read_text(encoding="utf-8")

        class SplitRunner(FakeRunner):
            def __call__(self, argv, timeout=None, engine_name=None, **kwargs):
                if engine_name == "llama-cpp":
                    self.calls.append((tuple(argv), timeout))
                    self.started_case_ids.append(str(argv[argv.index("--prompt") + 1]))
                    return {"returncode": 0, "stdout": llama_log, "stderr": ""}
                return super().__call__(argv, timeout=timeout, engine_name=engine_name, **kwargs)

        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 500))), SplitRunner())
        self.assertTrue(status.records)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in status.records]
        supersonic = next(record for record in records if record["engine"]["name"] == "supersonic")
        peer = next(record for record in records if record["engine"]["name"] == "llama-cpp")
        self.assertEqual(peer["engine"]["version"], "version: 5 (f8dd7c3)")
        self.assertNotEqual(peer["engine"]["version"].lower(), "unknown")
        self.assertEqual(peer["artifact"]["sha256"], self.execution._digest_file(self.peer_artifact))
        self.assertNotEqual(supersonic["artifact"]["sha256"], peer["artifact"]["sha256"])
        self.assertNotEqual(supersonic["artifact"]["semantic_id"], peer["artifact"]["semantic_id"])
        from tools.benchmark.compare import compare_records

        comparison = compare_records(supersonic, peer)
        self.assertFalse(comparison.comparable)
        self.assertIsNone(comparison.speedup)
        self.assertIn("sha256", comparison.reasons)

    def test_byte_identical_peer_artifact_shares_only_weight_identity(self):
        config = self.config(suite="full", include_peer=True, run_quality=False)
        config = self.execution.replace_config(
            config,
            peer_artifact=self.artifact,
            artifact_semantic_id="qwen3.8-27b-gqh-q3kxl",
            artifact_quantization="GQH-Q3KXL",
        )
        run_manifest = self.execution.preflight(config)
        supersonic = next(engine for engine in run_manifest.engines if engine.name == "supersonic")
        peer = next(engine for engine in run_manifest.engines if engine.name == "llama-cpp")

        primary_identity = self.execution._artifact_identity(config, supersonic)
        peer_identity = self.execution._artifact_identity(config, peer)

        self.assertEqual(peer_identity["sha256"], primary_identity["sha256"])
        self.assertEqual(peer_identity["semantic_id"], "qwen3.8-27b-gqh-q3kxl")
        self.assertEqual(peer_identity["quantization"], "GQH-Q3KXL")
        self.assertIsNone(peer_identity["tokenizer_sha256"])
        self.assertIsNone(peer_identity["chat_template_sha256"])

    def test_fractional_deadline_timeout_is_never_rounded_up(self):
        config = self.config(run_quality=False)

        class FractionalClock:
            def __init__(self):
                self.value = 0.0
                self.calls = 0

            def __call__(self):
                self.calls += 1
                if self.calls == 2:
                    self.value = 599.9
                return self.value

        clock = FractionalClock()
        class DeadlineRunner(FakeRunner):
            def __call__(self, argv, timeout=None, **kwargs):
                result = super().__call__(argv, timeout=timeout, **kwargs)
                clock.value = 600.0
                return result

        runner = DeadlineRunner()
        status = self.execution.run_suite(config, clock, runner)
        self.assertTrue(runner.calls)
        self.assertLessEqual(runner.calls[0][1], 0.1 + 1e-9)
        self.assertNotEqual(status.state, "complete")

    def test_deadline_rechecked_between_invocations(self):
        config = self.config(run_quality=False)

        class AdvancingClock:
            def __init__(self):
                self.value = 0.0
                self.phase = 0

            def __call__(self):
                if self.phase == 1:
                    self.value = 599.95
                self.phase += 1
                return self.value

        clock = AdvancingClock()

        class AdvancingRunner(FakeRunner):
            def __call__(self, argv, timeout=None, **kwargs):
                result = super().__call__(argv, timeout=timeout, **kwargs)
                clock.value += 0.1
                return result

        runner = AdvancingRunner()
        status = self.execution.run_suite(config, clock, runner)
        self.assertIn("budget_exhausted", status.errors)
        self.assertLess(len(runner.calls), 4)

    def test_missing_required_model_files_fail_preflight(self):
        for name in ("config.json", "tokenizer.json"):
            (self.model_dir / name).unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.execution.preflight(self.config())
            (self.model_dir / name).write_text("{}", encoding="utf-8")
        (self.model_dir / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
            self.execution.preflight(self.execution.replace_config(self.config(), chat=True))

    def test_prefix_cases_are_rejected_at_execution_boundary(self):
        suite = self.execution.manifest.load_suite("quick")
        from dataclasses import replace
        from tools.benchmark.model import PerformanceCase

        prefix_case = PerformanceCase(
            id="quick-prefix-rejected",
            prompt="prefix",
            max_new_tokens=1,
            warmups=0,
            repetitions=1,
            mode="ordinary",
            cache_state="prefix-cache-empty",
            timeout_seconds=1,
            decoding_policy="greedy",
            engines=("supersonic",),
        )
        custom_suite = replace(suite, performance_cases=(prefix_case,))
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.execution.preflight(self.execution.replace_config(self.config(suite=custom_suite)))

    def test_manifest_is_atomic_and_final_status_is_persisted(self):
        config = self.config(run_quality=False)
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 100))), FakeRunner())
        manifest_path = status.bundle / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"]["state"], status.state)
        self.assertNotIn(str(self.root), json.dumps(payload))
        self.assertIsNone(payload["seed"])
        self.assertEqual(payload["budgets"]["suite_seconds"], 600)

    def test_interrupted_run_updates_manifest_to_incomplete(self):
        config = self.config(run_quality=False)

        class InterruptRunner(FakeRunner):
            def __call__(self, argv, timeout=None, **kwargs):
                raise KeyboardInterrupt

        status = self.execution.run_suite(config, FakeClock([0, 1, 2]), InterruptRunner())
        payload = json.loads((status.bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"]["state"], "incomplete")

    def test_static_gpu_provenance_derives_identity_and_source_digest(self):
        status = self.execution.run_suite(
            self.config(run_quality=False),
            FakeClock([0] + list(range(1, 100))),
            FakeRunner(),
        )
        record = json.loads(status.records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["hardware"]["identity"], "0000:65:00.0")
        self.assertEqual(record["hardware"]["identity_kind"], "pci_bdf")
        self.assertEqual(
            record["hardware"]["identity_source_sha256"],
            self.execution._digest_file(self.static_json),
        )
        self.assertEqual(record["hardware"]["identity_fields"]["gpu"], "0")
        self.assertEqual(record["hardware"]["identity_fields"]["gfx_arch"], "gfx1201")

    def test_static_gpu_mismatched_architecture_fails_preflight(self):
        payload = json.loads(self.static_json.read_text(encoding="utf-8"))
        payload["gpu_data"][0]["asic"]["target_graphics_version"] = "gfx1100"
        self.static_json.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "architecture|gfx"):
            self.execution.preflight(self.config())

    def test_final_manifest_write_failure_is_not_swallowed(self):
        config = self.config(run_quality=False)
        original = self.execution._atomic_json_write
        calls = 0

        def fail_final(payload, target):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise OSError("injected final manifest fsync failure")
            return original(payload, target)

        with mock.patch.object(self.execution, "_atomic_json_write", side_effect=fail_final):
            with self.assertRaisesRegex(OSError, "final manifest fsync"):
                self.execution.run_suite(config, FakeClock([0] + list(range(1, 100))), FakeRunner())

    def test_final_manifest_read_failure_is_not_swallowed(self):
        config = self.config(run_quality=False)
        original = Path.read_text

        def fail_manifest(path, *args, **kwargs):
            if path.name == "manifest.json":
                raise OSError("injected final manifest read failure")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", fail_manifest):
            with self.assertRaisesRegex(OSError, "final manifest read"):
                self.execution.run_suite(config, FakeClock([0] + list(range(1, 100))), FakeRunner())

    def test_environment_snapshot_is_serialized_without_fabricated_telemetry(self):
        snapshot = json.loads((FIXTURES / "valid-result-v1.json").read_text(encoding="utf-8"))["environment"]
        config = self.execution.replace_config(
            self.config(run_quality=False),
            clock_policy={
                "name": "locked",
                "gpu_clock_mhz": 2400,
                "memory_clock_mhz": 1249,
                "power_cap_watts": 295,
                "performance_level": "manual",
            },
            environment_snapshot=snapshot,
        )
        status = self.execution.run_suite(config, FakeClock([0] + list(range(1, 100))), FakeRunner())
        record = json.loads(status.records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["environment"]["observed_before"]["gpu_clock_mhz"], 2400)
        self.assertEqual(record["environment"]["logical_gpu"], "0")
        self.assertFalse(record["environment"]["process_reuse"])


if __name__ == "__main__":
    unittest.main()
