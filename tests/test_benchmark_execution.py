from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest


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
        self.binary = self.root / "supersonic"
        self.binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.binary.chmod(0o755)
        self.peer_binary = self.root / "llama-cli"
        self.peer_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.peer_binary.chmod(0o755)

    def config(self, *, suite="quick", include_peer=False, run_quality=False, output=None):
        return self.execution.RunConfig(
            suite=suite,
            model_dir=self.model_dir,
            artifact=self.artifact,
            peer_artifact=self.peer_artifact if include_peer else None,
            physical_gpu="0",
            gpu_arch="gfx1201",
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
        for field in ("model_dir", "artifact", "physical_gpu", "gpu_arch"):
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
        status = self.execution.run_suite(config, FakeClock([0, 599, 601]), runner)
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


if __name__ == "__main__":
    unittest.main()
