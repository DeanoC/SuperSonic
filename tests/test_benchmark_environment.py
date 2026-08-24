from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "benchmark_fixtures"


def load_environment():
    path = ROOT / "tools" / "benchmark" / "environment.py"
    if not path.is_file():
        raise AssertionError("tools.benchmark.environment is absent")
    spec = importlib.util.spec_from_file_location("benchmark_environment", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_environment"] = module
    spec.loader.exec_module(module)
    return module


class _ProbeRunner:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> str:
        self.calls.append(argv)
        if not self.outputs:
            raise AssertionError("probe runner exhausted")
        return self.outputs.pop(0)


class EnvironmentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = load_environment()

    def setUp(self) -> None:
        self.policy = SimpleNamespace(
            name="locked",
            gpu_clock_mhz=2400,
            memory_clock_mhz=1249,
            power_cap_watts=295,
            performance_level="manual",
            clock_tolerance_mhz=20,
            memory_clock_tolerance_mhz=10,
            temperature_limit_celsius=90.0,
        )
        self.before = self.environment.ObservedTelemetry(
            gpu_clock_mhz=2400,
            memory_clock_mhz=1249,
            power_watts=245.0,
            temperature_celsius=67.0,
            gpu_utilization_percent=91.0,
            memory_utilization_percent=88.0,
            performance_level="manual",
        )
        self.after = replace(self.before, gpu_utilization_percent=0.0, memory_utilization_percent=3.0)
        self.drifted = replace(self.before, gpu_clock_mhz=2350)

    def snapshot(
        self,
        policy_name: str = "locked",
        *,
        cache_state: str = "cold-load",
        cache_evidence: dict[str, object] | None = None,
        environment_map: dict[str, str] | None = None,
        cpu_governor_reader=None,
    ):
        fixture = (FIXTURES / "rocm-smi-showallinfo.txt").read_text(encoding="utf-8")
        runner = _ProbeRunner([fixture, fixture, fixture])
        snapshot = self.environment.collect_snapshot(
            physical_gpu="1",
            clock_policy=replace(self.policy, name=policy_name)
            if hasattr(self.policy, "__dataclass_fields__")
            else SimpleNamespace(**{**self.policy.__dict__, "name": policy_name}),
            cache_state=cache_state,
            command_runner=runner,
            cache_evidence=cache_evidence,
            environment_map=environment_map
            or {
                "HIP_ARCH": "gfx1201",
                "HIP_VISIBLE_DEVICES": "1",
                "ROCM_PATH": "/opt/rocm",
                "HIP_PATH": "/opt/rocm/hip",
                "SUPERSONIC_DEVICE": "0",
                "RUSTFLAGS": "-C target-cpu=native",
                "IGNORED_SECRET": "drop-me",
            },
            cpu_governor_reader=cpu_governor_reader or (lambda: "performance\n"),
            sample_count=1,
            wall_clock=lambda: datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc),
            monotonic_clock=iter((0.0, 2.5, 5.0)).__next__,
        )
        return snapshot, runner.calls

    def test_uncontrolled_clocks_are_not_headline_eligible(self):
        snapshot, _ = self.snapshot("uncontrolled-clocks")
        self.assertFalse(snapshot.headline_eligible)

    def test_locked_clock_drift_fails(self):
        errors = self.environment.verify_clock_policy(
            self.before,
            [self.drifted],
            self.after,
            self.policy,
        )
        self.assertIn("clock drift", " ".join(errors).lower())

    def test_locked_performance_level_mismatch_fails(self):
        observed = replace(self.before, performance_level="auto")
        errors = self.environment.verify_clock_policy(self.before, [observed], self.after, self.policy)
        self.assertIn("performance level", " ".join(errors).lower())

    def test_unverified_flush_claim_fails(self):
        with self.assertRaisesRegex(ValueError, "verified"):
            self.environment.validate_cache_evidence(
                "cold-load",
                {"process_state": "fresh-process", "process_reuse": False, "filesystem_flush": "claimed"},
            )

    def test_prefix_cache_transitions_must_match_declared_state(self):
        for cache_state, transition in (
            ("prefix-cache-empty", "empty"),
            ("prefix-cache-populated", "populated"),
            ("prefix-cache-reset", "reset"),
        ):
            self.environment.validate_cache_evidence(
                cache_state,
                {"prefix_cache": transition, "process_reuse": False},
            )
        with self.assertRaisesRegex(ValueError, "prefix_cache"):
            self.environment.validate_cache_evidence(
                "prefix-cache-empty",
                {"prefix_cache": "populated", "process_reuse": False},
            )

    def test_warm_resident_uses_fresh_process_wording_and_no_process_reuse(self):
        snapshot, _ = self.snapshot("locked", cache_state="warm-resident")
        self.assertFalse(snapshot.process_reuse)
        self.assertEqual(snapshot.cache_evidence["process_state"], "fresh-process")

    def test_collect_snapshot_records_requested_observed_and_metadata(self):
        snapshot, calls = self.snapshot(
            cache_state="cold-load",
            cache_evidence={"process_state": "fresh-process", "process_reuse": False},
        )

        self.assertTrue(snapshot.headline_eligible)
        self.assertEqual(snapshot.requested.gpu_clock_mhz, 2400)
        self.assertEqual(snapshot.requested.memory_clock_mhz, 1249)
        self.assertEqual(snapshot.requested.power_cap_watts, 295)
        self.assertEqual(snapshot.requested.performance_level, "manual")
        self.assertEqual(snapshot.observed_before.temperature_celsius, 67.0)
        self.assertEqual(snapshot.observed_before.performance_level, "manual")
        self.assertEqual(snapshot.observed_after.power_watts, 245.0)
        self.assertEqual(snapshot.telemetry_samples[0].offset_seconds, 2.5)
        self.assertEqual(snapshot.cpu_governor, "performance")
        self.assertEqual(snapshot.physical_gpu, "1")
        self.assertEqual(snapshot.logical_gpu, "0")
        self.assertEqual(
            snapshot.allowlisted_environment,
            {
                "HIP_ARCH": "gfx1201",
                "HIP_VISIBLE_DEVICES": "1",
                "ROCM_PATH": "/opt/rocm",
                "HIP_PATH": "/opt/rocm/hip",
                "SUPERSONIC_DEVICE": "0",
                "RUSTFLAGS": "-C target-cpu=native",
            },
        )
        self.assertEqual(
            calls,
            [
                (
                    "timeout",
                    "--foreground",
                    "30s",
                    "rocm-smi",
                    "-d",
                    "1",
                    "--showallinfo",
                ),
                (
                    "timeout",
                    "--foreground",
                    "30s",
                    "rocm-smi",
                    "-d",
                    "1",
                    "--showallinfo",
                ),
                (
                    "timeout",
                    "--foreground",
                    "30s",
                    "rocm-smi",
                    "-d",
                    "1",
                    "--showallinfo",
                ),
            ],
        )

    def test_collect_snapshot_records_null_cpu_governor_with_evidence_note(self):
        snapshot, _ = self.snapshot(
            cpu_governor_reader=lambda: (_ for _ in ()).throw(FileNotFoundError("missing governor"))
        )
        self.assertIsNone(snapshot.cpu_governor)
        self.assertIn("cpu governor", " ".join(snapshot.evidence_notes).lower())


if __name__ == "__main__":
    unittest.main()
