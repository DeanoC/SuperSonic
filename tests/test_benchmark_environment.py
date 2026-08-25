from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
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
            power_cap_watts=295,
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
        fixture_text: str | None = None,
    ):
        fixture = fixture_text or (FIXTURES / "rocm-smi-showallinfo.txt").read_text(encoding="utf-8")
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
                "AMDSMI_GPU_METRICS_CACHE_MS": "0",
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

    def test_loaded_clock_drift_requires_three_consecutive_out_of_band_samples(self):
        self.assertEqual(
            self.environment.verify_clock_policy(self.before, [self.drifted], self.after, self.policy),
            (),
        )
        self.assertEqual(
            self.environment.verify_clock_policy(
                self.before,
                [self.drifted, self.drifted],
                self.after,
                self.policy,
            ),
            (),
        )
        errors = self.environment.verify_clock_policy(
            self.before,
            [self.drifted, self.drifted, self.drifted],
            self.after,
            self.policy,
        )
        self.assertIn("sustained clock drift", " ".join(errors).lower())

        reset_errors = self.environment.verify_clock_policy(
            self.before,
            [self.drifted, self.drifted, self.before, self.drifted, self.drifted],
            self.after,
            self.policy,
        )
        self.assertEqual(reset_errors, ())

    def test_snapshot_verifies_each_live_observation_once(self):
        drift_samples = tuple(
            self.environment.TelemetrySample(
                offset_seconds=float(index),
                gpu_clock_mhz=self.drifted.gpu_clock_mhz,
                memory_clock_mhz=self.drifted.memory_clock_mhz,
                power_cap_watts=self.drifted.power_cap_watts,
                power_watts=self.drifted.power_watts,
                temperature_celsius=self.drifted.temperature_celsius,
                gpu_utilization_percent=self.drifted.gpu_utilization_percent,
                memory_utilization_percent=self.drifted.memory_utilization_percent,
                performance_level=self.drifted.performance_level,
            )
            for index in (1, 2)
        )

        snapshot = self.environment.snapshot_from_observations(
            physical_gpu="1",
            logical_gpu="0",
            clock_policy=self.policy,
            cache_state="warm-resident",
            observed_before=self.before,
            observed_before_at="2026-08-24T12:00:00Z",
            telemetry_samples=drift_samples,
            observed_after=self.after,
            observed_after_at="2026-08-24T12:00:02Z",
            cpu_governor_reader=lambda: "performance\n",
        )

        self.assertTrue(snapshot.headline_eligible)
        self.assertEqual(snapshot.telemetry_samples, drift_samples)

    def test_locked_gpu_clock_uses_loaded_samples_and_nominal_tolerance(self):
        policy = SimpleNamespace(**{**self.policy.__dict__, "gpu_clock_mhz": 2350, "clock_tolerance_mhz": 100})
        idle_before = replace(self.before, gpu_clock_mhz=412, gpu_utilization_percent=0.0)
        loaded = replace(self.before, gpu_clock_mhz=2302, gpu_utilization_percent=99.0)
        idle_after = replace(self.after, gpu_clock_mhz=34, gpu_utilization_percent=0.0)

        errors = self.environment.verify_clock_policy(idle_before, [loaded], idle_after, policy)

        self.assertEqual(errors, ())

    def test_amd_smi_metric_records_throttle_as_advisory_clock_evidence(self):
        payload = {
            "gpu_data": [
                {
                    "gpu": 1,
                    "usage": {"gfx_activity": {"value": 97, "unit": "%"}},
                    "power": {
                        "socket_power": {"value": 144, "unit": "W"},
                        "throttle_status": "THROTTLED",
                    },
                    "clock": {
                        "gfx_0": {"clk": {"value": 2313, "unit": "MHz"}},
                        "mem_0": {"clk": {"value": 1258, "unit": "MHz"}},
                    },
                    "temperature": {"edge": {"value": 42, "unit": "C"}},
                    "perf_level": "AMDSMI_DEV_PERF_LEVEL_MANUAL",
                    "throttle": {"indep_throttle_status": 7},
                }
            ]
        }
        observed = self.environment.parse_amd_smi_metric(json.dumps(payload), physical_gpu="1")
        merged = self.environment.merge_telemetry(self.before, observed)

        self.assertEqual(merged.gpu_clock_mhz, 2313)
        self.assertEqual(merged.memory_clock_mhz, 1258)
        self.assertEqual(merged.performance_level, "manual")
        self.assertEqual(merged.throttle_label, "THROTTLED")
        self.assertIsNone(merged.throttle_status)
        self.assertEqual(merged.indep_throttle_status, 7)
        policy = SimpleNamespace(**{**self.policy.__dict__, "gpu_clock_mhz": 2350, "clock_tolerance_mhz": 100})
        self.assertEqual(self.environment.verify_clock_policy(self.after, [merged], self.after, policy), ())

    def test_loaded_clock_summary_ignores_idle_edges_and_retains_distribution(self):
        idle = replace(self.before, gpu_clock_mhz=4, gpu_utilization_percent=0.0)
        loaded = [
            replace(self.before, gpu_clock_mhz=2313, gpu_utilization_percent=97.0),
            replace(self.before, gpu_clock_mhz=2170, gpu_utilization_percent=92.0),
            replace(self.before, gpu_clock_mhz=2290, gpu_utilization_percent=99.0),
            replace(self.before, gpu_clock_mhz=2140, gpu_utilization_percent=50.0),
        ]

        self.assertEqual(
            self.environment.loaded_clock_summary([idle, *loaded, idle]),
            {"count": 4, "minimum_mhz": 2140, "median_mhz": 2230, "maximum_mhz": 2313},
        )

    def test_timing_dispersion_rejects_mad_above_three_percent(self):
        self.assertEqual(self.environment.verify_timing_dispersion([10.0] * 7), ())
        errors = self.environment.verify_timing_dispersion([8.0, 8.0, 8.0, 10.0, 12.0, 12.0, 12.0])
        self.assertIn("MAD", " ".join(errors))
        self.assertIn("3%", " ".join(errors))

    def test_locked_gpu_clock_requires_a_loaded_sample(self):
        policy = SimpleNamespace(**{**self.policy.__dict__, "gpu_clock_mhz": 2350, "clock_tolerance_mhz": 100})
        idle_before = replace(self.before, gpu_clock_mhz=412, gpu_utilization_percent=0.0)
        idle_sample = replace(self.before, gpu_clock_mhz=500, gpu_utilization_percent=5.0)
        idle_after = replace(self.after, gpu_clock_mhz=34, gpu_utilization_percent=0.0)

        errors = self.environment.verify_clock_policy(idle_before, [idle_sample], idle_after, policy)

        self.assertIn("loaded GPU clock", " ".join(errors))

    def test_snapshot_from_live_observations_preserves_loaded_telemetry(self):
        policy = SimpleNamespace(**{**self.policy.__dict__, "gpu_clock_mhz": 2350, "clock_tolerance_mhz": 100})
        idle_before = replace(self.before, gpu_clock_mhz=412, gpu_utilization_percent=0.0)
        loaded = self.environment.TelemetrySample(
            offset_seconds=1.25,
            gpu_clock_mhz=2302,
            memory_clock_mhz=1249,
            power_cap_watts=295,
            power_watts=245.0,
            temperature_celsius=67.0,
            gpu_utilization_percent=99.0,
            memory_utilization_percent=88.0,
            performance_level="manual",
        )
        idle_after = replace(self.after, gpu_clock_mhz=34, gpu_utilization_percent=0.0)

        snapshot = self.environment.snapshot_from_observations(
            physical_gpu="1",
            logical_gpu="0",
            clock_policy=policy,
            cache_state="warm-resident",
            observed_before=idle_before,
            observed_before_at="2026-08-24T12:00:00Z",
            telemetry_samples=(loaded,),
            observed_after=idle_after,
            observed_after_at="2026-08-24T12:00:02Z",
            environment_map={"HIP_ARCH": "gfx1201", "HIP_VISIBLE_DEVICES": "1"},
            cpu_governor_reader=lambda: "performance\n",
        )

        self.assertTrue(snapshot.headline_eligible)
        self.assertEqual(snapshot.telemetry_samples, (loaded,))
        self.assertEqual(snapshot.requested.clock_tolerance_mhz, 100)

    def test_locked_performance_level_mismatch_fails(self):
        observed = replace(self.before, performance_level="auto")
        errors = self.environment.verify_clock_policy(self.before, [observed], self.after, self.policy)
        self.assertIn("performance level", " ".join(errors).lower())

    def test_locked_power_cap_match_passes(self):
        errors = self.environment.verify_clock_policy(self.before, [self.before], self.after, self.policy)
        self.assertNotIn("power cap", " ".join(errors).lower())

    def test_locked_power_cap_missing_fails(self):
        observed = replace(self.before, power_cap_watts=None)
        errors = self.environment.verify_clock_policy(self.before, [observed], self.after, self.policy)
        self.assertIn("power cap", " ".join(errors).lower())

    def test_locked_power_cap_mismatch_fails(self):
        observed = replace(self.before, power_cap_watts=280)
        errors = self.environment.verify_clock_policy(self.before, [observed], self.after, self.policy)
        self.assertIn("power cap", " ".join(errors).lower())

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
        self.assertEqual(snapshot.requested.clock_tolerance_mhz, 20)
        self.assertEqual(snapshot.requested.memory_clock_mhz, 1249)
        self.assertEqual(snapshot.requested.power_cap_watts, 295)
        self.assertEqual(snapshot.requested.performance_level, "manual")
        self.assertEqual(snapshot.observed_before.power_cap_watts, 295)
        self.assertEqual(snapshot.observed_before.temperature_celsius, 67.0)
        self.assertEqual(snapshot.observed_before.performance_level, "manual")
        self.assertEqual(snapshot.observed_after.power_cap_watts, 295)
        self.assertEqual(snapshot.observed_after.power_watts, 245.0)
        self.assertEqual(snapshot.telemetry_samples[0].power_cap_watts, 295)
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
                "AMDSMI_GPU_METRICS_CACHE_MS": "0",
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

    def test_amd_metric_probe_disables_driver_metric_cache(self):
        command = self.environment.build_amd_metric_command("1")

        self.assertEqual(
            command,
            (
                "timeout",
                "--foreground",
                "30s",
                "env",
                "AMDSMI_GPU_METRICS_CACHE_MS=0",
                "amd-smi",
                "metric",
                "--gpu",
                "1",
                "--json",
            ),
        )

    def test_collect_snapshot_records_null_cpu_governor_with_evidence_note(self):
        snapshot, _ = self.snapshot(
            cpu_governor_reader=lambda: (_ for _ in ()).throw(FileNotFoundError("missing governor"))
        )
        self.assertIsNone(snapshot.cpu_governor)
        self.assertIn("cpu governor", " ".join(snapshot.evidence_notes).lower())

    def test_locked_snapshot_with_unsupported_probe_fields_is_not_headline_eligible(self):
        unsupported_fixture = (
            (FIXTURES / "rocm-smi-showallinfo.txt")
            .read_text(encoding="utf-8")
            .replace("GPU[1]          : Max Graphics Package Power (W): 295.0\n", "")
        )
        snapshot, _ = self.snapshot(fixture_text=unsupported_fixture)
        self.assertFalse(snapshot.headline_eligible)
        self.assertIn("power cap", " ".join(snapshot.evidence_notes).lower())

    def test_installed_rocm_smi_clock_labels_are_parsed(self):
        # Captured from this benchmark host's installed rocm-smi --showallinfo.
        # A parser that only accepts the synthetic "GPU Clock Level" wording
        # would silently make every locked-clock candidate ineligible.
        observed = self.environment._parse_showallinfo(
            """\
GPU[1]        : sclk clock level: S: (70Mhz)
GPU[1]        : mclk clock level: 0: (96Mhz)
GPU[1]        : Performance Level: auto
GPU[1]        : Max Graphics Package Power (W): 300.0
""",
            [],
        )

        self.assertEqual(observed.gpu_clock_mhz, 70)
        self.assertEqual(observed.memory_clock_mhz, 96)


if __name__ == "__main__":
    unittest.main()
