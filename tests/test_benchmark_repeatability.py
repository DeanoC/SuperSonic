from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.benchmark.execution import ProcessResult


class SequenceRunner:
    def __init__(self, outputs: list[str], *, manifest: Path | None = None) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[str, ...]] = []
        self.manifest = manifest
        self.manifests_seen_before_trace: list[dict[str, object]] = []

    def __call__(self, argv, *, timeout, physical_gpu, cwd=None, env=None):
        vector = tuple(argv)
        self.calls.append(vector)
        if self.manifest is not None and vector[0] == "rocprofv3":
            self.manifests_seen_before_trace.append(
                json.loads(self.manifest.read_text(encoding="utf-8"))
            )
            output_index = vector.index("--output-directory") + 1
            trace = Path(vector[output_index]) / "results.json"
            trace.write_text("{}\n", encoding="utf-8")
        output = next(self.outputs)
        return ProcessResult(vector, 0, output, "", 1.0), (), ()


class RaisingRunner:
    def __call__(self, argv, *, timeout, physical_gpu, cwd=None, env=None):
        raise KeyboardInterrupt


class NonzeroTraceRunner(SequenceRunner):
    def __call__(self, argv, *, timeout, physical_gpu, cwd=None, env=None):
        vector = tuple(argv)
        if vector[0] == "rocprofv3":
            self.calls.append(vector)
            return ProcessResult(vector, 1, "", "profiler failed", 1.0), (), ()
        return super().__call__(argv, timeout=timeout, physical_gpu=physical_gpu, cwd=cwd, env=env)


class AdvancingRunner(SequenceRunner):
    def __init__(self, outputs: list[str], clock: "FakeClock") -> None:
        super().__init__(outputs)
        self.clock = clock
        self.timeouts: list[float] = []

    def __call__(self, argv, *, timeout, physical_gpu, cwd=None, env=None):
        self.timeouts.append(timeout)
        self.clock.now += 4.0
        return super().__call__(argv, timeout=timeout, physical_gpu=physical_gpu, cwd=cwd, env=env)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def sample_output(*, persistent_ms: float, tokens: str = "1 2") -> str:
    return (
        '[generated_json] "ok"\n'
        f"[tokens] {tokens}\n"
        "[result] prompt_tokens=2 generated_tokens=2 decode_ms=100.0 ms_per_tok=50.0\n"
        f"[stage-timings] steps=2 persistent_ms={persistent_ms} total_native_decode_ms=100.0\n"
    )


class RepeatabilitySoakTests(unittest.TestCase):
    def setUp(self) -> None:
        from tools.benchmark import repeatability

        self.repeatability = repeatability
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "soak"
        self.config = repeatability.SoakConfig(
            argv=("supersonic", "--prompt", "fixed"),
            output=self.output,
            physical_gpu="1",
            slow_persistent_ms_per_token=55.0,
            max_runs=4,
            trace_attempts=2,
            timeout_seconds=60.0,
            max_duration_seconds=21600.0,
            rocprof_binary="rocprofv3",
            logical_gpu=0,
            hip_visible_devices="1",
        )

    def test_first_slow_sample_is_preserved_before_followup_traces(self):
        runner = SequenceRunner(
            [
                sample_output(persistent_ms=80.0),
                sample_output(persistent_ms=120.0),
                sample_output(persistent_ms=118.0),
                sample_output(persistent_ms=121.0),
            ],
            manifest=self.output / "manifest.json",
        )

        result = self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(result["state"], "slow-captured")
        self.assertEqual(result["trigger_run"], 2)
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(len(result["followup_traces"]), 2)
        self.assertEqual(result["followup_traces"][0]["relationship"], "followup-reproduction")
        self.assertIn("--kernel-trace", runner.calls[2])
        self.assertIn("--memory-allocation-trace", runner.calls[2])
        self.assertEqual(runner.manifests_seen_before_trace[0]["state"], "slow-triggered")
        self.assertEqual(runner.manifests_seen_before_trace[0]["trigger_run"], 2)
        self.assertEqual(len(runner.manifests_seen_before_trace[0]["samples"]), 2)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, result)
        self.assertIn("persistent_ms=120.0", (self.output / "logs" / "run-2.stdout.log").read_text())

    def test_token_mismatch_fails_without_running_a_trace(self):
        runner = SequenceRunner(
            [
                sample_output(persistent_ms=80.0, tokens="1 2"),
                sample_output(persistent_ms=120.0, tokens="1 3"),
            ]
        )

        result = self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"], "token-mismatch")
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(result["followup_traces"], [])

    def test_no_slow_sample_retains_complete_bounded_series(self):
        runner = SequenceRunner([sample_output(persistent_ms=80.0) for _ in range(4)])

        result = self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(result["state"], "no-slow-sample")
        self.assertEqual(result["trigger_run"], None)
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(result["followup_traces"], [])

    def test_running_manifest_and_each_completed_sample_are_durable(self):
        manifest = self.output / "manifest.json"

        class InspectingRunner(SequenceRunner):
            states: list[dict[str, object]] = []

            def __call__(inner_self, argv, *, timeout, physical_gpu, cwd=None, env=None):
                inner_self.states.append(json.loads(manifest.read_text(encoding="utf-8")))
                return super(InspectingRunner, inner_self).__call__(
                    argv, timeout=timeout, physical_gpu=physical_gpu, cwd=cwd, env=env
                )

        runner = InspectingRunner([sample_output(persistent_ms=80.0) for _ in range(4)])
        self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(runner.states[0]["state"], "running")
        self.assertEqual(len(runner.states[0]["samples"]), 0)
        self.assertEqual(len(runner.states[1]["samples"]), 1)

    def test_interrupt_retains_an_interrupted_manifest(self):
        result = self.repeatability.run_soak(self.config, sample_runner=RaisingRunner())

        self.assertEqual(result["state"], "interrupted")
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["state"], "interrupted")

    def test_malformed_output_retains_a_failed_manifest(self):
        result = self.repeatability.run_soak(
            self.config,
            sample_runner=SequenceRunner(["not benchmark output"]),
        )

        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"], "invalid-output")
        self.assertTrue((self.output / "logs" / "run-1.stdout.log").exists())

    def test_all_failed_trace_attempts_fail_the_soak(self):
        runner = NonzeroTraceRunner([sample_output(persistent_ms=120.0)])

        result = self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(result["state"], "trace-failed")
        self.assertEqual(len(result["followup_traces"]), 2)
        self.assertTrue(all("error" in trace for trace in result["followup_traces"]))

    def test_successful_profiler_without_json_trace_fails_validation(self):
        class MissingTraceRunner(SequenceRunner):
            def __call__(inner_self, argv, *, timeout, physical_gpu, cwd=None, env=None):
                vector = tuple(argv)
                inner_self.calls.append(vector)
                output = next(inner_self.outputs)
                return ProcessResult(vector, 0, output, "", 1.0), (), ()

        runner = MissingTraceRunner(
            [sample_output(persistent_ms=120.0), sample_output(persistent_ms=118.0), sample_output(persistent_ms=119.0)]
        )
        result = self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertEqual(result["state"], "trace-failed")
        self.assertTrue(all(trace["error"] == "missing-valid-json-trace" for trace in result["followup_traces"]))

    def test_manifest_declares_fresh_process_cache_and_device_mapping(self):
        result = self.repeatability.run_soak(
            self.config,
            sample_runner=SequenceRunner([sample_output(persistent_ms=80.0) for _ in range(4)]),
        )

        self.assertEqual(result["cache_state"], "cold-load")
        self.assertEqual(result["process_state"], "fresh-process")
        self.assertFalse(result["process_reuse"])
        self.assertEqual(result["device_mapping"], {"physical_gpu": "1", "logical_gpu": 0, "HIP_VISIBLE_DEVICES": "1"})

    def test_physical_to_logical_mapping_fails_closed(self):
        bad = replace(self.config, hip_visible_devices="0")
        with self.assertRaisesRegex(ValueError, "HIP_VISIBLE_DEVICES"):
            self.repeatability.run_soak(bad, sample_runner=SequenceRunner([]))

    def test_effective_child_environment_must_match_recorded_mapping(self):
        bad = replace(self.config, environment={"HIP_VISIBLE_DEVICES": "0"})
        with self.assertRaisesRegex(ValueError, "effective.*HIP_VISIBLE_DEVICES"):
            self.repeatability.run_soak(bad, sample_runner=SequenceRunner([]))

    def test_recorded_mapping_overrides_an_unconfigured_ambient_mapping(self):
        seen: list[dict[str, str]] = []

        class EnvironmentRunner(SequenceRunner):
            def __call__(inner_self, argv, *, timeout, physical_gpu, cwd=None, env=None):
                seen.append(dict(env))
                return super(EnvironmentRunner, inner_self).__call__(
                    argv, timeout=timeout, physical_gpu=physical_gpu, cwd=cwd, env=env
                )

        with mock.patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "0"}):
            self.repeatability.run_soak(
                self.config,
                sample_runner=EnvironmentRunner(
                    [sample_output(persistent_ms=80.0) for _ in range(4)]
                ),
            )

        self.assertTrue(seen)
        self.assertTrue(all(item["HIP_VISIBLE_DEVICES"] == "1" for item in seen))

    def test_aggregate_deadline_bounds_the_series(self):
        clock = FakeClock()
        runner = AdvancingRunner([sample_output(persistent_ms=80.0) for _ in range(4)], clock)
        config = replace(self.config, max_duration_seconds=10.0, timeout_seconds=60.0)

        result = self.repeatability.run_soak(config, sample_runner=runner, monotonic=clock)

        self.assertEqual(result["state"], "duration-complete")
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual(runner.timeouts, [10.0, 6.0, 2.0])

    def test_overnight_limits_cannot_be_expanded(self):
        for field, value in (
            ("max_runs", 2161),
            ("trace_attempts", 4),
            ("max_duration_seconds", 21601.0),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self.repeatability.run_soak(
                        replace(self.config, **{field: value}),
                        sample_runner=SequenceRunner([]),
                    )

    def test_streams_and_atomic_manifests_are_fsynced(self):
        runner = SequenceRunner([sample_output(persistent_ms=80.0) for _ in range(4)])
        with mock.patch.object(self.repeatability.os, "fsync") as fsync:
            self.repeatability.run_soak(self.config, sample_runner=runner)

        self.assertGreaterEqual(fsync.call_count, 7)


if __name__ == "__main__":
    unittest.main()
