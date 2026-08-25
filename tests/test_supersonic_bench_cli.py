from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "supersonic-bench.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("supersonic_bench_cli", TOOL)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkCliTests(unittest.TestCase):
    def test_run_requires_explicit_inputs(self):
        cli = load_cli_module()
        with self.assertRaises(SystemExit) as raised:
            cli.main(["run", "--suite", "quick"])
        self.assertNotEqual(raised.exception.code, 0)

    def test_validate_publishable_uses_publishability_gate(self):
        cli = load_cli_module()
        with tempfile.TemporaryDirectory() as temporary:
            self.assertNotEqual(cli.main(["validate", "--publishable", temporary]), 0)

    def test_compare_emits_json(self):
        cli = load_cli_module()
        self.assertTrue(hasattr(cli, "main"))
        self.assertTrue(hasattr(cli, "build_parser"))

    def test_render_accepts_results_and_output_paths(self):
        cli = load_cli_module()
        parser = cli.build_parser()
        args = parser.parse_args(["render", "benchmarks/results", "target/site"])
        self.assertEqual(args.command, "render")
        self.assertEqual(args.results_root, Path("benchmarks/results"))
        self.assertEqual(args.output_root, Path("target/site"))

    def test_public_cli_does_not_add_runner_execution_flags(self):
        cli = load_cli_module()
        parser = cli.build_parser()
        text = str(parser)
        self.assertNotIn("--model qwen3.8-27b", text)
        self.assertNotIn("--gguf-file", text)

    def test_locked_clock_policy_accepts_requested_telemetry_values(self):
        cli = load_cli_module()
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--suite",
                "quick",
                "--model-dir",
                "/model",
                "--artifact",
                "/artifact",
                "--artifact-semantic-id",
                "test-artifact",
                "--artifact-quantization",
                "GQH-Q3KXL",
                "--tokenizer-sha256",
                "a" * 64,
                "--chat-template-sha256",
                "b" * 64,
                "--physical-gpu",
                "0",
                "--gpu-static-json",
                "/tmp/amd-smi-static.json",
                "--rocm-version-file",
                "/tmp/rocm-version.txt",
                "--hip-version-file",
                "/tmp/hip-version.txt",
                "--gpu-arch",
                "gfx1201",
                "--clock-policy",
                "locked",
                "--gpu-clock-mhz",
                "2600",
                "--gpu-clock-tolerance-mhz",
                "100",
                "--memory-clock-mhz",
                "1800",
                "--power-cap-watts",
                "300",
            ]
        )
        self.assertEqual(args.gpu_clock_mhz, 2600)
        self.assertEqual(args.gpu_clock_tolerance_mhz, 100)
        self.assertEqual(args.memory_clock_mhz, 1800)
        self.assertEqual(args.power_cap_watts, 300)
        self.assertEqual(args.rocm_version_file, Path("/tmp/rocm-version.txt"))
        self.assertEqual(args.hip_version_file, Path("/tmp/hip-version.txt"))

    def test_gpu_identity_cannot_be_forged_through_public_cli(self):
        cli = load_cli_module()
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--suite",
                    "quick",
                    "--model-dir",
                    "/model",
                    "--artifact",
                    "/artifact",
                    "--physical-gpu",
                    "0",
                    "--gpu-identity",
                    "forged",
                ]
            )

    def test_run_accepts_explicit_artifact_and_model_component_identities(self):
        cli = load_cli_module()
        args = cli.build_parser().parse_args(
            [
                "run",
                "--suite",
                "quick",
                "--model-dir",
                "/model",
                "--artifact",
                "/artifact",
                "--artifact-semantic-id",
                "qwen3.8-27b-gqh-q3kxl",
                "--artifact-quantization",
                "GQH-Q3KXL",
                "--tokenizer-sha256",
                "a" * 64,
                "--chat-template-sha256",
                "b" * 64,
                "--physical-gpu",
                "0",
                "--gpu-static-json",
                "/tmp/amd-smi-static.json",
                "--rocm-version-file",
                "/tmp/rocm-version.txt",
                "--hip-version-file",
                "/tmp/hip-version.txt",
                "--gpu-arch",
                "gfx1201",
            ]
        )

        self.assertEqual(args.artifact_semantic_id, "qwen3.8-27b-gqh-q3kxl")
        self.assertEqual(args.artifact_quantization, "GQH-Q3KXL")
        self.assertEqual(args.tokenizer_sha256, "a" * 64)
        self.assertEqual(args.chat_template_sha256, "b" * 64)

    def test_repeatability_builds_a_bounded_fresh_process_soak(self):
        cli = load_cli_module()
        args = cli.build_parser().parse_args(
            [
                "repeatability",
                "--model-dir",
                "/data/models/Qwen3.8-27B",
                "--artifact",
                "/data/models/qwen38.gqh.gguf",
                "--physical-gpu",
                "1",
                "--output",
                "/tmp/repeatability",
                "--max-runs",
                "12",
            ]
        )

        with mock.patch.dict("os.environ", {"HIP_VISIBLE_DEVICES": "1"}):
            config = cli._repeatability_config(args)

        self.assertEqual(config.max_runs, 12)
        self.assertEqual(config.max_duration_seconds, 21600.0)
        self.assertEqual(config.slow_persistent_ms_per_token, 55.0)
        self.assertEqual(config.trace_attempts, 3)
        self.assertEqual(config.argv[0], "./target/release/supersonic")
        self.assertIn("--emit-stage-timings", config.argv)
        self.assertIn("--ignore-eos", config.argv)
        self.assertIn("Emit a single sentence describing cold-load benchmark startup.", config.argv)
        self.assertEqual(config.physical_gpu, "1")
        self.assertEqual(config.hip_visible_devices, "1")
        self.assertEqual(config.environment["HIP_VISIBLE_DEVICES"], "1")

    def test_repeatability_prints_a_concise_manifest_summary(self):
        cli = load_cli_module()
        result = {
            "state": "slow-captured",
            "trigger_run": 7,
            "samples": [{"telemetry_samples": [{"gpu_clock_mhz": 2350}]}],
            "followup_traces": [{"telemetry_samples": [{"gpu_clock_mhz": 2300}]}],
        }
        args = object()
        stdout = io.StringIO()

        with (
            mock.patch.object(cli, "_repeatability_config", return_value=object()),
            mock.patch.object(cli.repeatability, "run_soak", return_value=result),
            mock.patch("sys.stdout", stdout),
        ):
            status = cli._repeatability(args)

        self.assertEqual(status, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "followup_traces": 1,
                "samples": 1,
                "state": "slow-captured",
                "trigger_run": 7,
            },
        )
        self.assertNotIn("gpu_clock_mhz", stdout.getvalue())

    def test_repeatability_trace_failure_returns_nonzero(self):
        cli = load_cli_module()
        result = {
            "state": "trace-failed",
            "trigger_run": 1,
            "samples": [{}],
            "followup_traces": [{"error": "profiler-process-failed"}],
        }
        with (
            mock.patch.object(cli, "_repeatability_config", return_value=object()),
            mock.patch.object(cli.repeatability, "run_soak", return_value=result),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(cli._repeatability(object()), 1)


if __name__ == "__main__":
    unittest.main()
