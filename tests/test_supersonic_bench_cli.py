from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


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
                "--memory-clock-mhz",
                "1800",
                "--power-cap-watts",
                "300",
            ]
        )
        self.assertEqual(args.gpu_clock_mhz, 2600)
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


if __name__ == "__main__":
    unittest.main()
