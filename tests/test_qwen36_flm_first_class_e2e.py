import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parent / "gfx1100" / "run_qwen36_flm_first_class_e2e.py"
SPEC = importlib.util.spec_from_file_location("run_qwen36_flm_first_class_e2e", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Qwen36FlmFirstClassE2ETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.existing_flm = self.tmp_path / "existing.flm"
        self.existing_flm.write_bytes(b"existing")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **overrides):
        values = {
            "hf_source": Path("/models/Qwen3.6-35B-A3B"),
            "geoquant_root": Path("/repo/geo-quant"),
            "geoquant_python": Path("/venv/bin/python"),
            "flm": self.tmp_path / "model.flm",
            "quant_device": "cuda",
            "regenerate": False,
            "export_timeout": 3600,
            "validation_timeout": 1800,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_builds_strict_native_int4_export_command(self):
        args = self.args()
        self.assertEqual(
            runner.export_command(args, Path("/models/output.partial.flm")),
            [
                "/venv/bin/python",
                "scripts/quantize_qwen36_int4.py",
                "--bf16",
                "/models/Qwen3.6-35B-A3B",
                "--flm-out",
                "/models/output.partial.flm",
                "--flm-only",
                "--device",
                "cuda",
                "--bits",
                "4",
                "--group-size",
                "128",
                "--hf-compat-assets",
                "omit",
                "--flm-validate-profile",
                "supersonic-qwen36-moe-native-int4",
            ],
        )

    def test_builds_payload_verifying_validator_command(self):
        args = self.args()
        self.assertEqual(
            runner.validate_command(
                args,
                Path("/models/output.flm"),
                verify_payload_hashes=True,
            ),
            [
                "/venv/bin/python",
                "-m",
                "geoquant.formats.flm_validate",
                "/models/output.flm",
                "--profile",
                "supersonic-qwen36-moe-native-int4",
                "--verify-payload-hashes",
            ],
        )

    def test_missing_artifact_exports_to_partial_then_promotes(self):
        args = self.args()
        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "run_command") as run, \
                mock.patch.object(runner.os, "replace") as replace:
            result = runner.prepare_artifact(args)

        partial = self.tmp_path / ".model.flm.partial-42"
        self.assertEqual(result, args.flm)
        self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
        self.assertEqual(
            run.call_args_list[1].args[0],
            runner.validate_command(args, partial, verify_payload_hashes=True),
        )
        replace.assert_called_once_with(partial, args.flm)

    def test_valid_artifact_is_reused_and_hash_verified(self):
        args = self.args(flm=self.existing_flm)
        with mock.patch.object(runner, "probe_validation", return_value=True) as probe, \
                mock.patch.object(runner, "run_command") as run:
            result = runner.prepare_artifact(args)

        self.assertEqual(result, args.flm)
        probe.assert_called_once_with(args, args.flm)
        run.assert_called_once_with(
            runner.validate_command(args, args.flm, verify_payload_hashes=True),
            cwd=args.geoquant_root,
            timeout=args.validation_timeout,
            phase="payload validation",
        )

    def test_regenerate_preserves_existing_artifact_until_promotion(self):
        args = self.args(flm=self.existing_flm, regenerate=True)
        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "probe_validation") as probe, \
                mock.patch.object(runner, "run_command") as run, \
                mock.patch.object(runner.os, "replace") as replace:
            result = runner.prepare_artifact(args)

        partial = self.tmp_path / ".existing.flm.partial-42"
        self.assertEqual(result, args.flm)
        probe.assert_not_called()
        self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
        self.assertEqual(run.call_args_list[1].args[0], runner.validate_command(
            args, partial, verify_payload_hashes=True
        ))
        replace.assert_called_once_with(partial, args.flm)

    def test_invalid_artifact_selects_safe_regeneration(self):
        args = self.args(flm=self.existing_flm)
        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "probe_validation", return_value=False), \
                mock.patch.object(runner, "run_command") as run, \
                mock.patch.object(runner.os, "replace"):
            with mock.patch("builtins.print") as print_mock:
                result = runner.prepare_artifact(args)

        partial = self.tmp_path / ".existing.flm.partial-42"
        self.assertEqual(result, args.flm)
        self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
        print_mock.assert_called_once()
        self.assertIn("stale or incompatible", print_mock.call_args.args[0])

    def test_existing_partial_fails_closed(self):
        args = self.args()
        partial = self.tmp_path / ".model.flm.partial-42"
        partial.write_bytes(b"partial")
        with mock.patch.object(runner.os, "getpid", return_value=42):
            with self.assertRaisesRegex(runner.PhaseError, "export target already exists"):
                runner.prepare_artifact(args)

    def test_parse_args_uses_strict_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.quant_device, "cuda")
        self.assertEqual(args.geoquant_root, runner.DEFAULT_GEOQUANT_ROOT)
        self.assertEqual(args.geoquant_python, runner.DEFAULT_GEOQUANT_PYTHON)
        self.assertEqual(args.flm, runner.DEFAULT_FLM)


if __name__ == "__main__":
    unittest.main()
