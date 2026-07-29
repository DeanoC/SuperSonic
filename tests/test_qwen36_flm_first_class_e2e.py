import importlib.util
import io
import json
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
            "binary": Path("/repo/supersonic"),
            "limit": 1,
            "n_gen": 1,
            "context_size": 512,
            "inference_timeout": 900,
            "flm_virtual_transfer_backend": None,
            "out_json": self.tmp_path / "benchmark.json",
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
        self.assertEqual(args.limit, 1)
        self.assertEqual(args.n_gen, 1)
        self.assertEqual(args.context_size, 512)
        self.assertEqual(args.inference_timeout, 900)

    def test_supersonic_command_has_no_hf_model_or_quant_override(self):
        args = self.args()

        command = runner.supersonic_benchmark_command(args, args.flm)

        self.assertIn("qwen36-35b-a3b-flm", command)
        self.assertIn(str(args.flm), command)
        self.assertNotIn(str(args.hf_source), command)
        self.assertNotIn("--model", command)
        self.assertNotIn("--quant", command)
        self.assertNotIn("--int4", command)
        self.assertEqual(command[command.index("--limit") + 1], "1")
        self.assertEqual(command[command.index("--n-gen") + 1], "1")
        self.assertIn("--emit-stage-timings", command)
        self.assertIn("--hal-profile", command)
        self.assertEqual(command[command.index("--out-json") + 1], str(args.out_json))

    def test_supersonic_command_forwards_explicit_virtual_transfer_backend(self):
        args = self.args(flm_virtual_transfer_backend="gpu-direct-storage")

        command = runner.supersonic_benchmark_command(args, args.flm)

        self.assertEqual(
            command[-2:],
            ["--flm-virtual-transfer-backend", "gpu-direct-storage"],
        )

    def valid_report(self):
        return {
            "resolved_model": "qwen3.6-35b-a3b",
            "summary": {
                "count": 1,
                "flm_weight_modes": ["INT4 native FLM"],
                "flm_ready_for_decode_count": 1,
                "flm_direct_profiles": [{
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "bf16_fallback": 0,
                }],
                "flm_load_speed": {
                    "copy_h2d_bytes": 17179869184,
                    "copy_h2d_ms": 800.0,
                    "copy_h2d_gib_s": 20.0,
                },
            },
            "rows": [{
                "returncode": 0,
                "resolved_model": "qwen3.6-35b-a3b",
                "generated_tokens": 1,
                "flm_weight_mode": "INT4 native FLM",
                "flm_ready_for_decode": True,
                "flm_direct_profile": {
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "bf16_fallback": 0,
                },
            }],
        }

    def write_report(self, payload):
        path = self.tmp_path / "report.json"
        path.write_text(json.dumps(payload))
        return path

    def assert_report_rejected(self, payload, message):
        with self.assertRaisesRegex(runner.PhaseError, message):
            runner.validate_benchmark_report(self.write_report(payload))

    def test_valid_benchmark_report_is_accepted(self):
        payload = self.valid_report()

        self.assertEqual(
            runner.validate_benchmark_report(self.write_report(payload)),
            payload,
        )

    def test_report_rejects_wrong_resolved_model(self):
        payload = self.valid_report()
        payload["resolved_model"] = "qwen3.6-27b"

        self.assert_report_rejected(payload, "resolved model")

    def test_report_rejects_missing_row_resolved_model(self):
        payload = self.valid_report()
        payload["rows"][0].pop("resolved_model")

        self.assert_report_rejected(payload, "resolved model")

    def test_report_rejects_missing_rows(self):
        payload = self.valid_report()
        payload["rows"] = []

        self.assert_report_rejected(payload, "rows")

    def test_report_rejects_zero_summary_count(self):
        payload = self.valid_report()
        payload["summary"]["count"] = 0

        self.assert_report_rejected(payload, "summary count")

    def test_report_rejects_nonzero_row_return_code(self):
        payload = self.valid_report()
        payload["rows"][0]["returncode"] = 1

        self.assert_report_rejected(payload, "return code")

    def test_report_rejects_no_generated_tokens(self):
        payload = self.valid_report()
        payload["rows"][0]["generated_tokens"] = 0

        self.assert_report_rejected(payload, "generated tokens")

    def test_report_rejects_wrong_flm_weight_mode(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_weight_mode"] = "BF16 fallback"

        self.assert_report_rejected(payload, "weight mode")

    def test_report_rejects_unready_decode(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_ready_for_decode"] = False

        self.assert_report_rejected(payload, "ready for decode")

    def test_report_rejects_missing_decode_readiness(self):
        payload = self.valid_report()
        payload["rows"][0].pop("flm_ready_for_decode")

        self.assert_report_rejected(payload, "ready for decode")

    def test_report_rejects_no_native_int4_direct_plans(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["native_int4"] = 0

        self.assert_report_rejected(payload, "native INT4")

    def test_report_rejects_bf16_direct_plan_fallback(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["bf16_fallback"] = 1

        self.assert_report_rejected(payload, "BF16 fallback")

    def test_report_rejects_missing_transfer_bytes(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"].pop("copy_h2d_bytes")

        self.assert_report_rejected(payload, "transfer bytes")

    def test_report_rejects_zero_transfer_bytes(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"]["copy_h2d_bytes"] = 0

        self.assert_report_rejected(payload, "transfer bytes")

    def test_report_rejects_missing_transfer_speed(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"].pop("copy_h2d_gib_s")

        self.assert_report_rejected(payload, "transfer GiB/s")

    def test_report_rejects_zero_transfer_speed(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"]["copy_h2d_gib_s"] = 0.0

        self.assert_report_rejected(payload, "transfer GiB/s")

    def test_report_accepts_storage_direct_transfer_evidence(self):
        payload = self.valid_report()
        load_speed = payload["summary"]["flm_load_speed"]
        load_speed.pop("copy_h2d_bytes")
        load_speed.pop("copy_h2d_ms")
        load_speed.pop("copy_h2d_gib_s")
        load_speed.update({
            "copy_storage_to_device_bytes": 17179869184,
            "copy_storage_to_device_ms": 800.0,
            "copy_storage_to_device_gib_s": 20.0,
        })

        self.assertEqual(
            runner.validate_benchmark_report(self.write_report(payload)),
            payload,
        )

    def test_report_rejects_benchmark_validation_errors(self):
        payload = self.valid_report()
        payload["rows"][0]["benchmark_validation_errors"] = ["missing proof"]

        self.assert_report_rejected(payload, "benchmark validation errors")

    def test_parse_args_uses_positive_inference_defaults(self):
        args = runner.parse_args([
            "--limit", "2",
            "--n-gen", "3",
            "--context-size", "4",
            "--inference-timeout", "5",
        ])

        self.assertEqual(args.limit, 2)
        self.assertEqual(args.n_gen, 3)
        self.assertEqual(args.context_size, 4)
        self.assertEqual(args.inference_timeout, 5)

        for flag in ("--limit", "--n-gen", "--context-size", "--inference-timeout"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    with mock.patch("sys.stderr", new_callable=io.StringIO):
                        with self.assertRaises(SystemExit):
                            runner.parse_args([flag, value])

    def test_main_prepares_runs_and_validates_in_order(self):
        artifact = self.tmp_path / "prepared.flm"
        out_json = self.tmp_path / "benchmark.json"
        payload = self.valid_report()
        order = []

        def prepare(args):
            order.append("prepare")
            return artifact

        def run(*args, **kwargs):
            order.append("run")

        def validate(path):
            order.append("validate")
            self.assertEqual(path, out_json)
            return payload

        with mock.patch.object(runner, "prepare_artifact", side_effect=prepare), \
                mock.patch.object(runner, "run_command", side_effect=run) as run_mock, \
                mock.patch.object(runner, "validate_benchmark_report", side_effect=validate), \
                mock.patch.object(runner, "print_summary") as print_summary:
            result = runner.main([
                "--flm", str(artifact),
                "--binary", "/repo/supersonic",
                "--out-json", str(out_json),
            ])

        self.assertEqual(result, 0)
        self.assertEqual(order, ["prepare", "run", "validate"])
        self.assertEqual(run_mock.call_args.kwargs, {
            "cwd": runner.ROOT,
            "timeout": 900,
            "phase": "SuperSonic inference",
        })
        self.assertEqual(
            run_mock.call_args.args[0],
            runner.supersonic_benchmark_command(
                self.args(flm=artifact, out_json=out_json), artifact
            ),
        )
        print_summary.assert_called_once_with(payload, artifact)


if __name__ == "__main__":
    unittest.main()
