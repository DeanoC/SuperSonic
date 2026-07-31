import importlib.util
import hashlib
import io
import json
import signal
import subprocess
import sys
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
            "regenerate_output": None,
            "overwrite_artifact": False,
            "export_timeout": 3600,
            "validation_timeout": 1800,
            "binary": Path("/repo/supersonic"),
            "limit": 1,
            "n_gen": 1,
            "context_size": 512,
            "inference_timeout": 900,
            "inference_cleanup_grace": 30,
            "flm_virtual_transfer_backend": None,
            "out_json": self.tmp_path / "benchmark.json",
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_builds_strict_row_group_int4_export_command(self):
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
                "32",
                "--flm-int4-codec",
                "row-group",
                "--int4-recipe",
                "mse",
                "--hf-compat-assets",
                "omit",
                "--flm-validate-profile",
                "supersonic-qwen36-moe-row-group-int4",
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
                "supersonic-qwen36-moe-row-group-int4",
                "--verify-payload-hashes",
            ],
        )

    def test_missing_artifact_requires_explicit_regeneration(self):
        args = self.args(regenerate=False)

        with self.assertRaisesRegex(runner.PhaseError, "--regenerate"):
            runner.choose_artifact_action(args)

    def test_explicit_regeneration_of_missing_artifact_promotes_to_requested_path(self):
        args = self.args(regenerate=True)

        def run_export(command, **_kwargs):
            if "--flm-out" in command:
                Path(command[command.index("--flm-out") + 1]).write_bytes(b"generated")

        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "run_command", side_effect=run_export) as run:
            result = runner.prepare_artifact(args)

        partial = self.tmp_path / ".model.flm.partial-42"
        self.assertEqual(result.artifact, args.flm)
        self.assertEqual(result.action, "regenerate")
        self.assertIsNone(result.before_sha256)
        self.assertEqual(
            result.after_sha256,
            hashlib.sha256(b"generated").hexdigest(),
        )
        self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
        self.assertEqual(
            run.call_args_list[1].args[0],
            runner.validate_command(args, partial, verify_payload_hashes=True),
        )
        self.assertEqual(args.flm.read_bytes(), b"generated")

    def test_valid_artifact_is_reused_and_hash_verified(self):
        args = self.args(flm=self.existing_flm)
        with mock.patch.object(runner, "probe_validation", return_value=True) as probe, \
                mock.patch.object(runner, "run_command") as run:
            result = runner.prepare_artifact(args)

        expected_digest = hashlib.sha256(b"existing").hexdigest()
        self.assertEqual(result.artifact, args.flm)
        self.assertEqual(result.action, "reuse")
        self.assertEqual(result.source, args.flm)
        self.assertEqual(result.destination, args.flm)
        self.assertEqual(result.before_sha256, expected_digest)
        self.assertEqual(result.after_sha256, expected_digest)
        probe.assert_called_once_with(args, args.flm)
        run.assert_called_once_with(
            runner.validate_command(args, args.flm, verify_payload_hashes=True),
            cwd=args.geoquant_root,
            timeout=args.validation_timeout,
            phase="payload validation",
        )

    def test_regenerate_preserves_existing_artifact_and_prefers_distinct_output(self):
        args = self.args(flm=self.existing_flm, regenerate=True)

        def run_export(command, **_kwargs):
            if "--flm-out" in command:
                Path(command[command.index("--flm-out") + 1]).write_bytes(b"generated")

        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "probe_validation") as probe, \
                mock.patch.object(runner, "run_command", side_effect=run_export) as run:
            result = runner.prepare_artifact(args)

        destination = self.tmp_path / "existing.regenerated.flm"
        partial = self.tmp_path / ".existing.regenerated.flm.partial-42"
        self.assertEqual(result.artifact, destination)
        self.assertEqual(result.source, args.hf_source)
        self.assertEqual(result.destination, destination)
        self.assertEqual(self.existing_flm.read_bytes(), b"existing")
        self.assertEqual(destination.read_bytes(), b"generated")
        probe.assert_not_called()
        self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
        self.assertEqual(run.call_args_list[1].args[0], runner.validate_command(
            args, partial, verify_payload_hashes=True
        ))

    def test_invalid_reuse_fails_closed_without_export_or_overwrite(self):
        args = self.args(flm=self.existing_flm)
        with mock.patch.object(runner, "probe_validation", return_value=False), \
                mock.patch.object(runner, "run_command") as run:
            with self.assertRaisesRegex(runner.PhaseError, "reuse validation failed"):
                runner.prepare_artifact(args)

        run.assert_not_called()
        self.assertEqual(self.existing_flm.read_bytes(), b"existing")

    def test_regenerate_requires_overwrite_opt_in_for_same_destination(self):
        args = self.args(
            flm=self.existing_flm,
            regenerate=True,
            regenerate_output=self.existing_flm,
        )

        with self.assertRaisesRegex(runner.PhaseError, "--overwrite-artifact"):
            runner.prepare_artifact(args)

        self.assertEqual(self.existing_flm.read_bytes(), b"existing")

    def test_explicit_overwrite_opt_in_can_replace_regeneration_destination(self):
        args = self.args(
            flm=self.existing_flm,
            regenerate=True,
            regenerate_output=self.existing_flm,
            overwrite_artifact=True,
        )

        def run_export(command, **_kwargs):
            if "--flm-out" in command:
                Path(command[command.index("--flm-out") + 1]).write_bytes(b"generated")

        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "run_command", side_effect=run_export):
            result = runner.prepare_artifact(args)

        self.assertEqual(result.artifact, self.existing_flm)
        self.assertEqual(
            result.before_sha256,
            hashlib.sha256(b"existing").hexdigest(),
        )
        self.assertEqual(
            result.after_sha256,
            hashlib.sha256(b"generated").hexdigest(),
        )
        self.assertEqual(self.existing_flm.read_bytes(), b"generated")

    def test_existing_partial_fails_closed(self):
        args = self.args(regenerate=True)
        partial = self.tmp_path / ".model.flm.partial-42"
        partial.write_bytes(b"partial")
        with mock.patch.object(runner.os, "getpid", return_value=42):
            with self.assertRaisesRegex(runner.PhaseError, "export target already exists"):
                runner.prepare_artifact(args)

    def test_reuse_input_discovery_does_not_require_hf_source(self):
        args = self.args(
            flm=self.existing_flm,
            hf_source=self.tmp_path / "missing-hf-source",
            geoquant_root=self.tmp_path,
            geoquant_python=Path(sys.executable),
            binary=SCRIPT,
        )

        runner.discover_inputs(args, runner.ArtifactAction.REUSE)

    def test_export_input_discovery_requires_hf_source(self):
        args = self.args(
            hf_source=self.tmp_path / "missing-hf-source",
            geoquant_root=self.tmp_path,
            geoquant_python=Path(sys.executable),
            binary=SCRIPT,
        )

        with self.assertRaisesRegex(runner.PhaseError, "input discovery.*HF source"):
            runner.discover_inputs(args, runner.ArtifactAction.REGENERATE)

    def test_producer_oserror_names_producer_phase(self):
        self.assert_subprocess_oserror_names_phase("producer export")

    def test_validator_oserror_names_validation_phase(self):
        self.assert_subprocess_oserror_names_phase("strict validation")

    def test_benchmark_oserror_names_benchmark_phase(self):
        self.assert_subprocess_oserror_names_phase("SuperSonic inference")

    def assert_subprocess_oserror_names_phase(self, phase):
        command = ["/missing/executable"]
        with mock.patch.object(
            runner.subprocess,
            "Popen",
            side_effect=OSError("cannot execute"),
        ):
            with self.assertRaisesRegex(runner.PhaseError, rf"{phase}.*cannot execute"):
                runner.run_command(
                    command,
                    cwd=self.tmp_path,
                    timeout=5,
                    phase=phase,
                )

    def test_promotion_oserror_names_artifact_promotion_phase(self):
        args = self.args(regenerate=True)
        with mock.patch.object(runner.os, "getpid", return_value=42), \
                mock.patch.object(runner, "run_command"), \
                mock.patch.object(
                    runner.os,
                    "replace",
                    side_effect=OSError("cross-device failure"),
                ):
            with self.assertRaisesRegex(
                runner.PhaseError,
                "artifact promotion.*cross-device failure",
            ):
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
        self.assertEqual(args.inference_cleanup_grace, 30)
        self.assertFalse(args.regenerate)
        self.assertIsNone(args.regenerate_output)
        self.assertFalse(args.overwrite_artifact)

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
        self.assertEqual(command[command.index("--prompt-source") + 1], "jsonl")
        self.assertEqual(
            command[command.index("--lucebox-jsonl") + 1],
            str(runner.benchmark_prompt_path(args.out_json)),
        )

    def test_writes_portable_single_prompt_for_benchmark(self):
        prompt_path = self.tmp_path / "flm-e2e-prompts.jsonl"

        runner.write_benchmark_prompts(prompt_path)

        self.assertEqual(
            json.loads(prompt_path.read_text()),
            {"id": "flm-first-class-e2e", "prompt": "Hello"},
        )

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
            "flm_virtual_transfer_backend": None,
            "summary": {
                "count": 1,
                "flm_weight_modes": ["INT4 native FLM"],
                "flm_ready_for_decode_count": 1,
                "runtime_engine_ready_count": 1,
                "flm_direct_profiles": [{
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "row_group_int4": 330,
                    "tile_int4_v1": 0,
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
                "runtime_engine_ownership_markers": [{
                    "load_sequence": 1,
                    "source_open_count": 1,
                }],
                "flm_full_output_forbidden_markers": [],
                "flm_direct_profile": {
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "row_group_int4": 330,
                    "tile_int4_v1": 0,
                    "bf16_fallback": 0,
                },
            }],
        }

    def write_report(self, payload):
        path = self.tmp_path / "report.json"
        path.write_text(json.dumps(payload))
        return path

    def assert_report_rejected(self, payload, message, requested_backend=None):
        with self.assertRaisesRegex(runner.PhaseError, message):
            path = self.write_report(payload)
            if requested_backend is None:
                runner.validate_benchmark_report(path)
            else:
                runner.validate_benchmark_report(
                    path,
                    requested_backend=requested_backend,
                )

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

    def test_report_rejects_boolean_row_return_code(self):
        payload = self.valid_report()
        payload["rows"][0]["returncode"] = False

        self.assert_report_rejected(payload, "return code")

    def test_report_rejects_string_row_return_code(self):
        payload = self.valid_report()
        payload["rows"][0]["returncode"] = "0"

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

    def test_report_rejects_missing_runtime_engine_ownership(self):
        payload = self.valid_report()
        payload["rows"][0].pop("runtime_engine_ownership_markers")
        payload["summary"]["runtime_engine_ready_count"] = 0

        self.assert_report_rejected(payload, "runtime engine ownership")

    def test_report_rejects_duplicate_runtime_engine_ownership(self):
        payload = self.valid_report()
        payload["rows"][0]["runtime_engine_ownership_markers"].append({
            "load_sequence": 1,
            "source_open_count": 1,
        })
        payload["summary"]["runtime_engine_ready_count"] = 2

        self.assert_report_rejected(payload, "exactly one runtime engine ownership")

    def test_report_rejects_wrong_runtime_engine_load_sequence(self):
        payload = self.valid_report()
        payload["rows"][0]["runtime_engine_ownership_markers"][0]["load_sequence"] = 2

        self.assert_report_rejected(payload, "load_sequence=1")

    def test_report_rejects_wrong_runtime_engine_source_open_count(self):
        payload = self.valid_report()
        payload["rows"][0]["runtime_engine_ownership_markers"][0]["source_open_count"] = 2

        self.assert_report_rejected(payload, "source_open_count=1")

    def test_report_rejects_missing_full_output_hf_path_gate(self):
        payload = self.valid_report()
        payload["rows"][0].pop("flm_full_output_forbidden_markers")

        self.assert_report_rejected(payload, "full-output HF-path marker evidence")

    def test_report_rejects_full_output_hf_path_markers(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_full_output_forbidden_markers"] = ["config.json"]

        self.assert_report_rejected(payload, "forbidden HF-path markers")

    def test_report_rejects_noncanonical_native_int4_direct_plan_count(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["native_int4"] = 329

        self.assert_report_rejected(payload, "exactly 330 native INT4")

    def test_report_rejects_noncanonical_row_group_direct_plan_count(self):
        payload = self.valid_report()
        profile = payload["rows"][0]["flm_direct_profile"]
        profile["native_int4"] = 329
        profile["row_group_int4"] = 329

        self.assert_report_rejected(payload, "exactly 330 row-group INT4")

    def test_report_rejects_tile_v1_direct_plans(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["tile_int4_v1"] = 1

        self.assert_report_rejected(payload, "tile-v1 INT4")

    def test_report_rejects_inconsistent_native_int4_aggregate(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["native_int4"] = 331

        self.assert_report_rejected(payload, "aggregate native INT4")

    def test_report_rejects_bf16_direct_plan_fallback(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["bf16_fallback"] = 1

        self.assert_report_rejected(payload, "BF16 fallback")

    def test_report_rejects_fractional_bf16_direct_plan_fallback(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["bf16_fallback"] = 0.5
        payload["summary"]["flm_direct_profiles"][0]["bf16_fallback"] = 0.5

        self.assert_report_rejected(payload, "BF16 fallback")

    def test_report_rejects_nonnumeric_bf16_direct_plan_fallback(self):
        payload = self.valid_report()
        payload["rows"][0]["flm_direct_profile"]["bf16_fallback"] = "none"
        payload["summary"]["flm_direct_profiles"][0]["bf16_fallback"] = "none"

        self.assert_report_rejected(payload, "BF16 fallback")

    def test_report_rejects_boolean_summary_direct_profile_field(self):
        payload = self.valid_report()
        payload["summary"]["flm_direct_profiles"][0]["bf16_fallback"] = False

        self.assert_report_rejected(payload, "direct profiles")

    def test_report_rejects_extra_summary_direct_profile(self):
        payload = self.valid_report()
        payload["summary"]["flm_direct_profiles"].append({
            "required": 693,
            "raw_dense": 363,
            "native_int4": 329,
            "row_group_int4": 329,
            "tile_int4_v1": 0,
            "bf16_fallback": 1,
        })

        self.assert_report_rejected(payload, "direct profiles")

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

    def test_report_rejects_nan_transfer_speed(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"]["copy_h2d_gib_s"] = float("nan")

        self.assert_report_rejected(payload, "transfer GiB/s")

    def test_report_rejects_string_transfer_speed(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"]["copy_h2d_gib_s"] = "20.0"

        self.assert_report_rejected(payload, "transfer GiB/s")

    def test_report_rejects_boolean_transfer_speed(self):
        payload = self.valid_report()
        payload["summary"]["flm_load_speed"]["copy_h2d_gib_s"] = True

        self.assert_report_rejected(payload, "transfer GiB/s")

    def test_report_rejects_crossed_transfer_backend_pair(self):
        payload = self.valid_report()
        load_speed = payload["summary"]["flm_load_speed"]
        load_speed["copy_storage_to_device_bytes"] = 17179869184
        load_speed["copy_storage_to_device_gib_s"] = 0.0
        load_speed["copy_h2d_gib_s"] = 20.0
        load_speed["copy_h2d_bytes"] = 0

        self.assert_report_rejected(payload, "matching transfer")

    def test_report_rejects_h2d_only_when_storage_direct_requested(self):
        payload = self.valid_report()
        payload["flm_virtual_transfer_backend"] = "gpu-direct-storage"
        payload["rows"][0]["flm_virtual_transfer_backend"] = "gpu-direct-storage"

        self.assert_report_rejected(
            payload,
            "storage-to-device transfer",
            requested_backend="gpu-direct-storage",
        )

    def test_report_rejects_backend_selector_mismatch(self):
        payload = self.valid_report()
        payload["flm_virtual_transfer_backend"] = "pageable-h2d"

        self.assert_report_rejected(
            payload,
            "backend selector",
            requested_backend="gpu-direct-storage",
        )

    def test_report_accepts_storage_direct_transfer_evidence(self):
        payload = self.valid_report()
        payload["flm_virtual_transfer_backend"] = "gpu-direct-storage"
        payload["rows"][0]["flm_virtual_transfer_backend"] = "gpu-direct-storage"
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
            runner.validate_benchmark_report(
                self.write_report(payload),
                requested_backend="gpu-direct-storage",
            ),
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

        for flag in (
            "--limit",
            "--n-gen",
            "--context-size",
            "--inference-timeout",
            "--inference-cleanup-grace",
        ):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    with mock.patch("sys.stderr", new_callable=io.StringIO):
                        with self.assertRaises(SystemExit):
                            runner.parse_args([flag, value])

    def test_timeout_terminates_and_reaps_process_group_without_sleeping(self):
        process = mock.Mock(pid=4242, returncode=-signal.SIGTERM)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["benchmark"], 9),
            (None, None),
        ]
        with mock.patch.object(
            runner.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(runner.os, "killpg") as killpg:
            with self.assertRaisesRegex(
                runner.PhaseError,
                "SuperSonic inference timed out after 9s",
            ):
                runner.run_command(
                    ["benchmark"],
                    cwd=self.tmp_path,
                    timeout=9,
                    phase="SuperSonic inference",
                )

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.assertEqual(process.communicate.call_count, 2)

    def test_timeout_escalates_and_reaps_stubborn_process_group(self):
        process = mock.Mock(pid=4343, returncode=-signal.SIGKILL)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["benchmark"], 9),
            subprocess.TimeoutExpired(["benchmark"], 5),
            (None, None),
        ]
        with mock.patch.object(
            runner.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(runner.os, "killpg") as killpg:
            with self.assertRaisesRegex(runner.PhaseError, "timed out after 9s"):
                runner.run_command(
                    ["benchmark"],
                    cwd=self.tmp_path,
                    timeout=9,
                    phase="SuperSonic inference",
                )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4343, signal.SIGTERM),
                mock.call(4343, signal.SIGKILL),
            ],
        )
        self.assertEqual(process.communicate.call_count, 3)

    def test_report_read_oserror_names_report_phase(self):
        with self.assertRaisesRegex(runner.PhaseError, "report evidence.*read"):
            runner.validate_benchmark_report(self.tmp_path / "missing.json")

    def test_report_json_error_names_report_phase(self):
        report = self.tmp_path / "malformed.json"
        report.write_text("{not-json")

        with self.assertRaisesRegex(runner.PhaseError, "report evidence.*JSON"):
            runner.validate_benchmark_report(report)

    def test_main_prepares_runs_and_validates_in_order(self):
        artifact = self.tmp_path / "prepared.flm"
        artifact.write_bytes(b"prevalidated")
        digest = runner.artifact_sha256(artifact)
        out_json = self.tmp_path / "benchmark.json"
        payload = self.valid_report()
        order = []

        def prepare(args):
            order.append("prepare")
            return runner.ArtifactPreparation(
                artifact=artifact,
                action="reuse",
                source=artifact,
                destination=artifact,
                before_sha256=digest,
                after_sha256=digest,
            )

        def discover(args, action):
            order.append("discover")
            self.assertIs(action, runner.ArtifactAction.REUSE)

        def run(*args, **kwargs):
            order.append("run")

        def record(path, preparation):
            order.append("record")
            self.assertEqual(path, out_json)
            self.assertEqual(preparation.before_sha256, digest)
            self.assertEqual(preparation.after_sha256, digest)

        def validate(path, *, requested_backend):
            order.append("validate")
            self.assertEqual(path, out_json)
            self.assertIsNone(requested_backend)
            return payload

        with mock.patch.object(runner, "discover_inputs", side_effect=discover), \
                mock.patch.object(runner, "prepare_artifact", side_effect=prepare), \
                mock.patch.object(runner, "run_command", side_effect=run) as run_mock, \
                mock.patch.object(runner, "record_artifact_provenance", side_effect=record), \
                mock.patch.object(runner, "validate_benchmark_report", side_effect=validate), \
                mock.patch.object(runner, "print_summary") as print_summary:
            result = runner.main([
                "--flm", str(artifact),
                "--binary", "/repo/supersonic",
                "--out-json", str(out_json),
                "--limit", "3",
                "--inference-cleanup-grace", "17",
            ])

        self.assertEqual(result, 0)
        self.assertEqual(order, ["discover", "prepare", "run", "record", "validate"])
        self.assertEqual(run_mock.call_args.kwargs, {
            "cwd": runner.ROOT,
            "timeout": 2717,
            "phase": "SuperSonic inference",
        })
        self.assertEqual(
            run_mock.call_args.args[0],
            runner.supersonic_benchmark_command(
                self.args(
                    flm=artifact,
                    out_json=out_json,
                    limit=3,
                    inference_cleanup_grace=17,
                ),
                artifact,
            ),
        )
        print_summary.assert_called_once_with(payload, artifact)


if __name__ == "__main__":
    unittest.main()
