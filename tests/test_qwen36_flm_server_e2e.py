import contextlib
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "tests" / "gfx1100" / "run_qwen36_flm_server_e2e.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("qwen36_flm_server_e2e", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self, *, poll_result=None, pid=4242):
        self.pid = pid
        self.returncode = poll_result
        self.poll_result = poll_result
        self.communicate_calls = []

    def poll(self):
        return self.poll_result

    def communicate(self, timeout=None):
        self.communicate_calls.append(timeout)
        return ("", "")


class FakeHttpResponse:
    def __init__(self, payload, status=200):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class Qwen36FlmServerHarnessTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()
        self.args = SimpleNamespace(
            binary=Path("/repo/target/release/supersonic-serve"),
            flm=Path(
                "/mnt/data/runs/geo-quant/"
                "qwen36-35b-a3b-supersonic-native-int4-current.flm"
            ),
            backend="hip",
            device=0,
            max_context=4096,
            host="127.0.0.1",
            api_key="local",
            no_download=True,
            startup_timeout=900.0,
        )

    def valid_capabilities(self):
        return {
            "model": "qwen3.6-35b-a3b",
            "family": "qwen3.6-moe",
            "backend": "HIP",
            "ready": True,
            "max_context": 4096,
            "chat": True,
            "responses": True,
            "streaming": True,
            "stream_usage": True,
            "tools": True,
            "reasoning": True,
            "scheduler": {
                "active_requests": 0,
                "queued_requests": 0,
                "max_queued_requests": 32,
                "queue_timeout_ms": 30_000,
            },
            "flm": {
                "source": "flm",
                "file": "qwen36-native.flm",
                "architecture_id": 2,
                "model_id": 2,
                "storage_abi_ids": [8],
                "required_weights": 693,
                "raw_dense_weights": 363,
                "native_int4_direct_weights": 330,
                "bf16_fallback_weights": 0,
                "transfer_backend": "pageable-h2d",
                "source_bytes": 8_000_000_000,
                "device_upload_bytes": 7_000_000_000,
                "startup": {
                    "total_seconds": 1.25,
                    "exclusive_components": {
                        "source_open": {
                            "total_seconds": 0.12,
                            "exclusive_phases": {
                                "store_open_seconds": 0.08,
                                "config_seconds": 0.01,
                                "direct_plan_seconds": 0.02,
                            },
                        },
                        "descriptor_seconds": 0.04,
                        "tokenizer_seconds": 0.05,
                    },
                },
                "load_sequence": 1,
                "source_open_count": 1,
                "resident_allocation_count": 42,
                "features": {
                    "plain_prefill_decode": True,
                    "native_dflash_generate": False,
                    "prefix_snapshot": False,
                    "disk_prefix_snapshot": False,
                },
            },
        }

    def valid_metrics_text(self):
        return "\n".join(
            [
                "# TYPE supersonic_ready gauge",
                "supersonic_ready 1",
                "supersonic_active_requests 0",
                "supersonic_queued_requests 0",
                "supersonic_model_loads_total 1",
                "supersonic_flm_native_int4_direct_weights 330",
                "supersonic_flm_bf16_fallback_weights 0",
                "supersonic_flm_source_bytes 8000000000",
                "supersonic_flm_device_upload_bytes 7000000000",
                "supersonic_flm_startup_seconds 1.25",
            ]
        )

    def valid_report(self):
        return {
            "model": "qwen3.6-35b-a3b",
            "source": "flm",
            "load_sequence": 1,
            "native_int4": 330,
            "bf16_fallback": 0,
            "requests": {
                "compat": {
                    "auth": {"unauthorized_status": 401},
                    "models": {"listed": True, "retrieved": True},
                    "tokenizer": {"roundtrip": True},
                    "chat": {"assistant_result": True},
                    "chat_stream": {
                        "saw_delta": True,
                        "saw_terminal": True,
                        "saw_usage": True,
                    },
                    "responses": {"assistant_result": True},
                    "responses_stream": {
                        "saw_delta": True,
                        "saw_completed": True,
                    },
                    "reasoning": {
                        "assistant_result": True,
                        "request_accepted": True,
                        "reasoning_observed": False,
                        "visible_think_tags": False,
                    },
                    "usage_accounting": {
                        "chat_valid": True,
                        "chat_stream_valid": True,
                        "responses_valid": True,
                        "responses_stream_valid": True,
                    },
                    "repeated_request": {"assistant_result": True},
                },
                "agent": {
                    "chat_tool_loop": {"assistant_result": True},
                    "responses_tool_loop": {"assistant_result": True},
                },
            },
            "startup": {
                "ready_seconds": 2.0,
                "total_seconds": 1.25,
                "transfer_backend": "pageable-h2d",
                "source_bytes": 8_000_000_000,
                "device_upload_bytes": 7_000_000_000,
                "source_open_count": 1,
                "resident_allocation_count": 42,
            },
            "throughput": {
                "first_token_seconds": 0.5,
                "prefill_tokens_per_second": 120.0,
                "decode_tokens_per_second": 4.0,
            },
            "cancellation": {
                "aborted_after_first_delta": True,
                "saw_delta": True,
                "scheduler_released": True,
                "active_requests": 0,
                "queued_requests": 0,
                "release_seconds": 0.25,
            },
        }

    def test_server_command_is_first_class_flm_only(self):
        command = self.harness.server_command(self.args, 18765)

        self.assertEqual(
            command,
            [
                "/repo/target/release/supersonic-serve",
                "--flm-file",
                (
                    "/mnt/data/runs/geo-quant/"
                    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
                ),
                "--backend",
                "hip",
                "--device",
                "0",
                "--max-context",
                "4096",
                "--host",
                "127.0.0.1",
                "--port",
                "18765",
                "--api-key",
                "local",
                "--no-download",
            ],
        )
        for forbidden in ("--model", "--model-dir", "--int4", "--q4km", "--q4km-gptq"):
            self.assertNotIn(forbidden, command)

    def test_unused_loopback_port_can_be_rebound(self):
        port = self.harness.allocate_loopback_port()

        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))

    def test_wait_for_ready_accepts_only_ready_payload(self):
        process = FakeProcess()
        payload = {"ready": True, "model": "qwen3.6-35b-a3b"}
        opened = []

        def opener(request, timeout):
            opened.append((request.full_url, request.headers, timeout))
            return FakeHttpResponse(payload)

        result = self.harness.wait_for_ready(
            process,
            "http://127.0.0.1:18765",
            "local",
            timeout=1.0,
            opener=opener,
        )

        self.assertEqual(result, payload)
        self.assertEqual(opened[0][0], "http://127.0.0.1:18765/ready")
        self.assertEqual(opened[0][1]["Authorization"], "Bearer local")

    def test_wait_for_ready_reports_startup_timeout(self):
        process = FakeProcess()

        with self.assertRaisesRegex(self.harness.PhaseError, "startup timed out"):
            self.harness.wait_for_ready(
                process,
                "http://127.0.0.1:18765",
                "local",
                timeout=0.0,
                opener=mock.Mock(side_effect=urllib.error.URLError("refused")),
            )

    def test_wait_for_ready_reports_early_server_failure(self):
        process = FakeProcess(poll_result=17)

        with self.assertRaisesRegex(
            self.harness.PhaseError,
            "readiness failed.*exit 17.*FLM open failed",
        ):
            self.harness.wait_for_ready(
                process,
                "http://127.0.0.1:18765",
                "local",
                timeout=1.0,
                log_tail=lambda: "FLM open failed",
            )

    def test_running_server_starts_new_process_group_and_reaps_on_success(self):
        process = FakeProcess()
        calls = []

        def popen_factory(command, **kwargs):
            calls.append((command, kwargs))
            return process

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(self.harness.os, "killpg") as killpg:
                with self.harness.running_server(
                    self.args,
                    18765,
                    log_path,
                    popen_factory=popen_factory,
                    readiness=lambda *_args, **_kwargs: {"ready": True},
                ) as ready:
                    self.assertTrue(ready["ready"])

        self.assertTrue(calls[0][1]["start_new_session"])
        self.assertEqual(calls[0][1]["stderr"], subprocess.STDOUT)
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(process.communicate_calls, [self.harness.PROCESS_GRACE_SECONDS])

    def test_running_server_reaps_when_readiness_fails(self):
        process = FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(self.harness.os, "killpg") as killpg:
                with self.assertRaisesRegex(self.harness.PhaseError, "not ready"):
                    with self.harness.running_server(
                        self.args,
                        18765,
                        log_path,
                        popen_factory=lambda *_args, **_kwargs: process,
                        readiness=mock.Mock(
                            side_effect=self.harness.PhaseError("not ready")
                        ),
                    ):
                        self.fail("server should not be yielded")

        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(process.communicate_calls, [self.harness.PROCESS_GRACE_SECONDS])

    def test_parse_metrics_accepts_finite_scalar_samples(self):
        metrics = self.harness.parse_prometheus_metrics(self.valid_metrics_text())

        self.assertEqual(metrics["supersonic_model_loads_total"], 1)
        self.assertEqual(
            metrics["supersonic_flm_native_int4_direct_weights"],
            330,
        )
        self.assertEqual(metrics["supersonic_flm_startup_seconds"], 1.25)

    def test_parse_metrics_rejects_duplicate_or_nonfinite_samples(self):
        with self.assertRaisesRegex(self.harness.PhaseError, "duplicate metric"):
            self.harness.parse_prometheus_metrics(
                "supersonic_ready 1\nsupersonic_ready 1\n"
            )
        for value in ("NaN", "+Inf", "-Inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(self.harness.PhaseError, "finite"):
                    self.harness.parse_prometheus_metrics(
                        f"supersonic_ready {value}\n"
                    )

    def test_validate_flm_evidence_accepts_exact_capability_and_metrics(self):
        metrics = self.harness.parse_prometheus_metrics(self.valid_metrics_text())

        snapshot = self.harness.validate_flm_evidence(
            self.valid_capabilities(),
            metrics,
        )

        self.assertEqual(
            snapshot,
            {
                "model": "qwen3.6-35b-a3b",
                "source": "flm",
                "load_sequence": 1,
                "native_int4": 330,
                "bf16_fallback": 0,
                "model_loads_total": 1,
                "startup": self.valid_capabilities()["flm"]["startup"],
                "transfer_backend": "pageable-h2d",
                "source_bytes": 8_000_000_000,
                "device_upload_bytes": 7_000_000_000,
                "source_open_count": 1,
                "resident_allocation_count": 42,
                "scheduler": self.valid_capabilities()["scheduler"],
            },
        )

    def test_validate_flm_evidence_rejects_wrong_or_boolean_counts(self):
        cases = [
            ("load_sequence", 2),
            ("load_sequence", True),
            ("native_int4_direct_weights", 329),
            ("native_int4_direct_weights", True),
            ("bf16_fallback_weights", 1),
            ("bf16_fallback_weights", False),
            ("source_open_count", 2),
            ("resident_allocation_count", True),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                capabilities = self.valid_capabilities()
                capabilities["flm"][field] = value
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_flm_evidence(
                        capabilities,
                        self.harness.parse_prometheus_metrics(
                            self.valid_metrics_text()
                        ),
                    )

    def test_validate_load_invariance_requires_one_unchanged_load(self):
        before = {"load_sequence": 1, "model_loads_total": 1}
        after = {"load_sequence": 1, "model_loads_total": 1}
        self.harness.validate_load_invariance(before, after)

        invalid_pairs = [
            ({"load_sequence": True, "model_loads_total": 1}, after),
            (before, {"load_sequence": 2, "model_loads_total": 1}),
            (before, {"load_sequence": 1, "model_loads_total": 2}),
        ]
        for invalid_before, invalid_after in invalid_pairs:
            with self.subTest(before=invalid_before, after=invalid_after):
                with self.assertRaisesRegex(self.harness.PhaseError, "single-load"):
                    self.harness.validate_load_invariance(
                        invalid_before,
                        invalid_after,
                    )

    def test_parse_smoke_output_requires_one_structured_marker(self):
        payload = {"requests": {"chat": {"assistant_result": True}}}
        stdout = (
            "diagnostic line\n"
            f"{self.harness.SMOKE_JSON_PREFIX}{json.dumps(payload)}\n"
        )

        self.assertEqual(self.harness.parse_smoke_output(stdout, "compat"), payload)

        for invalid in (
            "no structured output",
            (
                f"{self.harness.SMOKE_JSON_PREFIX}{json.dumps(payload)}\n"
                f"{self.harness.SMOKE_JSON_PREFIX}{json.dumps(payload)}\n"
            ),
            f"{self.harness.SMOKE_JSON_PREFIX}not-json\n",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.parse_smoke_output(invalid, "compat")

    def test_validate_report_accepts_complete_structured_evidence(self):
        report = self.valid_report()

        self.assertIs(self.harness.validate_report(report), report)

    def test_validate_report_rejects_boolean_integer_and_nonfinite_numbers(self):
        mutations = [
            (("load_sequence",), True),
            (("native_int4",), True),
            (("bf16_fallback",), False),
            (("startup", "source_bytes"), True),
            (("startup", "total_seconds"), math.nan),
            (("throughput", "first_token_seconds"), math.inf),
            (("throughput", "decode_tokens_per_second"), True),
            (("cancellation", "active_requests"), False),
            (("cancellation", "release_seconds"), -math.inf),
        ]
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                report = self.valid_report()
                target = report
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_report(report)

    def test_validate_report_rejects_missing_protocol_or_release_evidence(self):
        missing_paths = [
            ("requests", "compat", "auth"),
            ("requests", "compat", "chat"),
            ("requests", "compat", "chat_stream"),
            ("requests", "compat", "responses"),
            ("requests", "compat", "responses_stream"),
            ("requests", "compat", "reasoning"),
            ("requests", "compat", "usage_accounting"),
            ("requests", "compat", "repeated_request"),
            ("requests", "agent", "chat_tool_loop"),
            ("requests", "agent", "responses_tool_loop"),
            ("cancellation", "scheduler_released"),
        ]
        for path in missing_paths:
            with self.subTest(path=path):
                report = self.valid_report()
                target = report
                for key in path[:-1]:
                    target = target[key]
                target.pop(path[-1])
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_report(report)

    def test_sdk_scripts_parse_as_javascript(self):
        scripts = [
            ROOT / "scripts" / "openai_compat_smoke.mjs",
            ROOT / "scripts" / "openai_agent_tool_smoke.mjs",
        ]
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["node", "--check", str(script)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_script_declares_both_tool_loops_and_stream_abort(self):
        source = (ROOT / "scripts" / "openai_agent_tool_smoke.mjs").read_text(
            encoding="utf-8"
        )

        for contract in (
            "client.chat.completions.create",
            "client.responses.create",
            "function_call_output",
            "controller.abort",
            "Your entire response must be exactly one call",
            "cancellation_release",
            "raw",
            "scheduler_released",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, source)

    def test_agent_script_checks_real_release_before_model_dependent_tool_calls(self):
        source = (ROOT / "scripts" / "openai_agent_tool_smoke.mjs").read_text(
            encoding="utf-8"
        )

        self.assertLess(
            source.index("const cancellationStream"),
            source.index("const chatStarted"),
        )


if __name__ == "__main__":
    unittest.main()
