import contextlib
import copy
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.openai_sdk_fixture import (
    API_KEY,
    CHAT_CALL_ID,
    CHAT_TOOL,
    CODING_PROMPT,
    MODEL,
    RESPONSE_CALL_ID,
    RESPONSE_ID,
    RESPONSES_TOOL,
    TOOL_OUTPUT,
    openai_sdk_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "tests" / "gfx1100" / "run_qwen36_flm_server_e2e.py"
SDK_DIR = ROOT / "target" / "openai-sdk-smoke"


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
    @classmethod
    def setUpClass(cls):
        probe = subprocess.run(
            ["npm", "list", "openai@6.49.0", "--depth=0", "--json"],
            cwd=SDK_DIR if SDK_DIR.is_dir() else ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:
            SDK_DIR.mkdir(parents=True, exist_ok=True)
            install = subprocess.run(
                [
                    "npm",
                    "install",
                    "--no-audit",
                    "--no-fund",
                    "--prefix",
                    str(SDK_DIR),
                    "openai@6.49.0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
            )
            if install.returncode != 0:
                raise RuntimeError(f"failed to install pinned OpenAI SDK: {install.stderr}")

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

    def run_sdk_fixture_script(self, script_name, base_url):
        env = os.environ.copy()
        env.update(
            {
                "SUPERSONIC_BASE_URL": base_url,
                "SUPERSONIC_API_KEY": API_KEY,
                "SUPERSONIC_SMOKE_MODEL": MODEL,
                "SUPERSONIC_REQUEST_TIMEOUT_MS": "5000",
            }
        )
        return subprocess.run(
            ["node", str(ROOT / "scripts" / script_name)],
            cwd=SDK_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )

    def valid_capabilities(self):
        return {
            "model": "qwen3.6-35b-a3b",
            "family": "qwen3.6-moe",
            "backend": "HIP",
            "ready": True,
            "max_context": 4096,
            "endpoints": [
                "/v1/models",
                "/v1/models/{model}",
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/tokenize",
                "/v1/detokenize",
                "/v1/responses",
                "/health",
                "/v1/health",
                "/ready",
                "/v1/ready",
                "/v1/capabilities",
                "/metrics",
            ],
            "chat": True,
            "completions": True,
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
            "prefix_cache": {
                "enabled": False,
                "dir": "",
                "min_tokens": 128,
                "max_entries": 1,
                "max_bytes": 1_000_000,
                "resident_bytes": 0,
                "entries": 0,
                "hits": 0,
                "misses": 0,
                "cached_tokens": 0,
                "evictions": 0,
                "disk_writes": 0,
                "disk_reads": 0,
                "restore_failures": 0,
                "admission_skips": 0,
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
                "startup_seconds": 1.25,
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

    def valid_health(self):
        capabilities = self.valid_capabilities()
        return {
            "status": "ok",
            "ready": True,
            "model": "qwen3.6-35b-a3b",
            "max_context": 4096,
            "active_requests": 0,
            "queued_requests": 0,
            "max_queued_requests": 32,
            "prefix_cache_entries": 0,
            "flm": copy.deepcopy(capabilities["flm"]),
        }

    def valid_metrics_text(self):
        return "\n".join(
            [
                "# TYPE supersonic_ready gauge",
                "supersonic_ready 1",
                "supersonic_active_requests 0",
                "supersonic_queued_requests 0",
                "supersonic_generation_active 0",
                "supersonic_generation_queued 0",
                "supersonic_max_queued_requests 32",
                "supersonic_queue_timeout_ms 30000",
                "supersonic_max_context 4096",
                "supersonic_prefix_cache_enabled 0",
                "supersonic_prefix_cache_entries 0",
                "supersonic_prefix_cache_resident_bytes 0",
                "supersonic_prefix_cache_max_bytes 1000000",
                "supersonic_prefix_cache_hits 0",
                "supersonic_prefix_cache_misses 0",
                "supersonic_prefix_cache_cached_tokens 0",
                "supersonic_prefix_cache_evictions 0",
                "supersonic_prefix_cache_disk_writes 0",
                "supersonic_prefix_cache_disk_reads 0",
                "supersonic_prefix_cache_restore_failures 0",
                "supersonic_prefix_cache_admission_skips 0",
                "supersonic_dflash_last_rounds 0",
                "supersonic_dflash_last_accepted_total 0",
                "supersonic_dflash_last_decode_ms 0",
                "supersonic_model_loads_total 1",
                "supersonic_flm_native_int4_direct_weights 330",
                "supersonic_flm_bf16_fallback_weights 0",
                "supersonic_flm_source_bytes 8000000000",
                "supersonic_flm_device_upload_bytes 7000000000",
                "supersonic_flm_startup_seconds 1.25",
            ]
        )

    def valid_usage(self):
        return {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }

    def valid_compat_report(self):
        usage = self.valid_usage()
        return {
            "transport": {
                "auth": {
                    "missing_key": {
                        "status": 401,
                        "error_type": "authentication_error",
                    },
                    "wrong_key": {
                        "status": 401,
                        "error_type": "authentication_error",
                    },
                    "protected_routes": {
                        path: {
                            "status": 401,
                            "error_type": "authentication_error",
                        }
                        for path in (
                            "/health",
                            "/ready",
                            "/metrics",
                            "/v1/capabilities",
                        )
                    },
                },
                "models": {"listed": True, "retrieved": True},
                "tokenizer": {"roundtrip": True, "token_count": 2},
                "chat": {"received": True},
                "chat_stream": {
                    "received_delta": True,
                    "received_terminal": True,
                    "received_usage": True,
                },
                "completions": {"received": True},
                "responses": {"received": True, "stored_roundtrip": True},
                "responses_stream": {
                    "received_delta": True,
                    "received_terminal": True,
                    "received_usage": True,
                },
                "reasoning": {"request_accepted": True},
                "repeated_request": {"received": True},
            },
            "semantic_quality": {
                "chat": {
                    "expected": "hello",
                    "actual": "hello",
                    "finish_reason": "stop",
                    "passed": True,
                },
                "chat_stream": {
                    "expected": "hello",
                    "actual": "hello",
                    "finish_reason": "stop",
                    "passed": True,
                    "terminal_count": 1,
                    "terminal_last_before_usage": True,
                    "usage_last": True,
                },
                "completions": {
                    "expected": "hello",
                    "actual": "hello",
                    "finish_reason": "stop",
                    "passed": True,
                },
                "responses": {
                    "expected": "hello",
                    "actual": "hello",
                    "status": "completed",
                    "stored_roundtrip": True,
                    "passed": True,
                },
                "responses_stream": {
                    "expected": "hello",
                    "actual": "hello",
                    "status": "completed",
                    "terminal_count": 1,
                    "terminal_last": True,
                    "passed": True,
                },
                "reasoning": {
                    "accepted": True,
                    "observed": True,
                    "visible_think_tags": False,
                    "passed": True,
                },
                "repeated_request": {
                    "expected": "ready",
                    "actual": "ready",
                    "finish_reason": "stop",
                    "passed": True,
                },
                "passed": True,
            },
            "usage": {
                section: dict(usage)
                for section in (
                    "chat",
                    "chat_stream",
                    "completions",
                    "responses",
                    "responses_stream",
                    "repeated_request",
                )
            },
            "throughput": {
                "first_token_seconds": 0.5,
                "prefill_tokens_per_second": 6.0,
                "decode_tokens_per_second": 2.0,
            },
        }

    def valid_cancellation(self):
        return {
            "nonterminal_delta": True,
            "abort_closed": True,
            "before": {
                "active_requests": 1,
                "queued_requests": 1,
                "model_loads_total": 1,
                "metric_active_requests": 1,
                "metric_queued_requests": 1,
            },
            "after": {
                "active_requests": 0,
                "queued_requests": 0,
                "model_loads_total": 1,
                "metric_active_requests": 0,
                "metric_queued_requests": 0,
            },
            "queued_request_completed": True,
            "release_seconds": 0.25,
        }

    def valid_agent_report(self):
        return {
            "requests": {
                "chat_tool_loop": {
                    "call_count": 1,
                    "valid_tool_call": True,
                    "call_id": "call_chat",
                    "tool_name": "read_source_file",
                    "arguments": {"path": "src/lib.rs"},
                    "finish_reason": "tool_calls",
                    "suffix_content": "",
                    "continuation": {
                        "text": "file read",
                        "finish_reason": "stop",
                        "tool_call_count": 0,
                    },
                    "elapsed_seconds": 0.5,
                },
                "responses_tool_loop": {
                    "call_count": 1,
                    "valid_tool_call": True,
                    "call_id": "call_response",
                    "tool_name": "read_source_file",
                    "arguments": {"path": "src/lib.rs"},
                    "status": "completed",
                    "suffix_content": "",
                    "continuation": {
                        "text": "file read",
                        "status": "completed",
                        "tool_call_count": 0,
                    },
                    "elapsed_seconds": 0.5,
                },
            },
            "cancellation": self.valid_cancellation(),
        }

    def valid_report(self):
        return {
            "model": "qwen3.6-35b-a3b",
            "source": "flm",
            "load_sequence": 1,
            "native_int4": 330,
            "bf16_fallback": 0,
            "requests": {
                "compat": self.valid_compat_report(),
                "agent": self.valid_agent_report(),
            },
            "startup": {
                "ready_seconds": 2.0,
                "total_seconds": 1.25,
                "transfer_backend": "pageable-h2d",
                "source_bytes": 8_000_000_000,
                "device_upload_bytes": 7_000_000_000,
                "source_open_count": 1,
                "resident_allocation_count": 42,
                "exclusive_components": {},
                "provenance": {
                    "artifact": {
                        "path": "/tmp/qwen.flm",
                        "sha256": "a" * 64,
                        "size_bytes": 8_000_000_000,
                    },
                    "sdk": {
                        "package": "openai",
                        "version": "6.49.0",
                    },
                },
            },
            "throughput": self.valid_compat_report()["throughput"],
            "cancellation": self.valid_cancellation(),
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
                with mock.patch.object(
                    self.harness,
                    "_process_group_exists",
                    side_effect=[True, False],
                ):
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
                with mock.patch.object(
                    self.harness,
                    "_process_group_exists",
                    side_effect=[True, False],
                ):
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

    def wait_pid_file(self, path):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists() and path.read_text().strip():
                return int(path.read_text().strip())
            time.sleep(0.02)
        self.fail(f"timed out waiting for pid file {path}")

    def assert_process_gone(self, pid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return
            time.sleep(0.02)
        self.fail(f"process {pid} survived cleanup")

    def test_process_group_cleanup_kills_descendant_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys;"
                        "p=subprocess.Popen([sys.executable,'-c',"
                        "'import time;time.sleep(60)'],"
                        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                        f"open({str(pid_file)!r},'w').write(str(p.pid))"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            child_pid = self.wait_pid_file(pid_file)
            leader.wait(timeout=5)
            with mock.patch.object(self.harness, "PROCESS_GRACE_SECONDS", 0.2):
                self.harness._terminate_and_reap_process_group(leader)

        try:
            self.assert_process_gone(child_pid)
        finally:
            if Path(f"/proc/{child_pid}").exists():
                os.kill(child_pid, signal.SIGKILL)

    def test_process_group_cleanup_sigkills_resistant_leader_and_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            child_file = Path(tmp) / "child.pid"
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import signal,subprocess,sys,time;"
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                        "p=subprocess.Popen([sys.executable,'-c',"
                        "'import signal,time;"
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                        "time.sleep(60)'],"
                        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                        f"open({str(child_file)!r},'w').write(str(p.pid));"
                        "time.sleep(60)"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            child_pid = self.wait_pid_file(child_file)
            with mock.patch.object(self.harness, "PROCESS_GRACE_SECONDS", 0.2):
                self.harness._terminate_and_reap_process_group(leader)

        try:
            self.assert_process_gone(leader.pid)
            self.assert_process_gone(child_pid)
        finally:
            for pid in (leader.pid, child_pid):
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, signal.SIGKILL)

    def run_process_with_detached_child(self, *, exit_code):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            command = [
                sys.executable,
                "-c",
                (
                    "import subprocess,sys;"
                    "p=subprocess.Popen([sys.executable,'-c',"
                    "'import time;time.sleep(60)'],"
                    "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                    "stderr=subprocess.DEVNULL);"
                    f"open({str(pid_file)!r},'w').write(str(p.pid));"
                    f"sys.exit({exit_code})"
                ),
            ]
            with mock.patch.object(self.harness, "PROCESS_GRACE_SECONDS", 0.2):
                result = self.harness.run_process(
                    command,
                    cwd=Path(tmp),
                    env=None,
                    timeout=5,
                    phase="child cleanup probe",
                    check=False,
                )
            child_pid = self.wait_pid_file(pid_file)
        try:
            self.assert_process_gone(child_pid)
        finally:
            if Path(f"/proc/{child_pid}").exists():
                os.kill(child_pid, signal.SIGKILL)
        return result

    def test_run_process_reaps_child_after_zero_exit(self):
        result = self.run_process_with_detached_child(exit_code=0)
        self.assertEqual(result.returncode, 0)

    def test_run_process_reaps_child_after_nonzero_exit(self):
        result = self.run_process_with_detached_child(exit_code=17)
        self.assertEqual(result.returncode, 17)

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

    def test_sdk_probe_install_and_verification_are_bounded_and_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                node="node",
                npm="npm",
                openai_sdk_dir=Path(tmp),
                sdk_probe_timeout=7.0,
                sdk_install_timeout=11.0,
            )
            results = [
                subprocess.CompletedProcess(
                    [],
                    1,
                    '{"dependencies":{"openai":{"version":"6.48.0"}}}',
                    "",
                ),
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess(
                    [],
                    0,
                    '{"dependencies":{"openai":{"version":"6.49.0"}}}',
                    "",
                ),
            ]
            with mock.patch.object(
                self.harness,
                "run_process",
                side_effect=results,
            ) as run_process:
                with mock.patch.object(
                    self.harness.subprocess,
                    "run",
                    side_effect=AssertionError("unbounded subprocess.run"),
                ):
                    sdk = self.harness.ensure_openai_sdk(args)

        self.assertEqual(
            sdk,
            {
                "directory": Path(tmp),
                "package": "openai",
                "version": "6.49.0",
            },
        )
        self.assertEqual(
            [call.kwargs["timeout"] for call in run_process.call_args_list],
            [7.0, 11.0, 7.0],
        )
        install_command = run_process.call_args_list[1].args[0]
        self.assertIn("openai@6.49.0", install_command)

    def test_run_supersedes_stale_success_with_phase_failure_and_final_evidence(self):
        capabilities = self.valid_capabilities()
        metrics = self.harness.parse_prometheus_metrics(self.valid_metrics_text())
        before = self.harness.validate_flm_evidence(capabilities, metrics)

        @contextlib.contextmanager
        def fake_server(*_args, **_kwargs):
            yield {"ready": True}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            flm = tmp_path / "model.flm"
            flm.write_bytes(b"flm\n")
            out_json = tmp_path / "result.json"
            out_json.write_text('{"status":"stale-success"}\n')
            args = SimpleNamespace(
                binary=tmp_path / "supersonic-serve",
                flm=flm,
                backend="hip",
                device=0,
                max_context=4096,
                host="127.0.0.1",
                api_key="local",
                no_download=True,
                startup_timeout=1.0,
                request_timeout=1.0,
                sdk_probe_timeout=1.0,
                sdk_install_timeout=1.0,
                openai_sdk_dir=tmp_path / "sdk",
                node="node",
                npm="npm",
                out_json=out_json,
            )
            health = self.valid_health()
            with mock.patch.object(self.harness, "discover_inputs"):
                with mock.patch.object(
                    self.harness,
                    "ensure_openai_sdk",
                    return_value={
                        "directory": tmp_path / "sdk",
                        "package": "openai",
                        "version": "6.49.0",
                    },
                ):
                    with mock.patch.object(
                        self.harness,
                        "running_server",
                        fake_server,
                    ):
                        with mock.patch.object(
                            self.harness,
                            "validate_flm_evidence",
                            side_effect=[before, before],
                        ):
                            with mock.patch.object(
                                self.harness,
                                "fetch_json",
                                side_effect=lambda _url, path, _key: (
                                    health if path == "/health" else capabilities
                                ),
                            ):
                                with mock.patch.object(
                                    self.harness,
                                    "fetch_metrics",
                                    return_value=metrics,
                                ):
                                    with mock.patch.object(
                                        self.harness,
                                        "run_sdk_smoke",
                                        side_effect=[
                                            self.valid_compat_report(),
                                            self.harness.PhaseError(
                                                "invalid real tool output"
                                            ),
                                        ],
                                    ):
                                        with self.assertRaisesRegex(
                                            self.harness.PhaseError,
                                            "invalid real tool output",
                                        ):
                                            self.harness.run(args)

            failure = json.loads(out_json.read_text())

        self.assertEqual(
            set(failure),
            {
                "schema_version",
                "status",
                "phase",
                "error",
                "provenance",
                "completed",
                "final",
            },
        )
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["phase"], "agent")
        self.assertIn("invalid real tool output", failure["error"]["message"])
        self.assertEqual(failure["provenance"]["sdk"]["version"], "6.49.0")
        self.assertEqual(len(failure["provenance"]["artifact"]["sha256"]), 64)
        self.assertTrue(failure["completed"]["compat"]["semantic_quality"]["passed"])
        self.assertEqual(failure["final"]["health"], health)
        self.assertEqual(failure["final"]["metrics"], metrics)
        self.assertTrue(failure["final"]["load_invariance"]["passed"])

    def valid_failure_report(self):
        metrics = self.harness.parse_prometheus_metrics(self.valid_metrics_text())
        before = self.harness.validate_flm_evidence(
            self.valid_capabilities(),
            metrics,
        )
        compat = self.valid_compat_report()
        compat["semantic_quality"]["chat"]["actual"] = "wrong"
        compat["semantic_quality"]["chat"]["passed"] = False
        compat["semantic_quality"]["passed"] = False
        agent = {
            "requests": {},
            "cancellation": self.valid_agent_report()["cancellation"],
            "failure": {
                "phase": "chat_tool_loop",
                "message": "raw model output was not a tool call",
                "raw": {"content": '{"path":"src/lib.rs"} trailing'},
            },
        }
        return {
            "schema_version": 1,
            "status": "failed",
            "phase": "compat_semantic+agent",
            "error": {
                "type": "PhaseError",
                "message": "compat and agent failed",
            },
            "provenance": {
                "artifact": {
                    "path": "/tmp/model.flm",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                },
                "sdk": {"package": "openai", "version": "6.49.0"},
            },
            "completed": {
                "initial_evidence": before,
                "compat": compat,
                "agent": agent,
                "phase_failures": [
                    {"phase": "compat_semantic", "message": "chat was wrong"},
                    {"phase": "agent", "message": "tool call was malformed"},
                ],
            },
            "final": {
                "health": self.valid_health(),
                "capabilities": self.valid_capabilities(),
                "metrics": metrics,
                "flm_evidence": before,
                "load_invariance": {"passed": True, "error": None},
                "collection_errors": [],
            },
        }

    def test_validate_failure_report_accepts_exact_partial_protocol_evidence(self):
        report = self.valid_failure_report()
        self.assertIs(self.harness.validate_failure_report(report), report)

    def test_validate_failure_report_accepts_unclassified_protocol_exception(self):
        report = self.valid_failure_report()
        report["phase"] = "protocol"
        report["completed"] = {
            "initial_evidence": report["completed"]["initial_evidence"]
        }

        self.assertIs(self.harness.validate_failure_report(report), report)

    def test_validate_failure_report_accepts_final_flm_crosscheck_failure(self):
        report = self.valid_failure_report()
        report["phase"] = "final_evidence"
        report["completed"].pop("phase_failures")
        report["completed"]["compat"]["semantic_quality"]["chat"]["actual"] = "hello"
        report["completed"]["compat"]["semantic_quality"]["chat"]["passed"] = True
        report["completed"]["compat"]["semantic_quality"]["passed"] = True
        report["completed"]["agent"] = self.valid_agent_report()
        report["final"]["flm_evidence"] = None
        report["final"]["load_invariance"] = {
            "passed": False,
            "error": "FLM evidence unavailable",
        }
        report["final"]["collection_errors"] = [
            "FLM evidence: model source mismatch"
        ]

        self.assertIs(self.harness.validate_failure_report(report), report)

    def test_validate_failure_report_rejects_nested_schema_and_type_mutations(self):
        mutations = [
            lambda report: report["error"].pop("type"),
            lambda report: report["error"].update({"extra": True}),
            lambda report: report["provenance"]["artifact"].update(
                {"size_bytes": True}
            ),
            lambda report: report["completed"]["phase_failures"].append(
                {"phase": "agent", "message": "duplicate"}
            ),
            lambda report: report["final"]["metrics"].update(
                {"supersonic_ready": math.nan}
            ),
            lambda report: report["final"]["load_invariance"].update(
                {"passed": "yes"}
            ),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = self.valid_failure_report()
                mutation(report)
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_failure_report(report)

    def test_validate_failure_report_rejects_partial_semantic_contradictions(self):
        def make_all_semantics_pass(report):
            chat = report["completed"]["compat"]["semantic_quality"]["chat"]
            chat["actual"] = "hello"
            chat["passed"] = True

        mutations = [
            lambda report: report["completed"]["compat"]["semantic_quality"][
                "chat"
            ].update({"expected": "wrong"}),
            lambda report: report["completed"]["compat"]["semantic_quality"][
                "chat"
            ].update({"finish_reason": "mystery"}),
            lambda report: report["completed"]["compat"]["semantic_quality"][
                "responses"
            ].update({"status": "mystery"}),
            lambda report: report["completed"]["compat"]["semantic_quality"][
                "chat"
            ].update({"passed": True}),
            lambda report: (
                make_all_semantics_pass(report),
                report["completed"]["compat"]["semantic_quality"]["chat"].update(
                    {"passed": False}
                ),
            ),
            lambda report: report["completed"]["compat"]["semantic_quality"].update(
                {"passed": True}
            ),
            lambda report: (
                make_all_semantics_pass(report),
                report["completed"]["compat"]["semantic_quality"].update(
                    {"passed": False}
                ),
            ),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = self.valid_failure_report()
                mutation(report)
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_failure_report(report)

    def test_validate_failure_report_rejects_extra_final_evidence_keys(self):
        mutations = [
            lambda report: report["final"]["health"].update({"extra": 0}),
            lambda report: report["final"]["capabilities"].update({"extra": True}),
            lambda report: report["final"]["metrics"].update({"extra_metric": 0}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = self.valid_failure_report()
                mutation(report)
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_failure_report(report)

    def test_validate_failure_report_rejects_phase_inconsistent_evidence(self):
        cases = [
            ("sdk", lambda report: report["provenance"].update({"sdk": None})),
            (
                "inputs",
                lambda report: report["completed"].update(
                    {"initial_evidence": report["final"]["flm_evidence"]}
                ),
            ),
            (
                "compat_semantic+agent",
                lambda report: report["completed"]["phase_failures"].reverse(),
            ),
            (
                "compat_semantic+unknown",
                lambda _report: None,
            ),
        ]
        for phase, mutation in cases:
            with self.subTest(phase=phase):
                report = self.valid_failure_report()
                report["phase"] = phase
                mutation(report)
                with self.assertRaises(self.harness.PhaseError):
                    self.harness.validate_failure_report(report)

    def test_protocol_phases_continue_after_semantic_failure(self):
        args = SimpleNamespace()
        compat = self.valid_compat_report()
        agent = self.valid_agent_report()
        with mock.patch.object(
            self.harness,
            "run_sdk_smoke",
            side_effect=[compat, agent],
        ) as run_sdk_smoke:
            with mock.patch.object(
                self.harness,
                "validate_compat_report",
                side_effect=self.harness.PhaseError("semantic canary failed"),
            ):
                result = self.harness.run_protocol_phases(
                    args,
                    Path("/tmp/sdk"),
                    "http://127.0.0.1:1234",
                )

        self.assertEqual(run_sdk_smoke.call_count, 2)
        self.assertIs(result["compat"], compat)
        self.assertIs(result["agent"], agent)
        self.assertEqual(
            result["failures"],
            [
                {
                    "phase": "compat_semantic",
                    "message": "semantic canary failed",
                }
            ],
        )

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
            ("requests", "compat", "transport", "auth"),
            ("requests", "compat", "semantic_quality", "chat"),
            ("requests", "compat", "semantic_quality", "chat_stream"),
            ("requests", "compat", "semantic_quality", "completions"),
            ("requests", "compat", "semantic_quality", "responses"),
            ("requests", "compat", "semantic_quality", "responses_stream"),
            ("requests", "compat", "semantic_quality", "reasoning"),
            ("requests", "compat", "usage", "completions"),
            ("requests", "compat", "semantic_quality", "repeated_request"),
            ("requests", "agent", "requests", "chat_tool_loop"),
            ("requests", "agent", "requests", "responses_tool_loop"),
            ("cancellation", "before"),
            ("startup", "provenance"),
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
            "queued_request_completed",
            "abort_closed",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, source)

    def test_agent_script_checks_real_release_before_model_dependent_tool_calls(self):
        source = (ROOT / "scripts" / "openai_agent_tool_smoke.mjs").read_text(
            encoding="utf-8"
        )

        self.assertLess(
            source.index("report.cancellation = await cancellationGate()"),
            source.index("report.requests.chat_tool_loop = await chatToolLoop()"),
        )

    def test_compat_script_executes_exact_semantics_with_official_sdk(self):
        source = (ROOT / "scripts" / "openai_compat_smoke.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn("wrongKeyClient.models.list()", source)

        with openai_sdk_fixture() as (base_url, _state):
            result = self.run_sdk_fixture_script(
                "openai_compat_smoke.mjs",
                base_url,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.harness.parse_smoke_output(result.stdout, "compat fixture")
        self.assertEqual(
            set(report),
            {"transport", "semantic_quality", "usage", "throughput"},
        )
        semantics = report["semantic_quality"]
        for section in ("chat", "chat_stream", "completions", "responses", "responses_stream"):
            with self.subTest(section=section):
                self.assertEqual(semantics[section]["expected"], "hello")
                self.assertEqual(semantics[section]["actual"], "hello")
                self.assertTrue(semantics[section]["passed"])
        self.assertEqual(semantics["chat"]["finish_reason"], "stop")
        self.assertEqual(semantics["chat_stream"]["finish_reason"], "stop")
        self.assertEqual(semantics["chat_stream"]["terminal_count"], 1)
        self.assertTrue(semantics["chat_stream"]["terminal_last_before_usage"])
        self.assertTrue(semantics["chat_stream"]["usage_last"])
        self.assertEqual(semantics["responses"]["status"], "completed")
        self.assertTrue(semantics["responses"]["stored_roundtrip"])
        self.assertEqual(semantics["responses_stream"]["status"], "completed")
        self.assertEqual(semantics["responses_stream"]["terminal_count"], 1)
        self.assertTrue(semantics["responses_stream"]["terminal_last"])
        self.assertTrue(semantics["reasoning"]["accepted"])
        self.assertTrue(semantics["reasoning"]["observed"])

    def test_agent_script_executes_exact_tool_loops_and_contention_abort(self):
        with openai_sdk_fixture() as (base_url, state):
            result = self.run_sdk_fixture_script(
                "openai_agent_tool_smoke.mjs",
                base_url,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.harness.parse_smoke_output(result.stdout, "agent fixture")
        chat = report["requests"]["chat_tool_loop"]
        self.assertEqual(chat["call_count"], 1)
        self.assertEqual(chat["call_id"], CHAT_CALL_ID)
        self.assertEqual(chat["tool_name"], "read_source_file")
        self.assertEqual(chat["arguments"], {"path": "src/lib.rs"})
        self.assertEqual(chat["finish_reason"], "tool_calls")
        self.assertEqual(chat["suffix_content"], "")
        self.assertEqual(chat["continuation"]["text"], "file read")
        self.assertEqual(chat["continuation"]["finish_reason"], "stop")
        self.assertEqual(chat["continuation"]["tool_call_count"], 0)

        responses = report["requests"]["responses_tool_loop"]
        self.assertEqual(responses["call_count"], 1)
        self.assertEqual(responses["call_id"], RESPONSE_CALL_ID)
        self.assertEqual(responses["tool_name"], "read_source_file")
        self.assertEqual(responses["arguments"], {"path": "src/lib.rs"})
        self.assertEqual(responses["status"], "completed")
        self.assertEqual(responses["suffix_content"], "")
        self.assertEqual(responses["continuation"]["text"], "file read")
        self.assertEqual(responses["continuation"]["status"], "completed")
        self.assertEqual(responses["continuation"]["tool_call_count"], 0)

        cancellation = report["cancellation"]
        self.assertTrue(cancellation["nonterminal_delta"])
        self.assertTrue(cancellation["abort_closed"])
        self.assertEqual(cancellation["before"]["active_requests"], 1)
        self.assertEqual(cancellation["before"]["queued_requests"], 1)
        self.assertEqual(cancellation["before"]["model_loads_total"], 1)
        self.assertEqual(cancellation["after"]["active_requests"], 0)
        self.assertEqual(cancellation["after"]["queued_requests"], 0)
        self.assertEqual(cancellation["after"]["model_loads_total"], 1)
        self.assertTrue(cancellation["queued_request_completed"])

        tool_output = json.dumps(
            {
                "path": "src/lib.rs",
                "contents": "pub fn protocol_ready() -> bool { true }\n",
            },
            separators=(",", ":"),
        )
        chat_bodies = [
            request["body"]
            for request in state.requests
            if request["path"] == "/v1/chat/completions"
            and any(
                "exactly one call to read_source_file" in str(message.get("content"))
                for message in request["body"].get("messages", [])
            )
        ]
        self.assertEqual(len(chat_bodies), 2)
        self.assertEqual(
            chat_bodies[0],
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": CODING_PROMPT}],
                "tools": [CHAT_TOOL],
                "tool_choice": "auto",
                "max_completion_tokens": 128,
                "temperature": 0,
            },
        )
        self.assertEqual(
            chat_bodies[1],
            {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": CODING_PROMPT},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": CHAT_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": "read_source_file",
                                    "arguments": '{"path":"src/lib.rs"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": CHAT_CALL_ID,
                        "content": tool_output,
                    },
                ],
                "tools": [CHAT_TOOL],
                "tool_choice": "auto",
                "max_completion_tokens": 64,
                "temperature": 0,
            },
        )
        responses_bodies = [
            request["body"]
            for request in state.requests
            if request["path"] == "/v1/responses"
            and (
                request["body"].get("previous_response_id") is not None
                or "exactly one call to read_source_file"
                in str(request["body"].get("input"))
            )
        ]
        self.assertEqual(len(responses_bodies), 2)
        self.assertEqual(
            responses_bodies,
            [
                {
                    "model": MODEL,
                    "input": CODING_PROMPT,
                    "tools": [RESPONSES_TOOL],
                    "tool_choice": "auto",
                    "max_output_tokens": 128,
                    "temperature": 0,
                },
                {
                    "model": MODEL,
                    "previous_response_id": RESPONSE_ID,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": RESPONSE_CALL_ID,
                            "output": tool_output,
                        }
                    ],
                    "tools": [RESPONSES_TOOL],
                    "tool_choice": "auto",
                    "max_output_tokens": 64,
                    "temperature": 0,
                },
            ],
        )

    def test_agent_fixture_rejects_corrupted_continuation_payloads(self):
        def mutate(case):
            def apply(path, body):
                if path == "/v1/chat/completions":
                    messages = body.get("messages", [])
                    if (
                        case == "chat_initial_extra"
                        and len(messages) == 1
                        and messages[0].get("content") == CODING_PROMPT
                    ):
                        body["unexpected"] = True
                    elif case == "chat_wrong_assistant_id" and len(messages) == 3:
                        body["messages"][1]["tool_calls"][0]["id"] = "wrong"
                    elif case == "chat_wrong_id" and len(messages) == 3:
                        body["messages"][2]["tool_call_id"] = "wrong"
                    elif case == "chat_wrong_output" and len(messages) == 3:
                        body["messages"][2]["content"] = "wrong"
                    elif case == "chat_extra" and len(messages) == 3:
                        body["messages"].append({"role": "user", "content": "extra"})
                if path == "/v1/responses":
                    if (
                        case == "responses_initial_extra"
                        and body.get("input") == CODING_PROMPT
                    ):
                        body["unexpected"] = True
                    elif case == "responses_wrong_previous_id" and body.get(
                        "previous_response_id"
                    ):
                        body["previous_response_id"] = "resp_wrong"
                    elif case == "responses_wrong_call_id" and body.get(
                        "previous_response_id"
                    ):
                        body["input"][0]["call_id"] = "call_wrong"
                    elif case == "responses_wrong_output" and body.get(
                        "previous_response_id"
                    ):
                        body["input"][0]["output"] = "wrong"
                    elif case == "responses_extra" and body.get(
                        "previous_response_id"
                    ):
                        body["input"].append(dict(body["input"][0]))
                    elif case == "responses_missing_correlation" and body.get(
                        "previous_response_id"
                    ):
                        body.pop("previous_response_id")
                return body

            return apply

        for case in (
            "chat_initial_extra",
            "chat_wrong_assistant_id",
            "chat_wrong_id",
            "chat_wrong_output",
            "chat_extra",
            "responses_initial_extra",
            "responses_wrong_previous_id",
            "responses_wrong_call_id",
            "responses_wrong_output",
            "responses_extra",
            "responses_missing_correlation",
        ):
            with self.subTest(case=case):
                with openai_sdk_fixture(body_mutator=mutate(case)) as (
                    base_url,
                    _state,
                ):
                    result = self.run_sdk_fixture_script(
                        "openai_agent_tool_smoke.mjs",
                        base_url,
                    )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid fixture request", result.stderr)

    def test_agent_script_preserves_malformed_raw_model_output(self):
        with openai_sdk_fixture(malformed_agent=True) as (base_url, _state):
            result = self.run_sdk_fixture_script(
                "openai_agent_tool_smoke.mjs",
                base_url,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Chat did not generate exactly one valid tool call",
            result.stderr,
        )
        self.assertIn("src/lib.rs", result.stderr)
        self.assertIn("trailing", result.stderr)
        partial = self.harness.parse_smoke_output(
            result.stdout,
            "malformed agent fixture",
        )
        self.harness.validate_agent_failure_report(partial)
        self.assertEqual(partial["failure"]["phase"], "chat_tool_loop")
        self.assertIn("src/lib.rs", json.dumps(partial["failure"]["raw"]))
        self.assertIn("trailing", json.dumps(partial["failure"]["raw"]))
        self.assertTrue(partial["cancellation"]["nonterminal_delta"])
        self.assertTrue(partial["cancellation"]["abort_closed"])

    def test_run_protocol_phases_preserves_structured_agent_failure(self):
        args = SimpleNamespace(
            node="node",
            api_key=API_KEY,
            request_timeout=5.0,
        )
        with openai_sdk_fixture(malformed_agent=True) as (base_url, _state):
            protocol = self.harness.run_protocol_phases(
                args,
                SDK_DIR,
                base_url,
            )

        self.assertIsNotNone(protocol["compat"])
        self.assertIsNotNone(protocol["agent"])
        self.assertEqual(protocol["agent"]["failure"]["phase"], "chat_tool_loop")
        self.assertTrue(protocol["agent"]["cancellation"]["queued_request_completed"])
        self.assertEqual(
            [failure["phase"] for failure in protocol["failures"]],
            ["agent"],
        )

    def test_agent_failure_report_revalidates_completed_tool_loops(self):
        agent = self.valid_agent_report()
        partial = {
            "requests": {
                "chat_tool_loop": agent["requests"]["chat_tool_loop"],
            },
            "cancellation": agent["cancellation"],
            "failure": {
                "phase": "responses_tool_loop",
                "message": "Responses continuation failed",
                "raw": {"status": "incomplete"},
            },
        }
        self.harness.validate_agent_failure_report(partial)

        partial["requests"]["chat_tool_loop"]["call_count"] = 2
        with self.assertRaises(self.harness.PhaseError):
            self.harness.validate_agent_failure_report(partial)


if __name__ == "__main__":
    unittest.main()
