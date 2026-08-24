from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_peer():
    path = ROOT / "tools" / "llama-cpp-peer.py"
    if not path.is_file():
        raise AssertionError("tools/llama-cpp-peer.py is absent")
    spec = importlib.util.spec_from_file_location("llama_cpp_peer", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llama_cpp_peer"] = module
    spec.loader.exec_module(module)
    return module


class LlamaCppPeerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.peer = load_peer()

    def test_server_command_disables_hidden_warmup_and_prompt_cache(self):
        command = self.peer.build_server_command(
            server_binary="/opt/llama/bin/llama-server",
            model=Path("/models/qwen.gguf"),
            context_size=4096,
            port=18321,
        )

        self.assertEqual(command[0], "/opt/llama/bin/llama-server")
        self.assertEqual(command[command.index("--model") + 1], "/models/qwen.gguf")
        self.assertEqual(command[command.index("--port") + 1], "18321")
        self.assertEqual(command[command.index("--ctx-size") + 1], "4096")
        self.assertIn("--no-cache-prompt", command)
        self.assertIn("--no-warmup", command)
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "99")

    def test_response_normalization_uses_exact_server_timing_fields(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "reasoning_content": "",
                        "content": "answer",
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "timings": {
                "prompt_n": 7,
                "predicted_n": 3,
                "predicted_ms": 12.0,
                "predicted_per_token_ms": 4.0,
                "predicted_per_second": 250.0,
            },
        }

        self.assertEqual(
            self.peer.normalize_response(response, expected_generated_tokens=3),
            {
                "decode_ms": 12.0,
                "generated_text": "answer",
                "generated_tokens": 3,
                "ms_per_tok": 4.0,
                "prompt_tokens": 7,
                "tokens_per_second": 250.0,
            },
        )

    def test_response_normalization_rejects_truncation_or_inconsistent_counts(self):
        response = {
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            "timings": {
                "prompt_n": 7,
                "predicted_n": 2,
                "predicted_ms": 8.0,
                "predicted_per_token_ms": 4.0,
                "predicted_per_second": 250.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "generated token count|finish_reason"):
            self.peer.normalize_response(response, expected_generated_tokens=3)

    def test_one_shot_lifecycle_posts_strict_request_and_stops_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            request_file = root / "request.json"
            pid_file = root / "pid.txt"
            fake_server = root / "fake-llama-server"
            fake_server.write_text(
                """#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', type=int, required=True)
args, _ = parser.parse_known_args()
Path(os.environ['FAKE_LLAMA_PID_FILE']).write_text(str(os.getpid()))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        Path(os.environ['FAKE_LLAMA_REQUEST_FILE']).write_bytes(raw)
        request = json.loads(raw)
        count = request['max_tokens']
        response = {
            'choices': [{'finish_reason': 'length', 'message': {'reasoning_content': '', 'content': 'answer'}}],
            'usage': {'prompt_tokens': 7, 'completion_tokens': count, 'total_tokens': 7 + count},
            'timings': {
                'prompt_n': 7,
                'predicted_n': count,
                'predicted_ms': count * 4.0,
                'predicted_per_token_ms': 4.0,
                'predicted_per_second': 250.0,
            },
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
""",
                encoding="utf-8",
            )
            fake_server.chmod(0o755)
            previous_request = os.environ.get("FAKE_LLAMA_REQUEST_FILE")
            previous_pid = os.environ.get("FAKE_LLAMA_PID_FILE")
            os.environ["FAKE_LLAMA_REQUEST_FILE"] = str(request_file)
            os.environ["FAKE_LLAMA_PID_FILE"] = str(pid_file)
            try:
                result = self.peer.run_one_shot(
                    server_binary=str(fake_server),
                    model=model,
                    prompt="hello",
                    max_new_tokens=3,
                    context_size=4096,
                    seed=42,
                    chat=True,
                    startup_timeout_seconds=5,
                )
            finally:
                if previous_request is None:
                    os.environ.pop("FAKE_LLAMA_REQUEST_FILE", None)
                else:
                    os.environ["FAKE_LLAMA_REQUEST_FILE"] = previous_request
                if previous_pid is None:
                    os.environ.pop("FAKE_LLAMA_PID_FILE", None)
                else:
                    os.environ["FAKE_LLAMA_PID_FILE"] = previous_pid

            self.assertEqual(result["generated_tokens"], 3)
            posted = json.loads(request_file.read_text(encoding="utf-8"))
            self.assertFalse(posted["cache_prompt"])
            self.assertTrue(posted["ignore_eos"])
            self.assertFalse(posted["stream"])
            self.assertEqual(posted["chat_template_kwargs"], {"enable_thinking": False})
            pid = int(pid_file.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_response_normalization_rejects_unexpected_reasoning_content(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"reasoning_content": "think", "content": "answer"},
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "timings": {
                "prompt_n": 7,
                "predicted_n": 3,
                "predicted_ms": 12.0,
                "predicted_per_token_ms": 4.0,
                "predicted_per_second": 250.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "reasoning"):
            self.peer.normalize_response(response, expected_generated_tokens=3)

    def test_response_normalization_accepts_early_eos_for_quality(self):
        response = {
            "choices": [{"finish_reason": "stop", "message": {"content": "ready"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
            "timings": {
                "prompt_n": 7,
                "predicted_n": 1,
                "predicted_ms": 4.0,
                "predicted_per_token_ms": 4.0,
                "predicted_per_second": 250.0,
            },
        }

        result = self.peer.normalize_response(
            response,
            expected_generated_tokens=4,
            fixed_token_count=False,
        )

        self.assertEqual(result["generated_text"], "ready")
        self.assertEqual(result["generated_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
