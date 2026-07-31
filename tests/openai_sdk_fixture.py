"""Deterministic OpenAI-shaped HTTP/SSE fixture for the SDK smoke scripts."""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator


MODEL = "qwen3.6-35b-a3b"
API_KEY = "local"
CHAT_CALL_ID = "call_chat_fixture"
RESPONSE_CALL_ID = "call_response_fixture"
RESPONSE_ID = "resp_fixture"
CODING_PROMPT = (
    "Your entire response must be exactly one call to read_source_file "
    "with path src/lib.rs. Do not write natural language before or after the call."
)
TOOL_OUTPUT = json.dumps(
    {
        "path": "src/lib.rs",
        "contents": "pub fn protocol_ready() -> bool { true }\n",
    },
    separators=(",", ":"),
)
FUNCTION_DEFINITION = {
    "name": "read_source_file",
    "description": "Read a UTF-8 source file from the current coding workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative source path to read.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}
CHAT_TOOL = {"type": "function", "function": FUNCTION_DEFINITION}
RESPONSES_TOOL = {"type": "function", **FUNCTION_DEFINITION}


def _usage(prompt: int = 3, completion: int = 1) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _response_usage(prompt: int = 3, completion: int = 1) -> dict[str, int]:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _chat_response(
    content: str | None,
    *,
    finish_reason: str,
    tool_calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl_fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(),
    }


def _response_object(
    output: list[dict[str, object]],
    *,
    response_id: str = RESPONSE_ID,
) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": MODEL,
        "output": output,
        "usage": _response_usage(),
        "parallel_tool_calls": False,
        "temperature": 0,
        "tool_choice": "auto",
        "tools": [],
    }


def _message_output(text: str) -> dict[str, object]:
    return {
        "id": "msg_fixture",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _expected_chat_initial() -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": CODING_PROMPT}],
        "tools": [CHAT_TOOL],
        "tool_choice": "auto",
        "max_completion_tokens": 128,
        "temperature": 0,
    }


def _expected_chat_continuation() -> dict[str, object]:
    return {
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
                "content": TOOL_OUTPUT,
            },
        ],
        "tools": [CHAT_TOOL],
        "tool_choice": "auto",
        "max_completion_tokens": 64,
        "temperature": 0,
    }


def _expected_responses_initial() -> dict[str, object]:
    return {
        "model": MODEL,
        "input": CODING_PROMPT,
        "tools": [RESPONSES_TOOL],
        "tool_choice": "auto",
        "max_output_tokens": 128,
        "temperature": 0,
    }


def _expected_responses_continuation() -> dict[str, object]:
    return {
        "model": MODEL,
        "previous_response_id": RESPONSE_ID,
        "input": [
            {
                "type": "function_call_output",
                "call_id": RESPONSE_CALL_ID,
                "output": TOOL_OUTPUT,
            }
        ],
        "tools": [RESPONSES_TOOL],
        "tool_choice": "auto",
        "max_output_tokens": 64,
        "temperature": 0,
    }


class FixtureState:
    def __init__(
        self,
        *,
        malformed_agent: bool = False,
        body_mutator: Callable[[str, dict[str, object]], dict[str, object]]
        | None = None,
    ) -> None:
        self.malformed_agent = malformed_agent
        self.body_mutator = body_mutator
        self.lock = threading.Lock()
        self.cancel_active = 0
        self.cancel_queued = 0
        self.cancel_released = threading.Event()
        self.stored_response = _response_object([_message_output("hello")])
        self.requests: list[dict[str, object]] = []

    def scheduler(self) -> tuple[int, int]:
        with self.lock:
            return self.cancel_active, self.cancel_queued


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: FixtureState) -> None:
        super().__init__(address, FixtureHandler)
        self.state = state


class FixtureHandler(BaseHTTPRequestHandler):
    server: FixtureServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _authorized(self) -> bool:
        if self.headers.get("authorization") == f"Bearer {API_KEY}":
            return True
        self._json(
            401,
            {
                "error": {
                    "message": "invalid API key",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
        return False

    def _json(self, status: int, payload: object) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload: object, event: str | None = None) -> None:
        if event is not None:
            self.wfile.write(f"event: {event}\n".encode())
        self.wfile.write(
            b"data: "
            + json.dumps(payload, separators=(",", ":")).encode()
            + b"\n\n"
        )
        self.wfile.flush()

    def do_GET(self) -> None:
        self.server.state.requests.append(
            {"method": "GET", "path": self.path, "body": None}
        )
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL,
                            "object": "model",
                            "created": 1,
                            "owned_by": "supersonic",
                        }
                    ],
                },
            )
            return
        if self.path == f"/v1/models/{MODEL}":
            self._json(
                200,
                {
                    "id": MODEL,
                    "object": "model",
                    "created": 1,
                    "owned_by": "supersonic",
                },
            )
            return
        if self.path == f"/v1/responses/{RESPONSE_ID}":
            self._json(200, self.server.state.stored_response)
            return
        if self.path in ("/health", "/ready", "/v1/capabilities"):
            active, queued = self.server.state.scheduler()
            if self.path == "/ready":
                self._json(200, {"ready": True, "model": MODEL})
            elif self.path == "/v1/capabilities":
                self._json(
                    200,
                    {
                        "ready": True,
                        "model": MODEL,
                        "scheduler": {
                            "active_requests": active,
                            "queued_requests": queued,
                        },
                    },
                )
            else:
                self._json(
                    200,
                    {
                        "status": "ok",
                        "active_requests": active,
                        "queued_requests": queued,
                    },
                )
            return
        if self.path == "/metrics":
            active, queued = self.server.state.scheduler()
            body = (
                f"supersonic_active_requests {active}\n"
                f"supersonic_queued_requests {queued}\n"
                "supersonic_model_loads_total 1\n"
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_DELETE(self) -> None:
        self.server.state.requests.append(
            {"method": "DELETE", "path": self.path, "body": None}
        )
        if not self._authorized():
            return
        if self.path == f"/v1/responses/{RESPONSE_ID}":
            self._json(
                200,
                {"id": RESPONSE_ID, "object": "response.deleted", "deleted": True},
            )
            return
        self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        body = self._body()
        if self.server.state.body_mutator is not None:
            body = self.server.state.body_mutator(self.path, body)
        self.server.state.requests.append(
            {"method": "POST", "path": self.path, "body": body}
        )
        if self.path == "/v1/tokenize":
            self._json(200, {"tokens": [10, 11]})
            return
        if self.path == "/v1/detokenize":
            self._json(200, {"text": "hello world"})
            return
        if self.path == "/v1/completions":
            self._json(
                200,
                {
                    "id": "cmpl_fixture",
                    "object": "text_completion",
                    "created": 1,
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "text": "hello",
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage(),
                },
            )
            return
        if self.path == "/v1/chat/completions":
            self._chat(body)
            return
        if self.path == "/v1/responses":
            self._responses(body)
            return
        self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def _chat(self, body: dict[str, object]) -> None:
        messages = body.get("messages") or []
        prompt = " ".join(
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict)
        )
        if body.get("stream") and "one to one hundred" in prompt:
            self._cancellation_stream()
            return
        if "queued cancellation probe" in prompt:
            self._queued_request()
            return
        if body.get("stream"):
            self._chat_stream()
            return
        if "read_source_file" in prompt:
            if len(messages) == 3:
                if body != _expected_chat_continuation():
                    self._invalid_fixture_request("Chat continuation body")
                    return
                self._json(200, _chat_response("file read", finish_reason="stop"))
                return
            if body != _expected_chat_initial():
                self._invalid_fixture_request("Chat initial body")
                return
            if self.server.state.malformed_agent:
                self._json(
                    200,
                    _chat_response(
                        '{"path":"src/lib.rs"} trailing',
                        finish_reason="stop",
                    ),
                )
                return
            self._json(
                200,
                _chat_response(
                    None,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": CHAT_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": "read_source_file",
                                "arguments": '{"path":"src/lib.rs"}',
                            },
                        }
                    ],
                ),
            )
            return
        if "1 + 1" in prompt:
            response = _chat_response("2", finish_reason="stop")
            response["choices"][0]["message"]["reasoning_content"] = "brief arithmetic"
            self._json(200, response)
            return
        if "single word ready" in prompt:
            self._json(200, _chat_response("ready", finish_reason="stop"))
            return
        self._json(200, _chat_response("hello", finish_reason="stop"))

    def _chat_stream(self) -> None:
        self._sse_start()
        base = {
            "id": "chatcmpl_stream_fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
        }
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hel"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "lo"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        self._sse({**base, "choices": [], "usage": _usage()})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _cancellation_stream(self) -> None:
        with self.server.state.lock:
            self.server.state.cancel_active = 1
        self._sse_start()
        self._sse(
            {
                "id": "chatcmpl_cancel_fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "one"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        try:
            while True:
                time.sleep(0.05)
                self.wfile.write(b": fixture-keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.error):
            with self.server.state.lock:
                self.server.state.cancel_active = 0
            self.server.state.cancel_released.set()

    def _queued_request(self) -> None:
        with self.server.state.lock:
            self.server.state.cancel_queued = 1
        if not self.server.state.cancel_released.wait(timeout=5):
            self._json(
                500,
                {"error": {"message": "cancellation was not released", "type": "fixture"}},
            )
            return
        with self.server.state.lock:
            self.server.state.cancel_queued = 0
            self.server.state.cancel_active = 1
        try:
            self._json(200, _chat_response("released", finish_reason="stop"))
        finally:
            with self.server.state.lock:
                self.server.state.cancel_active = 0

    def _responses(self, body: dict[str, object]) -> None:
        if body.get("stream"):
            self._responses_stream()
            return
        previous_id = body.get("previous_response_id")
        input_value = body.get("input")
        is_tool_output = (
            isinstance(input_value, list)
            and any(
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                for item in input_value
            )
        )
        if previous_id or is_tool_output:
            if body != _expected_responses_continuation():
                self._invalid_fixture_request("Responses continuation body")
                return
            response = _response_object(
                [_message_output("file read")],
                response_id="resp_tool_final_fixture",
            )
            self._json(200, response)
            return
        if isinstance(input_value, str) and "read_source_file" in input_value:
            if body != _expected_responses_initial():
                self._invalid_fixture_request("Responses initial body")
                return
            if self.server.state.malformed_agent:
                self._json(
                    200,
                    _response_object([_message_output('{"path":"src/lib.rs"} trailing')]),
                )
                return
            self._json(
                200,
                _response_object(
                    [
                        {
                            "id": "fc_fixture",
                            "type": "function_call",
                            "status": "completed",
                            "call_id": RESPONSE_CALL_ID,
                            "name": "read_source_file",
                            "arguments": '{"path":"src/lib.rs"}',
                        }
                    ]
                ),
            )
            return
        response = _response_object([_message_output("hello")])
        self.server.state.stored_response = response
        self._json(200, response)

    def _invalid_fixture_request(self, label: str) -> None:
        self._json(
            400,
            {
                "error": {
                    "message": f"invalid fixture request: {label}",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_fixture_request",
                }
            },
        )

    def _responses_stream(self) -> None:
        self._sse_start()
        response = _response_object([_message_output("hello")], response_id="resp_stream")
        events = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        **response,
                        "status": "in_progress",
                        "output": [],
                        "usage": None,
                    },
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hel",
                    "sequence_number": 1,
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "lo",
                    "sequence_number": 2,
                },
            ),
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "hello",
                    "sequence_number": 3,
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": response,
                    "sequence_number": 4,
                },
            ),
        ]
        for event, payload in events:
            self._sse(payload, event=event)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@contextmanager
def openai_sdk_fixture(
    *,
    malformed_agent: bool = False,
    body_mutator: Callable[[str, dict[str, object]], dict[str, object]]
    | None = None,
) -> Iterator[tuple[str, FixtureState]]:
    state = FixtureState(
        malformed_agent=malformed_agent,
        body_mutator=body_mutator,
    )
    server = FixtureServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
