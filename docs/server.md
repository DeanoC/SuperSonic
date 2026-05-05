# OpenAI-Compatible Server

`supersonic-serve` exposes a local OpenAI-compatible HTTP surface for common
OSS clients and harnesses such as Pi, OpenCode, and Hermes.

## Start

```bash
SUPERSONIC_BACKENDS=cuda cargo build --release -p server

target/release/supersonic-serve \
  --backend cuda \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --max-context 4096 \
  --host 127.0.0.1 \
  --port 8080
```

Optional deployment flags:

- `--api-key KEY` or `SUPERSONIC_API_KEY=KEY` requires
  `Authorization: Bearer KEY`.
- `--cors-allow-origin ORIGIN` or `SUPERSONIC_CORS_ALLOW_ORIGIN=ORIGIN`
  enables CORS for browser clients. Use a concrete origin when possible;
  `*` is accepted for local development.
- `--response-store-max-entries N` or
  `SUPERSONIC_RESPONSE_STORE_MAX_ENTRIES=N` caps the in-memory
  `/v1/responses` store. The default is `1024`; oldest entries are evicted.
- `--max-queued-requests N` or `SUPERSONIC_MAX_QUEUED_REQUESTS=N` caps the
  number of requests waiting for the single GPU generation slot. The default
  is `32`; excess requests fail with HTTP 429.
- `--queue-timeout-ms N` or `SUPERSONIC_QUEUE_TIMEOUT_MS=N` caps how long a
  queued request may wait before failing. The default is `30000`.
- `--no-download` disables release-bake downloads.

## Endpoints

- `GET /`, `GET /v1`, `GET /health`, `GET /v1/health`, `GET /ready`,
  and `GET /v1/ready`
- `GET /v1/capabilities`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `GET /metrics`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{id}`
- `DELETE /v1/responses/{id}`
- `POST /v1/tokenize` and `POST /tokenize`
- `POST /v1/detokenize` and `POST /detokenize`

## Compatibility Notes

Chat Completions accepts modern client fields including `developer` messages,
text content-part arrays, `max_completion_tokens`, `stream_options.include_usage`,
`tools`, `tool_choice`, `response_format`, and `reasoning_effort`.

Responses accepts string input or message-array input, plus `instructions`,
`max_output_tokens`, `tools`, `tool_choice`, `reasoning.effort`, `stream`, and
`previous_response_id`. Response objects are kept in process memory only and
are bounded by `--response-store-max-entries`.

`response_format: {"type":"json_object"}` and Responses
`text: {"format":{"type":"json_object"}}` add a JSON-object system hint.
Strict `json_schema` constrained decoding is not implemented and returns a
clear 400 error.

Tool calls are model-generated and are not executed by SuperSonic. For Qwen
templates, generated XML `<tool_call>` blocks are parsed into OpenAI-shaped
`tool_calls` or Responses `function_call` output items. The client remains
responsible for running tools and sending tool results back.

For Responses tool loops, `previous_response_id` restores prior assistant
`function_call` items as assistant tool calls before new
`function_call_output` items are rendered. This lets clients send the tool
result with the previous response id instead of rebuilding the assistant
tool-call message themselves.

Reasoning is off by default. Set `reasoning_effort` on Chat Completions or
`reasoning.effort` on Responses to a non-off value to enable model thinking.
When present, `<think>...</think>` is stripped from visible assistant text and
returned separately as `reasoning_content` or a Responses `reasoning` item.

Unsupported features fail explicitly: `n > 1`, `logprobs`, `echo=true`,
image/file content parts, and `tool_choice="required"`.

`/v1/completions` accepts either a raw string prompt or a single token-id
array prompt. Batched prompts are rejected because SuperSonic currently serves
one generation per request.

Generation is scheduled through a bounded one-at-a-time GPU queue. Queue state
is exposed on `/health` and `/v1/capabilities`; disconnecting streaming
clients releases queued work before it enters the GPU slot, and in-flight
generation stops at the next token boundary once the response stream is gone.

`/v1/tokenize` and `/v1/detokenize` are SuperSonic-local helpers for token
budgeting. They require the same auth as generation endpoints and accept the
same loaded-model id or local aliases.

## Smoke Tests

```bash
curl -fsS http://127.0.0.1:8080/v1/capabilities | jq .

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [
      {"role": "developer", "content": "Answer briefly."},
      {"role": "user", "content": "Say hi"}
    ],
    "max_completion_tokens": 8,
    "temperature": 0
  }' | jq .

curl -fsS http://127.0.0.1:8080/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "instructions": "Answer briefly.",
    "input": "Say hi",
    "max_output_tokens": 8,
    "temperature": 0
  }' | jq .
```

## OpenAI SDK Harness Smoke

For a real OpenAI-compatible client smoke, install the Node OpenAI SDK in a
temporary directory and point it at `supersonic-serve`:

```bash
tmpdir=$(mktemp -d /tmp/supersonic-openai-smoke.XXXXXX)
cd "$tmpdir"
npm init -y >/dev/null
npm install openai@6 >/dev/null

SUPERSONIC_BASE_URL=http://127.0.0.1:8080 \
SUPERSONIC_API_KEY=secret \
node /path/to/SuperSonic/scripts/openai_compat_smoke.mjs
```

The smoke covers model list/retrieve, Chat Completions, streaming Chat
Completions with usage, legacy Completions, Responses create/get/delete,
tokenization, and metrics.
