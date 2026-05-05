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
- Exact-prefix caching is enabled by default for repeated chat/agent turns.
  Useful knobs:
  - `--prefix-cache-disable` disables cache lookup and capture.
  - `--prefix-cache-dir DIR` sets the cache metadata directory; the default is
    `{model_dir}/.supersonic/serve-cache/v1/`.
  - `--prefix-cache-min-tokens N` sets the minimum prompt prefix eligible for
    caching. The default is `128`.
  - `--prefix-cache-max-entries N` caps resident prefix snapshots. Snapshots
    clone model state on GPU; the conservative default is `1`.
  - `--prefix-cache-memory-ttl-secs N` controls `in_memory` retention. The
    default is `600`.
  - `--prefix-cache-disk-ttl-secs N` controls `24h` metadata retention. The
    default is `86400`.

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
It also accepts OpenAI-style cache controls: `prompt_cache_key`,
`prompt_cache_retention`, `user`, and `metadata`. `metadata.thread_id`,
`metadata.conversation_id`, or `metadata.session_id` scopes repeated thread
turns when no explicit `prompt_cache_key` is provided.

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

## Prefix Cache

SuperSonic performs exact token-prefix reuse. On a cache hit, the server
restores the cached model state for the longest matching prefix, runs only the
uncached suffix, and reports the reused prompt tokens as
`usage.prompt_tokens_details.cached_tokens`. The cache is scoped by model, API
key, user/thread metadata, and `prompt_cache_key` to avoid accidental cross-user
reuse.

`prompt_cache_retention` accepts:

- `in_memory` (default): resident snapshot with a short idle TTL.
- `24h`: resident snapshot plus disk metadata under the prefix-cache directory.
- `none`: bypass cache lookup and capture for the request.

The disk files intentionally avoid prompt text and message JSON. They contain
only hashes, counts, retention, and expiry metadata; resident model-state
snapshots are what provide the live speedup.

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

## Prefix Cache Smoke

With a server already running, verify exact-prefix reuse and metrics:

```bash
SUPERSONIC_BASE_URL=http://127.0.0.1:8080 \
SUPERSONIC_API_KEY=secret \
node /path/to/SuperSonic/scripts/prefix_cache_smoke.mjs
```

The smoke uses `/v1/chat/completions` when `/v1/capabilities` reports chat
support, otherwise it falls back to `/v1/completions`. Set
`SUPERSONIC_PREFIX_CACHE_MODE=completions` or `chat` to force one endpoint.
Set `SUPERSONIC_PROMPT_CACHE_RETENTION=24h` to exercise disk metadata writes.
It checks both an exact repeat and an extended same-prefix request, matching
the common agent pattern where later turns replay prior transcript text and
append new user/tool content.

For short local prompts, start the server with `--prefix-cache-min-tokens 1`
or set `SUPERSONIC_PREFIX_CACHE_MIN_TOKENS=1`; production defaults avoid
caching tiny prompts.
