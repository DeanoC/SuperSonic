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

For the canonical Qwen3.6-35B-A3B native INT4 FLM, start the persistent server
from the FLM alone:

```bash
target/release/supersonic-serve \
  --flm-file /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --backend hip --device 0 --max-context 4096 \
  --host 127.0.0.1 --port 8080 --api-key local --no-download
```

This first-class path resolves the model, tokenizer, native chat template, and
native INT4 execution plan from the FLM. Do not add `--model`, `--model-dir`,
or an INT4 layout flag to this invocation.

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
  - `--prefix-cache-max-bytes N` caps resident prefix snapshot bytes. The
    default is an automatic conservative VRAM budget; set `0` to disable the
    byte cap.
  - `--prefix-cache-memory-ttl-secs N` controls `in_memory` retention. The
    default is `600`.
  - `--prefix-cache-disk-ttl-secs N` controls `24h` metadata retention. The
    default is `86400`.

### Native DFlash Server Mode

`supersonic-serve` can host the native DFlash speculative decoder directly.
This is the preferred OpenCode path; it keeps the target and draft resident in
one long-lived process and avoids the older Python CLI shim.

Current constraints:

- supported targets are `qwen3.5-9b` and `qwen3.6-27b`;
- the target must be loaded from a low-bit bake (`--int4`, `--q4km`, or
  `--q4km-gptq`);
- `--dflash-draft-dir` must point at a HuggingFace-style DFlash draft
  checkpoint directory;
- `--kv-fp8` is rejected for DFlash server mode;
- prefix caching should be disabled for this path until DFlash snapshot/restore
  support is added to the cache.

Example R9700 / `gfx1201` OpenCode server:

```bash
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100,gfx1201 \
target/release/supersonic-serve \
  --backend hip \
  --device 1 \
  --model qwen3.6-27b \
  --model-dir /mnt/data/tmp/supersonic-qwen36-27b-lucebox \
  --max-context 4096 \
  --q4km \
  --dflash \
  --dflash-draft-dir /mnt/data/tmp/qwen36-27b-dflash-q8-bf16 \
  --host 127.0.0.1 \
  --port 8013 \
  --api-key local \
  --no-download \
  --prefix-cache-disable
```

Optional DFlash tuning flags:

- `--dflash-block N` overrides the checkpoint block size. The override must
  divide the draft block size.
- `--dflash-tap-layers 1,16,31,46,61` overrides target tap layers. By default
  the runtime uses model-appropriate taps and the draft checkpoint metadata.

For OpenCode, configure an OpenAI-compatible provider with:

```json
{
  "provider": {
    "supersonic-dflash": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "SuperSonic native DFlash server",
      "options": {
        "baseURL": "http://127.0.0.1:8013/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen3.6 27B via SuperSonic DFlash Q4KM",
          "limit": {
            "context": 4096,
            "output": 512
          }
        }
      }
    }
  }
}
```

The initial OpenCode smoke profile is functional but prefill-bound: a tiny
direct chat request returns in about 0.52 s, while a minimal OpenCode agent
prompt of roughly 225 input tokens spends about 4.3 s in model prefill. Running
OpenCode attached to a warm `opencode serve` process trims roughly one second
of wrapper overhead, but the main optimization target is still DFlash/target
prefill for agent-sized prompts.

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

Resident snapshots are admitted only when they fit both the entry-count cap and
the byte budget. This avoids cloning large model states into cache on GPUs that
do not have enough spare VRAM. `/v1/capabilities` and `/metrics` expose current
resident bytes, byte budget, and admission skips.

`prompt_cache_retention` accepts:

- `in_memory` (default): resident snapshot with a short idle TTL.
- `24h`: resident snapshot plus prompt-free disk metadata. Qwen snapshots are
  also persisted and can be lazily restored after a server restart.
- `none`: bypass cache lookup and capture for the request.

The disk files intentionally avoid prompt text and message JSON. They contain
hashes, counts, retention, expiry metadata, logits, and model-state tensor
bytes keyed by token hashes. Gemma disk snapshots are not persisted yet.

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
npm install openai@6.49.0 >/dev/null

SUPERSONIC_BASE_URL=http://127.0.0.1:8080 \
SUPERSONIC_API_KEY=secret \
node /path/to/SuperSonic/scripts/openai_compat_smoke.mjs

SUPERSONIC_BASE_URL=http://127.0.0.1:8080 \
SUPERSONIC_API_KEY=secret \
node /path/to/SuperSonic/scripts/openai_agent_tool_smoke.mjs
```

The compatibility smoke covers missing and wrong-key auth, protected
operational routes, model list/retrieve, tokenize/detokenize, exact `hello`
canaries for Chat Completions, legacy Completions, Responses, and both
reconstructed streams, terminal ordering and usage, Responses
create/get/delete, and repeated warm requests. Transport compatibility and
semantic quality are reported independently. Reasoning acceptance is also
reported separately from observed reasoning and is not a green capability
unless reasoning content is actually observed.

The agent smoke requires exactly one model-generated `read_source_file` call
with a nonempty call ID and exactly `{"path":"src/lib.rs"}`, no suffix text,
and terminal tool state. It performs a terminal text-only continuation through
both Chat Completions and Responses. Its cancellation gate observes a
nonterminal stream delta, queues a second real request, awaits abort closure,
completes the queued request, and requires authenticated health and metrics to
report released active and queued work without a model reload.

The 2026-06-28 production refactor validation ran this harness against
`qwen3.6-27b` Q4KM-GPTQ DFlash on HIP at `127.0.0.1:8013`, using the then-current
`openai@6` from a temporary npm directory. It passed model list/retrieve, Chat
Completions, streaming Chat Completions with usage, legacy Completions,
Responses create/get/delete, `/tokenize`, and `/metrics`.

For production refactor PRs, this smoke is the minimum client-compatibility
gate. It must cover Chat Completions, Responses create/get/delete, tokenization,
streaming, and explicit unsupported-feature errors. For DFlash server mode, the
smoke should be run with `--prefix-cache-disable` or rely on runtime policy to
disable prefix-cache admission automatically.

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

To verify Qwen disk restore, run the smoke once with `24h`, restart the server
with the same `--prefix-cache-dir`, then run:

```bash
SUPERSONIC_BASE_URL=http://127.0.0.1:8080 \
SUPERSONIC_API_KEY=secret \
SUPERSONIC_PROMPT_CACHE_RETENTION=24h \
SUPERSONIC_PREFIX_CACHE_SMOKE_PHASE=restart-probe \
node /path/to/SuperSonic/scripts/prefix_cache_smoke.mjs
```

For short local prompts, start the server with `--prefix-cache-min-tokens 1`
or set `SUPERSONIC_PREFIX_CACHE_MIN_TOKENS=1`; production defaults avoid
caching tiny prompts.
