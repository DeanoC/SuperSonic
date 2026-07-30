# FLM Qwen3.6 First-Class Serving Design

## Status

Approved for implementation planning on 2026-07-30.

## Objective

Make a self-contained Qwen3.6-35B-A3B FLM a first-class model source for
`supersonic-serve`. The server must load the FLM once into a long-lived HIP
process and expose the existing OpenAI-compatible APIs needed by an agentic
coding client. Normal startup and request handling must not require a
Hugging Face snapshot, a second weight package, or a subprocess invocation of
the `supersonic` CLI.

This stage establishes the production architecture and a complete,
single-request-at-a-time HTTP vertical slice. Continuous batching,
Qwen3.6 prefix snapshots, speculative decoding, and hipFile-backed direct
storage remain later performance stages, but must be addable without changing
the public model-loading or HTTP interfaces introduced here.

## Current State

The merged first-class FLM verifier proves that geo-quant can produce a
Qwen3.6-35B-A3B native INT4 FLM which SuperSonic can validate, load, and run
without an HF snapshot. That path is still owned by the `supersonic` CLI:

- FLM opening, runtime descriptor parsing, tokenizer reconstruction, direct
  plan selection, layer loading, persistent scratch, prefill, and decode are
  orchestrated by `crates/runner/src/qwen36_moe/`.
- `supersonic-runtime` owns the persistent session abstraction used by the
  HTTP server, but its `InferenceSession` has no Qwen3.6 MoE variant.
- `supersonic-runtime::state::build` explicitly rejects
  `ModelFamily::Qwen36Moe`.
- `supersonic-serve` currently assumes `--model-dir` is an HF-style directory
  and loads `tokenizer.json` and `tokenizer_config.json` from that directory.

The server already implements the protocol behavior needed by coding clients:
Chat Completions, Responses, streaming, tools, reasoning separation,
authentication, bounded queueing, cancellation, tokenization helpers,
health/readiness, metrics, and a bounded response store. The missing work is
to give those routes a real persistent Qwen3.6 FLM session.

## Native Chat Template Format Extension

The current native FLM runtime directory contains the Qwen BPE tokenizer
descriptor and its vocab, merges, added-token, and regex assets. It does not
contain a native chat template. Agentic serving therefore requires one small
format extension shared by geo-quant and SuperSonic:

```text
ASSET_CHAT_TEMPLATE_UTF8 = 5
flags = ASSET_FLAG_REQUIRED_FOR_RUNTIME | ASSET_FLAG_TEXT_UTF8
name = "chat_template"
payload = exact UTF-8 Jinja template source
```

The asset is neither JSON nor an HF compatibility sidecar. The producer reads
and resolves the source checkpoint's selected default chat template at export
time, validates it as non-empty UTF-8, and stores the template source directly.
SuperSonic compiles that source with its existing `ChatTemplate` implementation
at startup.

The Qwen3.6 serving profile requires exactly one asset of this kind. Missing,
duplicate, non-UTF-8, or uncompileable template assets fail validation and
startup. Model special tokens continue to come from the existing native
tokenizer and Qwen3.6 config records; the serving path does not parse
`tokenizer_config.json`.

geo-quant must add this asset to new Qwen3.6 FLM exports even when
`--hf-compat-assets omit` is selected. The canonical native INT4 FLM used by
the real server gate must be regenerated after this extension. Optional HF
JSON assets remain compatibility-only and are not accepted as a substitute
for the native template in the first-class serving profile.

## Architectural Decision

The reusable Qwen3.6 engine lifecycle belongs in `supersonic-runtime`, not in
the CLI, server, or a worker subprocess.

`supersonic-runtime` already depends on `gpu-hal`, `kernel-ffi`,
`model-store`, and `qwen36_moe`, so this placement does not introduce a new
dependency direction. Both `runner` and `server` already depend on
`supersonic-runtime`. The CLI remains responsible for CLI-only diagnostics,
experimental switches, reports, and stdout rendering, while runtime owns the
model source, resident model state, and token-level inference lifecycle.

The implementation must not make `server` depend on `runner`, make `runtime`
depend on `runner`, duplicate the Qwen3.6 loader, or introduce an IPC protocol
around the CLI.

## Runtime Components

### `Qwen36MoeLoadConfig`

A runtime-owned load configuration describes only serving-relevant choices:

- FLM path;
- backend and device ordinal;
- maximum context length;
- persistent decode enablement;
- KV format and VMM policy;
- expert residency and prefetch policy;
- FLM virtual transfer backend;
- block-hash verification policy.

An FLM load configuration does not contain an HF model directory, a
quantization-selection flag, or an alternate bake path. The FLM runtime
descriptor and direct plans describe the model and storage layout.

### `Qwen36MoeEngine`

The long-lived engine owns:

- the open FLM store and source mapping;
- parsed Qwen3.6 model configuration;
- reconstructed tokenizer and model special-token metadata;
- validated runtime direct-plan selection and direct coverage profile;
- GPU-resident dense and native INT4 weights;
- virtual expert arena and residency manager;
- full-attention KV and linear-attention recurrent state;
- persistent decode descriptors and scratch buffers;
- embedding, final norm, LM-head, and logits buffers;
- startup timings and immutable load evidence.

Loading and allocation happen exactly once during server startup. Request
reset clears mutable sequence state while retaining weights, source mappings,
arenas, descriptors, and reusable scratch allocations.

The engine exposes these runtime-facing lifecycle operations:

```rust
pub struct Qwen36MoeLoadConfig { /* serving policy */ }
pub struct Qwen36MoeEngine { /* resident model and request state */ }

impl Qwen36MoeEngine {
    pub fn load(config: Qwen36MoeLoadConfig) -> anyhow::Result<Self>;
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer;
    pub fn eos_ids(&self) -> &[u32];
    pub fn load_evidence(&self) -> &Qwen36MoeLoadEvidence;
    pub fn reset(&mut self) -> anyhow::Result<()>;
    pub fn prefill(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>>;
    pub fn decode_step(&mut self, token_id: u32, pos: usize)
        -> anyhow::Result<Vec<f32>>;
}
```

Internal helper types are split by source loading, resident weights, mutable
sequence state, and telemetry, while these lifecycle semantics remain fixed.
`prefill` consumes the complete prompt and returns host-visible F32 logits for
the last prompt position. `decode_step` consumes one sampled token at its
absolute position and returns next-token logits. Both operations preserve
state for subsequent decode calls.

### Session Integration

`InferenceSession` gains a Qwen3.6 MoE variant backed by
`Qwen36MoeEngine`. Its first-stage feature report is:

```text
plain_prefill_decode = true
native_dflash_generate = false
prefix_snapshot = false
disk_prefix_snapshot = false
```

`reset`, `prefill`, and `decode_step` dispatch to the Qwen3.6 engine.
Snapshot, restore, and disk snapshot operations return explicit unsupported
errors. The prefix cache observes the feature report and bypasses snapshot
lookup and capture without disabling ordinary request serving.

## FLM-Only Server Startup

`supersonic-serve` adds `--flm-file PATH` as the first-class FLM source option.
It is mutually exclusive with a directory-valued `--model-dir`. A file-valued
`--model-dir model.flm` remains a compatibility spelling and is normalized to
the same source before policy validation. When the selected source is an FLM:

1. Open the FLM and parse its runtime descriptor.
2. Resolve the model variant from the descriptor.
3. Make `--model` optional for FLM startup. If present, require it to match the
   descriptor. Keep `--model` required for directory-valued HF startup.
4. Reject `--int4`, `--q4km`, `--q4km-gptq`, `--fp8-runtime`, DFlash target
   flags, and any other external weight-selection option.
5. Validate the SuperSonic Qwen3.6 direct plans before GPU allocation.
6. Reconstruct the tokenizer from FLM tokenizer assets.
7. Load and compile `ASSET_CHAT_TEMPLATE_UTF8`; absence or incompatibility is
   a startup error rather than a completions-only degradation.
8. Build one persistent `Qwen36MoeEngine`.
9. Mark readiness true only after the complete engine and protocol metadata
   are available.

No FLM startup branch calls `ensure_hf_metadata_present`, reads files adjacent
to the FLM, downloads a bake, or falls back to an HF path.

Directory-valued `--model-dir` behavior for existing models remains
unchanged.

## Direct-Plan Contract and Load Evidence

The server requires the same strict direct-plan contract as the merged
first-class verifier:

- every required logical model weight has a compatible direct plan;
- raw dense roles use raw-dense plans;
- quantized projection roles use native INT4 plans;
- mixed native INT4 and BF16 fallback projection modes are rejected;
- the accepted first-class serving profile has positive native INT4 coverage
  and zero BF16 fallback;
- optional full BLAKE3 payload verification is completed before readiness.

The engine records immutable `Qwen36MoeLoadEvidence` containing:

- FLM path and model identity;
- storage ABI and selected direct-plan profile;
- required, raw-dense, native-INT4, and BF16-fallback counts;
- selected transfer backend;
- source bytes and device-upload bytes;
- source-open, descriptor, tokenizer, plan, allocation, upload, and total
  startup durations;
- a process-local load sequence number.

`/health`, `/ready`, `/v1/capabilities`, and `/metrics` expose enough of this
evidence to prove that the serving process loaded the expected FLM once and
did not fall back. Filesystem paths must be reduced to a basename in HTTP
responses and metrics labels to avoid leaking deployment layout.

## HTTP and Agentic Coding Behavior

Qwen3.6 FLM uses the existing server routes and schemas rather than a
model-specific HTTP implementation.

The complete vertical slice must support:

- `GET /health`, `/ready`, `/v1/capabilities`, and `/metrics`;
- `GET /v1/models` and model retrieval;
- `POST /v1/tokenize` and `/v1/detokenize`;
- non-streaming and streaming `POST /v1/chat/completions`;
- non-streaming and streaming `POST /v1/responses`;
- response retrieval and deletion;
- system and developer messages;
- text content-part arrays;
- reasoning effort and separated reasoning output;
- OpenAI-shaped tool definitions, tool choice where already supported, Qwen
  tool-call parsing, and tool-result continuation;
- bounded queue admission and queue timeout;
- client-disconnect cancellation at the next token boundary;
- API-key authentication and existing CORS behavior.

The existing single-GPU generation permit remains in force for this stage.
Concurrent clients may queue, but model execution remains one request at a
time. The server must not reload weights between requests.

Prefix cache controls are accepted for protocol compatibility, but Qwen3.6
reports zero cached tokens and does not create snapshots in this stage.

## Request Lifecycle

For each generation request:

1. Validate and normalize the OpenAI-compatible request.
2. Render messages, tools, prior function calls, and tool outputs with the
   chat template reconstructed from FLM.
3. Tokenize with the FLM tokenizer and enforce the configured context limit.
4. Acquire the bounded generation permit.
5. Lock the persistent session and reset mutable request state.
6. Prefill the prompt through the Qwen3.6 batched-prefill path.
7. Sample through the server's existing sampling implementation and call
   incremental `decode_step` for subsequent tokens.
8. Emit protocol events as tokens become available.
9. Check cancellation between token boundaries and release the session and
   scheduler permit on every terminal path.
10. Preserve engine readiness unless an error indicates corrupted GPU or
    session state.

Streaming must be live token streaming. The engine must not generate an
entire response before the server begins emitting chunks.

## Error and Readiness Semantics

Startup failures are phase-labelled as FLM open, descriptor, direct plan,
tokenizer/template, allocation, upload, or engine initialization failures.
They terminate startup before the socket reports readiness.

Request validation and unsupported protocol features retain the existing 4xx
OpenAI-compatible error shapes. Queue overflow remains HTTP 429. Ordinary
generation errors produce the existing server error response and telemetry.

An engine error is classified as either request-local or integrity-losing.
Request-local errors reset the sequence before the next request. An
integrity-losing error marks readiness false and rejects new generation
requests until process restart. The initial classifier treats GPU device
loss, invalid resident pointers/descriptors, and failed state reset as
integrity-losing; sampling, context-limit, cancellation, and client protocol
errors are request-local.

## CLI Parity

The `supersonic` CLI must use the runtime-owned FLM source and resident engine
instead of retaining a second implementation. CLI-only behavior may wrap the
engine for dry-run reports, profiling, experimental speculative modes, sparse
prefill, trace taps, and stdout formatting.

Extraction is accepted only when the merged
`run_qwen36_flm_first_class_e2e.py` gate still proves:

- geo-quant FLM production or strict reuse;
- no HF snapshot requirement during FLM reuse;
- native INT4 direct coverage;
- zero BF16 fallback;
- successful generation with matching structured evidence.

## Testing and Acceptance

### Unit and Protocol Tests

Tests must cover:

- geo-quant emission and strict validation of
  `ASSET_CHAT_TEMPLATE_UTF8 = 5`;
- rejection of missing, duplicate, non-UTF-8, and uncompileable native
  template assets;
- FLM path detection and descriptor-based model resolution;
- explicit model mismatch and incompatible weight-flag rejection;
- proof that FLM startup bypasses HF metadata and bake download paths;
- direct-plan profile validation and load evidence serialization;
- Qwen3.6 `InferenceSession` feature dispatch;
- prefix-cache bypass when snapshots are unsupported;
- readiness transitions and integrity-error classification;
- normal Chat Completions, Responses, streaming, tools, reasoning, queueing,
  cancellation, and auth through a deterministic Qwen3.6 session test double.

The test double exists only at the session boundary. Protocol tests assert on
real route behavior and must not mock the route handlers themselves.

### CLI Regression Gate

The existing first-class FLM verifier must pass unchanged or with only
documented invocation updates caused by moving the engine API.

### Real ROCm Server Gate

On the local two-GPU ROCm machine, regenerate the canonical
geo-quant Qwen3.6-35B-A3B native INT4 FLM with the native chat template, build
`supersonic-serve` with HIP, and start it with `--flm-file`. The gate must:

1. Wait for readiness and assert Qwen3.6 model identity, FLM provenance,
   positive native INT4 coverage, zero fallback, and load sequence number one.
2. Exercise model list/retrieve and tokenizer endpoints.
3. Send a non-streaming Chat Completions request and receive at least one
   generated token.
4. Send a streaming Chat Completions request and observe a content or tool
   delta before the terminal event.
5. Exercise non-streaming and streaming Responses requests.
6. Perform a multi-turn coding-style tool loop: submit tools, receive a Qwen
   tool call, send the tool result, and receive a subsequent assistant result.
7. Disconnect a streaming request and verify cancellation telemetry and
   scheduler release.
8. Send a second successful request and verify the model load sequence and
   load count did not change.
9. Record startup time, first-token latency, prefill throughput, decode
   throughput, queue state, and transfer evidence in structured JSON.

The generated text need not pass a broad quality evaluation in this stage,
but tool-call structure and continuation must be syntactically usable by an
OpenAI-compatible coding client.

### Agent Client Smoke

After the protocol gate passes, run one real OpenCode or equivalent
OpenAI-compatible agent client against the warm server. The smoke must cause
at least one tool invocation and complete the tool-result continuation
without a protocol adapter specific to SuperSonic.

## Delivery Boundaries

This design is one implementation stage because all parts are required to
prove the persistent HTTP vertical slice. Work is committed and reviewed
in smaller extraction and integration tasks, but the branch is not complete
until the real ROCm server and agent-client gates pass.

The following are explicitly subsequent stages:

- continuous or paged batching across active sequences;
- Qwen3.6 in-memory or disk prefix snapshots;
- MTP, DFlash, or other speculative generation in the server;
- hipFile/GPU-direct-storage transfer;
- multi-GPU tensor, pipeline, or expert parallelism;
- broad model-family expansion;
- production orchestration such as TLS termination or distributed routing.

Those exclusions limit this stage's implementation, not the overall FLM
serving objective. The public FLM loader, persistent engine, session, metrics,
and HTTP contracts must remain suitable for those additions.
