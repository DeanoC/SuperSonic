# Task 14 Report: Qwen3.6 FLM Server Protocol Harness

Date: 2026-07-31

## Status

Implemented and committed the first-class real-server protocol harness. The
deterministic harness contracts, server route tests, HIP release build, real
Chat Completions and Responses compatibility smoke, and real cancellation
release check pass. The full real acceptance run intentionally remains red at
the required model-generated Chat tool-call gate because the canonical FLM
does not emit a valid protocol tool call.

## Requirements

- Start `supersonic-serve` from the FLM alone with `--flm-file`, HIP
  backend/device/context, loopback host/unused port, API key, and
  `--no-download`.
- Poll authenticated `/ready` and terminate/reap the new server process group
  on success, startup failure, protocol failure, timeout, and cancellation.
- Use the official OpenAI JavaScript SDK for real Chat Completions and
  Responses behavior against one resident server process.
- Constrain auth envelopes, model list/retrieve, tokenize/detokenize, Chat,
  legacy Completions, Responses create/retrieve/delete, streaming deltas and
  terminal events, terminal usage, reasoning request behavior, usage
  accounting, and repeated warm requests.
- Require model-generated coding tool calls, tool-result continuations, and
  subsequent assistant output through both Chat Completions and Responses.
- Abort a real stream after its first substantive delta, then require
  authenticated health and metrics evidence that active and queued work
  returned to zero without another model load.
- Strictly validate exact Qwen3.6 FLM identity/load evidence and the prescribed
  structured report. Reject booleans as integers, missing sections, duplicate
  Prometheus samples, and non-finite numbers.
- Preserve deterministic mock coverage for lifecycle/report behavior and real
  HIP/FLM coverage for route and model-dependent behavior.

## Files

- Created `tests/gfx1100/run_qwen36_flm_server_e2e.py`.
- Created `tests/test_qwen36_flm_server_e2e.py`.
- Expanded `scripts/openai_compat_smoke.mjs`.
- Created `scripts/openai_agent_tool_smoke.mjs`.
- Updated `docs/server.md`.
- Updated `docs/testing.md`.

No server/runtime implementation files or unrelated edits were changed.

## RED Evidence

1. Initial required command:

   ```bash
   python3 -m unittest tests.test_qwen36_flm_server_e2e -v
   ```

   Result: 18 errors because
   `tests/gfx1100/run_qwen36_flm_server_e2e.py` did not exist.

2. After implementing the Python lifecycle/report core, the same command ran
   18 tests: 16 passed and 2 failed because
   `scripts/openai_agent_tool_smoke.mjs` did not exist.

3. The first real run exposed the canonical wire value `backend: "HIP"`.
   The evidence test fixture was changed first and failed against the old
   lowercase validator; the validator was then corrected and the focused test
   passed.

4. A reasoning report fixture that truthfully recorded
   `reasoning_observed: false` failed while the validator required positive
   reasoning content. The contract was changed to require an accepted
   reasoning request, a boolean observation, an assistant result, and no
   visible `<think>` leakage. Positive extraction remains covered by the
   server route suite.

5. Source-contract tests were added before tightening the tool prompt and
   before moving cancellation ahead of model-dependent tool calls. Each test
   failed first, then passed after the corresponding script change.

6. The latest source-contract test first failed because the real cancellation
   result had no standalone diagnostic. It passed after adding the
   `cancellation_release` structured line.

## GREEN Evidence

The final deterministic command passed 19/19 tests:

```bash
python3 -m unittest tests.test_qwen36_flm_server_e2e -v
```

It covers the exact server argv, ephemeral loopback port, readiness success,
startup timeout, early-exit diagnostics, process-group cleanup, strict
capability/metric evidence, unchanged single-load invariants, structured SDK
markers, strict report validation, JavaScript syntax, tool-loop declarations,
raw-output failures, and cancellation-before-tool ordering.

Syntax checks passed:

```bash
python3 -m py_compile \
  tests/gfx1100/run_qwen36_flm_server_e2e.py \
  tests/test_qwen36_flm_server_e2e.py
node --check scripts/openai_compat_smoke.mjs
node --check scripts/openai_agent_tool_smoke.mjs
```

The requested server suite passed 46/46 tests:

```bash
CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo test -q -p server
```

The HIP release build passed:

```bash
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100 \
  CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target \
  cargo build --release -p server
```

Build result: release profile completed successfully in 44.78 seconds.
Compiler warnings were pre-existing warnings in kernel/runtime files outside
Task 14.

## Real Server Smoke

Command:

```bash
python3 tests/gfx1100/run_qwen36_flm_server_e2e.py \
  --binary /home/deano/projects/SuperSonicBase/target/release/supersonic-serve
```

Inputs:

- FLM:
  `/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm`
- Server:
  `/home/deano/projects/SuperSonicBase/target/release/supersonic-serve`
- SDK: `openai@6.49.0`
- Backend/device/context: `hip`, device `0`, context `4096`

The pre-request real evidence gate passed exact model/source/load checks:

- Model `qwen3.6-35b-a3b`
- Source `flm`
- Native INT4 direct weights `330`
- BF16 fallback weights `0`
- Load sequence `1`
- Source open count `1`
- Model loads total `1`
- Idle active and queued requests

The real compatibility smoke passed:

- Missing auth returned HTTP 401 with the OpenAI authentication error envelope.
- Model list and retrieve passed.
- Tokenize/detokenize passed.
- Chat non-streaming passed.
- Chat streaming produced substantive delta, terminal finish, and terminal
  usage.
- Legacy Completions passed.
- Responses create/retrieve/delete passed.
- Responses streaming produced output delta, `response.completed`, and usage.
- The reasoning request was accepted, returned an assistant result, and did
  not leak `<think>` tags. This artifact did not emit separate reasoning
  content.
- Chat and Responses usage totals were internally consistent.
- A repeated warm request passed.

Client-observed real throughput:

- First token: `0.283728523` seconds
- Prefill: `66.96542102677495` tokens/second
- Decode: `70.37276196913793` tokens/second

The real cancellation gate passed before model-dependent tool calls:

```json
{
  "saw_delta": true,
  "scheduler_released": true,
  "active_requests": 0,
  "queued_requests": 0,
  "release_seconds": 0.10438918200000001
}
```

The harness then failed as required when Chat did not produce a valid tool
call. Captured raw assistant content began with:

```text
{"path": "src/lib.rs"}
```

It was emitted as ordinary assistant content, followed by degraded tokens,
rather than as `tool_calls`. The harness reaped the server process group; a
subsequent `pgrep -af '[s]upersonic-serve'` found no remaining server.

## Commit

Task implementation:

`d52d543ae2caedbe13f4585fa77663ca90388476`

Commit message:

`test(server): gate Qwen3.6 FLM agent serving`

## Concerns

- The canonical FLM currently fails the first mandatory real tool-call gate.
  It emitted bare JSON as assistant text instead of the chat template's
  protocol tool-call form. The harness correctly preserves the raw response
  and fails instead of synthesizing or accepting the call.
- Because Chat tool-call generation failed, neither the Chat tool-result
  continuation nor the Responses tool-call/result continuation ran in the
  real acceptance process, and no final structured acceptance JSON was
  written. Their deterministic SDK/source contracts are green.
- The real reasoning request was accepted and safely surfaced, but this
  artifact produced no separate reasoning content. Positive reasoning
  extraction is covered by the Rust route tests.
- The server suite emits existing kernel/backend compiler warnings unrelated
  to the owned Task 14 files.
