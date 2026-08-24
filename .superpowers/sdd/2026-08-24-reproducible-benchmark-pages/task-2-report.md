# Task 2 Report: Engine Adapters

## Implementation

Implemented strict benchmark engine adapters in `tools/benchmark/adapters.py` and exported the public adapter API from `tools/benchmark/__init__.py`.

Added:

- `AdapterInputs` for explicit `model_dir`, `artifact`, `peer_artifact`, `chat`, `device`, and optional `context_size`.
- `ParsedOutput` for normalized engine output: engine identity, generated text, token ids, prompt/generated token counts, decode timing, and derived tok/s.
- `build_command(engine, case, inputs)` with argv-only adapters for:
  - `supersonic`: preserved the public CLI (`--model qwen3.8-27b`, `--model-dir`, `--gguf-file`, `--ignore-eos`, `--emit-generated-json`, `--emit-stage-timings`, `--device`, optional `--chat`, optional `--context-size`, optional `--speculative-decode` for MTP).
  - `llama-cpp`: uses the separate `peer_artifact`, deterministic greedy flags, and rejects unsupported modes.
- `parse_output(engine_name, stdout)` with fail-closed parsing:
  - requires exactly one `[generated_json]`, `[tokens]`, and `[result]` line;
  - rejects missing deterministic output, duplicate result lines, non-integer/negative token ids, non-finite timings, non-positive timings, inconsistent token counts, and inconsistent decode math;
  - keeps llama.cpp version identity by extracting a single `version: ...` line;
  - validates a single stage-timings line when present.

Added fixtures:

- `tests/benchmark_fixtures/supersonic-run.log`
- `tests/benchmark_fixtures/llama-cpp-run.log`

Added tests in `tests/test_benchmark_adapters.py` covering:

- SuperSonic public argv contract
- chat + MTP flagging
- fail-closed engine scope and unsupported mode handling
- llama.cpp peer artifact routing
- generated text, token id, version, and timing parsing
- duplicate result rejection
- non-finite, negative, inconsistent, and missing deterministic output rejection

## Files

- Modified: `tools/benchmark/__init__.py`
- Added: `tools/benchmark/adapters.py`
- Added: `tests/test_benchmark_adapters.py`
- Added: `tests/benchmark_fixtures/supersonic-run.log`
- Added: `tests/benchmark_fixtures/llama-cpp-run.log`

## RED/GREEN Evidence

### RED

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters -v
```

Output:

```text
FAILED (failures=8)
...
AssertionError: tools.benchmark.adapters is absent
```

This was the expected red state before the adapter module existed.

### GREEN (focused adapter tests)

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters -v
```

Output:

```text
Ran 8 tests in 0.012s

OK
```

### GREEN (new + legacy parser coverage)

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters tests.test_qwen38_reproducibility -v
```

Output:

```text
Ran 11 tests in 0.020s

OK
```

## Full CPU Python Suite Evidence

Command:

```bash
python3 -m unittest discover -s tests -v
```

Output:

```text
Ran 101 tests in 0.547s

OK
```

## Self-review

- Verified the adapter stays inside the narrow benchmark scope and does not change the public SuperSonic runner contract.
- Verified all command construction is tuple-based argv only; no shell templates, eval, or command strings.
- Verified llama-cpp uses `peer_artifact` instead of the SuperSonic GQH artifact.
- Verified parser failure modes are explicit and deterministic.
- Verified no whitespace errors with `git diff --check`.

## Concerns

- None at the adapter layer after Fix Round 1. llama.cpp now parses a raw timing transcript directly and sources version identity from the pinned manifest contract.

## Fix Round 1

Addressed review findings by replacing the invented llama.cpp marker protocol with direct parsing of a realistic raw combined `llama-cli` transcript. The llama.cpp adapter now:

- sources engine version from the pinned manifest contract instead of the run output;
- emits explicit `--perf`, `--show-timings`, and `--no-display-prompt`;
- maps non-chat to `--no-conversation`;
- maps chat to `--conversation --single-turn`; and
- parses raw `prompt eval time`, `eval time`, and `total time` lines while allowing deterministic peer text without token IDs.

### Covering test files

- `tests/test_benchmark_adapters.py`
- `tests/test_qwen38_reproducibility.py`

### RED

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters -v
```

Output:

```text
FAILED (failures=3, errors=1)
...
AssertionError: '--perf' not found in ('llama-cli', '--model', '/peers/qwen38-llama.gguf', ...
AssertionError: '--conversation' not found in ('llama-cli', '--model', '/peers/qwen38-llama.gguf', ...
ValueError: output must contain exactly one version line
```

### GREEN (adapter tests)

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters -v
```

Output:

```text
Ran 10 tests in 0.020s

OK
```

### GREEN (adapter + reproducibility tests)

Command:

```bash
python3 -m unittest tests.test_benchmark_adapters tests.test_qwen38_reproducibility -v
```

Output:

```text
Ran 13 tests in 0.020s

OK
```

### GREEN (full CPU Python suite)

Command:

```bash
python3 -m unittest discover -s tests -v
```

Output:

```text
Ran 103 tests in 0.587s

OK
```

### Fix-round self-review

- Verified the llama.cpp fixture now reflects raw timing output rather than invented local markers.
- Verified peer text parsing works without token IDs while keeping strict prompt/generated timing consistency checks.
- Verified the pinned engine version comes from `benchmarks/engines/llama-cpp.toml` plus `tools/external/llama-cpp-version.txt`, not the transcript.
- Verified the command remains argv-only and keeps the public SuperSonic CLI unchanged.
