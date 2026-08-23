# Contributor guidance

SuperSonic is a performance-specialized ROCm/HIP inference engine for one
public model: Qwen3.8-27B loaded from a project-specific GQH GGUF. The public
path supports single-sequence greedy generation, optional NextN/MTP generation,
and AMD `gfx1100` and `gfx1201` targets. Unsupported combinations fail
explicitly; do not add silent fallbacks or compatibility shims.

The canonical contributor guidance is this file. [CLAUDE.md](CLAUDE.md) is only
a pointer for tools that look for that filename.

## Repository ownership

Keep responsibilities within these crate boundaries:

- `crates/core`: Qwen3.8 model identity and the AMD architecture registry.
- `crates/gpu-hal`: HIP allocation, copies, events, and device operations.
- `crates/kernel-ffi`: HIP decode, prefill, GQH, and NextN/MTP FFI bridges.
- `crates/model-store`: custom GGUF/GQH readers and internal codec foundations.
- `crates/qwen38`: Qwen3.8 configuration, weights, state, and descriptors.
- `crates/runtime`: tokenization, chat rendering, prefill, decode, generation,
  and NextN/MTP state transitions.
- `crates/runner`: the `supersonic` binary, CLI, startup validation, and
  structured output.

The `tools/` directory owns narrow manifest, artifact, device-selection, and
documentation validators. The active public information architecture is the
README plus the six documents under `docs/` linked from it.

## Public contract changes

Keep `--model qwen3.8-27b` explicit. `--model-dir` and `--gguf-file` are a
paired startup contract: the former supplies configuration, tokenizer data,
and the chat template; the latter supplies custom GQH weights. HIP is the
runtime target selected by `HIP_ARCH`, `HIP_VISIBLE_DEVICES`, `ROCM_PATH`, and
`HIP_PATH` as appropriate for the host.

When a combination is outside this contract, reject it at parsing, startup,
or manifest validation with an actionable error. Do not reintroduce a broad
model registry, alternate weight source, multi-sequence path, or unvalidated
device fallback merely to make a test pass.

## Internal FLM foundation

FLM codec and model-store foundations remain contributor-only background work.
They must stay compile-tested and unreachable from the public runner contract.
Do not add a public loader flag, startup fallback, download route, or product
claim for FLM. Promotion requires a separate design and the same artifact,
correctness, generation, and performance gates as the direct GQH path.

## Kernel ABI and historical names

Some external wire keys and FFI helper symbols retain historical spellings.
For example, `build_int4_scale_descs` carries GQH sidecar/header pointers even
though its name predates the current product. Treat such names and layouts as
ABI until all call sites and C++ layout assertions support a coordinated
rename. Never rename a symbol or reorder a descriptor field as documentation
cleanup alone; add a focused parity test first and preserve the wire contract.

## Test tiers

Run the smallest relevant tier first, then the complete affected gate:

1. CPU-safe unit and contract tests cover GQH codec/header/pointer behavior,
   model geometry, CLI rejection, startup preflight, MTP acceptance policy,
   manifests, and active-document links and terms.
2. `cargo check --workspace --all-targets` and `cargo fmt --all --check` keep
   the retained workspace buildable without a GPU.
3. The serial `gfx1201` artifact gate validates the configured files, official
   vectors, deterministic decode/chat generation, and ordinary-versus-MTP
   token equality with `RUST_TEST_THREADS=1` and explicit ignored-test
   selection.
4. Performance telemetry runs only after correctness and records the exact
   commit, target, artifact, workload, and measurement method.

Artifact-dependent local tests may report a documented skip when no artifact
is configured. A configured CI artifact that is missing or unreadable must
fail preflight; it must never become a passing skip.

## Documentation and review

Keep public docs focused on measured Qwen3.8 ROCm performance, artifact pairing,
and reproducible gates. Do not preserve obsolete feature histories in the
active tree; Git history is the archive. Run the active-doc checker after any
public documentation change:

```bash
python3 tools/check-active-docs.py
python3 -m unittest tests.test_active_docs -v
git diff --check
```

Before claiming a change is complete, run the full relevant command and inspect
its exit status and output. A warning, skipped configured test, missing artifact,
or unexplained token mismatch is a review blocker.
