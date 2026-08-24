# Artifact format

The public weight input is a project-specific GQH GGUF for `qwen3.8-27b`.
It is not a generic GGUF compatibility promise. The GGUF supplies model
weights; the model directory supplies the rest of the startup contract.

## Required pair

`--model-dir` must contain:

- `config.json` with the fixed Qwen3.8-27B geometry;
- `tokenizer.json` with tokenizer data;
- `tokenizer_config.json` containing the chat template only when `--chat` is
  used.

`--gguf-file` must point to the matching GQH artifact. The canonical workflow
may also configure a separate 8192-context GQH file for its extended crawl;
that file is an additional artifact role, not a replacement for the primary
GGUF.

## Encodings and headers

The retained reader covers the encodings exercised by the canonical artifact,
including Q2_K, Q3_K, Q8_0, GGML-K variants, ROCmFP mixes, and GQH qtypes
108–111. GQH tensors carry their required sidecar headers and pointer
registrations. A file missing a required header or containing an incompatible
tensor geometry is rejected before device allocation where practical.

## Cheap preflight

The artifact preflight checks existence, readability, and required model
configuration/tokenizer sidecars. It does not claim to prove tensor parity;
the Rust startup validator and the serial GQH correctness tests perform header,
geometry, decode, and matvec checks.

Run the local preflight with the same variables used by CI:

```bash
export SUPERSONIC_GQH_GGUF=/path/to/qwen38.gqh.gguf
export SUPERSONIC_QWEN38_MODEL_DIR=/path/to/Qwen3.8-27B
export SUPERSONIC_GQH_8192_GGUF=/path/to/qwen38-8192.gqh.gguf
python3 tools/check-qwen38-artifacts.py --require-8192
```

The CPU workflow can compile and run codec checks without these files. The
self-hosted `gfx1201` workflow sets `SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1` and
fails during preflight when a configured file is missing, unreadable, or the
model directory lacks its required sidecars.
