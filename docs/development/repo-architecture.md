# Repository architecture

This document is contributor guidance for the retained Qwen3.8-27B ROCm/HIP
product. The public command enters through `crates/runner`, loads a custom GQH
GGUF with a model-directory sidecar pair, and executes through the runtime,
HIP kernel bridges, and `gpu-hal`.

## Ownership map

| Area | Owner | Responsibility |
| --- | --- | --- |
| Model identity | `crates/core` | Qwen3.8 identity and `gfx1100`/`gfx1201` registry data |
| Device operations | `crates/gpu-hal` | HIP allocation, copies, events, and device queries |
| Kernel bridge | `crates/kernel-ffi` | HIP decode/prefill, GQH codec, and NextN/MTP FFI |
| Artifact loading | `crates/model-store` | Custom GGUF/GQH readers and codec foundations |
| Model state | `crates/qwen38` | Configuration, weights, descriptors, and model state |
| Generation | `crates/runtime` | Tokenization, chat rendering, prefill, decode, and MTP state |
| Product entry point | `crates/runner` | CLI, host-side validation, generation, and structured output |

Validators under `tools/` own the support matrix, kernel groups, artifact
preflight, device selection, tool inventory, and active documentation checks.
The CPU workflow and serial `gfx1201` workflow are the executable definitions
of the two test tiers described in [testing](../testing.md).

## Runtime flow

```text
runner CLI
  -> model-directory and GQH preflight
  -> qwen38 loader and model state
  -> runtime prefill/decode and optional NextN/MTP
  -> kernel-ffi HIP bridges
  -> gpu-hal HIP device operations
```

The loader is intentionally strict. `config.json`, tokenizer data, and the
chat template come from `--model-dir`; weights and GQH sidecars come from
`--gguf-file`. A missing role or incompatible geometry fails before a large
device allocation where practical.

## ABI notes

The public product name does not imply that every private symbol has already
been renamed. A small number of wire keys and helper names retain historical
spellings for compatibility with the GQH artifact and C++ bridge. In
particular, the descriptor builder helper that creates INT4 scale descriptors
also carries GQH header pointers. Preserve its layout and the corresponding
FFI assertions until a dedicated rename changes both sides together.

When changing a descriptor, kernel argument, qtype mapping, or sidecar pointer:

1. add a focused CPU-safe layout or codec test;
2. run the GQH pointer/header tests;
3. run the serial artifact gate on `gfx1201` when an artifact is configured;
4. document any historical wire spelling in the code comment and review.

## Internal FLM foundation

The model-store FLM codec is compile-tested contributor background. It is not a
public runner input and must not become reachable through an implicit startup
fallback. Changes to this foundation need CPU-safe round-trip coverage and
must preserve the GQH reader boundary.

For public behavior and review commands, see [AGENTS.md](../../AGENTS.md),
[build and run](../build-and-run.md), and [testing](../testing.md).
