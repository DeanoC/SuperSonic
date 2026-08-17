# Kernel Build Groups

This page records the first scaffold for splitting `crates/kernel-ffi/build.rs`
into explicit backend/model compile groups. It does not change default build
behavior: when a backend is selected, the build script still compiles every
bridge group for that backend.

The checked source of truth is
[`crates/kernel-ffi/kernel-groups.toml`](../../crates/kernel-ffi/kernel-groups.toml).
Validate it with:

```bash
python3 tools/check-kernel-groups.py
```

## Current Groups

| Group | Backend | Owns |
| --- | --- | --- |
| `hip-qwen35` | HIP | Qwen3.5 dense attention, 4B attention, and prefill helpers. |
| `hip-gemma4` | HIP | Gemma 4 HIP bridge. |
| `hip-phi4` | HIP | Phi-4 HIP bridge. |
| `hip-dflash` | HIP | DFlash draft HIP bridge. |
| `hip-qwen36-moe` | HIP | Qwen3.6 MoE HIP bridge, persistent decode, and batched prefill support sources. |
| `hip-gqh` | HIP | GQH decode and fused dequant-matvec. |
| `hip-qwen3-moe` | HIP | Qwen3 MoE HIP bridge. |
| `cuda-qwen35` | CUDA | Qwen3.5 dense CUDA attention, 4B attention, and prefill helpers. |
| `cuda-llama31` | CUDA | Llama 3.1 certified-KV CUDA bridge. |
| `cuda-phi4` | CUDA | Phi-4 CUDA bridge. |
| `cuda-gemma4` | CUDA | Gemma 4 CUDA bridge. |
| `cuda-qwen36-moe` | CUDA | Qwen3.6 MoE CUDA bridge and shared Qwen3.6 source closure. |
| `metal-host-stubs` | Metal | Metal host/native link layer compiled by `cc` on macOS. |

## Guardrails

- Existing `SUPERSONIC_BACKENDS=hip`, `cuda`, `metal`, and `auto` behavior must
  stay compatible until grouped builds have validation coverage.
- Group ids describe backend and model family, not incidental source layout.
- The manifest must name bridge sources, support kernel/header sources, and
  Rust wrapper modules that reviewers should inspect together.
- A future grouped build must preserve the broad default path until at least one
  representative backend proves old-vs-new parity.

## Next Split Step

The next code PR can add an opt-in grouped build selector, such as a guarded
`SUPERSONIC_KERNEL_GROUPS` environment variable, but it should leave unset
behavior identical to the current broad backend build.
