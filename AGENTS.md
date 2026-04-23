# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SuperSonic is a persistent decode megakernel for LLM inference on AMD GPUs (ROCm/HIP), targeting two model families:
- **Qwen3.5** (0.8B, 2B, 4B, 9B) — hybrid linear + full-attention; supports BF16, FP8 runtime-dequant, INT4 GPTQ, FP8 KV cache, and batched decode.
- **Gemma 4** (E2B ~5B, E4B ~8B) — pure-attention with sliding + full layers and Per-Layer Embeddings; supports BF16 and INT4 GPTQ.

Instead of launching separate GPU kernels for each transformer operation, the entire decode step (attention + MLP + norms + PLE for Gemma 4) runs in a single monolithic kernel to minimize launch overhead and maximize occupancy. Each model family has its own megakernel (BF16 + INT4 variants) in an isolated compilation unit — hipcc's codegen on gfx11xx is fragile and cross-contamination between the Qwen and Gemma kernels caused regressions in the past.

## Build & Run

**Prerequisites**: Rust toolchain, ROCm/HIP stack (hipcc, amdhip64), Python 3 with PyTorch + transformers (for oracle validation only).

**Build**:
```bash
cargo build --release
# Override GPU arch if rocminfo isn't available:
HIP_ARCH=gfx90a cargo build --release
```

The `kernel-ffi` build script automatically compiles the HIP kernels via hipcc during `cargo build`. It detects GPU architecture from `rocminfo` or the `HIP_ARCH` env var.

**Run** (Qwen3.5):
```bash
cargo run --release --bin supersonic -- \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --validate  # optional: compare logits against PyTorch oracle
```

**Run** (Gemma 4):
```bash
cargo run --release --bin supersonic -- \
  --model gemma4-e2b \
  --model-dir /path/to/gemma-4-E2B \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --int4  # optional: use INT4 bake (must be baked first, see below)
```

The `--model` flag selects a model variant from the registry (`crates/runner/src/registry.rs`). At startup, SuperSonic detects the GPU architecture and VRAM, looks up the (model, backend, arch) combination in the registry, and exits with a clear error if the combo is unsupported or VRAM is insufficient. No fallback to unoptimized paths.

## Architecture

### Workspace Crates

- **`gpu-hal`** — Low-level HIP bindings. `GpuBuffer` provides typed device memory with shape/dtype metadata. Operations: alloc, H2D/D2H/D2D copy, memset, and RAII `GpuEvent` for kernel timing.
- **`kernel-ffi`** — FFI bridge between Rust and the HIP megakernels. `build.rs` compiles `kernels/*.hip` and `kernels/*.cpp` into a static library. Exports the Qwen `persistent_decode` entry plus the Gemma 4 `g4::*` family (primitives, `fused_attn_block[_int4]`, `fused_mlp_ple[_int4]`, `persistent_decode[_int4]`).
- **`model-store`** — Weight baking and loading. Converts HuggingFace safetensors into an optimized binary format (`weights.bin` + `manifest.json`) with 4096-byte aligned tensors and precomputed transforms. See "Weight Baking" section below.
- **`qwen35`** — Qwen3.5 model implementation: config loading, weight loading (safetensors or baked store), model state (KV caches, conv/recurrent state), RoPE tables, scratch buffer allocation, and `DecodeLayerDesc` builder.
- **`gemma4`** — Gemma 4 model implementation: config loading (sliding vs. full attention per layer, kv-share mapping), tensor specification, and RoPE helpers. Per-Layer Embeddings (PLE) and the tied lm-head are computed in the runner engine, not the kernel.
- **`runner`** — CLI binary (`supersonic`) plus family-specific engines (`decode_engine.rs` for Qwen, `gemma4_engine.rs` + `gemma4_int4_engine.rs` for Gemma 4). Contains the model/backend/GPU registry (`registry.rs`) and orchestrates prefill + decode. Uses PyTorch oracles (`oracle/`) for optional validation against the HF reference.

### Key Design Patterns

**Hybrid attention (Qwen3.5)**: Qwen3.5 alternates linear attention layers with full attention (every 4th layer). Linear layers use conv state + recurrent state; full layers use traditional KV caches.

**Sliding + full attention (Gemma 4)**: Gemma 4 alternates 4 sliding (window=512, head_dim=256) with 1 full-attention (head_dim=512) layer. `num_kv_shared_layers` tells later layers to inherit earlier layers' K/V caches via pointer aliasing in the descriptor array — no intra-kernel replication.

**KV cache**: Pre-allocated in configurable chunks. Qwen uses default 256 tokens with `grow_seq_dim` for expansion. Gemma 4 pre-allocates the full `max_t` per layer; shared layers alias the source layer's buffer.

**Work-stealing matmul**: Kernels use an atomic counter + barrier (`sync_buf` / `g4_grid_barrier`) for load-balanced matrix-vector operations across wavefronts. INT4 matmuls use an 8-wide packed-byte dequant path (`g4_int4_dequant_8`) that's bit-exact with the Python GPTQ bake.

**Oracle workflow**: Python runs HuggingFace prefill/decode, exports base64-encoded state (hidden, KV caches, conv/recurrent for Qwen; KV caches + per-layer PLI for Gemma 4) via JSON. Rust loads this state and runs decode, optionally comparing logits per step.

### GPU Kernel Source

Each model family lives in its own compilation unit — hipcc on gfx11xx is sensitive to cross-contamination and merging kernels has caused codegen regressions.

- **Qwen3.5**: `kernels/full_attention.hip` + `kernels/full_attention_bridge.cpp`. Layer config via `DecodeLayerDesc` (`kernel-ffi/src/layer_desc.rs`).
- **Qwen3.5 4B/larger**: `kernels/full_attention_4b.hip` — wider-tensor variant with different matmul tiling (8-wide INT4, wave-per-row for matvec).
- **Gemma 4**: `kernels/gemma4.hip` + `kernels/gemma4_bridge.cpp`. Layer config via `Gemma4DecodeLayerDesc` + parallel-struct `Gemma4Int4ScaleDesc` for INT4 scale/zero tables (`kernel-ffi/src/gemma4.rs`).

### Model/Backend/GPU Registry

`crates/runner/src/registry.rs` defines the set of supported (model, backend, GPU arch) combinations. Each entry specifies kernel parameters (scratch buffer sizes, weight prefix, KV chunk size) and VRAM budget. To add support for a new model size or GPU architecture, add a new `RegistryEntry` to the `REGISTRY` static slice.

### Weight Baking

On first run, SuperSonic automatically bakes HuggingFace safetensors into an optimized binary package stored at `{model_dir}/.supersonic/v{VERSION}/`. Subsequent runs load from the baked format — a single mmap + H2D copy per tensor with zero parsing or CPU transforms.

Bake-time transforms for Qwen3.5 linear attention layers:
- Conv1d weight squeeze: `[C_out, 1, K]` → `[C_out, K]`
- dt_bias reshape: `[H]` → `[1, 1, H]`
- A_log precompute: F32 exp() → BF16, reshape to `[1, 1, H]`

**INT4 GPTQ bake**: `--int4` selects a separate `.supersonic/v{VERSION}-int4-gptq/` directory containing packed-u8 weights + BF16 scale/zero triples at group_size=128. INT4 bakes are produced by `oracle/bake_int4.py` (Qwen) or `oracle/bake_int4_gemma4.py` (Gemma 4), both of which run GPTQ calibration over WikiText-2 (or a similar corpus) and self-verify each tensor. The Rust runtime is a pure reader — all calibration work lives in Python. The Gemma 4 bake is resumable (per-layer checkpoint) because 128×2048 calibration on shared-memory APUs is tight on host RAM.

Use `--no-bake` to bypass baking and load directly from safetensors (Qwen only; Gemma 4 uses safetensors directly when `--int4` is not set). Baked packages are version-checked; changing `FORMAT_VERSION` or `CONVERTER_VERSION` in `model-store` triggers automatic re-bake.

**Release-hosted bakes**: when a local bake is missing, SuperSonic auto-downloads the matching `{model}-{variant}-fmt{N}-cvt{M}.tar.zst` from the `DeanoC/SuperSonic` GitHub release tagged `bakes-v{FORMAT_VERSION}`. This is the only way small-VRAM machines can run INT4 variants whose calibration OOMs locally. Producers publish bakes with `oracle/upload_bake.py`; `--no-download` disables on the consumer. See `docs/bake-distribution.md`.
