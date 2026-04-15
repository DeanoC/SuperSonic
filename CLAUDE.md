# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperSonic is a persistent decode megakernel for LLM inference on AMD GPUs (ROCm/HIP), targeting Qwen3.5 models. Instead of launching separate GPU kernels for each transformer operation, the entire decode step (attention + MLP + norms) runs in a single monolithic kernel to minimize launch overhead and maximize occupancy.

## Build & Run

**Prerequisites**: Rust toolchain, ROCm/HIP stack (hipcc, amdhip64), Python 3 with PyTorch + transformers (for oracle validation only).

**Build**:
```bash
cargo build --release
# Override GPU arch if rocminfo isn't available:
HIP_ARCH=gfx90a cargo build --release
```

The `kernel-ffi` build script automatically compiles the HIP kernels via hipcc during `cargo build`. It detects GPU architecture from `rocminfo` or the `HIP_ARCH` env var.

**Run**:
```bash
cargo run --release --bin supersonic -- \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --validate  # optional: compare logits against PyTorch oracle
```

The `--model` flag selects a model variant from the registry (`crates/runner/src/registry.rs`). At startup, SuperSonic detects the GPU architecture and VRAM, looks up the (model, backend, arch) combination in the registry, and exits with a clear error if the combo is unsupported or VRAM is insufficient. No fallback to unoptimized paths.

## Architecture

### Workspace Crates

- **`gpu-hal`** — Low-level HIP bindings. `GpuBuffer` provides typed device memory with shape/dtype metadata. Operations: alloc, H2D/D2H/D2D copy, memset.
- **`kernel-ffi`** — FFI bridge between Rust and the HIP megakernel. `build.rs` compiles `kernels/*.hip` and `kernels/*.cpp` into a static library. Exports `persistent_decode`, `rms_norm`, `standalone_matvec`, `query_gpu_info`.
- **`qwen35`** — Model implementation: config loading, safetensors weight loading (memory-mapped), model state (KV caches, conv/recurrent state), RoPE tables, scratch buffer allocation, and `DecodeLayerDesc` builder. Kernel-specific parameters (scratch sizes, KV chunk size, weight prefix) are passed in from the caller.
- **`runner`** — CLI binary (`supersonic`). Contains the model/backend/GPU registry (`registry.rs`) and `DecodeEngine` which orchestrates the decode loop. Uses a PyTorch oracle (`oracle/`) for prefill and optional logit validation.

### Key Design Patterns

**Hybrid attention**: Qwen3.5 alternates linear attention layers with full attention (every 4th layer). Linear layers use conv state + recurrent state; full layers use traditional KV caches.

**KV cache**: Pre-allocated in configurable chunks (default 256 tokens). `grow_seq_dim` handles expansion via strided D2D copies.

**Work-stealing matmul**: The kernel uses an atomic counter + barrier (`sync_buf` in scratch) for load-balanced matrix-vector operations across wavefronts.

**Oracle workflow**: Python runs HuggingFace prefill, exports base64-encoded state (hidden, KV caches, conv/recurrent) via JSON. Rust loads this state and runs decode, optionally comparing logits per step.

### GPU Kernel Source

The megakernel lives in `kernels/full_attention.hip` with its Rust-callable bridge in `kernels/full_attention_bridge.cpp`. Layer configuration is passed via `DecodeLayerDesc` (a `#[repr(C)]` struct in `kernel-ffi/src/layer_desc.rs`).

### Model/Backend/GPU Registry

`crates/runner/src/registry.rs` defines the set of supported (model, backend, GPU arch) combinations. Each entry specifies kernel parameters (scratch buffer sizes, weight prefix, KV chunk size) and minimum VRAM. To add support for a new model size or GPU architecture, add a new `RegistryEntry` to the `REGISTRY` static slice.
