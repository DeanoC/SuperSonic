# Qwen3-30B-A3B HIP Bring-Up

This document tracks the Qwen3-30B-A3B implementation in SuperSonic. This is
the HuggingFace `Qwen/Qwen3-30B-A3B` model, not Qwen3.6. It has a separate
Rust crate, FFI module, and HIP compilation unit so it does not share the
Qwen3.6-MoE runtime implementation beyond generic infrastructure such as
`gpu-hal`, `model-store`, and registry plumbing.

## Scope

The first PR wires the model as an explicit HIP-only INT4 bring-up target:

- model registry alias: `qwen3-30b-a3b`
- HuggingFace id: `Qwen/Qwen3-30B-A3B`
- model family: `Qwen3Moe`
- implementation crate: `crates/qwen3_moe`
- FFI module: `kernel-ffi::qwen3_moe`
- HIP source: `kernels/qwen3_moe.hip`
- bridge source: `kernels/qwen3_moe_bridge.cpp`

The current code supports config validation, checkpoint tensor enumeration,
INT4 baked tensor contract validation, baked manifest indexing, GPU upload of
the INT4 weights, descriptor pointer construction, scratch allocation, full
single-token decode math, direct INT4 lm-head sampling, and a Qwen3-owned
persistent decode chain. The persistent path is separate from the Qwen3.6-MoE
runtime and is the default runtime path; pass `--no-persistent-decode` to run
the chained per-layer A/B path.

## Model Geometry

Qwen3-30B-A3B uses dense full-attention layers plus MoE feed-forward blocks:

- hidden size: 2048
- layers: 48
- query heads: 32
- KV heads: 4
- head dim: 128
- query projection width: 4096
- experts: 128
- experts per token: 8
- MoE intermediate size: 768
- vocab size: 151936

Unlike Qwen3.5 dense models, `num_attention_heads * head_dim` is larger than
`hidden_size`. The Qwen3 config parser deliberately validates this model's
real geometry instead of assuming `q_dim == hidden_size`.

## Runtime Policy

The first bring-up lane is intentionally narrow:

- HIP only
- `--int4` required
- `--batch-size 1` only
- `--fp8-runtime`, `--kv-fp8`, `--q4km`, and `--q4km-gptq` rejected
- no serve path yet
- no teacher-forced HF logit validation path yet

The CLI accepts dry-run validation:

```bash
cargo run --release --bin supersonic -- \
  --model qwen3-30b-a3b \
  --model-dir /path/to/Qwen3-30B-A3B \
  --int4 \
  --dry-run
```

Non-dry-run execution is supported for one-token-at-a-time decode. Decode runs
through the Qwen3 persistent chain by default. The chained per-layer path
remains available for A/B testing:

```bash
cargo run --release --bin supersonic -- \
  --model qwen3-30b-a3b \
  --model-dir /path/to/Qwen3-30B-A3B \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --context-size 8 \
  --int4 \
  --no-persistent-decode
```

Use `--emit-stage-timings` to print per-token averages for embedding upload,
decode, lm-head, and sampling.

## Decode Optimization Notes

The optimized HIP path now has two decode generations.

The first optimized persistent path made three targeted changes:

- the layer decode kernel now uses 1024 threads per cooperative block, which
  improves the scalar row-dot path substantially on gfx1100-class hardware
- the default runner path uploads the per-step descriptor table once and runs
  all 48 layers through `qwen3_moe_persistent_decode_kernel`
- the lm-head projection reads the INT4 baked `lm_head.weight` directly instead
  of expanding it to a roughly 593 MiB BF16 GPU buffer at startup

On the local gfx1100 24 GiB smoke setup with `/mnt/data/models/Qwen3-30B-A3B`,
the synced one-token smoke improved from roughly 5.2 s/token in the initial
256-thread bring-up path to roughly 2.46 s/token with the 1024-thread decode
path. Persistent-vs-chained timings are currently very close because the
single-token work is compute dominated; the persistent path is still useful as
the Qwen3-owned launch surface for future multi-block/grid-barrier work.

The multi-block persistent path keeps the Qwen3 implementation separate from
Qwen3.6 and uses a Qwen3-local grid barrier plus a work-stealing counter. It
launches one resident grid for all 48 layers and parallelizes the bulk decode
phases across blocks: RMSNorm reductions, Q/K/V projections, q/k norm and
RoPE, KV append, attention score/softmax/value aggregation, O projection,
router projection, selected expert gate/up/down matvecs, and residual writes.
The chained per-layer path remains available with `--no-persistent-decode` for
A/B correctness checks.

On the same local gfx1100 setup, synced one-token context-8 timings are:

- persistent multi-block: `decode_ms_avg=55.8`, `lm_head_ms_avg=3.9`
- chained single-block fallback: `decode_ms_avg=1905.1`, `lm_head_ms_avg=4.1`

A ROCTracer HIP trace for the persistent smoke reported
`qwen3_moe_persistent_decode_kernel` at about `55.15 ms` and
`qwen3_moe_lm_head_int4_kernel` at about `3.82 ms`. Persistent-vs-chained
logits for the same prompt were bit-identical in the local smoke
(`cos=1.0`, `max_abs=0.0`, same argmax and 5/5 top-5 overlap). A context-128
8-token smoke averaged `decode_ms_avg=125.2` and generated successfully.

## INT4 Bake Contract

The Qwen3 INT4 contract uses group size 128. Quantized tensors are stored as
packed `u8` weights with BF16 `_int4_scale` and `_int4_zero` sidecars.

Quantized tensors:

- `lm_head.weight`
- per-layer `self_attn.{q,k,v,o}_proj.weight`
- per-layer fused `mlp.experts.gate_up_proj`
- per-layer fused `mlp.experts.down_proj`

Raw BF16 tensors:

- `model.embed_tokens.weight`
- `model.norm.weight`
- per-layer input and post-attention norms
- per-layer `self_attn.{q,k}_norm.weight`
- per-layer router `mlp.gate.weight`

The split HuggingFace expert tensors are expected in the source checkpoint as
`experts.N.{gate,up,down}_proj.weight`. The baked contract replaces those with
the fused expert tensors consumed by the Qwen3 runtime.

### Producer Paths

`oracle/bake_int4.py` has two Qwen3-MoE producer modes:

- default `int4-gptq`: loads the HF model and runs the normal calibration
  workflow. On the 24 GiB GPU / 64 GiB host test machine this OOMed during the
  full `128 x 2048` calibration path with CPU/disk offload.
- `--qwen3-raw-minmax`: Qwen3-only fallback that never instantiates the HF
  model. It reads safetensors directly, writes the same runtime INT4 layout,
  fuses per-expert HF tensors into the runtime expert slabs, and uses
  calibration-free min/max scale search. This is OOM-safe and decode-capable,
  but should not be represented as Hessian-calibrated GPTQ quality.

Example OOM-safe bake:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3-30B-A3B \
  --device cuda \
  --qwen3-raw-minmax
```

## PR-Ready Verification

The expected local checks for this bring-up are:

```bash
cargo test -p qwen3_moe
cargo test -p supersonic-core qwen3
cargo test -p kernel-ffi qwen3_moe --lib
cargo check -p runner --bin supersonic
cargo check -p supersonic-runtime
python -m py_compile oracle/upload_bake.py
python -m py_compile oracle/bake_int4.py
```

These checks validate the model geometry, registry entry, descriptor ABI,
INT4 bake contract, Qwen3-owned baked index, and the runner/runtime policy
surface.

Local runtime smoke commands used during bring-up:

```bash
cargo run --release --bin supersonic -- \
  --model qwen3-30b-a3b \
  --model-dir /mnt/data/models/Qwen3-30B-A3B \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --context-size 8 \
  --int4 \
  --emit-stage-timings

cargo run --release --bin supersonic -- \
  --model qwen3-30b-a3b \
  --model-dir /mnt/data/models/Qwen3-30B-A3B \
  --prompt "Hello, can you explain HIP kernels in one sentence?" \
  --max-new-tokens 8 \
  --context-size 128 \
  --int4
```
