# PR 4c — Qwen3.6-MoE Multi-Layer Driver

Plan for stitching the parity-tested single-block kernels (full-attn,
linear-attn, FFN — all with INT4 paths after PR 4b6) into a real decode
that runs all `num_hidden_layers=40` layers and emits a token. This is
the integration milestone.

## Where we are after PR 4b6

Every quantizable tensor in a Qwen3.6-MoE layer has a parity-tested
INT4 dequant path against `oracle/bake_int4.py`'s reconstruction:

- **Full-attention** (`kernels/qwen36_moe.hip` `qwen36_moe_attn_step_kernel`):
  q_proj, k_proj, v_proj, o_proj.
- **Linear-attention** (`qwen36_moe_linear_step_kernel`):
  in_proj_qkv, in_proj_z, out_proj. (in_proj_a, in_proj_b stay BF16.)
- **MoE FFN** (`qwen36_moe_ffn_step_kernel`):
  gate_up_proj, down_proj (per-expert), shared gate_proj/up_proj/down_proj.

Each kernel is a *single-block* parity-test driver: it processes one
layer's worth of one decode step, with `kv_len=1` and a host-allocated
workspace per call. They don't yet chain. PR 4c chains them.

The Rust runner (`crates/runner/src/qwen36_moe_engine.rs`) is dry-run
only — `run_qwen36_moe_dry_run` enumerates the bake and prints VRAM
accounting. `run` bails on any non-`--dry-run` call. Replacing that
bail with a real decode path is one of PR 4c's deliverables.

## What PR 4c needs to deliver

Three pieces. They can land in this order, each parity-gated:

1. **Multi-layer Python oracle**. Chains `reference_full_attention_layer`
   + `reference_linear_attn_layer` + `reference_moe_ffn_block` per the
   hybrid pattern (3-of-4 linear, 1-of-4 full, every 4th layer full at
   indices 3/7/11/...). Reference for parity. Synthetic mode + a
   "minimum-real" mode that loads N layers from a checkpoint to keep
   host RAM honest.

2. **Host-orchestrated multi-launch driver**. Rust-side: per-layer
   weight pointers, per-layer KV state for full-attn layers, per-layer
   conv+recurrent state for linear-attn layers, dispatch loop that
   calls the existing `attn_step_launch` / `linear_step_launch` /
   `ffn_step_launch` in sequence. One HIP launch per kernel × N layers
   per token. High launch overhead, but correct, reviewable, and
   unblocks end-to-end testing.

3. **Persistent-decode megakernel** (optional for PR 4c — could land
   in a follow-up). A single `__global__` that walks the descriptor
   array (PR 4 `qwen36_moe_descriptor_walk_stub` is the prototype)
   and dispatches per-layer-type. Same compute, single launch. Goal
   is to amortize the ~40 launches/token down to 1.

The runner-side wiring (replace the bail at `qwen36_moe_engine.rs:639`
with a real path) lands as part of (2).

## Constraints

- **VRAM**: 24 GiB on the RX 7900 XTX. INT4 bake is ~17 GiB; KV
  caches + recurrent state are additional. The dry-run accountant
  in `crates/qwen36_moe/src/weights.rs::project_int4_total_bytes`
  already projects ~16-21 GiB for INT4 + sidecars + embed.
- **Host RAM**: 64 GiB. Full BF16 35B model is ~65 GiB → won't fit.
  Multi-layer oracle must avoid instantiating the full model. Either
  load tensors per-layer on demand, or run with synthetic weights at
  smaller geometry (e.g. 4 layers × small hidden) for parity work.
- **hipcc on gfx11xx is fragile**. Per CLAUDE.md, the qwen36_moe and
  full_attention_4b kernels are isolated compilation units. Adding
  more `__global__`s to `qwen36_moe.hip` is fine; merging it with
  `full_attention.hip` is not.
- **HF reference for end-to-end check**: the Qwen3.5 multimodal class
  needs ~70 GiB to instantiate. Skip it. Use the hand-rolled multi-
  layer oracle as the parity reference. If a sanity check against HF
  is wanted at the end, do it on a tiny model variant or via API.

## Step plan

### Step 1 — Multi-layer Python oracle

`oracle/qwen36_moe_multilayer_oracle.py`. Imports the reference functions
from the existing single-block oracles (move them to a shared module if
that's cleaner — there's already enough copy-paste between the three).

For one token at position `p`:
```
for layer_idx in 0..num_layers:
    if (layer_idx + 1) % 4 == 0:
        out = reference_full_attention_layer(input=h, ..., position=p)
        h = out["output_hidden"]                # post-attn residual
    else:
        out = reference_linear_attn_layer(input=h, ...,
              recurrent_state_before=rec_state[layer_idx],
              conv_state_before=conv_state[layer_idx])
        h = out["output_hidden"]
        rec_state[layer_idx] = out["state_after"]
        conv_state[layer_idx] = out["conv_state_after"]
    out = reference_moe_ffn_block(input=h, ...)
    h = out["output_hidden"]
final_norm = rms_norm(h, w_norm)
logits = final_norm @ lm_head.T
```

Output shape:
```json
{
    "schema": "qwen36-moe-oracle-multilayer-v1",
    "config": {...},
    "weights_per_layer": [...],          // optional in synthetic mode
    "intermediates_per_layer": [...],    // hidden after each layer's attn + FFN
    "final_hidden": "<bf16 base64>",
    "logits": "<bf16 base64>"
}
```

Modes:
- `--synthetic --num-layers 4`: 3 linear + 1 full at idx 3, small
  geometry (hidden=256 etc.). Fits in seconds, fits in <1 GiB host RAM.
  Primary parity gate.
- `--checkpoint --num-layers 8`: load layers 0..7 from a real bake.
  Sanity check at production geometry.
- `--int4`: same INT4 quantization pattern as the single-block
  oracles. Schema becomes `qwen36-moe-oracle-multilayer-int4-v1`.

Self-check: chain `reference_full_attention_layer` and
`reference_linear_attn_layer` + FFN from the existing references
and verify output_hidden has finite norm. Don't add new math here —
this oracle is pure orchestration.

### Step 2 — Host-orchestrated Rust multi-launch driver

New `crates/runner/src/qwen36_moe_decode.rs` (or extend
`qwen36_moe_engine.rs` — the existing dry-run code is already 700 LoC,
splitting into a separate module is probably cleaner).

Responsibilities:
1. **Build per-layer descriptor array.** One descriptor per layer, with
   pointers into the bake's mmap'd weight regions. The descriptor type
   already exists (`Qwen36MoeDecodeLayerDesc` in
   `crates/kernel-ffi/src/qwen36_moe.rs`). PR 4c's job is to populate
   it from the bake instead of from synthetic stubs.
2. **Allocate per-layer state.**
   - Full-attn layers (10 of 40): KV cache `[max_t, kv_dim]` × 2.
   - Linear-attn layers (30 of 40): conv_state `[qkv_dim, kernel-1]`
     + recurrent_state `[V, kd, vd]` (F32).
3. **Allocate scratch.** Workspace + sync_buf for each kernel kind.
   Reusable across layers within a step (workspaces are independent
   per kernel call, all overwritten each step).
4. **Decode loop.**
   ```rust
   for layer_idx in 0..num_layers {
       if is_full(layer_idx) {
           attn_step_launch(stage=5, ..., int4=&int4[layer_idx], ...)?;
       } else {
           linear_step_launch(stage=5, ..., int4=&int4[layer_idx], ...)?;
       }
       ffn_step_launch(stage=5, ..., int4=&int4[layer_idx], ...)?;
   }
   final_norm + lm_head_argmax → next token.
   ```
5. **Parity-test.** Compare final logits against the multi-layer
   oracle (cos_sim ≥ 0.999). Same pattern as the per-block tests.
6. **Replace the bail in `qwen36_moe_engine.rs::run`**. When `--dry-run`
   isn't set, route to the new decode path.

The key reuse here: every `*_step_launch` already runs a complete
single-layer step at `stage=5`. We're just calling them in sequence
with the right weight pointers and state buffers per layer.

### Step 3 — End-to-end smoke + first decoded token

Same idea as the existing `crates/runner/tests/qwen35_bughunt_smoke.rs`:
small synthetic model, run the full decode pipeline, verify the chosen
token matches what the multi-layer oracle picked. CI-friendly — no real
bake required.

Add a one-shot `--validate` path that runs the multi-layer oracle in
Python and the kernel side in Rust against the same synthesised
weights, compares logits.

### Step 4 (optional) — Persistent-decode megakernel

Single HIP launch that does what step 2 does in N launches. Walks
descriptor array via the work-stealing primitive in
`qwen36_moe_descriptor_walk_stub`. Each block claims a layer; per-layer
the kernel does attn (full or linear) + FFN inline using the existing
phase routines as `__device__` helpers. Significant refactor — the
current single-block kernels have phases-as-inlined-code; persistent
form needs them as functions.

Defer to PR 4c.1 if PR 4c step 2 already gets us first token. The
host-orchestrated path is good enough to ship; megakernel is perf
work.

## Open design decisions

1. **Where does `lm_head` go?** Could land in a new kernel
   (`qwen36_moe_lm_head_launch`) or run on the host (mmap'd weight,
   F32 GEMV → argmax). Host is simpler, kernel is ~1 ms faster per
   token. Recommend host for PR 4c, kernel for PR 4d.
2. **KV cache layout.** Existing parity tests fix `kv_len=1`. PR 4c
   step 2 still has `kv_len=1` for the *first* token after prefill;
   PR 4d (KV extension) lifts that. So step 2 needs the buffers
   pre-allocated at full size but only writes index 0.
3. **Final norm**. Loaded as a regular BF16 weight; applied in the
   host between the last layer's output and lm_head. Cheap.
4. **State persistence across decode steps**. Conv state shifts by 1
   each step; recurrent state is updated in place by the linear-attn
   kernel. Both buffers are *mut* — the kernel reads and writes them.
   Step 2 just needs to keep them alive across the decode loop.

## Checkpoint files / pointers

- **Kernel sources**: `kernels/qwen36_moe.hip`,
  `kernels/qwen36_moe_bridge.cpp`. The single-block kernels are
  `qwen36_moe_attn_step_kernel`, `qwen36_moe_linear_step_kernel`,
  `qwen36_moe_ffn_step_kernel`. Stage 5 = full layer.
- **Rust FFI**: `crates/kernel-ffi/src/qwen36_moe.rs` —
  `attn_step_launch`, `linear_step_launch`, `ffn_step_launch` are the
  safe wrappers. Each takes a `*StepWeights` + a `*StepInt4` struct.
- **Descriptor type**: `Qwen36MoeDecodeLayerDesc` (same file). Already
  has full-attn slots, linear-attn slots, and MoE block slots. Needs
  parallel INT4 sidecar fields added in PR 4c (or pass them per-launch).
- **Runner entry**: `crates/runner/src/qwen36_moe_engine.rs::run`.
  Replace the `anyhow::bail!` at line 639 with a real decode path.
- **Bake reader**: `crates/model-store::BakedStore` + the loader at
  `crates/qwen36_moe/src/loader.rs`. Already mmaps weights and
  exposes typed pointers.
- **Existing per-layer oracles**: `oracle/qwen36_moe_oracle.py` (full
  attn), `oracle/qwen36_moe_linear_oracle.py` (linear attn),
  `oracle/qwen36_moe_ffn_oracle.py` (MoE FFN). All have BF16 + INT4
  modes after PR 4b5/4b6.

## Test fixtures available

`SUPERSONIC_QWEN36_ORACLE_JSON` (full attn) and
`SUPERSONIC_QWEN36_LINEAR_ORACLE_JSON` (linear attn) plus
`SUPERSONIC_QWEN36_FFN_ORACLE_JSON` (FFN) drive the existing per-block
parity tests. PR 4c step 1 adds
`SUPERSONIC_QWEN36_MULTILAYER_ORACLE_JSON` for the multi-layer test.

Generate them with:
```bash
~/venvs/rocm/bin/python oracle/qwen36_moe_oracle.py \
    --mode synthetic --hidden 256 --num-attention-heads 4 \
    --num-kv-heads 2 --head-dim 128 --position 7 --int4 \
    --out /tmp/qwen36_attn_int4.json
```

## Acceptance criteria

PR 4c is done when:
- A synthesised 4-layer model decode in Rust produces logits matching
  the Python multi-layer oracle (cos_sim ≥ 0.999).
- `cargo run --release --bin supersonic --model qwen3.6-moe --model-dir
  <bake> --prompt "..."` runs without `--dry-run` and emits at least
  one token (no parity check on real model — just doesn't bail).
- The host-orchestrated multi-launch driver lands as a single PR
  (steps 1+2+3) without the persistent megakernel. Step 4 is a
  follow-up.
