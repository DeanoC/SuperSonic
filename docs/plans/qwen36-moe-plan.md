# Qwen3.6-35B-A3B MoE Runtime — Plan

Status: **planning, no code yet** · drafted 2026-04-30

This plan turns the existing `Qwen36Moe` registry placeholder into a working
end-to-end decode path. The placeholder bails out at startup
(`crates/runner/src/main.rs:1067`, also `crates/server/src/state.rs:159`); this
document is the bridge from "recognised but unimplemented" to "decodes 8
tokens that match a PyTorch oracle".

The work is large enough that follow-up sessions should break it into
PR-sized chunks before any code lands. Section 12 sketches that sequencing.

---

## 1. Background

The model is a Qwen3-Next-class **hybrid linear/full-attention MoE**:

- **Attention is the same family as Qwen3.5** — every fourth layer is full
  attention, the others are linear (conv1d + recurrent state). This is
  already in production in `crates/qwen35` and the decision in
  `oracle/bake_q4km.py:805–812` ("`linear_attention` if `(i+1) % 4 != 0`")
  applies unchanged. We do **not** need a new attention kernel.
- **The FFN is replaced with MoE**: a routing gate plus N experts (each a
  SwiGLU block) plus an always-on **shared expert**. Top-k routing (k≈8–10)
  with `norm_topk_prob=true` and a small load-balancing aux loss
  coefficient (irrelevant at inference).
- **Total params ≈ 35B, active ≈ 3B per token.** All expert weights must
  live in VRAM — there is no streaming path in SuperSonic today and adding
  one is out of scope for v1.

The Python bake side is **already MoE-aware**:

- `oracle/bake_q4km.py:1064–1066` and `oracle/q4km_stream_gptq_bake.py:297`
  emit `model_family: "qwen36-moe"` when the model name contains `35b-a3b`.
- `oracle/bake_fp8.py:120–135` recognises the same family.
- `oracle/upload_bake.py:78` maps `qwen3.6-35b-a3b → qwen36-moe`.
- `is_q4km_target` (`oracle/bake_q4km.py:600–620`) already includes
  `experts` in the include set and excludes `.gate.` / `router`, so router
  weights stay BF16 and expert FFN matrices get INT4-packed.
- `docs/bake-distribution.md:36` lists `qwen3.6-35b-a3b` as a supported
  upload model.

The **runtime side has nothing**. There is no engine, no kernel, no FFI
module, no model crate. Both `main.rs:1067–1074` and
`server/src/state.rs:159` bail before registry lookup. The
`ModelFamily::Qwen36Moe` and `ModelVariant::Qwen3_6_35B_A3B` enum variants,
plus the CLI alias parsing, are the only existing runtime hooks.

---

## 2. What we know about the architecture

Confirmed from the Qwen3-Next config family (the same `Qwen3NextForCausalLM`
class that 35B-A3B inherits) and from the existing Qwen3.5 implementation:

| Aspect              | Value (proxy from 80B-A3B; verify for 35B) |
|---------------------|---------------------------------------------|
| Architecture class  | `Qwen3NextForCausalLM`                      |
| Hidden size         | 2048                                        |
| Attention layers    | hybrid: 1 full per 4 layers, 3 linear       |
| Full-attn heads     | 16 Q heads, 2 KV heads (GQA), head_dim 256  |
| Linear-attn heads   | 16 K-heads, 32 V-heads, head_dim 128        |
| Linear conv kernel  | 4                                           |
| Partial RoPE        | 0.25 of head_dim rotated, theta 1e7         |
| Vocab               | 151,936                                     |
| Tied embeddings     | false                                       |
| Layer count         | 48 (verify for 35B)                         |
| Experts             | 512 routed (verify for 35B; likely 128–256) |
| Top-k               | 10 (verify; likely 6–8)                     |
| Expert FFN intermediate | 512 (verify)                            |
| Shared expert       | yes, intermediate 512                       |
| Router norm         | `norm_topk_prob=true` (renormalise top-k)   |
| Aux loss coef       | 0.001 (irrelevant at inference)             |
| Dense FFN intermediate | 5120 (used only if `mlp_only_layers`     |
|                     | non-empty; for 35B-A3B this should be empty) |

**Critical open spike:** the 35B-A3B variant's exact (num_experts, top_k,
moe_intermediate_size, num_hidden_layers) need to be read off the actual
HF `config.json` once we have weights. A first PR should land a
`crates/qwen36_moe/src/config.rs` that parses these fields and asserts the
shape against the bake manifest, **before** any kernel work begins.

### Tensor naming (Qwen3-Next / HF transformers convention)

Confirmed by `oracle/q4km_bake_oracle.py:101` (`.mlp.gate_proj.weight`) and
the standard HF Qwen3-MoE class naming. **Not Mixtral `w1/w2/w3`.**

```
model.language_model.embed_tokens.weight
model.language_model.norm.weight
lm_head.weight                                          # untied

# Per layer L ∈ [0, num_hidden_layers):
model.language_model.layers.{L}.input_layernorm.weight
model.language_model.layers.{L}.post_attention_layernorm.weight

# Full attention layers (every 4th):
model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.{L}.self_attn.{q,k}_norm.weight

# Linear attention layers (the other 3 of 4):
model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight
model.language_model.layers.{L}.linear_attn.in_proj_z.weight
model.language_model.layers.{L}.linear_attn.in_proj_a.weight
model.language_model.layers.{L}.linear_attn.in_proj_b.weight
model.language_model.layers.{L}.linear_attn.out_proj.weight
model.language_model.layers.{L}.linear_attn.conv1d.weight
model.language_model.layers.{L}.linear_attn.dt_bias
model.language_model.layers.{L}.linear_attn.A_log
model.language_model.layers.{L}.linear_attn.norm.weight

# MoE block (this is the new bit):
model.language_model.layers.{L}.mlp.gate.weight                              # router
model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight   # E ∈ [0, num_experts)
model.language_model.layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight
model.language_model.layers.{L}.mlp.shared_expert_gate.weight                # scalar gate
```

**Bake interaction with `is_q4km_target`** (already correct, no Python
changes needed):

- `mlp.gate.weight`: excluded by the `.gate.` substring check → stays BF16.
- `mlp.shared_expert_gate.weight`: also excluded by `.gate.` → stays BF16.
- `mlp.experts.{E}.gate_proj.weight` and friends: included via `experts` →
  packed INT4 + BF16 scale/zero sidecars.
- `mlp.shared_expert.gate_proj.weight` etc: included via `_proj` → packed
  INT4. Same as a dense FFN tensor.

Verify after first bake: `grep mlp.gate.weight manifest.json` should show
`"layout": "Raw"`, and `grep experts.0.gate_proj manifest.json` should show
`"layout": "Int4Quantized"` plus a sibling `…_int4_scale` and `…_int4_zero`.

---

## 3. Crate & file layout

We follow the **Phi4 template** — the smallest "added a new family" landed
recently — strictly. The CLAUDE.md rule about isolated compilation units is
non-negotiable on gfx11xx; merging kernels with Qwen3.5 has caused codegen
regressions before. New files only, never extending Qwen3.5 sources.

| New file                                         | Purpose                                          | Approx LoC |
|--------------------------------------------------|--------------------------------------------------|------------|
| `crates/qwen36_moe/Cargo.toml`                   | new crate                                        | ~25        |
| `crates/qwen36_moe/src/lib.rs`                   | re-exports                                       | ~10        |
| `crates/qwen36_moe/src/config.rs`                | parse `config.json`, MoE fields                  | ~250       |
| `crates/qwen36_moe/src/loader.rs`                | bake-store reader, glue                          | ~200       |
| `crates/qwen36_moe/src/weights.rs`               | per-layer expert/router weight slabs             | ~600       |
| `crates/qwen36_moe/src/state.rs`                 | KV caches + linear-attn state + MoE scratch      | ~350       |
| `crates/qwen36_moe/src/rotary.rs`                | partial-RoPE helpers                             | ~80        |
| `crates/qwen36_moe/src/desc_builder.rs`          | builds the FFI layer-desc array                  | ~250       |
| `crates/runner/src/qwen36_moe_engine.rs`         | top-level engine: prefill + decode loop          | ~1100      |
| `crates/kernel-ffi/src/qwen36_moe.rs`            | FFI declarations + safe wrapper                  | ~500       |
| `kernels/qwen36_moe.hip`                         | HIP megakernel                                   | ~6500      |
| `kernels/qwen36_moe_bridge.cpp`                  | HIP launch glue                                  | ~250       |
| `kernels/qwen36_moe_cuda.cuh`                    | CUDA megakernel (header-shared template)         | ~6500      |
| `kernels/qwen36_moe_bridge_cuda.cu`              | CUDA launch glue                                 | ~250       |
| `oracle/qwen36_moe_oracle.py`                    | HF reference forward, JSON state export          | ~600       |

Edits to existing files (small, surgical):

- `crates/kernel-ffi/build.rs` — add the four new kernel sources to the
  HIP and CUDA source arrays, plus `cargo:rerun-if-changed` lines (~10 LoC).
- `crates/kernel-ffi/src/lib.rs` — `pub mod qwen36_moe;` (~1 LoC).
- `crates/runner/src/main.rs` — replace the bail at line 1067 with the
  registry-lookup path; add `mod qwen36_moe_engine;`; add the new arm to
  the family `match` at line 1173 (~5 LoC delta).
- `crates/runner/src/registry.rs` — extend `FamilyParams` with a
  `Qwen36Moe(Qwen36MoeKernelParams)` variant; add at least one
  `RegistryEntry` (see §6); update test fixtures.
- `crates/server/src/state.rs:159` — drop the bail, route to the new family.
- `Cargo.toml` workspace members — add `crates/qwen36_moe`.

**Total budget: roughly 17–18k LoC, dominated by the kernel.** This is in
line with Gemma4 (~13k) and Qwen3.5 (~26k including dual kernels), and
larger than Phi4 (~7k) because MoE adds expert-dispatch logic.

### What `Qwen36MoeKernelParams` should carry

```rust
pub struct Qwen36MoeKernelParams {
    pub weight_prefix: &'static str,        // "model.language_model"
    pub kv_chunk_size: usize,               // 256 default, like Qwen3.5
    pub proj_buf_floats: usize,             // attention scratch
    pub attn_scratch_floats: usize,
    pub moe_scratch_floats: usize,          // top-k softmax + expert outs
    pub num_experts: u32,                   // pulled from config.json
    pub top_k: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}
```

The `num_experts` / `top_k` fields are **not** the source of truth — those
come from the parsed `config.json`. They live in the registry only as
sanity-check bounds (so we can refuse to launch a 35B-A3B kernel against a
weights file with 1024 experts, etc.).

---

## 4. Kernel design strategy

This is the design-critical section. The megakernel pattern means the
**entire** decode step (norms, attention, MoE FFN, residuals) lives in one
persistent kernel. The hard question is how to dispatch experts inside one
kernel: top-k routing makes per-token expert work irregular.

We outline three approaches, then pick.

### Option A — Per-token expert loop (simplest)

Each block strides over tokens. For each token it does:
1. Compute router logits via mat-vec against `mlp.gate.weight`.
2. Softmax over routes, pick top-k indices, renormalise (because
   `norm_topk_prob=true`).
3. Loop over the k selected experts in serial; each iteration is a
   SwiGLU mat-vec against that expert's `gate_proj`/`up_proj`/`down_proj`.
4. Add the shared expert (gated by `shared_expert_gate.weight` scalar).
5. Add residual.

**Pros:** trivial control flow, mirrors Qwen3.5's block-strided MLP exactly.
LDS budget identical to Qwen3.5 because at any one moment only one expert's
tile is loaded.

**Cons:** for batch=1 this serialises k expert mat-vecs across the same set
of blocks. With ~24 CUs on gfx1100 and `top_k≈8`, we leave most CUs idle
during each expert step — the per-expert mat-vec is small (`hidden ×
moe_intermediate ≈ 2048 × 512`) and a single mat-vec doesn't fill the GPU.

### Option B — Expert-major, token-compacted

Per layer:
1. Compute routing for all batch tokens, build a compacted token list per
   expert via in-kernel prefix-sum / atomics.
2. For each expert that has ≥1 routed token, do one batched mat-vec across
   those tokens.

**Pros:** at batch>1 this amortises expert weight reads — multiple tokens
sharing an expert pay one weight-traffic cost.

**Cons:** at batch=1 it degenerates to Option A but with extra compaction
overhead. The compaction itself is O(num_experts × batch) atomics + a
barrier, which is non-trivial inside the megakernel. Adds ~500 LoC of
in-kernel scratch management (per-expert offsets, atomic counters, sync).

### Option C — Per-block (token, expert) work units (chosen)

For each layer:
1. Compute routing for all batch tokens, write a flat list of `(token,
   expert, weight)` work units to scratch — total `batch × top_k` units
   plus `batch` shared-expert units.
2. Use the existing **work-stealing atomic-counter + grid barrier** pattern
   (`sync_buf` / `g4_grid_barrier`) to dispatch work units across all
   blocks. Each block fetches `next_unit = atomicAdd(counter, 1)` and
   processes one (token, expert) mat-vec.
3. Down-projection accumulates per-token outputs via atomics into a
   scratch buffer indexed by token_id.
4. After grid barrier, blocks add the residual and move to the next layer.

**Pros:**
- **Reuses an existing primitive.** The work-stealing matmul is already in
  `kernels/full_attention_4b.hip` and well-debugged; we are not inventing
  scheduling from scratch.
- **For batch=1 with top_k=8 and a shared expert: 9 work units per layer,
  spread across all 24+ CUs.** Each CU runs a different expert's
  mat-vec concurrently — that's exactly the parallelism we need to hide
  the small per-expert mat-vec size.
- **Down-proj atomics are cheap** because per-token writes are
  per-(token, hidden_dim) and conflicts are rare; on RDNA3 we can use
  `atomicAdd(half2*)` on BF16 pairs.
- **Trivially extends to batch>1** without a re-architecture.
- **LDS budget stays Qwen3.5-like** — one expert tile per block at any
  time.

**Cons:**
- The atomics-into-residual write needs a scratch-buffer cleanup pass.
- The work-unit list itself takes scratch (`batch × (top_k + 1) × 8 bytes`,
  trivial).

**Pick: Option C.** It's the only one that exploits inter-expert
parallelism at batch=1, which is the dominant decode case, and it leans on
a primitive we already trust.

### Routing precision

Router logits and softmax run in **FP32**. `mlp.gate.weight` is BF16
(excluded from INT4 by `is_q4km_target`). Top-k selection is a reduction
over `num_experts`; for ≤512 experts a single-block bitonic sort or a
selection-via-threshold-search works inside one block's LDS. Pick
selection-by-radix or threshold-bisection in the implementation PR; both
are O(num_experts × log num_experts) and fit in LDS for ≤512 experts.

### gfx1100 occupancy & LDS budget

Worst-case LDS use per block during MoE step:
- `hidden_size = 2048` BF16 input row: 4 KiB
- One expert tile (`block_n × group_size = 128 × 128` INT4 + scales):
  ~8 KiB
- Per-block `top_k` work-unit metadata: ~80 bytes
- Total: ~12 KiB — well under gfx1100's 64 KiB LDS per WG cap; we should
  comfortably hit ≥2 wavefronts/CU. Same envelope as Qwen3.5-9B which
  already runs.

Cross-check this against the actual config in the implementation PR — if
`moe_intermediate_size` for 35B is larger than 512 the tile size must
follow, and LDS pressure goes up.

---

## 5. Model state (mostly inherited from Qwen3.5)

Qwen3.5's state (`crates/qwen35/src/state.rs`) already covers everything we
need *except* the MoE-specific scratch:

- KV caches per full-attention layer (chunked, default 256-token chunks).
- Linear-attention conv state + recurrent state per linear layer.
- RoPE tables.
- Per-layer norms (kept in F32).

What Qwen3.6-MoE adds:

- **MoE scratch** (`moe_scratch_floats` in registry params): top-k
  softmax buffer (`batch × top_k × 2` for index+weight), per-token
  expert-output accumulator (`batch × hidden_size` BF16), router logits
  scratch (`batch × num_experts` FP32), work-unit list (`batch × (top_k +
  1) × 16` bytes).
- **Shared expert output buffer**: same shape as a single expert's down
  output; reused across layers.

We do **not** copy the Qwen3.5 state struct into `qwen36_moe::state`. The
crate gets its own struct so changes to one family can never silently
ripple into the other. Code-level duplication is fine and explicitly
encouraged by CLAUDE.md.

---

## 6. VRAM budgeting & the registry entry

### Per-quant size (35B params, ignoring norms/embeddings)

| Quant       | bits/weight | Weights only | + scales | + activations + KV (4K ctx) | Fits 24 GiB? |
|-------------|-------------|--------------|----------|------------------------------|--------------|
| BF16        | 16          | 70 GiB       | —        | ~71 GiB                      | no           |
| FP8 native  | 8           | 35 GiB       | + ~0.5%  | ~36 GiB                      | no           |
| INT4 GPTQ   | 4 + scales  | 17.5 GiB     | + ~0.6 GiB | ~19.5 GiB                  | yes          |
| q4km (GGML) | ~4.5        | 19.7 GiB     | inline   | ~21 GiB                      | yes (tight)  |
| q4km-gptq   | ~4.5        | ~19.7 GiB    | mixed    | ~21 GiB                      | yes (tight)  |

### Initial registry entries to add

For each entry, set `fixed_bytes` to weights-only (the runtime adds KV on
top via `estimate_total`). The `overhead_factor` stays at 1.1 to match
existing entries.

```rust
// HIP gfx1100, INT4 GPTQ — primary AMD target.
RegistryEntry {
    model: ModelVariant::Qwen3_6_35B_A3B,
    backend: Backend::Hip,
    arch: GpuArch::Gfx1100,
    vram: VramBudget { fixed_bytes: 19 * GIB, overhead_factor: 1.1 },
    params: FamilyParams::Qwen36Moe(Qwen36MoeKernelParams { /* … */ }),
},

// CUDA sm86, q4km — primary NVIDIA target.
RegistryEntry {
    model: ModelVariant::Qwen3_6_35B_A3B,
    backend: Backend::Cuda,
    arch: GpuArch::Sm86,
    vram: VramBudget { fixed_bytes: 22 * GIB, overhead_factor: 1.1 },
    params: FamilyParams::Qwen36Moe(Qwen36MoeKernelParams { /* … */ }),
},
```

The numbers above must be **re-confirmed against an actual bake**. The
plan PR should not freeze them; the implementation PR will measure and
adjust.

### What about gfx1150?

gfx1150 is an APU with unified memory; its VRAM is system RAM minus what
the OS holds. A 32 GiB APU could just about host a 19 GiB INT4 bake. We
do **not** add a gfx1150 registry entry in v1 — it's a "see if it works
opportunistically" scenario, not a committed combo. Users can pass
`--allow-untested-gpu=gfx1100` to force-reuse the gfx1100 kernel.

---

## 7. Quant matrix for v1

Given:

- **q4km is currently CUDA-only** (`crates/runner/src/main.rs:1035–1037`
  bails if `q4km_like && backend != Backend::Cuda`). Lifting that gate is
  a separate project — not in scope here.
- **INT4 GPTQ already works on HIP** for Qwen3.5; the existing
  `oracle/bake_int4.py` calibration path will pick up MoE expert tensors
  via `is_q4km_target`/`is_int4_target` matching `experts`/`_proj` with
  no MoE-specific changes (Hessians are pooled per-layer across experts —
  `oracle/bake_int4.py:72–82` confirms there is no per-expert logic, which
  is acceptable because all experts in a layer see calibration traffic).
- **FP8 bake is also already MoE-aware** (`oracle/bake_fp8.py`), but FP8
  weights at 35B don't fit 24 GiB so it's not useful for the primary
  targets.

**v1 quant matrix:**

| Backend | Arch    | Quant       | Bake script                    | Status  |
|---------|---------|-------------|--------------------------------|---------|
| HIP     | gfx1100 | INT4 GPTQ   | `oracle/bake_int4.py`          | **v1**  |
| CUDA    | sm86    | q4km        | `oracle/bake_q4km.py` + GGUF   | **v1**  |
| CUDA    | sm86    | q4km-gptq   | `oracle/bake_q4km.py --quantizer gptq` | **v1** |
| HIP     | gfx1100 | q4km        | (needs HIP q4km kernel — out of scope) | v2 |
| any     | any     | BF16 / FP8  | (won't fit 24 GiB)             | n/a     |

Picking **two quants** for v1 is deliberate: q4km is a separate code path
on the kernel side (GGML K-block dequant vs SuperSonic-native INT4
dequant), and we want both validated end-to-end before declaring the
runtime done. The CUDA megakernel must support both `Int4Quantized` and
`GgmlQ4K` weight layouts; both already exist for Qwen3.5 in the CUDA
build.

---

## 8. Backend matrix for v1

- **HIP gfx1100** — primary AMD target. Single megakernel
  (`kernels/qwen36_moe.hip`). INT4 GPTQ only.
- **CUDA sm86** — primary NVIDIA target. Single megakernel
  (`kernels/qwen36_moe_cuda.cuh` + `_bridge_cuda.cu`). q4km and INT4 GPTQ
  both supported.

**Out of v1 scope:** Metal (Apple M4), gfx1150 explicit support. Both can
be added later by registry-only changes if the kernel ports cleanly; no
new design work needed in this plan.

---

## 9. Bake distribution

35B at q4km is ~22 GiB compressed; the existing tarball pipeline already
handles >1.8 GiB by splitting into `.part01`, `.part02` etc. (see
`docs/bake-distribution.md:32–40`).

**Day-1 publish list** (each ≥10 GiB tar.zst, multi-part):

- `qwen3.6-35b-a3b-int4-gptq-fmt2-cvt1.tar.zst.part{01..NN}` (HIP target).
- `qwen3.6-35b-a3b-q4km-fmt2-cvt1.tar.zst.part{01..NN}` (CUDA target).
- `qwen3.6-35b-a3b-q4km-gptq-fmt2-cvt1.tar.zst.part{01..NN}` (CUDA, better
  perplexity).

The producer flow is unchanged — `oracle/upload_bake.py` already accepts
`qwen3.6-35b-a3b` (line 59). Bakes-index v1 schema covers it. **We do
need release-hosted bakes from day 1** because:

- 35B GPTQ calibration on a single 24 GiB GPU host takes hours and
  requires ~64 GiB system RAM for the layer-by-layer Hessian pass.
- The whole point of release-hosted bakes is to let small-VRAM users skip
  calibration.
- A user with a 24 GiB gfx1100 likely cannot calibrate locally; without a
  release bake they have no path to running the model.

The implementation PRs should ship a smoke-tested bake to `bakes-v2`
**before** announcing the model as supported.

---

## 10. Verification

### PyTorch oracle (`oracle/qwen36_moe_oracle.py`)

Modeled exactly on `oracle/qwen35_oracle.py`. It:

1. Loads the HF model in BF16 (or FP8 dequantised to BF16).
2. Runs prefill over a tiny prompt ("Hello, world").
3. Exports state per decode step: `hidden`, full-attention KV per layer,
   linear-attn conv + recurrent state, **and** per-layer router logits,
   chosen expert indices, expert outputs (sum over k), shared-expert
   output. JSON + base64.
4. Optionally records logits per step.

The Rust runtime then either (a) loads HF state and decodes from there
(parity check) or (b) runs prefill itself and compares logits at each step
to the oracle.

### Smoke test (CI-friendly when GPUs are available)

```
python oracle/bake_int4.py --model qwen3.6-35b-a3b --model-dir <hf-dir>
cargo run --release --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir <hf-dir> \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --validate
```

Expected:

- All 8 step logits match the oracle's BF16 forward to within INT4 GPTQ
  tolerance (typically `cos_sim ≥ 0.999` per step on Qwen3.5; MoE may
  show slightly looser numbers near the routing decision boundaries —
  pick a per-step threshold of `cos_sim ≥ 0.997` and `top1 token-id
  match` as the gate, refine after first measurement).
- Decode time < 200 ms/token on gfx1100; < 80 ms/token on sm86. These
  are coarse targets; final numbers go in `docs/performance.md`.

### Pre-merge unit tests

Inside `crates/qwen36_moe`:
- `config.rs` parses the test-fixture `config.json` correctly.
- `weights.rs` enumerates expert tensor names against a fake manifest
  with N=4 experts and verifies shape consistency.
- `desc_builder.rs` produces a layer-desc array whose pointers all fall
  inside the bake's tensor offset table.

Inside `crates/runner`:
- A registry test like `qwen36_aliases_are_public_and_canonical`
  (already exists at `registry.rs:690–706`) gets extended to assert a
  `RegistryEntry` exists for `(Qwen3_6_35B_A3B, Hip, Gfx1100)` and
  `(Qwen3_6_35B_A3B, Cuda, Sm86)`.

---

## 11. Risks & open spikes

1. **Actual 35B-A3B config** — 80B-A3B's (512, top-10, moe_int=512) is a
   proxy. First step in the implementation PR sequence is fetching and
   recording the real numbers.
2. **q4km-on-HIP gating** — if the user wants HIP gfx1100 + q4km in v1
   instead of INT4 GPTQ, that is a separate kernel project (the CUDA q4km
   path doesn't port to HIP without porting the GGML K-block dequant
   inline-asm). Decision needed before kernel work starts.
3. **Down-projection atomics on RDNA3** — Option C relies on
   `atomicAdd` into BF16 pairs. RDNA3 supports `__hip_atomic_fetch_add` on
   16-bit floats, but the codegen quality on hipcc 6.x is uneven. Build
   a 50-line standalone test before committing to this path; if codegen
   is bad, fall back to per-block scratch and a separate sum kernel.
4. **Top-k selection at num_experts=512** — a sort over 512 values is
   nontrivial inside the megakernel. Threshold-bisection is the safer
   bet than bitonic sort; the bisection variant takes ~`ceil(log2(512))
   = 9` reduction passes but uses no LDS. Worth a focused micro-bench
   before deciding.
5. **Shared-expert gating math** — Qwen3-Next applies the shared-expert
   gate as `sigmoid(shared_expert_gate · x) · shared_expert(x)`. Confirm
   against the HF source before implementing. Wrong math here is the
   classic silent-divergence bug.
6. **Hessian collection for 35B** — `oracle/bake_int4.py` accumulates
   forward-hook Hessians on the **full** model. For 35B that's ~280 GiB
   of FP32 Hessians if held in memory; the streaming GPTQ baker
   (`oracle/q4km_stream_gptq_bake.py`) was built for exactly this case
   for q4km but its INT4 equivalent may need work. Verify before
   committing the producer side of bake distribution.
7. **VRAM tightness on 24 GiB gfx1100** — at ~19.5 GiB weights + KV +
   scratch we are within a few hundred MiB of the cap. Any growth in
   `moe_scratch_floats` or in `kv_chunk_size` could push us over. The
   implementation PR should track peak alloc against the budget and fail
   the smoke test if it exceeds 22 GiB.

---

## 12. Suggested PR-sized chunks (for follow-up sessions)

1. **PR 1 — Crate scaffolding + config parsing.** New `crates/qwen36_moe`
   with `Cargo.toml`, `lib.rs`, `config.rs` (parses real 35B config.json,
   asserts hybrid layer pattern, MoE field presence). Workspace
   membership. No kernel work, no registry change. Closes the "what
   does the config actually look like" spike.

2. **PR 2 — Bake-side validation.** Run
   `python oracle/bake_int4.py --model qwen3.6-35b-a3b --model-dir <hf>`
   end-to-end on a tiny test fixture (artificially shrunk MoE: 4
   experts, 2 layers). Add `oracle/test_bake_qwen36_moe.py` covering
   manifest assertions: every expert weight is `Int4Quantized`, every
   gate is `Raw`. No runtime work yet.

3. **PR 3 — Runtime weight loader, no kernel.** `weights.rs`,
   `loader.rs`, `state.rs`. Add a `--dry-run` CLI flag that loads
   weights, builds state, prints VRAM accounting, and exits. Add the
   registry entries with placeholder `Qwen36MoeKernelParams`. Replace
   the bail at `main.rs:1067` with a route to the new dry-run path.

4. **PR 4 — CUDA kernel + bridge + FFI, q4km only.** First working
   decode path. Single arch (sm86). Skeleton `qwen36_moe_engine.rs`
   that drives one step. Pass a single token through and record
   logits to disk. No oracle comparison yet.

5. **PR 5 — Oracle + parity gate.** `oracle/qwen36_moe_oracle.py`,
   wire `--validate` through `qwen36_moe_engine`. Smoke test passes
   on sm86 with `cos_sim ≥ 0.997`, top-1 match.

6. **PR 6 — HIP kernel + bridge + FFI, INT4 GPTQ.** Port the CUDA
   megakernel (most of `qwen36_moe_cuda.cuh` becomes
   `qwen36_moe.hip`). Add gfx1100 registry entry. Smoke test passes
   on gfx1100.

7. **PR 7 — Bake distribution.** Run GPTQ calibration, upload to
   `bakes-v2`. Update `docs/bake-distribution.md`. Add an integration
   test that downloads the published bake, runs decode, compares
   logits — runs in CI behind a `SUPERSONIC_INTEGRATION` env gate.

8. **PR 8 — Performance pass.** Benchmark, write
   `docs/qwen36-moe-performance.md`, tune `moe_scratch_floats` and
   `kv_chunk_size` per arch, evaluate cooperative-launch preset like
   the Qwen3.5 4B kernel uses.

PR 1–3 are pure scaffolding and unblock parallel work on PR 4 and PR 6.
PRs 4–6 are the design-critical path. PRs 7–8 are productionisation.

---

## 13. Out of scope (explicit non-goals)

- **q4km on HIP.** Lifting the CUDA-only gate is a kernel project of its
  own; defer.
- **Streaming expert weights from CPU.** All experts must fit VRAM in
  v1.
- **Multi-GPU sharding.** Single-GPU only.
- **Dynamic expert pruning / hot-expert caching.** v1 routes by raw
  router logits without any caching layer.
- **Continuous batching.** Decode-only single-sequence first;
  many-sequence/batch is a follow-up after v1 ships.
- **Metal / Apple M4 support.** Future addition; nothing in this plan
  precludes it but no work is committed.

---

## 14. Approval gate before any code

The two decisions a reviewer should confirm before PR 1 lands:

1. **Quant choice for HIP gfx1100: INT4 GPTQ (this plan)** vs. blocking
   on q4km-on-HIP. INT4 GPTQ is the recommendation because it works
   today; q4km-on-HIP is months of extra work.
2. **Kernel dispatch strategy: Option C (per-block work units, this
   plan)** vs. Option A (per-token loop). Option C is the
   recommendation because it parallelises across experts at batch=1.

Once those are signed off, the work below is mechanical execution and
can proceed PR by PR without further architectural debate.

---

## 15. Schema reality check (post-PR 3 addendum, 2026-04-30)

Sections §1, §2, and §6 of this plan were drafted from the 80B-A3B proxy
config. PR 1 landed config parsing against the real 35B-A3B
`config.json`; PR 3 went further and enumerated the published
`Qwen/Qwen3.6-35B-A3B` safetensors checkpoint end-to-end. Several plan
assumptions did not survive contact with the real bake. **This section
overrides the corresponding rows of §1 / §2 / §6 / §7 wherever they
disagree.** Earlier sections are kept for historical context.

### Geometry — what 35B-A3B actually ships

| Aspect | Plan §1 (80B-A3B proxy) | **Real (35B-A3B)** |
|---|---|---|
| `num_hidden_layers` | 48 | **40** |
| `hidden_size` | 2048 | 2048 ✓ |
| Full-attn Q heads | 16 | 16 ✓ |
| Full-attn KV heads | 2 | 2 ✓ |
| `head_dim` | 256 | 256 ✓ |
| `num_experts` | 512 | **256** |
| `num_experts_per_tok` (top_k) | 10 | **8** |
| `moe_intermediate_size` | 512 | 512 ✓ |
| `shared_expert_intermediate_size` | 512 | 512 ✓ |
| `vocab_size` | 151,936 | **248,320** |
| `tie_word_embeddings` | false | false ✓ |
| Architecture class | `Qwen3NextForCausalLM` | **`Qwen3_5MoeForConditionalGeneration`** (multimodal — vision + text + MTP) |
| RoPE | partial 0.25, θ=1e7 | mRoPE, section [11,11,10], interleaved, partial 0.25, θ=1e7 |
| `attn_output_gate` | not in plan | **true** — see "q_proj is doubled" below |

Source of truth: `crates/qwen36_moe/src/config.rs` parses the live
`config.json`; the `parses_real_qwen36_35b_a3b_config` test pins these
values against a verbatim copy of the published config.

### Tensor naming — overrides §2

The §2 tensor list described an unfused, per-expert layout. **The real
checkpoint stores experts as fused-batched tensors per layer, with no
`.weight` suffix.** Cross-checked by directly enumerating
`Qwen/Qwen3.6-35B-A3B`'s safetensors index in PR 3.

```
# What plan §2 said (every layer, every expert E ∈ [0, num_experts)):
model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.weight   # NOT real
model.language_model.layers.{L}.mlp.experts.{E}.up_proj.weight     # NOT real
model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight   # NOT real

# What's actually in the checkpoint (note: no .weight suffix, 3D shapes):
model.language_model.layers.{L}.mlp.experts.gate_up_proj   [E, 2*moe_int, hidden]  bf16
model.language_model.layers.{L}.mlp.experts.down_proj      [E, hidden, moe_int]    bf16
```

Other MoE-block tensors are still per-projection and DO have `.weight`
suffixes:

```
model.language_model.layers.{L}.mlp.gate.weight                    [E, hidden]      bf16
model.language_model.layers.{L}.mlp.shared_expert_gate.weight      [1, hidden]      bf16
model.language_model.layers.{L}.mlp.shared_expert.gate_proj.weight [moe_int, hidden] bf16
model.language_model.layers.{L}.mlp.shared_expert.up_proj.weight   [moe_int, hidden] bf16
model.language_model.layers.{L}.mlp.shared_expert.down_proj.weight [hidden, moe_int] bf16
```

### `attn_output_gate=true` — overrides §1

Qwen3-Next MoE adds a sigmoid-gate on the attention output, fused into
`q_proj`'s output dimension. Effects:

```
self_attn.q_proj.weight: [2 * num_heads * head_dim, hidden]    # 2× the plan's assumption
self_attn.o_proj.weight: [hidden, num_heads * head_dim]        # unchanged
```

For 35B: `q_proj=[8192, 2048]` (not `[4096, 2048]`). The runtime must
split q_proj's output into `[Q | output_gate]` lanes and apply
`sigmoid(output_gate) * attn_output` before `o_proj`. Mechanically
identical to how Phi4 splits a fused `gate_up_proj` — a stride trick at
matmul time.

### Linear-attn dtype — overrides §2

Plan §2 (transcribed from a Qwen3.5-0.8B probe) implied
`linear_attn.norm.weight` and `linear_attn.A_log` are F32. The published
35B-A3B checkpoint stores **all** decoder weights — including these two —
as **BF16**. The runtime must accept both dtypes; the existing
`ensure_f32_on_gpu` helper in `crates/qwen35/src/weights.rs` handles BF16
norms by upcasting on demand.

### Things in the safetensors that v1 ignores

- **Vision tower** (`model.visual.blocks.0..26.*`, `model.visual.merger.*`,
  `model.visual.patch_embed.*`, `model.visual.pos_embed.weight`) —
  multimodal weights for the conditional-generation class. Text-only
  decode never references them.
- **MTP head** (`mtp.layers.0.*`, `mtp.fc.weight`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_*`) — multi-token-prediction layer used during
  training and optionally for speculative decoding. v1 single-sequence
  decode does not consume it.

The runtime should silently skip both groups during weight loading. The
on-disk total of 67 GiB (1045 tensors) breaks down as: 64.56 GiB / 693
tensors are language-decoder weights v1 actually uses, ~2.4 GiB / 352
tensors are vision + MTP overhead.

### The bake gap (the one that matters before PR 7)

Both `is_int4_target` and `is_q4km_target` reject the fused expert
tensors today: they require `name.endswith(".weight")`, and the q4km
predicate additionally requires a 2D shape. The real names are
`mlp.experts.gate_up_proj` (3D) and `mlp.experts.down_proj` (3D), with
no `.weight` suffix. **A naive `python oracle/bake_int4.py --model
qwen3.6-35b-a3b ...` run today would leave ~60 GiB of expert weight
unquantized**, busting any 24 GiB GPU budget.

There is also an upstream issue: `bake_int4.py` walks `nn.Linear`
modules (`for mod in layer.modules() if isinstance(mod, nn.Linear)`).
The HF `Qwen3_5Moe` implementation almost certainly stores experts as a
single `nn.Parameter` under a custom MoE module, *not* as a list of
`nn.Linear` per expert — so even if the predicates were updated, the
GPTQ driver would still skip the experts entirely.

**PR 7 (calibration) must close this gap before producing a usable
INT4 bake.** Three options, in increasing order of effort:

- (a) **Update predicates + extend the GPTQ driver to handle 3D fused
  tensors.** Treat the `[E, out, in]` slab as `E` parallel `[out, in]`
  matrices; share one Hessian per layer (since all experts see the same
  pre-MoE activations). Pack each expert's nibbles as a contiguous
  group_size tile, with one `[out/gs, in/gs]` BF16 scale+zero pair per
  expert.

- (b) **Unfuse before calibration, refuse after pack.** Add a state-dict
  pre-hook that splits the fused params into per-expert `nn.Linear`,
  run unmodified GPTQ, then re-pack the nibble layout into a fused INT4
  slab on save. More code, no kernel-side change.

- (c) **Skip Hessian-aware calibration on experts; min/max INT4 only.**
  Quantize experts via plain min/max group-quant (still group_size=128
  with BF16 scale+zero), keep GPTQ for dense projections only. Cheapest
  path. Quality cost is unknown but probably tolerable since experts
  only fire for ~3% of any given token's compute (top-8 of 256, at
  `moe_int=512` vs the much larger Q/K/V/O+linear-attn matmuls).

Recommendation: start with (c) for v1 to unblock the runtime quickly.
Revisit (a) in a follow-up if quality is below the cos_sim ≥ 0.997
gate.

### Updated VRAM measurement (replaces §6 estimates)

Numbers come from `--dry-run` against the live 67 GiB BF16 download:

| Component | Bytes | Notes |
|---|---|---|
| `embed_tokens` | 0.95 GiB | BF16, vocab 248,320 × hidden 2048 |
| `lm_head` | 0.95 GiB | untied, same shape |
| Full-attn (10 layers) | 0.51 GiB | q_proj doubled by attn_output_gate |
| Linear-attn (30 layers) | 1.88 GiB | qkv 8192×hidden, z 4096×hidden, conv 8192×4 |
| Routers (40 layers) | 0.04 GiB | `mlp.gate.weight` BF16 stays Raw |
| Shared experts (40 layers) | 0.23 GiB | per-projection BF16 |
| Routed experts (40 layers) | 60.00 GiB | fused `[256, 1024, 2048]` + `[256, 2048, 512]` per layer |
| **Total BF16 (decoder)** | **64.56 GiB** | byte-for-byte match between analytic + on-disk |
| **Total INT4 GPTQ projection** | **16.89 GiB** | gs=128, weights packed + BF16 scale/zero sidecars |

The plan's §6 19-GiB INT4 budget is correct with margin. After the
runtime adds activations + KV + scratch (≤ 4 GiB at 4K context), the
HIP gfx1100 entry projects 20.90 GiB total — fits a 24 GiB card.

### Sections of the plan still valid

- §3 crate layout and §4 kernel design (Option C work-stealing) are
  unaffected.
- §6 registry-entry layout is correct; only the per-quant byte numbers
  in the table needed verification (done above).
- §7 quant matrix is correct: INT4 GPTQ on HIP gfx1100, q4km on CUDA
  sm86.
- §10 verification + smoke-test approach is unchanged; the oracle just
  needs to know the fused-expert tensor names.
- §11 risks 4 (top-k at 256 is tighter than the plan's 512) and 5
  (shared-expert gating math) still apply.
- §12 PR sequencing is unchanged; this addendum lives inside PR 3.
