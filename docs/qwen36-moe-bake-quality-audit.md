# Qwen3.6-MoE INT4 GPTQ bake quality audit

**Status**: audit complete, no algorithmic bugs found. Quality issues are
design limitations of the in-house GPTQ implementation.

## Symptom

Multi-token decode against the local INT4 GPTQ bake at
`<model>/.supersonic/v2-int4-gptq/` produces near-random output:

  $ supersonic --model qwen3.6-35b-a3b --model-dir <snap> \
      --prompt "The quick brown fox" --max-new-tokens 8
  > The quick brown foxstoreparepareUDAUDAUDAUDAUDA

Per-prompt argmax IS responsive to input (different prompts → different
first tokens), but the model immediately settles into a small set of
tokens (UDA/store/pare/udas/lis). At T=0.8 + top_p=0.9 the entire vocab
is sampled near-uniformly including foreign scripts ("OLOG internsieß
第二段的统一 발표isk"), confirming the logit distribution is nearly flat.

## What we ruled out

### Engine + decode pipeline ✅
  - Multi-layer parity test (synthetic 4-layer): cos_sim 0.9999 final
    logits, BF16 + INT4 modes both pass.
  - Per-block parity tests at *production* geometry against real
    safetensors weights:
      - Full-attn (layer 3, H=16, Hkv=2, d=256, hidden=2048):
        n=2048 exact=2048 cos_sim=1.0  (bit-exact)
      - Linear-attn (layer 0): passes
      - FFN (layer 0, E=256, top_k=8, I=Is=512): passes
  - KV cache reads/writes verified by the kv_len=1 fast-path being
    bit-exact with the pre-cache PR 4b2 implementation.

### Sampling ✅
  - Greedy-vs-sampled output proves the issue isn't sampling — the
    underlying logit distribution itself is too flat for either to
    produce coherent text.

### Bake ↔ kernel format ✅
  - Python dequant `bf16(q*s - z*s)` matches the kernel's
    `int4_dequant_scalar` formula exactly (same nibble-packing
    convention, same tile indexing, same rounding).

## What we found

### Per-tensor INT4 reconstruction error from the bake

Sampled tensors compared bake-INT4-reconstruction vs safetensors BF16
(ground truth):

  tensor                                         shape    tiles    cos_sim
  layers.3.self_attn.q_proj                  (8192, 2048)  1024     0.961
  layers.3.self_attn.o_proj                  (2048, 4096)   512     0.900
  layers.0.linear_attn.in_proj_qkv           (8192, 2048)  1024     0.941
  layers.0.linear_attn.in_proj_z             (4096, 2048)   512     0.971
  layers.0.linear_attn.out_proj              (2048, 4096)   512     0.899
  layers.0.mlp.shared_expert.gate_proj        (512, 2048)    64     0.880
  layers.0.mlp.shared_expert.up_proj          (512, 2048)    64     0.952
  layers.0.mlp.shared_expert.down_proj        (2048, 512)    64     0.925

Per-tensor 3-12% directional error compounds across 40 layers ×
multiple tensors per layer to effectively-random hidden states by the
time the chain reaches the lm_head. Production AutoGPTQ INT4 typically
delivers ≥ 0.99 per-tensor cos_sim; we are 0.05–0.12 below that bar.

Notable: tensors that produce a `[..., out=2048 or 512]` (o_proj,
out_proj, shared_expert.gate_proj) consistently land at the bottom of
the cos_sim range. Those are the layer-output projections that GPTQ
quantizes LAST in each layer's order; they inherit the most accumulated
sequential-calibration error.

### Algorithm audit (oracle/bake_int4.py)

Walked the implementation against canonical GPTQ end to end. Every
piece checks out:

  - **Hessian** (`HessianHook`, line 513–536):
    `H = 2/N * sum_t x_t x_t^T` accumulated via forward-pre hooks.
    Recursive average update is correct; total `H` after all samples
    matches the canonical formula.

  - **Damping** (line 247–254): `H[i,i] += damp * mean(diag(H))`,
    damp=0.01 default (matches AutoGPTQ).

  - **Cholesky-of-inverse** (line 263–269): `L = chol(H)` (lower);
    `Hinv = L^-T L^-1`; `U = chol(Hinv, upper=True)` upper-triangular
    such that `U^T U = Hinv`. Stored back in `Hinv` (variable name is
    misleading but math is correct).

  - **Per-column error propagation** (line 307–340): standard GPTQ
    update `err = (w - q_dq) / U[i,i]`; `W[:, i+1:] -= err * U[i, i+1:]`.

  - **Block boundary error propagation** (line 343–344): standard
    `Err1 @ Hinv[b:e, e:]` propagates accumulated block errors to all
    columns past the block. With `blocksize == group_size == 128`, no
    column-group spans a block boundary.

  - **Scale derivation** (line 278–293): per-tile min/max,
    `sc = rng/15`, `zf = -tmin/sc`, both rounded through BF16 to match
    runtime kernel reads.

  - **Quantization** (line 326–332): `q = clamp(round(w/sc + zf), 0, 15)`;
    `q_dq = bf16(q*sc - zf*sc)` — matches kernel's `int4_dequant_scalar`
    exactly so error feedback uses what the kernel will actually read.

  - **Sequential calibration** (line 824–906): standard. Each layer's
    quantized outputs feed the next layer's Hessian collection.

  - **Calibration data** (line 1267–1280): defaults to 128 sequences ×
    2048 tokens from WikiText-2 train. Standard.

  - **lm_head GPTQ** (line 921–999): same algorithm applied to lm_head
    using post-final-norm hidden as activation. Correct.

### Identified design limitations (not bugs)

1. **No scale search.** `set_tile_scales` (line 278) uses plain min/max
   for every tile. AutoGPTQ tries multiple candidate scales per tile
   (typically `rng * f / 15` for `f in [1.0, 0.95, 0.9, …]`) and picks
   the one minimizing per-tile reconstruction MSE. Skipping scale
   search costs noticeable per-tensor accuracy; this is the most likely
   single contributor to our cos_sim gap vs AutoGPTQ-class output.

2. **Min/max for fused experts (no GPTQ).** `fused_expert_minmax_int4`
   (line 386) does plain min/max on each expert's `[out, in]` slab.
   Justified in code comments by the fused 3D layout being awkward for
   Hessian-aware quant on the producer host. ~half the model's
   parameters take this path on Qwen3.6-MoE.

3. **Coarse 128×128 tile granularity.** Each `(scale, zero)` pair
   covers a 16384-weight tile with 16 quantization levels. Narrower
   projections (e.g. shared_expert.gate_proj `[512, 2048]`, only 64
   tiles total) get the most coarse quantization — visible in the
   cos_sim survey.

## What to do

The audit confirms no bug to fix in the GPTQ algorithm. The realistic
quality lever options, in order of expected impact vs effort:

### Option 1: Add scale search (~1 day; modest gain per re-bake hour)
Patch `set_tile_scales` (line 278 of `oracle/bake_int4.py`) to try
~5 candidate scales per tile and pick the one minimizing per-tile MSE.
~30 LoC. Re-baking 35B-A3B takes hours (whole calibration loop), so
each iteration is expensive — but no infrastructure changes needed.

### Option 2: Migrate to AutoGPTQ (~1 week; biggest gain)
AutoGPTQ has been battle-tested on hundreds of production models and
includes scale search, activation reordering, static-vs-dynamic group
choices, and other tuning. Output format would need a small adapter
(their `[in, out]` packing vs our `[out, in/2]`) but their per-tensor
quality is well-documented at ≥ 0.99 cos_sim. Best long-term answer.

### Option 3: Less aggressive quantization (~1 day; large gain at runtime cost)
Switch from INT4 (4 bits + 16 levels) to INT8 (8 bits + 256 levels) for
dense projections. Per-tensor cos_sim would jump to ~0.999. Doubles the
weight VRAM (17 GiB → ~26 GiB), so no longer fits 24 GiB on this card —
only useful when the goal is "verify the engine works end-to-end on
*good* weights, even if not deployable on this card".

### Option 4: Hessian-aware MoE expert quant (~2 days; modest gain)
Replace `fused_expert_minmax_int4` with a router-aware GPTQ that only
includes calibration tokens routed to each expert. Substantial code
work; only worth it if dense projections (currently the worst tensors)
are first improved by Option 1 or 2.

## Recommendation

Start with **Option 1 (add scale search)**. Smallest blast radius, no
new infrastructure, directly addresses the worst-quality tensors in the
audit. If post-scale-search per-tensor cos_sim still under 0.98 across
the bake, escalate to Option 2.

## Verification protocol for any future bake

1. Run `oracle/bake_int4.py --model-dir <snap>` to produce
   `.supersonic/v2-int4-gptq/`.
2. Run the per-tensor cos_sim survey (see this doc's "What we found"
   section — script captured in shell history at audit time).
3. Acceptance: every quantized tensor cos_sim ≥ 0.98 vs safetensors.
4. End-to-end smoke: `cargo run --release --bin supersonic --model
   qwen3.6-35b-a3b --model-dir <snap> --prompt "The quick brown fox"
   --max-new-tokens 16 --temperature 0.8 --top-p 0.9` — output should
   be coherent English (not the current degenerate "UDAUDAUDA" pattern).

## Update: Option 1 (scale search) landed and re-baked

Scale search shipped in commit 91d7185. Re-bake completed in 32.8 min
on the local 35B-A3B with --num-samples 32 + cpu=20GiB max_memory.
Per-tensor cos_sim survey vs safetensors:

  tensor                                              old      new    delta
  layers.3.self_attn.q_proj                        0.9611   0.9787  +0.0176
  layers.3.self_attn.k_proj                        0.9711   0.9838  +0.0126
  layers.3.self_attn.o_proj                        0.9000   0.9357  +0.0357
  layers.0.linear_attn.in_proj_qkv                 0.9410   0.9617  +0.0208
  layers.0.linear_attn.in_proj_z                   0.9707   0.9823  +0.0116
  layers.0.linear_attn.out_proj                    0.8991   0.9423  +0.0432
  layers.0.mlp.shared_expert.gate_proj             0.8804   0.9278  +0.0474
  layers.0.mlp.shared_expert.up_proj               0.9523   0.9732  +0.0209
  layers.0.mlp.shared_expert.down_proj             0.9249   0.9569  +0.0319

Worst tensors gained the most (+0.04 to +0.05). Average +0.025.
Bake's own Python-side post-quant gen now produces a coherent first
token: "The quick brown fox jumps面面…" — "jumps" follows the prompt
correctly before degenerating into vocab gibberish.

**However**: our SuperSonic kernel chain on the same weights still
produces near-random output. Per-block parity tests at production
geometry against safetensors are bit-exact (separately verified), but
the 40-layer chain at production scale produces:

  greedy  "The quick brown fox" → "::.::.::.::.::.::.::.::.ivadspot…"
  T=0.8   "The quick brown fox" → "::. fundamentaisReviewer_color…"

Different prompts produce different first tokens (kernel responds to
input), but tokens aren't semantically appropriate. Since Python
forward on the same weights produces "jumps" but our chain doesn't,
the gap is in the kernel-side multi-layer chain at production scale —
NOT bake quality, NOT sampling, NOT the per-block kernels.

Acceptance criteria from this doc not met yet (worst tensor still
0.928 vs ≥ 0.98 target; coherent generation through kernel chain not
achieved). Next quality lever: localise the production-scale kernel
chain bug via a per-layer parity harness on real bake weights — load
layer N's INT4 from the bake, run kernel + Python single-layer ref
side-by-side, find which layer's output first diverges. That will
isolate the kernel bug independent of the bake.

## Update: HF parity bugs were the real blocker (PR #64)

The "kernel chain produces near-random output" symptom traced to two
HF-vs-SuperSonic decode-math discrepancies, NOT bake quality:

  1. **`Qwen3_5MoeRMSNorm` `(1.0 + weight)` unit offset** (line 819 of
     `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`). Our
     oracles + kernel + host code were doing plain `output * weight`,
     which scales norm output to ~zero against trained delta weights.
  2. **`q_proj` per-head interleaved `[q | gate]` split.** HF reshapes
     via `q_proj(x).view(..., H, head_dim*2).chunk(2, -1)`; we were
     splitting flat `[:H*d]`/`[H*d:]`, which only happens to work for
     head 0.

After PR #64 lands, end-to-end on the SAME `v2-int4-gptq` bake — no
rebake — produces:

    "The quick brown fox" → "jumps over the lazy dog."
    "The capital of France is" → "Paris. Paris is the most populous
                                  city in the European Union, …"
    "Q: List three primary colors. A:" → "Red, Yellow, and Blue.<EOS>"
    Python Fibonacci prompt → 80 tokens of correct, indented code

Per-tensor cos_sim survey on `v2-int4-gptq` post-scale-search,
re-measured with **F64 reductions** (F32 reductions on tensors
≥100M elements — e.g. lm_head — under-estimate norms enough to push
cos_sim above 1.0; it came out at 1.20 before fixing the precision):

| Class                                            | mean cos_sim | min    | tile count |
|--------------------------------------------------|--------------|--------|------------|
| `mlp.experts.gate_up_proj`  (fused, GPTQ)        | 0.987–0.993  | 0.957  | 128/expert |
| `mlp.experts.down_proj`     (fused, GPTQ)        | 0.987–0.993  | 0.960  | 64/expert  |
| `lm_head.weight`                                 | 0.987        | 0.987  | 31040      |
| `linear_attn.in_proj_z`                          | 0.982        | 0.982  | 512        |
| `self_attn.k_proj`                               | 0.984        | 0.984  | 64         |
| `self_attn.q_proj`                               | 0.978        | 0.978  | 1024       |
| `self_attn.v_proj`                               | 0.969        | 0.969  | 64         |
| `self_attn.o_proj`                               | 0.936        | 0.936  | 512        |
| `linear_attn.in_proj_qkv`                        | 0.961        | 0.961  | 1024       |
| `linear_attn.out_proj`                           | 0.942        | 0.942  | 512        |
| `mlp.shared_expert.gate_proj`                    | 0.928–0.967  | 0.928  | 64         |
| `mlp.shared_expert.up_proj`                      | 0.973–0.978  | 0.973  | 64         |
| `mlp.shared_expert.down_proj`                    | 0.949–0.957  | 0.949  | 64         |

The audit's earlier "experts are min/max-only" finding is now stale —
scale search (commit `91d7185`) was applied to BOTH the dense GPTQ
path AND the fused-expert path, lifting MoE-expert cos_sim into the
0.987–0.993 band. The remaining gap to ≥ 0.98 is concentrated in the
**dense layer-OUTPUT projections** (`o_proj`, `out_proj`, `down_proj`,
`shared_expert.gate_proj`) — they sit at the END of GPTQ's
sequential-calibration order and inherit the most accumulated error
from the projections quantized before them.

End-to-end output suggests the existing bake is now **functionally
adequate** for production use even though the formal ≥ 0.98 acceptance
bar isn't met on every tensor. Closing the remaining 1–7% gap on the
worst tensors is a separate, lower-priority quality push (Options 2,
4 from this doc) — no longer blocking coherent decode.

The survey is reproducible via:

    ~/venvs/rocm/bin/python oracle/qwen36_moe_bake_cossim_survey.py \
        --model-dir <snapshot> \
        --bake-dir <snapshot>/.supersonic/v2-int4-gptq \
        --layers 0,3 --out /tmp/qwen36_cossim_survey.tsv
