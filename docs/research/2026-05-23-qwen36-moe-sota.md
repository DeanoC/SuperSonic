# Qwen3.6 MoE Inference SOTA Notes for SuperSonic

Status: research snapshot, 2026-05-23

This note summarizes current public implementation practice for
`Qwen/Qwen3.6-35B-A3B` and adjacent Qwen3/Qwen3-Next MoE serving stacks, then
maps the useful ideas back to SuperSonic's HIP and Metal paths.

## Executive Takeaways

The state of the art is not a single-token expert GEMV with clever packing.
The best public stacks treat Qwen-style MoE as a scheduling and residency
problem:

- Batch or group routed expert work whenever there is more than one token.
- Keep expert representations resident in the layout consumed by the compute
  backend; avoid rebuilding active expert slabs per token.
- For multi-GPU servers, shard experts directly and make token dispatch/all2all
  a first-class backend.
- Use native MTP when the model ships MTP heads; it amortizes target-model work
  better than generic draft-model speculation for sparse MoE.
- For local single-device inference, the important question is where inactive
  experts live and whether active experts can be consumed without page faults or
  per-token transcodes.

For SuperSonic, the next useful implementation direction is therefore:

1. Treat decode FFN as a resident-expert scheduling problem, not a per-token
   pack/copy problem.
2. Keep the existing batched-prefill grouped-expert path as the template for any
   multi-token MoE work.
3. Prototype a resident FP16/MPS or Metal-tensor expert table only if it can be
   filled once per layer or updated by a measured hot-set policy.
4. Prototype native MTP only after the baseline decode path is not dominated by
   expert residency waits.

## Qwen3.6-35B-A3B Shape

The official model card describes Qwen3.6-35B-A3B as 35B total parameters with
about 3B active parameters, hidden size 2048, 40 language-model layers, and a
`10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))` hidden
layout. The MoE block has 256 routed experts, 8 active routed experts, one
shared expert, expert intermediate size 512, and native 262,144-token context,
with extension to roughly 1,010,000 tokens. It also says MTP was trained with
multiple steps and lists Transformers, vLLM, SGLang, KTransformers, and related
frameworks as compatible deployment targets.

Source: [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

Two details matter for SuperSonic:

- Decode has only one token of activation reuse, but every token touches 40
  layers x 8 routed experts plus the shared expert.
- Long-context prefill is a different regime: many tokens are available, so the
  right MoE primitive is grouped expert GEMM after router permutation.

## Public Serving Patterns

### vLLM

The Qwen3.6 model card recommends `vllm>=0.19.0`, uses tensor parallelism for
standard Qwen3.6 serving, exposes a Qwen3 MTP mode through
`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`,
and has a `--language-model-only` mode to skip multimodal overhead when text is
the target.

Source: [Qwen3.6-35B-A3B model card, vLLM section](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

For MoE specifically, vLLM's expert-parallel documentation states that when
expert parallelism is enabled, expert layers are sharded across the expert
parallel group while attention layers use tensor or data parallel behavior
depending on TP/DP shape. vLLM also supports DeepEP low-latency mode for
multi-node MoE dispatch.

Source: [vLLM expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

The vLLM fused-MoE design is modular: a prepare/finalize stage dispatches and
combines tokens, while expert kernels consume standard or batched activation
formats. The documented expert backends include Triton, DeepGEMM, CUTLASS FP8,
CUTLASS FP4, and FlashInfer; grouped GEMM is the core primitive rather than
launching independent per-expert rows.

Sources:

- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/v0.10.1.1/design/fused_moe_modular_kernel.html)
- [vLLM MoE kernel features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)

SuperSonic mapping:

- HIP prefill already matches the vLLM shape: router permutation -> grouped
  expert GEMM -> unpermute/combine.
- Metal decode experiments that pack active slabs per token are moving opposite
  to the vLLM design principle: they create a new prepare cost without enough
  token batch to amortize it.
- A SuperSonic equivalent for decode would need either persistent/resident
  expert layouts or a CUDA-graph-like command-buffer capture strategy, not just
  a faster copy.

### SGLang

The Qwen3.6 model card recommends `sglang>=0.5.10` and gives standard,
tool-use, and MTP commands. Its recommended MTP form uses `NEXTN` with multiple
speculative steps and draft tokens.

Source: [Qwen3.6-35B-A3B model card, SGLang section](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

SGLang's expert-parallel documentation is especially relevant because it names
the real serving decomposition: all-to-all dispatch backends and MoE computation
backends are separate knobs. Its all-to-all options include DeepEP, Mooncake,
NIXL, MORI for ROCm, FlashInfer, and Ascend fused EP. Its compute backends
include Triton grouped GEMM, DeepGEMM, CUTLASS, FlashInfer + TensorRT-LLM, and
MXFP4-oriented paths. The docs also call out DeepEP `normal` mode for prefill
and `low_latency` mode for decode.

Source: [SGLang expert parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)

SGLang also treats speculative decoding as a family of production mechanisms:
EAGLE-2/EAGLE-3, MTP, DFlash, draft-model speculation, and NGRAM variants.

Source: [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)

SuperSonic mapping:

- For single Apple GPU Metal, all-to-all is irrelevant, but the split between
  dispatch layout and expert compute is still useful.
- The `normal` versus `low_latency` distinction maps cleanly to SuperSonic's
  prefill versus decode split. Prefill can afford permutation and grouped GEMM;
  decode needs a low-latency resident path.
- SGLang's backend matrix argues for keeping the SuperSonic runner policy
  explicit: a backend should say whether it owns dispatch, expert compute,
  quantization format, and overlap.

### Hugging Face Transformers

Transformers is the reference model-definition path, not the performance path.
Its Qwen3MoE docs expose `Qwen3MoeModel` and `Qwen3MoeConfig`; the Qwen3 MoE
reference shape is 30.5B total, 3.3B active, 128 routed experts, 8 active
experts, and 48 layers. Qwen3.6-35B-A3B is a later Qwen3.5/Qwen3.6-family MoE
variant, but Transformers remains the canonical reference for configuration,
weight naming, and correctness.

Source: [Transformers Qwen3MoE docs](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)

SuperSonic mapping:

- Keep Transformers as the oracle/reference layer for config parsing,
  tokenization, and parity.
- Do not infer performance design from Transformers eager execution.

### llama.cpp

The official Qwen3.6 GitHub README says llama.cpp supports Qwen3.6 text and
vision models and points users to GGUF conversions.

Source: [QwenLM/Qwen3.6 README](https://github.com/QwenLM/Qwen3.6)

Mainline llama.cpp merged MTP support on 2026-05-16. The PR author reports
testing Qwen3.6 27B and Qwen3.6 35B-A3B, automatic MTP loading from the same
GGUF, around 75% steady-state acceptance with 3 draft tokens, and more than 2x
speedup over baseline in their posted benchmark. The same PR notes prompt
processing can take a negative hit because of device-to-host embedding
transfers, and parallel decoding with MTP is not fully optimized.

Source: [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)

There is also a community benchmark repository showing generic draft/ngram
speculative decoding on Qwen3.6-35B-A3B can lose on RTX 3090, even when the
draft tokens are high quality, because the MoE expert-routing and bandwidth
costs dominate. This is not an official framework result, but it is useful
primary evidence for the failure mode.

Source: [Qwen3.6 speculative decoding RTX 3090 benchmark repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)

SuperSonic mapping:

- Native MTP is worth tracking; generic draft-model speculation is less
  attractive for Qwen3.6 MoE unless the verification pass can amortize expert
  residency.
- A SuperSonic MTP implementation needs to account for prompt/prefill overhead,
  not just decode acceptance.

### KTransformers / KT-Kernel

KTransformers is relevant because it treats MoE as heterogeneous CPU/GPU
execution rather than assuming all experts live in accelerator memory. Its
KT-Kernel docs advertise CPU-optimized MoE kernels, AVX512 native precision
backends for FP8/BF16/INT4, AMX INT4/INT8 expert inference, universal CPU
fallback through a llamafile backend, and NUMA-aware execution. Release notes
also call out VNNI-256 support for GPTQ INT4 MoE and CPU-GPU expert scheduling.

Sources:

- [KT-Kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- [KTransformers releases](https://github.com/kvcache-ai/ktransformers/releases)

SuperSonic mapping:

- On UMA Apple silicon, "CPU versus GPU upload" is the wrong mental model, but
  residency and cache pollution still matter.
- The useful transplant is not a CPU fallback path by itself; it is an explicit
  expert scheduler with a known resident set, miss policy, and layout per
  backend.
- If SuperSonic keeps CPU-side INT4 transcode or FP16 MPS slab generation for
  Metal, it needs streaming writes and hot/cold expert accounting rather than
  per-token scalar packing.

## Technique Inventory

| Technique | Public-stack role | SuperSonic status | Recommendation |
| --- | --- | --- | --- |
| Router permutation + grouped expert GEMM | Standard MoE prefill/server batching shape in vLLM/SGLang | HIP batched prefill already has this shape | Keep investing here for long-context and batch workloads |
| Expert parallelism | Shard experts across GPUs, separate dispatch and compute | Not relevant to single Apple GPU, partially relevant to HIP multi-GPU future | Do not port now; preserve abstractions that separate dispatch from compute |
| DeepEP/low-latency dispatch | Reduce MoE all-to-all overhead in distributed decode | No single-device equivalent | Conceptually maps to a low-latency resident decode path |
| Native MTP / Qwen3 Next MTP | Multi-token decode without separate draft model | Not implemented in SuperSonic Qwen3.6 | Good next major feature after FFN residency is under control |
| Generic draft/ngram speculation | Workload-dependent; can lose on sparse MoE | SpecPrefill exists; generic spec decode is constrained | Prefer SpecPrefill for long prompts and native MTP for decode |
| Per-token active expert slab packing | Makes random expert gathers contiguous | Implemented as SuperSonic diagnostics; not promotable on M5 Max | Stop extending this path unless it becomes persistent/resident |
| Direct original-buffer gather | Avoids slab packing | Implemented as SuperSonic diagnostic; wait/residency dominated on M5 Max | Not enough by itself |
| Heterogeneous expert residency | Run hot experts on accelerator, cold experts elsewhere | Explored only indirectly | Worth designing as an explicit scheduler |
| FP8/FP4 MoE kernels | Production server sweet spot on Hopper/Blackwell | SuperSonic Qwen3.6 Metal lane is INT4 GPTQ | Track, but do not block Apple INT4 work on it |
| KV/cache compression | Enables long context and memory headroom | HIP has KV-FP8 work; Metal target does not claim it yet | Important after decode FFN and long-context harness stabilize |

## What SuperSonic Has Already Learned

Current local measurements in `docs/performance.md` show a tight negative
result:

- CPU active-slab packing proves the large original expert buffers cause real
  waits, but the pack cost is too high.
- Exact-route cache misses because active expert sets churn token-to-token.
- LRU hotset reduces copy bytes but not enough to beat the default lane.
- GPU-side active-slab packing is correct but slower.
- Direct original-buffer gather is correct but still wait/residency dominated.

This aligns with the public SOTA: real systems do not rely on rebuilding
per-token expert slabs in the decode hot path. They either batch/group across
tokens, shard experts with explicit dispatch, or keep a backend-native resident
representation.

Local sources:

- [Apple M5 Max Metal Qwen3.6 measurements](../performance.md#metal--apple-m5-max-apple-m5-max)
- [Qwen3.6 batched-prefill grouped MoE design](2026-05-05-qwen36-moe-batched-prefill-results.md)

## Recommended SuperSonic Roadmap

### 1. Make Expert Residency a First-Class Object

Create a small runtime concept for Qwen3.6 routed experts:

- `resident_format`: native INT4, dequantized FP16/MPS, MTLTensor, CPU INT4.
- `scope`: per-layer, per-device, per-session.
- `capacity`: number of experts or bytes.
- `miss_policy`: none, exact route, LRU, static top-N, offline profile.
- `metrics`: slot hits, misses, evictions, bytes touched, CPU cache policy,
  GPU command-buffer wait, GPU elapsed time.

The existing packed-cache/hotset probes can become implementations of this
interface instead of one-off environment branches.

### 2. Port the Batched-Prefill MoE Shape to Metal Before More Decode Tricks

Metal long-context prefill is still the place where grouped expert GEMM can pay
off naturally. The HIP design already has:

- router top-k over many prompt tokens,
- expert counting/permutation,
- grouped expert compute,
- unpermute/combine,
- shared expert batched path.

A Metal version should start with moderate chunks and correctness-first JSON
rows. This is closer to vLLM/SGLang SOTA than another single-token GEMV probe.

### 3. Evaluate Resident MPS/MPP Expert Tables With No Per-Token Rebuild

The MPS pilot rows suggest vendor dense GEMM can be much faster than the current
INT4 GEMV plumbing, but the current bridge pays transcode/rebuild costs. The
next experiment should fill a resident table once, then run several decode
tokens against it:

- static top-N experts per layer from a calibration prompt set,
- fallback to default host expert path for misses,
- profile quality and hit-rate on coding/agentic prompts, not only `"Hello"`.

This is the Apple-silicon analog of KTransformers-style heterogeneous expert
residency.

### 4. Add Native MTP as a Separate Feature Track

Qwen3.6 ships MTP-trained weights, and both Qwen's model card and public
llama.cpp work point to native MTP as the preferred speculative path. For
SuperSonic:

- Start with a model-file audit: confirm which MTP tensors exist in the local
  bake and whether the current INT4 bake preserves them.
- Build a tiny MTP loader/parity harness before touching the decode loop.
- Measure one-token and multi-token acceptance separately from FFN latency with
  `SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE=1` and
  `tests/metal/probe_qwen36_mtp_acceptance.py`.
- Keep Metal enablement env-gated while it is experimental:
  `SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT=1` allows sequential K=1 only.
- Keep SpecPrefill separate; it is a long-prompt prefill optimization, not a
  decode replacement.

### 5. Preserve Negative Results as Gates

Do not promote a Qwen3.6 Metal FFN path unless it beats the current default
under the same smoke:

- generated IDs match,
- headline `ms/token` improves,
- `ffn_ms_avg` improves,
- Metal profile does not merely move time into `command_buffer_wait`,
- full-attention, linear-attention, and lm-head do not regress materially.

## Concrete Next PR Candidates

1. **Resident expert scheduler scaffold**
   Add a small internal abstraction plus profile rows; initially wrap the
   existing exact-cache and hotset probes without changing behavior.
   Initial implementation landed as the `Qwen36ExpertResidency*` profile
   surface, which emits `[qwen36-expert-residency]` summary rows and
   `[qwen36-expert-residency-policy]` rows while preserving the legacy
   `[qwen36-pack-cache]` line for older bench parsers.

2. **Metal batched-prefill MoE feasibility harness**
   Implement only router permutation metadata and a no-op/reference combine
   first, then add grouped expert compute. This should target long-context
   prefill, not decode.
   Initial implementation landed a Metal-safe metadata lane: the long-context
   harness can run with `--batched-prefill-feasibility`, keeping the supported
   per-token Metal prefill execution while emitting
   `[qwen36-batched-prefill-feasibility]` rows derived from actual router
   choices. The row reports chunk count, permutation entries, touched expert
   segments, average rows per segment, max segment size, and WMMA16 coverage.
   The second slice adds `[qwen36-batched-prefill-plan]` rows for candidate
   chunk sizes. These rows keep the path `metadata_only=1`, but make the next
   grouped-compute PR concrete by exposing scalar-tail assignments, WMMA16
   padded assignments, and padding overhead for 64/128/256/512/1024-token
   chunks, with an env override for local what-if sweeps. The first v3
   512-token Metal smoke profiled 417 prefill tokens and showed 512/1024-token
   chunks tied at 82.7% WMMA16 assignment coverage and 54.6% padding overhead,
   while chunk 64 fell to 37.3% coverage and 228.0% padding overhead.

   The third slice adds an opt-in Metal grouped-compute prototype behind
   `SUPERSONIC_QWEN36_MOE_METAL_BATCHED_PREFILL_PROTOTYPE=1` and
   `tests/metal/bench_qwen36_longctx.py --batched-prefill-prototype`. It routes
   batched full-attention through the existing Metal prefill primitive and
   replaces the HIP M9/M10/M11 grouped expert sequence with a direct INT4 Metal
   routed-expert gate/up and down/combine kernel pair. This is still
   experimental, not policy-promoted: router/top-k and shared-expert work remain
   on the existing host/primitive path. The first normal 512-token smoke
   generated the same `[271]` sanity row and measured 22.15s prefill; the
   profiled run measured 34.20s prefill and showed 1.903s GPU time in
   gate/up, 2.045s in down/combine, 33.184s in command-buffer wait, and 4.335s
   in HAL `copy_h2d`. On Apple UMA this is buffer materialization/copy
   bookkeeping rather than PCIe upload. That shifts the next optimization
   question from "can the grouped MoE shape execute on Metal?" to "how do we
   cut prefill orchestration, UMA `copy_h2d` volume, and linear-attention
   command-buffer volume?"

   The fourth slice removes the shared-expert scalar-gate host broadcast from
   that prototype. A native Metal BF16 row-scalar sigmoid multiply consumes the
   `[N, 1]` shared gate directly and writes the `[N, hidden]` shared output,
   avoiding the D2H scalar read, expanded-gate H2D write, and per-layer temp
   allocation. The follow-up 512-token normal smoke measured 12.47s prefill and
   172.54 ms/token with the same `[271]` sanity row. The profiled
   `sigmoid_mul_row_scalar` row measured 11.859 ms native wall and 0.725 ms GPU
   time across 40 layers, so the next target remains orchestration plus the
   measured linear-attention, full-attention, and routed-expert direct rows.

   The fifth slice trims linear-attention orchestration without changing the
   linear kernel math. The Metal stage-5 native finalizer can now write its
   residual output to a separate chunk row while reusing the existing output
   scratch for normalized hidden state. The batched-prefill prototype opens one
   Metal command-buffer batch per linear-attention layer in normal runs,
   encodes each token sequentially with barriers, and avoids the per-token CPU
   D2D row copy. A 512-token control with
   `SUPERSONIC_QWEN36_MOE_METAL_LINEAR_PREFILL_DIRECT=0` measured 13.30s
   prefill; the direct-row batch measured 10.89s prefill and preserved the
   `[271]` sanity row. `SUPERSONIC_METAL_PROFILE=1` keeps the waited path so
   phase attribution remains comparable.

   The sixth slice tests the obvious full-attention layout shortcuts and keeps
   only the neutral scratch reuse on by default. Full-attention prefill now
   reuses Q/K-after-norm and KV-prefix scratch buffers across layers instead
   of allocating them inside every full-attention layer call. Two Metal layout
   probes remain env-gated negative results: a time-major KV attention kernel
   (`SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_TMAJOR=1`) measured 11.09s prefill,
   and a single-kernel Q/gate split
   (`SUPERSONIC_QWEN36_MOE_METAL_SPLIT_QGATE=1`) measured 11.00s prefill.
   The default path with both probes off measured 10.73s prefill and preserved
   the `[271]` sanity row, so neither layout probe is promoted. The next useful
   target is routed expert compute/residency rather than more launch-only
   full-attention reshuffling.

   The seventh slice tests two small routed-FFN orchestration ideas and leaves
   both disabled by default. A fused Metal residual-add kernel preserves the two
   BF16 rounding points from `chunk_hidden += combined; chunk_hidden +=
   shared_out`, but measured 11.03s prefill versus a 10.65s disabled-path
   control. A native Metal router softmax/top-k kernel avoids the router logits
   D2H readback and top-k H2D writes; standalone dispatch measured 11.51s
   prefill, and batching it with routed expert direct still measured 11.63s,
   while the host-top-k control measured 11.05s. The diagnostics remain
   available behind `SUPERSONIC_QWEN36_MOE_METAL_FUSED_FFN_RESIDUAL=1` and
   `SUPERSONIC_QWEN36_MOE_METAL_ROUTER_TOPK=1`, but the default lane stays on
   the measured faster host-top-k and two-add residual path.
   The long-context harness now has a `--batched-prefill-variant` selector for
   these measured env-gated probes (`linear-direct-off`, `full-attn-tmajor`,
   `split-qgate`, `router-topk`, and `fused-residual`) and records the selected
   variant plus env overrides in JSON rows. This keeps the negative/prototype
   gates reproducible without promoting them into the SuperSonic CLI.

   The eighth slice adds `tests/metal/sweep_qwen36_batched_prefill_variants.py`
   so those prototype gates can be compared as a small suite instead of
   one-off manual runs. The sweep includes the supported `baseline` lane,
   `prototype-default`, and selected named variants, uses the same
   deterministic NIAH prompt per context, records generated-ID parity, and
   writes JSON/Markdown rows with prefill ratios versus baseline plus optional
   Metal/HAL profiles. This keeps the "do not promote unless it wins" rule
   machine-readable for future prefill/orchestration changes.

   The ninth slice turns that rule into a nonfatal `promotion_gate` in the
   variant sweep schema. A candidate must preserve generated IDs, improve
   prefill, headline decode, and `ffn_ms_avg`, keep full-attention,
   linear-attention, and lm-head inside the configured regression threshold,
   and provide non-regressed `command_buffer_wait` profile evidence by default.

3. **Static top-N resident MPS table probe**
   Use route profiles to choose top-N experts per layer, materialize FP16 MPS
   RHS once, fall back on misses, and measure real prompts.
   Initial instrumentation landed the route data needed to make this
   experiment concrete: `[qwen36-route-topn-layer]` exposes per-layer expert
   IDs/counts for static tables, `[qwen36-route-call]` exposes real active
   expert sets, and `tests/metal/probe_qwen36_static_topn.py` turns calibration
   and evaluation prompt runs into coverage, fallback-call, worst-layer,
   resident native INT4 size, resident FP16 MPS RHS size, and `static_tables`
   rows. The first runtime slice consumes those tables as an opt-in native
   INT4 packed static top-N path; the larger static MPS/MPP table remains a
   follow-up.
   The first local smoke is a negative result for small capacities: top-N 16
   costs 3.75 GiB of FP16 RHS storage but covered only 35.8% of assignments and
   fully served 4/880 layer calls on a coding-shaped evaluation prompt.
   A regenerated v2 capacity-64 table improved evaluation assignment coverage
   to 69.858%, but the first one-token runtime smoke still hit only 9/40 layers
   exactly and measured 507 ms/token with 369.071 ms in FFN. Treat the native
   static table as validated plumbing for warm-reuse/hybrid-fallback experiments,
   not as a promoted path.
   The next harness slice is
   `tests/metal/sweep_qwen36_static_topn_runtime.py`, which compares warm decode
   modes (`default`, `static`, `static-hotset`, etc.) against the same generated
   IDs per prompt and records expert-residency policy rows so promotion
   decisions can be made on multi-token reuse rather than a cold first-token
   allocation. The sweep schema now also preserves parsed Metal/HAL profiles
   when `--metal-profile` is used, so coding/profiling prompt sets can be used
   as attribution gates instead of relying on a single Hello run.
   A follow-up v3 schema adds a nonfatal `promotion_gate` for resident FFN
   modes: generated IDs must match `default`, headline ms/token and
   `ffn_ms_avg` must improve, full-attention, linear-attention, and lm-head
   must stay inside the configured regression threshold, and
   `command_buffer_wait` profile evidence is required by default.
   The first four-token smoke preserved `[11, 353, 599, 264]` across default,
   static, and static+hotset, but default still won: `decode_ms=702` and
   `ffn_ms_avg=98.761` versus static at `decode_ms=951` / `ffn_ms_avg=177.563`
   and static+hotset at `decode_ms=1450` / `ffn_ms_avg=262.215`. Static top-N
   therefore remains a negative gate and a profiling scaffold, not the next
   runtime promotion.
   The resident MPS follow-up is now a viability harness rather than a runtime
   replacement. `tests/metal/probe_qwen36_mps_resident_table.py` consumes the
   static top-N report plus the `[qwen36-moe mps-expert-pilot]` row and writes
   cost rows for all-resident MPS, full-hit-only fallback, and optimistic
   partial-hit fallback. With the current v2 static table and direct-pilot
   timings (`gate_up_ms=1.312`, `down_ms=0.757`), capacity 64 needs 15.00 GiB
   of resident FP16 RHS data, covers 69.9% of routed assignments, fully serves
   only 9.8% of layer calls, estimates 97.20 ms/token for a full-hit-only
   bridge, and estimates 87.58 ms/token only under the optimistic partial-hit
   model. The next runtime implementation should therefore be a partial-hit
   resident table, not another full-hit-only static bridge.
   A follow-up v2 schema adds a nonfatal `viability_gate` to this probe. It
   evaluates full-hit-only and optimistic partial-hit candidates against
   resident RHS footprint, projected FFN speedup, route coverage, and full-hit
   thresholds, while keeping the runtime sweep as the authority for actual
   promotion.
   The first partial-hit runtime prototype is now env-gated behind
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_STATIC_TOPN_PARTIAL=1`.
   It caches a per-layer FP16 MPS RHS table for the static top-N experts,
   dispatches an indexed MPS bridge for resident-hit groups, and computes miss
   groups on the existing host INT4 path before combining the contributions.
   `tests/metal/sweep_qwen36_static_topn_runtime.py` includes the
   `mps-static-partial` mode so this path can be judged against the same
   generated-token parity and expert-residency rows as the native INT4 static
   probes.
   The v4 sweep adds `mps-static-partial-prewarm`, which sets
   `SUPERSONIC_METAL_PREWARM_QWEN36_FFN_MPS_STATIC_TOPN=1` and builds the
   static FP16 MPS RHS cache during layer setup. This keeps the normal decode
   row honest by separating one-time resident-table materialization from the
   steady-state per-token MPS bridge and host-miss costs.
   The first measurement is a useful rejection signal: a profiled one-token
   smoke preserved `[11]`, but measured `decode_ms=6324`,
   `ffn_ms_avg=6073.323`, `slot_hit_rate=0.731250`, and
   `copied_bytes=15753805824`; the indexed MPS bridge was `365.052 ms` across
   40 calls, while FP16 LUT packing alone took `5630.663 ms`. The warm
   four-token sweep also preserved `[11, 353, 599, 264]`, but regressed from
   `default` `decode_ms=702` / `ffn_ms_avg=94.930` to `mps-static-partial`
   `decode_ms=7839` / `ffn_ms_avg=1845.066`. Do not promote this path without
   eliminating FP16 RHS materialization/copy and the per-layer MPS/host split.

4. **Qwen3.6 MTP tensor audit**
   Parse local safetensors/bakes for MTP heads and write down the loader delta
   before implementing speculative decode.
   Initial implementation landed as `tests/metal/audit_qwen36_mtp.py`. The
   local M5 Max cache reports the FP8 source snapshot as complete with 1,560
   `mtp.*` tensors, and the INT4 bake manifest as complete with the 19 folded
   runtime tensors consumed by `mtp_loader.rs`. The loader delta is therefore
   "model files ready, Metal speculative decode still policy-blocked until the
   runtime path is enabled."

5. **Qwen3.6 MTP acceptance probe**
   Add machine-readable `[qwen36-mtp-acceptance]` telemetry to the runner and
   a harness that records either real acceptance or the current Metal policy
   block. The telemetry separates drafted tokens, accepted tokens, base verify
   chains, batched replay chains, and target steps per emitted token, so MTP can
   be judged independently of FFN kernel latency.

6. **Qwen3.6 Metal K=1 MTP experiment**
   Keep the supported Metal lane blocked by default, but add an explicit
   `SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT=1` escape hatch for sequential K=1
   acceptance runs. This is a measurement path only: batched verify and K>1 are
   still out of policy until K=1 proves useful and correct on local prompts.
   The first local Metal run measured `drafted_tokens=2`, `accepted_tokens=1`,
   `acceptance_rate=0.5`, and `target_steps_per_emitted=1.0` in 24.3s, so the
   current value is correctness/telemetry proof rather than throughput.

7. **Qwen3.6 MTP prompt-suite acceptance sweep**
   Promote the single-prompt K=1 experiment into a small smoke/comparison suite
   before changing Metal policy again. `tests/metal/sweep_qwen36_mtp_acceptance.py`
   runs the same env-gated path across profiling, coding, reasoning, and summary
   prompts, aggregates accepted/drafted tokens and target steps per emitted
   token, records policy-blocked rows when the experiment gate is absent, and
   now emits a `promotion_gate` summary plus optional Metal/HAL profile rows
   when `--metal-profile` is used.
   The first Metal smoke sweep measured 2/2 prompts in 34.7s with aggregate
   `accepted_tokens=1`, `drafted_tokens=4`, `acceptance_rate=0.25`, and
   `target_steps_per_emitted=1.0`; the profiling prompt accepted 0/2 drafts and
   the coding prompt accepted 1/2. K=1 therefore remains an instrumentation path,
   not a policy-promotion candidate.

8. **Qwen3.6 route residency decision sweep**
   Turn the route-locality telemetry into a prompt-suite decision before
   another resident-expert runtime experiment. `tests/metal/sweep_qwen36_route_residency.py`
   runs with `SUPERSONIC_QWEN36_ROUTE_PROFILE=1`, aggregates
   `[qwen36-route-profile]`, `[qwen36-route-cache-sim]`, and
   `[qwen36-route-topn]` rows across smoke or comparison prompts, and writes
   `target/qwen36_route_residency_sweep.{json,md}`. The v1 `decision_gate`
   compares per-layer LRU hit-rate with oracle static top-N coverage, then
   recommends a larger LRU resident cache, static resident table, or fused
   routed INT4 branch.

9. **Qwen3.6 LRU resident-cache runtime sweep**
   Turn `prototype_larger_lru_resident_cache` from a route-locality
   recommendation into a runtime promotion gate. `tests/metal/sweep_qwen36_lru_resident_cache.py`
   reuses the existing native INT4 packed hotset path, compares default decode
   with capacity-labeled `lru-hotset-N` modes, captures residency and
   Metal/HAL profile rows, and writes
   `target/qwen36_lru_resident_cache_sweep.{json,md}`. Its nonfatal
   `promotion_gate` uses the same runtime contract as the static and fused
   probes: generated IDs must match default, headline decode and `ffn_ms_avg`
   must improve, full-attention/linear-attention/lm-head must stay within the
   regression threshold, and `command_buffer_wait` evidence must be present and
   non-regressed when profile evidence is required.

10. **Qwen3.6 fused routed INT4 runtime sweep**
   Preserve the fused routed INT4 branch as a measured runtime gate rather than
   a note in the performance docs. `tests/metal/sweep_qwen36_fused_routed_int4.py`
   compares `default`, `direct-gather`, `gpu-pack`, and the larger
   `full-stage5` native FFN path under the same prompt suite, captures
   Metal/HAL profile rows when requested, and writes
   `target/qwen36_fused_routed_int4_sweep.{json,md}`. Its nonfatal
   `promotion_gate` uses the same promotion contract as the resident-runtime
   probes: generated IDs must match default, headline decode and `ffn_ms_avg`
   must improve, full-attention/linear-attention/lm-head must stay within the
   regression threshold, and `command_buffer_wait` evidence must be present and
   non-regressed when profile evidence is required. The v2 schema adds
   `full-stage5` via `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5=1`, so the
   doc's larger fused-native INT4 recommendation is now a repeatable gate
   instead of a hand-run smoke. The first one-token profiled v2 smoke preserved
   `[11]` but rejected `full-stage5`: `default` measured `264.2 ms/token` with
   `ffn_ms_avg=116.643`, while `full-stage5` measured `1190.1 ms/token` with
   `ffn_ms_avg=1035.294` and `command_buffer_wait=1121.367 ms`.
   The refreshed four-mode smoke wrote schema v2 and rejected every candidate:
   `default` generated `[11, 271, 40, 599]` at `201.4 ms/token` /
   `ffn_ms_avg=108.141`, while `direct-gather` and `full-stage5` generated
   `[11, 353, 599, 264]` and therefore failed generated-ID parity before their
   FFN/wait regressions are considered.

11. **Qwen3.6 SOTA gate summary**
   Preserve negative results as one roadmap-level artifact instead of leaving
   the promotion/viability decisions scattered across individual JSON files.
   `tests/metal/summarize_qwen36_sota_gates.py` reads the batched-prefill
   variant sweep, static top-N runtime sweep, fused routed INT4 runtime sweep,
   MPS resident-table probe, route residency sweep, MTP acceptance sweep, LRU
   resident-cache sweep, linear decode sweep, and full-attention decode sweep
   reports, then writes
   `target/qwen36_sota_gate_summary.{json,md}`. The v10 schema records input
   health, report age, passed and failed gate IDs, candidate failures, and a
   single `next_action`, plus the refresh command for each gate. It also marks
   passed estimate or decision gates as superseded when a newer runtime gate has
   already measured and rejected the corresponding candidate, so the next action
   cannot loop back to an already-negative prototype. Missing reports are nonfatal rows
   by default; `--require` turns absent, malformed, schema-mismatched, or
   missing-gate artifacts into a failed validation run, and `--max-age-hours`
   adds an mtime-based stale-report gate for local refresh runs.
   `tests/metal/refresh_qwen36_sota_gates.py` consumes the same gate
   surface and writes `target/qwen36_sota_gate_refresh_plan.{json,md}` so stale
   or missing SOTA gates can be selected in one dry-run artifact before the
   operator chooses `--run`.

12. **Qwen3.6 next-bottleneck selector**
   Close the loop when every SOTA runtime fork is negative. `tests/metal/select_qwen36_next_bottleneck.py`
   reads the refreshed SOTA gate summary plus the profiled default rows from
   the static top-N, fused routed INT4, LRU resident-cache, batched-prefill
   linear/full-attention/lm-head decode sweeps, and the latest optional `bench-perf`
   schema-v9 Qwen3.6 INT4
   artifact under `target/bench-runs`. It ranks decode buckets, records the
   prefill best-vs-baseline row, preserves top Metal/HAL profile ops, and writes
   `target/qwen36_next_bottleneck.{json,md}`. The v6 selector also reads
   adjacent `meta.json` worktree fingerprints and only auto-consumes bench
   artifacts whose git SHA plus diff hash match the current checkout, so
   rejected uncommitted experiments do not silently become the headline default.
   Historical JSON can still be supplied explicitly with `--bench-perf-json`.
   If FFN is still the dominant bucket but the resident/static/fused/MPS/LRU FFN
   forks have all failed or been superseded, the selector names the largest
   non-exhausted bucket instead of looping back to an already-negative FFN
   residency path.

13. **Qwen3.6 linear-attention decode direct-output handoff**
   Act on the selector's first recommendation with a small orchestration
   change before inventing a new linear-attention kernel. Decode now uses the
   existing Metal stage-5 INT4 linear launcher with a final-output override, so
   linear layers publish directly into the next residual buffer instead of
   writing `attn_output` and then issuing a D2D copy. The old handoff remains
   available for bisection with
   `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT=1`. A 2026-05-24
   `bench-perf` comparison measured a flat headline (`144.7 ms/token` direct
   versus `145.2 ms/token` disabled) while reducing HAL `copy_d2d` from 840
   calls / 3.44 MiB to 210 calls / 0.86 MiB, so the measured next target remains
   command-buffer/native linear work and FFN rather than the residual copy.

14. **Qwen3.6 bench attribution split**
   Keep the next-bottleneck selector honest by separating normal timing
   attribution from split-dispatch Metal profiling. `bench-perf` schema v9
   stores `stage_timings`, `chain_breakdown`, and `lifecycle_timings` from an
   unprofiled `--emit-stage-timings` run, then runs a separate
   `SUPERSONIC_METAL_PROFILE=1` attribution pass for `metal_profile` and
   `hal_profile`. The profiled timing maps remain available under
   `profile_stage_timings`, `profile_chain_breakdown`, and
   `profile_lifecycle_timings`, so command-buffer split overhead can be studied
   without making it the headline stage attribution. The same run directory's
   `meta.json` now includes git dirty paths and a diff hash for selector
   artifact hygiene. The next-bottleneck selector's v5 schema consumes this
   artifact when present, keeping the current headline run in the same evidence
   bundle as the narrower sweep rows.

15. **Qwen3.6 linear INT4 paired-nibble projection loop**
   Act on the selector's linear-attention target inside the measured dominant
   linear sub-ops. The Metal stage-5 linear INT4 projection and out-proj loops
   now consume both nibbles from each packed byte per SIMD lane load, while
   preserving the existing BF16 dequant rounding before accumulation. On the
   Apple M5 Max Qwen3.6 INT4 `bench-perf` lane this moved the headline median
   from `162.6` to `145.5 ms/token` and unprofiled `linear_attn_ms_avg` from
   `31.335` to `28.690 ms`. Split-profile GPU rows confirm the intended local
   effect: `qwen36_linear_int4_projections` dropped from `175.821` to
   `70.896 ms` total and `qwen36_linear_int4_out_proj_finalize` from `58.545`
   to `28.171 ms` total. The next measured bottleneck remains FFN host expert
   work by absolute time, but the current negative FFN residency gates still
   make the next actionable target the largest non-exhausted bucket selected by
   the v6 selector.

16. **Qwen3.6 linear recurrent beta/g hoist rejection**
   The next tiny linear-kernel idea was measured and rejected instead of
   promoted. A q/k-repeat precompute variant moved beta/g into the existing
   q/k repeat dispatch, but the 2026-05-24 `bench-perf` lane stayed headline
   flat (`145.1 ms/token` versus `145.5`) and did not improve normal
   `linear_attn_ms_avg` (`28.608` versus `28.690`); split-profile rows also
   moved the wrong way for recurrent/qk attribution
   (`qwen36_linear_int4_recurrent_update` `33.313 -> 48.187 ms` total and
   `qwen36_linear_int4_qk_norm_repeat` `3.611 -> 13.292 ms` total). A
   lane-0 threadgroup variant also stayed headline flat (`145.6 ms/token`) and
   kept the split recurrent row slower (`50.699 ms` total). A SIMD-broadcast
   smoke preserved `[11]`, but its one-token chain timing regressed, so it was
   not run as a promotion candidate. Keep the paired-nibble stage-5 linear
   kernel as the current default; the next useful work should be a larger
   orchestration/kernel change with a clearer normal-run win, not another
   beta/g hoist.

17. **Qwen3.6 linear decode variant sweep**
   The selector's linear-attention recommendation now has a repeatable gate
   instead of another hand-run experiment. `tests/metal/sweep_qwen36_linear_decode.py`
   compares the current default, the old residual handoff
   (`direct-off` / `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT=1`),
   and the host linear fallback
   (`host-linear` / `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5=1`)
   under the same deterministic prompts. It records generated-ID parity,
   `stage_timings`, `chain_breakdown`, lifecycle timings, optional Metal/HAL
   profiles, `qwen36_linear_int4_stage5`, subdispatch totals, `copy_d2d`, and
   `command_buffer_wait`. Its nonfatal promotion gate requires generated IDs
   to match default, at least two generated tokens before a row can pass,
   headline and `linear_attn_ms_avg` to improve, FFN/full-attention/lm-head to
   stay inside the regression threshold, and non-regressed command-buffer wait
   evidence when profiling is required. The SOTA gate summary now tracks this
   report, and the next-bottleneck selector marks `linear_attn_ms_avg`
   exhausted only after this gate has failed. That lets a measured negative
   linear sweep move the next target to full attention instead of cycling
   through smaller linear micro-tweaks. The first four-token Metal smoke
   rejected both candidates: `direct-off` preserved the default IDs
   `[11, 271, 40, 599]` but regressed headline decode (`860 ms` versus
   `827 ms`) and full-attention attribution, while `host-linear` generated
   `[11, 353, 599, 264]` and regressed headline, linear-attention, and lm-head.

18. **Qwen3.6 full-attention decode handoff gate**
   The selector's full-attention recommendation now starts with the smallest
   measurable orchestration change before attempting a real Metal attention
   kernel. Metal INT4 full-attention stage 5 can write its final residual
   directly into the decode ping-pong buffer through
   `attn_step_stage5_metal_host_into`, skipping the old `attn_output` to
   residual D2D copy for full-attention layers. Because the first measured
   profile showed the direct handoff slower than the old path, it is opt-in
   with `SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_DECODE_DIRECT=1` instead of
   the default.
   `tests/metal/sweep_qwen36_full_decode.py` compares default against the
   `direct` candidate, records generated-ID parity, timings, optional Metal/HAL
   profiles, full-attention host profile totals, `copy_d2d`, and
   `command_buffer_wait`, and writes
   `target/qwen36_full_decode_sweep.{json,md}`. Its nonfatal gate only passes
   a candidate if it preserves IDs, has enough generated tokens, improves
   headline and `full_attn_ms_avg`, and keeps FFN/linear/lm-head plus wait
   attribution within thresholds. The SOTA summary v10 tracks this report, and
   the next-bottleneck selector v6 marks `full_attn_ms_avg` exhausted after the
   full-attention gate fails so the workflow can move on to the lm-head tail or
   a larger measured kernel target.

19. **Qwen3.6 lm-head tail gate**
   The selector's lm-head recommendation now has a narrow runtime experiment
   before larger tail-kernel work. `SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX=1`
   keeps greedy top-1 selection on Metal after the existing RMSNorm + lm-head
   matmul, then reads back only the selected `u32` token instead of the full
   BF16 vocab logits. The gate is deliberately limited to Metal, greedy
   sampling (`temperature <= 0` or `top_k == 1`), and non-debug runs where
   `--dump-last-logits` / `SUPERSONIC_QWEN36_DUMP_LOGITS` do not require the
   full logits buffer on the host. The default path still downloads logits and
   samples on the CPU for parity, non-greedy sampling, and debugging.
   `tests/metal/sweep_qwen36_lm_head_tail.py` compares default against
   `gpu-argmax`, records generated-ID parity, stage and chain timings,
   optional Metal/HAL profiles, `argmax_bf16`, `copy_d2h`, and
   `command_buffer_wait`, and writes
   `target/qwen36_lm_head_tail_sweep.{json,md}`. The SOTA summary v10 tracks
   this report, and the next-bottleneck selector v6 marks `lm_head_ms_avg`
   exhausted only after the lm-head tail gate fails; once all narrow runtime
   gates are exhausted, it returns to the dominant measured bucket for a larger
   kernel or orchestration change. The first four-token M5 Max smoke preserved
   the default IDs `[11, 271, 40, 599]` and slightly improved headline
   ms/token (`235.1 -> 233.7`), but failed promotion because `lm_head_ms_avg`
   regressed (`9.044 -> 9.389`). The useful lesson is that full-logit D2H was
   already tiny (`copy_d2h` `0.067 -> 0.004 ms` total), so a future lm-head
   win needs a fused lm-head/top-1 tail or a larger dense-matmul change rather
   than just replacing host argmax with a separate Metal argmax dispatch.

20. **Qwen3.6 static MPS RHS prewarm gate**
   After all narrow decode buckets were exhausted, the selector returned to
   FFN for a larger measured path. The next resident-table slice is an
   opt-in setup prewarm, not a default runtime change:
   `SUPERSONIC_METAL_PREWARM_QWEN36_FFN_MPS_STATIC_TOPN=1` builds the existing
   static top-N FP16 MPS RHS cache during layer setup, using the same shared
   Metal buffers the decode bridge consumes. The static-topN runtime sweep's
   `mps-static-partial-prewarm` mode captures a `[qwen36-moe ffn-prewarm]`
   row with warmed layers, allocations, copied bytes, and elapsed setup time,
   while decode timings show whether the warm resident bridge is still slower
   after first-token materialization is removed from the measured token loop.
   The first prewarm smoke preserved `[11]` and warmed all 40 layers, with
   `resident_capacity=64`, `copied_bytes=15753805824` (`14.672 GiB`), and
   `elapsed_ms=6666.916`. Runtime expert-residency copies dropped to zero, but
   decode still regressed from `default` `333.2 ms/token` /
   `ffn_ms_avg=159.650` to `mps-static-partial-prewarm`
   `2473.8 ms/token` / `ffn_ms_avg=1543.206`; `command_buffer_wait` also
   regressed from `130.018 ms` to `861.632 ms`. This is the clean decision
   point before attempting a larger FFN kernel: the next FFN work should be a
   fused native INT4 compute path rather than more FP16 MPS table residency.
   That path is now wired into the fused-routed runtime sweep as
   `full-stage5`, keeping it env-gated and judged by the same generated-token,
   FFN, component-regression, and command-buffer-wait contract as the smaller
   direct-gather and GPU-pack FFN variants.
   The refreshed v4 static sweep keeps this negative decision intact:
   `mps-static-partial-prewarm` warmed all 40 layers and removed decode-loop
   residency copies (`copied_bytes=0` in the runtime row after a
   `15753805824` byte prewarm), but still generated `[11, 353, 599, 264]`
   instead of the default `[11, 271, 40, 599]` and regressed to
   `578.8 ms/token` / `ffn_ms_avg=480.816`.

21. **Qwen3.6 refreshed SOTA selector after FFN v4/static-v4 gates**
   With the static top-N runtime report refreshed to schema v4 and the
   fused-routed INT4 report refreshed to schema v4, the SOTA summary loads all
   ten gate reports cleanly. The latest selector still names
   `prototype_new_ffn_residency_or_compute_path` with `ffn_ms_avg` as both the
   target and dominant bucket: median default FFN is `111.557 ms`, versus
   `67.456 ms` for linear attention, `19.257 ms` for full attention, and
   `8.923 ms` for lm-head. Since all tracked narrow FFN/linear/full/lm-head
   gates are now exhausted, the next FFN change should be a larger native INT4
   kernel design, not another small env-gated fork of the current decode path.

22. **Qwen3.6 stage-5 router-in-Metal FFN probe**
   The fused-routed INT4 sweep includes `full-stage5-router`, guarded by
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER=1`, to test whether
   moving RMSNorm, router logits, softmax/top-k, and routed/shared INT4 stage-5
   projection work into the native Metal FFN path can remove the per-layer host
   router wait. The new kernel is deliberately narrow: Metal only, stage 5
   only, Qwen3.6 35B-A3B geometry (`hidden=2048`, `num_experts=256`,
   `moe_intermediate=512`, `shared_intermediate=512`, `top_k=8`), INT4
   `group_size=128`, complete sidecars, and the normal host fallback for all
   other cases or when `SUPERSONIC_METAL_FORCE_HOST_NATIVE=1`.
   One-token smokes compiled and generated the same ID as default (`[11]`),
   proving the wiring and shader pipeline are viable. The v4 four-token sweep
   keeps the mode disabled: default generated `[11, 271, 40, 599]` at
   `200.6 ms/token` with `ffn_ms_avg=106.849`, while
   `full-stage5-router` generated `[11, 353, 599, 264]` at
   `451.6 ms/token` with `ffn_ms_avg=353.337`. Its aggregate profile is still
   valuable: the stable op `qwen36_ffn_int4_stage5_with_router` reports
   `1512.266 ms` total, and command-buffer wait rises from `270.468 ms` to
   `1679.689 ms`. The next FFN attempt should keep this gate as evidence, but
   target a more substantial resident/fused INT4 compute design with fewer
   waited sub-dispatches and better reduction/tiling behavior before promotion
   is considered.

23. **Qwen3.6 aggregate FFN profile gate**
   The first `full-stage5-router` probe showed that the FFN profile path itself
   can dominate the gate when `SUPERSONIC_METAL_PROFILE=1` forces every FFN
   sub-phase into a separately waited command buffer. Qwen3.6 FFN Metal profile
   runs now keep the candidate stage aggregate by default, preserving the
   stable outer profile ops such as `qwen36_ffn_int4_stage5`,
   `qwen36_ffn_int4_stage5_with_router`, and
   `qwen36_batched_prefill_grouped_expert_direct` without converting the
   profile pass into a phase-split benchmark. The slower phase-attribution
   behavior remains available with
   `SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES=1` stacked on
   `SUPERSONIC_METAL_PROFILE=1`, and the fused-routed sweep exposes that as
   `--metal-profile-phases` for explicit attribution refreshes. The phase smoke
   confirmed those rows still work: `full-stage5-router` emitted GPU totals for
   `qwen36_ffn_int4_router_topk_stage5` (`12.470 ms`), expert gate/up
   (`6.275 ms`), expert down/finalize (`5.517 ms`), and the shared phases.
   This lets the promotion gate judge the fused FFN candidate with profiling
   enabled while still keeping per-phase GPU timestamps available for
   root-cause runs.

24. **Qwen3.6 FFN residency/submit-wait gap selector**
   The fused-routed INT4 sweep is now schema v5 and records separate native
   FFN wall time, command-buffer GPU time, wall/GPU ratio, wait/GPU ratio, and
   a stable `ffn_attribution_class` per candidate. The refreshed smoke sweep
   keeps the promotion gate negative, but removes the ambiguity in the next
   FFN target: every native candidate is classified as
   `residency_or_submit_wait`. The default row generated `[11, 271, 40, 599]`
   with `ffn_ms_avg=124.806`. The only candidate that preserved generated IDs
   was `gpu-pack`, but it regressed to `ffn_ms_avg=569.327` with
   `fused_wall_ms=2104.865`, `fused_gpu_ms=87.973`,
   `wall/GPU=23.93`, and `wait/GPU=28.10`. The other candidates also landed
   in the same gap class: `direct-gather` at `23.20` / `28.92`,
   `full-stage5` at `25.93` / `30.88`, and `full-stage5-router` at
   `18.44` / `21.28` for wall/GPU and wait/GPU respectively.
   The next-bottleneck selector is now schema v7 and keeps the broad action as
   `prototype_new_ffn_residency_or_compute_path`, but adds the measured
   `sub_action=prototype_ffn_residency_or_submit_wait_path`. The latest
   default-lane ranking remains FFN first (`118.882 ms` median), followed by
   linear attention (`69.149 ms`), full attention (`19.903 ms`), and lm-head
   (`8.453 ms`). So the next FFN implementation should not start as another
   arithmetic-only kernel fork; it should reduce waited command-buffer/native
   wall overhead for the resident stage-5 path, with correctness fixed before
   any promotion attempt.

25. **Qwen3.6 router FFN deferred-wait probe**
   The fused-routed INT4 sweep now includes `router-defer-wait`, guarded by
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER=1` plus
   `SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT=1`, to test whether
   simply letting the stage-5 router FFN command buffers run asynchronously can
   remove the dominant waited wall time. The native Metal profiling bridge also
   records `command_buffer_gpu:*` rows from completion handlers for async
   command buffers, so the deferred mode still has GPU timestamp attribution.
   The first four-token M5 Max smoke is a useful negative result. It reduced
   the measured FFN bucket from `full-stage5-router` `380.445 ms/token` to
   `3.612 ms/token`, but decode regressed from `481.6` to
   `559.7 ms/token` because the wait moved into the surrounding chain:
   `linear_attn_ms_avg` rose to `396.640`, `command_buffer_wait` rose to
   `2101.426 ms`, and `wait/GPU` stayed high at `21.79`. The deferred row also
   generated `[11, 353, 599, 264]` instead of the default
   `[11, 271, 40, 599]`, matching the existing router-FFN correctness gap.
   This rules out pure wait deferral as a promotion path. The next FFN attempt
   should either fix the router stage-5 arithmetic/parity gap first or reduce
   the real command-buffer/fence granularity with a larger encoded pipeline,
   rather than just shifting the same wait into linear attention or the next
   host read.

26. **Qwen3.6 direct-gather FFN parity fix**
   The raw expert direct-gather path now uses the same non-tiled
   down-finalize reduction as the earlier parity-preserving expert-tiled
   isolate, instead of the newer 256-thread tiled down-finalize leg. This keeps
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5=1` as the
   raw-resident expert candidate but removes the generated-token drift that
   made the mode hard to reason about. The first patched four-token smoke with
   Metal profiling generated the default IDs `[11, 271, 40, 599]`.
   The refreshed fused-routed sweep confirms that both `direct-gather` and
   `gpu-pack` now match default generated IDs, while `full-stage5`,
   `full-stage5-router`, and `router-defer-wait` still generate
   `[11, 353, 599, 264]`. Promotion remains negative because the fixed
   direct-gather path is still dominated by submit/wait cost:
   `default` ran at `214.2 ms/token` with `ffn_ms_avg=116.088`, while
   `direct-gather` ran at `434.4 ms/token` with `ffn_ms_avg=334.901`,
   `fused_wall_ms=1197.217`, `fused_gpu_ms=45.116`, `wall/GPU=26.54`, and
   `wait/GPU=32.66`. This is nevertheless the cleaner next base for FFN work:
   correctness no longer forces active expert packing, so the next
   implementation should target the real waited command-buffer/native-wall gap
   around the parity-preserving raw expert path, or fix the shared/router
   portions of `full-stage5*` before trying to promote those larger kernels.

27. **Qwen3.6 direct-gather FFN deferred-wait probe**
   The fused-routed INT4 sweep now includes `direct-defer-wait`, guarded by
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5=1` plus
   `SUPERSONIC_METAL_QWEN36_DEFER_FFN_DIRECT_GATHER_STAGE5_WAIT=1`. Unlike
   the router-deferred probe, this mode keeps the linear-attention wait in
   place because the host router/shared FFN setup still needs the attention
   output. It only skips the final direct-gather command-buffer wait and lets
   the runner synchronize before the next host full-attention read or final
   hidden download.
   The refreshed seven-mode smoke kept `direct-defer-wait` parity-clean:
   default, `direct-gather`, `direct-defer-wait`, and `gpu-pack` all generated
   `[11, 271, 40, 599]`; `full-stage5`, `full-stage5-router`, and
   `router-defer-wait` still generated `[11, 353, 599, 264]`. The result is a
   useful negative attribution point rather than a promotion candidate.
   `direct-defer-wait` collapsed the measured direct-gather fused wall from
   `692.446 ms` to `3.908 ms` while keeping comparable GPU time
   (`33.621 ms`), and the FFN bucket fell to `34.389 ms/token` versus the
   default `140.484`. But the wait moved into the surrounding chain:
   headline decode regressed to `384.9 ms/token` from default `245.1`,
   `linear_attn_ms_avg` rose to `243.713`, `command_buffer_wait` rose to
   `1294.473 ms`, and `wait/GPU` worsened to `38.50`. The SOTA selector still
   chooses FFN as the dominant default-lane bucket (`120.243 ms` median) and
   keeps `sub_action=prototype_ffn_residency_or_submit_wait_path`, but pure
   wait deferral is now ruled out for both the full-router and parity-clean
   direct-gather paths. The next attempt should reduce the actual number or
   granularity of waited Metal command buffers, or fix the larger
   `full-stage5*` parity issue so shared/router/expert work can be encoded as
   a single larger GPU pipeline.

28. **Qwen3.6 full-stage5 shared parity probe**
   The no-router `full-stage5` path is now a parity-clean diagnostic by using
   the tiled expert gate/up path plus host-order shared INT4 dot products and
   the original non-tiled expert down/finalize. This is not a promoted
   performance path: the shared kernels intentionally preserve host accumulation
   order to isolate correctness before reintroducing faster reductions.
   The refreshed seven-mode smoke shows the new boundary clearly. Default
   generated `[11, 271, 40, 599]` at `203.3 ms/token` with
   `ffn_ms_avg=107.419`. `direct-gather`, `direct-defer-wait`, `gpu-pack`, and
   the repaired `full-stage5` all preserved those IDs, while
   `full-stage5-router` and `router-defer-wait` still generated
   `[11, 353, 599, 264]`. The router mismatch persisted even after switching
   router down/finalize to the non-tiled finalize path, so the remaining
   correctness gap is in the router/RMSNorm/top-k side of
   `full-stage5-router`, not expert down finalize.
   The promotion gate remains negative. The parity-clean `full-stage5` row
   measured `401.5 ms/token`, `ffn_ms_avg=312.345`,
   `fused_wall_ms=1202.172`, `fused_gpu_ms=89.615`,
   `wall/GPU=13.41`, `command_buffer_wait=1426.349 ms`, and
   `wait/GPU=15.92`. A separate phase-attribution smoke now sums phase GPU
   rows correctly and kept `full-stage5` parity-clean, but still classified it
   as residency/submit-wait bound: `fused_gpu_ms=101.339`,
   `command_buffer_wait=1350.109 ms`, `wait/GPU=13.32`. The largest phase GPU
   rows were shared gate/up (`33.717 ms`), shared scalar (`22.224 ms`), expert
   gate/up (`19.618 ms`), expert down/finalize (`13.420 ms`), and shared down
   (`12.360 ms`). The next FFN attempt should keep no-router `full-stage5` as
   the correctness baseline, then either make the shared host-order math fast
   enough without token drift or fix the router stage so the whole FFN can be
   studied as one parity-clean larger pipeline.

29. **Qwen3.6 full-stage5 router parity repair**
   The router/RMSNorm/top-k side of `full-stage5-router` is now isolated and
   repaired. A narrow diagnostic tap,
   `SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP=1`, computes the host
   router reference beside the native Metal stage-5 router path and emits
   `[qwen36-ffn-router-parity]` rows for normalized hidden, router logits,
   top-k indices, and top-k weights. The sweep harness exposes this as
   `tests/metal/sweep_qwen36_fused_routed_int4.py --router-parity-tap` and
   records the rows in JSON/Markdown reports.

   The first tap proved the old four-token drift was not caused by top-k
   selection: all 160 tapped layer calls matched indices, but the Metal router
   RMSNorm reduction accumulated in a different order and allowed enough
   normalized-hidden/logit drift to perturb later tokens. The Metal router
   kernel now computes the stage-5 normalized hidden vector in host order before
   the router dot products. The refreshed four-token tap preserved the default
   IDs `[11, 271, 40, 599]`, reported zero top-k/workspace/output-index
   mismatches across 160 tapped calls, and bounded the remaining differences at
   `h_norm_max_abs=0.001953125`, `logits_max_abs=0.0`, and
   `topk_weight_max_abs=0.0009765625`.

   The refreshed seven-mode fused-routed sweep is now correctness-clean across
   every tracked mode: `default`, `direct-gather`, `direct-defer-wait`,
   `gpu-pack`, `full-stage5`, `full-stage5-router`, and `router-defer-wait`
   all generated `[11, 271, 40, 599]`. Promotion remains correctly negative.
   The default lane measured `209.8 ms/token` with `ffn_ms_avg=114.659`, while
   `full-stage5-router` measured `453.2 ms/token` with
   `ffn_ms_avg=361.197`, `fused_wall_ms=1444.370`,
   `fused_gpu_ms=181.980`, `wall/GPU=7.94`, `command_buffer_wait=1687.037 ms`,
   and `wait/GPU=9.27`. `router-defer-wait` collapsed the visible FFN bucket
   to `3.258 ms` but still regressed headline decode to `389.8 ms/token` and
   kept `command_buffer_wait=1435.791 ms`, confirming that pure wait deferral
   only moves the fence cost.

   The selector still chooses
   `prototype_new_ffn_residency_or_compute_path` with
   `sub_action=prototype_ffn_residency_or_submit_wait_path`; the default-lane
   median remains FFN first (`113.809 ms`) ahead of linear attention
   (`67.570 ms`), full attention (`19.479 ms`), and lm-head (`8.470 ms`).
   The next FFN work can now start from parity-clean larger stage-5 pipelines.
   It should target command-buffer granularity/residency and native-wall
   collapse, not another top-k or scalar wait-deferral fix.

30. **Qwen3.6 decode-batch command-buffer probe**
   The fused-routed sweep now includes `full-stage5-router-batch`, guarded by
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER=1` plus
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH=1`. The runner opens an experimental
   Metal batch around Qwen3.6 chained decode only when per-layer capture,
   accurate stage timings, router parity taps, phase profiling, and expert
   prefetch are all off. Host fallback boundaries flush the batch before CPU
   reads shared buffers, and D2D residual copies use a Metal blit when a batch
   is active. The sweep reports this mode in a fast-profile lane without
   `--emit-stage-timings`, because per-stage syncs intentionally defeat the
   command-buffer batching experiment.

   The smoke run kept generated-token parity with default:
   `[11, 271, 40, 599]`. The batch guard did what it was meant to do at the
   submission layer: the profiled `full-stage5-router-batch` row reduced Metal
   profile calls from `7396` to `764`, command-buffer waits from `1012` to
   `52`, and emitted `command_buffer_gpu:qwen36_decode_batch` with `44` batched
   chunks. However, it is not a promotion path. Default measured
   `219.7 ms/token`, non-batched `full-stage5-router` measured
   `476.1 ms/token`, and the batch row measured `435.9 ms/token`. The batch
   row moved the attribution from submit overhead into GPU work:
   `command_buffer_gpu:qwen36_decode_batch=1492.302 ms`,
   `command_buffer_wait=1628.769 ms`, `wait/GPU=1.09`, and the native wrapper
   wall around `qwen36_ffn_int4_stage5_with_router` collapsed to
   `9.250 ms` only because it now records enqueue time inside the open batch.

   This rules out command-buffer count alone as the next fix. The next FFN
   target should use the batch mode as an attribution tool, then optimize the
   actual batched GPU work: full-attention host-boundary frequency, the
   batched FFN router/shared/expert kernels, and the linear stage-5 kernels
   that now coexist inside the large decode-batch command buffers.

31. **Qwen3.6 decode-batch phase attribution**
   The decode-batch probe now has a phase-attribution mode,
   `full-stage5-router-batch-phases`, guarded by
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES=1` in addition to the
   stage-5 router and decode-batch envs. This keeps the no-stage-timing fast
   profile lane, but flushes the open Metal batch after each native
   linear-attention stage and each native router-FFN stage with stable labels:
   `command_buffer_gpu:qwen36_decode_batch_linear_attn` and
   `command_buffer_gpu:qwen36_decode_batch_ffn`. The sweep JSON/Markdown now
   records `decode_batch_linear_gpu_ms` and `decode_batch_ffn_gpu_ms` so the
   coarse `qwen36_decode_batch` GPU blob can be split without enabling
   per-dispatch `--emit-stage-timings`.

   The same patch also makes the batch guard flush before any full-attention
   host read, including the opt-in `attn_step_stage5_metal_host_into` direct
   handoff path. That path is a direct residual handoff, not a fully native
   full-attention kernel; it still computes the full-attention staged fallback
   on the host and therefore must not read Metal-backed hidden while earlier
   batched GPU work is pending.

   The first phase-attribution smoke preserved default generated IDs
   `[11, 271, 40, 599]`. The coarse batch row measured `482.1 ms/token` with
   `command_buffer_gpu:qwen36_decode_batch=1632.789 ms`. The phase row measured
   `472.5 ms/token`, and split that batched GPU time into
   `decode_batch_ffn_gpu_ms=507.387` versus
   `decode_batch_linear_gpu_ms=30.590`. Splitting increased command-buffer
   waits to `288` calls and `1759.923 ms`, as expected for attribution mode,
   but it proved the next arithmetic target is the FFN side of the batched
   pipeline rather than linear stage-5. The next kernel PR should now focus on
   reducing `qwen36_decode_batch_ffn` GPU time, with the phase mode kept as the
   proof harness for each FFN sub-kernel change.

32. **Qwen3.6 batched FFN subphase attribution**
   The FFN proof harness now has a narrower batch-compatible subphase mode,
   `full-stage5-router-batch-ffn-phases`, guarded by
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES=1` stacked on
   decode batching, stage-5 router FFN, and
   `SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES=1`. The runner only allows
   FFN phase profiling inside a decode batch when this env is present, and the
   Metal native split-profile path flushes after each labeled FFN subdispatch
   so the profile records stable per-phase GPU rows. The fused-routed INT4
   sweep is now schema v8 and records JSON fields for router top-k, shared
   gate/up, shared scalar, shared down, expert gate/up, expert down/finalize,
   and the summed FFN-subphase GPU total.

   The first smoke comparison preserved default generated IDs
   `[11, 271, 40, 599]` across `default`,
   `full-stage5-router-batch-phases`, and
   `full-stage5-router-batch-ffn-phases`. In that run, default measured
   `287.6 ms/token`. The coarse batch phase row measured `318.4 ms/token`,
   with `decode_batch_ffn_gpu_ms=391.358` and
   `decode_batch_linear_gpu_ms=32.729`. The FFN-subphase row measured
   `516.8 ms/token` because it intentionally flushes every FFN subdispatch;
   treat that row as attribution only, not as a promotion candidate.

   The subphase GPU sum was `223.285 ms`: router top-k `119.303 ms`,
   shared gate/up `32.349 ms`, shared scalar `22.823 ms`, expert gate/up
   `21.293 ms`, expert down/finalize `13.962 ms`, and shared down
   `13.555 ms`. Command-buffer wait rose to `1906.450 ms`
   (`wait/GPU=8.54`) in the split lane, confirming the lane is profiling
   overhead-heavy. The arithmetic signal is still clear: router top-k is about
   53% of the labeled FFN GPU work, so the next optimization should target
   `qwen36_ffn_int4_router_topk_stage5` before revisiting shared/expert matvec
   tiling.

33. **Qwen3.6 router-stage SIMD pilot**
   The router-stage optimization pilot is now wired behind
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD=1`. It keeps the
   existing serial RMSNorm order for parity, then splits router logits into a
   SIMD-per-expert dispatch before a second top-k-from-logits dispatch. The
   fused-routed INT4 sweep is schema v9 and adds
   `full-stage5-router-simd`, `full-stage5-router-simd-batch-phases`, and
   `full-stage5-router-simd-batch-ffn-phases` modes. The subphase parser now
   treats Metal GPU labels as exact labels, because
   `qwen36_ffn_int4_router_topk_stage5` is a prefix of the SIMD label
   `qwen36_ffn_int4_router_topk_stage5_simd`.

   This is not promoted. The full comparison smoke preserved IDs for the
   coarse SIMD batch row, but the FFN-subphase SIMD attribution row diverged:
   default and non-SIMD rows generated `[11, 271, 40, 599]`, while
   `full-stage5-router-simd-batch-ffn-phases` generated
   `[11, 353, 599, 264]`. In the same run, coarse SIMD also regressed the
   coarse FFN GPU bucket: non-SIMD
   `decode_batch_ffn_gpu_ms=393.556` versus SIMD `449.659`.

   The arithmetic signal is still interesting. In the FFN-subphase comparison,
   the SIMD router label reduced from `123.239 ms` to `99.278 ms`, and total
   labeled FFN GPU time moved from `230.097 ms` to `199.200 ms`; an isolated
   repeat of `default,full-stage5-router-simd-batch-ffn-phases` preserved
   generated IDs and measured router `113.686 ms` inside `228.659 ms` total
   FFN-subphase GPU time. Because the mixed comparison found an ID mismatch,
   the next router PR should add a router-logit/top-k parity tap for the SIMD
   path and chase deterministic route equality before using this path as a
   performance candidate.

34. **Qwen3.6 SIMD router parity evidence**
   The router parity tap now identifies the active router path and records the
   data needed to debug route-rank drift: `router_path`,
   `topk_first_mismatch`, workspace/output first-mismatch indices, host and
   Metal top-logit indices, top-logit values, and cross-read logits at the
   other path's top expert. The fused-routed sweep is schema v10 and summarizes
   tap counts, mismatch counts, observed paths, max hidden/logit/top-k-weight
   deltas, and mismatch examples in JSON. The Markdown renderer also samples
   every prompt/mode/path group so the SIMD rows are visible when a full 40
   layer tap is present.

   The full one-token Metal smoke used
   `--modes default,full-stage5-router,full-stage5-router-simd
   --metal-profile --router-parity-tap --router-parity-tap-max-calls 80`.
   It preserved generated IDs (`[11]`) across all modes and emitted 80 tap
   rows: 40 serial-router rows and 40 SIMD-router rows covering layers 0
   through 39. Both paths reported zero top-k mismatches, `topk_first_mismatch`
   stayed `-1`, host and Metal top-logit indices matched on every tapped
   layer, and the max deltas were all zero:
   `max_h_norm_abs=0.0`, `max_logits_abs=0.0`, and
   `max_topk_weight_abs=0.0`.

   This was a correctness/profiling tap, not a promotion measurement. The tap
   disables decode batching and adds per-layer synchronization/readback, so
   visible wall times are dominated by residency/submit waits. In that run,
   default measured `665.0 ms` decode with `ffn_ms_avg=239.519`.
   `full-stage5-router` measured `1581.0 ms` decode,
   `ffn_ms_avg=1397.756`, `fused_wall_ms=1377.424`,
   `fused_gpu_ms=97.045`, and `command_buffer_wait=1506.416 ms`.
   `full-stage5-router-simd` measured `1952.0 ms` decode,
   `ffn_ms_avg=1740.241`, `fused_wall_ms=1713.524`,
   `fused_gpu_ms=86.106`, and `command_buffer_wait=1866.179 ms`.

   The earlier SIMD batch-FFN phase drift is therefore not explained by the
   standalone SIMD router logit/top-k math. The next runtime step should move
   back to the decode-batch attribution lane and add a batch-compatible route
   checksum or route snapshot around
   `full-stage5-router-simd-batch-ffn-phases`. If that confirms route equality
   under batching, the measured bottleneck to attack is still the batched FFN
   router/shared/expert GPU work; if it finds route drift only inside the batch
   phase lane, fix that batch/profile interaction before promoting the SIMD
   router as a performance candidate.

35. **Qwen3.6 decode-batch route snapshot**
   The decode-batch attribution lane now has a batch-compatible route snapshot
   guard, `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT=1`. Instead of
   reading top-k routes after each layer, the runner copies each layer's
   `ffn_output_idx` into a per-decode Metal buffer while the batch is still
   open, then emits a single
   `[qwen36-decode-batch-route-snapshot]` row after the existing final-hidden
   synchronization. The fused-routed sweep is schema v11, exposes this with
   `--decode-batch-route-snapshot`, records raw route snapshots in JSON, and
   compares each snapshot-bearing mode against the serial batch reference by
   checksum and exact route string.

   The four-token Metal smoke used
   `--modes default,full-stage5-router-batch-ffn-phases,full-stage5-router-simd-batch-ffn-phases
   --metal-profile --decode-batch-route-snapshot`. It preserved generated IDs
   across all modes: `[11, 271, 40, 599]`. The snapshot report captured 8
   route rows total: 4 serial and 4 SIMD decode calls, each with all 40 layers
   captured. Every SIMD checksum matched the serial reference:
   `4123385977126816859`, `9921766482153130090`,
   `12710036671302543390`, and `2177400112220283478`; the route-summary
   mismatch count was zero.

   With route equality proven inside the batch FFN phase lane, the SIMD router
   path is correctness-clean for this smoke. The same run measured default at
   `959.0 ms` decode and `ffn_ms_avg=134.399`. The serial batch FFN phase row
   measured `2071.0 ms` decode, `decode_batch_ffn_gpu_ms=332.948`, and
   router top-k GPU `213.669 ms`. The SIMD batch FFN phase row measured
   `1497.0 ms` decode, `decode_batch_ffn_gpu_ms=224.532`, and router top-k GPU
   `132.618 ms`. That is a 32.6% reduction in labeled FFN GPU time and a 37.9%
   reduction in the router label in this attribution lane.

   This still is not a promotion lane: FFN phase profiling intentionally
   flushes around subdispatches, and native wrapper wall time remains far
   above GPU timestamps (`1358.920 ms` fused wall versus `224.532 ms` fused
   GPU for the SIMD row). The next measured target should keep SIMD router
   enabled, return to the coarse decode-batch lane without per-FFN subphase
   flushes, and decide whether the remaining blocker is batch/wait overhead or
   the still-large router/shared scalar GPU work.

36. **Qwen3.6 coarse decode-batch SIMD lane**
   The fused-routed sweep is now schema v12 and adds the missing phase-free
   SIMD batch mode, `full-stage5-router-simd-batch`, plus a
   `decode_batch_coarse` summary. The mode sets the same stage-5 router and
   decode-batch guards as `full-stage5-router-batch`, adds
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD=1`, and keeps
   `--emit-stage-timings` off so the measurement is one coarse
   `qwen36_decode_batch` profile lane rather than the per-FFN subphase
   attribution path. The summary compares the serial and SIMD coarse rows by
   generated IDs, decode wall time, exact `command_buffer_gpu:qwen36_decode_batch`
   time, command-buffer wait, and wait/GPU ratios.

   The Metal smoke used
   `--modes default,full-stage5-router-batch,full-stage5-router-simd-batch
   --metal-profile` with no route snapshot and no FFN phase flushing. It
   preserved generated IDs across all rows: `[11, 271, 40, 599]`. Default
   remained much faster at `849.0 ms` decode with `ffn_ms_avg=114.380`, so the
   decode-batch lane is still not a promotion candidate. Inside the coarse
   batch lane, SIMD helped but only modestly: serial batch measured
   `1482.0 ms` decode, `1263.000 ms` batch GPU, and `1360.363 ms`
   command-buffer wait; SIMD batch measured `1447.0 ms` decode,
   `1220.971 ms` batch GPU, and `1328.013 ms` command-buffer wait. That is a
   2.4% decode improvement, 3.3% batch-GPU reduction, and 2.4% wait reduction
   versus the serial coarse batch row.

   The blocker classification is now sharper: the subphase flushes were not
   hiding a SIMD correctness problem, and the phase-free coarse batch row does
   benefit from SIMD. However, the large `qwen36_decode_batch` GPU blob still
   dominates and command-buffer wait tracks it closely (`wait/GPU ~= 1.10`),
   rather than exploding into the earlier per-phase wall/GPU gap. The next
   implementation should keep the SIMD router enabled for batch probes, but
   keep the batch path gated. The next optimization target is the internal
   work inside the coarse `qwen36_decode_batch` blob, starting with a
   non-flushing attribution split or a narrower fused FFN arithmetic reduction;
   batch mode should not be promoted over the current default chained decode
   path until its headline decode time beats default.

37. **Qwen3.6 deferred decode-batch phase attribution**
   The decode-batch attribution lane now has non-waiting phase labels,
   `full-stage5-router-batch-deferred-phases` and
   `full-stage5-router-simd-batch-deferred-phases`, guarded by
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED=1`.
   Instead of flushing after a phase, the Metal batch helper closes and commits
   the current command buffer with a stable label, starts a fresh command
   buffer for later work, and waits for all pending buffers only at the next
   host boundary or final batch close. This keeps Metal queue ordering intact
   while letting the profile report `qwen36_decode_batch_linear_attn` and
   `qwen36_decode_batch_ffn` without inserting a per-phase fence. The sweep is
   schema v13 and summarizes these rows in
   `decode_batch_deferred_phase`.

   The four-token Metal smoke used
   `--modes default,full-stage5-router-batch,full-stage5-router-simd-batch,full-stage5-router-batch-deferred-phases,full-stage5-router-simd-batch-deferred-phases
   --metal-profile`. All rows generated the same IDs:
   `[11, 271, 40, 599]`. The coarse batch comparison was noisy and did not
   promote SIMD in this run: serial batch measured `1692.0 ms` decode and
   `1424.394 ms` batch GPU, while SIMD batch measured `1805.0 ms` decode and
   `1548.916 ms` batch GPU. That keeps the coarse batch lane gated.

   The deferred phase labels still provide the needed bottleneck split. The
   serial deferred row measured `linear=39.053 ms`, `ffn=615.180 ms`,
   `phase_total=654.233 ms`, and `ffn_share=0.940`. The SIMD deferred row
   measured `linear=26.962 ms`, `ffn=383.723 ms`,
   `phase_total=410.685 ms`, and `ffn_share=0.934`. Command-buffer wait still
   exceeded labeled GPU work (`wait/GPU=2.75` serial and `2.84` SIMD), so this
   is an attribution lane rather than a promotion path. The important result is
   that, once phase labels avoid per-phase waits, FFN still dominates the
   labeled native batch work by a wide margin. The next implementation target
   should be batched FFN arithmetic, starting with the routed expert and shared
   expert work inside `qwen36_decode_batch_ffn`, not another linear-attention
   split.

38. **Qwen3.6 shared-expert tiled SIMD rejection**
   The first batched-FFN arithmetic probe targeted the scalar shared-expert
   kernels because `shared_gate_up`, `shared_scalar`, and `shared_down` were
   still mostly one-lane dot products while routed expert gate/up already used
   256-thread reductions. The Metal path now has gated shared-expert probes:
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_TILED`,
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_SIMD`,
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_TILED`, plus the combined
   `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED`. The fused-routed sweep is
   schema v14 and exposes these through
   `full-stage5-router-simd-batch-shared-gate-up-tiled`,
   `full-stage5-router-simd-batch-shared-scalar-simd`,
   `full-stage5-router-simd-batch-shared-down-tiled`, and
   `full-stage5-router-simd-batch-shared-tiled`.

   The combined FFN-subphase attribution row proved the arithmetic idea works
   locally but is not correctness-clean: shared gate/up dropped to `4.227 ms`,
   shared scalar to `1.234 ms`, and shared down to `5.626 ms`, but generated
   IDs changed from `[11, 271, 40, 599]` to `[11, 353, 599, 264]`. The
   component sweep isolated the drift: gate/up tiled and scalar SIMD each
   produced `[11, 353, 599, 264]`; shared-down tiled preserved
   `[11, 271, 40, 599]` but regressed coarse batch GPU from `1351.984 ms` to
   `1948.279 ms`. The combined tiled row also drifted and measured
   `1571.437 ms` batch GPU, worse than the SIMD batch baseline in that run.

   This path should stay gated as a negative/probe result. The useful lesson is
   that naive parallel reductions change enough shared-expert numerical order
   to alter decode, and the one parity-clean shared component is slower at
   coarse batch scope. The next FFN arithmetic step should not be a direct
   tiled rewrite of shared reductions. Either build a route/output parity tap
   for shared-expert reductions before changing accumulation order, or move to
   a larger routed-expert arithmetic target where the existing tiled reductions
   are already accepted and the profile still has significant GPU work.

39. **Qwen3.6 shared-expert parity tap**
   The shared-expert parity tap now mirrors the router tap shape with
   `SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP=1` and
   `SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS`. It emits
   `[qwen36-ffn-shared-parity]` rows for `shared_gate`, `shared_up`,
   `shared_mid`, `shared_scalar`, and `shared_out`; the v15 fused-routed sweep
   parses those rows into `shared_parity` JSON and Markdown. The Metal native
   shared gate/up kernels now also publish gate and up intermediates into the
   already-reserved workspace slots, so the tap is reading real Metal state
   rather than unwritten scratch.

   A non-batched control run,
   `--modes default,full-stage5-router,full-stage5-router-simd
   --shared-parity-tap --shared-parity-tap-max-calls 80`, passed with
   `rows=3`, `ok=3`, and matching generated IDs. Its shared tap was clean:
   `max_shared_gate_abs=0`, `max_shared_up_abs=0`,
   `max_shared_mid_abs=7.15e-7`, `max_shared_scalar_abs=2.98e-8`, and
   `max_shared_out_abs=6.10e-5`. That validates the tap and the default
   non-batched Metal shared path against the host-order reference.

   The batched shared probe,
   `--modes default,full-stage5-router-simd-batch,
   full-stage5-router-simd-batch-shared-gate-up-tiled,
   full-stage5-router-simd-batch-shared-scalar-simd,
   full-stage5-router-simd-batch-shared-tiled --shared-parity-tap`, also
   passed `rows=5`, `ok=5`, and one-token ID matching, but the shared tap
   reported large deltas: `max_shared_gate_abs=15.61`,
   `max_shared_up_abs=17.32`, `max_shared_mid_abs=126.21`,
   `max_shared_scalar_abs=0.566`, and `max_shared_out_abs=6.828`. Router tap
   rows stayed disabled in that run, confirming the shared and router tap gates
   are independent.

   Because the same tap is clean in non-batched stage-5 and noisy in batched
   decode, the next correctness step is not another shared reduction rewrite.
   The next implementation target should be a batch-native shared parity tap
   inside the decode-batch FFN path, using the exact token/layer hidden row and
   workspace buffers that `qwen36_decode_batch_ffn` consumes. Only after that
   tap can distinguish true shared arithmetic drift from batch tap/reference
   sequencing should the shared tiled probes be reconsidered for promotion.

40. **Qwen3.6 decode-batch shared parity tap**
   The decode-batch lane now has the batch-native shared-expert tap called for
   in section 39. When `--shared-parity-tap` is used, the sweep sets both the
   original non-batch env and
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP=1`. During a
   Metal decode batch, the runtime suppresses the old live-buffer shared tap,
   enqueues D2D snapshots of the exact FFN input row and FFN workspace after
   each native stage-5 router call, and emits
   `[qwen36-decode-batch-shared-parity]` after the batch flush. The fused
   routed sweep schema is now `v16` and records those rows under
   `decode_batch_shared_parity`.

   The one-token smoke,
   `--modes default,full-stage5-router-simd-batch-shared-tiled
   --max-new-tokens 1 --shared-parity-tap --shared-parity-tap-max-calls 80`,
   passed with `rows=2`, `ok=2`, and matching generated IDs. The old
   non-batch `shared_parity` summary stayed empty in the batch row, while the
   new decode-batch tap captured `40` rows on `all_tiled` with
   `max_shared_gate_abs=8.58e-6`, `max_shared_up_abs=7.63e-6`,
   `max_shared_mid_abs=4.96e-5`, `max_shared_scalar_abs=2.98e-7`, and
   `max_shared_out_abs=2.44e-4`.

   The four-token repeat,
   `--max-new-tokens 4 --shared-parity-tap --shared-parity-tap-max-calls 160`,
   reproduced the previous correctness failure:
   default generated `[11,271,40,599]`, while
   `full-stage5-router-simd-batch-shared-tiled` generated
   `[11,353,599,264]`. However, the new batch-native shared tap remained
   tight over `160` captured layer/token rows:
   `max_shared_gate_abs=1.57e-4`, `max_shared_up_abs=1.59e-4`,
   `max_shared_mid_abs=7.23e-5`, `max_shared_scalar_abs=9.02e-7`, and
   `max_shared_out_abs=2.44e-4`.

   This changes the next target. The shared tiled path is still not a
   promotion candidate because the four-token IDs diverge and the profiled row
   remains slower (`decode_ms=1630`, `qwen36_decode_batch=1146.375 ms`,
   `command_buffer_wait=1201.580 ms`). But the batch-native evidence no longer
   points at shared-expert arithmetic as the primary correctness gap. The next
   probe should move downstream: capture per-token final hidden and/or lm-head
   logits for decode-batch versus the default chained path, then bisect whether
   the divergence appears before FFN finalize, in residual accumulation, or at
   lm-head/sampling.

41. **Qwen3.6 downstream hidden/logits parity tap**
   The fused-routed sweep is now schema `v17` and adds
   `--downstream-parity-tap`, which sets
   `SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP=1`. The runner emits
   `[qwen36-final-hidden-tap]` after final hidden materialization and
   `[qwen36-logits-tap]` after lm-head logits are available. Both records carry
   `step`, `gen_index`, `position`, `path`, `lm_head_folded`, element count, and
   a stable checksum; the final-hidden row also reports norm/max/head values,
   while the logits row reports top-1 and top-5. The sweep compares candidate
   taps against the default row per prompt and generation index, then renders
   JSON/Markdown summaries for checksum and logits top-1 drift.

   The four-token Metal run used
   `--modes default,full-stage5-router-simd-batch-shared-tiled
   --max-new-tokens 4 --metal-profile --shared-parity-tap
   --shared-parity-tap-max-calls 160 --downstream-parity-tap`. It reproduced
   the known generated-ID divergence: default generated `[11,271,40,599]`,
   while the decode-batch shared-tiled row generated `[11,353,599,264]`.
   The batch-native shared parity tap stayed tight over `160` rows
   (`max_shared_out_abs=2.44e-4`), so this run again does not implicate shared
   expert arithmetic as the primary drift source.

   The new downstream tap moved the failure boundary. Final-hidden checksums
   mismatched for all four generated tokens, starting immediately at
   `gen_index=0` (`d30ef3d258b59586` default versus `ec6f0d09ab0f9b73`
   decode-batch). Logits checksums also mismatched from `gen_index=0`, but
   top-1 still matched for the first sample (`11` versus `11`); top-1 diverged
   at `gen_index=1` (`271` default versus `353` decode-batch) and stayed
   divergent for the remaining samples (`40` versus `599`, then `599` versus
   `264`). The profiled row remains a correctness probe, not a promotion
   candidate: default measured `1059.0 ms` decode, while the decode-batch
   shared-tiled row measured `1945.0 ms` decode with
   `qwen36_decode_batch=1417.532 ms` and `command_buffer_wait=1506.119 ms`.

   This rules out lm-head/sampling as the first place to look. The logits
   decision changes only after the hidden vector has already diverged, and the
   first-token top-1 match shows that checksum drift can be masked by the
   margin. The next correctness probe should bisect the final-hidden producer:
   capture per-token layer-output signatures around decode-batch finalize,
   residual add, and final RMSNorm for the default chained path versus
   `qwen36_decode_batch`. If layer outputs are clean until the last layer, focus
   on final RMSNorm/residual materialization; if they diverge earlier, bisect
   within the decode-batch FFN finalize and residual accumulation for the first
   mismatching layer.

42. **Qwen3.6 layer-output parity tap**
   The fused-routed sweep is now schema `v18` and adds `--layer-output-tap`,
   which sets `SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP=1`. The decode runtime
   snapshots each layer's post-attention residual and post-FFN residual into a
   compact Metal buffer while the batch is still open, then emits
   `[qwen36-layer-output-tap]` rows after the batch flush. Each row carries
   `position`, `cache_pos`, `path`, phase-profile state, `layer`, `phase`
   (`attn` or `ffn`), element count, checksum, norm/max, and head values. The
   sweep compares candidate rows against the default row by prompt, position,
   layer, and phase.

   The four-token Metal run used
   `--modes default,full-stage5-router-simd-batch-shared-tiled
   --max-new-tokens 4 --metal-profile --shared-parity-tap
   --shared-parity-tap-max-calls 160 --downstream-parity-tap
   --layer-output-tap`. It reproduced the generated-ID failure:
   default generated `[11,271,40,599]`, while the decode-batch shared-tiled
   row generated `[11,353,599,264]`. The shared parity tap again stayed tight
   over `160` rows with `max_shared_out_abs=2.44e-4`, and final-hidden/logits
   drift matched section 41.

   The layer-output tap found the first hidden-stream divergence at the start
   of the chain, not at final RMSNorm. At position `0`, layer `0` post-attn
   matched exactly (`c3790558cfcd8fe4` in both rows), but layer `0` post-FFN
   diverged (`f1872ddeb26a59b0` default versus `68a1cc5202bd41e4`
   decode-batch). The run captured `640` layer-output rows and compared `320`
   candidate rows; `317` mismatched. The only matching comparisons were
   position `0` layer `0` post-attn, plus position `1` layer `0` post-attn and
   post-FFN. From position `2` onward every tapped layer phase differed, which
   is expected once earlier generated-token history has diverged.

   This narrows the next correctness target to the first layer's FFN finalize
   path. Since layer `0` post-attn is clean and the batch-native shared-expert
   tap is still within BF16-scale tolerance, the next probe should capture the
   routed expert side of layer `0`: router logits/top-k for decode-batch,
   routed gate/up, routed down output, combined routed output, pre-residual
   shared+routed sum, and the final residual add. If routed expert output is
   clean, focus on residual accumulation/materialization. If routed output is
   already different, disable or parity-tap the batched expert gate/up and
   down/combine kernels before attempting another performance promotion.

43. **Qwen3.6 decode-batch routed parity tap**
   The fused-routed sweep is now schema `v19` and adds `--routed-parity-tap`,
   which sets
   `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP=1`. The
   decode-batch runtime reuses the batch-native FFN snapshots from section 40
   and, when the routed tap is enabled, also snapshots the stage-5 FFN output
   row plus `output_idx`. After the batch flush it emits
   `[qwen36-decode-batch-routed-parity]` rows comparing the exact Metal
   workspace against a host reference for top-k indices/weights, routed
   `expert_mid`, combined `moe_out`, and final residual output.

   The four-token Metal run used
   `--modes default,full-stage5-router-simd-batch-shared-tiled
   --max-new-tokens 4 --metal-profile --shared-parity-tap
   --shared-parity-tap-max-calls 160 --downstream-parity-tap
   --layer-output-tap --routed-parity-tap
   --routed-parity-tap-max-calls 160`. It again reproduced the generated-ID
   failure: default generated `[11,271,40,599]`, while the decode-batch
   shared-tiled row generated `[11,353,599,264]`. The first layer-output
   mismatch stayed at position `0`, layer `0`, phase `ffn`; final-hidden
   checksums diverged from `gen_index=0`, and logits top-1 diverged at
   `gen_index=1`.

   The new routed tap captured `160` rows and found no top-k index mismatch.
   Routed arithmetic was tight: `max_topk_weight_abs=9.77e-4`,
   `max_expert_mid_abs=1.39e-4`, `max_moe_out_abs=4.88e-4`, and
   `max_final_out_abs=9.77e-4`. The shared tap in the same run also stayed in
   the same BF16-scale band (`max_shared_out_abs=2.44e-4`). This rules out a
   gross routed-expert indexing or down/combine bug as the source of the layer
   `0` FFN checksum mismatch.

   The remaining correctness gap looks like deterministic BF16-scale output
   drift that checksum-level taps classify as a mismatch and that later
   generation steps amplify into token divergence. Before any more FFN
   performance promotion work, the next diagnostic should add a numeric
   default-versus-decode-batch layer-output delta tap for the first
   mismatching FFN row, then decide whether the policy can accept a small
   tolerance or whether the native path must be made bit-exact with the
   default chained FFN rounding order.

44. **Qwen3.6 layer-output numeric delta tap**
   The fused-routed sweep is now schema `v20` and adds
   `--layer-output-delta-tap`, which sets
   `SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP=1`. The runtime reuses the
   layer-output tap buffer and, for a filtered position/layer/phase only,
   emits `[qwen36-layer-output-delta-tap]` rows with the full BF16 row as
   comma-separated hex words. The sweep compares the default chained row
   against the candidate decode-batch row and reports element count, checksum
   agreement, max absolute delta, max BF16-ordered ULP delta, and the first
   mismatching element details. The extra full-row payload is intentionally
   filtered by `SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION`,
   `SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER`, and
   `SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE` so attribution runs do not
   dump every layer output.

   The four-token Metal run used
   `--modes default,full-stage5-router-simd-batch-shared-tiled
   --max-new-tokens 4 --metal-profile --shared-parity-tap
   --shared-parity-tap-max-calls 160 --downstream-parity-tap
   --layer-output-tap --layer-output-delta-tap
   --layer-output-delta-position 0 --layer-output-delta-layer 0
   --layer-output-delta-phase ffn --routed-parity-tap
   --routed-parity-tap-max-calls 160`. It reproduced the existing
   generated-ID divergence: default generated `[11,271,40,599]`, while the
   decode-batch shared-tiled row generated `[11,353,599,264]`. The first
   checksum mismatch stayed at position `0`, layer `0`, phase `ffn`
   (`f1872ddeb26a59b0` default versus `68a1cc5202bd41e4` decode-batch).

   The numeric delta tap emitted the two filtered rows and one comparison. The
   layer `0` FFN row has `2048` BF16 elements, only `2` differing elements,
   `max_abs_delta=1.220703125e-4`, and `max_ulp_delta=1`. The largest
   difference was at index `225`: default BF16 `3c93`
   (`0.0179443359375`) versus decode-batch BF16 `3c92`
   (`0.017822265625`). Shared and routed attribution stayed tight in the same
   run: shared captured `160` rows with `max_shared_out_abs=2.44e-4`, and
   routed captured `160` rows with no top-k mismatch plus
   `max_topk_weight_abs=9.77e-4`, `max_expert_mid_abs=1.39e-4`,
   `max_moe_out_abs=4.88e-4`, and `max_final_out_abs=9.77e-4`.

   This turns the first-layer correctness issue from an indexing suspicion
   into a deterministic materialization/rounding target. The batch-native FFN
   path is numerically very close, but not bit-exact with the default chained
   path, and the one-ULP BF16 drift is enough to amplify into generated-token
   divergence by the second sampled token. Promotion should stay blocked until
   stage-5 decode-batch FFN finalize/residual materialization matches the
   default path bit-for-bit, or until the acceptance policy explicitly moves
   from generated-ID parity to an eval/tolerance gate. For this lane, the next
   correctness target is bit-exact rounding/materialization, not another
   performance variant.

45. **Qwen3.6 layer-output promotion gate**
   The fused-routed sweep is now schema `v21`. The promotion gate now treats
   `--layer-output-tap` as a correctness contract: if a default row and a
   candidate row both include layer-output tap rows, every tapped
   position/layer/phase checksum must match before the candidate can pass. A
   generated-ID match is no longer sufficient for promotion when the hidden
   stream was sampled.

   The two-token component sweep used `--modes
   default,full-stage5-router-simd-batch,
   full-stage5-router-simd-batch-shared-gate-up-tiled,
   full-stage5-router-simd-batch-shared-scalar-simd,
   full-stage5-router-simd-batch-shared-down-tiled,
   full-stage5-router-simd-batch-shared-tiled --max-new-tokens 2
   --metal-profile --shared-parity-tap --shared-parity-tap-max-calls 80
   --downstream-parity-tap --layer-output-tap --layer-output-delta-tap
   --layer-output-delta-position 0 --layer-output-delta-layer 0
   --layer-output-delta-phase ffn`. The plain
   `full-stage5-router-simd-batch` row generated `[11,271]` like default and
   matched the filtered layer `0` FFN row exactly, but the broader
   layer-output tap still found its first hidden checksum mismatch later at
   position `0`, layer `7`, phase `ffn`. That means sampled generated tokens
   can hide hidden-stream drift over a short run.

   The shared component rows all diverged by the second sampled token and
   generated `[11,353]`. In the filtered numeric delta row,
   `full-stage5-router-simd-batch-shared-gate-up-tiled` was the first
   mismatching component: one BF16 element differed at index `225`, default
   `3c93` (`0.0179443359375`) versus candidate `3c92`
   (`0.017822265625`), with `max_abs_delta=1.220703125e-4` and
   `max_ulp_delta=1`. The full shared-tiled row had two differing BF16
   elements with the same one-ULP maximum.

   The immediate policy result is that shared-tiled remains a negative/probe
   path until the materialization is bit-exact, or until a future eval-based
   tolerance gate is explicitly chosen. The next correctness target should be
   stage-5 decode-batch hidden-stream parity under the layer-output tap, with
   performance promotion gated on those checksums before headline speed.

46. **Qwen3.6 FFN residual delta attribution**
   The fused-routed sweep is now schema `v22` and adds a derived
   `ffn_residual_delta_attribution` summary. When
   `--layer-output-delta-tap` is combined with decode-batch shared and routed
   parity taps, the reporter now joins the first FFN row delta with the
   matching shared/routed parity rows for the same prompt, position, and
   layer. The JSON and Markdown rows name the delta index, shared-out argmax,
   MoE-out argmax, final-out argmax, top-k parity, and a coarse source label
   such as `shared_out_residual_rounding_boundary`.

   The verification run used `--modes default,full-stage5-router-simd-batch
   --max-new-tokens 2 --metal-profile --shared-parity-tap
   --shared-parity-tap-max-calls 80 --routed-parity-tap
   --routed-parity-tap-max-calls 80 --downstream-parity-tap
   --layer-output-tap --layer-output-delta-tap
   --layer-output-delta-position 0 --layer-output-delta-layer 7
   --layer-output-delta-phase ffn`. Generated IDs still matched
   (`[11,271]`), while promotion stayed blocked by the tapped hidden stream.
   The first layer-output mismatch remained position `0`, layer `7`, phase
   `ffn`, and the filtered numeric row had a single BF16 element different:
   index `1621`, default `-0.09423828125`, decode-batch `-0.0947265625`,
   `max_abs_delta=0.00048828125`, `max_ulp_delta=1`.

   The new attribution row connected that same index to shared and final
   residual parity: top-k indices and top-k weights matched exactly,
   `shared_out_argmax=1621` with host `0.0123901367` versus Metal
   `0.0123291016`, while `final_out_argmax=1621` reproduced the one-ULP final
   hidden delta. The MoE max was elsewhere (`moe_out_argmax=1107`,
   `max_moe_out_abs=1.52587891e-5`), so the current root-cause label is
   `shared_out_residual_rounding_boundary`.

   The next implementation target is therefore the native shared path's
   materialization math, especially `silu(shared_gate) * shared_up` feeding
   `shared_down`, and the final residual BF16 rounding boundary. Router SIMD
   and top-k are not the blocker for this measured row. A bit-exact fix should
   first try to make the host-order Metal shared path reproduce the chained
   CPU/shared result at index `1621`; if that proves too expensive or
   inherently cross-math-library sensitive, the alternative is an explicitly
   documented eval/tolerance promotion policy rather than silently relaxing the
   checksum gate.

47. **Qwen3.6 shared-mid drift tap**
   A precise-exp probe was tried in both Qwen3.6 Metal SiLU helpers and did not
   move the measured layer-7 FFN mismatch. The probe was reverted rather than
   keeping slower math on the hot path: the first mismatch stayed at position
   `0`, layer `7`, phase `ffn`, index `1621`, with the same one-BF16-step
   final hidden delta and the same shared-out residual attribution.

   The fused-routed sweep is now schema `v23` and extends both
   `[qwen36-ffn-shared-parity]` and
   `[qwen36-decode-batch-shared-parity]` with host/Metal values at the
   `shared_mid_argmax`: gate, up, and mid. The Markdown shared parity rows now
   surface those values, and the residual attribution row reports the derived
   gate-at-mid, up-at-mid, and mid deltas.

   The refreshed Metal run used the same two-token layer-7 attribution command
   and wrote
   `/private/tmp/qwen36_shared_mid_tap_layer7_delta_2tok.{json,md}`. Generated
   IDs still matched (`[11,271]`) and promotion remained blocked by the layer
   checksum. For the first residual delta, `shared_out_argmax=1621` still
   matched the layer-output delta index, with host shared-out `0.0123901367`
   versus Metal `0.0123291016` (`shared_out_delta_at_argmax=-6.10351e-5`).
   The new mid tap shows `shared_mid_argmax=13`: gate and up matched exactly at
   that index, while mid differed only by `-6.0e-8` (`0.336338878` versus
   `0.336338818`).

   That rules out `exp` precision as the primary fix for the observed layer-7
   row. The next useful correctness probe should target the shared-down
   accumulation and final shared-out BF16 materialization for output index
   `1621`, because a tiny shared-mid drift is being amplified or rounded into
   the shared-out residual boundary rather than showing as a large gate/up/mid
   mismatch.

48. **Qwen3.6 shared-down recompute probe**
   The fused-routed sweep is now schema `v25`. It extends the shared parity
   rows again with a host recompute of the shared-down row at
   `shared_out_argmax`, using the Metal-produced `shared_mid` and shared scalar
   as inputs. The residual attribution now records whether that host recompute
   matches the host shared-out or the Metal shared-out, and refines the source
   to `shared_mid_to_shared_out_bf16_boundary` when the recompute lands on
   Metal.

   The refreshed Metal run wrote
   `/private/tmp/qwen36_shared_down_probe_layer7_delta_2tok.{json,md}`. The
   first FFN mismatch is unchanged at position `0`, layer `7`, index `1621`,
   with one BF16 element differing. The recompute result is decisive:
   `host_shared_gated_at_out_argmax=0.012359621`, while recomputing from the
   Metal mid vector gives `0.0123596173`; that tiny `-3.7e-9` gated delta
   rounds to the Metal shared-out `0.0123291016` rather than the host shared-out
   `0.0123901367`.

   The remaining correctness issue is therefore not a large shared-down
   accumulation error. It is a BF16 cliff: exact-enough gate/up and a
   `-6.0e-8` shared-mid difference are sufficient to flip the final shared-out
   rounding bit at index `1621`. The next implementation choice should be
   explicit: either make shared-mid bit-exact with the CPU reference before
   shared-down, or promote this path under a documented tolerance/eval policy
   instead of a checksum gate.

49. **Qwen3.6 explicit layer-output tolerance gate**
   The fused-routed sweep is now schema `v26`. It adds
   `--layer-output-delta-all`, which leaves the runtime's layer-output delta
   filters unset so every tapped position/layer/phase row emits the full BF16
   payload for numeric comparison. It also adds an explicit opt-in promotion
   policy:
   `--promotion-allow-layer-output-tolerance`,
   `--promotion-layer-output-max-abs-delta`,
   `--promotion-layer-output-max-ulp-delta`, and
   `--promotion-layer-output-max-differing-elems`. Strict checksum parity
   remains the default. With tolerance enabled, a layer-output checksum
   mismatch is waived only when a matching numeric delta row exists and proves
   the mismatch is within all configured limits; missing or over-limit delta
   evidence still fails the nonfatal promotion gate.

   The validation run used `--modes default,full-stage5-router-simd-batch
   --max-new-tokens 1 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all
   --promotion-allow-layer-output-tolerance
   --promotion-layer-output-max-abs-delta 0.0005
   --promotion-layer-output-max-ulp-delta 1
   --promotion-layer-output-max-differing-elems 1
   --no-promotion-require-profile`, writing
   `/private/tmp/qwen36_layer_output_tolerance_v26_1tok.{json,md}`. The run
   emitted `160` delta rows and compared `80` candidate rows. Generated IDs
   matched for the one sampled token, but the tolerance gate still rejected the
   candidate: `65` layer-output checksums differed, only the first layer-7 FFN
   row was within the one-ULP/one-element policy, and `64` rows exceeded or
   propagated beyond that tolerance.

   The first tolerated row is the known BF16 cliff at position `0`, layer `7`,
   phase `ffn`, index `1621`: `differing_elems=1`,
   `max_abs_delta=0.00048828125`, and `max_ulp_delta=1`. The first
   untolerated row is immediately next, position `0`, layer `8`, phase `attn`:
   `differing_elems=125`, `max_abs_delta=0.0009765625`, and
   `max_ulp_delta=128`. Across the full one-token hidden stream,
   `max_abs_delta` reached `0.15625`, `max_ulp_delta` reached `31081`, and
   `max_differing_elems` reached `1991`.

   This makes the promotion policy explicit without weakening the default
   correctness gate. A hidden-row one-ULP tolerance can explain the first
   rounding cliff, but it cannot promote the current decode-batch path once the
   drift propagates through later layers. The next useful implementation choice
   is now sharper: either make shared-mid/output materialization bit-exact
   before layer `8`, or move to a separate eval/logit-level acceptance policy
   that is intentionally broader than hidden-stream parity.

50. **Qwen3.6 shared-output host-correction diagnostic**
   The fused-routed sweep is now schema `v27`. It adds a diagnostic-only mode,
   `full-stage5-router-simd-batch-shared-host-corrected`, controlled by
   `SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION=1`. The mode
   still runs the native decode-batch/router-SIMD FFN path, then flushes and
   patches only the shared FFN output from the host reference before
   recomputing the final residual BF16 row. It is explicitly rejected by the
   promotion gate through `diagnostic_mode_not_promotable`, because the
   per-layer flush/sync/readback cost is for attribution only.

   The one-token Metal smoke wrote
   `/private/tmp/qwen36_shared_host_corrected_v27_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch,full-stage5-router-simd-batch-shared-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all`. Generated IDs matched
   across all rows (`[11]`), but promotion stayed false. The uncorrected
   decode-batch/router-SIMD row had `65` layer-output checksum mismatches, first
   at layer `7` FFN with `max_abs_delta=0.00048828125`,
   `max_ulp_delta=1`, and one differing element. With the shared-output host
   correction enabled, mismatches dropped to `13`, and the first untolerated
   row moved to layer `33` FFN with `max_abs_delta=0.0001220703125`,
   `max_ulp_delta=2`, and one differing element.

   The correction log emitted `40` per-layer rows. Only one layer changed the
   final hidden output: layer `7`, where the patched shared output differed by
   `6.10351562e-5` at index `1621` and changed one BF16 output element by
   `0.00048828125`. This confirms that the layer-7 cliff is caused by shared
   output materialization/residual rounding, not by router top-k selection.
   It also proves that fixing that first cliff is not sufficient for full
   hidden-stream parity: a later layer-33 FFN source remains.

   The profile rows underline why this stays diagnostic. The normal
   decode-batch/router-SIMD profile in this tap-heavy run reported
   `command_buffer_gpu:qwen36_decode_batch` at `1211.089 ms` and
   `command_buffer_wait` at `1264.159 ms`, while the host-corrected row split
   work into `93` command-buffer waits with `120` HAL syncs. Its top GPU rows
   were `command_buffer_gpu:qwen36_ffn_int4_stage5_with_router`
   (`91.159 ms`) and
   `command_buffer_gpu:qwen36_linear_int4_out_proj_finalize` (`87.653 ms`).
   Those numbers are attribution overhead, not a performance candidate.

   The next measured correctness target is the late layer-33 FFN mismatch under
   shared correction. A good follow-up is a similarly narrow diagnostic that
   patches the routed/MoE contribution or final residual at layer `33`, so we
   can tell whether the remaining drift is routed expert math, MoE combine, or
   final residual ordering before changing the hot Metal kernels.

51. **Qwen3.6 routed-output host-correction diagnostic**
   The fused-routed sweep is now schema `v28`. It adds a second diagnostic-only
   mode, `full-stage5-router-simd-batch-shared-routed-host-corrected`, which
   enables both
   `SUPERSONIC_METAL_QWEN36_FFN_STAGE5_SHARED_HOST_CORRECTION=1` and
   `SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_HOST_CORRECTION=1`. The routed
   correction computes the host routed reference from the same router/shared
   reference path, compares Metal `expert_mid`, `moe_out`, and final output,
   patches only the routed `moe_out` plus final BF16 output, and emits
   `[qwen36-ffn-routed-host-correction]` rows. Like the shared correction, this
   mode is explicitly diagnostic and cannot pass promotion.

   The Metal smoke wrote
   `/private/tmp/qwen36_routed_host_corrected_v28_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch,full-stage5-router-simd-batch-shared-host-corrected,full-stage5-router-simd-batch-shared-routed-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all`. Generated IDs matched
   across all four rows (`[11]`). The uncorrected row again had `65`
   layer-output checksum mismatches starting at layer `7` FFN. Shared-only
   correction again reduced that to `13` mismatches, first at layer `33` FFN
   with `max_abs_delta=0.0001220703125`, `max_ulp_delta=2`, and one differing
   element. Shared+routed correction reduced layer-output mismatches to `0`.

   The routed correction emitted `40` rows and changed final output on exactly
   one layer: layer `33`. There, `expert_mid_max_abs=4.76837158e-6` at expert
   mid index `111`, while the routed `moe_out` BF16 boundary flipped by
   `0.000122070312` at hidden index `8`
   (`host_moe_out_at_argmax=-0.0219726562`, Metal
   `-0.0220947266`). The final output patch was the same magnitude at the same
   hidden index (`host_final_out_at_argmax=0.0126953125`, Metal
   `0.0125732422`) and changed one BF16 element. That makes the remaining
   post-shared-correction source a routed MoE output materialization cliff, not
   final residual ordering.

   The correction also demonstrates that the current decode-batch/router-SIMD
   hidden-stream drift is a chain of two BF16 cliffs: shared output at layer `7`
   and routed MoE output at layer `33`. The next hot-path correctness target is
   therefore the routed expert down/combine materialization. Before touching the
   optimized Metal path, the useful follow-up diagnostic is a row probe for
   layer `33`, hidden index `8`, that recomputes the expert down/combine from
   Metal-produced `expert_mid` and top-k weights. If that recompute lands on the
   Metal `moe_out`, the fix has to make `expert_mid` bit-exact; if it lands on
   the host `moe_out`, the fix is in the routed down/combine accumulation or
   BF16 materialization.

52. **Qwen3.6 routed down/combine row probe**
   The fused-routed sweep is now schema `v29`. The routed host-correction row
   now also recomputes the `moe_out_argmax` row three ways: host
   `expert_mid` plus host top-k, Metal `expert_mid` plus host top-k, and Metal
   `expert_mid` plus Metal top-k. The Markdown/JSON summary records whether
   the Metal-mid recompute lands on the host or Metal `moe_out`, plus the
   top-k index and weight deltas.

   The shortest Metal smoke wrote
   `/private/tmp/qwen36_routed_row_probe_v29_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch-shared-routed-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all`. Generated IDs matched
   (`[11]`), and the shared+routed diagnostic still produced zero layer-output
   mismatches. The row is diagnostic-only and still fails promotion only for
   `diagnostic_mode_not_promotable` plus the expected missing headline/stage
   timing fields.

   The layer `33`, hidden index `8` routed cliff is now localized one step
   earlier. Top-k indices matched and `topk_weight_max_abs=0.0`. The host
   down/combine recompute at the failing row was
   `host_routed_moe_out_recomputed_at_argmax=-0.0219726562`, matching the host
   `moe_out`. Recomputing with Metal `expert_mid` produced
   `metal_mid_host_topk_moe_out_at_argmax=-0.0220947266`, matching the Metal
   `moe_out`; using Metal top-k gave the same value. The accumulator values
   straddle a BF16 boundary:
   `host_routed_down_acc_at_moe_argmax=-0.0220336821` versus
   `metal_mid_host_topk_down_acc_at_moe_argmax=-0.022033738`.

   That rules out routed down/combine ordering and top-k weights as the direct
   source for this measured row. The next correctness target is the routed
   expert gate/up/SwiGLU materialization that creates `expert_mid` for layer
   `33`, expert-mid index `111`. A useful next probe should log host and Metal
   routed gate, up, SiLU, and mid values at that index, then decide whether the
   fix is precise SiLU/multiply behavior, BF16/F32 staging, or an explicit
   hidden-stream tolerance/eval policy.

53. **Qwen3.6 routed gate/up/SwiGLU probe**
   The fused-routed sweep is now schema `v30`. The routed host-correction
   diagnostic now has an env-gated Metal tap for routed expert gate/up: when
   `SUPERSONIC_METAL_QWEN36_FFN_STAGE5_ROUTED_HOST_CORRECTION=1`, the tiled
   expert gate/up shader writes the raw gate and up reductions into the
   existing `off_expert_gu` workspace before materializing `expert_mid`. Normal
   runs keep the sentinel offset and do not pay for the diagnostic writes. The
   JSON/Markdown summary now records host and Metal gate, up, SiLU, and
   recomputed mid values at the `expert_mid_argmax`.

   The shortest Metal smoke wrote
   `/private/tmp/qwen36_routed_gate_up_probe_v30_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch-shared-routed-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all`. Generated IDs matched
   (`[11]`), and the shared+routed diagnostic still produced zero layer-output
   mismatches. The routed summary had `40` correction rows, `1` changed output
   row, `topk_weight_max_abs=0.0`, `max_expert_gate_abs=5.7220459e-6`,
   `max_expert_up_abs=3.33786011e-6`,
   `max_expert_silu_abs=5.7220459e-6`, and
   `max_expert_mid_recompute_abs=2.67028809e-5`.

   The layer `33` cliff is now directly attributable to routed gate/up/SwiGLU
   materialization. At expert-mid index `111` (`group=0`, `row=111`), host
   gate/up/SwiGLU were `2.38652968`, `-2.25330496`, and `2.18557024`; Metal
   wrote `2.38653088`, `-2.25330591`, and `2.18557143`. Those small deltas
   exactly recreate the observed mid difference:
   `host_expert_mid_recomputed_at_argmax=-4.92475605` versus
   `metal_expert_mid_recomputed_at_argmax=-4.92476082`
   (`expert_mid_recompute_delta_at_argmax=4.76837158e-6`). Feeding that Metal
   mid into the already-probed down/combine path reproduces the Metal BF16 MoE
   output (`-0.0220947266`) rather than the host BF16 output
   (`-0.0219726562`) at hidden index `8`.

   This rules out routed down/combine, top-k, and final residual ordering for
   the remaining one-token correctness cliff. The next correctness target is
   routed expert gate/up accumulation/materialization: either make the tiled
   reduction land on the host-side value for boundary-sensitive rows, or promote
   this path under an explicit layer-output tolerance policy that accepts the
   measured BF16-scale drift. The performance profile from this diagnostic is
   still submit/residency-heavy (`command_buffer_wait=743.468 ms` versus
   `qwen36_ffn_int4_stage5_with_router` GPU `104.130 ms`), so the next runtime
   optimization should not start until the correctness policy for this routed
   gate/up BF16 boundary is decided.

54. **Qwen3.6 layer-output tolerance policy**
   The fused-routed sweep is now schema `v31`. The sweep now summarizes an
   explicit `layer_output_tolerance_policy` beside the existing promotion gate.
   For each candidate mode and prompt it records whether layer-output checksum
   mismatches have complete delta-tap evidence, the minimum required BF16
   absolute-delta / ordered-ULP / differing-element thresholds, whether the
   current promotion tolerance covers those mismatches, and whether the mode is
   diagnostic-only. The Markdown report also renders a dedicated Layer Output
   Tolerance Policy table so tolerance decisions are visible in review logs.

   The shortest Metal smoke wrote
   `/private/tmp/qwen36_layer_tolerance_policy_v31_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch,full-stage5-router-simd-batch-shared-host-corrected,full-stage5-router-simd-batch-shared-routed-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all --no-promotion-require-profile`.
   Generated IDs still matched (`[11]`), and no candidate was promotable. The
   policy recommendation was
   `choose_explicit_layer_output_tolerance_or_fix_kernel`, with no missing
   delta evidence. The non-diagnostic `full-stage5-router-simd-batch` row had
   `65` layer-output checksum mismatches. Its first mismatch was still layer
   `7` FFN with `max_abs_delta=0.00048828125`, `max_ulp_delta=1`, and one
   differing BF16 element, but tolerating the whole downstream stream would
   require `max_abs_delta=0.15625`, `max_ulp_delta=31081`, and `1991`
   differing elements. The shared-only diagnostic row had `13` mismatches and
   would still require `max_abs_delta=0.0625`, `max_ulp_delta=30737`, and
   `1951` differing elements. The shared+routed diagnostic remained at zero
   layer-output mismatches.

   That makes the policy decision clear: the current routed SIMD batch path
   should not be promoted under a broad layer-output tolerance. The local first
   cliffs are BF16-scale (`0.00048828125` at layer `7`, then `0.0001220703125`
   at layer `33` after shared correction), but the uncorrected downstream
   hidden stream diverges too far for an acceptable promotion threshold. The
   next correctness step is a kernel fix for the routed expert gate/up/SwiGLU
   accumulation/materialization boundary, not a tolerance-only gate change.
   The performance profile remains useful after that fix lands: the measured
   v31 run still classifies the FFN path as submit/residency heavy for the
   correction diagnostics, while the uncorrected batch row shows large GPU work
   under profile (`command_buffer_wait=829.747 ms`,
   `fused_gpu_ms=799.781 ms`). Runtime optimization should resume only after a
   non-diagnostic row reaches layer-output parity or a much narrower,
   locally-proven tolerance.

55. **Qwen3.6 routed gate/up host-order Metal rejection**
   The fused-routed sweep is now schema `v33`. It adds an env-gated routed
   expert gate/up Metal probe,
   `SUPERSONIC_METAL_QWEN36_FFN_EXPERT_GATE_UP_HOST_ORDER_STAGE5=1`, and two
   sweep modes:
   `full-stage5-router-simd-batch-routed-gate-up-host-order` and the diagnostic
   `full-stage5-router-simd-batch-shared-host-corrected-routed-gate-up-host-order`.
   The new Metal kernel uses one active thread per `(top_k, row)` item and
   mirrors the Rust reference's packed-pair INT4 expression before writing
   `expert_mid`. It keeps the normal tiled path as the default and labels phase
   profiling as `qwen36_ffn_int4_expert_gate_up_host_order_stage5`.

   A release rebuild was required before measuring this Objective-C++ change:
   `cargo build --release -p runner --bin supersonic`. A one-mode phase smoke
   wrote `/private/tmp/qwen36_routed_gate_up_host_order_v33_phase_1tok.{json,md}`
   and confirmed the new profile label was live:
   `command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_host_order_stage5` had
   `40` calls and `71.125 ms` total GPU time. Generated IDs still matched
   (`[11]`).

   The rebuilt five-mode smoke wrote
   `/private/tmp/qwen36_routed_gate_up_host_order_v33_1tok.{json,md}` with
   `--modes default,full-stage5-router-simd-batch,full-stage5-router-simd-batch-routed-gate-up-host-order,full-stage5-router-simd-batch-shared-host-corrected-routed-gate-up-host-order,full-stage5-router-simd-batch-shared-routed-host-corrected
   --max-new-tokens 1 --context-size 64 --metal-profile --layer-output-tap
   --layer-output-delta-tap --layer-output-delta-all --no-promotion-require-profile`.
   All rows ran and generated IDs matched (`[11]`), but no row was promotable.
   The normal SIMD batch row still had `65` layer-output mismatches, first at
   layer `7` FFN with `max_abs_delta=0.00048828125`, `max_ulp_delta=1`, and one
   differing BF16 element. The routed gate/up host-order row regressed the
   layer-output policy summary to `67` mismatches, first at layer `6` FFN with
   `max_abs_delta=0.000244140625`, `max_ulp_delta=1`, and one differing BF16
   element; tolerating the whole downstream stream would require
   `max_abs_delta=0.71875`, `max_ulp_delta=31348`, and `2028` differing
   elements. Adding shared host correction did not rescue the host-order routed
   gate/up path: the shared-corrected host-order diagnostic also had `67`
   mismatches. The full shared+routed host-corrected diagnostic remained at zero
   layer-output mismatches.

   That rejects single-thread Metal host-order routed gate/up as a correctness
   fix. The remaining problem is narrower than routed down/combine and top-k,
   but it is not solved by replacing the tiled reduction with a scalar Metal dot
   in the current kernel. The next correctness probe should add a non-patching
   routed gate/up tap, or split the existing routed host-correction tap, so the
   host-order Metal gate/up/SwiGLU values can be compared directly against the
   host reference at the new layer `6` cliff and the older layer `33` cliff.
   Until that probe names whether the source is Metal `h_norm`, dot expression
   evaluation, SiLU/exp, or product materialization, the tiled FFN optimization
   path should stay behind the layer-output parity gate.

## Sources

- [Qwen/Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [QwenLM/Qwen3.6 README](https://github.com/QwenLM/Qwen3.6)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Transformers Qwen3MoE docs](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)
- [vLLM expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM fused MoE modular kernel](https://docs.vllm.ai/en/v0.10.1.1/design/fused_moe_modular_kernel.html)
- [vLLM MoE kernel features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
- [vLLM MTP docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [vLLM Qwen3-Next blog](https://vllm.ai/blog/2025-09-11-qwen3-next)
- [SGLang expert parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [llama.cpp MTP PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- [KTransformers KT-Kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- [KTransformers releases](https://github.com/kvcache-ai/ktransformers/releases)
- [Qwen3.6 speculative decoding RTX 3090 benchmark repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
- [SuperSonic Apple M5 Max Metal Qwen3.6 measurements](../performance.md#metal--apple-m5-max-apple-m5-max)
- [SuperSonic Qwen3.6 batched-prefill grouped MoE design](2026-05-05-qwen36-moe-batched-prefill-results.md)
