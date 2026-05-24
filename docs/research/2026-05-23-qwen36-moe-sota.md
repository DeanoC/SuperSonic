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
   IDs and records expert-residency policy rows so promotion decisions can be
   made on multi-token reuse rather than a cold first-token allocation.
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
   token, and records policy-blocked rows when the experiment gate is absent.
   The first Metal smoke sweep measured 2/2 prompts in 34.7s with aggregate
   `accepted_tokens=1`, `drafted_tokens=4`, `acceptance_rate=0.25`, and
   `target_steps_per_emitted=1.0`; the profiling prompt accepted 0/2 drafts and
   the coding prompt accepted 1/2. K=1 therefore remains an instrumentation path,
   not a policy-promotion candidate.

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
