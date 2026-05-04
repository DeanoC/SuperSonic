# KV Cache Quantization Research

Status: research branch `research/kv-quant`, HIP/Qwen first. CUDA certified-KV remains a parity baseline, not the production target for this track.

## SuperSonic Baseline

Qwen3.5 currently stores full-attention KV caches as BF16 buffers shaped by layer state. `--kv-fp8` switches those buffers to byte storage with per-head/token F32 scales and optional BF16 sidecar support in the persistent HIP megakernel. Linear-attention layers do not use a transformer KV cache; they maintain convolution and recurrent state.

Gemma 4 preallocates per-layer KV to `max_t`; shared layers alias earlier layer buffers. This note focuses on Qwen because the intended implementation point is `kernels/full_attention*.hip` and the Qwen state/descriptor path.

The first measurement target is attention-output error, not logits. A low-bit KV format must preserve:

- score ordering for K, because K perturbation changes softmax mass;
- weighted value sums for V, where error is attenuated by final attention mass;
- GQA layout, where one KV head feeds multiple query heads;
- decode write/read cost in the persistent megakernel.

## Certified-KV Parity

The existing certified path is Llama 3.1 CUDA-only. It stores quantized Tier-1 keys/values plus CPU-pinned original Tier-2 blocks and uses bounded promoted BF16 key/value scratch. This branch adds `--certified-kv-preset legacy|paper-v2` while keeping `legacy` as default.

`paper-v2` maps the research plan defaults:

- `v_tol=0.5`
- `rung1_threshold=0.02`
- `tau_cov=0.995`
- `k_min=2`
- `rung1_multiplier=2.0`
- `k_max=None`, spelled as `--certified-kv-k-max 0`

The pinned paper repo `DeanoC/certified-quantised-attention@7925ee8199faf95f336a3e916d558f83794039f9` represents unclamped `k_max` as `None` in `dotcache/kernels/certified_attention.py`. Its benchmark provenance constants include `PAPER_RUNG1_THRESHOLD=0.02`; the plan’s `paper-v2` preset intentionally uses the higher `v_tol=0.5` for this branch.

Operational cache caps remain caps. If unclamped selection asks for more promoted key blocks than the live key cache can hold, SuperSonic routes affected heads through the existing dense fallback instead of evicting selected blocks under the selector.

Value promotion parity is already aligned with the tighter rule: after initial attention, SuperSonic reduces final token probabilities to per-block mass and compares `final_block_mass * per_block_value_error` to `v_tol`. The Python oracle now has explicit coverage for that rule.

## Literature Notes

- KIVI, arXiv:2402.02750: tuning-free asymmetric 2-bit KV quantization. Main implementation lesson is asymmetric treatment: per-channel keys and per-token values. Good first INT2 baseline for Qwen, but Qwen RoPE/head dimensions need direct error measurement. Source: https://arxiv.org/abs/2402.02750
- KVQuant, arXiv:2401.18079: combines per-channel key quantization, pre-RoPE key quantization, non-uniform datatypes, dense-and-sparse outlier handling, and Q-Norm. Strong for ultra-low bits, but pre-RoPE capture is invasive in the current HIP megakernel. Source: https://arxiv.org/abs/2401.18079
- KVTuner, arXiv:2502.04420: layer-wise mixed precision driven by sensitivity. This maps well to SuperSonic’s registry/config style: per-layer KV format decisions can be baked into descriptors before adding kernels. Source: https://arxiv.org/abs/2502.04420
- LogQuant, arXiv:2503.19950: log-distributed 2-bit quantization. Interesting for outlier-heavy activations; HIP cost depends on whether decode can use table/log dequant cheaply enough. Source: https://arxiv.org/abs/2503.19950
- MILLION, arXiv:2504.03661: outlier-immunized product quantization for long context. Likely too complex for the first HIP pass, but useful if scalar INT2/INT4 loses too much Qwen accuracy. Source: https://arxiv.org/abs/2504.03661
- TurboQuant, arXiv:2504.19874: online vector quantization with random rotation and unbiased inner-product preservation; reports KV-cache quality neutrality at 3.5 bits/channel and marginal degradation at 2.5 bits/channel. Attractive for K score preservation, but rotation cost and HIP implementation complexity are second phase. Source: https://arxiv.org/abs/2504.19874
- KVLinC, arXiv:2510.05373: Hadamard rotation plus linear correction. Similar implementation concern to TurboQuant; candidate if scalar per-channel K cannot preserve attention scores. Source: https://arxiv.org/abs/2510.05373
- KVTC, arXiv:2511.01815: transform coding with PCA decorrelation, adaptive quantization, and entropy coding. Storage efficient, but entropy coding is awkward inside a persistent decode kernel. Source: https://arxiv.org/abs/2511.01815
- InnerQ, arXiv:2602.23200: hardware-aware, tuning-free quantization using inner-dimension grouping, scale reuse, hybrid strategies, and high-precision windows. This is directly relevant to HIP because scale reuse and coalesced dequant matter as much as nominal bits. Source: https://arxiv.org/abs/2602.23200
- VQKV, arXiv:2603.16435: training-free vector-quantization cache compression. Keep as a later vector-codebook candidate after scalar baselines. Source: https://arxiv.org/abs/2603.16435
- Adaptive bit allocation, arXiv:2604.04722 and KV-AdaQuant arXiv:2502.15075: assign precision by token/head/layer importance. This is promising for Qwen because full-attention layers are sparse in the hybrid stack, but needs a cheap online importance signal. Sources: https://arxiv.org/abs/2604.04722 and https://huggingface.co/papers/2502.15075

## Harness

`oracle/kv_quant_research.py` consumes `q/k/v` tensors from NPZ or generates deterministic Qwen-shaped synthetic tensors:

```bash
python3 oracle/kv_quant_research.py \
  --layers 4 --q-heads 16 --kv-heads 4 --tokens 512 --head-dim 256
```

It reports BF16, FP8-style, INT4, INT2, and KIVI-like INT2 estimates:

- estimated resident KV bytes;
- per-layer `max_abs`, `mean_l2`, and per-head relative L2 attention-output error;
- first recommended HIP candidate by lowest measured relative error among compressed schemes.

The oracle now has the debug capture path:

```bash
/home/deano/venvs/rocm/bin/python oracle/qwen35_oracle.py \
  --model-id /path/to/Qwen3.5-0.8B \
  --prompt-ids 9707,11,1879,374,697,829 \
  --max-new-tokens 0 \
  --dtype bf16 \
  --device cuda:0 \
  --kv-quant-capture-npz /tmp/qwen35-kv-capture.npz \
  --kv-quant-capture-only

/home/deano/venvs/rocm/bin/python oracle/kv_quant_research.py \
  --input /tmp/qwen35-kv-capture.npz \
  --output /tmp/qwen35-kv-quant-report.json
```

The NPZ stores post-RoPE query/key tensors and prepared value tensors with shape `q=[full_layers,q_heads,head_dim]`, `k/v=[full_layers,kv_heads,tokens,head_dim]`, plus `layer_ids`, `prompt_position`, and `prompt_tokens` metadata.

### Qwen3.5-0.8B 128-Token Capture

Local capture:

```bash
/home/deano/venvs/rocm/bin/python oracle/qwen35_oracle.py \
  --model-id /mnt/data/models/Qwen3.5-0.8B \
  --prompt-ids <128 tokenizer-generated ids> \
  --max-new-tokens 0 \
  --dtype bf16 \
  --device cuda:0 \
  --kv-quant-capture-npz /tmp/qwen35-0.8b-kv-capture-128.npz \
  --kv-quant-capture-only

/home/deano/venvs/rocm/bin/python oracle/kv_quant_research.py \
  --input /tmp/qwen35-0.8b-kv-capture-128.npz \
  --max-rel-l2-threshold 0.035 \
  --fail-on-threshold \
  --output /tmp/qwen35-0.8b-kv-quant-report-128.json
```

Captured layers were full-attention layers `[3, 7, 11, 15, 19, 23]` with `q=[6,8,256]` and `k/v=[6,2,128,256]`.

| Scheme | Estimated KV | Max layer relative L2 | Notes |
| --- | ---: | ---: | --- |
| BF16 | 1.500 MiB | 0.0000 | Dense baseline |
| FP8 token | 0.762 MiB | 0.0233 | Best compressed candidate |
| INT4 token group64 | 0.469 MiB | 0.4038 | Too much attention-output error for first HIP pass |
| INT4 K-only group64 | 0.984 MiB | 0.3985 | K quantization alone breaks score quality on this capture |
| INT4 V-only group64 | 0.984 MiB | 0.1624 | Better error than full INT4, but worse memory than FP8 because K remains BF16 |
| FP8 K + INT4 V | 0.615 MiB | 0.1633 | Saves memory versus FP8, but V INT4 dominates the error |
| INT2 token group64 | 0.281 MiB | 1.6290 | Not viable without a better quantizer |
| KIVI-like INT2 | 0.305 MiB | 1.0378 | Better than scalar INT2, still far behind FP8 |
 
This points to the existing Qwen `--kv-fp8` implementation as the correct first production-facing HIP path to validate and tune. INT4 should stay in simulation until it has a V-only or mixed-precision strategy with much lower attention-output error.

The same 128-token capture on `/mnt/data/models/Qwen3.5-2B` produced the same tensor shapes and layer ids. FP8 again passed the `0.035` threshold with max layer relative L2 `0.0162`; full INT4 reached `0.2053`, INT4 K-only reached `0.1602`, INT4 V-only reached `0.1911`, and FP8 K + INT4 V reached `0.1906`. All INT4 variants remain above the FP8-derived threshold.

`/mnt/data/models/Qwen3.5-4B` captured eight full-attention layers `[3, 7, 11, 15, 19, 23, 27, 31]` with `q=[8,16,256]` and `k/v=[8,4,128,256]`. FP8 passed with max layer relative L2 `0.0244`. Full INT4 reached `0.5516`, INT4 K-only `0.5292`, INT4 V-only `0.2930`, and FP8 K + INT4 V `0.2905`. This strengthens the recommendation to avoid INT4 HIP work until the quantizer changes substantially.

`/mnt/data/models/Qwen3.5-9B` has the same captured shape and full-attention layers as 4B. FP8 reached max layer relative L2 `0.0330`, just above the initial `0.03` threshold but below a cross-size `0.035` acceptance gate. Full INT4 reached `0.5399`, INT4 K-only `0.4317`, INT4 V-only `0.2626`, and FP8 K + INT4 V `0.2622`.

Summary with `max_rel_l2_threshold=0.035`:

| Model | Captured shape | FP8 max rel L2 | Full INT4 max rel L2 | Passing schemes |
| --- | --- | ---: | ---: | --- |
| Qwen3.5-0.8B | `q=[6,8,256]`, `k/v=[6,2,128,256]` | 0.0233 | 0.4038 | FP8 |
| Qwen3.5-2B | `q=[6,8,256]`, `k/v=[6,2,128,256]` | 0.0162 | 0.2053 | FP8 |
| Qwen3.5-4B | `q=[8,16,256]`, `k/v=[8,4,128,256]` | 0.0244 | 0.5516 | FP8 |
| Qwen3.5-9B | `q=[8,16,256]`, `k/v=[8,4,128,256]` | 0.0330 | 0.5399 | FP8 |

The capture matrix can be rerun with:

```bash
/home/deano/venvs/rocm/bin/python oracle/kv_quant_capture_matrix.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --model-dir /mnt/data/models/Qwen3.5-2B \
  --model-dir /mnt/data/models/Qwen3.5-4B \
  --model-dir /mnt/data/models/Qwen3.5-9B \
  --tokens 128 \
  --threshold 0.035 \
  --out-dir /tmp/supersonic-kv-quant
```

### Runtime KV-FP8 Validation

The existing HIP Qwen `--kv-fp8` runtime path was validated on the local BF16 bakes with a 4-token decode:

```bash
PATH=/home/deano/venvs/rocm/bin:$PATH cargo run --release --bin supersonic -- \
  --model qwen3.5-<size> \
  --model-dir /mnt/data/models/Qwen3.5-<SIZE> \
  --prompt "KV cache quantization preserves attention output." \
  --max-new-tokens 4 \
  --kv-fp8 \
  --validate
```

| Model | Prefill delta | Decode max delta | Decode ms/token | Result |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B | 0.1719 | 0.2031 | 102 | Pass |
| Qwen3.5-2B | 0.1328 | 0.2656 | 159 | Pass |
| Qwen3.5-4B | 0.1875 | 0.1875 | 214 | Pass |
| Qwen3.5-9B | 0.1875 | 0.1719 | 447 | Pass |

Single-sequence KV-FP8 currently uses replayed GPU prefill for correctness. The validation result supports treating the existing FP8 cache path as the implementation baseline while keeping lower-bit formats in the offline harness.

## HIP Implementation Order

1. Validate and tune the existing Qwen `--kv-fp8` path on HIP with real captures and corpus prompts.
2. Add pass/fail thresholds to the harness so FP8 remains the acceptance baseline for lower-bit experiments.
3. Prototype INT4 V-only simulation before touching HIP kernels; V errors are weighted by attention mass and are less likely to disrupt score ordering.
4. Add per-channel or KIVI-like K quantization only if attention score/order error stays inside the FP8-derived threshold.
5. Promote any new low-bit format into descriptors and registry entries only after corpus/oracle validation passes; keep it off by default.
