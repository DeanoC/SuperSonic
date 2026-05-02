# Phi-4 CUDA INT4 parity notes

Status as of 2026-05-02:

- CUDA build passes with `SUPERSONIC_BACKENDS=cuda cargo build --release --bin supersonic`.
- The Phi-4 INT4 corpus is `12/12` against the kernel-accurate deterministic Python oracle.
- The same corpus remains `10/12` against the live PyTorch BF16 oracle.
- The live-PyTorch misses are generation differences caused by accumulated BF16 matmul sensitivity, not a localized CUDA projection mismatch.
- `SUPERSONIC_PHI4_DUMP_LAYER_TRACE` is intentionally kept as diagnostic plumbing for this lane.

Kernel-accurate corpus command:

```bash
HF_HOME=/dev/shm/hf_home SUPERSONIC_BACKENDS=cuda python3 oracle/int4_corpus_compare.py \
  --model-dir /dev/shm/Phi-4-mini-int4-oracle \
  --bake-subdir .supersonic/v2-int4-gptq \
  --binary target/release/supersonic \
  --model-variant phi4-mini \
  --max-new-tokens 8 \
  --device cuda:0 \
  --deterministic-projection-oracle \
  --deterministic-lm-head \
  --deterministic-attention-mode hybrid \
  --report /tmp/phi4_corpus_deterministic_projection_oracle_hybrid.json
```

Live PyTorch corpus command:

```bash
HF_HOME=/dev/shm/hf_home SUPERSONIC_BACKENDS=cuda python3 oracle/int4_corpus_compare.py \
  --model-dir /dev/shm/Phi-4-mini-int4-oracle \
  --bake-subdir .supersonic/v2-int4-gptq \
  --binary target/release/supersonic \
  --model-variant phi4-mini \
  --max-new-tokens 8 \
  --device cuda:0 \
  --report /tmp/phi4_corpus_after_qkv_trace.json
```

Known live-PyTorch misses:

- `The quick brown fox`: Python starts with `jumps`; CUDA/Rust starts with `foxes`.
- `Water boils at 100 degrees Celsius at sea level because`: Python ends the 8-token window with `transitions`; CUDA/Rust emits `turns`.

Layer-trace diagnostics show:

- Layer 1 starts from identical embeddings.
- Given Rust-traced inputs, Python deterministic replay matches CUDA for QKV, attention `o_proj` residual, MLP gate/up/SwiGLU/down, and final hidden at the traced boundaries.
- The live PyTorch BF16 hook differs from deterministic F32 dot plus BF16 output rounding at Q/K in layer 1, then that small difference accumulates through 32 layers.
- Forcing `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False` does not remove the layer-1 QKV mismatch. It changes which corpus prompts miss, so it is not a clean parity definition.
- `--deterministic-projection-oracle --deterministic-lm-head --deterministic-attention-mode hybrid` patches the Python oracle's decoder Linear, RMSNorm, attention, SwiGLU, and lm-head boundaries to use F32 arithmetic plus explicit BF16 rounding. In hybrid attention mode, prompts that start at 10 or more tokens use the explicit decode-order attention loop from the first token; shorter prompts recompute the attention path before each prompt/decode token and switch to the loop once the resulting KV length is at least 10 tokens.

Useful reproducibility commands:

```bash
HF_HOME=/dev/shm/hf_home SUPERSONIC_BACKENDS=cuda python3 oracle/phi4_int4_layer_drift.py \
  --model-dir /dev/shm/Phi-4-mini-int4-oracle \
  --bake-subdir .supersonic/v2-int4-gptq \
  --binary target/release/supersonic \
  --model-variant phi4-mini \
  --prompt "The quick brown fox" \
  --layers 1 \
  --single-launch-trace \
  --component-dump-layers 1 \
  --device cuda:0 \
  --report /tmp/phi4_quick_l1_trace.json
```

```bash
HF_HOME=/dev/shm/hf_home SUPERSONIC_BACKENDS=cuda python3 oracle/int4_corpus_compare.py \
  --model-dir /dev/shm/Phi-4-mini-int4-oracle \
  --bake-subdir .supersonic/v2-int4-gptq \
  --binary target/release/supersonic \
  --model-variant phi4-mini \
  --max-new-tokens 8 \
  --device cuda:0 \
  --disable-bf16-reduced-precision-reduction \
  --report /tmp/phi4_corpus_bf16_reduce_off.json
```

The next kernel experiment, if exact live-PyTorch parity is still required, should start by changing QKV matvec accumulation behavior in CUDA. The evidence so far does not justify a broad projection rewrite: every traced Rust input replays to the current CUDA output essentially exactly, while PyTorch's live BF16 GEMM output remains backend-sensitive.
