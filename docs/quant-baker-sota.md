# Quant Baker Profiles

This branch introduces a unified weight quantization profile surface:

```bash
cargo run --release --bin supersonic -- \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --weight-quant int4-gptq
```

Legacy aliases remain valid:

- `--int4` selects `int4-gptq`.
- `--fp8-runtime` selects `fp8-native`.
- `--q4km` selects `q4km`.
- `--q4km-gptq` selects `q4km-gptq`.

## Implemented Qwen Profiles

### `int4-gptq`

Existing calibration path. Emits SuperSonic native INT4 runtime layout:

- packed `u8` weights, two 4-bit nibbles per byte
- BF16 `_int4_scale` sidecar
- BF16 `_int4_zero` sidecar
- `LayoutTag::Int4Quantized`

Bake:

```bash
python oracle/bake_int4.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --profile int4-gptq \
  --num-samples 128 \
  --seqlen 2048
```

### `int4-hqq`

Data-free HQQ-style alternating least-squares path. It emits the same native
INT4 runtime layout as GPTQ, so no kernel changes are required.

Bake:

```bash
python oracle/bake_int4.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --profile int4-hqq \
  --hqq-iters 4 \
  --skip-ppl
```

`--skip-ppl` is optional, but useful for first smoke bakes. The HQQ bake does
not load WikiText-2 calibration data.

GPTQ calibration windows are cached under:

```text
{model_dir}/.supersonic/calib/
```

The cache key includes corpus, sample count, sequence length, and seed.

### `int4-awq`

Calibration-backed AWQ-style path. It uses mean absolute input activations to
choose per-input-channel AWQ scales, quantizes `W * awq_scale`, and emits the
native INT4 packed `u8` weights plus BF16 scale/zero sidecars. It also writes a
BF16 `<tensor>_awq_inv_scale` sidecar. The Qwen decode descriptor exposes that
sidecar to the HIP kernel, and both the Qwen decode INT4 paths and the shared
HIP prefill INT4 matmul helper multiply each dequanted column by
`awq_inv_scale[col]` before accumulation. When the sidecar pointer is null,
GPTQ/HQQ/Qwen36 runtime math is unchanged.

Bake:

```bash
python oracle/bake_int4.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --profile int4-awq \
  --num-samples 128 \
  --seqlen 2048
```

### `int4-autoround`

Calibration-backed AutoRound/SignRound-style path. It initializes native INT4
scale/zero with the AWQ weighted range search, then optimizes rounding choices
against sampled calibration activations using a sigmoid relaxation. Optimization
is row-chunked, and very large tensors fall back to the AWQ rounding path unless
`--autoround-max-elements` is raised.

Bake:

```bash
python oracle/bake_int4.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --profile int4-autoround \
  --num-samples 128 \
  --seqlen 2048 \
  --autoround-steps 20 \
  --autoround-max-rows 4096
```

## Reserved Profiles

These names are wired through CLI parsing, manifest metadata, bake directories,
fetch variants, and upload artifact naming, but the full baker/runtime support
is intentionally not enabled yet:

- `higgs4`
- `quip-e8`
- `qtip-trellis2`

`higgs4`, `quip-e8`, and `qtip-trellis2` also have reserved Qwen low-bit type
codes, but the Qwen HIP dequant loaders are not implemented. The runner rejects
them clearly rather than falling back to BF16.

## Release Upload

GPTQ keeps the legacy upload flag:

```bash
python oracle/upload_bake.py \
  --model qwen3.5-0.8b \
  --int4 \
  --model-dir /path/to/Qwen3.5-0.8B
```

Named profiles use `--weight-quant`:

```bash
python oracle/upload_bake.py \
  --model qwen3.5-0.8b \
  --weight-quant int4-hqq \
  --model-dir /path/to/Qwen3.5-0.8B
```

## Local Smoke Ranking

The reusable comparison harness is:

```bash
python oracle/quant_bake/compare_profiles.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile bf16 \
  --profile int4-gptq \
  --profile int4-hqq=/mnt/data/tmp/supersonic-hqq-smoke \
  --profile int4-awq=/mnt/data/tmp/supersonic-awq-smoke \
  --profile int4-autoround=/mnt/data/tmp/supersonic-autoround-smoke \
  --out-json target/quant_profile_compare_smoke.json \
  --out-md target/quant_profile_compare_smoke.md
```

Use `label:profile=/path/to/bake` when comparing multiple bakes that share the
same runtime profile, for example `hqq-i16:int4-hqq=/mnt/data/tmp/supersonic-hqq-i16`.

Current local smoke results on RX 7900 XTX, Qwen3.5 0.8B, short
teacher-forced prompt:

| Profile | PPL proxy | avg NLL | generation ms/token | notes |
|:--|--:|--:|--:|:--|
| bf16 | 26.315 | 3.270 | 9.0 | baseline |
| int4-gptq | 145.207 | 4.978 | 11.0 | legacy bake, no quant metadata |
| int4-hqq | 126.943 | 4.844 | 11.0 | data-free smoke bake |
| int4-awq | 146.390 | 4.986 | 11.0 | smoke bake, `2x128` calibration |
| int4-autoround | 138.670 | 4.932 | 11.0 | smoke bake, `1x64`, 2 steps |

Interpretation: these smoke bakes validate package/runtime mechanics, not
production quality. The next useful research step is a controlled Qwen3.5 0.8B
quality sweep where GPTQ, AWQ, and AutoRound all use the same calibration
budget, e.g. `32x512` for screening and `128x2048` for final ranking. The
existing `int4-gptq` bake predates `quant_method` metadata, so it should not be
treated as the final GPTQ reference.

## Reconstruction Audits

Native INT4 packages can be audited against the source safetensors with:

```bash
python oracle/quant_bake/audit_reconstruction.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --bake-dir /mnt/data/tmp/supersonic-hqq-i16 \
  --out-json target/recon_hqq_i16.json \
  --out-md target/recon_hqq_i16.md
```

Current Qwen3.5 0.8B reconstruction results:

| Bake | weighted MSE | mean rel L2 | max rel L2 | notes |
|:--|--:|--:|--:|:--|
| HQQ, 4 iters | 5.200750e-06 | 0.153314 | 0.217269 | data-free smoke |
| HQQ, 16 iters | 4.820526e-06 | 0.141790 | 0.199951 | includes tied `lm_head`; better reconstruction and proxy PPL |
| AWQ, `32x512` | 6.366651e-06 | 0.168656 | 0.259097 | calibration-backed, weak short-proxy gain |
| AWQ, `128x2048` | 6.427213e-06 | 0.168399 | 0.259097 | full calibration; no reconstruction gain from larger sample set |
| AutoRound smoke | 6.368463e-06 | 0.168677 | 0.259175 | tiny settings, effectively AWQ quality |
| AutoRound s8, `128x2048` | 6.431777e-06 | 0.168479 | 0.259540 | 150 tensors optimized, `lm_head` AWQ fallback; still AWQ-like |
| GPTQ, `32x512` | 1.075389e-05 | 0.227036 | 0.424536 | fresh calibrated bake; quality proxy beats HQQ despite worse reconstruction |
| GPTQ, `128x2048` | 1.014622e-05 | 0.216242 | 0.406790 | full calibration budget; slightly better reconstruction than `32x512` |
| existing GPTQ bake | 1.986569e-05 | 0.300992 | 0.597653 | legacy local bake, no `quant_method` metadata |
| Qwen3.5 2B HQQ, 16 iters | 3.103073e-06 | 0.142462 | 0.235331 | scales to 2B; all 151 tensors audited |

For tied-output Qwen checkpoints, the audit compares baked `lm_head.weight`
against `model.language_model.embed_tokens.weight` when the checkpoint does not
expose a separate `lm_head.weight` tensor.

## HQQ Iteration Sweep

The first useful local signal is that HQQ benefits from more alternating LSQ
iterations without changing runtime cost:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token |
|:--|--:|--:|--:|--:|--:|
| bf16 | 26.315 | 3.270 | 454.6 | 769.0 | 8.0 |
| HQQ, 4 iters | 126.943 | 4.844 | 364.2 | 662.8 | 11.0 |
| HQQ, 16 iters | 82.006 | 4.407 | 381.0 | 662.1 | 11.0 |

HQQ 16 remains the best data-free native INT4 candidate, but a fresh calibrated
GPTQ `32x512` bake beats it on the short quality proxy. The HQQ finalization
path avoids materializing full-size scale/zero expansions, which keeps
vocab-sized tensors such as Qwen3.5 2B `lm_head` under the 24 GiB RX 7900 XTX
VRAM limit.

## Qwen3.5 0.8B `32x512` Calibration Screen

A fresh GPTQ reference was baked with the same `32x512` calibration budget used
for the AWQ screen:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-gptq \
  --num-samples 32 \
  --seqlen 512 \
  --out-dir /mnt/data/tmp/supersonic-gptq-32x512 \
  --skip-ppl \
  --device cuda
```

The GPTQ bake took 2.2 minutes; `lm_head` accounted for 94.2 seconds of that.
It wrote an 885.9 MiB package and passed dense INT4 self-check with `0/151`
mismatches.

Short runtime proxy:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 26.315 | 3.270 | 435.1 | 714.6 | 9.0 | baseline |
| GPTQ, `32x512` | 51.116 | 3.934 | 363.3 | 661.9 | 11.0 | fresh calibrated bake |
| AWQ, `32x512` | 145.134 | 4.978 | 376.4 | 665.9 | 11.0 | current AWQ implementation |
| HQQ, 16 iters | 82.006 | 4.407 | 377.3 | 661.7 | 11.0 | data-free bake |

This is a useful caution: reconstruction error alone is not enough for ranking
native INT4 methods. HQQ has lower weight reconstruction error, but calibrated
GPTQ preserves the short language-model proxy better.

## Qwen3.5 0.8B Full GPTQ Calibration

The full GPTQ bake used the target production calibration budget:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-gptq \
  --num-samples 128 \
  --seqlen 2048 \
  --out-dir /mnt/data/tmp/supersonic-gptq-128x2048 \
  --skip-ppl \
  --device cuda
```

It created the `wikitext-2-raw-v1_train-n128-t2048-seed0.json` calibration
cache, took 3.1 minutes total, wrote an 885.9 MiB package, and passed dense
INT4 self-check with `0/151` mismatches. The `lm_head` step took 79.3 seconds.

Short runtime proxy:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 26.315 | 3.270 | 432.3 | 714.7 | 8.0 | baseline |
| GPTQ, `128x2048` | 63.822 | 4.156 | 361.2 | 658.9 | 11.0 | full calibration |
| GPTQ, `32x512` | 51.116 | 3.934 | 377.7 | 661.8 | 11.0 | screening calibration |
| HQQ, 16 iters | 82.006 | 4.407 | 378.0 | 660.7 | 11.0 | data-free bake |

The full calibration bake reconstructs weights slightly better than `32x512`,
but this short prompt proxy ranks it worse. Treat this as a prompt-sensitivity
warning: final ranking needs a real corpus metric, not only the single
teacher-forced prompt baked into the quick harness.

The comparison harness now accepts repeated `--prompt` arguments and aggregates
teacher-forced NLL over all scored tokens. A four-prompt smoke set reverses the
single-prompt GPTQ result:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 44.269 | 3.790 | 419.6 | 711.6 | 8.0 | baseline |
| GPTQ, `128x2048` | 70.005 | 4.249 | 373.3 | 659.6 | 11.0 | full calibration |
| GPTQ, `32x512` | 76.501 | 4.337 | 376.5 | 661.1 | 11.0 | screening calibration |
| HQQ, 16 iters | 83.276 | 4.422 | 377.6 | 662.4 | 11.0 | data-free bake |

So GPTQ `128x2048` remains the best current native INT4 baseline once scoring
uses more than one prompt.

## Qwen3.5 0.8B Full AWQ Calibration

AWQ was baked with the same `128x2048` calibration cache:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-awq \
  --num-samples 128 \
  --seqlen 2048 \
  --out-dir /mnt/data/tmp/supersonic-awq-128x2048 \
  --skip-ppl \
  --device cuda
```

It took 1.2 minutes, wrote an 885.9 MiB package, and passed dense INT4
self-check with `0/151` mismatches. The `lm_head` activation-stat pass took
5.4 seconds.

Four-prompt proxy:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 44.269 | 3.790 | 430.8 | 724.8 | 8.0 | baseline |
| GPTQ, `128x2048` | 70.005 | 4.249 | 373.2 | 660.9 | 11.0 | current baseline |
| AWQ, `128x2048` | 166.052 | 5.112 | 377.8 | 660.6 | 11.0 | current implementation |
| HQQ, 16 iters | 83.276 | 4.422 | 377.3 | 662.0 | 11.0 | data-free bake |

This pre-sidecar AWQ path should stay experimental. The larger calibration set
does not improve reconstruction, and the multi-prompt proxy is substantially
worse than GPTQ and HQQ. That suggested the simplified native-format AWQ
implementation was missing AWQ's activation scaling behavior rather than
merely needing more calibration samples. The branch now emits, loads, and
consumes a per-input-channel inverse-scale sidecar in Qwen prefill and decode,
so these measurements must be rerun with a fresh AWQ bake.

## Qwen3.5 0.8B AWQ Sidecar Screen

After wiring `<tensor>_awq_inv_scale` through the Qwen loader, prefill INT4
helper, and decode megakernel, AWQ was re-baked with the screening calibration
budget:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-awq \
  --num-samples 32 \
  --seqlen 512 \
  --out-dir /mnt/data/tmp/supersonic-awq-sidecar-32x512 \
  --skip-ppl \
  --device cuda
```

The bake took 0.4 minutes, wrote a 886.6 MiB package, emitted 151 BF16
`_awq_inv_scale` tensors, and passed dense sidecar reconstruction self-check
with `0/151` mismatches.

The comparison below was run with the rebuilt sidecar-aware binary and an
isolated symlinked model directory under `/mnt/data/tmp/Qwen3.5-0.8B-compare`
so existing local bakes were not overwritten:

```bash
python oracle/quant_bake/compare_profiles.py \
  --binary target/release/supersonic \
  --model-dir /mnt/data/tmp/Qwen3.5-0.8B-compare \
  --profile bf16 \
  --profile gptq32:int4-gptq=/mnt/data/tmp/supersonic-gptq-32x512 \
  --profile gptq128:int4-gptq=/mnt/data/tmp/supersonic-gptq-128x2048 \
  --profile hqq-i16:int4-hqq=/mnt/data/tmp/supersonic-hqq-i16 \
  --profile awq-sidecar32:int4-awq=/mnt/data/tmp/supersonic-awq-sidecar-32x512
```

Four-prompt proxy:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 62.058 | 4.128 | 419.2 | 710.1 | 8.0 | rebuilt binary |
| GPTQ, `32x512` | 112.586 | 4.724 | 384.0 | 683.5 | 18.0 | screening calibration |
| GPTQ, `128x2048` | 136.490 | 4.916 | 388.7 | 684.7 | 18.0 | full calibration |
| HQQ, 16 iters | 144.336 | 4.972 | 388.0 | 684.4 | 18.0 | data-free bake |
| AWQ sidecar, `32x512` | 246.800 | 5.509 | 393.4 | 689.7 | 19.0 | sidecar alpha=0.5 |

Conclusion: the sidecar runtime contract is now working, but the current AWQ
producer is still not competitive. The likely problem is the simplistic
per-input-channel scale formula/objective (`alpha=0.5`, direct mean-abs
importance) rather than missing runtime support. Do not spend a full
`128x2048` AWQ sidecar bake until a cheap alpha/objective grid improves the
`32x512` screen.

The baker now exposes that grid directly:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-awq \
  --awq-scale-mode activation-weight \
  --awq-scale-alpha 0.25 \
  --num-samples 32 \
  --seqlen 512 \
  --out-dir /mnt/data/tmp/supersonic-awq-sidecar-activation_weight_a025-32x512 \
  --skip-ppl \
  --device cuda
```

The first grid tested direct activation scaling, weight-normalized scaling,
and inverted activation scaling:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 62.058 | 4.128 | 420.3 | 716.8 | 8.0 | |
| GPTQ, `32x512` | 112.586 | 4.724 | 383.9 | 680.7 | 18.0 | screening calibration |
| HQQ, 16 iters | 144.336 | 4.972 | 386.6 | 681.6 | 18.0 | data-free bake |
| AWQ activation, `alpha=0.25` | 277.137 | 5.625 | 392.2 | 688.4 | 19.0 | sidecar |
| AWQ activation, `alpha=0.50` | 246.800 | 5.509 | 393.0 | 688.6 | 19.0 | sidecar |
| AWQ activation-weight, `alpha=0.25` | 240.139 | 5.481 | 393.7 | 687.5 | 18.0 | best AWQ grid point |
| AWQ activation-weight, `alpha=0.50` | 326.209 | 5.788 | 392.8 | 687.7 | 18.0 | sidecar |
| AWQ activation-weight, `alpha=0.75` | 17974.629 | 9.797 | 392.0 | 688.2 | 19.0 | unstable |
| AWQ inverse activation, `alpha=0.50` | 32572155.360 | 17.299 | 394.3 | 688.2 | 19.0 | invalid direction |

This rules out a simple alpha or denominator fix. AWQ should remain
experimental until the producer switches to a layerwise search that minimizes
actual layer-output error, or until its scale search is borrowed more directly
from a known-good AWQ implementation. The sidecar load/runtime path is useful
infrastructure, but the current producer should not be promoted.

## Qwen3.5 0.8B AutoRound Screen

A bounded AutoRound screen used the same `128x2048` calibration cache:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-0.8B \
  --profile int4-autoround \
  --num-samples 128 \
  --seqlen 2048 \
  --autoround-steps 8 \
  --autoround-max-rows 2048 \
  --autoround-row-chunk 512 \
  --out-dir /mnt/data/tmp/supersonic-autoround-128x2048-s8 \
  --skip-ppl \
  --device cuda
```

It took 1.3 minutes, optimized 150 tensors, fell back to AWQ for `lm_head`, and
passed dense INT4 self-check with `0/151` mismatches.

Four-prompt proxy:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 44.269 | 3.790 | 423.9 | 722.9 | 8.0 | baseline |
| GPTQ, `128x2048` | 70.005 | 4.249 | 372.7 | 660.7 | 11.0 | current baseline |
| AutoRound s8 | 158.964 | 5.069 | 377.2 | 661.4 | 11.0 | bounded screen |
| AWQ, `128x2048` | 166.052 | 5.112 | 377.3 | 661.9 | 11.0 | current AWQ implementation |
| HQQ, 16 iters | 83.276 | 4.422 | 377.6 | 661.8 | 11.0 | data-free bake |

The current AutoRound implementation is not worth scaling to more steps yet.
It tracks AWQ reconstruction and quality, which means the rounding relaxation
is not correcting the missing activation-scaling behavior in the native INT4
format.

## Qwen3.5 2B HQQ Screen

Qwen3.5 2B was baked with:

```bash
python oracle/bake_int4.py \
  --model-dir /mnt/data/models/Qwen3.5-2B \
  --profile int4-hqq \
  --hqq-iters 16 \
  --out-dir /mnt/data/tmp/supersonic-qwen35-2b-hqq-i16 \
  --skip-ppl \
  --device cuda
```

The bake wrote a 1987.4 MiB package, quantized 151 tensors, and passed dense
INT4 self-check with `0/151` mismatches.

Short runtime proxy on RX 7900 XTX:

| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |
|:--|--:|--:|--:|--:|--:|:--|
| bf16 | 13.483 | 2.601 | 534.3 | 804.6 | 12.0 | baseline |
| int4-gptq | 92.555 | 4.528 | 472.0 | 685.4 | 12.0 | legacy local bake |
| HQQ, 16 iters | 44.358 | 3.792 | 410.1 | 679.3 | 12.0 | data-free bake |

This confirms the 0.8B HQQ signal is not isolated to the smallest model, but
the fresh 0.8B GPTQ results mean calibrated GPTQ `128x2048` is now the
production baseline to beat. HQQ is the best non-calibrated fallback. AWQ now
has the sidecar contract wired through Qwen prefill and decode, but the first
fresh sidecar bake is worse than GPTQ and HQQ; the AWQ producer needs
alpha/objective tuning before another full calibration run. AutoRound should
remain experimental until its rounding path is retuned on top of the corrected
AWQ activation-scaling format.
