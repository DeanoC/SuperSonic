# Supported matrix

SuperSonic has one public model and one public weight source. The rows below
are the complete product contract, not a promise of compatibility with
unlisted artifacts or devices.

The machine-readable source is [`support/matrix.toml`](../support/matrix.toml).
Run `python3 tools/check-support-matrix.py` after changing a row.

| Model | Weight source | Architecture | Status | Generation | Correctness gate |
| --- | --- | --- | --- | --- | --- |
| `qwen3.8-27b` | custom `gqh-gguf` | `gfx1100` | `experimental` | single-sequence greedy, optional NextN/MTP | `qwen38-gqh-correctness` |
| `qwen3.8-27b` | custom `gqh-gguf` | `gfx1201` | `experimental` | single-sequence greedy, optional NextN/MTP | `qwen38-gqh-correctness` |

## HIP on `gfx1100`

The `gfx1100` row is a supported ROCm/HIP target for component and artifact
correctness. Its named gate uses the serial GQH artifact tests with ignored
correctness cases selected explicitly.

## HIP on `gfx1201`

The `gfx1201` row is the self-hosted R9700 lane. Its workflow validates the
physical device, waits for selected-device idleness, builds with
`HIP_ARCH=gfx1201`, checks the configured artifact pair, and compares ordinary
and NextN/MTP token arrays.

## Failure boundary

The `--model` value must be `qwen3.8-27b`; the model directory and custom GQH
GGUF must be supplied together. Unlisted model names, architectures, artifact
sources, multi-sequence requests, and non-greedy controls fail explicitly.
There is no silent fallback to another row or loader.

The workflow's throughput telemetry is diagnostic until repeated measurements
establish variance and a separately documented threshold. Correctness gates
remain authoritative for support status.
