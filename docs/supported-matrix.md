# Supported matrix

The active matrix contains one model, one custom artifact source, one GQH
quantization lane, and two HIP architectures. Rows are correctness claims;
performance measurements belong in [performance](performance.md).

The machine-readable source is [`support/matrix.toml`](../support/matrix.toml).
Run `python3 tools/check-support-matrix.py` after changing a row.

## Active contract

| Model | Source | Quantization | Architecture | Gate |
| --- | --- | --- | --- | --- |
| `qwen3.8-27b` | `gqh-gguf` | `gqh` | `gfx1100` | `qwen38-gqh-correctness` |
| `qwen3.8-27b` | `gqh-gguf` | `gqh` | `gfx1201` | `qwen38-gqh-correctness` |

### HIP on `gfx1100`

The `gfx1100` row is the component and bring-up lane. Its named gate uses
the full serial GQH artifact crawl with ignored tests enabled.

### HIP on `gfx1201`

The `gfx1201` row is the R9700 serial lane. Its workflow validates the
physical device, waits for selected-device idleness, builds a release
workspace, crawls the artifact, and compares ordinary and MTP tokens.

## Status vocabulary

`experimental` means the row is wired to a named correctness gate and is
still subject to promotion from runner evidence. A row is not implied to be
portable to an architecture absent from this document.
