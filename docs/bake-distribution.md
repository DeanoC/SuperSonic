# Bake distribution via GitHub releases

Some SuperSonic bakes — notably Gemma 4 INT4 GPTQ and Qwen 9B INT4 — cannot
be produced on smaller machines because the calibration pass needs host/GPU
memory those machines don't have. To make these models usable on small-VRAM
hardware, SuperSonic can download a pre-baked package from a GitHub release
instead of baking locally.

This document covers the asset layout, the producer/consumer flow, and the
trust model.

## Overview

1. A **producer** (big-box machine) runs the existing Python baker
   (`oracle/bake_int4.py` for Qwen, `oracle/bake_int4_gemma4.py` for Gemma 4)
   to create a bake at `{model_dir}/.supersonic/v{N}[-variant]/`.
2. The producer runs `oracle/upload_bake.py`, which tars + zstd-compresses the
   bake, splits it if needed to stay under GitHub's 2 GiB per-asset cap, and
   uploads to a GitHub release with a `bakes-index.json` describing everything.
3. A **consumer** (small machine) runs `supersonic` normally. When the local
   bake is missing, the runner fetches the release index, downloads the
   matching asset(s), verifies SHA-256, and extracts into
   `{model_dir}/.supersonic/v{N}[-variant]/`. The rest of the pipeline is
   unchanged.

## Asset layout

One release per `FORMAT_VERSION`; tag `bakes-v{FORMAT_VERSION}` on
`DeanoC/SuperSonic`. Each release contains:

- `bakes-index.json` — single JSON file listing available bakes.
- One `{model}-{variant}-fmt{N}-cvt{M}.tar.zst` (or
  `…tar.zst.partNN` when >1800 MiB) per bake.

Models: `qwen3.5-{0.8b,2b,4b,9b}`, `gemma4-{e2b,e4b}`.
Variants: `bf16`, `fp8-native`, `int4-gptq`.

Example: `gemma4-e4b-int4-gptq-fmt1-cvt2.tar.zst`.

Tarballs contain `manifest.json` + `weights.bin` at the root plus the small
HuggingFace metadata files under an `hf/` prefix:

```
manifest.json
weights.bin
hf/config.json
hf/tokenizer.json
hf/tokenizer_config.json     (optional)
hf/special_tokens_map.json   (optional)
hf/generation_config.json    (optional)
hf/chat_template.json        (optional)
hf/tokenizer.model           (optional, SentencePiece models)
hf/preprocessor_config.json  (optional)
hf/processor_config.json     (optional)
```

The consumer extracts the root-level files into
`{model_dir}/.supersonic/v{N}[-variant]/` and the `hf/*` files into
`{model_dir}/` itself. A consumer can therefore point `--model-dir` at an
empty directory and the fetch populates everything needed — no separate HF
checkpoint download required.

The downloader rejects any tar entry with an unknown name, a path outside
the documented structure, or a non-file type.

### `bakes-index.json`

```json
{
  "schema_version": 1,
  "format_version": 1,
  "converter_version": 2,
  "generated_at": "2026-04-18T12:34:56Z",
  "producer": {"host": "...", "system": "...", "python": "..."},
  "bakes": [
    {
      "model": "qwen3.5-4b",
      "variant": "int4-gptq",
      "model_family": "qwen35",
      "asset": "qwen3.5-4b-int4-gptq-fmt1-cvt2.tar.zst",
      "parts": null,
      "sha256": "<hex>",
      "compressed_bytes": 2345678901,
      "uncompressed_bytes": 6123456789,
      "bake_manifest_sha256": "<hex>",
      "format_version": 1,
      "converter_version": 2
    }
  ]
}
```

When an asset is split, `parts` is a list of `{name, bytes, sha256}` objects
(one per part) and the top-level `sha256` is over the concatenated stream.

## Producer: publishing a bake

On the big-box machine, after running the appropriate baker:

```bash
pip install -r oracle/requirements-upload.txt    # zstandard
# First-time-only: ensure `gh auth login` is complete

# Dry-run first to inspect what would be uploaded.
python oracle/upload_bake.py \
  --model qwen3.5-4b \
  --int4 \
  --model-dir /path/to/Qwen3.5-4B \
  --dry-run

# Real upload.
python oracle/upload_bake.py \
  --model qwen3.5-4b \
  --int4 \
  --model-dir /path/to/Qwen3.5-4B
```

For a batch run across every model the producer has available, use
`oracle/bake_all.sh`:

```bash
QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
QWEN_4B_DIR=/models/Qwen3.5-4B \
GEMMA_E4B_DIR=/models/gemma-4-E4B \
./oracle/bake_all.sh --upload
```

Unset env vars are silently skipped; existing valid bakes are reused unless
you pass `--force`. See `./oracle/bake_all.sh --help` for the full flag list.

The uploader:

- Validates `{model-dir}/.supersonic/v1-int4-gptq/` contains a
  `manifest.json` + `weights.bin` with matching `format_version` /
  `converter_version` and at least one `Int4Quantized` tensor (catches the
  case where you point `--int4` at a BF16 bake directory).
- Bundles `config.json` + `tokenizer.json` (required) and any of the
  optional HF metadata files present in `--model-dir` (see asset layout
  above) under `hf/` in the tarball, so consumers don't need a separate HF
  checkpoint.
- Tars + zstd-compresses into `/tmp/supersonic-upload-*/…tar.zst`, teeing the
  compressed stream through SHA-256.
- Splits into `.partNN` at 1800 MiB if needed.
- Fetches the existing `bakes-index.json` (if any), upserts the
  `(model, variant)` entry, and uploads everything via `gh release upload
  --clobber`.
- Re-running with the same bake overwrites the asset bit-for-bit (idempotent).

## Consumer: downloading a bake

On the small machine, just run SuperSonic normally. When the local bake is
missing, the runner logs:

```
[fetch] downloading qwen3.5-4b int4-gptq from DeanoC/SuperSonic/bakes-v1
[fetch] resolving release index...
[fetch] downloading part 1/2 (1800 MiB)
[fetch]   10% (180 / 1800 MiB)
…
[fetch] verifying SHA-256...
[fetch] extracting tarball...
[fetch] done
[fetch] installed int4-gptq bake at /path/to/model/.supersonic/v1-int4-gptq
```

Flags:

- `--no-download`: never touch the network. For air-gapped machines. Fails
  with the existing "run bake_int4.py" error if the local bake is missing.
- `--download-bake`: force a re-download over an existing local bake. Useful
  to verify release content matches what you baked.
- `--bake-release <tag-or-url>`: override the release. Also readable from
  the `SUPERSONIC_BAKE_RELEASE` env var. Accepts either a bare tag
  (`bakes-v1`) or a full URL
  (`https://github.com/<owner>/<repo>/releases/tag/<tag>`).

### Atomicity

The downloader writes parts into
`{model_dir}/.supersonic/.bake-cache-<tag>/` with HTTP `Range:` resume and
per-part SHA-256 verification. It then extracts into a
`.partial-<tag>-<variant>-<pid>/` sibling directory and atomically renames
into the final bake dir on success. Interrupted downloads never leave a
half-installed bake.

### Concurrency

`{model_dir}/.supersonic/.lock` is held exclusively for the duration of both
bake and fetch. Two concurrent `supersonic` invocations will serialize on
this lock rather than race.

## Producer architecture: CUDA ↔ HIP

Bakes are raw bytes — no GPU bytecode, no arch-specific layout. A CUDA
machine can produce a bake that a HIP consumer will load verbatim (and
vice-versa). The bakers use `torch.device("cuda")` abstractly; real CUDA on
NVIDIA and ROCm's CUDA-shim on AMD both write the same `uint8` nibbles and
`bf16` scales/zeros.

Consumer integrity is guaranteed by:

- Every packed-INT4 weight is self-verified by the baker against its BF16
  source before it's written. A producer-side numerical glitch fails at the
  producer, never silently at a consumer.
- The downloader SHA-256-verifies every part plus the inner `manifest.json`,
  so any in-transit corruption is caught.

Caveats:

- GPTQ calibration is not bit-reproducible across producers. Two CUDA-made
  bakes of the same model — let alone one CUDA + one HIP bake — will differ
  at the byte level. We don't promise cross-producer reproducibility; we
  promise that whatever SHA-256 the release index names is exactly what the
  consumer receives. If you need bit-reproducibility, produce the bake
  yourself.
- The `producer` block in `bakes-index.json` (`host`, `system`, `python`) is
  informational only; nothing checks it.

## Trust model

v1 trust boundary: the repo owner's release. SuperSonic enforces:

- SHA-256 verification per part and across the concatenated stream.
- The inner `manifest.json`'s SHA-256 is cross-checked against
  `bakes-index.json` after extraction.
- URL allowlist: only `https://github.com/<repo>/releases/download/...` URLs
  are downloaded. Overrides are parsed, not redirected.
- Tarball contents: only `manifest.json` and `weights.bin` are permitted.
  Directory components, symlinks, and other entry types are rejected.

Not in v1: GPG/signify signatures on the index, reproducible bakes, content
addressing by manifest hash across releases. If you need stronger guarantees,
produce the bake yourself.

## Troubleshooting

**"release asset not found: no entry for model=… variant=…"** — The tag
exists but there's no matching bake. Publish one from a big-box producer.

**"version mismatch: got fmt=… cvt=…"** — Your SuperSonic's
`FORMAT_VERSION` / `CONVERTER_VERSION` don't match the release. Either
upgrade SuperSonic to match the release, or re-bake and re-upload to match
your runtime.

**"sha256 mismatch"** — Corrupt download (often a partial HTTP response).
The corrupt file is removed; rerun to re-fetch.

**Stalled downloads / slow networks** — Read timeout is 5 minutes per chunk.
For very slow links, rerun — parts resume via HTTP `Range:`.

**Want to bypass entirely** — Download the asset(s) manually with
`gh release download bakes-v1 -p 'qwen3.5-4b-*' -D /tmp/bake`, extract with
`zstd -d -c ...tar.zst | tar -x -C {model_dir}/.supersonic/v1-int4-gptq/`,
and run with `--no-download`.
