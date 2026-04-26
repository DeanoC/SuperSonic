#!/usr/bin/env python3
"""
Upload a SuperSonic bake package to a GitHub Release so smaller machines can
auto-download it. Reads `{model_dir}/.supersonic/v{FORMAT_VERSION}[-variant]/`,
streams tar + zstd, optionally splits into parts at 1800 MiB, computes SHA-256,
upserts a single `bakes-index.json` asset, and uploads everything via `gh`.

Typical use on a big-box producer:
    python oracle/upload_bake.py --model qwen3.5-4b --int4 --model-dir /path/to/Qwen3.5-4B
    python oracle/upload_bake.py --model gemma4-e4b --int4 --model-dir /path/to/gemma-4-E4B

Dry-run leaves the tarball in /tmp/ and prints the `gh` commands without
uploading.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Keep in lockstep with crates/model-store/src/manifest.rs
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

INDEX_ASSET = "bakes-index.json"
INDEX_SCHEMA_VERSION = 1

PART_SIZE_BYTES = 1800 * 1024 * 1024  # < GitHub's 2 GiB per-asset cap
ZSTD_LEVEL = 19

# Small HF metadata files bundled into every tarball under `hf/` so a
# consumer with an empty --model-dir still has config + tokenizer after the
# fetch. config.json and tokenizer.json are required; the rest are pulled in
# when present.
REQUIRED_HF_FILES = ["config.json", "tokenizer.json"]
OPTIONAL_HF_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "tokenizer.model",          # SentencePiece models (Gemma etc.)
    "preprocessor_config.json",
    "processor_config.json",
]

KNOWN_MODELS = {
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
    "qwen3.6-27b", "qwen3.6-35b-a3b",
    "gemma4-e2b", "gemma4-e4b",
    "phi4-mini",
}

VARIANT_DIR_SUFFIX = {
    "bf16": "",
    "fp8-native": "-fp8",
    "int4-gptq": "-int4-gptq",
    "q4km": "-q4km",
}

FAMILY_FOR = {
    "qwen3.5-0.8b": "qwen35",
    "qwen3.5-2b": "qwen35",
    "qwen3.5-4b": "qwen35",
    "qwen3.5-9b": "qwen35",
    "qwen3.6-27b": "qwen35",
    "qwen3.6-35b-a3b": "qwen36-moe",
    "gemma4-e2b": "gemma4",
    "gemma4-e4b": "gemma4",
    "phi4-mini": "phi4",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def bake_dir_for(model_dir: Path, variant: str) -> Path:
    suffix = VARIANT_DIR_SUFFIX[variant]
    return model_dir / ".supersonic" / f"v{FORMAT_VERSION}{suffix}"


def asset_basename(model: str, variant: str) -> str:
    return f"{model}-{variant}-fmt{FORMAT_VERSION}-cvt{CONVERTER_VERSION}.tar.zst"


def gh(args: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    log(f"  $ gh {' '.join(args)}")
    return subprocess.run(
        ["gh", *args],
        check=check,
        text=True,
        capture_output=capture,
    )


def validate_bake(bake_dir: Path, variant: str) -> dict:
    manifest_path = bake_dir / "manifest.json"
    weights_path = bake_dir / "weights.bin"
    if not manifest_path.is_file():
        sys.exit(f"error: no manifest.json at {manifest_path}")
    if not weights_path.is_file():
        sys.exit(f"error: no weights.bin at {weights_path}")
    manifest = json.loads(manifest_path.read_text())
    fmt = manifest.get("format_version")
    cvt = manifest.get("converter_version")
    if fmt != FORMAT_VERSION or cvt != CONVERTER_VERSION:
        sys.exit(
            f"error: bake at {bake_dir} has format_version={fmt} converter_version={cvt} "
            f"but uploader expects {FORMAT_VERSION}/{CONVERTER_VERSION}. "
            f"Update FORMAT_VERSION/CONVERTER_VERSION in upload_bake.py or re-bake."
        )
    # variant-specific sanity: the Int4 bake must contain Int4Quantized layouts,
    # the Fp8Native bake must contain Fp8Native layouts. This catches the case
    # where a user points --int4 at a BF16 bake directory.
    layouts = {t.get("layout") for t in manifest.get("tensors", [])}
    lowbit_layouts = {"Int4Quantized", "GgmlQ4K", "GgmlQ5K", "GgmlQ6K"}
    if variant in ("int4-gptq", "q4km") and not (layouts & lowbit_layouts):
        sys.exit(f"error: {bake_dir} has no Int4Quantized tensors — wrong variant?")
    if variant == "q4km" and manifest.get("quant_profile") != "q4km-ggml-v1":
        sys.exit(f"error: {bake_dir} is missing quant_profile=q4km-ggml-v1")
    if variant == "fp8-native" and "Fp8Native" not in layouts:
        sys.exit(f"error: {bake_dir} has no Fp8Native tensors — wrong variant?")
    return manifest


def collect_hf_files(model_dir: Path) -> list[Path]:
    """Return the set of HF metadata files to bundle. Errors if a required
    file is missing; optional files are included silently when present."""
    found: list[Path] = []
    missing_required: list[str] = []
    for name in REQUIRED_HF_FILES:
        p = model_dir / name
        if p.is_file():
            found.append(p)
        else:
            missing_required.append(name)
    if missing_required:
        sys.exit(
            f"error: required HF metadata missing in {model_dir}: "
            f"{', '.join(missing_required)}. The consumer needs these to load "
            f"the model; ensure the HF checkpoint is complete before baking."
        )
    for name in OPTIONAL_HF_FILES:
        p = model_dir / name
        if p.is_file():
            found.append(p)
    return found

class MultipartHashingWriter:
    """Stream compressed output directly into GitHub-sized asset parts."""

    def __init__(self, out_dir: Path, base_name: str, part_size: int) -> None:
        self.out_dir = out_dir
        self.base_name = base_name
        self.part_size = part_size
        self.sha = hashlib.sha256()
        self.total_bytes = 0
        self.parts: list[Path] = []
        self.part_metas: list[dict] = []
        self._idx = 0
        self._size = 0
        self._part_sha = hashlib.sha256()
        self._path = self._part_path(self._idx)
        self._raw = open(self._path, "wb")

    def _part_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.base_name}.part{idx:02d}"

    def _close_part(self) -> None:
        if self._raw.closed:
            return
        self._raw.close()
        size = self._path.stat().st_size
        self.parts.append(self._path)
        self.part_metas.append({
            "name": self._path.name,
            "bytes": size,
            "sha256": self._part_sha.hexdigest(),
        })

    def _open_next_part(self) -> None:
        self._idx += 1
        self._size = 0
        self._part_sha = hashlib.sha256()
        self._path = self._part_path(self._idx)
        self._raw = open(self._path, "wb")

    def write(self, b) -> int:
        view = memoryview(b)
        written = 0
        while view:
            if self._size == self.part_size:
                self._close_part()
                self._open_next_part()
            take = min(len(view), self.part_size - self._size)
            chunk = view[:take]
            self._raw.write(chunk)
            self.sha.update(chunk)
            self._part_sha.update(chunk)
            self._size += take
            self.total_bytes += take
            written += take
            view = view[take:]
        return written

    def finish(self) -> tuple[list[Path], list[dict] | None, str, int]:
        self._close_part()
        if len(self.parts) == 1:
            final_path = self.out_dir / self.base_name
            self.parts[0].rename(final_path)
            return [final_path], None, self.sha.hexdigest(), self.total_bytes
        return self.parts, self.part_metas, self.sha.hexdigest(), self.total_bytes


def stream_tar_zst(
    bake_dir: Path,
    hf_files: list[Path],
    out_dir: Path,
    base_name: str,
) -> tuple[list[Path], list[dict] | None, str, int, str]:
    """Write tar.zst of manifest.json + weights.bin + hf/* directly into upload-sized parts."""
    try:
        import zstandard
    except ImportError:
        sys.exit(
            "error: zstandard not installed. Run:\n"
            "  pip install -r oracle/requirements-upload.txt"
        )

    manifest_sha = hashlib.sha256()

    manifest_bytes = (bake_dir / "manifest.json").read_bytes()
    manifest_sha.update(manifest_bytes)

    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
    writer = MultipartHashingWriter(out_dir, base_name, PART_SIZE_BYTES)
    with cctx.stream_writer(writer) as zout:
        with tarfile.open(fileobj=zout, mode="w|") as tar:
            # manifest.json from bytes we already have
            ti = tarfile.TarInfo(name="manifest.json")
            ti.size = len(manifest_bytes)
            ti.mtime = 0
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(manifest_bytes))
            # weights.bin streamed from disk
            wb = bake_dir / "weights.bin"
            ti = tarfile.TarInfo(name="weights.bin")
            ti.size = wb.stat().st_size
            ti.mtime = 0
            ti.mode = 0o644
            with open(wb, "rb") as f:
                tar.addfile(ti, f)
            # HF metadata under hf/ prefix
            for hf_path in hf_files:
                ti = tarfile.TarInfo(name=f"hf/{hf_path.name}")
                ti.size = hf_path.stat().st_size
                ti.mtime = 0
                ti.mode = 0o644
                with open(hf_path, "rb") as f:
                    tar.addfile(ti, f)
    parts, part_metas, sha256_hex, total_bytes = writer.finish()
    return parts, part_metas, sha256_hex, total_bytes, manifest_sha.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_existing_index(tag: str) -> dict | None:
    """Try to download bakes-index.json from an existing release. Returns dict, or None if absent."""
    res = subprocess.run(
        ["gh", "release", "download", tag, "-p", INDEX_ASSET, "-O", "-"],
        text=True, capture_output=True,
    )
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return None


def fresh_index() -> dict:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "generated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "producer": {
            "host": platform.node(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "bakes": [],
    }


def upsert_bake(index: dict, entry: dict) -> None:
    key = (entry["model"], entry["variant"])
    index["bakes"] = [
        b for b in index.get("bakes", [])
        if (b.get("model"), b.get("variant")) != key
    ]
    index["bakes"].append(entry)
    index["bakes"].sort(key=lambda b: (b.get("model", ""), b.get("variant", "")))
    index["generated_at"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def prune_old_format_assets(index: dict) -> None:
    keep = []
    for b in index.get("bakes", []):
        if b.get("format_version") == FORMAT_VERSION and b.get("converter_version") == CONVERTER_VERSION:
            keep.append(b)
    index["bakes"] = keep


def ensure_release(tag: str, dry_run: bool) -> None:
    res = subprocess.run(
        ["gh", "release", "view", tag],
        text=True, capture_output=True,
    )
    if res.returncode == 0:
        return
    notes = (
        f"SuperSonic pre-baked weights for format_version={FORMAT_VERSION} "
        f"converter_version={CONVERTER_VERSION}.\n\n"
        f"See docs/bake-distribution.md for the asset layout and trust model."
    )
    log(f"[release] creating release {tag}")
    if dry_run:
        log(f"  (dry-run) would: gh release create {tag} --title 'Bakes v{FORMAT_VERSION}' --notes '...'")
        return
    subprocess.run(
        ["gh", "release", "create", tag, "--title", f"Bakes v{FORMAT_VERSION}", "--notes", notes],
        check=True,
    )


def upload_asset(tag: str, path: Path, dry_run: bool) -> None:
    log(f"[upload] {path.name} ({path.stat().st_size:,} bytes)")
    if dry_run:
        log(f"  (dry-run) would: gh release upload {tag} {path} --clobber")
        return
    subprocess.run(
        ["gh", "release", "upload", tag, str(path), "--clobber"],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload SuperSonic bake to GitHub release")
    ap.add_argument("--model", required=True, choices=sorted(KNOWN_MODELS))
    ap.add_argument("--model-dir", required=True, type=Path)
    variant_group = ap.add_mutually_exclusive_group(required=True)
    variant_group.add_argument("--bf16", action="store_true")
    variant_group.add_argument("--fp8-native", action="store_true", dest="fp8_native")
    variant_group.add_argument("--int4", action="store_true", help="INT4 GPTQ bake")
    variant_group.add_argument("--q4km", action="store_true", help="GGUF q4km raw GGML K-block bake")
    ap.add_argument("--tag", default=f"bakes-v{FORMAT_VERSION}")
    ap.add_argument("--dry-run", action="store_true", help="Print gh commands, don't upload")
    ap.add_argument("--prune-old-cvt", action="store_true",
                    help="Remove index entries with older converter_version")
    args = ap.parse_args()

    if args.bf16:
        variant = "bf16"
    elif args.fp8_native:
        variant = "fp8-native"
    elif args.q4km:
        variant = "q4km"
    else:
        variant = "int4-gptq"

    bake_dir = bake_dir_for(args.model_dir, variant)
    log(f"[bake] reading {bake_dir}")
    manifest = validate_bake(bake_dir, variant)
    if manifest.get("model_family") != FAMILY_FOR[args.model]:
        sys.exit(
            f"error: manifest says model_family={manifest.get('model_family')} "
            f"but --model={args.model} implies family {FAMILY_FOR[args.model]}"
        )

    hf_files = collect_hf_files(args.model_dir)
    log(f"[bake] bundling {len(hf_files)} HF metadata files: {[p.name for p in hf_files]}")

    base = asset_basename(args.model, variant)
    workdir = Path(tempfile.mkdtemp(prefix="supersonic-upload-"))
    try:
        log(f"[compress] tar+zstd(level={ZSTD_LEVEL}) → {workdir / base}")
        parts, part_metas, sha256_hex, total_bytes, inner_manifest_sha = stream_tar_zst(
            bake_dir, hf_files, workdir, base
        )
        log(f"[compress] {total_bytes:,} bytes, sha256={sha256_hex[:16]}...")
        if part_metas is not None:
            log(f"[split] wrote {len(parts)} parts")

        # Size of the uncompressed tar content (weights.bin + manifest.json + hf/*)
        uncompressed_bytes = (
            (bake_dir / "weights.bin").stat().st_size
            + (bake_dir / "manifest.json").stat().st_size
            + sum(p.stat().st_size for p in hf_files)
        )

        entry = {
            "model": args.model,
            "variant": variant,
            "model_family": manifest["model_family"],
            "asset": base,
            "parts": part_metas,
            "sha256": sha256_hex,
            "compressed_bytes": total_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "bake_manifest_sha256": inner_manifest_sha,
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
        }

        ensure_release(args.tag, args.dry_run)

        log(f"[index] fetching existing {INDEX_ASSET}")
        index = fetch_existing_index(args.tag) or fresh_index()
        upsert_bake(index, entry)
        if args.prune_old_cvt:
            prune_old_format_assets(index)

        index_path = workdir / INDEX_ASSET
        index_path.write_text(json.dumps(index, indent=2) + "\n")

        for p in parts:
            upload_asset(args.tag, p, args.dry_run)
        upload_asset(args.tag, index_path, args.dry_run)

        if args.dry_run:
            log(f"[dry-run] artifacts kept in {workdir} — inspect and delete manually")
        else:
            log("[done] upload complete")
    finally:
        if not args.dry_run:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
