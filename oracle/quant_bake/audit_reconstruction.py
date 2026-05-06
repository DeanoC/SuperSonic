#!/usr/bin/env python3
"""Audit native INT4 bake reconstruction against source safetensors weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def bf16_bytes_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    f32 = (u16 << 16).view(np.float32)
    return f32.reshape(shape)


def unpack_native_int4(packed: np.ndarray, cols: int) -> np.ndarray:
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = packed >> 4
    return out[..., :cols]


class BakeReader:
    def __init__(self, bake_dir: Path):
        self.bake_dir = bake_dir
        self.manifest = json.loads((bake_dir / "manifest.json").read_text())
        self.entries = {e["name"]: e for e in self.manifest["tensors"]}
        self.weights = open(bake_dir / "weights.bin", "rb")

    def read_entry(self, name: str) -> tuple[dict[str, Any], bytes]:
        entry = self.entries[name]
        self.weights.seek(entry["offset"])
        return entry, self.weights.read(entry["byte_len"])

    def tensor_u8(self, name: str) -> np.ndarray:
        entry, raw = self.read_entry(name)
        return np.frombuffer(raw, dtype=np.uint8).reshape(entry["shape"])

    def tensor_bf16_f32(self, name: str) -> np.ndarray:
        entry, raw = self.read_entry(name)
        return bf16_bytes_to_f32(raw, entry["shape"])


class SafeTensorSource:
    def __init__(self, model_dir: Path):
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            idx = json.loads(index.read_text())
            self.weight_map = dict(idx["weight_map"])
        else:
            self.weight_map = {}
            for path in sorted(model_dir.glob("model*.safetensors")):
                with safe_open(str(path), framework="pt") as f:
                    for key in f.keys():
                        self.weight_map[key] = path.name
        self.model_dir = model_dir
        self.handles: dict[str, Any] = {}

    def get(self, name: str) -> np.ndarray | None:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        handle = self.handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self.handles[shard] = handle
        tensor = handle.get_tensor(name)
        return tensor.detach().to(dtype=torch.float32).cpu().numpy()

    def get_with_aliases(self, name: str, shape: list[int]) -> tuple[np.ndarray | None, str | None]:
        aliases = [name]
        if name == "lm_head.weight":
            aliases.extend([
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ])
        for source_name in aliases:
            tensor = self.get(source_name)
            if tensor is None:
                continue
            if tuple(tensor.shape) == tuple(shape):
                return tensor, source_name
        return None, None


def dequantize(entry: dict[str, Any], packed: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    shape = entry["shape"]
    if len(shape) == 2:
        rows, packed_cols = shape
        cols = packed_cols * 2
        q = unpack_native_int4(packed, cols).astype(np.float32)
        group_size = 128
        row_gr = np.arange(rows) // group_size
        col_gc = np.arange(cols) // group_size
        sc = scale[row_gr[:, None], col_gc[None, :]]
        zf = zero[row_gr[:, None], col_gc[None, :]]
        return q * sc - zf * sc
    if len(shape) == 3:
        experts, rows, packed_cols = shape
        cols = packed_cols * 2
        q = unpack_native_int4(packed, cols).astype(np.float32)
        group_size = 128
        row_gr = np.arange(rows) // group_size
        col_gc = np.arange(cols) // group_size
        sc = scale[:, row_gr][:, :, col_gc]
        zf = zero[:, row_gr][:, :, col_gc]
        return q * sc - zf * sc
    raise ValueError(f"unsupported INT4 shape {shape}")


def metrics(recon: np.ndarray, source: np.ndarray) -> dict[str, float]:
    r = recon.reshape(-1).astype(np.float64)
    s = source.reshape(-1).astype(np.float64)
    diff = r - s
    denom = float(np.linalg.norm(s))
    rel_l2 = float(np.linalg.norm(diff) / denom) if denom else float("nan")
    cos_denom = float(np.linalg.norm(r) * np.linalg.norm(s))
    cos = float(np.dot(r, s) / cos_denom) if cos_denom else float("nan")
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": rel_l2,
        "cosine": cos,
    }


def audit(model_dir: Path, bake_dir: Path, max_tensors: int | None) -> dict[str, Any]:
    reader = BakeReader(bake_dir)
    source = SafeTensorSource(model_dir)
    rows = []
    skipped = []
    int4_entries = [
        e for e in reader.manifest["tensors"]
        if e.get("layout") == "Int4Quantized"
    ]
    if max_tensors is not None:
        int4_entries = int4_entries[:max_tensors]
    for entry in int4_entries:
        name = entry["name"]
        expected_shape = list(entry["shape"])
        if expected_shape:
            expected_shape[-1] *= 2
        src, source_name = source.get_with_aliases(name, expected_shape)
        if src is None:
            skipped.append({"name": name, "reason": "missing source"})
            continue
        packed = reader.tensor_u8(name)
        scale = reader.tensor_bf16_f32(f"{name}_int4_scale")
        zero = reader.tensor_bf16_f32(f"{name}_int4_zero")
        recon = dequantize(entry, packed, scale, zero)
        awq_name = f"{name}_awq_inv_scale"
        if awq_name in reader.entries:
            awq_inv = reader.tensor_bf16_f32(awq_name)
            if recon.ndim == 2:
                recon = recon * awq_inv.reshape(1, -1)
            elif recon.ndim == 3:
                recon = recon * awq_inv.reshape(1, 1, -1)
        if tuple(recon.shape) != tuple(src.shape):
            skipped.append({
                "name": name,
                "reason": f"shape mismatch recon={recon.shape} source={src.shape}",
            })
            continue
        row = {"name": name, "source_name": source_name or name, "shape": list(src.shape)}
        row.update(metrics(recon, src))
        rows.append(row)
    worst_rel = sorted(rows, key=lambda r: r["rel_l2"], reverse=True)[:10]
    weighted_mse_num = 0.0
    weighted_mse_den = 0
    rel_l2_vals = []
    for row in rows:
        n = math.prod(row["shape"])
        weighted_mse_num += row["mse"] * n
        weighted_mse_den += n
        rel_l2_vals.append(row["rel_l2"])
    return {
        "bake_dir": str(bake_dir),
        "quant_profile": reader.manifest.get("quant_profile"),
        "quant_method": reader.manifest.get("quant_method"),
        "tensor_count": len(rows),
        "skipped": skipped,
        "summary": {
            "weighted_mse": weighted_mse_num / weighted_mse_den if weighted_mse_den else None,
            "mean_rel_l2": float(np.mean(rel_l2_vals)) if rel_l2_vals else None,
            "max_rel_l2": max(rel_l2_vals) if rel_l2_vals else None,
        },
        "worst_rel_l2": worst_rel,
        "rows": rows,
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        f"# Reconstruction Audit: {Path(result['bake_dir']).name}",
        "",
        f"- profile: `{result.get('quant_profile')}`",
        f"- tensors audited: {result['tensor_count']}",
        f"- weighted MSE: {result['summary']['weighted_mse']:.6e}",
        f"- mean rel L2: {result['summary']['mean_rel_l2']:.6f}",
        f"- max rel L2: {result['summary']['max_rel_l2']:.6f}",
        "",
        "| Tensor | rel L2 | cosine | MSE | max abs |",
        "|:--|--:|--:|--:|--:|",
    ]
    for row in result["worst_rel_l2"]:
        lines.append(
            f"| `{row['name']}` | {row['rel_l2']:.6f} | {row['cosine']:.6f} | "
            f"{row['mse']:.6e} | {row['max_abs']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bake-dir", type=Path, required=True)
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.model_dir, args.bake_dir, args.max_tensors)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    args.out_md.write_text(render_md(result))
    print(render_md(result))
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")


if __name__ == "__main__":
    main()
