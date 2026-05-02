#!/usr/bin/env python3
"""
Per-block FP8-E4M3 calibration bake for SuperSonic — Qwen3.5.

Quantizes BF16 projection weights to FP8-E4M3-FN with per-block (128×128)
absmax scales, writing a SuperSonic baked package at
  {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8/

Mirrors the Qwen 3.6 native FP8 layout: each `*_proj.weight` becomes
  - `name`            : u8  [rows, cols]            FP8-E4M3 bytes (Fp8Native)
  - `name_scale_inv`  : bf16 [rows/128, cols/128]   per-block scale (Raw)

so the existing Rust runtime FP8-dequant path (used today for Qwen 3.6
FP8-native checkpoints) reads it without any new code on the C++ / kernel
side. Non-projection tensors (norms, embeddings, conv1d, dt_bias, A_log)
get the same layout transforms the Rust BF16 baker applies — squeezed
conv1d, reshaped dt_bias, A_log → exp(A_log) → BF16.

Usage:
    python3 oracle/bake_fp8.py --model-dir /path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import struct
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
except ModuleNotFoundError:
    torch = None
    safe_open = None

# -- constants mirrored from crates/model-store/src/manifest.rs --
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

LAYOUT_RAW = "Raw"
LAYOUT_FP8_NATIVE = "Fp8Native"
LAYOUT_DEPTHWISE_CONV_SQUEEZED = "DepthwiseConvSqueezed"
LAYOUT_HEAD_BIAS_RESHAPED = "HeadBiasReshaped"
LAYOUT_HEAD_EXP_RESHAPED = "HeadExpReshaped"

ALIGN = 4096
BLOCK_SIZE = 128
MAX_E4M3 = 448.0  # finite max of float8_e4m3fn

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
QWEN36_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(?:_scale_inv)?$"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------
def is_fp8_quant_target(
    name: str, shape: tuple[int, ...], block_size: int = BLOCK_SIZE
) -> bool:
    """True if this 2D weight should be quantized to FP8.

    Match every per-layer projection by looking for `_proj` anywhere in the
    leaf module name. Covers:

      * `gate_proj` / `up_proj` / `down_proj`     (MLP)
      * `q_proj` / `k_proj` / `v_proj` / `o_proj` (full attention)
      * `out_proj`                                (linear-attn output)
      * `in_proj_qkv` / `in_proj_z`               (linear-attn input)
      * `in_proj_a` / `in_proj_b`                 (linear-attn input;
                                                   filtered by the shape
                                                   divisibility check below
                                                   — Qwen 3.5 stores those
                                                   at `[16, hidden]`, 16 < 128)

    The previous check `"_proj.weight" in name` matched only names where
    `_proj` was the immediate parent of `.weight`, so every linear-attention
    `in_proj_qkv.weight` / `in_proj_z.weight` was silently left BF16,
    leaving the bake only partially quantized for the linear layers (about
    a third of the tensors in a Qwen 3.5 hybrid model).

    Skip everything else: norms, embeddings, conv1d / dt_bias / A_log
    transforms, layer scalars, rotary buffers. **Also skip `lm_head.weight`**
    — the runtime does not currently consume `lm_head_scale` (decode lm_head
    matmul is BF16-only across `decode_engine.rs` and `compute_logits_for_range`),
    so quantizing it would feed FP8 bytes into a BF16 matmul and either
    corrupt logits or page-fault. Tied-embedding checkpoints (most Qwen3.5
    sizes) don't have a standalone `lm_head.weight` so this never mattered;
    Qwen3.5-9B does, and was producing GPU memory faults until this exclusion
    landed.
    """
    if len(shape) != 2:
        return False
    if not name.endswith(".weight"):
        return False
    # Embedding / lm_head tables — skip until runtime lm_head FP8 support
    # is wired (runtime currently only knows BF16 lm_head).
    if "embed_tokens" in name or name == "lm_head.weight":
        return False
    # Norms (caught by 1D check above mostly, but be explicit).
    if "layernorm" in name or name.endswith(".norm.weight"):
        return False
    if "_proj" not in name:
        return False
    return shape[0] % block_size == 0 and shape[1] % block_size == 0


def detect_model_family(config_path: Path) -> str:
    """Map a HF config to the SuperSonic model_family name in the manifest.

    Mirrors `FAMILY_FOR` in oracle/upload_bake.py:
      * qwen36-moe → Qwen 3.6 MoE (qwen3.6-35b-a3b)
      * qwen35     → Qwen 3.5 dense and Qwen 3.6 27B (uses Qwen35 hybrid kernel)
    Anything else falls back to "qwen35", which is fine for the Qwen3.5
    family the script targets; users producing non-Qwen bakes should pass
    `--model-family` explicitly.
    """
    cfg = json.load(open(config_path))
    archs: list[str] = []
    if isinstance(cfg.get("architectures"), list):
        archs.extend(str(a) for a in cfg["architectures"])
    text_cfg = cfg.get("text_config") or {}
    if isinstance(text_cfg.get("architectures"), list):
        archs.extend(str(a) for a in text_cfg["architectures"])
    model_type = str(text_cfg.get("model_type", cfg.get("model_type", "")))
    if any("Moe" in a or "MoE" in a for a in archs) or "moe" in model_type.lower():
        return "qwen36-moe"
    return "qwen35"


def linear_attn_layer_indices(config_path: Path) -> set[int]:
    """Return the set of linear-attention layer indices for Qwen3.5.

    Qwen3.5 alternates: full attention every 4th layer starting at index 3.
    Reads the model's config.json to compute this; falls back to the
    standard `(idx + 1) % 4 != 0` rule if the config doesn't carry the
    attention pattern explicitly.
    """
    cfg = json.load(open(config_path))
    text = cfg.get("text_config", cfg)
    n = int(text["num_hidden_layers"])
    pattern = text.get("attention_pattern") or text.get("layer_types")
    if isinstance(pattern, list) and len(pattern) == n:
        return {
            i for i, t in enumerate(pattern)
            if "linear" in str(t).lower() or "ssm" in str(t).lower()
        }
    full_attn_period = int(text.get("full_attn_period", 4))
    return {i for i in range(n) if (i + 1) % full_attn_period != 0}


def classify_layout(name: str, shape: tuple[int, ...], linear_layer_idx: set[int]) -> str:
    """Return the layout tag a tensor should have in the bake.

    Mirrors `classify_tensor` in crates/model-store/src/baker.rs. Only
    linear-attention layers' conv1d/dt_bias/A_log get reshape transforms;
    everything else is `Raw`.
    """
    m = LAYER_RE.match(name)
    if not m:
        return LAYOUT_RAW
    idx = int(m.group(1))
    if idx not in linear_layer_idx:
        return LAYOUT_RAW
    if name.endswith(".conv1d.weight") and len(shape) == 3 and shape[1] == 1:
        return LAYOUT_DEPTHWISE_CONV_SQUEEZED
    if name.endswith(".dt_bias") and len(shape) == 1:
        return LAYOUT_HEAD_BIAS_RESHAPED
    if name.endswith(".A_log") and len(shape) == 1:
        return LAYOUT_HEAD_EXP_RESHAPED
    return LAYOUT_RAW


# ---------------------------------------------------------------------------
# FP8 quantization
# ---------------------------------------------------------------------------
def quantize_bf16_to_fp8(
    weight: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [rows, cols] → (FP8-E4M3 [rows, cols] uint8, scale [rows/B, cols/B] BF16).

    Per-block (B×B) absmax: scale = absmax / MAX_E4M3, then
    quantized_byte = round_to_e4m3(value / scale). The kernel reads back as
    `value ≈ fp8_to_f32(byte) * bf16_to_f32(scale)`.
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    rows, cols = weight.shape
    assert rows % block_size == 0 and cols % block_size == 0, \
        f"shape {tuple(weight.shape)} not divisible by {block_size}"

    w_f32 = weight.float()
    # Reshape into [rows/B, B, cols/B, B] and absmax over the inner
    # (1, 3) dims to get [rows/B, cols/B].
    sr = rows // block_size
    sc = cols // block_size
    blocks = w_f32.view(sr, block_size, sc, block_size).permute(0, 2, 1, 3).contiguous()
    # blocks: [sr, sc, block_size, block_size]
    absmax = blocks.abs().amax(dim=(2, 3))  # [sr, sc]
    # Avoid /0 for all-zero blocks.
    scale = (absmax / MAX_E4M3).clamp_min(torch.finfo(torch.float32).tiny)
    # Per-element scale matrix [rows, cols] via repeat-interleave.
    scale_full = (
        scale.repeat_interleave(block_size, dim=0)
              .repeat_interleave(block_size, dim=1)
    )
    # Divide by scale, cast to E4M3 (rounds + clamps to representable range),
    # reinterpret bytes as uint8.
    q_e4m3 = (w_f32 / scale_full).to(torch.float8_e4m3fn)
    q_bytes = q_e4m3.view(torch.uint8)
    return q_bytes.contiguous(), scale.to(torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------
# Layout transforms (1:1 with crates/model-store/src/transforms.rs)
# ---------------------------------------------------------------------------
def apply_squeeze_dim1(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 3 and t.shape[1] == 1, \
        f"squeeze_dim1: expected [C, 1, K], got {tuple(t.shape)}"
    return t.squeeze(1).contiguous()


def apply_head_bias_reshape(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 1, f"head_bias_reshape: expected [H], got {tuple(t.shape)}"
    return t.view(1, 1, -1).contiguous()


def apply_a_log_to_exp_bf16(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 1, f"a_log_to_exp_bf16: expected [H], got {tuple(t.shape)}"
    return t.float().exp().to(torch.bfloat16).view(1, 1, -1).contiguous()


# ---------------------------------------------------------------------------
# Manifest dtype names (mirror dtype_name in baker.rs)
# ---------------------------------------------------------------------------
def torch_dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float32:
        return "f32"
    if dt == torch.float16:
        return "f16"
    if dt == torch.uint8:
        return "u8"
    if dt == torch.float8_e4m3fn:
        return "f8_e4m3"
    raise ValueError(f"unsupported dtype: {dt}")


def tensor_to_bytes(t: torch.Tensor, dtype_str: str) -> bytes:
    """Return the LE byte representation of `t`, validating the requested dtype.

    f8_e4m3 accepts either a `torch.uint8` reinterpret (what `quantize_bf16_to_fp8`
    produces) or a native `torch.float8_e4m3fn` tensor — viewed as uint8 before
    `numpy.tobytes()` because numpy doesn't know about float8.
    """
    if dtype_str == "f8_e4m3":
        if t.dtype == torch.float8_e4m3fn:
            t = t.view(torch.uint8)
        elif t.dtype != torch.uint8:
            raise AssertionError(
                f"f8_e4m3 must be uint8 or float8_e4m3fn, got {t.dtype}"
            )
        return t.contiguous().cpu().numpy().tobytes()
    expected = {"bf16": torch.bfloat16, "f32": torch.float32, "f16": torch.float16, "u8": torch.uint8}[dtype_str]
    if t.dtype != expected:
        t = t.to(expected)
    return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------
class StreamingTensorWriter:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.weights_path = out_dir / "weights.bin"
        self.f = open(self.weights_path, "wb")
        self.entries: list[dict] = []
        self.cursor = 0

    def add(self, name: str, data: bytes, shape: list[int], dtype_str: str, layout: str) -> None:
        offset = align_up(self.cursor)
        if offset > self.cursor:
            self.f.write(b"\x00" * (offset - self.cursor))
        self.f.write(data)
        self.entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "layout": layout,
            "offset": offset,
            "byte_len": len(data),
        })
        self.cursor = offset + len(data)

    def close(self, model_family: str = "qwen35") -> None:
        self.f.close()
        sorted_entries = sorted(self.entries, key=lambda e: e["name"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": model_family,
            "tensors": sorted_entries,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[bake-fp8] wrote {self.cursor / (1024 * 1024):.1f} MiB to {self.weights_path}")


class RawSafeTensorIndex:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        weight_map_path = model_dir / "model.safetensors.index.json"
        if weight_map_path.exists():
            wm = json.load(open(weight_map_path)).get("weight_map", {})
            self.weight_map = {str(k): str(v) for k, v in wm.items()}
            shard_names = sorted(set(self.weight_map.values()))
        else:
            shard_names = [p.name for p in sorted(model_dir.glob("*.safetensors"))]
            self.weight_map = {}
        if not shard_names:
            raise SystemExit(f"no .safetensors found in {model_dir}")

        self.shards: dict[str, tuple[int, dict[str, Any]]] = {}
        for shard_name in shard_names:
            path = model_dir / shard_name
            with path.open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            self.shards[shard_name] = (8 + header_len, tensors)
            if not self.weight_map:
                for k in tensors:
                    self.weight_map[k] = shard_name

    def keys(self) -> list[str]:
        return list(self.weight_map)

    def meta(self, name: str) -> dict[str, Any]:
        shard = self.weight_map[name]
        return self.shards[shard][1][name]

    def raw_bytes(self, name: str) -> bytes:
        shard = self.weight_map[name]
        data_start, tensors = self.shards[shard]
        meta = tensors[name]
        start, end = meta["data_offsets"]
        with (self.model_dir / shard).open("rb") as f:
            f.seek(data_start + start)
            return f.read(end - start)


def _bf16_to_f32_bits(u16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def _f32_to_bf16_bytes(x: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def _a_log_raw_to_exp_bf16(raw: bytes, dtype: str) -> bytes:
    out = bytearray()
    if dtype == "BF16":
        for i in range(0, len(raw), 2):
            v = _bf16_to_f32_bits(struct.unpack_from("<H", raw, i)[0])
            out.extend(_f32_to_bf16_bytes(math.exp(v)))
        return bytes(out)
    if dtype == "F32":
        for i in range(0, len(raw), 4):
            v = struct.unpack_from("<f", raw, i)[0]
            out.extend(_f32_to_bf16_bytes(math.exp(v)))
        return bytes(out)
    raise SystemExit(f"A_log raw fallback supports BF16/F32 only, got {dtype}")


def _safetensors_dtype_to_manifest(dtype: str) -> str:
    return {
        "BF16": "bf16",
        "F32": "f32",
        "F16": "f16",
        "U8": "u8",
        "F8_E4M3": "f8_e4m3",
    }[dtype]


def bake_qwen36_fp8_raw(model_dir: Path, out_dir: Path, block_size: int) -> None:
    """Dependency-free FP8-native bake for Qwen3.6-MoE checkpoints."""
    if block_size != BLOCK_SIZE:
        raise SystemExit("raw qwen36 FP8 bake currently requires block_size=128")
    config_path = model_dir / "config.json"
    model_family = detect_model_family(config_path)
    if model_family != "qwen36-moe":
        raise SystemExit("raw fallback only supports qwen36-moe FP8 checkpoints")
    linear_idx = linear_attn_layer_indices(config_path)
    raw = RawSafeTensorIndex(model_dir)
    keys = raw.keys()
    log(f"[bake-fp8/raw] {len(set(raw.weight_map.values()))} shard(s), {len(keys)} tensors")

    weight_prefix = "model.language_model"
    eligible = sorted(n for n in keys if n.startswith(f"{weight_prefix}.") or n == "lm_head.weight")
    writer = StreamingTensorWriter(out_dir)

    per_layer: dict[int, set[int]] = {}
    skip_names: set[str] = set()
    for name in keys:
        m = QWEN36_EXPERT_RE.match(name)
        if not m:
            continue
        layer = int(m.group(1))
        expert = int(m.group(2))
        per_layer.setdefault(layer, set()).add(expert)
        skip_names.add(name)

    log(f"[bake-fp8/raw] qwen36-moe fused experts: {len(per_layer)} layer(s)")
    for layer in sorted(per_layer):
        experts = sorted(per_layer[layer])
        base = f"{weight_prefix}.layers.{layer}.mlp.experts"
        gate_up = bytearray()
        gate_up_scale = bytearray()
        down = bytearray()
        down_scale = bytearray()
        gate_shape = up_shape = down_shape = None
        gate_scale_shape = up_scale_shape = down_scale_shape = None
        for expert in experts:
            eb = f"{base}.{expert}"
            gate_n = f"{eb}.gate_proj.weight"
            up_n = f"{eb}.up_proj.weight"
            down_n = f"{eb}.down_proj.weight"
            gate_m = raw.meta(gate_n)
            up_m = raw.meta(up_n)
            down_m = raw.meta(down_n)
            if gate_m["dtype"] != "F8_E4M3" or up_m["dtype"] != "F8_E4M3" or down_m["dtype"] != "F8_E4M3":
                raise SystemExit(f"qwen36 raw FP8 experts must be F8_E4M3 at layer={layer} expert={expert}")
            gate_shape, up_shape, down_shape = gate_m["shape"], up_m["shape"], down_m["shape"]
            gate_up.extend(raw.raw_bytes(gate_n))
            gate_up.extend(raw.raw_bytes(up_n))
            down.extend(raw.raw_bytes(down_n))
            gate_s = f"{gate_n}_scale_inv"
            up_s = f"{up_n}_scale_inv"
            down_s = f"{down_n}_scale_inv"
            gate_scale_shape = raw.meta(gate_s)["shape"]
            up_scale_shape = raw.meta(up_s)["shape"]
            down_scale_shape = raw.meta(down_s)["shape"]
            gate_up_scale.extend(raw.raw_bytes(gate_s))
            gate_up_scale.extend(raw.raw_bytes(up_s))
            down_scale.extend(raw.raw_bytes(down_s))

        writer.add(
            f"{base}.gate_up_proj",
            bytes(gate_up),
            [len(experts), gate_shape[0] + up_shape[0], gate_shape[1]],
            "f8_e4m3",
            LAYOUT_FP8_NATIVE,
        )
        writer.add(
            f"{base}.gate_up_proj_scale_inv",
            bytes(gate_up_scale),
            [len(experts), gate_scale_shape[0] + up_scale_shape[0], gate_scale_shape[1]],
            "bf16",
            LAYOUT_RAW,
        )
        writer.add(
            f"{base}.down_proj",
            bytes(down),
            [len(experts), down_shape[0], down_shape[1]],
            "f8_e4m3",
            LAYOUT_FP8_NATIVE,
        )
        writer.add(
            f"{base}.down_proj_scale_inv",
            bytes(down_scale),
            [len(experts), down_scale_shape[0], down_scale_shape[1]],
            "bf16",
            LAYOUT_RAW,
        )
        log(f"[bake-fp8/raw]   layer {layer}: fused {len(experts)} experts")

    n_skip = 0
    for name in eligible:
        if name in skip_names:
            continue
        meta = raw.meta(name)
        shape = list(meta["shape"])
        dtype = meta["dtype"]
        data = raw.raw_bytes(name)
        layout = classify_layout(name, tuple(shape), linear_idx)
        if layout == LAYOUT_DEPTHWISE_CONV_SQUEEZED:
            shape = [shape[0], shape[2]]
        elif layout == LAYOUT_HEAD_BIAS_RESHAPED:
            shape = [1, 1, shape[0]]
        elif layout == LAYOUT_HEAD_EXP_RESHAPED:
            data = _a_log_raw_to_exp_bf16(data, dtype)
            shape = [1, 1, shape[0]]
            dtype = "BF16"
        elif dtype == "F8_E4M3" and is_fp8_quant_target(name, tuple(shape), block_size):
            layout = LAYOUT_FP8_NATIVE
        writer.add(name, data, shape, _safetensors_dtype_to_manifest(dtype), layout)
        n_skip += 1

    writer.close(model_family=model_family)
    log(f"[bake-fp8/raw] wrote {n_skip} non-fused tensors. Output: {out_dir}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_tensor_index(model_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """Return (shards, name → shard_idx) over every .safetensors in model_dir."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors found in {model_dir}")
    index: dict[str, int] = {}
    for i, sf in enumerate(shards):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                index[key] = i
    return shards, index


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-block FP8 bake for Qwen3.5")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="Override output dir (default: {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8)")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                    help=f"Per-tile FP8 block size (default {BLOCK_SIZE}; "
                         f"the Rust runtime currently only supports 128).")
    ap.add_argument("--model-family", default=None,
                    help="Override the manifest model_family field (default: "
                         "auto-detect from config.json — qwen36-moe for MoE "
                         "checkpoints, qwen35 otherwise).")
    args = ap.parse_args()

    block_size: int = args.block_size
    if block_size != BLOCK_SIZE:
        log(f"[bake-fp8] WARNING: --block-size={block_size} is non-default; "
            "the Rust runtime currently assumes 128 — set this only for "
            "calibration/perplexity experiments, not for production bakes.")

    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir or (model_dir / ".supersonic" / f"v{FORMAT_VERSION}-fp8")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {model_dir}")
    model_family = args.model_family or detect_model_family(config_path)
    log(f"[bake-fp8] model_family={model_family}")
    if (torch is None or safe_open is None) and model_family == "qwen36-moe":
        log("[bake-fp8] torch/safetensors not available; using raw qwen36 FP8 path")
        bake_qwen36_fp8_raw(model_dir, out_dir, block_size)
        return
    if torch is None or safe_open is None:
        raise SystemExit("bake_fp8.py needs torch+safetensors for non-qwen36 FP8 bakes")
    linear_idx = linear_attn_layer_indices(config_path)
    log(f"[bake-fp8] linear-attn layer indices ({len(linear_idx)}): "
        f"{sorted(linear_idx)[:8]}{' ...' if len(linear_idx) > 8 else ''}")

    shards, index = collect_tensor_index(model_dir)
    log(f"[bake-fp8] {len(shards)} shard(s), {len(index)} tensors")

    # Filter to language-model tensors + lm_head; sort for deterministic output.
    weight_prefix = "model.language_model"
    eligible = sorted(
        n for n in index
        if n.startswith(f"{weight_prefix}.") or n == "lm_head.weight"
    )
    log(f"[bake-fp8] {len(eligible)} eligible tensors")

    writer = StreamingTensorWriter(out_dir)
    open_shards: dict[int, Any] = {}

    def get_handle(idx: int):
        h = open_shards.get(idx)
        if h is None:
            h = safe_open(str(shards[idx]), framework="pt")
            h.__enter__()
            open_shards[idx] = h
        return h

    def load_tensor(name: str) -> torch.Tensor:
        shard_idx = index[name]
        return get_handle(shard_idx).get_tensor(name)

    def fuse_qwen36_experts() -> set[str]:
        """Emit runtime-fused Qwen3.6-MoE expert slabs.

        The published FP8 checkpoint stores experts as
        `experts.{id}.{gate,up,down}_proj.weight` plus per-expert
        `*_scale_inv`. SuperSonic's qwen36 runtime consumes the same fused
        names as the INT4 bake: `experts.gate_up_proj` and
        `experts.down_proj`, each with one fused `_scale_inv` sidecar.
        """
        if model_family != "qwen36-moe":
            return set()

        per_layer: dict[int, set[int]] = {}
        skipped: set[str] = set()
        for name in index:
            m = QWEN36_EXPERT_RE.match(name)
            if not m:
                continue
            layer = int(m.group(1))
            expert = int(m.group(2))
            per_layer.setdefault(layer, set()).add(expert)
            skipped.add(name)

        if not per_layer:
            return skipped

        log(f"[bake-fp8] qwen36-moe fused experts: {len(per_layer)} layer(s)")
        for layer in sorted(per_layer):
            experts = sorted(per_layer[layer])
            base = f"model.language_model.layers.{layer}.mlp.experts"

            gate_up_bytes = bytearray()
            gate_up_scale_chunks: list[torch.Tensor] = []
            down_bytes = bytearray()
            down_scale_chunks: list[torch.Tensor] = []

            for expert in experts:
                eb = f"{base}.{expert}"
                gate = load_tensor(f"{eb}.gate_proj.weight")
                up = load_tensor(f"{eb}.up_proj.weight")
                down = load_tensor(f"{eb}.down_proj.weight")
                if gate.dtype != torch.float8_e4m3fn or up.dtype != torch.float8_e4m3fn or down.dtype != torch.float8_e4m3fn:
                    raise SystemExit(
                        f"qwen36 FP8 fused expert source must be float8_e4m3fn "
                        f"(layer={layer}, expert={expert})"
                    )
                gate_up_bytes.extend(tensor_to_bytes(gate, "f8_e4m3"))
                gate_up_bytes.extend(tensor_to_bytes(up, "f8_e4m3"))
                down_bytes.extend(tensor_to_bytes(down, "f8_e4m3"))

                gate_s = load_tensor(f"{eb}.gate_proj.weight_scale_inv").to(torch.bfloat16)
                up_s = load_tensor(f"{eb}.up_proj.weight_scale_inv").to(torch.bfloat16)
                down_s = load_tensor(f"{eb}.down_proj.weight_scale_inv").to(torch.bfloat16)
                gate_up_scale_chunks.append(torch.cat([gate_s, up_s], dim=0).unsqueeze(0))
                down_scale_chunks.append(down_s.unsqueeze(0))

            gate_shape = [len(experts), int(gate.shape[0] + up.shape[0]), int(gate.shape[1])]
            down_shape = [len(experts), int(down.shape[0]), int(down.shape[1])]
            writer.add(
                f"{base}.gate_up_proj",
                bytes(gate_up_bytes),
                gate_shape,
                "f8_e4m3",
                LAYOUT_FP8_NATIVE,
            )
            gate_up_scale = torch.cat(gate_up_scale_chunks, dim=0).contiguous()
            writer.add(
                f"{base}.gate_up_proj_scale_inv",
                tensor_to_bytes(gate_up_scale, "bf16"),
                list(gate_up_scale.shape),
                "bf16",
                LAYOUT_RAW,
            )
            writer.add(
                f"{base}.down_proj",
                bytes(down_bytes),
                down_shape,
                "f8_e4m3",
                LAYOUT_FP8_NATIVE,
            )
            down_scale = torch.cat(down_scale_chunks, dim=0).contiguous()
            writer.add(
                f"{base}.down_proj_scale_inv",
                tensor_to_bytes(down_scale, "bf16"),
                list(down_scale.shape),
                "bf16",
                LAYOUT_RAW,
            )
            log(f"[bake-fp8]   layer {layer}: fused {len(experts)} experts")
            del gate_up_bytes, down_bytes, gate_up_scale_chunks, down_scale_chunks
            del gate_up_scale, down_scale

        return skipped

    t0 = time.perf_counter()
    n_quant = 0
    n_skip = 0
    try:
        skip_names = fuse_qwen36_experts()
        for name in eligible:
            if name in skip_names:
                continue
            shard_idx = index[name]
            handle = get_handle(shard_idx)
            t = handle.get_tensor(name)
            shape = tuple(t.shape)

            # FP8 quantization / pass-through candidate.
            if is_fp8_quant_target(name, shape, block_size):
                if t.dtype == torch.bfloat16:
                    # BF16 source → quantize to E4M3 with per-block scales.
                    packed, scale_bf16 = quantize_bf16_to_fp8(t, block_size)
                    writer.add(
                        name,
                        tensor_to_bytes(packed, "f8_e4m3"),
                        list(shape),
                        "f8_e4m3",
                        LAYOUT_FP8_NATIVE,
                    )
                    writer.add(
                        f"{name}_scale_inv",
                        tensor_to_bytes(scale_bf16, "bf16"),
                        list(scale_bf16.shape),
                        "bf16",
                        LAYOUT_RAW,
                    )
                    n_quant += 1
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    # FP8-native source (e.g. Qwen 3.6 *-FP8 checkpoints) —
                    # store the bytes as-is with the Fp8Native layout tag so
                    # `upload_bake.py --fp8-native` accepts the bake. The
                    # companion `_scale_inv` tensor is in source safetensors
                    # already and falls through to the raw write path below.
                    fp8_bytes = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    writer.add(
                        name,
                        fp8_bytes,
                        list(shape),
                        "f8_e4m3",
                        LAYOUT_FP8_NATIVE,
                    )
                    n_quant += 1
                    continue
                # Other dtype on a quant-target name: warn loudly and fall
                # through to the raw path so we don't silently drop tensors.
                log(f"[bake-fp8] WARNING: quant target {name} has unexpected "
                    f"dtype {t.dtype}; storing raw")

            # Layout-transform path.
            layout = classify_layout(name, shape, linear_idx)
            if layout == LAYOUT_DEPTHWISE_CONV_SQUEEZED:
                t2 = apply_squeeze_dim1(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            elif layout == LAYOUT_HEAD_BIAS_RESHAPED:
                t2 = apply_head_bias_reshape(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            elif layout == LAYOUT_HEAD_EXP_RESHAPED:
                t2 = apply_a_log_to_exp_bf16(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            else:
                writer.add(name, tensor_to_bytes(t, torch_dtype_to_str(t.dtype)),
                           list(shape), torch_dtype_to_str(t.dtype), LAYOUT_RAW)
            n_skip += 1
    finally:
        for h in open_shards.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass

    writer.close(model_family=model_family)
    elapsed = time.perf_counter() - t0
    log(f"[bake-fp8] quantized {n_quant} projection tensors, "
        f"passed through {n_skip} layout/raw tensors in {elapsed:.1f}s")
    log(f"[bake-fp8] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
