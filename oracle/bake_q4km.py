#!/usr/bin/env python3
"""
Create a SuperSonic q4km bake.

The runtime format preserves GGML K-block tensors from GGUF for target
projection weights. CUDA interprets Q4_K/Q5_K/Q6_K blocks directly instead of
requantizing them into SuperSonic's older native INT4 layout.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any

import torch
import numpy as np
from safetensors import safe_open

try:
    from bake_int4 import gptq_quantize
except Exception:
    gptq_quantize = None

FORMAT_VERSION = 2
CONVERTER_VERSION = 1
LAYOUT_RAW = "Raw"
LAYOUT_CONV_SQ = "DepthwiseConvSqueezed"
LAYOUT_HEAD_BIAS = "HeadBiasReshaped"
LAYOUT_HEAD_EXP = "HeadExpReshaped"
LAYOUT_INT4 = "Int4Quantized"
LAYOUT_GGML_Q4K = "GgmlQ4K"
LAYOUT_GGML_Q5K = "GgmlQ5K"
LAYOUT_GGML_Q6K = "GgmlQ6K"
QUANT_MINMAX = "minmax"
QUANT_GPTQ = "gptq"

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30
GGUF_METADATA_ARRAY = 9
QK_K = 256
QK_0 = 32
BLOCK_Q8_0_BYTES = 2 + QK_0
BLOCK_Q4_K_BYTES = 2 + 2 + 12 + QK_K // 2
BLOCK_Q5_K_BYTES = 2 + 2 + 12 + QK_K // 8 + QK_K // 2
BLOCK_Q6_K_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2


def log(msg: str) -> None:
    print(msg, flush=True)


def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def prod(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def bf16_to_bytes(t: torch.Tensor) -> bytes:
    t = t.to(torch.bfloat16).contiguous().cpu()
    arr = t.view(torch.int16).numpy()
    if sys.byteorder != "little":
        arr = arr.byteswap()
    return arr.tobytes()


def tensor_to_bytes(t: torch.Tensor, dtype_str: str) -> bytes:
    t = t.contiguous().cpu()
    if dtype_str == "bf16":
        return bf16_to_bytes(t)
    if dtype_str == "f32":
        return t.to(torch.float32).numpy().tobytes()
    if dtype_str == "f16":
        return t.to(torch.float16).numpy().tobytes()
    if dtype_str == "u8":
        return t.to(torch.uint8).numpy().tobytes()
    if dtype_str == "u32":
        return t.to(torch.uint32).numpy().tobytes()
    if dtype_str == "i64":
        return t.to(torch.int64).numpy().tobytes()
    raise ValueError(f"unsupported dtype_str {dtype_str}")


def f16_le_to_f32(buf: bytes, offset: int = 0) -> float:
    return float(struct.unpack_from("<e", buf, offset)[0])


def bf16_u16_to_f32_tensor(u16: torch.Tensor) -> torch.Tensor:
    arr = (u16.cpu().numpy().astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(arr.copy())


def torch_dtype_to_str(dt: torch.dtype) -> str:
    return {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float16: "f16",
        torch.uint8: "u8",
        torch.int64: "i64",
    }.get(dt, "bf16")


class GgufReader:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("rb")
        self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0

    def tell(self) -> int:
        return self.pos

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise SystemExit(f"{self.path}: unexpected EOF")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return out[0] if len(out) == 1 else out

    def string(self) -> str:
        n = self.unpack("<Q")
        return self.read(n).decode("utf-8")


@dataclass
class GgufTensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int


@dataclass
class GgufFile:
    version: int
    metadata: dict[str, object]
    tensors: list[GgufTensorInfo]
    data_start: int
    blob: mmap.mmap
    file_handle: BinaryIO

    def close(self) -> None:
        self.blob.close()
        self.file_handle.close()


def read_gguf_scalar(r: GgufReader, ty: int):
    if ty == 0:
        return r.unpack("<B")
    if ty == 1:
        return r.unpack("<b")
    if ty == 2:
        return r.unpack("<H")
    if ty == 3:
        return r.unpack("<h")
    if ty == 4:
        return r.unpack("<I")
    if ty == 5:
        return r.unpack("<i")
    if ty == 6:
        return r.unpack("<f")
    if ty == 7:
        return bool(r.unpack("<?"))
    if ty == 8:
        return r.string()
    if ty == 10:
        return r.unpack("<Q")
    if ty == 11:
        return r.unpack("<q")
    if ty == 12:
        return r.unpack("<d")
    raise SystemExit(f"unsupported GGUF metadata value type {ty}")


def read_gguf_value(r: GgufReader, ty: int):
    if ty != GGUF_METADATA_ARRAY:
        return read_gguf_scalar(r, ty)
    elem_ty = r.unpack("<I")
    n = r.unpack("<Q")
    return [read_gguf_value(r, elem_ty) for _ in range(n)]


def parse_gguf(path: Path) -> GgufFile:
    r = GgufReader(path)
    if r.read(4) != b"GGUF":
        raise SystemExit(f"{path} is not a GGUF file")
    version = r.unpack("<I")
    if version < 2 or version > 3:
        raise SystemExit(f"unsupported GGUF version {version}")
    tensor_count = r.unpack("<Q")
    metadata_count = r.unpack("<Q")
    metadata: dict[str, object] = {}
    for _ in range(metadata_count):
        key = r.string()
        ty = r.unpack("<I")
        metadata[key] = read_gguf_value(r, ty)
    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.unpack("<I")
        dims = [int(r.unpack("<Q")) for _ in range(n_dims)]
        ggml_type = int(r.unpack("<I"))
        offset = int(r.unpack("<Q"))
        tensors.append(GgufTensorInfo(name, dims, ggml_type, offset))
    alignment = int(metadata.get("general.alignment", 32) or 32)
    data_start = align_up(r.tell(), alignment)
    return GgufFile(version, metadata, tensors, data_start, r.data, r._fh)


def ggml_row_size(ggml_type: int, cols: int) -> int:
    if ggml_type == GGML_TYPE_F32:
        return cols * 4
    if ggml_type in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return cols * 2
    if ggml_type == GGML_TYPE_Q8_0:
        if cols % QK_0 != 0:
            raise SystemExit(f"Q8_0 row length {cols} is not divisible by {QK_0}")
        return (cols // QK_0) * BLOCK_Q8_0_BYTES
    if ggml_type == GGML_TYPE_Q4_K:
        if cols % QK_K != 0:
            raise SystemExit(f"Q4_K row length {cols} is not divisible by {QK_K}")
        return (cols // QK_K) * BLOCK_Q4_K_BYTES
    if ggml_type == GGML_TYPE_Q5_K:
        if cols % QK_K != 0:
            raise SystemExit(f"Q5_K row length {cols} is not divisible by {QK_K}")
        return (cols // QK_K) * BLOCK_Q5_K_BYTES
    if ggml_type == GGML_TYPE_Q6_K:
        if cols % QK_K != 0:
            raise SystemExit(f"Q6_K row length {cols} is not divisible by {QK_K}")
        return (cols // QK_K) * BLOCK_Q6_K_BYTES
    raise SystemExit(f"unsupported GGML tensor type {ggml_type} ({ggml_type_name(ggml_type)})")


def ggml_type_name(ggml_type: int) -> str:
    return {
        GGML_TYPE_F32: "F32",
        GGML_TYPE_F16: "F16",
        GGML_TYPE_Q8_0: "Q8_0",
        GGML_TYPE_Q4_K: "Q4_K",
        GGML_TYPE_Q5_K: "Q5_K",
        GGML_TYPE_Q6_K: "Q6_K",
        GGML_TYPE_BF16: "BF16",
    }.get(ggml_type, "unknown")


def gguf_logical_shape(info: GgufTensorInfo) -> list[int]:
    return list(reversed(info.dims))


def gguf_tensor_nbytes(info: GgufTensorInfo) -> int:
    if not info.dims:
        raise SystemExit(f"{info.name}: scalar GGUF tensor is not supported")
    rows = prod(info.dims[1:]) if len(info.dims) > 1 else 1
    return ggml_row_size(info.ggml_type, info.dims[0]) * rows


def raw_gguf_tensor_bytes(g: GgufFile, info: GgufTensorInfo) -> bytes:
    nbytes = gguf_tensor_nbytes(info)
    start = g.data_start + info.offset
    raw = bytes(g.blob[start:start + nbytes])
    if len(raw) != nbytes:
        raise SystemExit(f"{info.name}: tensor data extends past EOF")
    return raw


def ggml_k_layout(ggml_type: int) -> str | None:
    if ggml_type == GGML_TYPE_Q4_K:
        return LAYOUT_GGML_Q4K
    if ggml_type == GGML_TYPE_Q5_K:
        return LAYOUT_GGML_Q5K
    if ggml_type == GGML_TYPE_Q6_K:
        return LAYOUT_GGML_Q6K
    return None


def get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequantize_q4_k(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    blocks_per_row = cols // QK_K
    blocks = np.frombuffer(buf, dtype=np.uint8).reshape(rows * blocks_per_row, BLOCK_Q4_K_BYTES)
    d = blocks[:, 0:2].view(np.float16).reshape(-1).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).reshape(-1).astype(np.float32)
    packed_scales = blocks[:, 4:16]
    scales = np.empty((blocks.shape[0], 8), dtype=np.float32)
    mins = np.empty((blocks.shape[0], 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            scales[:, j] = packed_scales[:, j] & 63
            mins[:, j] = packed_scales[:, j + 4] & 63
        else:
            scales[:, j] = (packed_scales[:, j + 4] & 0x0F) | ((packed_scales[:, j - 4] >> 6) << 4)
            mins[:, j] = (packed_scales[:, j + 4] >> 4) | ((packed_scales[:, j] >> 6) << 4)
    qs = blocks[:, 16:144]
    out = np.empty((blocks.shape[0], QK_K), dtype=np.float32)
    for g in range(4):
        q = qs[:, g * 32:(g + 1) * 32].astype(np.float32)
        d1 = (d * scales[:, 2 * g + 0])[:, None]
        m1 = (dmin * mins[:, 2 * g + 0])[:, None]
        d2 = (d * scales[:, 2 * g + 1])[:, None]
        m2 = (dmin * mins[:, 2 * g + 1])[:, None]
        out[:, g * 64:g * 64 + 32] = d1 * (q.astype(np.uint8) & 0x0F).astype(np.float32) - m1
        out[:, g * 64 + 32:g * 64 + 64] = d2 * (q.astype(np.uint8) >> 4).astype(np.float32) - m2
    return torch.from_numpy(out.reshape(rows, blocks_per_row, QK_K).reshape(rows, cols).copy())


def dequantize_q6_k(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    blocks_per_row = cols // QK_K
    blocks = np.frombuffer(buf, dtype=np.uint8).reshape(rows * blocks_per_row, BLOCK_Q6_K_BYTES)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].view(np.float16).reshape(-1).astype(np.float32)
    out = np.empty((blocks.shape[0], QK_K), dtype=np.float32)
    l = np.arange(32)
    iscale = l // 16
    for half in range(2):
        ql_pos = half * 64
        qh_pos = half * 32
        sc_pos = half * 8
        y_pos = half * 128
        qh_chunk = qh[:, qh_pos:qh_pos + 32]
        q1 = ((ql[:, ql_pos:ql_pos + 32] & 0x0F) | (((qh_chunk >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql[:, ql_pos + 32:ql_pos + 64] & 0x0F) | (((qh_chunk >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql[:, ql_pos:ql_pos + 32] >> 4) | (((qh_chunk >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql[:, ql_pos + 32:ql_pos + 64] >> 4) | (((qh_chunk >> 6) & 3) << 4)).astype(np.int16) - 32
        out[:, y_pos:y_pos + 32] = (d[:, None] * scales[:, sc_pos + iscale + 0]) * q1
        out[:, y_pos + 32:y_pos + 64] = (d[:, None] * scales[:, sc_pos + iscale + 2]) * q2
        out[:, y_pos + 64:y_pos + 96] = (d[:, None] * scales[:, sc_pos + iscale + 4]) * q3
        out[:, y_pos + 96:y_pos + 128] = (d[:, None] * scales[:, sc_pos + iscale + 6]) * q4
    return torch.from_numpy(out.reshape(rows, blocks_per_row, QK_K).reshape(rows, cols).copy())


def dequantize_q5_k(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    blocks_per_row = cols // QK_K
    blocks = np.frombuffer(buf, dtype=np.uint8).reshape(rows * blocks_per_row, BLOCK_Q5_K_BYTES)
    d = blocks[:, 0:2].view(np.float16).reshape(-1).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).reshape(-1).astype(np.float32)
    packed_scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    ql = blocks[:, 48:176]
    scales = np.empty((blocks.shape[0], 8), dtype=np.float32)
    mins = np.empty((blocks.shape[0], 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            scales[:, j] = packed_scales[:, j] & 63
            mins[:, j] = packed_scales[:, j + 4] & 63
        else:
            scales[:, j] = (packed_scales[:, j + 4] & 0x0F) | ((packed_scales[:, j - 4] >> 6) << 4)
            mins[:, j] = (packed_scales[:, j + 4] >> 4) | ((packed_scales[:, j] >> 6) << 4)
    out = np.empty((blocks.shape[0], QK_K), dtype=np.float32)
    for g in range(4):
        q = ql[:, g * 32:(g + 1) * 32]
        u1 = 1 << (2 * g)
        u2 = 2 << (2 * g)
        hi1 = ((qh & u1) != 0).astype(np.float32) * 16.0
        hi2 = ((qh & u2) != 0).astype(np.float32) * 16.0
        d1 = (d * scales[:, 2 * g + 0])[:, None]
        m1 = (dmin * mins[:, 2 * g + 0])[:, None]
        d2 = (d * scales[:, 2 * g + 1])[:, None]
        m2 = (dmin * mins[:, 2 * g + 1])[:, None]
        out[:, g * 64:g * 64 + 32] = d1 * ((q & 0x0F).astype(np.float32) + hi1) - m1
        out[:, g * 64 + 32:g * 64 + 64] = d2 * ((q >> 4).astype(np.float32) + hi2) - m2
    return torch.from_numpy(out.reshape(rows, blocks_per_row, QK_K).reshape(rows, cols).copy())


def dequantize_q8_0(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    blocks_per_row = cols // QK_0
    row_size = blocks_per_row * BLOCK_Q8_0_BYTES
    out = torch.empty((rows, cols), dtype=torch.float32)
    for row in range(rows):
        row_base = row * row_size
        dst_base = 0
        for block in range(blocks_per_row):
            base = row_base + block * BLOCK_Q8_0_BYTES
            d = f16_le_to_f32(buf, base)
            qs = np.frombuffer(buf, dtype=np.int8, count=QK_0, offset=base + 2).astype(np.float32)
            out[row, dst_base:dst_base + QK_0] = torch.from_numpy(qs * d)
            dst_base += QK_0
    return out


def load_gguf_tensor(g: GgufFile, info: GgufTensorInfo) -> torch.Tensor:
    cols = info.dims[0]
    rows = prod(info.dims[1:]) if len(info.dims) > 1 else 1
    nbytes = gguf_tensor_nbytes(info)
    start = g.data_start + info.offset
    raw = g.blob[start:start + nbytes]
    if len(raw) != nbytes:
        raise SystemExit(f"{info.name}: tensor data extends past EOF")
    shape = gguf_logical_shape(info)
    if info.ggml_type == GGML_TYPE_F32:
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(shape)
    if info.ggml_type == GGML_TYPE_F16:
        return torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().reshape(shape)
    if info.ggml_type == GGML_TYPE_BF16:
        u16 = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
        return bf16_u16_to_f32_tensor(u16).reshape(shape).to(torch.bfloat16)
    if info.ggml_type == GGML_TYPE_Q8_0:
        return dequantize_q8_0(raw, rows, cols).reshape(shape)
    if info.ggml_type == GGML_TYPE_Q4_K:
        return dequantize_q4_k(raw, rows, cols).reshape(shape)
    if info.ggml_type == GGML_TYPE_Q5_K:
        return dequantize_q5_k(raw, rows, cols).reshape(shape)
    if info.ggml_type == GGML_TYPE_Q6_K:
        return dequantize_q6_k(raw, rows, cols).reshape(shape)
    raise SystemExit(
        f"{info.name}: unsupported GGML tensor type {info.ggml_type} "
        f"({ggml_type_name(info.ggml_type)})"
    )


def discover_safetensors(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        raw = json.loads(index.read_text())
        files = sorted(set(raw.get("weight_map", {}).values()))
        return [model_dir / f for f in files]
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors files found in {model_dir}")
    return files


def tensor_index(files: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = path
    return out


def load_tensor(index: dict[str, Path], name: str) -> torch.Tensor:
    path = index[name]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


@torch.no_grad()
def dequant_fp8_blocks(w: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    if w.ndim != 2:
        raise SystemExit(f"FP8 tensor {tuple(w.shape)} has _scale_inv but is not 2D")
    rows, cols = w.shape
    expected = (math.ceil(rows / block_size), math.ceil(cols / block_size))
    if tuple(scale_inv.shape) != expected:
        raise SystemExit(
            f"FP8 _scale_inv shape {tuple(scale_inv.shape)} does not match "
            f"weight shape {tuple(w.shape)} with block size {block_size}; expected {expected}"
        )
    out = w.to(torch.float32)
    scales = scale_inv.to(torch.float32)
    for br in range(expected[0]):
        r0 = br * block_size
        r1 = min(r0 + block_size, rows)
        for bc in range(expected[1]):
            c0 = bc * block_size
            c1 = min(c0 + block_size, cols)
            out[r0:r1, c0:c1].mul_(scales[br, bc])
    return out


def load_source_tensor(index: dict[str, Path], name: str) -> tuple[torch.Tensor, bool]:
    t = load_tensor(index, name)
    scale_name = f"{name}_scale_inv"
    if scale_name not in index:
        return t, False
    return dequant_fp8_blocks(t, load_tensor(index, scale_name)), True


def infer_weight_prefix(keys: list[str]) -> str:
    for key in sorted(keys):
        if key.endswith(".embed_tokens.weight"):
            return key[: -len(".embed_tokens.weight")]
    for key in sorted(keys):
        if ".layers.0." in key:
            return key.split(".layers.")[0]
    raise SystemExit("could not infer weight prefix from safetensors keys")


def classify_tensor(name: str, shape: list[int], weight_prefix: str, layer_types: list[str]) -> str:
    prefix = f"{weight_prefix}.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        dot = rest.find(".")
        if dot > 0:
            try:
                idx = int(rest[:dot])
                if idx < len(layer_types) and layer_types[idx] == "linear_attention":
                    if name.endswith(".conv1d.weight") and len(shape) == 3 and shape[1] == 1:
                        return LAYOUT_CONV_SQ
                    if name.endswith(".conv1d.weight") and len(shape) == 2:
                        return LAYOUT_CONV_SQ
                    if name.endswith(".dt_bias") and len(shape) == 1:
                        return LAYOUT_HEAD_BIAS
                    if name.endswith(".A_log") and len(shape) == 1:
                        return LAYOUT_HEAD_EXP
            except ValueError:
                pass
    return LAYOUT_RAW


def map_gguf_name(name: str, weight_prefix: str) -> str | None:
    top_level = {
        "token_embd.weight": f"{weight_prefix}.embed_tokens.weight",
        "output_norm.weight": f"{weight_prefix}.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if name in top_level:
        return top_level[name]

    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    layer = int(m.group(1))
    rest = m.group(2)
    lp = f"{weight_prefix}.layers.{layer}"
    mapped = {
        "attn_norm.weight": f"{lp}.input_layernorm.weight",
        "ffn_norm.weight": f"{lp}.post_attention_layernorm.weight",
        "post_attention_norm.weight": f"{lp}.post_attention_layernorm.weight",
        "ffn_gate.weight": f"{lp}.mlp.gate_proj.weight",
        "ffn_up.weight": f"{lp}.mlp.up_proj.weight",
        "ffn_down.weight": f"{lp}.mlp.down_proj.weight",
        "attn_q.weight": f"{lp}.self_attn.q_proj.weight",
        "attn_k.weight": f"{lp}.self_attn.k_proj.weight",
        "attn_v.weight": f"{lp}.self_attn.v_proj.weight",
        "attn_output.weight": f"{lp}.self_attn.o_proj.weight",
        "attn_q_norm.weight": f"{lp}.self_attn.q_norm.weight",
        "attn_k_norm.weight": f"{lp}.self_attn.k_norm.weight",
        # Qwen3Next/Qwen3.5 linear-attention tensors in llama.cpp naming.
        "attn_qkv.weight": f"{lp}.linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": f"{lp}.linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": f"{lp}.linear_attn.in_proj_a.weight",
        "ssm_beta.weight": f"{lp}.linear_attn.in_proj_b.weight",
        "ssm_out.weight": f"{lp}.linear_attn.out_proj.weight",
        "ssm_conv1d.weight": f"{lp}.linear_attn.conv1d.weight",
        "ssm_dt.bias": f"{lp}.linear_attn.dt_bias",
        "ssm_a.weight": f"{lp}.linear_attn.A_log",
        "ssm_a": f"{lp}.linear_attn.A_log",
        "ssm_norm.weight": f"{lp}.linear_attn.norm.weight",
    }
    return mapped.get(rest)


def undo_gguf_tensor_transform(name: str, t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Undo llama.cpp conversion transforms so emitted names match HF bake semantics.

    Returns `(tensor, a_log_is_precomputed)`; Qwen3Next GGUF stores SSM A as
    `-exp(A_log)`, while SuperSonic's baked A_log slot stores `exp(A_log)`.
    """
    if name.endswith(".linear_attn.A_log"):
        return -t.to(torch.float32), True
    return t, False


def apply_layout(t: torch.Tensor, shape: list[int], layout: str, dtype_str: str) -> tuple[bytes, list[int], str]:
    if layout == LAYOUT_RAW:
        if dtype_str in ("f32", "f16", "bf16"):
            return tensor_to_bytes(t, "bf16"), shape, "bf16"
        return tensor_to_bytes(t, dtype_str), shape, dtype_str
    if layout == LAYOUT_CONV_SQ:
        if len(shape) == 3:
            return tensor_to_bytes(t.squeeze(1), "bf16"), [shape[0], shape[2]], "bf16"
        return tensor_to_bytes(t, "bf16"), shape, "bf16"
    if layout == LAYOUT_HEAD_BIAS:
        return tensor_to_bytes(t.reshape(1, 1, shape[0]), "bf16"), [1, 1, shape[0]], "bf16"
    if layout == LAYOUT_HEAD_EXP:
        return bf16_to_bytes(torch.exp(t.to(torch.float32)).reshape(1, 1, shape[0])), [1, 1, shape[0]], "bf16"
    raise ValueError(f"unknown layout {layout}")


# Fused 3D MoE expert tensors (Qwen3.6-MoE): names lack `.weight` and ship 3D.
# Treated as `E` parallel `[out, in]` matrices for INT4 packing — see
# `quantize_minmax_fused_experts` for the driver path.
FUSED_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)


def is_fused_expert_target(name: str) -> bool:
    return any(name.endswith(s) for s in FUSED_EXPERT_SUFFIXES)


def is_q4km_target(name: str, shape: list[int], group_size: int) -> bool:
    if is_fused_expert_target(name):
        # Real shapes: gate_up_proj [E, 2*moe_int, hidden],
        #              down_proj    [E, hidden, moe_int].
        # Validate divisibility on the per-expert (out, in) axes and accept.
        if len(shape) != 3:
            return False
        out_dim, in_dim = shape[1], shape[2]
        if in_dim % group_size != 0 or in_dim % 2 != 0:
            return False
        if out_dim % group_size != 0:
            return False
        return True
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if shape[1] % group_size != 0 or shape[1] % 2 != 0:
        return False
    if shape[0] % group_size != 0:
        # GPTQ scale tile is sized [out/gs, in/gs]; out must align too.
        return False
    lowered = name.lower()
    if any(s in lowered for s in ("layernorm", "norm.weight", "embed_tokens", "conv1d")):
        return False
    if "in_proj_b.weight" in name or "in_proj_a.weight" in name:
        return False
    if ".gate." in name or "router" in lowered:
        return False
    if name == "lm_head.weight":
        # The runtime (Qwen35Weights) loads lm_head_int4_scale/zero when present
        # and dispatches the INT4 matmul on Metal — saves ~4× device-read traffic
        # on the dominant decode-side matmul.
        return True
    return "_proj" in name or "experts" in name or "ffn_" in name


def pack_nibbles_3d(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack `[E, rows, cols]` uint8 nibbles into `[E, rows, cols/2]` u8 bytes."""
    if nibbles.dim() != 3:
        raise ValueError(f"expected 3D nibble tensor, got shape {tuple(nibbles.shape)}")
    e, rows, cols = nibbles.shape
    if cols % 2 != 0:
        raise ValueError(f"cols must be even, got {cols}")
    r = nibbles.reshape(e, rows, cols // 2, 2).to(torch.uint8)
    return (r[..., 0] | (r[..., 1] << 4)).contiguous()


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    rows, cols = nibbles.shape
    if cols % 2 != 0:
        raise ValueError(f"cols must be even, got {cols}")
    r = nibbles.reshape(rows, cols // 2, 2).to(torch.uint8)
    return (r[..., 0] | (r[..., 1] << 4)).contiguous()


def sanitize_tensor_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace("/", "_")


def load_hessian(hessian_dir: Path | None, name: str) -> torch.Tensor | None:
    """Load a per-tensor Hessian for GPTQ requantization.

    Supported layouts:
      * hessian_dir/index.json mapping tensor names to relative .pt/.npy files
      * hessian_dir/<sanitized tensor name>.pt
      * hessian_dir/<sanitized tensor name>.npy
    """
    if hessian_dir is None:
        return None
    candidates: list[Path] = []
    index_path = hessian_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        rel = index.get(name)
        if rel is not None:
            candidates.append(hessian_dir / rel)
    stem = sanitize_tensor_name(name)
    candidates.extend([hessian_dir / f"{stem}.pt", hessian_dir / f"{stem}.npy"])
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pt":
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                obj = obj.get("H", obj.get("hessian"))
            if not isinstance(obj, torch.Tensor):
                raise SystemExit(f"{path}: expected tensor or {{'H': tensor}}")
            return obj.to(torch.float32)
        if path.suffix == ".npy":
            return torch.from_numpy(np.load(path)).to(torch.float32)
    return None


@torch.no_grad()
def quantize_minmax(W: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = W.to(torch.float32)
    rows, cols = W.shape
    scale_rows = math.ceil(rows / group_size)
    scale_cols = cols // group_size
    nibbles = torch.empty((rows, cols), dtype=torch.uint8)
    scales = torch.empty((scale_rows, scale_cols), dtype=torch.float32)
    zeros = torch.empty((scale_rows, scale_cols), dtype=torch.float32)

    for gr in range(scale_rows):
        r0 = gr * group_size
        r1 = min(r0 + group_size, rows)
        for gc in range(scale_cols):
            c0 = gc * group_size
            c1 = c0 + group_size
            tile = W[r0:r1, c0:c1]
            tmin = tile.amin()
            tmax = tile.amax()
            rng = tmax - tmin
            if float(rng) > 0.0:
                sc = (rng / 15.0).to(torch.bfloat16).to(torch.float32)
                zf = (-tmin / sc).to(torch.bfloat16).to(torch.float32)
            else:
                sc = torch.tensor(1.0, dtype=torch.float32)
                zf = torch.tensor(0.0, dtype=torch.float32)
            q = torch.clamp(torch.round(tile / sc + zf), 0, 15).to(torch.uint8)
            nibbles[r0:r1, c0:c1] = q
            scales[gr, gc] = sc
            zeros[gr, gc] = zf
    return pack_nibbles(nibbles), scales, zeros


@torch.no_grad()
def quantize_minmax_fused_experts(
    W: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert min/max INT4 group-quant for a 3D fused MoE weight slab.

    Input shape: `[E, out, in]`. Each `[out, in]` expert is independently
    quantized with `group_size`-tile BF16 scale + zero. Mirrors
    `oracle/bake_int4.fused_expert_minmax_int4` so both bake paths produce the
    same on-disk layout (packed nibbles `[E, out, in/2]`, scale/zero
    `[E, out/gs, in/gs]`).
    """
    if W.dim() != 3:
        raise SystemExit(f"fused expert tensor must be 3D, got shape {tuple(W.shape)}")
    E, out_f, in_f = W.shape
    gs = group_size
    if in_f % gs != 0 or in_f % 2 != 0:
        raise SystemExit(
            f"fused expert in_features {in_f} must be divisible by "
            f"group_size={gs} and even"
        )
    if out_f % gs != 0:
        raise SystemExit(
            f"fused expert out_features {out_f} must be divisible by group_size={gs}"
        )
    scale_rows = out_f // gs
    scale_cols = in_f // gs
    nibbles = torch.empty((E, out_f, in_f), dtype=torch.uint8)
    scales = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    zeros = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    for e in range(E):
        slab = W[e].to(torch.float32)
        tiles = slab.reshape(scale_rows, gs, scale_cols, gs)
        tmax = tiles.amax(dim=(1, 3))
        tmin = tiles.amin(dim=(1, 3))
        rng = tmax - tmin
        sc = torch.where(rng > 0, rng / 15.0, torch.ones_like(rng))
        zf = torch.where(rng > 0, -tmin / sc, torch.zeros_like(rng))
        sc = sc.to(torch.bfloat16).to(torch.float32)
        zf = zf.to(torch.bfloat16).to(torch.float32)
        sc_full = sc.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        zf_full = zf.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        q = torch.clamp(torch.round(slab / sc_full + zf_full), 0, 15).to(torch.uint8)
        nibbles[e] = q
        scales[e] = sc
        zeros[e] = zf
    return pack_nibbles_3d(nibbles), scales, zeros


@torch.no_grad()
def quantize_gptq(
    name: str,
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
    damp: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gptq_quantize is None:
        raise SystemExit("oracle/bake_int4.py GPTQ quantizer could not be imported")
    rows, cols = W.shape
    if rows % group_size != 0 or cols % group_size != 0:
        raise SystemExit(
            f"{name}: GPTQ requires both dimensions divisible by group_size={group_size}; "
            f"got {rows}x{cols}"
        )
    if tuple(H.shape) != (cols, cols):
        raise SystemExit(f"{name}: Hessian shape {tuple(H.shape)} does not match in_features={cols}")
    dev = torch.device(device)
    _, nibbles, scales, zeros = gptq_quantize(
        W.to(device=dev, dtype=torch.float32),
        H.to(device=dev, dtype=torch.float32),
        group_size=group_size,
        damp=damp,
    )
    return pack_nibbles(nibbles.cpu()), scales.cpu(), zeros.cpu()


class BakePackageWriter:
    def __init__(
        self,
        out_dir: Path,
        family: str,
        source_format: str,
        source_quant: str | None,
        quant_profile: str,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = out_dir / "weights.bin"
        self._wf = self.weights_path.open("wb")
        self._cursor = 0
        self._entries: list[dict[str, Any]] = []
        self._manifest_base = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": family,
            "quant_profile": quant_profile,
            "source_format": source_format,
            "source_quant": source_quant,
        }

    def write_tensor(self, name: str, data: bytes, shape: list[int], dtype: str, layout: str) -> None:
        aligned = align_up(self._cursor, 4096)
        if aligned > self._cursor:
            self._wf.write(b"\0" * (aligned - self._cursor))
            self._cursor = aligned
        self._wf.write(data)
        self._entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "layout": layout,
            "offset": self._cursor,
            "byte_len": len(data),
        })
        self._cursor += len(data)

    def write_entries(self, entries: list[tuple[str, bytes, list[int], str, str]]) -> None:
        for name, data, shape, dtype, layout in entries:
            self.write_tensor(name, data, shape, dtype, layout)

    def close(self, write_manifest: bool = True) -> None:
        self._wf.close()
        if not write_manifest:
            return
        manifest = dict(self._manifest_base)
        manifest["tensors"] = self._entries
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log(f"[q4km] wrote {self._cursor / (1024 * 1024):.1f} MiB to {self.weights_path}")
        log(f"[q4km] manifest: {self.out_dir / 'manifest.json'}")

    def __enter__(self) -> "BakePackageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(write_manifest=exc_type is None)


def write_bake(
    out_dir: Path,
    tensors: list[tuple[str, bytes, list[int], str, str]],
    family: str,
    source_format: str,
    source_quant: str | None,
    quant_profile: str,
) -> None:
    with BakePackageWriter(out_dir, family, source_format, source_quant, quant_profile) as writer:
        writer.write_entries(tensors)


def load_config_context(model_dir: Path) -> tuple[dict, list[str]]:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    text_cfg = config.get("text_config", config)
    layer_types = list(text_cfg.get("layer_types") or [])
    num_layers = int(text_cfg.get("num_hidden_layers", 0) or 0)
    if not layer_types and num_layers:
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
    return config, layer_types


def emit_tensor(
    tensors_out: list[tuple[str, bytes, list[int], str, str]],
    name: str,
    t: torch.Tensor,
    weight_prefix: str,
    layer_types: list[str],
    group_size: int,
    quantizer: str = QUANT_MINMAX,
    hessian_dir: Path | None = None,
    gptq_damp: float = 0.01,
    gptq_device: str = "cpu",
    a_log_precomputed: bool = False,
    raw_dtype_override: str | None = None,
) -> bool:
    entries, quantized = encode_tensor_entries(
        name,
        t,
        weight_prefix,
        layer_types,
        group_size,
        quantizer,
        hessian_dir,
        gptq_damp,
        gptq_device,
        a_log_precomputed,
        raw_dtype_override,
    )
    tensors_out.extend(entries)
    return quantized


def encode_tensor_entries(
    name: str,
    t: torch.Tensor,
    weight_prefix: str,
    layer_types: list[str],
    group_size: int,
    quantizer: str = QUANT_MINMAX,
    hessian_dir: Path | None = None,
    gptq_damp: float = 0.01,
    gptq_device: str = "cpu",
    a_log_precomputed: bool = False,
    raw_dtype_override: str | None = None,
) -> tuple[list[tuple[str, bytes, list[int], str, str]], bool]:
    entries: list[tuple[str, bytes, list[int], str, str]] = []
    shape = list(t.shape)
    if is_q4km_target(name, shape, group_size):
        if is_fused_expert_target(name):
            # Fused MoE experts (Qwen3.6-MoE) — 3D `[E, out, in]`. Hessian-aware
            # GPTQ doesn't apply cleanly to the fused layout (router decides
            # which expert sees a token, so per-tensor Hessians are not
            # well-defined). Always use min/max group-quant per expert; reject
            # GPTQ requests loudly so the operator switches to --quantizer minmax.
            if quantizer != QUANT_MINMAX:
                raise SystemExit(
                    f"{name}: fused MoE experts require --quantizer minmax "
                    f"(got {quantizer!r}); see docs/qwen36-moe-plan.md §15."
                )
            packed, scales, zeros = quantize_minmax_fused_experts(t, group_size)
        elif quantizer == QUANT_GPTQ:
            H = load_hessian(hessian_dir, name)
            if H is None:
                raise SystemExit(
                    f"{name}: --quantizer gptq requires a Hessian in {hessian_dir}; "
                    "use --quantizer minmax for calibration-free baking"
                )
            packed, scales, zeros = quantize_gptq(name, t, H, group_size, gptq_damp, gptq_device)
        elif quantizer == QUANT_MINMAX:
            packed, scales, zeros = quantize_minmax(t, group_size)
        else:
            raise SystemExit(f"unsupported quantizer {quantizer!r}")
        entries.append((name, packed.numpy().tobytes(), list(packed.shape), "u8", LAYOUT_INT4))
        entries.append((f"{name}_int4_scale", bf16_to_bytes(scales), list(scales.shape), "bf16", LAYOUT_RAW))
        entries.append((f"{name}_int4_zero", bf16_to_bytes(zeros), list(zeros.shape), "bf16", LAYOUT_RAW))
        return entries, True

    layout = classify_tensor(name, shape, weight_prefix, layer_types)
    if a_log_precomputed:
        if len(shape) != 1:
            raise SystemExit(f"{name}: precomputed GGUF A_log tensor must be rank 1, got {shape}")
        data = bf16_to_bytes(t.to(torch.float32).reshape(1, 1, shape[0]))
        entries.append((name, data, [1, 1, shape[0]], "bf16", LAYOUT_HEAD_EXP))
        return entries, False
    dtype = raw_dtype_override or torch_dtype_to_str(t.dtype)
    data, final_shape, final_dtype = apply_layout(t, shape, layout, dtype)
    entries.append((name, data, final_shape, final_dtype, layout))
    return entries, False


def bake_from_safetensors(args, weight_prefix: str, layer_types: list[str], family: str, out_dir: Path) -> None:
    quantizer = getattr(args, "quantizer", QUANT_MINMAX)
    hessian_dir = getattr(args, "hessian_dir", None)
    gptq_damp = getattr(args, "gptq_damp", 0.01)
    gptq_device = getattr(args, "gptq_device", "cpu")
    files = discover_safetensors(args.model_dir)
    index = tensor_index(files)
    keys = sorted(index)
    log(f"[q4km] weight_prefix={weight_prefix!r} tensors={len(keys)} out={out_dir}")

    eligible = [
        k for k in keys
        if not k.endswith("_scale_inv")
        and (k.startswith(f"{weight_prefix}.") or k == "lm_head.weight")
        and not k.startswith(f"{weight_prefix}.visual.")
        and ".mtp." not in k
    ]
    tensors_out: list[tuple[str, bytes, list[int], str, str]] = []
    quantized = 0
    for i, name in enumerate(eligible, 1):
        t, dequantized_fp8 = load_source_tensor(index, name)
        if dequantized_fp8 and not is_q4km_target(name, list(t.shape), args.group_size):
            t = t.to(torch.bfloat16)
        if emit_tensor(
            tensors_out,
            name,
            t,
            weight_prefix,
            layer_types,
            args.group_size,
            quantizer,
            hessian_dir,
            gptq_damp,
            gptq_device,
        ):
            quantized += 1
        if i % 100 == 0:
            log(f"[q4km] processed {i}/{len(eligible)} tensors")

    # Tied embeddings (Qwen3.5-0.8B): when the safetensors index has no
    # standalone `lm_head.weight`, synthesize one from the embedding table so
    # we can produce an INT4 lm_head separately. The extra ~half-vocab-sized
    # tensor is worth it — it lets the runtime read INT4 nibbles instead of
    # BF16 on what is the dominant decode-side matmul.
    if "lm_head.weight" not in index:
        embed_name = f"{weight_prefix}.embed_tokens.weight"
        if embed_name in index:
            t, _ = load_source_tensor(index, embed_name)
            if is_q4km_target("lm_head.weight", list(t.shape), args.group_size):
                log(
                    f"[q4km] synthesising lm_head.weight from {embed_name} "
                    f"(tied embeddings) for INT4 lm_head bake"
                )
                if emit_tensor(
                    tensors_out,
                    "lm_head.weight",
                    t,
                    weight_prefix,
                    layer_types,
                    args.group_size,
                    quantizer,
                    hessian_dir,
                    gptq_damp,
                    gptq_device,
                ):
                    quantized += 1

    tensors_out.sort(key=lambda x: x[0])
    log(f"[q4km] quantized {quantized} tensors")
    source_quant = "q4km-gptq-hessian" if quantizer == QUANT_GPTQ else "q4km-minmax"
    quant_profile = "q4km-native-int4-v1"
    write_bake(out_dir, tensors_out, family, "safetensors", source_quant, quant_profile)


def bake_from_gguf(args, weight_prefix: str, layer_types: list[str], family: str, out_dir: Path) -> None:
    quantizer = getattr(args, "quantizer", QUANT_MINMAX)
    hessian_dir = getattr(args, "hessian_dir", None)
    gptq_damp = getattr(args, "gptq_damp", 0.01)
    gptq_device = getattr(args, "gptq_device", "cpu")
    gguf = parse_gguf(args.gguf_file)
    try:
        log(f"[q4km] GGUF v{gguf.version} tensors={len(gguf.tensors)} out={out_dir}")
        tensors_out: list[tuple[str, bytes, list[int], str, str]] = []
        quantized = 0
        skipped = 0
        for i, info in enumerate(gguf.tensors, 1):
            mapped = map_gguf_name(info.name, weight_prefix)
            if mapped is None:
                skipped += 1
                continue
            if mapped.startswith(f"{weight_prefix}.visual.") or ".mtp." in mapped:
                skipped += 1
                continue
            shape = gguf_logical_shape(info)
            raw_layout = ggml_k_layout(info.ggml_type)
            if (
                raw_layout is not None
                # The runtime lm-head path expects BF16 or native INT4 sidecars,
                # not raw GGML K-block bytes.
                and mapped != "lm_head.weight"
                and is_q4km_target(mapped, shape, args.group_size)
            ):
                cols = info.dims[0]
                rows = prod(info.dims[1:]) if len(info.dims) > 1 else 1
                row_bytes = ggml_row_size(info.ggml_type, cols)
                tensors_out.append((
                    mapped,
                    raw_gguf_tensor_bytes(gguf, info),
                    [rows, row_bytes],
                    "u8",
                    raw_layout,
                ))
                quantized += 1
                if i % 100 == 0:
                    log(f"[q4km] processed {i}/{len(gguf.tensors)} GGUF tensors")
                continue
            t = load_gguf_tensor(gguf, info)
            t, a_log_precomputed = undo_gguf_tensor_transform(mapped, t)
            raw_dtype = None
            if info.ggml_type not in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
                raw_dtype = "bf16"
            if emit_tensor(
                tensors_out,
                mapped,
                t,
                weight_prefix,
                layer_types,
                args.group_size,
                quantizer,
                hessian_dir,
                gptq_damp,
                gptq_device,
                a_log_precomputed,
                raw_dtype,
            ):
                quantized += 1
            if i % 100 == 0:
                log(f"[q4km] processed {i}/{len(gguf.tensors)} GGUF tensors")

        if not tensors_out:
            raise SystemExit("GGUF import produced no SuperSonic tensors; unsupported tensor naming?")
        tensors_out.sort(key=lambda x: x[0])
        log(f"[q4km] quantized {quantized} tensors, skipped {skipped} unmapped GGUF tensors")
        source_quant = "ggml-q4-k-family+gptq-hessian" if quantizer == QUANT_GPTQ else "ggml-q4-k-family"
        quant_profile = "q4km-ggml-v1"
        write_bake(out_dir, tensors_out, family, "gguf", source_quant, quant_profile)
    finally:
        gguf.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake SuperSonic q4km weights")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gguf-file", type=Path, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quantizer", choices=[QUANT_MINMAX, QUANT_GPTQ], default=QUANT_MINMAX)
    ap.add_argument("--hessian-dir", type=Path, default=None)
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--gptq-device", default="cpu")
    args = ap.parse_args()

    if args.quantizer == QUANT_GPTQ and args.hessian_dir is None:
        raise SystemExit("--quantizer gptq requires --hessian-dir")

    config, layer_types = load_config_context(args.model_dir)
    if args.gguf_file is not None:
        weight_prefix = "model.language_model"
    else:
        files = discover_safetensors(args.model_dir)
        weight_prefix = infer_weight_prefix(sorted(tensor_index(files)))
    model_name = (args.model or "").lower()
    family = "qwen36-moe" if "35b-a3b" in model_name else "qwen35"
    out_dir = args.out_dir or (args.model_dir / ".supersonic" / f"v{FORMAT_VERSION}-q4km")

    if args.gguf_file is not None:
        bake_from_gguf(args, weight_prefix, layer_types, family, out_dir)
    else:
        bake_from_safetensors(args, weight_prefix, layer_types, family, out_dir)


if __name__ == "__main__":
    main()
