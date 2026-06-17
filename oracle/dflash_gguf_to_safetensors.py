#!/usr/bin/env python3
"""Convert a Lucebox DFlash draft GGUF into SuperSonic's DFlash safetensors layout.

The SuperSonic DFlash draft loader consumes a HF-style directory:

  config.json
  model.safetensors

Lucebox publishes Qwen3.6-27B drafters as GGUF. This converter handles the
Q8_0 draft path used by the apples-to-apples HumanEval benchmark and writes
BF16 tensors with the names expected by crates/qwen35_dflash.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

DEFAULT_QWEN36_27B_TAPS = [1, 16, 31, 46, 61]


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int


class GgufReader:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0

    def close(self) -> None:
        self.data.close()
        self.file.close()

    def read(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        if len(out) != n:
            raise EOFError(f"{self.path}: short read")
        self.pos += n
        return out

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        return self.read(n).decode("utf-8")

    def value(self, ty: int):
        if ty == GGUF_TYPE_UINT8:
            return self.unpack("<B")
        if ty == GGUF_TYPE_INT8:
            return self.unpack("<b")
        if ty == GGUF_TYPE_UINT16:
            return self.unpack("<H")
        if ty == GGUF_TYPE_INT16:
            return self.unpack("<h")
        if ty == GGUF_TYPE_UINT32:
            return self.unpack("<I")
        if ty == GGUF_TYPE_INT32:
            return self.unpack("<i")
        if ty == GGUF_TYPE_FLOAT32:
            return self.unpack("<f")
        if ty == GGUF_TYPE_BOOL:
            return self.unpack("<?")
        if ty == GGUF_TYPE_STRING:
            return self.string()
        if ty == GGUF_TYPE_UINT64:
            return self.unpack("<Q")
        if ty == GGUF_TYPE_INT64:
            return self.unpack("<q")
        if ty == GGUF_TYPE_FLOAT64:
            return self.unpack("<d")
        if ty == GGUF_TYPE_ARRAY:
            elem_ty = self.unpack("<I")
            n = self.unpack("<Q")
            return [self.value(elem_ty) for _ in range(n)]
        raise ValueError(f"unsupported GGUF metadata type {ty}")


@dataclass
class GgufFile:
    version: int
    metadata: dict[str, object]
    tensors: list[TensorInfo]
    data_start: int
    data: mmap.mmap
    reader: GgufReader

    def close(self) -> None:
        self.reader.close()


def align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def parse_gguf(path: Path) -> GgufFile:
    r = GgufReader(path)
    if r.read(4) != b"GGUF":
        raise SystemExit(f"{path} is not a GGUF file")
    version = r.unpack("<I")
    if version != 3:
        raise SystemExit(f"{path}: unsupported GGUF version {version}")
    tensor_count = r.unpack("<Q")
    kv_count = r.unpack("<Q")
    metadata = {}
    for _ in range(kv_count):
        key = r.string()
        ty = r.unpack("<I")
        metadata[key] = r.value(ty)
    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        ndims = r.unpack("<I")
        dims = [int(r.unpack("<Q")) for _ in range(ndims)]
        ggml_type = int(r.unpack("<I"))
        offset = int(r.unpack("<Q"))
        tensors.append(TensorInfo(name, dims, ggml_type, offset))
    alignment = int(metadata.get("general.alignment", 32))
    data_start = align_up(r.pos, alignment)
    return GgufFile(version, metadata, tensors, data_start, r.data, r)


def gguf_shape(info: TensorInfo) -> list[int]:
    if len(info.dims) == 1:
        return [info.dims[0]]
    return list(reversed(info.dims))


def gguf_nbytes(info: TensorInfo) -> int:
    cols = info.dims[0]
    rows = int(np.prod(info.dims[1:], dtype=np.int64)) if len(info.dims) > 1 else 1
    if info.ggml_type == GGML_TYPE_F32:
        return rows * cols * 4
    if info.ggml_type == GGML_TYPE_F16:
        return rows * cols * 2
    if info.ggml_type == GGML_TYPE_Q8_0:
        if cols % 32 != 0:
            raise SystemExit(f"{info.name}: Q8_0 cols {cols} not divisible by 32")
        return rows * (cols // 32) * 34
    raise SystemExit(f"{info.name}: unsupported GGML type {info.ggml_type}; use the Q8_0 draft")


def raw_tensor(g: GgufFile, info: TensorInfo) -> bytes:
    start = g.data_start + info.offset
    end = start + gguf_nbytes(info)
    if end > len(g.data):
        raise SystemExit(f"{info.name}: tensor overruns GGUF file")
    return g.data[start:end]


def f32_to_bf16_bytes(x: np.ndarray) -> bytes:
    f32 = np.asarray(x, dtype="<f4")
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounded = u32 + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    bf16 = (rounded >> 16).astype("<u2", copy=False)
    return bf16.tobytes(order="C")


def dequant_q8_0(raw: bytes, rows: int, cols: int) -> np.ndarray:
    blocks = cols // 32
    q = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks, 34)
    d = q[:, :, 0:2].copy().view("<f2").reshape(rows, blocks).astype(np.float32)
    qs = q[:, :, 2:34].copy().view(np.int8).astype(np.float32)
    return (qs * d[:, :, None]).reshape(rows, cols)


def tensor_to_bf16(info: TensorInfo, raw: bytes) -> bytes:
    shape = gguf_shape(info)
    if info.ggml_type == GGML_TYPE_F32:
        return f32_to_bf16_bytes(np.frombuffer(raw, dtype="<f4").reshape(shape))
    if info.ggml_type == GGML_TYPE_F16:
        return f32_to_bf16_bytes(np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32))
    if info.ggml_type == GGML_TYPE_Q8_0:
        rows = shape[0]
        cols = shape[1]
        return f32_to_bf16_bytes(dequant_q8_0(raw, rows, cols))
    raise AssertionError(info.ggml_type)


def map_name(name: str) -> str | None:
    if name == "dflash.fc.weight":
        return "fc.weight"
    if name == "dflash.hidden_norm.weight":
        return "hidden_norm.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if not name.startswith("blk."):
        return None
    parts = name.split(".")
    if len(parts) < 4:
        return None
    layer = int(parts[1])
    suffix = ".".join(parts[2:])
    mapped = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }.get(suffix)
    if mapped is None:
        return None
    return f"layers.{layer}.{mapped}"


def write_safetensors(path: Path, entries: list[tuple[str, list[int], TensorInfo, bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape, _info, data in entries:
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _name, _shape, _info, data in entries:
            f.write(data)


def make_config(g: GgufFile, target_layer_ids: list[int], num_target_layers: int) -> dict[str, object]:
    arch = str(g.metadata["general.architecture"])
    p = f"{arch}."
    head_dim = int(g.metadata.get(p + "attention.key_length", 128))
    return {
        "vocab_size": int(g.metadata[p + "vocab_size"]),
        "hidden_size": int(g.metadata[p + "embedding_length"]),
        "intermediate_size": int(g.metadata[p + "feed_forward_length"]),
        "num_hidden_layers": int(g.metadata[p + "block_count"]),
        "num_attention_heads": int(g.metadata[p + "attention.head_count"]),
        "num_key_value_heads": int(g.metadata[p + "attention.head_count_kv"]),
        "head_dim": head_dim,
        "max_position_embeddings": int(g.metadata[p + "context_length"]),
        "rope_theta": float(g.metadata[p + "rope.freq_base"]),
        "rms_norm_eps": float(g.metadata[p + "attention.layer_norm_rms_epsilon"]),
        "block_size": int(g.metadata[p + "dflash.block_size"]),
        "num_target_layers": num_target_layers,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "dflash_config": {
            "mask_token_id": int(g.metadata[p + "dflash.mask_token_id"]),
            "target_layer_ids": target_layer_ids,
        },
    }


def parse_taps(raw: str) -> list[int]:
    vals = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("tap list must not be empty")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--target-layer-ids",
        type=parse_taps,
        default=DEFAULT_QWEN36_27B_TAPS,
        help="Comma-separated target tap layer IDs. Default: Qwen3.6-27B Lucebox taps.",
    )
    ap.add_argument(
        "--num-target-layers",
        type=int,
        default=64,
        help="Target model layer count for config validation. Default: Qwen3.6-27B (64).",
    )
    args = ap.parse_args()

    g = parse_gguf(args.gguf)
    try:
        by_name = {info.name: info for info in g.tensors}
        entries = []
        for src_name in sorted(by_name):
            dst_name = map_name(src_name)
            if dst_name is None:
                continue
            info = by_name[src_name]
            if info.ggml_type not in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0):
                raise SystemExit(
                    f"{src_name}: GGML type {info.ggml_type} is not supported by this BF16 converter"
                )
            print(f"[dflash-gguf] {src_name} -> {dst_name} {gguf_shape(info)}")
            data = tensor_to_bf16(info, raw_tensor(g, info))
            entries.append((dst_name, gguf_shape(info), info, data))
        if len(entries) != 58:
            raise SystemExit(f"expected 58 mapped DFlash tensors, got {len(entries)}")
        write_safetensors(args.out_dir / "model.safetensors", entries)
        config = make_config(g, args.target_layer_ids, args.num_target_layers)
        (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"[dflash-gguf] wrote {args.out_dir}")
    finally:
        g.close()


if __name__ == "__main__":
    main()
