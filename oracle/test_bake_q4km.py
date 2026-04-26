#!/usr/bin/env python3

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_q4km


def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _metadata_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _metadata_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    out = bytearray()
    out += _gguf_string(name)
    out += struct.pack("<I", len(dims))
    for dim in dims:
        out += struct.pack("<Q", dim)
    out += struct.pack("<I", ggml_type)
    out += struct.pack("<Q", offset)
    return bytes(out)


def _q4_k_block(value: int = 3, scale: float = 0.5) -> bytes:
    # d, dmin, 12 scale/min bytes, then 128 packed q4 bytes.
    # scale byte 1 means every nibble dequantizes to d * value.
    return (
        struct.pack("<e", scale)
        + struct.pack("<e", 0.0)
        + bytes([1] * 8 + [0] * 4)
        + bytes([(value & 0xF) | ((value & 0xF) << 4)] * 128)
    )


def _q8_0_block(values: list[int] | None = None, scale: float = 0.25) -> bytes:
    vals = values or list(range(-16, 16))
    if len(vals) != bake_q4km.QK_0:
        raise ValueError("Q8_0 synthetic block must have exactly 32 values")
    return struct.pack("<e", scale) + struct.pack("32b", *vals)


def write_tiny_gguf(path: Path) -> None:
    tensors = [
        ("token_embd.weight", [4, 2], bake_q4km.GGML_TYPE_F32, torch.arange(8, dtype=torch.float32).reshape(2, 4).numpy().tobytes()),
        ("output_norm.weight", [4], bake_q4km.GGML_TYPE_F32, torch.full((4,), 2.0, dtype=torch.float32).numpy().tobytes()),
        ("blk.0.ffn_gate.weight", [256, 1], bake_q4km.GGML_TYPE_Q4_K, _q4_k_block()),
        ("blk.0.ffn_up.weight", [32, 1], bake_q4km.GGML_TYPE_Q8_0, _q8_0_block()),
    ]
    offset = 0
    infos = []
    data = bytearray()
    for name, dims, ggml_type, blob in tensors:
        infos.append(_tensor_info(name, dims, ggml_type, offset))
        data += blob
        offset += len(blob)

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(tensors), 3)
    header += _metadata_string("general.architecture", "qwen3next")
    header += _metadata_u32("general.alignment", 32)
    header += _metadata_string("general.file_type", "synthetic")
    for info in infos:
        header += info
    padding = bytes(b"\0" * (bake_q4km.align_up(len(header), 32) - len(header)))
    path.write_bytes(bytes(header) + padding + bytes(data))


class BakeQ4KmGgufTest(unittest.TestCase):
    def test_tiny_gguf_bake_writes_native_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            out_dir = root / "out"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "layer_types": ["full_attention"]}))
            gguf_path = root / "tiny.gguf"
            write_tiny_gguf(gguf_path)

            args = type("Args", (), {"gguf_file": gguf_path, "group_size": 128})()
            _, layer_types = bake_q4km.load_config_context(model_dir)
            bake_q4km.bake_from_gguf(args, "model.language_model", layer_types, "qwen35", out_dir)

            manifest = json.loads((out_dir / "manifest.json").read_text())
            by_name = {entry["name"]: entry for entry in manifest["tensors"]}
            self.assertEqual(manifest["source_format"], "gguf")
            self.assertEqual(manifest["source_quant"], "ggml-q4-k-family")
            self.assertEqual(by_name["model.language_model.embed_tokens.weight"]["shape"], [2, 4])
            self.assertEqual(by_name["model.language_model.norm.weight"]["dtype"], "f32")
            self.assertEqual(by_name["model.language_model.layers.0.mlp.gate_proj.weight"]["layout"], "Int4Quantized")
            self.assertEqual(by_name["model.language_model.layers.0.mlp.gate_proj.weight"]["shape"], [1, 128])
            self.assertEqual(by_name["model.language_model.layers.0.mlp.gate_proj.weight_int4_scale"]["shape"], [1, 2])
            self.assertEqual(by_name["model.language_model.layers.0.mlp.up_proj.weight"]["dtype"], "bf16")
            self.assertEqual(by_name["model.language_model.layers.0.mlp.up_proj.weight"]["shape"], [1, 32])

    def test_gptq_quantizer_uses_hessian_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hdir = Path(td)
            name = "model.language_model.layers.0.mlp.gate_proj.weight"
            W = torch.linspace(-1.0, 1.0, 128 * 256, dtype=torch.float32).reshape(128, 256)
            H = torch.eye(256, dtype=torch.float32)
            torch.save({"H": H}, hdir / f"{bake_q4km.sanitize_tensor_name(name)}.pt")

            out: list[tuple[str, bytes, list[int], str, str]] = []
            quantized = bake_q4km.emit_tensor(
                out,
                name,
                W,
                "model.language_model",
                ["full_attention"],
                128,
                bake_q4km.QUANT_GPTQ,
                hdir,
                0.01,
                "cpu",
            )

            self.assertTrue(quantized)
            by_name = {entry[0]: entry for entry in out}
            self.assertEqual(by_name[name][2], [128, 128])
            self.assertEqual(by_name[name][3], "u8")
            self.assertEqual(by_name[f"{name}_int4_scale"][2], [1, 2])
            self.assertEqual(by_name[f"{name}_int4_zero"][2], [1, 2])


if __name__ == "__main__":
    unittest.main()
