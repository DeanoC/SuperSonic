"""Shared helpers for loading SuperSonic INT4 GPTQ bakes from `weights.bin`
into BF16-reconstructed tensors + parallel (packed, scale, zero) sidecars.

The per-block oracles (`qwen36_moe_oracle.py`, `qwen36_moe_linear_oracle.py`,
`qwen36_moe_ffn_oracle.py`) use this when their `--bake-dir` is set so the
parity tests run against the *bake's actual quantized weights* — the
canonical "does the kernel produce what Python expects on these specific
INT4 reconstructions" check.

Bake naming convention (matches `oracle/bake_int4.py`):
  - Dense projections: `<name>.weight` (packed u8 [out, in/2]) +
    `<name>.weight_int4_scale` + `<name>.weight_int4_zero` (BF16 tiles).
  - Fused experts (no `.weight` suffix in HF state-dict): `<name>` packed +
    `<name>_int4_scale` + `<name>_int4_zero`.
  - Non-quantized: stored as their native dtype (BF16 for norms; F32 for A_log).

Reconstruction matches the kernel's `int4_dequant_scalar` exactly:
`bf16(q*s - z*s)` where (q, s, z) come from the bake's (packed, scale, zero).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


GROUP_SIZE = 128  # pinned across the runtime + bake


@lru_cache(maxsize=8)
def _open_bake(bake_dir: str) -> tuple[dict, bytes]:
    """Read manifest + mmap weights.bin once per bake-dir per process."""
    bake_path = Path(bake_dir)
    manifest = json.loads((bake_path / "manifest.json").read_text())
    # Read weights.bin into memory. For 17 GiB on a 64 GiB host this is fine
    # for a single-layer harness; the per-block oracles only touch a tiny
    # slice. (mmap would be lighter but adds complexity and risks of
    # unaligned-numpy-view bugs across torch versions — keep it simple.)
    weights_data = (bake_path / "weights.bin").read_bytes()
    return manifest, weights_data


def _slab(manifest_idx: dict, data: bytes, name: str) -> tuple[bytes, list[int]]:
    """Return raw bytes + shape for `name` from the bake's index."""
    if name not in manifest_idx:
        raise KeyError(f"tensor not in bake manifest: {name}")
    t = manifest_idx[name]
    return data[t["offset"]:t["offset"] + t["byte_len"]], t["shape"]


def has_tensor(bake_dir: str, name: str) -> bool:
    manifest, _ = _open_bake(bake_dir)
    return any(t["name"] == name for t in manifest["tensors"])


def load_bf16(bake_dir: str, name: str) -> torch.Tensor:
    """Load a BF16 tensor from the bake. Used for norms, conv1d, dt_bias,
    A_log — anything not INT4-quantized."""
    manifest, data = _open_bake(bake_dir)
    idx = {t["name"]: t for t in manifest["tensors"]}
    raw, shape = _slab(idx, data, name)
    arr = np.frombuffer(raw, dtype=np.int16).copy()
    return torch.tensor(arr).view(torch.bfloat16).reshape(shape)


def load_f32(bake_dir: str, name: str) -> torch.Tensor:
    """Load an F32 tensor from the bake. A_log lives here in some
    converters."""
    manifest, data = _open_bake(bake_dir)
    idx = {t["name"]: t for t in manifest["tensors"]}
    raw, shape = _slab(idx, data, name)
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return torch.tensor(arr).reshape(shape)


def load_native(bake_dir: str, name: str) -> torch.Tensor:
    """Dtype-agnostic loader — picks BF16 or F32 based on the manifest's
    declared dtype. Useful for tensors that vary between converters."""
    manifest, data = _open_bake(bake_dir)
    idx = {t["name"]: t for t in manifest["tensors"]}
    if name not in idx:
        raise KeyError(f"tensor not in bake manifest: {name}")
    t = idx[name]
    raw = data[t["offset"]:t["offset"] + t["byte_len"]]
    shape = t["shape"]
    dt = t["dtype"]
    if dt == "bf16":
        arr = np.frombuffer(raw, dtype=np.int16).copy()
        return torch.tensor(arr).view(torch.bfloat16).reshape(shape)
    if dt == "f32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.tensor(arr).reshape(shape)
    if dt == "u8":
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        return torch.tensor(arr).reshape(shape)
    raise ValueError(f"unsupported dtype {dt} for {name}")


def load_int4(
    bake_dir: str,
    name: str,
    sidecar_suffix_style: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a 2D INT4-quantized tensor from the bake and return
    `(recon, packed, scale, zero)`:

      recon  : BF16 [out, in]      — `bf16(q*s - z*s)` matching the kernel.
      packed : u8   [out, in/2]    — raw nibble-packed bytes from the bake.
      scale  : BF16 [out/gs, in/gs]
      zero   : BF16 [out/gs, in/gs]

    `sidecar_suffix_style`:
      'dotweight'  → `<name>_int4_scale` (e.g. `q_proj.weight_int4_scale`)
      'noweight'   → `<name>_int4_scale` (e.g. `experts.gate_up_proj_int4_scale`)
      'auto'       → infer from `name`: if it ends in `.weight`, use
                     `<name>_int4_scale` (which becomes
                     `<base>.weight_int4_scale`); else
                     `<name>_int4_scale` (raw experts).
                     In both cases the literal sidecar name is
                     `f"{name}_int4_scale"` — bake just glues the suffix on.
    """
    if sidecar_suffix_style not in ("auto", "dotweight", "noweight"):
        raise ValueError(f"unknown sidecar_suffix_style: {sidecar_suffix_style}")

    manifest, data = _open_bake(bake_dir)
    idx = {t["name"]: t for t in manifest["tensors"]}

    p_raw, ps = _slab(idx, data, name)
    s_raw, ss = _slab(idx, data, f"{name}_int4_scale")
    z_raw, _ = _slab(idx, data, f"{name}_int4_zero")

    out_dim, in_half = ps[-2], ps[-1]
    in_dim = in_half * 2
    sr, sc = ss[-2], ss[-1]
    if sr * GROUP_SIZE != out_dim:
        raise ValueError(
            f"{name}: scale rows {sr} × gs {GROUP_SIZE} != out_dim {out_dim}"
        )
    if sc * GROUP_SIZE != in_dim:
        raise ValueError(
            f"{name}: scale cols {sc} × gs {GROUP_SIZE} != in_dim {in_dim}"
        )

    packed = np.frombuffer(p_raw, dtype=np.uint8).copy().reshape(out_dim, in_half)
    scale_bf = (
        torch.tensor(np.frombuffer(s_raw, dtype=np.int16).copy())
        .view(torch.bfloat16)
        .reshape(sr, sc)
    )
    zero_bf = (
        torch.tensor(np.frombuffer(z_raw, dtype=np.int16).copy())
        .view(torch.bfloat16)
        .reshape(sr, sc)
    )

    # Reconstruct BF16 weights via the kernel's bf16(q*s - z*s) formula.
    s_f32 = scale_bf.to(torch.float32)
    z_f32 = zero_bf.to(torch.float32)
    q = np.empty((out_dim, in_dim), dtype=np.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = (packed >> 4) & 0x0F
    s_full = s_f32.repeat_interleave(GROUP_SIZE, 0).repeat_interleave(GROUP_SIZE, 1)
    z_full = z_f32.repeat_interleave(GROUP_SIZE, 0).repeat_interleave(GROUP_SIZE, 1)
    recon = (
        (torch.tensor(q, dtype=torch.float32) * s_full - z_full * s_full)
        .to(torch.bfloat16)
    )

    return recon, torch.tensor(packed), scale_bf, zero_bf


def load_int4_3d(
    bake_dir: str,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like `load_int4` but for 3D fused-expert tensors (`[E, out, in/2]`
    packed + `[E, out/gs, in/gs]` scale/zero). Returns recon `[E, out, in]`,
    packed `[E, out, in/2]`, scale/zero `[E, out/gs, in/gs]`."""
    manifest, data = _open_bake(bake_dir)
    idx = {t["name"]: t for t in manifest["tensors"]}

    p_raw, ps = _slab(idx, data, name)
    s_raw, ss = _slab(idx, data, f"{name}_int4_scale")
    z_raw, _ = _slab(idx, data, f"{name}_int4_zero")

    e, out_dim, in_half = ps
    in_dim = in_half * 2
    e2, sr, sc = ss
    if e != e2:
        raise ValueError(f"{name}: expert axis mismatch {e} vs {e2}")
    if sr * GROUP_SIZE != out_dim or sc * GROUP_SIZE != in_dim:
        raise ValueError(f"{name}: scale shape doesn't match group_size")

    packed = (
        np.frombuffer(p_raw, dtype=np.uint8).copy().reshape(e, out_dim, in_half)
    )
    scale_bf = (
        torch.tensor(np.frombuffer(s_raw, dtype=np.int16).copy())
        .view(torch.bfloat16)
        .reshape(e, sr, sc)
    )
    zero_bf = (
        torch.tensor(np.frombuffer(z_raw, dtype=np.int16).copy())
        .view(torch.bfloat16)
        .reshape(e, sr, sc)
    )

    s_f32 = scale_bf.to(torch.float32)
    z_f32 = zero_bf.to(torch.float32)
    recon = torch.empty((e, out_dim, in_dim), dtype=torch.bfloat16)
    for j in range(e):
        q = np.empty((out_dim, in_dim), dtype=np.uint8)
        q[:, 0::2] = packed[j] & 0x0F
        q[:, 1::2] = (packed[j] >> 4) & 0x0F
        s_full = s_f32[j].repeat_interleave(GROUP_SIZE, 0).repeat_interleave(GROUP_SIZE, 1)
        z_full = z_f32[j].repeat_interleave(GROUP_SIZE, 0).repeat_interleave(GROUP_SIZE, 1)
        recon[j] = (
            (torch.tensor(q, dtype=torch.float32) * s_full - z_full * s_full)
            .to(torch.bfloat16)
        )
    return recon, torch.tensor(packed), scale_bf, zero_bf
