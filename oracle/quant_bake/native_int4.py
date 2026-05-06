from __future__ import annotations

import numpy as np


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """Pack int4 values with low nibble = even column, matching the runtime."""
    if q.ndim != 2:
        raise ValueError("pack_nibbles expects a 2D [rows, cols] array")
    rows, cols = q.shape
    padded_cols = cols + (cols & 1)
    if padded_cols != cols:
        q = np.pad(q, ((0, 0), (0, 1)), constant_values=0)
    q = np.clip(q.astype(np.uint8), 0, 15)
    lo = q[:, 0::2]
    hi = q[:, 1::2] << 4
    return (lo | hi).reshape(rows, padded_cols // 2)


def unpack_nibbles(packed: np.ndarray, cols: int) -> np.ndarray:
    if packed.ndim != 2:
        raise ValueError("unpack_nibbles expects a 2D [rows, packed_cols] array")
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out[:, :cols]


def minmax_quantize(
    weight: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small CPU reference quantizer for tests and method prototyping.

    Returns packed uint8 weights plus float32 scale and zero arrays shaped as
    [ceil(rows/group_size), ceil(cols/group_size)].
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D [rows, cols] matrix")
    w = weight.astype(np.float32, copy=False)
    rows, cols = w.shape
    scale = np.empty(
        ((rows + group_size - 1) // group_size, (cols + group_size - 1) // group_size),
        dtype=np.float32,
    )
    zero = np.empty_like(scale)
    q = np.empty((rows, cols), dtype=np.uint8)
    for r0 in range(0, rows, group_size):
        r1 = min(r0 + group_size, rows)
        rg = r0 // group_size
        for c0 in range(0, cols, group_size):
            c1 = min(c0 + group_size, cols)
            cg = c0 // group_size
            block = w[r0:r1, c0:c1]
            mn = float(block.min())
            mx = float(block.max())
            s = max((mx - mn) / 15.0, 1.0e-8)
            z = -mn / s
            scale[rg, cg] = s
            zero[rg, cg] = z
            q[r0:r1, c0:c1] = np.clip(np.rint(block / s + z), 0, 15).astype(np.uint8)
    return pack_nibbles(q), scale, zero


def hqq_lsq_quantize(
    weight: np.ndarray,
    group_size: int = 128,
    iters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data-free HQQ-style alternating least-squares INT4 quantizer.

    This keeps SuperSonic's native asymmetric INT4 runtime layout, but refines
    each tile's scale and zero after an initial min/max estimate by alternating:
    quantize with current `(scale, zero)`, then solve the best affine
    reconstruction `w ~= scale*q + bias` in least-squares form and convert
    `bias` back to `zero=-bias/scale`.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D [rows, cols] matrix")
    w = weight.astype(np.float32, copy=False)
    rows, cols = w.shape
    scale = np.empty(
        ((rows + group_size - 1) // group_size, (cols + group_size - 1) // group_size),
        dtype=np.float32,
    )
    zero = np.empty_like(scale)
    q = np.empty((rows, cols), dtype=np.uint8)
    for r0 in range(0, rows, group_size):
        r1 = min(r0 + group_size, rows)
        rg = r0 // group_size
        for c0 in range(0, cols, group_size):
            c1 = min(c0 + group_size, cols)
            cg = c0 // group_size
            block = w[r0:r1, c0:c1]
            mn = float(block.min())
            mx = float(block.max())
            s = max((mx - mn) / 15.0, 1.0e-8)
            z = -mn / s
            q_block = np.zeros(block.shape, dtype=np.float32)
            for _ in range(max(1, iters)):
                q_block = np.clip(np.rint(block / s + z), 0, 15).astype(np.float32)
                q_mean = float(q_block.mean())
                w_mean = float(block.mean())
                q_centered = q_block - q_mean
                w_centered = block - w_mean
                denom = float((q_centered * q_centered).sum())
                if denom <= 1.0e-12:
                    continue
                s_new = float((q_centered * w_centered).sum()) / denom
                if not np.isfinite(s_new) or s_new <= 1.0e-8:
                    continue
                bias = w_mean - s_new * q_mean
                s = s_new
                z = -bias / s
            scale[rg, cg] = s
            zero[rg, cg] = z
            q[r0:r1, c0:c1] = np.clip(np.rint(block / s + z), 0, 15).astype(np.uint8)
    return pack_nibbles(q), scale, zero


def dequantize_native_int4(
    packed: np.ndarray,
    scale: np.ndarray,
    zero: np.ndarray,
    cols: int,
    group_size: int = 128,
) -> np.ndarray:
    q = unpack_nibbles(packed, cols).astype(np.float32)
    rows = q.shape[0]
    out = np.empty((rows, cols), dtype=np.float32)
    for r0 in range(0, rows, group_size):
        r1 = min(r0 + group_size, rows)
        rg = r0 // group_size
        for c0 in range(0, cols, group_size):
            c1 = min(c0 + group_size, cols)
            cg = c0 // group_size
            out[r0:r1, c0:c1] = (
                q[r0:r1, c0:c1] * scale[rg, cg] - zero[rg, cg] * scale[rg, cg]
            )
    return out
