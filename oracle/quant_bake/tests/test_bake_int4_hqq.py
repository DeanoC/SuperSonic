import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from oracle.bake_int4 import (
    _selfcheck_dense,
    autoround_quantize_torch,
    awq_channel_scale_torch,
    awq_quantize_with_sidecar_torch,
    awq_quantize_torch,
    hqq_lsq_quantize_torch,
    pack_nibbles,
)


def test_hqq_lsq_quantize_torch_native_int4_shapes():
    torch.manual_seed(7)
    w = torch.randn(16, 24, dtype=torch.float32)
    q_dq, nibbles, scale, zero = hqq_lsq_quantize_torch(w, group_size=8, iters=3)
    packed = pack_nibbles(nibbles)
    assert q_dq.shape == w.shape
    assert nibbles.shape == w.shape
    assert packed.shape == (16, 12)
    assert scale.shape == (2, 3)
    assert zero.shape == (2, 3)
    assert int(nibbles.min()) >= 0
    assert int(nibbles.max()) <= 15
    assert torch.isfinite(q_dq).all()


def test_awq_quantize_torch_native_int4_shapes():
    torch.manual_seed(13)
    w = torch.randn(16, 24, dtype=torch.float32)
    act = torch.linspace(0.2, 2.0, steps=24)
    q_dq, nibbles, scale, zero = awq_quantize_torch(w, act, group_size=8)
    packed = pack_nibbles(nibbles)
    assert q_dq.shape == w.shape
    assert nibbles.shape == w.shape
    assert packed.shape == (16, 12)
    assert scale.shape == (2, 3)
    assert zero.shape == (2, 3)
    assert int(nibbles.min()) >= 0
    assert int(nibbles.max()) <= 15
    assert torch.isfinite(q_dq).all()


def test_awq_quantize_with_sidecar_native_int4_shapes():
    torch.manual_seed(14)
    w = torch.randn(16, 24, dtype=torch.float32)
    act = torch.linspace(0.2, 2.0, steps=24)
    q_dq, nibbles, scale, zero, inv_scale = awq_quantize_with_sidecar_torch(
        w, act, group_size=8
    )
    assert q_dq.shape == w.shape
    assert nibbles.shape == w.shape
    assert scale.shape == (2, 3)
    assert zero.shape == (2, 3)
    assert inv_scale.shape == (24,)
    assert int(nibbles.min()) >= 0
    assert int(nibbles.max()) <= 15
    assert torch.isfinite(q_dq).all()
    assert torch.isfinite(inv_scale).all()
    matched, linf = _selfcheck_dense(
        nibbles.cpu(),
        scale.cpu(),
        zero.cpu(),
        q_dq.cpu(),
        group_size=8,
        awq_inv_scale=inv_scale.cpu(),
    )
    assert matched, linf


def test_awq_sidecar_scale_modes_are_finite_and_distinct():
    torch.manual_seed(15)
    w = torch.randn(16, 24, dtype=torch.float32)
    act = torch.linspace(0.2, 2.0, steps=24)
    activation = awq_channel_scale_torch(act, W=w, mode="activation")
    inverse = awq_channel_scale_torch(act, W=w, mode="inverse-activation")
    weight = awq_channel_scale_torch(act, W=w, mode="activation-weight")
    assert torch.isfinite(activation).all()
    assert torch.isfinite(inverse).all()
    assert torch.isfinite(weight).all()
    assert not torch.allclose(activation, inverse)
    assert not torch.allclose(activation, weight)


def test_autoround_quantize_torch_native_int4_shapes():
    torch.manual_seed(17)
    w = torch.randn(16, 24, dtype=torch.float32)
    x = torch.randn(10, 24, dtype=torch.float32)
    q_dq, nibbles, scale, zero, optimized = autoround_quantize_torch(
        w,
        x,
        group_size=8,
        steps=2,
        row_chunk=8,
        max_optim_elements=10_000,
    )
    packed = pack_nibbles(nibbles)
    assert optimized
    assert q_dq.shape == w.shape
    assert nibbles.shape == w.shape
    assert packed.shape == (16, 12)
    assert scale.shape == (2, 3)
    assert zero.shape == (2, 3)
    assert int(nibbles.min()) >= 0
    assert int(nibbles.max()) <= 15
    assert torch.isfinite(q_dq).all()
