import numpy as np

from oracle.quant_bake.calibration import cache_path, sample_windows
from oracle.quant_bake.native_int4 import (
    dequantize_native_int4,
    hqq_lsq_quantize,
    minmax_quantize,
    pack_nibbles,
    unpack_nibbles,
)
from oracle.quant_bake.profiles import parse_profile


def test_pack_unpack_nibbles_odd_cols():
    q = np.arange(15, dtype=np.uint8).reshape(3, 5)
    packed = pack_nibbles(q)
    assert packed.shape == (3, 3)
    np.testing.assert_array_equal(unpack_nibbles(packed, 5), q)


def test_minmax_quantize_reconstructs_shape():
    rng = np.random.default_rng(123)
    w = rng.normal(size=(9, 17)).astype(np.float32)
    packed, scale, zero = minmax_quantize(w, group_size=8)
    out = dequantize_native_int4(packed, scale, zero, cols=w.shape[1], group_size=8)
    assert out.shape == w.shape
    assert np.isfinite(out).all()


def test_hqq_lsq_quantize_reconstructs_shape_and_is_deterministic():
    rng = np.random.default_rng(321)
    w = rng.normal(size=(11, 19)).astype(np.float32)
    packed_a, scale_a, zero_a = hqq_lsq_quantize(w, group_size=8, iters=3)
    packed_b, scale_b, zero_b = hqq_lsq_quantize(w, group_size=8, iters=3)
    out = dequantize_native_int4(packed_a, scale_a, zero_a, cols=w.shape[1], group_size=8)
    assert out.shape == w.shape
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(packed_a, packed_b)
    np.testing.assert_allclose(scale_a, scale_b)
    np.testing.assert_allclose(zero_a, zero_b)


def test_profile_aliases():
    assert parse_profile("awq").name == "int4-awq"
    assert parse_profile("signround").name == "int4-autoround"
    assert parse_profile("qtip").layout == "QtipTrellisQuantized"


def test_calibration_windows_are_deterministic(tmp_path):
    ids = list(range(100))
    a = sample_windows(ids, num_samples=4, seqlen=8, seed=11)
    b = sample_windows(ids, num_samples=4, seqlen=8, seed=11)
    c = sample_windows(ids, num_samples=4, seqlen=8, seed=12)
    assert a == b
    assert a != c
    assert all(len(row) == 8 for row in a)
    assert cache_path(tmp_path, "wikitext/a:b", 4, 8, 11).name == "wikitext_a_b-n4-t8-seed11.json"
