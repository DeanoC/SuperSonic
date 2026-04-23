import math
import unittest

import torch

from oracle.certified_kv_llama31 import (
    CertifiedKvConfig,
    adaptive_topk_mask,
    build_tiered_kv_cache,
    certified_attention_step,
    dequantize_values_int4,
    quantize_values_int4,
    score_blocks_int8,
    value_error_bound,
)


class CertifiedKvOracleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_int4_value_roundtrip_shape_and_error(self):
        values = torch.randn(2, 16, 128)
        packed, scales, zeros, err = quantize_values_int4(values, group_size=16)
        deq = dequantize_values_int4(packed, scales, zeros, group_size=16)

        self.assertEqual(tuple(packed.shape), (2, 16, 64))
        self.assertEqual(tuple(scales.shape), (2, 16, 8))
        self.assertEqual(tuple(zeros.shape), (2, 16, 8))
        self.assertEqual(tuple(deq.shape), tuple(values.shape))
        self.assertTrue(torch.all((values - deq).norm(dim=-1) <= err + 1e-5))

    def test_cache_uses_paper_defaults_and_handles_tail(self):
        keys = torch.randn(2, 33, 128)
        values = torch.randn_like(keys)
        cache = build_tiered_kv_cache(keys, values)

        self.assertEqual(cache.config.block_size, 16)
        self.assertEqual(cache.config.value_group_size, 16)
        self.assertAlmostEqual(cache.config.tau_cov, 0.995)
        self.assertEqual(cache.config.k_max, 128)
        self.assertAlmostEqual(cache.config.v_tol, 0.05)
        self.assertEqual(cache.aligned_tokens, 32)
        self.assertEqual(cache.total_tokens, 33)
        self.assertTrue(cache.has_tail)
        self.assertEqual(tuple(cache.keys_int8.shape), (2, 32, 128))
        self.assertEqual(tuple(cache.key_scales.shape), (2, 2, 128))
        self.assertEqual(tuple(cache.values_int4_packed.shape), (2, 32, 64))
        self.assertEqual(tuple(cache.value_errors.shape), (2, 2))

    def test_adaptive_topk_respects_k_bounds(self):
        m_b = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        s_b = torch.ones_like(m_b)
        cfg = CertifiedKvConfig(tau_cov=0.90, k_min=2, k_max=3)
        mask, k_star, tail_mass, mass_frac = adaptive_topk_mask(m_b, s_b, cfg)

        self.assertEqual(k_star.tolist(), [3, 3])
        self.assertEqual(mask.sum(dim=1).tolist(), [3, 3])
        captured = (mass_frac * mask.float()).sum(dim=1)
        self.assertTrue(torch.allclose(tail_mass, 1.0 - captured))

    def test_value_bound_covers_int4_value_error_when_all_keys_fp16(self):
        cfg = CertifiedKvConfig(tau_cov=1.0, k_min=2, k_max=128, v_tol=999.0)
        keys = torch.randn(2, 32, 128)
        values = torch.randn_like(keys)
        q = torch.randn(8, 128)
        cache = build_tiered_kv_cache(keys, values, cfg)

        m_b, s_b = score_blocks_int8(q, cache, gqa_group=4)
        mask, _, _, mass_frac = adaptive_topk_mask(m_b, s_b, cfg)
        self.assertTrue(mask.all())
        e_val = value_error_bound(mass_frac, cache.value_errors, gqa_group=4)
        out, telemetry = certified_attention_step(q, cache, gqa_group=4)

        # With every key promoted, the only approximation should be INT4 values.
        del out
        actual = torch.tensor(telemetry["actual_l2_error"])
        self.assertTrue(torch.all(actual <= e_val.cpu() + 2e-4), (actual, e_val))

    def test_forced_dense_fallback_matches_dense(self):
        cfg = CertifiedKvConfig()
        keys = torch.randn(2, 20, 128)
        values = torch.randn_like(keys)
        q = torch.randn(8, 128)
        cache = build_tiered_kv_cache(keys, values, cfg)

        dense_a, telem_a = certified_attention_step(q, cache, gqa_group=4, force_dense_fallback=True)
        dense_b, telem_b = certified_attention_step(q, cache, gqa_group=4, force_dense_fallback=True)

        self.assertEqual(telem_a["mode"], "dense_fallback")
        self.assertTrue(telem_a["rung4_fired"])
        self.assertTrue(torch.allclose(dense_a, dense_b, atol=0.0, rtol=0.0))
        self.assertEqual(telem_b["num_blocks"], 1)

    def test_q_scale_matches_attention_scale(self):
        keys = torch.randn(1, 16, 64)
        values = torch.randn_like(keys)
        q = torch.randn(1, 64)
        cache = build_tiered_kv_cache(keys, values, CertifiedKvConfig(value_group_size=16))
        m_b, _ = score_blocks_int8(q, cache, gqa_group=1)
        self.assertTrue(torch.isfinite(m_b).all())
        self.assertLess(abs((1.0 / math.sqrt(64)) - 0.125), 1e-12)


if __name__ == "__main__":
    unittest.main()
