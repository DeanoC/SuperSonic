import math
import unittest

import torch

from oracle.certified_kv_llama31 import (
    CertifiedKvConfig,
    adaptive_topk_mask,
    build_tiered_kv_cache,
    certified_attention_step,
    dequantize_keys,
    dequantize_values_int4,
    quantize_values_int4,
    ranking_consistency_fallback_heads,
    score_blocks_int8,
    score_delta_bound,
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
        self.assertEqual(tuple(cache.key_zeros.shape), (2, 2, 128))
        self.assertEqual(tuple(cache.values_int4_packed.shape), (2, 32, 64))
        self.assertEqual(tuple(cache.value_errors.shape), (2, 2))
        self.assertEqual(tuple(cache.value_norms.shape), (2, 2))

    def test_key_quantization_is_asymmetric_per_block_channel(self):
        keys = torch.zeros(1, 16, 4)
        keys[0, :, 0] = torch.linspace(1.0, 2.0, 16)
        keys[0, :, 1] = torch.linspace(-3.0, -1.0, 16)
        keys[0, :, 2] = 7.0
        keys[0, :, 3] = torch.linspace(-0.25, 0.5, 16)
        values = torch.randn_like(keys)
        cache = build_tiered_kv_cache(keys, values, CertifiedKvConfig(value_group_size=2))
        deq = dequantize_keys(cache)

        self.assertTrue(torch.all(cache.key_zeros[0, 0] != 0.0))
        self.assertTrue(torch.all(cache.keys_int8[0, :, 0] >= -128))
        self.assertLess((deq - keys).abs().max().item(), 0.01)
        self.assertAlmostEqual(cache.key_zeros[0, 0, 2].item(), 7.0, places=5)

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
        cfg = CertifiedKvConfig(
            tau_cov=1.0,
            k_min=2,
            k_max=128,
            v_tol=999.0,
            require_certified_tail_bound=False,
        )
        keys = torch.randn(2, 32, 128)
        values = torch.randn_like(keys)
        q = torch.randn(8, 128)
        cache = build_tiered_kv_cache(keys, values, cfg)

        m_b, s_b = score_blocks_int8(q, cache, gqa_group=4)
        mask, _, _, mass_frac = adaptive_topk_mask(
            m_b, s_b, cfg, score_delta_bound(q, cache.key_scales, gqa_group=4)
        )
        self.assertTrue(mask.all())
        e_val = value_error_bound(mass_frac, cache.value_errors, gqa_group=4)
        out, telemetry = certified_attention_step(q, cache, gqa_group=4)

        # With every key promoted, the only approximation should be INT4 values.
        del out
        actual = torch.tensor(telemetry["actual_l2_error"])
        self.assertTrue(torch.all(actual <= e_val.cpu() + 2e-4), (actual, e_val))
        self.assertIn("true_tail_bound", telemetry)
        self.assertIn("vmax", telemetry)

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

    def test_rung2_value_promotion_zeroes_achieved_e_val_for_promoted_blocks(self):
        cfg = CertifiedKvConfig(v_tol=0.0, require_certified_tail_bound=False)
        keys = torch.randn(2, 32, 128)
        values = torch.randn_like(keys)
        q = torch.randn(8, 128)
        cache = build_tiered_kv_cache(keys, values, cfg)

        _, telemetry = certified_attention_step(q, cache, gqa_group=4)

        self.assertTrue(telemetry["rung2_fired"])
        self.assertTrue(all(v == 0.0 for v in telemetry["e_val"]))

    def test_ranking_boundary_check_uses_delta_not_eps_guard(self):
        int8_m = torch.tensor([[10.0, 9.8]])
        int8_s = torch.ones_like(int8_m)
        fp16_log = torch.tensor([[10.0, 9.0]])
        mask = torch.tensor([[True, False]])
        delta = torch.tensor([[0.0, 0.25]])

        order, boundary, _ = ranking_consistency_fallback_heads(
            int8_m, int8_s, fp16_log, mask, delta, ranking_r=1
        )

        self.assertFalse(order[0].item())
        self.assertTrue(boundary[0].item())

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
