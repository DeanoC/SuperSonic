import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "kernels" / "full_attention_bridge_4b.cpp"


class Qwen38BridgeIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BRIDGE.read_text(encoding="utf-8")

    def test_persistent_decode_joins_final_launch_and_sync_fail_stop(self):
        start = self.source.index("int persistent_decode_device(")
        end = self.source.index("// Restore conv+rec", start)
        body = self.source[start:end]

        self.assertRegex(
            body,
            r"const hipError_t sync_err = hipDeviceSynchronize\(\);\s+"
            r"return persistent_decode_post_enqueue_status\(",
        )
        final_check = body[body.rindex("const hipError_t sync_err") :]
        self.assertNotRegex(final_check, r"return\s+25[45]\s*;")

    def test_unsupported_kv_fp8_bridge_does_not_enqueue_work(self):
        start = self.source.index(
            "extern \"C\" int supersonic_qwen35_4b_hip_quantize_kv_to_fp8("
        )
        body = self.source[start:]

        self.assertIn("KV-FP8 is outside the narrowed Qwen3.8 product contract", body)
        self.assertIn("return 256;", body)
        self.assertNotIn("quantize_kv_to_fp8_kernel", body)
        self.assertNotIn("hipLaunchKernelGGL", body)

    def test_persistent_decode_rejects_kv_fp8_descriptors_before_device_work(self):
        start = self.source.index("int persistent_decode_device(")
        body = self.source[start:]
        guard = body.index("if (kv_fp8_descs != nullptr)")
        prefix = body[:guard]
        self.assertIn("return 256;", body[guard : guard + 180])
        self.assertNotIn("DecodeBridgeLockGuard guard", prefix)
        self.assertNotIn("hipLaunchKernelGGL", body[guard : guard + 180])

    def test_prepare_only_sync_failures_use_integrity_policy(self):
        start = self.source.index("int persistent_decode_device(")
        end = self.source.index("// Restore conv+rec", start)
        body = self.source[start:end]

        self.assertEqual(
            body.count("persistent_decode_prepare_only_status(\n                    hipDeviceSynchronize()"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
