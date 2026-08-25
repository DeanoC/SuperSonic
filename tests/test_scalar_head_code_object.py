import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check-scalar-head-code-object.py"
SPEC = importlib.util.spec_from_file_location("check_scalar_head_code_object", SCRIPT)
scalar_head = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scalar_head
SPEC.loader.exec_module(scalar_head)


SYMBOL = "supersonic_qwen38_q6_k_scalar_head_f32_kernel"

DISASSEMBLY = f"""
0000000000000000 <{SYMBOL}>:
    v_mul_f32 v0, v1, v2
    v_fma_f32 v3, v4, v5, v6
    ds_bpermute_b32 v0, v1
    v_add_f32 v0, v0, v1
    ds_bpermute_b32 v0, v1
    v_add_f32 v0, v0, v1
    ds_bpermute_b32 v0, v1
    v_add_f32 v0, v0, v1
    ds_bpermute_b32 v0, v1
    v_add_f32 v0, v0, v1
    ds_bpermute_b32 v0, v1
    v_add_f32 v0, v0, v1
0000000000000040 <another_kernel>:
    v_fma_mix_f32 v0, v1, v2, v3
"""

METADATA = f"""
AMDGPU HSA Kernel Descriptor
  .symbol: {SYMBOL}
  .vgpr_count: 24
  .private_segment_fixed_size: 0
  COMPUTE_PGM_RSRC1: 0x00030002
"""


class ScalarHeadCodeObjectTests(unittest.TestCase):
    def test_accepts_scalar_f32_kernel_with_required_reduction_and_metadata(self):
        report = scalar_head.analyze(DISASSEMBLY, METADATA, SYMBOL)

        self.assertEqual(report["symbol"], SYMBOL)
        self.assertEqual(report["vgpr_count"], 24)
        self.assertEqual(report["spill_count"], 0)
        self.assertEqual(report["fp32_round_mode"], "RNE")
        self.assertEqual(report["fp32_denorm_mode"], "preserve")
        self.assertEqual(
            report["instruction_counts"],
            {
                "ds_bpermute_b32": 5,
                "v_add_f32": 5,
                "v_fma_f32": 1,
                "v_fma_mix_f32": 0,
                "v_mfma_f32_16x16x16bf16": 0,
                "v_mul_f32": 1,
                "v_wmma_f32_16x16x16_bf16": 0,
            },
        )
        self.assertEqual(scalar_head.find_violations(report), [])

    def test_rejects_forbidden_or_contract_breaking_kernel_variants(self):
        cases = {
            "mixed fma": (DISASSEMBLY.replace("v_fma_f32", "v_fma_mix_f32", 1), METADATA, "v_fma_mix_f32"),
            "wmma": (DISASSEMBLY.replace("v_mul_f32", "v_wmma_f32_16x16x16_bf16", 1), METADATA, "v_wmma_f32_16x16x16_bf16"),
            "mfma": (DISASSEMBLY.replace("v_mul_f32", "v_mfma_f32_16x16x16bf16", 1), METADATA, "v_mfma_f32_16x16x16bf16"),
            "scratch": (DISASSEMBLY, METADATA.replace(".private_segment_fixed_size: 0", ".private_segment_fixed_size: 16"), "spill_count"),
            "round mode": (DISASSEMBLY, METADATA.replace("0x00030002", "0x00031002"), "fp32_round_mode"),
            "denorm mode": (DISASSEMBLY, METADATA.replace("0x00030002", "0x00020002"), "fp32_denorm_mode"),
            "missing named symbol": (DISASSEMBLY.replace(SYMBOL, "other_kernel", 1), METADATA.replace(SYMBOL, "other_kernel"), "missing symbol"),
            "unexpected shuffle count": (DISASSEMBLY.replace("    ds_bpermute_b32 v0, v1\n", "", 1), METADATA, "ds_bpermute_b32"),
            "unexpected add count": (DISASSEMBLY.replace("    v_add_f32 v0, v0, v1\n", "", 1), METADATA, "v_add_f32"),
        }

        for label, (disassembly, metadata, expected) in cases.items():
            with self.subTest(label=label):
                report = scalar_head.analyze(disassembly, metadata, SYMBOL)
                self.assertTrue(
                    any(expected in violation for violation in scalar_head.find_violations(report)),
                    scalar_head.find_violations(report),
                )

    def test_decodes_fp32_modes_from_the_real_code_object_descriptor_layout(self):
        descriptor_disassembly = DISASSEMBLY + """
Contents of section .rodata:
 7fc40 00000000 00000000 20000000 00000000  ........ .......
 7fc50 c0350500 00000000 00000000 00000000  .5..............
 7fc60 00000000 00000000 00000000 c0000000  ................
 7fc70 05000fe0 84000000 08040000 00000000  ................
"""
        descriptor_metadata = """
amdhsa.kernels:
  - .args:
    .name:           _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii
    .private_segment_fixed_size: 0
    .sgpr_spill_count: 0
    .symbol:         _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii.kd
    .vgpr_count:     44
    .vgpr_spill_count: 0
Symbols [
  Symbol {
    Name: _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii.kd (1300)
    Value: 0x7FC40
  }
]
"""

        report = scalar_head.analyze(descriptor_disassembly, descriptor_metadata, SYMBOL)

        self.assertEqual(report["vgpr_count"], 44)
        self.assertEqual(report["fp32_round_mode"], "RNE")
        self.assertEqual(report["fp32_denorm_mode"], "preserve")

    def test_decodes_descriptor_when_readobj_symbols_precede_kernel_metadata(self):
        metadata_with_real_ordering = """
Symbols [
  Symbol {
    Name: _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii.kd (1300)
    Value: 0x7FC40
  }
]
amdhsa.kernels:
  - .args:
    .name:           _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii
    .private_segment_fixed_size: 0
    .sgpr_spill_count: 0
    .symbol:         _Z45supersonic_qwen38_q6_k_scalar_head_f32_kernelPKtPKhPfii.kd
    .vgpr_count:     44
    .vgpr_spill_count: 0
  - .args:
    .name:           another_kernel
    .symbol:         another_kernel.kd
"""
        descriptor_disassembly = DISASSEMBLY + """
Contents of section .rodata:
 7fc40 00000000 00000000 20000000 00000000  ........ .......
 7fc50 c0350500 00000000 00000000 00000000  .5..............
 7fc60 00000000 00000000 00000000 c0000000  ................
 7fc70 05000fe0 84000000 08040000 00000000  ................
"""

        report = scalar_head.analyze(descriptor_disassembly, metadata_with_real_ordering, SYMBOL)

        self.assertEqual(report["fp32_round_mode"], "RNE")
        self.assertEqual(report["fp32_denorm_mode"], "preserve")
