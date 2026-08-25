import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch


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
  .sgpr_spill_count: 0
  .vgpr_spill_count: 0
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

    def test_rejects_missing_spill_metadata_evidence(self):
        for field in ("private_segment_fixed_size", "sgpr_spill_count", "vgpr_spill_count"):
            with self.subTest(field=field):
                report = scalar_head.analyze(
                    DISASSEMBLY,
                    METADATA.replace(f"  .{field}: 0\n", ""),
                    SYMBOL,
                )
                self.assertIn(
                    f"missing {field} metadata",
                    scalar_head.find_violations(report),
                )

    def test_rejects_forbidden_instruction_families(self):
        cases = {
            "fma mix variant": ("v_fma_f32", "v_fma_mixlo_f32", "v_fma_mix_f32"),
            "dual fma mix variant": ("v_fma_f32", "v_dual_fma_mix_f32", "v_fma_mix_f32"),
            "wmma variant": ("v_mul_f32", "v_wmma_f32_16x16x16_fp8", "v_wmma_f32_16x16x16_bf16"),
            "mfma variant": ("v_mul_f32", "v_mfma_f32_32x32x8f16", "v_mfma_f32_16x16x16bf16"),
        }
        for label, (original, variant, expected) in cases.items():
            with self.subTest(label=label):
                report = scalar_head.analyze(DISASSEMBLY.replace(original, variant, 1), METADATA, SYMBOL)
                self.assertTrue(
                    any(expected in violation for violation in scalar_head.find_violations(report)),
                    scalar_head.find_violations(report),
                )

    def test_rejects_missing_ambiguous_and_prefix_colliding_symbol_matches(self):
        cases = {
            "missing": (DISASSEMBLY.replace(SYMBOL, "different_kernel", 1), METADATA.replace(SYMBOL, "different_kernel")),
            "prefix collision": (
                DISASSEMBLY.replace(SYMBOL, f"{SYMBOL}_different_kernel", 1),
                METADATA.replace(SYMBOL, f"{SYMBOL}_different_kernel"),
            ),
            "ambiguous labels": (
                DISASSEMBLY + f"0000000000000080 <{SYMBOL}>:\n    s_endpgm\n",
                METADATA,
            ),
            "ambiguous metadata": (
                DISASSEMBLY,
                METADATA + f"\n  .symbol: {SYMBOL}\n  .vgpr_count: 24\n",
            ),
        }
        for label, (disassembly, metadata) in cases.items():
            with self.subTest(label=label):
                report = scalar_head.analyze(disassembly, metadata, SYMBOL)
                self.assertIn("missing symbol in disassembly or metadata", scalar_head.find_violations(report))

    def test_cli_rejects_abbreviated_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            object_path = Path(temporary) / "code.o"
            object_path.write_bytes(b"object")
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                scalar_head.main(["--obj", str(object_path), "--sym", SYMBOL])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--object", stderr.getvalue())

    def test_inspection_extracts_the_gfx1201_offload_image_with_bounded_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            object_path = Path(temporary) / "full_attention_bridge_4b.o"
            object_path.write_bytes(b"fat-object")

            def run_tool(command):
                if command[1:] == ["--offloading", command[-1]]:
                    Path(command[-1] + ".0.hipv4-amdgcn-amd-amdhsa--gfx1201").write_bytes(b"device-object")
                    return "Extracting offload bundle"
                if command[1] == "--disassemble":
                    return DISASSEMBLY
                if command[1] == "--notes":
                    return METADATA
                self.fail(f"unexpected command: {command}")

            with patch.object(scalar_head, "_run_tool", side_effect=run_tool) as run:
                disassembly, metadata = scalar_head._inspect_object(
                    object_path,
                    "llvm-objdump",
                    "llvm-readobj",
                )

        self.assertEqual(disassembly, DISASSEMBLY)
        self.assertEqual(metadata, METADATA)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["llvm-objdump", "--offloading"])
        self.assertTrue(commands[1][-1].endswith(".0.hipv4-amdgcn-amd-amdhsa--gfx1201"))
        self.assertEqual(commands[1][1:4], ["--disassemble", "--full-contents", "--mcpu=gfx1201"])
        self.assertEqual(commands[2][0:3], ["llvm-readobj", "--notes", "--symbols"])

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
