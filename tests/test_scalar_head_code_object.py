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

# This is deliberately the fixture's own pinned digest, not the supported
# ROCm/gfx1201 object digest. Pure parser tests inject it explicitly so they do
# not accidentally become coupled to the real release artifact.
BASE_FIXTURE_FINGERPRINT = "8f04084f4a7abcf6cd7e412082316fb1da01fdfcf163e9b25e549a400c36a435"

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
    v_xor_b32 v6, 16, v3
    v_xor_b32 v7, 8, v3
    v_xor_b32 v7, 4, v3
    v_xor_b32 v7, 2, v3
    v_xor_b32 v7, 1, v3
    v_mul_f32 v8, v9, v10
    v_mul_f32 v9, v10, v11
    v_mul_f32 v10, v11, v12
    v_mul_f32 v11, v12, v13
    v_mul_f32 v12, v13, v14
    v_mul_f32 v13, v14, v15
    v_mul_f32 v14, v15, v16
    v_mul_f32 v15, v16, v17
    v_mul_f32 v16, v17, v18
    v_mul_f32 v17, v18, v19
    v_mul_f32 v18, v19, v20
    v_mul_f32 v19, v20, v21
    v_mul_f32 v20, v21, v22
    v_mul_f32 v21, v22, v23
    v_mul_f32 v22, v23, v24
    v_fma_f32 v8, v9, v10, v11
    v_fma_f32 v9, v10, v11, v12
    v_fma_f32 v10, v11, v12, v13
    v_fma_f32 v11, v12, v13, v14
    v_fma_f32 v12, v13, v14, v15
    v_fma_f32 v13, v14, v15, v16
    v_fma_f32 v14, v15, v16, v17
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
                "v_fma_f32": 8,
                "v_fma_mix_f32": 0,
                "v_mfma_f32_16x16x16bf16": 0,
                "v_mul_f32": 16,
                "v_wmma_f32_16x16x16_bf16": 0,
            },
        )
        self.assertEqual(report["xor_offsets"], [16, 8, 4, 2, 1])
        self.assertEqual(
            scalar_head.find_violations(
                report,
                expected_instruction_stream_sha256=BASE_FIXTURE_FINGERPRINT,
            ),
            [],
        )

    def test_reports_canonical_instruction_stream_digest(self):
        report = scalar_head.analyze(DISASSEMBLY, METADATA, SYMBOL)

        self.assertEqual(
            report["instruction_stream_sha256"],
            BASE_FIXTURE_FINGERPRINT,
        )

    def test_canonical_digest_ignores_addresses_and_encoding_comments(self):
        annotated = DISASSEMBLY.replace(
            f"0000000000000000 <{SYMBOL}>:",
            f"0000000000001000 <{SYMBOL}>:",
        ).replace(
            "    v_mul_f32 v0, v1, v2",
            "    v_mul_f32 v0, v1, v2 // 0000000000001000: 10343512",
            1,
        )

        base = scalar_head.analyze(DISASSEMBLY, METADATA, SYMBOL)
        changed_addresses = scalar_head.analyze(annotated, METADATA, SYMBOL)

        self.assertEqual(
            changed_addresses["instruction_stream_sha256"],
            base["instruction_stream_sha256"],
        )

    def test_rejects_order_offset_and_dependency_mutations_even_when_counts_match(self):
        mutations = {
            "wrong offset": DISASSEMBLY.replace(
                "    v_xor_b32 v6, 16, v3\n",
                "    v_xor_b32 v6, 32, v3\n",
                1,
            ),
            "wrong order": DISASSEMBLY.replace(
                "    v_xor_b32 v7, 8, v3\n    v_xor_b32 v7, 4, v3\n",
                "    v_xor_b32 v7, 4, v3\n    v_xor_b32 v7, 8, v3\n",
                1,
            ),
            "changed dependency": DISASSEMBLY.replace(
                "    v_mul_f32 v0, v1, v2\n",
                "    v_mul_f32 v0, v1, v9\n",
                1,
            ),
            "changed reduction dependency": DISASSEMBLY.replace(
                "    v_add_f32 v0, v0, v1\n",
                "    v_add_f32 v0, v0, v2\n",
                1,
            ),
        }

        for label, disassembly in mutations.items():
            with self.subTest(label=label):
                report = scalar_head.analyze(disassembly, METADATA, SYMBOL)
                self.assertEqual(
                    report["instruction_counts"]["v_mul_f32"],
                    16,
                )
                self.assertEqual(report["instruction_counts"]["v_fma_f32"], 8)
                violations = scalar_head.find_violations(
                    report,
                    expected_instruction_stream_sha256=BASE_FIXTURE_FINGERPRINT,
                )
                self.assertTrue(
                    any(
                        "instruction_stream_sha256" in violation
                        for violation in violations
                    ),
                    violations,
                )

        wrong_offset = scalar_head.analyze(mutations["wrong offset"], METADATA, SYMBOL)
        wrong_order = scalar_head.analyze(mutations["wrong order"], METADATA, SYMBOL)
        self.assertNotEqual(wrong_offset["xor_offsets"], [16, 8, 4, 2, 1])
        self.assertNotEqual(wrong_order["xor_offsets"], [16, 8, 4, 2, 1])

    def test_rejects_extra_mul_and_fma_instructions(self):
        extra = DISASSEMBLY.replace(
            f"0000000000000040 <another_kernel>:",
            "    v_mul_f32 v25, v26, v27\n"
            "    v_fma_f32 v25, v26, v27, v28\n"
            f"0000000000000040 <another_kernel>:",
            1,
        )
        report = scalar_head.analyze(extra, METADATA, SYMBOL)
        violations = scalar_head.find_violations(
            report,
            expected_instruction_stream_sha256=BASE_FIXTURE_FINGERPRINT,
        )

        self.assertEqual(report["instruction_counts"]["v_mul_f32"], 17)
        self.assertEqual(report["instruction_counts"]["v_fma_f32"], 9)
        self.assertTrue(any("v_mul_f32 count must be 16" in v for v in violations), violations)
        self.assertTrue(any("v_fma_f32 count must be 8" in v for v in violations), violations)

    def test_rejects_reassociation_opcode_families(self):
        for opcode in ("v_add3_f32", "v_mad_f32"):
            with self.subTest(opcode=opcode):
                disassembly = DISASSEMBLY.replace(
                    "0000000000000040 <another_kernel>:",
                    f"    {opcode} v25, v26, v27, v28\n"
                    "0000000000000040 <another_kernel>:",
                    1,
                )
                report = scalar_head.analyze(disassembly, METADATA, SYMBOL)
                violations = scalar_head.find_violations(report)
                self.assertTrue(
                    any(f"forbidden {opcode}" in v for v in violations),
                    violations,
                )

    def test_cli_enforces_real_pinned_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            object_path = Path(temporary) / "code.o"
            object_path.write_bytes(b"object")
            stderr = io.StringIO()
            with patch.object(
                scalar_head,
                "_inspect_object",
                return_value=(DISASSEMBLY, METADATA),
            ), redirect_stderr(stderr):
                result = scalar_head.main(
                    ["--object", str(object_path), "--symbol", SYMBOL]
                )

        self.assertEqual(result, 1)
        self.assertIn("instruction_stream_sha256", stderr.getvalue())

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
