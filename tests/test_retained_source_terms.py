import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check-retained-source-terms.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_retained_source_terms", CHECKER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {CHECKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_retained_source_terms"] = module
    spec.loader.exec_module(module)
    return module


class RetainedSourceTermsTests(unittest.TestCase):
    def test_rejects_legacy_mtp_fields_helpers_and_envs_outside_mtp_module(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "crates" / "runtime" / "src" / "decode_engine.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                """
                struct DecodeEngine {
                    dflash_mtp_cache: Option<usize>,
                }

                fn dflash_verify_step() {}

                fn profile_enabled() -> bool {
                    std::env::var_os("SUPERSONIC_DFLASH_PROFILE_VERIFY").is_some()
                }
                """,
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        rendered = "\n".join(term for _, _, term, _ in violations)
        self.assertIn("dflash_mtp_cache", rendered)
        self.assertIn("dflash_verify_step", rendered)
        self.assertIn("SUPERSONIC_DFLASH_PROFILE_VERIFY", rendered)

    def test_rejects_legacy_mtp_identifiers_in_both_orders_without_comment_or_abi_hits(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "decode_engine.rs").write_text(
                "// MtpDFlashCache and MtpMetalV2Scratch are historical names.\n"
                "unsafe extern \"C\" {\n"
                "    #[link_name = \"supersonic_qwen35_hip_mtp_restore_linear_prefix\"]\n"
                "    fn supersonic_qwen35_hip_mtp_restore_linear_prefix();\n"
                "}\n",
                encoding="utf-8",
            )
            (runtime / "prefill_engine.rs").write_text(
                "struct MtpState {\n"
                "    cache: Option<MtpDFlashCache>,\n"
                "    reverse_cache: Option<DFlashMtpCache>,\n"
                "    scratch: Option<MtpMetalV2Scratch>,\n"
                "    reverse_scratch: Option<MetalV2MtpScratch>,\n"
                "}\n"
                "fn mtp_metal_v2_decode_step() {}\n"
                "fn dflash_mtp_decode_step() {}\n",
                encoding="utf-8",
            )
            (runtime / "lib.rs").write_text(
                "fn legacy_profile_enabled() -> bool {\n"
                "    std::env::var_os(\"SUPERSONIC_QWEN35_DRAFT_MTP_GREEDY\").is_some()\n"
                "}\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        rendered = "\n".join(term for _, _, term, _ in violations)
        for term in (
            "MtpDFlashCache",
            "DFlashMtpCache",
            "MtpMetalV2Scratch",
            "MetalV2MtpScratch",
            "mtp_metal_v2_decode_step",
            "dflash_mtp_decode_step",
            "SUPERSONIC_QWEN35_DRAFT_MTP_GREEDY",
        ):
            self.assertIn(term, rendered)
        self.assertTrue(violations)
        self.assertEqual(
            {path for path, *_ in violations},
            {
                Path("crates/runtime/src/prefill_engine.rs"),
                Path("crates/runtime/src/lib.rs"),
            },
            violations,
        )


if __name__ == "__main__":
    unittest.main()
