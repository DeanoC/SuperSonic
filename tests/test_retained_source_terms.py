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
                "fn retained_decode_path() {}\n",
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
            abi = root / "crates" / "kernel-ffi" / "src" / "qwen38.rs"
            abi.parent.mkdir(parents=True)
            abi.write_text(
                "#[link_name = \"supersonic_qwen35_hip_mtp_restore_linear_prefix\"]\n",
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

    def test_lexes_all_nested_runtime_rust_and_rejects_legacy_tokens_and_envs(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            nested = runtime / "new" / "deep"
            nested.mkdir(parents=True)
            (nested / "future_module.rs").write_text(
                "fn qwen35_draft_mtp_forward() {}\n"
                "fn qwen35_mtpfoo() {}\n"
                "fn mtp_qwen35_forward() {}\n"
                "fn mtpqwen35foo() {}\n"
                "fn mtp_dflash_cache() {}\n"
                "fn mtp_metal_v2_decode_step() {}\n"
                "fn standalone_dflash_verify() {}\n"
                "fn standalone_spec_prefill() {}\n"
                "fn standalone_certified_kv() {}\n"
                "fn allowed_dflashback() {}\n"
                "fn allowed_uncertified() {}\n"
                "fn allowed_certifiedness() {}\n",
                encoding="utf-8",
            )
            (runtime / "lexer_cases.rs").write_text(
                "/* outer dflash /* nested SUPERSONIC_DFLASH_PROFILE */ still comment */\n"
                "fn char_dflash_after_quote() { let quote = '\"'; }\n"
                "let lower = \"supersonic_dflash_profile\";\n"
                "let escaped = \"SUPERSONIC_\\x44FLASH_PROFILE\";\n"
                "let unicode = \"SUPERSONIC_\\u{44}FLASH_PROFILE\";\n"
                "let byte = b\"SUPERSONIC_METALV2_PROFILE\";\n"
                "let raw = br###\"SUPERSONIC_qwen35_draft_mtp_greedy\"###;\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        rendered = "\n".join(term for _, _, term, _ in violations)
        for term in (
            "qwen35_draft_mtp_forward",
            "qwen35_mtpfoo",
            "mtp_qwen35_forward",
            "mtpqwen35foo",
            "mtp_dflash_cache",
            "mtp_metal_v2_decode_step",
            "standalone_dflash_verify",
            "standalone_spec_prefill",
            "standalone_certified_kv",
            "dflash_after_quote",
            "supersonic_dflash_profile",
            "SUPERSONIC_DFLASH_PROFILE",
            "SUPERSONIC_METALV2_PROFILE",
            "SUPERSONIC_qwen35_draft_mtp_greedy",
        ):
            self.assertIn(term, rendered)
        self.assertNotIn("allowed_dflashback", rendered)
        self.assertNotIn("allowed_uncertified", rendered)
        self.assertNotIn("allowed_certifiedness", rendered)
        self.assertNotIn("still comment", rendered)
        self.assertEqual(
            {path for path, *_ in violations},
            {
                Path("crates/runtime/src/new/deep/future_module.rs"),
                Path("crates/runtime/src/lexer_cases.rs"),
            },
            violations,
        )

    def test_lexer_edge_cases_for_escapes_suffixes_lifetimes_and_concat(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "edge_cases.rs").write_text(
                "fn dflash2_cache() {}\n"
                "fn DFlash2Cache() {}\n"
                "fn r#dflash2_cache() {}\n"
                "fn lifetime_only<'dflash, 'mtp>() {}\n"
                "fn lifetime_label() { 'dflash: loop { break 'dflash; } }\n"
                "fn actual_dflash_after_lifetime() {}\n"
                "let escaped = \"SUPERSONIC_\\u{4_4}FLASH_PROFILE\";\n"
                "let concat_qwen = concat!(\n"
                "    \"SUPERSONIC_\", /* split legacy control */\n"
                "    \"qwen35\", \"mtp_PROFILE\",\n"
                ");\n"
                "let assembled_control = concat!(\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\",);\n"
                "let no_separator = \"superSONIC_qwen35mtp_profile\";\n"
                "let raw_no_separator = r#\"SUPERSONIC_QWEN35MTP_PROFILE\"#;\n"
                "// 'dflash 'mtp SUPERSONIC_DFLASH are historical prose only.\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        terms = [term for _, _, term, _ in violations]
        rendered = "\n".join(terms)
        for term in (
            "dflash2_cache",
            "DFlash2Cache",
            "SUPERSONIC_DFLASH_PROFILE",
            "SUPERSONIC_qwen35mtp_PROFILE",
            "SUPERSONIC_DFLASH_PROFILE",
            "superSONIC_qwen35mtp_profile",
            "SUPERSONIC_QWEN35MTP_PROFILE",
            "actual_dflash_after_lifetime",
        ):
            self.assertIn(term, rendered)
        self.assertNotIn("dflash\n", rendered)
        self.assertNotIn("mtp\n", rendered)
        self.assertEqual(
            {path for path, *_ in violations},
            {Path("crates/runtime/src/edge_cases.rs")},
            violations,
        )

    def test_concat_uses_all_rust_delimiters_and_raw_identifiers(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "concat_delimiters.rs").write_text(
                "let parens = concat!(\n"
                "    \"SUPERSONIC_\", /* ) ] } /* nested ( [ { */ */\n"
                "    \"D\", \"FLASH_PARENS_PROFILE\",\n"
                ");\n"
                "let brackets = concat![\n"
                "    \"SUPERSONIC_\", // ] } )\n"
                "    \"METAL\", \"V2_BRACKET_PROFILE\",\n"
                "];\n"
                "let braces = concat! {\n"
                "    \"SUPERSONIC_\", /* { [ ( ] } ) */\n"
                "    \"QWEN35\", \"MTP_BRACE_PROFILE\",\n"
                "};\n"
                "let raw_parens = r#concat!(\n"
                "    \"SUPERSONIC_\", \"D\", \"FLASH_RAW_PARENS_PROFILE\",\n"
                ");\n"
                "let raw_brackets = r#concat /* between raw id and bang */ ![\n"
                "    \"SUPERSONIC_\", \"METAL\", \"V2_RAW_BRACKET_PROFILE\",\n"
                "];\n"
                "let raw_braces = r#concat!{\n"
                "    \"SUPERSONIC_\", /* nested /* ( [ { */ */\n"
                "    \"QWEN35\", \"MTP_RAW_BRACE_PROFILE\",\n"
                "};\n"
                "fn r#concat_suffix() {}\n"
                "fn r#concat() {}\n"
                "fn r#dflash_cache() {}\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        rendered = "\n".join(term for _, _, term, _ in violations)
        for term in (
            "SUPERSONIC_DFLASH_PARENS_PROFILE",
            "SUPERSONIC_METALV2_BRACKET_PROFILE",
            "SUPERSONIC_QWEN35MTP_BRACE_PROFILE",
            "SUPERSONIC_DFLASH_RAW_PARENS_PROFILE",
            "SUPERSONIC_METALV2_RAW_BRACKET_PROFILE",
            "SUPERSONIC_QWEN35MTP_RAW_BRACE_PROFILE",
            "dflash_cache",
        ):
            self.assertIn(term, rendered)
        self.assertNotIn("concat_suffix", rendered)
        self.assertEqual(
            {path for path, *_ in violations},
            {Path("crates/runtime/src/concat_delimiters.rs")},
            violations,
        )


if __name__ == "__main__":
    unittest.main()
