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
                "/* outer dflash /* nested SUPERSONIC_DFLASH_LEGACY */ still comment */\n"
                "fn char_dflash_after_quote() { let quote = '\"'; }\n"
                "let lower = \"supersonic_dflash_profile\";\n"
                "let escaped = \"SUPERSONIC_\\x44FLASH_LEGACY\";\n"
                "let unicode = \"SUPERSONIC_\\u{44}FLASH_LEGACY\";\n"
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
            "SUPERSONIC_DFLASH_LEGACY",
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
                "let escaped = \"SUPERSONIC_\\u{4_4}FLASH_LEGACY\";\n"
                "let concat_qwen = concat!(\n"
                "    \"SUPERSONIC_\", /* split legacy control */\n"
                "    \"qwen35\", \"mtp_PROFILE\",\n"
                ");\n"
                "let assembled_control = concat!(\"SUPERSONIC_\", \"D\", \"FLASH_LEGACY\",);\n"
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
            "SUPERSONIC_DFLASH_LEGACY",
            "SUPERSONIC_qwen35mtp_PROFILE",
            "SUPERSONIC_DFLASH_LEGACY",
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

    def test_concat_renders_char_numeric_and_bool_literals_for_all_forms(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            source = (
                "let parens = concat!(\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", 'x');\n"
                "let brackets = concat![\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", '\\x78'];\n"
                "let braces = concat!{\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", '\\u{78}'};\n"
                "let raw_parens = r#concat!(\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", 'x');\n"
                "let raw_brackets = r#concat![\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", '\\x78'];\n"
                "let raw_braces = r#concat!{\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", '\\u{78}'};\n"
                "let supported = concat!(\"SUPERSONIC_\", \"D\", \"FLASH_PROFILE\", '\\x78', 42u8, 3.50f32, true, false);\n"
                "let unrelated = concat!(\"hello\", 'x', 42u32, 3.50f64, true, false);\n"
            )
            (runtime / "literal_arguments.rs").write_text(source, encoding="utf-8")

            violations = checker.find_violations(root)
            lexemes = checker._lex_rust(source)

        terms = [term for _, _, term, _ in violations]
        rendered = "\n".join(terms)
        self.assertEqual(terms.count("SUPERSONIC_DFLASH_PROFILEx"), 6, rendered)
        self.assertIn(
            "SUPERSONIC_DFLASH_PROFILEx423.50truefalse",
            [lexeme.value for lexeme in lexemes if lexeme.kind == "string"],
        )
        self.assertNotIn("hello", rendered)

    def test_concat_unknown_arguments_fail_closed_only_with_supersonic_prefix(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "unknown_arguments.rs").write_text(
                "fn unknown_fragment() -> &'static str { \"D\" }\n"
                "let after = concat!(\"SUPERSONIC_\", unknown_fragment());\n"
                "let before = concat!(unknown_fragment(), \"SUPERSONIC_\");\n"
                "let split = concat![\"SUPER\", \"SONIC_\", unknown_fragment()];\n"
                "let unsupported = r#concat!{\"SUPERSONIC_\", b\"D\"};\n"
                "let unsupported_prefix = concat!(b\"SUPERSONIC_\", unknown_fragment());\n"
                "let unrelated = concat!(\"hello\", unknown_fragment(), b\"world\");\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        terms = [term for _, _, term, _ in violations]
        prefix_terms = [term for term in terms if "SUPERSONIC_" in term.upper()]
        self.assertEqual(len(prefix_terms), 5, violations)
        self.assertTrue(all("SUPERSONIC_" in term.upper() for term in prefix_terms))
        self.assertNotIn("hello", "\n".join(terms))

    def test_allows_retained_dflash2_spec_decode_but_rejects_other_dflash(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "dflash_spec.rs").write_text(
                "use model_store::dflash::{DraftConfig, DraftGpuWeights};\n"
                "use crate::prefill_engine::{self, DflashTargetCapture};\n"
                "pub struct DflashSpecDecoder { capture: DflashTargetCapture }\n"
                "pub struct DflashSpecRound;\n"
                "pub struct DflashSpecSummary;\n"
                "pub fn dflash_spec() {}\n"
                "pub fn verify_block_dflash() {}\n"
                "pub fn capture_block_dflash() {}\n"
                "pub fn prefill_with_dflash_capture() {}\n"
                "pub fn dflash_capture() {}\n"
                "pub fn dflash_dyn_conv() {}\n"
                "pub fn dflash_scatter_cols_raw() {}\n"
                "fn profile() -> bool {\n"
                "    std::env::var_os(\"SUPERSONIC_DFLASH_PROFILE\").is_some()\n"
                "}\n"
                "fn dflash_fused_verify_cache() {}\n"
                "fn DFlashFusedVerifyCache() {}\n"
                "fn dflash_mtp_cache() {}\n"
                "fn lower_evasion() -> bool {\n"
                "    std::env::var_os(\"supersonic_dflash_profile\").is_some()\n"
                "}\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        terms = {term for _, _, term, _ in violations}
        for allowed in (
            "dflash", "dflash_spec", "DflashSpecDecoder", "DflashSpecRound",
            "DflashSpecSummary", "DflashTargetCapture", "dflash_capture",
            "dflash_dyn_conv", "dflash_scatter_cols_raw", "verify_block_dflash",
            "capture_block_dflash", "prefill_with_dflash_capture",
            "SUPERSONIC_DFLASH_PROFILE",
        ):
            self.assertNotIn(allowed, terms, allowed)
        for legacy in (
            "dflash_fused_verify_cache", "DFlashFusedVerifyCache",
            "dflash_mtp_cache", "supersonic_dflash_profile",
        ):
            self.assertIn(legacy, terms, legacy)
        self.assertEqual(len(violations), 4, violations)

    def test_allows_retained_dflash2_rollback_and_commit_contracts(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "crates" / "runtime" / "src"
            runtime.mkdir(parents=True)
            (runtime / "dflash_spec.rs").write_text(
                "use crate::prefill_engine::DflashRollbackCapture;\n"
                "pub enum DflashVerifyPath { Component }\n"
                "struct DflashCommitPlan;\n"
                "pub fn verify_block_dflash_with_rollback() {}\n"
                "pub fn rollback_dflash_prefix() {}\n"
                "pub fn replay_committed_prefix_dflash() {}\n"
                "fn dflash_commit_plan() {}\n"
                "fn dflash_fast_rollback_plan() {}\n"
                "fn dflash_tokens_from_selector() {}\n"
                "fn dflash_next_token() {}\n"
                "fn trace() -> Option<usize> {\n"
                "    std::env::var(\"SUPERSONIC_DFLASH_TRACE_CTX\").ok()?\n"
                "        .parse::<usize>().ok()\n"
                "}\n"
                "mod dflash_commit_tests {}\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        terms = {term for _, _, term, _ in violations}
        for allowed in (
            "DflashRollbackCapture", "DflashVerifyPath", "DflashCommitPlan",
            "verify_block_dflash_with_rollback", "rollback_dflash_prefix",
            "replay_committed_prefix_dflash", "dflash_commit_plan",
            "dflash_fast_rollback_plan", "dflash_tokens_from_selector",
            "dflash_next_token", "dflash_commit_tests",
            "SUPERSONIC_DFLASH_TRACE_CTX",
        ):
            self.assertNotIn(allowed, terms, allowed)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
