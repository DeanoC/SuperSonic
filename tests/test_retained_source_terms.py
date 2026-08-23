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


if __name__ == "__main__":
    unittest.main()
