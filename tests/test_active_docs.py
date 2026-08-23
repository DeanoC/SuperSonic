#!/usr/bin/env python3
"""Tests for the active public-documentation contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    path = ROOT / "tools" / "check-active-docs.py"
    spec = importlib.util.spec_from_file_location("check_active_docs", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_active_docs"] = module
    spec.loader.exec_module(module)
    return module


class ActiveDocsTests(unittest.TestCase):
    def test_repository_active_docs_have_no_removed_product_contract(self):
        checker = load_checker()
        self.assertEqual([], checker.find_violations(ROOT))

    def test_removed_cli_backend_and_identity_terms_are_rejected(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n\nCurrent HIP guidance.\n", encoding="utf-8")
            (root / "README.md").write_text(
                "# Product\n\n"
                "Run with --backend cuda for Qwen3.5.\n"
                "SUPERSONIC_BACKENDS=cuda QWEN35_MODEL_DIR=/tmp/Gemma4\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("--backend" in violation for violation in violations))
        self.assertTrue(any("Qwen3.5" in violation for violation in violations))
        self.assertTrue(any("SUPERSONIC_BACKENDS" in violation for violation in violations))
        self.assertTrue(any("QWEN35" in violation for violation in violations))
        self.assertTrue(any("Gemma4" in violation for violation in violations))

    def test_internal_flm_term_is_allowed_only_in_explicit_internal_section(self):
        checker = load_checker()
        allowed = "## Internal FLM foundation\n\nThis is contributor-only context.\n"
        self.assertEqual([], checker.find_text_violations(Path("README.md"), allowed))
        self.assertNotEqual(
            [], checker.find_text_violations(Path("README.md"), "The FLM backend is public.")
        )

    def test_local_documentation_anchor_is_checked(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n", encoding="utf-8")
            (root / "README.md").write_text(
                "# Product\n\n[broken](docs/testing.md#missing)\n", encoding="utf-8"
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("missing" in violation for violation in violations))


if __name__ == "__main__":
    unittest.main()
