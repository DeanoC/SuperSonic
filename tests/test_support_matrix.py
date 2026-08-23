import importlib.util
import contextlib
import io
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check-support-matrix.py"
SPEC = importlib.util.spec_from_file_location("check_support_matrix", SCRIPT)
support_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support_matrix
SPEC.loader.exec_module(support_matrix)


class SupportMatrixTests(unittest.TestCase):
    def test_lane_key_distinguishes_gqh_source_lanes(self):
        base = {
            "backend": "hip",
            "arch": "gfx1100",
            "models": ["qwen3.8-27b"],
            "quants": ["gqh"],
            "model_sources": ["gqh-gguf"],
        }
        other_source = {
            **base,
            "model_sources": ["unsupported-source"],
        }

        self.assertEqual(
            support_matrix.model_sources_for_entry("base", base, []),
            ["gqh-gguf"],
        )
        self.assertNotEqual(
            support_matrix.lane_key_for_entry(base),
            support_matrix.lane_key_for_entry(other_source),
        )

    def test_active_matrix_is_exact_qwen38_gqh_product_contract(self):
        with support_matrix.MANIFEST.open("rb") as handle:
            data = tomllib.load(handle)

        entries = data["entry"]
        self.assertEqual(
            {entry["arch"] for entry in entries},
            {"gfx1100", "gfx1201"},
        )
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["backend"], "hip")
            self.assertEqual(entry["models"], ["qwen3.8-27b"])
            self.assertEqual(entry["model_sources"], ["gqh-gguf"])
            self.assertEqual(entry["quants"], ["gqh"])
            self.assertIsInstance(entry.get("correctness_gate"), str)
            self.assertTrue(entry["correctness_gate"].strip())
            self.assertTrue(entry.get("gate_commands"))

    def test_validator_rejects_non_product_model_backend_and_source(self):
        invalid_manifest = """
version = 1

[[entry]]
id = "cuda-old-model"
backend = "cuda"
arch = "sm90"
status = "validated"
models = ["qwen3.6-35b-a3b"]
model_sources = ["hf-snapshot"]
quants = ["int4"]
support_doc = "docs/supported-matrix.md"
correctness_gate = "old-gate"
gate_commands = ["cargo test"]
"""

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "matrix.toml"
            manifest.write_text(invalid_manifest, encoding="utf-8")
            original = support_matrix.MANIFEST
            support_matrix.MANIFEST = manifest
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    result = support_matrix.main()
            finally:
                support_matrix.MANIFEST = original

        self.assertNotEqual(result, 0)
        self.assertIn("qwen3.8-27b", stderr.getvalue())
        self.assertIn("gqh-gguf", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
