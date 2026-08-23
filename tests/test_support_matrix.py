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

    def test_each_active_gate_fails_closed_before_serial_artifact_crawl(self):
        with support_matrix.MANIFEST.open("rb") as handle:
            data = tomllib.load(handle)

        for entry in data["entry"]:
            self.assertEqual(entry["status"], "experimental")
            self.assertEqual(entry["correctness_gate"], "qwen38-gqh-correctness")
            commands = "\n".join(entry["gate_commands"])
            self.assertIn("SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1", commands)
            self.assertIn("tools/check-qwen38-artifacts.py --require-8192", commands)
            self.assertIn("qwen38_gqh_gguf_crawl", commands)
            self.assertIn("--include-ignored", commands)
            self.assertIn("--test-threads=1", commands)

    def test_supported_document_displays_status_and_named_gate_for_each_arch(self):
        document = (Path(__file__).resolve().parents[1] / "docs" / "supported-matrix.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("| Status |", document)
        for arch in ("gfx1100", "gfx1201"):
            rows = [
                line
                for line in document.splitlines()
                if line.startswith("| `qwen3.8-27b`") and f"`{arch}`" in line
            ]
            self.assertEqual(len(rows), 1, arch)
            self.assertIn("`experimental`", rows[0])
            self.assertIn("`qwen38-gqh-correctness`", rows[0])

    def test_validator_enforces_real_strict_preflight_and_serial_crawl_steps(self):
        original_text = support_matrix.MANIFEST.read_text(encoding="utf-8")
        preflight = "SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 python3 tools/check-qwen38-artifacts.py --require-8192"
        crawl = (
            "SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 RUST_TEST_THREADS=1 cargo test --release "
            "-p qwen38 --test qwen38_gqh_gguf_crawl -- --include-ignored --test-threads=1"
        )
        mutations = {
            "echo-only preflight": original_text.replace(preflight, f"echo {preflight}", 1),
            "help preflight": original_text.replace(preflight, f"{preflight} --help", 1),
            "version preflight": original_text.replace(preflight, f"{preflight} --version", 1),
            "unknown preflight flag": original_text.replace(preflight, f"{preflight} --bogus", 1),
            "extra environment assignment": original_text.replace(
                preflight, f"NOOP=1 {preflight}", 1
            ),
            "missing preflight": original_text.replace(
                preflight,
                "SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 python3 tools/check-qwen38-artifacts.py",
                1,
            ),
            "reordered gate": original_text.replace(
                f'  "{preflight}",\n  "{crawl}",',
                f'  "{crawl}",\n  "{preflight}",',
                1,
            ),
            "non-serial crawl": original_text.replace("--test-threads=1", "--test-threads=2", 1),
            "wrong crawl test": original_text.replace(
                "qwen38_gqh_gguf_crawl", "wrong_test", 1
            ),
            "echo-only crawl": original_text.replace(crawl, f"echo {crawl}", 1),
            "short-circuit preflight": original_text.replace(preflight, f"{preflight} || true", 1),
            "substitution preflight": original_text.replace(
                preflight, f"{preflight} $(echo bypass)", 1
            ),
        }

        for label, manifest_text in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                manifest = Path(temporary) / "matrix.toml"
                manifest.write_text(manifest_text, encoding="utf-8")
                original_manifest = support_matrix.MANIFEST
                stderr = io.StringIO()
                support_matrix.MANIFEST = manifest
                try:
                    with contextlib.redirect_stderr(stderr):
                        result = support_matrix.main()
                finally:
                    support_matrix.MANIFEST = original_manifest
                self.assertNotEqual(result, 0, stderr.getvalue())

    def test_markdown_anchor_parser_handles_github_heading_forms(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "doc.md"
            path.write_text(
                "## Repeat\n"
                "## Repeat\n"
                "Repeat\n"
                "=======\n"
                "   # Indented title\n"
                "    # Code title\n",
                encoding="utf-8",
            )
            self.assertEqual(
                support_matrix.anchors_for(path),
                {"repeat", "repeat-1", "repeat-2", "indented-title"},
            )

    def test_markdown_anchor_parser_ignores_variable_fenced_code_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "doc.md"
            path.write_text(
                "# Before\n"
                "# Before\n"
                "````markdown\n"
                "# Hidden backtick\n"
                "```\n"
                "# Still hidden\n"
                "````\n"
                "~~~text\n"
                "## Hidden tilde\n"
                "~~~~\n"
                "## After\n"
                "## After\n",
                encoding="utf-8",
            )
            self.assertEqual(
                support_matrix.anchors_for(path),
                {"before", "before-1", "after", "after-1"},
            )

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
