#!/usr/bin/env python3
"""Tests for the active public-documentation contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
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
                "SUPERSONIC_BACKENDS=cuda QWEN35_MODEL_DIR=/tmp/Gemma4\n"
                "KV-FP8 VMM MoE Q4_K_M safetensors --int4\n"
                "Gemma 4 Phi 4 Llama 3\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("--backend" in violation for violation in violations))
        self.assertTrue(any("Qwen3.5" in violation for violation in violations))
        self.assertTrue(any("SUPERSONIC_BACKENDS" in violation for violation in violations))
        self.assertTrue(any("QWEN35" in violation for violation in violations))
        self.assertTrue(any("Gemma4" in violation for violation in violations))
        self.assertTrue(any("KV-FP8" in violation for violation in violations))
        self.assertTrue(any("Q4_K_M" in violation for violation in violations))
        self.assertTrue(any("Gemma 4" in violation for violation in violations))
        self.assertTrue(any("Phi 4" in violation for violation in violations))
        self.assertTrue(any("Llama 3" in violation for violation in violations))

    def test_artifact_doc_matches_cheap_preflight_scope(self):
        artifact_doc = (ROOT / "docs" / "artifact-format.md").read_text(encoding="utf-8")
        self.assertIn("existence, readability", artifact_doc)
        self.assertNotIn("hashes", artifact_doc.lower())
        self.assertNotIn("metadata", artifact_doc.lower())

    def test_gfx1201_examples_use_validated_physical_selection(self):
        for relative in ("README.md", "docs/build-and-run.md", "docs/benchmarks.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("HIP_VISIBLE_DEVICES=0", text, relative)
            self.assertIn("SUPERSONIC_R9700_GPU_ID", text, relative)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("amd-smi static --asic --json", readme)
        self.assertIn("tools/select-r9700-device.py", readme)
        self.assertIn("SUPERSONIC_R9700_GPU_ARCH", readme)

    def test_readme_selector_snippet_fails_closed_and_exports_valid_selection(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        snippet = readme.split("```bash", 1)[1].split("```", 1)[0]
        self.assertIn("set -euo pipefail", snippet)
        self.assertIn('selection="$(', snippet)
        self.assertNotIn("done < <(", snippet)
        self.assertIn("declare -A", snippet)

        valid_payload = {
            "gpu_data": [
                {
                    "gpu": 0,
                    "asic": {
                        "market_name": "AMD Radeon RX 7900 XTX",
                        "device_id": "0x744c",
                        "target_graphics_version": "gfx1100",
                    },
                },
                {
                    "gpu": 1,
                    "asic": {
                        "market_name": "AMD Radeon AI PRO R9700",
                        "device_id": "0x7551",
                        "target_graphics_version": "gfx1201",
                    },
                },
            ]
        }
        invalid_payload = {
            "gpu_data": [
                {"gpu": 0, "asic": {"target_graphics_version": "gfx1100"}}
            ]
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_amd_smi = fake_bin / "amd-smi"
            fake_amd_smi.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "[[ \"$*\" == \"static --asic --json\" ]]\n"
                "cat \"$FAKE_AMD_SMI_JSON\"\n",
                encoding="utf-8",
            )
            fake_amd_smi.chmod(0o755)

            def run_snippet(payload: dict, *, override: str = ""):
                payload_path = root / "payload.json"
                payload_path.write_text(json.dumps(payload), encoding="utf-8")
                marker = root / "selection.env"
                if marker.exists():
                    marker.unlink()
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                        "TMPDIR": str(root),
                        "FAKE_AMD_SMI_JSON": str(payload_path),
                        "MARKER": str(marker),
                        "HIP_VISIBLE_DEVICES": "stale",
                        "SUPERSONIC_R9700_GPU_ARCH": "stale",
                        "SUPERSONIC_DEVICE": "stale",
                    }
                )
                if override:
                    environment["SUPERSONIC_R9700_GPU_ID"] = override
                else:
                    environment.pop("SUPERSONIC_R9700_GPU_ID", None)
                script = (
                    snippet
                    + '\nprintf \'%s\\n\' "${SUPERSONIC_R9700_GPU_ID-}" '
                    + '"${SUPERSONIC_R9700_GPU_ARCH-}" "${HIP_VISIBLE_DEVICES-}" '
                    + '"${SUPERSONIC_DEVICE-}" > "$MARKER"\n'
                )
                result = subprocess.run(
                    ["bash", "-c", script],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                return result, marker

            result, marker = run_snippet(valid_payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["1", "gfx1201", "1", "0"])

            result, marker = run_snippet(valid_payload, override="0")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

            result, marker = run_snippet(invalid_payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

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

    def test_public_positioning_is_measured_performance_first(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        performance = (ROOT / "docs" / "performance.md").read_text(encoding="utf-8")

        self.assertIn("performance-specialized", readme.lower())
        self.assertIn("maximum measured performance", readme.lower())
        self.assertIn("reproducible", performance.lower())
        self.assertNotIn("megakernel", readme.lower())

    def test_direct_gqh_quickstart_names_both_artifact_roles(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Quick start", readme)
        self.assertIn("cargo run --release --bin supersonic", readme)
        self.assertIn("--model qwen3.8-27b", readme)
        self.assertIn("--model-dir", readme)
        self.assertIn("--gguf-file", readme)
        self.assertIn("config.json", readme)
        self.assertIn("tokenizer", readme.lower())
        self.assertIn("chat template", readme.lower())

    def test_performance_number_is_omitted_or_fully_qualified(self):
        public_text = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in load_checker().ACTIVE_DOCS
        )
        if not re.search(r"37\.2\s*tok/s", public_text, re.IGNORECASE):
            return

        lowered = public_text.lower()
        self.assertRegex(lowered, r"commit\s*[:=]?\s*[0-9a-f]{7,40}")
        self.assertIn("artifact", lowered)
        self.assertRegex(lowered, r"gfx1100|gfx1201")
        self.assertRegex(lowered, r"prompt|workload")
        self.assertRegex(lowered, r"correctness|parity")

    def test_contributor_guidance_is_canonical_and_complete(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        lowered = agents.lower()
        for crate in (
            "crates/core",
            "crates/gpu-hal",
            "crates/kernel-ffi",
            "crates/model-store",
            "crates/qwen38",
            "crates/runtime",
            "crates/runner",
        ):
            self.assertIn(crate, lowered)
        self.assertIn("internal flm foundation", lowered)
        self.assertIn("abi", lowered)
        self.assertRegex(lowered, r"cpu[- ]safe|test tiers?|testing tiers?")
        self.assertRegex(lowered, r"unsupported .*fail|fail .*unsupported")

        claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        self.assertRegex(claude.lower(), r"canonical guidance|see .*agents\.md|agents\.md")
        self.assertLessEqual(len(claude.splitlines()), 12)

    def test_obsolete_documentation_is_removed_from_active_tree(self):
        obsolete = (
            "docs/bake-distribution.md",
            "docs/detailed_performance.md",
            "docs/dflash.md",
            "docs/feature-compatibility.md",
            "docs/lowlevel-memory.md",
            "docs/quality.md",
            "docs/server.md",
            "docs/specprefill.md",
            "docs/development/consolidation-roadmap.md",
            "docs/development/kernel-build-groups.md",
            "docs/development/kernel-lab.md",
        )
        for relative in obsolete:
            self.assertFalse((ROOT / relative).exists(), relative)

        for directory in ("docs/bringup", "docs/optimization", "docs/papers", "docs/plans", "docs/research"):
            self.assertFalse((ROOT / directory).exists(), directory)

        superpowers_root = ROOT / "docs" / "superpowers"
        for path in superpowers_root.glob("specs/*.md"):
            self.assertEqual(
                path.name,
                "2026-08-23-qwen38-rocm-product-slimming-design.md",
                path.as_posix(),
            )
        for path in superpowers_root.glob("plans/*.md"):
            self.assertEqual(
                path.name,
                "2026-08-23-qwen38-rocm-product-slimming.md",
                path.as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
