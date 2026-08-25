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
    def test_benchmark_docs_define_tiers_clocks_and_cache(self):
        text = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8").lower()
        for term in (
            "quick",
            "10 minutes",
            "full",
            "six hours",
            "locked",
            "uncontrolled-clocks",
            "cold-load",
            "warm-resident",
        ):
            self.assertIn(term, text)

    def test_peer_claims_require_comparability_evidence(self):
        text = (ROOT / "docs" / "performance.md").read_text(encoding="utf-8").lower()
        for term in ("comparability", "artifact", "cache state", "clock", "sample count"):
            self.assertIn(term, text)

    def test_full_duration_rounds_and_sustained_clock_policy_are_documented(self):
        benchmark_text = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        performance_text = (ROOT / "docs" / "performance.md").read_text(encoding="utf-8")
        testing_text = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

        for text in (benchmark_text, performance_text, testing_text):
            self.assertIn("20,700-second minimum", text)
            self.assertIn("21,600-second hard budget", text)
        self.assertIn("balanced rounds", benchmark_text)
        self.assertIn("three consecutive loaded", benchmark_text)
        self.assertIn("three consecutive loaded", performance_text)

    def test_benchmark_recipes_match_versioned_cli_and_local_prerequisites(self):
        text = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        self.assertIn("HIP_ARCH=gfx1201 cargo build --release --workspace", text)
        self.assertIn("--rocm-version-file", text)
        self.assertIn("--hip-version-file", text)
        self.assertIn('run_id="quick-manual-', text)
        self.assertIn('run_id="full-manual-', text)
        self.assertIn("command -v llama-server", text)
        self.assertIn('test -r "$SUPERSONIC_LLAMA_CPP_ARTIFACT"', text)
        self.assertIn("The validator validates raw sample values", text)
        self.assertIn("The renderer deterministically derives sample", text)
        full = text.split("## Full candidate", 1)[1].split("## Cache and clock terminology", 1)[0]
        self.assertIn('export SUPERSONIC_LLAMA_CPP_ARTIFACT=', full)
        self.assertIn('--peer-artifact "$SUPERSONIC_LLAMA_CPP_ARTIFACT"', full)
        self.assertNotIn("--peer-artifact /path/to/pinned-peer-artifact.gguf", full)
        testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
        self.assertIn('run_id="quick-manual-', testing)
        self.assertIn('--run-id "$run_id"', testing)

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

    def test_removed_terms_are_case_insensitive_and_bare_identities_fail(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n", encoding="utf-8")
            (root / "README.md").write_text(
                "# Product\n\n"
                "supersonic_backends=cuda\n"
                "Gemma Phi Llama\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("supersonic_backends" in violation.lower() for violation in violations))
        self.assertTrue(any("gemma" in violation.lower() for violation in violations))
        self.assertTrue(any("phi" in violation.lower() for violation in violations))
        self.assertTrue(any("llama" in violation.lower() for violation in violations))

    def test_removed_stream_gemv_lifecycle_terms_are_rejected(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n\nCurrent GQH guidance.\n", encoding="utf-8")
            (root / "docs" / "testing.md").write_text(
                "# Testing\n\n"
                "The bridge has held GEMV arguments and a learned gate/up path.\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("held GEMV" in violation for violation in violations))
        self.assertTrue(any("learned gate/up" in violation for violation in violations))

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
                        "pci_bdf": "0000:03:00.0",
                    },
                    "logical_gpu": 1,
                },
                {
                    "gpu": 1,
                    "asic": {
                        "market_name": "AMD Radeon AI PRO R9700",
                        "device_id": "0x7551",
                        "target_graphics_version": "gfx1201",
                        "pci_bdf": "0000:65:00.0",
                    },
                    "logical_gpu": 0,
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

    def test_readme_device_probe_is_bounded_and_build_docs_do_not_claim_local_idle_polling(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        snippet = readme.split("```bash", 1)[1].split("```", 1)[0]
        self.assertIn("timeout --foreground 30s amd-smi static --asic --json", snippet)

        build_docs = (ROOT / "docs" / "build-and-run.md").read_text(encoding="utf-8").lower()
        self.assertNotIn("busy selected device is a failed run", build_docs)
        self.assertIn("self-hosted workflow", build_docs)

    def test_internal_flm_term_is_allowed_only_in_explicit_internal_section(self):
        checker = load_checker()
        allowed = "## Internal FLM foundation\n\nThis is contributor-only context.\n"
        self.assertEqual([], checker.find_text_violations(Path("README.md"), allowed))
        self.assertNotEqual(
            [], checker.find_text_violations(Path("README.md"), "The FLM backend is public.")
        )

    def test_internal_flm_section_is_bounded_by_same_or_higher_heading(self):
        checker = load_checker()
        nested = (
            "## Internal FLM foundation\n"
            "### Contributor details\n"
            "FLM codec work stays internal here.\n"
        )
        self.assertEqual([], checker.find_text_violations(Path("AGENTS.md"), nested))

        after_same_level = nested + "## Public contract\nFLM is public.\n"
        self.assertNotEqual([], checker.find_text_violations(Path("AGENTS.md"), after_same_level))

        after_higher_level = nested + "# Public contract\nFLM is public.\n"
        self.assertNotEqual([], checker.find_text_violations(Path("AGENTS.md"), after_higher_level))

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

    def test_same_document_anchor_is_checked(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n## Existing\n", encoding="utf-8")
            (root / "README.md").write_text(
                "# Product\n\n[broken](#missing)\n", encoding="utf-8"
            )

            violations = checker.find_violations(root)

        self.assertTrue(any("#missing" in violation for violation in violations))

    def test_github_anchor_parser_handles_duplicates_setext_and_indentation(self):
        checker = load_checker()
        markdown = (
            "## Repeat\n"
            "## Repeat\n"
            "Repeat\n"
            "=======\n"
            "   # Indented title\n"
            "    # Code title\n"
        )
        self.assertEqual(
            checker.anchors_for(markdown),
            {"repeat", "repeat-1", "repeat-2", "indented-title"},
        )

    def test_github_anchor_parser_ignores_variable_fenced_code_blocks(self):
        checker = load_checker()
        markdown = (
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
            "## After\n"
        )
        self.assertEqual(
            checker.anchors_for(markdown),
            {"before", "before-1", "after", "after-1"},
        )

    def test_anchor_links_accept_github_duplicate_and_setext_ids_but_reject_missing(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n", encoding="utf-8")
            (root / "README.md").write_text(
                "## Repeat\n## Repeat\n"
                "Setext\n=======\n"
                "[first](#repeat) [second](#repeat-1) [setext](#setext) "
                "[missing](#repeat-2)\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertEqual(1, len(violations))
        self.assertIn("#repeat-2", violations[0])

    def test_fenced_heading_anchor_is_invalid_while_real_headings_link(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in checker.ACTIVE_DOCS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Product\n", encoding="utf-8")
            (root / "README.md").write_text(
                "# Before\n"
                "```\n"
                "# Hidden\n"
                "```\n"
                "## After\n"
                "[before](#before) [after](#after) [hidden](#hidden)\n",
                encoding="utf-8",
            )

            violations = checker.find_violations(root)

        self.assertEqual(1, len(violations))
        self.assertIn("#hidden", violations[0])

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

    def test_tokenizer_config_and_chat_template_wording_matches_chat_only_code_path(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        build = (ROOT / "docs" / "build-and-run.md").read_text(encoding="utf-8").lower()
        chat_template = (ROOT / "crates" / "runtime" / "src" / "chat_template.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn("only when `--chat` is used", readme)
        self.assertRegex(build, r"when\s+`--chat` is used")
        self.assertNotIn("gemma", chat_template.lower())
        self.assertNotIn("server", chat_template.lower())

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

    def test_every_throughput_number_requires_colocated_strong_evidence(self):
        checker = load_checker()
        weak = (
            "A benchmark measured 37.2 tok/s.\n\n"
            "The commit is abcdef1 and the artifact is documented elsewhere.\n"
        )
        weak_violations = checker.find_performance_violations(Path("README.md"), weak)
        self.assertTrue(weak_violations)

        strong = (
            "engine: supersonic, version: 1.0, commit: abcdef1, "
            "artifact: /tmp/qwen38.gqh.gguf, target: gfx1201, prompt=Hello, "
            "clock policy: locked, cache state: warm-resident, process_reuse=false, "
            "warmup=1, statistic: median, sample count: 3, correctness: pass, "
            "direct run: benchmarks/results/run.json, decode measurement: 37.2 tok/s.\n"
        )
        self.assertEqual([], checker.find_performance_violations(Path("README.md"), strong))

        mixed = strong + "\nA second run reports 100 tok/s without its run record.\n"
        self.assertTrue(checker.find_performance_violations(Path("README.md"), mixed))

        generic_context = (
            "commit: abcdef1, artifact: /tmp/qwen38.gqh.gguf, target: gfx1201; "
            "workload and measurement are documented elsewhere: 37.2 tok/s.\n"
        )
        self.assertTrue(checker.find_performance_violations(Path("README.md"), generic_context))

        placeholder = (
            "commit: abcdef1, artifact: /path/to/qwen38.gqh.gguf, target: gfx1201, "
            "prompt=Hello, warmup=1, median decode measurement: 37.2 tok/s.\n"
        )
        self.assertTrue(checker.find_performance_violations(Path("README.md"), placeholder))

        shared_evidence = (
            "commit: abcdef1, artifact: /tmp/qwen38.gqh.gguf, target: gfx1201, "
            "prompt=Hello, warmup=1, median decode measurement: "
            "37.2 tok/s and 38.4 tok/s.\n"
        )
        self.assertTrue(checker.find_performance_violations(Path("README.md"), shared_evidence))

        structured_records = (
            "engine: supersonic, version: 1.0, commit: abcdef1, "
            "artifact: /tmp/qwen38-a.gqh.gguf, target: gfx1201, prompt=Hello, "
            "clock policy: locked, cache state: warm-resident, process_reuse=false, "
            "warmup=1, statistic: median, sample count: 3, correctness: pass, "
            "direct run: benchmarks/results/a.json, decode measurement: 37.2 tok/s; "
            "engine: peer, version: 2.0, commit: abcdef2, "
            "artifact: /tmp/qwen38-b.gqh.gguf, target: gfx1100, prompt=World, "
            "clock policy: locked, cache state: warm-resident, process_reuse=false, "
            "warmup=2, statistic: median, sample count: 3, correctness: pass, "
            "direct run: benchmarks/results/b.json, decode measurement: 38.4 tok/s.\n"
        )
        self.assertEqual(
            [], checker.find_performance_violations(Path("README.md"), structured_records)
        )

    def test_numeric_claim_requires_peer_evidence_and_speedup_is_qualified(self):
        checker = load_checker()
        current_contract = (
            "commit: abcdef1, artifact: /tmp/qwen38.gqh.gguf, target: gfx1201, "
            "prompt=Hello, warmup=1, median decode measurement: 37.2 tok/s.\n"
        )
        violations = checker.find_performance_violations(Path("README.md"), current_contract)
        self.assertTrue(violations)
        self.assertTrue(any("engine" in violation for violation in violations))
        self.assertTrue(any("clock" in violation for violation in violations))
        self.assertTrue(any("cache" in violation for violation in violations))
        self.assertTrue(any("sample" in violation for violation in violations))
        self.assertTrue(any("correctness" in violation for violation in violations))
        self.assertTrue(any("direct run" in violation for violation in violations))

        self.assertTrue(
            checker.find_performance_violations(
                Path("README.md"),
                "The 2.0x speedup claim has no run record or comparability evidence.\n",
            )
        )
        structured_evidence = (
            "engine=supersonic, version=1.0, commit=abcdef1, artifact=qwen38.gqh.gguf, "
            "gpu=gfx1201, workload=quick-short, clock=locked, "
            "cache_state=warm-resident, process_reuse=false, statistic=median, "
            "sample_count=3, correctness=pass, direct_run=run-1, 37.2 tok/s.\n"
        )
        self.assertEqual(
            [], checker.find_performance_violations(Path("README.md"), structured_evidence)
        )
        self.assertEqual(
            [],
            checker.find_performance_violations(
                Path("README.md"),
                "A prose explanation can mention a 2.0x multiplier without making a speed claim.\n",
            ),
        )

    def test_common_numeric_performance_phrasings_require_complete_evidence(self):
        checker = load_checker()
        for phrase in (
            "37 tokens per second",
            "1.2 milliseconds per token",
            "speedup is 2.0x",
            "50% faster",
            "1 token per second",
            "1 millisecond per token",
            "37 tok/s",
            "1.2 ms/token",
            "speed-up is 2.0x",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(checker.find_performance_violations(Path("README.md"), phrase))

    def test_numeric_claim_bypass_and_non_performance_multiplier_are_safe(self):
        checker = load_checker()
        self.assertEqual(
            [],
            checker.find_performance_violations(
                Path("README.md"), "The 2.0x multiplier is a scale factor."
            ),
        )
        for phrase in ("37 tokens\u00a0per\u00a0second", "speedup is\u00a02.0x", "50 percent faster"):
            with self.subTest(phrase=phrase):
                self.assertTrue(checker.find_performance_violations(Path("README.md"), phrase))

    def test_testing_artifact_block_defines_and_propagates_strict_environment(self):
        document = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
        section = document.split("## `gfx1201` artifact gate", 1)[1]
        block = section.split("```bash", 1)[1].split("```", 1)[0]
        for name in (
            "SUPERSONIC_GQH_GGUF",
            "SUPERSONIC_QWEN38_MODEL_DIR",
            "SUPERSONIC_GQH_8192_GGUF",
        ):
            self.assertRegex(block, rf"export {name}=\"\$\{{{name}:-[^\"]+\}}\"")
        self.assertIn("SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1", block)
        self.assertIn("tools/check-qwen38-artifacts.py --require-8192", block)
        self.assertIn("--include-ignored", block)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                """#!/usr/bin/python3
import os
import sys
from pathlib import Path

if sys.argv[1:] != ["tools/check-qwen38-artifacts.py", "--require-8192"]:
    raise SystemExit(f"unexpected python args: {sys.argv[1:]}")
line = (
    "python|GQH=" + os.environ["SUPERSONIC_GQH_GGUF"]
    + "|MODEL=" + os.environ["SUPERSONIC_QWEN38_MODEL_DIR"]
    + "|8192=" + os.environ["SUPERSONIC_GQH_8192_GGUF"]
    + "|REQ=" + os.environ["SUPERSONIC_REQUIRE_GQH_ARTIFACTS"] + "\\n"
)
with Path(os.environ["MARKER"]).open("a", encoding="utf-8") as out:
    out.write(line)
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_cargo = fake_bin / "cargo"
            fake_cargo.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf 'cargo|%s|GQH=%s|MODEL=%s|8192=%s|REQ=%s\\n' "$*" \
  "$SUPERSONIC_GQH_GGUF" "$SUPERSONIC_QWEN38_MODEL_DIR" \
  "$SUPERSONIC_GQH_8192_GGUF" "$SUPERSONIC_REQUIRE_GQH_ARTIFACTS" >> "$MARKER"
""",
                encoding="utf-8",
            )
            fake_cargo.chmod(0o755)
            marker = root / "propagation.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                    "MARKER": str(marker),
                }
            )
            result = subprocess.run(
                ["bash", "-c", "set -euo pipefail\n" + block],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = marker.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len([line for line in lines if line.startswith("python|")]), 1)
        cargo_lines = [line for line in lines if line.startswith("cargo|")]
        self.assertEqual(len(cargo_lines), 3)
        for line in lines:
            self.assertIn("GQH=/home/deano/gqh-artifacts/", line)
            self.assertIn("MODEL=/data/models/Qwen3.8-27B", line)
            self.assertIn("8192=/home/deano/gqh-artifacts/", line)
            self.assertIn("REQ=1", line)
        self.assertTrue(any("--include-ignored" in line for line in cargo_lines))

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
        self.assertIn("slim evolution policy", lowered)
        self.assertRegex(lowered, r"one (maintained )?implementation")
        self.assertRegex(lowered, r"tag .*before .*remov|before .*remov.*tag")
        self.assertIn("github", lowered)
        self.assertRegex(lowered, r"backward compatibility .*not .*default|no backward compatibility")
        self.assertRegex(lowered, r"maintenance cost")
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
        retained_specs = {
            "2026-08-23-qwen38-rocm-product-slimming-design.md",
            "2026-08-24-reproducible-benchmark-pages-design.md",
            "2026-08-25-deterministic-raw-q6-output-head-design.md",
        }
        for path in superpowers_root.glob("specs/*.md"):
            self.assertIn(path.name, retained_specs, path.as_posix())
        retained_plans = {
            "2026-08-23-qwen38-rocm-product-slimming.md",
            "2026-08-24-reproducible-benchmark-pages.md",
            "2026-08-24-six-hour-balanced-full-benchmark.md",
            "2026-08-25-deterministic-raw-q6-output-head.md",
        }
        for path in superpowers_root.glob("plans/*.md"):
            self.assertIn(path.name, retained_plans, path.as_posix())


if __name__ == "__main__":
    unittest.main()
