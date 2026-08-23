import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class WorkflowContractTests(unittest.TestCase):
    def test_cpu_pull_request_gate_is_gpu_free_and_uses_retained_commands(self):
        workflow = WORKFLOWS / "ci.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertIn("pull_request:", text)
        self.assertNotIn("self-hosted", text)
        self.assertNotIn("HIP_VISIBLE_DEVICES", text)
        self.assertNotIn("/dev/kfd", text)
        self.assertNotIn("/dev/dri", text)
        self.assertIn("HIP_ARCH=gfx1201", text)
        for command in (
            "cargo fmt --all --check",
            "cargo check --workspace --all-targets",
            "cargo test -p model-store --lib 'gqh::tests::'",
            "cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids'",
            "cargo test -p supersonic-runtime --lib mtp_accept_tests",
            "cargo test -p runner --test qwen38_cli_contract",
            "cargo test -p runner --test qwen38_startup_contract",
            "python3 tools/check-support-matrix.py",
            "python3 tools/check-kernel-groups.py",
            "python3 tools/check-tool-inventory.py",
            "python3 -m unittest discover -s tests -p 'test_*.py' -v",
        ):
            self.assertIn(command, text, command)

    def test_r9700_gate_is_serial_strict_and_publishes_results(self):
        workflow = WORKFLOWS / "qwen38-gfx1201.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertRegex(
            text,
            re.compile(r"runs-on:\s*\[self-hosted,\s*linux,\s*rocm,\s*gfx1201\]"),
        )
        self.assertIn("timeout-minutes: 45", text)
        self.assertIn("HIP_VISIBLE_DEVICES", text)
        self.assertIn("--device 0", text)
        self.assertIn("SUPERSONIC_REQUIRE_GQH_ARTIFACTS", text)
        self.assertIn("SUPERSONIC_GQH_GGUF", text)
        self.assertIn("SUPERSONIC_QWEN38_MODEL_DIR", text)
        self.assertIn("python3 tools/check-qwen38-artifacts.py", text)
        self.assertIn("RUST_TEST_THREADS=1", text)
        self.assertIn("--test-threads=1", text)
        self.assertIn("--include-ignored", text)
        self.assertIn("cargo build --release", text)
        self.assertIn("cargo test --release -p kernel-ffi", text)
        self.assertIn("cargo test --release -p qwen38 --test qwen38_gqh_gguf_crawl", text)
        self.assertIn(
            "cargo test --release -p supersonic-runtime --test qwen38_gqh_decode_rung11",
            text,
        )
        self.assertIn("--speculative-decode", text)
        self.assertIn("--emit-generated-json", text)
        self.assertIn("rocm-smi", text)
        self.assertRegex(text, re.compile(r"(?i)gpu.*idle"))
        self.assertIn("continue-on-error: true", text)
        self.assertIn("actions/upload-artifact@v4", text)
        self.assertRegex(text, re.compile(r"if:\s*always\(\)"))

    def test_obsolete_kernel_lab_workflow_is_not_reintroduced(self):
        self.assertFalse((WORKFLOWS / "kernel-lab.yml").exists())


if __name__ == "__main__":
    unittest.main()
