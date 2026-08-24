import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

ACTIVE_DOC_PATHS = (
    "README.md",
    "docs/build-and-run.md",
    "docs/supported-matrix.md",
    "docs/artifact-format.md",
    "docs/testing.md",
    "docs/benchmarks.md",
    "docs/performance.md",
    "tools/check-active-docs.py",
    "tools/check-kernel-groups.py",
    "tools/check-pr-diff.py",
    "tools/check-qwen38-artifacts.py",
    "tools/check-retained-source-terms.py",
    "tools/check-support-matrix.py",
    "tools/check-tool-inventory.py",
    "tools/parse-rocm-smi.py",
    "tools/select-r9700-device.py",
    "tests/test_active_docs.py",
    "tests/test_ci_workflows.py",
    "tests/test_kernel_groups.py",
    "tests/test_r9700_helpers.py",
    "tests/test_pr_diff.py",
    "tests/test_qwen38_artifact_preflight.py",
    "tests/test_support_matrix.py",
)


def load_helper(filename: str, module_name: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class WorkflowContractTests(unittest.TestCase):
    def test_quick_benchmark_is_serial_gpu_candidate_job(self):
        workflow = WORKFLOWS / "benchmark-quick.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", text)
        self.assertRegex(
            text,
            re.compile(r"runs-on:\s*\[self-hosted,\s*linux,\s*rocm,\s*gfx1201\]"),
        )
        self.assertIn("timeout-minutes: 30", text)
        self.assertIn("--suite quick", text)
        self.assertIn("concurrency:", text)
        self.assertIn("RUST_TEST_THREADS: \"1\"", text)
        self.assertIn("amd-smi static --asic --bus --json", text)
        self.assertIn("amd-smi list -e --json", text)
        self.assertIn("merge-amd-smi-provenance.py", text)
        self.assertIn("tools/select-r9700-device.py", text)
        self.assertIn("HIP_VISIBLE_DEVICES", text)
        self.assertIn("SUPERSONIC_DEVICE", text)
        self.assertRegex(text, re.compile(r"(?i)gpu.*idle"))
        self.assertIn("--clock-policy locked", text)
        self.assertIn("--gpu-clock-mhz", text)
        self.assertIn("--memory-clock-mhz", text)
        self.assertIn("--power-cap-watts", text)
        self.assertIn("--gpu-static-json", text)
        self.assertIn("SUPERSONIC_GQH_GGUF", text)
        self.assertIn("SUPERSONIC_QWEN38_MODEL_DIR", text)
        self.assertIn("if: always()", text)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", text)
        self.assertNotIn("actions/upload-artifact@v", text)
        self.assertNotRegex(text, re.compile(r"\bgit\s+(?:commit|push)\b"))
        self.assertNotIn("deploy-pages", text)
        self.assertNotIn("continue-on-error: true", text)

    def test_full_is_manual_serial_and_six_hours(self):
        workflow = WORKFLOWS / "benchmark-full.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", text)
        self.assertIn("timeout-minutes: 390", text)
        self.assertIn("concurrency:", text)
        self.assertIn("--suite full", text)
        self.assertNotIn("continue-on-error: true", text)
        self.assertRegex(
            text,
            re.compile(r"runs-on:\s*\[self-hosted,\s*linux,\s*rocm,\s*gfx1201\]"),
        )
        self.assertIn("--peer-artifact", text)
        self.assertIn("tools/external/llama-cpp-version.txt", text)
        self.assertIn("SUPERSONIC_LLAMA_CPP_SERVER", text)
        self.assertIn("SUPERSONIC_LLAMA_CPP_ARTIFACT", text)
        self.assertIn('test "$(basename "$llama_server")" = "llama-server"', text)
        self.assertIn('export PATH="$(dirname "$SUPERSONIC_LLAMA_CPP_SERVER"):$PATH"', text)
        self.assertIn('"$llama_server" --version 2>&1 | head -n 1', text)
        self.assertIn("amd-smi static --asic --bus --json", text)
        self.assertIn("amd-smi list -e --json", text)
        self.assertIn("merge-amd-smi-provenance.py", text)
        self.assertIn("amd-smi-provenance.json", text)
        self.assertIn("--clock-policy locked", text)
        self.assertIn("RUST_TEST_THREADS: \"1\"", text)
        self.assertIn("if: always()", text)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", text)
        self.assertNotRegex(text, re.compile(r"\bgit\s+(?:commit|push)\b"))
        self.assertNotIn("deploy-pages", text)

    def test_pages_validates_before_deploy(self):
        workflow = WORKFLOWS / "benchmark-pages.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertLess(text.index("validate --publishable"), text.index(" render "))
        self.assertLess(text.index(" render "), text.index("deploy-pages"))
        self.assertNotIn("pull_request_target", text)
        self.assertIn("pull_request:", text)
        self.assertIn("pages: write", text)
        self.assertIn("id-token: write", text)
        self.assertRegex(text, re.compile(r"github\.ref\s*==\s*['\"]refs/heads/main['\"]"))
        for action in (
            "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
            "actions/configure-pages@983d7736d9b0ae728b81ab479565c72886d7745b",
            "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa",
            "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e",
        ):
            self.assertIn(action, text)

    def test_pages_readme_only_results_skip_baseline_cleanly(self):
        workflow = WORKFLOWS / "benchmark-pages.yml"
        text = workflow.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "benchmarks" / "results"
            results.mkdir(parents=True)
            (results / "README.md").write_text(
                "Results are produced by the GPU workflow.\n",
                encoding="utf-8",
            )
            self.assertEqual(tuple(results.rglob("*.json")), ())

        self.assertIn("id: detect-results", text)
        self.assertIn("has_results", text)
        self.assertIn("No committed benchmark baseline", text)
        self.assertIn("steps.detect-results.outputs.has_results", text)
        self.assertRegex(text, re.compile(r"has_results\s*==\s*['\"]true['\"]"))

    def test_pages_malformed_first_record_remains_blocking(self):
        workflow = WORKFLOWS / "benchmark-pages.yml"
        text = workflow.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "benchmarks" / "results"
            results.mkdir(parents=True)
            malformed = results / "000-malformed.json"
            malformed.write_text("{not-json\n", encoding="utf-8")
            (results / "001-valid.json").write_text(
                (ROOT / "tests" / "benchmark_fixtures" / "valid-result-v1.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                tuple(sorted(results.rglob("*.json"))),
                (malformed, results / "001-valid.json"),
            )
            from tools.benchmark import validation

            with self.assertRaises(ValueError):
                validation.validate_bundle(results, require_complete=True)

        self.assertIn("find benchmarks/results -type f -name '*.json'", text)
        self.assertIn("validate --publishable benchmarks/results", text)
        self.assertNotIn("head -n 1", text)

    def test_gpu_workflows_share_one_serial_per_device_group(self):
        paths = (
            WORKFLOWS / "benchmark-quick.yml",
            WORKFLOWS / "benchmark-full.yml",
            WORKFLOWS / "qwen38-gfx1201.yml",
        )
        groups = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            match = re.search(r"^  group:\s*(.+)$", text, re.MULTILINE)
            self.assertIsNotNone(match, path)
            groups.append(match.group(1).strip())
            self.assertRegex(text, re.compile(r"^  cancel-in-progress:\s*false\s*$", re.MULTILINE))
        self.assertEqual(len(set(groups)), 1)
        self.assertIn("gfx1201", groups[0])

    def test_gpu_workflows_keep_raw_hip_mapping_separate_from_visible_device(self):
        for path in (
            WORKFLOWS / "benchmark-quick.yml",
            WORKFLOWS / "benchmark-full.yml",
            WORKFLOWS / "qwen38-gfx1201.yml",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn('[[ "$SUPERSONIC_GPU_LOGICAL" =~ ^[0-9]+$ ]]', text)
            self.assertNotIn('[[ "$SUPERSONIC_GPU_LOGICAL" == "$SUPERSONIC_DEVICE" ]]', text)

    def test_benchmark_workflows_pass_captured_toolchain_version_files(self):
        for name in ("benchmark-quick.yml", "benchmark-full.yml"):
            with self.subTest(workflow=name):
                text = (WORKFLOWS / name).read_text(encoding="utf-8")
                self.assertIn('--rocm-version-file "$BENCHMARK_OUTPUT_ROOT/rocm-driver-version.txt"', text)
                self.assertIn('--hip-version-file "$BENCHMARK_OUTPUT_ROOT/hipcc-version.txt"', text)
                self.assertIn("run_id=", text)
                self.assertIn('--run-id "$run_id"', text)

    def test_benchmark_workflows_fail_closed_on_non_q3kxl_artifact_digests(self):
        digest = "c710b03bf5bf224107d0ae1567b97f1c8638ef35c5f431c39479a3ecc963bd98"
        for name in ("benchmark-quick.yml", "benchmark-full.yml"):
            with self.subTest(workflow=name):
                text = (WORKFLOWS / name).read_text(encoding="utf-8")
                self.assertIn(f'expected_q3kxl_sha256="{digest}"', text)
                self.assertIn('test "$SUPERSONIC_GQH_GGUF_SHA256" = "$expected_q3kxl_sha256"', text)
        full = (WORKFLOWS / "benchmark-full.yml").read_text(encoding="utf-8")
        self.assertIn(
            'test "$SUPERSONIC_LLAMA_CPP_ARTIFACT_SHA256" = "$expected_q3kxl_sha256"',
            full,
        )

    def test_cpu_ci_validates_benchmark_fixtures_without_gpu(self):
        workflow = WORKFLOWS / "ci.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("python3 tools/supersonic-bench.py render tests/benchmark_fixtures", text)
        self.assertIn("tests.test_benchmark_manifests", text)
        self.assertIn("tests.test_benchmark_validation", text)
        self.assertIn("tests.test_amd_smi_provenance", text)
        self.assertNotIn("self-hosted", text)

    def test_r9700_workflow_data_flow_validates_physical_to_logical_mapping(self):
        selector = load_helper("select-r9700-device.py", "workflow_select_r9700")
        devices = selector.parse_devices(
            json.dumps(
                {
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
            )
        )
        selected = selector.select_device(devices)
        environment = selector.render_environment(selected)
        self.assertEqual(environment["SUPERSONIC_R9700_GPU_ID"], "1")
        self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(environment["SUPERSONIC_DEVICE"], "0")

        parser = load_helper("parse-rocm-smi.py", "workflow_parse_rocm_smi")
        utilization = parser.parse_utilization(
            "GPU use (%): 0\nGPU Memory Allocated (VRAM%): 3\n"
        )
        self.assertLessEqual(utilization.gpu_use_percent, 10)
        self.assertLessEqual(utilization.vram_use_percent, 20)

    def test_cpu_pull_request_gate_is_gpu_free_and_uses_retained_commands(self):
        workflow = WORKFLOWS / "ci.yml"
        self.assertTrue(workflow.is_file(), workflow)
        text = workflow.read_text(encoding="utf-8")

        self.assertIn("pull_request:", text)
        self.assertNotIn("self-hosted", text)
        self.assertNotIn("HIP_VISIBLE_DEVICES", text)
        self.assertNotIn("/dev/kfd", text)
        self.assertNotIn("/dev/dri", text)
        self.assertNotIn("container:", text)
        self.assertIn("ROCM_VERSION: \"7.2.4\"", text)
        self.assertIn("ROCM_HIP_SDK_VERSION: \"7.2.4.70204-93~24.04\"", text)
        self.assertIn("repo.radeon.com/rocm/apt/${ROCM_VERSION}", text)
        self.assertIn("rocm-hip-sdk", text)
        self.assertIn("CA8BB4727A47B4D09B4EE8969386B48A1A693C5C", text)
        self.assertIn("key_fingerprint", text)
        self.assertIn("GITHUB_PATH", text)
        self.assertIn("git --version", text)
        self.assertIn("python3", text)
        self.assertIn("python3 --version", text)
        self.assertIn("import tomllib", text)
        self.assertIn("command -v hipcc", text)
        self.assertIn("hipcc --version", text)
        self.assertLess(text.index("rocm-hip-sdk"), text.index("hipcc --version"))
        self.assertLess(text.index("git --version"), text.index("python3"))
        self.assertLess(text.index("hipcc --version"), text.index("command -v cargo"))
        self.assertLess(text.index("hipcc --version"), text.index("cargo fmt --all --check"))
        self.assertIn("fetch-depth: 0", text)
        self.assertIn("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683", text)
        self.assertNotIn("actions/checkout@v", text)
        self.assertIn("github.event.pull_request.base.sha", text)
        self.assertIn("github.event.pull_request.head.sha", text)
        self.assertIn("python3 tools/check-pr-diff.py", text)
        self.assertIn("git merge-base", text)
        for path in ACTIVE_DOC_PATHS:
            self.assertIn(f'      - "{path}"', text, path)
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
            "python3 tools/check-active-docs.py",
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
        self.assertNotIn("|| '0'", text)
        self.assertNotIn(":-0", text)
        self.assertIn("amd-smi static --asic --bus --json", text)
        self.assertIn("amd-smi list -e --json", text)
        self.assertIn("merge-amd-smi-provenance.py", text)
        self.assertIn("tools/select-r9700-device.py", text)
        self.assertIn("tools/parse-rocm-smi.py", text)
        self.assertIn("GITHUB_ENV", text)
        self.assertIn('rocm-smi -d "$physical_gpu"', text)
        self.assertIn("timeout --foreground", text)
        self.assertIn("deadline", text)
        for line in text.splitlines():
            if re.search(r"\b(?:rocm-smi|amd-smi)\s", line) and not line.lstrip().startswith("#"):
                self.assertIn("timeout", line, line)
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
        self.assertIn("hipcc --version", text)
        self.assertIn("rocm-driver-version.txt", text)
        self.assertIn("tools/qwen38-reproducibility.py", text)
        self.assertIn("reproducibility.json", text)
        for option in (
            '--commit "$(git rev-parse HEAD)"',
            "--hip-version-file",
            "--rocm-version-file",
            "--gpu-json",
            "--physical-gpu",
            "--gpu-arch",
            "--artifact",
            "--model-dir",
            "--ordinary",
            "--mtp",
            "--telemetry-root",
            "--prompt \"Hello\"",
            "--chat",
            "--max-new-tokens 8",
        ):
            self.assertIn(option, text, option)
        record_tool = (ROOT / "tools" / "qwen38-reproducibility.py").read_text(encoding="utf-8")
        for field in (
            '"commit"',
            '"toolchain"',
            '"physical_gpu"',
            '"artifact"',
            '"workload"',
            '"correctness"',
            '"ordinary_vs_mtp"',
            '"warmup_runs"',
            '"measured_runs"',
        ):
            self.assertIn(field, record_tool)
        self.assertRegex(text, re.compile(r"(?i)gpu.*idle"))
        self.assertIn("continue-on-error: true", text)
        self.assertIn("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683", text)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", text)
        self.assertNotIn("actions/upload-artifact@v", text)
        self.assertRegex(text, re.compile(r"if:\s*always\(\)"))

    def test_obsolete_kernel_lab_workflow_is_not_reintroduced(self):
        self.assertFalse((WORKFLOWS / "kernel-lab.yml").exists())


if __name__ == "__main__":
    unittest.main()
