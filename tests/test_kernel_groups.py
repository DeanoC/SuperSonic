import importlib.util
import re
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "crates" / "kernel-ffi" / "kernel-groups.toml"
KERNEL_BUILD = ROOT / "crates" / "kernel-ffi" / "build.rs"
HAL_BUILD = ROOT / "crates" / "gpu-hal" / "build.rs"
HAL_BACKEND = ROOT / "crates" / "gpu-hal" / "src" / "backend.rs"
HAL_LIB = ROOT / "crates" / "gpu-hal" / "src" / "lib.rs"
HAL_OPS = ROOT / "crates" / "gpu-hal" / "src" / "ops.rs"
CORE_LIB = ROOT / "crates" / "core" / "src" / "lib.rs"
CORE_BACKEND = ROOT / "crates" / "core" / "src" / "backend.rs"
KERNEL_FFI_SRC = ROOT / "crates" / "kernel-ffi" / "src"
RETAINED_SOURCE_VALIDATOR = ROOT / "tools" / "check-retained-source-terms.py"
RETAINED_BRIDGES = (
    ROOT / "kernels" / "full_attention_bridge.cpp",
    ROOT / "kernels" / "full_attention_bridge_4b.cpp",
)
RETAINED_SOURCE_ROOTS = (
    ROOT / "crates" / "core" / "src",
    ROOT / "crates" / "gpu-hal" / "src",
    ROOT / "crates" / "kernel-ffi" / "src",
    ROOT / "crates" / "qwen38" / "src",
    ROOT / "crates" / "runtime" / "src",
    ROOT / "kernels",
)


LEGACY_CONTENT_RE = re.compile(
    r"(?:certified[-_]kv|certifiedkv|dflash|specprefill|spec_prefill|"
    r"metal|cuda|qwen3[.]6|qwen36)",
    re.IGNORECASE,
)


def retained_source_text():
    for root in RETAINED_SOURCE_ROOTS:
        for path in root.rglob("*"):
            if path.suffix not in {".rs", ".c", ".cc", ".cpp", ".h", ".hip"}:
                continue
            yield path, path.read_text(encoding="utf-8")


def manifest_sources(group):
    """Return all source/module names that a kernel group retains."""

    sources = list(group.get("kernel_sources", []))
    sources.extend(group.get("native_sources", []))
    sources.extend(group.get("rust_modules", []))
    sources.extend(bridge["source"] for bridge in group.get("bridge", []))
    return sources


class KernelGroupManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with MANIFEST.open("rb") as handle:
            cls.data = tomllib.load(handle)

    def test_all_groups_are_hip_and_have_no_removed_source_names(self):
        groups = self.data["group"]
        self.assertTrue(groups)

        forbidden = re.compile(r"(?:_cuda|metal|gemma|phi|dflash|moe)", re.IGNORECASE)
        for group in groups:
            self.assertEqual(group["backend"], "hip", group["id"])
            for source in manifest_sources(group):
                self.assertIsNone(
                    forbidden.search(source),
                    f"{group['id']} retains removed kernel source {source}",
                )

    def test_retained_manifest_contains_dense_4b_prefill_and_gqh_sources(self):
        all_sources = {
            source for group in self.data["group"] for source in manifest_sources(group)
        }
        for source in (
            "kernels/full_attention.hip",
            "kernels/full_attention_4b.hip",
            "kernels/prefill_helpers.hip",
            "kernels/gqh.hip",
        ):
            self.assertIn(source, all_sources)


class HipOnlyBuildSurfaceTests(unittest.TestCase):
    def test_active_bridges_have_no_legacy_qwen35_environment_controls(self):
        legacy_env = re.compile(
            r"(?:getenv|var(?:_os)?)\s*\(\s*\"[^\"]*QWEN35[^\"]*\"",
            re.IGNORECASE,
        )
        violations = []
        for path in RETAINED_BRIDGES:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if legacy_env.search(line):
                    violations.append(f"{path}:{line_number}: {line.strip()}")
        self.assertEqual([], violations)

    def test_retained_qwen38_kernel_comments_use_canonical_geometry(self):
        sources = (
            ROOT / "kernels" / "full_attention.hip",
            ROOT / "kernels" / "full_attention_4b.hip",
        )
        product_ref = re.compile(r"qwen3[.]5", re.IGNORECASE)
        stale_refs = []
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if product_ref.search(line):
                    stale_refs.append(f"{path}:{line_number}: {line.strip()}")
            self.assertIn(
                "64 total for canonical Qwen3.8-27B",
                text,
                f"{path} must state the canonical 64-layer geometry",
            )
            self.assertNotRegex(
                text,
                r"\bProcesses\s+all\s+(?!64\b)\d+\s+decoder\s+layers\b",
                f"{path} must not describe a non-canonical layer count",
            )
        self.assertEqual([], stale_refs)
        self.assertIn(
            "partial rotary dimension (64 for canonical Qwen3.8-27B)",
            (sources[1]).read_text(encoding="utf-8"),
        )

    def test_retained_source_validator_scans_kernel_geometry(self):
        spec = importlib.util.spec_from_file_location(
            "check_retained_source_terms", RETAINED_SOURCE_VALIDATOR
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)

        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            kernel_root = root / "kernels"
            kernel_root.mkdir()
            stale = (
                "// qwen3.5 product identity\n"
                "// One struct per decoder layer (24 total for the retained Qwen3.8 geometry).\n"
            )
            for name in ("full_attention.hip", "full_attention_4b.hip"):
                (kernel_root / name).write_text(stale, encoding="utf-8")

            violations = validator.find_violations(root)

        rendered = "\n".join(term for _, _, term, _ in violations).lower()
        self.assertIn("qwen3.5", rendered)
        self.assertIn("24 total", rendered)

    def test_removed_ffi_source_surfaces_are_absent(self):
        removed_names = {
            "certified_kv.rs",
            "dflash.rs",
            "metal_host.rs",
            "metal_native.rs",
            "metal_link_stubs.cc",
            "metal_native.mm",
            "metal_native_ffi.h",
            "metal_native_ffi_contract.cc",
            "qwen36_moe",
        }
        for path in KERNEL_FFI_SRC.rglob("*"):
            self.assertNotIn(path.name, removed_names, path)

    def test_build_scripts_do_not_select_or_link_removed_backends(self):
        kernel_build = KERNEL_BUILD.read_text(encoding="utf-8")
        hal_build = HAL_BUILD.read_text(encoding="utf-8")
        for text in (kernel_build, hal_build):
            self.assertNotIn("SUPERSONIC_BACKENDS", text)
            self.assertNotIn("supersonic_backend_cuda", text)
            self.assertNotIn("supersonic_backend_metal", text)
            self.assertNotRegex(text, r"\b(?:nvcc|CUDA|Metal|metal)\b")

    def test_build_manifest_is_the_only_kernel_group_source_of_truth(self):
        build = KERNEL_BUILD.read_text(encoding="utf-8")
        self.assertIn("kernel-groups.toml", build)
        self.assertNotRegex(
            build,
            r"\b(?:HIP_GROUPS|HIP_BRIDGES|KERNEL_RERUN_PATHS)\b",
        )

    def test_retained_sources_have_no_removed_implementation_content(self):
        violations = []
        for path, source in retained_source_text():
            for line_number, line in enumerate(source.splitlines(), start=1):
                match = LEGACY_CONTENT_RE.search(line)
                if match:
                    violations.append(f"{path}:{line_number}: {match.group(0)}")
        self.assertEqual([], violations[:20], "legacy retained source content: " + "; ".join(violations[:20]))
        self.assertEqual([], violations)

    def test_public_ffi_and_backend_api_has_no_removed_surfaces(self):
        api_files = (
            ROOT / "crates" / "kernel-ffi" / "src" / "lib.rs",
            ROOT / "crates" / "kernel-ffi" / "src" / "prefill_ffi.rs",
            ROOT / "crates" / "kernel-ffi" / "src" / "qwen38.rs",
            ROOT / "crates" / "gpu-hal" / "src" / "backend.rs",
            ROOT / "crates" / "gpu-hal" / "src" / "lib.rs",
        )
        forbidden_public = re.compile(
            r"\b(?:Backend::(?:Cuda|Metal)|(?:pub\s+)?(?:fn|struct|enum|type|const|static)\s+"
            r"[^\n]*(?:certified|dflash|specprefill|metal|cuda|qwen36))",
            re.IGNORECASE,
        )
        violations = []
        for path in api_files:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if forbidden_public.search(line):
                    violations.append(f"{path}:{line_number}: {line.strip()}")
        self.assertEqual([], violations)

    def test_gpu_hal_exposes_only_hip_backend_surfaces(self):
        backend = HAL_BACKEND.read_text(encoding="utf-8")
        lib = HAL_LIB.read_text(encoding="utf-8")
        ops = HAL_OPS.read_text(encoding="utf-8")
        self.assertRegex(backend, r"pub enum Backend\s*\{\s*Hip,\s*\}")
        self.assertNotIn("compiled_backends", backend)
        self.assertIn("Backend::Hip\n}", backend)
        self.assertNotIn("supersonic_backend_cuda", backend)
        self.assertNotIn("supersonic_backend_metal", backend)
        self.assertNotIn("mod cuda_sys", lib)
        self.assertNotIn("mod metal_sys", lib)
        self.assertNotIn("use crate::cuda_sys", ops)
        self.assertNotIn("use crate::metal_sys", ops)
        self.assertFalse((ROOT / "crates" / "gpu-hal" / "src" / "vmm.rs").exists())

    def test_hip_surfaces_have_no_disabled_non_hip_branches(self):
        for path in (
            HAL_OPS,
            KERNEL_FFI_SRC / "qwen38.rs",
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "cfg(not(supersonic_backend_hip))",
                source,
                f"disabled non-HIP branch remains in {path}",
            )

    def test_core_and_hal_have_no_backend_selector_compatibility_surfaces(self):
        self.assertFalse(CORE_BACKEND.exists(), "legacy core backend parser remains")
        self.assertNotIn("pub mod backend", CORE_LIB.read_text(encoding="utf-8"))
        self.assertNotIn("compiled_backends", HAL_BACKEND.read_text(encoding="utf-8"))
        self.assertNotIn("set_backend", HAL_BACKEND.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
