import re
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
HAL_VMM = ROOT / "crates" / "gpu-hal" / "src" / "vmm.rs"
KERNEL_FFI_SRC = ROOT / "crates" / "kernel-ffi" / "src"


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

    def test_gpu_hal_exposes_only_hip_backend_surfaces(self):
        backend = HAL_BACKEND.read_text(encoding="utf-8")
        lib = HAL_LIB.read_text(encoding="utf-8")
        ops = HAL_OPS.read_text(encoding="utf-8")
        vmm = HAL_VMM.read_text(encoding="utf-8")
        self.assertIn("vec![Backend::Hip]", backend)
        self.assertIn("Backend::Hip\n}", backend)
        self.assertNotIn("supersonic_backend_cuda", backend)
        self.assertNotIn("supersonic_backend_metal", backend)
        self.assertNotIn("mod cuda_sys", lib)
        self.assertNotIn("mod metal_sys", lib)
        self.assertNotIn("use crate::cuda_sys", ops)
        self.assertNotIn("use crate::metal_sys", ops)
        self.assertNotIn("use crate::cuda_sys", vmm)


if __name__ == "__main__":
    unittest.main()
