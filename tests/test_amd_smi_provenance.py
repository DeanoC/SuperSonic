from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


ASIC_BUS = json.loads(
    (ROOT / "tests" / "amd_smi_fixtures" / "asic-bus.json").read_text(encoding="utf-8")
)
ENUMERATION = json.loads(
    (ROOT / "tests" / "amd_smi_fixtures" / "enumeration.json").read_text(encoding="utf-8")
)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AmdSmiProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = load_module(
            ROOT / "tools" / "merge-amd-smi-provenance.py",
            "merge_amd_smi_provenance",
        )
        self.selector = load_module(
            ROOT / "tools" / "select-r9700-device.py",
            "select_r9700_for_provenance",
        )

    def test_merge_joins_physical_bus_identity_to_logical_enumeration(self):
        merged = self.tool.merge_sources(ASIC_BUS, ENUMERATION)

        devices = self.selector.parse_devices(json.dumps(merged))
        selected = self.selector.select_device(devices)
        self.assertEqual(selected.physical_index, 1)
        self.assertEqual(selected.stable_identity, "0000:85:00.0")
        self.assertEqual(selected.logical_gpu, "1")
        self.assertEqual(
            merged["gpu_data"][1]["asic"]["enumeration"]["hip_uuid"],
            "gpu-73f8101733408480",
        )
        self.assertEqual(merged["provenance"]["join_key"], "pci_bdf")

    def test_installed_capture_with_distinct_uuid_and_hip_uuid_selects_by_bdf(self):
        merged = self.tool.merge_sources(ASIC_BUS, ENUMERATION)
        devices = self.selector.parse_devices(json.dumps(merged))
        selected = self.selector.select_device(devices)
        self.assertEqual(selected.physical_index, 1)
        self.assertEqual(selected.stable_identity, "0000:85:00.0")
        self.assertEqual(selected.logical_gpu, "1")
        self.assertEqual(
            merged["gpu_data"][1]["asic"]["pci_bdf"],
            "0000:85:00.0",
        )

    def test_merge_rejects_bus_and_enumeration_identity_mismatch(self):
        enumeration = json.loads(json.dumps(ENUMERATION))
        enumeration[1]["bdf"] = "0000:66:00.0"
        with self.assertRaisesRegex(ValueError, "mismatch|matching|BDF"):
            self.tool.merge_sources(ASIC_BUS, enumeration)

    def test_merge_rejects_duplicate_physical_bus_identity(self):
        asic_bus = json.loads(json.dumps(ASIC_BUS))
        asic_bus["gpu_data"][1]["bus"]["bdf"] = "0000:04:00.0"
        with self.assertRaisesRegex(ValueError, "duplicate|ambiguous"):
            self.tool.merge_sources(asic_bus, ENUMERATION)

    def test_merge_rejects_conflicting_same_kind_standard_uuids(self):
        enumeration = json.loads(json.dumps(ENUMERATION))
        enumeration[1]["nested_duplicate"] = {
            "uuid": "73007551-0000-1000-80f8-conflicting"
        }
        with self.assertRaisesRegex(ValueError, "conflicting UUID"):
            self.tool.merge_sources(ASIC_BUS, enumeration)

    def test_merge_rejects_duplicate_logical_mapping(self):
        enumeration = json.loads(json.dumps(ENUMERATION))
        enumeration[1]["hip_id"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate logical"):
            self.tool.merge_sources(ASIC_BUS, enumeration)

    def test_merge_records_source_digests_in_canonical_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asic_path = root / "asic-bus.json"
            enumeration_path = root / "enumeration.json"
            output_path = root / "provenance.json"
            asic_path.write_text(json.dumps(ASIC_BUS), encoding="utf-8")
            enumeration_path.write_text(json.dumps(ENUMERATION), encoding="utf-8")

            self.assertEqual(
                self.tool.main(
                    [
                        "--asic-bus",
                        str(asic_path),
                        "--enumeration",
                        str(enumeration_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(result["provenance"]["sources"]["asic_bus"]["sha256"]),
                64,
            )
            self.assertEqual(
                len(result["provenance"]["sources"]["enumeration"]["sha256"]),
                64,
            )

    def test_merged_capture_is_accepted_by_benchmark_gpu_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "provenance.json"
            self.assertEqual(
                self.tool.main(
                    [
                        "--asic-bus",
                        str(ROOT / "tests" / "amd_smi_fixtures" / "asic-bus.json"),
                        "--enumeration",
                        str(ROOT / "tests" / "amd_smi_fixtures" / "enumeration.json"),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            gpu = load_module(ROOT / "tools" / "benchmark" / "gpu.py", "benchmark_gpu_for_provenance")
            provenance = gpu.resolve_static_gpu(
                output_path,
                physical_gpu="1",
                gpu_arch="gfx1201",
                logical_gpu="1",
            )
            self.assertEqual(provenance.identity, "0000:85:00.0")
            self.assertEqual(provenance.logical_gpu, "1")


if __name__ == "__main__":
    unittest.main()
