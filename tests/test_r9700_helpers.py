#!/usr/bin/env python3
"""Behavioral tests for the R9700 workflow's host-side helper scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]

AMD_SMI_STATIC_ASIC_FIXTURE = {
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


def load_helper(filename: str, module_name: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class R9700SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = load_helper("select-r9700-device.py", "select_r9700_device")
        self.devices = self.selector.parse_devices(json.dumps(AMD_SMI_STATIC_ASIC_FIXTURE))

    def test_captured_amd_smi_static_schema_uses_gpu_not_device_id(self):
        self.assertEqual(
            [(device.physical_index, device.gfx_arch) for device in self.devices],
            [(0, "gfx1100"), (1, "gfx1201")],
        )
        self.assertEqual(self.selector.select_device(self.devices).physical_index, 1)

    def test_list_schema_without_architecture_is_not_treated_as_static_schema(self):
        with self.assertRaises(ValueError):
            self.selector.parse_devices(
                json.dumps({"gpu_data": [{"gpu": 1, "device_id": "0x7551"}]})
            )

    def test_nested_identifiers_cannot_override_record_gpu_ordinal(self):
        payload = {
            "gpu_data": [
                {
                    "gpu": 1,
                    "asic": {
                        "market_name": "AMD Radeon AI PRO R9700",
                        "device_id": "0x7551",
                        "target_graphics_version": "gfx1201",
                        "subsystem": {
                            "gpu": 99,
                            "target_graphics_version": "gfx1201",
                        },
                    },
                }
            ]
        }
        devices = self.selector.parse_devices(json.dumps(payload))
        self.assertEqual(
            [(device.physical_index, device.gfx_arch) for device in devices],
            [(1, "gfx1201")],
        )

    def test_top_level_gpu_key_is_the_only_legacy_wrapper_ordinal(self):
        payload = {
            "GPU": "GPU 1",
            "asic": {
                "market_name": "AMD Radeon AI PRO R9700",
                "target_graphics_version": "gfx1201",
            },
        }
        devices = self.selector.parse_devices(json.dumps(payload))
        self.assertEqual(
            [(device.physical_index, device.gfx_arch) for device in devices],
            [(1, "gfx1201")],
        )

    def test_discovery_selects_valid_physical_device_and_maps_it_to_logical_zero(self):
        selected = self.selector.select_device(self.devices)

        self.assertEqual(selected.physical_index, 1)
        self.assertEqual(selected.gfx_arch, "gfx1201")
        environment = self.selector.render_environment(selected)
        self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(environment["SUPERSONIC_DEVICE"], "0")

    def test_explicit_override_is_allowed_only_after_gfx_validation(self):
        selected = self.selector.select_device(self.devices, override="1")
        self.assertEqual(selected.physical_index, 1)

        with self.assertRaises(ValueError):
            self.selector.select_device(self.devices, override="0")

    def test_ambiguous_or_missing_gfx1201_devices_fail_without_override(self):
        with self.assertRaises(ValueError):
            self.selector.select_device(
                self.devices
                + [
                    self.selector.Device(
                        physical_index=2,
                        gfx_arch="gfx1201",
                        market_name="AMD Radeon AI PRO R9700",
                    )
                ]
            )

        with self.assertRaises(ValueError):
            self.selector.select_device(
                [
                    self.selector.Device(
                        physical_index=0,
                        gfx_arch="gfx1100",
                        market_name="other",
                    )
                ]
            )

        with self.assertRaises(ValueError):
            self.selector.select_device(
                [
                    self.selector.Device(
                        physical_index=1,
                        gfx_arch="gfx1201",
                        market_name="AMD Radeon RX 8900",
                    )
                ]
            )


class RocmSmiParserTests(unittest.TestCase):
    def test_parser_returns_selected_device_utilization(self):
        parser = load_helper("parse-rocm-smi.py", "parse_rocm_smi")
        utilization = parser.parse_utilization(
            """
===================== ROCm System Management Interface =====================
GPU use (%): 7
GPU Memory Allocated (VRAM%): 4
"""
        )
        self.assertEqual(utilization.gpu_use_percent, 7.0)
        self.assertEqual(utilization.vram_use_percent, 4.0)

    def test_parser_rejects_incomplete_or_ambiguous_probe_output(self):
        parser = load_helper("parse-rocm-smi.py", "parse_rocm_smi")
        with self.assertRaises(ValueError):
            parser.parse_utilization("GPU use (%): 7")


if __name__ == "__main__":
    unittest.main()
