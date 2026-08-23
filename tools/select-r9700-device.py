#!/usr/bin/env python3
"""Select the physical gfx1201/R9700 device from authoritative AMD SMI JSON.

The ROCm ordinal discovered here is physical.  The workflow writes it to
``HIP_VISIBLE_DEVICES`` so that the selected device is logical device zero for
all subsequent Rust tests.  There is intentionally no GPU-zero fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


@dataclass(frozen=True)
class Device:
    physical_index: int
    gfx_arch: str
    market_name: str


_INDEX_KEYS = ("gpu", "GPU")
_ARCH_KEYS = (
    "gfx_target_version",
    "target_graphics_version",
    "target_graphics_core",
    "gfx_arch",
    "architecture",
    "gpu_arch",
)
_MARKET_KEYS = ("market_name", "product_name", "device_name", "name")
_GFX_RE = re.compile(r"\bgfx[0-9]+\b", re.IGNORECASE)
_INDEX_RE = re.compile(r"(?:gpu\s*[\[(:#]?\s*)?([0-9]+)", re.IGNORECASE)


def _as_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        match = _INDEX_RE.search(value.strip())
        if match:
            return int(match.group(1))
    return None


def _as_gfx(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _GFX_RE.search(value)
    return match.group(0).lower() if match else None


def _direct_index(node: dict[str, Any]) -> int | None:
    for key in _INDEX_KEYS:
        if key in node:
            index = _as_index(node[key])
            if index is not None:
                return index
    return None


def _direct_gfx(node: dict[str, Any]) -> str | None:
    for key in _ARCH_KEYS:
        gfx = _as_gfx(node.get(key))
        if gfx:
            return gfx
    return None


def _direct_market(node: dict[str, Any]) -> str:
    for key in _MARKET_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_devices(output: str) -> list[Device]:
    """Parse ``amd-smi static --asic --json`` into indexed devices.

    The static ASIC schema carries the physical ordinal in ``gpu`` and the
    architecture under ``asic``.  Walking nested objects while carrying that
    record ordinal supports both current and older wrappers, but deliberately
    never treats a PCI/device/subsystem identifier as a physical ordinal.
    """

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        # A few AMD SMI builds prefix JSON with a one-line diagnostic.  Keep
        # the authoritative JSON payload strict while tolerating that wrapper.
        starts = [position for position in (output.find("{"), output.find("[")) if position >= 0]
        if not starts:
            raise ValueError(f"amd-smi JSON is invalid: {exc}") from exc
        start = min(starts)
        try:
            payload = json.loads(output[start:])
        except json.JSONDecodeError as nested_exc:
            raise ValueError(f"amd-smi JSON is invalid: {nested_exc}") from nested_exc

    records: dict[int, dict[str, str | int]] = {}
    seen_record_indexes: set[int] = set()

    def visit(
        node: Any,
        inherited_index: int | None = None,
        *,
        record_context: bool = False,
    ) -> None:
        if isinstance(node, dict):
            current_index = inherited_index
            if record_context:
                direct_index = _direct_index(node)
                if direct_index is not None:
                    if direct_index in seen_record_indexes:
                        raise ValueError(
                            f"amd-smi JSON contains duplicate physical GPU ordinal {direct_index}"
                        )
                    seen_record_indexes.add(direct_index)
                    current_index = direct_index
            gfx = _direct_gfx(node)
            market = _direct_market(node)
            if current_index is not None and (gfx or market):
                record = records.setdefault(current_index, {"physical_index": current_index})
                if gfx:
                    record["gfx_arch"] = gfx
                if market:
                    record["market_name"] = market
            for key, value in node.items():
                # Only the device-record containers may introduce a new
                # physical ordinal.  Nested ASIC/subsystem objects inherit
                # their record's ordinal and cannot reinterpret identifiers.
                child_record_context = key in {"gpu_data", "devices"}
                visit(
                    value,
                    current_index,
                    record_context=child_record_context,
                )
        elif isinstance(node, list):
            for value in node:
                visit(value, inherited_index, record_context=record_context)

    root_record_context = isinstance(payload, list) or (
        isinstance(payload, dict) and _direct_index(payload) is not None
    )
    visit(payload, record_context=root_record_context)
    devices = [
        Device(
            physical_index=int(record["physical_index"]),
            gfx_arch=str(record.get("gfx_arch", "")),
            market_name=str(record.get("market_name", "")),
        )
        for record in records.values()
        if record.get("gfx_arch")
    ]
    devices.sort(key=lambda device: device.physical_index)
    if not devices:
        raise ValueError("amd-smi JSON contained no indexed devices with gfx architecture")
    return devices


def select_device(devices: list[Device], override: str | None = None) -> Device:
    """Select one validated device; never infer GPU zero as a default."""

    physical_indexes = [device.physical_index for device in devices]
    if len(set(physical_indexes)) != len(physical_indexes):
        raise ValueError(
            "device discovery contains duplicate physical GPU ordinals: "
            f"{physical_indexes}"
        )

    if override is not None and override.strip():
        value = override.strip()
        if not value.isdigit():
            raise ValueError(f"R9700 override must be a physical numeric ordinal, got {value!r}")
        physical_index = int(value)
        matches = [device for device in devices if device.physical_index == physical_index]
        if len(matches) != 1:
            raise ValueError(f"R9700 override physical GPU {physical_index} was not discovered")
        selected = matches[0]
        if selected.gfx_arch != "gfx1201":
            raise ValueError(
                f"R9700 override physical GPU {physical_index} reports {selected.gfx_arch}, "
                "not gfx1201"
            )
        if "r9700" not in selected.market_name.lower():
            raise ValueError(
                f"R9700 override physical GPU {physical_index} is not a named R9700 device"
            )
        return selected

    gfx1201 = [device for device in devices if device.gfx_arch == "gfx1201"]
    if len(gfx1201) != 1:
        indexes = [device.physical_index for device in gfx1201]
        raise ValueError(
            "exactly one physical gfx1201/R9700 device is required; "
            f"discovered candidates={indexes}"
        )
    selected = gfx1201[0]
    if selected.market_name and "r9700" not in selected.market_name.lower():
        raise ValueError(
            f"selected physical GPU {selected.physical_index} is not a named R9700 device"
        )
    if selected.gfx_arch != "gfx1201":
        raise ValueError(f"selected physical GPU is {selected.gfx_arch}, not gfx1201")
    return selected


def render_environment(selected: Device) -> dict[str, str]:
    """Return the environment contract consumed by subsequent workflow steps."""

    return {
        "SUPERSONIC_R9700_GPU_ID": str(selected.physical_index),
        "SUPERSONIC_R9700_GPU_ARCH": selected.gfx_arch,
        "HIP_VISIBLE_DEVICES": str(selected.physical_index),
        "SUPERSONIC_DEVICE": "0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="amd-smi JSON file (default: stdin)")
    parser.add_argument(
        "--override",
        default="",
        help="optional physical ordinal; it is accepted only after named R9700/gfx1201 validation",
    )
    args = parser.parse_args(argv)
    output = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    try:
        selected = select_device(parse_devices(output), args.override)
    except (OSError, ValueError) as exc:
        print(f"R9700 device selection failed: {exc}", file=sys.stderr)
        return 1
    for name, value in render_environment(selected).items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
