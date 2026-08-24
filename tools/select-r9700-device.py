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
    # These fields come from the same authoritative static record as the
    # ordinal/architecture.  Defaults preserve the small helper's historical
    # constructor shape; selection rejects devices that do not carry them.
    stable_identity: str = ""
    identity_kind: str = ""
    logical_gpu: str = ""


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
_IDENTITY_KEYS = (
    ("pci_bdf", "pci_bdf"),
    ("pci_bus_id", "pci_bdf"),
    ("bus_id", "pci_bdf"),
    ("bdf", "pci_bdf"),
    ("uuid", "uuid"),
    ("gpu_uuid", "uuid"),
    ("gpu_uuid_id", "uuid"),
    ("unique_id", "uuid"),
)
_LOGICAL_KEYS = (
    "logical_gpu",
    "logical_device",
    "logical_index",
    "device_index",
    "rocm_device",
)
_GFX_RE = re.compile(r"\bgfx[0-9]+\b", re.IGNORECASE)
_INDEX_RE = re.compile(r"(?:gpu\s*[\[(:#]?\s*)?([0-9]+)", re.IGNORECASE)
_R9700_RE = re.compile(r"\br9700\b", re.IGNORECASE)
_BDF_RE = re.compile(r"^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{7,}$", re.IGNORECASE)


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


def _direct_identity(node: dict[str, Any]) -> tuple[str, str] | None:
    """Return a stable physical identity explicitly present in ``node``.

    Ordinals, market names, and AMD ``device_id`` values identify a product
    or an enumeration slot, not a physical device.  Only a PCI BDF or UUID
    from the authoritative static record is accepted here.
    """

    lowered = {str(key).lower(): value for key, value in node.items()}
    for key, kind in _IDENTITY_KEYS:
        value = lowered.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        identity = value.strip().lower()
        if kind == "pci_bdf":
            if not _BDF_RE.fullmatch(identity):
                raise ValueError(f"invalid PCI BDF identity {value!r}")
        elif identity in {"unknown", "n/a", "na", "none", "null"}:
            raise ValueError("static GPU record contains an unknown UUID identity")
        elif _UUID_RE.fullmatch(identity) is None:
            raise ValueError(f"invalid UUID identity {value!r}")
        return identity, kind
    return None


def _direct_logical(node: dict[str, Any]) -> str | None:
    lowered = {str(key).lower(): value for key, value in node.items()}
    for key in _LOGICAL_KEYS:
        if key not in lowered:
            continue
        value = _as_index(lowered[key])
        if value is None:
            raise ValueError(f"static GPU record has invalid logical GPU mapping {lowered[key]!r}")
        return str(value)
    return None


def _is_named_r9700(device: Device) -> bool:
    return bool(device.market_name.strip()) and bool(_R9700_RE.search(device.market_name))


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
            identity = _direct_identity(node)
            logical = _direct_logical(node)
            if current_index is not None and (gfx or market or identity or logical is not None):
                record = records.setdefault(current_index, {"physical_index": current_index})
                if gfx:
                    record["gfx_arch"] = gfx
                if market:
                    record["market_name"] = market
                if identity:
                    identity_value, identity_kind = identity
                    prior_identity = record.get("stable_identity")
                    if prior_identity is not None and str(prior_identity) != identity_value:
                        raise ValueError(
                            f"physical GPU {current_index} has conflicting stable identities"
                        )
                    record["stable_identity"] = identity_value
                    record["identity_kind"] = identity_kind
                if logical is not None:
                    prior_logical = record.get("logical_gpu")
                    if prior_logical is not None and str(prior_logical) != logical:
                        raise ValueError(
                            f"physical GPU {current_index} has conflicting logical mappings"
                        )
                    record["logical_gpu"] = logical
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
            stable_identity=str(record.get("stable_identity", "")),
            identity_kind=str(record.get("identity_kind", "")),
            logical_gpu=str(record.get("logical_gpu", "")),
        )
        for record in records.values()
        if record.get("gfx_arch")
    ]
    devices.sort(key=lambda device: device.physical_index)
    if not devices:
        raise ValueError("amd-smi JSON contained no indexed devices with gfx architecture")
    _validate_device_provenance(devices)
    return devices


def _validate_device_provenance(devices: list[Device]) -> None:
    """Require a one-to-one physical/logical map with stable identities."""

    physical_indexes = [device.physical_index for device in devices]
    if len(set(physical_indexes)) != len(physical_indexes):
        raise ValueError(
            "device discovery contains duplicate physical GPU ordinals: "
            f"{physical_indexes}"
        )
    identities = [device.stable_identity.strip().lower() for device in devices]
    if any(not identity for identity in identities):
        raise ValueError("static GPU record is missing a stable PCI BDF or UUID identity")
    if any(device.identity_kind not in {"pci_bdf", "uuid"} for device in devices):
        raise ValueError("static GPU record is missing a stable PCI BDF or UUID identity")
    if len(set(identities)) != len(identities):
        raise ValueError("device discovery contains duplicate stable GPU identities")
    logical = [device.logical_gpu.strip() for device in devices]
    if any(not value or not value.isdigit() for value in logical):
        raise ValueError("static GPU record is missing a logical GPU mapping")
    if len(set(logical)) != len(logical):
        raise ValueError("device discovery contains duplicate logical GPU mappings")


def select_physical_device(devices: list[Device], physical_gpu: str) -> Device:
    """Select the requested physical ordinal after validating static evidence.

    Unlike :func:`select_device`, this helper is architecture-neutral for the
    benchmark runner, which supports both published AMD targets.  The caller
    still validates that the selected architecture matches its explicit
    configuration.
    """

    _validate_device_provenance(devices)
    value = str(physical_gpu).strip()
    if not value.isdigit():
        raise ValueError(f"physical GPU must be a numeric ordinal, got {physical_gpu!r}")
    matches = [device for device in devices if device.physical_index == int(value)]
    if len(matches) != 1:
        raise ValueError(f"physical GPU {value} was not uniquely discovered in static evidence")
    return matches[0]


def select_device(devices: list[Device], override: str | None = None) -> Device:
    """Select one validated device; never infer GPU zero as a default."""

    _validate_device_provenance(devices)

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
        if not _is_named_r9700(selected):
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
    if not _is_named_r9700(selected):
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
        "SUPERSONIC_GPU_IDENTITY": selected.stable_identity,
        "SUPERSONIC_GPU_IDENTITY_KIND": selected.identity_kind,
        "SUPERSONIC_GPU_LOGICAL": selected.logical_gpu,
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
