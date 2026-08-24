#!/usr/bin/env python3
"""Merge AMD SMI ASIC/bus and enumeration captures into safe GPU provenance.

``amd-smi static --asic --bus`` describes a physical device and its stable
bus identity.  ``amd-smi list -e`` describes the logical ROCm/HIP ordinal.
The two captures are joined by a stable identity, never by the ``gpu`` array
ordinal (which is an enumeration index in AMD SMI output).  The result is a
small JSON document accepted by both the device selector and the benchmark
recording path, while retaining digests and counts for the original captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


_BDF_RE = re.compile(r"^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", re.IGNORECASE)
_UUID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{7,}$", re.IGNORECASE)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_INDEX_RE = re.compile(r"(?:gpu|hip|device|ordinal|id)\s*[\[(:#=]?\s*([0-9]+)", re.IGNORECASE)
_GFX_RE = re.compile(r"\bgfx[0-9]+\b", re.IGNORECASE)

_BDF_KEYS = (
    "pci_bdf",
    "pci_bus_id",
    "bus_id",
    "bdf",
    "bus_address",
    "location",
)
_UUID_KEYS = ("uuid", "gpu_uuid", "gpu_uuid_id", "unique_id", "hip_uuid")
_ARCH_KEYS = (
    "gfx_target_version",
    "target_graphics_version",
    "target_graphics_core",
    "gfx_arch",
    "architecture",
    "gpu_arch",
)
_MARKET_KEYS = ("market_name", "product_name", "device_name", "name")
_LOGICAL_KEYS = (
    "hip_id",
    "logical_gpu",
    "logical_device",
    "logical_index",
    "device_index",
    "rocm_device",
)
_RECORD_LIST_KEYS = ("gpu_data", "devices", "gpus", "cards", "items", "data")


def _casefold_mapping(node: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in node.items()}


def _direct_value(node: Mapping[str, Any], keys: Iterable[str]) -> Any:
    lowered = _casefold_mapping(node)
    for key in keys:
        if key.casefold() in lowered:
            return lowered[key.casefold()]
    return None


def _find_values(node: Any, keys: Iterable[str]) -> list[Any]:
    """Find scalar-bearing fields recursively, preserving source order."""

    wanted = {key.casefold() for key in keys}
    found: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            lowered = _casefold_mapping(value)
            for key in wanted:
                if key in lowered:
                    found.append(lowered[key])
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return found


def _as_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        if value >= 0:
            return value
        raise ValueError(f"{field} must be a non-negative integer")
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        match = _INDEX_RE.search(text)
        if match:
            return int(match.group(1))
    raise ValueError(f"{field} is not a numeric ordinal: {value!r}")


def _as_gfx(values: Iterable[Any]) -> str:
    for value in values:
        if isinstance(value, str):
            match = _GFX_RE.search(value)
            if match:
                return match.group(0).lower()
    raise ValueError("AMD SMI ASIC record is missing a gfx architecture")


def _as_market(values: Iterable[Any]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("AMD SMI ASIC record is missing a market name")


def _normal_identity(value: Any, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AMD SMI {kind} identity is empty")
    identity = value.strip().lower()
    if kind == "pci_bdf":
        if _BDF_RE.fullmatch(identity) is None:
            raise ValueError(f"invalid PCI BDF identity {value!r}")
    elif identity in {"unknown", "n/a", "na", "none", "null"} or _UUID_RE.fullmatch(identity) is None:
        raise ValueError(f"invalid UUID identity {value!r}")
    return identity


def _identities(node: Any) -> dict[str, str]:
    """Return consistent BDF/UUID identities carried by one source record."""

    result: dict[str, str] = {}
    bdfs = [_normal_identity(value, "pci_bdf") for value in _find_values(node, _BDF_KEYS)]
    uuids = [_normal_identity(value, "uuid") for value in _find_values(node, _UUID_KEYS)]
    if bdfs:
        if len(set(bdfs)) != 1:
            raise ValueError("AMD SMI record contains conflicting PCI BDF identities")
        result["pci_bdf"] = bdfs[0]
    if uuids:
        if len(set(uuids)) != 1:
            raise ValueError("AMD SMI record contains conflicting UUID identities")
        result["uuid"] = uuids[0]
    if not result:
        raise ValueError("AMD SMI record is missing a stable PCI BDF or UUID identity")
    return result


def _records(payload: Any, source_name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = None
        lowered = _casefold_mapping(payload)
        for key in _RECORD_LIST_KEYS:
            candidate = lowered.get(key.casefold())
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            # A single record is useful for diagnostic captures and remains
            # strict because it must carry a physical/enumeration marker.
            records = [dict(payload)] if _direct_value(payload, ("gpu", "GPU")) is not None else []
    else:
        records = []
    if not records or not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"AMD SMI {source_name} capture contains no device records")
    return [dict(record) for record in records]


def _physical_records(payload: Any) -> list[dict[str, Any]]:
    records = _records(payload, "ASIC/bus")
    result: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    seen_keys: set[tuple[str, str]] = set()
    for record in records:
        raw_index = _direct_value(record, ("gpu", "GPU"))
        if raw_index is None:
            raise ValueError("AMD SMI ASIC/bus record is missing its physical GPU ordinal")
        physical_index = _as_index(raw_index, field="physical GPU ordinal")
        if physical_index in seen_ordinals:
            raise ValueError(f"duplicate physical GPU ordinal {physical_index} in ASIC/bus capture")
        identities = _identities(record)
        for kind, identity in identities.items():
            key = (kind, identity)
            if key in seen_keys:
                raise ValueError(f"duplicate physical stable identity {identity}")
            seen_keys.add(key)
        result.append(
            {
                "physical_index": physical_index,
                "identities": identities,
                "gfx_arch": _as_gfx(_find_values(record, _ARCH_KEYS)),
                "market_name": _as_market(_find_values(record, _MARKET_KEYS)),
            }
        )
        seen_ordinals.add(physical_index)
    return result


def _enumeration_records(payload: Any) -> list[dict[str, Any]]:
    records = _records(payload, "enumeration")
    result: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_logical: set[int] = set()
    for record in records:
        identities = _identities(record)
        for kind, identity in identities.items():
            key = (kind, identity)
            if key in seen_keys:
                raise ValueError(f"duplicate enumeration stable identity {identity}")
            seen_keys.add(key)
        logical_values = _find_values(record, _LOGICAL_KEYS)
        if logical_values:
            logical_candidates = {
                _as_index(value, field="logical GPU ID") for value in logical_values
            }
            if len(logical_candidates) != 1:
                raise ValueError("AMD SMI enumeration record contains conflicting logical GPU IDs")
            logical_gpu = next(iter(logical_candidates))
        else:
            # AMD SMI's top-level gpu field is an enumeration ordinal.  It is
            # accepted only as a last-resort logical value after joining by a
            # separate stable identity; it is never used as the join key.
            raw_logical = _direct_value(record, ("gpu", "GPU"))
            if raw_logical is None:
                raise ValueError("AMD SMI enumeration record is missing HIP/logical GPU ID")
            logical_gpu = _as_index(raw_logical, field="logical GPU ID")
        if logical_gpu in seen_logical:
            raise ValueError(f"duplicate logical GPU mapping {logical_gpu} in enumeration capture")
        seen_logical.add(logical_gpu)
        result.append({"identities": identities, "logical_gpu": logical_gpu})
    return result


def _select_join_key(physical: list[dict[str, Any]], enumeration: list[dict[str, Any]]) -> str:
    for key in ("pci_bdf", "uuid"):
        physical_any = any(key in item["identities"] for item in physical)
        enumeration_any = any(key in item["identities"] for item in enumeration)
        if not physical_any and not enumeration_any:
            continue
        if physical_any or enumeration_any:
            if not (
                all(key in item["identities"] for item in physical)
                and all(key in item["identities"] for item in enumeration)
            ):
                raise ValueError(f"ASIC/bus and enumeration captures have incomplete {key} identities")
        physical_complete = all(key in item["identities"] for item in physical)
        enumeration_complete = all(key in item["identities"] for item in enumeration)
        if not (physical_complete and enumeration_complete):
            continue
        physical_ids = {item["identities"][key] for item in physical}
        enumeration_ids = {item["identities"][key] for item in enumeration}
        if physical_ids != enumeration_ids:
            raise ValueError(
                f"ASIC/bus and enumeration {key} identities mismatch: "
                f"physical={sorted(physical_ids)}, enumeration={sorted(enumeration_ids)}"
            )
        return key
    raise ValueError("ASIC/bus and enumeration captures have no complete common stable identity")


def merge_sources(
    asic_bus: Any,
    enumeration: Any,
    *,
    source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Join two decoded AMD SMI payloads and return selector-safe JSON."""

    physical = _physical_records(asic_bus)
    logical = _enumeration_records(enumeration)
    join_key = _select_join_key(physical, logical)
    physical_by_key = {item["identities"][join_key]: item for item in physical}
    logical_by_key = {item["identities"][join_key]: item for item in logical}
    missing_logical = sorted(set(physical_by_key) - set(logical_by_key))
    missing_physical = sorted(set(logical_by_key) - set(physical_by_key))
    if missing_logical or missing_physical:
        raise ValueError(
            "ASIC/bus and enumeration identities mismatch: "
            f"missing enumeration={missing_logical}, missing ASIC/bus={missing_physical}"
        )

    source_sha256 = dict(source_sha256 or {})
    asic_digest = source_sha256.get("asic_bus") or _payload_sha256(asic_bus)
    enumeration_digest = source_sha256.get("enumeration") or _payload_sha256(enumeration)
    for source_name, digest in (("asic_bus", asic_digest), ("enumeration", enumeration_digest)):
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError(f"{source_name} source digest is not a SHA-256 value")
    devices = []
    for identity in sorted(physical_by_key, key=lambda value: physical_by_key[value]["physical_index"]):
        item = physical_by_key[identity]
        enum = logical_by_key[identity]
        asic: dict[str, str] = {
            "market_name": item["market_name"],
            "target_graphics_version": item["gfx_arch"],
        }
        for kind, value in item["identities"].items():
            asic[kind] = value
        devices.append(
            {
                "gpu": item["physical_index"],
                "asic": asic,
                "logical_gpu": enum["logical_gpu"],
            }
        )
    return {
        "gpu_data": devices,
        "provenance": {
            "schema_version": 1,
            "join_key": join_key,
            "sources": {
                "asic_bus": {
                    "sha256": asic_digest,
                    "record_count": len(physical),
                },
                "enumeration": {
                    "sha256": enumeration_digest,
                    "record_count": len(logical),
                },
            },
        },
    }


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError(f"AMD SMI capture is empty: {path.name}")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
        return payload, hashlib.sha256(raw).hexdigest()
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"AMD SMI capture is invalid JSON: {path.name}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"AMD SMI capture contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise ValueError(f"AMD SMI capture contains non-finite JSON value {value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asic-bus", required=True, type=Path)
    parser.add_argument("--enumeration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        asic_bus, asic_digest = _read_json(args.asic_bus)
        enumeration, enumeration_digest = _read_json(args.enumeration)
        merged = merge_sources(
            asic_bus,
            enumeration,
            source_sha256={"asic_bus": asic_digest, "enumeration": enumeration_digest},
        )
        encoded = json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"AMD SMI provenance merge failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
