"""Trusted GPU provenance derived from one captured AMD SMI static record."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Mapping


_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = _ROOT / "tools" / "select-r9700-device.py"


@dataclass(frozen=True, slots=True)
class StaticGpuProvenance:
    """Portable identity and selected static evidence for one physical GPU."""

    identity: str
    identity_kind: str
    source_sha256: str
    physical_gpu: str
    architecture: str
    logical_gpu: str
    selected_fields: Mapping[str, str]


def resolve_static_gpu(
    static_json: Path,
    *,
    physical_gpu: str,
    gpu_arch: str,
    logical_gpu: str,
) -> StaticGpuProvenance:
    """Read and validate static AMD SMI evidence for the configured device.

    The caller supplies ordinals/architecture only as expectations.  The
    physical identity, architecture, and logical mapping are all selected
    from the captured record and are never inferred from a CLI assertion.
    """

    path = Path(static_json)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"gpu_static_json unavailable: {path}") from exc
    if not raw.strip():
        raise ValueError("gpu_static_json is empty")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        output = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("gpu_static_json must be UTF-8 JSON") from exc

    selector = _load_selector()
    try:
        devices = selector.parse_devices(output)
        selected = selector.select_physical_device(devices, str(physical_gpu))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gpu_static_json does not prove selected GPU: {exc}") from exc

    expected_physical = str(physical_gpu).strip()
    expected_arch = str(gpu_arch).strip().lower()
    expected_logical = str(logical_gpu).strip()
    if str(selected.physical_index) != expected_physical:
        raise ValueError(
            "gpu_static_json physical ordinal mismatch: "
            f"selected {selected.physical_index}, configured {expected_physical}"
        )
    if selected.gfx_arch != expected_arch:
        raise ValueError(
            "gpu_static_json architecture mismatch: "
            f"selected {selected.gfx_arch}, configured {expected_arch}"
        )
    if not expected_logical.isdigit() or selected.logical_gpu != expected_logical:
        raise ValueError(
            "gpu_static_json logical mapping mismatch: "
            f"selected {selected.logical_gpu!r}, configured {expected_logical!r}"
        )
    identity = str(selected.stable_identity).strip().lower()
    identity_kind = str(selected.identity_kind).strip().lower()
    if not identity or identity_kind not in {"pci_bdf", "uuid"}:
        raise ValueError("gpu_static_json lacks a stable PCI BDF or UUID identity")

    selected_fields = {
        "gpu": str(selected.physical_index),
        "gfx_arch": selected.gfx_arch,
        "logical_gpu": selected.logical_gpu,
        "identity": identity,
        "identity_kind": identity_kind,
    }
    return StaticGpuProvenance(
        identity=identity,
        identity_kind=identity_kind,
        source_sha256=source_sha256,
        physical_gpu=expected_physical,
        architecture=expected_arch,
        logical_gpu=expected_logical,
        selected_fields=selected_fields,
    )


def _load_selector():
    module_name = "_supersonic_static_gpu_selector"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, _SELECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("static GPU selector is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["StaticGpuProvenance", "resolve_static_gpu"]
