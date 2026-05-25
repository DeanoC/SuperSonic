#!/usr/bin/env python3
"""Audit Qwen3.6-MoE MTP tensors in a local source snapshot and INT4 bake.

The runtime MTP loader consumes a compact baked view with 19 ``mtp.*`` tensors.
The Hugging Face source snapshot keeps routed experts split by expert id and by
gate/up/down projection, so this script checks both views explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


MODEL = "qwen3.6-35b-a3b"
SCHEMA = "qwen36-moe-mtp-audit-v1"
DEFAULT_NUM_EXPERTS = 256
DEFAULT_BAKE_GLOB = "v*-int4-gptq"

REQUIRED_BAKE_TENSORS = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
]

SOURCE_EXACT_TENSORS = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
]

SOURCE_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def resolve_model_dir(raw_model_dir: Path | None, env: dict[str, str]) -> Path:
    if raw_model_dir is not None:
        return raw_model_dir
    if env.get("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"):
        return Path(env["SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"])
    if env.get("SUPERSONIC_TEST_MODEL_ROOT"):
        return Path(env["SUPERSONIC_TEST_MODEL_ROOT"]) / MODEL
    return Path.home() / ".cache" / "supersonic-metal-models" / MODEL


def resolve_bake_dir(model_dir: Path, raw_bake_dir: Path | None) -> Path:
    if raw_bake_dir is not None:
        return raw_bake_dir
    supersonic_dir = model_dir / ".supersonic"
    candidates = sorted(
        path
        for path in supersonic_dir.glob(DEFAULT_BAKE_GLOB)
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if candidates:
        return candidates[-1]
    return supersonic_dir / "v2-int4-gptq"


def load_source_index(model_dir: Path) -> tuple[set[str] | None, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    meta: dict[str, Any] = {
        "path": str(index_path),
        "kind": "safetensors_index",
        "available": index_path.is_file(),
    }
    if not index_path.is_file():
        meta["status"] = "missing_index"
        meta["error"] = "model.safetensors.index.json not found"
        return None, meta

    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        meta["status"] = "malformed_index"
        meta["error"] = "index has no object weight_map"
        return None, meta

    names = set(weight_map)
    mtp_names = sorted(name for name in names if name.startswith("mtp."))
    meta["status"] = "loaded"
    meta["tensor_count"] = len(names)
    meta["mtp_tensor_count"] = len(mtp_names)
    meta["sample_mtp_tensors"] = mtp_names[:12]
    return names, meta


def load_bake_manifest(bake_dir: Path) -> tuple[set[str] | None, dict[str, Any]]:
    manifest_path = bake_dir / "manifest.json"
    meta: dict[str, Any] = {
        "path": str(manifest_path),
        "kind": "bake_manifest",
        "available": manifest_path.is_file(),
    }
    if not manifest_path.is_file():
        meta["status"] = "missing_manifest"
        meta["error"] = "manifest.json not found"
        return None, meta

    payload = json.loads(manifest_path.read_text())
    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        meta["status"] = "malformed_manifest"
        meta["error"] = "manifest has no tensor list"
        return None, meta

    tensor_meta: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        tensor_meta[name] = {
            key: item.get(key)
            for key in ("shape", "dtype", "layout", "byte_len")
            if key in item
        }

    names = set(tensor_meta)
    mtp_names = sorted(name for name in names if name.startswith("mtp."))
    meta.update(
        {
            "status": "loaded",
            "format_version": payload.get("format_version"),
            "converter_version": payload.get("converter_version"),
            "quant_profile": payload.get("quant_profile"),
            "source_quant": payload.get("source_quant"),
            "quant_method": payload.get("quant_method"),
            "tensor_count": len(names),
            "mtp_tensor_count": len(mtp_names),
            "sample_mtp_tensors": mtp_names[:12],
            "mtp_tensor_metadata": {
                name: tensor_meta[name]
                for name in REQUIRED_BAKE_TENSORS
                if name in tensor_meta
            },
        }
    )
    return names, meta


def audit_required(names: set[str] | None, required: list[str]) -> dict[str, Any]:
    if names is None:
        return {
            "status": "unavailable",
            "present_count": 0,
            "required_count": len(required),
            "present": [],
            "missing": required,
        }
    present = [name for name in required if name in names]
    missing = [name for name in required if name not in names]
    return {
        "status": "complete" if not missing else ("absent" if not present else "partial"),
        "present_count": len(present),
        "required_count": len(required),
        "present": present,
        "missing": missing,
    }


def audit_source_expert_projection(
    names: set[str] | None,
    projection: str,
    num_experts: int,
) -> dict[str, Any]:
    expected = [
        f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
        for expert in range(num_experts)
    ]
    result = audit_required(names, expected)
    result["projection"] = projection
    result["missing_experts"] = [
        idx
        for idx, name in enumerate(expected)
        if names is None or name not in names
    ]
    return result


def audit_source(names: set[str] | None, meta: dict[str, Any], num_experts: int) -> dict[str, Any]:
    exact = audit_required(names, SOURCE_EXACT_TENSORS)
    expert_projection_rows = [
        audit_source_expert_projection(names, projection, num_experts)
        for projection in SOURCE_EXPERT_PROJECTIONS
    ]
    all_complete = exact["status"] == "complete" and all(
        row["status"] == "complete" for row in expert_projection_rows
    )
    source = dict(meta)
    source.update(
        {
            "exact_tensors": exact,
            "expert_projection_tensors": expert_projection_rows,
            "status": "complete" if all_complete else ("unavailable" if names is None else "partial"),
        }
    )
    return source


def audit_bake(names: set[str] | None, meta: dict[str, Any]) -> dict[str, Any]:
    required = audit_required(names, REQUIRED_BAKE_TENSORS)
    bake = dict(meta)
    bake.update(
        {
            "required_tensors": required,
            "runtime_probe_present": bool(names is not None and "mtp.fc.weight" in names),
            "status": required["status"] if names is not None else meta.get("status", "unavailable"),
        }
    )
    return bake


def loader_delta(source: dict[str, Any], bake: dict[str, Any]) -> dict[str, Any]:
    source_status = source.get("status")
    bake_required = bake.get("required_tensors") or {}
    bake_status = bake_required.get("status")
    missing = bake_required.get("missing") or []
    notes: list[str] = []

    if bake_status == "complete":
        status = "ready"
        notes.append("INT4 bake contains all 19 runtime MTP tensors expected by the loader.")
    elif bake.get("runtime_probe_present"):
        status = "partial_bake"
        notes.append(
            "Bake contains mtp.fc.weight but is missing other loader tensors; runtime loader will fail closed."
        )
    elif source_status == "complete":
        status = "rebake_required"
        notes.append(
            "Source snapshot has the MTP block, but the current INT4 bake lacks the runtime MTP tensors."
        )
    elif source_status == "partial":
        status = "source_incomplete"
        notes.append("Source snapshot has only a partial MTP block; refresh or verify the model snapshot.")
    else:
        status = "unavailable"
        notes.append("Could not prove MTP availability from the local source snapshot or bake.")

    if missing:
        notes.append(f"Missing bake tensors: {', '.join(missing)}")
    if status == "ready":
        notes.append(
            "Metal speculative decode remains policy-blocked until an MTP parity/acceptance harness is wired."
        )

    return {
        "status": status,
        "runtime_probe": "mtp.fc.weight",
        "runtime_required_count": len(REQUIRED_BAKE_TENSORS),
        "metal_policy": "qwen3.6 Metal speculative decode is still unsupported in v1",
        "notes": notes,
    }


def build_report(model_dir: Path, bake_dir: Path, num_experts: int) -> dict[str, Any]:
    source_names, source_meta = load_source_index(model_dir)
    bake_names, bake_meta = load_bake_manifest(bake_dir)
    source = audit_source(source_names, source_meta, num_experts)
    bake = audit_bake(bake_names, bake_meta)
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "model_dir": str(model_dir),
        "bake_dir": str(bake_dir),
        "num_experts": num_experts,
        "required_bake_tensors": REQUIRED_BAKE_TENSORS,
        "source_required_exact_tensors": SOURCE_EXACT_TENSORS,
        "source": source,
        "bake": bake,
        "loader_delta": loader_delta(source, bake),
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report.get("source") or {}
    bake = report.get("bake") or {}
    bake_required = bake.get("required_tensors") or {}
    delta = report.get("loader_delta") or {}
    lines = [
        "# Qwen3.6 MTP Tensor Audit",
        "",
        f"- model: `{report.get('model')}`",
        f"- model_dir: `{report.get('model_dir')}`",
        f"- bake_dir: `{report.get('bake_dir')}`",
        f"- loader status: `{delta.get('status')}`",
        "",
        "| View | Status | Tensors | MTP tensors | Required present | Required missing |",
        "|---|---:|---:|---:|---:|---:|",
        "| Source safetensors | `{status}` | {tensors} | {mtp} | {present} | {missing} |".format(
            status=source.get("status"),
            tensors=source.get("tensor_count", ""),
            mtp=source.get("mtp_tensor_count", ""),
            present=(source.get("exact_tensors") or {}).get("present_count", ""),
            missing=len((source.get("exact_tensors") or {}).get("missing") or []),
        ),
        "| INT4 bake | `{status}` | {tensors} | {mtp} | {present} | {missing} |".format(
            status=bake.get("status"),
            tensors=bake.get("tensor_count", ""),
            mtp=bake.get("mtp_tensor_count", ""),
            present=bake_required.get("present_count", ""),
            missing=len(bake_required.get("missing") or []),
        ),
        "",
        "## Source Expert Families",
        "",
        "| Projection | Status | Present | Missing experts |",
        "|---|---:|---:|---:|",
    ]
    for row in source.get("expert_projection_tensors") or []:
        missing_experts = row.get("missing_experts") or []
        lines.append(
            "| `{projection}` | `{status}` | {present}/{required} | {missing} |".format(
                projection=row.get("projection"),
                status=row.get("status"),
                present=row.get("present_count"),
                required=row.get("required_count"),
                missing=len(missing_experts),
            )
        )

    missing_bake = bake_required.get("missing") or []
    if missing_bake:
        lines.extend(["", "## Missing Bake Tensors", ""])
        lines.extend(f"- `{name}`" for name in missing_bake)

    lines.extend(["", "## Loader Delta", ""])
    lines.extend(f"- {note}" for note in delta.get("notes") or [])
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--bake-dir", type=Path)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_mtp_audit.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_mtp_audit.md"))
    parser.add_argument(
        "--require-complete-bake",
        action="store_true",
        help="exit nonzero unless the INT4 bake has every runtime MTP tensor",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model_dir = resolve_model_dir(args.model_dir, os.environ)
    bake_dir = resolve_bake_dir(model_dir, args.bake_dir)
    report = build_report(model_dir, bake_dir, args.num_experts)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    print(f"[qwen36-mtp-audit] status={report['loader_delta']['status']} source={report['source']['status']} bake={report['bake']['status']}")
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    if args.require_complete_bake and report["loader_delta"]["status"] != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
