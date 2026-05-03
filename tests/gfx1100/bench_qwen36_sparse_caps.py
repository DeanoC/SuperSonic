#!/usr/bin/env python3
"""Sweep Qwen3.6-MoE sparse VMM expert caps on HIP/gfx1100.

Runs the dense virtual-slab baseline plus sparse MoE island caps, captures
stage timings and VMM residency, and emits both JSON and Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple


MIB = 1024 * 1024
GIB = 1024 * MIB


class BenchCase(NamedTuple):
    label: str
    cap: int | None
    prefetch_mode: str | None
    prefetch_ranks: str | None


def parse_prefetch_rank_policy(raw: str) -> tuple[str, str | None]:
    value = raw.strip().lower()
    if value in {"none", "off", "disabled", "0"}:
        return ("none", None)
    if value == "all":
        return ("all", "all")
    ranks = int(value)
    if ranks <= 0:
        raise ValueError("prefetch ranks must be positive")
    return (f"r{ranks}", str(ranks))


def parse_prefetch_rank_policies(raw: str) -> list[tuple[str, str | None]]:
    policies = [parse_prefetch_rank_policy(part) for part in raw.split(",") if part.strip()]
    if not policies:
        raise ValueError("prefetch rank policy list is empty")
    seen: set[str] = set()
    deduped: list[tuple[str, str | None]] = []
    for label, ranks in policies:
        if label in seen:
            continue
        seen.add(label)
        deduped.append((label, ranks))
    return deduped


def parse_prefetch_mode_policy(raw: str) -> tuple[str, str | None]:
    value = raw.strip().lower().replace("_", "-")
    if value in {"none", "off", "disabled", "0"}:
        return ("none", None)
    if value in {"previous-token", "prev-token"}:
        return ("previous-token", "previous-token")
    if value in {
        "previous-token-resident",
        "prev-token-resident",
        "resident-previous-token",
    }:
        return ("previous-token-resident", "previous-token-resident")
    raise ValueError(f"unknown prefetch mode {raw!r}")


def parse_prefetch_mode_policies(raw: str) -> list[tuple[str, str | None]]:
    policies = [parse_prefetch_mode_policy(part) for part in raw.split(",") if part.strip()]
    if not policies:
        raise ValueError("prefetch mode policy list is empty")
    seen: set[str] = set()
    deduped: list[tuple[str, str | None]] = []
    for label, mode in policies:
        if label in seen:
            continue
        seen.add(label)
        deduped.append((label, mode))
    return deduped


def build_cases(
    caps: list[int],
    prefetch_mode_sweep: str | None,
    prefetch_rank_sweep: str | None,
    prefetch_mode: str | None,
    prefetch_ranks: str | None,
) -> list[BenchCase]:
    cases: list[BenchCase] = [BenchCase("dense", None, None, None)]
    if prefetch_mode_sweep:
        mode_policies = parse_prefetch_mode_policies(prefetch_mode_sweep)
        rank_policies = (
            parse_prefetch_rank_policies(prefetch_rank_sweep)
            if prefetch_rank_sweep
            else [parse_prefetch_rank_policy(prefetch_ranks or "all")]
        )
        for cap in caps:
            emitted_disabled = False
            for mode_label, mode in mode_policies:
                if mode is None:
                    if not emitted_disabled:
                        cases.append(BenchCase(f"cap{cap}-none", cap, None, None))
                        emitted_disabled = True
                    continue
                for rank_label, ranks in rank_policies:
                    if ranks is None:
                        if not emitted_disabled:
                            cases.append(BenchCase(f"cap{cap}-none", cap, None, None))
                            emitted_disabled = True
                        continue
                    cases.append(
                        BenchCase(f"cap{cap}-{mode_label}-{rank_label}", cap, mode, ranks)
                    )
    elif prefetch_rank_sweep:
        rank_policies = parse_prefetch_rank_policies(prefetch_rank_sweep)
        for cap in caps:
            for policy_label, ranks in rank_policies:
                mode = prefetch_mode if ranks is not None else None
                cases.append(BenchCase(f"cap{cap}-{policy_label}", cap, mode, ranks))
    else:
        cases.extend(BenchCase(f"cap{cap}", cap, prefetch_mode, prefetch_ranks) for cap in caps)
    return cases


def parse_stage_timings(output: str) -> dict[str, float]:
    match = re.search(r"\[qwen36-moe stage-timings\]\s+(.+)", output)
    if not match:
        return {}
    out: dict[str, float] = {}
    for part in match.group(1).split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def parse_generated_ids(output: str) -> list[int]:
    match = re.search(r"Generated ids:\s*\[([^\]]*)\]", output)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_mib_field(output: str, prefix: str, field: str) -> int | None:
    line = next((line for line in output.splitlines() if prefix in line), "")
    if not line:
        return None
    match = re.search(rf"(?:^|\s){re.escape(field)}=([0-9.]+)MiB(?:\s|$)", line)
    if not match:
        return None
    return int(round(float(match.group(1)) * MIB))


def load_sparse_telemetry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload.get("summary") or {}


def fmt_gib(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / GIB:.2f}"


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_rank_pct(route_summary: dict[str, Any], key: str, rank: int = 0) -> str:
    observations = route_summary.get("observations_by_rank") or []
    values = route_summary.get(key) or []
    if rank >= len(observations) or rank >= len(values) or not observations[rank]:
        return "-"
    return f"{100.0 * float(values[rank]) / float(observations[rank]):.1f}%"


def fmt_rank_transition_pct(
    route_summary: dict[str, Any],
    current_rank: int = 0,
    previous_rank: int = 0,
) -> str:
    observations = route_summary.get("observations_by_rank") or []
    matrix = route_summary.get("repeated_previous_rank_by_current_rank") or []
    if (
        current_rank >= len(observations)
        or current_rank >= len(matrix)
        or previous_rank >= len(matrix[current_rank])
        or not observations[current_rank]
    ):
        return "-"
    return f"{100.0 * float(matrix[current_rank][previous_rank]) / float(observations[current_rank]):.1f}%"


def fmt_probability_pct(values: list[Any], rank: int = 0) -> str:
    if rank >= len(values):
        return "-"
    return f"{100.0 * float(values[rank]):.1f}%"


def fmt_same_rank_repeat_pct(route_summary: dict[str, Any], rank: int = 0) -> str:
    values = route_summary.get("same_rank_repeat_probability_by_rank") or []
    if values:
        return fmt_probability_pct(values, rank)
    return fmt_rank_transition_pct(route_summary, current_rank=rank, previous_rank=rank)


def fmt_previous_rank_reused_pct(route_summary: dict[str, Any], previous_rank: int = 0) -> str:
    values = route_summary.get("repeated_current_probability_by_previous_rank") or []
    if values:
        return fmt_probability_pct(values, previous_rank)
    observations = route_summary.get("observations_by_rank") or []
    matrix = route_summary.get("repeated_previous_rank_by_current_rank") or []
    if previous_rank >= len(observations) or not observations[previous_rank]:
        return "-"
    count = sum(
        row[previous_rank]
        for row in matrix
        if previous_rank < len(row)
    )
    return f"{100.0 * float(count) / float(observations[previous_rank]):.1f}%"


def fmt_best_transition(route_summary: dict[str, Any]) -> str:
    transition = route_summary.get("best_transition") or {}
    current_rank = transition.get("current_rank")
    previous_rank = transition.get("previous_rank")
    probability = transition.get("probability_by_current_rank")
    if current_rank is None or previous_rank is None or probability is None:
        return "-"
    return f"{previous_rank}->{current_rank} ({100.0 * float(probability):.1f}%)"


def run_case(
    args: argparse.Namespace,
    case: BenchCase,
    tmp: Path,
    warmup: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = args.backend
    env["SUPERSONIC_VMM_MOE_ISLANDS"] = "1"
    env.pop("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS", None)
    env.pop("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON", None)
    env.pop("SUPERSONIC_MOE_ISLAND_PREFETCH", None)
    env.pop("SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS", None)
    telemetry_path = tmp / f"{case.label}_telemetry.json"
    if case.cap is not None:
        env["SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"] = str(case.cap)
        env["SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON"] = str(telemetry_path)
        if case.prefetch_mode:
            env["SUPERSONIC_MOE_ISLAND_PREFETCH"] = case.prefetch_mode
        if case.prefetch_ranks is not None:
            env["SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS"] = case.prefetch_ranks

    cmd = [
        str(args.binary),
        "--backend", args.backend,
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", str(args.model_dir),
        "--int4",
        "--prompt", args.prompt,
        "--context-size", str(args.context_size),
        "--max-new-tokens", str(args.warmup_new_tokens if warmup else args.max_new_tokens),
        "--temperature", "0",
        "--top-k", "1",
        "--sampling-seed", str(args.seed),
        "--no-download",
        "--emit-stage-timings",
    ]
    if args.no_persistent_decode:
        cmd.append("--no-persistent-decode")

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    output = proc.stdout + proc.stderr
    row: dict[str, Any] = {
        "label": case.label,
        "cap_experts": case.cap,
        "prefetch_mode_requested": case.prefetch_mode,
        "prefetch_ranks_requested": case.prefetch_ranks,
        "returncode": proc.returncode,
        "generated_ids": parse_generated_ids(output),
        "stage": parse_stage_timings(output),
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-2400:],
    }
    if proc.returncode != 0:
        row["error"] = output[-4000:]
        return row

    if case.cap is None:
        moe_resident = parse_mib_field(output, "routed expert slabs active", "resident")
        moe_reserved = parse_mib_field(output, "routed expert slabs active", "reserved")
        kv_resident = parse_mib_field(output, "KV active", "resident") or 0
        kv_reserved = parse_mib_field(output, "KV active", "reserved") or 0
        row["vmm_residency"] = {
            "moe_resident_bytes": moe_resident,
            "moe_reserved_bytes": moe_reserved,
            "kv_resident_bytes": kv_resident,
            "kv_reserved_bytes": kv_reserved,
            "total_vmm_resident_bytes": (
                moe_resident + kv_resident if moe_resident is not None else None
            ),
            "total_vmm_reserved_bytes": (
                moe_reserved + kv_reserved if moe_reserved is not None else None
            ),
        }
    elif telemetry_path.exists():
        row["vmm_residency"] = load_sparse_telemetry(telemetry_path)
    else:
        row["vmm_residency"] = {}

    total_ms = row["stage"].get("total_ms_avg")
    row["tok_per_s"] = 1000.0 / total_ms if total_ms else None
    return row


def markdown(rows: list[dict[str, Any]]) -> str:
    out = [
        "| Mode | total ms/tok | tok/s | total resident GiB | MoE resident GiB | KV resident GiB | prefetch | ranks | peak pages | page misses | prefetch page misses | prefetch skipped | rank0 resident | rank0 repeat | rank0 same-rank | prev-rank0 reused | best transition | evicted pages | ids match |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for row in rows:
        residency = row.get("vmm_residency") or {}
        route_summary = residency.get("route_summary") or {}
        stage = row.get("stage") or {}
        ids_match = row.get("generated_ids_match")
        out.append(
            "| {label} | {ms} | {tok} | {total_gib} | {moe_gib} | {kv_gib} | {prefetch_mode} | {prefetch_ranks} | {peak_pages} | {page_misses} | {prefetch_page_misses} | {prefetch_skipped} | {rank0_resident} | {rank0_repeat} | {rank0_same_rank} | {prev_rank0_reused} | {best_transition} | {evicted_pages} | {ids_match} |".format(
                label=row["label"],
                ms=fmt_num(stage.get("total_ms_avg")),
                tok=fmt_num(row.get("tok_per_s")),
                total_gib=fmt_gib(residency.get("total_vmm_resident_bytes")),
                moe_gib=fmt_gib(residency.get("moe_resident_bytes")),
                kv_gib=fmt_gib(residency.get("kv_resident_bytes")),
                prefetch_mode=residency.get("prefetch_mode", "-"),
                prefetch_ranks=residency.get("prefetch_ranks", "-"),
                peak_pages=residency.get("peak_resident_pages", "-"),
                page_misses=residency.get("page_misses", "-"),
                prefetch_page_misses=residency.get("prefetch_page_misses", "-"),
                prefetch_skipped=residency.get("prefetch_skipped", "-"),
                rank0_resident=fmt_rank_pct(route_summary, "resident_before_by_rank"),
                rank0_repeat=fmt_rank_pct(route_summary, "repeated_previous_by_rank"),
                rank0_same_rank=fmt_same_rank_repeat_pct(route_summary),
                prev_rank0_reused=fmt_previous_rank_reused_pct(route_summary),
                best_transition=fmt_best_transition(route_summary),
                evicted_pages=residency.get("evicted_pages", "-"),
                ids_match="yes" if ids_match else "NO",
            )
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/data/models/Qwen3.6-35B-A3B"))
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--caps", default="8,32,64,128,256,320")
    parser.add_argument("--prompt", default="The quick brown fox jumps over")
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-persistent-decode", action="store_true")
    parser.add_argument(
        "--prefetch",
        choices=["previous-token", "previous-token-resident"],
        help="set SUPERSONIC_MOE_ISLAND_PREFETCH for sparse cap rows",
    )
    parser.add_argument(
        "--prefetch-mode-sweep",
        help=(
            "comma-separated sparse prefetch modes to sweep per cap, e.g. "
            "disabled,previous-token,previous-token-resident"
        ),
    )
    parser.add_argument(
        "--prefetch-ranks",
        help="set SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS for sparse cap rows",
    )
    parser.add_argument(
        "--prefetch-rank-sweep",
        help=(
            "comma-separated sparse prefetch policies to sweep per cap, e.g. "
            "none,1,2,4,all; overrides --prefetch/--prefetch-ranks"
        ),
    )
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_sparse_cap_sweep.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_sparse_cap_sweep.md"))
    args = parser.parse_args()

    if args.prefetch and args.prefetch_mode_sweep:
        parser.error("--prefetch cannot be combined with --prefetch-mode-sweep")
    if args.prefetch_rank_sweep and args.prefetch_ranks is not None:
        parser.error("--prefetch-rank-sweep cannot be combined with --prefetch-ranks")
    if args.prefetch_ranks is not None and not (args.prefetch or args.prefetch_mode_sweep):
        parser.error("--prefetch-ranks requires --prefetch or --prefetch-mode-sweep")
    if args.prefetch_ranks is not None:
        try:
            _, ranks = parse_prefetch_rank_policy(args.prefetch_ranks)
        except ValueError as exc:
            parser.error(str(exc))
        if ranks is None:
            parser.error("--prefetch-ranks must be a positive integer or all")
        args.prefetch_ranks = ranks
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    caps = [int(part.strip()) for part in args.caps.split(",") if part.strip()]
    tmp_owner = tempfile.TemporaryDirectory(prefix="qwen36-sparse-cap-sweep-")
    tmp = Path(tmp_owner.name)

    try:
        prefetch_mode = args.prefetch
        if prefetch_mode is None and args.prefetch_rank_sweep and not args.prefetch_mode_sweep:
            prefetch_mode = "previous-token"
        cases = build_cases(
            caps,
            args.prefetch_mode_sweep,
            args.prefetch_rank_sweep,
            prefetch_mode,
            args.prefetch_ranks,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rows: list[dict[str, Any]] = []
    for case in cases:
        if not args.no_warmup:
            print(f"[warmup] {case.label}", flush=True)
            run_case(args, case, tmp, warmup=True)
        print(f"[bench] {case.label}", flush=True)
        row = run_case(args, case, tmp, warmup=False)
        if row.get("returncode") != 0:
            print(row.get("error") or row.get("stderr_tail") or "run failed", file=sys.stderr)
            return 1
        rows.append(row)
        stage = row.get("stage") or {}
        residency = row.get("vmm_residency") or {}
        print(
            f"  total_ms={stage.get('total_ms_avg')} tok/s={row.get('tok_per_s')} "
            f"resident_gib={fmt_gib(residency.get('total_vmm_resident_bytes'))} "
            f"ids={row.get('generated_ids')}",
            flush=True,
        )

    dense_ids = rows[0].get("generated_ids") or []
    for row in rows:
        row["generated_ids_match"] = (row.get("generated_ids") or []) == dense_ids

    payload = {
        "schema": "qwen36-moe-sparse-cap-sweep-v2",
        "model": "qwen3.6-35b-a3b",
        "model_dir": str(args.model_dir),
        "backend": args.backend,
        "prompt": args.prompt,
        "context_size": args.context_size,
        "max_new_tokens": args.max_new_tokens,
        "prefetch": args.prefetch,
        "prefetch_mode_sweep": args.prefetch_mode_sweep,
        "prefetch_rank_sweep": args.prefetch_rank_sweep,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown(rows))
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    print(markdown(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
