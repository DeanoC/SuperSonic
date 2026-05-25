#!/usr/bin/env python3
"""Plan or run refresh commands for Qwen3.6 Metal SOTA gate reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import summarize_qwen36_sota_gates as gate_summary
except ModuleNotFoundError:
    script_path = Path(__file__).with_name("summarize_qwen36_sota_gates.py")
    spec = importlib.util.spec_from_file_location(
        "summarize_qwen36_sota_gates",
        script_path,
    )
    gate_summary = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = gate_summary
    spec.loader.exec_module(gate_summary)


SCHEMA = "qwen36-sota-gate-refresh-plan-v1"
DEFAULT_OUT_JSON = Path("target/qwen36_sota_gate_refresh_plan.json")
DEFAULT_OUT_MD = Path("target/qwen36_sota_gate_refresh_plan.md")


def known_gate_ids() -> list[str]:
    return [spec.gate_id for spec in gate_summary.GATE_SPECS]


def normalize_only(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for value in values:
        selected.update(part.strip() for part in value.split(",") if part.strip())
    unknown = selected - set(known_gate_ids())
    if unknown:
        raise ValueError(
            "unknown gate id(s): {}; expected one of {}".format(
                ",".join(sorted(unknown)),
                ",".join(known_gate_ids()),
            )
        )
    return selected


def should_select_row(row: dict[str, Any], only: set[str] | None, refresh_all: bool) -> bool:
    if only is not None:
        return row["gate_id"] in only
    return refresh_all or row.get("status") != "ok"


def selection_reason(row: dict[str, Any], only: set[str] | None, refresh_all: bool) -> str:
    if only is not None:
        return "requested" if row["gate_id"] in only else "not_requested"
    if refresh_all:
        return "all"
    status = str(row.get("status", "unknown"))
    if status == "ok":
        return "already_ok"
    return status


def build_plan_rows(
    summary_rows: list[dict[str, Any]],
    only: set[str] | None = None,
    refresh_all: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        selected = should_select_row(row, only, refresh_all)
        rows.append(
            {
                "gate_id": row["gate_id"],
                "label": row["label"],
                "path": row["path"],
                "input_status": row["status"],
                "input_error": row.get("error"),
                "selected": selected,
                "selection_reason": selection_reason(row, only, refresh_all),
                "refresh_command": row["refresh_command"],
                "run_status": "planned" if selected else "skipped",
                "returncode": None,
                "duration_seconds": None,
                "started_at_utc": None,
                "finished_at_utc": None,
            }
        )
    return rows


def run_selected_rows(rows: list[dict[str, Any]], shell: str) -> None:
    for row in rows:
        if not row["selected"]:
            continue
        started = datetime.now(timezone.utc)
        row["run_status"] = "running"
        row["started_at_utc"] = started.isoformat().replace("+00:00", "Z")
        t0 = time.perf_counter()
        completed = subprocess.run(
            row["refresh_command"],
            shell=True,
            executable=shell,
            check=False,
        )
        row["duration_seconds"] = time.perf_counter() - t0
        row["returncode"] = completed.returncode
        row["finished_at_utc"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        row["run_status"] = "passed" if completed.returncode == 0 else "failed"


def count_rows(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_refresh_summary(
    rows: list[dict[str, Any]],
    dry_run: bool,
    pre_summary: dict[str, Any],
    post_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["selected"]]
    failed_rows = [row for row in selected_rows if row.get("run_status") == "failed"]
    return {
        "dry_run": dry_run,
        "gate_count": len(rows),
        "selected_count": len(selected_rows),
        "skipped_count": len(rows) - len(selected_rows),
        "ran_count": len(
            [row for row in selected_rows if row.get("run_status") in {"passed", "failed"}]
        ),
        "failed_count": len(failed_rows),
        "selected_gate_ids": [str(row["gate_id"]) for row in selected_rows],
        "failed_gate_ids": [str(row["gate_id"]) for row in failed_rows],
        "input_status_counts": count_rows(rows, "input_status"),
        "run_status_counts": count_rows(rows, "run_status"),
        "pre_next_action": pre_summary["next_action"],
        "post_next_action": (post_summary or {}).get("next_action"),
        "post_inputs_ok": None if post_summary is None else post_summary["all_inputs_ok"],
    }


def build_refresh_report(
    pre_report: dict[str, Any],
    rows: list[dict[str, Any]],
    dry_run: bool,
    post_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    post_summary = post_report["summary"] if post_report is not None else None
    return {
        "schema": SCHEMA,
        "model": pre_report["model"],
        "backend": pre_report["backend"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "summary": build_refresh_summary(
            rows,
            dry_run=dry_run,
            pre_summary=pre_report["summary"],
            post_summary=post_summary,
        ),
        "rows": rows,
        "pre_gate_summary": pre_report["summary"],
        "post_gate_summary": post_summary,
    }


def fmt_bool(value: Any) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def fmt_duration(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "-"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Qwen3.6 Metal SOTA Gate Refresh Plan",
        "",
        f"- dry_run: `{fmt_bool(summary['dry_run'])}`",
        f"- selected_gates: `{summary['selected_count']}`",
        f"- ran_gates: `{summary['ran_count']}`",
        f"- failed_gates: `{summary['failed_count']}`",
        f"- pre_next_action: `{summary['pre_next_action']['action']}`",
        f"- post_inputs_ok: `{fmt_bool(summary['post_inputs_ok'])}`",
        "",
        "| Gate | Input Status | Selected | Run Status | Return | Duration | Command |",
        "|:---|:---|:---:|:---|---:|---:|:---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {label} | {status} | {selected} | {run_status} | {returncode} | {duration} | `{cmd}` |".format(
                label=row["label"],
                status=row["input_status"],
                selected=fmt_bool(row["selected"]),
                run_status=row["run_status"],
                returncode="-" if row["returncode"] is None else row["returncode"],
                duration=fmt_duration(row["duration_seconds"]),
                cmd=row["refresh_command"],
            )
        )
    lines.extend(
        [
            "",
            "Dry-run is the default. Add `--run` to execute selected refresh commands in order.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def add_gate_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefill-json",
        type=Path,
        default=gate_summary.GATE_SPECS[0].default_path,
        help="batched-prefill variant sweep JSON",
    )
    parser.add_argument(
        "--static-runtime-json",
        type=Path,
        default=gate_summary.GATE_SPECS[1].default_path,
        help="static top-N runtime sweep JSON",
    )
    parser.add_argument(
        "--fused-json",
        type=Path,
        default=gate_summary.GATE_SPECS[2].default_path,
        help="fused routed INT4 runtime sweep JSON",
    )
    parser.add_argument(
        "--mps-json",
        type=Path,
        default=gate_summary.GATE_SPECS[3].default_path,
        help="MPS resident table probe JSON",
    )
    parser.add_argument(
        "--route-json",
        type=Path,
        default=gate_summary.GATE_SPECS[4].default_path,
        help="route residency sweep JSON",
    )
    parser.add_argument(
        "--mtp-json",
        type=Path,
        default=gate_summary.GATE_SPECS[5].default_path,
        help="MTP acceptance sweep JSON",
    )
    parser.add_argument(
        "--lru-json",
        type=Path,
        default=gate_summary.GATE_SPECS[6].default_path,
        help="LRU resident cache runtime sweep JSON",
    )
    parser.add_argument(
        "--linear-json",
        type=Path,
        default=gate_summary.GATE_SPECS[7].default_path,
        help="linear decode variant sweep JSON",
    )
    parser.add_argument(
        "--full-json",
        type=Path,
        default=gate_summary.GATE_SPECS[8].default_path,
        help="full-attention decode variant sweep JSON",
    )
    parser.add_argument(
        "--lm-head-json",
        type=Path,
        default=gate_summary.GATE_SPECS[9].default_path,
        help="lm-head tail variant sweep JSON",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_path_args(parser)
    parser.add_argument(
        "--only",
        action="append",
        help="refresh only the named gate id; may be repeated or comma-separated",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="refresh every configured gate, including reports that are already OK",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute selected refresh commands; default is to write a dry-run plan",
    )
    parser.add_argument(
        "--shell",
        default="/bin/zsh",
        help="shell used to execute generated refresh commands when --run is set",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        help="select loaded reports as stale when their file mtime is older than this many hours",
    )
    parser.add_argument(
        "--require-post-ok",
        action="store_true",
        help="with --run, exit non-zero if refreshed gate inputs are still not all OK",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        only = normalize_only(args.only)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    max_age_seconds = (
        args.max_age_hours * 3600.0 if args.max_age_hours is not None else None
    )
    paths = gate_summary.paths_from_args(args)
    pre_report = gate_summary.build_report(paths, max_age_seconds=max_age_seconds)
    rows = build_plan_rows(
        pre_report["rows"],
        only=only,
        refresh_all=args.all,
    )

    post_report = None
    if args.run:
        run_selected_rows(rows, shell=args.shell)
        post_report = gate_summary.build_report(paths, max_age_seconds=max_age_seconds)

    report = build_refresh_report(
        pre_report,
        rows,
        dry_run=not args.run,
        post_report=post_report,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    refresh_summary = report["summary"]
    print(
        "[qwen36-sota-gate-refresh] dry_run={} selected={} ran={} failed={}".format(
            str(refresh_summary["dry_run"]).lower(),
            refresh_summary["selected_count"],
            refresh_summary["ran_count"],
            refresh_summary["failed_count"],
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    if refresh_summary["failed_count"]:
        return 1
    if args.require_post_ok and args.run and not refresh_summary["post_inputs_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
