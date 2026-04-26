#!/usr/bin/env python3
"""Summarize PG-19 smoke JSON outputs.

The PG-19 smoke benchmark records both high-level metrics and optional stage
timings. This helper keeps dense/certified comparisons reproducible without
manually spelunking result JSONs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_KEYS = (
    "layer_compute",
    "full_attn",
    "full_attn_core",
    "cert_selector",
    "cert_gather",
    "cert_attend",
    "mlp",
    "rms_norm",
    "lm_head",
    "logits_d2h",
)


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    results = doc.get("results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    summary = doc.get("summary")
    if isinstance(summary, list):
        return [r for r in summary if isinstance(r, dict)]
    raise ValueError(f"{path}: expected `results` or `summary` list")


def metric(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if value is None and key == "config":
        value = row.get("mode", default)
    return value


def fmt_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def print_table(rows: list[tuple[Path, dict[str, Any]]]) -> None:
    print("file\tcontext\tconfig\tms/token\tppl\tppl/ref\ttokens")
    for path, row in rows:
        context = metric(row, "context_length", metric(row, "prompt_tokens", "-"))
        config = metric(row, "config", metric(row, "mode", "-"))
        ppl = metric(row, "perplexity")
        ref = metric(row, "reference_perplexity")
        ratio = "-"
        if isinstance(ppl, (int, float)) and isinstance(ref, (int, float)) and ref != 0:
            ratio = f"{ppl / ref:.4f}"
        tokens = metric(row, "total_tokens", metric(row, "scored_tokens", "-"))
        print(
            f"{path}\t{context}\t{config}\t"
            f"{fmt_float(metric(row, 'ms_per_token'))}\t"
            f"{fmt_float(ppl, 4)}\t{ratio}\t{tokens}"
        )


def print_dense_ratios(rows: list[tuple[Path, dict[str, Any]]]) -> None:
    dense_by_context: dict[Any, float] = {}
    for _, row in rows:
        mode = str(metric(row, "config", metric(row, "mode", ""))).lower()
        ms = metric(row, "ms_per_token")
        context = metric(row, "context_length", metric(row, "prompt_tokens", "-"))
        if isinstance(ms, (int, float)) and ("dense" in mode or mode == "fp16"):
            dense_by_context[context] = float(ms)

    printed = False
    for path, row in rows:
        context = metric(row, "context_length", metric(row, "prompt_tokens", "-"))
        dense_ms = dense_by_context.get(context)
        ms = metric(row, "ms_per_token")
        if dense_ms and isinstance(ms, (int, float)):
            if not printed:
                print("\ndense ratios")
                print("file\tcontext\tms/token\tdense_ms\tratio")
                printed = True
            print(f"{path}\t{context}\t{ms:.3f}\t{dense_ms:.3f}\t{ms / dense_ms:.3f}x")


def print_stage(rows: list[tuple[Path, dict[str, Any]]]) -> None:
    printed_header = False
    for path, row in rows:
        stage = row.get("stage")
        if not isinstance(stage, dict) or not stage:
            continue
        if not printed_header:
            print("\nstage timings (ms total)")
            print("file\t" + "\t".join(STAGE_KEYS))
            printed_header = True
        values = [fmt_float(stage.get(key), 1) for key in STAGE_KEYS]
        print(f"{path}\t" + "\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", nargs="+", type=Path, help="PG-19 smoke JSON result files")
    args = parser.parse_args()

    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in args.json:
        for row in load_results(path):
            rows.append((path, row))
    print_table(rows)
    print_dense_ratios(rows)
    print_stage(rows)


if __name__ == "__main__":
    main()
