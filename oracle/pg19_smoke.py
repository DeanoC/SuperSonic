#!/usr/bin/env python3
"""PG-19 teacher-forced smoke harness for Llama 3.1 CUDA.

This is intentionally small and systems-oriented. It drives SuperSonic's Rust
`--teacher-forced` scorer for dense INT8 and certified-KV INT8, aggregates
perplexity, and optionally compares against DotCache arxiv_v1 PG-19 reference
JSON for matching context/config cells.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pg19Reference:
    metric: str
    value: float
    dense_value: float | None
    path: Path


def parse_contexts(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower().endswith("k"):
            out.append(int(part[:-1]) * 1024)
        else:
            out.append(int(part))
    if not out:
        raise ValueError("no contexts specified")
    return out


def context_label(ctx: int) -> str:
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)


def parse_teacher_forced_json(output: str) -> dict[str, Any]:
    match = re.search(r"^\[teacher_forced_json\] (.+)$", output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("SuperSonic output missing [teacher_forced_json]")
    return json.loads(match.group(1))


def load_reference(
    reference_dir: Path,
    config: str,
    context: int,
    smoke: bool,
) -> Pg19Reference | None:
    label = context_label(context)
    suffix = ".smoke.json" if smoke else ".json"
    candidates = sorted(reference_dir.glob(f"*pg19_{label}_{config}{suffix}"))
    candidates = [p for p in candidates if ".native" not in p.name]
    if not candidates:
        return None
    path = candidates[0]
    payload = json.loads(path.read_text())
    quality = payload.get("quality") or {}
    value = quality.get("value")
    if value is None:
        return None
    return Pg19Reference(
        metric=str(quality.get("metric") or "perplexity"),
        value=float(value),
        dense_value=(
            float(quality["dense_value"])
            if quality.get("dense_value") is not None
            else None
        ),
        path=path,
    )


def load_text_chunks(
    tokenizer: Any,
    context: int,
    num_chunks: int,
    stride: int | None,
    source_text: Path | None,
) -> list[str]:
    if stride is None:
        stride = context
    token_chunks: list[list[int]] = []
    if source_text is not None:
        text_iter = [source_text.read_text()]
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "datasets is required unless --source-text is provided"
            ) from exc
        text_iter = (row["text"] for row in load_dataset("emozilla/pg19", split="test", streaming=True))

    for text in text_iter:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, max(len(token_ids) - context + 1, 0), stride):
            token_chunks.append(token_ids[start : start + context])
            if len(token_chunks) >= num_chunks:
                return [
                    tokenizer.decode(chunk, skip_special_tokens=True)
                    for chunk in token_chunks
                ]
    if len(token_chunks) < num_chunks:
        raise RuntimeError(f"only found {len(token_chunks)} PG-19 chunks, requested {num_chunks}")
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]


def run_supersonic(
    binary: Path,
    model_dir: Path,
    prompt: str,
    context: int,
    config: str,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        str(binary),
        "--backend",
        "cuda",
        "--model",
        "llama3.1-8b",
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--prompt-no-special-tokens",
        "--context-size",
        str(context),
        "--max-new-tokens",
        "0",
        "--int8",
        "--teacher-forced",
    ]
    if config == "certified":
        cmd.append("--certified-kv")
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"SuperSonic failed for config={config} context={context} rc={proc.returncode}\n{combined[-4000:]}"
        )
    result = parse_teacher_forced_json(combined)
    result["stdout_tail"] = proc.stdout[-1000:]
    result["stderr_tail"] = proc.stderr[-1000:]
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_nll = sum(float(r["total_nll"]) for r in results)
    total_tokens = sum(int(r["scored_tokens"]) for r in results)
    total_ms = sum(float(r["total_ms"]) for r in results)
    ppl = math.exp(total_nll / total_tokens)
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "total_ms": total_ms,
        "ms_per_token": total_ms / total_tokens if total_tokens else 0.0,
        "chunks": len(results),
        "per_chunk": results,
    }


def evaluate_gates(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    by_key = {
        (row["config"], row["context_length"]): row
        for row in payload["summary"]
    }
    for row in payload["summary"]:
        ref = row.get("reference_perplexity")
        if args.fail_above_reference and ref is not None:
            limit = ref + args.reference_tolerance
            if row["perplexity"] > limit:
                failures.append(
                    f"{row['config']}:{row['context_length']} ppl {row['perplexity']:.6f} > ref {ref:.6f} + {args.reference_tolerance}"
                )
    if args.max_certified_delta is not None:
        for row in payload["summary"]:
            if row["config"] != "certified":
                continue
            dense = by_key.get(("dense", row["context_length"]))
            if dense is None:
                continue
            delta = row["perplexity"] - dense["perplexity"]
            row["dense_perplexity"] = dense["perplexity"]
            row["delta_vs_dense"] = delta
            if delta > args.max_certified_delta:
                failures.append(
                    f"certified:{row['context_length']} delta {delta:.6f} > {args.max_certified_delta}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--contexts", default="512")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--config", choices=["dense", "certified", "both"], default="both")
    parser.add_argument("--source-text", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("target/pg19_smoke.json"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/workspace/DotCache/benchmarks/results/arxiv_v1_20260420"),
    )
    parser.add_argument("--reference-smoke", action="store_true")
    parser.add_argument("--fail-above-reference", action="store_true")
    parser.add_argument("--reference-tolerance", type=float, default=0.05)
    parser.add_argument("--max-certified-delta", type=float, default=0.10)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("transformers is required for PG-19 smoke chunking") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    contexts = parse_contexts(args.contexts)
    configs = ["dense", "certified"] if args.config == "both" else [args.config]
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for ctx in contexts:
        chunks = load_text_chunks(tokenizer, ctx, args.num_chunks, args.stride, args.source_text)
        for config in configs:
            per_chunk = []
            for idx, prompt in enumerate(chunks):
                print(f"[pg19] context={ctx} config={config} chunk={idx + 1}/{len(chunks)}", flush=True)
                result = run_supersonic(args.binary, args.model_dir, prompt, ctx, config, args.timeout)
                result.update({"chunk_idx": idx, "context_length": ctx, "config": config})
                per_chunk.append(result)
                rows.append(result)
            agg = aggregate(per_chunk)
            ref = load_reference(args.reference_dir, config, ctx, args.reference_smoke)
            summary.append({
                "context_length": ctx,
                "config": config,
                "perplexity": agg["perplexity"],
                "total_nll": agg["total_nll"],
                "total_tokens": agg["total_tokens"],
                "ms_per_token": agg["ms_per_token"],
                "chunks": agg["chunks"],
                "reference_perplexity": ref.value if ref else None,
                "reference_path": str(ref.path) if ref else None,
            })

    payload = {
        "benchmark": "pg19_teacher_forced_smoke",
        "model": "llama3.1-8b",
        "contexts": contexts,
        "configs": configs,
        "num_chunks": args.num_chunks,
        "summary": summary,
        "results": rows,
    }
    failures = evaluate_gates(payload, args)
    payload["gate_failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"summary": summary, "gate_failures": failures}, indent=2))
    print(f"Wrote {args.output}")
    if failures:
        print("QUALITY GATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
