"""Quality check for Qwen3.6 CUDA SpecPrefill against dense INT4 logits.

Writes one quality cell per keep ratio:
  quality/qwen3.6-35b-a3b_int4-spec050_specprefill_logits.json

The metric value is argmax match rate against dense INT4. Extras include
cosine mean/min, top-5 overlap, kept-token counts, and per-prompt rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .render.schema import validate_quality_cell
except ModuleNotFoundError:  # pragma: no cover - bare local env fallback
    def validate_quality_cell(cell: dict) -> None:
        required = {"schema_version", "model", "quant", "eval", "metric", "value"}
        missing = required - set(cell)
        if missing:
            raise ValueError(f"quality cell missing fields: {sorted(missing)}")


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    prompt: str


@dataclass(frozen=True)
class RunResult:
    logits: list[float]
    generated_ids: list[int]
    kept: str


THRESHOLDS = {
    "int4-spec050": {
        "argmax_match_rate_min": 0.70,
        "cosine_min": 0.75,
        "top5_overlap_min": 2,
        "lane_status": "balanced",
    },
    "int4-spec075": {
        "argmax_match_rate_min": 1.0,
        "cosine_min": 0.90,
        "top5_overlap_min": 3,
        "lane_status": "conservative",
    },
    "int4-spec025": {
        "argmax_match_rate_min": 1.0,
        "cosine_min": 0.80,
        "top5_overlap_min": 3,
        "lane_status": "exploratory",
    },
}

DEFAULT_KEEP_RATIOS = "0.75"


def cossim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def topk(v: list[float], k: int = 5) -> set[int]:
    return set(sorted(range(len(v)), key=v.__getitem__, reverse=True)[:k])


def quality_cell(
    binary: Path,
    model_dir: Path,
    draft_dir: Path,
    prompts: Iterable[PromptCase],
    keep_ratio: float,
    timeout: int,
    dense_cache: dict[str, RunResult] | None = None,
) -> dict:
    rows = []
    for case in prompts:
        if dense_cache is not None and case.case_id in dense_cache:
            dense = dense_cache[case.case_id]
        else:
            dense = run_supersonic(
                binary,
                model_dir,
                case.prompt,
                keep_ratio=None,
                draft_dir=None,
                timeout=timeout,
            )
            if dense_cache is not None:
                dense_cache[case.case_id] = dense
        sparse = run_supersonic(binary, model_dir, case.prompt, keep_ratio=keep_ratio, draft_dir=draft_dir, timeout=timeout)
        dense_argmax = argmax(dense.logits)
        sparse_argmax = argmax(sparse.logits)
        dense_top5 = topk(dense.logits)
        sparse_top5 = topk(sparse.logits)
        rows.append({
            "case_id": case.case_id,
            "kept": sparse.kept,
            "cosine": cossim(dense.logits, sparse.logits),
            "argmax_match": dense_argmax == sparse_argmax,
            "top5_overlap": len(dense_top5 & sparse_top5),
            "dense_argmax": dense_argmax,
            "sparse_argmax": sparse_argmax,
            "dense_generated_ids": dense.generated_ids,
            "sparse_generated_ids": sparse.generated_ids,
        })

    argmax_match_rate = sum(1 for r in rows if r["argmax_match"]) / max(1, len(rows))
    cosines = [float(r["cosine"]) for r in rows]
    top5 = [int(r["top5_overlap"]) for r in rows]
    quant = f"int4-spec{int(round(keep_ratio * 100)):03d}"
    thresholds = THRESHOLDS.get(quant, {})
    cell = {
        "schema_version": 1,
        "model": "qwen3.6-35b-a3b",
        "quant": quant,
        "eval": "specprefill_logits",
        "metric": "argmax_match_rate",
        "value": argmax_match_rate,
        "extras": {
            "dense_quant": "int4",
            "keep_ratio": keep_ratio,
            "cases": len(rows),
            "cosine_mean": sum(cosines) / max(1, len(cosines)),
            "cosine_min": min(cosines) if cosines else 0.0,
            "top5_overlap_mean": sum(top5) / max(1, len(top5)),
            "top5_overlap_min": min(top5) if top5 else 0,
            "thresholds": thresholds,
            "lane_status": thresholds.get("lane_status", "unknown"),
            "per_prompt": rows,
        },
    }
    failures = threshold_failures(cell)
    cell["extras"]["threshold_pass"] = not failures
    cell["extras"]["threshold_failures"] = failures
    validate_quality_cell(cell)
    return cell


def threshold_failures(cell: dict) -> list[str]:
    thresholds = cell.get("extras", {}).get("thresholds") or {}
    if not thresholds:
        return []
    failures = []
    value = float(cell["value"])
    cosine_min = float(cell["extras"]["cosine_min"])
    top5_min = int(cell["extras"]["top5_overlap_min"])
    if value < float(thresholds["argmax_match_rate_min"]):
        failures.append(
            f"argmax_match_rate {value:.3f} < {thresholds['argmax_match_rate_min']:.3f}"
        )
    if cosine_min < float(thresholds["cosine_min"]):
        failures.append(f"cosine_min {cosine_min:.6f} < {thresholds['cosine_min']:.6f}")
    if top5_min < int(thresholds["top5_overlap_min"]):
        failures.append(f"top5_overlap_min {top5_min} < {thresholds['top5_overlap_min']}")
    return failures


def run_supersonic(
    binary: Path,
    model_dir: Path,
    prompt: str,
    keep_ratio: float | None,
    draft_dir: Path | None,
    timeout: int,
) -> RunResult:
    cmd = [
        str(binary),
        "--backend", "cuda",
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", str(model_dir),
        "--prompt", prompt,
        "--max-new-tokens", "1",
        "--dump-last-logits",
    ]
    if keep_ratio is not None:
        if draft_dir is None:
            raise ValueError("draft_dir is required for SpecPrefill runs")
        cmd.extend([
            "--specprefill-draft-dir", str(draft_dir),
            "--specprefill-algorithm", "cosine",
            "--specprefill-keep-ratio", f"{keep_ratio:.2f}",
            "--specprefill-unload-draft",
        ])
    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = "cuda"
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"supersonic failed with code {proc.returncode}; output tail:\n{output[-4000:]}")
    logits_match = re.search(r"LAST_LOGITS:([^\n]+)", output)
    if not logits_match:
        raise RuntimeError(f"LAST_LOGITS not found; output tail:\n{output[-4000:]}")
    logits = [float(x) for x in logits_match.group(1).split(",") if x.strip()]
    ids_match = re.search(r"Generated ids:\s*\[([^\]]*)\]", output)
    generated_ids = (
        [int(x.strip()) for x in ids_match.group(1).split(",") if x.strip()]
        if ids_match else []
    )
    kept_match = re.search(r"\[specprefill\] kept ([0-9]+)/([0-9]+) tokens", output)
    kept = f"{kept_match.group(1)}/{kept_match.group(2)}" if kept_match else "-"
    return RunResult(logits=logits, generated_ids=generated_ids, kept=kept)


def default_prompts() -> list[PromptCase]:
    prompts = [
        PromptCase("repeat20", ("SuperSonic CUDA dense prefill profiling sentence. " * 2).strip()),
        PromptCase("repeat120", ("SuperSonic CUDA dense prefill profiling sentence. " * 12).strip()),
        PromptCase(
            "code_python",
            "def summarize_numbers(values):\n"
            "    total = sum(values)\n"
            "    count = len(values)\n"
            "    if count == 0:\n"
            "        return",
        ),
        PromptCase(
            "qa_science",
            "Question: Why does the sky appear blue during the day?\n"
            "Answer in one concise sentence:",
        ),
        PromptCase(
            "json_completion",
            '{"name": "Ada", "language": "Python", "scores": [10, 20,',
        ),
    ]
    fixture = Path("tests/fixtures/specprefill/specprefill_c0088_target88_actual88.txt")
    if fixture.exists():
        prompts.append(PromptCase("specfixture88", fixture.read_text()[:1800]))
    fixture_long = Path("tests/fixtures/specprefill/specprefill_c0349_target349_actual349.txt")
    if fixture_long.exists():
        prompts.append(PromptCase("specfixture349_slice", fixture_long.read_text()[:3600]))
    return prompts


def quick_prompts() -> list[PromptCase]:
    prompts = [
        PromptCase("repeat20", ("SuperSonic CUDA dense prefill profiling sentence. " * 2).strip()),
        PromptCase("repeat120", ("SuperSonic CUDA dense prefill profiling sentence. " * 12).strip()),
    ]
    fixture = Path("tests/fixtures/specprefill/specprefill_c0088_target88_actual88.txt")
    if fixture.exists():
        prompts.append(PromptCase("specfixture88", fixture.read_text()[:1800]))
    return prompts


def latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"no run-dirs at {root}; run bench-perf first")
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


def parse_keep_ratios(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("no keep ratios specified")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=Path("./target/release/supersonic"))
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--draft-dir", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, default=Path("./target/bench-runs"))
    ap.add_argument("--run", type=Path, help="specific bench run directory; defaults to latest under --run-root")
    ap.add_argument(
        "--keep-ratios",
        default=DEFAULT_KEEP_RATIOS,
        help=(
            f"comma-separated SpecPrefill keep ratios to evaluate; default {DEFAULT_KEEP_RATIOS} "
            "is the conservative sm86 lane, use 0.50,0.75 to include the balanced lane"
        ),
    )
    ap.add_argument("--prompt-set", choices=["quick", "standard"], default="standard")
    ap.add_argument("--enforce-thresholds", action="store_true")
    ap.add_argument(
        "--fail-exploratory",
        action="store_true",
        help="also fail enforced runs when exploratory lanes such as int4-spec025 miss thresholds",
    )
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    run_dir = args.run or latest_run_dir(args.run_root)
    quality_dir = run_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    prompts = quick_prompts() if args.prompt_set == "quick" else default_prompts()

    all_failures = []
    dense_cache: dict[str, RunResult] = {}
    for keep in parse_keep_ratios(args.keep_ratios):
        cell = quality_cell(
            args.binary,
            args.model_dir,
            args.draft_dir,
            prompts,
            keep,
            args.timeout,
            dense_cache=dense_cache,
        )
        out = quality_dir / f"qwen3.6-35b-a3b_{cell['quant']}_specprefill_logits.json"
        out.write_text(json.dumps(cell, indent=2))
        failures = cell["extras"]["threshold_failures"]
        lane_status = cell["extras"]["lane_status"]
        if args.enforce_thresholds and failures and (lane_status != "exploratory" or args.fail_exploratory):
            all_failures.extend(f"{cell['quant']}: {failure}" for failure in failures)
        print(
            f"[specprefill-quality] {cell['quant']} argmax={cell['value']:.3f} "
            f"cos_min={cell['extras']['cosine_min']:.6f} "
            f"top5_min={cell['extras']['top5_overlap_min']} "
            f"threshold={'pass' if cell['extras']['threshold_pass'] else 'FAIL'} "
            f"lane={lane_status} wrote {out}",
            flush=True,
        )
    if all_failures:
        raise SystemExit("SpecPrefill quality gate failed:\n  - " + "\n  - ".join(all_failures))


if __name__ == "__main__":
    main()
