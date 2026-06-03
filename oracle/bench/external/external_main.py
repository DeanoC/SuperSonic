"""CLI: python -m oracle.bench.external.external_main --engine llama.cpp --models qwen3.5-35b-a3b --quants q4km"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .hipfire import HipfireAdapter
from .llama_cpp import LlamaCppAdapter
from .mlx_lm import MlxLmAdapter
from .common import ExternalWorkload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="hipfire", choices=["hipfire", "llama.cpp", "mlx-lm"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--quants", default="all")
    ap.add_argument("--prompt", default="The quick brown fox jumps over")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--context-size", type=int)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--measurement-runs", type=int, default=5)
    ap.add_argument("--prompt-tokens", type=int)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[], help="KEY=PATH")
    ap.add_argument("--binary", help="External engine binary override (hipfire or llama.cpp)")
    ap.add_argument("--python", default="python3", help="Python executable for mlx-lm")
    ap.add_argument("--skip-version-check", action="store_true")
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    adapter = _adapter(args)
    if not args.skip_version_check:
        adapter.assert_version_match()

    run_dir = _latest_run_dir(args.run_root)
    out_dir = run_dir / "external" / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)

    models = _default_models(args.engine) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    quants = _default_quants(args.engine) if args.quants == "all" else [q.strip() for q in args.quants.split(",")]
    workload = ExternalWorkload(
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        context_size=args.context_size,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
    )

    for model in models:
        for quant in quants:
            mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
            if hasattr(adapter, "measure_workload"):
                cell = adapter.measure_workload(model, quant, mdir, workload)
            else:
                cell = adapter.measure_speed(model, quant, args.prompt, args.max_new_tokens, mdir)
            (out_dir / f"{model}_{quant}.json").write_text(json.dumps(cell, indent=2))

    print(f"[bench-external] wrote {out_dir}")


def _adapter(args):
    if args.engine == "hipfire":
        return HipfireAdapter(binary=args.binary or "hipfire")
    if args.engine == "llama.cpp":
        return LlamaCppAdapter(binary=args.binary or "llama-bench")
    if args.engine == "mlx-lm":
        return MlxLmAdapter(python=args.python)
    raise AssertionError(args.engine)


def _default_models(engine: str) -> list[str]:
    if engine in {"llama.cpp", "mlx-lm"}:
        return ["qwen3.5-35b-a3b"]
    return ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]


def _default_quants(engine: str) -> list[str]:
    if engine in {"llama.cpp", "mlx-lm"}:
        return ["q4km"]
    return ["bf16", "int4"]


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
