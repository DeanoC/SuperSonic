"""CLI: python -m oracle.bench.external.external_main --engine hipfire --models all --quants all"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .hipfire import HipfireAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="hipfire", choices=["hipfire"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--quants", default="all")
    ap.add_argument("--prompt", default="The quick brown fox jumps over")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[], help="KEY=PATH")
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    adapter = HipfireAdapter()
    adapter.assert_version_match()

    run_dir = _latest_run_dir(args.run_root)
    out_dir = run_dir / "external" / args.engine
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"] \
             if args.models == "all" else [m.strip() for m in args.models.split(",")]
    quants = ["bf16", "int4"] if args.quants == "all" else [q.strip() for q in args.quants.split(",")]

    for model in models:
        for quant in quants:
            mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
            cell = adapter.measure_speed(model, quant, args.prompt, args.max_new_tokens, mdir)
            (out_dir / f"{model}_{quant}.json").write_text(json.dumps(cell, indent=2))

    print(f"[bench-external] wrote {out_dir}")


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
