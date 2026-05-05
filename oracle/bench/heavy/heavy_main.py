"""CLI: python -m oracle.bench.heavy.heavy_main --combos qwen3.5-9b:int4,gemma4-e4b:int4"""
from __future__ import annotations
import argparse
from pathlib import Path

from .longctx import run_longctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", required=True,
                    help="Comma-separated model:quant entries, e.g. qwen3.5-9b:int4,gemma4-e4b:int4")
    ap.add_argument("--contexts", default="4096,8192,16384,32768",
                    help="Comma-separated context lengths for longctx")
    ap.add_argument("--binary", default="./target/release/supersonic", type=Path)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[], help="KEY=PATH")
    ap.add_argument("--evals", default="longctx",
                    help="Comma-separated: longctx, niah, ruler, lm_eval. Phase 1 ships longctx; others stubbed.")
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    contexts = [int(c) for c in args.contexts.split(",")]
    combos = [tuple(c.split(":", 1)) for c in args.combos.split(",")]
    evals = [e.strip() for e in args.evals.split(",")]

    run_dir = _latest_run_dir(args.run_root)

    for (model, quant) in combos:
        mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
        if "longctx" in evals:
            print(f"[heavy] longctx {model}/{quant}")
            run_longctx(args.binary, model, mdir, quant, contexts, run_dir)
        if "niah" in evals:
            from .niah import run_niah
            print(f"[heavy] niah {model}/{quant}")
            run_niah(args.binary, model, mdir, quant,
                     contexts=[c for c in contexts if c <= 16384], run_dir=run_dir)
        if "ruler" in evals:
            from .ruler import run_ruler
            print(f"[heavy] ruler {model}/{quant}")
            run_ruler(args.binary, model, mdir, quant,
                      contexts=[c for c in contexts if c <= 32768], run_dir=run_dir)
        if "lm_eval" in evals:
            from .lm_eval import run_lm_eval, DEFAULT_TASKS
            print(f"[heavy] lm_eval {model}/{quant}")
            run_lm_eval(model, mdir, quant, DEFAULT_TASKS, run_dir)


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
