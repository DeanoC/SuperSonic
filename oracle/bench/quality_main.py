"""CLI: python -m oracle.bench.quality_main --arch gfx1100 --models all --quants all"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .perplexity import PerplexityRequest, score_perplexity
from .golden import GoldenRequest, run_golden
from .render.schema import validate_quality_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1100")
    ap.add_argument("--models", default="all")
    ap.add_argument("--quants", default="all")
    ap.add_argument("--binary", default="./target/release/supersonic", type=Path)
    ap.add_argument("--run-root", default="./target/bench-runs", type=Path)
    ap.add_argument("--model-dir", action="append", default=[],
                    help="KEY=PATH; repeatable")
    ap.add_argument("--ppl-contexts", type=int, default=2048)
    ap.add_argument("--ppl-num-chunks", type=int, default=4)
    args = ap.parse_args()

    model_dirs = dict(s.split("=", 1) for s in args.model_dir)
    run_dir = _latest_run_dir(args.run_root)
    out_dir = run_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
              "gemma4-e2b", "gemma4-e4b", "phi4-mini"] if args.models == "all" \
             else [m.strip() for m in args.models.split(",")]
    quants = ["bf16", "int4", "fp8r", "kv-fp8"] if args.quants == "all" \
             else [q.strip() for q in args.quants.split(",")]

    for model in models:
        # BF16 must run first to populate golden reference.
        ordered_quants = sorted(quants, key=lambda q: 0 if q == "bf16" else 1)
        for quant in ordered_quants:
            mdir = Path(model_dirs.get(model, f"/mnt/data/models/{model}"))
            for ds in ("pg19", "wikitext2"):
                req = PerplexityRequest(args.binary, model, mdir, quant, ds,
                                        args.ppl_contexts, args.ppl_num_chunks)
                cell = score_perplexity(req)
                validate_quality_cell(cell)
                (out_dir / f"{model}_{quant}_perplexity_{ds}.json").write_text(
                    json.dumps(cell, indent=2))
            golden = run_golden(GoldenRequest(args.binary, model, mdir, quant))
            (out_dir / f"{model}_{quant}_golden.json").write_text(json.dumps(golden, indent=2))

    print(f"[bench-quality] wrote {out_dir}")


def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"no run-dirs at {root} — run bench-perf first to create one")
    candidates = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no run-dirs in {root}")
    return candidates[0]


if __name__ == "__main__":
    main()
