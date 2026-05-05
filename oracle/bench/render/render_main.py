"""CLI: python -m oracle.bench.render.render_main"""
import argparse
from pathlib import Path

from .markdown import render_perf_table, replace_autogen_zone


def main():
    ap = argparse.ArgumentParser(prog="render")
    sub = ap.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("markdown")
    render.add_argument("--run", required=True, type=Path)
    render.add_argument("--out", required=True, type=Path,
                        help="Repo root containing docs/quality.md and docs/performance.md")

    args = ap.parse_args()
    if args.cmd == "markdown":
        perf_md = render_perf_table(args.run / "perf")
        perf_doc = (args.out / "docs" / "performance.md")
        if perf_doc.exists():
            updated = replace_autogen_zone(perf_doc.read_text(), "bench-perf-matrix", perf_md)
            perf_doc.write_text(updated)
            print(f"updated {perf_doc}")


if __name__ == "__main__":
    main()
