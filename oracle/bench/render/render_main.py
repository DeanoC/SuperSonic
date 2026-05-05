"""CLI: python -m oracle.bench.render.render_main"""
import argparse
from pathlib import Path

from .markdown import render_perf_table, render_quality_table, render_external_comparison_table, replace_autogen_zone


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

        quality_md = render_quality_table(args.run / "quality")
        quality_doc_path = args.out / "docs" / "quality.md"
        quality_doc = quality_doc_path.read_text() if quality_doc_path.exists() else "# Model Quality\n\nMeasured (model, quant) quality for shipping models.\n"
        updated = replace_autogen_zone(quality_doc, "bench-quality-table", quality_md)
        quality_doc_path.write_text(updated)
        print(f"updated {quality_doc_path}")

        ext_dir = args.run / "external" / "hipfire"
        if ext_dir.exists() and perf_doc.exists():
            hipfire_md = render_external_comparison_table(args.run / "perf", ext_dir, "hipfire")
            updated = replace_autogen_zone(perf_doc.read_text(), "hipfire-comparison", hipfire_md)
            perf_doc.write_text(updated)
            print(f"updated hipfire-comparison zone in {perf_doc}")


if __name__ == "__main__":
    main()
