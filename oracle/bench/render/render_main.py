"""CLI: python -m oracle.bench.render.render_main"""
import argparse
import json
from pathlib import Path

from .markdown import render_perf_table, render_quality_table, render_external_comparison_table, replace_autogen_zone


def main():
    ap = argparse.ArgumentParser(prog="render")
    sub = ap.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("markdown")
    render.add_argument("--run", required=True, type=Path)
    render.add_argument("--out", required=True, type=Path,
                        help="Repo root containing docs/quality.md and docs/performance.md")
    render.add_argument("--perf-zone-key", default="bench-perf-matrix",
                        help="AUTOGEN zone key to update in docs/performance.md")

    diff = sub.add_parser("diff")
    diff.add_argument("--run-a", required=True, type=Path)
    diff.add_argument("--run-b", required=True, type=Path)
    diff.add_argument("--threshold-pct", type=float, default=5.0)

    args = ap.parse_args()
    if args.cmd == "markdown":
        perf_md = render_perf_table(args.run / "perf")
        perf_zone_key = args.perf_zone_key
        meta_path = args.run / "meta.json"
        if perf_zone_key == "bench-perf-matrix" and meta_path.exists():
            arch = json.loads(meta_path.read_text()).get("arch")
            if arch == "apple-m5-max":
                perf_zone_key = "apple-m5-max-metal"
        perf_doc = (args.out / "docs" / "performance.md")
        if perf_doc.exists():
            updated = replace_autogen_zone(perf_doc.read_text(), perf_zone_key, perf_md)
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

    if args.cmd == "diff":
        from .diff import diff_runs, render_diff_table
        rows = diff_runs(args.run_a, args.run_b, threshold_pct=args.threshold_pct)
        print(render_diff_table(rows))


if __name__ == "__main__":
    main()
