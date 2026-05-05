"""Two-run regression diff: surface cells where the metric moved beyond threshold."""
import json
from pathlib import Path


def diff_runs(run_a: Path, run_b: Path, threshold_pct: float = 5.0) -> list[dict]:
    """Compare perf cells between two run-dirs. Returns rows where ms_per_step moved by > threshold_pct."""
    a_cells = _load_perf(run_a)
    b_cells = _load_perf(run_b)
    rows = []
    for key in sorted(set(a_cells) | set(b_cells)):
        a, b = a_cells.get(key), b_cells.get(key)
        if a is None or b is None:
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        before, after = a["ms_per_step"], b["ms_per_step"]
        if before == 0:
            continue
        delta_pct = (after - before) / before * 100
        if abs(delta_pct) >= threshold_pct:
            rows.append({"model": key[0], "quant": key[1],
                         "before": before, "after": after, "delta_pct": delta_pct})
    return rows


def _load_perf(run: Path) -> dict[tuple[str, str], dict]:
    out = {}
    perf_dir = run / "perf"
    if not perf_dir.exists():
        return out
    for f in perf_dir.glob("*.json"):
        d = json.loads(f.read_text())
        out[(d["model"], d["quant"])] = d
    return out


def render_diff_table(rows: list[dict]) -> str:
    if not rows:
        return "(no cells exceeded threshold)\n"
    header = "| Model           | Quant | Before ms | After ms | Δ%      |"
    sep    = "|-----------------|-------|----------:|---------:|--------:|"
    body = [
        f"| {r['model'].ljust(15)} | {r['quant'].ljust(5)} | {r['before']:>9.2f} | {r['after']:>8.2f} | {r['delta_pct']:+7.2f} |"
        for r in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"
