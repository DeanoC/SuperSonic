"""Pure-function JSON → markdown renderer for bench run-dirs."""
import json
import re
from pathlib import Path
from typing import Iterable

from .schema import validate_perf_cell

QUANT_COL_ORDER = [
    "bf16",
    "int4",
    "int4-spec025",
    "int4-spec050",
    "int4-spec075",
    "fp8r",
    "kv-fp8",
    "int8",
]
QUANT_LABELS = {
    "bf16": "BF16",
    "int4": "INT4",
    "int4-spec025": "Spec025",
    "int4-spec050": "Spec050",
    "int4-spec075": "Spec075",
    "fp8r": "FP8r",
    "kv-fp8": "KV-FP8",
    "int8": "INT8",
}


def render_perf_table(perf_dir: Path) -> str:
    cells = []
    for f in sorted(perf_dir.glob("*.json")):
        d = json.loads(f.read_text())
        validate_perf_cell(d)
        cells.append(d)
    if not cells:
        return ""

    by_model: dict[str, dict[str, dict]] = {}
    for c in cells:
        by_model.setdefault(c["model"], {})[c["quant"]] = c

    quants_present = [q for q in QUANT_COL_ORDER if any(q in m for m in by_model.values())]
    headers = ["Model"] + [QUANT_LABELS[q] for q in quants_present]
    sep_cells = ["-" * 15] + ["-" * 4 + ":" for _ in quants_present]

    rows = ["| " + " | ".join(h.ljust(15) if i == 0 else h.rjust(5) for i, h in enumerate(headers)) + " |",
            "| " + " | ".join(sep_cells) + " |"]
    for model in sorted(by_model):
        cells_for_model = by_model[model]
        cols = []
        for q in quants_present:
            cell = cells_for_model.get(q)
            if cell is None:
                cols.append("—")
            elif cell["status"] == "ok":
                cols.append(f"{cell['ms_per_step']:.1f}")
            elif cell["status"] == "skipped":
                cols.append("—")
            elif cell["status"] == "error":
                cols.append("ERR")
        cols_str = [c.rjust(5) for c in cols]
        rows.append(f"| {model.ljust(15)} | " + " | ".join(cols_str) + " |")
    return "\n".join(rows) + "\n"


def render_external_comparison_table(perf_dir: Path, external_dir: Path, engine: str) -> str:
    perf_cells = {}
    for f in perf_dir.glob("*.json"):
        d = json.loads(f.read_text())
        perf_cells[(d["model"], d["quant"])] = d
    ext_cells = {}
    for f in external_dir.glob("*.json"):
        d = json.loads(f.read_text())
        ext_cells[(d["model"], d["quant"])] = d
    keys = sorted(set(perf_cells) & set(ext_cells))
    if not keys:
        return ""

    header = f"| Model           | Quant | SuperSonic ms/step | {engine} ms/step | Δ vs {engine} |"
    sep    =  "|-----------------|-------|-------------------:|----------------:|-------------:|"
    rows = [header, sep]
    for (model, quant) in keys:
        p, e = perf_cells[(model, quant)], ext_cells[(model, quant)]
        if p["status"] != "ok" or e["status"] != "ok":
            continue
        ss_ms, ext_ms = p["ms_per_step"], e["ms_per_step"]
        delta = (ss_ms - ext_ms) / ext_ms * 100 if ext_ms > 0 else 0
        rows.append(f"| {model.ljust(15)} | {quant.ljust(5)} | {ss_ms:>18.1f} | {ext_ms:>15.1f} | {delta:+11.1f}% |")
    return "\n".join(rows) + "\n"


def render_quality_table(quality_dir: Path) -> str:
    cells = []
    for f in sorted(quality_dir.glob("*.json")):
        cells.append(json.loads(f.read_text()))
    if not cells:
        return ""

    by_model_quant: dict[tuple[str, str], dict[str, dict]] = {}
    for c in cells:
        by_model_quant.setdefault((c["model"], c["quant"]), {})[c["eval"]] = c

    evals = sorted({c["eval"] for c in cells})
    headers = ["Model", "Quant"] + evals
    rows = ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|"]
    for (model, quant) in sorted(by_model_quant):
        row_evals = by_model_quant[(model, quant)]
        cols = [model, quant]
        for ev in evals:
            cell = row_evals.get(ev)
            if cell is None:
                cols.append("—")
            else:
                cols.append(f"{cell['value']:.3f}")
        rows.append("| " + " | ".join(cols) + " |")
    return "\n".join(rows) + "\n"


_AUTOGEN_BEGIN = "<!-- AUTOGEN BELOW: {key} -->"
_AUTOGEN_END = "<!-- AUTOGEN END: {key} -->"


def replace_autogen_zone(doc: str, key: str, new_content: str) -> str:
    begin = _AUTOGEN_BEGIN.format(key=key)
    end = _AUTOGEN_END.format(key=key)
    if begin in doc and end in doc:
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        replacement = f"{begin}\n{new_content}\n{end}"
        return pattern.sub(replacement, doc, count=1)
    # Append a new zone at the end.
    if not doc.endswith("\n"):
        doc += "\n"
    return f"{doc}\n{begin}\n{new_content}\n{end}\n"
