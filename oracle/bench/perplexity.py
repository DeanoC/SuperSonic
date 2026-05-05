"""Perplexity sweep over PG-19 and WikiText-2 for any (model, quant).

Generalizes oracle/pg19_smoke.py to drive `./target/release/supersonic
--teacher-forced` on any model family. Requires Tasks 12-14 to have
extended --teacher-forced beyond the Llama lane.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_TF_JSON = re.compile(r"^\[teacher_forced_json\] (.+)$", re.MULTILINE)


class PerplexityChunkError(RuntimeError):
    """Raised when a teacher-forced chunk invocation fails to produce a
    parseable [teacher_forced_json] line. Surfacing this as a hard error
    prevents partial sweeps from quietly publishing biased numbers."""


def aggregate_ppl_from_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("no chunks to aggregate")
    total_nll = sum(c["nll"] for c in chunks)
    total_tokens = sum(c["tokens"] for c in chunks)
    avg_nll = total_nll / total_tokens
    return {"nll": total_nll, "tokens": total_tokens,
            "avg_nll": avg_nll, "ppl": math.exp(avg_nll)}


@dataclass
class PerplexityRequest:
    binary: Path
    model: str
    model_dir: Path
    quant: str
    dataset: str            # "pg19" or "wikitext2"
    contexts: int           # context length per chunk
    num_chunks: int


def score_perplexity(req: PerplexityRequest) -> dict:
    """Run --teacher-forced over `num_chunks` chunks of the dataset.
    Returns a dict suitable for writing as a quality cell:
       {schema_version: 1, model, quant, eval, metric: "ppl", value: ppl, extras: {...}}
    """
    chunks_per_run = _load_chunks(req.dataset, req.contexts, req.num_chunks)
    chunk_results = []
    for chunk_idx, prompt in enumerate(chunks_per_run):
        out = _run_supersonic_teacher_forced(req, prompt)
        m = _TF_JSON.search(out)
        if not m:
            # A missing [teacher_forced_json] line means the runner crashed,
            # the kernel OOMed, or the model variant doesn't support
            # --teacher-forced. Failing loudly is the right move: silently
            # averaging fewer chunks than requested would publish biased
            # numbers. Surface the failure with an output tail for triage.
            tail = out[-2000:] if out else "<no output>"
            raise PerplexityChunkError(
                f"missing [teacher_forced_json] in chunk {chunk_idx} "
                f"({req.model}/{req.quant}/{req.dataset}); output tail:\n{tail}"
            )
        d = json.loads(m.group(1))
        chunk_results.append({"nll": d["total_nll"], "tokens": d["scored_tokens"]})
    agg = aggregate_ppl_from_chunks(chunk_results)
    return {
        "schema_version": 1,
        "model": req.model,
        "quant": req.quant,
        "eval": f"perplexity_{req.dataset}",
        "metric": "ppl",
        "value": agg["ppl"],
        "extras": {"avg_nll": agg["avg_nll"], "tokens": agg["tokens"],
                   "contexts": req.contexts, "num_chunks": req.num_chunks},
    }


def _load_chunks(dataset: str, contexts: int, num_chunks: int) -> list[str]:
    """Load `num_chunks` text chunks of `contexts` tokens from the named dataset.
    Reuses oracle/pg19_smoke.py loading logic where possible.
    """
    if dataset == "pg19":
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        # PG-19 is char streams; the supersonic --teacher-forced path tokenizes
        # internally. Slice each chunk to ~contexts*5 chars (rough byte-per-token).
        out = []
        char_budget = contexts * 5
        for row in ds:
            text = row["text"][:char_budget]
            out.append(text)
            if len(out) >= num_chunks:
                break
        return out
    if dataset == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined = "\n".join(ds["text"])
        out, char_budget = [], contexts * 5
        for i in range(num_chunks):
            chunk = joined[i * char_budget:(i + 1) * char_budget]
            if not chunk.strip():
                break
            out.append(chunk)
        return out
    raise ValueError(f"unknown dataset {dataset!r}")


def _run_supersonic_teacher_forced(req: PerplexityRequest, prompt: str) -> str:
    cmd = [str(req.binary), "--model", req.model, "--model-dir", str(req.model_dir),
           "--prompt", prompt, "--max-new-tokens", "1", "--teacher-forced"]
    cmd.extend({"bf16": [], "int4": ["--int4"], "fp8r": ["--fp8-runtime"],
                "kv-fp8": ["--kv-fp8"], "int8": ["--int8"]}[req.quant])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (res.stdout or "") + "\n" + (res.stderr or "")
