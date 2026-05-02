#!/usr/bin/env python3
"""Deterministic Qwen3.6 MoE verification suite.

The suite shells out to `supersonic`, captures Qwen3.6 final-hidden/logit
dumps, and checks repeatability across identical runs. It intentionally keeps
the first gate simple: if the same prompt/backend/mode does not produce the
same dumped bytes, the lane cannot be used as a quantized verification base.

Prompt families:
  - pg19: fixed token windows from PG-19, a local text file, or a synthetic
    fallback for CI.
  - ruler: deterministic RULER/NIAH-like retrieval haystacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FILLER_BLOCK = (
    "The archive describes a sequence of journeys, letters, instruments, "
    "weather observations, family names, and quiet digressions about distant "
    "cities. Each paragraph adds ordinary detail so the important facts are "
    "embedded in a long and repetitive context rather than placed near the "
    "question. The reader must preserve exact associations across the whole "
    "passage while ignoring unrelated prose.\n\n"
)

SYNTHETIC_PG19_BLOCK = (
    "In the late afternoon the house stood silent above the road, and the "
    "pages of the old book stirred whenever the window admitted a thread of "
    "wind. There were accounts of voyages, inventories of rooms, arguments "
    "between cousins, and careful descriptions of towns that no one in the "
    "family had visited for many years. The style was patient and elaborate, "
    "as though the narrator meant to preserve every small change of light.\n\n"
)

RULER_KEYS = [
    "amber", "basil", "cobalt", "dorian", "ember", "fable", "garnet",
    "harbor", "ivory", "jasper", "kepler", "linen", "mistral", "nickel",
    "opal", "prairie", "quartz", "raven", "saffron", "tundra",
]


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    context: int
    prompt: str
    references: list[str]
    prompt_tokens: int


class TokenizerAdapter:
    def __init__(self, model_dir: Path):
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "tokenizers is required for qwen36_verify_suite.py; install it with "
                "`python3 -m pip install tokenizers`"
            ) from exc
        tokenizer_json = model_dir / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(tokenizer_json)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=True).ids)

    def encode_plain(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode_plain(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))


def parse_contexts(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower().endswith("k"):
            out.append(int(part[:-1]) * 1024)
        else:
            out.append(int(part))
    if not out:
        raise ValueError("no contexts specified")
    return out


def stable_seed(seed_base: int, *parts: object) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return seed_base + int(hashlib.md5(key).hexdigest()[:8], 16)


def fit_to_context(text: str, tokenizer: TokenizerAdapter, target_tokens: int) -> tuple[str, int]:
    ids = tokenizer.encode_plain(text)
    if len(ids) >= target_tokens:
        prompt = tokenizer.decode_plain(ids[:target_tokens])
        return prompt, len(tokenizer.encode(prompt))

    pieces = [text]
    while len(tokenizer.encode_plain("".join(pieces))) < target_tokens:
        pieces.append(FILLER_BLOCK)
    ids = tokenizer.encode_plain("".join(pieces))
    prompt = tokenizer.decode_plain(ids[:target_tokens])
    return prompt, len(tokenizer.encode(prompt))


def load_pg19_texts(args: argparse.Namespace) -> Iterable[str]:
    if args.pg19_source == "text":
        if args.pg19_text is None:
            raise ValueError("--pg19-text is required with --pg19-source=text")
        yield args.pg19_text.read_text()
        return
    if args.pg19_source == "synthetic":
        yield SYNTHETIC_PG19_BLOCK * 20000
        return

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("datasets is required with --pg19-source=dataset") from exc
    for row in load_dataset("emozilla/pg19", split="test", streaming=True):
        text = str(row.get("text") or "")
        if text.strip():
            yield text


def build_pg19_cases(
    args: argparse.Namespace,
    tokenizer: TokenizerAdapter,
    contexts: list[int],
) -> list[Case]:
    cases: list[Case] = []
    for ctx in contexts:
        wanted = max(1, ctx - args.max_new_tokens)
        made = 0
        for doc_idx, text in enumerate(load_pg19_texts(args)):
            ids = tokenizer.encode_plain(text)
            if len(ids) < wanted:
                continue
            stride = args.pg19_stride or wanted
            for start in range(0, len(ids) - wanted + 1, stride):
                prompt = tokenizer.decode_plain(ids[start : start + wanted])
                prompt_tokens = len(tokenizer.encode(prompt))
                cases.append(Case(
                    case_id=f"pg19_ctx{ctx}_doc{doc_idx}_off{start}",
                    family="pg19",
                    context=ctx,
                    prompt=prompt,
                    references=[],
                    prompt_tokens=prompt_tokens,
                ))
                made += 1
                if made >= args.pg19_samples:
                    break
            if made >= args.pg19_samples:
                break
        if made < args.pg19_samples:
            raise RuntimeError(f"only built {made} PG-19 cases for context {ctx}")
    return cases


def make_ruler_prompt(
    rng: random.Random,
    context: int,
    tokenizer: TokenizerAdapter,
    max_new: int,
) -> tuple[str, int, list[str]]:
    key = rng.choice(RULER_KEYS)
    value = "".join(rng.choice(string.digits) for _ in range(8))
    payload = f"\nThe verification code paired with key {key} is {value}.\n\n"
    question = (
        f"\nQuestion: What is the verification code paired with key {key}? "
        f"Answer with only the code.\nAnswer:"
    )
    target = max(1, context - max_new)

    filler_tokens = max(1, len(tokenizer.encode_plain(FILLER_BLOCK)))
    payload_tokens = len(tokenizer.encode_plain(payload + question))
    filler_count = max(0, (target - payload_tokens) // filler_tokens)
    while True:
        insert_at = rng.randrange(0, filler_count + 1) if filler_count else 0
        parts: list[str] = []
        for idx in range(filler_count):
            if idx == insert_at:
                parts.append(payload)
            parts.append(FILLER_BLOCK)
        if insert_at == filler_count:
            parts.append(payload)
        parts.append(question)
        prompt = "".join(parts)
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens <= target or filler_count == 0:
            break
        filler_count -= 1
    return prompt, prompt_tokens, [value]


def build_ruler_cases(
    args: argparse.Namespace,
    tokenizer: TokenizerAdapter,
    contexts: list[int],
) -> list[Case]:
    cases: list[Case] = []
    for ctx in contexts:
        for sample_idx in range(args.ruler_samples):
            rng = random.Random(stable_seed(args.seed, "ruler", ctx, sample_idx))
            prompt, prompt_tokens, refs = make_ruler_prompt(rng, ctx, tokenizer, args.max_new_tokens)
            cases.append(Case(
                case_id=f"ruler_niah_ctx{ctx}_sample{sample_idx}",
                family="ruler",
                context=ctx,
                prompt=prompt,
                references=refs,
                prompt_tokens=prompt_tokens,
            ))
    return cases


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_values(path: Path) -> list[float]:
    data = path.read_bytes()
    out: list[float] = []
    for i in range(0, len(data), 2):
        u = int.from_bytes(data[i : i + 2], "little")
        out.append(float_from_u32(u << 16))
    return out


def float_from_u32(raw: int) -> float:
    import struct
    return struct.unpack("<f", raw.to_bytes(4, "little"))[0]


def vector_stats(path: Path) -> dict[str, Any]:
    vals = bf16_values(path)
    finite = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"len": 0, "finite": 0, "nan": 0, "all_zero": True}
    argmax = max(range(len(vals)), key=vals.__getitem__)
    return {
        "len": len(vals),
        "finite": len(finite),
        "nan": len(vals) - len(finite),
        "all_zero": all(v == 0.0 for v in vals),
        "argmax": argmax,
        "max": vals[argmax],
        "min": min(vals),
    }


def compare_vectors(a_path: Path, b_path: Path) -> dict[str, float]:
    a = bf16_values(a_path)
    b = bf16_values(b_path)
    n = min(len(a), len(b))
    if n == 0:
        return {"max_abs": math.inf, "rmse": math.inf, "cosine": 0.0}
    max_abs = 0.0
    ss = 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for x, y in zip(a[:n], b[:n]):
        d = x - y
        max_abs = max(max_abs, abs(d))
        ss += d * d
        dot += x * y
        aa += x * x
        bb += y * y
    denom = math.sqrt(aa) * math.sqrt(bb)
    return {
        "max_abs": max_abs,
        "rmse": math.sqrt(ss / n),
        "cosine": dot / denom if denom else 0.0,
    }


def parse_generated_ids(output: str) -> list[int]:
    match = re.search(r"Generated ids:\s*\[([^\]]*)\]", output)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_stage_timings(output: str) -> dict[str, float]:
    match = re.search(r"\[qwen36-moe stage-timings\]\s+(.+)", output)
    if not match:
        return {}
    out: dict[str, float] = {}
    for part in match.group(1).split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def run_once(
    args: argparse.Namespace,
    case: Case,
    mode: str,
    decode_path: str,
    repeat_idx: int,
    tmp: Path,
) -> dict[str, Any]:
    # Dump filenames include `decode_path` so chained/persistent runs of the
    # same (case, mode, repeat_idx) don't clobber each other — the
    # bit-identity check below diff's their SHA256s.
    logits = tmp / f"{case.case_id}_{mode}_{decode_path}_{repeat_idx}_logits.bin"
    hidden = tmp / f"{case.case_id}_{mode}_{decode_path}_{repeat_idx}_hidden.bin"
    env = os.environ.copy()
    env["SUPERSONIC_QWEN36_DUMP_LOGITS"] = str(logits)
    env["SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN"] = str(hidden)
    env.setdefault("SUPERSONIC_BACKENDS", args.backend)

    cmd = [
        str(args.binary),
        "--backend", args.backend,
        "--model", "qwen3.6-35b-a3b",
        "--model-dir", str(args.model_dir),
        "--prompt", case.prompt,
        "--context-size", str(max(case.context, case.prompt_tokens + args.max_new_tokens)),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", "0",
        "--top-k", "1",
        "--sampling-seed", str(args.seed),
    ]
    if mode == "fp8":
        cmd.append("--fp8-runtime")
    elif mode == "int4":
        cmd.append("--int4")
    else:
        raise ValueError(f"unknown mode {mode}")
    if args.emit_stage_timings:
        cmd.append("--emit-stage-timings")
    if decode_path == "persistent":
        cmd.append("--persistent-decode")
    elif decode_path != "chained":
        raise ValueError(f"unknown decode_path {decode_path}")

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        output = stdout + stderr
        return {
            "mode": mode,
            "decode_path": decode_path,
            "repeat_idx": repeat_idx,
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": args.timeout,
            "wall_seconds": wall,
            "generated_ids": parse_generated_ids(output),
            "stage": parse_stage_timings(output),
            "stdout_tail": stdout[-1200:],
            "stderr_tail": stderr[-1200:],
            "error": f"supersonic timed out after {args.timeout}s",
        }
    wall = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    row: dict[str, Any] = {
        "mode": mode,
        "decode_path": decode_path,
        "repeat_idx": repeat_idx,
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "generated_ids": parse_generated_ids(output),
        "stage": parse_stage_timings(output),
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }
    if proc.returncode != 0:
        row["error"] = output[-4000:]
        return row
    if not logits.exists() or not hidden.exists():
        row["error"] = "missing qwen36 dump files"
        return row
    row["logits_path"] = str(logits)
    row["hidden_path"] = str(hidden)
    row["logits_sha256"] = sha256_file(logits)
    row["hidden_sha256"] = sha256_file(hidden)
    row["logits_stats"] = vector_stats(logits)
    row["hidden_stats"] = vector_stats(hidden)
    return row


def _cell_key(row: dict[str, Any]) -> str:
    """Two-axis cell key: `<mode>/<decode_path>`. Older payloads pre-Phase
    3e.2 only had a `mode` axis; default `decode_path=chained` for
    backwards compatibility on those rows."""
    return f"{row['mode']}/{row.get('decode_path', 'chained')}"


def _stage_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of `row['stage'][key]` across `rows`. Used for the
    chained-vs-persistent perf delta. Skips rows whose stage block is
    missing the key (timeouts, runs that crashed before stage timings
    printed, etc.)."""
    vals = [
        float(r["stage"][key])
        for r in rows
        if isinstance(r.get("stage"), dict) and key in r["stage"]
    ]
    return statistics.mean(vals) if vals else None


def summarize_case(case: Case, runs: list[dict[str, Any]]) -> dict[str, Any]:
    # Two-axis grouping: cell = (mode × decode_path). Each cell gets its
    # own determinism + perf stats. The `chained_vs_persistent` block
    # below cross-references cells when both decode paths were run for
    # the same mode.
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_cell.setdefault(_cell_key(row), []).append(row)
    summary: dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "context": case.context,
        "prompt_tokens": case.prompt_tokens,
        "references": case.references,
        "modes": {},
    }
    for cell, rows in by_cell.items():
        ok_rows = [r for r in rows if r.get("returncode") == 0 and not r.get("error")]
        logits_hashes = {r.get("logits_sha256") for r in ok_rows}
        hidden_hashes = {r.get("hidden_sha256") for r in ok_rows}
        generated = {tuple(r.get("generated_ids") or []) for r in ok_rows}
        all_zero = [bool((r.get("logits_stats") or {}).get("all_zero")) for r in ok_rows]
        wall = [float(r.get("wall_seconds") or 0.0) for r in ok_rows]
        cell_summary = {
            "runs": len(rows),
            "ok_runs": len(ok_rows),
            "deterministic_logits": len(logits_hashes) == 1 and len(ok_rows) == len(rows),
            "deterministic_hidden": len(hidden_hashes) == 1 and len(ok_rows) == len(rows),
            "deterministic_generated_ids": len(generated) == 1 and len(ok_rows) == len(rows),
            "any_all_zero_logits": any(all_zero),
            "logits_hashes": sorted(h for h in logits_hashes if h),
            "hidden_hashes": sorted(h for h in hidden_hashes if h),
            "generated_ids": [list(x) for x in sorted(generated)],
            "wall_seconds_mean": statistics.mean(wall) if wall else None,
            "chain_ms_avg_mean": _stage_mean(ok_rows, "chain_ms_avg"),
            "total_ms_avg_mean": _stage_mean(ok_rows, "total_ms_avg"),
            "errors": [r.get("error") for r in rows if r.get("error") or r.get("returncode") != 0],
        }
        summary["modes"][cell] = cell_summary

    # FP8↔INT4 quantization comparison (existing). Look for the chained
    # variants of each mode by default — chained is the canonical
    # numerical reference. Falls back to whichever decode_path was run if
    # only one is present.
    def _pick_for_mode(mode: str) -> dict[str, Any] | None:
        for decode_path in ("chained", "persistent"):
            cell_rows = by_cell.get(f"{mode}/{decode_path}", [])
            ok = [r for r in cell_rows if r.get("logits_path") and not r.get("error")]
            if ok:
                return ok[0]
        return None

    fp8_ref = _pick_for_mode("fp8")
    int4_ref = _pick_for_mode("int4")
    if fp8_ref and int4_ref:
        summary["fp8_vs_int4_logits"] = compare_vectors(
            Path(fp8_ref["logits_path"]),
            Path(int4_ref["logits_path"]),
        )

    # Chained-vs-persistent comparison per mode. Bit-identity is the
    # gate (Phase 3e parity test asserts it on synthetic fixtures); this
    # block surfaces it for every real prompt + verifies the perf delta
    # matches the projected ~9-10% from launch-overhead reclaim.
    chained_vs_persistent: dict[str, Any] = {}
    for mode in {row["mode"] for row in runs}:
        chained_rows = by_cell.get(f"{mode}/chained", [])
        persistent_rows = by_cell.get(f"{mode}/persistent", [])
        if not chained_rows or not persistent_rows:
            continue
        c_ok = [r for r in chained_rows if r.get("logits_path") and not r.get("error")]
        p_ok = [r for r in persistent_rows if r.get("logits_path") and not r.get("error")]
        if not c_ok or not p_ok:
            continue
        block: dict[str, Any] = {}
        # Bit-identity: the persistent kernel runs the IDENTICAL device
        # functions as the chained step kernels, so logits + hidden + ids
        # should match byte-for-byte across every (case, mode, repeat).
        c_logits = {r["logits_sha256"] for r in c_ok if r.get("logits_sha256")}
        p_logits = {r["logits_sha256"] for r in p_ok if r.get("logits_sha256")}
        c_hidden = {r["hidden_sha256"] for r in c_ok if r.get("hidden_sha256")}
        p_hidden = {r["hidden_sha256"] for r in p_ok if r.get("hidden_sha256")}
        c_ids = {tuple(r.get("generated_ids") or []) for r in c_ok}
        p_ids = {tuple(r.get("generated_ids") or []) for r in p_ok}
        block["logits_bit_identical"] = bool(c_logits) and c_logits == p_logits
        block["hidden_bit_identical"] = bool(c_hidden) and c_hidden == p_hidden
        block["generated_ids_match"] = bool(c_ids) and c_ids == p_ids
        # Perf delta. Use stage-level chain_ms_avg when available (most
        # accurate), wall_seconds otherwise.
        c_chain = _stage_mean(c_ok, "chain_ms_avg")
        p_chain = _stage_mean(p_ok, "chain_ms_avg")
        if c_chain is not None and p_chain is not None and c_chain > 0:
            block["chain_ms_avg_chained"] = c_chain
            block["chain_ms_avg_persistent"] = p_chain
            block["chain_speedup_pct"] = 100.0 * (c_chain - p_chain) / c_chain
        c_total = _stage_mean(c_ok, "total_ms_avg")
        p_total = _stage_mean(p_ok, "total_ms_avg")
        if c_total is not None and p_total is not None and c_total > 0:
            block["total_ms_avg_chained"] = c_total
            block["total_ms_avg_persistent"] = p_total
            block["total_speedup_pct"] = 100.0 * (c_total - p_total) / c_total
        chained_vs_persistent[mode] = block
    if chained_vs_persistent:
        summary["chained_vs_persistent"] = chained_vs_persistent

    return summary


def evaluate_failures(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for case in payload["summary"]:
        # Cell labels are now `<mode>/<decode_path>`; older payloads
        # written before Phase 3e.2 stayed as plain `<mode>` and the
        # default decode_path is chained, so the "/" delimiter just
        # disambiguates and stays human-readable.
        for cell, row in case["modes"].items():
            label = f"{case['case_id']}:{cell}"
            if row["ok_runs"] != row["runs"]:
                failures.append(f"{label} had {row['runs'] - row['ok_runs']} failed runs")
            if args.fail_on_nondeterminism:
                if not row["deterministic_logits"]:
                    failures.append(f"{label} logits are nondeterministic")
                if not row["deterministic_hidden"]:
                    failures.append(f"{label} hidden state is nondeterministic")
                if not row["deterministic_generated_ids"]:
                    failures.append(f"{label} generated ids are nondeterministic")
            if row["any_all_zero_logits"]:
                failures.append(f"{label} produced all-zero logits")
        # Phase 3e.2 gate: when both decode paths are run for a mode,
        # the persistent kernel must produce byte-identical
        # logits/hidden/generated_ids vs the chained reference. This is
        # the production-prompt analog of the
        # `multilayer_persistent_decode_matches_chained` parity test;
        # any divergence here means a regression in the megakernel or
        # its descriptor wiring on a real prompt the parity test
        # doesn't cover.
        cvp = case.get("chained_vs_persistent") or {}
        for mode, block in cvp.items():
            label = f"{case['case_id']}:{mode}"
            if not block.get("logits_bit_identical", True):
                failures.append(
                    f"{label} chained vs persistent logits diverged "
                    f"(should be byte-identical)"
                )
            if not block.get("hidden_bit_identical", True):
                failures.append(
                    f"{label} chained vs persistent hidden state diverged"
                )
            if not block.get("generated_ids_match", True):
                failures.append(
                    f"{label} chained vs persistent generated_ids diverged"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--contexts", default="512,2K,8K")
    parser.add_argument("--families", default="pg19,ruler", help="comma list: pg19,ruler")
    parser.add_argument("--modes", default="fp8,int4", help="comma list: fp8,int4")
    parser.add_argument(
        "--decode-paths",
        default="chained",
        help=(
            "comma list: chained,persistent. `persistent` adds "
            "`--persistent-decode` to the supersonic command (Phase 3e.2 "
            "megakernel — one cooperative HIP launch per token instead of "
            "80 step launches). When both are present the suite gates "
            "byte-identity of logits/hidden/generated_ids and reports the "
            "perf delta in `chained_vs_persistent` per case."
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--out", type=Path, default=Path("qwen36_verify_results.json"))
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--emit-stage-timings", action="store_true")
    parser.add_argument("--fail-on-nondeterminism", action="store_true", default=True)
    parser.add_argument("--no-fail-on-nondeterminism", dest="fail_on_nondeterminism", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--pg19-source", choices=["dataset", "text", "synthetic"], default="synthetic")
    parser.add_argument("--pg19-text", type=Path)
    parser.add_argument("--pg19-samples", type=int, default=1)
    parser.add_argument("--pg19-stride", type=int)
    parser.add_argument("--ruler-samples", type=int, default=1)
    args = parser.parse_args()

    contexts = parse_contexts(args.contexts)
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    decode_paths = [x.strip() for x in args.decode_paths.split(",") if x.strip()]
    if not decode_paths:
        decode_paths = ["chained"]
    for dp in decode_paths:
        if dp not in ("chained", "persistent"):
            raise ValueError(f"--decode-paths entry {dp!r} not in {{chained, persistent}}")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)

    tokenizer = TokenizerAdapter(args.model_dir)
    cases: list[Case] = []
    if "pg19" in families:
        cases.extend(build_pg19_cases(args, tokenizer, contexts))
    if "ruler" in families:
        cases.extend(build_ruler_cases(args, tokenizer, contexts))

    tmp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.tmp_dir is None:
        tmp_owner = tempfile.TemporaryDirectory(prefix="qwen36-verify-")
        tmp = Path(tmp_owner.name)
    else:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = args.tmp_dir

    all_runs: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_runs: list[dict[str, Any]] = []
            print(
                f"[case] {case.case_id} family={case.family} "
                f"ctx={case.context} prompt_tokens={case.prompt_tokens}",
                flush=True,
            )
            for mode in modes:
                for decode_path in decode_paths:
                    for repeat_idx in range(args.repeats):
                        row = run_once(args, case, mode, decode_path, repeat_idx, tmp)
                        row["case_id"] = case.case_id
                        row["family"] = case.family
                        row["context"] = case.context
                        row["prompt_tokens"] = case.prompt_tokens
                        case_runs.append(row)
                        all_runs.append(row)
                        status = "ok" if row.get("returncode") == 0 and not row.get("error") else "FAIL"
                        cell_label = f"{mode}/{decode_path}"
                        print(
                            f"  {cell_label:<18} repeat={repeat_idx} {status} "
                            f"ids={row.get('generated_ids')} "
                            f"logits={str(row.get('logits_sha256', ''))[:12]} "
                            f"zero={(row.get('logits_stats') or {}).get('all_zero')}",
                            flush=True,
                        )
                        if status != "ok" and not args.continue_on_error:
                            raise RuntimeError(str(row.get("error") or "run failed"))
            case_summary = summarize_case(case, case_runs)
            summary.append(case_summary)
            for cell, row in case_summary["modes"].items():
                print(
                    f"  summary {cell:<18} logits_det={row['deterministic_logits']} "
                    f"hidden_det={row['deterministic_hidden']} ids_det={row['deterministic_generated_ids']} "
                    f"all_zero={row['any_all_zero_logits']}",
                    flush=True,
                )
            cvp = case_summary.get("chained_vs_persistent")
            if cvp:
                for mode, block in cvp.items():
                    bits_ok = (
                        block.get("logits_bit_identical")
                        and block.get("hidden_bit_identical")
                        and block.get("generated_ids_match")
                    )
                    speedup = block.get("chain_speedup_pct")
                    parts = [
                        f"  cvp     {mode:<4} bit_identical={'yes' if bits_ok else 'NO!'}",
                    ]
                    if speedup is not None:
                        parts.append(
                            f"chain_speedup={speedup:+.2f}% "
                            f"(chained={block['chain_ms_avg_chained']:.2f}ms "
                            f"→ persistent={block['chain_ms_avg_persistent']:.2f}ms)"
                        )
                    print(" ".join(parts), flush=True)

        payload = {
            "schema": "qwen36-verify-suite-v1",
            "model": "qwen3.6-35b-a3b",
            "model_dir": str(args.model_dir),
            "contexts": contexts,
            "families": families,
            "modes": modes,
            "decode_paths": decode_paths,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "summary": summary,
            "runs": all_runs,
        }
        failures = evaluate_failures(payload, args)
        payload["failures"] = failures
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"[qwen36-verify] wrote {args.out}", flush=True)
        if failures:
            print("[qwen36-verify] FAIL", flush=True)
            for failure in failures:
                print(f"  - {failure}", flush=True)
            return 1
        print("[qwen36-verify] PASS", flush=True)
        return 0
    finally:
        if tmp_owner is not None:
            tmp_owner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
