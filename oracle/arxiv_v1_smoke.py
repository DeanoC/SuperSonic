#!/usr/bin/env python3
"""arXiv v1 smoke QA for Llama 3.1 certified KV work.

This is a small, reproducible subset of the DotCache paper benchmarks.  It
uses the same synthetic RULER/NIAH prompt builders and can compare the resulting
scores against the normalized arxiv_v1 JSON files produced by DotCache.

The harness intentionally shells out to the SuperSonic binary for generation so
the same prompts can gate both the native CUDA dense INT8 path and the certified
KV path.  It does not attempt to run PG-19 perplexity yet because SuperSonic's
CLI does not expose token-level logprobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FILLER_BLOCK = (
    "The history of mathematics spans thousands of years and encompasses many "
    "different cultures and civilizations. From the earliest counting systems "
    "developed by ancient peoples to the sophisticated abstract algebras of the "
    "modern era, mathematical knowledge has grown through a process of discovery, "
    "invention, and refinement. The Babylonians developed a base-60 number system "
    "that still influences how we measure time and angles today. The ancient Greeks "
    "made foundational contributions to geometry, number theory, and logic. During "
    "the Islamic Golden Age, scholars preserved and extended Greek mathematics while "
    "making original advances in algebra and trigonometry. The Renaissance saw a "
    "flowering of mathematical activity in Europe, leading eventually to the "
    "development of calculus by Newton and Leibniz in the seventeenth century. "
    "In the nineteenth and twentieth centuries, mathematics became increasingly "
    "abstract and specialized, with new fields emerging at a rapid pace. Today, "
    "mathematics is essential to science, engineering, economics, and many other "
    "domains of human activity.\n\n"
)

NIAH_PREAMBLE = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize them. I will quiz you about the numbers afterwards.\n\n"
)
NIAH_SINGLE_PREAMBLE = (
    "A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards.\n\n"
)
NIAH_KEY_WORDS = [
    "pelican",
    "quarry",
    "moonlit",
    "cipher",
    "lantern",
    "nebula",
    "prism",
    "velvet",
    "orbit",
    "rustic",
    "ember",
    "zenith",
    "hollow",
    "falcon",
    "meadow",
    "silver",
    "coral",
    "breeze",
    "ivory",
    "twilight",
]

_CWE_POOL = [
    "apple",
    "river",
    "stone",
    "cloud",
    "forest",
    "mountain",
    "valley",
    "ocean",
    "desert",
    "island",
    "meadow",
    "garden",
    "bridge",
    "castle",
    "tower",
    "harbor",
    "market",
    "temple",
    "palace",
    "village",
    "country",
    "highway",
    "railway",
    "canyon",
    "glacier",
    "volcano",
    "prairie",
    "jungle",
    "marsh",
    "lagoon",
]

SUBTASK_MAX_NEW = {
    "niah_single": 32,
    "niah_multikey": 32,
    "niah_multivalue": 96,
    "niah_multiquery": 48,
    "vt": 96,
    "cwe": 96,
    "fwe": 32,
}


@dataclass(frozen=True)
class ReferenceMetric:
    benchmark: str
    config: str
    context_length: int
    metric: str
    value: float | None
    source: str
    subtask_values: dict[str, float]


def _random_number_value(rng: random.Random, digits: int = 7) -> str:
    return "".join(rng.choice(string.digits) for _ in range(digits))


def _pick_keys(rng: random.Random, k: int) -> list[str]:
    return rng.sample(NIAH_KEY_WORDS, k)


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fill_haystack(
    target_tokens: int,
    tokenizer: Any,
    payload_blocks: list[str],
    rng: random.Random,
) -> str:
    filler_tpb = _token_len(tokenizer, FILLER_BLOCK)
    payload_total = sum(_token_len(tokenizer, p) for p in payload_blocks)
    available = target_tokens - payload_total - 100
    num_filler = max(available // filler_tpb, len(payload_blocks) + 1)
    positions = sorted(rng.sample(range(num_filler), len(payload_blocks)))

    parts: list[str] = []
    payload_idx = 0
    for idx in range(num_filler):
        if payload_idx < len(positions) and idx == positions[payload_idx]:
            parts.append(payload_blocks[payload_idx])
            payload_idx += 1
        parts.append(FILLER_BLOCK)
    return "".join(parts)


def make_niah_single(rng: random.Random, ctx_tokens: int, tokenizer: Any) -> tuple[str, list[str]]:
    key = _pick_keys(rng, 1)[0]
    value = _random_number_value(rng)
    payload = f"\nOne of the special magic numbers for {key} is: {value}.\n\n"
    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    question = (
        f"\nWhat is the special magic number for {key} mentioned in the "
        f"provided text?\nThe special magic number for {key} mentioned in "
        f"the provided text is"
    )
    return NIAH_SINGLE_PREAMBLE + haystack + question, [value]


def make_niah_multikey(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    num_keys: int = 4,
) -> tuple[str, list[str]]:
    keys = _pick_keys(rng, num_keys)
    values = [_random_number_value(rng) for _ in keys]
    target_idx = rng.randrange(num_keys)
    target_key = keys[target_idx]
    target_value = values[target_idx]
    payloads = [
        f"\nOne of the special magic numbers for {key} is: {value}.\n\n"
        for key, value in zip(keys, values)
    ]
    haystack = _fill_haystack(ctx_tokens, tokenizer, payloads, rng)
    question = (
        f"\nWhat is the special magic number for {target_key} mentioned in "
        f"the provided text?\nThe special magic number for {target_key} "
        f"mentioned in the provided text is"
    )
    return NIAH_PREAMBLE + haystack + question, [target_value]


def make_niah_multivalue(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    num_values: int = 4,
) -> tuple[str, list[str]]:
    key = _pick_keys(rng, 1)[0]
    values = [_random_number_value(rng) for _ in range(num_values)]
    payloads = [
        f"\nOne of the special magic numbers for {key} is: {value}.\n\n"
        for value in values
    ]
    haystack = _fill_haystack(ctx_tokens, tokenizer, payloads, rng)
    question = (
        f"\nWhat are all the special magic numbers for {key} mentioned in "
        f"the provided text?\nThe special magic numbers for {key} mentioned "
        f"in the provided text are"
    )
    return NIAH_PREAMBLE + haystack + question, values


def make_niah_multiquery(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    num_queries: int = 4,
) -> tuple[str, list[str]]:
    keys = _pick_keys(rng, num_queries)
    planted_idx = rng.randrange(num_queries)
    planted_key = keys[planted_idx]
    planted_value = _random_number_value(rng)
    payload = f"\nOne of the special magic numbers for {planted_key} is: {planted_value}.\n\n"
    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    query_list = ", ".join(keys)
    question = (
        f"\nFor which of the following keys was a special magic number "
        f"mentioned in the provided text: {query_list}? Provide the key "
        f"and its magic number.\nThe key with the special magic number is"
    )
    return NIAH_PREAMBLE + haystack + question, [planted_key, planted_value]


def make_vt(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    chain_len: int = 4,
    noise_chains: int = 3,
) -> tuple[str, list[str]]:
    target_value = _random_number_value(rng, digits=5)
    target_vars = [f"VAR_{rng.randint(10, 99)}_{idx}" for idx in range(chain_len)]
    while len(set(target_vars)) < chain_len:
        target_vars = [f"VAR_{rng.randint(10, 99)}_{idx}" for idx in range(chain_len)]

    chain_lines = [f"{target_vars[0]} = {target_value}"]
    for idx in range(1, chain_len):
        chain_lines.append(f"{target_vars[idx]} = {target_vars[idx - 1]}")

    noise_lines: list[str] = []
    for _ in range(noise_chains):
        noise_value = _random_number_value(rng, digits=5)
        noise_vars = [f"VAR_{rng.randint(100, 999)}_{idx}" for idx in range(chain_len)]
        noise_lines.append(f"{noise_vars[0]} = {noise_value}")
        for idx in range(1, chain_len):
            noise_lines.append(f"{noise_vars[idx]} = {noise_vars[idx - 1]}")

    lines = chain_lines + noise_lines
    rng.shuffle(lines)
    payload = "\n\n" + "\n".join(lines) + "\n\n"
    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    preamble = (
        "Memorize and track the chains of variable assignment hidden within "
        "the following text.\n\n"
    )
    question = (
        f"\nFind all variables that are assigned the value {target_value} "
        f"in the text above.\nAnswer: The variables are"
    )
    return preamble + haystack + question, target_vars


def make_cwe(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    top_k: int = 10,
    freq_high: int = 30,
    freq_low: int = 3,
) -> tuple[str, list[str]]:
    pool = list(_CWE_POOL)
    rng.shuffle(pool)
    targets = pool[:top_k]
    noise = pool[top_k : top_k + 15]
    words: list[str] = []
    for word in targets:
        words.extend([word] * freq_high)
    for word in noise:
        words.extend([word] * freq_low)
    rng.shuffle(words)

    numbered = "\n".join(f"{idx + 1}. {word}" for idx, word in enumerate(words))
    payload = "\n" + numbered + "\n"
    filler_tpb = _token_len(tokenizer, FILLER_BLOCK)
    payload_tpb = _token_len(tokenizer, payload)
    num_filler = max((ctx_tokens - payload_tpb - 100) // filler_tpb, 0)
    haystack = FILLER_BLOCK * num_filler + payload
    preamble = (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n\n"
    )
    question = (
        f"\nQuestion: What are the {top_k} most common words in the above "
        f"list?\nAnswer: The {top_k} most common words are"
    )
    return preamble + haystack + question, targets


def make_fwe(
    rng: random.Random,
    ctx_tokens: int,
    tokenizer: Any,
    top_k: int = 3,
    alpha: float = 2.0,
    pool_size: int = 20,
) -> tuple[str, list[str]]:
    pool = list(_CWE_POOL)
    rng.shuffle(pool)
    pool = pool[:pool_size]
    weights = [1.0 / ((idx + 1) ** alpha) for idx in range(len(pool))]
    total_word_tokens = max(200, ctx_tokens // 4)
    words = rng.choices(pool, weights=weights, k=total_word_tokens)

    from collections import Counter

    refs = [word for word, _ in Counter(words).most_common(top_k)]
    payload = "\n" + " ".join(f".... {word}" for word in words) + "\n"
    filler_tpb = _token_len(tokenizer, FILLER_BLOCK)
    payload_tpb = _token_len(tokenizer, payload)
    num_filler = max((ctx_tokens - payload_tpb - 100) // filler_tpb, 0)
    haystack = FILLER_BLOCK * num_filler + payload
    preamble = (
        "Read the following coded text and track the frequency of each coded "
        "word. Find the three most frequently appeared coded words.\n\n"
    )
    question = (
        "\nQuestion: Do not provide any explanation. Please ignore the dots "
        "'....'. What are the three most frequently appeared words in the "
        "above coded text?\nAnswer: The three most frequently appeared words "
        "are"
    )
    return preamble + haystack + question, refs


SUBTASK_BUILDERS = {
    "niah_single": make_niah_single,
    "niah_multikey": make_niah_multikey,
    "niah_multivalue": make_niah_multivalue,
    "niah_multiquery": make_niah_multiquery,
    "vt": make_vt,
    "cwe": make_cwe,
    "fwe": make_fwe,
}


def stable_seed(seed_base: int, subtask: str, context_length: int, sample_idx: int) -> int:
    key = f"{subtask}|{context_length}|{sample_idx}".encode()
    return seed_base + int(hashlib.md5(key).hexdigest()[:8], 16)


def score_string_match_all(generated: str, references: list[str]) -> float:
    if not references:
        return 1.0
    lowered = generated.lower()
    hits = sum(1 for ref in references if ref.lower() in lowered)
    return hits / len(references)


def parse_supersonic_generation(combined_output: str, prompt: str) -> str:
    """Extract generated suffix from SuperSonic output.

    SuperSonic prints log lines, then the raw prompt plus generated text, then
    `[tokens]`/`[result]`.  Scoring against the raw segment would be invalid for
    retrieval because the reference answer is embedded in the prompt.
    """
    json_match = re.search(r"^\[generated_json\] (.+)$", combined_output, flags=re.MULTILINE)
    if json_match:
        return str(json.loads(json_match.group(1)))

    marker = "[decode]"
    marker_pos = combined_output.find(marker)
    if marker_pos < 0:
        raise RuntimeError("SuperSonic output did not contain a [decode] marker")
    after_decode_line = combined_output.find("\n", marker_pos)
    if after_decode_line < 0:
        raise RuntimeError("SuperSonic output ended at the [decode] line")
    end = combined_output.find("\n[tokens]", after_decode_line)
    if end < 0:
        end = combined_output.find("\n[result]", after_decode_line)
    if end < 0:
        raise RuntimeError("SuperSonic output did not contain [tokens] or [result]")
    segment = combined_output[after_decode_line + 1 : end]
    if segment.startswith(prompt):
        return segment[len(prompt) :]

    # Some terminals normalize trailing whitespace; preserve correctness by
    # falling back to a suffix alignment instead of scoring the whole prompt.
    overlap = min(len(prompt), len(segment))
    for keep in range(overlap, max(overlap - 4096, 0), -1):
        if segment.startswith(prompt[-keep:]):
            return segment[keep:]
    raise RuntimeError("could not strip known prompt from SuperSonic output")


def load_reference(
    results_dir: Path,
    benchmark: str,
    config: str,
    context_length: int,
    smoke: bool = True,
) -> ReferenceMetric | None:
    suffix = ".smoke.json" if smoke else ".json"
    pattern = f"*_{benchmark}_{context_length // 1024}K_{config}{suffix}"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        return None
    path = matches[0]
    data = json.loads(path.read_text())
    quality = data.get("quality", {})
    summary = quality.get("summary") or data.get("native", {}).get("summary") or {}
    subtask_values: dict[str, float] = {}
    for key, row in summary.items():
        if config == "dense":
            value = row.get("dense_mean")
        else:
            value = row.get("cert_mean")
        if value is not None:
            subtask_values[key] = float(value)
    return ReferenceMetric(
        benchmark=str(data.get("benchmark", benchmark)),
        config=str(data.get("config", config)),
        context_length=int(data.get("context_length", context_length)),
        metric=str(quality.get("metric", "")),
        value=(float(quality["value"]) if quality.get("value") is not None else None),
        source=str(path),
        subtask_values=subtask_values,
    )


def run_supersonic(
    binary: Path,
    model_dir: Path,
    prompt: str,
    max_new_tokens: int,
    config: str,
    timeout_s: int,
    emit_stage_timings: bool = False,
) -> tuple[str, dict[str, Any]]:
    cmd = [
        str(binary),
        "--backend",
        "cuda",
        "--model",
        "llama3.1-8b",
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--int8",
        "--emit-generated-json",
    ]
    if config == "certified":
        cmd.append("--certified-kv")
    if emit_stage_timings:
        cmd.append("--emit-stage-timings")
    env = os.environ.copy()
    env.setdefault("SUPERSONIC_BACKENDS", "cuda")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    elapsed = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"SuperSonic failed with exit code {proc.returncode}\n{output}")
    generated = parse_supersonic_generation(output, prompt)
    result = parse_result_line(output)
    result["wall_seconds"] = elapsed
    return generated, result


def parse_result_line(output: str) -> dict[str, Any]:
    match = re.search(
        r"\[result\] prompt_tokens=(\d+) generated_tokens=(\d+) "
        r"decode_ms=([0-9.]+) ms_per_(?:step|tok)=([0-9.]+)",
        output,
    )
    if not match:
        return {}
    return {
        "prompt_tokens": int(match.group(1)),
        "generated_tokens": int(match.group(2)),
        "decode_ms": float(match.group(3)),
        "ms_per_step": float(match.group(4)),
    }


def run_ruler_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    configs = ["dense", "certified"] if args.config == "both" else [args.config]
    contexts = args.contexts
    subtasks = args.subtasks
    references = {
        cfg: {
            ctx: load_reference(args.reference_dir, "ruler", cfg, ctx, smoke=args.reference_smoke)
            for ctx in contexts
        }
        for cfg in configs
    }

    all_results: list[dict[str, Any]] = []
    for ctx in contexts:
        for subtask in subtasks:
            builder = SUBTASK_BUILDERS[subtask]
            max_new = args.max_new_tokens or SUBTASK_MAX_NEW[subtask]
            for sample_idx in range(args.samples):
                rng = random.Random(stable_seed(args.seed, subtask, ctx, sample_idx))
                prompt, refs = builder(rng, ctx, tokenizer)
                for cfg in configs:
                    generated, timing = run_supersonic(
                        args.binary,
                        args.model_dir,
                        prompt,
                        max_new,
                        cfg,
                        args.timeout,
                        emit_stage_timings=args.emit_stage_timings,
                    )
                    score = score_string_match_all(generated, refs)
                    ref = references[cfg][ctx]
                    ref_key = f"{subtask}_{ctx // 1024}K"
                    ref_score = ref.subtask_values.get(ref_key) if ref else None
                    row = {
                        "benchmark": "ruler",
                        "config": cfg,
                        "context_length": ctx,
                        "subtask": subtask,
                        "sample_idx": sample_idx,
                        "references": refs,
                        "score": score,
                        "generated": generated[:400],
                        "timing": timing,
                        "reference_score": ref_score,
                        "reference_source": ref.source if ref else None,
                    }
                    all_results.append(row)
                    ref_part = "" if ref_score is None else f" ref={ref_score:.3f}"
                    print(
                        f"{cfg:<9} ctx={ctx:<5} {subtask:<16} "
                        f"sample={sample_idx:<2} score={score:.3f}{ref_part} "
                        f"ms_step={timing.get('ms_per_step', 0.0):.2f}"
                    )

    summary: dict[str, dict[str, Any]] = {}
    for row in all_results:
        key = f"{row['config']}:{row['subtask']}_{row['context_length'] // 1024}K"
        bucket = summary.setdefault(
            key,
            {
                "config": row["config"],
                "subtask": row["subtask"],
                "context_length": row["context_length"],
                "scores": [],
                "reference_score": row["reference_score"],
                "reference_source": row["reference_source"],
            },
        )
        bucket["scores"].append(row["score"])

    for bucket in summary.values():
        scores = bucket.pop("scores")
        bucket["mean_score"] = sum(scores) / len(scores) if scores else 0.0
        bucket["n"] = len(scores)
        ref = bucket.get("reference_score")
        bucket["delta_vs_reference"] = (
            bucket["mean_score"] - float(ref) if ref is not None else None
        )

    payload = {
        "benchmark": "arxiv_v1_ruler_smoke",
        "model": "llama3.1-8b",
        "model_dir": str(args.model_dir),
        "configs": configs,
        "contexts": contexts,
        "subtasks": subtasks,
        "samples": args.samples,
        "seed": args.seed,
        "reference_dir": str(args.reference_dir),
        "summary": summary,
        "results": all_results,
    }
    return payload


def require_reference_dir(path: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"reference dir does not exist: {path}")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SuperSonic arxiv_v1 smoke QA")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument(
        "--reference-dir",
        type=require_reference_dir,
        default=Path("/workspace/DotCache/benchmarks/results/arxiv_v1_20260420"),
    )
    parser.add_argument("--reference-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--subtasks", nargs="+", default=["niah_single", "niah_multikey"])
    parser.add_argument("--config", choices=["dense", "certified", "both"], default="both")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--emit-stage-timings", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("target/arxiv_v1_smoke.json"))
    args = parser.parse_args(argv)
    unknown = [name for name in args.subtasks if name not in SUBTASK_BUILDERS]
    if unknown:
        raise SystemExit(f"unknown subtask(s): {', '.join(unknown)}")
    if not args.binary.exists():
        raise SystemExit(f"SuperSonic binary not found: {args.binary}; run cargo build --release --bin supersonic")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    payload = run_ruler_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nJSON -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
