#!/usr/bin/env python3
"""
Run vLLM's Qwen3.6/Qwen3.5 MTP speculative path and compare draft-token
proposals against fixtures produced by `qwen36_moe_mtp_oracle.py --mode prefill`.

This is intentionally a big-box workflow. It expects a vLLM ROCm/CUDA
environment that can load Qwen3.6-35B-A3B-FP8 plus the MTP drafter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _install_vllm_draft_trace() -> None:
    """Monkeypatch vLLM's proposer in every spawned worker process.

    vLLM uses Python multiprocessing on ROCm. Top-level imports are re-run in
    spawned workers, so installing this patch before `main()` lets the worker
    append actual draft-token rows to a JSONL log.
    """
    log_path = os.environ.get("SUPERSONIC_VLLM_DRAFT_LOG")
    if not log_path:
        return

    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    if getattr(SpecDecodeBaseProposer.propose, "_supersonic_traced", False):
        return
    orig_propose = SpecDecodeBaseProposer.propose

    def traced_propose(self, *args, **kwargs):
        out = orig_propose(self, *args, **kwargs)
        try:
            rows = out.detach().cpu().tolist()
        except Exception as ex:
            rows = {"error": repr(ex)}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "draft_token_ids": rows}) + "\n")
        print("VLLM_DRAFT_TOKENS", json.dumps(rows), flush=True)
        return out

    traced_propose._supersonic_traced = True
    SpecDecodeBaseProposer.propose = traced_propose


_install_vllm_draft_trace()


DEFAULT_PROMPTS = [
    "The quick brown fox jumps over",
    "def factorial(n):",
    "Once upon a time",
]

DEFAULT_FIXTURES = [
    "mtp_real_prefill_quick_brown_fox.json",
    "mtp_real_prefill_factorial.json",
    "mtp_real_prefill_once_upon.json",
]


def _load_expected(fixture_dir: Path) -> list[dict]:
    expected = []
    for name in DEFAULT_FIXTURES:
        raw = json.loads((fixture_dir / name).read_text())
        expected.append(
            {
                "fixture": name,
                "prompt": raw["prompt"],
                "draft_token_ids": raw["draft_token_ids"],
            }
        )
    return expected


def _read_draft_rows(log_path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        draft = item["draft_token_ids"]
        if isinstance(draft, list):
            rows.extend(draft)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--fixture-dir", type=Path, default=Path("tests/fixtures/qwen36_moe"))
    p.add_argument("--out", type=Path, default=Path("/tmp/qwen36_mtp_vllm_reference.json"))
    p.add_argument("--draft-log", type=Path, default=Path("/tmp/qwen36_mtp_vllm_drafts.jsonl"))
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.draft_log.unlink(missing_ok=True)
    os.environ["SUPERSONIC_VLLM_DRAFT_LOG"] = str(args.draft_log)
    _install_vllm_draft_trace()

    expected = _load_expected(args.fixture_dir)
    prompts = [x["prompt"] for x in expected]
    if prompts != DEFAULT_PROMPTS:
        raise SystemExit(f"unexpected fixture prompts: {prompts!r}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_dir),
        speculative_config={"method": "qwen3_next_mtp", "num_speculative_tokens": 3},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=args.max_new_tokens),
    )

    generated = []
    for prompt, out in zip(prompts, outputs):
        item = {
            "prompt": prompt,
            "token_ids": list(out.outputs[0].token_ids),
            "text": out.outputs[0].text,
        }
        generated.append(item)
        print("VLLM_OUTPUT", json.dumps(item), flush=True)

    rows = _read_draft_rows(args.draft_log)
    matches = []
    for item in expected:
        draft = item["draft_token_ids"]
        matches.append({**item, "matched": draft in rows})

    result = {
        "vllm_reference": {
            "model_dir": str(args.model_dir),
            "speculative_config": {
                "method": "qwen3_next_mtp",
                "num_speculative_tokens": 3,
            },
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
        },
        "expected": expected,
        "draft_rows": rows,
        "matches": matches,
        "outputs": generated,
    }
    args.out.write_text(json.dumps(result, indent=2))

    missing = [m for m in matches if not m["matched"]]
    if missing:
        raise SystemExit(f"vLLM draft mismatch: {missing}")
    print(f"[vllm-mtp] all {len(matches)} fixture draft rows matched")


if __name__ == "__main__":
    main()
