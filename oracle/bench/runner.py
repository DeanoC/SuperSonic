"""Subprocess driver for ./target/release/supersonic.

Mirrors the cooldown/warmup/median discipline of the Rust crates/bench
orchestrator. Used by quality runs (perplexity, golden) that need to capture
more than `ms_per_step` from the runner subprocess.
"""
from __future__ import annotations
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

_RESULT_LINE = re.compile(r"^\[result\] .*?ms_per_step=([0-9.]+)", re.MULTILINE)


@dataclass
class SupersonicInvocation:
    binary: Path
    model: str
    model_dir: Path
    quant: str
    prompt: str
    max_new_tokens: int
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RunPolicy:
    measurement_runs: int = 3
    cooldown_seconds: int = 3
    warmup_tokens: int = 2


def run_with_capture(inv: SupersonicInvocation, policy: RunPolicy) -> dict:
    """Run supersonic with cooldown+warmup+median; return a dict ready to
    serialize as a perf cell or quality cell.
    """
    if policy.cooldown_seconds > 0:
        time.sleep(policy.cooldown_seconds)
    _invoke(inv, policy.warmup_tokens, capture=False)

    samples: list[float] = []
    last_err: str | None = None
    for _ in range(policy.measurement_runs):
        try:
            stdout = _invoke(inv, inv.max_new_tokens, capture=True)
            ms = _extract_ms(stdout)
            if ms is not None:
                samples.append(ms)
            else:
                last_err = stdout.splitlines()[-50:]
        except subprocess.CalledProcessError as e:
            last_err = (e.stderr or e.stdout or "")[-2000:]

    if samples:
        sorted_s = sorted(samples)
        median = sorted_s[len(sorted_s) // 2]
        return {"status": "ok", "ms_per_step": median,
                "ms_per_tok": median, "samples": samples}
    return {"status": "error", "stderr_tail": str(last_err)[-2000:]}


def _invoke(inv: SupersonicInvocation, max_new: int, capture: bool) -> str:
    cmd = [str(inv.binary),
           "--model", inv.model,
           "--model-dir", str(inv.model_dir),
           "--prompt", inv.prompt,
           "--max-new-tokens", str(max_new),
           *_quant_flags(inv.quant),
           *inv.extra_args]
    res = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if capture:
        return (res.stdout or "") + "\n" + (res.stderr or "")
    return ""


def _extract_ms(stdout: str) -> float | None:
    m = _RESULT_LINE.search(stdout)
    return float(m.group(1)) if m else None


def _quant_flags(quant: str) -> list[str]:
    return {
        "bf16": [],
        "int4": ["--int4"],
        "fp8r": ["--fp8-runtime"],
        "kv-fp8": ["--kv-fp8"],
        "int8": ["--int8"],
    }.get(quant, [])
