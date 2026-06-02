"""Base class and helpers for external engine benchmark adapters."""
from __future__ import annotations
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class ExternalWorkload:
    """Single-stream generation workload shared by SuperSonic and external engines."""

    prompt: str
    max_new_tokens: int
    context_size: int | None = None
    warmup_runs: int = 1
    measurement_runs: int = 5
    prompt_tokens: int | None = None
    temperature: float = 0.0
    top_k: int = 1
    seed: int = 20260504

    def metadata(self) -> dict:
        return {
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "max_new_tokens": self.max_new_tokens,
            "context_size": self.context_size,
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "seed": self.seed,
        }


class ExternalAdapter(ABC):
    """Common API for benchmarking against an external inference engine."""

    name: str = "unknown"

    @abstractmethod
    def assert_version_match(self) -> None:
        """Raise if the installed engine does not match the pinned version."""

    @abstractmethod
    def supports(self, model: str, quant: str) -> bool:
        """Return True if this adapter can run the given (model, quant)."""

    @abstractmethod
    def measure_speed(self, model: str, quant: str, prompt: str,
                      max_new_tokens: int, model_dir: Path) -> dict:
        """Run the engine and return a dict matching the EXTERNAL_CELL_SCHEMA."""


def read_pinned_version(pin_file: Path) -> str:
    for line in pin_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"{pin_file} has no version line")


def get_engine_version(version_cmd: list[str]) -> str:
    res = subprocess.run(version_cmd, capture_output=True, text=True, check=False)
    return (res.stdout or res.stderr or "").splitlines()[0].strip() if res.stdout or res.stderr else ""


def build_external_cell(
    *,
    engine: str,
    engine_version: str,
    model: str,
    quant: str,
    status: str,
    workload: ExternalWorkload | None = None,
    command: list[str] | None = None,
    ms_samples: list[float] | None = None,
    stderr_tail: str | None = None,
    extras: dict | None = None,
) -> dict:
    samples = sorted(ms_samples) if ms_samples else None
    cell = {
        "schema_version": 1,
        "engine": engine,
        "engine_version": engine_version,
        "model": model,
        "quant": quant,
        "status": status,
        "ms_per_step": median(samples) if samples else None,
        "tok_per_s": (1000.0 / median(samples)) if samples else None,
        "samples": samples,
        "stderr_tail": stderr_tail,
        "command": command,
        "workload": workload.metadata() if workload else None,
        "extras": extras or {},
    }
    return cell


def parse_ms_per_token_samples(text: str) -> list[float]:
    """Extract generation speed samples from common LLM benchmark output."""
    samples: list[float] = []
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|tokens?/s(?:ec)?|t/s)", text, re.I):
        tps = float(match.group(1))
        if tps > 0:
            samples.append(1000.0 / tps)
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*ms/(?:tok|token|step)", text, re.I):
        samples.append(float(match.group(1)))
    for match in re.finditer(r"ms/(?:tok|token|step)[:\s=]+([0-9]+(?:\.[0-9]+)?)", text, re.I):
        samples.append(float(match.group(1)))
    return samples
