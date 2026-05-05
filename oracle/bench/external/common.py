"""Base class for external engine adapters (hipfire today; llama.cpp/vLLM later)."""
from __future__ import annotations
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


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
