#!/usr/bin/env python3
"""Parse one selected-device ``rocm-smi`` utilization probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class Utilization:
    gpu_use_percent: float
    vram_use_percent: float


_GPU_RE = re.compile(r"GPU\s+use\s*\(%\)\s*:?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_VRAM_RE = re.compile(
    r"GPU\s+Memory\s+Allocated\s*\(VRAM%\)\s*:?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _one(pattern: re.Pattern[str], output: str, label: str) -> float:
    matches = pattern.findall(output)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} value, found {len(matches)}")
    value = float(matches[0])
    if not 0 <= value <= 100:
        raise ValueError(f"{label} value is outside 0..100: {value}")
    return value


def parse_utilization(output: str) -> Utilization:
    """Parse both utilization fields and reject partial/ambiguous probes."""

    return Utilization(
        gpu_use_percent=_one(_GPU_RE, output, "GPU use (%)"),
        vram_use_percent=_one(_VRAM_RE, output, "GPU Memory Allocated (VRAM%)"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="rocm-smi output file (default: stdin)")
    args = parser.parse_args(argv)
    try:
        output = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        utilization = parse_utilization(output)
    except (OSError, ValueError) as exc:
        print(f"rocm-smi utilization parse failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"gpu_use_percent={utilization.gpu_use_percent:g} "
        f"vram_use_percent={utilization.vram_use_percent:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
