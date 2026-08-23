#!/usr/bin/env python3
"""Check retained Qwen3.8 source boundaries for legacy implementation names.

This is a source-boundary check for the retained MTP path.  It rejects the
legacy MTP field, helper, and environment spellings that must not be added to
the retained runtime again, while leaving unrelated historical words outside
this narrow boundary alone.  The retained full-attention HIP sources are also
checked for product-facing Qwen3.5 comments and stale model-geometry counts;
their qwen35 spellings are ABI/compiler identifiers and are intentionally
outside this product check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


RUNTIME_FILES = (
    Path("crates/runtime/src/decode_engine.rs"),
    Path("crates/runtime/src/prefill_engine.rs"),
    Path("crates/runtime/src/mtp.rs"),
    Path("crates/runtime/src/lib.rs"),
)

KERNEL_FILES = (
    Path("kernels/full_attention.hip"),
    Path("kernels/full_attention_4b.hip"),
)

# Keep these exact spellings for readable diagnostics and compatibility with
# the original boundary check.  The prefix expressions below cover new
# fields/helpers/envs that use a variant of one of these legacy names.
FORBIDDEN_MTP_TERMS = (
    "DFlashFusedVerifyCache",
    "dflash_fused_verify_cache",
    "MetalV2DecodeScratch",
    "metal_v2_scratch",
    # Legacy names this checker must continue to reject.
    "qwen35_mtp_forward",
    "qwen35_mtp_draft_greedy",
)

# Catch future legacy field/helper spellings in any retained runtime module,
# including names that are not themselves Mtp*/mtp_* declarations.  The
# runtime no longer carries the old DFlash/Metal MTP implementation, so these
# prefixes are unambiguous in the retained source boundary.
FORBIDDEN_MTP_IDENTIFIER_RE = re.compile(
    r"\b(?:dflash[A-Za-z0-9_]*|metalv2[A-Za-z0-9_]*|metal_v2[A-Za-z0-9_]*|"
    r"qwen35_[A-Za-z0-9_]*mtp[A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
FORBIDDEN_MTP_ENV_RE = re.compile(
    r"\bSUPERSONIC_(?:DFLASH[A-Z0-9_]*|METALV2[A-Z0-9_]*|METAL_V2[A-Z0-9_]*|"
    r"QWEN35_[A-Z0-9_]*MTP[A-Z0-9_]*)\b",
    re.IGNORECASE,
)
FORBIDDEN_KERNEL_PRODUCT_RE = re.compile(r"qwen\s*3[.]5", re.IGNORECASE)
STALE_KERNEL_GEOMETRY_RE = re.compile(
    r"(?:\b(?!64\b)\d+\s+total\b[^\n]*(?:decoder\s+layer|qwen3[.]8)|"
    r"\bProcesses\s+all\s+(?!64\b)\d+\s+decoder\s+layers\b|"
    r"\bpartial\s+rotary\s+dimension\s*\(\s*(?!64\b)\d+\s+for\s+"
    r"(?:canonical\s+)?qwen3[.]8\b)",
    re.IGNORECASE,
)
REQUIRED_KERNEL_GEOMETRY = {
    Path("kernels/full_attention.hip"): (
        "64 total for canonical Qwen3.8-27B",
        "Processes all 64 decoder layers",
    ),
    Path("kernels/full_attention_4b.hip"): (
        "64 total for canonical Qwen3.8-27B",
        "Processes all 64 decoder layers",
        "partial rotary dimension (64 for canonical Qwen3.8-27B)",
    ),
}


def find_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for relative in RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for term in FORBIDDEN_MTP_TERMS:
                if term in line:
                    violations.append((relative, line_number, term, line.strip()))

            for pattern in (FORBIDDEN_MTP_ENV_RE, FORBIDDEN_MTP_IDENTIFIER_RE):
                match = pattern.search(line)
                if match:
                    violations.append(
                        (relative, line_number, match.group(0), line.strip())
                    )

    for relative in KERNEL_FILES:
        path = root / relative
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        source = "\n".join(lines)
        for line_number, line in enumerate(lines, start=1):
            match = FORBIDDEN_KERNEL_PRODUCT_RE.search(line)
            if match:
                violations.append((relative, line_number, match.group(0), line.strip()))

            match = STALE_KERNEL_GEOMETRY_RE.search(line)
            if match:
                violations.append((relative, line_number, match.group(0), line.strip()))

        for required in REQUIRED_KERNEL_GEOMETRY[relative]:
            if required not in source:
                violations.append(
                    (relative, 0, required, "required canonical kernel geometry marker missing")
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to scan (default: repository containing this tool)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    violations = find_violations(root)
    if violations:
        print("retained Qwen3.8 MTP source-boundary violations:", file=sys.stderr)
        for path, line_number, term, line in violations:
            print(f"  {path}:{line_number}: {term}: {line}", file=sys.stderr)
        return 1
    print("retained Qwen3.8 MTP source-boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
