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
from bisect import bisect_right
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

# Catch legacy field/helper identifiers in either order, such as
# `MtpDFlashCache`, `DFlashMtpCache`, `MtpMetalV2Scratch`, and
# `mtp_metal_v2_decode_step`.  Matching complete Rust identifiers keeps prose
# comments and historical ABI strings out of this check.  The case-sensitive
# boundary classes still permit CamelCase and snake_case spellings while
# avoiding unrelated words such as `dflashback`.
RUST_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FORBIDDEN_MTP_IDENTIFIER_RE = re.compile(
    r"(?:^|_|[a-z0-9])(?i:dflash|metalv2|metal_v2|qwen35_mtp)"
    r"(?=$|_|[A-Z0-9])"
)
FORBIDDEN_MTP_ENV_RE = re.compile(
    r"\bSUPERSONIC_(?i:DFLASH[A-Z0-9_]*|METALV2[A-Z0-9_]*|"
    r"METAL_V2[A-Z0-9_]*|QWEN35_[A-Z0-9_]*MTP[A-Z0-9_]*)\b"
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


def _mask_rust_comments_and_strings(source: str) -> tuple[str, list[tuple[int, str]]]:
    """Mask comments/string syntax while retaining string literals separately."""

    masked = list(source)
    literals: list[tuple[int, str]] = []

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if source[index] != "\n":
                masked[index] = " "

    def raw_string_bounds(start: int) -> tuple[int, int, int] | None:
        if source[start] != "r":
            return None
        index = start + 1
        while index < len(source) and source[index] == "#":
            index += 1
        if index >= len(source) or source[index] != '"':
            return None
        hashes = index - start - 1
        content_start = index + 1
        closing = '"' + ("#" * hashes)
        content_end = source.find(closing, content_start)
        if content_end < 0:
            return content_start, len(source), len(source)
        return content_start, content_end, content_end + len(closing)

    def quoted_string_end(start: int) -> int:
        index = start + 1
        escaped = False
        while index < len(source):
            char = source[index]
            if char == "\n":
                break
            if char == '"' and not escaped:
                return index + 1
            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
            index += 1
        return index

    index = 0
    while index < len(source):
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = len(source) if end < 0 else end
            blank(index, end)
            index = end
            continue

        if source.startswith("/*", index):
            start = index
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            blank(start, index)
            continue

        raw_bounds = raw_string_bounds(index)
        if raw_bounds is not None:
            content_start, content_end, end = raw_bounds
            literals.append((index, source[content_start:content_end]))
            blank(index, end)
            index = end
            continue

        if source[index] == '"':
            end = quoted_string_end(index)
            literals.append((index, source[index + 1 : max(index + 1, end - 1)]))
            blank(index, end)
            index = end
            continue

        index += 1

    return "".join(masked), literals


def _line_starts(source: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(source) if char == "\n")
    return starts


def _line_number(starts: list[int], offset: int) -> int:
    return bisect_right(starts, offset)


def _legacy_mtp_violations(
    lines: list[str], relative: Path
) -> list[tuple[Path, int, str, str]]:
    source = "\n".join(lines)
    masked, literals = _mask_rust_comments_and_strings(source)
    starts = _line_starts(source)
    violations: list[tuple[Path, int, str, str]] = []

    for match in RUST_IDENTIFIER_RE.finditer(masked):
        token = match.group(0)
        if FORBIDDEN_MTP_IDENTIFIER_RE.search(token):
            line_number = _line_number(starts, match.start())
            violations.append((relative, line_number, token, lines[line_number - 1].strip()))

    for offset, literal in literals:
        match = FORBIDDEN_MTP_ENV_RE.search(literal)
        if match:
            line_number = _line_number(starts, offset)
            violations.append((relative, line_number, match.group(0), lines[line_number - 1].strip()))

    return violations


def find_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for relative in RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        violations.extend(_legacy_mtp_violations(lines, relative))

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
