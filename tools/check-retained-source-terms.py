#!/usr/bin/env python3
"""Check retained Qwen3.8 source boundaries for legacy implementation names.

This is deliberately a source-boundary check rather than a blanket ban on
historical words in the runtime.  Task 5 leaves the outer tree/tap/rollback
experiments in place for the later kernel reduction, so those names are not
part of this check.  The symbols below are the shared MTP cache/scratch and
the Qwen3.8 MTP entry points that are part of the retained product path.  The
retained full-attention HIP sources are also checked for product-facing
Qwen3.5 comments and stale model-geometry counts; their qwen35 spellings are
ABI/compiler identifiers and are intentionally outside this product check.
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

# Keep this list specific.  Broadly rejecting `dflash` would also reject the
# outer tree/tap/rollback implementation that Task 5 explicitly leaves in
# place until the later kernel reduction task.
FORBIDDEN_MTP_TERMS = (
    "DFlashFusedVerifyCache",
    "dflash_fused_verify_cache",
    "MetalV2DecodeScratch",
    "metal_v2_scratch",
    # Legacy names this checker must continue to reject.
    "qwen35_mtp_forward",
    "qwen35_mtp_draft_greedy",
)

# The exact names above catch the old shared symbols.  This second check is
# scoped to MTP declarations so a newly introduced `Mtp*DFlash` or
# `mtp_*_metal_v2` identifier cannot slip through while the outer tree/tap
# implementation remains in the same runtime files for later tasks.
FORBIDDEN_MTP_DECL_RE = re.compile(r"(?:dflash|metalv2|metal_v2|qwen35_mtp)", re.IGNORECASE)
MTP_DECL_RE = re.compile(
    r"\b(?:struct|enum|type|impl|fn)\s+"
    r"(?:Mtp[A-Za-z0-9_]*|mtp_[A-Za-z0-9_]*)\b"
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

        # `mtp.rs` is a dedicated retained module, so every legacy spelling
        # there is a boundary violation.  In the larger runtime files, walk
        # only declarations whose own name starts with Mtp/mtp_; this avoids
        # flagging the legacy tree/tap code and its preserved FFI calls.
        if relative == Path("crates/runtime/src/mtp.rs"):
            for line_number, line in enumerate(lines, start=1):
                match = FORBIDDEN_MTP_DECL_RE.search(line)
                if match:
                    violations.append(
                        (relative, line_number, match.group(0), line.strip())
                    )
            continue

        active_decl: str | None = None
        brace_depth = 0
        saw_open_brace = False
        for line_number, line in enumerate(lines, start=1):
            if active_decl is None and MTP_DECL_RE.search(line):
                active_decl = line.strip()
                brace_depth = 0
                saw_open_brace = False

            if active_decl is None:
                continue

            match = FORBIDDEN_MTP_DECL_RE.search(line)
            if match:
                violations.append(
                    (relative, line_number, match.group(0), line.strip())
                )

            # Rust declarations in the retained modules all use a braced
            # body.  Counting braces is intentionally lightweight here: the
            # check is a naming guard, not a Rust parser, and declarations
            # with an unusual body simply remain active through EOF.
            brace_depth += line.count("{") - line.count("}")
            saw_open_brace = saw_open_brace or "{" in line
            if saw_open_brace and brace_depth <= 0:
                active_decl = None

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
