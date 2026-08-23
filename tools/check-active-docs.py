#!/usr/bin/env python3
"""Validate the active public documentation product boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ACTIVE_DOCS = (
    Path("README.md"),
    Path("docs/build-and-run.md"),
    Path("docs/supported-matrix.md"),
    Path("docs/artifact-format.md"),
    Path("docs/testing.md"),
    Path("docs/benchmarks.md"),
    Path("docs/performance.md"),
)

# These are removed public contract terms.  The list intentionally names
# identities and CLI/env spellings rather than banning ordinary prose such as
# the word "model" or implementation-only source identifiers.
FORBIDDEN_PATTERNS = (
    re.compile(r"--backend\b", re.IGNORECASE),
    re.compile(r"\bSUPERSONIC_BACKENDS?\b", re.IGNORECASE),
    re.compile(r"(?:\b(?:cuda|metal)\b|\b(?:cuda|metal)[_-][a-z0-9_]+)", re.IGNORECASE),
    re.compile(r"\b(?:gemma|phi|llama)(?:\s*[0-9]|[_-][a-z0-9])", re.IGNORECASE),
    re.compile(r"\b(?:gemma|phi|llama)\b", re.IGNORECASE),
    re.compile(r"qwen[-_ ]*3[.]?[56](?![0-9])", re.IGNORECASE),
    re.compile(r"\bDFlash\b|\bSpecPrefill\b|\bCertified[-_ ]?KV\b", re.IGNORECASE),
    re.compile(r"\bKV[-_ ]?FP8\b|\bFP8\b|\bVMM\b|\bMoE\b", re.IGNORECASE),
    re.compile(r"\b(?:Q4KM|Q4_K_M|safetensors|oracle)\b|--gptq\b", re.IGNORECASE),
    re.compile(r"--flm(?:-file)?\b|\bflm[_-]file\b", re.IGNORECASE),
    re.compile(r"--q4km\b|\bq4[_-]k[_-]m\b", re.IGNORECASE),
    re.compile(r"--(?:int4|bf16|fp8|batch-size|force-kernel-decode)\b", re.IGNORECASE),
)
FLM_RE = re.compile(r"\bFLM\b|--flm(?:-file)?\b|\bflm[_-]file\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)")
INTERNAL_FLM_HEADING_RE = re.compile(r"^##\s+Internal FLM foundation\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{1,6}\s+")
TOK_PER_SECOND_RE = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?)\s*tok/s\b", re.IGNORECASE)
COMMIT_EVIDENCE_RE = re.compile(
    r"\b(?:commit|revision)(?:\s+hash)?\s*[:=]?\s*`?([0-9a-f]{7,40})\b",
    re.IGNORECASE,
)
ARTIFACT_EVIDENCE_RE = re.compile(
    r"(?:\b[\w./-]+\.gguf\b|\b(?:artifact|gguf)\s*(?:path|id|identifier)?\s*[:=]\s*[\w./:@-]+)",
    re.IGNORECASE,
)
TARGET_EVIDENCE_RE = re.compile(r"\bgfx(?:1100|1201)\b", re.IGNORECASE)
WORKLOAD_EVIDENCE_RE = re.compile(
    r"(?:\bprompt\s*(?:[:=]\s*|[\"'])[^,\n.;]+"
    r"|\bworkload\s*[:=]\s*(?!documented\b|elsewhere\b)[^,\n.;]+"
    r"|\bcontext(?:[- ]size)?\s*[:=]\s*\d+"
    r"|\bgenerated[- ]tokens?\s*[:=]\s*\d+"
    r"|\binput\s*[:=]\s*(?!documented\b|elsewhere\b)[^,\n.;]+)",
    re.IGNORECASE,
)
MEASUREMENT_EVIDENCE_RE = re.compile(
    r"(?:\bwarmups?(?:\s+count)?\s*[:=]\s*\d+"
    r"|\bmeasured[- ]runs?\s*[:=]\s*\d+"
    r"|\bms_per_tok\s*[:=]\s*\d+(?:\.\d+)?"
    r"|\bmeasurement\s*[:=]\s*(?!documented\b|elsewhere\b)[^,\n.;]+"
    r"|\bmedian\s+(?:decode|prefill)?\s*(?:measurement|latency|ms_per_tok)"
    r"\s*[:=]?\s*\d+)",
    re.IGNORECASE,
)


def slugify_heading(text: str) -> str:
    text = text.strip().lower().replace("`", "")
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def anchors_for(text: str) -> set[str]:
    anchors: set[str] = set()
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                anchors.add(slugify_heading(heading))
    return anchors


def _format_violation(path: Path, line_number: int, term: str, line: str) -> str:
    return f"{path}:{line_number}: {term}: {line.strip()}"


def _heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else None


def find_performance_violations(path: Path, text: str) -> list[str]:
    """Require colocated evidence for every numeric ``tok/s`` claim."""

    violations: list[str] = []
    for match in TOK_PER_SECOND_RE.finditer(text):
        separator = text.rfind("\n\n", 0, match.start())
        paragraph_start = 0 if separator < 0 else separator + 2
        paragraph_end = text.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(text)
        paragraph = text[paragraph_start:paragraph_end]
        missing: list[str] = []
        if not COMMIT_EVIDENCE_RE.search(paragraph):
            missing.append("exact commit hash")
        if not ARTIFACT_EVIDENCE_RE.search(paragraph):
            missing.append("exact GGUF artifact identifier/path")
        if not TARGET_EVIDENCE_RE.search(paragraph):
            missing.append("gfx1100/gfx1201 target")
        if not WORKLOAD_EVIDENCE_RE.search(paragraph):
            missing.append("workload context")
        if not MEASUREMENT_EVIDENCE_RE.search(paragraph):
            missing.append("measurement context")
        if missing:
            line_number = text.count("\n", 0, match.start()) + 1
            violations.append(
                _format_violation(
                    path,
                    line_number,
                    match.group(0),
                    "missing colocated performance evidence: " + ", ".join(missing),
                )
            )
    return violations


def find_text_violations(path: Path, text: str, root: Path | None = None) -> list[str]:
    violations: list[str] = []
    internal_flm_level: int | None = None
    document_anchors = anchors_for(text)
    for line_number, line in enumerate(text.splitlines(), start=1):
        heading_level = _heading_level(line)
        if INTERNAL_FLM_HEADING_RE.match(line.strip()):
            internal_flm_level = heading_level or 2
        elif (
            internal_flm_level is not None
            and heading_level is not None
            and heading_level <= internal_flm_level
        ):
            internal_flm_level = None

        for pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(line)
            if match:
                violations.append(_format_violation(path, line_number, match.group(0), line))

        flm_match = FLM_RE.search(line)
        if flm_match and internal_flm_level is None:
            violations.append(_format_violation(path, line_number, flm_match.group(0), line))

        if root is not None:
            for target in LINK_RE.findall(line):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if target.startswith("#"):
                    if target[1:] not in document_anchors:
                        violations.append(
                            _format_violation(
                                path,
                                line_number,
                                target,
                                f"missing link anchor: {target}",
                            )
                        )
                    continue
                target_name, _, anchor = target.partition("#")
                target_path = (root / path).parent / target_name
                if not target_path.is_file():
                    violations.append(
                        _format_violation(path, line_number, target, f"missing link target: {target}")
                    )
                elif anchor:
                    target_anchors = anchors_for(target_path.read_text(encoding="utf-8"))
                    if anchor not in target_anchors:
                        violations.append(
                            _format_violation(
                                path,
                                line_number,
                                target,
                                f"missing link anchor: {target}",
                            )
                        )
    violations.extend(find_performance_violations(path, text))
    return violations


def find_violations(root: Path) -> list[str]:
    root = root.resolve()
    violations: list[str] = []
    for relative in ACTIVE_DOCS:
        path = root / relative
        if not path.is_file():
            violations.append(f"{relative}: active public document is missing")
            continue
        violations.extend(
            find_text_violations(relative, path.read_text(encoding="utf-8"), root)
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    violations = find_violations(args.root)
    if violations:
        print("active public documentation violations:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("active public documentation check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
