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
    re.compile(r"\b(?:gemma|phi|llama)(?!-(?:cli|cpp)\b)(?:\s*[0-9]|[_-][a-z0-9])", re.IGNORECASE),
    re.compile(r"\b(?:gemma|phi|llama)(?!-(?:cli|cpp)\b)\b", re.IGNORECASE),
    re.compile(r"qwen[-_ ]*3[.]?[56](?![0-9])", re.IGNORECASE),
    re.compile(r"\bDFlash\b|\bSpecPrefill\b|\bCertified[-_ ]?KV\b", re.IGNORECASE),
    re.compile(r"\bKV[-_ ]?FP8\b|\bFP8\b|\bVMM\b|\bMoE\b", re.IGNORECASE),
    re.compile(r"\b(?:Q4KM|Q4_K_M|safetensors|oracle)\b|--gptq\b", re.IGNORECASE),
    re.compile(r"--flm(?:-file)?\b|\bflm[_-]file\b", re.IGNORECASE),
    re.compile(r"--q4km\b|\bq4[_-]k[_-]m\b", re.IGNORECASE),
    re.compile(r"--(?:int4|bf16|fp8|batch-size|force-kernel-decode)\b", re.IGNORECASE),
    # Stream GEMV no longer retains a launch between public calls. Keep this
    # lifecycle wording out of active docs while allowing ordinary GEMV prose.
    re.compile(r"\bheld\s+GEMV\b|\blearned\s+(?:GEMV|gate/up)\b", re.IGNORECASE),
)
FLM_RE = re.compile(r"\bFLM\b|--flm(?:-file)?\b|\bflm[_-]file\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)")
INTERNAL_FLM_HEADING_RE = re.compile(r"^##\s+Internal FLM foundation\s*$", re.IGNORECASE)
TOK_PER_SECOND_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?)\s*(?:tok(?:ens?)?\s*/\s*s|t(?:ok(?:ens?)?)?\s+per\s+second)\b",
    re.IGNORECASE,
)
PERFORMANCE_METRIC_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?)\s*(?:"
    r"t(?:ok(?:ens?)?)?\s*(?:/|per)\s*(?:second|sec|s)|"
    r"tps|"
    r"milliseconds?\s*(?:per|/)\s*tokens?|"
    r"ms\s*(?:_per_tok|/\s*(?:tok(?:en)?s?)|per\s+tokens?)"
    r")\b",
    re.IGNORECASE,
)
SPEEDUP_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?)\s*[x×]\s*(?:speed[- ]?up|faster)\b"
    r"|\bspeed[- ]?up\s*(?:is\s*)?[:=]?\s*(?:\d+(?:\.\d+)?)\s*[x×]\b"
    r"|(?<![\w.])(?:\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)\s+faster\b",
    re.IGNORECASE,
)
CLAIM_RE = re.compile(
    rf"(?:{PERFORMANCE_METRIC_RE.pattern})|(?:{SPEEDUP_RE.pattern})",
    re.IGNORECASE,
)
COMMIT_EVIDENCE_RE = re.compile(
    r"\b(?:commit|revision)(?:\s+hash)?\s*[:=]?\s*`?([0-9a-f]{7,64})\b",
    re.IGNORECASE,
)
ARTIFACT_PATH_RE = re.compile(r"(?<![\w<$])[\w./~:-]+\.gguf\b(?!>)", re.IGNORECASE)
ARTIFACT_FIELD_RE = re.compile(
    r"\b(?:artifact|gguf)\s*(?:path|id|identifier)?\s*[:=]\s*"
    r"(?P<value>`[^`]+`|\"[^\"]+\"|'[^']+'|[^\s,;]+)",
    re.IGNORECASE,
)
ARTIFACT_PLACEHOLDER_RE = re.compile(
    r"(?:<[^>]+>|\$\{?[A-Z_][A-Z0-9_]*\}?|(?:^|[\s/_-])"
    r"(?:path|to|your|example|placeholder|documented|elsewhere|unknown|tbd)"
    r"(?:[\s/_.:-]|$))",
    re.IGNORECASE,
)
TARGET_EVIDENCE_RE = re.compile(r"\bgfx(?:1100|1201)\b", re.IGNORECASE)
WORKLOAD_EVIDENCE_RE = re.compile(
    r"(?:\bprompt\s*(?:[:=]\s*|[\"'])"
    r"(?!documented\b|elsewhere\b|<|\$\{?)[^,\n.;]+"
    r"|\bworkload\s*[:=]\s*(?!documented\b|elsewhere\b)[^,\n.;]+"
    r"|\bcontext(?:[- ]size)?\s*[:=]\s*\d+"
    r"|\b(?:generated[- ]tokens?|token[- ]count|max[- ]new[- ]tokens?)\s*[:=]\s*\d+"
    r"|\bbatch(?:[- ]size)?\s*[:=]\s*\d+"
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
ENGINE_EVIDENCE_RE = re.compile(r"\bengine\s*[:=]\s*[^\s,;]+", re.IGNORECASE)
VERSION_EVIDENCE_RE = re.compile(
    r"\b(?:engine[-_ ]?)?version\s*[:=]\s*[^\s,;]+", re.IGNORECASE
)
ENGINE_VERSION_EVIDENCE_RE = re.compile(
    r"\bengine\s*/\s*version\s*[:=]\s*[^\n,;]+", re.IGNORECASE
)
CLOCK_EVIDENCE_RE = re.compile(
    r"(?:\b(?:clock|clocks)(?:[-_ ]policy|[-_ ]verification|[-_ ]verified)?\s*[:=]\s*"
    r"(?:locked|verified|pass|yes|true)\b|\b(?:verified[-_ ]?clock|clock[-_ ]?verified)\s*[:=]\s*"
    r"(?:locked|verified|pass|yes|true)\b|\bclocks?\s+(?:are\s+)?verified\b)",
    re.IGNORECASE,
)
CACHE_STATE_EVIDENCE_RE = re.compile(
    r"\b(?:cache(?:[-_ ]state)?|cache_state)\s*[:=]\s*"
    r"(?:cold-load|warm-resident|prefix-cache-(?:empty|populated|reset))\b",
    re.IGNORECASE,
)
PROCESS_STATE_EVIDENCE_RE = re.compile(
    r"\b(?:process(?:[-_ ]state)?|process_reuse)\s*[:=]\s*"
    r"(?:fresh-process|false|true)\b",
    re.IGNORECASE,
)
STATISTIC_EVIDENCE_RE = re.compile(
    r"\b(?:statistic|summary\s+statistic|metric)\s*[:=]\s*"
    r"(?:median|mean|minimum|maximum|mad|p\d+(?:\.\d+)?)\b"
    r"|\bmedian\b",
    re.IGNORECASE,
)
SAMPLE_COUNT_EVIDENCE_RE = re.compile(
    r"\b(?:sample(?:s)?\s*(?:count|number)|samples?|sample_count|"
    r"measured[-_ ]runs?(?:[-_ ]count)?|n)\s*[:=]\s*\d+\b",
    re.IGNORECASE,
)
CORRECTNESS_EVIDENCE_RE = re.compile(
    r"(?:\b(?:correctness|quality|token(?:[-_ ]sequence)?[-_ ]?equality|"
    r"mtp(?:[-_ /]+)?equality)\s*[:=]\s*"
    r"(?:pass(?:ed)?|ok|true|yes|verified|equal(?:ity)?|match(?:ed)?)\b"
    r"|\b(?:correctness|quality)\s+(?:pass(?:ed)?|verified|matches?)\b)",
    re.IGNORECASE,
)
DIRECT_RUN_EVIDENCE_RE = re.compile(
    r"(?:\b(?:direct[-_ ]run|run\s+(?:id|record|manifest|evidence|command)|"
    r"run[-_]?(?:id|record|manifest|evidence|command)|command)\s*[:=]\s*"
    r"(?:`[^`]+`|[^\s,;]+)|\bbenchmarks/results/[^\s)]+\.json\b|"
    r"\btarget/benchmarks/[^\s)]+)",
    re.IGNORECASE,
)
DIRECT_RUN_PLACEHOLDER_RE = re.compile(
    r"(?:<[^>]+>|\$\{?[A-Z_][A-Z0-9_]*\}?|\b(?:documented|elsewhere|unknown|tbd|placeholder)\b)",
    re.IGNORECASE,
)


def slugify_heading(text: str) -> str:
    text = text.strip().lower().replace("`", "")
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def _heading_texts(text: str) -> list[str]:
    lines = text.splitlines()
    headings: list[str] = []
    active_fence: tuple[str, int] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if active_fence is not None:
            if (
                fence
                and fence.group(1)[0] == active_fence[0]
                and len(fence.group(1)) >= active_fence[1]
                and not fence.group(2).strip()
            ):
                active_fence = None
            index += 1
            continue
        if fence:
            active_fence = (fence.group(1)[0], len(fence.group(1)))
            index += 1
            continue
        atx = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)$", line)
        if atx:
            heading = atx.group(2) or ""
            heading = re.sub(r"[ \t]+#+[ \t]*$", "", heading).strip()
            if heading:
                headings.append(heading)
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and line.strip()
            and len(line) - len(line.lstrip(" ")) <= 3
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[index + 1])
        ):
            headings.append(line.strip())
            index += 2
            continue
        index += 1
    return headings


def anchors_for(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in _heading_texts(text):
        slug = slugify_heading(heading)
        if not slug:
            continue
        suffix = counts.get(slug, 0)
        candidate = slug if suffix == 0 else f"{slug}-{suffix}"
        while candidate in anchors:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        counts[slug] = suffix + 1
        anchors.add(candidate)
    return anchors


def _format_violation(path: Path, line_number: int, term: str, line: str) -> str:
    return f"{path}:{line_number}: {term}: {line.strip()}"


def _heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else None


def find_performance_violations(path: Path, text: str) -> list[str]:
    """Require colocated evidence for numeric performance and peer claims.

    Ordinary prose is intentionally outside this check.  A paragraph is
    inspected only when it contains a numeric metric (for example ``tok/s``)
    or an explicit numeric speedup/faster claim.  Each claim receives its own
    evidence slice so one run record cannot accidentally qualify a second
    number in the same paragraph.
    """

    violations: list[str] = []
    def artifact_evidence(value: str) -> bool:
        candidates = [match.group(0) for match in ARTIFACT_PATH_RE.finditer(value)]
        candidates.extend(match.group("value") for match in ARTIFACT_FIELD_RE.finditer(value))
        for candidate in candidates:
            candidate = candidate.strip("`\"'").lower()
            if not candidate or ARTIFACT_PLACEHOLDER_RE.search(candidate):
                continue
            if candidate in {"artifact", "gguf", "path", "identifier"}:
                continue
            return True
        return False

    def evidence_for_claim(paragraph: str, claim_index: int, claims: list[re.Match[str]]) -> str:
        if len(claims) == 1:
            return paragraph
        start = 0 if claim_index == 0 else claims[claim_index - 1].end()
        end = claims[claim_index + 1].start() if claim_index + 1 < len(claims) else len(paragraph)
        return paragraph[start:end]

    def has_direct_run_evidence(value: str) -> bool:
        for candidate in DIRECT_RUN_EVIDENCE_RE.finditer(value):
            if not DIRECT_RUN_PLACEHOLDER_RE.search(candidate.group(0)):
                return True
        return False

    for match in CLAIM_RE.finditer(text):
        separator = text.rfind("\n\n", 0, match.start())
        paragraph_start = 0 if separator < 0 else separator + 2
        paragraph_end = text.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(text)
        paragraph = text[paragraph_start:paragraph_end]
        relative_start = match.start() - paragraph_start
        claims = list(CLAIM_RE.finditer(paragraph))
        claim_index = next(
            index for index, claim in enumerate(claims) if claim.start() == relative_start
        )
        evidence = evidence_for_claim(paragraph, claim_index, claims)
        missing: list[str] = []
        if not COMMIT_EVIDENCE_RE.search(evidence):
            missing.append("exact commit hash")
        if not artifact_evidence(evidence):
            missing.append("exact GGUF artifact identifier/path")
        if not TARGET_EVIDENCE_RE.search(evidence):
            missing.append("gfx1100/gfx1201 target")
        if not WORKLOAD_EVIDENCE_RE.search(evidence):
            missing.append("workload context")
        if not (MEASUREMENT_EVIDENCE_RE.search(evidence) or STATISTIC_EVIDENCE_RE.search(evidence)):
            missing.append("measurement context")
        if not (
            ENGINE_VERSION_EVIDENCE_RE.search(evidence)
            or (ENGINE_EVIDENCE_RE.search(evidence) and VERSION_EVIDENCE_RE.search(evidence))
        ):
            missing.append("engine/version evidence")
        if not CLOCK_EVIDENCE_RE.search(evidence):
            missing.append("verified clock policy")
        if not (
            CACHE_STATE_EVIDENCE_RE.search(evidence)
            and PROCESS_STATE_EVIDENCE_RE.search(evidence)
        ):
            missing.append("cache/process state")
        if not STATISTIC_EVIDENCE_RE.search(evidence) or not SAMPLE_COUNT_EVIDENCE_RE.search(evidence):
            missing.append("statistic and sample count")
        if not CORRECTNESS_EVIDENCE_RE.search(evidence):
            missing.append("correctness result")
        if not has_direct_run_evidence(evidence):
            missing.append("direct run evidence")
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
                    if slugify_heading(target[1:]) not in document_anchors:
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
                    if slugify_heading(anchor) not in target_anchors:
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
