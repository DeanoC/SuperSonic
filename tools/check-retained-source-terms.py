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


RUNTIME_ROOT = Path("crates/runtime/src")

KERNEL_FILES = (
    Path("kernels/full_attention.hip"),
    Path("kernels/full_attention_4b.hip"),
)

# Keep these exact spellings for readable diagnostics and compatibility with
# the original boundary check.  The lexer below also catches their CamelCase,
# snake_case, and mixed-order variants.
FORBIDDEN_MTP_TERMS = (
    "DFlashFusedVerifyCache",
    "dflash_fused_verify_cache",
    "MetalV2DecodeScratch",
    "metal_v2_scratch",
    # Legacy names this checker must continue to reject.
    "qwen35_mtp_forward",
    "qwen35_mtp_draft_greedy",
)

# Rust environment controls are conventionally uppercase, but the boundary
# check is deliberately case-insensitive so a lowercase or escaped spelling
# cannot evade it.  These are inspected only in decoded Rust string literals;
# historical ABI symbols outside `crates/runtime` are therefore not in scope.
FORBIDDEN_MTP_ENV_RE = re.compile(
    r"\bSUPERSONIC_(?:DFLASH[A-Z0-9_]*|METALV2[A-Z0-9_]*|"
    r"METAL_V2[A-Z0-9_]*|QWEN35_?[A-Z0-9_]*MTP[A-Z0-9_]*)\b",
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


class _RustLexeme:
    __slots__ = ("kind", "value", "offset")

    def __init__(self, kind: str, value: str, offset: int) -> None:
        self.kind = kind
        self.value = value
        self.offset = offset


_SIMPLE_RUST_ESCAPES = {
    "0": "\0",
    "\\": "\\",
    "\"": '"',
    "'": "'",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _is_identifier_continue(char: str) -> bool:
    return _is_identifier_start(char) or char.isdigit()


def _consume_identifier(source: str, start: int) -> tuple[str, int]:
    if source.startswith("r#", start) and start + 2 < len(source):
        if _is_identifier_start(source[start + 2]):
            index = start + 3
            while index < len(source) and _is_identifier_continue(source[index]):
                index += 1
            return source[start + 2 : index], index

    index = start + 1
    while index < len(source) and _is_identifier_continue(source[index]):
        index += 1
    return source[start:index], index


def _raw_string_opener(source: str, start: int) -> tuple[int, int] | None:
    for prefix in ("br", "cr", "r"):
        if not source.startswith(prefix, start):
            continue
        index = start + len(prefix)
        while index < len(source) and source[index] == "#":
            index += 1
        if index < len(source) and source[index] == '"':
            return index, index - start - len(prefix)
    return None


def _normal_string_quote(source: str, start: int) -> int | None:
    if source[start] == '"':
        return start
    if source[start] in "bc" and start + 1 < len(source) and source[start + 1] == '"':
        return start + 1
    return None


def _decode_rust_escape(source: str, slash: int) -> tuple[str, int]:
    index = slash + 1
    if index >= len(source):
        return "\\", index

    escaped = source[index]
    if escaped in _SIMPLE_RUST_ESCAPES:
        return _SIMPLE_RUST_ESCAPES[escaped], index + 1

    if escaped == "\n":
        index += 1
        while index < len(source) and source[index] in " \t":
            index += 1
        return "", index
    if escaped == "\r":
        index += 1
        if index < len(source) and source[index] == "\n":
            index += 1
        while index < len(source) and source[index] in " \t":
            index += 1
        return "", index

    if escaped == "x":
        digits = source[index + 1 : index + 3]
        if len(digits) == 2 and all(char in _HEX_DIGITS for char in digits):
            return chr(int(digits, 16)), index + 3
        return "x", index + 1

    if escaped == "u" and index + 1 < len(source) and source[index + 1] == "{":
        close = source.find("}", index + 2)
        if close >= 0:
            digits = source[index + 2 : close].replace("_", "")
            if 1 <= len(digits) <= 6 and all(char in _HEX_DIGITS for char in digits):
                codepoint = int(digits, 16)
                if codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF:
                    return chr(codepoint), close + 1
        return "u", index + 1

    # Invalid Rust escapes are retained conservatively as their escaped
    # character.  This still catches a legacy control written as `\\D...` in
    # a fixture instead of silently discarding the suspicious character.
    return escaped, index + 1


def _parse_normal_string(source: str, quote: int) -> tuple[str, int]:
    value: list[str] = []
    index = quote + 1
    while index < len(source):
        char = source[index]
        if char == '"':
            return "".join(value), index + 1
        if char == "\\":
            decoded, index = _decode_rust_escape(source, index)
            value.append(decoded)
            continue
        if char in "\r\n":
            return "".join(value), index
        value.append(char)
        index += 1
    return "".join(value), index


def _parse_raw_string(source: str, quote: int, hashes: int) -> tuple[str, int]:
    content_start = quote + 1
    closing = '"' + ("#" * hashes)
    content_end = source.find(closing, content_start)
    if content_end < 0:
        return source[content_start:], len(source)
    return source[content_start:content_end], content_end + len(closing)


def _skip_rust_space_and_comments(source: str, start: int) -> int:
    index = start
    while index < len(source):
        if source[index].isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
            continue
        if source.startswith("/*", index):
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
            continue
        break
    return index


_RUST_GROUP_CLOSERS = {"(": ")", "[": "]", "{": "}"}
_RUST_GROUP_OPENERS = frozenset(_RUST_GROUP_CLOSERS)


def _matching_rust_group_end(source: str, opener: int) -> int | None:
    """Return the end of a balanced Rust delimiter group, if it is valid."""

    first_close = _RUST_GROUP_CLOSERS.get(source[opener])
    if first_close is None:
        return None

    expected_closers = [first_close]
    index = opener + 1
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
            continue

        if source.startswith("/*", index):
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
            if depth:
                return None
            continue

        raw_opener = _raw_string_opener(source, index)
        if raw_opener is not None:
            quote, hashes = raw_opener
            closing = '"' + ("#" * hashes)
            content_end = source.find(closing, quote + 1)
            if content_end < 0:
                return None
            index = content_end + len(closing)
            continue

        quote = _normal_string_quote(source, index)
        if quote is not None:
            _, end = _parse_normal_string(source, quote)
            if end <= quote or end > len(source) or source[end - 1] != '"':
                return None
            index = end
            continue

        if source[index] == "'":
            char_end = _char_literal_end(source, index)
            if char_end is not None:
                index = char_end
                continue

        char = source[index]
        if char in _RUST_GROUP_OPENERS:
            expected_closers.append(_RUST_GROUP_CLOSERS[char])
            index += 1
            continue
        if char in _RUST_GROUP_CLOSERS.values():
            if char != expected_closers[-1]:
                return None
            expected_closers.pop()
            index += 1
            if not expected_closers:
                return index
            continue
        index += 1

    return None


def _parse_concat_macro(source: str, name_end: int) -> tuple[str, int] | None:
    index = _skip_rust_space_and_comments(source, name_end)
    if index >= len(source) or source[index] != "!":
        return None
    index = _skip_rust_space_and_comments(source, index + 1)
    if index >= len(source) or source[index] not in _RUST_GROUP_OPENERS:
        return None

    group_end = _matching_rust_group_end(source, index)
    if group_end is None:
        return None
    body_end = group_end - 1
    index += 1
    values: list[str] = []
    while index < body_end:
        index = _skip_rust_space_and_comments(source, index)
        if index >= body_end:
            break
        if source[index] in _RUST_GROUP_OPENERS:
            return None

        raw_opener = _raw_string_opener(source, index)
        if raw_opener is not None:
            quote, hashes = raw_opener
            value, index = _parse_raw_string(source, quote, hashes)
        else:
            quote = _normal_string_quote(source, index)
            if quote is None:
                return None
            value, index = _parse_normal_string(source, quote)
        if index > body_end:
            return None
        values.append(value)

        index = _skip_rust_space_and_comments(source, index)
        if index == body_end:
            break
        if source[index] == ",":
            index += 1
            continue
        return None

    return "".join(values), group_end


def _char_literal_end(source: str, start: int) -> int | None:
    index = start + 1
    if index >= len(source) or source[index] in "\r\n'":
        return None
    if source[index] == "\\":
        _, index = _decode_rust_escape(source, index)
    else:
        index += 1
    if index < len(source) and source[index] == "'":
        return index + 1
    return None


def _lex_rust(source: str) -> list[_RustLexeme]:
    """Lex enough Rust syntax to separate identifiers from comments/literals."""

    lexemes: list[_RustLexeme] = []
    index = 0
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
            continue

        if source.startswith("/*", index):
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
            continue

        raw_opener = _raw_string_opener(source, index)
        if raw_opener is not None:
            quote, hashes = raw_opener
            value, index = _parse_raw_string(source, quote, hashes)
            lexemes.append(_RustLexeme("string", value, quote))
            continue

        quote = _normal_string_quote(source, index)
        if quote is not None:
            value, index = _parse_normal_string(source, quote)
            lexemes.append(_RustLexeme("string", value, quote))
            continue

        if source[index] == "'":
            end = _char_literal_end(source, index)
            if end is not None:
                index = end
            elif index + 1 < len(source) and _is_identifier_start(source[index + 1]):
                _, index = _consume_identifier(source, index + 1)
            else:
                index += 1
            continue

        if source.startswith("r#", index) and index + 2 < len(source):
            if _is_identifier_start(source[index + 2]):
                token_start = index
                value, index = _consume_identifier(source, index)
                if value == "concat":
                    parsed = _parse_concat_macro(source, index)
                    if parsed is not None:
                        value, index = parsed
                        lexemes.append(_RustLexeme("string", value, token_start))
                        continue
                lexemes.append(_RustLexeme("identifier", value, token_start))
                continue

        if _is_identifier_start(source[index]):
            token_start = index
            value, index = _consume_identifier(source, index)
            if value == "concat":
                parsed = _parse_concat_macro(source, index)
                if parsed is not None:
                    value, index = parsed
                    lexemes.append(_RustLexeme("string", value, token_start))
                    continue
            lexemes.append(_RustLexeme("identifier", value, token_start))
            continue

        index += 1

    return lexemes


def _line_starts(source: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(source) if char == "\n")
    return starts


def _line_number(starts: list[int], offset: int) -> int:
    return bisect_right(starts, offset)


def _identifier_parts(identifier: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for index, char in enumerate(identifier):
        if char == "_":
            if start < index:
                parts.append(identifier[start:index].lower())
            start = index + 1
            continue
        if index <= start or not char.isupper():
            continue
        previous = identifier[index - 1]
        next_char = identifier[index + 1] if index + 1 < len(identifier) else ""
        acronym_start = previous.isupper() and next_char.islower()
        uppercase_run = index - start
        if previous.islower() or previous.isdigit() or (acronym_start and uppercase_run > 1):
            parts.append(identifier[start:index].lower())
            start = index
    if start < len(identifier):
        parts.append(identifier[start:].lower())
    return parts


def _has_compound(parts: list[str], compound: tuple[str, ...]) -> bool:
    width = len(compound)
    return any(tuple(parts[index : index + width]) == compound for index in range(len(parts)))


def _legacy_identifier_reason(identifier: str) -> str | None:
    parts = _identifier_parts(identifier)
    normalized = "".join(parts)
    has_qwen35 = "qwen35" in normalized
    has_mtp = "mtp" in normalized
    if has_qwen35 and has_mtp:
        return "qwen35/mtp legacy identifier"
    if "dflash" in normalized and has_mtp:
        return "dflash/mtp legacy identifier"
    if "metalv2" in normalized and has_mtp:
        return "metalv2/mtp legacy identifier"

    if "dflash" in parts or any(
        part.startswith("dflash") and part[6:7].isdigit() for part in parts
    ):
        return "standalone dflash legacy identifier"
    if _has_compound(parts, ("metal", "v2")) or any(
        part.startswith("metalv2") for part in parts
    ):
        return "standalone metalv2 legacy identifier"
    if _has_compound(parts, ("spec", "prefill")) or any(
        part.startswith("specprefill") for part in parts
    ):
        return "standalone specprefill legacy identifier"
    if any(part == "certified" or part.startswith("certifiedkv") for part in parts):
        return "standalone certified legacy identifier"
    return None


def _legacy_mtp_violations(
    lines: list[str], relative: Path
) -> list[tuple[Path, int, str, str]]:
    source = "\n".join(lines)
    starts = _line_starts(source)
    violations: list[tuple[Path, int, str, str]] = []

    for lexeme in _lex_rust(source):
        line_number = _line_number(starts, lexeme.offset)
        line = lines[line_number - 1].strip() if lines else ""
        if lexeme.kind == "identifier":
            if _legacy_identifier_reason(lexeme.value) is not None:
                violations.append((relative, line_number, lexeme.value, line))
            continue
        match = FORBIDDEN_MTP_ENV_RE.search(lexeme.value)
        if match:
            violations.append((relative, line_number, match.group(0), line))

    return violations


def _runtime_files(root: Path) -> list[Path]:
    runtime_root = root / RUNTIME_ROOT
    if not runtime_root.is_dir():
        return []
    return sorted(path.relative_to(root) for path in runtime_root.rglob("*.rs"))


def find_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for relative in _runtime_files(root):
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
