#!/usr/bin/env python3
"""Audit the generated AMDGPU code object for the raw-Q6 scalar output head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_REDUCTION_STAGES = 5
EXPECTED_MUL_COUNT = 16
EXPECTED_FMA_COUNT = 8
EXPECTED_XOR_OFFSETS = [16, 8, 4, 2, 1]
INSTRUCTION_PATTERNS = {
    "ds_bpermute_b32": r"\bds_bpermute_b32\b",
    "v_add_f32": r"\bv_(?:dual_)?add_f32(?:_[a-z0-9]+)?\b",
    "v_fma_f32": r"\bv_(?:dual_)?fma[ck]?_f32(?:_[a-z0-9]+)?\b",
    "v_fma_mix_f32": r"\b(?:v_fma_mix|v_dual_fma_mix)[a-z0-9_]*\b",
    "v_mfma_f32_16x16x16bf16": r"\bv_mfma[a-z0-9_]*\b",
    "v_mul_f32": r"\bv_(?:dual_)?mul_f32(?:_[a-z0-9]+)?\b",
    "v_wmma_f32_16x16x16_bf16": r"\bv_wmma[a-z0-9_]*\b",
}
FORBIDDEN_REASSOCIATION_PATTERNS = {
    "v_add3_f32": r"\bv_(?:dual_)?add3_f32(?:_[a-z0-9]+)?\b",
    "v_mad_f32": r"\bv_(?:dual_)?mad(?:_mix)?_f32(?:_[a-z0-9]+)?\b",
    "v_muladd_f32": r"\bv_(?:dual_)?muladd_f32(?:_[a-z0-9]+)?\b",
}
OFFLOAD_IMAGE_SUFFIX = ".0.hipv4-amdgcn-amd-amdhsa--gfx1201"

# This is the canonical instruction-stream digest for the supported ROCm
# 7.14 gfx1201 kernel object. It is deliberately distinct from the container
# object digest, which contains relocatable addresses and bundle metadata.
PINNED_INSTRUCTION_STREAM_SHA256 = "a72c8ffba00db69dc5b29aa61238f23ed3fb1540325497d4d84a7c2608d69667"


def _matches_symbol(candidate: str, symbol: str) -> bool:
    return candidate == symbol or (
        candidate.startswith("_Z")
        and not candidate.endswith(".kd")
        and f"{len(symbol)}{symbol}" in candidate
    )


def _kernel_disassembly(disassembly: str, symbol: str) -> str:
    label = re.compile(r"^[0-9A-Fa-f]+ <([^>]+)>:$")
    lines = disassembly.splitlines()
    starts: list[int] = []
    for index, line in enumerate(lines):
        match = label.match(line.strip())
        if (
            match
            and _matches_symbol(match.group(1), symbol)
            and "__device_stub__" not in match.group(1)
        ):
            starts.append(index + 1)
    if len(starts) != 1:
        return ""
    start = starts[0]
    end = len(lines)
    for index in range(start, len(lines)):
        if label.match(lines[index].strip()):
            end = index
            break
    return "\n".join(lines[start:end])


def _kernel_metadata(metadata: str, symbol: str) -> str:
    blocks = re.split(r"(?m)^  - \.args:\s*$", metadata)
    matches: list[str] = []
    for block in blocks:
        candidates = re.findall(r"(?m)^\s*\.(?:name|symbol):\s*(\S+)", block)
        matching = [candidate for candidate in candidates if _matches_symbol(candidate, symbol)]
        if len(matching) > 1:
            return ""
        if matching:
            matches.append(block)
    return matches[0] if len(matches) == 1 else ""


def canonical_instruction_stream(kernel: str) -> str:
    """Return address/comment-free selected-symbol instructions.

    LLVM's AMDGPU disassembler appends the instruction address and encoding as
    a ``//`` comment. Those bytes are not part of the generated operation
    graph and vary with placement, while the opcode, operands, register
    dependencies, and immediates must remain unchanged.
    """

    canonical: list[str] = []
    for raw_line in kernel.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        # Accept objdump variants that put the address before the instruction,
        # while leaving immediates such as ``0x100`` untouched.
        line = re.sub(r"^[0-9A-Fa-f]+:\s+(?=[A-Za-z_])", "", line)
        canonical.append(" ".join(line.split()))
    return "\n".join(canonical)


def instruction_stream_sha256(kernel: str) -> str:
    """Hash the selected symbol's canonical instruction stream."""

    return hashlib.sha256(canonical_instruction_stream(kernel).encode("utf-8")).hexdigest()


def _xor_offsets(kernel: str) -> list[int | str | None]:
    offsets: list[int | str | None] = []
    opcode = re.compile(r"^v_(?:dual_)?xor_b32(?:_[a-z0-9]+)?\s+(.+)$")
    for line in canonical_instruction_stream(kernel).splitlines():
        match = opcode.match(line)
        if not match:
            continue
        operands = [operand.strip() for operand in match.group(1).split(",")]
        if len(operands) < 3:
            offsets.append(None)
            continue
        try:
            offsets.append(int(operands[1], 0))
        except ValueError:
            offsets.append(operands[1])
    return offsets


def _metadata_int(metadata: str, key: str) -> int | None:
    match = re.search(rf"(?m)^\s*\.{re.escape(key)}:\s*(\d+)\s*$", metadata)
    return int(match.group(1)) if match else None


def _descriptor_rsrc1(disassembly: str, kernel_metadata: str, full_metadata: str) -> int | None:
    direct = re.search(r"(?im)^\s*COMPUTE_PGM_RSRC1:\s*(0x[0-9a-f]+|\d+)\s*$", kernel_metadata)
    if direct:
        return int(direct.group(1), 0)

    descriptor = re.search(r"(?m)^\s*\.symbol:\s*(\S+\.kd)\s*$", kernel_metadata)
    if not descriptor:
        return None
    descriptor_name = re.escape(descriptor.group(1))
    value = re.search(
        rf"(?ms)^\s*Symbol \{{\s+Name:\s*{descriptor_name}\s+\(\d+\)\s+Value:\s*(0x[0-9A-Fa-f]+)",
        full_metadata,
    )
    if not value:
        return None
    address = int(value.group(1), 16) + 48

    rodata: dict[int, int] = {}
    in_rodata = False
    for line in disassembly.splitlines():
        if line.startswith("Contents of section "):
            in_rodata = line.rstrip().endswith(".rodata:")
            continue
        if not in_rodata:
            continue
        match = re.match(r"^\s*([0-9A-Fa-f]+)\s+((?:[0-9A-Fa-f]{8}\s*)+)", line)
        if not match:
            continue
        chunk = re.sub(r"\s", "", match.group(2))
        base = int(match.group(1), 16)
        for offset in range(0, len(chunk), 2):
            rodata[base + offset // 2] = int(chunk[offset : offset + 2], 16)
    try:
        encoded = bytes(rodata[address + offset] for offset in range(4))
    except KeyError:
        return None
    return int.from_bytes(encoded, "little")


def analyze(disassembly: str, metadata: str, symbol: str) -> dict:
    """Parse LLVM output into the scalar-head generated-code contract report."""
    kernel = _kernel_disassembly(disassembly, symbol)
    kernel_metadata = _kernel_metadata(metadata, symbol)
    rsrc1 = _descriptor_rsrc1(disassembly, kernel_metadata, metadata)
    canonical = canonical_instruction_stream(kernel)
    instruction_counts = {
        name: len(re.findall(pattern, canonical))
        for name, pattern in INSTRUCTION_PATTERNS.items()
    }
    forbidden_reassociation_counts = {
        name: len(re.findall(pattern, canonical))
        for name, pattern in FORBIDDEN_REASSOCIATION_PATTERNS.items()
    }
    spill_fields = {
        "private_segment_fixed_size": _metadata_int(kernel_metadata, "private_segment_fixed_size"),
        "sgpr_spill_count": _metadata_int(kernel_metadata, "sgpr_spill_count"),
        "vgpr_spill_count": _metadata_int(kernel_metadata, "vgpr_spill_count"),
    }
    missing_spill_fields = [name for name, value in spill_fields.items() if value is None]
    return {
        "symbol": symbol if kernel and kernel_metadata else None,
        "vgpr_count": _metadata_int(kernel_metadata, "vgpr_count"),
        "spill_count": None if missing_spill_fields else sum(spill_fields.values()),
        "fp32_round_mode": None if rsrc1 is None else ("RNE" if ((rsrc1 >> 12) & 3) == 0 else str((rsrc1 >> 12) & 3)),
        "fp32_denorm_mode": None if rsrc1 is None else ("preserve" if ((rsrc1 >> 16) & 3) == 3 else str((rsrc1 >> 16) & 3)),
        "instruction_counts": instruction_counts,
        "instruction_stream_sha256": instruction_stream_sha256(kernel),
        "xor_offsets": _xor_offsets(kernel),
        "_forbidden_reassociation_counts": forbidden_reassociation_counts,
        "_missing_spill_fields": missing_spill_fields,
    }


def find_violations(
    report: dict,
    *,
    expected_instruction_stream_sha256: str | None = None,
) -> list[str]:
    """Return every generated-code contract violation in stable order."""
    violations: list[str] = []
    if report["symbol"] is None:
        violations.append("missing symbol in disassembly or metadata")
    if report["vgpr_count"] is None:
        violations.append("missing vgpr_count metadata")
    for field in report["_missing_spill_fields"]:
        violations.append(f"missing {field} metadata")
    if not report["_missing_spill_fields"] and report["spill_count"] != 0:
        violations.append(f"spill_count must be 0 (got {report['spill_count']})")
    if report["fp32_round_mode"] != "RNE":
        violations.append(f"fp32_round_mode must be RNE (got {report['fp32_round_mode']})")
    if report["fp32_denorm_mode"] != "preserve":
        violations.append(f"fp32_denorm_mode must be preserve (got {report['fp32_denorm_mode']})")
    counts = report["instruction_counts"]
    for forbidden in (
        "v_fma_mix_f32",
        "v_wmma_f32_16x16x16_bf16",
        "v_mfma_f32_16x16x16bf16",
    ):
        if counts[forbidden]:
            violations.append(f"forbidden {forbidden} count is {counts[forbidden]}")
    for forbidden, count in report["_forbidden_reassociation_counts"].items():
        if count:
            violations.append(f"forbidden {forbidden} count is {count}")
    if counts["v_mul_f32"] != EXPECTED_MUL_COUNT:
        violations.append(
            f"v_mul_f32 count must be {EXPECTED_MUL_COUNT} (got {counts['v_mul_f32']})"
        )
    if counts["v_fma_f32"] != EXPECTED_FMA_COUNT:
        violations.append(
            f"v_fma_f32 count must be {EXPECTED_FMA_COUNT} (got {counts['v_fma_f32']})"
        )
    for instruction in ("ds_bpermute_b32", "v_add_f32"):
        if counts[instruction] != EXPECTED_REDUCTION_STAGES:
            violations.append(
                f"{instruction} count must be {EXPECTED_REDUCTION_STAGES} (got {counts[instruction]})"
            )
    if report["xor_offsets"] != EXPECTED_XOR_OFFSETS:
        violations.append(
            f"xor offsets must be {EXPECTED_XOR_OFFSETS} (got {report['xor_offsets']})"
        )
    if expected_instruction_stream_sha256 is not None and report["instruction_stream_sha256"] != expected_instruction_stream_sha256:
        violations.append(
            "instruction_stream_sha256 mismatch: "
            f"expected {expected_instruction_stream_sha256} "
            f"(got {report['instruction_stream_sha256']})"
        )
    return violations


def _run_tool(command: list[str]) -> str:
    return subprocess.run(
        command,
        timeout=30,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _inspect_object(object_path: Path, objdump: str, readobj: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="supersonic-scalar-head-") as temporary:
        bundled = Path(temporary) / "code-object.o"
        shutil.copyfile(object_path, bundled)
        _run_tool([objdump, "--offloading", str(bundled)])
        extracted = Path(f"{bundled}{OFFLOAD_IMAGE_SUFFIX}")
        inspected = extracted if extracted.is_file() and extracted.stat().st_size else object_path
        disassembly = _run_tool(
            [objdump, "--disassemble", "--full-contents", "--mcpu=gfx1201", str(inspected)]
        )
        metadata = _run_tool([readobj, "--notes", "--symbols", str(inspected)])
    return disassembly, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--object", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args(argv)
    if not args.object.is_file() or args.object.stat().st_size == 0:
        print(f"object must be a nonempty regular file: {args.object}", file=sys.stderr)
        return 1

    objdump = os.environ.get("LLVM_OBJDUMP", "llvm-objdump")
    readobj = os.environ.get("LLVM_READOBJ", "llvm-readobj")
    try:
        disassembly, metadata = _inspect_object(args.object, objdump, readobj)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"LLVM inspection failed: {error}", file=sys.stderr)
        return 1

    report = analyze(disassembly, metadata, args.symbol)
    violations = find_violations(
        report,
        expected_instruction_stream_sha256=PINNED_INSTRUCTION_STREAM_SHA256,
    )
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    public_report["sha256"] = hashlib.sha256(args.object.read_bytes()).hexdigest()
    print(json.dumps(public_report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
