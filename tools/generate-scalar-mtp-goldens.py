#!/usr/bin/env python3
"""Generate reviewed, engine-specific MTP token goldens from fresh processes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark import adapters, manifest
from tools.benchmark.execution import APPROVED_ARTIFACT
from tools.benchmark.model import PerformanceCase


CASES = (
    (
        "ordinary-vs-mtp-token-equality-1",
        "Reply with the token ids for 1 2 3.",
    ),
    (
        "ordinary-vs-mtp-token-equality-2",
        "Reply with the token ids for 4 5 6.",
    ),
)


def review_repeated_outputs(outputs, *, max_new_tokens: int) -> dict[str, object]:
    values = tuple(outputs)
    if len(values) != 2:
        raise ValueError("golden generation requires exactly two independent runs")
    first, second = values
    if first.token_ids is None or second.token_ids is None:
        raise ValueError("independent runs must expose exact token ids")
    if not first.token_ids or len(first.token_ids) > max_new_tokens:
        raise ValueError("independent runs produced an invalid token count")
    if first.token_ids != second.token_ids or first.generated_text != second.generated_text:
        raise ValueError("independent runs did not produce identical tokens and text")
    return {"token_ids": list(first.token_ids), "generated_text": first.generated_text}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wmma-binary", default="target/release/supersonic")
    parser.add_argument("--scalar-binary", default="tools/supersonic-scalar-lab.py")
    parser.add_argument("--scalar-instruction-stream-sha256", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = args.artifact.resolve()
    _validate_approved_artifact(artifact)
    model_dir = args.model_dir.resolve()
    tokenizer_sha256 = _digest_file(model_dir / "tokenizer.json")
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer_config.json must contain chat_template")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()

    engines = {
        "supersonic-wmma": replace(
            manifest.load_engine("supersonic-wmma"),
            binary=str(Path(args.wmma_binary).resolve()),
        ),
        "supersonic-scalar-lab": replace(
            manifest.load_engine("supersonic-scalar-lab"),
            binary=str(Path(args.scalar_binary).resolve()),
        ),
    }
    captures = args.output.parent / f"{args.output.stem}-captures"
    captures.mkdir(parents=True, exist_ok=False)
    engine_entries: dict[str, object] = {}
    inputs = adapters.AdapterInputs(
        model_dir=model_dir,
        artifact=artifact,
        peer_artifact=None,
        chat=True,
        device=args.device,
        context_size=32768,
        fixed_token_count=False,
        sampling_seed=1,
    )
    for engine_name, engine in engines.items():
        case_entries: dict[str, object] = {}
        for case_id, prompt in CASES:
            case = PerformanceCase(
                id=case_id,
                prompt=prompt,
                max_new_tokens=8,
                warmups=0,
                repetitions=1,
                mode="ordinary",
                cache_state="cold-load",
                timeout_seconds=int(args.timeout_seconds),
                decoding_policy="greedy",
                engines=(engine_name,),
            )
            outputs = []
            for run in (1, 2):
                command = adapters.build_command(engine, case, inputs)
                completed = subprocess.run(
                    command,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                stem = captures / f"{engine_name}-{case_id}-run-{run}"
                stem.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
                stem.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise ValueError(f"{engine_name}/{case_id} run {run} failed")
                outputs.append(adapters.parse_output(engine_name, completed.stdout, completed.stderr))
            reviewed = review_repeated_outputs(outputs, max_new_tokens=8)
            reviewed["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            case_entries[case_id] = reviewed
        binary_sha256 = _digest_file(Path(engine.binary))
        engine_entries[engine_name] = {
            "binary_sha256": binary_sha256,
            "instruction_stream_sha256": (
                args.scalar_instruction_stream_sha256
                if engine_name == "supersonic-scalar-lab"
                else None
            ),
            "cases": case_entries,
        }
    payload = {
        "version": "v1",
        "artifact": {
            "semantic_id": APPROVED_ARTIFACT["semantic_id"],
            "source_revision": APPROVED_ARTIFACT["source_revision"],
            "sha256": APPROVED_ARTIFACT["sha256"],
        },
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "engines": engine_entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "captures": str(captures)}, sort_keys=True))
    return 0


def _validate_approved_artifact(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("approved artifact must be a regular non-symlink file")
    if path.name != APPROVED_ARTIFACT["filename"]:
        raise ValueError("approved artifact filename mismatch")
    if path.stat().st_size != APPROVED_ARTIFACT["size_bytes"]:
        raise ValueError("approved artifact size mismatch")
    if _digest_file(path) != APPROVED_ARTIFACT["sha256"]:
        raise ValueError("approved artifact SHA-256 mismatch")


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
