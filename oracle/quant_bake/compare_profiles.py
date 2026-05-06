#!/usr/bin/env python3
"""Compare Qwen quant profiles with small SuperSonic runtime probes.

The harness intentionally does not bake. It consumes existing local bake
directories, temporarily exposes them under the model's `.supersonic/vN-profile`
path when needed, runs a short teacher-forced quality proxy, and writes JSON +
Markdown summaries for later full-bake ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TF_JSON_RE = re.compile(r"^\[teacher_forced_json\] (.+)$", re.MULTILINE)
RESULT_RE = re.compile(
    r"^\[result\].*?decode_ms=(?P<decode_ms>[0-9.]+)\s+"
    r"ms_per_tok=(?P<ms_per_tok>[0-9.]+).*?batch_size=(?P<batch_size>[0-9]+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ProfileSpec:
    label: str
    runtime_profile: str
    bake_dir: Path | None


@dataclass(frozen=True)
class LinkState:
    path: Path | None
    made_link: bool = False
    replaced_target: str | None = None


def _run(cmd: list[str], timeout: int) -> str:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n{proc.stdout[-4000:]}"
        )
    return proc.stdout


def _link_profile(
    model_dir: Path,
    runtime_profile: str,
    bake_dir: Path | None,
    replace_existing_symlink: bool,
) -> LinkState:
    if bake_dir is None:
        return LinkState(None)
    manifest = bake_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{runtime_profile}: missing manifest.json in {bake_dir}")
    version = json.loads(manifest.read_text()).get("format_version")
    if not isinstance(version, int):
        raise ValueError(f"{runtime_profile}: manifest has no integer format_version")
    link = model_dir / ".supersonic" / f"v{version}-{runtime_profile}"
    if link.exists() or link.is_symlink():
        if link.is_symlink() and Path(os.readlink(link)) == bake_dir:
            return LinkState(link)
        if link.is_symlink() and replace_existing_symlink:
            previous = os.readlink(link)
            link.unlink()
            os.symlink(bake_dir, link)
            return LinkState(link, replaced_target=previous)
        raise FileExistsError(f"refusing to overwrite existing bake path {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(bake_dir, link)
    return LinkState(link, made_link=True)


def _restore_link(state: LinkState) -> None:
    if state.path is None:
        return
    if state.made_link and state.path.is_symlink():
        state.path.unlink()
        return
    if state.replaced_target is not None:
        if state.path.is_symlink():
            state.path.unlink()
        os.symlink(state.replaced_target, state.path)


def _profile_args(profile: str) -> list[str]:
    if profile == "bf16":
        return []
    return ["--weight-quant", profile]


def _score_teacher_forced(
    binary: Path,
    model: str,
    model_dir: Path,
    profile: str,
    prompt: str,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        str(binary),
        "--model",
        model,
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "1",
        "--teacher-forced",
        "--no-download",
        "--emit-stage-timings",
        *_profile_args(profile),
    ]
    out = _run(cmd, timeout)
    match = TF_JSON_RE.search(out)
    if not match:
        raise RuntimeError(f"{profile}: missing teacher_forced_json\n{out[-4000:]}")
    row = json.loads(match.group(1))
    row["runtime_profile"] = profile
    return row


def _score_teacher_forced_many(
    binary: Path,
    model: str,
    model_dir: Path,
    profile: str,
    prompts: list[str],
    timeout: int,
) -> dict[str, Any]:
    runs = [
        _score_teacher_forced(binary, model, model_dir, profile, prompt, timeout)
        for prompt in prompts
    ]
    total_nll = sum(float(row.get("total_nll", 0.0)) for row in runs)
    scored_tokens = sum(int(row.get("scored_tokens", 0)) for row in runs)
    prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in runs)
    avg_nll = total_nll / scored_tokens if scored_tokens else float("nan")
    return {
        "runtime_profile": profile,
        "runs": runs,
        "prompt_count": len(runs),
        "prompt_tokens": prompt_tokens,
        "scored_tokens": scored_tokens,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "perplexity": float("inf") if avg_nll > 709.0 else math.exp(avg_nll),
        "prefill_ms": sum(float(row.get("prefill_ms", 0.0)) for row in runs) / max(1, len(runs)),
        "ms_per_token": sum(float(row.get("ms_per_token", 0.0)) for row in runs) / max(1, len(runs)),
    }


def _probe_generation(
    binary: Path,
    model: str,
    model_dir: Path,
    profile: str,
    prompt: str,
    max_new_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        str(binary),
        "--model",
        model,
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--no-download",
        "--emit-stage-timings",
        *_profile_args(profile),
    ]
    out = _run(cmd, timeout)
    match = RESULT_RE.search(out)
    result = {
        "runtime_profile": profile,
        "stdout_tail": out[-2000:],
    }
    if match:
        result.update(
            {
                "decode_ms": float(match.group("decode_ms")),
                "ms_per_tok": float(match.group("ms_per_tok")),
                "batch_size": int(match.group("batch_size")),
            }
        )
    return result


def _load_manifest(profile: str, bake_dir: Path | None, model_dir: Path) -> dict[str, Any] | None:
    if profile == "bf16":
        path = model_dir / ".supersonic" / "v2" / "manifest.json"
    elif bake_dir is None:
        path = model_dir / ".supersonic" / f"v2-{profile}" / "manifest.json"
    else:
        path = bake_dir / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    return {
        "format_version": manifest.get("format_version"),
        "converter_version": manifest.get("converter_version"),
        "quant_profile": manifest.get("quant_profile"),
        "source_quant": manifest.get("source_quant"),
        "quant_method": manifest.get("quant_method"),
    }


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Profile | PPL proxy | avg NLL | TF prefill ms | TF ms/token | gen ms/token | notes |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in rows:
        tf = row.get("teacher_forced") or {}
        gen = row.get("generation") or {}
        manifest = row.get("manifest") or {}
        method = manifest.get("quant_method") or {}
        params = method.get("parameters") or {}
        notes = []
        if method.get("calibration_samples") is not None:
            notes.append(
                f"calib={method.get('calibration_samples')}x{method.get('calibration_seqlen')}"
            )
        if params.get("steps") is not None:
            notes.append(f"steps={params.get('steps')}")
        if not method and row["profile"] != "bf16":
            notes.append("legacy manifest")
        lines.append(
            "| {profile} | {ppl:.3f} | {nll:.3f} | {prefill:.1f} | {tf_ms:.1f} | {gen_ms:.1f} | {notes} |".format(
                profile=row["profile"],
                ppl=float(tf.get("perplexity", float("nan"))),
                nll=float(tf.get("avg_nll", float("nan"))),
                prefill=float(tf.get("prefill_ms", float("nan"))),
                tf_ms=float(tf.get("ms_per_token", float("nan"))),
                gen_ms=float(gen.get("ms_per_tok", float("nan"))),
                notes=", ".join(notes),
            )
        )
    return "\n".join(lines) + "\n"


def parse_profile_spec(raw: str) -> ProfileSpec:
    label: str | None = None
    if ":" in raw:
        label, raw = raw.split(":", 1)
        label = label.strip()
    if "=" in raw:
        name, path = raw.split("=", 1)
        runtime_profile = name.strip()
        return ProfileSpec(
            label=label or runtime_profile,
            runtime_profile=runtime_profile,
            bake_dir=Path(path).expanduser(),
        )
    runtime_profile = raw.strip()
    return ProfileSpec(label=label or runtime_profile, runtime_profile=runtime_profile, bake_dir=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        help=(
            "Profile name, profile=/path/to/bake-dir, or label:profile=/path. "
            "Repeat for each profile."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help=(
            "Teacher-forced prompt. Repeat for multi-prompt scoring. If omitted, "
            "uses the built-in short smoke prompt."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional UTF-8 text file with one teacher-forced prompt per non-empty line.",
    )
    parser.add_argument("--gen-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-json", type=Path, default=Path("target/quant_profile_compare.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/quant_profile_compare.md"))
    parser.add_argument(
        "--replace-existing-symlinks",
        action="store_true",
        help=(
            "Allow a profile bake symlink in model_dir/.supersonic to be "
            "temporarily replaced and restored. Refuses to replace real dirs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompt or [
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is used for testing language model scoring."
    ]
    if args.prompt_file is not None:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text().splitlines()
            if line.strip()
        )
    if not prompts:
        raise ValueError("at least one prompt is required")
    rows: list[dict[str, Any]] = []
    for spec in [parse_profile_spec(raw) for raw in args.profile]:
        link_state = _link_profile(
            args.model_dir,
            spec.runtime_profile,
            spec.bake_dir,
            args.replace_existing_symlinks,
        )
        try:
            print(f"[compare] {spec.label} ({spec.runtime_profile})", flush=True)
            tf = _score_teacher_forced_many(
                args.binary,
                args.model,
                args.model_dir,
                spec.runtime_profile,
                prompts,
                args.timeout,
            )
            gen = _probe_generation(
                args.binary,
                args.model,
                args.model_dir,
                spec.runtime_profile,
                prompts[0],
                args.gen_new_tokens,
                args.timeout,
            )
            rows.append(
                {
                    "profile": spec.label,
                    "runtime_profile": spec.runtime_profile,
                    "bake_dir": str(spec.bake_dir) if spec.bake_dir else None,
                    "manifest": _load_manifest(spec.runtime_profile, spec.bake_dir, args.model_dir),
                    "teacher_forced": tf,
                    "generation": gen,
                }
            )
            print(
                f"  ppl={tf['perplexity']:.3f} tf_ms/tok={tf['ms_per_token']:.1f} "
                f"gen_ms/tok={gen.get('ms_per_tok')}",
                flush=True,
            )
        finally:
            _restore_link(link_state)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2))
    args.out_md.write_text(_render_markdown(rows))
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
