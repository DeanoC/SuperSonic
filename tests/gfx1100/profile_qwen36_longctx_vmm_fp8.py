#!/usr/bin/env python3
"""Run Qwen3.6 long-context VMM/KV-FP8 profile rows with GPU-idle gating.

This wraps bench_qwen36_longctx.py but starts one row at a time. Before each
row it polls rocm-smi and waits until the GPU is idle, which is useful when
another local agent may occasionally be using the same device.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROFILES: dict[str, dict[str, str]] = {
    "vmm-fp8-large": {
        "contexts": "8192,16384",
        "modes": "int4-vmm,int4-kv-fp8",
        "sparse_caps": "320",
    },
    "vmm-fp8-xl": {
        "contexts": "8192,16384,32768",
        "modes": "int4-vmm,int4-kv-fp8",
        "sparse_caps": "320",
    },
    "sparse-vmm-fp8": {
        "contexts": "8192,16384",
        "modes": "int4-vmm,int4-kv-fp8,sparse,sparse-kv-fp8",
        "sparse_caps": "320",
    },
}


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def rocm_smi_snapshot() -> tuple[int | None, int | None, list[int], str]:
    proc = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--showpidgpus"],
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return None, None, [], output

    gpu_use = None
    mem_use = None
    pids: list[int] = []
    for line in output.splitlines():
        gpu_match = re.search(r"GPU use \(%\):\s*([0-9]+)", line)
        if gpu_match:
            gpu_use = int(gpu_match.group(1))
        mem_match = re.search(r"VRAM%\):\s*([0-9]+)", line)
        if mem_match:
            mem_use = int(mem_match.group(1))
        pid_match = re.search(r"PID\s+([0-9]+)\s+is using", line)
        if pid_match:
            pids.append(int(pid_match.group(1)))
    return gpu_use, mem_use, pids, output


def wait_for_gpu_idle(
    max_gpu_use: int,
    max_mem_use: int,
    poll_seconds: float,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        gpu_use, mem_use, pids, raw = rocm_smi_snapshot()
        if gpu_use is None or mem_use is None:
            raise RuntimeError(f"could not read rocm-smi output:\n{raw}")
        if gpu_use <= max_gpu_use and mem_use <= max_mem_use and not pids:
            print(
                f"[gpu-idle] gpu_use={gpu_use}% mem_use={mem_use}% pids=[]",
                flush=True,
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "GPU did not become idle before timeout: "
                f"gpu_use={gpu_use}% mem_use={mem_use}% pids={pids}"
            )
        print(
            f"[gpu-busy] gpu_use={gpu_use}% mem_use={mem_use}% pids={pids}; "
            f"sleeping {poll_seconds:.0f}s",
            flush=True,
        )
        time.sleep(poll_seconds)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    return list(payload.get("rows") or [])


def write_combined_report(out_json: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "qwen36-moe-longctx-vmm-fp8-profile-v1",
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="vmm-fp8-large")
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/data/models/Qwen3.6-35B-A3B"))
    parser.add_argument("--contexts")
    parser.add_argument("--modes")
    parser.add_argument("--sparse-caps")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--out-dir", type=Path, default=Path("target/qwen36_longctx_profiles"))
    parser.add_argument("--max-gpu-use", type=int, default=10)
    parser.add_argument("--max-mem-use", type=int, default=5)
    parser.add_argument("--gpu-idle-timeout", type=int, default=7200)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-wait-gpu-idle", action="store_true")
    args = parser.parse_args()

    defaults = PROFILES[args.profile]
    contexts = parse_csv_ints(args.contexts or defaults["contexts"])
    modes = parse_csv(args.modes or defaults["modes"])
    sparse_caps = args.sparse_caps or defaults["sparse_caps"]

    rows: list[dict[str, Any]] = []
    for context in contexts:
        for mode in modes:
            if not args.no_wait_gpu_idle:
                wait_for_gpu_idle(
                    args.max_gpu_use,
                    args.max_mem_use,
                    args.gpu_poll_seconds,
                    args.gpu_idle_timeout,
                )
            label = f"{args.profile}_{context}_{mode}"
            out_json = args.out_dir / f"{label}.json"
            out_md = args.out_dir / f"{label}.md"
            cmd = [
                sys.executable,
                "tests/gfx1100/bench_qwen36_longctx.py",
                "--binary",
                str(args.binary),
                "--model-dir",
                str(args.model_dir),
                "--contexts",
                str(context),
                "--modes",
                mode,
                "--sparse-caps",
                sparse_caps,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--no-warmup",
                "--timeout",
                str(args.timeout),
                "--seed",
                str(args.seed),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
            print(f"[profile-row] context={context} mode={mode}", flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                return proc.returncode
            rows.extend(load_rows(out_json))

    write_combined_report(args.out_dir / f"{args.profile}_combined.json", rows)
    print(f"[wrote] {args.out_dir / f'{args.profile}_combined.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
