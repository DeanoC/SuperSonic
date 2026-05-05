"""hipfire (https://github.com/Kaden-Schutt/hipfire) external adapter.

Speed-only Phase 1. Quality comparison out of scope (tokenizer alignment).
"""
from __future__ import annotations
import re
import subprocess
import time
from pathlib import Path

from .common import ExternalAdapter, get_engine_version, read_pinned_version

DEFAULT_PIN = Path(__file__).parent.parent.parent.parent / "tools" / "external" / "hipfire-version.txt"


class HipfireVersionMismatch(RuntimeError):
    pass


class HipfireAdapter(ExternalAdapter):
    name = "hipfire"

    def __init__(self, version_pin_file: Path = DEFAULT_PIN, binary: str = "hipfire"):
        self.binary = binary
        self.pin_file = version_pin_file

    def assert_version_match(self) -> None:
        pinned = read_pinned_version(self.pin_file)
        actual = get_engine_version([self.binary, "--version"])
        if pinned != actual:
            raise HipfireVersionMismatch(
                f"hipfire pinned={pinned!r} actual={actual!r}; bump {self.pin_file} or install pinned version"
            )

    def supports(self, model: str, quant: str) -> bool:
        # hipfire's supported model set as of 2026-05-05 (verify with `hipfire --list-models`).
        # Conservative default: only the most-likely-supported pairs; widen as confirmed.
        supported = {
            ("qwen3.5-0.8b", "bf16"), ("qwen3.5-0.8b", "int4"),
            ("qwen3.5-2b", "bf16"),   ("qwen3.5-2b", "int4"),
            ("qwen3.5-4b", "bf16"),   ("qwen3.5-4b", "int4"),
            ("qwen3.5-9b", "bf16"),   ("qwen3.5-9b", "int4"),
        }
        return (model, quant) in supported

    def measure_speed(self, model: str, quant: str, prompt: str,
                      max_new_tokens: int, model_dir: Path) -> dict:
        version = get_engine_version([self.binary, "--version"])
        if not self.supports(model, quant):
            return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                    "model": model, "quant": quant, "status": "unsupported_by_engine",
                    "ms_per_step": None, "samples": None, "stderr_tail": None}

        # 3s cooldown + 1 warmup + 3 measurement passes, mirror SuperSonic discipline.
        time.sleep(3)
        # Warmup
        self._invoke(model, quant, prompt, 2, model_dir)
        samples = []
        last_err = None
        for _ in range(3):
            try:
                out = self._invoke(model, quant, prompt, max_new_tokens, model_dir)
                ms = self._extract_ms_per_step(out)
                if ms is not None:
                    samples.append(ms)
                else:
                    last_err = out[-2000:]
            except subprocess.CalledProcessError as e:
                last_err = (e.stderr or e.stdout or "")[-2000:]

        if samples:
            samples.sort()
            median = samples[len(samples) // 2]
            return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                    "model": model, "quant": quant, "status": "ok",
                    "ms_per_step": median, "samples": samples, "stderr_tail": None}
        return {"schema_version": 1, "engine": "hipfire", "engine_version": version,
                "model": model, "quant": quant, "status": "error",
                "ms_per_step": None, "samples": None, "stderr_tail": str(last_err)}

    def _invoke(self, model: str, quant: str, prompt: str, max_new: int, model_dir: Path) -> str:
        # Adjust the CLI shape to match actual hipfire conventions; below is a placeholder
        # that needs verification against `hipfire --help` on the dev machine.
        cmd = [self.binary, "generate", "--model", str(model_dir),
               "--prompt", prompt, "--n", str(max_new)]
        if quant == "int4":
            cmd.extend(["--quant", "int4"])
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (res.stdout or "") + "\n" + (res.stderr or "")

    def _extract_ms_per_step(self, stdout: str) -> float | None:
        # hipfire output format TBD; common shapes:
        #   "tokens/sec: 95.3" → ms_per_step = 1000 / 95.3
        #   "ms/token: 10.5"   → ms_per_step = 10.5
        m = re.search(r"tokens?/sec[:\s=]+([0-9.]+)", stdout)
        if m:
            tps = float(m.group(1))
            return 1000.0 / tps if tps > 0 else None
        m = re.search(r"ms/(?:token|step)[:\s=]+([0-9.]+)", stdout)
        if m:
            return float(m.group(1))
        return None
