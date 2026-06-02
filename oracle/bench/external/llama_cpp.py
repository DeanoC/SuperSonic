"""llama.cpp speed adapter for the Qwen3.5 Q4_K_M M5 Max comparison lane."""
from __future__ import annotations
import subprocess
import time
from pathlib import Path

from .common import (
    ExternalAdapter,
    ExternalWorkload,
    build_external_cell,
    get_engine_version,
    parse_ms_per_token_samples,
    read_pinned_version,
)

DEFAULT_PIN = Path(__file__).parent.parent.parent.parent / "tools" / "external" / "llama-cpp-version.txt"


class LlamaCppVersionMismatch(RuntimeError):
    pass


class LlamaCppAdapter(ExternalAdapter):
    name = "llama.cpp"

    def __init__(self, version_pin_file: Path = DEFAULT_PIN, binary: str = "llama-bench"):
        self.binary = binary
        self.pin_file = version_pin_file

    def assert_version_match(self) -> None:
        pinned = read_pinned_version(self.pin_file)
        actual = get_engine_version([self.binary, "--version"])
        if pinned != actual:
            raise LlamaCppVersionMismatch(
                f"llama.cpp pinned={pinned!r} actual={actual!r}; bump {self.pin_file} or install pinned version"
            )

    def supports(self, model: str, quant: str) -> bool:
        return (model, quant) == ("qwen3.5-35b-a3b", "q4km")

    def measure_speed(self, model: str, quant: str, prompt: str,
                      max_new_tokens: int, model_dir: Path) -> dict:
        workload = ExternalWorkload(
            prompt=prompt,
            prompt_tokens=0 if prompt == "" else None,
            max_new_tokens=max_new_tokens,
            context_size=1024,
            warmup_runs=1,
            measurement_runs=5,
        )
        return self.measure_workload(model, quant, model_dir, workload)

    def measure_workload(self, model: str, quant: str, model_dir: Path,
                         workload: ExternalWorkload) -> dict:
        version = get_engine_version([self.binary, "--version"])
        if not self.supports(model, quant):
            return build_external_cell(
                engine=self.name,
                engine_version=version,
                model=model,
                quant=quant,
                status="unsupported_by_engine",
                workload=workload,
            )

        try:
            model_path = self._resolve_model_path(model_dir)
        except FileNotFoundError as exc:
            return build_external_cell(
                engine=self.name,
                engine_version=version,
                model=model,
                quant=quant,
                status="error",
                workload=workload,
                stderr_tail=str(exc),
            )
        cmd = self._command(model_path, workload)
        time.sleep(3)
        try:
            subprocess.run(self._warmup_command(model_path, workload), capture_output=True, text=True, check=True)
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            combined = (out.stdout or "") + "\n" + (out.stderr or "")
            samples = parse_ms_per_token_samples(combined)
            if samples:
                return build_external_cell(
                    engine=self.name,
                    engine_version=version,
                    model=model,
                    quant=quant,
                    status="ok",
                    workload=workload,
                    command=cmd,
                    ms_samples=samples,
                    extras={
                        "model_path": str(model_path),
                        "batch_context": "llama-bench defaults unless overridden by local binary",
                    },
                )
            return build_external_cell(
                engine=self.name,
                engine_version=version,
                model=model,
                quant=quant,
                status="error",
                workload=workload,
                command=cmd,
                stderr_tail=combined[-2000:],
            )
        except subprocess.CalledProcessError as exc:
            tail = ((exc.stdout or "") + "\n" + (exc.stderr or ""))[-2000:]
            return build_external_cell(
                engine=self.name,
                engine_version=version,
                model=model,
                quant=quant,
                status="error",
                workload=workload,
                command=cmd,
                stderr_tail=tail,
            )

    def _resolve_model_path(self, model_dir: Path) -> Path:
        if model_dir.is_file():
            return model_dir
        matches = sorted(model_dir.glob("*.gguf"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"no .gguf file found in {model_dir}")
        raise FileNotFoundError(f"multiple .gguf files found in {model_dir}; pass the exact GGUF path")

    def _command(self, model_path: Path, workload: ExternalWorkload) -> list[str]:
        cmd = [
            self.binary,
            "-m", str(model_path),
            "-p", str(workload.prompt_tokens if workload.prompt_tokens is not None else 0),
            "-n", str(workload.max_new_tokens),
            "-r", str(workload.measurement_runs),
        ]
        if workload.context_size is not None:
            cmd.extend(["-c", str(workload.context_size)])
        return cmd

    def _warmup_command(self, model_path: Path, workload: ExternalWorkload) -> list[str]:
        warm = ExternalWorkload(
            prompt=workload.prompt,
            prompt_tokens=workload.prompt_tokens,
            max_new_tokens=min(16, workload.max_new_tokens),
            context_size=workload.context_size,
            warmup_runs=0,
            measurement_runs=max(1, workload.warmup_runs),
            temperature=workload.temperature,
            top_k=workload.top_k,
            seed=workload.seed,
        )
        return self._command(model_path, warm)
