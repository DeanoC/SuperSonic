"""MLX-LM speed adapter for the Qwen3.5 large-model comparison lane."""
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

DEFAULT_PIN = Path(__file__).parent.parent.parent.parent / "tools" / "external" / "mlx-lm-version.txt"


class MlxLmVersionMismatch(RuntimeError):
    pass


class MlxLmAdapter(ExternalAdapter):
    name = "mlx-lm"

    def __init__(self, version_pin_file: Path = DEFAULT_PIN, python: str = "python3"):
        self.python = python
        self.pin_file = version_pin_file

    def assert_version_match(self) -> None:
        pinned = read_pinned_version(self.pin_file)
        actual = get_engine_version([self.python, "-m", "mlx_lm", "--version"])
        if pinned != actual:
            raise MlxLmVersionMismatch(
                f"mlx-lm pinned={pinned!r} actual={actual!r}; bump {self.pin_file} or install pinned version"
            )

    def supports(self, model: str, quant: str) -> bool:
        # MLX consumes an MLX model directory, not the raw GGUF itself. The
        # comparison lane still records quant=q4km so reports group the same
        # public target while the artifact kind is captured in extras.
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
        version = get_engine_version([self.python, "-m", "mlx_lm", "--version"])
        if not self.supports(model, quant):
            return build_external_cell(
                engine=self.name,
                engine_version=version,
                model=model,
                quant=quant,
                status="unsupported_by_engine",
                workload=workload,
            )

        cmd = self._command(model_dir, workload)
        time.sleep(3)
        try:
            subprocess.run(self._warmup_command(model_dir, workload), capture_output=True, text=True, check=True)
            samples: list[float] = []
            last_output = ""
            for _ in range(workload.measurement_runs):
                out = subprocess.run(cmd, capture_output=True, text=True, check=True)
                combined = (out.stdout or "") + "\n" + (out.stderr or "")
                last_output = combined
                parsed = parse_ms_per_token_samples(combined)
                if parsed:
                    samples.append(parsed[-1])
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
                        "model_path": str(model_dir),
                        "artifact_kind": "mlx-model-dir",
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
                stderr_tail=last_output[-2000:],
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

    def _command(self, model_dir: Path, workload: ExternalWorkload) -> list[str]:
        return [
            self.python,
            "-m", "mlx_lm.generate",
            "--model", str(model_dir),
            "--prompt", workload.prompt,
            "--max-tokens", str(workload.max_new_tokens),
            "--temp", str(workload.temperature),
            "--seed", str(workload.seed),
        ]

    def _warmup_command(self, model_dir: Path, workload: ExternalWorkload) -> list[str]:
        warm = ExternalWorkload(
            prompt=workload.prompt,
            prompt_tokens=workload.prompt_tokens,
            max_new_tokens=min(16, workload.max_new_tokens),
            context_size=workload.context_size,
            warmup_runs=0,
            measurement_runs=1,
            temperature=workload.temperature,
            top_k=workload.top_k,
            seed=workload.seed,
        )
        return self._command(model_dir, warm)
