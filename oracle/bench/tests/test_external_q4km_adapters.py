from unittest.mock import patch

import pytest

from oracle.bench.external.common import ExternalWorkload, parse_ms_per_token_samples
from oracle.bench.external.llama_cpp import LlamaCppAdapter, LlamaCppVersionMismatch
from oracle.bench.external.mlx_lm import MlxLmAdapter, MlxLmVersionMismatch
from oracle.bench.render.schema import validate_external_cell


def test_common_speed_parser_handles_tok_s_and_ms_token():
    samples = parse_ms_per_token_samples(
        "generation: 91.0 tok/s\n"
        "fallback ms/token: 11.5\n"
        "| test | tg 512 | 139.0 t/s |\n"
        "| qwen35moe 35B.A3B Q4_K - Medium | 20.49 GiB | BLAS,MTL | 6 | tg512 | 90.31 ± 0.42 |\n"
    )
    assert sorted(samples) == pytest.approx(
        sorted([1000.0 / 91.0, 11.5, 1000.0 / 139.0, 1000.0 / 90.31])
    )


def test_llama_cpp_version_check_passes_when_pinned(tmp_path):
    pin = tmp_path / "llama-cpp-version.txt"
    pin.write_text("llama.cpp build 1234\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "llama.cpp build 1234\n"
            stderr = ""
            returncode = 0
        return R()

    with patch("oracle.bench.external.llama_cpp.subprocess.run", side_effect=fake_run):
        LlamaCppAdapter(version_pin_file=pin).assert_version_match()


def test_llama_cpp_version_check_raises_on_mismatch(tmp_path):
    pin = tmp_path / "llama-cpp-version.txt"
    pin.write_text("llama.cpp build 1234\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "llama.cpp build 5678\n"
            stderr = ""
            returncode = 0
        return R()

    with patch("oracle.bench.external.llama_cpp.subprocess.run", side_effect=fake_run):
        with pytest.raises(LlamaCppVersionMismatch):
            LlamaCppAdapter(version_pin_file=pin).assert_version_match()


def test_llama_cpp_adapter_builds_tg512_command_and_schema(tmp_path):
    gguf = tmp_path / "qwen35-q4km.gguf"
    gguf.write_bytes(b"fake")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "llama.cpp version test\n" if "--version" in cmd else "tg 512 91.0 t/s\n"
            stderr = ""
            returncode = 0
        return R()

    workload = ExternalWorkload(prompt="", prompt_tokens=0, max_new_tokens=512, context_size=1024)
    with patch("oracle.bench.external.llama_cpp.subprocess.run", side_effect=fake_run), \
         patch("oracle.bench.external.llama_cpp.time.sleep"):
        cell = LlamaCppAdapter(binary="llama-bench").measure_workload(
            "qwen3.5-35b-a3b", "q4km", gguf, workload
        )

    validate_external_cell(cell)
    assert cell["status"] == "ok"
    assert cell["ms_per_step"] == pytest.approx(1000.0 / 91.0)
    assert cell["workload"]["max_new_tokens"] == 512
    assert "-n" in cell["command"]
    assert "512" in cell["command"]
    assert "-c" not in cell["command"]
    assert cell["command"][cell["command"].index("-p") + 1] == "1"
    assert cell["command"][cell["command"].index("-r") + 1] == "1"
    assert len(cell["samples"]) == workload.measurement_runs
    assert cell["extras"]["version_command"] == ["llama-cli", "--version"]


def test_mlx_lm_version_check_raises_on_mismatch(tmp_path):
    pin = tmp_path / "mlx-lm-version.txt"
    pin.write_text("mlx-lm 0.0.1\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "mlx-lm 0.0.2\n"
            stderr = ""
            returncode = 0
        return R()

    with patch("oracle.bench.external.mlx_lm.subprocess.run", side_effect=fake_run):
        with pytest.raises(MlxLmVersionMismatch):
            MlxLmAdapter(version_pin_file=pin).assert_version_match()


def test_mlx_lm_adapter_records_repeated_generation_samples(tmp_path):
    measurement_outputs = iter(["generation: 100.0 tok/s\n", "generation: 125.0 tok/s\n"])

    def fake_run(cmd, *args, **kw):
        class R:
            stderr = ""
            returncode = 0
        r = R()
        if "--version" in cmd:
            r.stdout = "mlx-lm version test\n"
        elif "--max-tokens" in cmd and cmd[cmd.index("--max-tokens") + 1] == "16":
            r.stdout = "generation: 100.0 tok/s\n"
        else:
            r.stdout = next(measurement_outputs)
        return r

    workload = ExternalWorkload(prompt="", prompt_tokens=0, max_new_tokens=512, context_size=1024, measurement_runs=2)
    with patch("oracle.bench.external.mlx_lm.subprocess.run", side_effect=fake_run), \
         patch("oracle.bench.external.mlx_lm.time.sleep"):
        cell = MlxLmAdapter(python="python3").measure_workload(
            "qwen3.5-35b-a3b", "q4km", tmp_path, workload
        )

    validate_external_cell(cell)
    assert cell["status"] == "ok"
    assert cell["samples"] == pytest.approx([8.0, 10.0])
    assert cell["extras"]["artifact_kind"] == "mlx-model-dir"
