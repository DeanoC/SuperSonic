"""Golden-prompt scoring math. No GPU."""
from pathlib import Path
from unittest.mock import patch

import pytest

from oracle.bench.golden import (
    GoldenGenerationError, GoldenRequest, _generate,
    score_pair, aggregate_golden_results,
)


def test_exact_match_scores_1():
    s = score_pair("hello world", "hello world")
    assert s["exact_match"] == 1.0
    assert s["chrf"] == 1.0


def test_total_mismatch_scores_low():
    s = score_pair("hello world", "completely different output here xyz")
    assert s["exact_match"] == 0.0
    assert s["chrf"] < 0.5


def test_aggregate_returns_means_and_failure_count():
    per_prompt = [
        {"prompt_id": "a", "exact_match": 1.0, "chrf": 1.0},
        {"prompt_id": "b", "exact_match": 0.0, "chrf": 0.4},
        {"prompt_id": "c", "exact_match": 0.0, "chrf": 0.05},
    ]
    out = aggregate_golden_results(per_prompt, chrf_threshold=0.20)
    assert abs(out["exact_match_mean"] - (1.0/3.0)) < 1e-6
    assert abs(out["chrf_mean"] - (1.0 + 0.4 + 0.05) / 3) < 1e-6
    assert out["below_threshold_count"] == 1   # only "c" is < 0.20
    assert out["below_threshold_ids"] == ["c"]


def _fake_run(returncode=0, stdout="", stderr=""):
    def runner(cmd, *args, **kw):
        class R:
            pass
        r = R()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r
    return runner


def test_generate_raises_when_supersonic_exits_nonzero():
    req = GoldenRequest(binary=Path("/bin/false"), model="m", model_dir=Path("/x"), quant="bf16")
    with patch("oracle.bench.golden.subprocess.run",
               side_effect=_fake_run(returncode=1, stderr="oom\n")):
        with pytest.raises(GoldenGenerationError) as ei:
            _generate(req, {"id": "p1", "prompt": "x", "max_new_tokens": 1})
        assert "exited with code 1" in str(ei.value)


def test_generate_raises_when_only_status_lines_returned():
    req = GoldenRequest(binary=Path("/bin/false"), model="m", model_dir=Path("/x"), quant="bf16")
    stdout_only_status = "[gpu] backend=HIP\n[tokenizer] prompt_tokens=1\n"
    with patch("oracle.bench.golden.subprocess.run",
               side_effect=_fake_run(returncode=0, stdout=stdout_only_status)):
        with pytest.raises(GoldenGenerationError) as ei:
            _generate(req, {"id": "p2", "prompt": "x", "max_new_tokens": 1})
        assert "no generation text" in str(ei.value)


def test_generate_returns_filtered_text_when_runner_succeeds():
    req = GoldenRequest(binary=Path("/bin/true"), model="m", model_dir=Path("/x"), quant="bf16")
    stdout = ("[gpu] backend=HIP\n"
              "Hello,\n"
              "I am a model.\n"
              "[tokens] 1 2 3\n"
              "[result] ms_per_step=8\n")
    with patch("oracle.bench.golden.subprocess.run",
               side_effect=_fake_run(returncode=0, stdout=stdout)):
        text = _generate(req, {"id": "p3", "prompt": "x", "max_new_tokens": 4})
        assert text == "Hello,\nI am a model."
