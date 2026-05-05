"""Golden-prompt scoring math. No GPU."""
from oracle.bench.golden import (
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
