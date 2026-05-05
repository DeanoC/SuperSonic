"""Math sanity for perplexity aggregation. No GPU."""
import math

from oracle.bench.perplexity import aggregate_ppl_from_chunks


def test_aggregate_ppl_from_single_chunk():
    chunks = [{"nll": 10.0, "tokens": 5}]
    out = aggregate_ppl_from_chunks(chunks)
    assert math.isclose(out["avg_nll"], 2.0, rel_tol=1e-9)
    assert math.isclose(out["ppl"], math.exp(2.0), rel_tol=1e-9)
    assert out["tokens"] == 5


def test_aggregate_ppl_from_multiple_chunks():
    chunks = [
        {"nll": 10.0, "tokens": 5},
        {"nll": 6.0, "tokens": 3},
    ]
    out = aggregate_ppl_from_chunks(chunks)
    expected_avg_nll = (10.0 + 6.0) / (5 + 3)
    assert math.isclose(out["avg_nll"], expected_avg_nll, rel_tol=1e-9)
    assert math.isclose(out["ppl"], math.exp(expected_avg_nll), rel_tol=1e-9)
    assert out["tokens"] == 8


def test_aggregate_ppl_empty_chunks_raises():
    import pytest
    with pytest.raises(ValueError):
        aggregate_ppl_from_chunks([])
