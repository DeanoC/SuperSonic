"""Math sanity for perplexity aggregation. No GPU."""
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from oracle.bench.perplexity import (
    PerplexityChunkError, PerplexityRequest,
    aggregate_ppl_from_chunks, score_perplexity,
)


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


def test_score_perplexity_raises_on_missing_teacher_forced_json():
    req = PerplexityRequest(
        binary=Path("/bin/false"),
        model="qwen3.5-0.8b",
        model_dir=Path("/x"),
        quant="bf16",
        dataset="wikitext2",
        contexts=64,
        num_chunks=2,
    )
    # Stub _load_chunks so we don't hit HF; stub the runner so its output is
    # missing the [teacher_forced_json] line. The chunk error must propagate.
    with patch("oracle.bench.perplexity._load_chunks",
               return_value=["chunk a", "chunk b"]), \
         patch("oracle.bench.perplexity._run_supersonic_teacher_forced",
               return_value="[gpu] backend=HIP\n[result] ms_per_step=8\n"):
        with pytest.raises(PerplexityChunkError) as ei:
            score_perplexity(req)
        assert "missing [teacher_forced_json]" in str(ei.value)
        assert "chunk 0" in str(ei.value)
