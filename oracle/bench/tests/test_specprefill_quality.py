from oracle.bench.specprefill_quality import (
    DEFAULT_KEEP_RATIOS,
    cossim,
    parse_keep_ratios,
    quality_cell,
    threshold_failures,
)


def test_cossim_identity():
    assert abs(cossim([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9


def test_default_keep_ratios_are_conservative_lane_only():
    assert DEFAULT_KEEP_RATIOS == "0.75"
    assert parse_keep_ratios(DEFAULT_KEEP_RATIOS) == [0.75]


def test_quality_cell_aggregates_rows(monkeypatch, tmp_path):
    from oracle.bench import specprefill_quality as mod

    def fake_run(binary, model_dir, prompt, keep_ratio, draft_dir, timeout):
        del binary, model_dir, prompt, draft_dir, timeout
        if keep_ratio is None:
            return mod.RunResult([0.0, 2.0, 1.0], [1], "-")
        return mod.RunResult([0.0, 1.5, 1.0], [1], "2/3")

    monkeypatch.setattr(mod, "run_supersonic", fake_run)
    prompts = [mod.PromptCase("p0", "hello")]
    cell = quality_cell(
        tmp_path / "supersonic",
        tmp_path / "model",
        tmp_path / "draft",
        prompts,
        0.50,
        3,
    )

    assert cell["quant"] == "int4-spec050"
    assert cell["metric"] == "argmax_match_rate"
    assert cell["value"] == 1.0
    assert cell["extras"]["threshold_pass"] is True
    assert cell["extras"]["per_prompt"][0]["kept"] == "2/3"


def test_threshold_failures_report_each_failed_bar():
    cell = {
        "quant": "int4-spec075",
        "value": 0.5,
        "extras": {
            "cosine_min": 0.5,
            "top5_overlap_min": 1,
            "thresholds": {
                "argmax_match_rate_min": 1.0,
                "cosine_min": 0.9,
                "top5_overlap_min": 3,
            },
        },
    }
    failures = threshold_failures(cell)
    assert len(failures) == 3
    assert "argmax_match_rate" in failures[0]
