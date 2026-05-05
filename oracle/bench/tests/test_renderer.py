"""Renderer is a pure function: same JSON in → same markdown out."""
from pathlib import Path

from oracle.bench.render.markdown import (
    render_perf_table, replace_autogen_zone,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "run_minimal"
GOLDEN_PERF = (Path(__file__).parent / "fixtures" / "golden_perf_fragment.md").read_text()


def test_perf_table_matches_golden():
    rendered = render_perf_table(FIXTURE_DIR / "perf")
    assert rendered.strip() == GOLDEN_PERF.strip()


def test_autogen_zone_replaces_only_the_zone():
    original = (
        "# Performance\n\nProse goes here.\n\n"
        "<!-- AUTOGEN BELOW: hipfire-comparison -->\n"
        "OLD CONTENT THAT GETS REPLACED\n"
        "<!-- AUTOGEN END: hipfire-comparison -->\n\n"
        "Trailing prose stays.\n"
    )
    new_content = "NEW CONTENT"
    out = replace_autogen_zone(original, "hipfire-comparison", new_content)
    assert "Prose goes here." in out
    assert "Trailing prose stays." in out
    assert "OLD CONTENT" not in out
    assert "NEW CONTENT" in out
    assert "<!-- AUTOGEN BELOW: hipfire-comparison -->" in out
    assert "<!-- AUTOGEN END: hipfire-comparison -->" in out


def test_autogen_zone_inserted_when_absent():
    original = "# Empty doc\n"
    out = replace_autogen_zone(original, "hipfire-comparison", "NEW")
    assert "<!-- AUTOGEN BELOW: hipfire-comparison -->" in out
    assert "NEW" in out


GOLDEN_HIPFIRE = (Path(__file__).parent / "fixtures" / "golden_hipfire_fragment.md").read_text()


def test_hipfire_table_matches_golden():
    from oracle.bench.render.markdown import render_external_comparison_table
    rendered = render_external_comparison_table(
        FIXTURE_DIR / "perf",
        FIXTURE_DIR / "external" / "hipfire",
        engine="hipfire",
    )
    assert rendered.strip() == GOLDEN_HIPFIRE.strip()
