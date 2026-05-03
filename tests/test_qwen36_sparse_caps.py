import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "gfx1100" / "bench_qwen36_sparse_caps.py"
SPEC = importlib.util.spec_from_file_location("bench_qwen36_sparse_caps", SCRIPT)
bench_qwen36_sparse_caps = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(bench_qwen36_sparse_caps)


class Qwen36SparseCapBenchTests(unittest.TestCase):
    def assert_cases_equal(self, actual, expected):
        self.assertEqual(
            [(case.label, case.cap, case.prefetch_mode, case.prefetch_ranks) for case in actual],
            expected,
        )

    def test_parse_prefetch_rank_policy_accepts_none_rank_and_all(self):
        parse = bench_qwen36_sparse_caps.parse_prefetch_rank_policy
        self.assertEqual(parse("none"), ("none", None))
        self.assertEqual(parse("0"), ("none", None))
        self.assertEqual(parse("1"), ("r1", "1"))
        self.assertEqual(parse("4"), ("r4", "4"))
        self.assertEqual(parse("all"), ("all", "all"))

    def test_parse_prefetch_rank_policies_dedupes_by_label(self):
        parse = bench_qwen36_sparse_caps.parse_prefetch_rank_policies
        self.assertEqual(
            parse("none,1,1,all,off"),
            [("none", None), ("r1", "1"), ("all", "all")],
        )

    def test_parse_prefetch_rank_policies_rejects_empty_or_negative_values(self):
        parse = bench_qwen36_sparse_caps.parse_prefetch_rank_policies
        with self.assertRaises(ValueError):
            parse("")
        with self.assertRaises(ValueError):
            parse("-1")

    def test_parse_prefetch_mode_policies_accepts_aliases_and_dedupes(self):
        parse = bench_qwen36_sparse_caps.parse_prefetch_mode_policies
        self.assertEqual(
            parse(
                "disabled,previous-token,previous_token,previous-token-resident,"
                "transition,transition_weighted"
            ),
            [
                ("none", None),
                ("previous-token", "previous-token"),
                ("previous-token-resident", "previous-token-resident"),
                ("transition", "transition"),
            ],
        )
        with self.assertRaises(ValueError):
            parse("unknown")

    def test_build_cases_expands_prefetch_rank_sweep_per_cap(self):
        build_cases = bench_qwen36_sparse_caps.build_cases
        self.assert_cases_equal(
            build_cases([8, 320], None, "none,1,all", "previous-token", None),
            [
                ("dense", None, None, None),
                ("cap8-none", 8, None, None),
                ("cap8-r1", 8, "previous-token", "1"),
                ("cap8-all", 8, "previous-token", "all"),
                ("cap320-none", 320, None, None),
                ("cap320-r1", 320, "previous-token", "1"),
                ("cap320-all", 320, "previous-token", "all"),
            ],
        )

    def test_build_cases_preserves_single_policy_mode(self):
        build_cases = bench_qwen36_sparse_caps.build_cases
        self.assert_cases_equal(
            build_cases([64, 128], None, None, "previous-token-resident", "4"),
            [
                ("dense", None, None, None),
                ("cap64", 64, "previous-token-resident", "4"),
                ("cap128", 128, "previous-token-resident", "4"),
            ],
        )

    def test_build_cases_expands_mode_and_rank_sweeps(self):
        build_cases = bench_qwen36_sparse_caps.build_cases
        self.assert_cases_equal(
            build_cases(
                [320],
                "disabled,previous-token,previous-token-resident,transition",
                "none,1,all",
                None,
                None,
            ),
            [
                ("dense", None, None, None),
                ("cap320-none", 320, None, None),
                ("cap320-previous-token-r1", 320, "previous-token", "1"),
                ("cap320-previous-token-all", 320, "previous-token", "all"),
                (
                    "cap320-previous-token-resident-r1",
                    320,
                    "previous-token-resident",
                    "1",
                ),
                (
                    "cap320-previous-token-resident-all",
                    320,
                    "previous-token-resident",
                    "all",
                ),
                ("cap320-transition-r1", 320, "transition", "1"),
                ("cap320-transition-all", 320, "transition", "all"),
            ],
        )

    def test_fmt_rank_transition_pct_uses_current_rank_denominator(self):
        fmt = bench_qwen36_sparse_caps.fmt_rank_transition_pct
        summary = {
            "observations_by_rank": [10, 5],
            "repeated_previous_rank_by_current_rank": [[3, 2], [1, 4]],
        }
        self.assertEqual(fmt(summary, current_rank=0, previous_rank=0), "30.0%")
        self.assertEqual(fmt(summary, current_rank=0, previous_rank=1), "20.0%")
        self.assertEqual(fmt(summary, current_rank=1, previous_rank=1), "80.0%")
        self.assertEqual(fmt({}, current_rank=0, previous_rank=0), "-")

    def test_transition_summary_formatters_use_derived_fields(self):
        summary = {
            "observations_by_rank": [10, 5],
            "same_rank_repeat_probability_by_rank": [0.3, 0.8],
            "repeated_current_probability_by_previous_rank": [0.4, 0.6],
            "best_transition": {
                "previous_rank": 1,
                "current_rank": 0,
                "probability_by_current_rank": 0.2,
            },
        }
        self.assertEqual(
            bench_qwen36_sparse_caps.fmt_same_rank_repeat_pct(summary, rank=0),
            "30.0%",
        )
        self.assertEqual(
            bench_qwen36_sparse_caps.fmt_previous_rank_reused_pct(summary, previous_rank=1),
            "60.0%",
        )
        self.assertEqual(
            bench_qwen36_sparse_caps.fmt_best_transition(summary),
            "1->0 (20.0%)",
        )

    def test_transition_summary_formatters_fallback_to_matrix(self):
        summary = {
            "observations_by_rank": [10, 5],
            "repeated_previous_rank_by_current_rank": [[3, 2], [1, 4]],
        }
        self.assertEqual(
            bench_qwen36_sparse_caps.fmt_same_rank_repeat_pct(summary, rank=0),
            "30.0%",
        )
        self.assertEqual(
            bench_qwen36_sparse_caps.fmt_previous_rank_reused_pct(summary, previous_rank=0),
            "40.0%",
        )
        self.assertEqual(bench_qwen36_sparse_caps.fmt_best_transition({}), "-")


if __name__ == "__main__":
    unittest.main()
