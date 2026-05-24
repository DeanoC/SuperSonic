import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "sweep_qwen36_mtp_acceptance.py"
SPEC = importlib.util.spec_from_file_location("sweep_qwen36_mtp_acceptance", SCRIPT)
sweep_qwen36_mtp_acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sweep_qwen36_mtp_acceptance
SPEC.loader.exec_module(sweep_qwen36_mtp_acceptance)


class Qwen36MtpAcceptanceSweepTests(unittest.TestCase):
    def test_build_summary_aggregates_measured_rows_only(self):
        rows = [
            {
                "status": "measured",
                "wall_seconds": 10.0,
                "acceptance": {
                    "drafted_tokens": 2,
                    "accepted_tokens": 1,
                    "emitted_tokens": 3,
                    "base_steps": 3,
                    "replay_steps": 0,
                    "full_accept_steps": 1,
                    "zero_accept_steps": 1,
                },
            },
            {
                "status": "measured",
                "wall_seconds": 12.0,
                "acceptance": {
                    "drafted_tokens": 4,
                    "accepted_tokens": 3,
                    "emitted_tokens": 5,
                    "base_steps": 5,
                    "replay_steps": 1,
                    "full_accept_steps": 2,
                    "zero_accept_steps": 0,
                },
            },
            {
                "status": "policy_blocked",
                "wall_seconds": 0.1,
                "acceptance": {},
            },
        ]

        summary = sweep_qwen36_mtp_acceptance.build_summary(rows)

        self.assertEqual(summary["prompt_count"], 3)
        self.assertEqual(summary["measured_count"], 2)
        self.assertEqual(summary["status_counts"]["policy_blocked"], 1)
        self.assertEqual(summary["drafted_tokens"], 6)
        self.assertEqual(summary["accepted_tokens"], 4)
        self.assertAlmostEqual(summary["acceptance_rate"], 4 / 6)
        self.assertEqual(summary["target_steps"], 9)
        self.assertAlmostEqual(summary["target_steps_per_emitted"], 9 / 8)
        self.assertFalse(summary["promotion_gate"]["passed"])
        self.assertIn("policy_blocked_rows", summary["promotion_gate"]["failures"])
        self.assertIn(
            "target_steps_per_emitted_above_threshold",
            summary["promotion_gate"]["failures"],
        )

    def test_promotion_gate_can_pass_when_all_prompts_are_efficient(self):
        rows = [
            {
                "status": "measured",
                "wall_seconds": 10.0,
                "acceptance": {
                    "drafted_tokens": 4,
                    "accepted_tokens": 3,
                    "emitted_tokens": 8,
                    "base_steps": 6,
                    "replay_steps": 1,
                },
            }
        ]

        summary = sweep_qwen36_mtp_acceptance.build_summary(rows)

        self.assertTrue(summary["promotion_gate"]["passed"])

    def test_render_markdown_includes_prompt_rows(self):
        rows = [
            {
                "prompt_id": "coding",
                "prompt": "Write code.",
                "backend": "metal",
                "mode": "sequential",
                "status": "measured",
                "returncode": 0,
                "wall_seconds": 24.0,
                "command": ["supersonic"],
                "policy_blocked": False,
                "acceptance": {
                    "drafted_tokens": 2,
                    "accepted_tokens": 1,
                    "acceptance_rate": 0.5,
                    "emitted_tokens": 3,
                    "base_steps": 3,
                    "replay_steps": 0,
                    "target_steps_per_emitted": 1.0,
                },
                "metal_profile": {
                    "summary": {"total_ms": 12.0},
                    "entries": [{"op": "qwen36_mtp_verify", "total_ms": 10.0}],
                },
                "hal_profile": {"summary": {"total_ms": 3.0}, "entries": []},
            }
        ]
        report = sweep_qwen36_mtp_acceptance.build_report(
            rows,
            "metal",
            "sequential",
            "smoke",
            {"SUPERSONIC_BACKENDS": "metal"},
        )

        md = sweep_qwen36_mtp_acceptance.render_markdown(report)

        self.assertIn("Qwen3.6 MTP Acceptance Sweep", md)
        self.assertIn("promotion_gate_passed", md)
        self.assertIn(
            "| coding | measured | 2 | 1 | 50.0% | 3 | 1.000 | qwen36_mtp_verify | 10.000 | 3.000 | 24.0 |",
            md,
        )

    def test_select_prompts_prefers_custom_prompts(self):
        args = type(
            "Args",
            (),
            {
                "prompt": ["one", "two"],
                "prompt_set": "smoke",
            },
        )()

        prompts = sweep_qwen36_mtp_acceptance.select_prompts(args)

        self.assertEqual(prompts, [("custom_1", "one"), ("custom_2", "two")])


if __name__ == "__main__":
    unittest.main()
