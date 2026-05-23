import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "probe_qwen36_mtp_acceptance.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen36_mtp_acceptance", SCRIPT)
probe_qwen36_mtp_acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = probe_qwen36_mtp_acceptance
SPEC.loader.exec_module(probe_qwen36_mtp_acceptance)


ACCEPTANCE_OUTPUT = """
[qwen36-mtp-acceptance] mode=batched steps=4 drafted_tokens=10 accepted_tokens=7 acceptance_rate=0.700000 emitted_tokens=11 emitted_per_step=2.750000 base_steps=14 replay_steps=2 target_steps_per_emitted=1.454545 full_accept_steps=2 zero_accept_steps=1 max_accept=3
"""

POLICY_OUTPUT = """
Error: Qwen3.6-35B-A3B Metal v1 does not wire the MTP/speculative decode path yet.
"""


class Qwen36MtpAcceptanceProbeTests(unittest.TestCase):
    def test_parse_mtp_acceptance_summary(self):
        row = probe_qwen36_mtp_acceptance.parse_mtp_acceptance(ACCEPTANCE_OUTPUT)

        self.assertEqual(row["mode"], "batched")
        self.assertEqual(row["steps"], 4)
        self.assertEqual(row["drafted_tokens"], 10)
        self.assertEqual(row["accepted_tokens"], 7)
        self.assertAlmostEqual(row["acceptance_rate"], 0.7)
        self.assertAlmostEqual(row["target_steps_per_emitted"], 1.454545)

    def test_build_report_marks_measured_output(self):
        report = probe_qwen36_mtp_acceptance.build_report(
            ACCEPTANCE_OUTPUT,
            0,
            ["supersonic"],
            1.25,
            "hip",
            batched_spec_verify=True,
        )

        self.assertEqual(report["schema"], probe_qwen36_mtp_acceptance.SCHEMA)
        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["mode"], "batched")
        self.assertFalse(report["policy_blocked"])
        self.assertEqual(report["acceptance"]["base_steps"], 14)

    def test_build_report_marks_metal_policy_block(self):
        report = probe_qwen36_mtp_acceptance.build_report(
            POLICY_OUTPUT,
            1,
            ["supersonic"],
            0.02,
            "metal",
            batched_spec_verify=False,
        )

        self.assertEqual(report["status"], "policy_blocked")
        self.assertTrue(report["policy_blocked"])
        self.assertEqual(report["acceptance"], {})
        md = probe_qwen36_mtp_acceptance.render_markdown(report)
        self.assertIn("policy-blocked", md)

    def test_missing_acceptance_success_is_not_measured(self):
        report = probe_qwen36_mtp_acceptance.build_report(
            "Generated ids: [1, 2]",
            0,
            ["supersonic"],
            0.1,
            "hip",
            batched_spec_verify=False,
        )

        self.assertEqual(report["status"], "missing_acceptance")


if __name__ == "__main__":
    unittest.main()
