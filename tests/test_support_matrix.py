import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check-support-matrix.py"
SPEC = importlib.util.spec_from_file_location("check_support_matrix", SCRIPT)
support_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support_matrix
SPEC.loader.exec_module(support_matrix)


class SupportMatrixTests(unittest.TestCase):
    def test_lane_key_distinguishes_flm_model_source(self):
        base = {
            "backend": "hip",
            "arch": "gfx1100",
            "models": ["qwen3.6-35b-a3b"],
            "quants": ["int4"],
        }
        flm = {
            **base,
            "model_sources": ["flm"],
        }

        self.assertEqual(
            support_matrix.model_sources_for_entry("base", base, []),
            ["hf-snapshot"],
        )
        self.assertNotEqual(
            support_matrix.lane_key_for_entry(base),
            support_matrix.lane_key_for_entry(flm),
        )


if __name__ == "__main__":
    unittest.main()
