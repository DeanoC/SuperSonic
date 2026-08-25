import importlib.util
from pathlib import Path
import unittest

from tools.benchmark.adapters import ParsedOutput


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "generate-scalar-mtp-goldens.py"


def load_generator():
    if not SCRIPT.is_file():
        raise AssertionError("scalar MTP golden generator is absent")
    spec = importlib.util.spec_from_file_location("generate_scalar_mtp_goldens", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScalarMtpGoldenGeneratorTests(unittest.TestCase):
    def output(self, tokens):
        return ParsedOutput(
            engine_name="supersonic-scalar-lab",
            engine_version="scalar-head-lab-v1",
            generated_text="reviewed text",
            token_ids=tuple(tokens),
            prompt_tokens=25,
            generated_tokens=len(tokens),
            decode_ms=8.0,
            ms_per_tok=1.0,
            tokens_per_second=1000.0,
        )

    def test_repeated_fresh_process_outputs_must_agree_before_freezing(self):
        generator = load_generator()

        reviewed = generator.review_repeated_outputs(
            [self.output((40, 4021)), self.output((40, 4021))],
            max_new_tokens=8,
        )

        self.assertEqual(reviewed, {"token_ids": [40, 4021], "generated_text": "reviewed text"})
        with self.assertRaisesRegex(ValueError, "independent runs"):
            generator.review_repeated_outputs(
                [self.output((40, 4021)), self.output((40, 4022))],
                max_new_tokens=8,
            )


if __name__ == "__main__":
    unittest.main()
