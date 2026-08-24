import importlib
from dataclasses import replace
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "benchmark_fixtures"


def load_adapters_module():
    try:
        return importlib.import_module("tools.benchmark.adapters")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.adapters is absent") from exc


def load_manifest_module():
    try:
        return importlib.import_module("tools.benchmark.manifest")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.manifest is absent") from exc


class BenchmarkAdapterTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.adapters = load_adapters_module()
        self.manifest = load_manifest_module()
        self.supersonic = self.manifest.load_engine("supersonic")
        self.llama_cpp = self.manifest.load_engine("llama-cpp")
        self.full = self.manifest.load_suite("full")
        self.supersonic_case = next(case for case in self.full.performance_cases if case.mode == "ordinary")
        self.mtp_case = next(case for case in self.full.performance_cases if case.mode == "mtp")
        self.llama_case = next(
            case for case in self.full.performance_cases if case.mode == "ordinary" and "llama-cpp" in case.engines
        )
        self.inputs = self.adapters.AdapterInputs(
            model_dir=Path("/models/Qwen3.8-27B"),
            artifact=Path("/artifacts/qwen38-gqh.gguf"),
            peer_artifact=Path("/peers/qwen38-llama.gguf"),
            chat=False,
            device=0,
        )
        self.supersonic_log = (FIXTURES / "supersonic-run.log").read_text(encoding="utf-8")
        self.llama_log = (FIXTURES / "llama-cpp-run.log").read_text(encoding="utf-8")

    def test_supersonic_argv_uses_public_contract(self):
        argv = self.adapters.build_command(self.supersonic, self.supersonic_case, self.inputs)
        self.assertEqual(argv[0], "./target/release/supersonic")
        self.assertIn("qwen3.8-27b", argv)
        self.assertIn("--emit-generated-json", argv)
        self.assertIn("--emit-stage-timings", argv)
        self.assertIn("--ignore-eos", argv)
        self.assertEqual(argv.count("--model-dir"), 1)
        self.assertEqual(argv.count("--gguf-file"), 1)
        self.assertNotIn("|", argv)

    def test_supersonic_chat_and_mtp_flags_follow_case_and_inputs(self):
        inputs = self.adapters.AdapterInputs(
            model_dir=self.inputs.model_dir,
            artifact=self.inputs.artifact,
            peer_artifact=self.inputs.peer_artifact,
            chat=True,
            device=3,
            context_size=2048,
        )
        argv = self.adapters.build_command(self.supersonic, self.mtp_case, inputs)
        self.assertIn("--chat", argv)
        self.assertIn("--speculative-decode", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "3")
        self.assertEqual(argv[argv.index("--context-size") + 1], "2048")

    def test_build_command_fails_closed_for_out_of_scope_engine_or_mode(self):
        with self.assertRaisesRegex(ValueError, "case.*engine|outside|scope"):
            self.adapters.build_command(self.llama_cpp, self.mtp_case, self.inputs)

        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.adapters.build_command(
                self.llama_cpp,
                replace(self.llama_case, mode="mtp", engines=("llama-cpp",)),
                self.inputs,
            )

    def test_llama_cpp_argv_uses_peer_artifact_and_ordinary_mode(self):
        argv = self.adapters.build_command(self.llama_cpp, self.llama_case, self.inputs)
        self.assertEqual(argv[0], "llama-cli")
        self.assertNotIn(str(self.inputs.artifact), argv)
        self.assertIn(str(self.inputs.peer_artifact), argv)
        self.assertIn("--perf", argv)
        self.assertIn("--no-display-prompt", argv)
        self.assertIn("--no-conversation", argv)
        self.assertNotIn("--conversation", argv)
        self.assertIn("--ignore-eos", argv)
        self.assertIn("--temp", argv)
        self.assertIn("--seed", argv)

    def test_llama_cpp_chat_argv_is_explicit_single_turn_conversation(self):
        inputs = self.adapters.AdapterInputs(
            model_dir=self.inputs.model_dir,
            artifact=self.inputs.artifact,
            peer_artifact=self.inputs.peer_artifact,
            chat=True,
            context_size=4096,
        )
        argv = self.adapters.build_command(self.llama_cpp, self.llama_case, inputs)
        self.assertIn("--conversation", argv)
        self.assertIn("--single-turn", argv)
        self.assertNotIn("--no-conversation", argv)
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "4096")

    def test_supersonic_parser_extracts_generated_text_tokens_and_timings(self):
        parsed = self.adapters.parse_output("supersonic", self.supersonic_log)
        self.assertEqual(parsed.engine_name, "supersonic")
        self.assertIsNone(parsed.engine_version)
        self.assertEqual(parsed.generated_text, " world")
        self.assertEqual(parsed.token_ids, (42, 43, 44))
        self.assertEqual(parsed.prompt_tokens, 7)
        self.assertEqual(parsed.generated_tokens, 3)
        self.assertEqual(parsed.decode_ms, 9.0)
        self.assertEqual(parsed.ms_per_tok, 3.0)
        self.assertAlmostEqual(parsed.tokens_per_second, 333.3333333333333)

    def test_llama_cpp_parser_keeps_version_identity(self):
        parsed = self.adapters.parse_output("llama-cpp", self.llama_log)
        self.assertEqual(parsed.engine_version, self.llama_cpp.pinned_version)
        self.assertEqual(parsed.generated_text, " world")
        self.assertIsNone(parsed.token_ids)
        self.assertEqual(parsed.prompt_tokens, 7)
        self.assertEqual(parsed.generated_tokens, 3)
        self.assertEqual(parsed.decode_ms, 12.0)
        self.assertEqual(parsed.ms_per_tok, 4.0)
        self.assertAlmostEqual(parsed.tokens_per_second, 250.0)

    def test_llama_cpp_parser_combines_stdout_and_stderr_streams(self):
        text_line = "text emitted by llama"
        parsed = self.adapters.parse_output(
            "llama-cpp",
            text_line,
            self.llama_log.split("\n", 1)[1],
        )
        self.assertEqual(parsed.generated_text, text_line)
        self.assertEqual(parsed.token_ids, None)
        self.assertEqual(parsed.decode_ms, 12.0)

    def test_duplicate_result_line_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.adapters.parse_output("supersonic", self.supersonic_log + self.supersonic_log)

    def test_llama_cpp_raw_timing_lines_must_be_complete_and_consistent(self):
        with self.assertRaisesRegex(ValueError, "exactly one.*prompt eval time"):
            self.adapters.parse_output(
                "llama-cpp",
                self.llama_log.replace("prompt eval time", "prefill eval time"),
            )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.adapters.parse_output(
                "llama-cpp",
                self.llama_log.replace("eval time =      12.00 ms /     3 runs", "eval time =      13.00 ms /     3 runs"),
            )

    def test_non_finite_negative_and_inconsistent_values_fail(self):
        bad_logs = (
            self.supersonic_log.replace("decode_ms=9", "decode_ms=nan"),
            self.supersonic_log.replace("decode_ms=9", "decode_ms=-1"),
            self.supersonic_log.replace("generated_tokens=3", "generated_tokens=4"),
            self.supersonic_log.replace('[generated_json] " world"\n', ""),
            self.llama_log.replace("eval time =      12.00 ms", "eval time =      nan ms"),
        )
        for log in bad_logs:
            with self.assertRaises(ValueError):
                engine_name = "llama-cpp" if "llama_perf_context_print" in log else "supersonic"
                self.adapters.parse_output(engine_name, log)


if __name__ == "__main__":
    unittest.main()
