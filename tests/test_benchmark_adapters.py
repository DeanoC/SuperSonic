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
        self.llama_log = (FIXTURES / "llama-cpp-peer-run.log").read_text(encoding="utf-8")

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
            sampling_seed=7,
        )
        argv = self.adapters.build_command(self.supersonic, self.mtp_case, inputs)
        self.assertIn("--chat", argv)
        self.assertIn("--speculative-decode", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "3")
        self.assertEqual(argv[argv.index("--context-size") + 1], "2048")
        self.assertEqual(argv[argv.index("--sampling-seed") + 1], "7")

    def test_build_command_fails_closed_for_out_of_scope_engine_or_mode(self):
        with self.assertRaisesRegex(ValueError, "case.*engine|outside|scope"):
            self.adapters.build_command(self.llama_cpp, self.mtp_case, self.inputs)

        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.adapters.build_command(
                self.llama_cpp,
                replace(self.llama_case, mode="mtp", engines=("llama-cpp",)),
                self.inputs,
            )

    def test_llama_cpp_argv_uses_one_shot_server_adapter_and_peer_artifact(self):
        argv = self.adapters.build_command(self.llama_cpp, self.llama_case, self.inputs)
        self.assertEqual(argv[0], "tools/llama-cpp-peer.py")
        self.assertEqual(argv[argv.index("--server-binary") + 1], "llama-server")
        self.assertNotIn(str(self.inputs.artifact), argv)
        self.assertIn(str(self.inputs.peer_artifact), argv)
        self.assertNotIn("--chat", argv)
        self.assertIn("--max-new-tokens", argv)
        self.assertIn("--seed", argv)

    def test_llama_cpp_chat_argv_is_explicit_single_turn_conversation(self):
        inputs = self.adapters.AdapterInputs(
            model_dir=self.inputs.model_dir,
            artifact=self.inputs.artifact,
            peer_artifact=self.inputs.peer_artifact,
            chat=True,
            context_size=4096,
            sampling_seed=7,
        )
        argv = self.adapters.build_command(self.llama_cpp, self.llama_case, inputs)
        self.assertIn("--chat", argv)
        self.assertEqual(argv[argv.index("--context-size") + 1], "4096")
        self.assertEqual(argv[argv.index("--seed") + 1], "7")

    def test_quality_commands_honor_eos_instead_of_forcing_the_token_cap(self):
        inputs = replace(self.inputs, chat=True, fixed_token_count=False)

        supersonic = self.adapters.build_command(self.supersonic, self.supersonic_case, inputs)
        peer = self.adapters.build_command(self.llama_cpp, self.llama_case, inputs)

        self.assertNotIn("--ignore-eos", supersonic)
        self.assertIn("--honor-eos", peer)

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

    def test_supersonic_parser_combines_real_split_stdout_and_stderr(self):
        stdout = '[generated_json] "ready"\n[tokens] 2232\n'
        stderr = (
            "[result] prompt_tokens=19 generated_tokens=1 decode_ms=49 ms_per_tok=49\n"
            "[stage-timings] steps=1 total_native_decode_ms=48.244\n"
        )

        parsed = self.adapters.parse_output("supersonic", stdout, stderr)

        self.assertEqual(parsed.generated_text, "ready")
        self.assertEqual(parsed.prompt_tokens, 19)
        self.assertEqual(parsed.generated_tokens, 1)

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
        parsed = self.adapters.parse_output(
            "llama-cpp",
            "",
            self.llama_log,
        )
        self.assertEqual(parsed.generated_text, " world")
        self.assertEqual(parsed.token_ids, None)
        self.assertEqual(parsed.decode_ms, 12.0)

    def test_duplicate_result_line_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.adapters.parse_output("supersonic", self.supersonic_log + self.supersonic_log)

    def test_llama_cpp_normalized_timing_must_be_complete_and_consistent(self):
        with self.assertRaisesRegex(ValueError, "exactly.*normalized peer fields"):
            self.adapters.parse_output(
                "llama-cpp",
                self.llama_log.replace(',"prompt_tokens":7', ""),
            )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.adapters.parse_output(
                "llama-cpp",
                self.llama_log.replace('"decode_ms":12.0', '"decode_ms":13.0'),
            )

    def test_non_finite_negative_and_inconsistent_values_fail(self):
        bad_logs = (
            self.supersonic_log.replace("decode_ms=9", "decode_ms=nan"),
            self.supersonic_log.replace("decode_ms=9", "decode_ms=-1"),
            self.supersonic_log.replace("generated_tokens=3", "generated_tokens=4"),
            self.supersonic_log.replace('[generated_json] " world"\n', ""),
            self.llama_log.replace('"decode_ms":12.0', '"decode_ms":"nan"'),
        )
        for log in bad_logs:
            with self.assertRaises(ValueError):
                engine_name = "llama-cpp" if "llama_perf_context_print" in log else "supersonic"
                self.adapters.parse_output(engine_name, log)


if __name__ == "__main__":
    unittest.main()
