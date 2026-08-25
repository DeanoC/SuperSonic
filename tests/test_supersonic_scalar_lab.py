from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]


def load_scalar_lab():
    path = ROOT / "tools" / "supersonic-scalar-lab.py"
    if not path.is_file():
        raise AssertionError("tools/supersonic-scalar-lab.py is absent")
    spec = importlib.util.spec_from_file_location("supersonic_scalar_lab", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["supersonic_scalar_lab"] = module
    spec.loader.exec_module(module)
    return module


class SupersonicScalarLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lab = load_scalar_lab()

    def test_command_is_source_fixed_greedy_and_propagates_mode_and_chat(self):
        command = self.lab.build_command(
            binary=Path("/repo/target/release/examples/scalar_head_lab"),
            model_dir=Path("/models/qwen38"),
            artifact=Path("/models/qwen38.gguf"),
            prompt="Hello",
            max_new_tokens=32,
            device=0,
            mode="mtp",
            chat=True,
            honor_eos=False,
        )

        self.assertEqual(command[0], "/repo/target/release/examples/scalar_head_lab")
        self.assertEqual(command[command.index("--model-dir") + 1], "/models/qwen38")
        self.assertEqual(command[command.index("--artifact") + 1], "/models/qwen38.gguf")
        self.assertEqual(command[command.index("--mode") + 1], "mtp")
        self.assertIn("--chat", command)
        self.assertIn("--ignore-eos", command)
        forbidden = {"--route", "--temperature", "--top-k", "--top-p", "--sampling-seed"}
        self.assertTrue(forbidden.isdisjoint(command), command)

    def test_normalization_requires_one_exact_consistent_scalar_record(self):
        payload = {
            "decode_ms": 12.0,
            "engine_name": "supersonic-scalar-lab",
            "engine_version": "scalar-head-lab-v1",
            "generated_text": "answer",
            "generated_tokens": 3,
            "lm_head_ms": 3.0,
            "ms_per_tok": 4.0,
            "prompt_tokens": 7,
            "token_ids": [11, 12, 13],
            "timed_decode_steps": 2,
            "tokens_per_second": 250.0,
        }
        stdout = "[supersonic_json] " + json.dumps(payload, sort_keys=True, separators=(",", ":"))

        self.assertEqual(self.lab.normalize_output(stdout, ""), payload)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.lab.normalize_output(stdout + "\n" + stdout, "")
        broken = dict(payload, decode_ms=99.0)
        with self.assertRaisesRegex(ValueError, "timing"):
            self.lab.normalize_output("[supersonic_json] " + json.dumps(broken), "")
        wrong_route = dict(payload, engine_name="supersonic-wmma")
        with self.assertRaisesRegex(ValueError, "engine_name"):
            self.lab.normalize_output("[supersonic_json] " + json.dumps(wrong_route), "")
        bad_head = dict(payload, lm_head_ms=0.0)
        with self.assertRaisesRegex(ValueError, "lm_head_ms"):
            self.lab.normalize_output("[supersonic_json] " + json.dumps(bad_head), "")

    def test_cli_rejects_route_and_route_environment(self):
        with self.assertRaises(SystemExit):
            self.lab.build_parser().parse_args(
                [
                    "--model-dir", "/models/qwen38",
                    "--artifact", "/models/qwen38.gguf",
                    "--prompt", "Hello",
                    "--max-new-tokens", "2",
                    "--mode", "ordinary",
                    "--route", "wmma",
                ]
            )
        with self.assertRaisesRegex(ValueError, "route environment"):
            self.lab.reject_route_environment({"SUPERSONIC_SCALAR_HEAD_ROUTE": "wmma"})

    def test_version_is_available_without_model_arguments(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(self.lab.main(["--version"]), 0)
        self.assertEqual(stdout.getvalue(), "supersonic-scalar-lab scalar-head-lab-v1\n")

    def test_script_version_is_available_without_model_arguments(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "supersonic-scalar-lab.py"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "supersonic-scalar-lab scalar-head-lab-v1")

    def test_timeout_kills_the_child_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / "child.pid"
            fake = root / "slow-scalar"
            fake.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env python3
                    import os
                    from pathlib import Path
                    import subprocess
                    import sys
                    import time
                    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                    Path({str(pid_file)!r}).write_text(str(child.pid))
                    time.sleep(60)
                    """
                ),
                encoding="utf-8",
            )
            fake.chmod(0o755)
            command = (str(fake),)

            with self.assertRaisesRegex(TimeoutError, "timed out"):
                self.lab.run_command(command, timeout_seconds=0.2)

            child_pid = int(pid_file.read_text(encoding="utf-8"))
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text(encoding="utf-8").split()[2]
            except FileNotFoundError:
                return
            self.assertEqual(state, "Z", "timed-out descendant must not remain runnable")


if __name__ == "__main__":
    unittest.main()
