import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check-qwen38-artifacts.py"
ARTIFACT_ENV = (
    "SUPERSONIC_GQH_GGUF",
    "SUPERSONIC_QWEN38_MODEL_DIR",
    "SUPERSONIC_GQH_8192_GGUF",
    "SUPERSONIC_REQUIRE_GQH_ARTIFACTS",
)


class Qwen38ArtifactPreflightTests(unittest.TestCase):
    def run_checker(self, env, *args):
        clean_env = os.environ.copy()
        for name in ARTIFACT_ENV:
            clean_env.pop(name, None)
        clean_env.update(env)
        return subprocess.run(
            [sys.executable, str(CHECKER), *args],
            cwd=ROOT,
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def write_model_fixture(root):
        model_dir = root / "qwen38-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return model_dir

    @staticmethod
    def write_gguf_fixture(root, name="qwen38.gqh.gguf"):
        path = root / name
        path.write_bytes(b"GGUF\x03\x00\x00\x00")
        return path

    def test_missing_required_environment_names_every_required_item(self):
        result = self.run_checker({})

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("SUPERSONIC_GQH_GGUF", output)
        self.assertIn("SUPERSONIC_QWEN38_MODEL_DIR", output)
        self.assertNotIn("SUPERSONIC_GQH_8192_GGUF", output)

    def test_missing_canonical_gguf_names_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = self.write_model_fixture(root)
            missing = root / "missing.gqh.gguf"
            result = self.run_checker(
                {
                    "SUPERSONIC_GQH_GGUF": str(missing),
                    "SUPERSONIC_QWEN38_MODEL_DIR": str(model_dir),
                }
            )

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("SUPERSONIC_GQH_GGUF", output)
        self.assertIn(str(missing), output)

    def test_missing_config_names_required_model_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "qwen38-model"
            model_dir.mkdir()
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            gguf = self.write_gguf_fixture(root)
            result = self.run_checker(
                {
                    "SUPERSONIC_GQH_GGUF": str(gguf),
                    "SUPERSONIC_QWEN38_MODEL_DIR": str(model_dir),
                }
            )

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("config.json", output)
        self.assertIn(str(model_dir / "config.json"), output)

    def test_valid_fixture_layout_passes_without_optional_8192(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = self.write_model_fixture(root)
            gguf = self.write_gguf_fixture(root)
            result = self.run_checker(
                {
                    "SUPERSONIC_GQH_GGUF": str(gguf),
                    "SUPERSONIC_QWEN38_MODEL_DIR": str(model_dir),
                }
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_require_8192_reports_missing_optional_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = self.write_model_fixture(root)
            gguf = self.write_gguf_fixture(root)
            result = self.run_checker(
                {
                    "SUPERSONIC_GQH_GGUF": str(gguf),
                    "SUPERSONIC_QWEN38_MODEL_DIR": str(model_dir),
                },
                "--require-8192",
            )

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("SUPERSONIC_GQH_8192_GGUF", output)

    def test_configured_optional_8192_path_is_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = self.write_model_fixture(root)
            gguf = self.write_gguf_fixture(root)
            missing = root / "missing-8192.gqh.gguf"
            result = self.run_checker(
                {
                    "SUPERSONIC_GQH_GGUF": str(gguf),
                    "SUPERSONIC_QWEN38_MODEL_DIR": str(model_dir),
                    "SUPERSONIC_GQH_8192_GGUF": str(missing),
                }
            )

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("SUPERSONIC_GQH_8192_GGUF", output)
        self.assertIn(str(missing), output)


if __name__ == "__main__":
    unittest.main()
