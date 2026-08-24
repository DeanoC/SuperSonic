import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "qwen38-reproducibility.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("qwen38_reproducibility", TOOL)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen38_reproducibility"] = module
    spec.loader.exec_module(module)
    return module


class Qwen38ReproducibilityTests(unittest.TestCase):
    def test_record_contains_safe_identity_correctness_and_timings(self):
        tool = load_tool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "private" / "qwen38.gqh.gguf"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")
            model_dir = root / "private" / "Qwen3.8-27B"
            model_dir.mkdir()
            for name, value in (
                ("config.json", b"config"),
                ("tokenizer.json", b"tokenizer"),
                ("tokenizer_config.json", b"chat"),
            ):
                (model_dir / name).write_bytes(value)
            ordinary = root / "ordinary.log"
            ordinary.write_text(
                "[prefill] native HIP prefill done in 12.5ms\n"
                "[tokens] 10 20 30\n"
                "[result] prompt_tokens=7 generated_tokens=3 decode_ms=9.0 ms_per_tok=3.0\n",
                encoding="utf-8",
            )
            mtp = root / "mtp.log"
            mtp.write_text(
                "[prefill] native HIP prefill done in 13.5ms\n"
                "[tokens] 10 20 30\n"
                "[result] prompt_tokens=7 generated_tokens=3 decode_ms=8.0 ms_per_tok=2.7\n",
                encoding="utf-8",
            )
            telemetry = root / "telemetry"
            telemetry.mkdir()
            (telemetry / "warmup-1.log").write_text(
                "[prefill] native HIP prefill done in 20ms\n"
                "[result] prompt_tokens=7 generated_tokens=3 decode_ms=12ms ms_per_tok=4ms\n",
                encoding="utf-8",
            )
            for index, value in enumerate((3.0, 2.0, 4.0), start=1):
                (telemetry / f"run-{index}.log").write_text(
                    f"[prefill] native HIP prefill done in {10 + index}ms\n"
                    f"[result] prompt_tokens=7 generated_tokens=3 decode_ms={value * 3}ms "
                    f"ms_per_tok={value}ms\n",
                    encoding="utf-8",
                )
            hip_version = root / "hipcc-version.txt"
            hip_version.write_text("HIP version: 7.2.4\n", encoding="utf-8")
            rocm_version = root / "rocm-driver-version.txt"
            rocm_version.write_text("Driver version: 7.2.4\n", encoding="utf-8")
            gpu_json = root / "amd-smi.json"
            gpu_json.write_text(
                json.dumps(
                    {
                        "gpu_data": [
                            {
                                "gpu": 1,
                                "asic": {
                                    "market_name": "AMD Radeon AI PRO R9700",
                                    "target_graphics_version": "gfx1201",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            record = tool.build_record(
                commit="a" * 40,
                hip_version_file=hip_version,
                rocm_version="unknown",
                rocm_version_file=rocm_version,
                gpu_json=gpu_json,
                physical_gpu="1",
                gpu_arch="gfx1201",
                artifact=artifact,
                model_dir=model_dir,
                ordinary_log=ordinary,
                mtp_log=mtp,
                telemetry_root=telemetry,
                prompt="Hello",
                chat=False,
                max_new_tokens=3,
            )
            (model_dir / "tokenizer_config.json").unlink()
            non_chat_record = tool.build_record(
                commit="a" * 40,
                hip_version_file=hip_version,
                rocm_version="unknown",
                rocm_version_file=rocm_version,
                gpu_json=gpu_json,
                physical_gpu="1",
                gpu_arch="gfx1201",
                artifact=artifact,
                model_dir=model_dir,
                ordinary_log=ordinary,
                mtp_log=mtp,
                telemetry_root=telemetry,
                prompt="Hello",
                chat=False,
                max_new_tokens=3,
            )
            self.assertEqual(non_chat_record["model_directory"]["name"], "Qwen3.8-27B")
            with self.assertRaises(ValueError):
                tool.build_record(
                    commit="a" * 40,
                    hip_version_file=hip_version,
                    rocm_version="unknown",
                    rocm_version_file=rocm_version,
                    gpu_json=gpu_json,
                    physical_gpu="1",
                    gpu_arch="gfx1201",
                    artifact=artifact,
                    model_dir=model_dir,
                    ordinary_log=ordinary,
                    mtp_log=mtp,
                    telemetry_root=telemetry,
                    prompt="Hello",
                    chat=True,
                    max_new_tokens=3,
                )

        self.assertEqual(record["commit"], "a" * 40)
        self.assertEqual(record["toolchain"]["hip_version"], "7.2.4")
        self.assertEqual(record["physical_gpu"]["id"], "1")
        self.assertEqual(record["physical_gpu"]["architecture"], "gfx1201")
        self.assertEqual(record["physical_gpu"]["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(record["artifact"]["name"], "qwen38.gqh.gguf")
        self.assertEqual(record["artifact"]["identity"], "qwen38.gqh.gguf")
        self.assertEqual(
            record["artifact"]["sha256"],
            hashlib.sha256(b"artifact").hexdigest(),
        )
        self.assertTrue(record["artifact"]["digest"].startswith("sha256:"))
        self.assertNotIn(str(root), json.dumps(record))
        self.assertEqual(record["workload"]["token_count"], 3)
        self.assertTrue(record["correctness"]["ordinary_vs_mtp"]["equal"])
        self.assertEqual(
            record["correctness"]["correctness_hash"],
            tool.token_sequence_hash([10, 20, 30]),
        )
        self.assertEqual(record["timings"]["warmup_runs"], 1)
        self.assertEqual(record["timings"]["measured_runs"], 3)
        self.assertEqual(record["timings"]["median_ms_per_tok"], 3.0)
        self.assertEqual(record["timings"]["warmup"][0]["prefill_ms"], 20.0)

    def test_validator_rejects_absolute_paths_and_missing_record_fields(self):
        tool = load_tool()
        with self.assertRaises(ValueError):
            tool.validate_record({"commit": "a" * 40, "artifact": {"path": "/secret/file"}})

        with self.assertRaises(ValueError):
            tool.validate_record(
                {
                    "schema_version": 1,
                    "commit": "a" * 40,
                    "toolchain": {"hip_version": "7.2.4", "rocm_version": "7.2.4"},
                    "physical_gpu": {"id": "1", "architecture": "gfx1201", "name": "R9700"},
                    "artifact": {"name": "qwen38.gguf", "sha256": "0" * 64},
                    "workload": {
                        "prompt": "Hello",
                        "token_count": 3,
                        "max_new_tokens": 3,
                    },
                    "correctness": {
                        "correctness_hash": "0" * 64,
                        "ordinary_vs_mtp": {"applicable": True, "equal": None},
                    },
                    "timings": {
                        "warmup_runs": 0,
                        "measured_runs": 0,
                        "warmup": [],
                        "measured": [],
                    },
                }
            )

    def test_validator_rejects_partial_or_mismatched_telemetry_counts(self):
        tool = load_tool()
        valid = {
            "schema_version": 1,
            "commit": "a" * 40,
            "toolchain": {"hip_version": "7.2.4", "rocm_version": "7.2.4"},
            "target_architecture": "gfx1201",
            "physical_gpu": {"id": "1", "architecture": "gfx1201", "name": "R9700"},
            "artifact": {"name": "qwen38.gguf", "sha256": "0" * 64},
            "model_directory": {"name": "Qwen3.8-27B", "required_files": {}},
            "workload": {"prompt": "Hello", "token_count": 3, "max_new_tokens": 3},
            "correctness": {
                "correctness_hash": "0" * 64,
                "ordinary_vs_mtp": {
                    "applicable": True,
                    "equal": True,
                    "ordinary_hash": "0" * 64,
                    "mtp_hash": "0" * 64,
                },
            },
            "timings": {
                "warmup_runs": 1,
                "measured_runs": 3,
                "warmup": [{}],
                "measured": [{}, {}, {}],
                "median_ms_per_tok": 1.0,
                "status": "complete",
            },
        }
        for warmup_runs, measured_runs, warmup, measured in (
            (0, 3, [], [{}, {}, {}]),
            (1, 1, [{}], [{}]),
            (1, 4, [{}], [{}, {}, {}, {}]),
        ):
            candidate = json.loads(json.dumps(valid))
            candidate["timings"].update(
                {
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "warmup": warmup,
                    "measured": measured,
                }
            )
            with self.assertRaises(ValueError):
                tool.validate_record(candidate)


if __name__ == "__main__":
    unittest.main()
