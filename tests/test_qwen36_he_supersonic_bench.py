import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parent / "gfx1100" / "bench_qwen36_he_supersonic.py"
SPEC = importlib.util.spec_from_file_location("bench_qwen36_he_supersonic", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class BenchQwen36HeSuperSonicTests(unittest.TestCase):
    def test_render_chatml_wraps_messages_for_lucebox_jsonl(self):
        prompt = bench.render_chatml(
            [{"role": "user", "content": "Complete this function."}]
        )

        self.assertEqual(
            prompt,
            "<|im_start|>user\nComplete this function.<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_render_chatml_can_include_qwen3_no_thinking_prefill(self):
        prompt = bench.render_chatml(
            [{"role": "user", "content": "Complete this function."}],
            no_thinking=True,
        )

        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_load_jsonl_prompts_supports_raw_and_chatml(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as tmp:
            tmp.write(
                json.dumps(
                    {
                        "id": "he_x",
                        "messages": [
                            {"role": "user", "content": "def f():\n    return"}
                        ],
                    }
                )
                + "\n"
            )
            tmp.flush()
            path = Path(tmp.name)

            self.assertEqual(
                bench.load_lucebox_jsonl_prompts(path, "raw"),
                [("he_x", "def f():\n    return")],
            )
            chatml = bench.load_lucebox_jsonl_prompts(path, "chatml")
            no_thinking = bench.load_lucebox_jsonl_prompts(
                path, "chatml-no-thinking"
            )
        self.assertEqual(chatml[0][0], "he_x")
        self.assertIn("<|im_start|>user\n", chatml[0][1])
        self.assertTrue(chatml[0][1].endswith("<|im_start|>assistant\n"))
        self.assertTrue(no_thinking[0][1].endswith("<think>\n\n</think>\n\n"))

    def test_resolve_lucebox_q4_alias_uses_config_dir_and_gguf(self):
        args = types.SimpleNamespace(
            dflash_draft_variant="lucebox-q4-k-m",
            dflash_draft_dir=Path("/custom/config"),
            dflash_draft_gguf=None,
        )

        config_dir, gguf, label = bench.resolve_dflash_draft(args)

        self.assertEqual(label, "lucebox-q4-k-m")
        self.assertEqual(config_dir, bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR)
        self.assertEqual(
            gguf,
            bench.DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q4_k_m.gguf",
        )

    def test_apply_target_profile_sets_qwen36_35b_a3b_defaults(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            model=None,
            model_dir=None,
            quant=None,
            out_json=None,
        )

        bench.apply_target_profile(args)

        self.assertEqual(args.model, "qwen3.6-35b-a3b")
        self.assertEqual(args.model_dir, bench.DEFAULT_35B_A3B_MODEL_DIR)
        self.assertEqual(args.quant, "int4")
        self.assertEqual(args.out_json, bench.DEFAULT_35B_A3B_OUT_JSON)

    def test_apply_target_profile_sets_qwen36_35b_a3b_flm_defaults(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b-flm",
            model=None,
            model_dir=None,
            quant=None,
            out_json=None,
        )

        bench.apply_target_profile(args)

        self.assertIsNone(args.model)
        self.assertEqual(args.model_dir, bench.DEFAULT_35B_A3B_FLM_MODEL_DIR)
        self.assertEqual(args.quant, "none")
        self.assertEqual(args.out_json, bench.DEFAULT_35B_A3B_FLM_OUT_JSON)

    def test_qwen36_35b_a3b_flm_profile_points_at_current_e2e_artifact(self):
        self.assertEqual(
            bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            Path(
                "/mnt/data/runs/geo-quant/"
                "qwen36-35b-a3b-supersonic-native-int4-current.flm"
            ),
        )

    def test_apply_target_profile_preserves_explicit_args(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            model="custom-model",
            model_dir=Path("/custom/model"),
            quant="none",
            out_json=Path("/tmp/custom.json"),
        )

        bench.apply_target_profile(args)

        self.assertEqual(args.model, "custom-model")
        self.assertEqual(args.model_dir, Path("/custom/model"))
        self.assertEqual(args.quant, "none")
        self.assertEqual(args.out_json, Path("/tmp/custom.json"))

    def test_lucebox_serving_mode_does_not_enable_dflash_for_35b_a3b(self):
        args = types.SimpleNamespace(
            target_profile="qwen36-35b-a3b",
            prompt_source="script",
            prompt_format="raw",
            ignore_eos=True,
            dflash=False,
            dflash_draft_variant=None,
            context_size=bench.DEFAULT_CONTEXT_SIZE,
        )

        bench.apply_lucebox_serving_mode(args)

        self.assertEqual(args.prompt_source, "jsonl")
        self.assertEqual(args.prompt_format, "chatml-no-thinking")
        self.assertFalse(args.ignore_eos)
        self.assertFalse(args.dflash)
        self.assertIsNone(args.dflash_draft_variant)
        self.assertEqual(args.context_size, bench.LUCEBOX_SERVING_CONTEXT_SIZE)

    def test_parse_flm_direct_execution_signals(self):
        output = (
            "[qwen36-moe] FLM weight mode: INT4 native FLM\n"
            "[qwen36-moe] FLM direct plans: required=693 raw_dense=363 "
            "native_int4=330 bf16_fallback=0\n"
            "[FLM runtime weights] ready-for-decode: YES "
            "(source=/tmp/qwen36.flm)\n"
        )

        self.assertEqual(bench.parse_flm_weight_mode(output), "INT4 native FLM")
        self.assertEqual(
            bench.parse_flm_direct_profile(output),
            {
                "required": 693,
                "raw_dense": 363,
                "native_int4": 330,
                "bf16_fallback": 0,
            },
        )
        self.assertEqual(
            bench.parse_flm_ready_for_decode(output),
            {"ready": True, "detail": "source=/tmp/qwen36.flm"},
        )

    def test_build_summary_includes_token_weighted_throughput(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": True,
            },
            {"returncode": 1},
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean_tok_s"], 37.5)
        self.assertEqual(summary["weighted_tok_s"], 37.5)
        self.assertEqual(summary["total_generated_tokens"], 30)
        self.assertEqual(summary["total_decode_ms"], 800.0)
        self.assertEqual(summary["stopped_early_count"], 1)

    def test_build_summary_includes_mean_lifecycle_timings_when_present(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "model_source_ms": 0.010,
                    "layer_load_ms": 6000.0,
                    "generation_wall_ms": 40.0,
                    "total_wall_ms": 6400.0,
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "model_source_ms": 0.030,
                    "layer_load_ms": 5000.0,
                    "generation_wall_ms": 50.0,
                    "total_wall_ms": 5600.0,
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["mean_lifecycle_timings"],
            {
                "model_source_ms": 0.020,
                "layer_load_ms": 5500.0,
                "generation_wall_ms": 45.0,
                "total_wall_ms": 6000.0,
            },
        )

    def test_parse_qwen36_startup_timings(self):
        output = (
            "[qwen36-moe startup-timings] flm_source_open_ms=123.456 "
            "bake_prepare_ms=0.000 dry_run_ms=4.500 "
            "pre_decode_total_ms=127.956\n"
        )

        self.assertEqual(
            bench.parse_startup_timings(output),
            {
                "flm_source_open_ms": 123.456,
                "bake_prepare_ms": 0.0,
                "dry_run_ms": 4.5,
                "pre_decode_total_ms": 127.956,
            },
        )

    def test_parse_hal_profile_ops(self):
        output = (
            "[hal-profile-op] op=copy_h2d calls=1518 mean_ms=0.5289 "
            "total_ms=802.919 max_ms=38.764 total_bytes=17908409344\n"
            "[hal-profile-op] op=vmm_map_no_sync calls=80 mean_ms=0.0243 "
            "total_ms=1.942 max_ms=0.043 total_bytes=16106127360\n"
        )

        self.assertEqual(
            bench.parse_hal_profile_ops(output),
            {
                "copy_h2d": {
                    "calls": 1518,
                    "mean_ms": 0.5289,
                    "total_ms": 802.919,
                    "max_ms": 38.764,
                    "total_bytes": 17908409344,
                },
                "vmm_map_no_sync": {
                    "calls": 80,
                    "mean_ms": 0.0243,
                    "total_ms": 1.942,
                    "max_ms": 0.043,
                    "total_bytes": 16106127360,
                },
            },
        )

    def test_parse_sparse_residency_and_breakdown_metrics(self):
        output = (
            "[vmm] MoE island residency: resident_slices=17 peak_slices=19 "
            "resident_pages=16 peak_pages=16 uploaded=161448.00MiB "
            "unmapped=161416.00MiB resident=32.00MiB peak_resident=32.00MiB "
            "total_vmm_resident=72.00MiB total_vmm_reserved=15400.00MiB\n"
            "[qwen36-moe sparse-breakdown] route_d2h_calls=3510 "
            "route_d2h_ms=84284.259 route_d2h_avg_ms=24.013 "
            "demand_prefetch_calls=160 demand_prefetch_ms=16185.783 "
            "demand_prefetch_avg_ms=101.161\n"
        )

        self.assertEqual(
            bench.parse_sparse_residency(output),
            {
                "resident_slices": 17.0,
                "peak_slices": 19.0,
                "resident_pages": 16.0,
                "peak_pages": 16.0,
                "uploaded": 161448.0,
                "unmapped": 161416.0,
                "resident": 32.0,
                "peak_resident": 32.0,
                "total_vmm_resident": 72.0,
                "total_vmm_reserved": 15400.0,
            },
        )
        self.assertEqual(
            bench.parse_sparse_breakdown(output),
            {
                "route_d2h_calls": 3510.0,
                "route_d2h_ms": 84284.259,
                "route_d2h_avg_ms": 24.013,
                "demand_prefetch_calls": 160.0,
                "demand_prefetch_ms": 16185.783,
                "demand_prefetch_avg_ms": 101.161,
            },
        )

    def test_build_summary_includes_mean_startup_timings_when_present(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "startup_timings": {
                    "flm_source_open_ms": 100.0,
                    "bake_prepare_ms": 0.0,
                    "dry_run_ms": 4.0,
                    "pre_decode_total_ms": 104.0,
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "startup_timings": {
                    "flm_source_open_ms": 200.0,
                    "bake_prepare_ms": 0.0,
                    "dry_run_ms": 6.0,
                    "pre_decode_total_ms": 206.0,
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["mean_startup_timings"],
            {
                "flm_source_open_ms": 150.0,
                "bake_prepare_ms": 0.0,
                "dry_run_ms": 5.0,
                "pre_decode_total_ms": 155.0,
            },
        )

    def test_build_summary_includes_mean_hal_profile_ops_when_present(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "hal_profile_ops": {
                    "copy_h2d": {
                        "calls": 10,
                        "mean_ms": 1.0,
                        "total_ms": 10.0,
                        "max_ms": 4.0,
                        "total_bytes": 100,
                    },
                    "vmm_map_no_sync": {
                        "calls": 2,
                        "mean_ms": 0.5,
                        "total_ms": 1.0,
                        "max_ms": 0.7,
                        "total_bytes": 200,
                    },
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "hal_profile_ops": {
                    "copy_h2d": {
                        "calls": 20,
                        "mean_ms": 2.0,
                        "total_ms": 40.0,
                        "max_ms": 8.0,
                        "total_bytes": 300,
                    },
                    "vmm_map": {
                        "calls": 1,
                        "mean_ms": 0.4,
                        "total_ms": 0.4,
                        "max_ms": 0.4,
                        "total_bytes": 50,
                    },
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["mean_hal_profile_ops"],
            {
                "copy_h2d": {
                    "calls": 15.0,
                    "mean_ms": 1.5,
                    "total_ms": 25.0,
                    "max_ms": 6.0,
                    "total_bytes": 200.0,
                }
            },
        )

    def test_build_summary_derives_flm_load_speed_metrics(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "layer_load_copy_h_to_d_ms": 500.0,
                    "layer_load_copy_h_to_d_bytes": 8 * 1024 * 1024 * 1024,
                },
                "hal_profile_ops": {
                    "copy_h2d": {
                        "calls": 8,
                        "mean_ms": 25.0,
                        "total_ms": 200.0,
                        "max_ms": 50.0,
                        "total_bytes": 6 * 1024 * 1024 * 1024,
                    },
                    "copy_storage_to_device": {
                        "calls": 2,
                        "mean_ms": 100.0,
                        "total_ms": 200.0,
                        "max_ms": 125.0,
                        "total_bytes": 4 * 1024 * 1024 * 1024,
                    },
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "lifecycle_timings": {
                    "layer_load_copy_h_to_d_ms": 1000.0,
                    "layer_load_copy_h_to_d_bytes": 16 * 1024 * 1024 * 1024,
                },
                "hal_profile_ops": {
                    "copy_h2d": {
                        "calls": 12,
                        "mean_ms": 50.0,
                        "total_ms": 600.0,
                        "max_ms": 100.0,
                        "total_bytes": 18 * 1024 * 1024 * 1024,
                    },
                    "copy_storage_to_device": {
                        "calls": 3,
                        "mean_ms": 200.0,
                        "total_ms": 600.0,
                        "max_ms": 250.0,
                        "total_bytes": 12 * 1024 * 1024 * 1024,
                    },
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["flm_load_speed"],
            {
                "layer_load_copy_h_to_d_bytes": 24 * 1024 * 1024 * 1024,
                "layer_load_copy_h_to_d_ms": 1500.0,
                "layer_load_copy_h_to_d_gib_s": 16.0,
                "copy_h2d_bytes": 24 * 1024 * 1024 * 1024,
                "copy_h2d_ms": 800.0,
                "copy_h2d_gib_s": 30.0,
                "copy_storage_to_device_bytes": 16 * 1024 * 1024 * 1024,
                "copy_storage_to_device_ms": 800.0,
                "copy_storage_to_device_gib_s": 20.0,
            },
        )

    def test_build_summary_includes_mean_sparse_metrics_when_present(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "sparse_residency": {
                    "uploaded": 160000.0,
                    "resident": 32.0,
                    "peak_resident": 32.0,
                },
                "sparse_breakdown": {
                    "route_d2h_avg_ms": 24.0,
                    "demand_prefetch_avg_ms": 100.0,
                    "cap8_only": 1.0,
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "sparse_residency": {
                    "uploaded": 170000.0,
                    "resident": 64.0,
                    "peak_resident": 64.0,
                },
                "sparse_breakdown": {
                    "route_d2h_avg_ms": 26.0,
                    "demand_prefetch_avg_ms": 110.0,
                },
            },
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(
            summary["mean_sparse_residency"],
            {
                "peak_resident": 48.0,
                "resident": 48.0,
                "uploaded": 165000.0,
            },
        )
        self.assertEqual(
            summary["mean_sparse_breakdown"],
            {
                "demand_prefetch_avg_ms": 105.0,
                "route_d2h_avg_ms": 25.0,
            },
        )

    def test_build_summary_includes_flm_direct_execution_evidence(self):
        rows = [
            {
                "returncode": 0,
                "tok_s": 25.0,
                "generated_tokens": 10,
                "decode_ms": 400.0,
                "ms_per_step": 40.0,
                "stopped_early": False,
                "flm_weight_mode": "INT4 native FLM",
                "flm_ready_for_decode": True,
                "flm_direct_profile": {
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "bf16_fallback": 0,
                },
            },
            {
                "returncode": 0,
                "tok_s": 50.0,
                "generated_tokens": 20,
                "decode_ms": 400.0,
                "ms_per_step": 20.0,
                "stopped_early": False,
                "flm_weight_mode": "INT4 native FLM",
            },
            {"returncode": 1, "flm_weight_mode": "BF16"},
        ]

        summary = bench.build_summary(rows)

        self.assertEqual(summary["flm_weight_modes"], ["INT4 native FLM"])
        self.assertEqual(summary["flm_ready_for_decode_count"], 1)
        self.assertEqual(
            summary["flm_direct_profiles"],
            [
                {
                    "required": 693,
                    "raw_dense": 363,
                    "native_int4": 330,
                    "bf16_fallback": 0,
                }
            ],
        )

    def test_run_one_sets_gguf_env_and_stop_on_eos_command(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model="qwen3.6-27b",
            model_dir=Path("/models/target"),
            context_size=1024,
            warmup_new_tokens=2,
            n_gen=256,
            seed=1,
            prompt_no_special_tokens=True,
            quant="q4km",
            ignore_eos=False,
            emit_stage_timings=False,
            kv_fp8=False,
            dflash=True,
            dflash_block=0,
            dflash_draft_variant="lucebox-q8-0",
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
        )

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="[result] prompt_tokens=12 generated_tokens=97 decode_ms=970 ms_per_tok=10\n",
        )
        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "prompt")

        cmd = run.call_args.args[0]
        env = run.call_args.kwargs["env"]
        self.assertIn("--dflash", cmd)
        self.assertIn("--prompt-no-special-tokens", cmd)
        self.assertNotIn("--ignore-eos", cmd)
        self.assertEqual(
            env["SUPERSONIC_DFLASH_DRAFT_GGUF"],
            str(bench.DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q8_0.gguf"),
        )
        self.assertEqual(row["generated_tokens"], 97)
        self.assertTrue(row["stopped_early"])
        self.assertEqual(row["dflash_draft_label"], "lucebox-q8-0")

    def test_run_one_hal_profile_sets_runner_profile_env(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
            hal_profile=True,
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="[result] prompt_tokens=1 generated_tokens=1 decode_ms=10 ms_per_step=10\n",
        )

        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            bench.run_one(args, "case", "Hello")

        env = run.call_args.kwargs["env"]
        self.assertEqual(env["SUPERSONIC_METAL_PROFILE"], "1")

    def test_run_one_sets_runner_env_and_records_sparse_metrics(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
            hal_profile=True,
            runner_env=["SUPERSONIC_MOE_ISLAND_CAP_EXPERTS=8"],
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=(
                "[vmm] MoE island residency: resident_pages=16 uploaded=161448.00MiB "
                "resident=32.00MiB peak_resident=32.00MiB\n"
                "[result] prompt_tokens=1 generated_tokens=1 decode_ms=10 "
                "ms_per_step=10\n"
                "[qwen36-moe sparse-breakdown] route_d2h_calls=3510 "
                "route_d2h_ms=84284.259 demand_prefetch_avg_ms=101.161\n"
            ),
        )

        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "Hello")

        env = run.call_args.kwargs["env"]
        self.assertEqual(env["SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"], "8")
        self.assertEqual(row["runner_env"], {"SUPERSONIC_MOE_ISLAND_CAP_EXPERTS": "8"})
        self.assertEqual(
            row["sparse_residency"],
            {
                "resident_pages": 16.0,
                "uploaded": 161448.0,
                "resident": 32.0,
                "peak_resident": 32.0,
            },
        )
        self.assertEqual(
            row["sparse_breakdown"],
            {
                "route_d2h_calls": 3510.0,
                "route_d2h_ms": 84284.259,
                "demand_prefetch_avg_ms": 101.161,
            },
        )

    def test_run_one_forwards_flm_virtual_transfer_backend_flag(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
            hal_profile=False,
            runner_env=[],
            flm_virtual_transfer_backend="hipfile",
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="[result] prompt_tokens=1 generated_tokens=1 decode_ms=10 ms_per_step=10\n",
        )

        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "Hello")

        cmd = run.call_args.args[0]
        self.assertIn("--flm-virtual-transfer-backend", cmd)
        self.assertIn("hipfile", cmd)
        self.assertEqual(row["flm_virtual_transfer_backend"], "hipfile")

    def test_run_one_flm_profile_omits_int4_and_records_lifecycle_timings(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
        )

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=(
                "[flm] inferred model qwen3.6-35b-a3b from runtime descriptor\n"
                "[qwen36-moe] FLM weight mode: INT4 native FLM\n"
                "[qwen36-moe] FLM direct plans: required=693 raw_dense=363 "
                "native_int4=330 bf16_fallback=0\n"
                "[FLM runtime weights] ready-for-decode: YES "
                "(source=/tmp/qwen36.flm)\n"
                "[qwen36-moe startup-timings] flm_source_open_ms=123.456 "
                "flm_tokenizer_parse_vocab_ms=0.000 "
                "flm_tokenizer_parse_vocab_ids_ms=0.000 "
                "flm_tokenizer_parse_merges_ms=0.000 "
                "flm_tokenizer_parse_added_tokens_ms=0.000 "
                "flm_tokenizer_parse_regex_ms=0.000 "
                "bake_prepare_ms=0.000 dry_run_ms=4.500 "
                "pre_decode_total_ms=127.956\n"
                "[result] prompt_tokens=1 generated_tokens=1 decode_ms=41 "
                "ms_per_step=41.0\n"
                "[qwen36-moe lifecycle-timings] prompt_setup_ms=55.351 "
                "flm_tokenizer_parse_vocab_ms=10.000 "
                "flm_tokenizer_parse_vocab_ids_ms=0.500 "
                "flm_tokenizer_parse_merges_ms=5.000 "
                "flm_tokenizer_parse_added_tokens_ms=0.250 "
                "flm_tokenizer_parse_regex_ms=0.125 "
                "model_source_ms=0.009 layer_load_ms=5744.555 "
                "layer_load_buffers_ms=5400.000 "
                "layer_load_vmm_setup_ms=10.000 "
                "layer_load_prewarm_ms=0.000 "
                "layer_load_hal_ms=4500.000 "
                "layer_load_alloc_ms=1200.000 "
                "layer_load_copy_h_to_d_ms=3000.000 "
                "layer_load_memset_ms=100.000 "
                "layer_load_vmm_ms=200.000 "
                "layer_load_alloc_bytes=123456 "
                "layer_load_copy_h_to_d_bytes=654321 "
                "layer_load_memset_bytes=4096 "
                "layer_load_vmm_bytes=8192 "
                "session_ms=466.101 "
                "prefill_steps=0 prefill_embed_ms=0.000 prefill_chain_ms=0.000 "
                "prefill_total_ms=0.000 generation_wall_ms=41.006 "
                "total_wall_ms=6307.287\n"
                "[hal-profile-op] op=copy_h2d calls=1518 mean_ms=0.5289 "
                "total_ms=802.919 max_ms=38.764 total_bytes=17908409344\n"
            ),
        )
        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            row = bench.run_one(args, "case", "Hello")

        cmd = run.call_args.args[0]
        self.assertIn("--model-dir", cmd)
        self.assertIn(str(bench.DEFAULT_35B_A3B_FLM_MODEL_DIR), cmd)
        self.assertNotIn("--model", cmd)
        self.assertNotIn("qwen3.6-35b-a3b", cmd)
        self.assertNotIn("--int4", cmd)
        self.assertIn("--emit-stage-timings", cmd)
        self.assertEqual(row["generated_tokens"], 1)
        self.assertEqual(row.get("resolved_model"), "qwen3.6-35b-a3b")
        self.assertEqual(row["flm_weight_mode"], "INT4 native FLM")
        self.assertTrue(row["flm_ready_for_decode"])
        self.assertEqual(row["flm_ready_for_decode_detail"], "source=/tmp/qwen36.flm")
        self.assertEqual(
            row["flm_direct_profile"],
            {
                "required": 693,
                "raw_dense": 363,
                "native_int4": 330,
                "bf16_fallback": 0,
            },
        )
        self.assertEqual(
            row["startup_timings"],
            {
                "flm_source_open_ms": 123.456,
                "flm_tokenizer_parse_vocab_ms": 0.0,
                "flm_tokenizer_parse_vocab_ids_ms": 0.0,
                "flm_tokenizer_parse_merges_ms": 0.0,
                "flm_tokenizer_parse_added_tokens_ms": 0.0,
                "flm_tokenizer_parse_regex_ms": 0.0,
                "bake_prepare_ms": 0.0,
                "dry_run_ms": 4.5,
                "pre_decode_total_ms": 127.956,
            },
        )
        self.assertEqual(
            row["lifecycle_timings"],
            {
                "prompt_setup_ms": 55.351,
                "flm_tokenizer_parse_vocab_ms": 10.0,
                "flm_tokenizer_parse_vocab_ids_ms": 0.5,
                "flm_tokenizer_parse_merges_ms": 5.0,
                "flm_tokenizer_parse_added_tokens_ms": 0.25,
                "flm_tokenizer_parse_regex_ms": 0.125,
                "model_source_ms": 0.009,
                "layer_load_ms": 5744.555,
                "layer_load_buffers_ms": 5400.0,
                "layer_load_vmm_setup_ms": 10.0,
                "layer_load_prewarm_ms": 0.0,
                "layer_load_hal_ms": 4500.0,
                "layer_load_alloc_ms": 1200.0,
                "layer_load_copy_h_to_d_ms": 3000.0,
                "layer_load_memset_ms": 100.0,
                "layer_load_vmm_ms": 200.0,
                "layer_load_alloc_bytes": 123456.0,
                "layer_load_copy_h_to_d_bytes": 654321.0,
                "layer_load_memset_bytes": 4096.0,
                "layer_load_vmm_bytes": 8192.0,
                "session_ms": 466.101,
                "prefill_steps": 0.0,
                "prefill_embed_ms": 0.0,
                "prefill_chain_ms": 0.0,
                "prefill_total_ms": 0.0,
                "generation_wall_ms": 41.006,
                "total_wall_ms": 6307.287,
            },
        )
        self.assertEqual(
            row["hal_profile_ops"],
            {
                "copy_h2d": {
                    "calls": 1518,
                    "mean_ms": 0.5289,
                    "total_ms": 802.919,
                    "max_ms": 38.764,
                    "total_bytes": 17908409344,
                }
            },
        )

    def test_run_one_flm_profile_fails_without_first_class_flm_evidence(self):
        args = types.SimpleNamespace(
            binary=Path("target/release/supersonic"),
            backend="hip",
            model=None,
            model_dir=bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
            context_size=16,
            warmup_new_tokens=1,
            n_gen=1,
            seed=1,
            prompt_no_special_tokens=False,
            quant="none",
            ignore_eos=True,
            emit_stage_timings=True,
            kv_fp8=False,
            dflash=False,
            dflash_block=0,
            dflash_draft_variant=None,
            dflash_draft_dir=bench.DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
            dflash_draft_gguf=None,
            timeout=10,
            tail_chars=200,
            allow_untested_gpu=None,
            hal_profile=False,
            runner_env=[],
            flm_virtual_transfer_backend=None,
        )
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="[result] prompt_tokens=1 generated_tokens=1 decode_ms=10 ms_per_step=10\n",
        )

        with mock.patch.object(bench.subprocess, "run", return_value=completed):
            row = bench.run_one(args, "case", "Hello")

        self.assertEqual(row["returncode"], 1)
        self.assertEqual(row["runner_returncode"], 0)
        self.assertEqual(
            row["benchmark_validation_errors"],
            [
                "FLM run did not report inferred model from runtime descriptor",
                "FLM run did not report INT4 native FLM weight mode",
                "FLM run did not report ready-for-decode YES",
                "FLM run did not report native INT4 direct plan coverage",
            ],
        )


if __name__ == "__main__":
    unittest.main()
