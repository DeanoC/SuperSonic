#!/usr/bin/env python3
"""Generate golden test data from SuperSonic native output (GPU path).

Runs the actual supersonic binary on each test case and records what it produces.
This ensures the golden data matches the GPU BF16 path exactly.

Usage:
    python3 tests/corpus/generate_golden_native.py \
        --binary target/release/supersonic \
        --model qwen3.5-0.8b \
        --model-dir /path/to/model \
        --test-defs tests/corpus/golden_0.8b.json \
        --output tests/corpus/golden_0.8b.json
"""

import argparse
import json
import subprocess
import time


def generate(binary, model, model_dir, test_defs_path, output_path, extra_flags=None):
    with open(test_defs_path) as f:
        test_defs = json.load(f)

    golden = {
        "model_id": test_defs.get("model_id", model),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "native_gpu",
        "test_cases": [],
    }

    extra = extra_flags or []

    for tc in test_defs["test_cases"]:
        name = tc["name"]
        prompt = tc["prompt"]
        max_new = tc["max_new_tokens"]
        print(f"  [{name}] ", end="", flush=True)

        proc = subprocess.run(
            [binary, "--model", model, "--model-dir", model_dir,
             "--prompt", prompt, "--max-new-tokens", str(max_new)] + extra,
            capture_output=True, text=True, timeout=600,
        )

        if proc.returncode != 0:
            print(f"FAILED (exit {proc.returncode})")
            continue

        full_output = proc.stdout.strip()
        if full_output.startswith(prompt):
            generated = full_output[len(prompt):]
        else:
            generated = full_output

        result = {
            "name": name,
            "type": tc.get("type", "exact_prefix"),
            "description": tc.get("description", ""),
            "prompt": prompt,
            "prompt_tokens": tc.get("prompt_tokens", 0),
            "max_new_tokens": max_new,
            "expected_token_ids": tc.get("expected_token_ids", []),
            "expected_text": generated,
            "prefill_top5": tc.get("prefill_top5", []),
        }
        golden["test_cases"].append(result)
        print(f"→ {repr(generated[:60])}")

    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(golden['test_cases'])} test cases to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-defs", required=True, help="Existing golden JSON to read test definitions from")
    parser.add_argument("--output", required=True)
    parser.add_argument("extra_flags", nargs="*", help="Extra flags to pass to supersonic (e.g. --fp8-runtime)")
    args = parser.parse_args()
    generate(args.binary, args.model, args.model_dir, args.test_defs, args.output, args.extra_flags)
