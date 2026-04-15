#!/usr/bin/env python3
"""Generate golden test data for SuperSonic regression tests.

Runs each test case through the HuggingFace model and saves the expected
token IDs and logit metadata. This golden data allows offline validation
without re-running the Python oracle every time.

Usage:
    python3 tests/corpus/generate_golden.py --model-id Qwen/Qwen3.5-0.8B --output tests/corpus/golden_0.8b.json
    python3 tests/corpus/generate_golden.py --model-id Qwen/Qwen3.5-4B --output tests/corpus/golden_4b.json
"""

import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- Test case definitions ----

COMMON_TESTS = [
    # Basic completion
    {
        "name": "hello_world",
        "type": "exact_prefix",
        "prompt": "Hello, world",
        "max_new_tokens": 8,
        "description": "Basic greeting completion",
    },
    {
        "name": "capital_france",
        "type": "exact_prefix",
        "prompt": "The capital of France is",
        "max_new_tokens": 4,
        "min_prefix_tokens": 1,  # at least "Paris" or similar
        "description": "Simple factual completion",
    },
    {
        "name": "arithmetic",
        "type": "exact_prefix",
        "prompt": "2 + 2 =",
        "max_new_tokens": 4,
        "description": "Simple arithmetic",
    },
    {
        "name": "counting",
        "type": "exact_prefix",
        "prompt": "1, 2, 3, 4,",
        "max_new_tokens": 8,
        "description": "Sequence continuation",
    },
    # Longer context
    {
        "name": "medium_context",
        "type": "exact_prefix",
        "prompt": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. What animal jumps?",
        "max_new_tokens": 8,
        "description": "Medium-length context with question",
    },
    # NIAH style — fact embedded in padding
    {
        "name": "niah_short",
        "type": "exact_prefix",
        "prompt": "Random text here. Random text here. The secret code is 42. Random text here. Random text here. What is the secret code?",
        "max_new_tokens": 8,
        "description": "NIAH: retrieve embedded fact from short context",
    },
    {
        "name": "niah_medium",
        "type": "exact_prefix",
        "prompt": (
            "The weather today is sunny. Birds are singing in the trees. "
            "The river flows gently through the valley. Mountains rise in the distance. "
            "The password to the vault is diamond. "
            "Clouds drift lazily across the blue sky. Flowers bloom in the meadow. "
            "Children play in the park nearby. The wind whispers through the leaves. "
            "What is the password to the vault?"
        ),
        "max_new_tokens": 8,
        "description": "NIAH: retrieve embedded fact from medium context",
    },
    # Edge cases
    {
        "name": "single_token",
        "type": "exact_prefix",
        "prompt": "A",
        "max_new_tokens": 8,
        "description": "Single token prompt",
    },
    {
        "name": "long_prompt",
        "type": "exact_prefix",
        "prompt": " ".join(["The"] * 50) + " end.",
        "max_new_tokens": 8,
        "description": "Long repetitive prompt (50+ tokens)",
    },
    # Multilingual
    {
        "name": "chinese_prompt",
        "type": "exact_prefix",
        "prompt": "中国的首都是",
        "max_new_tokens": 8,
        "description": "Chinese factual completion (capital of China)",
    },
]


def generate_golden(model_id: str, output_path: str, device: str = "cpu"):
    print(f"Loading {model_id} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"dtype": torch.bfloat16}
    if device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    golden = {
        "model_id": model_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "test_cases": [],
    }

    for tc in COMMON_TESTS:
        name = tc["name"]
        prompt = tc["prompt"]
        max_new = tc["max_new_tokens"]
        print(f"  [{name}] prompt={repr(prompt[:60])}...", end="", flush=True)

        input_ids = tok.encode(prompt, return_tensors="pt")
        if device != "cpu":
            input_ids = input_ids.to(device)
        prompt_tokens = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        generated_ids = out.sequences[0, prompt_tokens:].tolist()
        generated_text = tok.decode(generated_ids)

        # Get prefill logits (last logit before first generated token)
        # output_logits gives one tensor per generated step
        prefill_top5 = []
        if out.logits:
            first_logits = out.logits[0][0].float().cpu()
            top5_vals, top5_idx = first_logits.topk(5)
            prefill_top5 = [
                {"token_id": int(idx), "logit": float(val), "text": tok.decode([int(idx)])}
                for idx, val in zip(top5_idx.tolist(), top5_vals.tolist())
            ]

        result = {
            "name": name,
            "type": tc["type"],
            "description": tc.get("description", ""),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new,
            "expected_token_ids": generated_ids,
            "expected_text": generated_text,
            "prefill_top5": prefill_top5,
        }
        golden["test_cases"].append(result)
        print(f" → {repr(generated_text[:60])}")

    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(golden['test_cases'])} test cases to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    generate_golden(args.model_id, args.output, args.device)
