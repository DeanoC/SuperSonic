#!/usr/bin/env python3
"""
Bake-side classification tests for Qwen3.6-35B-A3B (qwen36-moe).

PR 2 of docs/qwen36-moe-plan.md asks: "manifest assertions: every expert
weight is `Int4Quantized`, every gate is `Raw`". The decision is made by
`is_int4_target` (INT4 GPTQ — primary HIP path) and `is_q4km_target`
(q4km — primary CUDA path). This file pins their behavior against the
ACTUAL Qwen3.6-MoE tensor naming, as discovered post-PR 3 by enumerating
the published Qwen/Qwen3.6-35B-A3B safetensors.

Two facts the published checkpoint forces on us:

1. **Experts are fused, not per-expert.** The real checkpoint stores ONE
   tensor per layer for gate+up across all 256 experts:
       mlp.experts.gate_up_proj      [256, 1024, 2048]  bf16
       mlp.experts.down_proj         [256, 2048, 512]   bf16
   Note the *missing `.weight` suffix*. Plan §2 listed
   `mlp.experts.{E}.{gate,up,down}_proj.weight` (768 tensors per layer);
   that layout does not exist on disk for this model.

2. **Bake predicates pick up the fused tensors as INT4 targets** so the
   experts get quantized into the runtime's expected layout. Plan §15
   option (c) drove this: predicates accept the bare `mlp.experts.*` names
   (no `.weight` suffix, 3D shape), and a separate fused-experts driver in
   `bake_int4.py` / `bake_q4km.py` packs each expert's `[out, in]` slab
   independently with `group_size=128` BF16 scale+zero — min/max group-quant,
   no Hessian/GPTQ for the experts (GPTQ stays on the dense projections).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_int4 import is_int4_target
from bake_q4km import is_q4km_target


HIDDEN = 2048
MOE_INTER = 512
NUM_EXPERTS = 256
VOCAB = 248320
HEAD_DIM = 256
KV_HEADS = 2
Q_HEADS = 16
GROUP_SIZE = 128


def shape_for(name: str) -> list[int]:
    """Return the on-disk shape for a real Qwen3.6-MoE tensor name."""
    if name == "lm_head.weight":
        return [VOCAB, HIDDEN]
    if name.endswith(".embed_tokens.weight"):
        return [VOCAB, HIDDEN]
    if name.endswith(".norm.weight") or name.endswith("layernorm.weight"):
        return [HIDDEN]
    if name.endswith(".q_norm.weight") or name.endswith(".k_norm.weight"):
        return [HEAD_DIM]
    # `attn_output_gate=true` doubles q_proj's output dim (Q + output gate).
    if name.endswith(".q_proj.weight"):
        return [2 * Q_HEADS * HEAD_DIM, HIDDEN]
    if name.endswith(".k_proj.weight") or name.endswith(".v_proj.weight"):
        return [KV_HEADS * HEAD_DIM, HIDDEN]
    if name.endswith(".o_proj.weight"):
        return [HIDDEN, Q_HEADS * HEAD_DIM]
    # 35B linear-attn: 16 K-heads × 128 + 16 K-heads × 128 + 32 V-heads × 128 = 8192.
    if name.endswith(".linear_attn.in_proj_qkv.weight"):
        return [8192, HIDDEN]
    if name.endswith(".linear_attn.in_proj_z.weight"):
        return [32 * 128, HIDDEN]
    if name.endswith(".linear_attn.in_proj_a.weight"):
        return [32, HIDDEN]
    if name.endswith(".linear_attn.in_proj_b.weight"):
        return [32, HIDDEN]
    if name.endswith(".linear_attn.out_proj.weight"):
        return [HIDDEN, 32 * 128]
    if name.endswith(".linear_attn.conv1d.weight"):
        return [8192, 1, 4]
    if name.endswith(".linear_attn.dt_bias"):
        return [32]
    if name.endswith(".linear_attn.A_log"):
        return [32]
    if name.endswith(".mlp.gate.weight"):
        return [NUM_EXPERTS, HIDDEN]
    if name.endswith(".mlp.shared_expert_gate.weight"):
        return [1, HIDDEN]
    if name.endswith(".shared_expert.gate_proj.weight"):
        return [MOE_INTER, HIDDEN]
    if name.endswith(".shared_expert.up_proj.weight"):
        return [MOE_INTER, HIDDEN]
    if name.endswith(".shared_expert.down_proj.weight"):
        return [HIDDEN, MOE_INTER]
    if name.endswith(".mlp.experts.gate_up_proj"):
        return [NUM_EXPERTS, 2 * MOE_INTER, HIDDEN]
    if name.endswith(".mlp.experts.down_proj"):
        return [NUM_EXPERTS, HIDDEN, MOE_INTER]
    raise KeyError(f"no shape rule for {name}")


# Fixed expectations: the tensors that DO ship in Qwen/Qwen3.6-35B-A3B.
# (name, expected_int4_target, expected_q4km_target)
EXPECTATIONS: list[tuple[str, bool, bool]] = [
    # Embeddings / final norm / lm_head.
    ("model.language_model.embed_tokens.weight", False, False),
    ("model.language_model.norm.weight", False, False),
    ("lm_head.weight", True, True),

    # Per-layer norms.
    ("model.language_model.layers.0.input_layernorm.weight", False, False),
    ("model.language_model.layers.0.post_attention_layernorm.weight", False, False),
    ("model.language_model.layers.3.input_layernorm.weight", False, False),
    ("model.language_model.layers.3.post_attention_layernorm.weight", False, False),

    # Full-attention layer (every 4th: indices 3, 7, 11, ...). Note q_proj
    # is doubled by attn_output_gate=true.
    ("model.language_model.layers.3.self_attn.q_proj.weight", True, True),
    ("model.language_model.layers.3.self_attn.k_proj.weight", True, True),
    ("model.language_model.layers.3.self_attn.v_proj.weight", True, True),
    ("model.language_model.layers.3.self_attn.o_proj.weight", True, True),
    ("model.language_model.layers.3.self_attn.q_norm.weight", False, False),
    ("model.language_model.layers.3.self_attn.k_norm.weight", False, False),

    # Linear-attention layer.
    ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", True, True),
    ("model.language_model.layers.0.linear_attn.in_proj_z.weight", True, True),
    ("model.language_model.layers.0.linear_attn.in_proj_a.weight", False, False),
    ("model.language_model.layers.0.linear_attn.in_proj_b.weight", False, False),
    ("model.language_model.layers.0.linear_attn.out_proj.weight", True, True),
    ("model.language_model.layers.0.linear_attn.conv1d.weight", False, False),
    ("model.language_model.layers.0.linear_attn.norm.weight", False, False),
    # Non-.weight tensors must not be targeted regardless of name.
    ("model.language_model.layers.0.linear_attn.dt_bias", False, False),
    ("model.language_model.layers.0.linear_attn.A_log", False, False),

    # MoE block.
    ("model.language_model.layers.0.mlp.gate.weight", False, False),                  # router stays BF16
    ("model.language_model.layers.0.mlp.shared_expert_gate.weight", False, False),    # scalar gate stays BF16
    ("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", True, True),
    ("model.language_model.layers.0.mlp.shared_expert.up_proj.weight", True, True),
    ("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", True, True),
]

# The two real fused-expert tensor names. Expectations encode TODAY'S bake
# behavior — both predicates currently return False (gap to fix in PR 7).
FUSED_EXPERT_NAMES = (
    "model.language_model.layers.0.mlp.experts.gate_up_proj",
    "model.language_model.layers.0.mlp.experts.down_proj",
)


class Qwen36MoeBakeClassificationTest(unittest.TestCase):
    def test_int4_target_split(self) -> None:
        for name, want_int4, _ in EXPECTATIONS:
            with self.subTest(name=name):
                self.assertEqual(
                    is_int4_target(name),
                    want_int4,
                    msg=f"is_int4_target({name!r}) classification changed",
                )

    def test_q4km_target_split(self) -> None:
        for name, _, want_q4km in EXPECTATIONS:
            with self.subTest(name=name):
                shape = shape_for(name)
                self.assertEqual(
                    is_q4km_target(name, shape, GROUP_SIZE),
                    want_q4km,
                    msg=f"is_q4km_target({name!r}, shape={shape}) classification changed",
                )

    def test_router_and_shared_gate_stay_raw(self) -> None:
        # Plan §2 is explicit: routers and the scalar shared-expert gate must
        # not be quantized. Pin both axes for both bake paths.
        for layer in (0, 3, 39):
            router = f"model.language_model.layers.{layer}.mlp.gate.weight"
            shared = f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"
            self.assertFalse(is_int4_target(router), router)
            self.assertFalse(is_int4_target(shared), shared)
            self.assertFalse(is_q4km_target(router, shape_for(router), GROUP_SIZE))
            self.assertFalse(is_q4km_target(shared, shape_for(shared), GROUP_SIZE))

    def test_shared_expert_projs_quantize(self) -> None:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.language_model.layers.0.mlp.shared_expert.{proj}.weight"
            shape = shape_for(name)
            self.assertTrue(is_int4_target(name), name)
            self.assertTrue(is_q4km_target(name, shape, GROUP_SIZE), name)

    def test_fused_experts_are_quantized(self) -> None:
        """
        Pin the post-PR 7-option-(c) reality: both bake paths classify the
        fused MoE expert tensors as INT4 targets, so the runtime sees a
        manifest with `Int4Quantized` packed nibbles + BF16 scale/zero
        sidecars instead of ~60 GiB of BF16 expert weight.

        Predicates accept the bare names (no `.weight` suffix). For
        `is_q4km_target` the 2D-only constraint is relaxed for fused names:
        3D shapes `[E, out, in]` are accepted as long as the per-expert
        `(out, in)` axes match the group_size / evenness rules. The actual
        quantization runs through `fused_expert_minmax_int4` (bake_int4.py)
        / `quantize_minmax_fused_experts` (bake_q4km.py) — plain min/max
        group-quant per expert, no Hessian / GPTQ.
        """
        for name in FUSED_EXPERT_NAMES:
            shape = shape_for(name)
            self.assertTrue(
                is_int4_target(name),
                f"is_int4_target({name!r}) must classify fused experts as INT4",
            )
            self.assertTrue(
                is_q4km_target(name, shape, GROUP_SIZE),
                f"is_q4km_target({name!r}, shape={shape}) must accept "
                f"3D fused expert tensors",
            )

    def test_fused_expert_predicate_rejects_bad_shapes(self) -> None:
        """
        Sanity-pin the divisibility gates that `is_q4km_target` enforces on
        fused experts. The runtime layout requires per-expert `(out, in)`
        axes to be divisible by `group_size`, with `in` even (for nibble
        packing). Catching mis-shaped fused tensors here prevents a silent
        partial-tile bake that would diverge from the runtime kernel.
        """
        bad_in_dim = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        # `in` not divisible by group_size -> reject.
        self.assertFalse(is_q4km_target(bad_in_dim, [4, 1024, 100], GROUP_SIZE))
        # `out` not divisible by group_size -> reject.
        self.assertFalse(is_q4km_target(bad_in_dim, [4, 100, 2048], GROUP_SIZE))
        # 2D shape on a fused-expert name -> reject (must be 3D).
        self.assertFalse(is_q4km_target(bad_in_dim, [1024, 2048], GROUP_SIZE))
        # `is_int4_target` doesn't see shape, but it must still accept the name.
        self.assertTrue(is_int4_target(bad_in_dim))

    def test_fused_expert_naming_invariants(self) -> None:
        """
        Sanity-pin the structural facts the rest of the runtime depends on:
          - exactly two fused tensors per layer (gate_up_proj, down_proj)
          - neither carries a `.weight` suffix
          - both are 3D with `num_experts` on axis 0
        """
        for name in FUSED_EXPERT_NAMES:
            self.assertFalse(name.endswith(".weight"), name)
            shape = shape_for(name)
            self.assertEqual(len(shape), 3, f"{name} shape={shape}")
            self.assertEqual(shape[0], NUM_EXPERTS, f"{name} expert axis")
        gate_up = shape_for(FUSED_EXPERT_NAMES[0])
        down = shape_for(FUSED_EXPERT_NAMES[1])
        # gate_up_proj fuses gate (moe_int) + up (moe_int) along axis 1.
        self.assertEqual(gate_up[1], 2 * MOE_INTER)
        self.assertEqual(gate_up[2], HIDDEN)
        self.assertEqual(down[1], HIDDEN)
        self.assertEqual(down[2], MOE_INTER)


if __name__ == "__main__":
    unittest.main()
