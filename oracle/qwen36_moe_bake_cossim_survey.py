#!/usr/bin/env python3
"""Per-tensor cos_sim survey of an INT4 GPTQ bake vs the safetensors
ground truth. Prints a simple table so we can see which tensors are
limiting bake quality after the latest baker changes (scale search on
both dense + fused expert paths).

Run:
  ~/venvs/rocm/bin/python oracle/qwen36_moe_bake_cossim_survey.py \\
      --model-dir <snapshot> --bake-dir <snapshot>/.supersonic/v2-int4-gptq \\
      --layers 0,3 --out /tmp/qwen36_cossim_survey.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

# Make sibling modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bake_loader import load_int4, load_int4_3d  # noqa: E402


def find_shard_for(model_dir: Path, name: str) -> Path:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wm = json.loads(idx_path.read_text())["weight_map"]
        if name not in wm:
            raise SystemExit(f"tensor not in safetensors index: {name}")
        return model_dir / wm[name]
    single = model_dir / "model.safetensors"
    if single.exists():
        return single
    raise SystemExit(f"no safetensors at {model_dir}")


def load_safetensors(model_dir: Path, name: str) -> torch.Tensor:
    path = find_shard_for(model_dir, name)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def cos_sim_2d(recon: torch.Tensor, ref: torch.Tensor) -> float:
    # F64 reductions: F32 dot/norm on 100M+ element tensors loses enough
    # precision to push cos_sim above 1.0 (lm_head.weight @ 508M elements
    # comes out at 1.20 in F32 but the correct value is 0.987 in F64).
    a = recon.to(torch.float64).reshape(-1)
    b = ref.to(torch.float64).reshape(-1)
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((a @ b).item() / denom)


def survey_dense(
    bake_dir: str, model_dir: Path, name: str
) -> tuple[tuple[int, ...], int, float]:
    recon, _packed, _s, _z = load_int4(bake_dir, name)
    ref = load_safetensors(model_dir, name)
    if recon.shape != ref.shape:
        raise SystemExit(
            f"{name}: shape mismatch recon {tuple(recon.shape)} vs "
            f"safetensors {tuple(ref.shape)}"
        )
    out, in_ = ref.shape[-2], ref.shape[-1]
    tiles = (out // 128) * (in_ // 128) if (out % 128 == 0 and in_ % 128 == 0) else 0
    return tuple(ref.shape), tiles, cos_sim_2d(recon, ref)


def survey_fused(
    bake_dir: str, model_dir: Path, name: str
) -> tuple[tuple[int, ...], int, float, float, float]:
    """Returns (shape, tiles_per_expert, mean cos_sim, min, max)."""
    recon, _packed, _s, _z = load_int4_3d(bake_dir, name)
    ref = load_safetensors(model_dir, name)
    if recon.shape != ref.shape:
        raise SystemExit(
            f"{name}: shape mismatch recon {tuple(recon.shape)} vs "
            f"safetensors {tuple(ref.shape)}"
        )
    e, out, in_ = ref.shape
    tiles = (out // 128) * (in_ // 128)
    cs = []
    for i in range(e):
        cs.append(cos_sim_2d(recon[i], ref[i]))
    cs_t = torch.tensor(cs)
    return tuple(ref.shape), tiles, float(cs_t.mean()), float(cs_t.min()), float(cs_t.max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to HuggingFace safetensors directory.")
    p.add_argument("--bake-dir", type=Path, required=True,
                   help="Path to SuperSonic INT4 GPTQ bake directory.")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--layers", default="0,3",
                   help="Comma-separated layer indices to survey "
                        "(default: 0 = first linear-attn, 3 = first full-attn).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional TSV path; otherwise stdout only.")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    bake_dir = str(args.bake_dir)

    rows: list[tuple[str, str, str, float, float, float]] = []  # (name, shape, tiles, mean, min, max)

    def add_dense(name: str) -> None:
        try:
            shape, tiles, cs = survey_dense(bake_dir, args.model_dir, name)
        except (SystemExit, KeyError) as e:
            return  # tensor doesn't exist for this layer type — quiet skip.
        rows.append((name, str(shape), str(tiles), cs, cs, cs))

    def add_fused(name: str) -> None:
        try:
            shape, tiles, cs_mean, cs_min, cs_max = survey_fused(
                bake_dir, args.model_dir, name
            )
        except (SystemExit, KeyError):
            return
        rows.append((name, str(shape), f"{tiles}/expert", cs_mean, cs_min, cs_max))

    for li in layers:
        lp = f"{args.weight_prefix}.layers.{li}"
        # Linear attn projections (only when this layer is linear).
        for sub in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                    "linear_attn.out_proj"):
            add_dense(f"{lp}.{sub}.weight")
        # Full attn projections (only when this layer is full attn).
        for sub in ("self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj"):
            add_dense(f"{lp}.{sub}.weight")
        # MoE shared expert (always present).
        for sub in ("shared_expert.gate_proj", "shared_expert.up_proj",
                    "shared_expert.down_proj"):
            add_dense(f"{lp}.mlp.{sub}.weight")
        # MoE fused experts.
        add_fused(f"{lp}.mlp.experts.gate_up_proj")
        add_fused(f"{lp}.mlp.experts.down_proj")

    # lm_head if quantized in the bake.
    add_dense("lm_head.weight")

    # Print table.
    name_w = max(20, max(len(r[0]) for r in rows) if rows else 20)
    shape_w = max(18, max(len(r[1]) for r in rows) if rows else 18)
    print(f"{'tensor'.ljust(name_w)}  {'shape'.ljust(shape_w)}  "
          f"{'tiles'.rjust(12)}  {'cos_sim'.rjust(8)}  "
          f"{'min'.rjust(8)}  {'max'.rjust(8)}")
    print("-" * (name_w + shape_w + 12 + 8 + 8 + 8 + 10))
    for name, shape, tiles, mean, mn, mx in rows:
        flag = "  <0.98" if mean < 0.98 else ""
        print(f"{name.ljust(name_w)}  {shape.ljust(shape_w)}  "
              f"{tiles.rjust(12)}  {mean:8.4f}  {mn:8.4f}  {mx:8.4f}{flag}")

    if args.out is not None:
        with args.out.open("w") as f:
            f.write("name\tshape\ttiles\tcos_sim_mean\tcos_sim_min\tcos_sim_max\n")
            for name, shape, tiles, mean, mn, mx in rows:
                f.write(f"{name}\t{shape}\t{tiles}\t{mean:.6f}\t{mn:.6f}\t{mx:.6f}\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
