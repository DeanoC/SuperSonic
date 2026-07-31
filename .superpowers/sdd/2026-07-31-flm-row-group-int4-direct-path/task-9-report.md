# Task 9 Report: G32 Row-Group INT4 Production Execution

## Status

Complete against base commit `1472c74`. Qwen3.6 full attention, linear
attention, routed/shared FFN, grouped routed-expert prefill, chained decode,
and persistent decode now consume the Task 8 projection descriptor for every
INT4 scalar, 8-wide, pair, and WMMA operation. Encoding 2 is admitted at the
converted chained and persistent boundaries; unconverted entry points remain
fail-closed.

## TDD Record

### RED

The required release `gfx1100` no-run command was run before implementation.
The starting worktree already exposed stale Task 7 aggregate consumers in the
runner engine/tests, which did not initialize the new descriptor fields.

After parameterizing the independent oracles and adding focused G32 tests, the
same command remained genuinely red for Task 9 behavior:

- the grouped-expert safe/raw launcher accepted global `group_size` and flat
  scale/zero pointers instead of per-projection descriptors;
- staged FFN/attention/linear parameter structs did not carry the exact
  projection descriptor;
- runtime and diagnostic struct literals still constructed only legacy flat
  sidecar pointers.

A final focused routing test was also introduced red. It referenced
`optimized_batched_ffn_supports_group_size` before that predicate existed and
failed with `E0425`; this proved G32 had no explicit fail-closed routing around
the still-G128-only generic shared-expert batch workflow.

### GREEN

The implementation made each staged launcher accept one aggregate
`Qwen36MoeInt4ScaleDesc` and pass its exact leaf descriptor by value into the
phase kernel. The grouped routed-expert launcher accepts separate gate-up and
down descriptors. Runtime constructors now derive these descriptors from the
loaded storage view, so row-group strides and the null zero plane are not
reconstructed from a global group size.

The final routing predicate admits only G128 to the complete optimized batch
workflow. G32 therefore falls through to the descriptor-driven per-token
workflow; the converted low-level grouped routed-expert launcher remains
directly available and covered by G32 GPU parity.

## Path Inventory

| Material path | Projection descriptors | Compute coverage | G32 evidence |
| --- | --- | --- | --- |
| Full attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | dq8 scalar and WMMA | multilayer chained/persistent parity; descriptor assertions |
| Linear attention | `linear_in_proj_qkv`, `linear_in_proj_z`, `linear_out_proj` | dq8 scalar and WMMA | multilayer chained/persistent parity; descriptor assertions |
| Routed FFN | `experts_gate_up`, `experts_down` | dq8 scalar and WMMA | multilayer parity and grouped prefill parity |
| Shared FFN gate/up | `shared_expert_gate_proj`, `shared_expert_up_proj` | paired dq8 and scalar fallback | multilayer isolated shared/routed diagnostics |
| Shared FFN down | `shared_expert_down_proj` | dq8 scalar and WMMA | multilayer isolated shared diagnostics |
| Grouped prefill | explicit routed gate-up/down leaves | descriptor 8-wide and WMMA | N=4/E=8, N=16/E=64, N=64/E=256 GPU shapes |
| Chained decode | per-step aggregate descriptor | full/linear/FFN phase launches | tracked four-layer oracle and boundary checks |
| Persistent decode | per-layer aggregate descriptor array | full/linear/FFN phase launches | persistent vs chained exact hidden parity |

Every expected row-group descriptor assertion requires:

- `encoding == 2`;
- `zero == nullptr`;
- `input_group_size == 32`;
- `output_group_size == 1`;
- `implicit_zero_code == 8`.

## Oracle Model

All four independent Python oracles accept `layout="row_group"` and require
G32 for that layout. Row-group scale shape is `[O, K/32]` or
`[E, O, K/32]`, and the zero plane is omitted.

The canonical row-group value is `(code - 8) * BF16(scale)`. Full and linear
attention references BF16-round reconstructed weights because those gfx1100
paths feed BF16 WMMA operands. FFN row-group references retain the exact F32
canonical value because scalar/dq8 accumulation consumes it directly. This
distinction was established from the GPU output while keeping all existing
layer-boundary tolerances unchanged; globally BF16-rounding FFN reconstruction
produced an incorrect `0.0625` expert-boundary delta.

The tracked multilayer fixture was regenerated with:

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python \
  oracle/qwen36_moe_multilayer_oracle.py \
  --state fresh --int4 --int4-layout row_group --int4-group-size 32 \
  --out oracle/fixtures/qwen36_moe_multilayer_int4_v1.json
```

## Guard Transition

`ensure_legacy_int4_execution_supported` retains its historical name but now
validates row-group descriptors instead of rejecting the storage kind
unconditionally. Encoding 2 is admitted only when:

- the caller identifies the execution path as `chained decode` or
  `persistent decode`;
- every row-group projection successfully builds the canonical encoding-2,
  null-zero descriptor.

Any other named path still rejects row-group before device work. Bridge-side
validation independently checks each projection's expected expert/output/input
geometry and encoding contract. Tile-v1 encoding 1 continues to require G128
plus an explicit zero plane. FP8 encoding 3 continues to require a scale plane
and no zero plane, and is never treated as INT4 for WMMA selection.

The complete optimized batched prefill workflow remains fail-closed for G32
because its generic shared-expert matmuls are still tile-v1 G128 code. G32 uses
the descriptor-driven per-token workflow instead. The grouped routed-expert
launcher itself is converted and accepts G32 descriptors.

## Compile-Required Scope

The brief's core files required these tightly scoped supporting edits:

- `helpers.cuh` and `qwen36_moe.hip`: descriptor dq8/pair/WMMA wrappers and
  descriptor-valued staged kernel wrappers;
- kernel-ffi descriptor/launch declarations: a disabled leaf constructor and
  ABI signature updates;
- runtime decode/prefill/persistent constructors and runner prewarm: build and
  propagate exact descriptors at every production struct literal;
- the layer loader guard and diagnostic test: encoding transition and stale
  tile-v1 struct-literal repair;
- the FFN oracle and tracked fixture: independent routed/shared row-group
  reference data needed by the required multilayer gate.

Legacy direct scale/zero helpers remain only for isolated Task 8 tile-v1
known-byte coverage. Metal native helpers and the generic optimized batched
shared-expert workflow remain explicitly G128-only. Neither is selectable for
G32 production execution.

## Verification

All commands used `SUPERSONIC_BACKENDS=hip`, `HIP_ARCH=gfx1100`, and
`CARGO_TARGET_DIR=/home/deano/projects/SuperSonicBase/target` where applicable.

- Required three-test release no-run: passed; all three test executables built.
- `qwen36_moe_batched_prefill_attn_kernel_parity`: 1 passed.
- `qwen36_moe_batched_prefill_grouped_expert_parity`: 1 passed across all
  three G32 shapes; production shape max absolute error `9.76562e-4`, minimum
  cosine `0.999998`.
- `qwen36_moe_multilayer_parity`: 4 passed; chained/persistent hidden states
  were exact, folded lm-head max absolute error `0.03125`, and both negative
  perturbation tests were rejected by their unchanged bounds.
- Task 8 known-byte suites: tile-v1 2 passed; row-group scalar/dq8/WMMA and
  validation 13 passed.
- Guard/encoding/routing focused unit tests: 3 passed.
- `kernel-ffi --lib --no-run`: passed.
- `qwen36_layer0_diagnostic --no-run`: passed.
- Python `py_compile` for all four changed oracles: passed.
- Targeted `rustfmt --check` for every changed Rust file: passed.
- `git diff --check 1472c74`: passed.

Repository-wide `cargo fmt --all -- --check` still reports pre-existing drift
in untouched `crates/gpu-hal/build.rs` and
`crates/runner/src/bin/int4_test.rs`. Those unrelated files were not modified;
all Task 9 Rust files pass direct `rustfmt --check`.

## Self-Review

The exact `1472c74` base-to-worktree diff was reviewed for ABI agreement,
projection-to-descriptor mapping, encoding selection, pointer/stride geometry,
remaining direct phase indexing, G128 assumptions, and tolerance changes.
No direct phase-specific scale/zero indexing remains in full attention, linear
attention, FFN, grouped expert, or persistent phase code. No existing parity
tolerance was relaxed.

## Concerns

The G32 path intentionally gives up the complete optimized batched prefill
workflow until its generic shared-expert branch is descriptor-converted. This
is a performance limitation, not a correctness fallback: G32 executes through
the converted per-token workflow, while the grouped routed-expert primitive is
already G32-capable and parity-tested at production geometry.
