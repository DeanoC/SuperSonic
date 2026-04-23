# Metal Qwen3.5 0.8B Bring-Up Notes

## Scope

These notes track the current Apple Metal bring-up for `qwen3.5-0.8b` on the
M4 Mac mini. The target remains intentionally narrow:

- get the workspace to build on macOS without HIP/CUDA toolchains
- get the CLI path running on `--backend metal`
- keep v1 functionality-first
- use replayed prefill for decode on Metal
- use the Python oracle as the main correctness reference

Performance tuning and a full native-kernel port are still explicitly deferred.

## Current implementation shape

The current Metal path is a pragmatic bridge, not a full native-kernel backend:

- the repo now compiles with a public `metal` backend
- Metal can be selected explicitly and is chosen by `auto` on this Mac when it
  is the only usable GPU backend
- the server is compile-safe but Metal is still runner-focused
- the Qwen v1 decode path on Metal uses replayed prefill by design
- the Python oracle launcher now prefers:
  - `SUPERSONIC_ORACLE_PYTHON`
  - `/opt/homebrew/bin/python3.11`
  - `python3.11`
  - `python3`

That last point mattered on this machine because the Command Line Tools
`python3` was `3.9.6` and was too old for the modern `qwen3_5` oracle stack.

## Bughunt gate

The repeatable parity gate for the current Qwen3.5 0.8B Metal bring-up is:

```bash
tests/metal/qwen35_bughunt_gate.sh
```

The script builds `qwen35_bughunt`, runs the checked-in token-ID manifest, uses
the Python oracle on CPU as truth, and writes JSON to
`/tmp/qwen35_bughunt_gate.json` by default.

Useful overrides:

- `QWEN35_MODEL_DIR=/path/to/Qwen3.5-0.8B`
- `QWEN35_BUGHUNT_PROMPT=code_prompt`
- `QWEN35_BUGHUNT_REPORT_JSON=/tmp/report.json`
- `QWEN35_ORACLE_DEVICE=cpu`

## Replay drift prompt family

The most informative prompt family so far is:

`The quick brown fox jumps over the lazy dog. Then it ran into the forest before sunset.`

Exact prompt tokenization:

- `01 760 The`
- `02 3841 Ġquick`
- `03 13477 Ġbrown`
- `04 37550 Ġfox`
- `05 33075 Ġjumps`
- `06 888 Ġover`
- `07 279 Ġthe`
- `08 15217 Ġlazy`
- `09 5388 Ġdog`
- `10 13 .`
- `11 4844 ĠThen`
- `12 424 Ġit`
- `13 10298 Ġran`
- `14 1083 Ġinto`
- `15 279 Ġthe`
- `16 13245 Ġforest`
- `17 1518 Ġbefore`
- `18 41564 Ġsunset`
- `19 13 .`

## Exact-prefix replay sweep

Using `--prompt-ids` and `--trace-replay-decode-step 0`, replay over prefix
lengths `12..19` produced:

- `12`: prefill `0.2498`, decode `0.1644`, replay `0.1595`, max replay layer `23`
- `13`: prefill `0.1985`, decode `0.1563`, replay `0.1645`, max replay layer `22`
- `14`: prefill `0.2282`, decode `0.2123`, replay `0.1711`, max replay layer `20`
- `15`: prefill `0.1711`, decode `0.1833`, replay `0.2056`, max replay layer `23`
- `16`: prefill `0.1865`, decode `0.1476`, replay `0.1486`, max replay layer `20`
- `17`: prefill `0.1608`, decode `0.1968`, replay `0.2030`, max replay layer `23`
- `18`: prefill `0.1739`, decode `0.1683`, replay `0.1425`, max replay layer `18`
- `19`: prefill `0.1425`, decode `0.1648`, replay `0.1283`, max replay layer `23`

Two takeaways from that sweep:

- replay drift is not monotonic in prompt length
- layer `23` is the most common hotspot, but not the only one

The interesting suffix is:

- `15`: `Ġthe`
- `16`: `Ġforest`
- `17`: `Ġbefore`
- `18`: `Ġsunset`
- `19`: `.`

## 17-token vs 18-token follow-up

The `17` vs `18` prefix comparison showed that the improvement at `18` is not a
generic "everything got better" change.

For prefix `17`:

- prefill delta: `0.1608`
- decode delta: `0.1968`
- replay delta: `0.2030`
- max replay layer: `23`
- layer-23 full-attention internals:
  - `q_prepared_delta=0.0726`
  - `k_prepared_delta=0.0562`
  - `attn_pregate_delta=0.0469`
  - `attn_output_delta=0.0625`
- layer-23 MLP internals:
  - `post_norm_delta=0.1250`
  - `swiglu_delta=0.0938`
  - `down_proj_delta=0.0781`

For prefix `18`:

- prefill delta: `0.1739`
- decode delta: `0.1683`
- replay delta: `0.1425`
- max replay layer: `18`
- layer-23 full-attention internals:
  - `q_prepared_delta=0.0521`
  - `k_prepared_delta=0.0453`
  - `attn_pregate_delta=0.0781`
  - `attn_output_delta=0.0234`
- layer-23 MLP internals:
  - `post_norm_delta=0.0938`
  - `swiglu_delta=0.0625`
  - `down_proj_delta=0.0312`

The most consistent improvement is in the late MLP tail, not in one clean
layer-23 attention fix.

## Layer-18 vs layer-23 read

Replay summaries for prefixes `17`, `18`, and `19`:

- prefix `17`
  - layer `18`: `attn=0.0117 post_norm=0.0625 mlp_out=0.0107 layer=0.0156`
  - layer `23`: `attn=0.0312 post_norm=0.1250 mlp_out=0.0781 layer=0.0469`
- prefix `18`
  - layer `18`: `attn=0.0156 post_norm=0.0781 mlp_out=0.0026 layer=0.0312`
  - layer `23`: `attn=0.0234 post_norm=0.0938 mlp_out=0.0312 layer=0.0312`
- prefix `19`
  - layer `18`: `attn=0.0078 post_norm=0.0625 mlp_out=0.0078 layer=0.0078`
  - layer `23`: `attn=0.0098 post_norm=0.0469 mlp_out=0.0156 layer=0.0312`

That kept layer `18` interesting as a visibility point, but not as the best
root-cause candidate. The more actionable target stayed the late MLP tail.

## MLP precision split

The Metal-only debug toggles for the Qwen MLP were split into:

- projection stage
- `swiglu`
- `down_proj`

Single toggles:

- `MLP_PROJ`: no useful effect
- `MLP_SWIGLU`: no useful effect
- `MLP_DOWN_PROJ`: no useful effect

Pairwise toggles:

- `proj + swiglu`
  - prefix `17`: `0.2030 -> 0.2253`
  - prefix `19`: `0.1283 -> 0.1462`
- `proj + down_proj`
  - no meaningful effect
- `swiglu + down_proj`
  - prefix `17`: `0.2030 -> 0.1877`
  - prefix `18`: `0.1425 -> 0.1320`
  - prefix `19`: `0.1283 -> 0.1539`

That strongly suggests the precision-sensitive boundary is the
`swiglu -> down_proj` handoff, not the gate/up projection stage.

## Late-layer filter

The composite debug toggle is:

- `SUPERSONIC_METAL_FORCE_F32_LATE_MLP_TAIL=1`

It defaults to F32 `swiglu + down_proj` from layer `23` onward. Additional
range filters are available with:

- `SUPERSONIC_METAL_FORCE_F32_MLP_LAYER_START`
- `SUPERSONIC_METAL_FORCE_F32_MLP_LAYER_END`

On the forest/sunset prompt family:

- baseline
  - prefix `17`: `0.2030`
  - prefix `18`: `0.1425`
  - prefix `19`: `0.1283`
- global `swiglu + down_proj`
  - prefix `17`: `0.1877`
  - prefix `18`: `0.1320`
  - prefix `19`: `0.1539`
- layers `22..23`
  - prefix `17`: `0.1934`
  - prefix `18`: `0.1487`
  - prefix `19`: `0.1338`
- layer `23` only
  - prefix `17`: `0.1932`
  - prefix `18`: `0.1389`
  - prefix `19`: `0.1273`

Single nearby late layers:

- `21-only`
  - prefix `17`: `0.2015`
  - prefix `18`: `0.1570`
  - prefix `19`: `0.1292`
- `22-only`
  - prefix `17`: `0.1929`
  - prefix `18`: `0.1462`
  - prefix `19`: `0.1305`
- `23-only`
  - prefix `17`: `0.1932`
  - prefix `18`: `0.1389`
  - prefix `19`: `0.1273`

So layer `23` remains the best local tradeoff. Layer `22` helps the worst
prefix a little, but is worse overall.

## Generalization check

The late-tail candidate was checked on a few unrelated prompts:

- `Hello world`
- `Write a haiku about rain on a tin roof.`
- `The quick brown fox jumps over the lazy dog. Then it ran into the forest before sunset.`
- `Explain what this Rust function does: fn add(a: i32, b: i32) -> i32 { a + b }`

Baseline vs `SUPERSONIC_METAL_FORCE_F32_LATE_MLP_TAIL=1`:

- `hello_world`
  - prefill: `0.1707 -> 0.1685`
  - decode: `0.2017 -> 0.2048`
  - replay: `0.1714 -> 0.1653`
  - max layer: `20 -> 20`
- `haiku_rain`
  - prefill: `0.1784 -> 0.1803`
  - decode: `0.2447 -> 0.2402`
  - replay: `0.2522 -> 0.2575`
  - max layer: `23 -> 23`
- `forest`
  - prefill: `0.1425 -> 0.1389`
  - decode: `0.1648 -> 0.1613`
  - replay: `0.1283 -> 0.1273`
  - max layer: `23 -> 23`
- `code_prompt`
  - prefill: `0.1798 -> 0.1786`
  - decode: `0.1701 -> 0.1668`
  - replay: `0.2150 -> 0.2199`
  - max layer: `18 -> 18`

So the late-tail candidate is real, but not universal:

- it helps on `hello_world`
- it helps on the `forest` prompt
- it slightly hurts replay on `haiku_rain`
- it hurts replay on the `code_prompt`

That keeps it in the "debug/localization aid" bucket, not the "new default"
bucket.

## Oracle runtime recovery and higher-sensitivity metrics

After restoring the Python oracle on Homebrew Python `3.11`, the replay trace
reporter grew two more distribution-sensitive summaries:

- mean absolute error (`*_mae`)
- scientific-notation MSE output (`*_mse`)

That was specifically to test the "final logits move, but the max-abs layer
trace still looks unchanged" cases.

### `haiku_rain`

Prompt IDs:

- `7734,264,6185,36974,883,10849,383,264,24007,14693,13`

Baseline vs `SUPERSONIC_METAL_FORCE_F32_LATE_MLP_TAIL=1`:

- replay logit max delta: `0.2522 -> 0.2575`
- final hidden MSE: `4.6e-5 -> 4.6e-5`
- final norm MSE: `7.905e-3 -> 7.961e-3`
- logit MSE: `2.484e-3 -> 2.484e-3`
- layer-23 MLP-out MSE: `1.2e-5 -> 1.2e-5`
- layer-23 layer-hidden MSE: `4.6e-5 -> 4.6e-5`

So on this prompt the late-tail toggle nudges scalar maxima, but the
layer-by-layer MAE/MSE profile is effectively unchanged.

### `code_prompt`

Prompt IDs:

- `814,20139,1092,411,32671,709,1503,25,5003,884,2784,25,585,18,17,11,292,25,585,18,17,8,1411,585,18,17,313,264,478,292,333`

Baseline vs `SUPERSONIC_METAL_FORCE_F32_LATE_MLP_TAIL=1`:

- replay logit max delta: `0.2150 -> 0.2199`
- final hidden MAE: `4.1449e-3 -> 4.1487e-3`
- final hidden MSE: `3.0768e-5 -> 3.0777e-5`
- final norm MAE: `5.1340e-2 -> 5.1449e-2`
- final norm MSE: `4.7332e-3 -> 4.7348e-3`
- logit MAE: `3.8205e-2 -> 3.9842e-2`
- logit MSE: `2.2904e-3 -> 2.4570e-3`
- layer-23 MLP-out MAE: `2.0060e-3 -> 2.0196e-3`
- layer-23 MLP-out MSE: `1.0182e-5 -> 1.0179e-5`
- layer-23 layer-hidden MAE: `4.1449e-3 -> 4.1487e-3`
- layer-23 layer-hidden MSE: `3.0768e-5 -> 3.0777e-5`

That changes the interpretation in a useful way:

- the late-tail toggle really is perturbing the tail
- the perturbation is tiny and distributed
- the final RMSNorm + LM-head projection amplifies it more than the per-layer
  hidden-state summaries do

## Current conclusion

The current best read is:

- the most promising remaining drift still lives in the late MLP tail
- the most precision-sensitive local boundary is `swiglu -> down_proj`
- `layer 23 only` remains the best local late-tail experiment so far
- the remaining regressions are not explained by one new giant visible block
  error
- instead, the replay differences now look like many tiny distributed
  hidden-state changes that become more obvious after final normalization and LM
  head projection

## Next step

The next high-value debugging pass is probably not another blunt F32 toggle.
The better next stage is to make the final projection more explainable, for
example by:

- tracing how final hidden-state deltas project into a small set of high-error
  logits
- ranking final-norm dimensions by contribution to observed logit drift
- comparing a few top-error logit rows against the late-tail hidden-state
  changes directly

## Final-projection instrumentation results

That next stage is now partially in place in the trace reporter:

- `top_logit_deltas`
- `logit_row_detail rank=N`
- `logit_dim_aggregate`

So instead of only explaining one worst logit row, replay traces now show:

- the top few logits by absolute replay error
- the dominant LM-head contributors for each of those rows
- the final-norm dimensions whose hidden-state drift contributes most across
  the selected bad logits as a group

### `code_prompt` projection summary

For the code prompt, the baseline top bad logits were:

- `95933`
- `101688`
- `95815`

The aggregate dominant final-norm dimensions across those rows were:

- `3`
- `750`
- `415`
- `540`
- `713`
- `38`

With the late-tail toggle enabled, the top bad logits changed to:

- `95933`
- `95852`
- `144278`

But the aggregate dominant dimensions still overlapped heavily:

- `3`
- `415`
- `750`
- `829`
- `38`
- `540`

So the late-tail precision tweak is not introducing a brand-new final
projection failure mode. It mostly changes which logit rows surface as the
worst offenders, while the important hidden dimensions remain substantially the
same.

That narrows the next debugging step further:

- focus on a small stable set of final-norm dimensions such as `3`, `415`,
  `750`, `540`, and `38`
- trace those dimensions backward through the late layers
- check whether they are coming primarily from one residual stream branch
  (attention vs MLP) rather than from the LM head itself

## Late-layer branch trace for the stable dimensions

The replay trace now prints:

- `tracked_logit_dims`
- `tracked_dims layer=N`

For each tracked dimension, the late-layer line reports signed native-minus-
oracle deltas at:

- `a`: attention residual state
- `pn`: post-attention norm
- `m`: MLP output
- `l`: final layer output

### Baseline code-prompt run

Tracked dimensions:

- `3,750,415,540,713,38`

From layers `18..22`, the pattern is mostly gradual accumulation in the
attention/post-norm path:

- dim `3`
  - layer `18`: `a=+0.0049 pn=+0.0312 m=+0.0000 l=+0.0039`
  - layer `21`: `a=+0.0078 pn=+0.0469 m=+0.0000 l=+0.0078`
  - layer `22`: `a=+0.0078 pn=+0.0312 m=-0.0001 l=+0.0078`
- dim `540`
  - layer `18`: `a=+0.0017 pn=+0.0117 m=+0.0005 l=+0.0020`
  - layer `22`: `a=+0.0059 pn=+0.0293 m=+0.0000 l=+0.0059`

But at layer `23`, several of the stable dims pick up their largest shift in
the MLP output itself:

- dim `3`: `a=+0.0000 pn=+0.0000 m=-0.0376 l=-0.0312`
- dim `415`: `a=-0.0039 pn=-0.0156 m=-0.0254 l=-0.0293`
- dim `713`: `a=-0.0059 pn=-0.0303 m=-0.0112 l=-0.0166`
- dim `750`: `a=+0.0029 pn=+0.0156 m=+0.0117 l=+0.0156`

So the baseline branch trace strengthens the working theory:

- layers `18..22` create a small upstream bias through attention/post-norm
- layer `23` MLP is where the biggest per-dimension jump shows up for the most
  important final-projection dimensions

### Late-tail code-prompt run

With `SUPERSONIC_METAL_FORCE_F32_LATE_MLP_TAIL=1`, the tracked set becomes:

- `3,415,750,829,38,540`

The layer-by-layer pattern is almost the same through layers `18..22`. The main
differences are:

- dim `829` replaces dim `713` in the aggregate top set
- layer `23` still shows the largest branch-local jump in the MLP output for
  dims `3`, `415`, and `750`

Layer `23` under the late-tail toggle:

- dim `3`: `a=+0.0000 pn=+0.0000 m=-0.0371 l=-0.0312`
- dim `415`: `a=-0.0039 pn=-0.0156 m=-0.0254 l=-0.0293`
- dim `750`: `a=+0.0029 pn=+0.0156 m=+0.0117 l=+0.0156`
- dim `829`: `a=+0.0078 pn=+0.0469 m=+0.0156 l=+0.0234`

That means the F32 late-tail experiment is not fundamentally changing the
branch diagnosis. It perturbs which dimensions and logits end up worst, but the
same broad structure remains:

- small bias already exists before the final layer
- the final layer's MLP branch is where the strongest dimension-local jump
  occurs for the stable bad dimensions

## Updated next step

The next best debugging pass is now more specific than "trace dimensions
backward":

- focus on the layer-23 MLP internals for dims `3`, `415`, `750`, and nearby
  replacements like `713` / `829`
- compare those dims across:
  - post-attention norm input
  - gate/up projection outputs if available
  - `swiglu`
  - `down_proj`
- check whether the dominant discrepancy is created at the `swiglu` activation
  itself or in the `down_proj` accumulation that writes back into hidden space

## Layer-23 MLP contributor aggregate

The latest trace pass adds a layer-23 aggregate over the dominant intermediate
channels that feed the tracked hidden dims through `down_proj`.

For the code prompt, the strongest recurrent intermediate channels were:

- `83`
- `47`
- `39`
- `23`
- then smaller contributors like `19` and `28`

The aggregate line now includes a simple decomposition:

- `gate_only`: `silu(g_native) * u_oracle - swiglu_oracle`
- `up_only`: `silu(g_oracle) * u_native - swiglu_oracle`

That gives a useful directional diagnosis:

- channel `83`
  - `sd=-0.0703`
  - `gate_only=-0.0775`
  - `up_only=+0.0107`
- channel `47`
  - `sd=+0.0508`
  - `gate_only=+0.0500`
  - `up_only=+0.0003`
- channel `39`
  - `sd=+0.0469`
  - `gate_only=+0.0437`
  - `up_only=-0.0018`
- channel `23`
  - `sd=+0.0586`
  - `gate_only=+0.0572`
  - `up_only=-0.0016`

So for the dominant channels, the `swiglu` drift is explained almost entirely
by the gate side, not the up-projection side.

There are a couple of smaller channels where `up_only` matters more:

- channel `19`
  - `sd=+0.0625`
  - `gate_only=+0.0043`
  - `up_only=+0.0551`
- channel `28`
  - `sd=-0.0273`
  - `gate_only=-0.0027`
  - `up_only=-0.0262`

But those channels have materially smaller aggregate contribution than
`83/47/39/23`, so the overall picture is still:

- the important layer-23 MLP drift is mostly gate-driven
- `up_proj` differences exist, but they are usually secondary on the channels
  that matter most for the bad hidden dims and bad logits

## Refined next step

The best next debugging pass is now even narrower:

- focus on the gate path in layer 23
- inspect whether the gate drift is already present in `gate_proj` output or is
  being amplified/nonlinearly skewed by `silu`
- compare the dominant channels `83`, `47`, `39`, and `23` under:
  - `gate_proj`
  - `silu(gate)`
  - `swiglu`

If that confirms the same pattern, the next likely implementation experiment is
not a full `swiglu + down_proj` precision toggle, but a much narrower gate-path
precision or accumulation experiment for the final MLP layer.

## Gate-path precision follow-up

That narrower gate-path experiment has now been wired up explicitly in the
Metal prefill path.

New debug toggles:

- `SUPERSONIC_METAL_FORCE_F32_MLP_GATE_PROJ`
- `SUPERSONIC_METAL_FORCE_F32_MLP_UP_PROJ`
- `SUPERSONIC_METAL_FORCE_F32_MLP_GATE_PATH`

The first two isolate the projection matmuls only. On the replayed decode trace
for the `code_prompt`, both were bit-for-bit unchanged from baseline:

- replay logit delta stayed `0.2150`
- final hidden / final norm / logit MAE+MSE stayed unchanged
- tracked layer-23 branch deltas stayed unchanged

That result is consistent with the current plumbing: toggling only one
projection stage still casts back to BF16 before `swiglu`, so this does not
meaningfully test whether the gate path needs to stay live in F32 through the
activation/multiply.

### Layer-23-only gate-path experiment

To test that boundary directly, the new
`SUPERSONIC_METAL_FORCE_F32_MLP_GATE_PATH=1` path keeps the gate side in F32
through `swiglu`, while `SUPERSONIC_METAL_FORCE_F32_MLP_LAYER_START=23` and
`SUPERSONIC_METAL_FORCE_F32_MLP_LAYER_END=23` confine the experiment to the
known hotspot.

The first implementation accidentally routed `up_proj` through the F32-input /
F32-output helper as well, which exploded parity and exposed a real experiment
bug:

- prefill logit delta jumped to `14.0309`
- `up_proj_delta` jumped to `6.1250`
- `swiglu_delta` jumped to `15.9629`

That routing bug is now fixed so `gate_path` only changes the gate side and
still casts the existing BF16 `up_proj` result to F32 for the multiply.

With that fix in place, the layer-23-only prefill trace for the `code_prompt`
shows:

- baseline
  - prefill logit delta: `0.1798`
  - final norm MAE: `4.2422e-2`
  - final norm MSE: `3.2329e-3`
  - logit MAE: `2.5819e-2`
  - logit MSE: `1.0687e-3`
- `GATE_PATH + layer23-only`
  - prefill logit delta: `0.1865`
  - final norm MAE: `4.2153e-2`
  - final norm MSE: `3.1335e-3`
  - logit MAE: `2.6293e-2`
  - logit MSE: `1.1086e-3`

So the controlled gate-path precision experiment does not help overall parity on
this prompt. It slightly improves some final-norm aggregates while making the
actual logit comparison slightly worse.

That is an important narrowing result:

- the dominant gate-side mismatch is not fixed just by keeping the gate live in
  F32 through `silu * up`
- the earlier gate-heavy contributor aggregates were real, but they do not
  imply a simple precision-lifetime fix
- the next likely target is the exact math/implementation boundary inside the
  gate-side nonlinear path or the later projection back into hidden space, not
  merely "make the gate side more F32"

## MLP self-check follow-up

The next pass added direct matmul/op self-checks to the `code_prompt`
layer-23 prefill trace so we could distinguish "bad operator math" from
"faithful operators acting on already-diverged inputs."

### `swiglu` op

New trace line:

- `swiglu_selfcheck native=0.0156 oracle=0.0189`

Interpretation:

- the Metal/host `swiglu_mul` path is internally self-consistent
- it is not materially worse than the oracle side
- `swiglu_delta=0.0703` is therefore mostly explained by input drift arriving
  at `swiglu`, not by a broken `swiglu` implementation

### `down_proj`

For the worst `down_proj` output row on the same prompt:

- `down_proj_matmul_selfcheck idx=0 native=1.5312 ref=1.5343 delta=0.0031`
- `oracle=1.5547 ref_oracle=1.5557 oracle_delta=0.0010`
- `native_vs_oracle=0.0234`

This is a strong sign that `down_proj` itself is not the main problem:

- the native output matches a direct row recomputation very closely
- the native-vs-oracle row gap is much larger than the native self-check error
- the row is faithfully multiplying slightly different `swiglu` inputs

### `gate_proj` and `up_proj`

Projection self-checks were then added for the worst output rows:

- `gate_proj_matmul_selfcheck idx=252 native=-2.8281 ref=-2.8356 delta=0.0075`
- `gate_proj ... oracle=-2.8906 ref_oracle=-2.8879 oracle_delta=0.0027`
- `gate_proj native_vs_oracle=0.0625`
- `up_proj_matmul_selfcheck idx=108 native=-5.5938 ref=-5.6046 delta=0.0108`
- `up_proj ... oracle=-5.6250 ref_oracle=-5.6099 oracle_delta=0.0151`
- `up_proj native_vs_oracle=0.0312`

Those rows tell the same story as `down_proj`:

- both projection matmuls are internally consistent
- their self-check errors are meaningfully smaller than their
  native-vs-oracle gaps
- the layer-23 MLP operators are mostly doing the expected math on slightly
  different inputs rather than injecting large extra error themselves

## Updated read on the hotspot

At this point the layer-23 MLP path looks numerically coherent end to end:

- `gate_proj` matmul looks correct
- `up_proj` matmul looks correct
- `swiglu` looks correct
- `down_proj` matmul looks correct

So the remaining late-tail drift on the `code_prompt` is more likely explained
by upstream hidden-state divergence feeding the MLP than by a broken MLP op.

The best next-stage investigation is now:

- move back upstream of the layer-23 MLP
- focus on the full-attention / post-attention-norm path feeding that MLP
- keep the current MLP self-checks in place as guardrails while tracing the
  earlier source of hidden-state drift

## Layer-23 attention self-check

The next pass moved one step earlier and traced the final full-attention layer
on the same `code_prompt`:

- `q_and_gate_delta=0.0625`
- `q_prepared_delta=0.0633`
- `k_prepared_delta=0.0619`
- `attn_pregate_delta=0.0469`
- `attn_output_delta=0.0391`

But the operator-level self-checks stayed materially smaller:

- `q_proj_matmul_selfcheck_delta=0.0162`
- `q_norm_selfcheck_delta=0.0146`
- `k_norm_selfcheck_delta=0.0150`
- `q_rope_selfcheck_delta=0.0077`
- `k_rope_selfcheck_delta=0.0038`

That points to the same interpretation as the MLP:

- layer-23 attention math looks mostly coherent on its current inputs
- it does not look like the primary source of the late-tail drift
- the hidden-state mismatch feeding the layer matters more than any obvious
  local operator bug in that layer

## Layer-22 linear self-check

The next trace moved to the last linear-attention block before layer 23:

- `normed_delta=0.1328`
- `qkv_delta=0.0938`
- `z_delta=0.1719`
- `post_conv_delta=0.0311`
- `attn_delta=0.0001`
- `proj_out_delta=0.0049`

Again, the operator checks were much smaller than the native-vs-oracle gaps:

- `qkv_matmul_selfcheck_delta=0.0318`
- `z_matmul_selfcheck_delta=0.0176`
- `conv_matmul_selfcheck_delta=0.0295`

To separate "bad linear input RMSNorm" from "correct RMSNorm of bad hidden",
the trace was extended with a direct `input_norm_selfcheck`:

- `input_norm_selfcheck native delta=0.0252`
- `input_norm_selfcheck oracle delta=0.0147`
- `native_vs_oracle normed_delta=0.1328`

That is another strong narrowing result:

- the layer-22 input RMSNorm is also internally consistent
- the large `normed_delta` is mostly explained by different hidden-state input
  arriving at the norm, not by the norm implementation itself

## Layer-18 linear self-check

The same direct trace was then run on layer 18, because the restart sweeps
started to suggest that this earlier linear block was one of the bigger
amplifiers in the late tail.

Layer-18 showed the same broad shape as layer 22:

- `input_norm_selfcheck native delta=0.0431`
- `input_norm_selfcheck oracle delta=0.0303`
- `native_vs_oracle normed_delta=0.1562`
- `qkv_matmul_selfcheck_delta=0.0531`
- `z_matmul_selfcheck_delta=0.0298`
- `conv_matmul_selfcheck_delta=0.0520`
- `attn_delta=0.0008`
- `proj_out_delta=0.0039`

That supports the same interpretation:

- layer 18 is not obviously failing inside the local linear-attention math
- the larger mismatch is already present in the hidden state arriving at the
  layer input norm
- the late linear blocks appear to be amplifying inherited drift more than
  inventing a new, isolated operator bug

## Layer-20 and layer-21 linear self-checks

The same tracing pass was then repeated for layers 20 and 21, since the
restart curve made them look like the highest-value remaining late-tail
contributors.

Layer 20:

- `input_norm_selfcheck native delta=0.0396`
- `input_norm_selfcheck oracle delta=0.0169`
- `native_vs_oracle normed_delta=0.1250`
- `qkv_matmul_selfcheck_delta=0.0382`
- `z_matmul_selfcheck_delta=0.0253`
- `conv_matmul_selfcheck_delta=0.0256`
- `attn_delta=0.0018`
- `proj_out_delta=0.0078`

Layer 21:

- `input_norm_selfcheck native delta=0.0519`
- `input_norm_selfcheck oracle delta=0.0549`
- `native_vs_oracle normed_delta=0.1562`
- `qkv_matmul_selfcheck_delta=0.0716`
- `z_matmul_selfcheck_delta=0.0187`
- `conv_matmul_selfcheck_delta=0.0812`
- `direct_recurrent_delta=0.0009`
- `attn_delta=0.0009`
- `proj_out_delta=0.0039`

Across layers `18`, `20`, `21`, and `22`, the same pattern now repeats:

- the input RMSNorm self-checks are not perfect, but they are materially
  smaller than the native-vs-oracle normalized-input mismatch
- the inner attention/recurrent outputs stay comparatively close once they are
  operating on the current native inputs
- the final projection deltas stay small
- where measured directly, the recurrent update itself also stays very close to
  the oracle (`direct_recurrent_delta=0.0009` on layer 21)

That is a strong sign that the whole late linear-attention run is mostly
faithful to its inputs, and that the larger bug is probably earlier hidden-state
or recurrent-state drift getting carried forward and then amplified.

## Earlier restart sweep and mid-stack traces

The restart sweep was pushed earlier through layers `10..13`:

- restart at layer `10`: `0.1822`
- restart at layer `11`: `0.1565`
- restart at layer `12`: `0.1308`
- restart at layer `13`: `0.1632`

That curve is not perfectly monotonic, but it does still say something useful:

- restarting as early as layer `10` does essentially nothing
- restarting after layer `11` helps somewhat
- restarting after layer `12` helps a lot more
- by layer `13`, the restart metric becomes noisy enough that it is not safe to
  treat it as a strict monotone boundary detector on its own

So the direct traces became the tiebreaker.

### Layer-12 and layer-13 linear traces

Layer 12:

- `native_vs_oracle normed_delta=0.1250`
- `qkv_matmul_selfcheck_delta=0.0343`
- `z_matmul_selfcheck_delta=0.0196`
- `conv_matmul_selfcheck_delta=0.0073`
- `attn_delta=0.0003`
- `proj_out_delta=0.0013`

Layer 13:

- `native_vs_oracle normed_delta=0.1250`
- `qkv_matmul_selfcheck_delta=0.0325`
- `z_matmul_selfcheck_delta=0.0149`
- `conv_matmul_selfcheck_delta=0.0047`
- `attn_delta=0.0001`
- `proj_out_delta=0.0020`

These two layers still look like the later linear blocks:

- the input-side mismatch is much larger than the local operator self-checks
- the direct recurrent / attention-style output remains very close
- there is still no concrete sign that the linear primitive chain itself is the
  first failing computation

### Layer-11 full-attention trace

Because restarting after layer 11 helped more than restarting before it, the
mid-stack full-attention block at layer 11 was traced directly too.

It also looked locally healthy:

- `q_proj_matmul_selfcheck_delta=0.0212`
- `q_norm_selfcheck_delta=0.0147`
- `k_norm_selfcheck_delta=0.0156`
- `q_rope_selfcheck_delta=0.0220`
- `attn_output_delta=0.0065`

So layer 11 joins layers `19` and `23` in the "full-attention math mostly fine
on the current inputs" bucket.

## Restart-layer localization

Once the late operators all looked locally sane, the existing
`--trace-prefill-restart-layer` path became the highest-value tool. It lets the
native tail restart from an oracle-correct hidden row at a chosen layer.

### Restart from layer 22

Restarting the native tail from the oracle hidden at layer 22 produced:

- baseline prefill logit max delta: `0.1798`
- restarted tail logit max delta: `0.0602`
- restarted layer-23 hidden delta: `0.0078`

That is the clearest evidence so far that:

- the final layer itself is mostly healthy
- most of the final error is already present before layer 23 begins

### Restart from layer 21

Restarting one layer earlier produced:

- restarted tail logit max delta: `0.0864`
- restarted layer-22 hidden delta: `0.0078`
- restarted layer-23 hidden delta: `0.0156`

Comparing the two restarts:

- layer `21 -> 22 -> 23` contributes some additional drift
- but the larger share of the original `0.1798` final logit delta is already
  baked in before layer 21

### Restart curve through the late tail

The restart sweep was pushed farther back through layers `14..22`:

- baseline prefill logit max delta: `0.1798`
- restart at layer `14`: `0.1224`
- restart at layer `15`: `0.1250`
- restart at layer `16`: `0.1318`
- restart at layer `17`: `0.1234`
- restart at layer `18`: `0.0874`
- restart at layer `19`: `0.0867`
- restart at layer `20`: `0.1094`
- restart at layer `21`: `0.0864`
- restart at layer `22`: `0.0602`

The restarted hidden deltas stayed relatively small even when the remaining
final logit delta was still noticeable. Representative tails:

- restart at `18`
  - layer `19` hidden delta: `0.0039`
  - layer `20` hidden delta: `0.0039`
  - layer `21` hidden delta: `0.0078`
  - layer `22` hidden delta: `0.0078`
  - layer `23` hidden delta: `0.0156`
- restart at `19`
  - layer `20` hidden delta: `0.0039`
  - layer `21` hidden delta: `0.0078`
  - layer `22` hidden delta: `0.0078`
  - layer `23` hidden delta: `0.0156`
- restart at `20`
  - layer `21` hidden delta: `0.0039`
  - layer `22` hidden delta: `0.0078`
  - layer `23` hidden delta: `0.0078`
- restart at `21`
  - layer `22` hidden delta: `0.0078`
  - layer `23` hidden delta: `0.0156`
- restart at `22`
  - layer `23` hidden delta: `0.0078`

The useful read from that wider curve is:

- the full-attention checkpoints at layers `15` and `19` barely change the
  remaining final delta when used as restart boundaries
- the biggest late-tail drops come from crossing layer `18`, then the
  `20..22` linear-attention run
- the earlier sweep suggests the drift is already materially established by the
  layer-11 / layer-12 region, even if the restart metric there is not perfectly
  monotone
- there is still no sign of a single catastrophic operator failure
- the more likely shape is "hidden drift accumulates earlier, then the late
  linear stack amplifies it"

## Updated next stage

The current best read is:

- layers `18`, `20`, `21`, and `22` are the highest-value late-tail
  amplifiers to study
- the surrounding full-attention checkpoints at `15` and `19` look relatively
  benign as restart boundaries
- direct traces on layers `18`, `20`, `21`, and `22` all say "local math mostly
  fine, incoming hidden already drifted"
- layer `12`, layer `13`, and even the mid-stack full-attention layer `11`
  now tell the same story
- that points the investigation away from late full-attention and toward the
  point where upstream hidden/state drift first appears, not where it is most
  visible

So the next stage should prioritize:

- pushing the restart sweep earlier than layer `10`, since the direct traces now
  say the visible mid/late-stack operators are not the root cause
- adding or using instrumentation that can compare earlier hidden/state
  carry-over checkpoints directly, instead of relying only on final-logit
  restart curves
- treating the recurrent update itself as lower priority for suspicion than the
  upstream hidden/input state that feeds it

## Position-0 code-prompt follow-up

The next pass stayed on the same `code_prompt` and switched to direct
position-`0` traces with more explicit self-checks.

### Layer 6 gated stage is locally healthy

Layer `6` had looked suspicious because it showed a large
`gated_delta=0.2500`. That turned out to be misleading.

Once the trace added a direct gated RMSNorm reference check, layer `6` produced:

- `gated_selfcheck native delta=0.0052`
- `gated_selfcheck oracle delta=0.0051`
- `native_vs_oracle gated delta=0.2500`
- `direct_recurrent_delta=0.0001`
- `attn_delta=0.0001`
- `proj_out_delta=0.0156`

So layer `6` is not a local gated-RMSNorm bug. The gated tensor is already
different going in, but both the native and oracle implementations track their
own local reference closely.

### Layer 4 is an amplifier, not a uniquely native failure

Layer `4` remains the earliest place where the position-`0` linear trace gets
dramatically louder:

- `input_norm ... native_vs_oracle=0.1250`
- `qkv_delta=0.5000`
- `conv_window_delta=0.5000`
- `z_delta=0.0625`

But the new oracle-side self-checks changed the interpretation of that result.
For layer `4`:

- `qkv_matmul_selfcheck_delta=0.1841`
- `oracle_qkv_selfcheck_delta=0.2206`
- `z_matmul_selfcheck_delta=0.0245`
- `oracle_z_selfcheck_delta=0.0288`
- `gated_selfcheck native delta=0.0055`
- `gated_selfcheck oracle delta=0.0059`
- `direct_recurrent_delta=0.0014`
- `attn_delta=0.0021`
- `proj_out_delta=0.0009`

That is important because the big layer-4 `qkv` mismatch is not unique to the
Metal/native path. Both native and oracle projections are comparably far from
the same neutral matmul reference on this prompt/position, while the later
stages in the linear block still look locally tight.

An additional input-side detail made that boundary easier to read:

- `input_hidden_detail idx=0 hidden_native=-0.4141 hidden_oracle=-0.4219 hidden_delta=0.0078`
- `input_norm_detail idx=0 hidden_native=-0.4141 hidden_oracle=-0.4219`
  `inv_rms_native=11.622281 inv_rms_oracle=11.640912 scale=1.1758`
  `normed_native=-5.6562 normed_oracle=-5.7812`

So the loud `normed_delta=0.1250` at layer `4` is not because the RMSNorm is
broken. It is amplifying a larger pre-norm hidden mismatch (`0.0078`) at a
dimension whose row scale is already in a high-gain regime.

### Layer 0 is still a clean anchor

To make sure the new layer-4 story was not just a global trace artifact, the
front-of-model linear layer was re-checked on the same `code_prompt`:

- `normed_delta=0.0000`
- `qkv_matmul_selfcheck_delta=0.0313`
- `oracle_qkv_selfcheck_delta=0.0312`
- `z_matmul_selfcheck_delta=0.0101`
- `oracle_z_selfcheck_delta=0.0075`
- `gated_selfcheck native delta=0.0088`
- `gated_selfcheck oracle delta=0.0112`
- `direct_recurrent_delta=0.0002`
- `attn_delta=0.0002`
- `proj_out_delta=0.0010`

The preceding full-attention checkpoint at layer `3` also still looks locally
healthy on the same prompt/position:

- `q_and_gate_delta=0.0312`
- `q_proj_delta=0.0312`
- `k_proj_delta=0.0156`
- `v_proj_delta=0.0117`
- `q_prepared_delta=0.0452`
- `k_prepared_delta=0.0388`
- `attn_pregate_delta=0.0117`
- `attn_output_delta=0.0039`
- `q_proj_matmul_selfcheck_delta=0.0286`
- `q_norm_selfcheck_delta=0.0262`
- `k_norm_selfcheck_delta=0.0104`
- `q_rope_selfcheck_delta=0.0000`
- `k_rope_selfcheck_delta=0.0000`

The linear layers between those endpoints also still look much closer to layer
`0` than to layer `4`:

- layer `1`
  - `input_norm native_vs_oracle=0.0469`
  - `input_hidden_detail idx=399 hidden_native=-0.1006 hidden_oracle=-0.1021 hidden_delta=0.0015`
  - `input_norm_detail idx=399 inv_rms_native=22.229784 inv_rms_oracle=22.176138 scale=1.3926`
  - `qkv_delta=0.0625`
  - `gated_delta=0.0156`
  - `proj_out_delta=0.0010`
- layer `2`
  - `input_norm native_vs_oracle=0.0625`
  - `input_hidden_detail idx=0 hidden_native=-0.4961 hidden_oracle=-0.5000 hidden_delta=0.0039`
  - `input_norm_detail idx=0 inv_rms_native=17.822954 inv_rms_oracle=17.794245 scale=1.2500`
  - `qkv_delta=0.0625`
  - `gated_delta=0.0078`
  - `proj_out_delta=0.0078`

That comparison matters because layer `1` actually shows an even larger
`inv_rms`, but the hidden mismatch there is much smaller, so the normalized
output remains relatively tame. Layer `4` looks loud primarily because the
incoming hidden mismatch itself has already grown by that point.

So the current best read is:

- layer `0` is locally healthy on the exact prompt where the final prefill
  delta is `0.1798`
- layer `3` full attention is also locally healthy there
- layer `6` is also locally healthy in the stage that looked worst at first
- layer `4` is still the first strong visible amplifier, but not as a
  Metal-only projection failure
- layer `1` and `2` show that high-gain RMSNorm alone is not enough to explain
  the layer-4 jump; the pre-norm hidden mismatch itself is already larger there
- the working shape is now "small earlier hidden/input drift gets amplified by
  layernorm/projection sensitivity around layer `4`", not "layer `6` gated math
  is broken"
