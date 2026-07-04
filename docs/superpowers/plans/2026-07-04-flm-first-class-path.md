# FLM First-Class Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make geo-quant FLM output load and run as the normal SuperSonic model path without `--int4`, without HF files, and with measurable FLM load/inference timings.

**Architecture:** SuperSonic treats an effective FLM source as the authority for executable weight mode. The loader opens FLM with runtime aliases enabled, reads config/tokenizer from FLM, probes the open store for native INT4 or BF16 fallback views, and rejects only truly incompatible FLM/flag combinations.

**Tech Stack:** Rust workspace (`runner`, `model-store`), Qwen3.6 MoE HIP path, geo-quant FLM Stage 3 native INT4 artifacts, cargo tests, env-gated e2e smoke tests.

---

### Task 1: Red Tests For File-Driven FLM Selection

**Files:**
- Modify: `crates/runner/src/bakes.rs`
- Modify: `crates/runner/src/qwen36_moe/flm_source.rs`
- Modify: `crates/runner/tests/flm_moe_main_path.rs`

- [ ] **Step 1: Add failing unit coverage for FLM aliases without `--int4`**

Add a test in `crates/runner/src/bakes.rs` that constructs the default Qwen3.6 MoE CLI with `.flm` `model_dir` and asserts `flm_source_open_options(&cli).unwrap().int4_runtime` is true even when `cli.int4` is false.

- [ ] **Step 2: Add failing unit coverage for probe-driven MoE mode**

Add tests in `crates/runner/src/qwen36_moe/flm_source.rs` that call the probe selection helper without a quant profile and assert:

```rust
assert_eq!(native_selection.mode, Qwen36WeightMode::Int4);
assert_eq!(native_selection.label, "INT4 native FLM");
assert_eq!(fallback_selection.mode, Qwen36WeightMode::Bf16);
assert_eq!(fallback_selection.label, "BF16");
```

- [ ] **Step 3: Remove `--int4` from FLM integration tests**

In `crates/runner/tests/flm_moe_main_path.rs`, remove `--int4` from the native FLM smoke command. Keep `--verify-flm-hashes` only where the test is explicitly checking hash option threading.

- [ ] **Step 4: Run the focused tests and confirm the expected red failures**

Run:

```bash
cargo test -p runner bakes::tests::flm_source_open_options -- --nocapture
cargo test -p runner qwen36_moe::flm_source::tests -- --nocapture
cargo test -p runner --test flm_moe_main_path moe_flm_main_path_output_contract_accepts_expected_logs -- --nocapture
```

Expected: at least one test fails because FLM INT4 aliases and MoE weight selection still depend on CLI quant flags.

### Task 2: Implement File-Driven FLM Open And MoE Selection

**Files:**
- Modify: `crates/runner/src/bakes.rs`
- Modify: `crates/runner/src/qwen36_moe/flm_source.rs`
- Modify: `crates/runner/src/flm_model_source.rs`

- [ ] **Step 1: Make FLM open options source-driven**

Change `flm_source_open_options` so `int4_runtime` is true for effective FLM sources and no longer depends on `effective_quant_profile(cli)`.

- [ ] **Step 2: Make Qwen3.6 MoE selection probe-driven**

Remove the required `QuantProfile` argument from `qwen36_moe_flm_weight_selection_for_store` and `qwen36_moe_flm_weight_selection_from_probe`. Select:

```rust
Some((LayoutTag::Int4Quantized, "u8")) => INT4 native FLM
Some((LayoutTag::Raw, "bf16")) => BF16 fallback
None => INT4 native FLM for unit fixtures without a store probe
other => error naming the incompatible probe
```

- [ ] **Step 3: Preserve explicit incompatible-flag rejection**

Keep `validate_flm_weight_source_options` rejecting q4km-like and int8 flags. Do not add CT or HF fallback logic to the native path.

- [ ] **Step 4: Run focused tests to green**

Run the same commands from Task 1 and confirm they pass.

### Task 3: Canonical Docs And Commands

**Files:**
- Modify: `docs/testing.md`
- Modify: geo-quant `docs/fast-load-model-format-design.md`

- [ ] **Step 1: Update SuperSonic FLM testing docs**

Document the canonical SuperSonic command without `--int4`:

```bash
cargo run -q -p runner --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir /mnt/data/tmp/flm-native-complete-path/qwen36-35b-a3b-supersonic-native-int4.flm \
  --backend hip \
  --device 0 \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --emit-stage-timings
```

- [ ] **Step 2: Update geo-quant FLM format docs**

Document that normal FLM consumers should not need a Hugging Face snapshot. HF JSON assets are optional transition material, and the runtime descriptor/config/tokenizer/tensor tables are the required fast-load contract.

### Task 4: Real Artifact Verification

**Files:**
- Modify docs only if verification output changes recommended commands.

- [ ] **Step 1: Validate the current geo-quant artifact**

Run:

```bash
/home/deano/.config/superpowers/worktrees/geo-quant/flm-first-class-path/.venv-rocm/bin/python scripts/flm_validate.py \
  /mnt/data/tmp/flm-native-complete-path/qwen36-35b-a3b-supersonic-native-int4.flm \
  --profile runnable-no-hf
```

Expected: validator prints OK with zero warnings.

- [ ] **Step 2: Run SuperSonic dry-run without `--int4`**

Run:

```bash
cargo run -q -p runner --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir /mnt/data/tmp/flm-native-complete-path/qwen36-35b-a3b-supersonic-native-int4.flm \
  --backend hip \
  --device 0 \
  --dry-run \
  --max-new-tokens 1 \
  --emit-stage-timings
```

Expected: command succeeds, reports `FLM weight mode: INT4 native FLM`, reports ready-for-decode, and does not mention HF files, fetch, bake, safetensors, or `INT4 GPTQ`.

- [ ] **Step 3: Run one-token inference without `--int4`**

Run:

```bash
cargo run -q -p runner --bin supersonic -- \
  --model qwen3.6-35b-a3b \
  --model-dir /mnt/data/tmp/flm-native-complete-path/qwen36-35b-a3b-supersonic-native-int4.flm \
  --backend hip \
  --device 0 \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --context-size 16 \
  --emit-stage-timings
```

Expected: command succeeds, prints generated ids/result summary, and emits stage timings that can be used as the baseline for subsequent fast-load and decode-speed work.
