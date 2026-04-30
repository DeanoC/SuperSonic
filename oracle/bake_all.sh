#!/usr/bin/env bash
# bake_all.sh — produce INT4/Q4KM (and optionally BF16) bakes for every
# configured model on a big-box producer, and optionally publish each to the
# GitHub release as it finishes.
#
# Point env vars at the model directories you want to bake. Any unset var is
# silently skipped, so you can bake a subset without editing the script.
#
#   QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
#   QWEN_2B_DIR=/models/Qwen3.5-2B \
#   QWEN_4B_DIR=/models/Qwen3.5-4B \
#   QWEN_9B_DIR=/models/Qwen3.5-9B \
#   QWEN_36_27B_DIR=/models/Qwen3.6-27B \
#   QWEN_36_27B_GGUF=/models/Qwen3.6-27B-Q4_K_M.gguf \
#   QWEN_36_27B_ACTIVATION_DIR=/models/Qwen3.6-27B-FP8 \
#   Q4KM_GPTQ_EXTRA_ARGS='--device-map auto --max-memory {"0":"22GiB","cpu":"48GiB"} --offload-folder /tmp/qwen36-offload' \
#   GEMMA_E2B_DIR=/models/gemma-4-E2B \
#   GEMMA_E4B_DIR=/models/gemma-4-E4B \
#   PHI4_MINI_DIR=/models/Phi-4-mini-instruct \
#   ./oracle/bake_all.sh --upload
#
# Flags:
#   --upload        Publish each bake via oracle/upload_bake.py once it succeeds.
#   --bf16          Also produce Qwen BF16 bakes (triggered via a throwaway
#                   `supersonic` run). Gemma 4 has no BF16 bake format.
#   --fp8-native    Also produce Qwen FP8-native bakes (needs a model source
#                   with FP8 tensors; e.g. lovedheart/Qwen3.5-*-FP8). Gemma 4
#                   has no FP8-native bake format.
#   --q4km          Also produce Q4KM bakes (Qwen only; Gemma/Phi4 not yet
#                   supported by oracle/bake_q4km.py).
#   --q4km-gptq     Also produce Q4KM-sourced GPTQ/native INT4 bakes (Qwen
#                   only; requires the matching ${ENV_VAR%_DIR}_GGUF path).
#   --no-int4       Skip INT4 bakes (useful with --bf16 for a BF16-only run).
#   --force         Re-bake even if a valid bake already exists.
#   --stop-on-error Abort the whole batch if any single bake fails.
#                   (Default: keep going and report failures at the end.)
#
# 9B INT4 note: GPTQ calibration for qwen3.5-9b loads the full BF16 model
# (~18 GiB) into GPU memory, which OOMs on ≤16 GiB cards (including gfx1150).
# Bake it on a box with ≥24 GiB GPU RAM or skip it by leaving QWEN_9B_DIR
# unset. `oracle/bake_int4.py --help` documents the memory requirements.
#
# Auto-update on lm_head INT4: existing Qwen INT4/Q4KM bakes that predate
# the lm_head-INT4 work (no `lm_head.weight_int4_scale` tensor in the
# manifest) are treated as needing re-bake even without --force, so a plain
# `./bake_all.sh` brings every Qwen bake up to the current format. Override
# this auto-detection by passing --skip-lm-head-int4-refresh.
#
# The script is safely re-runnable — existing valid bakes are skipped unless
# --force is set. Both Python bakers checkpoint per-layer, so an interrupted
# INT4 run resumes without redoing prior work.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UPLOAD=0
INCLUDE_BF16=0
INCLUDE_FP8=0
INCLUDE_INT4=1
INCLUDE_Q4KM=0
INCLUDE_Q4KM_GPTQ=0
FORCE=0
STOP_ON_ERROR=0
SKIP_LM_HEAD_INT4_REFRESH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) UPLOAD=1 ;;
        --bf16) INCLUDE_BF16=1 ;;
        --fp8-native) INCLUDE_FP8=1 ;;
        --q4km) INCLUDE_Q4KM=1 ;;
        --q4km-gptq) INCLUDE_Q4KM_GPTQ=1 ;;
        --no-int4) INCLUDE_INT4=0 ;;
        --force) FORCE=1 ;;
        --skip-lm-head-int4-refresh) SKIP_LM_HEAD_INT4_REFRESH=1 ;;
        --stop-on-error) STOP_ON_ERROR=1 ;;
        -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0" ; exit 0 ;;
        *) echo "unknown flag: $1" >&2 ; exit 2 ;;
    esac
    shift
done

# (cli_name, env_var, family)
MODELS=(
    "qwen3.5-0.8b:QWEN_0_8B_DIR:qwen"
    "qwen3.5-2b:QWEN_2B_DIR:qwen"
    "qwen3.5-4b:QWEN_4B_DIR:qwen"
    "qwen3.5-9b:QWEN_9B_DIR:qwen"
    "qwen3.6-27b:QWEN_36_27B_DIR:qwen"
    "qwen3.6-35b-a3b:QWEN_36_35B_A3B_DIR:qwen"
    "gemma4-e2b:GEMMA_E2B_DIR:gemma"
    "gemma4-e4b:GEMMA_E4B_DIR:gemma"
    "phi4-mini:PHI4_MINI_DIR:phi4"
)

FORMAT_VERSION=2   # must match crates/model-store/src/manifest.rs

declare -a SUCCEEDED FAILED SKIPPED

bake_dir_for() {
    local model_dir="$1" variant="$2"
    case "$variant" in
        bf16) echo "$model_dir/.supersonic/v${FORMAT_VERSION}" ;;
        fp8-native) echo "$model_dir/.supersonic/v${FORMAT_VERSION}-fp8" ;;
        int4) echo "$model_dir/.supersonic/v${FORMAT_VERSION}-int4-gptq" ;;
        q4km) echo "$model_dir/.supersonic/v${FORMAT_VERSION}-q4km" ;;
        q4km-gptq) echo "$model_dir/.supersonic/v${FORMAT_VERSION}-q4km-gptq" ;;
        *) echo "bad variant: $variant" >&2 ; return 2 ;;
    esac
}

bake_exists_and_valid() {
    local dir="$1"
    [[ -f "$dir/manifest.json" && -f "$dir/weights.bin" ]] || return 1
    # cheap check — the full version check runs again in the Rust consumer
    grep -q "\"format_version\": *$FORMAT_VERSION" "$dir/manifest.json"
}

# True iff the bake at `$1` includes an INT4-quantized lm_head. Used to
# auto-detect bakes that predate the lm_head-INT4 work and force them to
# re-bake (without requiring callers to remember --force). Only meaningful
# for Qwen bakes; Gemma/Phi4 bakes intentionally don't quantize lm_head.
bake_has_lm_head_int4() {
    local dir="$1"
    [[ -f "$dir/manifest.json" ]] || return 1
    grep -q '"name": *"lm_head.weight_int4_scale"' "$dir/manifest.json"
}

gguf_file_for() {
    local env_var="$1"
    local gguf_env="${env_var%_DIR}_GGUF"
    local gguf_file="${!gguf_env:-}"
    if [[ -z "$gguf_file" ]]; then
        echo "missing $gguf_env; set it to the matching GGUF source" >&2
        return 1
    fi
    if [[ ! -f "$gguf_file" ]]; then
        echo "$gguf_env=$gguf_file is not a file" >&2
        return 1
    fi
    echo "$gguf_file"
}

activation_dir_for() {
    local env_var="$1"
    local activation_env="${env_var%_DIR}_ACTIVATION_DIR"
    local activation_dir="${!activation_env:-}"
    if [[ -z "$activation_dir" ]]; then
        return 1
    fi
    if [[ ! -d "$activation_dir" ]]; then
        echo "$activation_env=$activation_dir is not a directory" >&2
        return 1
    fi
    echo "$activation_dir"
}

run_or_record() {
    local label="$1" ; shift
    echo
    echo "==== $label ===="
    echo "    $*"
    if "$@"; then
        SUCCEEDED+=("$label")
        return 0
    else
        local code=$?
        echo "!! FAILED ($code): $label"
        FAILED+=("$label")
        if [[ $STOP_ON_ERROR -eq 1 ]]; then
            echo "stopping on first error as requested" >&2
            print_summary
            exit 1
        fi
        return $code
    fi
}

bake_int4() {
    local cli_name="$1" model_dir="$2" family="$3"
    local dir ; dir="$(bake_dir_for "$model_dir" int4)" || return 2
    local label="$cli_name int4"
    local should_skip=0
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
        should_skip=1
        # Auto-refresh Qwen INT4 bakes that predate the lm_head-INT4 work.
        # Gemma/Phi4 bakes intentionally don't quantize lm_head, so the
        # check is qwen-only.
        if [[ "$family" == "qwen" && $SKIP_LM_HEAD_INT4_REFRESH -eq 0 ]] \
            && ! bake_has_lm_head_int4 "$dir"; then
            echo "[refresh] $label — existing bake lacks lm_head_int4_scale; re-baking"
            should_skip=0
        fi
    fi
    if [[ $should_skip -eq 1 ]]; then
        echo "[skip] $label — valid bake at $dir"
        SKIPPED+=("$label")
        return 0
    fi
    case "$family" in
        qwen)
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_int4.py" \
                --model-dir "$model_dir" || return $?
            ;;
        gemma)
            # gemma4-e4b needs group_size=64 — at the default 128 the
            # per-tensor INT4 rounding error compounds across the deeper
            # 42-layer stack and produces post-calibration gibberish.
            # Verified: gs=128 → "The quick brown fox 0 � 0",
            #          gs=64  → "The quick brown fox is a bit of a joke".
            # E2B is fine at gs=128.
            local extra_args=()
            if [[ "$cli_name" == "gemma4-e4b" ]]; then
                extra_args+=("--group-size" "64")
            fi
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_int4_gemma4.py" \
                --model-dir "$model_dir" "${extra_args[@]}" || return $?
            ;;
        phi4)
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_int4_phi4.py" \
                --model-dir "$model_dir" || return $?
            ;;
        *)
            echo "[skip] $label — no INT4 baker for family '$family'"
            SKIPPED+=("$label")
            return 0
            ;;
    esac
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --int4 --model-dir "$model_dir"
    fi
}

bake_q4km() {
    local cli_name="$1" model_dir="$2" family="$3" env_var="$4"
    if [[ "$family" != "qwen" ]]; then
        echo "[skip] $cli_name q4km — Q4KM bake supports Qwen only"
        SKIPPED+=("$cli_name q4km")
        return 0
    fi
    local dir ; dir="$(bake_dir_for "$model_dir" q4km)" || return 2
    local label="$cli_name q4km"
    local should_skip=0
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
        should_skip=1
        if [[ $SKIP_LM_HEAD_INT4_REFRESH -eq 0 ]] && ! bake_has_lm_head_int4 "$dir"; then
            echo "[refresh] $label — existing bake lacks lm_head_int4_scale; re-baking"
            should_skip=0
        fi
    fi
    if [[ $should_skip -eq 1 ]]; then
        echo "[skip] $label — valid bake at $dir"
        SKIPPED+=("$label")
        return 0
    fi
    local gguf_file ; gguf_file="$(gguf_file_for "$env_var")" || {
        echo "[skip] $label — no GGUF source configured"
        SKIPPED+=("$label")
        return 0
    }
    run_or_record "$label" python3 "$SCRIPT_DIR/bake_q4km.py" \
        --model "$cli_name" \
        --model-dir "$model_dir" \
        --gguf-file "$gguf_file" || return $?
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --q4km --model-dir "$model_dir"
    fi
}

bake_q4km_gptq() {
    local cli_name="$1" model_dir="$2" family="$3" env_var="$4"
    if [[ "$family" != "qwen" ]]; then
        echo "[skip] $cli_name q4km-gptq — Q4KM-GPTQ bake supports Qwen only"
        SKIPPED+=("$cli_name q4km-gptq")
        return 0
    fi
    local dir ; dir="$(bake_dir_for "$model_dir" q4km-gptq)" || return 2
    local label="$cli_name q4km-gptq"
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
        echo "[skip] $label — valid bake at $dir"
        SKIPPED+=("$label")
        return 0
    fi
    local gguf_file ; gguf_file="$(gguf_file_for "$env_var")" || {
        echo "[skip] $label — no GGUF source configured"
        SKIPPED+=("$label")
        return 0
    }
    local -a cmd=(
        python3 "$SCRIPT_DIR/q4km_stream_gptq_bake.py"
        --model "$cli_name" \
        --model-dir "$model_dir" \
        --gguf-file "$gguf_file"
    )
    local activation_dir
    if activation_dir="$(activation_dir_for "$env_var")"; then
        cmd+=(--activation-model-dir "$activation_dir")
    else
        cmd+=(--use-gguf-loader)
    fi
    if [[ -n "${Q4KM_GPTQ_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        local -a extra_args=( $Q4KM_GPTQ_EXTRA_ARGS )
        cmd+=("${extra_args[@]}")
    fi
    run_or_record "$label" "${cmd[@]}" || return $?
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --q4km-gptq --model-dir "$model_dir"
    fi
}

bake_bf16() {
    local cli_name="$1" model_dir="$2" family="$3"
    if [[ "$family" == "gemma" ]]; then
        echo "[skip] $cli_name bf16 — Gemma 4 has no BF16 bake format"
        SKIPPED+=("$cli_name bf16")
        return 0
    fi
    local dir ; dir="$(bake_dir_for "$model_dir" bf16)" || return 2
    local label="$cli_name bf16"
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
        echo "[skip] $label — valid bake at $dir"
        SKIPPED+=("$label")
        return 0
    fi
    # Triggering the BF16 baker is a side-effect of a `supersonic` invocation.
    # We pass --no-download so that a flaky network on the producer doesn't
    # substitute an existing release bake for the fresh local one.
    run_or_record "$label" cargo run --release --manifest-path "$REPO_ROOT/Cargo.toml" \
        --bin supersonic -- \
        --model "$cli_name" \
        --model-dir "$model_dir" \
        --prompt "bake-trigger" \
        --max-new-tokens 1 \
        --no-download || return $?
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --bf16 --model-dir "$model_dir"
    fi
}

bake_fp8_native() {
    local cli_name="$1" model_dir="$2" family="$3"
    if [[ "$family" == "gemma" ]]; then
        echo "[skip] $cli_name fp8-native — Gemma 4 has no FP8-native bake format"
        SKIPPED+=("$cli_name fp8-native")
        return 0
    fi
    local dir ; dir="$(bake_dir_for "$model_dir" fp8-native)" || return 2
    local label="$cli_name fp8-native"
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
        echo "[skip] $label — valid bake at $dir"
        SKIPPED+=("$label")
        return 0
    fi
    # Per-block FP8-E4M3 calibration from the BF16 source. Produces FP8
    # weights + BF16 scale_inv tensors at v{FORMAT_VERSION}-fp8/, the same
    # layout the Rust runtime uses for Qwen 3.6 native FP8 checkpoints. The
    # bake scripts read BF16 source safetensors and write the bake directly
    # — no need to round-trip through the Rust auto-baker.
    case "$family" in
        qwen)
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_fp8.py" \
                --model-dir "$model_dir" || return $?
            ;;
        phi4)
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_fp8_phi4.py" \
                --model-dir "$model_dir" || return $?
            ;;
        *)
            echo "[skip] $label — no FP8 baker for family '$family'"
            SKIPPED+=("$label")
            return 0
            ;;
    esac
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --fp8-native --model-dir "$model_dir"
    fi
}

print_summary() {
    local succeeded_count=0
    local skipped_count=0
    local failed_count=0
    [[ ${SUCCEEDED+x} ]] && succeeded_count=${#SUCCEEDED[@]}
    [[ ${SKIPPED+x} ]] && skipped_count=${#SKIPPED[@]}
    [[ ${FAILED+x} ]] && failed_count=${#FAILED[@]}
    echo
    echo "==== bake_all summary ===="
    echo "succeeded: $succeeded_count"
    if [[ $succeeded_count -gt 0 ]]; then
        for s in "${SUCCEEDED[@]}"; do echo "  OK   $s"; done
    fi
    echo "skipped:   $skipped_count"
    if [[ $skipped_count -gt 0 ]]; then
        for s in "${SKIPPED[@]}"; do echo "  --   $s"; done
    fi
    echo "failed:    $failed_count"
    if [[ $failed_count -gt 0 ]]; then
        for s in "${FAILED[@]}"; do echo "  FAIL $s"; done
    fi
}

any_configured=0
for entry in "${MODELS[@]}"; do
    IFS=':' read -r cli_name env_var family <<< "$entry"
    model_dir="${!env_var:-}"
    if [[ -z "$model_dir" ]]; then
        continue
    fi
    if [[ ! -d "$model_dir" ]]; then
        echo "warning: $env_var=$model_dir is not a directory; skipping $cli_name" >&2
        continue
    fi
    any_configured=1
    echo
    echo "#### $cli_name @ $model_dir"
    if [[ $INCLUDE_BF16 -eq 1 ]]; then
        bake_bf16 "$cli_name" "$model_dir" "$family" || true
    fi
    if [[ $INCLUDE_FP8 -eq 1 ]]; then
        bake_fp8_native "$cli_name" "$model_dir" "$family" || true
    fi
    if [[ $INCLUDE_INT4 -eq 1 ]]; then
        bake_int4 "$cli_name" "$model_dir" "$family" || true
    fi
    if [[ $INCLUDE_Q4KM -eq 1 ]]; then
        bake_q4km "$cli_name" "$model_dir" "$family" "$env_var" || true
    fi
    if [[ $INCLUDE_Q4KM_GPTQ -eq 1 ]]; then
        bake_q4km_gptq "$cli_name" "$model_dir" "$family" "$env_var" || true
    fi
done

if [[ $any_configured -eq 0 ]]; then
    echo "error: no model directory env vars set. See $0 --help" >&2
    exit 2
fi

print_summary
# Exit non-zero iff anything failed, so CI can gate on the script's return.
failed_count=0
[[ ${FAILED+x} ]] && failed_count=${#FAILED[@]}
[[ $failed_count -eq 0 ]]
