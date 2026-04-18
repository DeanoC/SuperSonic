#!/usr/bin/env bash
# bake_all.sh — produce INT4 (and optionally BF16) bakes for every configured
# model on a big-box producer, and optionally publish each to the GitHub
# release as it finishes.
#
# Point env vars at the model directories you want to bake. Any unset var is
# silently skipped, so you can bake a subset without editing the script.
#
#   QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
#   QWEN_2B_DIR=/models/Qwen3.5-2B \
#   QWEN_4B_DIR=/models/Qwen3.5-4B \
#   QWEN_9B_DIR=/models/Qwen3.5-9B \
#   GEMMA_E2B_DIR=/models/gemma-4-E2B \
#   GEMMA_E4B_DIR=/models/gemma-4-E4B \
#   ./oracle/bake_all.sh --upload
#
# Flags:
#   --upload        Publish each bake via oracle/upload_bake.py once it succeeds.
#   --bf16          Also produce Qwen BF16 bakes (triggered via a throwaway
#                   `supersonic` run). Gemma 4 has no BF16 bake format.
#   --no-int4       Skip INT4 bakes (useful with --bf16 for a BF16-only run).
#   --force         Re-bake even if a valid bake already exists.
#   --stop-on-error Abort the whole batch if any single bake fails.
#                   (Default: keep going and report failures at the end.)
#
# The script is safely re-runnable — existing valid bakes are skipped unless
# --force is set. Both Python bakers checkpoint per-layer, so an interrupted
# INT4 run resumes without redoing prior work.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UPLOAD=0
INCLUDE_BF16=0
INCLUDE_INT4=1
FORCE=0
STOP_ON_ERROR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) UPLOAD=1 ;;
        --bf16) INCLUDE_BF16=1 ;;
        --no-int4) INCLUDE_INT4=0 ;;
        --force) FORCE=1 ;;
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
    "gemma4-e2b:GEMMA_E2B_DIR:gemma"
    "gemma4-e4b:GEMMA_E4B_DIR:gemma"
)

FORMAT_VERSION=1   # must match crates/model-store/src/manifest.rs

declare -a SUCCEEDED FAILED SKIPPED

bake_dir_for() {
    local model_dir="$1" variant="$2"
    case "$variant" in
        bf16) echo "$model_dir/.supersonic/v${FORMAT_VERSION}" ;;
        int4) echo "$model_dir/.supersonic/v${FORMAT_VERSION}-int4-gptq" ;;
        *) echo "bad variant: $variant" >&2 ; return 2 ;;
    esac
}

bake_exists_and_valid() {
    local dir="$1"
    [[ -f "$dir/manifest.json" && -f "$dir/weights.bin" ]] || return 1
    # cheap check — the full version check runs again in the Rust consumer
    grep -q "\"format_version\": *$FORMAT_VERSION" "$dir/manifest.json"
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
    if [[ $FORCE -eq 0 ]] && bake_exists_and_valid "$dir"; then
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
            run_or_record "$label" python3 "$SCRIPT_DIR/bake_int4_gemma4.py" \
                --model-dir "$model_dir" || return $?
            ;;
    esac
    if [[ $UPLOAD -eq 1 ]]; then
        run_or_record "$label upload" python3 "$SCRIPT_DIR/upload_bake.py" \
            --model "$cli_name" --int4 --model-dir "$model_dir"
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

print_summary() {
    echo
    echo "==== bake_all summary ===="
    echo "succeeded: ${#SUCCEEDED[@]}"
    for s in "${SUCCEEDED[@]:-}"; do [[ -n "$s" ]] && echo "  OK   $s"; done
    echo "skipped:   ${#SKIPPED[@]}"
    for s in "${SKIPPED[@]:-}"; do [[ -n "$s" ]] && echo "  --   $s"; done
    echo "failed:    ${#FAILED[@]}"
    for s in "${FAILED[@]:-}"; do [[ -n "$s" ]] && echo "  FAIL $s"; done
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
    if [[ $INCLUDE_INT4 -eq 1 ]]; then
        bake_int4 "$cli_name" "$model_dir" "$family" || true
    fi
done

if [[ $any_configured -eq 0 ]]; then
    echo "error: no model directory env vars set. See $0 --help" >&2
    exit 2
fi

print_summary
# Exit non-zero iff anything failed, so CI can gate on the script's return.
[[ ${#FAILED[@]} -eq 0 ]]
