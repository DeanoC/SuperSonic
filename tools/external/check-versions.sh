#!/usr/bin/env bash
# Verify installed external benchmark engines match pinned versions.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

check_one() {
    local engine="$1"
    local binary="$2"
    local version_cmd="$3"
    local pin_file="$SCRIPT_DIR/${engine}-version.txt"
    if [ ! -f "$pin_file" ]; then
        echo "[check-versions] no pin file for $engine ($pin_file); skipping" >&2
        return 0
    fi
    if ! command -v "$binary" >/dev/null 2>&1; then
        echo "[check-versions] $engine binary '$binary' is not installed; skipping" >&2
        return 0
    fi
    local pinned actual
    pinned="$(grep -v '^#' "$pin_file" | grep -v '^$' | head -n1)"
    set +e
    actual="$(eval "$version_cmd" 2>&1 | head -n1)"
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "[check-versions] $engine version command failed; skipping ($actual)" >&2
        return 0
    fi
    if [ "$pinned" != "$actual" ]; then
        echo "[check-versions] MISMATCH: $engine pinned=$pinned actual=$actual" >&2
        return 1
    fi
    echo "[check-versions] OK: $engine = $actual"
}

failed=0
check_one hipfire hipfire "hipfire --version" || failed=1
check_one llama-cpp llama-cli "llama-cli --version" || failed=1
check_one mlx-lm python3 "python3 -m mlx_lm --version" || failed=1
exit $failed
