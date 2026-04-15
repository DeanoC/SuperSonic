#!/usr/bin/env bash
# Run golden corpus tests for a specific model.
# Called from machine-specific test scripts (e.g. tests/gfx1150/run.sh).
#
# Usage:
#   tests/corpus/run_golden.sh <model-variant> <model-dir> <golden-json> <supersonic-binary>
#
# Returns 0 if all tests pass, 1 if any fail.

set -euo pipefail

MODEL="$1"
MODEL_DIR="$2"
GOLDEN="$3"
SUPERSONIC="$4"
TIMEOUT="${CORPUS_TIMEOUT:-300}"

if [ ! -f "$GOLDEN" ]; then
    echo "Golden file not found: $GOLDEN"
    exit 1
fi

NUM_TESTS=$(python3 -c "import json; print(len(json.load(open('$GOLDEN'))['test_cases']))")
echo ""
echo "--- Golden Corpus Tests ($NUM_TESTS cases) ---"

PASSED=0
FAILED=0

for i in $(seq 0 $((NUM_TESTS - 1))); do
    NAME=$(python3 -c "import json; print(json.load(open('$GOLDEN'))['test_cases'][$i]['name'])")
    MAX_NEW=$(python3 -c "import json; print(json.load(open('$GOLDEN'))['test_cases'][$i]['max_new_tokens'])")

    printf "  %-20s " "$NAME"

    # Use python to invoke the binary — avoids shell escaping issues with prompts
    RESULT=$(python3 -c "
import json, subprocess, sys

golden = json.load(open('$GOLDEN'))
tc = golden['test_cases'][$i]
prompt = tc['prompt']
expected = tc['expected_text']

try:
    proc = subprocess.run(
        ['$SUPERSONIC', '--model', '$MODEL', '--model-dir', '$MODEL_DIR',
         '--prompt', prompt, '--max-new-tokens', str(tc['max_new_tokens'])],
        capture_output=True, text=True, timeout=$TIMEOUT
    )
    if proc.returncode != 0:
        print('CRASH')
        print(proc.stderr[-200:] if proc.stderr else 'no stderr', file=sys.stderr)
        sys.exit(0)

    full = proc.stdout.strip()

    # Extract generated suffix
    if full.startswith(prompt):
        generated = full[len(prompt):]
    else:
        generated = full

    # Compare
    if generated == expected:
        print('EXACT')
    elif len(expected) >= 3 and generated.startswith(expected[:min(len(expected), 20)]):
        print('PREFIX')
    elif len(generated) >= 3 and expected.startswith(generated[:min(len(generated), 20)]):
        print('PREFIX')
    else:
        print('MISMATCH')
        print(f'expected: {repr(expected[:80])}', file=sys.stderr)
        print(f'got:      {repr(generated[:80])}', file=sys.stderr)
except subprocess.TimeoutExpired:
    print('TIMEOUT')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)

    STATUS=$(echo "$RESULT" | head -1)

    case "$STATUS" in
        EXACT)
            echo "PASS"
            PASSED=$((PASSED + 1))
            ;;
        PREFIX)
            echo "PASS (prefix)"
            PASSED=$((PASSED + 1))
            ;;
        CRASH)
            echo "FAIL (crash)"
            echo "$RESULT" | tail -n +2 | head -2 | sed 's/^/    /'
            FAILED=$((FAILED + 1))
            ;;
        TIMEOUT)
            echo "FAIL (timeout)"
            FAILED=$((FAILED + 1))
            ;;
        MISMATCH)
            echo "FAIL"
            echo "$RESULT" | tail -n +2 | head -2 | sed 's/^/    /'
            FAILED=$((FAILED + 1))
            ;;
        *)
            echo "FAIL ($STATUS)"
            FAILED=$((FAILED + 1))
            ;;
    esac
done

echo ""
echo "Golden corpus: $PASSED passed, $FAILED failed / $NUM_TESTS total"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
