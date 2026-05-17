#!/usr/bin/env bash
# ASI-Evolve eval wrapper for artificial_atom v3.
#
# ASI-Evolve interface:
#   - Invoked as `bash <this_script>` with NO args
#   - cwd is the per-step directory (.../steps/step_N/)
#   - Candidate code is at $cwd/code (no extension)
#   - We MUST write $cwd/results.json with 'eval_score' field
#
# Pipeline:
#   1. Run candidate (~7-10 min on CPU; ~2-4 min on GPU)
#   2. Parse metrics, write results.json
#
set -o pipefail

WORK_DIR="$(pwd)"
CANDIDATE="$WORK_DIR/code"
LOG="$WORK_DIR/candidate_stdout.log"
RESULTS="$WORK_DIR/results.json"
EVAL_DIR="$(dirname "$(realpath "$0")")"

write_error() {
    local reason="$1"
    python -c "
import json
with open('$RESULTS', 'w') as f:
    json.dump({
        'eval_score': -1.0e6,
        'success': False,
        'error': '''$reason'''[:500],
        'composite': float('inf'),
        'checks_passed': 0,
    }, f, indent=2)
print('wrote error result.json: $reason', file=__import__('sys').stderr)
"
    exit 0
}

if [ ! -f "$CANDIDATE" ]; then
    write_error "candidate file not found at $CANDIDATE"
fi

# Set PYTHONPATH so artificial_atom imports work
export PYTHONPATH="/content/cell:${PYTHONPATH}"

# Run candidate (15-min timeout — generous for slow CPU)
echo "[eval] running candidate ($CANDIDATE)" >&2
timeout 900 python "$CANDIDATE" > "$LOG" 2>&1
EXIT=$?
if [ $EXIT -eq 124 ]; then
    write_error "candidate timeout 900s"
fi
if [ $EXIT -ne 0 ]; then
    TAIL=$(tail -20 "$LOG" 2>/dev/null | tr -d "'\"" | tr '\n' ' ')
    write_error "candidate exited with code $EXIT: $TAIL"
fi

# Parse metrics
python "$EVAL_DIR/evaluator.py" "$LOG" "$RESULTS"
if [ $? -ne 0 ] || [ ! -f "$RESULTS" ]; then
    write_error "evaluator failed to write results.json"
fi

SCORE=$(python -c "import json; print(json.load(open('$RESULTS'))['eval_score'])")
PASSED=$(python -c "import json; print(json.load(open('$RESULTS')).get('checks_passed', '?'))")
echo "[eval] done; eval_score=$SCORE  checks=$PASSED/8" >&2
