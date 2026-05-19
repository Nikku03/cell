#!/usr/bin/env bash
# ASI-Evolve eval wrapper for snn_scaling reversal.
#
# ASI-Evolve interface:
#   - Invoked as `bash <this_script>` with NO args
#   - cwd is the per-step directory (.../steps/step_N/)
#   - Candidate code is at $cwd/code (no extension)
#   - We MUST write $cwd/results.json with 'eval_score' field
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
    }, f, indent=2)
print('wrote error result.json: $reason', file=__import__('sys').stderr)
"
    exit 0
}

if [ ! -f "$CANDIDATE" ]; then
    write_error "candidate file not found at $CANDIDATE"
fi

export PYTHONPATH="/content/cell:${PYTHONPATH}"

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

python "$EVAL_DIR/evaluator.py" "$LOG" "$RESULTS"
if [ $? -ne 0 ] || [ ! -f "$RESULTS" ]; then
    write_error "evaluator failed to write results.json"
fi

SCORE=$(python -c "import json; print(json.load(open('$RESULTS'))['eval_score'])")
BEST=$(python -c "import json; print(json.load(open('$RESULTS')).get('best_integrated_cum', '?'))")
GAP=$(python -c "import json; print(json.load(open('$RESULTS')).get('gap_memrec_naive', '?'))")
echo "[eval] done; eval_score=$SCORE  best_integrated_cum=$BEST  gap_memrec_naive=$GAP" >&2
