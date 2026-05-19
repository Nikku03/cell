#!/usr/bin/env bash
# ASI-Evolve eval wrapper for snn_scaling reversal.
#
# Defensive workaround for asi_evolve's diff-application step: when the
# Engineer fails to apply the Researcher's diff (and instead writes the
# raw "<<<<<<< SEARCH ... >>>>>>> REPLACE" text into $cwd/code), this
# script detects that, applies the diff itself against initial_program.py,
# and overwrites $cwd/code with the patched program before running it.
set -o pipefail

WORK_DIR="$(pwd)"
CANDIDATE="$WORK_DIR/code"
LOG="$WORK_DIR/candidate_stdout.log"
RESULTS="$WORK_DIR/results.json"
EVAL_DIR="$(dirname "$(realpath "$0")")"
BASE_CODE="$EVAL_DIR/initial_program.py"

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

# --- Defensive diff-applier ----------------------------------------
# If asi_evolve's Engineer step wrote the raw diff to $CANDIDATE
# instead of applying it, do the application ourselves.
if head -1 "$CANDIDATE" | grep -Eq '^<{4,10}[[:space:]]*SEARCH'; then
    if [ ! -f "$BASE_CODE" ]; then
        write_error "base code missing: $BASE_CODE"
    fi
    echo "[eval] candidate is raw diff; applying to base $BASE_CODE" >&2
    python3 - "$CANDIDATE" "$BASE_CODE" <<'PYEOF'
import re, sys
diff_path, base_path = sys.argv[1], sys.argv[2]
diff = open(diff_path).read()
base = open(base_path).read()
pat = re.compile(
    r'<{4,10}\s*SEARCH\s*\n(.*?)\n={4,10}\s*\n(.*?)\n>{4,10}\s*REPLACE',
    re.DOTALL,
)
matches = pat.findall(diff)
if not matches:
    print(f'no diff blocks found in {diff_path}', file=sys.stderr)
    sys.exit(2)
for i, (search, replace) in enumerate(matches):
    if search not in base:
        print(f'SEARCH block {i} not found in base code', file=sys.stderr)
        print(f'  SEARCH text was:\n{search!r}', file=sys.stderr)
        sys.exit(3)
    base = base.replace(search, replace, 1)
open(diff_path, 'w').write(base)
print(f'[diff-apply] applied {len(matches)} block(s); '
      f'output {len(base)} chars to {diff_path}', file=sys.stderr)
PYEOF
    APPLY_EXIT=$?
    if [ $APPLY_EXIT -ne 0 ]; then
        TAIL=$(head -20 "$CANDIDATE" 2>/dev/null | tr -d "'\"" | tr '\n' ' ')
        write_error "diff-apply failed (exit $APPLY_EXIT); head of diff was: $TAIL"
    fi
fi
# -------------------------------------------------------------------

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
