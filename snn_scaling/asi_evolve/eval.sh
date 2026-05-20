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

# --- Defensive diff-applier (comment-tolerant) ---------------------
# If asi_evolve's Engineer step wrote the raw diff to $CANDIDATE
# instead of applying it, do the application ourselves. Tolerates
# inline-comment drift in the SEARCH text (a common GPT-5 failure mode
# where the LLM hallucinates slightly different trailing comments).
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

def strip_inline_comment(line: str) -> str:
    """Strip trailing inline Python comment (preserves indentation +
    string-literal #'s by only honouring a # that isn't inside quotes)."""
    in_s = in_d = False
    out = []
    for ch in line:
        if ch == "'" and not in_d: in_s = not in_s
        elif ch == '"' and not in_s: in_d = not in_d
        elif ch == '#' and not in_s and not in_d:
            break
        out.append(ch)
    return ''.join(out).rstrip()

def apply_one(base, search, replace):
    """Try exact match first; if it fails, retry with inline-comment
    tolerance (allow base or search to have a different trailing comment
    on each line). Returns (new_base, mode) or (None, reason)."""
    if search in base:
        return base.replace(search, replace, 1), 'exact'
    # Build a comment-tolerant regex
    s_lines = search.split('\n')
    pat_lines = []
    for ln in s_lines:
        code = strip_inline_comment(ln)
        if code == ln:
            # No comment in SEARCH -> still allow base to have one
            pat_lines.append(re.escape(code) + r'[ \t]*(?:#[^\n]*)?')
        else:
            # SEARCH had a comment -> still escape full line but allow drift
            pat_lines.append(re.escape(code) + r'[ \t]*(?:#[^\n]*)?')
    pattern = '\n'.join(pat_lines)
    m = re.search(pattern, base)
    if m is None:
        return None, 'no match (exact or comment-tolerant)'
    return base[:m.start()] + replace + base[m.end():], 'comment-tolerant'

for i, (search, replace) in enumerate(matches):
    new_base, mode = apply_one(base, search, replace)
    if new_base is None:
        print(f'SEARCH block {i} not applicable to base:', file=sys.stderr)
        print(f'  reason: {mode}', file=sys.stderr)
        print(f'  SEARCH was:\n{search!r}', file=sys.stderr)
        sys.exit(3)
    base = new_base
    print(f'[diff-apply] block {i}: matched via {mode}', file=sys.stderr)
open(diff_path, 'w').write(base)
print(f'[diff-apply] wrote {len(base)} chars to {diff_path}', file=sys.stderr)
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
WHO=$(python -c "import json; print(json.load(open('$RESULTS')).get('best_integrated_agent', '?'))")
EVOL=$(python -c "import json; print(json.load(open('$RESULTS')).get('evolved_cum_mean', '?'))")
echo "[eval] done; eval_score=$SCORE  best=$WHO ($BEST)  evolved=$EVOL" >&2
