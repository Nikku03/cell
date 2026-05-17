#!/usr/bin/env bash
# ASI-Evolve M7-Evolve v5 eval wrapper.
#
# ASI-Evolve INTERFACE (verified from pipeline/engineer/engineer.py):
#   - Invoked as `bash <this_script>` with NO command-line arguments.
#   - cwd is the per-step experiment directory (e.g. .../steps/step_5/).
#   - Candidate code is at $cwd/code (no .py extension).
#   - We MUST write $cwd/results.json with at least {"eval_score": float}.
#   - Non-zero exit code is interpreted as success=False.
#
# Three-phase pipeline:
#   1. Run candidate (writes /tmp/m7_candidate.pt with state_dict + metrics)
#   2. Snapshot safety tests (~30 sec, rejects broken mutations early)
#   3. Full multi-objective benchmark (accuracy + speed + memory)
#   4. Parse the bench log -> results.json with eval_score

set -o pipefail

WORK_DIR="$(pwd)"
CANDIDATE="$WORK_DIR/code"
LOG="$WORK_DIR/candidate_stdout.log"
RESULTS="$WORK_DIR/results.json"
EVAL_DIR="$(dirname "$(realpath "$0")")"
CKPT="${M7_OUTPUT_CKPT:-/tmp/m7_candidate.pt}"

# Helper: write an error result.json and exit 0 so ASI-Evolve picks up the
# error message instead of treating it as an opaque non-zero exit.
write_error() {
    local reason="$1"
    python -c "
import json
with open('$RESULTS', 'w') as f:
    json.dump({
        'eval_score': -1.0e6,
        'success': False,
        'error': '''$reason'''[:500],
        'composite_relative': float('inf'),
    }, f, indent=2)
print('wrote error result.json: $reason', file=__import__('sys').stderr)
"
    exit 0
}

if [ ! -f "$CANDIDATE" ]; then
    write_error "candidate file not found at $CANDIDATE"
fi

# ----------------------------------------------------------------
# Phase 1 - run candidate (~5-12 min)
# ----------------------------------------------------------------
echo "[eval] Phase 1: run candidate ($CANDIDATE)" >&2
rm -f "$CKPT"
# Set PYTHONPATH so `import cell_sim.*` works no matter where the LLM
# puts (or removes) sys.path.insert calls in the candidate.
export PYTHONPATH="/content/cell:${PYTHONPATH}"
timeout 900 python "$CANDIDATE" > "$LOG" 2>&1
EXIT=$?
if [ $EXIT -eq 124 ]; then
    write_error "candidate timeout 900s"
fi
if [ $EXIT -ne 0 ]; then
    TAIL=$(tail -20 "$LOG" 2>/dev/null | tr -d "'\"" | tr '\n' ' ')
    write_error "candidate exited with code $EXIT: $TAIL"
fi

if [ ! -f "$CKPT" ]; then
    TAIL=$(tail -10 "$LOG" 2>/dev/null | tr -d "'\"" | tr '\n' ' ')
    write_error "candidate did not produce $CKPT: $TAIL"
fi

# ----------------------------------------------------------------
# Phase 2 - snapshot safety tests (~30 sec)
# ----------------------------------------------------------------
echo "[eval] Phase 2: snapshot tests" >&2
timeout 120 python "$EVAL_DIR/snapshot_tests.py" "$CKPT" \
    > "$WORK_DIR/snapshot.log" 2>&1
if [ $? -ne 0 ]; then
    REASON=$(grep -m1 'SNAPSHOT_FAIL:' "$WORK_DIR/snapshot.log" 2>/dev/null | head -1 | tr -d "'\"")
    [ -z "$REASON" ] && REASON="unknown snapshot failure"
    write_error "$REASON"
fi

# ----------------------------------------------------------------
# Phase 3 - full multi-objective benchmark (~3-5 min)
# ----------------------------------------------------------------
echo "[eval] Phase 3: full benchmark" >&2
timeout 600 python "$EVAL_DIR/full_benchmark.py" "$CKPT" \
    > "$WORK_DIR/bench.log" 2>&1
EXIT=$?
if [ $EXIT -eq 124 ]; then
    write_error "benchmark timeout 600s"
fi
if [ $EXIT -ne 0 ]; then
    TAIL=$(tail -20 "$WORK_DIR/bench.log" 2>/dev/null | tr -d "'\"" | tr '\n' ' ')
    write_error "benchmark exited with code $EXIT: $TAIL"
fi

# ----------------------------------------------------------------
# Phase 4 - parse benchmark metrics, write results.json
# ----------------------------------------------------------------
echo "[eval] Phase 4: write results.json" >&2
python "$EVAL_DIR/evaluator.py" "$WORK_DIR/bench.log" "$RESULTS"
if [ $? -ne 0 ] || [ ! -f "$RESULTS" ]; then
    write_error "evaluator.py failed to write results.json"
fi

echo "[eval] done; eval_score=$(python -c "import json; print(json.load(open('$RESULTS'))['eval_score'])")" >&2
