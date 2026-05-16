#!/usr/bin/env bash
# ASI-Evolve M7-Evolve v2 eval wrapper.
#
# Three-phase evaluation:
#   1. Run candidate (typically a fine-tune) -> produces /tmp/m7_candidate.pt
#   2. Snapshot tests (~30 sec) reject obviously broken mutations early
#   3. Full multi-objective benchmark (accuracy + speed + memory)
#
# If candidate fails any phase, returns a sentinel score so ASI-Evolve
# correctly logs the failure without crashing.
#
set -e -o pipefail

if [ -z "$1" ]; then
    echo '{"score": -1e6, "error": "no candidate program provided"}'
    exit 1
fi

CANDIDATE="$1"
WORK_DIR="$(dirname "$CANDIDATE")"
LOG="$WORK_DIR/candidate_stdout.log"
EVAL_DIR="$(dirname "$0")"
CKPT="${M7_OUTPUT_CKPT:-/tmp/m7_candidate.pt}"

# Phase 1 — run the candidate (~5-12 min)
echo "[eval] Phase 1: run candidate" >&2
timeout 900 python "$CANDIDATE" > "$LOG" 2>&1 || {
    EXIT=$?
    tail -50 "$LOG" >&2
    if [ $EXIT -eq 124 ]; then
        echo '{"score": -1e6, "error": "candidate timeout 900s"}'
    else
        echo "{\"score\": -1e6, \"error\": \"candidate exit code $EXIT\"}"
    fi
    exit 0
}

if [ ! -f "$CKPT" ]; then
    echo "{\"score\": -1e6, \"error\": \"candidate did not produce $CKPT\"}"
    exit 0
fi

# Phase 2 — snapshot safety tests (~30 sec)
echo "[eval] Phase 2: snapshot safety tests" >&2
timeout 120 python "$EVAL_DIR/snapshot_tests.py" "$CKPT" > "$WORK_DIR/snapshot.log" 2>&1 || {
    REASON=$(grep -m1 'SNAPSHOT_FAIL:' "$WORK_DIR/snapshot.log" | head -1 || echo 'unknown')
    echo "{\"score\": -1e6, \"error\": \"snapshot test failed: $REASON\"}"
    exit 0
}

# Phase 3 — full multi-objective benchmark (~3-5 min)
echo "[eval] Phase 3: full benchmark" >&2
timeout 600 python "$EVAL_DIR/full_benchmark.py" "$CKPT" > "$WORK_DIR/bench.log" 2>&1 || {
    EXIT=$?
    tail -20 "$WORK_DIR/bench.log" >&2
    if [ $EXIT -eq 124 ]; then
        echo '{"score": -1e6, "error": "benchmark timeout 600s"}'
    else
        echo "{\"score\": -1e6, \"error\": \"benchmark exit code $EXIT\"}"
    fi
    exit 0
}

# Phase 4 — parse metrics, compute final score
python "$EVAL_DIR/evaluator.py" "$WORK_DIR/bench.log"
