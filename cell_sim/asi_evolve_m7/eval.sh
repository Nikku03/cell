#!/usr/bin/env bash
# ASI-Evolve eval wrapper for M7 experiment.
#
# Invoked by ASI-Evolve's Engineer with one argument: the path to the
# candidate program to evaluate. The candidate program is a Python
# file with the same structure as initial_program.py (the Researcher
# generates these via diff-based mutations).
#
# Flow:
#   1. Run the candidate, capture stdout+stderr to log
#   2. Run evaluator.py on the log, which parses the metrics block
#      and prints the score dict
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

# Run the candidate (with a 15-minute wall timeout)
timeout 900 python "$CANDIDATE" > "$LOG" 2>&1 || {
    EXIT=$?
    cat "$LOG" >&2
    if [ $EXIT -eq 124 ]; then
        echo '{"score": -1e6, "error": "timeout 900s exceeded"}'
    else
        echo "{\"score\": -1e6, \"error\": \"candidate exited with code $EXIT\"}"
    fi
    exit 0
}

# Parse metrics
python "$EVAL_DIR/evaluator.py" "$LOG"
