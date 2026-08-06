"""Merge the closed-book solvers' answers out of their own transcripts into the scorer's answer file.

Reading the workflow's return value would be easier and is the wrong choice: that value is a summary produced
downstream of the agents, and if it were ever truncated or reshaped the loss would be silent. The transcripts
are the primary record. Each agent's own transcript names the batch file it read -- `tie_b3.txt` -- so the arm
and batch are recovered from the agent's behaviour rather than from bookkeeping that could drift.

Refuses to write a partial answer file. A missing batch would silently score as 200-minus-40 wrong answers and
depress an arm by up to 0.2, which is larger than every effect being measured.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT = A.OUT
ARMS = ("tie", "open", "scram")
N_EXPECT = 200


def collect(tdir):
    """arm -> {pid: gene}, taken from each agent's StructuredOutput tool call."""
    got = {a: {} for a in ARMS}
    seen = defaultdict(set)
    for f in sorted(Path(tdir).glob("agent-*.jsonl")):
        txt = f.read_text(errors="replace")
        tags = set(re.findall(r"(tie|open|scram)_b(\d+)\.txt", txt))
        if len(tags) != 1:
            print(f"  SKIP {f.name}: batch file ambiguous ({sorted(tags)})")
            continue
        arm, b = tags.pop()
        seen[arm].add(b)
        n0 = len(got[arm])
        for line in txt.splitlines():
            if '"answers"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stack = [rec]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("answers"), list):
                        for a_ in x["answers"]:
                            if isinstance(a_, dict) and "pid" in a_ and "gene" in a_:
                                got[arm][int(a_["pid"])] = str(a_["gene"]).strip().upper()
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
        print(f"  {f.name[:22]}  {arm}_b{b}  +{len(got[arm])-n0} answers")
    return got, seen


def main():
    tdir = sys.argv[1]
    print("=" * 100)
    print("MERGE closed-book answers from solver transcripts")
    print("=" * 100)
    got, seen = collect(tdir)

    ok = True
    for a in ARMS:
        pids = set(got[a])
        missing = sorted(set(range(N_EXPECT)) - pids)
        print(f"\n  {a:<7} batches {sorted(seen[a])} | {len(pids)}/{N_EXPECT} answered"
              + (f" | MISSING {len(missing)}: {missing[:12]}{'...' if len(missing) > 12 else ''}"
                 if missing else ""))
        if missing:
            ok = False
    if not ok:
        print("\n  REFUSING TO WRITE a partial answer file. A missing batch scores as wrong answers and would")
        print("  depress that arm by more than any effect being measured. Re-run the missing solvers first.")
        return 1

    merged = {str(p): {a: got[a][p] for a in ARMS} for p in range(N_EXPECT)}
    json.dump(merged, open(OUT / "llm_answers.json", "w"), indent=1)
    for a in ARMS:
        v = list(got[a].values())
        print(f"  {a:<7} distinct genes {len(set(v))}/{len(v)}")
    print(f"\n  wrote {OUT/'llm_answers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
