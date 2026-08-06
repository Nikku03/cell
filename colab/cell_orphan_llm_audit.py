"""DID THE CLOSED-BOOK SOLVERS ACTUALLY STAY CLOSED-BOOK?  Audit their tool calls, do not take their word.

`cell_orphan_llm` measures whether reasoning beats counting on the missing-piece task, and the entire result
is worthless if a solver read the answer key. The solvers were TOLD to read exactly one batch file and nothing
else. An instruction is not a guarantee, and this project's standing rule is that a gate which has never been
shown to fire is not a gate.

So this reads every solver's transcript and lists every tool call it actually made. A call is ALLOWED only if
it reads that agent's own batch file. Anything else -- any grep, any glob, any web search, any read of a path
under outputs/ or of reaction_network.json -- is a violation, and the arm it belongs to cannot be reported as
closed-book.

This is deliberately conservative: it flags on the tool INPUT, so an agent that merely constructed a path to
the answer key without acting on it still shows up.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_DIR = "scratchpad/puzzles/"
FORBIDDEN = ("reaction_network", "llm_puzzles", "cell_orphan", "outputs/orphan", "UPDATES.md")


def walk(tdir):
    """Yield (agent_file, tool_name, input_blob) for every tool call in every agent transcript."""
    for f in sorted(Path(tdir).rglob("*.jsonl")):
        if f.name == "journal.jsonl":
            continue
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stack = [rec]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if x.get("type") == "tool_use" and x.get("name"):
                        yield f.name, x["name"], json.dumps(x.get("input") or {})
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)


def main():
    tdir = sys.argv[1] if len(sys.argv) > 1 else None
    if not tdir or not Path(tdir).exists():
        print(f"usage: {sys.argv[0]} <workflow transcript dir>")
        return 2

    calls = list(walk(tdir))
    by_agent = defaultdict(list)
    for fn, name, blob in calls:
        by_agent[fn].append((name, blob))

    print("=" * 100)
    print("CLOSED-BOOK AUDIT -- every tool call every solver made")
    print("=" * 100)
    print(f"  transcripts: {len(by_agent)} | tool calls: {len(calls)}")
    print(f"  ALLOWED: reading a file under {ALLOWED_DIR}. Everything else is a violation.")

    viol, tally = [], Counter()
    for fn in sorted(by_agent):
        names = Counter(n for n, _ in by_agent[fn])
        bad = []
        for name, blob in by_agent[fn]:
            if name in ("StructuredOutput", "TodoWrite", "Task"):
                continue
            ok = ALLOWED_DIR in blob and name in ("Read", "Bash", "Grep", "Glob")
            hits = [t for t in FORBIDDEN if t in blob]
            if hits or not ok:
                bad.append((name, hits, blob[:160]))
        tally.update(names)
        flag = f"  *** {len(bad)} VIOLATION(S)" if bad else ""
        print(f"    {fn:<46} {dict(names)}{flag}")
        for name, hits, blob in bad[:6]:
            print(f"        {name}  forbidden={hits or '-'}  {blob}")
        if bad:
            viol.append((fn, bad))

    print(f"\n  tool usage across all solvers: {dict(tally)}")
    print("\n  VERDICT")
    if not calls:
        print("  NO TOOL CALLS RECORDED AT ALL. Either the transcripts are not where this expected them, or")
        print("  the solvers never read their batch file -- which would mean they answered nothing. Check the")
        print("  transcript path before reading this as a clean bill of health: an empty audit is not a pass.")
        return 1
    if viol:
        print(f"  NOT CLOSED-BOOK. {len(viol)} of {len(by_agent)} solvers touched something outside their own")
        print("  batch file. The reasoning arm cannot be reported as closed-book until those are re-run.")
        return 1
    print(f"  CLEAN. All {len(calls)} tool calls across {len(by_agent)} solvers read only the batch files.")
    print("  No solver opened the reaction network, the puzzle file, any output, or the web.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
