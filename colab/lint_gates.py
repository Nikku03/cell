"""Scan loop files for the four gate-machinery defects gate_guard.Gates now prevents.

Written instead of blanket-retrofitting every loop. Most loops that already ran are correct as
they stand -- each defect was patched where it appeared -- and rewriting a file that produced a
committed result risks changing it silently, which is worse than leaving it alone. This finds the
files where the defect is still LATENT, so the retrofit is aimed rather than swept.

  B  a verdict message that indexes something which may not exist on the branch not taken.
     GG.verdict builds BOTH f-strings before the call, so `if_true=f"...{best[0]}..."` crashes
     when best is empty and the gate had already decided FAIL. Loop 196's X4 and loop 197's Y4.
  C  a hand-rolled void set, which is where loop 196's X4 printed VOID and FAIL into one summary.
  A  a gate whose verdict is a threshold comparison on a value that could be nan -- the boolean
     swallows the undefinedness, which is loop 187's B6.
  D  a control gate comparing signed values, which assumes the sign of its own answer -- loop
     199's Q5.

Run: python3 colab/lint_gates.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RISKY_INDEX = re.compile(r"if_true\s*=\s*f?\"[^\"]*\{[^}]*\[[^\]]+\][^}]*\}")
HAND_VOID = re.compile(r"^\s*void\s*(=|\|=)\s*", re.M)
NAN_CMP = re.compile(r"np\.isfinite\(\s*(\w+)\s*\)\s*and\s*\1\s*[<>]")
SIGNED_CTRL = re.compile(r"\b(\w*real\w*|d\d)\[?[\"']?\w*[\"']?\]?\s*>\s*(\w*swap\w*|\w*perm\w*|d\d)\b",
                         re.I)


def scan(p):
    s = p.read_text()
    if "gate_guard" not in s or "def main" not in s:
        return None
    hits = {}
    # ONLY flag an eager message whose indexed NAME is assigned None somewhere in the file.
    # The first version flagged every f-string containing a subscript, which is most of them and
    # almost all benign -- a dict that always exists indexes fine. The dangerous shape is
    # specifically `d = None` followed by `if_true=f"...{d['k']}..."`, because Python builds that
    # string at the CALL SITE before verdict() can intercept it. That is exactly loop 196's X4
    # (winner = None) and loop 197's Y4 (qual = []). An over-flagging linter gets ignored, which
    # is worse than no linter.
    nullable = set(re.findall(r"^\s*(\w+)\s*=\s*(?:None|False,\s*None|\[\])\s*$", s, re.M))
    nullable |= set(re.findall(r"^\s*\w+,\s*(\w+)\s*=\s*\w+,\s*None\s*$", s, re.M))
    b = []
    for m in RISKY_INDEX.finditer(s):
        txt = m.group(0)
        names = set(re.findall(r"\{(\w+)\[", txt))
        if names & nullable:
            b.append(f"{sorted(names & nullable)} may be None: {txt[:70]}")
    if b:
        hits["B eager message indexes a name assigned None"] = b
    if HAND_VOID.search(s) and "Gates(" not in s:
        hits["C hand-rolled void set"] = [f"{len(HAND_VOID.findall(s))} sites"]
    # A: a gate guarded by isfinite is SAFE; flag comparisons with no such guard on a z/rho name
    unguarded = [m.group(0)[:70] for m in
                 re.finditer(r"^\s*\w+ = bool\((?!.*isfinite)[^)]*\b(z\w*|rho|r_)\b[^)]*[<>][^)]*\)",
                             s, re.M)]
    if unguarded:
        hits["A threshold on a possibly-undefined statistic, unguarded"] = unguarded
    c = [m.group(0)[:70] for m in SIGNED_CTRL.finditer(s)]
    if c:
        hits["D control compared by sign, not magnitude"] = c
    return hits


def main():
    files = sorted(HERE.glob("loop_*.py"))
    flagged = 0
    for p in files:
        h = scan(p)
        if not h:
            continue
        if any(h.values()):
            flagged += 1
            print(f"\n{p.name}")
            for k, v in h.items():
                print(f"   {k}")
                for x in v[:3]:
                    print(f"      {x}")
    print(f"\n{flagged} of {len(files)} loop files carry a latent gate defect")
    return flagged


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
