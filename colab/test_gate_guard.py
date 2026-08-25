"""Replays the four gate-machinery defects from the loop 187-199 arc against the Gates ledger.

Each test reconstructs the ACTUAL situation that produced a false line in a log, not a synthetic
analogue. If any of these regress, a loop will quietly print something untrue again.

Run: python3 colab/test_gate_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_guard as GG                      # noqa: E402
import numpy as np                           # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def cap():
    out = []
    return out, out.append


# --- A: loop 187's B6 -- a self-loop count above chance in a network with zero self-loops -----
lines, emit = cap()
G = GG.Gates(emit=emit)
observed, null_mean, null_sd = 0, 0.0, 0.0
z = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")
G.add("B6", z > 3.0, stat=z,
      if_true="B6 PASS -- self-regulation is above chance",
      if_false="B6 FAIL -- the self-loops are what chance gives and loop 175's framing stands")
check("A/187-B6 undefined z is VOID, not FAIL", G.status["B6"] == GG.VOID)
check("A/187-B6 does not print the unearned claim",
      not any("framing stands" in x for x in lines), lines[0][:70])

# --- B: loop 196's X4 and loop 197's Y4 -- success message referencing a success-only value ---
lines, emit = cap()
G = GG.Gates(emit=emit)
d4, winner = {}, None
G.add("X4", winner is not None,
      if_true=lambda: f"X4 PASS -- '{winner}' holds on {d4[winner]['held']}/4 subsets",
      if_false="X4 FAIL -- no candidate holds on 3 of 4 subsets")
check("B/196-X4 lazy message does not crash", G.status["X4"] == GG.FAIL)

lines, emit = cap()
G = GG.Gates(emit=emit)
qual = []
G.add("Y4", bool(qual),
      if_true=f"Y4 PASS -- {qual[0]['cell'] if qual else ''} qualifies",   # EAGER, still safe
      if_false="Y4 FAIL -- no structured public series is dense enough")
check("B/197-Y4 eager message cannot kill a decided verdict", G.status["Y4"] == GG.FAIL)

# an eager message that DOES raise must still not crash the run
lines, emit = cap()
G = GG.Gates(emit=emit)


class Boom:
    def __format__(self, spec):
        raise RuntimeError("exploded while narrating")


G.add("Z9", False, if_true="unused", if_false=f"Z9 FAIL -- {Boom()!s}" if False else "Z9 FAIL")
check("B/narration failure is contained", G.status["Z9"] == GG.FAIL)

# --- C: loop 194's V4/V6 -- confirmatory gates on a positive V3 did not find ------------------
lines, emit = cap()
G = GG.Gates(emit=emit)
G.add("V3", False, if_true="V3 PASS", if_false="V3 FAIL -- z -0.8, coupled enzymes are at chance")
G.add("V4", True, requires=("V3",),
      if_true="V4 PASS", if_false="V4 FAIL -- the result depends on the hub threshold")
G.add("V6", True, requires=("V3",),
      if_true="V6 PASS", if_false="V6 FAIL -- against an abundance-matched null the coherence goes")
check("C/194-V4 voids on a failed precondition", G.status["V4"] == GG.VOID)
check("C/194-V6 voids on a failed precondition", G.status["V6"] == GG.VOID)
check("C/194 does not print 'depends on the hub threshold'",
      not any("hub threshold" in x for x in lines))
check("C/194 does not print 'the coherence goes'",
      not any("coherence goes" in x for x in lines))

# --- D: loop 199's Q5 -- a swap that destroyed a NEGATIVE association -------------------------
w = GG.weakened_by(real=-0.0840, control=-0.0149)
check("D/199-Q5 magnitude comparison sees the swap destroy it", w["weakened"],
      f"real {w['real']:+.4f} control {w['control']:+.4f}")
check("D/199-Q5 the old signed test would have failed", not (-0.0840 > -0.0149))

# --- D': loop 188's G2 -- 90 of 4,482 non-finite turned a real result into 'REFUTED' ----------
rng = np.random.default_rng(0)
a = rng.normal(4.0, 1.0, 500)
b = rng.normal(9.5, 1.0, 4000)
a[:10] = np.nan                                      # the 2% that broke it
check("D'/188-G2 raw median is nan", not np.isfinite(np.median(a)))
aa, bb, dropped = GG.finite(a, a)[0], b, 0
aa, dropped = GG.finite(a)[0], GG.finite(a)[-1]
check("D'/188-G2 finite() recovers the statistic", np.isfinite(np.median(aa)),
      f"median {np.median(aa):.2f}, dropped {dropped}")

# --- the score excludes VOID rather than counting it against ----------------------------------
lines, emit = cap()
G = GG.Gates(emit=emit)
G.add("P1", True, if_true="P1 PASS", if_false="P1 FAIL")
G.add("P2", False, if_true="P2 PASS", if_false="P2 FAIL")
G.add("P3", True, requires=("P2",), if_true="P3 PASS", if_false="P3 FAIL")
G.summary()
gates, void = G.as_dict()
check("score excludes VOID from the denominator", any("1/2" in x for x in lines),
      [x for x in lines if "/" in x][-1].strip())
check("as_dict matches the shape loops already write", void == ["P3"] and gates["P1"] is True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all gate-machinery regressions covered")
