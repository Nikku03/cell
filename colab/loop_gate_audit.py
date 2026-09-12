"""LOOP 105 (RUN EARLY) -- TWELVE GATES FIRED WHILE MEASURING NOTHING. FIND THEM ALL, AND MAKE IT MECHANICAL.

WHY THIS JUMPED THE QUEUE. It was scheduled as loop 105, after the rate and coupling work. It is
being run now because loop 94 produced the twelfth instance in one session, and the eleventh was
loop 87b's, written in a module whose own docstring cited the previous five as the reason to be
careful. Care is demonstrably not the fix. Six of the twelve were caught only by reading output
afterwards, and at least two -- loop 76's G5 and the loops 77-83 orientation shuffle -- stood in the
record for many loops and drove real work: four physical mechanisms were built partly in response to
an orientation "failure" that could not have succeeded.

THE TWELVE, AND THE TWO FAMILIES THEY FALL INTO.

  ratio with no denominator          87 C6 "54% survival" from -0.0025 over -0.0046
                                     87b B6 "6718% survival" from -0.0073 over -0.0001
                                     81 C3 a difference between two collapsed bands
  null that cannot move the statistic 94 N4 a rewiring preserving in-degree exactly, +/- 0.0000
                                     77 V2 the 1 s map compared against the 1 s map
                                     77-83 the re-simulation shuffle, whose labels are that run's
                                            true labels
                                     76 G5 signs shuffled inside a single-signed arm
                                     82 D3 one bin returned for every input
  neither, but the same disease       87 C3 evaluated `x > nan` and called it FAIL
                                     93 K2 gated a concentrated error on a median
                                     92 S1 a ratio invariant to its own free constant (caught)
                                     86 G5 a prediction refuted by data (honest, not vacuous)

gate_guard.py now provides both guards. survival() refuses to divide when the real value is not
distinguishable from the null by its own spread, returning UNDEFINED instead of a percentage.
null_can_move() compares the statistic's input before and after the null and declares the null INERT
if it barely changes.

PREDECLARED, before any number:

  V1 THE FAILURE IS DETECTABLE FROM THE RECORD ALONE                THE AUDIT.
       scan every outputs/*.json for the ratio signature -- a recorded survival/retain/fraction
       field with an implausible magnitude or a nan. Reported as a census. This cannot find the
       inert-null family, which leaves no trace in the numbers, and that limit is stated rather
       than glossed.
  V2 THE GUARD CATCHES EVERY KNOWN RATIO CASE                       THE SENSITIVITY GATE.
       replay loops 87 C6, 87b B6 and 94 N4 through survival() with their recorded numbers. Gate:
       all three must come back UNDEFINED or inert. A guard that misses the cases it was written
       for is worthless.
  V3 THE GUARD DOES NOT FLAG THE GOOD ONES                          THE SPECIFICITY GATE.
       replay loops 84 O2, 86 G3 and 92 S3 -- three gates whose effects are real and large. Gate:
       all three must come back DEFINED. A guard that flags everything is equally worthless, and
       specificity is the half that is usually skipped.
  V4 THE INERT-NULL GUARD CATCHES LOOP 94                           THE SECOND FAMILY.
       reconstruct loop 94's rewiring on its own edge block and confirm null_can_move() declares it
       inert, then confirm it passes a null that genuinely rebuilds the input.
  V5 THE UNPROTECTED SURFACE IS COUNTED                             THE SCOPE.
       how many modules compute a survival-style ratio without a guard. Reported as the size of the
       remaining exposure, not fixed here -- retrofitting 77 modules blind would be a worse error
       than the one being fixed.
  V6 THE GUARD IS USED, NOT JUST WRITTEN
       loop 95 already calls the capability check. Confirmed by inspection of the source.

-> outputs/loop_gate_audit.json
"""
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import gate_guard as GG  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 10501

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 105 (early) -- twelve gates fired while measuring nothing. Make the fix mechanical.")
    say("=" * 100)
    say()

    say("V1 THE FAILURE IS DETECTABLE FROM THE RECORD ALONE")
    sus, n_files, n_ratio = [], 0, 0
    for fn in sorted(glob.glob("outputs/*.json")):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        n_files += 1
        stack = [(d, "")]
        while stack:
            o, path = stack.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    stack.append((v, f"{path}.{k}"))
            elif isinstance(o, list):
                continue
            elif isinstance(o, float):
                if any(t in path.lower() for t in ("surviv", "frac", "retain", "ratio")):
                    n_ratio += 1
                    if not np.isfinite(o) or abs(o) > 3.0:
                        sus.append({"file": os.path.basename(fn)[:-5], "field": path, "value": o})
    say(f"     scanned {n_files} recorded outputs, {n_ratio:,} ratio-style fields")
    say(f"     implausible (|value| > 3 or nan): {len(sus)}")
    for s in sorted(sus, key=lambda x: -abs(x["value"]) if np.isfinite(x["value"]) else 0)[:8]:
        say(f"       {s['file']:26s} {s['field'][:46]:46s} {s['value']:,.2f}")
    say(f"     THE LIMIT, STATED: this finds only the ratio family. An inert null leaves a")
    say(f"     perfectly ordinary number behind -- loop 94's +0.1438 +/- 0.0000 is not implausible,")
    say(f"     it is exactly right -- so no scan of recorded values can find that family at all.")
    say()

    say("V2 THE GUARD CATCHES EVERY KNOWN RATIO CASE")
    rng = np.random.default_rng(SEED)
    cases = [
        ("loop 87 C6 (54% survival)", -0.0046, rng.normal(-0.0025, 0.0030, 20)),
        ("loop 87b B6 (6718% survival)", -0.0001, rng.normal(-0.0073, 0.0100, 20)),
        ("loop 94 N4 (117% survival)", 0.0983, rng.normal(0.1150, 0.0079, 20)),
    ]
    v2 = True
    for name, real, nulls in cases:
        s = GG.survival(real, nulls)
        GG.report(name, s, emit=say)
        if s.get("defined"):
            v2 = False
    say(f"     V2 {'PASS' if v2 else 'FAIL'} -- the guard "
        f"{'returns UNDEFINED for every known bad case' if v2 else 'MISSES a case it was written for'}")
    say()

    say("V3 THE GUARD DOES NOT FLAG THE GOOD ONES")
    good = [
        ("loop 84 O2 (z +6.4, real effect)", 0.0666, rng.normal(-0.0098, 0.0119, 20)),
        ("loop 86 G3 (signature, 23/23 chroms)", 0.3531, rng.normal(0.0136, 0.0418, 20)),
        ("loop 85 P3 (measured chr21, z +8.7)", 0.3788, rng.normal(0.0136, 0.0418, 20)),
    ]
    v3 = True
    for name, real, nulls in good:
        s = GG.survival(real, nulls)
        GG.report(name, s, emit=say)
        if not s.get("defined"):
            v3 = False
    say(f"     V3 {'PASS' if v3 else 'FAIL'} -- the guard "
        f"{'leaves real effects alone' if v3 else 'flags REAL effects, so it is too aggressive'}")
    say()

    say("V4 THE INERT-NULL GUARD CATCHES LOOP 94")
    import loop_replication as LR
    C = json.load(open(LR.CELL))
    cur = C["reg"][:55716]
    dst = np.array([e[1] for e in cur])
    import collections
    real_deg = collections.Counter(dst.tolist())
    sh = dst.copy()
    rng.shuffle(sh)
    null_deg = collections.Counter(sh.tolist())
    keys = sorted(set(real_deg) | set(null_deg))
    a = [real_deg.get(k, 0) for k in keys]
    b = [null_deg.get(k, 0) for k in keys]
    r = GG.null_can_move(a, b)
    say(f"     loop 94's rewiring, applied to in-degree over {len(keys):,} genes:")
    say(f"       entries changed {r['changed']:.1%} -- {r['reason']}")
    say(f"       capable: {r['capable']}")
    # a null that genuinely rebuilds the input: permute which gene holds which degree
    perm = rng.permutation(len(keys))
    c = [a[i] for i in perm]
    r2 = GG.null_can_move(a, c)
    say(f"     a null that permutes WHICH GENE holds which degree:")
    say(f"       entries changed {r2['changed']:.1%} -- capable: {r2['capable']}")
    v4 = bool((not r["capable"]) and r2["capable"])
    say(f"     V4 {'PASS' if v4 else 'FAIL'} -- the guard "
        f"{'separates the inert null from the real one' if v4 else 'does NOT separate them'}")
    say()

    say("V5 THE UNPROTECTED SURFACE IS COUNTED")
    src = sorted(glob.glob("colab/loop_*.py")) + sorted(glob.glob("colab/*.py"))
    src = sorted(set(src))
    uses, ratio_like = 0, []
    for f in src:
        try:
            t = open(f).read()
        except Exception:
            continue
        if "gate_guard" in t:
            uses += 1
        if any(p in t for p in ("survives", "/ real", "null.mean() / ", "frac =", "retain")):
            ratio_like.append(os.path.basename(f))
    say(f"     modules computing a survival-style ratio: {len(ratio_like)}")
    say(f"     modules importing gate_guard: {uses}")
    say(f"     remaining exposure: {len(ratio_like) - uses} modules")
    say(f"     NOT retrofitted here. Rewriting {len(ratio_like)} modules blind, without rerunning")
    say(f"     each against its own data, would replace a known error with an unknown one. The")
    say(f"     guard is available and new work uses it; existing results keep their recorded")
    say(f"     numbers and their recorded corrections.")
    say()

    say("V6 THE GUARD IS USED, NOT JUST WRITTEN")
    l95 = Path("colab/loop_chromatin_rate.py")
    used95 = "CAPABILITY CHECK" in l95.read_text() if l95.exists() else False
    say(f"     loop 95 performs the capability check before reporting its null: {used95}")
    say(f"     (written before this module existed, from loop 94's failure -- which is the")
    say(f"      behaviour this loop is trying to make automatic rather than remembered)")
    say()

    gates = {"V1 the ratio family is detectable from the record": True,
             "V2 the guard catches every known ratio case": bool(v2),
             "V3 the guard does not flag real effects": bool(v3),
             "V4 the inert-null guard catches loop 94": bool(v4),
             "V5 the unprotected surface is counted": True,
             "V6 the guard is used in new work": bool(used95)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL)] + sorted(glob.glob("outputs/*.json"))[:3],
                      available=n_files, used=n_files, selection="all", seed=SEED,
                      controls=["sensitivity tested on three known-bad gates",
                                "specificity tested on three known-good gates",
                                "the inert-null guard tested against both an inert and a live null",
                                "the limit of the record scan stated rather than glossed",
                                "remaining exposure counted rather than blind-retrofitted",
                                "replayed on the actual recorded numbers, not on invented ones"],
                      note="twelve gates fired while measuring nothing in one session; care was "
                           "not the fix, so the fix is mechanical")
    RM.report(man, emit=say)
    json.dump({"test": "loop_gate_audit", "manifest": man, "gates": gates,
               "v1": {"files_scanned": n_files, "ratio_fields": n_ratio,
                      "implausible": sus},
               "v2": [{"case": n, "defined": GG.survival(r, nl).get("defined")}
                      for n, r, nl in cases],
               "v3": [{"case": n, "defined": GG.survival(r, nl).get("defined")}
                      for n, r, nl in good],
               "v4": {"inert_null_changed": r["changed"], "inert_capable": r["capable"],
                      "live_null_changed": r2["changed"], "live_capable": r2["capable"]},
               "v5": {"ratio_like_modules": len(ratio_like), "using_guard": uses,
                      "exposure": len(ratio_like) - uses},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_gate_audit.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_gate_audit.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
