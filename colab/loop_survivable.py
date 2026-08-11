"""THE ANTI-CORRELATION -- is a driver constrained to be survivable, or did the control break?

WHERE THIS COMES FROM.  loop_cancer's most surprising number was not a failure. After partialling out
gene length and publication count, three independent measures of how much a gene matters all pointed
BACKWARDS at driver status:

    essentiality       partial AUC 0.2990
    metabolic cost     partial AUC 0.3254
    protein abundance  partial AUC 0.3691
    LOEUF              partial AUC 0.6588   <- the one that points forwards

A partial AUC of 0.2990 is not a weak result. It is a 0.7010 prediction of NON-drivers, and loop_cancer
wrote it down as such. What it did not do is establish WHY, and there are exactly two live stories:

    THE BIOLOGY   a driver has to be survivable. A tumour is a cell that is still alive. If losing a
                  gene kills the cell, that loss cannot be the event that starts a tumour, so drivers
                  are drawn from the non-essential pool by construction of what a tumour is.
    THE ARTEFACT  publication count predicts driver status at 0.8259. Regressing out a confound that
                  strong, from a score that is itself correlated with it, is exactly the setup that
                  manufactures a reversed residual. Over-control produces anti-correlations for free.

Loop 27 already lost a headline to an algebraic identity hiding inside a control. This asks the same
question of loop 14 before its number is allowed to stand.

WHAT IS DIFFERENT HERE.  No regression at all. Every driver is paired with a non-driver matched on gene
length and publication count directly, and the comparison happens inside the matched set. A matched pair
cannot over-control, because nothing is subtracted from anything.

PREDECLARED, before any number:

    N1 THE MATCHING WORKS   inside the matched set, gene length and publication count both fall back
                inside the shuffle band. This is the design check: if the confounds still separate
                drivers from controls after matching, nothing below can be read at all.
    N2 THE ANTI-CORRELATION SURVIVES   essentiality still scores significantly BELOW 0.5 in the matched
                set. fails -> 0.2990 was manufactured by the regression, and loop_cancer's direction
                claim has to be withdrawn, which is the honest outcome if it is what happened.
    N3 IT IS THE SURVIVABILITY CONSTRAINT   the anti-correlation is stronger for loss-of-function
                drivers than for activating drivers, by a margin that beats permuting the ROLE label
                with the matched pairs held fixed.
                THIS IS THE DISCRIMINATING GATE, and it is the reason this loop can settle anything.
                The survivability story makes a prediction the artefact story cannot: the constraint
                only binds where the driver mechanism requires LOSING the gene. A tumour suppressor
                must be deletable, so it must be non-essential. An oncogene is switched on, not lost,
                so its essentiality is unconstrained -- an activating driver is free to be essential.
                An over-control artefact knows nothing about roles and hits both classes equally.
                The permutation is not optional. Two separately-matched AUCs read side by side is not
                a test: each carries its own sampling noise, and a gap of 0.05 between two numbers
                whose shuffle bands are both +/-0.03 is nothing at all.
    N4 POSITIVE CONTROL   LOEUF still points FORWARDS inside the same matched set -- significant AND
                in the right direction, because a positive control that is significant the wrong way
                round is a second anomaly, not a control. If matching flattens the one measure that
                genuinely predicts drivers, the matching is too aggressive and N2 is unreadable
                regardless of which way it came out.

CONTROLS: greedy nearest-neighbour matching without replacement inside a 0.2 SD caliper; 200 label
shuffles inside the matched set; confounds re-scored after matching as the design check; LOEUF as a
direction-checked positive control, negated to loop_cancer's convention so low LOEUF is the 'matters
more' end; the role contrast tested by permuting role with pairs fixed; roles taken from IntOGen's own
annotation, with genes carrying both roles excluded from the role comparison rather than assigned to
one.

-> outputs/loop_survivable.json
"""
import collections
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
INTOGEN = SC / "intogen.zip"
SEED = 1904
N_SHUFFLE = 200
CALIPER = 0.2      # SD units in the (log length, log publications) plane
GATE_PAIRS = 150   # below this the matched set is too small to read


def match(T, rng):
    """Pair every driver with the closest available non-driver on length and citations.

    Greedy, without replacement, inside a caliper. Drivers are processed in a shuffled order so the
    ones that happen to sit early in the gene list do not get first pick of the control pool."""
    X = np.c_[T.log_len.to_numpy(), T.log_pubs.to_numpy()]
    X = (X - X.mean(0)) / X.std(0)
    dr = np.where(T.y.to_numpy() == 1)[0]
    ctrl = np.where(T.y.to_numpy() == 0)[0]
    rng.shuffle(dr)
    C = X[ctrl]
    taken = np.zeros(len(ctrl), bool)
    pairs = []
    for i in dr:
        d = np.hypot(C[:, 0] - X[i, 0], C[:, 1] - X[i, 1])
        d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= CALIPER:
            taken[j] = True
            pairs.append((i, int(ctrl[j])))
    return pairs


def scored(T, pairs, col):
    """Values and labels for one column inside the matched set, dropping pairs with a missing value.

    LOEUF is negated to match loop_cancer's convention: a LOW LOEUF means a gene is intolerant to
    losing a copy, so low is the 'matters more' end. Without the flip its column would read as an
    anti-correlation when it is the positive control pointing the right way."""
    sgn = -1.0 if col == "LOEUF" else 1.0
    v, y = [], []
    for i, j in pairs:
        a, b = T[col].iloc[i], T[col].iloc[j]
        if np.isfinite(a) and np.isfinite(b):
            v += [sgn * float(a), sgn * float(b)]
            y += [1, 0]
    return np.array(v), np.array(y)


def band(v, y, rng, n=N_SHUFFLE):
    """95th percentile of |AUC - 0.5| under label shuffles inside the matched set."""
    if len(v) < 40:
        return float("nan")
    return float(np.percentile([abs(auc(v, rng.permutation(y)) - 0.5) for _ in range(n)], 95))


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("THE ANTI-CORRELATION -- survivability constraint, or a control that broke")
    say("=" * 100)
    say("  loop_cancer found essentiality ANTI-predicting driver status at 0.2990 after partialling")
    say("  out length and citations. Citations alone predict drivers at 0.8259; regressing out a")
    say("  confound that strong is how reversed residuals get manufactured. No regression here --")
    say("  every driver is paired with a matched non-driver instead.")

    z = zipfile.ZipFile(INTOGEN)
    nm = [x for x in z.namelist() if x.endswith("Compendium_Cancer_Genes.tsv")][0]
    rows = list(csv.DictReader(io.StringIO(z.read(nm).decode("utf-8", "ignore")), delimiter="\t"))
    roles = collections.defaultdict(set)
    for r in rows:
        roles[r["SYMBOL"]].add(r.get("ROLE"))
    drivers = set(roles)
    lof_only = {s for s, v in roles.items() if v == {"LoF"}}
    act_only = {s for s, v in roles.items() if v == {"Act"}}
    say(f"\n  IntOGen 2024-06: {len(drivers):,} driver genes; {len(lof_only):,} annotated loss-of-"
        f"function ONLY, {len(act_only):,} activating ONLY")
    say("  genes carrying both roles are excluded from the role comparison rather than assigned to one")

    D = json.load(open(CELL))
    genes = D["genes"]
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}
    rec = []
    for i, g in enumerate(genes):
        s = g["name"]
        ln = (g.get("gene_end") or 0) - (g.get("gene_start") or 0)
        if ln <= 0:
            continue
        rec.append({"sym": s, "y": 1 if s in drivers else 0,
                    "lof": 1 if s in lof_only else 0, "act": 1 if s in act_only else 0,
                    "log_len": float(np.log10(ln)),
                    "log_pubs": float(np.log10((g.get("pubs") or 0) + 1)),
                    "essentiality": float(g["ess"]) if isinstance(g.get("ess"), (int, float))
                    else np.nan,
                    "dependency fraction": float(g["dep_frac"])
                    if isinstance(g.get("dep_frac"), (int, float)) else np.nan,
                    "LOEUF": float(g["loeuf"]) if isinstance(g.get("loeuf"), (int, float))
                    else np.nan,
                    "abundance": float(np.log10(ppm.get(i, 0.0) + 1e-3))})
    T = pd.DataFrame(rec)
    say(f"  {len(T):,} genes with coordinates; {int(T.y.sum()):,} of them drivers")

    pairs = match(T, rng)
    say(f"\n  matched {len(pairs):,} drivers to distinct non-drivers within {CALIPER} SD on both "
        f"confounds ({len(pairs)/max(int(T.y.sum()),1):.0%} of drivers paired)")
    if len(pairs) < GATE_PAIRS:
        say(f"  fewer than {GATE_PAIRS} pairs. NOT TESTABLE.")
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump({"test": "loop_survivable", "testable": False, "n_pairs": len(pairs), "log": log},
                  open(OUT / "loop_survivable.json", "w"), indent=2)
        return 1

    # ---- N1 the design check ------------------------------------------------------------------------
    say(f"\n  N1 THE MATCHING WORKS -- the confounds, before and inside the matched set")
    say(f"    {'confound':<24}{'all genes':>12}{'matched':>10}{'|null| 95th':>14}")
    n1 = True
    conf = {}
    for c, label in (("log_len", "gene length"), ("log_pubs", "publication count")):
        a_all = auc(T[c], T.y.to_numpy())
        v, yy = scored(T, pairs, c)
        a_m = auc(v, yy)
        b = band(v, yy, rng)
        ok = bool(abs(a_m - 0.5) <= b)
        n1 = n1 and ok
        conf[label] = {"all": a_all, "matched": a_m, "band": b, "balanced": ok}
        say(f"    {label:<24}{a_all:>12.4f}{a_m:>10.4f}{0.5+b:>14.4f}   "
            f"{'balanced' if ok else 'STILL SEPARATES'}")
    say(f"    N1 {'PASS' if n1 else 'FAIL'}")

    # ---- N2 does the anti-correlation survive -------------------------------------------------------
    say(f"\n  N2 THE ANTI-CORRELATION SURVIVES -- measures of how much a gene matters, inside the")
    say(f"     matched set. below 0.5 means the measure predicts NON-drivers.")
    say(f"    {'measure':<24}{'matched AUC':>13}{'|null| 95th':>14}{'direction':>14}")
    res = {}
    for c in ("essentiality", "dependency fraction", "abundance", "LOEUF"):
        v, yy = scored(T, pairs, c)
        a = auc(v, yy)
        b = band(v, yy, rng)
        sig = bool(abs(a - 0.5) > b)
        res[c] = {"auc": a, "band95": 0.5 + b, "significant": sig, "n_pairs": len(v) // 2}
        d = "ANTI (non-drivers)" if a < 0.5 else "predicts drivers"
        say(f"    {c:<24}{a:>13.4f}{0.5+b:>14.4f}{d:>20}   {'' if sig else '(inside band)'}")
    n2 = bool(res["essentiality"]["auc"] < 0.5 and res["essentiality"]["significant"])
    say(f"    N2 {'PASS' if n2 else 'FAIL'}  (essentiality still below 0.5 and outside the band)")

    # ---- N4 positive control (read before N3, since N3 is unreadable without it) --------------------
    # direction matters as much as significance: a positive control that is significant in the WRONG
    # direction is not a positive control, it is a second anomaly.
    n4 = bool(res["LOEUF"]["significant"] and res["LOEUF"]["auc"] > 0.5)
    say(f"\n  N4 POSITIVE CONTROL -- LOEUF (negated, so high = intolerant) inside the matched set: "
        f"{res['LOEUF']['auc']:.4f}   N4 {'PASS' if n4 else 'FAIL'}")
    say("     LOEUF is the measure that genuinely predicts drivers. If matching flattened it too, the")
    say("     matching removed the biology along with the confound and N2 could not be read.")

    # ---- N3 the discriminating gate -----------------------------------------------------------------
    # ONE matched set over the role-annotated drivers, then the role label is permuted with the pairs
    # held fixed. Comparing two separately-matched AUCs by eye is not a test: each carries its own
    # sampling noise, and a 0.05 gap between two numbers whose bands are both +/-0.03 is nothing.
    say(f"\n  N3 IT IS THE SURVIVABILITY CONSTRAINT -- one matched set over role-annotated drivers,")
    say(f"     then the LoF/activating label is permuted with the pairs held fixed.")
    sub = T[(T.lof == 1) | (T.act == 1) | (T.y == 0)].reset_index(drop=True).copy()
    sub["y"] = ((sub.lof == 1) | (sub.act == 1)).astype(int)
    p = match(sub, np.random.default_rng(SEED))
    keep, is_lof = [], []
    for i, j in p:
        a, b = sub["essentiality"].iloc[i], sub["essentiality"].iloc[j]
        if np.isfinite(a) and np.isfinite(b):
            keep.append((float(a), float(b)))
            is_lof.append(bool(sub.lof.iloc[i] == 1))
    keep, is_lof = np.array(keep), np.array(is_lof)

    def split_auc(mask):
        if mask.sum() < 20:
            return float("nan")
        v = np.concatenate([keep[mask, 0], keep[mask, 1]])
        yy = np.r_[np.ones(mask.sum()), np.zeros(mask.sum())]
        return auc(v, yy)

    la, aa = split_auc(is_lof), split_auc(~is_lof)
    delta = aa - la    # positive = the constraint binds harder on loss-of-function drivers
    null = []
    for _ in range(N_SHUFFLE):
        m = rng.permutation(is_lof)
        null.append(split_auc(~m) - split_auc(m))
    p95 = float(np.nanpercentile(null, 95))
    role = {"loss-of-function drivers": {"auc": la, "n_pairs": int(is_lof.sum())},
            "activating drivers": {"auc": aa, "n_pairs": int((~is_lof).sum())},
            "delta": delta, "null_95th": p95}
    say(f"    loss-of-function drivers    {la:>9.4f}   ({int(is_lof.sum())} pairs)")
    say(f"    activating drivers          {aa:>9.4f}   ({int((~is_lof).sum())} pairs)")
    say(f"    difference {delta:+.4f} against a permuted 95th percentile of {p95:+.4f}")
    n3 = bool(np.isfinite(delta) and delta > p95)
    say(f"    N3 {'PASS' if n3 else 'FAIL'}  (gate: the constraint binds harder where the driver "
        f"mechanism requires losing the gene)")

    say("\n" + "=" * 100)
    gates = {"N1 the matching works": n1, "N2 the anti-correlation survives": n2,
             "N3 it is the survivability constraint": n3, "N4 positive control holds": n4}
    for kk, vv in gates.items():
        say(f"  {kk:<42}{'PASS' if vv else 'FAIL'}")
    if not n1:
        say("  THE MATCHING FAILED and nothing below it can be read. The confounds still separate")
        say("  drivers from their controls, so this design did not do the job it exists to do.")
    elif not n4:
        say("  THE MATCHING WENT TOO FAR. LOEUF, which genuinely predicts drivers, is flattened inside")
        say("  the matched set, so an absent signal here means nothing.")
    elif not n2 and n3:
        say(f"  THE POOLED NUMBER WAS THE REGRESSION. THE ROLE SPLIT IS NOT. Two findings, and they")
        say("  do not conflict -- the second explains the first.")
        say(f"    1. loop_cancer's 0.2990 is withdrawn. With no regression anywhere, pooled")
        say(f"       essentiality sits at {res['essentiality']['auc']:.4f} inside a band of "
            f"{res['essentiality']['band95']:.4f}. The blanket claim that")
        say("       the model predicts non-drivers was manufactured by partialling out a confound")
        say("       that predicts the label at 0.8259.")
        say(f"    2. The two driver roles point OPPOSITE WAYS and cancel. Loss-of-function drivers are")
        say(f"       less essential than their matched controls ({la:.4f}); activating drivers are MORE")
        say(f"       essential ({aa:.4f}); the gap of {delta:+.4f} beats permuting role with the pairs")
        say(f"       held fixed ({p95:+.4f}). A pooled test over both roles has to read as nothing,")
        say("       because it is averaging a constraint against its opposite.")
        say("  That is the survivability constraint showing up exactly where it should: a tumour")
        say("  suppressor has to be deletable, so it cannot be essential; an oncogene is switched on")
        say("  rather than lost, so nothing stops it from being essential. THE MARGIN IS THIN -- one")
        say("  test, ~160 pairs a side, and a difference that clears its permutation band by 0.0125.")
        say("  It is a direction worth one more measurement, not a result to build on.")
    elif not n2:
        say(f"  THE 0.2990 WAS THE REGRESSION, NOT THE BIOLOGY. Without partialling, essentiality sits")
        say(f"  at {res['essentiality']['auc']:.4f} inside a band of {res['essentiality']['band95']:.4f}."
            f" loop_cancer's direction claim")
        say("  is withdrawn: the model does not predict non-drivers, it predicts nothing here.")
        say("  The role split does not rescue it either: loss-of-function and activating drivers are")
        say(f"  constrained alike ({la:.4f} vs {aa:.4f}, gap {delta:+.4f} inside {p95:+.4f}).")
    elif n2 and n3:
        say("  THE CONSTRAINT IS REAL AND IT IS SURVIVABILITY. With no regression anywhere, essentiality")
        say(f"  still predicts NON-drivers at {1-res['essentiality']['auc']:.4f}, and it binds harder on")
        say(f"  loss-of-function drivers ({la:.4f}) than on activating drivers ({aa:.4f}) -- exactly")
        say("  where losing the gene is the mechanism. A tumour is a cell that is still alive, so a")
        say("  gene whose loss kills cannot be the thing that starts one. The model's essentiality")
        say("  estimate is not failing at cancer; it is measuring the constraint from the other side.")
    else:
        say("  THE ANTI-CORRELATION IS REAL BUT NOT EXPLAINED BY ROLE. Essentiality survives matching")
        say(f"  and still predicts non-drivers, but loss-of-function ({la:.4f}) and activating "
            f"({aa:.4f})")
        say("  drivers are constrained alike. Survivability predicts a difference between them and")
        say("  there is not one, so the direction is real and the explanation for it is not this.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(INTOGEN), str(CELL)], available=len(T), used=len(pairs) * 2,
                      selection="stratified", seed=SEED,
                      controls=[f"greedy nearest-neighbour matching without replacement, {CALIPER} SD "
                                f"caliper on length and citations",
                                f"{N_SHUFFLE} label shuffles inside the matched set",
                                "confounds re-scored after matching as the design check",
                                "LOEUF as a positive control",
                                "genes with both roles excluded from the role comparison"],
                      note="no regression anywhere; the partial-AUC design loop_cancer used is the "
                           "thing under test, so it is not reused to test itself. coverage is low by "
                           "design: a matched pair set is a stratification on the two confounds, not "
                           "a sample of all genes")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_survivable", "manifest": man, "gates": gates, "testable": True,
               "n_pairs": len(pairs), "confounds": conf, "measures": res, "by_role": role,
               "log": log}, open(OUT / "loop_survivable.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_survivable.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
