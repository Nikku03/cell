"""LOOP 64 -- IS THE CHROMOSOME DISCONNECTED, OR JUST SMALL?

WHERE THIS COMES FROM. Loop 61 scored 28 layers and found 26 fillable from the rest of the cell.
loops3d was not: model 0.5071, best single 0.5332, null 0.5211. Chance. That is the whole 3D genome
failing to connect to anything else in the model, and it was reported as the loop's most interesting
negative. But loops3d is also the SECOND-SMALLEST layer in the table at 767 present genes, and small
is a boring explanation that has to be eliminated before an interesting one is allowed to stand.

TWO EXPLANATIONS, and they make opposite predictions:

    SPARSE      767 positives is simply too few to fit 32 features against, and any layer this
                small would fail the same way
    DISCONNECTED  3D genome organisation genuinely does not covary with the rest of this model,
                and a layer of the same size built on anything else would be fine

ONE CHECK ALREADY DONE, because it would have made both explanations wrong. Chromosome-grouped folds
could make a layer unpredictable BY CONSTRUCTION if its positives sat on a few chromosomes, since
training folds would then hold none. Measured before this module was written: loops3d spans 23
chromosomes, and under loop 61's exact fold assignment every fold carries 90-203 held-out positives
against 564-677 in training. No fold is starved. The failure is not an artifact of how it was split.

THE TEST, which needs no data this project does not have. Take every layer loop 61 found EARNED and
big enough to cut down, SUBSAMPLE ITS POSITIVES TO 767 -- exactly loops3d's size -- keep the absent
set whole, and re-score with the same folds, baselines and null. This turns "is 767 too few?" from an
argument into a measurement, and it costs nothing but compute.

WHAT WAS NOT AVAILABLE, stated so the design is not mistaken for a preference. A denser 3D layer
would be the direct test, and there is no genome-wide one to hand: this project holds real Hi-C for
chr21 and chr22 only, and ENCODE returns no genome-wide loop calls under the filters tried. Loop 63
covers the complementary angle by adding replication timing, which is a genome-wide A/B compartment
measurement, as a FEATURE for predicting loops3d. This module asks the other half -- whether the
target's size is what breaks it.

PREDECLARED, before any number:

  C1 NO FOLD IS STARVED
       recomputed here rather than trusted to the paragraph above: loops3d must span >= 10
       chromosomes and every fold must carry >= 30 training and >= 30 held-out positives. If this
       fails, loop 61's loops3d row is void and nothing else in this module means anything.
  C2 LAYERS CUT TO 767 POSITIVES ARE STILL PREDICTABLE
       every EARNED layer with >= 2,000 positives, subsampled to 767, three independent draws,
       median margin reported. The gate is that the MEDIAN subsampled layer still earns. If most
       layers collapse at 767 then sparsity is a sufficient explanation, loops3d is unremarkable,
       and loop 61's headline negative must be withdrawn.
  C3 loops3d IS AN OUTLIER AT MATCHED SIZE
       its margin must fall below the 5th percentile of the subsampled distribution. Passing C2 and
       failing C3 would mean 767 is survivable in general but loops3d is merely at the weak end of
       normal, which is a much softer claim than "disconnected" and would be reported as such.
  C4 THE NULLS ARE MEASURED AT MATCHED SIZE
       label permutation inside each subsample, since a null computed at n=14,000 does not describe
       a fit at n=767.
  C5 THE VERDICT IS WRITTEN FROM C2 AND C3 TOGETHER
       sparse, disconnected, or undecided -- and undecided is a permitted outcome that gets said
       out loud rather than resolved by preference.

-> outputs/loop_chrom_connect.json
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_state_vector as SV
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"

TARGET_N = 767
MIN_FOR_CUT = 2000
DRAWS = 3
MIN_CHROM = 10
MIN_FOLD_POS = 30
SEED = 6401

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 64 -- is the chromosome disconnected, or just small?")
    say("=" * 100)
    say()

    D = json.load(open(CELL))
    G = D["genes"]
    n = len(G)
    L, pubs, chrom = SV.build_layers(D)
    ug = sorted(set(chrom))
    fold = {g: i for g, i in zip(ug, np.random.default_rng(SV.SEED).permutation(len(ug)) % SV.FOLDS)}
    fid = np.array([fold[c] for c in chrom])
    rng = np.random.default_rng(SEED)

    say("C1 NO FOLD IS STARVED")
    y3 = (L["loops3d"] > 0).astype(int)
    nch = len({G[i]["chrom"] for i in range(n) if y3[i]})
    ok = nch >= MIN_CHROM
    say(f"     loops3d present on {nch} chromosomes, {int(y3.sum())} genes")
    for k in range(SV.FOLDS):
        tr, te = fid != k, fid == k
        a, b = int(y3[tr].sum()), int(y3[te].sum())
        say(f"       fold {k}: train pos {a:4d}   held-out pos {b:4d}   held-out n {int(te.sum()):6,d}")
        if a < MIN_FOLD_POS or b < MIN_FOLD_POS:
            ok = False
    c1 = ok
    say(f"     C1 {'PASS' if c1 else 'FAIL'}")
    say()

    def score(target_y, name):
        feats = [k for k in sorted(L) if k != name]
        X = np.column_stack([L[k] for k in feats] + [pubs])
        keep = ~np.isnan(target_y)
        a_model, Pm = SV.cv_auc(X[keep], target_y[keep].astype(int), fid[keep])
        a_pubs, _ = SV.cv_auc(pubs[keep].reshape(-1, 1), target_y[keep].astype(int), fid[keep])
        best = float("nan")
        for k in feats:
            s, _ = SV.cv_auc(L[k][keep].reshape(-1, 1), target_y[keep].astype(int), fid[keep])
            if not np.isnan(s) and (np.isnan(best) or s > best):
                best = s
        okm = ~np.isnan(Pm)
        yy = target_y[keep].astype(int)[okm]
        nn = []
        for _ in range(SV.N_NULL):
            yp = yy.copy()
            rng.shuffle(yp)
            nn.append(SV.roc_auc_score(yp, Pm[okm]))
        nq = float(np.percentile(nn, 95))
        mg = a_model - max(a_pubs, best)
        return {"model": a_model, "pubs": a_pubs, "best1": best, "null95": nq, "margin": mg,
                "earned": bool(mg >= SV.MARGIN and a_model > nq)}

    prev = json.load(open(OUT / "loop_state_vector.json"))
    earned = [r["layer"] for r in prev["layers"]
              if r["earned"] and r["n_present"] >= MIN_FOR_CUT and r["layer"] != "loops3d"]
    say(f"C2 LAYERS CUT TO {TARGET_N} POSITIVES ARE STILL PREDICTABLE")
    say(f"     {len(earned)} EARNED layers have >= {MIN_FOR_CUT:,} positives and can be cut")
    say(f"     {'layer':13s} {'full n':>8s} {'full mg':>8s} {'cut mg (median of 3)':>22s}  verdict")
    cut_margins = []
    detail = []
    for t in earned:
        y = (L[t] > 0).astype(int)
        pos = np.where(y == 1)[0]
        full = next(r for r in prev["layers"] if r["layer"] == t)
        ms = []
        for d in range(DRAWS):
            sub = rng.choice(pos, size=TARGET_N, replace=False)
            yy = np.zeros(n, float)
            yy[sub] = 1.0
            yy[np.setdiff1d(pos, sub)] = np.nan      # dropped positives are removed, not called absent
            r = score(yy, t)
            ms.append(r["margin"])
        med = float(np.median(ms))
        cut_margins.append(med)
        e = med >= SV.MARGIN
        detail.append({"layer": t, "full_margin": full["margin"], "cut_margins": ms,
                       "cut_median": med, "cut_earned": bool(e)})
        say(f"     {t:13s} {full['n_present']:8,d} {full['margin']:+8.4f} {med:+22.4f}  "
            f"{'EARNED' if e else 'defer'}")
    n_earn = sum(1 for m in cut_margins if m >= SV.MARGIN)
    c2 = n_earn > len(cut_margins) / 2
    say(f"     {n_earn} of {len(cut_margins)} still earn at n={TARGET_N}; "
        f"median cut margin {np.median(cut_margins):+.4f}")
    say(f"     C2 {'PASS' if c2 else 'FAIL'}  -- {TARGET_N} positives "
        f"{'IS enough in general' if c2 else 'is NOT enough, sparsity explains loops3d'}")
    say()

    say("C3 loops3d IS AN OUTLIER AT MATCHED SIZE")
    l3 = next(r for r in prev["layers"] if r["layer"] == "loops3d")
    p5 = float(np.percentile(cut_margins, 5)) if cut_margins else float("nan")
    say(f"     loops3d margin        {l3['margin']:+.4f}   (model {l3['model']:.4f}, "
        f"null95 {l3['null95']:.4f})")
    say(f"     subsampled peers      5th pct {p5:+.4f}, median {np.median(cut_margins):+.4f}, "
        f"min {min(cut_margins):+.4f}")
    c3 = l3["margin"] < p5
    say(f"     C3 {'PASS' if c3 else 'FAIL'}")
    say()

    c4 = all(np.isfinite(m) for m in cut_margins)
    say(f"C4 nulls permuted inside each subsample, {SV.N_NULL} per draw, "
        f"{len(cut_margins) * DRAWS} fits")
    say(f"     C4 {'PASS' if c4 else 'FAIL'}")
    say()

    say("C5 THE VERDICT")
    if not c1:
        verdict = "VOID -- a fold was starved and loop 61's loops3d row cannot be interpreted"
    elif c2 and c3:
        verdict = ("DISCONNECTED -- 767 positives is enough for most layers, and loops3d falls below "
                   "the 5th percentile of the matched-size distribution. NOT below its minimum: at "
                   "least one subsampled layer does worse, so this is an outlier claim and not a "
                   "worst-in-class one. The 3D genome does not covary with the rest of this model")
    elif c2 and not c3:
        verdict = ("UNDECIDED -- 767 is survivable in general, but loops3d sits inside the range of "
                   "matched-size peers rather than below it. It is at the weak end of normal, which "
                   "is a softer claim than disconnected and loop 61's wording should be softened")
    else:
        verdict = ("SPARSE -- most layers also collapse at 767 positives, so size is a sufficient "
                   "explanation and loop 61's headline negative is WITHDRAWN")
    say(f"     {verdict}")
    c5 = True
    say()

    gates = {"C1 no fold is starved": bool(c1),
             "C2 767 positives is enough in general": bool(c2),
             "C3 loops3d is an outlier at matched size": bool(c3),
             "C4 nulls measured at matched size": bool(c4),
             "C5 verdict written from C2 and C3": bool(c5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(CELL), str(OUT / "loop_state_vector.json")],
                      available=len(prev["layers"]), used=len(earned), selection="filtered",
                      seed=SEED,
                      controls=["fold positive counts recomputed rather than assumed",
                                "positives subsampled to loops3d's exact size",
                                "dropped positives removed, never relabelled absent",
                                "three independent draws per layer, median reported",
                                "nulls permuted inside each subsample",
                                "loop 61 folds, baselines and margin threshold reused"],
                      note="no genome-wide 3D layer exists here to test the other direction; this "
                           "project holds Hi-C for chr21 and chr22 only")
    RM.report(man, emit=say)
    json.dump({"test": "loop_chrom_connect", "manifest": man, "gates": gates,
               "n_chromosomes": nch, "loops3d": l3, "cut_detail": detail,
               "cut_median": float(np.median(cut_margins)) if cut_margins else None,
               "cut_p5": p5, "n_still_earning": n_earn, "n_cut": len(cut_margins),
               "verdict": verdict, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_chrom_connect.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_chrom_connect.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
