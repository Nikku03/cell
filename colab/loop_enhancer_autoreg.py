"""Loop 175. Drop the self-regulating factors and run the enhancer test on the rest.

THE PROPOSAL, AND THE ONE PART OF IT THAT DOES NOT SURVIVE CONTACT WITH THE DATA. The suggestion
was that a factor binds many sites and many promoters, that around half of transcription factors
are self-activating or self-repressing, that such factors do not need a distal element to influence
transcription, and that removing them should therefore clean up the enhancer search. The mechanism
is real and the instinct is right -- a gene held by an autoregulatory loop is not a gene whose
expression a distal enhancer has to explain. Two things about the premise are measured here rather
than assumed, and they change what the test can be.

  THE 50% IS AN E. COLI NUMBER. Thieffry, Huerta, Perez-Rueda & Collado-Vides (BioEssays 1998)
  found about half of E. coli's characterised factors bind their own promoter; Rosenfeld, Elowitz &
  Alon (JMB 2002) explained why negative autoregulation is so common there. For human, TRRUST v2
  curation gives 24 self-loops among 795 curated regulators, and 21 of the 736 JASPAR matrices used
  here -- 4.9% of the curated ones, not 50%. Dropping 21 matrices out of 736 cannot do much, so the
  literal version of the request is run (A3) and reported, but it is not the interesting arm.

  CURATION CANNOT GIVE A 50/50 SPLIT WITHOUT MEASURING SOMETHING ELSE. The factors TRRUST calls
  self-regulating have a median out-degree of 39 against 5 for the curated factors it does not, so
  the label is substantially a statement about how much a factor has been studied. A split on it
  would be a split on fame.

SO THE SPLIT IS MEASURED INSTEAD, with the machinery already built: scan every factor's OWN
promoter with its OWN motif, and z-score that against the same motif's score across all the other
factor promoters. The z-score is the quantity, not the raw score -- a motif that scores highly on
every promoter is not autoregulating, it is merely easy. That gives a uniform, curation-free number
for all 736 matrices, and splitting it at its own median gives exactly the 50/50 the plan asked
for. Whether the prediction means anything is gated (A2) against curation, not asserted.

WHAT IS MEASURED. Loop 173's evaluation, unchanged: EP CRISPR benchmark K562 arm, 11,933 powered
pairs, 482 positives, within-gene recall at 1, chromosome-held-out, 5 folds x 5 seeds, identical
folds. The only thing that changes is WHICH MATRICES contribute to the sequence features. The bar
is the same bar: distance-only R@1 0.5930, and loop 173's full stack at 0.6050 with the
dinucleotide-shuffled control at 0.6090.

PREDECLARED, BEFORE ANY NUMBER.

  A1 IS THE SPLIT OVER A COMPLETE SET? The self-binding z must be defined for essentially every
     matrix, or "the non-self-regulating half" would silently mean "the half we could annotate".
     Gate: PASS iff z is defined for >= 95% of the 736 matrices.

  A2 DOES THE PREDICTED CALL MEASURE AUTOREGULATION AT ALL? The matrices TRRUST curates as
     self-regulating should carry a higher self-binding z than the curated matrices it does not.
     Gate: PASS iff the SELF group exceeds the NO_SELF group by a one-sided Mann-Whitney at
     p < 0.05. A FAIL means the 50/50 split is splitting on something, but not on autoregulation,
     and every arm below has to be read that way.

  A3 THE LITERAL REQUEST: drop the 21 curated self-loop matrices.
     Gate: within-gene R@1 improves over loop 173's all-matrix full stack in >= 4/5 seeds and by
     more than 3 sem.

  A4 THE 50/50 SPLIT: drop the 368 matrices with the highest self-binding z.
     Gate: same as A3.

  A5 THE SIZE-MATCHED CONTROL, and it is the one that decides what A4 means. Drop the SAME NUMBER
     of matrices chosen at random, 12 draws. If a random 368 helps just as much, the finding is
     that fewer matrices make a better feature set -- less noise summed into every column -- and
     has nothing to do with autoregulation.
     Gate: PASS iff the real drop beats the matched drop in >= 90% of draws.

  A6 DOES ANY OF IT BEAT THE FLOOR? The best retained-matrix arm against distance alone, on
     identical folds.
     Gate: PASS iff paired R@1 is positive in >= 4/5 seeds past 3 sem AND paired AUPRC is
     >= +0.01 in >= 4/5 seeds -- the same bar loop 173's E3 failed.

  A7 THE SHUFFLE STILL DECIDES. The retained-matrix stack against the same stack on
     dinucleotide-shuffled elements. No subset of matrices can be said to be reading binding sites
     while a composition-matched shuffle matches it.
     Gate: PASS iff real beats shuffled in >= 4/5 seeds and by more than 3 sem.

  A8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_autoreg.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import autoregulation as AR         # noqa: E402
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_autoreg.json"
N_MATCHED_DRAWS = 12
MATCHED_SEEDS = [0, 1]
MIN_SEEDS = 4
MIN_Z_COVERAGE = 0.95
L173_FULL_R1 = 0.6050
L173_DIST_R1 = 0.5930

MOTIF_AXIS_0 = ["el_MX", "el_LZ", "el_NS", "sh_MX", "sh_LZ", "sh_NS",
                "pr_MX", "pr_LZ", "pr_NS", "bg_MX", "bg_LZ", "bg_NS",
                "motif_ids", "motif_width", "motif_maxscore"]
MOTIF_AXIS_1 = ["el_SH", "sh_SH"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def mask_payload(P, keep):
    """A copy of the scan payload with only the kept matrices. Every array carrying a motif axis is
    subset on it; nothing else is touched, so the pairs, folds and labels are bit-identical."""
    Q = dict(P)
    for k in MOTIF_AXIS_0:
        if k in Q:
            Q[k] = Q[k][keep]
    for k in MOTIF_AXIS_1:
        if k in Q:
            Q[k] = Q[k][:, keep]
    return Q


def arm(P, keep, y, chrom, g_idx, jitter, tag, elem="el"):
    Q = mask_payload(P, keep)
    F, _, _ = L.build_features(Q, elem, report=lambda *_: None)
    for c in F:
        F[c] = np.nan_to_num(F[c], nan=0.0, posinf=0.0, neginf=0.0)
    X, _ = L.matrix(F, L.ARMS["FULL"])
    return L.run_arm(X, y, chrom, g_idx, jitter, tag, say)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 175  DROP THE SELF-REGULATING FACTORS AND RUN THE ENHANCER TEST ON THE REST")
    say("=" * 104)
    say(f"  PREDECLARED: z defined for >= {MIN_Z_COVERAGE:.0%} of matrices; the curated split must")
    say(f"  separate on the predicted one (Mann-Whitney p < 0.05); each drop judged against loop")
    say(f"  173's all-matrix R@1 {L173_FULL_R1:.4f} in >= {MIN_SEEDS}/5 seeds past 3 sem; the")
    say(f"  50/50 drop must beat a size-matched random drop in >= 90% of {N_MATCHED_DRAWS} draws;")
    say(f"  and the winner must clear the distance floor {L173_DIST_R1:.4f} on loop 173's E3 bar.")
    say()

    P = SC.load(say)
    y = P["y"].astype(int)
    g_idx = P["g_idx"]
    chrom = np.array([str(c) for c in P["chrom"]])
    jitter = np.random.default_rng(L.TIE_SEED).uniform(0, 1e-9, size=len(y))
    ids = [str(m) for m in P["motif_ids"]]
    nm = len(ids)

    # ---- A1 ------------------------------------------------------------------------------------
    say("A1 IS THE SPLIT OVER A COMPLETE SET?")
    cur = AR.load()["matrices"]
    sb = AR.self_binding(lambda s: say("   " + s))
    zid = {str(m): i for i, m in enumerate(sb["motif_ids"])}
    z = np.array([sb["self_z"][zid[m]] if m in zid else np.nan for m in ids])
    cov = float(np.isfinite(z).mean())
    say(f"     self-binding z defined for {int(np.isfinite(z).sum())}/{nm} matrices ({cov:.4f})")
    cls = np.array([cur.get(m, {}).get("cls", "UNCURATED") for m in ids])
    for c in ("SELF", "NO_SELF", "UNCURATED"):
        say(f"     {c:10} {int((cls == c).sum()):4d} matrices")
    a1 = bool(cov >= MIN_Z_COVERAGE)
    GG.verdict(a1, emit=say,
               if_true=f"A1 PASS -- the 50/50 split is over {int(np.isfinite(z).sum())} matrices, "
                       f"not over the curated subset",
               if_false=f"A1 FAIL -- z is defined for only {cov:.1%}, so any split is confounded "
                        f"with which factors could be annotated")

    # ---- A2 ------------------------------------------------------------------------------------
    say()
    say("A2 DOES THE PREDICTED CALL MEASURE AUTOREGULATION AT ALL?")
    zs = z[(cls == "SELF") & np.isfinite(z)]
    zn = z[(cls == "NO_SELF") & np.isfinite(z)]
    u, p = stats.mannwhitneyu(zs, zn, alternative="greater") if len(zs) and len(zn) else (0, 1.0)
    say(f"     curated SELF    n {len(zs):3d}  self-binding z median {np.median(zs):+.3f}")
    say(f"     curated NO_SELF n {len(zn):3d}  self-binding z median {np.median(zn):+.3f}")
    say(f"     one-sided Mann-Whitney p = {p:.4g}")
    a2 = bool(p < 0.05)
    GG.verdict(a2, emit=say,
               if_true="A2 PASS -- the factors curation calls self-regulating do score higher on "
                       "their own promoters, so the predicted split is tracking autoregulation",
               if_false="A2 FAIL -- the predicted score does not recover the curated label, so the "
                        "50/50 split below is splitting on something else and must be read that way")

    # ---- the arms ------------------------------------------------------------------------------
    say()
    say("   the arms, all on loop 173's pairs, folds and seeds")
    res = {}
    res["all_matrices"] = arm(P, np.ones(nm, bool), y, chrom, g_idx, jitter, "all 736 matrices")
    keep_cur = cls != "SELF"
    res["drop_curated"] = arm(P, keep_cur, y, chrom, g_idx, jitter,
                              f"drop curated SELF ({int((~keep_cur).sum())})")
    zz = np.where(np.isfinite(z), z, -np.inf)
    med = np.median(zz[np.isfinite(z)])
    keep_half = zz <= med
    res["drop_top_half"] = arm(P, keep_half, y, chrom, g_idx, jitter,
                               f"drop top-half self-binding ({int((~keep_half).sum())})")
    res["drop_top_half_shuffled"] = arm(P, keep_half, y, chrom, g_idx, jitter,
                                        "drop top-half, SHUFFLED elements", elem="sh")
    X, _ = L.matrix({c: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                     for c, v in L.build_features(P, "el", report=lambda *_: None)[0].items()},
                    L.ARMS["distance"])
    res["distance"] = L.run_arm(X, y, chrom, g_idx, jitter, "distance", say)

    # ---- A3, A4 --------------------------------------------------------------------------------
    say()
    say("A3 THE LITERAL REQUEST: drop the curated self-loop matrices")
    d3 = L.paired(res["drop_curated"], res["all_matrices"])
    say(f"     {L.fmt(d3)}")
    a3 = L.gate_pair(d3, use_ap=False)
    GG.verdict(a3, emit=say,
               if_true="A3 PASS -- removing the curated self-regulating factors helps",
               if_false="A3 FAIL -- removing them changes nothing measurable, which is what 21 "
                        "matrices out of 736 was always likely to do")

    say()
    say("A4 THE 50/50 SPLIT: drop the half that binds its own promoter most")
    d4 = L.paired(res["drop_top_half"], res["all_matrices"])
    say(f"     {L.fmt(d4)}")
    a4 = L.gate_pair(d4, use_ap=False)
    GG.verdict(a4, emit=say,
               if_true="A4 PASS -- keeping only the least self-binding half of the factors helps",
               if_false="A4 FAIL -- the least self-binding half does no better than all of them")

    # ---- A5 ------------------------------------------------------------------------------------
    say()
    say(f"A5 THE SIZE-MATCHED CONTROL: {N_MATCHED_DRAWS} random drops of the same size")
    ndrop = int((~keep_half).sum())
    real = float(res["drop_top_half"]["r1"][:len(MATCHED_SEEDS)].mean())
    draws = []
    for d in range(N_MATCHED_DRAWS):
        rr = np.random.default_rng(5000 + d)
        km = np.ones(nm, bool)
        km[rr.choice(nm, ndrop, replace=False)] = False
        Q = mask_payload(P, km)
        F, _, _ = L.build_features(Q, "el", report=lambda *_: None)
        for c in F:
            F[c] = np.nan_to_num(F[c], nan=0.0, posinf=0.0, neginf=0.0)
        Xm, _ = L.matrix(F, L.ARMS["FULL"])
        vals = []
        for s in MATCHED_SEEDS:
            fold = L.folds_for(chrom, s)
            sc = L.oof_scores(Xm, y, fold, s)
            a, b, n = L.within_gene(sc, y, g_idx, jitter)
            vals.append(a)
        draws.append(float(np.mean(vals)))
        say(f"     draw {d+1:2d}/{N_MATCHED_DRAWS}  R@1 {draws[-1]:.4f}")
    draws = np.array(draws)
    frac = float((real > draws).mean())
    say(f"     real drop {real:.4f} against matched draws mean {draws.mean():.4f} "
        f"(min {draws.min():.4f}, max {draws.max():.4f}); real beats {frac:.0%} of them")
    a5 = bool(frac >= 0.90)
    GG.verdict(a5, emit=say,
               if_true="A5 PASS -- dropping the self-binding half beats dropping the same number at "
                       "random, so the split is about autoregulation and not about count",
               if_false="A5 FAIL -- a random drop of the same size does as well, so whatever A4 "
                        "showed is about how many matrices are summed, not about which ones")

    # ---- A6 ------------------------------------------------------------------------------------
    say()
    say("A6 DOES ANY OF IT BEAT THE DISTANCE FLOOR?")
    best = max(("drop_curated", "drop_top_half", "all_matrices"),
               key=lambda k: res[k]["r1"].mean())
    d6 = L.paired(res[best], res["distance"])
    say(f"     best arm is {best} at R@1 {res[best]['r1'].mean():.4f}")
    say(f"     {best} vs distance   {L.fmt(d6)}")
    a6 = L.gate_pair(d6)
    GG.verdict(a6, emit=say,
               if_true=f"A6 PASS -- {best} clears the bar loop 173's full stack failed",
               if_false="A6 FAIL -- no matrix subset clears the distance floor on loop 173's bar")

    # ---- A7 ------------------------------------------------------------------------------------
    say()
    say("A7 THE SHUFFLE STILL DECIDES")
    d7 = L.paired(res["drop_top_half"], res["drop_top_half_shuffled"])
    say(f"     drop_top_half vs the same on shuffled elements   {L.fmt(d7)}")
    a7 = L.gate_pair(d7, use_ap=False)
    GG.verdict(a7, emit=say,
               if_true="A7 PASS -- on the retained half, real sequence beats a composition-matched "
                       "shuffle, so this subset is reading binding sites",
               if_false="A7 FAIL -- the shuffle matches it on the retained half too, so removing "
                        "the self-regulating factors did not turn composition into sites")

    say()
    say("A8 WHAT THIS CANNOT SHOW")
    say("     Autoregulation is a statement about a factor's OWN gene. That a factor holds itself")
    say("     in a loop does not mean its TARGETS are enhancer-independent, and nothing here tests")
    say("     that step -- it is the load-bearing assumption of the whole request and it is")
    say("     untested by this loop.")
    say("     The predicted split uses a 1 kb promoter window and a single motif per factor, so a")
    say("     factor that autoregulates through a distal element of its own is scored as not")
    say("     self-regulating.")
    say("     TRRUST curation is biased toward studied factors, which is why A5 exists and why the")
    say("     curated arm is reported but not leaned on.")
    a8 = True
    say(f"     A8 {'PASS' if a8 else 'FAIL'}")

    gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "A8": a8}
    man = RM.manifest(inputs=[Path("colab/data/tf_autoregulation.json"),
                              Path("colab/data/tf_self_binding.npz")],
                      available=nm, used=int(keep_half.sum()), selection="lowest-half self-binding z",
                      seed=5000,
                      controls=[f"{N_MATCHED_DRAWS} size-matched random matrix drops",
                                "the curated split checked against the predicted one",
                                "dinucleotide shuffle on the retained half",
                                "identical pairs, folds and seeds as loop 173"],
                      note="does removing self-regulating factors rescue the sequence chain")
    out = dict(test="enhancer autoregulation split", gates=gates,
               n_matrices=nm, n_curated_self=int((cls == "SELF").sum()),
               n_dropped_half=ndrop, z_coverage=cov,
               curated_vs_predicted=dict(n_self=len(zs), n_no_self=len(zn),
                                         median_self=float(np.median(zs)) if len(zs) else None,
                                         median_no_self=float(np.median(zn)) if len(zn) else None,
                                         mannwhitney_p=float(p)),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "mrr", "ap")}
                     for k, v in res.items()},
               matched_draws=[float(x) for x in draws], matched_frac_beaten=frac,
               increments={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                               for kk, vv in d.items()}
                           for k, d in (("A3", d3), ("A4", d4), ("A6", d6), ("A7", d7))},
               manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
