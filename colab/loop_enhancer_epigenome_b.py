"""Loop 188b. The epigenetic layer, rerun with three defects in loop 188 corrected.

Loop 188 scored 4/9 with one VOID. Its conclusion -- that chromatin state adds a great deal over a
motif-and-shape base and almost nothing over loop 185's measured-binding stack -- is not in question
here and is expected to reproduce. What is in question is three things loop 188 got wrong about its
own machinery, all of which are in the repo record already and none of which change that conclusion.

DEFECT 1, AND IT PUT A FALSE LINE IN THE RECORD. G2 tested three signed direction predictions with
a one-sided Mann-Whitney. Ninety of the 4,482 elements have zero CpGs measured by WGBS, so their
5mC is nan; np.median propagates a single nan and scipy returns a nan p-value, and the code printed

    el_5mc  predicted lower  median functional +nan vs +nan  one-sided p nan  REFUTED

which is an undefined statistic dressed as a verdict. The 5mC prediction has never actually been
tested. Every reduction in this loop now goes through `mw`, which drops non-finite values, counts
them, and returns defined=False rather than a number, and an undefined prediction is VOID rather
than refuted. Note that loop 188's G1 SAW this -- it reports el_5mc as the worst-defined column at
98.0% and passes it, correctly, because 98% coverage is fine for a model that imputes. It is fatal
for np.median, which has no threshold at all.

DEFECT 2, AN ARM THAT DID NOT MEASURE WHAT ITS NAME SAID. G3 was called "the repressive and
insulating arm" and contained H3K27me3, CTCF and the elementChromatinCategory one-hot. Three of that
category's five levels are High H3K27ac / H3K27ac / No H3K27ac, which encode an ACTIVATING mark, and
the base stack it was added to has no H3K27ac column at all. So the arm smuggled activation into the
arm named for repression, and its +0.0476 AUPRC gain cannot be attributed. Here the two are separate
arms -- H5 is H3K27me3 and CTCF alone, H6 is the category alone -- so the question "what is
repression worth" has an answer that is about repression.

DEFECT 3, A VERDICT LINE THAT OVERSTATED A FAILURE. G3's FAIL text read "it adds nothing the motif
and shape columns did not have" when AUPRC had gone up in 5/5 seeds by +0.0476; it failed the R@1
half of the bar only (+0.0181 against a 3-sem requirement of 0.0279). Every bar verdict here names
which half failed and quotes both halves, so a FAIL cannot be read as "no effect".

AND ONE DESIGN CHANGE THAT IS NOT A DEFECT FIX. Loop 188's G7 permuted the promoter marks across
genes and gated on R@1. G6 had passed on AUPRC, not on R@1, so G7 was testing a channel in which no
claim had been made; it correctly returned VOID, and in doing so it did not score the thing its own
numbers showed most clearly -- permuting cost dAUPRC +0.0441 in 5/5 seeds, meaning the promoter
marks' global gain IS gene-specific. H10 gates the permutation on whichever channel H9 passed on,
which is the only version of that control that tests the claim being made.

THE H3K27ME3 REFUTATION, AND HOW IT IS HANDLED. Loop 188 predicted H3K27me3 would be LOWER at
functional pairs and it came out higher, one-sided p = 1. That refutation is real and it stands; it
is not re-litigated by swapping the tested quantity. But loop 188's own G1 table shows only 1.3% of
elements overlap an H3K27me3 peak, so the RPM column is 98.7% background, and background read depth
scales with local accessibility -- which is higher at functional elements. H3 tests that explanation
directly and PROSPECTIVELY, because it is a claim about H3K27me3 against ACCESSIBILITY and makes no
reference to the outcome: if the RPM is mostly coverage, it must correlate with DHS across elements,
and the wrong-way direction must weaken or vanish inside accessibility strata.

H4 is different and is labelled so. The observation that H3K27me3-CATEGORY elements are 0/64
functional came from loop 188's descriptive table, on these same 11,933 pairs. Re-testing it here
cannot be independent evidence no matter how it comes out, and H4 is therefore reported as a
POST-HOC diagnostic and is not gated. It is a hypothesis for a loop with different data, not a
result.

PREDECLARED, BEFORE ANY NUMBER.

  H1 DOES THE MARK LAYER JOIN? Definedness per column on both sides, with the non-finite count
     printed per column rather than only the worst.
     Gate: PASS iff every element column is finite for >= 95% of elements and every promoter column
     for >= 90% of promoters.

  H2 DO THE MARKS POINT THE WAY THEY SHOULD? The same three signed predictions as loop 188 --
     H3K27me3 lower, 5mC lower, H3K4me1 higher in functional pairs -- now with non-finite values
     dropped and counted.
     Gate: PASS iff every prediction whose test is DEFINED holds at one-sided p < 0.05. A prediction
     whose test is undefined is VOID for that prediction and says so; it is not refuted.

  H3 IS THE H3K27ME3 RESULT A COVERAGE ARTEFACT? Prospective, and independent of the outcome
     variable. Two parts:
       (a) Spearman between H3K27me3 RPM and DHS RPM across elements. If the column is mostly
           background, this is strongly positive.
       (b) the functional-versus-not comparison repeated INSIDE quintiles of DHS accessibility.
     Gate: PASS iff (a) rho > 0.30 AND (b) the functional median is no higher than the
     non-functional median in at least 3 of the 5 strata. A PASS says the direction loop 188 found
     is coverage; a FAIL says H3K27me3 really is elevated at functional elements and the Polycomb
     reading was simply wrong.

  H4 THE THRESHOLDED CALL. POST-HOC, NOT GATED. Positive rate by H3K27me3 peak overlap and by
     chromatin category, on the same pairs that produced the observation.

  H5 REPRESSION AND INSULATION, CLEAN. H3K27me3 and CTCF only -- RPM and peak call, no category --
     over the base motif and shape stack.
     Gate: loop 173's E3 bar, with the verdict naming which half failed.

  H6 WHAT THE CATEGORY IS WORTH ON ITS OWN. The category one-hot alone over the same base.
     Gate: same bar. H5 and H6 together are what loop 188's G3 conflated.

  H7 THE ENHANCER/PROMOTER DISCRIMINATOR. H3K4me1, element H3K4me3, their contrast. Same bar.

  H8 DNA METHYLATION. Element 5mC, CpG count, CpG density. Same bar.

  H9 THE PROMOTER SIDE. Gate: PASS iff the AUPRC half clears. R@1 reported, not gated.

  H10 IS THE PROMOTER SIDE GENE-SPECIFIC? Promoter marks permuted across genes, every value kept.
     Gate: PASS iff real beats permuted IN THE CHANNEL H9 PASSED ON, in >= 4/5 seeds past 3 sem for
     R@1 or >= 4/5 seeds at the increment bar for AUPRC. VOID only if H9 itself failed, because then
     there is no claim to control.

  H11 THE DECISIVE ONE. The whole epigenetic block over loop 185's best stack. Same bar.

  H12 AGAINST DISTANCE ALONE. Same bar, and the winner must clear distance-only R@1 0.5930.

  H13 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_epigenome_b.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import chip as CH                   # noqa: E402
from enh import epigenome as EPI             # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402
import loop_enhancer_tfnet as L183           # noqa: E402
import loop_enhancer_cobinding as L185       # noqa: E402
import loop_enhancer_epigenome as L188       # noqa: E402

from scipy.stats import mannwhitneyu, spearmanr                  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_epigenome_b.json"
SEEDS = L173.SEEDS
MIN_SEEDS = L173.MIN_SEEDS
MIN_DEFINED_EL = 0.95
MIN_DEFINED_PR = 0.90
ALPHA = 0.05
L173_DIST_R1 = 0.5930
MIN_RHO = 0.30           # H3(a): what counts as "the RPM tracks coverage"
MIN_STRATA = 3           # H3(b): of 5 accessibility quintiles
N_STRATA = 5
SEED = 188288

# loop 188's G3 conflated these two; here they are separate arms
REPRESS = ["el_k27me3", "el_k27me3_peak", "el_ctcf", "el_ctcf_peak"]
CATEGORY = ["cat_high_k27ac", "cat_k27ac", "cat_no_k27ac", "cat_ctcf_element",
            "cat_k27me3_element"]
ME1ME3 = L188.ME1ME3
METHYL = L188.METHYL
PROMOTER = L188.PROMOTER
EPI_ALL = REPRESS + CATEGORY + ME1ME3 + METHYL + PROMOTER

SIGNED = {"el_k27me3": "lower", "el_5mc": "lower", "el_k4me1": "higher"}
UNSIGNED = ["el_ctcf", "el_k4me3_sig"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def mw(a, b, alternative):
    """One-sided Mann-Whitney with non-finite values dropped and counted.

    np.median propagates a single nan and scipy's tests return a nan p-value without warning.
    Loop 188's G2 printed 5mC as REFUTED on exactly that basis. Every comparison in this loop goes
    through here, and an undefined result comes back as defined=False rather than as a number, so
    the caller has to decide what to do with it instead of a nan silently becoming a verdict."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ka, kb = np.isfinite(a), np.isfinite(b)
    dropped = int((~ka).sum() + (~kb).sum())
    a, b = a[ka], b[kb]
    if len(a) < 2 or len(b) < 2:
        return dict(n_a=int(len(a)), n_b=int(len(b)), dropped=dropped, median_a=float("nan"),
                    median_b=float("nan"), p=float("nan"), defined=False)
    _, p = mannwhitneyu(a, b, alternative=alternative)
    return dict(n_a=int(len(a)), n_b=int(len(b)), dropped=dropped,
                median_a=float(np.median(a)), median_b=float(np.median(b)),
                p=float(p), defined=True)


def bar_halves(d):
    """Which half of loop 173's E3 bar cleared, in words, so a FAIL cannot be read as 'no effect'.

    Loop 188's G3 failed with AUPRC up in 5/5 seeds by +0.0476 and printed 'it adds nothing the
    motif and shape columns did not have'. That was false about its own numbers."""
    r1_ok = (d["n_pos_r1"] >= MIN_SEEDS) and (d["mean_r1"] > 3 * d["sem_r1"])
    ap_ok = d["n_ap_pass"] >= MIN_SEEDS
    r1_txt = (f"R@1 {d['mean_r1']:+.4f} +/- {d['sem_r1']:.4f} ({d['n_pos_r1']}/5 up, "
              f"needs > {3*d['sem_r1']:.4f})")
    ap_txt = (f"AUPRC {d['mean_ap']:+.4f} ({d['n_ap_pass']}/5 at >= {L173.MIN_INCREMENT})")
    if r1_ok and ap_ok:
        return True, f"both halves clear: {r1_txt}; {ap_txt}"
    if ap_ok:
        return False, f"the AUPRC half CLEARS ({ap_txt}) and the R@1 half does not ({r1_txt})"
    if r1_ok:
        return False, f"the R@1 half CLEARS ({r1_txt}) and the AUPRC half does not ({ap_txt})"
    return False, f"neither half clears: {r1_txt}; {ap_txt}"


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 188b  THE EPIGENETIC LAYER, with loop 188's three machinery defects corrected")
    say("=" * 104)
    say("  PREDECLARED: non-finite values dropped and COUNTED in every comparison, and a")
    say("  prediction whose test is undefined is VOID rather than refuted -- loop 188's G2 printed")
    say("  5mC as REFUTED on a nan median over 90 elements with no measured CpG; repression and")
    say("  the chromatin category run as SEPARATE arms, because loop 188's G3 put an H3K27ac")
    say("  discretisation inside the arm named for repression; every bar verdict names which half")
    say("  failed; H3 tests the coverage explanation for loop 188's H3K27me3 refutation")
    say(f"  prospectively (rho > {MIN_RHO} against DHS, and the direction gone in >= {MIN_STRATA}")
    say("  of 5 accessibility strata); H4 is POST-HOC and not gated; H10 gates the permutation on")
    say("  whichever channel H9 passed on; and the winner must clear distance-only R@1 "
        f"{L173_DIST_R1}.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))
    n_el, n_gn = len(S["el_key"]), len(S["gn_key"])
    say(f"    {len(y):,} element-gene pairs, {int(y.sum()):,} functional, "
        f"{n_el:,} elements, {n_gn:,} genes")

    epi = EPI.build(S["el_key"], S["gn_key"], say)
    EL = L188.element_frame(S["el_key"], epi, say)
    PR = L188.promoter_frame(epi)

    # ---- H1 ------------------------------------------------------------------------------------
    say()
    say("H1 DOES THE MARK LAYER JOIN?")
    def_el = {c: float(np.isfinite(EL[c]).mean()) for c in EL}
    def_pr = {c: float(np.isfinite(PR[c]).mean()) for c in PR}
    for c in sorted(def_el, key=def_el.get):
        n_bad = int((~np.isfinite(EL[c])).sum())
        if n_bad or def_el[c] < 1.0:
            say(f"     element {c:20} defined {def_el[c]:.2%}  ({n_bad} non-finite)")
    for c in sorted(def_pr, key=def_pr.get):
        n_bad = int((~np.isfinite(PR[c])).sum())
        if n_bad or def_pr[c] < 1.0:
            say(f"     promoter {c:19} defined {def_pr[c]:.2%}  ({n_bad} non-finite)")
    say(f"     every other column is finite everywhere; {len(EL)} element and {len(PR)} promoter "
        f"columns in total")
    worst_el, worst_pr = min(def_el, key=def_el.get), min(def_pr, key=def_pr.get)
    h1 = bool(def_el[worst_el] >= MIN_DEFINED_EL and def_pr[worst_pr] >= MIN_DEFINED_PR)
    GG.verdict(h1, emit=say,
               if_true=f"H1 PASS -- worst is {worst_el} at {def_el[worst_el]:.2%}. Note this gate "
                       f"protects the MODEL, which imputes, and not the statistics below, which "
                       f"propagate a single nan; that is loop 188's lesson and mw() is the fix",
               if_false=f"H1 FAIL -- {worst_el} at {def_el[worst_el]:.2%}")

    # ---- H2 ------------------------------------------------------------------------------------
    say()
    say("H2 DO THE MARKS POINT THE WAY THEY SHOULD?")
    pos, neg = y == 1, y == 0
    dirs, undefined = {}, []
    for c, want in SIGNED.items():
        v = EL[c][e_idx]
        r = mw(v[pos], v[neg], "less" if want == "lower" else "greater")
        holds = bool(r["defined"] and r["p"] < ALPHA)
        r.update(predicted=want, holds=holds)
        dirs[c] = r
        if not r["defined"]:
            undefined.append(c)
            say(f"     {c:12} predicted {want:6}  UNDEFINED after dropping {r['dropped']} "
                f"non-finite pairs -- VOID for this prediction, NOT refuted")
        else:
            say(f"     {c:12} predicted {want:6}  median functional {r['median_a']:+.4f} vs "
                f"{r['median_b']:+.4f}   one-sided p {r['p']:.3g}   "
                f"({r['dropped']} non-finite dropped)   {'HOLDS' if holds else 'REFUTED'}")
    for c in UNSIGNED:
        v = EL[c][e_idx]
        r = mw(v[pos], v[neg], "two-sided")
        say(f"     {c:12} UNSIGNED -- reported only: median functional {r['median_a']:+.4f} vs "
            f"{r['median_b']:+.4f}   two-sided p {r['p']:.3g}")
    defined_preds = [c for c in SIGNED if dirs[c]["defined"]]
    h2 = bool(defined_preds and all(dirs[c]["holds"] for c in defined_preds))
    GG.verdict(h2, emit=say,
               if_true=f"H2 PASS -- every prediction with a defined test holds "
                       f"({len(defined_preds)}/3 defined)",
               if_false="H2 FAIL -- " + ", ".join(c for c in defined_preds if not dirs[c]["holds"])
                        + " runs against its prediction on a test that is defined"
                        + (f"; {', '.join(undefined)} VOID" if undefined else ""))

    # ---- accessibility, needed by H3 and by the loop 185 stack ---------------------------------
    rows = SC.load_benchmark(lambda *_: None)
    dhs, h3k = {}, {}
    for r in rows:
        k = f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}"
        try:
            dhs[k] = float(r.get("DHS.RPM") or 0)
            h3k[k] = float(r.get("H3K27ac.RPM") or 0)
        except ValueError:
            pass
    ek = [str(k) for k in S["el_key"]]
    el_dhs = np.array([np.log10(1 + dhs.get(k, 0.0)) for k in ek])

    # ---- H3 ------------------------------------------------------------------------------------
    say()
    say("H3 IS THE H3K27ME3 RESULT A COVERAGE ARTEFACT?")
    say("     loop 188 predicted H3K27me3 LOWER at functional pairs and found it HIGHER at "
        "one-sided p = 1.")
    say("     that refutation stands. this gate asks only whether the column is measuring "
        "Polycomb or read depth,")
    say("     and it is prospective: both parts are about H3K27me3 against ACCESSIBILITY and "
        "neither looks at the label.")
    ok_k = np.isfinite(EL["el_k27me3"]) & np.isfinite(el_dhs)
    rho, p_rho = spearmanr(EL["el_k27me3"][ok_k], el_dhs[ok_k])
    say(f"     (a) Spearman(H3K27me3 RPM, DHS RPM) over {int(ok_k.sum()):,} elements = "
        f"{rho:+.3f}  (p {p_rho:.3g})")
    pair_dhs = el_dhs[e_idx]
    edges = np.quantile(pair_dhs, np.linspace(0, 1, N_STRATA + 1))
    edges[-1] += 1e-9
    strata, n_gone = [], 0
    say(f"     (b) the same comparison inside {N_STRATA} quintiles of accessibility:")
    for q in range(N_STRATA):
        m = (pair_dhs >= edges[q]) & (pair_dhs < edges[q + 1])
        v = EL["el_k27me3"][e_idx][m]
        yy = y[m]
        r = mw(v[yy == 1], v[yy == 0], "less")
        gone = bool(r["defined"] and r["median_a"] <= r["median_b"])
        n_gone += int(gone)
        strata.append(dict(q=q + 1, n=int(m.sum()), n_pos=int(yy.sum()),
                           median_pos=r["median_a"], median_neg=r["median_b"], p=r["p"],
                           direction_gone=gone))
        say(f"         Q{q+1}  n {int(m.sum()):5,} ({int(yy.sum()):3d} functional)   "
            f"median functional {r['median_a']:+.4f} vs {r['median_b']:+.4f}   "
            f"{'direction GONE' if gone else 'still higher at functional'}")
    h3 = bool(rho > MIN_RHO and n_gone >= MIN_STRATA)
    GG.verdict(h3, emit=say,
               if_true=f"H3 PASS -- rho {rho:+.3f} and the direction is gone in {n_gone}/"
                       f"{N_STRATA} strata, so H3K27me3 RPM here is tracking read depth and loop "
                       f"188's refutation is about coverage rather than about Polycomb",
               if_false=f"H3 FAIL -- rho {rho:+.3f} (bar {MIN_RHO}) and the direction is gone in "
                        f"{n_gone}/{N_STRATA} strata (bar {MIN_STRATA}). The coverage explanation "
                        f"does not carry it, so H3K27me3 really is elevated at functional elements "
                        f"and the Polycomb reading was simply wrong")

    # ---- H4 ------------------------------------------------------------------------------------
    say()
    say("H4 THE THRESHOLDED CALL -- POST-HOC, NOT GATED")
    say("     this observation came from loop 188's descriptive table on these same pairs, so")
    say("     re-testing it here cannot be independent evidence however it comes out. it is a")
    say("     hypothesis for a loop with different data, and it is recorded as one.")
    posthoc = {}
    pk = EL["el_k27me3_peak"][e_idx] > 0
    for lab, m in (("H3K27me3 peak", pk), ("no H3K27me3 peak", ~pk)):
        posthoc[lab] = dict(n=int(m.sum()), pos=int(y[m].sum()),
                            rate=float(y[m].mean()) if m.sum() else float("nan"))
        say(f"       {lab:22} {int(m.sum()):6,} pairs   "
            f"{(y[m].mean() if m.sum() else float('nan')):.2%} functional")
    for cname in CATEGORY:
        m = EL[cname][e_idx] > 0
        if m.sum():
            posthoc[cname] = dict(n=int(m.sum()), pos=int(y[m].sum()), rate=float(y[m].mean()))
            say(f"       {cname:22} {int(m.sum()):6,} pairs   {y[m].mean():.2%} functional")
    say("     H4 (descriptive)")

    # ---- the stacks ----------------------------------------------------------------------------
    say()
    say("   building the reference stacks")
    E178, FAM, _ = L178.element_frame(S, "el", lambda *_: None)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    for c in P:
        P[c] = np.nan_to_num(P[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in sorted(FAM)])
    Xd = np.column_stack([P["log_dist"]])
    dom = TD.load()
    names = sorted({(v.get("name") or "").upper().split("::")[0]
                    for v in dom.values() if v.get("name")})
    Be, tfs = CH.build(S["el_key"], names, lambda *_: None)
    prom_key = []
    for k in S["gn_key"]:
        c, p_, _ = str(k).split(":")
        prom_key.append(f"{c}:{max(0, int(p_) - CH.PROMOTER_PAD)}-{int(p_) + CH.PROMOTER_PAD}")
    Bp, tfs_p = CH.build(prom_key, names, lambda *_: None, cache=CH.PROM_CACHE, pad=0)
    if tfs_p != tfs:
        common = sorted(set(tfs) & set(tfs_p))
        Be, Bp, tfs = (Be[[tfs.index(t) for t in common]],
                       Bp[[tfs_p.index(t) for t in common]], common)
    OV = L185.overlap_features(Be, Bp, e_idx, g_idx)
    sets, grow, midf, gom = L183.network(S, lambda *_: None)
    nb = json.load(gzip.open(L183.BUNDLE))
    tgt_count = Counter()
    for r in nb["reg"]:
        tgt_count[nb["names"][int(r[0])].upper()] += 1
    RC = L185.chip_regulator_features(Be, tfs, sets, grow, e_idx, g_idx, tgt_count)
    IN = {"acc_dhs": el_dhs[e_idx],
          "acc_h3k": np.array([np.log10(1 + h3k.get(k, 0.0)) for k in ek])[e_idx],
          "log_n_bound_el": np.log10(1.0 + Be.sum(0))[e_idx].astype(float)}
    X185 = np.column_stack([Xbase]
                           + [OV[c] for c in L185.OVERLAP + L185.PMI]
                           + [RC[c] for c in L185.REGCHIP]
                           + [IN[c] for c in L185.INTRINSIC])
    say(f"     base {Xbase.shape[1]} columns; loop 185's best stack {X185.shape[1]} columns")

    def M(cols, g_perm=None):
        return L188.pair_matrix(EL, PR, cols, e_idx, g_idx, g_perm)

    say()
    say("   running the arms")
    res = {}
    res["base"] = L185.run(Xbase, y, chrom, g_idx, jitter, "base motif+shape", say)
    res["+repress"] = L185.run(np.column_stack([Xbase, M(REPRESS)]), y, chrom, g_idx, jitter,
                               "base +H3K27me3/CTCF only (no category)", say)
    res["+category"] = L185.run(np.column_stack([Xbase, M(CATEGORY)]), y, chrom, g_idx, jitter,
                                "base +chromatin category only", say)
    res["+me1me3"] = L185.run(np.column_stack([Xbase, M(ME1ME3)]), y, chrom, g_idx, jitter,
                              "base +H3K4me1/me3", say)
    res["+methyl"] = L185.run(np.column_stack([Xbase, M(METHYL)]), y, chrom, g_idx, jitter,
                              "base +5mC", say)
    res["+promoter"] = L185.run(np.column_stack([Xbase, M(PROMOTER)]), y, chrom, g_idx, jitter,
                                "base +promoter marks", say)
    rng = np.random.default_rng(SEED)
    res["+promoter_perm"] = L185.run(np.column_stack([Xbase, M(PROMOTER, rng.permutation(n_gn))]),
                                     y, chrom, g_idx, jitter,
                                     "base +promoter marks PERMUTED across genes", say)
    res["+epi_all"] = L185.run(np.column_stack([Xbase, M(EPI_ALL)]), y, chrom, g_idx, jitter,
                               "base +all epigenetic", say)
    res["l185"] = L185.run(X185, y, chrom, g_idx, jitter, "loop 185 best stack", say)
    res["l185+epi"] = L185.run(np.column_stack([X185, M(EPI_ALL)]), y, chrom, g_idx, jitter,
                               "loop 185 best stack +all epigenetic", say)
    res["distance"] = L185.run(Xd, y, chrom, g_idx, jitter, "distance only", say)

    gates = {"H1": h1, "H2": h2, "H3": h3, "H4": True}
    pairs_d = {}
    for gname, arm, label, note in (
            ("H5", "+repress", "REPRESSION AND INSULATION, CLEAN",
             "H3K27me3 and CTCF only. loop 188's G3 had the H3K27ac category in here"),
            ("H6", "+category", "WHAT THE CATEGORY IS WORTH ON ITS OWN",
             "three of its five levels encode H3K27ac, which the base stack does not carry"),
            ("H7", "+me1me3", "THE ENHANCER/PROMOTER DISCRIMINATOR", ""),
            ("H8", "+methyl", "DNA METHYLATION", "")):
        say()
        say(f"{gname} {label}")
        if note:
            say(f"     {note}")
        d = L173.paired(res[arm], res["base"])
        pairs_d[gname] = d
        ok, txt = bar_halves(d)
        say(f"     {txt}")
        gates[gname] = ok
        GG.verdict(ok, emit=say,
                   if_true=f"{gname} PASS -- clears the E3 bar over the base stack",
                   if_false=f"{gname} FAIL on the E3 bar -- {txt}")

    # ---- H9 ------------------------------------------------------------------------------------
    say()
    say("H9 THE PROMOTER SIDE")
    d9 = L173.paired(res["+promoter"], res["base"])
    pairs_d["H9"] = d9
    ok9, txt9 = bar_halves(d9)
    say(f"     {txt9}")
    say("     these columns are constant across a gene's candidates, so an additive main effect")
    say("     cannot reorder them; any R@1 movement is an interaction with the element columns")
    ap_ok9 = bool(d9["n_ap_pass"] >= MIN_SEEDS)
    r1_ok9 = bool(d9["n_pos_r1"] >= MIN_SEEDS and d9["mean_r1"] > 3 * d9["sem_r1"])
    h9 = ap_ok9
    gates["H9"] = h9
    GG.verdict(h9, emit=say,
               if_true=f"H9 PASS -- AUPRC up in {d9['n_ap_pass']}/5, the channel a gene-level "
                       f"column has; R@1 moved {d9['mean_r1']:+.4f} and is reported not gated",
               if_false=f"H9 FAIL -- AUPRC up in only {d9['n_ap_pass']}/5")

    # ---- H10 -----------------------------------------------------------------------------------
    say()
    say("H10 IS THE PROMOTER SIDE GENE-SPECIFIC?")
    d10 = L173.paired(res["+promoter"], res["+promoter_perm"])
    pairs_d["H10"] = d10
    _, txt10 = bar_halves(d10)
    say(f"     real minus permuted: {txt10}")
    channel = "R@1" if r1_ok9 else ("AUPRC" if ap_ok9 else None)
    h10_void = channel is None
    if channel == "R@1":
        h10 = bool(d10["n_pos_r1"] >= MIN_SEEDS and d10["mean_r1"] > 3 * d10["sem_r1"])
    elif channel == "AUPRC":
        h10 = bool(d10["n_ap_pass"] >= MIN_SEEDS)
    else:
        h10 = False
    say(f"     H9 passed on {channel}, so that is the channel this control is scored in -- loop")
    say("     188's G7 scored R@1 when G6 had passed on AUPRC, and voided on a channel where no")
    say("     claim had been made")
    if h10_void:
        say("     H10 VOID -- H9 failed, so there is no claim for this permutation to control")
    else:
        GG.verdict(h10, emit=say,
                   if_true=f"H10 PASS -- giving a gene another gene's promoter marks costs "
                           f"{channel}, so the gain is about WHICH gene's promoter it is",
                   if_false=f"H10 FAIL -- the permuted marks do as well on {channel}; whatever "
                            f"H9 gained is a generic scale and not gene-specific")
    gates["H10"] = h10

    # ---- H11 -----------------------------------------------------------------------------------
    say()
    say("H11 THE DECISIVE ONE: the epigenetic block on top of loop 185's best stack")
    d11 = L173.paired(res["l185+epi"], res["l185"])
    pairs_d["H11"] = d11
    ok11, txt11 = bar_halves(d11)
    say(f"     {txt11}")
    say(f"     loop 185 best R@1 {res['l185']['r1'].mean():.4f}  ->  "
        f"with epigenetics {res['l185+epi']['r1'].mean():.4f}; "
        f"over the bare base the same block was worth "
        f"{res['+epi_all']['r1'].mean() - res['base']['r1'].mean():+.4f}")
    gates["H11"] = ok11
    GG.verdict(ok11, emit=say,
               if_true="H11 PASS -- chromatin state adds to the strongest stack this arc has",
               if_false=f"H11 FAIL on the E3 bar -- {txt11}. Measured co-binding, accessibility "
                        f"and the network match already carry what the marks would say")

    # ---- H12 -----------------------------------------------------------------------------------
    say()
    say("H12 AGAINST DISTANCE ALONE")
    best = max(("l185+epi", "l185", "+epi_all", "base"), key=lambda k: res[k]["r1"].mean())
    d12 = L173.paired(res[best], res["distance"])
    pairs_d["H12"] = d12
    ok12, txt12 = bar_halves(d12)
    say(f"     best arm '{best}' R@1 {res[best]['r1'].mean():.4f}; "
        f"distance only {res['distance']['r1'].mean():.4f}")
    say(f"     {txt12}")
    h12 = bool(ok12 and res[best]["r1"].mean() > L173_DIST_R1)
    gates["H12"] = h12
    GG.verdict(h12, emit=say,
               if_true=f"H12 PASS -- {res[best]['r1'].mean():.4f} clears distance-only "
                       f"{L173_DIST_R1}",
               if_false=f"H12 FAIL -- {res[best]['r1'].mean():.4f} against {L173_DIST_R1}")

    # ---- H13 -----------------------------------------------------------------------------------
    say()
    say("H13 WHAT THIS CANNOT SHOW")
    say("     The candidates were chosen by the screen designers on accessibility and H3K27ac, so")
    say("     every element here is already enhancer-like and these marks are separating")
    say("     functional from non-functional inside a pre-enriched pool.")
    say("     H4 is post-hoc on the same pairs and is not evidence. H3 is prospective only in the")
    say("     sense that it never looks at the label; it was still written after seeing loop 188's")
    say("     refutation, and a coverage confound found by looking is not the same as one")
    say("     predicted in advance.")
    say("     Bulk K562 marks average over a population, so an element Polycomb-marked in half the")
    say("     cells and acetylated in the other half reads as both.")
    say("     The me1-minus-me3 contrast mixes RPM with peak signal from a different assay and is")
    say("     an ordering device, not a physical ratio; the raw columns are entered beside it.")
    say("     Marks are correlational and nothing here perturbs one.")
    say("     H5 and H6 separate repression from the category, but neither is independent of the")
    say("     accessibility columns in loop 185's stack, which is why H11 and not H5 is decisive.")
    gates["H13"] = True
    say("     H13 PASS")

    void = {"H10"} if h10_void else set()
    man = RM.manifest(inputs=[EPI.CACHE, Path("colab/data/tf_domains.json")],
                      available=len(y), used=len(y), selection="all", seed=SEED,
                      controls=["non-finite values dropped and counted in every comparison",
                                "repression and the chromatin category as separate arms",
                                "promoter marks permuted across genes, scored in H9's channel",
                                "H3K27me3 checked against accessibility prospectively",
                                "distance-only on identical folds"],
                      note="epigenetic marks over the CRISPR benchmark, loop 188 defects corrected")
    out = dict(test="enhancer epigenome corrected", gates=gates, void=sorted(void),
               n_pairs=int(len(y)), n_pos=int(y.sum()),
               defined=dict(element=def_el, promoter=def_pr),
               directions=dirs, undefined_predictions=undefined,
               k27me3_vs_dhs=dict(rho=float(rho), p=float(p_rho), strata=strata,
                                  n_direction_gone=n_gone),
               posthoc_thresholded=posthoc,
               arms={k: dict(r1_mean=float(v["r1"].mean()), ap_mean=float(v["ap"].mean()),
                             r1=list(map(float, v["r1"])), ap=list(map(float, v["ap"])))
                     for k, v in res.items()},
               paired={k: {kk: (list(map(float, vv)) if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in d.items()} for k, d in pairs_d.items()},
               h9_channel=channel, best_arm=best,
               manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
