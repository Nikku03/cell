"""Loop 188. The epigenetic layer: does chromatin state say which element a gene uses?

WHAT IS NEW HERE, AND WHY IT IS NOT MORE OF THE SAME. Every element-level signal this arc has
tested is ACTIVATING. Loop 173 used motif occupancy and DNA shape; loop 184 ranked co-binding
(0.8455) above accessibility (0.7902) above H3K27ac (0.7510) above motif (0.6228); loop 185's Z6
added DHS and H3K27ac RPM and the bound-factor count. All of those say "something is happening
here". None of them can say "something is being SHUT OFF here", and none of them can say "this
element belongs to a different compartment from that promoter". Three marks in the benchmark have
never been touched, and two of them carry exactly that missing information:

    H3K27me3   Polycomb. A silenced element is not the one a gene is using, and no activating
               mark reports that -- an element can be accessible, acetylated and Polycomb-marked
               at once in a mixed population.
    CTCF       insulation. A CTCF element is a boundary object, and boundaries constrain which
               promoter an element can reach, which is the stage-two question itself.
    H3K4me1    enhancer priming. With H3K4me3 it forms the classical enhancer/promoter
               discriminator (Heintzman et al., Nat Genet 2007): enhancers me1-high me3-low,
               promoters the reverse.

And two more were fetched for this loop and have never been used at all: H3K4me3 over the elements
and the promoters (ENCODE replicated peaks) and CpG methylation over both (ENCODE WGBS). Active
regulatory DNA is hypomethylated, and 5mC is the one mark here that is a property of the DNA rather
than of a histone.

THE STRUCTURAL FACT THAT DECIDES HOW EACH COLUMN CAN HELP, and which loop 185's Z6 got wrong in
one direction and this loop must not get wrong in the other. R@1 here is WITHIN-GENE: for each gene
with at least two tested elements, is the true one ranked first. So:

    an ELEMENT column varies across a gene's candidates and reorders them. Loop 185's Z6
    predeclared that element-intrinsic columns could NOT move R@1, citing loop 178's 0.4422
    leave-one-gene-out ceiling, and was refuted by its own arm: they moved it +0.0693. The error
    was conflating "element-intrinsic" with "constant within a gene". They are not the same thing.

    a PROMOTER column is constant across a gene's candidates. As an ADDITIVE main effect it cannot
    reorder them, so it cannot move R@1 that way, and it can still move pooled AUPRC. But it is not
    inert either: a tree can use it to condition on how the element columns are read -- rank by
    H3K4me1 when the promoter is a CpG island, by H3K27ac otherwise -- and that DOES reorder. So
    G6 gates the promoter arm on AUPRC and REPORTS its R@1 rather than predeclaring a ceiling on
    it. Predeclaring a ceiling is what Z6 did, and Z6 was wrong.

WHAT THIS LOOP CANNOT ESCAPE, stated before any number. The benchmark's candidate elements were
chosen by the screen designers using accessibility and H3K27ac, so the marks are not independent of
the candidate set: every element here is already enhancer-like, and these columns are being asked
to separate functional from non-functional inside a pre-enriched pool, not to find enhancers in the
genome. `elementChromatinCategory` is worse -- it is a discretisation of H3K27ac, CTCF and H3K27me3,
the same columns entered beside it -- so it is included for its compact "No H3K27ac" and "CTCF
element" distinctions and is not treated as independent evidence.

PREDECLARED, BEFORE ANY NUMBER.

  G1 DOES THE MARK LAYER JOIN? Every column defined on both sides, plus the positive rate by
     chromatin category as a description of what the categories are worth.
     Gate: PASS iff every element column is finite for >= 95% of elements and every promoter column
     for >= 90% of promoters. This is a gate on DEFINEDNESS, not on peak overlap -- an element with
     no H3K27me3 peak has a defined value of zero and is not missing data.

  G2 DO THE MARKS POINT THE WAY THEY SHOULD? Three signed predictions, written down before the
     comparison, over the element-gene pairs:
         H3K27me3 LOWER in functional pairs than non-functional
         5mC      LOWER
         H3K4me1  HIGHER
     CTCF and element H3K4me3 are entered UNSIGNED -- reported, not predicted, because an argument
     runs both ways for each.
     Gate: PASS iff all three signed predictions hold at one-sided Mann-Whitney p < 0.05. A model
     that gains from a column running the wrong way is fitting a confound, and this gate is here to
     catch that before any arm is scored.

  G3 THE REPRESSIVE AND INSULATING ARM. H3K27me3, CTCF and the category one-hot over the base
     motif and shape stack.
     Gate: loop 173's E3 bar -- paired per-seed R@1 positive in >= 4/5 past 3 sem AND paired
     AUPRC >= +0.01 in >= 4/5. Unchanged since loop 173.

  G4 THE ENHANCER/PROMOTER DISCRIMINATOR. H3K4me1, element H3K4me3 coverage and signal, and their
     contrast, over the same base.
     Gate: same bar.

  G5 DNA METHYLATION. Element 5mC, CpG count and CpG density, over the same base.
     Gate: same bar.

  G6 THE PROMOTER SIDE. Promoter H3K4me3 and 5mC over the same base.
     Gate: PASS iff the AUPRC half of the bar is cleared. R@1 is reported and interpreted, not
     gated -- see the structural note above for why predeclaring a ceiling here would repeat Z6's
     mistake.

  G7 IS THE PROMOTER SIDE GENE-SPECIFIC? The promoter marks permuted across genes, every value
     kept. If G6 moved R@1 through a real gene-by-element interaction the permutation should
     destroy it; if the marks act as a generic scale it will not.
     Gate: PASS iff real beats permuted on R@1 in >= 4/5 seeds past 3 sem. VOID if G6's arm did not
     move R@1 at all, because then there is no interaction for this control to test and a FAIL
     would be a statement about nothing.

  G8 THE DECISIVE ONE. The whole epigenetic block on top of loop 185's best stack -- base, factor
     overlap, PMI, ChIP regulator match and the Z6 intrinsic columns -- which is the strongest
     thing this arc has built.
     Gate: same E3 bar, against that stack rather than against the bare base. This is the honest
     test: not "does chromatin state help a weak model" but "does it add anything to everything
     already known".

  G9 AGAINST DISTANCE ALONE. The best arm against log distance, identical folds.
     Gate: same bar, and the winner must clear distance-only R@1 0.5930. Every loop in this arc is
     held to this and most have failed it.

  G10 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_epigenome.json
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

from scipy.stats import mannwhitneyu                             # noqa: E402
from sklearn.metrics import average_precision_score              # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_epigenome.json"
SEEDS = L173.SEEDS
MIN_SEEDS = L173.MIN_SEEDS
MIN_DEFINED_EL = 0.95
MIN_DEFINED_PR = 0.90
ALPHA = 0.05
L173_DIST_R1 = 0.5930
SEED = 188188

REPRESSIVE = ["el_k27me3", "el_k27me3_peak", "el_ctcf", "el_ctcf_peak",
              "cat_high_k27ac", "cat_k27ac", "cat_no_k27ac", "cat_ctcf_element",
              "cat_k27me3_element"]
ME1ME3 = ["el_k4me1", "el_k4me1_peak", "el_k4me3_cov", "el_k4me3_sig", "el_me1_minus_me3"]
METHYL = ["el_5mc", "el_ncpg", "el_cpg_dens"]
PROMOTER = ["pr_k4me3_cov", "pr_k4me3_sig", "pr_5mc", "pr_ncpg"]
EPI_ALL = REPRESSIVE + ME1ME3 + METHYL + PROMOTER

# the three predictions written down before the comparison, and the two deliberately left unsigned
SIGNED = {"el_k27me3": "lower", "el_5mc": "lower", "el_k4me1": "higher"}
UNSIGNED = ["el_ctcf", "el_k4me3_sig"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def bench_columns(el_key, report=print):
    """Per-element benchmark marks, keyed the way loop 185 keys them.

    An element that appears in several pairs has one row per pair with identical mark values, so
    the last write wins and the dictionary is one value per element, which is what it should be --
    these are properties of the DNA, not of the pair."""
    rows = SC.load_benchmark(lambda *_: None)
    num = {c: {} for c in ("H3K27me3.RPM", "CTCF.RPM", "H3K4me1.RPM",
                           "H3K27me3_peak_overlap", "CTCF_peak_overlap", "H3K4me1_peak_overlap")}
    cat = {}
    for r in rows:
        k = f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}"
        cat[k] = r.get("elementChromatinCategory") or ""
        for c in num:
            try:
                num[c][k] = float(r.get(c) or 0)
            except ValueError:
                pass
    ek = [str(k) for k in el_key]
    miss = sum(1 for k in ek if k not in cat)
    report(f"     benchmark marks joined for {len(ek)-miss:,}/{len(ek):,} elements")
    return num, cat, ek


def element_frame(el_key, epi, report=print):
    """One row per ELEMENT (not per pair) for every epigenetic column."""
    num, cat, ek = bench_columns(el_key, report)
    n = len(ek)

    def col(c):
        d = num[c]
        return np.array([d.get(k, np.nan) for k in ek], dtype=float)

    F = {}
    F["el_k27me3"] = np.log10(1.0 + col("H3K27me3.RPM"))
    F["el_ctcf"] = np.log10(1.0 + col("CTCF.RPM"))
    F["el_k4me1"] = np.log10(1.0 + col("H3K4me1.RPM"))
    F["el_k27me3_peak"] = col("H3K27me3_peak_overlap")
    F["el_ctcf_peak"] = col("CTCF_peak_overlap")
    F["el_k4me1_peak"] = col("H3K4me1_peak_overlap")
    cats = {"High H3K27ac": "cat_high_k27ac", "H3K27ac": "cat_k27ac", "No H3K27ac": "cat_no_k27ac",
            "CTCF element": "cat_ctcf_element", "H3K27me3 element": "cat_k27me3_element"}
    for lab, name in cats.items():
        F[name] = np.array([1.0 if cat.get(k, "") == lab else 0.0 for k in ek])
    F["el_k4me3_cov"] = np.asarray(epi["el_h3k4me3_cov"], dtype=float)
    F["el_k4me3_sig"] = np.log10(1.0 + np.asarray(epi["el_h3k4me3_sig"], dtype=float))
    # the two terms come from different assays and different units, so this contrast is a monotone
    # ordering device rather than a physical quantity, and the raw columns are entered beside it
    F["el_me1_minus_me3"] = F["el_k4me1"] - F["el_k4me3_sig"]
    F["el_5mc"] = np.asarray(epi["el_5mc"], dtype=float)
    ncpg = np.asarray(epi["el_ncpg"], dtype=float)
    F["el_ncpg"] = np.log10(1.0 + ncpg)
    width = np.array([max(1, int(k.split(":")[1].split("-")[1]) - int(k.split(":")[1].split("-")[0]))
                      for k in ek], dtype=float)
    F["el_cpg_dens"] = 1000.0 * ncpg / width
    assert all(len(v) == n for v in F.values())
    return F


def promoter_frame(epi):
    """One row per GENE. Constant across that gene's candidate elements by construction."""
    return {"pr_k4me3_cov": np.asarray(epi["pr_h3k4me3_cov"], dtype=float),
            "pr_k4me3_sig": np.log10(1.0 + np.asarray(epi["pr_h3k4me3_sig"], dtype=float)),
            "pr_5mc": np.asarray(epi["pr_5mc"], dtype=float),
            "pr_ncpg": np.log10(1.0 + np.asarray(epi["pr_ncpg"], dtype=float))}


def pair_matrix(EL, PR, cols, e_idx, g_idx, g_perm=None):
    """Expand element and promoter frames to one row per pair.

    `g_perm` reassigns which gene's promoter marks a pair receives -- G7's control. Every value is
    kept and only the assignment moves, so the marginal distribution of every promoter column is
    identical between the real and permuted arms."""
    out = []
    for c in cols:
        if c in EL:
            out.append(EL[c][e_idx])
        else:
            gi = g_idx if g_perm is None else g_perm[g_idx]
            out.append(PR[c][gi])
    return np.column_stack(out).astype(np.float64)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 188  THE EPIGENETIC LAYER: does chromatin state say which element a gene uses?")
    say("=" * 104)
    say(f"  PREDECLARED: every element column finite for >= {MIN_DEFINED_EL:.0%} of elements and")
    say(f"  every promoter column for >= {MIN_DEFINED_PR:.0%} of promoters; three SIGNED direction")
    say("  predictions -- H3K27me3 lower, 5mC lower, H3K4me1 higher in functional pairs -- all")
    say(f"  three required at one-sided p < {ALPHA}; every increment arm on loop 173's E3 bar,")
    say(f"  paired R@1 positive in >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= "
        f"+{L173.MIN_INCREMENT} in >= {MIN_SEEDS}/5; the promoter arm gated on AUPRC only, because")
    say("  predeclaring an R@1 ceiling for it would repeat loop 185 Z6's refuted mistake; the")
    say("  promoter marks must not be permutable across genes at no cost; the decisive arm judged")
    say("  on top of loop 185's best stack rather than the bare base; and the winner must clear")
    say(f"  distance-only R@1 {L173_DIST_R1}.")
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
    EL = element_frame(S["el_key"], epi, say)
    PR = promoter_frame(epi)

    # ---- G1 ------------------------------------------------------------------------------------
    say()
    say("G1 DOES THE MARK LAYER JOIN?")
    def_el = {c: float(np.isfinite(EL[c]).mean()) for c in EL}
    def_pr = {c: float(np.isfinite(PR[c]).mean()) for c in PR}
    worst_el = min(def_el, key=def_el.get)
    worst_pr = min(def_pr, key=def_pr.get)
    say(f"     element columns: {len(EL)}, worst defined {worst_el} at {def_el[worst_el]:.1%}")
    say(f"     promoter columns: {len(PR)}, worst defined {worst_pr} at {def_pr[worst_pr]:.1%}")
    say(f"     peak overlap, for description rather than for the gate: "
        f"H3K27me3 {EL['el_k27me3_peak'].mean():.1%}, CTCF {EL['el_ctcf_peak'].mean():.1%}, "
        f"H3K4me1 {EL['el_k4me1_peak'].mean():.1%} of elements; "
        f"H3K4me3 covers {float((EL['el_k4me3_cov'] > 0).mean()):.1%} of elements and "
        f"{float((PR['pr_k4me3_cov'] > 0).mean()):.1%} of promoters")
    say("     positive rate by chromatin category (a discretisation of columns entered beside it):")
    cats = [("cat_high_k27ac", "High H3K27ac"), ("cat_k27ac", "H3K27ac"),
            ("cat_no_k27ac", "No H3K27ac"), ("cat_ctcf_element", "CTCF element"),
            ("cat_k27me3_element", "H3K27me3 element")]
    cat_rate = {}
    for cname, lab in cats:
        m = EL[cname][e_idx] > 0
        if m.sum():
            cat_rate[lab] = dict(n=int(m.sum()), pos=int(y[m].sum()),
                                 rate=float(y[m].mean()))
            say(f"       {lab:20} {int(m.sum()):6,} pairs   {y[m].mean():.2%} functional")
    g1 = bool(def_el[worst_el] >= MIN_DEFINED_EL and def_pr[worst_pr] >= MIN_DEFINED_PR)
    GG.verdict(g1, emit=say,
               if_true=f"G1 PASS -- every column is defined for essentially every element and "
                       f"promoter, so an arm that fails below fails on the mark and not on absence",
               if_false=f"G1 FAIL -- {worst_el} at {def_el[worst_el]:.1%} and {worst_pr} at "
                        f"{def_pr[worst_pr]:.1%}; missingness would be doing the work")

    # ---- G2 ------------------------------------------------------------------------------------
    say()
    say("G2 DO THE MARKS POINT THE WAY THEY SHOULD?")
    pos, neg = y == 1, y == 0
    dirs = {}
    for c, want in SIGNED.items():
        v = EL[c][e_idx]
        a, b = v[pos], v[neg]
        alt = "less" if want == "lower" else "greater"
        u, p = mannwhitneyu(a, b, alternative=alt)
        ok = bool(p < ALPHA)
        dirs[c] = dict(predicted=want, median_pos=float(np.median(a)),
                       median_neg=float(np.median(b)), p=float(p), holds=ok)
        say(f"     {c:12} predicted {want:6}  median functional {np.median(a):+.4f} vs "
            f"{np.median(b):+.4f}   one-sided p {p:.3g}   {'HOLDS' if ok else 'REFUTED'}")
    for c in UNSIGNED:
        v = EL[c][e_idx]
        u, p = mannwhitneyu(v[pos], v[neg], alternative="two-sided")
        say(f"     {c:12} UNSIGNED -- reported only: median functional "
            f"{np.median(v[pos]):+.4f} vs {np.median(v[neg]):+.4f}   two-sided p {p:.3g}")
    g2 = bool(all(d["holds"] for d in dirs.values()))
    GG.verdict(g2, emit=say,
               if_true="G2 PASS -- all three signed predictions hold, so a gain below is a gain "
                       "from marks running the direction the biology says they should",
               if_false="G2 FAIL -- " + ", ".join(c for c, d in dirs.items() if not d["holds"])
                        + " does not run the predicted way; any arm that gains from it is gaining "
                          "from something other than the mechanism the mark was chosen for")

    # ---- the stacks ----------------------------------------------------------------------------
    say()
    say("   building the reference stacks")
    E178, FAM, _ = L178.element_frame(S, "el", lambda *_: None)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    for c in P:
        P[c] = np.nan_to_num(P[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xd = np.column_stack([P["log_dist"]])
    say(f"     base motif+shape stack: {Xbase.shape[1]} columns")

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
        Be, Bp, tfs = Be[[tfs.index(t) for t in common]], Bp[[tfs_p.index(t) for t in common]], common
    OV = L185.overlap_features(Be, Bp, e_idx, g_idx)
    sets, grow, midf, gom = L183.network(S, lambda *_: None)
    nb = json.load(gzip.open(L183.BUNDLE))
    tgt_count = Counter()
    for r in nb["reg"]:
        tgt_count[nb["names"][int(r[0])].upper()] += 1
    RC = L185.chip_regulator_features(Be, tfs, sets, grow, e_idx, g_idx, tgt_count)
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
    IN = {"acc_dhs": np.array([np.log10(1 + dhs.get(ek[int(i)], 0.0)) for i in e_idx]),
          "acc_h3k": np.array([np.log10(1 + h3k.get(ek[int(i)], 0.0)) for i in e_idx]),
          "log_n_bound_el": np.log10(1.0 + Be.sum(0))[e_idx].astype(float)}
    X185 = np.column_stack([Xbase]
                           + [OV[c] for c in L185.OVERLAP + L185.PMI]
                           + [RC[c] for c in L185.REGCHIP]
                           + [IN[c] for c in L185.INTRINSIC])
    say(f"     loop 185's best stack: {X185.shape[1]} columns")

    def M(cols, g_perm=None):
        return pair_matrix(EL, PR, cols, e_idx, g_idx, g_perm)

    say()
    say("   running the arms")
    res = {}
    res["base"] = L185.run(Xbase, y, chrom, g_idx, jitter, "base motif+shape", say)
    res["+repressive"] = L185.run(np.column_stack([Xbase, M(REPRESSIVE)]), y, chrom, g_idx, jitter,
                                  "base +repressive (K27me3, CTCF, category)", say)
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

    # ---- G3, G4, G5 -----------------------------------------------------------------------------
    gates = {"G1": g1, "G2": g2}
    pairs_d = {}
    for gname, arm, label in (("G3", "+repressive", "THE REPRESSIVE AND INSULATING ARM"),
                              ("G4", "+me1me3", "THE ENHANCER/PROMOTER DISCRIMINATOR"),
                              ("G5", "+methyl", "DNA METHYLATION")):
        say()
        say(f"{gname} {label}")
        d = L173.paired(res[arm], res["base"])
        pairs_d[gname] = d
        say(f"     {L173.fmt(d)}")
        ok = L173.gate_pair(d)
        gates[gname] = ok
        GG.verdict(ok, emit=say,
                   if_true=f"{gname} PASS -- the block clears the E3 bar over the base stack",
                   if_false=f"{gname} FAIL -- the block does not clear the E3 bar; on this "
                            f"benchmark it adds nothing the motif and shape columns did not have")

    # ---- G6 ------------------------------------------------------------------------------------
    say()
    say("G6 THE PROMOTER SIDE")
    d6 = L173.paired(res["+promoter"], res["base"])
    pairs_d["G6"] = d6
    say(f"     {L173.fmt(d6)}")
    say("     the promoter columns are constant across a gene's candidates, so an additive main")
    say("     effect cannot reorder them; any R@1 movement here is an interaction with the element")
    say("     columns, and G7 tests whether that interaction is gene-specific")
    g6 = bool(d6["n_ap_pass"] >= MIN_SEEDS)
    GG.verdict(g6, emit=say,
               if_true=f"G6 PASS -- AUPRC up in {d6['n_ap_pass']}/5 seeds, which is the channel a "
                       f"gene-level column has; R@1 moved {d6['mean_r1']:+.4f} and is reported "
                       f"rather than gated",
               if_false=f"G6 FAIL -- AUPRC up in only {d6['n_ap_pass']}/5; the promoter marks do "
                        f"not separate functional from non-functional pairs even globally")

    # ---- G7 ------------------------------------------------------------------------------------
    say()
    say("G7 IS THE PROMOTER SIDE GENE-SPECIFIC?")
    d7 = L173.paired(res["+promoter"], res["+promoter_perm"])
    pairs_d["G7"] = d7
    say(f"     real minus permuted: {L173.fmt(d7)}")
    moved = bool(d6["mean_r1"] > 3 * d6["sem_r1"] and d6["n_pos_r1"] >= MIN_SEEDS)
    g7_void = not moved
    g7 = bool(d7["n_pos_r1"] >= MIN_SEEDS and d7["mean_r1"] > 3 * d7["sem_r1"])
    if g7_void:
        say(f"     G7 VOID -- G6's arm did not move R@1 ({d6['mean_r1']:+.4f} +/- "
            f"{d6['sem_r1']:.4f}, {d6['n_pos_r1']}/5 up), so there is no gene-by-element "
            f"interaction for this permutation to destroy and a verdict either way would be a "
            f"statement about nothing")
    else:
        GG.verdict(g7, emit=say,
                   if_true="G7 PASS -- giving a gene another gene's promoter marks costs R@1, so "
                           "the interaction is gene-specific and not a generic scale",
                   if_false="G7 FAIL -- the permuted marks do as well, so whatever G6 gained is "
                            "not about WHICH gene's promoter it is")
    gates["G6"], gates["G7"] = g6, g7

    # ---- G8 ------------------------------------------------------------------------------------
    say()
    say("G8 THE DECISIVE ONE: the epigenetic block on top of loop 185's best stack")
    d8 = L173.paired(res["l185+epi"], res["l185"])
    pairs_d["G8"] = d8
    say(f"     {L173.fmt(d8)}")
    say(f"     loop 185 best R@1 {res['l185']['r1'].mean():.4f}  ->  "
        f"with epigenetics {res['l185+epi']['r1'].mean():.4f}")
    g8 = L173.gate_pair(d8)
    gates["G8"] = g8
    GG.verdict(g8, emit=say,
               if_true="G8 PASS -- chromatin state adds to the strongest stack this arc has, "
                       "which is a harder thing to do than adding to the bare base",
               if_false="G8 FAIL -- the epigenetic block adds nothing on top of co-binding, "
                        "accessibility and the network match. Those columns already carry whatever "
                        "the marks were going to say")

    # ---- G9 ------------------------------------------------------------------------------------
    say()
    say("G9 AGAINST DISTANCE ALONE")
    best = max(("l185+epi", "l185", "+epi_all", "base"), key=lambda k: res[k]["r1"].mean())
    d9 = L173.paired(res[best], res["distance"])
    pairs_d["G9"] = d9
    say(f"     best arm is '{best}' at R@1 {res[best]['r1'].mean():.4f}; "
        f"distance only {res['distance']['r1'].mean():.4f}")
    say(f"     {L173.fmt(d9)}")
    g9 = bool(L173.gate_pair(d9) and res[best]["r1"].mean() > L173_DIST_R1)
    gates["G9"] = g9
    GG.verdict(g9, emit=say,
               if_true=f"G9 PASS -- {res[best]['r1'].mean():.4f} clears distance-only "
                       f"{L173_DIST_R1} on the bar every loop in this arc is held to",
               if_false=f"G9 FAIL -- {res[best]['r1'].mean():.4f} against the {L173_DIST_R1} floor")

    # ---- G10 -----------------------------------------------------------------------------------
    say()
    say("G10 WHAT THIS CANNOT SHOW")
    say("     The candidate elements were chosen by the screen designers using accessibility and")
    say("     H3K27ac, so every element here is already enhancer-like. These marks are separating")
    say("     functional from non-functional inside a pre-enriched pool. Nothing here says how they")
    say("     would perform at finding enhancers in unselected genome.")
    say("     elementChromatinCategory is a discretisation of H3K27ac, CTCF and H3K27me3, which are")
    say("     entered as columns beside it. It is compact, not independent, and a gain that came")
    say("     only from it would be a gain from re-binning columns already present.")
    say("     Bulk K562 marks average over a population. An element that is Polycomb-marked in half")
    say("     the cells and acetylated in the other half reads as both, and no bulk assay can")
    say("     separate that from an element carrying both marks in every cell.")
    say("     H3K4me1 and H3K27me3 come from the benchmark as RPM; H3K4me3 comes from ENCODE")
    say("     replicated peaks as coverage and signal. The me1-minus-me3 contrast therefore mixes")
    say("     units and is an ordering device, not a physical ratio. The raw columns are entered")
    say("     beside it so nothing depends on that contrast alone.")
    say("     Marks are correlational. A mark that predicts function is not thereby a cause of it,")
    say("     and this loop has no perturbation of any mark to distinguish the two.")
    g10 = True
    say(f"     G10 {'PASS' if g10 else 'FAIL'}")
    gates["G10"] = g10

    void = {"G7"} if g7_void else set()
    man = RM.manifest(inputs=[EPI.CACHE, Path("colab/data/tf_domains.json")],
                      available=len(y), used=len(y), selection="all",
                      seed=SEED,
                      controls=["promoter marks permuted across genes, every value kept",
                                "three signed direction predictions made before the comparison",
                                "the decisive arm judged over loop 185's best stack, not the base",
                                "distance-only on identical folds"],
                      note="epigenetic marks over the CRISPR element-gene benchmark")
    out = dict(test="enhancer epigenome", gates=gates, void=sorted(void),
               n_pairs=int(len(y)), n_pos=int(y.sum()), n_el=int(n_el), n_gn=int(n_gn),
               defined=dict(element=def_el, promoter=def_pr),
               category_rate=cat_rate, directions=dirs,
               arms={k: dict(r1=list(map(float, v["r1"])), ap=list(map(float, v["ap"])),
                             r1_mean=float(v["r1"].mean()), ap_mean=float(v["ap"].mean()))
                     for k, v in res.items()},
               paired={k: {kk: (list(map(float, vv)) if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in d.items()} for k, d in pairs_d.items()},
               best_arm=best, manifest=man, seconds=time.time() - t0, log=log)
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
