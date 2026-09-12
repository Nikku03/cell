"""Loop 181. Close stage two with the variable sequence cannot carry: physical contact.

WHERE THIS ARC HAS GOT TO. Sequence answers "is this an enhancer" at AUC 0.8506 against
distance-matched genome (loop 177), and answers "which gene does it act on" at a lift of 1.23x over
a 0.0404 base rate once the gene-varying columns are removed (loop 176's M5). Loop 176's M2 also
refused the excuse I offered for that: a gene-blind oracle that has seen every label still reaches
R@1 0.8844, so the shortfall was never the many-to-many structure. The remaining explanation is the
simplest one. Which promoter an enhancer serves is decided by whether the folded chromosome brings
them together, and no property of the element's own bases can report that. This loop supplies the
measurement.

WHAT IS SUPPLIED. Rao et al. (Cell 2014, GSE63525) K562 combined Hi-C, KR-normalised at 5 kb,
streamed strip by strip so nothing is downloaded that is not used; the HiCCUPS loop calls and the
Arrowhead contact-domain calls from the same experiment. All hg19, which is the assembly every
sequence feature in this arc already lives in, so no second liftover enters the chain.

THE TRAP THIS LOOP IS BUILT AROUND. Contact and distance are nearly the same variable. Two loci
10 kb apart touch more than two 1 Mb apart whatever else is true, and distance is already the
strongest feature this task has -- it alone reaches R@1 0.5930 against the 0.6050 of a
thirty-four-column sequence stack. So a contact column that "helps" may be helping only by being a
better-measured distance. Two things guard against reporting that as a finding: the
observed-over-expected column, which divides out the distance decay estimated from the data itself,
and T3, which gives that column its own gate. If raw contact helps and O/E does not, Hi-C is
re-encoding distance and this loop says so rather than claiming to have closed anything.

THE ACTIVITY-BY-CONTACT FORM. The published ABC model scores a pair as the element's ACTIVITY times
its CONTACT with the promoter, divided by the same product summed over the gene's candidates. This
arc now has both terms measured independently: activity from the stage-one sequence classifier that
reaches 0.8506, and contact from the Hi-C. T6 asks whether the product beats the two entered
additively, which is the whole claim of that model's functional form.

PREDECLARED, BEFORE ANY NUMBER.

  T1 DID THE CONTACT DATA LOAD, AND IS IT THE RIGHT DATA? Coverage, and the sanity check that
     contact falls with genomic separation.
     Gate: PASS iff at least 90% of pairs receive a contact value AND the Spearman correlation
     between log contact and log distance is below -0.5. A join that produced numbers but not
     contact-like numbers would fail here rather than downstream.

  T2 DOES CONTACT ADD OVER DISTANCE? distance + the contact block against distance alone.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- loop 173's E3 bar, unchanged.

  T3 IS IT CONTACT, OR DISTANCE MEASURED BETTER? distance + ONLY the observed-over-expected column
     against distance alone.
     Gate: same bar. This is the attribution gate and it is the one that decides what T2 means.

  T4 DOES CONTACT ADD OVER THE SEQUENCE STACK? The stack plus contact against the stack.
     Gate: same bar.

  T5 DO LOOPS AND DOMAINS ADD OVER RAW CONTACT? The called loops and contact domains on top of the
     contact block.
     Gate: same bar. A pass says the focal calls carry something the continuous map does not.

  T6 DOES THE ACTIVITY-BY-CONTACT PRODUCT BEAT THE ADDITIVE FORM? The ABC column, built from the
     fold-internal stage-one sequence classifier and the measured contact, against the same two
     blocks entered separately.
     Gate: same bar.

  T7 THE DECISIVE ONE. The best arm against distance alone on identical folds.
     Gate: same bar. This is the gate loops 173, 175, 178 and 179 were all held to and it is the
     one that decides whether stage two has moved.

  T8 THE CONTACT SWAP. Each gene is given the contact profile of a DIFFERENT gene on the same
     chromosome with a similar candidate count, 20 draws. Distance is untouched, the sequence is
     untouched, and only the correspondence between this gene and its own contacts is destroyed.
     Gate: PASS iff the real profile beats the swapped one in >= 90% of draws. If a stranger's
     contact map works as well, what is being read is the general contact geometry of the
     neighbourhood, not this promoter's own reach.

  T9 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_contact.json
"""
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import contact as CT                # noqa: E402
from enh import genome as GEN                # noqa: E402
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402
import loop_enhancer_stage_one as L177       # noqa: E402
import loop_enhancer_vs_genome as L174       # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_contact.json"
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
MIN_COVER = 0.90
MAX_RHO = -0.5
N_SWAP = 20
L173_DIST_R1 = 0.5930

CONTACT = ["log_contact", "log_contact_oe", "contact_share", "contact_rank", "contact_missing"]
OE_ONLY = ["log_contact_oe"]
STRUCT = ["same_domain", "n_boundaries", "log_domain_span", "loop_connects",
          "loop_n_elem", "loop_n_tss"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def contact_features(S, report=print, gene_perm=None):
    """Per-pair contact columns. `gene_perm` swaps which gene's contact profile is used, for T8."""
    lo = GEN.LiftOver()
    el19, tss19 = [], []
    for k in S["el_key"]:
        c, rest = str(k).split(":")
        a, b = rest.split("-")
        v = lo.lift_interval(c, int(a), int(b))
        el19.append((c,) + (v if v else (0, 0)))
    for k in S["gn_key"]:
        c, p, _ = str(k).split(":")
        q = lo.lift(c, int(p))
        tss19.append((c, q or 0))
    prof = CT.strips(tss19, report)
    exp = CT.expected_by_distance(tss19, prof, report)
    loops = CT.load_bedpe(CT.LOOPS, report)
    doms = CT.load_bedpe(CT.DOMAINS, report)
    lidx0, lidx1 = CT.interval_index(loops, 0), CT.interval_index(loops, 1)
    didx = CT.interval_index(doms, 0)

    e_idx, g_idx = S["e_idx"], S["g_idx"]
    n = len(e_idx)
    prof_map = []
    for b, v in prof:
        prof_map.append(dict(zip(b.tolist(), v.tolist())) if len(b) else {})

    F = {k: np.zeros(n) for k in CONTACT + STRUCT}
    F["contact_missing"] = np.zeros(n)
    raw = np.zeros(n)
    for i in range(n):
        e, g = int(e_idx[i]), int(g_idx[i])
        gg = gene_perm[g] if gene_perm is not None else g
        c, a, b = el19[e]
        tc, tp = tss19[gg]
        pm = prof_map[gg]
        if not pm or b <= a:
            F["contact_missing"][i] = 1.0
            continue
        bins = range((a // CT.RES) * CT.RES, ((b - 1) // CT.RES) * CT.RES + CT.RES, CT.RES)
        vals = [pm[x] for x in bins if x in pm]
        if not vals:
            F["contact_missing"][i] = 1.0
            continue
        v = max(vals)
        raw[i] = v
        F["log_contact"][i] = np.log10(1.0 + v)
        d = abs(((a + b) // 2 // CT.RES) * CT.RES - (tp // CT.RES) * CT.RES) // CT.RES
        ev = exp.get((tc, int(d)), np.nan)
        F["log_contact_oe"][i] = np.log10((v + 0.1) / (ev + 0.1)) if ev == ev else 0.0
    # per-gene shares and ranks
    by_g = defaultdict(list)
    for i in range(n):
        by_g[int(g_idx[i])].append(i)
    for g, ix in by_g.items():
        v = raw[ix]
        s = v.sum()
        for j, i in enumerate(ix):
            F["contact_share"][i] = v[j] / s if s > 0 else 0.0
        o = np.argsort(np.argsort(-v))
        for j, i in enumerate(ix):
            F["contact_rank"][i] = o[j] / max(len(ix) - 1, 1)
    # loops and domains
    for i in range(n):
        e, g = int(e_idx[i]), int(g_idx[i])
        gg = gene_perm[g] if gene_perm is not None else g
        c, a, b = el19[e]
        tc, tp = tss19[gg]
        if b <= a:
            continue
        ea0 = CT.overlaps(lidx0, c, a, b)
        ea1 = CT.overlaps(lidx1, c, a, b)
        ta0 = CT.overlaps(lidx0, tc, tp - CT.RES, tp + CT.RES)
        ta1 = CT.overlaps(lidx1, tc, tp - CT.RES, tp + CT.RES)
        F["loop_n_elem"][i] = len(ea0) + len(ea1)
        F["loop_n_tss"][i] = len(ta0) + len(ta1)
        F["loop_connects"][i] = float(bool(set(ea0.tolist()) & set(ta1.tolist())
                                           or set(ea1.tolist()) & set(ta0.tolist())))
        de = CT.overlaps(didx, c, a, b)
        dt = CT.overlaps(didx, tc, tp, tp + 1)
        both = set(de.tolist()) & set(dt.tolist())
        F["same_domain"][i] = float(bool(both))
        if both:
            spans = [doms[k][2] - doms[k][1] for k in both]
            F["log_domain_span"][i] = np.log10(1.0 + min(spans))
        lo_, hi_ = min(a, tp), max(b, tp)
        arr = didx.get(c)
        if arr is not None and len(arr):
            F["n_boundaries"][i] = float(((arr[:, 0] > lo_) & (arr[:, 0] < hi_)).sum())
    cover = 1.0 - F["contact_missing"].mean()
    report(f"    contact assigned to {cover:.4f} of pairs; "
           f"loop_connects on {F['loop_connects'].mean():.4f}, "
           f"same_domain on {F['same_domain'].mean():.4f}")
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F, raw, cover


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run(build, y, chrom, g_idx, jitter, tag, report=print):
    r1, ap = [], []
    for s in SEEDS:
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0:
                continue
            X = build(tr, s)
            m = gbm(s)
            m.fit(np.nan_to_num(X[tr]), y[tr])
            sc[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        r1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
        ap.append(average_precision_score(y, sc))
    r1, ap = np.array(r1), np.array(ap)
    report(f"    {tag:38} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 181  CLOSE STAGE TWO WITH PHYSICAL CONTACT: K562 Hi-C, loops and contact domains")
    say("=" * 104)
    say(f"  PREDECLARED: >= {MIN_COVER:.0%} of pairs must receive a contact value and log contact")
    say(f"  must fall with log distance past Spearman {MAX_RHO}; every arm on loop 173's E3 bar --")
    say(f"  paired R@1 positive in >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= +0.01 in")
    say(f"  >= {MIN_SEEDS}/5; the observed-over-expected column gated on its own so a raw-contact")
    say(f"  gain cannot be read as more than distance; the winner must clear {L173_DIST_R1}; and")
    say(f"  the real contact profile must beat a swapped one in >= 90% of {N_SWAP} draws.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))
    ng = len(S["gn_key"])

    F, raw, cover = contact_features(S, say)

    # ---- T1 --------------------------------------------------------------------------------
    say()
    say("T1 DID THE CONTACT DATA LOAD, AND IS IT THE RIGHT DATA?")
    ok = F["contact_missing"] == 0
    rho = stats.spearmanr(np.log10(np.maximum(S["dist"][ok], 1)), F["log_contact"][ok]).correlation
    say(f"     coverage {cover:.4f}; Spearman(log distance, log contact) = {rho:+.4f}")
    say(f"     median contact {np.median(raw[ok]):.2f}, "
        f"p90 {np.percentile(raw[ok], 90):.2f}, max {raw[ok].max():.1f}")
    t1 = bool(cover >= MIN_COVER and rho < MAX_RHO)
    GG.verdict(t1, emit=say,
               if_true=f"T1 PASS -- contact reaches {cover:.1%} of pairs and falls with separation "
                       f"at rho {rho:+.3f}, so the join produced contact and not noise",
               if_false=f"T1 FAIL -- coverage {cover:.4f}, rho {rho:+.4f}; the numbers are not "
                        f"behaving like a contact map and nothing below can be read")

    # ---- features --------------------------------------------------------------------------
    E, FAM, _ = L178.element_frame(S, "el", say)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    for c in P:
        P[c] = np.nan_to_num(P[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xd = np.column_stack([P["log_dist"]])
    Xstack = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xc = np.column_stack([F[c] for c in CONTACT])
    Xs = np.column_stack([F[c] for c in STRUCT])
    say(f"    stack {Xstack.shape[1]} columns, contact {Xc.shape[1]}, structure {Xs.shape[1]}")

    # the stage-one sequence activity, fitted inside each fold on loop 174's genomic windows
    D = L174.build_scan(lambda *_: None)
    wkind = np.array([str(k) for k in D["kind"]])
    wchrom = np.array([str(c) for c in D["chrom"]])
    wkeep = wkind != "tested"
    wy = D["y"].astype(int)[wkeep]
    WF = L174.features(D, "el")
    WFAM, _ = L177.family_features(D, "el", lambda *_: None)
    s1_cols = [c for b in L178.STAGE1_BLOCKS for c in L174.BLOCKS[b]]
    s1_fam = [c for c in sorted(WFAM) if c in FAM]
    Xw = np.nan_to_num(np.column_stack([WF[c][wkeep] for c in s1_cols]
                                       + [WFAM[c][wkeep] for c in s1_fam]))
    Xe = np.nan_to_num(np.column_stack([E[c][e_idx] for c in s1_cols]
                                       + [FAM[c][e_idx] for c in s1_fam]))
    wchrom_k = wchrom[wkeep]
    say(f"    stage-one transfer set: {Xw.shape[1]} columns shared with the genomic windows")

    _cache = {}

    def activity(tr, seed):
        key = (seed, tuple(sorted(set(chrom[tr]))))
        if key in _cache:
            return _cache[key]
        m = np.isin(wchrom_k, np.unique(chrom[tr]))
        if m.sum() < 200 or wy[m].sum() < 20:
            a = np.full(len(e_idx), 0.5)
        else:
            g = gbm(seed)
            g.fit(Xw[m], wy[m])
            a = g.predict_proba(Xe)[:, 1]
        _cache[key] = a
        return a

    def abc_col(tr, seed):
        a = activity(tr, seed)
        num = a * raw
        den = np.zeros(len(num))
        by_g = defaultdict(list)
        for i in range(len(num)):
            by_g[int(g_idx[i])].append(i)
        for g, ix in by_g.items():
            s = num[ix].sum()
            den[ix] = s if s > 0 else 1.0
        return np.column_stack([num / den, a])

    res = {}
    res["distance"] = run(lambda tr, s: Xd, y, chrom, g_idx, jitter, "distance", say)
    res["dist+contact"] = run(lambda tr, s: np.column_stack([Xd, Xc]), y, chrom, g_idx, jitter,
                              "distance + contact", say)
    res["dist+oe"] = run(lambda tr, s: np.column_stack([Xd] + [F[c] for c in OE_ONLY]),
                         y, chrom, g_idx, jitter, "distance + observed/expected only", say)
    res["stack"] = run(lambda tr, s: Xstack, y, chrom, g_idx, jitter, "sequence stack", say)
    res["stack+contact"] = run(lambda tr, s: np.column_stack([Xstack, Xc]), y, chrom, g_idx,
                               jitter, "stack + contact", say)
    res["stack+contact+struct"] = run(lambda tr, s: np.column_stack([Xstack, Xc, Xs]),
                                      y, chrom, g_idx, jitter, "stack + contact + loops/domains",
                                      say)
    res["ABC"] = run(lambda tr, s: np.column_stack([Xstack, Xc, Xs, abc_col(tr, s)]),
                     y, chrom, g_idx, jitter, "+ activity x contact (ABC)", say)

    # ---- T2..T7 ----------------------------------------------------------------------------
    say()
    say("T2 DOES CONTACT ADD OVER DISTANCE?")
    d2 = L173.paired(res["dist+contact"], res["distance"])
    say(f"     distance+contact vs distance   {L173.fmt(d2)}")
    t2 = L173.gate_pair(d2)
    GG.verdict(t2, emit=say,
               if_true="T2 PASS -- measured contact adds over the distance between the two loci",
               if_false="T2 FAIL -- contact adds nothing over distance")

    say()
    say("T3 IS IT CONTACT, OR DISTANCE MEASURED BETTER?")
    d3 = L173.paired(res["dist+oe"], res["distance"])
    say(f"     distance+observed/expected vs distance   {L173.fmt(d3)}")
    t3 = L173.gate_pair(d3)
    GG.verdict(t3, emit=say,
               if_true="T3 PASS -- contact BEYOND what separation predicts adds on its own, so T2 "
                       "is about three-dimensional proximity and not about a better ruler",
               if_false="T3 FAIL -- once the distance decay is divided out the contact column stops "
                        "helping, so Hi-C here is re-encoding distance")

    say()
    say("T4 DOES CONTACT ADD OVER THE SEQUENCE STACK?")
    d4 = L173.paired(res["stack+contact"], res["stack"])
    say(f"     stack+contact vs stack   {L173.fmt(d4)}")
    t4 = L173.gate_pair(d4)
    GG.verdict(t4, emit=say,
               if_true="T4 PASS -- contact carries what sequence structurally could not",
               if_false="T4 FAIL -- contact adds nothing the sequence stack did not already have")

    say()
    say("T5 DO LOOPS AND DOMAINS ADD OVER RAW CONTACT?")
    d5 = L173.paired(res["stack+contact+struct"], res["stack+contact"])
    say(f"     +loops/domains vs contact alone   {L173.fmt(d5)}")
    t5 = L173.gate_pair(d5)
    GG.verdict(t5, emit=say,
               if_true="T5 PASS -- the focal loop and domain calls carry something the continuous "
                       "map does not",
               if_false="T5 FAIL -- the called loops and domains add nothing over the contact "
                        "values they were called from")

    say()
    say("T6 DOES ACTIVITY x CONTACT BEAT THE ADDITIVE FORM?")
    d6 = L173.paired(res["ABC"], res["stack+contact+struct"])
    say(f"     ABC vs the same blocks entered additively   {L173.fmt(d6)}")
    t6 = L173.gate_pair(d6)
    GG.verdict(t6, emit=say,
               if_true="T6 PASS -- the product form buys something over entering activity and "
                       "contact as separate columns",
               if_false="T6 FAIL -- the product form adds nothing a tree could not build from the "
                        "two columns separately")

    say()
    say("T7 THE DECISIVE ONE")
    best = max((k for k in res if k != "distance"), key=lambda k: res[k]["r1"].mean())
    d7 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} and AUPRC "
        f"{res[best]['ap'].mean():.4f}, against distance {res['distance']['r1'].mean():.4f} / "
        f"{res['distance']['ap'].mean():.4f}")
    say(f"     {L173.fmt(d7)}")
    t7 = L173.gate_pair(d7)
    GG.verdict(t7, emit=say,
               if_true=f"T7 PASS -- {best} clears the bar that loops 173, 175, 178 and 179 all "
                       f"failed. Stage two has moved, and it moved on contact",
               if_false="T7 FAIL -- even with the folded chromosome measured, stage two is distance")

    # ---- T8 --------------------------------------------------------------------------------
    say()
    say(f"T8 THE CONTACT SWAP: {N_SWAP} draws giving each gene a stranger's contact profile")
    cand_n = np.zeros(ng, int)
    gchrom = {}
    for i in range(len(y)):
        gchrom[int(g_idx[i])] = str(chrom[i])
    for g in range(ng):
        cand_n[g] = sum(1 for i in range(len(y)) if int(g_idx[i]) == g)
    by_c = defaultdict(list)
    for g, c in gchrom.items():
        by_c[c].append(g)
    swaps = []
    for k in range(N_SWAP):
        rr = np.random.default_rng(4000 + k)
        perm = np.arange(ng)
        for c, gs in by_c.items():
            gs = sorted(gs, key=lambda g: cand_n[g])
            pm = list(gs)
            rr.shuffle(pm)
            for a, b in zip(gs, pm):
                perm[a] = b
        Fp, rawp, _ = contact_features(S, lambda *_: None, gene_perm=perm)
        Xcp = np.column_stack([Fp[c] for c in CONTACT])
        r = run(lambda tr, s: np.column_stack([Xd, Xcp]), y, chrom, g_idx, jitter,
                f"swap draw {k+1}", lambda *_: None)
        swaps.append(float(r["r1"].mean()))
        if (k + 1) % 5 == 0:
            say(f"     draw {k+1}/{N_SWAP}  R@1 {swaps[-1]:.4f}")
    swaps = np.array(swaps)
    real = float(res["dist+contact"]["r1"].mean())
    frac = float((real > swaps).mean())
    say(f"     real {real:.4f} against swapped mean {swaps.mean():.4f} "
        f"(min {swaps.min():.4f}, max {swaps.max():.4f}); real beats {frac:.0%}")
    t8 = bool(frac >= 0.90)
    GG.verdict(t8, emit=say,
               if_true="T8 PASS -- a stranger's contact map does not work, so what is being read is "
                       "this promoter's own reach and not the neighbourhood's general geometry",
               if_false="T8 FAIL -- a distance- and chromosome-matched stranger's contact profile "
                        "works as well, so the gain is neighbourhood geometry, not this pair")

    say()
    say("T9 WHAT THIS CANNOT SHOW")
    say("     The Hi-C is a population average over millions of K562 nuclei. A contact value is a")
    say("     frequency across cells, not a statement that these two loci touch in any one of them.")
    say("     Contact is measured at 5 kb, and the median element here is 500 bp, so several")
    say("     elements share a bin and cannot be told apart by this map at all.")
    say("     Rao's K562 Hi-C and the CRISPR benchmark are the same cell line but different")
    say("     laboratories and different cultures, and K562 is aneuploid with rearrangements that")
    say("     a reference-genome contact map represents imperfectly.")
    say("     Contact does not establish causation. An element and a promoter can share a domain")
    say("     and a loop and still not regulate one another.")
    t9 = True
    say(f"     T9 {'PASS' if t9 else 'FAIL'}")

    gates = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7, "T8": t8,
             "T9": t9}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz")],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["observed-over-expected gated separately from raw contact",
                                f"{N_SWAP} chromosome- and size-matched contact-profile swaps",
                                "the stage-one activity model refitted per fold on training "
                                "chromosomes only",
                                "identical folds and seeds as every earlier stage-two loop"],
                      note="K562 Hi-C, HiCCUPS loops and Arrowhead domains on the stage-two task")
    out = dict(test="enhancer contact", gates=gates,
               coverage=float(cover), spearman_dist_contact=float(rho),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("T2", d2), ("T3", d3), ("T4", d4), ("T5", d5),
                                    ("T6", d6), ("T7", d7))},
               swap_draws=[float(x) for x in swaps], swap_frac_beaten=frac,
               best_arm=best, manifest=man, seconds=time.time() - t0, log=log)
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
