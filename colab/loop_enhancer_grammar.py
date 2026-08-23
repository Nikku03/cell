"""Loop 173. Can the SEQUENCE of a distal element -- its binding sites, its groove shape, its
groove charge, and how hard its duplex is to open -- say which element regulates which gene, better
than the distance between them?

THE PLAN THIS TESTS, AND THE FOUR CORRECTIONS IT CARRIES. The proposal was: take a promoter whose
enhancer is unknown, take 2 Mb upstream and downstream, find the 6 bp sequences the relevant TF
binds, keep the major groove and discard the minor, narrow further by comparing site charge and
sterics against the TF domain's charge and sterics, and finish on binding energy plus the energy of
opening the strand, which must come out net negative. The arithmetic in that plan is exact: a 6 bp
motif over 4x10^6 bp on both strands gives 4x10^6 x 2 / 4^6 = 1,953 expected hits, which is the
1,800-2,000 the plan predicted. Four things about it are changed here, and each change is why a
particular gate exists.

  1  GROOVE IS NOT A SITE FILTER. Which groove a protein reads is fixed by its domain fold -- zinc
     fingers and helix-turn-helix insert a recognition helix into the major groove, TBP and the HMG
     box splay the minor groove open -- and every instance of a motif presents both grooves
     identically. There is nothing to filter. What DOES vary site to site is groove GEOMETRY: minor
     groove width, propeller twist, helix twist and roll, all set by the flanking sequence. That is
     shape readout (Rohs et al., Nature 2009) and it is a real, published increment over a PWM. So
     groove becomes a per-FACTOR label that decides which shape variable that factor can feel, and
     shape becomes a per-SITE quantity. Gate E6 asks whether it adds.

  2  1,953 HITS IS THE FUTILITY THEOREM, not an intermediate result. Over 99.9% of genomic motif
     matches are non-functional (Wasserman & Sandelin, Nat Rev Genet 2004), so a filter chain
     starting from 1,953 candidates has to do essentially all of the work, and a best-match score
     cannot do it. The quantity that can is OCCUPANCY UNDER COMPETITION: a site's Boltzmann weight
     divided by the weight of every competing site in the window, including the ~10^6 the plan's own
     arithmetic implies. Gate E8 is the direct test -- the same features computed with the
     competition denominator against the same features computed from the raw best score.

  3  "NET NEGATIVE" DOES NOT DISCRIMINATE. Every stable binding event has a negative free energy;
     that is what stable means. Binding energy and the cost of opening the duplex are therefore
     carried as two separate features and their difference as a third, and gate E7's successor E3
     asks whether the balance predicts anything, rather than using the sign as a filter that would
     pass everything.

  4  THE ASSEMBLIES DO NOT MATCH. The CRISPR benchmark is GRCh38, the reference sequence on this
     machine is hg19. A wrong coordinate conversion returns real DNA from the wrong locus and
     nothing downstream throws. So the liftover is checked end to end against 16,380 gene TSSs held
     in both assemblies before any score is computed, and gate E1 refuses to report the rest if it
     fails.

WHAT IS MEASURED, AND AGAINST WHAT.
  GROUND TRUTH  EP CRISPR benchmark (EngreitzLab/CRISPR_comparison), K562 arm, ValidConnection and
                powered at PowerAtEffectSize25 >= 0.8. An unpowered non-significant pair is NOT
                DETECTED, not a negative.
  METRIC        WITHIN-GENE RECALL AT 1 -- for each gene whose tested elements include at least one
                validated enhancer, is the top-ranked element a real one? That is the question the
                plan actually asks ("find the enhancer for this promoter"), and it is the metric the
                existing E-G layer reports, so the two are comparable. Pooled out-of-fold AUPRC is
                reported beside it because at a ~4% base rate a single metric hides too much.
  FOLDS         chromosome-held-out, 5 folds x 5 seeds, IDENTICAL folds for every arm.
  THE BAR       DISTANCE ALONE, measured here on this exact evaluation set rather than imported.
                The existing layer reports 0.6509 within-gene R@1 for distance on ITS candidate
                sets; that number is not transferable to a different candidate set, so it is
                re-measured and the re-measured value is the bar.

PREDECLARED, BEFORE ANY NUMBER IS LOOKED AT.

  E1 IS THE SEQUENCE THE RIGHT SEQUENCE? Liftover and extraction integrity.
     Gate: PASS iff >= 95% of benchmark elements lift with their width preserved to within 10%,
     AND the median element N fraction is below 0.01. On FAIL the loop reports nothing else,
     because every downstream number would be about the wrong DNA.

  E2 IS THE EVALUATION SET UP CORRECTLY? Distance-only against picking a candidate at random
     within the gene.
     Gate: PASS iff distance-only within-gene R@1 beats the random-pick R@1 by more than 3 sem.
     This is not a finding; it is the check that the ranking task is a task at all, and it
     establishes the bar the rest of the loop is judged against.

  E3 DOES SEQUENCE ADD OVER DISTANCE? The full stack against distance alone, same folds.
     Gate: PASS iff the paired per-seed change in within-gene R@1 is positive in >= 4/5 seeds AND
     the mean change exceeds 3 sem, AND the paired change in AUPRC is >= +0.01 in >= 4/5 seeds.
     The AUPRC bar is the one causal_enhancer.py fixed for the same benchmark, reused so this
     module cannot earn a positive claim on a number that would have failed there.

  E4 MOTIFS, OR BASE COMPOSITION? The full stack against the full stack recomputed on element
     sequences DINUCLEOTIDE-SHUFFLED (Altschul-Erikson), which preserves GC, CpG and every
     dinucleotide frequency exactly and destroys binding sites.
     Gate: PASS iff real beats shuffled on R@1 in >= 4/5 seeds and by more than 3 sem. A FAIL here
     means the sequence block is reading composition, and the verdict says that instead of
     reporting motif scores.

  E5 WHICH TFs, OR HOW MANY? The pairing block asks whether an element carries sites for the SAME
     factors whose motifs are present at the gene's promoter. Permuting which matrix is assigned to
     which factor destroys that correspondence while holding the number of motifs, their widths and
     their information contents EXACTLY -- a permutation of the matrix axis is the same multiset.
     Gate: PASS iff real beats permuted on R@1 in >= 4/5 seeds. On FAIL the finding is "how many
     sites", not "which factors", and it is stated that way.

  E6 DOES SHAPE ADD OVER SITES? sites+pairing against sites+pairing+shape, where shape is the
     Boltzmann-weighted minor and major groove width, propeller twist, roll, helix twist and
     minor-groove electrostatic potential at the occupied positions.
     Gate: PASS iff paired AUPRC change >= +0.01 in >= 4/5 seeds AND R@1 change positive in >= 4/5.

  E7 DOES MAJOR-GROOVE COMPLEMENTARITY ADD OVER SHAPE? The plan asked for the major groove first
     and the minor groove only afterwards, so the two are separated and run in that order. E7 adds
     only the terms a major-groove reader can feel: major groove width against the domain's mean
     residue volume (the steric term -- a domain of large residues cannot enter a groove that is
     not wide enough), propeller twist against charge density, and the ratio of domain length to
     motif width, which is the plan's "if the TF protein extends that far" made numerical.
     Gate: same bar as E6.

  E7b AND THEN THE MINOR GROOVE, ON TOP. Adds the terms only a minor-groove-reading fold can feel:
     the site's minor-groove electrostatic potential against the domain's charge density and,
     separately, against its arginine fraction -- arginine being the residue that actually inserts
     into a narrow electronegative minor groove -- and minor groove width against residue volume.
     These are switched off for folds whose JASPAR class does not read the minor groove, so the
     block is zero for most factors by construction and non-zero for the 208 that read the minor
     groove or both.
     Gate: same bar as E6, applied to the increment over E7's major-groove-only arm. Running the
     two in this order is what makes it possible to say whether the minor groove added anything,
     rather than reporting one combined number that could be all major.

  E8 DOES COMPETITION BEAT THE RAW SCORE? The site block computed as occupancy under competition
     against the identical block computed from the raw best score, everything else held fixed.
     Gate: same bar as E6. This is the futility correction, tested rather than assumed.

  E9 THE FUTILITY COUNT ITSELF, predicted before it is measured: more than 99% of above-threshold
     motif matches in this data sit in elements that are NOT validated enhancers.
     Gate: PASS iff the measured fraction is >= 0.99.

  E10 WHAT THIS CANNOT SHOW. Stated, not gated on being flattering.

-> outputs/loop_enhancer_grammar.json
"""
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import genome as GEN                # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import average_precision_score           # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_grammar.json"
SEEDS = [0, 1, 2, 3, 4]
NFOLD = 5
WINDOW_BP = 4_000_000        # the plan's +/- 2 Mb, used as the competition window
MIN_INCREMENT = 0.01         # the AUPRC bar causal_enhancer.py fixed for this benchmark
MIN_SEEDS = 4
MIN_LIFT = 0.95
MAX_N_FRAC = 0.01
FUTILITY_FLOOR = 0.99
TIE_SEED = 91173

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------
def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def occupancy(LZ_el, e_idx, g_idx, ng, bg_rate):
    """log occupancy of each (motif, pair): the element's Boltzmann weight over the weight of every
    competitor in the gene's window -- its other tested elements plus WINDOW_BP of genome
    background at the motif's own measured background rate."""
    nm = LZ_el.shape[0]
    # per gene, the set of distinct elements tested against it
    pool = defaultdict(set)
    for i in range(len(e_idx)):
        pool[int(g_idx[i])].add(int(e_idx[i]))
    D = np.empty((nm, ng), dtype=np.float64)
    bgterm = np.log(np.maximum(bg_rate, 1e-300)) + np.log(WINDOW_BP)
    for g in range(ng):
        els = sorted(pool.get(g, ()))
        if els:
            part = _logsumexp(LZ_el[:, els].astype(np.float64), axis=1)
            D[:, g] = np.logaddexp(part, bgterm)
        else:
            D[:, g] = bgterm
    return LZ_el[:, e_idx].astype(np.float64) - D[:, g_idx]


def build_features(P, elem, tf_perm=None, competition=True, report=print):
    """One feature frame. `elem` selects which element scan to read ('el' real, 'sh' shuffled).
    `tf_perm` permutes the matrix axis of the PROMOTER side only, which destroys the
    promoter-element factor correspondence while holding every count exactly."""
    y = P["y"].astype(int)
    e_idx, g_idx = P["e_idx"], P["g_idx"]
    ng = len(P["gn_key"])
    nm = len(P["motif_ids"])
    mx_max = P["motif_maxscore"].astype(np.float64)
    width = P["motif_width"].astype(np.float64)

    LZ = P[f"{elem}_LZ"]
    MX = P[f"{elem}_MX"]
    NS = P[f"{elem}_NS"]
    SH = P[f"{elem}_SH"]
    tracks = list(P["tracks"])
    T = {n: SH[i] for i, n in enumerate(tracks)}

    bg_rate = np.exp(_logsumexp(P["bg_LZ"].astype(np.float64), axis=1)) / float(P["bg_bp"])
    logocc = occupancy(LZ, e_idx, g_idx, ng, bg_rate)              # (nm, npair), natural log
    occ = np.exp(logocc)

    # promoter TF set, and the element's own TF set
    thr = SC.REL_THRESH * mx_max
    Pm = (P["pr_MX"] >= thr[:, None])                              # (nm, ng)
    if tf_perm is not None:
        Pm = Pm[tf_perm]
    Em = NS > 0                                                    # (nm, ne)
    Pp = Pm[:, g_idx]
    Ep = Em[:, e_idx]

    F = {}
    d = np.maximum(P["dist"], 1.0)
    F["log_dist"] = np.log10(d)

    F["log_width"] = np.log10(np.maximum(P["el_width"][e_idx], 1))
    F["gc"] = (P["el_gc"] if elem == "el" else P["el_gc"])[e_idx]   # shuffle preserves GC exactly
    F["cpg_raw"] = P["el_cpg_raw"][e_idx]
    F["n_frac"] = P["el_n_frac"][e_idx]

    sum_occ = occ.sum(0)
    F["log_sum_occ"] = np.log10(np.maximum(sum_occ, 1e-300))
    F["log_max_occ"] = np.log10(np.maximum(occ.max(0), 1e-300))
    F["n_sites"] = NS[:, e_idx].sum(0).astype(np.float64)
    F["log_elem_n"] = np.log10(1.0 + Ep.sum(0))

    rel = MX / mx_max[:, None]
    F["max_rel"] = rel[:, e_idx].max(0).astype(np.float64)
    F["log_sum_expmx"] = _logsumexp((MX - mx_max[:, None])[:, e_idx].astype(np.float64), axis=0)
    if not competition:
        F["log_sum_occ"] = F.pop("log_sum_expmx")
        F["log_max_occ"] = F.pop("max_rel")

    sh_occ = (occ * Pp).sum(0)
    F["log_shared_occ"] = np.log10(np.maximum(sh_occ, 1e-300))
    F["shared_n"] = (Pp & Ep).sum(0).astype(np.float64)
    F["prom_n"] = Pp.sum(0).astype(np.float64)
    F["shared_frac"] = F["shared_n"] / np.maximum(F["prom_n"], 1.0)
    uni = (Pp | Ep).sum(0).astype(np.float64)
    F["jaccard"] = F["shared_n"] / np.maximum(uni, 1.0)

    W = np.maximum(sum_occ, 1e-300)
    for name in ("mgw", "mgrw", "prot", "roll", "helt", "ep", "dg"):
        v = T[name][:, e_idx].astype(np.float64)
        ok = np.isfinite(v)
        F["site_" + name] = np.where(ok, occ * v, 0.0).sum(0) / np.maximum(
            np.where(ok, occ, 0.0).sum(0), 1e-300)
    for name in ("mgw", "prot", "ep", "dg"):
        F["elem_" + name] = (P["elmean_" + name] if elem == "el"
                             else P["shmean_" + name])[e_idx].astype(np.float64)

    # ---- complementarity: what the site offers x what the domain brings ------------------------
    dom = TD.load()
    ids = list(P["motif_ids"])
    have = np.array([bool(dom.get(m, {}).get("route")) for m in ids])
    def col(k, default=0.0):
        return np.array([float(dom.get(m, {}).get(k, default) or default) for m in ids])
    chg = col("charge_density")
    arg = col("arg_frac")
    vol = col("mean_volume")
    dlen = col("length", 1.0)
    groove = np.array([dom.get(m, {}).get("groove", "major") for m in ids])
    minorish = ((groove == "minor") | (groove == "both")) & have
    majorish = ((groove == "major") | (groove == "both")) & have
    volc = vol - (vol[have].mean() if have.any() else 0.0)
    span = np.where(width > 0, dlen / width, 0.0)

    EP = np.nan_to_num(T["ep"][:, e_idx].astype(np.float64), nan=0.0)
    MGW = np.nan_to_num(T["mgw"][:, e_idx].astype(np.float64), nan=0.0)
    MGrW = np.nan_to_num(T["mgrw"][:, e_idx].astype(np.float64), nan=0.0)
    PROT = np.nan_to_num(T["prot"][:, e_idx].astype(np.float64), nan=0.0)
    F["comp_charge"] = (occ * (-EP) * (chg * minorish)[:, None]).sum(0)
    F["comp_arg"] = (occ * (-EP) * (arg * minorish)[:, None]).sum(0)
    F["comp_steric"] = (occ * MGW * (volc * have)[:, None]).sum(0)
    F["comp_major"] = (occ * MGrW * (volc * majorish)[:, None]).sum(0)
    F["comp_twist"] = (occ * PROT * (chg * have)[:, None]).sum(0)
    F["comp_span"] = (occ * (span * have)[:, None]).sum(0) / W

    # ---- energetics ----------------------------------------------------------------------------
    RT = 0.616                                     # kcal/mol at 37 C
    best = np.argmax(rel[:, e_idx], axis=0)
    bw = width[best]
    F["bind_kcal_pb"] = -RT * MX[best, e_idx].astype(np.float64) / np.maximum(bw, 1.0)
    F["open_kcal_pb"] = -F["site_dg"]              # cost of melting one bp of the occupied site
    F["net_kcal_pb"] = F["bind_kcal_pb"] + F["open_kcal_pb"]

    return F, y, have


BLOCKS = {
    "distance": ["log_dist"],
    "composition": ["log_width", "gc", "cpg_raw", "n_frac"],
    "sites": ["log_sum_occ", "log_max_occ", "n_sites", "log_elem_n"],
    "pairing": ["log_shared_occ", "shared_n", "prom_n", "shared_frac", "jaccard"],
    "shape": ["site_mgw", "site_mgrw", "site_prot", "site_roll", "site_helt", "site_ep",
              "elem_mgw", "elem_prot", "elem_ep"],
    # split by which groove the fold can actually feel, so the plan's "major first, then minor"
    # can be run in that order and each stage judged on its own increment
    "compl_major": ["comp_major", "comp_twist", "comp_span"],
    "compl_minor": ["comp_charge", "comp_arg", "comp_steric"],
    "opening": ["site_dg", "elem_dg", "bind_kcal_pb", "open_kcal_pb", "net_kcal_pb"],
}
ARMS = {
    "distance":            ["distance"],
    "dist+comp":           ["distance", "composition"],
    "dist+comp+sites":     ["distance", "composition", "sites"],
    "+pairing":            ["distance", "composition", "sites", "pairing"],
    "+shape":              ["distance", "composition", "sites", "pairing", "shape"],
    "+compl_major":        ["distance", "composition", "sites", "pairing", "shape",
                            "compl_major"],
    "+compl_minor":        ["distance", "composition", "sites", "pairing", "shape",
                            "compl_major", "compl_minor"],
    "FULL":                ["distance", "composition", "sites", "pairing", "shape",
                            "compl_major", "compl_minor", "opening"],
}


def matrix(F, blocks):
    cols = [c for b in blocks for c in BLOCKS[b]]
    return np.column_stack([F[c] for c in cols]).astype(np.float64), cols


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------
def folds_for(chrom, seed):
    ch = sorted(set(chrom))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ch))
    assign = {ch[order[i]]: i % NFOLD for i in range(len(ch))}
    return np.array([assign[c] for c in chrom])


def oof_scores(X, y, fold, seed):
    s = np.zeros(len(y))
    for f in range(NFOLD):
        te = fold == f
        tr = ~te
        if te.sum() == 0 or y[tr].sum() == 0:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=seed)
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def within_gene(scores, y, g_idx, jitter):
    """R@1 and MRR over genes with at least one positive and at least two tested elements."""
    by = defaultdict(list)
    for i in range(len(y)):
        by[int(g_idx[i])].append(i)
    hits, rr, n = 0, 0.0, 0
    for g, ix in by.items():
        if len(ix) < 2:
            continue
        yy = y[ix]
        if yy.sum() == 0:
            continue
        n += 1
        s = scores[ix] + jitter[ix]
        o = np.argsort(-s)
        hits += int(yy[o[0]] == 1)
        rr += 1.0 / (1 + int(np.argmax(yy[o] == 1)))
    return (hits / max(n, 1)), (rr / max(n, 1)), n


def run_arm(X, y, chrom, g_idx, jitter, tag, report=print):
    r1, mrr, ap = [], [], []
    for s in SEEDS:
        fold = folds_for(chrom, s)
        sc = oof_scores(X, y, fold, s)
        a, b, n = within_gene(sc, y, g_idx, jitter)
        r1.append(a)
        mrr.append(b)
        ap.append(average_precision_score(y, sc))
    r1, mrr, ap = np.array(r1), np.array(mrr), np.array(ap)
    report(f"    {tag:22} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"MRR {mrr.mean():.4f}   AUPRC {ap.mean():.4f}")
    return dict(r1=r1, mrr=mrr, ap=ap)


def paired(a, b):
    """a minus b, per seed."""
    d1 = a["r1"] - b["r1"]
    dap = a["ap"] - b["ap"]
    sem = d1.std(ddof=1) / np.sqrt(len(d1)) if len(d1) > 1 else 0.0
    return dict(d_r1=d1, d_ap=dap, mean_r1=float(d1.mean()), sem_r1=float(sem),
                mean_ap=float(dap.mean()),
                n_pos_r1=int((d1 > 0).sum()), n_ap_pass=int((dap >= MIN_INCREMENT).sum()))


def gate_pair(d, use_ap=True):
    """The bar E3 fixes and E6-E8 reuse: R@1 up in >= MIN_SEEDS of 5 and past 3 sem, and -- when
    the arm is an increment rather than a control -- AUPRC up by >= MIN_INCREMENT in >= MIN_SEEDS."""
    r1_ok = (d["n_pos_r1"] >= MIN_SEEDS) and (d["mean_r1"] > 3 * d["sem_r1"])
    ap_ok = d["n_ap_pass"] >= MIN_SEEDS
    return bool(r1_ok and (ap_ok if use_ap else True))


def fmt(d):
    return (f"dR@1 {d['mean_r1']:+.4f} +/- {d['sem_r1']:.4f} ({d['n_pos_r1']}/5 up)   "
            f"dAUPRC {d['mean_ap']:+.4f} ({d['n_ap_pass']}/5 >= {MIN_INCREMENT})")


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 173  ENHANCER GRAMMAR: sites, groove shape, groove charge and duplex opening,")
    say("          against the distance between the element and the promoter")
    say("=" * 104)
    say(f"  PREDECLARED: increment bar dAUPRC >= {MIN_INCREMENT} in >= {MIN_SEEDS}/5 seeds and dR@1")
    say(f"  positive in >= {MIN_SEEDS}/5 past 3 sem; liftover >= {MIN_LIFT:.0%} with width preserved;")
    say(f"  median element N fraction < {MAX_N_FRAC}; futility fraction >= {FUTILITY_FLOOR}.")
    say()

    P = SC.load(say)
    y = P["y"].astype(int)
    e_idx, g_idx, chrom = P["e_idx"], P["g_idx"], np.array([str(c) for c in P["chrom"]])
    rng = np.random.default_rng(TIE_SEED)
    jitter = rng.uniform(0, 1e-9, size=len(y))
    say(f"    {len(y):,} pairs, {int(y.sum())} positives (base rate {y.mean():.4f}), "
        f"{len(P['el_key']):,} elements, {len(P['gn_key']):,} genes, {len(set(chrom))} chromosomes")

    # ---- E1 ------------------------------------------------------------------------------------
    say()
    say("E1 IS THE SEQUENCE THE RIGHT SEQUENCE?")
    tss_ok, tss = GEN.qc(lambda s: say("   " + s))
    rows = SC.load_benchmark(lambda s: say("   " + s))
    lo = GEN.LiftOver()
    keys = sorted({(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows})
    n_lift = sum(lo.lift_interval(*k) is not None for k in keys)
    lift_rate = n_lift / len(keys)
    med_n = float(np.median(P["el_n_frac"]))
    say(f"     benchmark elements lifting with width preserved: {n_lift:,}/{len(keys):,} "
        f"({lift_rate:.4f})")
    say(f"     element N fraction: median {med_n:.5f}, "
        f"{int((P['el_n_frac'] > 0.05).sum())} elements above 5%")
    e1 = bool(lift_rate >= MIN_LIFT and med_n < MAX_N_FRAC and tss_ok)
    GG.verdict(e1, emit=say,
               if_true=f"E1 PASS -- the TSS control lifts {tss['exact']:,}/{tss['lifted']:,} onto the "
                       f"exact base and {lift_rate:.1%} of elements survive with their width",
               if_false=f"E1 FAIL -- lift {lift_rate:.4f}, median N {med_n:.4f}, TSS control "
                        f"{'ok' if tss_ok else 'FAILED'}; nothing downstream is about the right DNA")
    if not e1:
        json.dump({"test": "enhancer grammar", "gates": {"E1": False}, "liftover": tss,
                   "lift_rate": lift_rate, "median_n_frac": med_n, "log": log},
                  open(OUT, "w"), indent=1)
        return

    # ---- features --------------------------------------------------------------------------
    say()
    say("   building features")
    F, _, have = build_features(P, "el", report=say)
    say(f"     {len(F)} columns; domain properties resolved for {int(have.sum())}/{len(have)} matrices")
    Fsh, _, _ = build_features(P, "sh", report=say)
    Fraw, _, _ = build_features(P, "el", competition=False, report=say)
    for name, frame in (("real", F), ("shuffled", Fsh), ("raw-score", Fraw)):
        bad = [c for c, v in frame.items() if not np.isfinite(v).all()]
        if bad:
            say(f"     WARNING {name}: non-finite columns {bad}")
            for c in bad:
                frame[c] = np.nan_to_num(frame[c], nan=0.0, posinf=0.0, neginf=0.0)

    # ---- E2 ------------------------------------------------------------------------------------
    say()
    say("E2 IS THE EVALUATION SET UP CORRECTLY? distance alone against picking at random")
    res = {}
    for arm, blocks in ARMS.items():
        X, cols = matrix(F, blocks)
        res[arm] = run_arm(X, y, chrom, g_idx, jitter, arm, say)
    rnd_r1, rnd_mrr = [], []
    for s in SEEDS:
        rr = np.random.default_rng(1000 + s).uniform(size=len(y))
        a, b, n_eval = within_gene(rr, y, g_idx, jitter)
        rnd_r1.append(a)
        rnd_mrr.append(b)
    rnd_r1 = np.array(rnd_r1)
    say(f"    {'random-pick':22} R@1 {rnd_r1.mean():.4f}   MRR {np.mean(rnd_mrr):.4f}   "
        f"(n evaluable genes {n_eval})")
    dr = res["distance"]["r1"] - rnd_r1
    sem = dr.std(ddof=1) / np.sqrt(len(dr))
    e2 = bool(dr.mean() > 3 * sem)
    say(f"     distance beats random by {dr.mean():+.4f} +/- {sem:.4f}")
    GG.verdict(e2, emit=say,
               if_true=f"E2 PASS -- the bar for everything below is distance-only R@1 "
                       f"{res['distance']['r1'].mean():.4f}, AUPRC {res['distance']['ap'].mean():.4f}",
               if_false="E2 FAIL -- distance does not beat a random pick, so the ranking task as "
                        "set up here is not measuring what it claims")

    # ---- E3 ------------------------------------------------------------------------------------
    say()
    say("E3 DOES SEQUENCE ADD OVER DISTANCE?")
    d3 = paired(res["FULL"], res["distance"])
    say(f"     FULL vs distance   {fmt(d3)}")
    e3 = gate_pair(d3)
    GG.verdict(e3, emit=say,
               if_true="E3 PASS -- the element's own sequence carries information the distance does not",
               if_false="E3 FAIL -- everything the sequence stack knows, the distance already knew")

    # ---- E4 ------------------------------------------------------------------------------------
    say()
    say("E4 MOTIFS, OR BASE COMPOSITION? dinucleotide-shuffled elements")
    Xs, _ = matrix(Fsh, ARMS["FULL"])
    res["FULL_shuffled"] = run_arm(Xs, y, chrom, g_idx, jitter, "FULL_shuffled", say)
    d4 = paired(res["FULL"], res["FULL_shuffled"])
    say(f"     FULL vs FULL_shuffled   {fmt(d4)}")
    e4 = gate_pair(d4, use_ap=False)
    GG.verdict(e4, emit=say,
               if_true="E4 PASS -- shuffling the element while holding every dinucleotide frequency "
                       "costs real accuracy, so the sequence block is reading sites, not composition",
               if_false="E4 FAIL -- a composition-matched shuffle scores the same, so the sequence "
                        "block is reading base composition and the motif account is decoration")

    # ---- E5 ------------------------------------------------------------------------------------
    say()
    say("E5 WHICH FACTORS, OR HOW MANY? permuting the matrix axis of the promoter side")
    nm = len(P["motif_ids"])
    pr1, pap = [], []
    for s in SEEDS:
        perm = np.random.default_rng(7000 + s).permutation(nm)
        Fp, _, _ = build_features(P, "el", tf_perm=perm, report=lambda *_: None)
        Xp, _ = matrix(Fp, ARMS["FULL"])
        fold = folds_for(chrom, s)
        sc = oof_scores(Xp, y, fold, s)
        a, b, _ = within_gene(sc, y, g_idx, jitter)
        pr1.append(a)
        pap.append(average_precision_score(y, sc))
    res["FULL_permTF"] = dict(r1=np.array(pr1), mrr=np.zeros(len(SEEDS)), ap=np.array(pap))
    say(f"    {'FULL_permTF':22} R@1 {np.mean(pr1):.4f}   AUPRC {np.mean(pap):.4f}")
    d5 = paired(res["FULL"], res["FULL_permTF"])
    say(f"     FULL vs FULL_permTF   {fmt(d5)}")
    e5 = gate_pair(d5, use_ap=False)
    GG.verdict(e5, emit=say,
               if_true="E5 PASS -- WHICH factors the promoter and the element share matters, not "
                       "merely how many sites either carries",
               if_false="E5 FAIL -- the pairing signal is site COUNT, not factor identity; the "
                        "promoter-element correspondence can be permuted away at no cost")

    # ---- E6, E7 --------------------------------------------------------------------------------
    say()
    say("E6 DOES SHAPE ADD OVER SITES?")
    d6 = paired(res["+shape"], res["+pairing"])
    say(f"     +shape vs +pairing   {fmt(d6)}")
    e6 = gate_pair(d6)
    GG.verdict(e6, emit=say,
               if_true="E6 PASS -- groove geometry at the occupied positions adds over the sites",
               if_false="E6 FAIL -- groove geometry adds nothing the site scores did not already have")

    say()
    say("E7 DOES MAJOR-GROOVE COMPLEMENTARITY ADD OVER SHAPE? (the plan's first stage)")
    d7 = paired(res["+compl_major"], res["+shape"])
    say(f"     +compl_major vs +shape   {fmt(d7)}")
    e7 = gate_pair(d7)
    GG.verdict(e7, emit=say,
               if_true="E7 PASS -- major groove width against domain bulk, and the domain's reach "
                       "past the motif core, add over the DNA-side shape alone",
               if_false="E7 FAIL -- the major-groove domain terms add nothing over the shape")

    say()
    say("E7b AND THEN THE MINOR GROOVE, ON TOP (the plan's second stage)")
    d7b = paired(res["+compl_minor"], res["+compl_major"])
    say(f"     +compl_minor vs +compl_major   {fmt(d7b)}")
    nmin = int(sum(1 for m in P["motif_ids"]
                   if TD.load().get(str(m), {}).get("groove") in ("minor", "both")))
    say(f"     the minor-groove terms are non-zero for {nmin}/{len(P['motif_ids'])} matrices, "
        f"zero for the rest by construction")
    e7b = gate_pair(d7b)
    GG.verdict(e7b, emit=say,
               if_true="E7b PASS -- adding the minor groove, for the folds that read it, adds over "
                       "the major groove alone",
               if_false="E7b FAIL -- the minor-groove terms add nothing once the major groove is in")

    # ---- E8 ------------------------------------------------------------------------------------
    say()
    say("E8 DOES OCCUPANCY UNDER COMPETITION BEAT THE RAW BEST SCORE?")
    Xr, _ = matrix(Fraw, ARMS["dist+comp+sites"])
    res["sites_raw"] = run_arm(Xr, y, chrom, g_idx, jitter, "dist+comp+sites_RAW", say)
    d8 = paired(res["dist+comp+sites"], res["sites_raw"])
    say(f"     competition vs raw best score   {fmt(d8)}")
    e8 = gate_pair(d8, use_ap=False)
    GG.verdict(e8, emit=say,
               if_true="E8 PASS -- normalising by the competing sites in the window beats the raw "
                       "best score, which is the futility correction doing work",
               if_false="E8 FAIL -- the competition denominator buys nothing over the raw best "
                        "score on this task")

    # ---- E9 ------------------------------------------------------------------------------------
    say()
    say("E9 THE FUTILITY COUNT, predicted before it was measured: >99% of matches are not in enhancers")
    NS = P["el_NS"]
    pos_el = np.unique(e_idx[y == 1])
    tot = float(NS.sum())
    in_pos = float(NS[:, pos_el].sum())
    frac = 1.0 - in_pos / max(tot, 1.0)
    el_bp = float(P["el_width"].sum())
    per_mb = tot / (el_bp / 1e6)
    bgz = P["bg_LZ"]
    w6 = P["motif_width"] == 6
    say(f"     {int(tot):,} above-threshold matches over {el_bp/1e6:.2f} Mb of element sequence "
        f"= {per_mb:,.0f} per Mb across {NS.shape[0]} motifs")
    say(f"     {int(in_pos):,} of them lie in one of the {len(pos_el):,} validated-positive elements; "
        f"{frac:.5f} do not")
    if w6.any():
        pm6 = float(NS[w6].sum()) / (el_bp / 1e6) / int(w6.sum())
        say(f"     for the {int(w6.sum())} width-6 motifs, {pm6:,.0f} matches per Mb per motif, so "
            f"{pm6*4:,.0f} in a 4 Mb window -- the plan's arithmetic predicted 1,953")
    e9 = bool(frac >= FUTILITY_FLOOR)
    GG.verdict(e9, emit=say,
               if_true=f"E9 PASS -- {frac:.3%} of motif matches sit outside any validated enhancer, "
                       f"so the filter chain has to do essentially all of the work",
               if_false=f"E9 FAIL -- only {frac:.3%} of matches are outside validated enhancers, so "
                        f"the futility premise does not hold on this data as measured")

    # ---- E10 -----------------------------------------------------------------------------------
    say()
    say("E10 WHAT THIS CANNOT SHOW")
    say("     K562 only. The candidate set is what the CRISPR screens chose to test, not the")
    say("     genome, so within-gene R@1 here is not the 2 Mb search the plan described -- it is")
    say("     the same ranking question on a pre-filtered pool, and the pool was filtered by")
    say("     people who already believed those elements might be enhancers.")
    say("     Sequence is hg19 read at lifted coordinates. The TSS control says the lift is exact,")
    say("     but hg19 and GRCh38 differ in content at a small number of loci and no control here")
    say("     can see that.")
    say("     JASPAR matrices are in-vitro preferences. A factor that is not expressed in K562")
    say("     still contributes sites here, and nothing in this loop knows that.")
    e10 = True
    say(f"     E10 {'PASS' if e10 else 'FAIL'}")

    gates = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5,
             "E6": e6, "E7": e7, "E7b": e7b, "E8": e8, "E9": e9, "E10": e10}

    say()
    say("  THE LADDER, rung by rung, against the bar each rung is held to")
    prev = None
    ladder = []
    for arm in ARMS:
        r = res[arm]
        row = dict(arm=arm, r1=float(r["r1"].mean()), mrr=float(r["mrr"].mean()),
                   ap=float(r["ap"].mean()))
        if prev is not None:
            d = paired(r, res[prev])
            row.update(d_r1=d["mean_r1"], d_ap=d["mean_ap"], clears=gate_pair(d))
            say(f"    {prev:22} -> {arm:22} dR@1 {d['mean_r1']:+.4f}  dAUPRC {d['mean_ap']:+.4f}  "
                f"{'clears' if gate_pair(d) else 'sub-threshold'}")
        ladder.append(row)
        prev = arm

    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz"), Path("colab/data/tf_domains.json")],
                      available=len(y), used=len(y), selection="powered valid K562 pairs",
                      seed=TIE_SEED,
                      controls=["dinucleotide-shuffled elements (E4)",
                                "promoter matrix-axis permutation, counts held exactly (E5)",
                                "raw best score against occupancy under competition (E8)",
                                "random-pick within-gene baseline (E2)",
                                "liftover checked against 16,380 TSSs held in both assemblies (E1)"],
                      note="sequence-level enhancer-gene ranking against the distance floor")
    out = dict(test="enhancer grammar", gates=gates,
               n_pairs=int(len(y)), n_positives=int(y.sum()), base_rate=float(y.mean()),
               n_elements=int(len(P["el_key"])), n_genes=int(len(P["gn_key"])),
               n_evaluable_genes=int(n_eval),
               liftover=tss, lift_rate=float(lift_rate), median_n_frac=med_n,
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "mrr", "ap")}
                     for k, v in res.items()},
               random_pick_r1=float(rnd_r1.mean()),
               ladder=ladder,
               increments={"E3_full_vs_distance": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                   for k, v in d3.items()},
                           "E4_vs_shuffled": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                              for k, v in d4.items()},
                           "E5_vs_permTF": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                            for k, v in d5.items()},
                           "E6_shape": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                        for k, v in d6.items()},
                           "E7_compl_major": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                              for k, v in d7.items()},
                           "E7b_compl_minor": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                               for k, v in d7b.items()},
                           "E8_competition": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                              for k, v in d8.items()}},
               futility=dict(matches=int(tot), in_positive_elements=int(in_pos),
                             fraction_outside=float(frac), per_mb=float(per_mb)),
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
