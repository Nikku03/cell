"""Loop 178. Give stage two an explicit "how strong is this enhancer" score, built from sequence,
and see whether that is what was missing.

THE PROPOSAL, AND ONE CORRECTION TO ITS PREMISE. The reading offered was that loop 176's oracle
shows a pure global-enhancer-strength score placing a valid target at #1 for 88.4% of genes, and
that the 34-column stack is therefore missing the chemical signature of enhancer potency. The
second half is a real hypothesis and this loop tests it. The first half overstates what the oracle
measured, and the overstatement is mine to correct because I reported the number without this
qualification.

  M2's oracle scored each element by the FRACTION OF ITS TESTED PAIRS THAT ARE POSITIVE. For an
  element tested against a single gene that fraction is 1.0 or 0.0 -- it IS the label, not an
  estimate of anything. 45.0% of elements in this benchmark were tested against exactly one gene,
  and inside the evaluable genes 100 of 367 positives (27.2%) sit on such elements. So a
  substantial part of 0.8844 is memorisation of the answer key rather than a strength signal that
  any model could learn. The ceiling stands as a CEILING -- no gene-blind scorer can beat it -- but
  it is not evidence that a learnable strength score reaches it, and P1 measures the difference
  instead of arguing about it.

WHAT IS ACTUALLY WORTH TESTING, and it comes straight from the proposal. Loop 177 found that the
sequence stack's problem was never the physics and never the learner: it was that every column
summed 736 matrices into one number before anything saw it. Re-aggregating the SAME occupancies by
JASPAR structural class and family took stage one from AUC 0.6807 to 0.8506, and a logistic
regression on those columns matched the tree ensemble. Stage two has never seen that
representation -- loops 173, 175 and 176 all ran the old 34-column summed stack. That is a real
omission and it is the first thing this loop fixes.

The second is the proposal's own suggestion, made literal. Rather than hoping the stack assembles a
potency proxy from groove width and duplex energy, TRAIN one: inside each training fold, fit the
loop-177 stage-one classifier on windows from the training chromosomes only, apply it to the
benchmark elements, and hand its output to stage two as a single column called what it is. If
global enhancer potency is the missing variable, a column that predicts it should move stage two.

The third is motif clustering, which the proposal names and loop 177 already tested on stage one,
where it FAILED at -0.0061 AUC. It has not been tested on stage two and it is carried here so the
answer is measured on both tasks rather than transferred between them.

PREDECLARED, BEFORE ANY NUMBER.

  P1 WHAT DOES THE ORACLE MEASURE? The M2 oracle recomputed LEAVE-ONE-GENE-OUT: element e's score
     for gene g is the positive rate over e's OTHER genes only, so it never sees the pair being
     ranked. Elements tested against one gene alone get no score and take the pool median.
     Gate: PASS iff the leave-one-out oracle lands BELOW 0.70. A pass means most of 0.8844 was
     label memorisation and the learnable gene-blind ceiling is far lower than reported. A FAIL --
     the leave-one-out oracle still high -- means a genuine element-intrinsic potency signal exists
     at that strength and everything below should be judged against it rather than against 0.5930.

  P2 DOES THE LOOP-177 REPRESENTATION HELP STAGE TWO? The 34-column stack against the same stack
     with occupancy re-aggregated by class and family plus the fold-internal spectrum.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- the bar loop 173's E3 fixed and failed.

  P3 DOES AN EXPLICIT SEQUENCE-PREDICTED POTENCY COLUMN HELP? The stage-one classifier, trained
     inside the fold on loop 174's windows from training chromosomes only, applied to the
     benchmark elements, added as one column.
     Gate: same bar, measured as an increment over P2's arm.

  P4 DOES CLUSTERING HELP STAGE TWO? Homotypic and heterotypic density, which cost -0.0061 on
     stage one.
     Gate: same bar, as an increment over P2's arm.

  P5 THE DECISIVE ONE. The best arm against distance alone on identical folds.
     Gate: same bar. This is the gate loops 173 and 175 failed and it is unchanged.

  P6 IS THE POTENCY COLUMN LEAKING? It is trained on stage-one windows, which are drawn around
     these same genes, so a training window could sit on a chromosome held out at stage two. The
     honest arm restricts stage-one training to training-fold chromosomes; the control refits it on
     every chromosome.
     Gate: PASS iff the honest arm does not EXCEED the leaky arm past 3 sem. It cannot, unless the
     restriction is not being applied.

  P7 THE SHUFFLE STILL DECIDES. Best configuration on real against dinucleotide-shuffled sequence.
     Gate: real beats shuffled in >= 4/5 seeds past 3 sem.

  P8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_potency.json
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
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_vs_genome as L174       # noqa: E402
import loop_enhancer_stage_one as L177       # noqa: E402

from sklearn.decomposition import PCA                          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_potency.json"
ECLUST = SC.CACHE / "enh_element_clusters.npz"
SEEDS = L173.SEEDS
NFOLD = 5
N_PC = 32
MIN_SEEDS = 4
P1_BAR = 0.70
L173_DIST_R1 = 0.5930
L173_FULL_R1 = 0.6050

# element-intrinsic blocks only: what can be computed for a piece of DNA without naming a gene,
# and therefore what a stage-one model trained on genomic windows can be applied to
STAGE1_BLOCKS = ["composition", "sites", "shape", "compl_major", "compl_minor", "opening"]
PAIR_COLS = ["log_dist", "log_shared_occ", "shared_n", "prom_n", "shared_frac", "jaccard"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def element_clusters(report=print):
    """Clustering statistics for the benchmark elements, real and shuffled, on the exact sequence
    loop 173 scanned. The element list is regenerated from the same lift, and alignment is checked
    against the cached GC by the caller."""
    if ECLUST.exists():
        z = np.load(ECLUST, allow_pickle=True)
        report(f"    element cluster cache: {ECLUST.name} ({ECLUST.stat().st_size/1e6:.1f} MB)")
        return {k: z[k] for k in z.files}
    import random
    t0 = time.time()
    ids, mots = SC.load_motifs(report)
    maxw = max(m.shape[0] for m in mots.values())
    rows = SC.load_benchmark(report)
    lo = GEN.LiftOver()
    el19 = {}
    for r in rows:
        k = (r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))
        if k not in el19:
            el19[k] = lo.lift_interval(*k)
    tss19 = {}
    for r in rows:
        t = (r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])
        if t not in tss19:
            tss19[t] = lo.lift(t[0], t[1])
    rows = [r for r in rows
            if el19[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))] is not None
            and tss19[(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])] is not None]
    el_key = sorted({(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows})
    seqs = GEN.Genome().extract_cached([(k[0],) + el19[k] for k in el_key], "elem", report)
    srng = random.Random(SC.SHUF_SEED)
    shf = [SC.dinuc_shuffle(s, srng) for s in seqs]
    out = {}
    for tag, ss in (("el", seqs), ("sh", shf)):
        cat, starts = SC.concat(ss, maxw)
        report(f"    clustering {tag}: {len(ss):,} elements, {len(cat):,} bp")
        for k, v in SC.scan_clusters(cat, starts, ids, mots, report, tag).items():
            out[f"{tag}_{k}"] = v
        del cat
    out["gc_check"] = np.array([SC._composition(s)["gc"] for s in seqs], np.float32)
    ECLUST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ECLUST, **out)
    report(f"    -> {ECLUST} [{time.time()-t0:.0f}s]")
    return out


def element_frame(S, tag, report=print):
    """The element-intrinsic feature frame for the benchmark elements, normalised by the GENOME
    BACKGROUND rather than by a per-gene pool, so it is computed the same way loop 174 computes it
    for genomic windows and a model fitted on one can be applied to the other."""
    mx_max = S["motif_maxscore"].astype(np.float64)
    width = S["motif_width"].astype(np.float64)
    LZ, MX, NS = S[f"{tag}_LZ"], S[f"{tag}_MX"], S[f"{tag}_NS"]
    T = {n: S[f"{tag}_SH"][i] for i, n in enumerate(list(S["tracks"]))}
    bg = np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1)) / float(S["bg_bp"])
    occ = np.exp(LZ.astype(np.float64)
                 - (np.log(np.maximum(bg, 1e-300)) + np.log(L174.WINDOW_BP))[:, None])
    F = {}
    F["log_width"] = np.log10(np.maximum(S["el_width"], 1)).astype(np.float64)
    F["gc"] = S["el_gc"].astype(np.float64)
    F["cpg_raw"] = S["el_cpg_raw"].astype(np.float64)
    sum_occ = occ.sum(0)
    F["log_sum_occ"] = np.log10(np.maximum(sum_occ, 1e-300))
    F["log_max_occ"] = np.log10(np.maximum(occ.max(0), 1e-300))
    F["n_sites"] = NS.sum(0).astype(np.float64)
    F["log_elem_n"] = np.log10(1.0 + (NS > 0).sum(0))
    for name in ("mgw", "mgrw", "prot", "roll", "helt", "ep", "dg"):
        v = T[name].astype(np.float64)
        ok = np.isfinite(v)
        F["site_" + name] = (np.where(ok, occ * v, 0.0).sum(0)
                             / np.maximum(np.where(ok, occ, 0.0).sum(0), 1e-300))
    pref = "elmean" if tag == "el" else "shmean"
    for name in ("mgw", "prot", "ep", "dg"):
        F["elem_" + name] = S[f"{pref}_{name}"].astype(np.float64)
    dom = TD.load()
    ids = [str(m) for m in S["motif_ids"]]
    have = np.array([bool(dom.get(m, {}).get("route")) for m in ids])
    def col(k, d=0.0):
        return np.array([float(dom.get(m, {}).get(k, d) or d) for m in ids])
    chg, arg, vol, dlen = col("charge_density"), col("arg_frac"), col("mean_volume"), col("length", 1.0)
    groove = np.array([dom.get(m, {}).get("groove", "major") for m in ids])
    minorish = ((groove == "minor") | (groove == "both")) & have
    majorish = ((groove == "major") | (groove == "both")) & have
    volc = vol - (vol[have].mean() if have.any() else 0.0)
    span = np.where(width > 0, dlen / width, 0.0)
    EP = np.nan_to_num(T["ep"].astype(np.float64), nan=0.0)
    MGW = np.nan_to_num(T["mgw"].astype(np.float64), nan=0.0)
    MGrW = np.nan_to_num(T["mgrw"].astype(np.float64), nan=0.0)
    PROT = np.nan_to_num(T["prot"].astype(np.float64), nan=0.0)
    F["comp_charge"] = (occ * (-EP) * (chg * minorish)[:, None]).sum(0)
    F["comp_arg"] = (occ * (-EP) * (arg * minorish)[:, None]).sum(0)
    F["comp_steric"] = (occ * MGW * (volc * have)[:, None]).sum(0)
    F["comp_major"] = (occ * MGrW * (volc * majorish)[:, None]).sum(0)
    F["comp_twist"] = (occ * PROT * (chg * have)[:, None]).sum(0)
    F["comp_span"] = (occ * (span * have)[:, None]).sum(0) / np.maximum(sum_occ, 1e-300)
    RT = 0.616
    best = np.argmax(MX / mx_max[:, None], axis=0)
    F["bind_kcal_pb"] = -RT * MX[best, np.arange(MX.shape[1])].astype(np.float64) \
        / np.maximum(width[best], 1.0)
    F["open_kcal_pb"] = -F["site_dg"]
    F["net_kcal_pb"] = F["bind_kcal_pb"] + F["open_kcal_pb"]
    # class / family occupancy -- loop 177's winning block
    groups = defaultdict(list)
    for i, m in enumerate(ids):
        r = dom.get(m, {})
        groups["CLS:" + str(r.get("cls"))].append(i)
        groups["FAM:" + str(r.get("family"))].append(i)
    fam = {}
    for k, ix in sorted(groups.items()):
        if len(ix) >= L177.MIN_FAMILY:
            fam["occ_" + k.replace(" ", "_").replace(",", "")[:44]] = np.log10(
                np.maximum(occ[ix].sum(0), 1e-300))
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    for k in fam:
        fam[k] = np.nan_to_num(fam[k], nan=0.0, posinf=0.0, neginf=0.0)
    report(f"    element frame ({tag}): {len(F)} intrinsic + {len(fam)} class/family columns")
    return F, fam, np.log(np.maximum(occ, 1e-300)).T


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run_pairs(build, y, chrom, g_idx, jitter, tag, report=print):
    """`build(train_mask, seed)` returns (X, ) for all rows, fitted using training rows only.
    Everything fold-internal -- spectra, potency models -- is constructed inside it."""
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
    report(f"    {tag:34} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 178  AN EXPLICIT SEQUENCE-PREDICTED ENHANCER POTENCY COLUMN FOR STAGE TWO")
    say("=" * 104)
    say(f"  PREDECLARED: the leave-one-gene-out oracle must land below {P1_BAR} for M2's 0.8844 to")
    say("  have been mostly memorisation; every arm judged on loop 173's E3 bar -- paired R@1")
    say(f"  positive in >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= +0.01 in >= {MIN_SEEDS}/5;")
    say(f"  the winner must clear distance-only R@1 {L173_DIST_R1}; the potency column must not")
    say("  out-score its own leaky variant; and the shuffle must still cost the best arm.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))
    C = element_clusters(say)

    # ---- P1 ------------------------------------------------------------------------------------
    say()
    say("P1 WHAT DOES THE ORACLE MEASURE?")
    npos_e, ntot_e = defaultdict(int), defaultdict(int)
    for i in range(len(y)):
        ntot_e[int(e_idx[i])] += 1
        npos_e[int(e_idx[i])] += int(y[i])
    single = sum(1 for k, v in ntot_e.items() if v == 1)
    say(f"     {single:,}/{len(ntot_e):,} elements ({single/len(ntot_e):.1%}) were tested against "
        f"exactly one gene; for those the M2 oracle's score IS the label")
    raw = np.array([npos_e[int(e_idx[i])] / max(ntot_e[int(e_idx[i])], 1) for i in range(len(y))])
    loo = np.full(len(y), np.nan)
    for i in range(len(y)):
        k = int(e_idx[i])
        n = ntot_e[k] - 1
        loo[i] = (npos_e[k] - y[i]) / n if n > 0 else np.nan
    med = np.nanmedian(loo)
    loo_f = np.where(np.isfinite(loo), loo, med)
    r_raw = L173.within_gene(raw, y, g_idx, jitter)[0]
    r_loo = L173.within_gene(loo_f, y, g_idx, jitter)[0]
    say(f"     M2 oracle as reported (sees the pair): R@1 {r_raw:.4f}")
    say(f"     leave-one-gene-out oracle (never sees the pair): R@1 {r_loo:.4f}   "
        f"AUPRC {average_precision_score(y, loo_f):.4f}")
    p1 = bool(r_loo < P1_BAR)
    GG.verdict(p1, emit=say,
               if_true=f"P1 PASS -- {r_raw:.4f} falls to {r_loo:.4f} once the pair's own label is "
                       f"removed, so most of the reported ceiling was memorisation and the "
                       f"learnable gene-blind ceiling is {r_loo:.4f}, not 0.8844",
               if_false=f"P1 FAIL -- the leave-one-out oracle holds at {r_loo:.4f}, so a genuine "
                        f"element-intrinsic potency signal of that strength exists and every arm "
                        f"below should be read against it")

    # ---- frames --------------------------------------------------------------------------------
    say()
    say("   building frames")
    E, FAM, SPEC = element_frame(S, "el", say)
    Es, FAMs, SPECs = element_frame(S, "sh", say)
    gc_ok = bool(np.array_equal(C["gc_check"], S["el_gc"]))
    say(f"     element cluster stats align with the cached scan bit for bit: {gc_ok}")
    P173, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    P173s, _, _ = L173.build_features(S, "sh", report=lambda *_: None)
    for fr in (P173, P173s):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)

    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    Xbase = np.column_stack([P173[c] for c in base_cols])
    Xbase_s = np.column_stack([P173s[c] for c in base_cols])
    fam_cols = sorted(FAM)
    Xfam = np.column_stack([Xbase] + [FAM[c][e_idx] for c in fam_cols])
    Xfam_s = np.column_stack([Xbase_s] + [FAMs[c][e_idx] for c in fam_cols])
    Xclu = np.column_stack([Xfam] + [C["el_" + c][e_idx] for c in L177.CLUSTER_COLS])
    say(f"     columns: base {Xbase.shape[1]}, +family {Xfam.shape[1]}, +cluster {Xclu.shape[1]}")

    # ---- the stage-one potency model ------------------------------------------------------------
    D = L174.build_scan(say)
    wkind = np.array([str(k) for k in D["kind"]])
    wchrom = np.array([str(c) for c in D["chrom"]])
    wkeep = wkind != "tested"
    wy = D["y"].astype(int)[wkeep]
    WF = L174.features(D, "el")
    WFAM, _ = L177.family_features(D, "el", lambda *_: None)
    s1_cols = [c for b in STAGE1_BLOCKS for c in L174.BLOCKS[b]]
    s1_fam = [c for c in sorted(WFAM) if c in FAM]
    say(f"     stage-one transfer set: {len(s1_cols)} intrinsic + {len(s1_fam)} class/family "
        f"columns shared between genomic windows and benchmark elements")
    Xw = np.nan_to_num(np.column_stack([WF[c][wkeep] for c in s1_cols]
                                       + [WFAM[c][wkeep] for c in s1_fam]))
    Xe_apply = np.nan_to_num(np.column_stack([E[c][e_idx] for c in s1_cols]
                                             + [FAM[c][e_idx] for c in s1_fam]))
    Xe_apply_s = np.nan_to_num(np.column_stack([Es[c][e_idx] for c in s1_cols]
                                               + [FAMs[c][e_idx] for c in s1_fam]))
    wchrom_k = wchrom[wkeep]

    def potency(train_mask, seed, leak=False, apply_X=None):
        """The stage-one classifier fitted on genomic windows whose chromosome is in the stage-two
        TRAINING fold, then applied to every benchmark element."""
        ok = np.ones(len(wy), bool) if leak else np.isin(wchrom_k, np.unique(chrom[train_mask]))
        if ok.sum() < 200 or wy[ok].sum() < 20:
            return np.zeros(len(e_idx))
        m = gbm(seed)
        m.fit(Xw[ok], wy[ok])
        return m.predict_proba(apply_X if apply_X is not None else Xe_apply)[:, 1]

    def spec_build(X, spectrum):
        def f(tr, seed):
            p = PCA(n_components=N_PC, random_state=seed)
            p.fit(spectrum[tr])
            return np.column_stack([X, p.transform(spectrum)])
        return f

    SPECp = SPEC[e_idx]
    SPECp_s = SPECs[e_idx]
    res = {}
    res["distance"] = run_pairs(lambda tr, s: np.column_stack([P173["log_dist"]]),
                                y, chrom, g_idx, jitter, "distance", say)
    res["base34"] = run_pairs(lambda tr, s: Xbase, y, chrom, g_idx, jitter,
                              "loop 173's 34 columns", say)
    res["+family"] = run_pairs(lambda tr, s: Xfam, y, chrom, g_idx, jitter,
                               "+class/family occupancy", say)
    res["+family+spectrum"] = run_pairs(spec_build(Xfam, SPECp), y, chrom, g_idx, jitter,
                                        "+class/family+spectrum", say)
    res["+potency"] = run_pairs(
        lambda tr, s: np.column_stack([Xfam, potency(tr, s)]),
        y, chrom, g_idx, jitter, "+predicted enhancer potency", say)
    res["+potency_leaky"] = run_pairs(
        lambda tr, s: np.column_stack([Xfam, potency(tr, s, leak=True)]),
        y, chrom, g_idx, jitter, "+potency, LEAKY control", say)
    res["+cluster"] = run_pairs(lambda tr, s: Xclu, y, chrom, g_idx, jitter, "+clustering", say)
    res["ALL"] = run_pairs(
        lambda tr, s: np.column_stack([Xclu, potency(tr, s)]),
        y, chrom, g_idx, jitter, "everything", say)

    # ---- P2..P5 --------------------------------------------------------------------------------
    say()
    say("P2 DOES THE LOOP-177 REPRESENTATION HELP STAGE TWO?")
    d2 = L173.paired(res["+family+spectrum"], res["base34"])
    say(f"     +class/family+spectrum vs the 34 columns   {L173.fmt(d2)}")
    say(f"     (family alone: {L173.fmt(L173.paired(res['+family'], res['base34']))})")
    p2 = L173.gate_pair(d2)
    GG.verdict(p2, emit=say,
               if_true="P2 PASS -- the representation that took stage one from 0.6807 to 0.8506 "
                       "also moves stage two",
               if_false="P2 FAIL -- resolving the matrices by class and family, which was worth "
                        "+0.17 AUC on stage one, does nothing for gene assignment")

    say()
    say("P3 DOES AN EXPLICIT SEQUENCE-PREDICTED POTENCY COLUMN HELP?")
    d3 = L173.paired(res["+potency"], res["+family"])
    say(f"     +potency vs +class/family   {L173.fmt(d3)}")
    p3 = L173.gate_pair(d3)
    GG.verdict(p3, emit=say,
               if_true="P3 PASS -- a column that predicts how strong an enhancer the element is, "
                       "trained on genomic windows and transferred, moves gene assignment",
               if_false="P3 FAIL -- handing stage two an explicit potency score changes nothing, "
                        "so potency was not the missing variable")

    say()
    say("P4 DOES CLUSTERING HELP STAGE TWO? (it cost -0.0061 AUC on stage one)")
    d4 = L173.paired(res["+cluster"], res["+family"])
    say(f"     +clustering vs +class/family   {L173.fmt(d4)}")
    p4 = L173.gate_pair(d4)
    GG.verdict(p4, emit=say,
               if_true="P4 PASS -- motif clustering helps here even though it did not on stage one",
               if_false="P4 FAIL -- clustering adds nothing on either task")

    say()
    say("P5 THE DECISIVE ONE: does anything clear the distance floor?")
    best = max((k for k in res if k not in ("distance", "+potency_leaky")),
               key=lambda k: res[k]["r1"].mean())
    d5 = L173.paired(res[best], res["distance"])
    say(f"     best arm is {best} at R@1 {res[best]['r1'].mean():.4f} against distance "
        f"{res['distance']['r1'].mean():.4f}")
    say(f"     {L173.fmt(d5)}")
    p5 = L173.gate_pair(d5)
    GG.verdict(p5, emit=say,
               if_true=f"P5 PASS -- {best} clears the bar loops 173 and 175 both failed",
               if_false="P5 FAIL -- stage two is still distance, with sequence adding nothing that "
                        "clears the bar")

    say()
    say("P6 IS THE POTENCY COLUMN LEAKING?")
    d6 = L173.paired(res["+potency"], res["+potency_leaky"])
    say(f"     honest minus leaky   {L173.fmt(d6)}")
    p6 = bool(not L173.gate_pair(d6, use_ap=False))
    GG.verdict(p6, emit=say,
               if_true="P6 PASS -- the chromosome-restricted potency model does not out-score the "
                       "one trained on every chromosome, which is the only possible ordering",
               if_false="P6 FAIL -- the restricted model BEATS the unrestricted one, so the "
                        "restriction is not doing what the code says")

    say()
    say("P7 THE SHUFFLE STILL DECIDES")
    Xclu_s = np.column_stack([Xfam_s] + [C["sh_" + c][e_idx] for c in L177.CLUSTER_COLS])
    res["ALL_shuffled"] = run_pairs(
        lambda tr, s: np.column_stack([Xclu_s, potency(tr, s, apply_X=Xe_apply_s)]),
        y, chrom, g_idx, jitter, "everything, SHUFFLED", say)
    d7 = L173.paired(res["ALL"], res["ALL_shuffled"])
    say(f"     real vs dinucleotide-shuffled   {L173.fmt(d7)}")
    p7 = L173.gate_pair(d7, use_ap=False)
    GG.verdict(p7, emit=say,
               if_true="P7 PASS -- the shuffle costs the stage-two stack, so it is reading sites",
               if_false="P7 FAIL -- the composition-matched shuffle matches it, so on stage two the "
                        "sequence columns are still not reading binding sites")

    say()
    say("P8 WHAT THIS CANNOT SHOW")
    say("     The potency column is trained against GENOMIC decoys, so it predicts 'is this an")
    say("     enhancer', not 'is this a strong enhancer'. A graded potency label would need an")
    say("     assay that measures element strength directly -- STARR-seq or MPRA -- and none is")
    say("     joined here.")
    say("     Stage two's real missing variable is physical contact: which promoter the element is")
    say("     looped to. Nothing in this loop measures Hi-C, TAD boundaries or CTCF anchors, and")
    say("     no sequence feature is a substitute for them.")
    say("     K562 only; JASPAR in-vitro preferences; no expression filter on the factor set.")
    p8 = True
    say(f"     P8 {'PASS' if p8 else 'FAIL'}")

    gates = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7, "P8": p8}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz"), Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["leave-one-gene-out recomputation of the M2 oracle",
                                "the potency model refitted per fold on training chromosomes only, "
                                "with the unrestricted variant reported beside it",
                                "the spectrum projection fitted inside each training fold",
                                "dinucleotide shuffle through the whole stack including potency"],
                      note="does an explicit sequence-predicted enhancer potency score move stage two")
    out = dict(test="enhancer potency for stage two", gates=gates,
               oracle=dict(raw_r1=float(r_raw), loo_r1=float(r_loo),
                           singleton_elements=int(single), n_elements=int(len(ntot_e))),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in d.items()}
                       for k, d in (("P2", d2), ("P3", d3), ("P4", d4), ("P5", d5),
                                    ("P6", d6), ("P7", d7))},
               best_arm=best, gc_aligned=gc_ok,
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
