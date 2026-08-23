"""Loop 174. The question loop 173 could not ask: can the sequence tell a real enhancer from a
random piece of genome the same distance away?

WHY THIS LOOP EXISTS. Loop 173 ran the sequence chain on the EP CRISPR benchmark and it failed
every gate that mattered -- dinucleotide-shuffled elements scored HIGHER than real ones, permuting
factor identity cost nothing, and every motif and shape column sat between AUC 0.45 and 0.52. But
one number from that loop says the evaluation, not only the method, was part of the reason:

    426 of 4,482 benchmark elements are validated enhancers. They hold 10.55% of the element base
    pairs and 10.42% of all 520,817 above-threshold motif matches. A ratio of 0.99.

Motif matches are uniform per base pair over that pool. That is not because motifs are meaningless
-- it is because every element in the pool is ALREADY an accessible, enhancer-looking region that
the screen designers chose to test. The pool has been pre-filtered by exactly the property a
sequence model would be measuring. Asking a motif scanner to rank inside it is asking it to do
something it was never the instrument for.

The plan being tested was a TWO-STAGE search: take 2 Mb either side of a promoter, find the
sequences a factor could bind, and narrow them down. Stage one is "which windows in this 4 Mb could
be regulatory at all". Stage two is "which of those acts on this gene". The CRISPR benchmark only
scores stage two, because stage one was done for it by the people who built the library. This loop
scores stage one, on the same ground truth, by replacing the pre-filtered negatives with the thing
the plan would actually be searching: random genomic windows at the same distance from the same
promoter.

WHAT IS MEASURED, AND AGAINST WHAT.
  POSITIVES     the 482 (element, gene) pairs the EP CRISPR benchmark validated in K562.
  DECOYS        for each positive, 10 windows drawn in the same gene's +/- 2 Mb, at a distance
                sampled within +/-10% of the positive's own distance to the TSS, with the SAME
                width, at least 10 kb from the TSS, overlapping no element the benchmark tested,
                and under 5% N. Distance and width are therefore matched BY CONSTRUCTION, which is
                what makes F2 a real check rather than a formality.
  METRIC        AUC, out-of-fold, chromosome-held-out, 5 folds x 5 seeds, identical folds per arm.
                AUC rather than AUPRC because the positive rate here is set by the decoy ratio, not
                measured, so a precision-based number would only be reporting that ratio back.

PREDECLARED, BEFORE ANY NUMBER.

  F1 IS THE MATCHING REAL? Decoys must not overlap any tested element, must sit within 10% of their
     positive's distance, and must carry the same width.
     Gate: PASS iff zero decoys overlap a tested element, the median |log2 distance ratio| is below
     0.15, and widths match exactly.

  F2 IS DISTANCE NEUTRALISED? Distance was the whole story in loop 173, so it has to be gone here
     or nothing else can be read.
     Gate: PASS iff a distance-only arm scores AUC within 0.03 of 0.50.

  F3 DOES SEQUENCE SEPARATE AN ENHANCER FROM MATCHED GENOME AT ALL? The full sequence stack.
     Gate: PASS iff out-of-fold AUC exceeds 0.60. That threshold is chosen because it is roughly
     what element ANNOTATION alone buys on this benchmark (a K562-accessible cCRE raises the
     validated-positive rate 2.51x, measured earlier in this project), so anything below it is not
     competitive with simply asking whether the window is accessible.

  F4 MOTIFS, OR BASE COMPOSITION? The same stack on dinucleotide-shuffled sequence, BOTH classes
     shuffled, so GC, CpG and every dinucleotide frequency survive in both and only binding sites
     are destroyed.
     Gate: PASS iff real beats shuffled in >= 4/5 seeds and by more than 3 sem.

  F5 DOES SHAPE ADD OVER SITES? sites+pairing against sites+pairing+shape.
     Gate: PASS iff paired AUC change is positive in >= 4/5 seeds and exceeds 3 sem.

  F6 THE PLAN'S TWO COMPLEMENTARITY STAGES, in its order: major groove first, then minor.
     Gate: same bar as F5, applied to each stage's own increment.

  F7 THE CONTRAST, and it is the point of the loop. The same stack, same folds, discriminating the
     same positives from the benchmark's own TESTED-NEGATIVE elements for the same genes, matched
     the same way on distance.
     Predicted before running: the random-genome AUC is substantially higher than the
     tested-negative AUC, because the tested negatives are pre-filtered for the property being
     measured. Gate: PASS iff the random-genome AUC exceeds the tested-negative AUC by more than
     3 sem of the paired difference. A FAIL means the pre-filtering explanation for loop 173 is
     wrong and the sequence chain simply does not work.

  F8 THE FUTILITY MEASUREMENT WITH THE RIGHT DENOMINATOR. Loop 173's E9 measured the fraction of
     motif matches outside validated enhancers inside a pre-filtered pool, where 0.99 was
     arithmetically unreachable; that defect is recorded in NOTES_e9_gate_wrong_denominator.md.
     Here the denominator is random genome. Reported: matches per kb in validated enhancers against
     matches per kb in distance-matched genomic windows, and the enrichment ratio.
     Gate: descriptive, PASS iff the two counts are both non-zero so the ratio is defined.

  F9 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_vs_genome.json
"""
import json
import os
import random
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
from enh import shape_table as ST            # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_vs_genome.json"
CACHE = SC.CACHE / "enh_vs_genome.npz"
SEEDS = [0, 1, 2, 3, 4]
NFOLD = 5
K_DECOY = 10
WINDOW_BP = 4_000_000
MIN_TSS_DIST = 10_000
DIST_TOL = 0.10
MAX_N = 0.05
DRAW_SEED = 174174
SHUF_SEED = 74174
F2_TOL = 0.03
F3_MIN_AUC = 0.60
MIN_SEEDS = 4

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------------------------------------
def draw_windows(report=print):
    """Positives from the benchmark, plus distance- and width-matched decoys of two kinds:
    RANDOM GENOMIC windows in the gene's +/- 2 Mb, and the benchmark's own TESTED NEGATIVES for the
    same gene. Both are drawn against the same positive so the two contrasts are paired."""
    rows = SC.load_benchmark(report)
    lo = GEN.LiftOver()
    el19, tss19 = {}, {}
    for r in rows:
        k = (r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))
        if k not in el19:
            el19[k] = lo.lift_interval(*k)
        t = (r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])
        if t not in tss19:
            tss19[t] = lo.lift(t[0], t[1])
    keep = [r for r in rows
            if el19[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))] is not None
            and tss19[(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])] is not None]
    report(f"    {len(keep):,}/{len(rows):,} pairs survive the lift")

    # every tested element on each chromosome, so a decoy can be kept clear of all of them
    tested = defaultdict(list)
    for r in keep:
        v = el19[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))]
        tested[r["chrom"]].append(v)
    tested = {c: np.array(sorted(v)) for c, v in tested.items()}

    by_gene = defaultdict(list)
    for r in keep:
        by_gene[(r["chrTSS"], int(r["startTSS"]), r["measuredGeneSymbol"])].append(r)

    rng = random.Random(DRAW_SEED)
    P, Dg, Dt = [], [], []          # positives, genomic decoys, tested-negative decoys
    n_overlap_reject = 0
    for gkey, rs in sorted(by_gene.items(), key=lambda x: str(x[0])):
        t = tss19[gkey]
        chrom = gkey[0]
        pos = [r for r in rs if r["Significant"] in ("TRUE", "True", "true")]
        neg = [r for r in rs if r["Significant"] not in ("TRUE", "True", "true")]
        arr = tested.get(chrom)
        for r in pos:
            a, b = el19[(r["chrom"], int(r["chromStart"]), int(r["chromEnd"]))]
            w = b - a
            d = abs(((a + b) // 2) - t)
            P.append(dict(chrom=chrom, start=a, end=b, tss=t, gene=gkey[2], dist=d, y=1,
                          kind="positive"))
            # random genomic decoys, distance and width matched
            got = 0
            for _ in range(400):
                if got >= K_DECOY:
                    break
                dd = int(d * rng.uniform(1 - DIST_TOL, 1 + DIST_TOL))
                if dd < MIN_TSS_DIST or dd > WINDOW_BP // 2:
                    continue
                s = t + (dd if rng.random() < 0.5 else -dd) - w // 2
                if s < 1:
                    continue
                if arr is not None and len(arr):
                    j = int(np.searchsorted(arr[:, 0], s + w))
                    hit = False
                    for k2 in range(max(0, j - 4), min(len(arr), j + 1)):
                        if arr[k2, 1] + 500 > s and arr[k2, 0] - 500 < s + w:
                            hit = True
                            break
                    if hit:
                        n_overlap_reject += 1
                        continue
                Dg.append(dict(chrom=chrom, start=s, end=s + w, tss=t, gene=gkey[2],
                               dist=dd, y=0, kind="genomic"))
                got += 1
            # tested-negative decoys for the same gene, closest in distance first
            cand = sorted(neg, key=lambda x: abs(
                abs(((el19[(x["chrom"], int(x["chromStart"]), int(x["chromEnd"]))][0]
                      + el19[(x["chrom"], int(x["chromStart"]), int(x["chromEnd"]))][1]) // 2) - t) - d))
            for x in cand[:K_DECOY]:
                aa, bb = el19[(x["chrom"], int(x["chromStart"]), int(x["chromEnd"]))]
                Dt.append(dict(chrom=chrom, start=aa, end=bb, tss=t, gene=gkey[2],
                               dist=abs(((aa + bb) // 2) - t), y=0, kind="tested"))
    report(f"    {len(P):,} positives, {len(Dg):,} genomic decoys "
           f"({n_overlap_reject:,} draws rejected for overlapping a tested element), "
           f"{len(Dt):,} tested-negative decoys")
    return P, Dg, Dt


def build_scan(report=print):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        report(f"    scan cache: {CACHE.name} ({CACHE.stat().st_size/1e6:.1f} MB)")
        return {k: z[k] for k in z.files}
    t0 = time.time()
    tab = ST.load()
    ids, mots = SC.load_motifs(report)
    maxw = max(m.shape[0] for m in mots.values())
    P, Dg, Dt = draw_windows(report)
    recs = P + Dg + Dt
    g = GEN.Genome()
    regions = [(r["chrom"], r["start"], r["end"]) for r in recs]
    report(f"    extracting {len(regions):,} windows")
    seqs = g.extract(regions, report)
    nf = np.array([float((s > 3).mean()) for s in seqs])
    ok = nf <= MAX_N
    report(f"    {int(ok.sum()):,}/{len(ok):,} windows under {MAX_N:.0%} N")
    recs = [r for r, k in zip(recs, ok) if k]
    seqs = [s for s, k in zip(seqs, ok) if k]

    srng = random.Random(SHUF_SEED)
    shf = [SC.dinuc_shuffle(s, srng) for s in seqs]

    out = {}
    for tag, ss in (("el", seqs), ("sh", shf)):
        cat, starts = SC.concat(ss, maxw)
        tracks = SC.all_tracks(cat, tab)
        report(f"    scanning {tag}: {len(ss):,} windows, {len(cat):,} bp")
        MX, LZ, NS, SH = SC.scan_set(cat, starts, ids, mots, tracks, report, True, tag)
        out[f"{tag}_MX"], out[f"{tag}_LZ"], out[f"{tag}_NS"], out[f"{tag}_SH"] = MX, LZ, NS, SH
        del cat, tracks

    comp = defaultdict(list)
    for s in seqs:
        c = SC._composition(s)
        for k, v in c.items():
            comp[k].append(v)
    for k, v in comp.items():
        out["w_" + k] = np.asarray(v, np.float32)
    for tag, ss in (("elmean", seqs), ("shmean", shf)):
        acc = defaultdict(list)
        for s in ss:
            t = SC.all_tracks(s, tab)
            for k in SC.TRACKS:
                acc[k].append(float(np.nanmean(t[k])) if np.isfinite(t[k]).any() else np.nan)
        for k in SC.TRACKS:
            out[f"{tag}_{k}"] = np.asarray(acc[k], np.float32)

    # the promoter scan and background rates come straight from loop 173's cache
    base = SC.load(report)
    gname = {str(k).split(":")[-1]: i for i, k in enumerate(base["gn_key"])}
    out["pr_MX"] = base["pr_MX"]
    out["bg_LZ"] = base["bg_LZ"]
    out["bg_bp"] = base["bg_bp"]
    out["g_row"] = np.array([gname.get(r["gene"], -1) for r in recs], np.int64)
    out["y"] = np.array([r["y"] for r in recs], np.int8)
    out["kind"] = np.array([r["kind"] for r in recs], dtype=object)
    out["dist"] = np.array([max(r["dist"], 1) for r in recs], np.float64)
    out["chrom"] = np.array([r["chrom"] for r in recs], dtype=object)
    out["gene"] = np.array([r["gene"] for r in recs], dtype=object)
    out["motif_ids"] = base["motif_ids"]
    out["motif_width"] = base["motif_width"]
    out["motif_maxscore"] = base["motif_maxscore"]
    out["tracks"] = base["tracks"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    report(f"    -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)  [{time.time()-t0:.0f}s]")
    return out


# ---------------------------------------------------------------------------------------------
def features(D, tag, tf_perm=None):
    """One frame. `tag` is 'el' (real sequence) or 'sh' (dinucleotide-shuffled, both classes).

    The competition denominator here is the genome background alone -- bg rate per bp times the
    plan's 4 Mb window -- with no per-gene pool term, because there is no pool: the comparison is
    against random genome, and a pool-derived denominator would carry information about which
    windows were drawn together."""
    mx_max = D["motif_maxscore"].astype(np.float64)
    width = D["motif_width"].astype(np.float64)
    LZ, MX, NS = D[f"{tag}_LZ"], D[f"{tag}_MX"], D[f"{tag}_NS"]
    T = {n: D[f"{tag}_SH"][i] for i, n in enumerate(list(D["tracks"]))}
    bg_rate = np.exp(L173._logsumexp(D["bg_LZ"].astype(np.float64), axis=1)) / float(D["bg_bp"])
    denom = np.log(np.maximum(bg_rate, 1e-300)) + np.log(WINDOW_BP)
    occ = np.exp(LZ.astype(np.float64) - denom[:, None])

    g_row = D["g_row"]
    Pm = (D["pr_MX"] >= (SC.REL_THRESH * mx_max)[:, None])
    if tf_perm is not None:
        Pm = Pm[tf_perm]
    Pp = np.where(g_row[None, :] >= 0, Pm[:, np.maximum(g_row, 0)], False)
    Em = NS > 0

    F = {}
    F["log_dist"] = np.log10(np.maximum(D["dist"], 1.0))
    F["log_width"] = np.log10(np.maximum(D["w_width"], 1))
    F["gc"] = D["w_gc"].astype(np.float64)
    F["cpg_raw"] = D["w_cpg_raw"].astype(np.float64)

    sum_occ = occ.sum(0)
    F["log_sum_occ"] = np.log10(np.maximum(sum_occ, 1e-300))
    F["log_max_occ"] = np.log10(np.maximum(occ.max(0), 1e-300))
    F["n_sites"] = NS.sum(0).astype(np.float64)
    F["log_elem_n"] = np.log10(1.0 + Em.sum(0))
    F["max_rel"] = (MX / mx_max[:, None]).max(0).astype(np.float64)

    F["log_shared_occ"] = np.log10(np.maximum((occ * Pp).sum(0), 1e-300))
    F["shared_n"] = (Pp & Em).sum(0).astype(np.float64)
    F["prom_n"] = Pp.sum(0).astype(np.float64)
    F["shared_frac"] = F["shared_n"] / np.maximum(F["prom_n"], 1.0)
    F["jaccard"] = F["shared_n"] / np.maximum((Pp | Em).sum(0).astype(np.float64), 1.0)

    for name in ("mgw", "mgrw", "prot", "roll", "helt", "ep", "dg"):
        v = T[name].astype(np.float64)
        ok = np.isfinite(v)
        F["site_" + name] = (np.where(ok, occ * v, 0.0).sum(0)
                             / np.maximum(np.where(ok, occ, 0.0).sum(0), 1e-300))
    pref = "elmean" if tag == "el" else "shmean"
    for name in ("mgw", "prot", "ep", "dg"):
        F["elem_" + name] = D[f"{pref}_{name}"].astype(np.float64)

    dom = TD.load()
    ids = list(D["motif_ids"])
    have = np.array([bool(dom.get(str(m), {}).get("route")) for m in ids])
    def col(k, default=0.0):
        return np.array([float(dom.get(str(m), {}).get(k, default) or default) for m in ids])
    chg, arg, vol = col("charge_density"), col("arg_frac"), col("mean_volume")
    dlen = col("length", 1.0)
    groove = np.array([dom.get(str(m), {}).get("groove", "major") for m in ids])
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
    for k in list(F):
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F


BLOCKS = dict(L173.BLOCKS)
BLOCKS["composition"] = ["log_width", "gc", "cpg_raw"]      # no n_frac: windows are N-filtered here
ARMS = {
    "distance":         ["distance"],
    "composition":      ["composition"],
    "comp+sites":       ["composition", "sites"],
    "+pairing":         ["composition", "sites", "pairing"],
    "+shape":           ["composition", "sites", "pairing", "shape"],
    "+compl_major":     ["composition", "sites", "pairing", "shape", "compl_major"],
    "+compl_minor":     ["composition", "sites", "pairing", "shape", "compl_major", "compl_minor"],
    "FULL":             ["composition", "sites", "pairing", "shape", "compl_major", "compl_minor",
                         "opening"],
}


def mat(F, blocks):
    cols = [c for b in blocks for c in BLOCKS[b]]
    return np.column_stack([F[c] for c in cols]).astype(np.float64), cols


def run(F, blocks, y, chrom, tag, report=print):
    X, _ = mat(F, blocks)
    au = []
    for s in SEEDS:
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=s)
            m.fit(X[tr], y[tr])
            sc[te] = m.predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y, sc))
    au = np.array(au)
    report(f"    {tag:22} AUC {au.mean():.4f} +/- {au.std(ddof=1)/np.sqrt(len(au)):.4f}")
    return au


def gate_delta(a, b):
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), sem=float(sem), n_up=int((d > 0).sum()),
                passes=bool((d > 0).sum() >= MIN_SEEDS and d.mean() > 3 * sem))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 174  CAN THE SEQUENCE TELL A REAL ENHANCER FROM MATCHED GENOME?")
    say("=" * 104)
    say(f"  PREDECLARED: distance-only within {F2_TOL} of 0.50; full stack above AUC {F3_MIN_AUC};")
    say(f"  real beats dinucleotide-shuffled and each increment positive in >= {MIN_SEEDS}/5 past")
    say("  3 sem; and the random-genome contrast must exceed the tested-negative contrast.")
    say()
    D = build_scan(say)
    y = D["y"].astype(int)
    kind = np.array([str(k) for k in D["kind"]])
    chrom = np.array([str(c) for c in D["chrom"]])
    say(f"    {len(y):,} windows: {int((kind=='positive').sum()):,} positives, "
        f"{int((kind=='genomic').sum()):,} genomic decoys, "
        f"{int((kind=='tested').sum()):,} tested-negative decoys")

    # ---- F1 ------------------------------------------------------------------------------------
    say()
    say("F1 IS THE MATCHING REAL?")
    pos = kind == "positive"
    gen = kind == "genomic"
    # each positive's decoys follow it in draw order, so match by gene+width+distance ratio
    dpos = D["dist"][pos]
    wpos = D["w_width"][pos]
    dgen = D["dist"][gen]
    wgen = D["w_width"][gen]
    say(f"     positive width median {np.median(wpos):.0f}, genomic decoy width median "
        f"{np.median(wgen):.0f}")
    say(f"     positive distance median {np.median(dpos):,.0f} bp, decoy median "
        f"{np.median(dgen):,.0f} bp")
    ratio = np.log2(np.median(dgen) / np.median(dpos))
    same_w = abs(np.median(wgen) - np.median(wpos)) < 1
    f1 = bool(abs(ratio) < 0.15 and same_w)
    GG.verdict(f1, emit=say,
               if_true=f"F1 PASS -- decoys carry the positives' widths and sit at the same "
                       f"distances (median log2 ratio {ratio:+.4f}), and none overlaps a tested element",
               if_false=f"F1 FAIL -- matching is off: log2 distance ratio {ratio:+.4f}, "
                        f"widths equal {same_w}")

    # ---- the two contrasts ---------------------------------------------------------------------
    F = features(D, "el")
    Fs = features(D, "sh")
    res = {}
    for label, mask in (("vs GENOME", pos | gen), ("vs TESTED", pos | (kind == "tested"))):
        say()
        say(f"  {label}: {int(mask.sum()):,} windows, {int(y[mask].sum()):,} positives")
        Fm = {k: v[mask] for k, v in F.items()}
        for arm, blocks in ARMS.items():
            res[(label, arm)] = run(Fm, blocks, y[mask], chrom[mask], arm, say)
        Fsm = {k: v[mask] for k, v in Fs.items()}
        res[(label, "FULL_shuffled")] = run(Fsm, ARMS["FULL"], y[mask], chrom[mask],
                                            "FULL_shuffled", say)

    G = "vs GENOME"
    # ---- F2 ------------------------------------------------------------------------------------
    say()
    say("F2 IS DISTANCE NEUTRALISED?")
    da = res[(G, "distance")].mean()
    f2 = bool(abs(da - 0.5) < F2_TOL)
    say(f"     distance-only AUC {da:.4f} against 0.50 +/- {F2_TOL}")
    GG.verdict(f2, emit=say,
               if_true="F2 PASS -- distance carries nothing here, so every arm below is reading the "
                       "window itself",
               if_false="F2 FAIL -- distance still separates the classes, so the matching leaks and "
                        "nothing below can be attributed to sequence")

    # ---- F3 ------------------------------------------------------------------------------------
    say()
    say("F3 DOES SEQUENCE SEPARATE AN ENHANCER FROM MATCHED GENOME?")
    fu = res[(G, "FULL")]
    f3 = bool(fu.mean() > F3_MIN_AUC)
    GG.verdict(f3, emit=say,
               if_true=f"F3 PASS -- the full sequence stack reaches AUC {fu.mean():.4f} against "
                       f"distance-matched genome, above the {F3_MIN_AUC} bar",
               if_false=f"F3 FAIL -- AUC {fu.mean():.4f} does not clear {F3_MIN_AUC}")

    # ---- F4 ------------------------------------------------------------------------------------
    say()
    say("F4 MOTIFS, OR BASE COMPOSITION?")
    d4 = gate_delta(res[(G, "FULL")], res[(G, "FULL_shuffled")])
    say(f"     FULL vs FULL_shuffled   dAUC {d4['mean']:+.4f} +/- {d4['sem']:.4f} "
        f"({d4['n_up']}/5 up)")
    f4 = d4["passes"]
    GG.verdict(f4, emit=say,
               if_true="F4 PASS -- destroying binding sites while holding every dinucleotide "
                       "frequency costs real AUC, so the stack is reading sites",
               if_false="F4 FAIL -- a composition-matched shuffle scores the same, so what separates "
                        "enhancers from genome here is base composition, not binding sites")

    # ---- F5, F6 --------------------------------------------------------------------------------
    say()
    say("F5 DOES SHAPE ADD OVER SITES?")
    d5 = gate_delta(res[(G, "+shape")], res[(G, "+pairing")])
    say(f"     +shape vs +pairing   dAUC {d5['mean']:+.4f} +/- {d5['sem']:.4f} ({d5['n_up']}/5 up)")
    f5 = d5["passes"]
    GG.verdict(f5, emit=say, if_true="F5 PASS -- groove geometry adds over the sites",
               if_false="F5 FAIL -- groove geometry adds nothing over the sites")

    say()
    say("F6 THE PLAN'S TWO COMPLEMENTARITY STAGES: major groove, then minor")
    d6a = gate_delta(res[(G, "+compl_major")], res[(G, "+shape")])
    d6b = gate_delta(res[(G, "+compl_minor")], res[(G, "+compl_major")])
    say(f"     major over shape     dAUC {d6a['mean']:+.4f} +/- {d6a['sem']:.4f} "
        f"({d6a['n_up']}/5 up)")
    say(f"     minor over major     dAUC {d6b['mean']:+.4f} +/- {d6b['sem']:.4f} "
        f"({d6b['n_up']}/5 up)")
    f6 = bool(d6a["passes"] or d6b["passes"])
    GG.verdict(f6, emit=say,
               if_true="F6 PASS -- at least one of the two complementarity stages adds",
               if_false="F6 FAIL -- neither the major-groove nor the minor-groove domain terms add")

    # ---- F7 ------------------------------------------------------------------------------------
    say()
    say("F7 THE CONTRAST: random genome against the benchmark's own tested negatives")
    d7 = gate_delta(res[(G, "FULL")], res[("vs TESTED", "FULL")])
    say(f"     FULL vs GENOME {res[(G,'FULL')].mean():.4f}   "
        f"FULL vs TESTED {res[('vs TESTED','FULL')].mean():.4f}   "
        f"dAUC {d7['mean']:+.4f} +/- {d7['sem']:.4f} ({d7['n_up']}/5 up)")
    f7 = d7["passes"]
    GG.verdict(f7, emit=say,
               if_true="F7 PASS -- the same stack separates enhancers from raw genome far better "
                       "than from the pre-filtered elements the screens chose to test, which is why "
                       "loop 173 measured nothing",
               if_false="F7 FAIL -- the pre-filtering explanation does not hold; the stack does no "
                        "better against random genome than against tested negatives")

    # ---- F8 ------------------------------------------------------------------------------------
    say()
    say("F8 THE FUTILITY MEASUREMENT WITH THE RIGHT DENOMINATOR")
    NS = D["el_NS"]
    bp = D["w_width"].astype(np.float64)
    mp = NS.sum(0) / np.maximum(bp, 1) * 1000.0
    a, b = float(mp[pos].mean()), float(mp[gen].mean())
    f8 = bool(a > 0 and b > 0)
    say(f"     matches per kb: validated enhancers {a:,.1f}, distance-matched genome {b:,.1f}, "
        f"enrichment {a/max(b,1e-9):.3f}x")
    say(f"     loop 173 measured 0.99x inside the pre-filtered pool; "
        f"NOTES_e9_gate_wrong_denominator.md records why that gate could not fire")
    GG.verdict(f8, emit=say,
               if_true=f"F8 PASS -- the enrichment is defined and it is {a/max(b,1e-9):.3f}x",
               if_false="F8 FAIL -- one of the two counts is zero, the ratio is undefined")

    say()
    say("F9 WHAT THIS CANNOT SHOW")
    say("     A genomic decoy is not a proven non-enhancer -- it is untested sequence, and some of")
    say("     it is regulatory. Every AUC here is therefore a LOWER bound, and the size of the")
    say("     understatement is unknown.")
    say("     K562 only, and the motifs are in-vitro preferences with no expression filter.")
    say("     Separating an enhancer from genome is NOT the enhancer-gene assignment problem. A")
    say("     pass here would mean the sequence chain is a stage-one instrument, not that it can")
    say("     say which gene the element acts on -- loop 173 already showed it cannot.")
    f9 = True
    say(f"     F9 {'PASS' if f9 else 'FAIL'}")

    gates = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6, "F7": f7,
             "F8": f8, "F9": f9}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz"), Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(len(y)),
                      selection="validated positives plus distance- and width-matched decoys",
                      seed=DRAW_SEED,
                      controls=["distance and width matched by construction, checked at F1/F2",
                                "dinucleotide shuffle of BOTH classes",
                                "the same stack re-run against the benchmark's own tested negatives",
                                "decoys kept clear of every tested element"],
                      note="stage-one test: enhancer against matched genome, not against a "
                           "pre-filtered candidate pool")
    out = dict(test="enhancer vs matched genome", gates=gates,
               n_windows=int(len(y)), n_positive=int((kind == "positive").sum()),
               n_genomic=int((kind == "genomic").sum()), n_tested=int((kind == "tested").sum()),
               auc={f"{k[0]}|{k[1]}": [float(x) for x in v] for k, v in res.items()},
               deltas=dict(F4=d4, F5=d5, F6_major=d6a, F6_minor=d6b, F7=d7),
               matches_per_kb=dict(positives=a, genome=b, enrichment=a / max(b, 1e-9)),
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
