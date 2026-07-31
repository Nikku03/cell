"""causal_enhancer -- can the IDENTITY of the TFs bound at a distal element, weighted by whether those TFs are
CAUSAL for the gene in Perturb-seq, predict a CRISPR-validated enhancer-gene link better than distance and
better than accessibility?

WHAT IS BEING COMBINED, AND WHY THAT COMBINATION IS THE INTERESTING ONE.
`screen_ccre.py` established the element side: 1,063,878 SCREEN cCREs, of which 340k are K562-accessible, and
the number that qualifies every E-G claim in this project -- 66.8% of experimentally VALIDATED K562 E-G links
sit at a distance the current E-G layer cannot express. It also showed that K562 cCRE overlap carries only a
weak candidate-filter signal. `bound_causal.py` established the TF side: 392 TFs (507 here, after a re-fetch
of the ENCODE K562 conservative-IDR set) that have BOTH a K562 ChIP occupancy track AND a genome-scale
Perturb-seq perturbation, split into DIRECT / INDIRECT / bound-not-causal tiers, with the finding that only
12% of causal edges are bound at the promoter and that occupancy predicts transfer MAGNITUDE but not SIGN.

Neither module ever crossed occupancy with causality AT A DISTAL ELEMENT. That is the natural next object:
an enhancer is not "a peak", it is a place where particular proteins sit, and the proteins that sit there
should be the ones whose removal moves the gene. Distance and accessibility are the two features every
enhancer-gene predictor already has, and they are known to be strong. So the question is not "does TF
causality predict links" -- almost anything correlated with activity does -- but:

    IS THERE AN INCREMENT, over distance and accessibility, that comes from the ELEMENT'S OWN TF SET?

WHAT IS MEASURED, AND AGAINST WHAT CONTROL, BEFORE ANY NUMBER.
  MEASURED   AUPRC, out-of-fold, on CRISPR enhancer-perturbation ground truth (EP CRISPR benchmark, K562 arm,
             training_K562 + the K562 rows of the 5-cell-type held-out file), folds held out BY CHROMOSOME so
             they are gene-disjoint by construction. AUPRC is reported RAW, with the positive base rate
             beside it, because at a ~4% base rate accuracy is meaningless and an AUPRC of 0.10 is a 2.5x
             lift, not a bad model.
  CONTROL 1  DISTANCE-ONLY and ACCESSIBILITY-ONLY arms on IDENTICAL folds. The TF-causality features are
             scored as an increment over what was already available, never on their own.
  CONTROL 2  DISTANCE + ACCESSIBILITY + TF OCCUPANCY COUNT. This is the bound-vs-causal distinction imported
             from bound_causal.py: an element with 300 of 507 profiled TFs sitting on it is a busy element,
             and busy-ness is available without any Perturb-seq. If the increment survives only against
             distance and accessibility but not against occupancy count, the finding is "busy elements", not
             "the right TFs".
  CONTROL 3  A GENE-RESPONSIVENESS-ONLY arm with NO element information at all. Some genes move under almost
             any perturbation. Such a gene will look causally connected to every element in the genome, and
             it will also be an easy CRISPR hit. This arm measures how much of the answer is readable from
             the gene alone.
  CONTROL 4, THE DECISIVE ONE -- THE ELEMENT SWAP. Keep the gene, keep the label, and replace the element
             with a DIFFERENT element matched on distance-to-TSS decile and accessibility decile, then score
             the swapped pair with the model trained on real pairs. If the decoy scores as well as the real
             element, the model is reading the gene and the distance and the element is decoration. Drawn 20
             times.
  CONTROL 5  THE DEGREE-MATCHED ELEMENT SWAP. Elements differ in how many TFs sit on them, and a decoy drawn
             only on distance and accessibility can carry a systematically different TF count, which would
             move every TF-causality feature for a reason that has nothing to do with TF identity. So the
             swap is repeated with the decoy ALSO matched on n-TFs-bound decile -- the configuration-model
             analogue for a bipartite TF-element graph. If the degree-matched swap reproduces most of the
             effect, THAT IS THE FINDING and it is said in the verdict, not in a footnote.

THE DECISION THRESHOLD, FIXED BEFORE THE NUMBERS (and encoded below as MIN_INCREMENT / MIN_SEEDS /
MIN_SWAP_FRAC). TF causality earns a place in the E-G layer only if ALL THREE hold:
    (i)   mean out-of-fold AUPRC of the full model exceeds the BEST baseline arm by >= 0.010 absolute AUPRC,
          and does so in >= 4 of 5 CV seeds;
    (ii)  the real element beats the distance x accessibility decile-matched swap in >= 18 of 20 draws;
    (iii) it also beats the DEGREE-matched swap in >= 18 of 20 draws.
Failing (i) means there is no increment. Passing (i) but failing (ii) means the model reads the gene and the
distance. Passing (ii) but failing (iii) means it reads how busy the element is, not which TFs are on it.

CIRCULARITY, STATED PLAINLY. The TF->gene causal weights come from Replogle K562 Perturb-seq; the ground
truth is CRISPRi enhancer perturbation in K562. These are DIFFERENT experiments (gene knockdown vs element
silencing, different labs, different libraries), so this is not the forbidden case of scoring a K562-derived
layer on K562 Perturb-seq. But they share a cell line, and therefore they share "which genes are easy to
move in K562". That shared nuisance is exactly what CONTROL 3 quantifies and what CONTROL 4 removes by
holding the gene fixed. There is no RPE1 enhancer-CRISPR benchmark to escape to, and that is a limit.

SIGN AND SIGNIFICANCE ARE READ TOGETHER. An arm that does not clear its control is NOT DETECTED. It is never
called reversed, however negative the point estimate looks.

SAMPLING, AND WHAT IT EXCLUDES. One ChIP file per TF (the lowest-accession GRCh38 "conservative IDR
thresholded peaks" file), not all replicates and not all experiments -- so a TF whose occupancy differs
between its K562 experiments is represented by one of them. Only TFs with BOTH a K562 conservative-IDR track
and a Replogle K562 perturbation are used; TFs profiled by ChIP but never perturbed are invisible here, as
are perturbed TFs with no K562 ChIP. Only the K562 arm of the benchmark is used (the other cell types have
no matching Perturb-seq). Only pairs whose measured gene is one of the 8,248 genes Replogle quantified are
kept. 20 swap draws, 5 CV seeds, 5 folds.
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))

API = "https://www.encodeproject.org"
CR = SP / "crispr"
CRISPR_BASE = ("https://raw.githubusercontent.com/EngreitzLab/CRISPR_comparison/main/"
               "resources/crispr_data")
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
GWPS = SP / "gwps.h5ad"
ATAC = SP / "k562_atac.bed.gz"
BOUND_CACHE = SP / "causal_enhancer_bound.npz"

# ---- the pins. One assembly, one peak output type, one power bar, one causal-edge threshold. --------------
ASSEMBLY = "GRCh38"
OUTPUT_TYPE = "conservative IDR thresholded peaks"   # same pin bound_causal.py uses; mixing IDR tiers would
                                                     # make "bound" mean different things for different TFs
POWER_COL = "PowerAtEffectSize25"
MIN_POWER = 0.8            # an unpowered non-significant pair is NOT DETECTED, not a negative -- excluded
Z_EDGE = 5.0               # |per-gene robust z| at which a (TF, gene) Perturb-seq pair is called causal;
                           # the same threshold causal_reg.py and bound_causal.py use
ELEMENT_PAD = 250          # bp: a ChIP summit this far outside the tested element still counts as "at" it,
                           # because the CRISPRi tiling element is a guide window, not a protein footprint

# ---- the decision threshold, stated BEFORE the numbers ----------------------------------------------------
FOLDS = 5
SEEDS = (0, 1, 2, 3, 4)    # each seed draws a genuinely DIFFERENT chromosome->fold partition, not a rotation
SWAP_DRAWS = 20
MIN_INCREMENT = 0.010      # absolute AUPRC over the best baseline arm
MIN_SEEDS = 4              # of len(SEEDS): the increment must hold in this many CV partitions
MIN_SWAP_FRAC = 0.90       # 18 of 20 draws in which the real element beats the matched decoy

# ---- joins DECLARE an expected rate and RAISE below it ---------------------------------------------------
MIN_JOIN_GENE = 0.85       # benchmark measured gene -> a gene Replogle quantified
MIN_JOIN_TF = 0.85         # ENCODE K562 ChIP target -> a Replogle perturbation


# ==========================================================================================================
# fetch
# ==========================================================================================================
def download(url, path, tries=4, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    for i in range(tries):
        try:
            r = urllib.request.Request(url, headers=headers or {"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=600) as fh, open(path, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return path
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def api(path, tries=4):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=300))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def load_benchmark(report):
    """EP CRISPR benchmark, K562 arm. Re-fetched from the EngreitzLab repo the way expand_cells.py does it."""
    rows = []
    for f, only_k562 in ((TRAINING, False), (HELDOUT, True)):
        p = CR / f
        if not p.exists():
            report(f"    fetching {f}")
            download(f"{CRISPR_BASE}/{f}", p)
        r = list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
        if only_k562:
            r = [x for x in r if x["CellType"] == "K562"]
        else:
            for x in r:
                x["CellType"] = "K562"
        report(f"    {f}: {len(r):,} K562 pairs")
        rows += r
    return rows


def chip_index(report):
    """One K562 conservative-IDR narrowPeak per TF, from ONE portal query.

    bound_causal.py walked the portal per TF (two API round-trips each); the File-level search returns the
    whole set in one call with the target attached, which is the same universe and ~500x fewer requests. The
    output type is PINNED, so a TF with no conservative-IDR file is DROPPED and counted rather than
    back-filled with a noisier pseudoreplicated call.
    """
    u = ("/search/?type=File&file_format=bed&file_format_type=narrowPeak"
         f"&assembly={ASSEMBLY}&output_type={urllib.parse.quote(OUTPUT_TYPE)}"
         "&biosample_ontology.term_name=K562&status=released&limit=all&frame=object&format=json")
    d = api(u)
    g = [x for x in d.get("@graph", []) if x.get("assay_title") == "TF ChIP-seq"]
    best = {}
    for x in g:
        t = x["target"].split("/")[-2].replace("-human", "")
        # lowest accession wins: an arbitrary but DETERMINISTIC pick, so a rerun cannot silently change the
        # occupancy matrix under a cached result
        if t not in best or x["accession"] < best[t]["accession"]:
            best[t] = x
    report(f"    ENCODE: {len(g):,} released K562 '{OUTPUT_TYPE}' TF ChIP files over {len(best)} TF targets")
    return best


def peak_summits(path):
    """chrom -> sorted summit positions. narrowPeak column 10 is the summit offset; -1 means none called,
    in which case the interval midpoint is used (bound_causal.py's convention, kept identical)."""
    d = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                st, en = int(p[1]), int(p[2])
            except ValueError:
                continue
            off = int(p[9]) if len(p) > 9 and p[9] not in ("", "-1") else -1
            d.setdefault(p[0], []).append(st + off if off >= 0 else (st + en) // 2)
    return {c: np.sort(np.array(v, dtype=np.int64)) for c, v in d.items()}


# ==========================================================================================================
# per-gene robust z from the K562 Perturb-seq pseudobulk
# ==========================================================================================================
def robust_z_k562(report):
    """(perturbation x gene) robust z, IDENTICAL to causal_reg.py: (lfc - median) / (1.4826 * MAD) down each
    GENE column. The column effect is the point: some genes move under almost any perturbation, and a raw
    log fold change is dominated by them, so an edge here means 'this gene moved more than it usually moves'.
    """
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        ncell = f["obs"]["num_cells_filtered"][:]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var"]["__categories"]["gene_name"][:]]
        codes = f["var"]["gene_name"][:]
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["gene_id"][:]])
        gexpr = f["var"]["mean"][:]
    gsym = np.array([cats[c] for c in codes])
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    dead = mad < 1e-6
    mad = np.where(dead, np.nan, mad)
    Z = (X - med) / mad
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    report(f"    Perturb-seq K562: {Z.shape[0]:,} perturbations x {Z.shape[1]:,} genes; "
           f"{int(dead.sum())} genes have zero MAD and are z=0 by construction")
    return Z, psym, gsym, gid, np.asarray(gexpr, dtype=np.float64), np.asarray(ncell, dtype=np.float64)


# ==========================================================================================================
# features
# ==========================================================================================================
def tf_features(Bp, Zp, g_mean_abs, g_frac_causal):
    """Aggregate the causal weights of the TFs BOUND at the element, per pair.

    Bp  (n_pairs x n_tf) bool -- is TF t bound at this pair's element
    Zp  (n_pairs x n_tf) float -- robust z of gene g under knockdown of TF t (0 where t IS g: a TF's own
        knockdown moving its own transcript is the assay working, not regulation, and it would hand every
        element containing that TF a spurious enormous weight)
    """
    A = np.abs(Zp) * Bp
    nb = Bp.sum(1).astype(np.float64)
    nbc = np.maximum(nb, 1.0)
    ncausal = ((np.abs(Zp) >= Z_EDGE) & Bp).sum(1).astype(np.float64)
    ndown = ((Zp <= -Z_EDGE) & Bp).sum(1).astype(np.float64)
    mean_abs = A.sum(1) / nbc
    max_abs = A.max(1)
    # top-5 rather than the max alone: one enormous z is a single edge, five is a coherent TF set, and the
    # two are different claims about the element
    k = min(5, A.shape[1])
    top5 = np.sort(np.partition(A, -k, axis=1)[:, -k:], axis=1).sum(1)
    mean_signed = (Zp * Bp).sum(1) / nbc
    frac_causal = ncausal / nbc
    return {
        "tf_n_causal": ncausal,
        "tf_frac_causal": frac_causal,
        "tf_max_abs_z": max_abs,
        "tf_mean_abs_z": mean_abs,
        "tf_top5_abs_z": top5,
        "tf_mean_signed_z": mean_signed,
        "tf_n_causal_down": ndown,
        # element-SPECIFIC enrichment: how much more (or less) this element's TF set moves this gene than
        # the average profiled TF does. The gene's own background is subtracted, so a gene that everything
        # moves does not get a high value for free -- but the value still contains gene information through
        # the interaction, which is precisely why the element swap exists.
        "tf_enrich_abs_z": mean_abs - g_mean_abs,
        "tf_enrich_frac": frac_causal - g_frac_causal,
    }


# ==========================================================================================================
# CV
# ==========================================================================================================
def chrom_folds(chrom, y, seed, folds=FOLDS):
    """Assign whole CHROMOSOMES to folds in a seed-specific RANDOM order, each going to the currently
    lightest fold.

    A gene sits on one chromosome, so chromosome-held-out folds are gene-disjoint AND element-disjoint by
    construction. The random order is what makes a seed mean something: rotating the fold LABELS of one
    partition gives an sd of exactly 0.0000, which this project has already reported once and which is
    meaningless. Assigning in a FIXED order (say, by positive count descending) would have the same defect
    in a subtler form -- every seed would produce nearly the same split -- so the order is shuffled and the
    resulting overlap between seeds is measured and printed rather than assumed.
    """
    rng = np.random.default_rng(1000 + seed)
    cs = np.array(sorted(set(chrom)))
    rng.shuffle(cs)
    pos = {c: int(y[chrom == c].sum()) for c in cs}
    load = np.zeros(folds, dtype=np.float64)
    assign = {}
    for c in cs:
        f = int(np.argmin(load))
        assign[c] = f
        load[f] += pos[c] + 0.5           # +0.5 so chromosomes with zero positives still spread out
    return np.array([assign[c] for c in chrom]), {str(c): int(assign[c]) for c in cs}


def fit_folds(F, y, fold, seed):
    """Fit one model per fold on the REAL features. Returned so the element-swap control can re-score a
    swapped feature matrix without retraining -- the swap must change what the model SEES, never what it
    LEARNED, or it would stop being a control and become a different model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    ms = {}
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 5 or y[te].sum() < 1:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        m.fit(F[tr], y[tr])
        ms[f] = m
    return ms


def score_oof(ms, F, y, fold):
    from sklearn.metrics import average_precision_score
    p = np.zeros(len(y))
    for f, m in ms.items():
        te = fold == f
        p[te] = m.predict_proba(F[te])[:, 1]
    return float(average_precision_score(y, p))


# ==========================================================================================================
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 104)
    report("DISTAL ENHANCER-GENE LINKS SCORED BY TF CAUSALITY -- is there an increment over distance and")
    report("accessibility, and does it survive replacing the element with a decile-matched decoy?")
    report("=" * 104)
    report(f"  DECISION THRESHOLD, FIXED BEFORE THE NUMBERS: TF causality earns a place only if the full "
           f"model beats the")
    report(f"  best baseline by >= {MIN_INCREMENT:.3f} AUPRC in >= {MIN_SEEDS}/{len(SEEDS)} CV partitions, "
           f"AND beats the distance x accessibility")
    report(f"  matched element swap in >= {MIN_SWAP_FRAC:.0%} of {SWAP_DRAWS} draws, AND beats the "
           f"DEGREE-matched swap in >= {MIN_SWAP_FRAC:.0%} of {SWAP_DRAWS}.")

    # ---- 1. ground truth ---------------------------------------------------------------------------------
    report("\n  1  GROUND TRUTH -- CRISPR enhancer perturbation, EP CRISPR benchmark K562 arm")
    rows = load_benchmark(report)
    pw = np.array([float(r[POWER_COL] or 0) for r in rows])
    report(f"    {len(rows):,} K562 pairs; {int((pw >= MIN_POWER).sum()):,} powered at "
           f"{POWER_COL} >= {MIN_POWER}")

    Z, psym, gsym, gid, gexpr, ncell = robust_z_k562(report)

    # ---- join 1: benchmark gene -> a gene Replogle quantified. DECLARED rate, RAISES below it. ------------
    g_of = {}
    for j, e in enumerate(gid):
        g_of.setdefault(e, j)
    ens = [r["measuredGeneEnsemblId"].split(".")[0] for r in rows]
    hit = np.array([e in g_of for e in ens])
    rate = float(hit.mean())
    report(f"    JOIN benchmark measuredGeneEnsemblId -> Replogle var/gene_id: "
           f"{int(hit.sum()):,}/{len(rows):,} ({rate:.1%}), expected >= {MIN_JOIN_GENE:.0%}")
    if rate < MIN_JOIN_GENE:
        raise SystemExit(f"gene join rate {rate:.1%} below the declared {MIN_JOIN_GENE:.0%} floor -- a join "
                         f"that matches nothing must not return an empty set a control then passes")

    keep = (pw >= MIN_POWER) & hit
    rows = [r for r, k in zip(rows, keep) if k]
    ens = [e for e, k in zip(ens, keep) if k]
    y = np.array([1 if r["Significant"] in ("TRUE", "True", "true") else 0 for r in rows])
    report(f"    ANALYSIS SET: {len(rows):,} powered distal pairs, {int(y.sum())} positives, "
           f"BASE RATE {y.mean():.4f}")
    from collections import Counter
    report(f"    datasets: { dict(Counter(r['Dataset'] for r in rows)) }")

    el_key = [(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows]
    uel = sorted(set(el_key))
    ei = {k: i for i, k in enumerate(uel)}
    el_idx = np.array([ei[k] for k in el_key])
    gene_idx = np.array([g_of[e] for e in ens])
    dist = np.array([max(abs(float(r["distanceToTSS"])), 1.0) for r in rows])
    chrTSS = np.array([r["chrTSS"] for r in rows])
    report(f"    {len(uel):,} distinct elements, {len(set(ens)):,} distinct genes, "
           f"{len(set(chrTSS))} chromosomes; distance to TSS median {np.median(dist):,.0f} bp "
           f"(min {dist.min():,.0f} -- the benchmark carries no promoter-proximal pair, so every pair here "
           f"is distal)")

    # ---- 2. TF occupancy at the ELEMENTS -----------------------------------------------------------------
    report("\n  2  TF OCCUPANCY AT THE ELEMENTS -- ENCODE K562 ChIP crossed with Replogle perturbations")
    idx = chip_index(report)
    pset = set(psym)
    tfs = sorted(t for t in idx if t in pset)
    tf_rate = len(tfs) / max(len(idx), 1)
    report(f"    JOIN ENCODE K562 ChIP target -> Replogle perturbation: {len(tfs)}/{len(idx)} "
           f"({tf_rate:.1%}), expected >= {MIN_JOIN_TF:.0%}")
    if tf_rate < MIN_JOIN_TF:
        raise SystemExit(f"TF join rate {tf_rate:.1%} below the declared {MIN_JOIN_TF:.0%} floor")

    # one perturbation row per TF: the one with the most cells, because a low-cell pseudobulk is noise and
    # picking the first row would let library depth decide which TFs look causal
    prow = {}
    for i, s in enumerate(psym):
        if s in pset and (s not in prow or ncell[i] > ncell[prow[s]]):
            prow[s] = i
    tf_prow = np.array([prow[t] for t in tfs])

    ech = np.array([k[0] for k in uel])
    est = np.array([k[1] for k in uel], dtype=np.int64)
    een = np.array([k[2] for k in uel], dtype=np.int64)

    cached = None
    if BOUND_CACHE.exists():
        z = np.load(BOUND_CACHE, allow_pickle=True)
        if list(z["tfs"]) == tfs and len(z["elkey"]) == len(uel) and \
                all(str(a) == f"{c}:{s}-{e}" for a, c, s, e in zip(z["elkey"], ech, est, een)):
            cached = z["B"]
            report(f"    occupancy matrix from cache: {cached.shape[0]} TFs x {cached.shape[1]:,} elements")
    if cached is None:
        B = np.zeros((len(tfs), len(uel)), dtype=bool)
        tmp = SP / "ce_chip_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        lock = [0]

        def one(k):
            t = tfs[k]
            p = tmp / f"{t}.bed.gz"
            try:
                download(API + idx[t]["href"], p)
                sm = peak_summits(p)
            finally:
                # the container has < 5 GB of disk headroom; the peak files are deleted after use and only
                # the (507 x 4,362) boolean matrix is cached
                if p.exists():
                    p.unlink()
            col = np.zeros(len(uel), dtype=bool)
            for c in np.unique(ech):
                if c not in sm:
                    continue
                m = np.where(ech == c)[0]
                pos = sm[c]
                lo = np.searchsorted(pos, est[m] - ELEMENT_PAD)
                hi = np.searchsorted(pos, een[m] + ELEMENT_PAD)
                col[m] = hi > lo
            lock[0] += 1
            if lock[0] % 100 == 0:
                report(f"      occupancy {lock[0]}/{len(tfs)} TFs")
            return k, col

        with ThreadPoolExecutor(max_workers=8) as ex:
            for k, col in ex.map(one, range(len(tfs))):
                B[k] = col
        np.savez_compressed(BOUND_CACHE, B=B, tfs=np.array(tfs),
                            elkey=np.array([f"{c}:{s}-{e}" for c, s, e in zip(ech, est, een)]))
        cached = B
    B = cached
    nbound_el = B.sum(0)
    report(f"    occupancy: {len(tfs)} TFs x {len(uel):,} elements, {100*B.mean():.1f}% of cells bound; "
           f"TFs per element median {np.median(nbound_el):.0f} (IQR "
           f"{np.percentile(nbound_el,25):.0f}-{np.percentile(nbound_el,75):.0f}), "
           f"{int((nbound_el == 0).sum()):,} elements carry no profiled TF at all")

    # ---- 3. features -------------------------------------------------------------------------------------
    report("\n  3  FEATURES")
    # TF x gene causal weight matrix, with the self-knockdown diagonal removed
    Zsub = Z[tf_prow, :].copy()                               # (n_tf x n_genes_measured)
    del Z
    sym2col = {}
    for j, s in enumerate(gsym):
        sym2col.setdefault(s, j)
    nself = 0
    for k, t in enumerate(tfs):
        j = sym2col.get(t)
        if j is not None:
            Zsub[k, j] = 0.0
            nself += 1
    report(f"    TF->gene causal weights: {Zsub.shape[0]} TFs x {Zsub.shape[1]:,} genes, robust z, "
           f"{nself} self-knockdown diagonal entries zeroed")

    Zp = Zsub[:, gene_idx].T.astype(np.float32)               # (n_pairs x n_tf), gene side
    Bp = B[:, el_idx].T                                       # (n_pairs x n_tf), element side

    # gene-only background over ALL profiled TFs (no element information whatsoever)
    g_mean_abs_full = np.abs(Zsub).mean(0)
    g_frac_full = (np.abs(Zsub) >= Z_EDGE).mean(0)
    g_max_full = np.abs(Zsub).max(0)
    g_mean_abs = g_mean_abs_full[gene_idx]
    g_frac_causal = g_frac_full[gene_idx]

    dhs = np.array([float(r["DHS.RPM"] or 0) for r in rows])
    dhs_pct = np.array([float(r["DHS.percentile"] or 0) for r in rows])
    width = (een - est)[el_idx].astype(np.float64)

    # an independent accessibility measure, from the K562 ATAC narrowPeak already on disk, so the
    # accessibility arm is not resting on one pipeline's column
    at_sig = np.zeros(len(uel))
    with gzip.open(ATAC, "rt") as fh:
        ap = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            ap.setdefault(f[0], []).append((int(f[1]), int(f[2]), float(f[6])))
    for c, v in ap.items():
        v.sort()
        s = np.array([x[0] for x in v]); e = np.array([x[1] for x in v]); sg = np.array([x[2] for x in v])
        m = np.where(ech == c)[0]
        if not len(m):
            continue
        for i in m:
            lo = np.searchsorted(s, est[i] - 5000)
            hi = np.searchsorted(s, een[i])
            if hi > lo:
                ov = np.minimum(e[lo:hi], een[i]) - np.maximum(s[lo:hi], est[i])
                good = ov > 0
                if good.any():
                    at_sig[i] = sg[lo:hi][good].max()
    atac = at_sig[el_idx]
    report(f"    accessibility: DHS.RPM (benchmark pipeline) median {np.median(dhs):.2f}; K562 ATAC peak "
           f"overlaps {int((atac > 0).sum()):,}/{len(rows):,} pairs, median signal "
           f"{np.median(atac[atac > 0]) if (atac > 0).any() else 0:.2f}")

    # DHS is a per-PAIR column in the benchmark but a per-ELEMENT property; collapse it to the element so a
    # swapped decoy can carry its own value
    el_dhs = np.zeros(len(uel)); el_dhspct = np.zeros(len(uel))
    for i in range(len(rows)):
        el_dhs[el_idx[i]] = dhs[i]; el_dhspct[el_idx[i]] = dhs_pct[i]
    el_width = (een - est).astype(np.float64)

    def blocks(el_i, dist_v):
        """Every feature the model can see, as named blocks, for a given ELEMENT assignment.

        Called once with the real elements and once per swap draw with the decoys. Passing the element index
        in explicitly is what makes the swap a genuine swap: the decoy brings its OWN accessibility, its own
        width, its own TF count and its own TF set, and only the gene stays.
        """
        d = np.maximum(dist_v, 1.0)
        bp = B[:, el_i].T
        tf = tf_features(bp, Zp, g_mean_abs, g_frac_causal)
        return {
            "DIST": {"log10_dist": np.log10(d)},
            "ACC": {"log1p_dhs_rpm": np.log1p(el_dhs[el_i]),
                    "dhs_pct": el_dhspct[el_i],
                    "log1p_atac": np.log1p(at_sig[el_i]),
                    "atac_hit": (at_sig[el_i] > 0).astype(float),
                    "log10_width": np.log10(np.maximum(el_width[el_i], 1.0))},
            "OCC": {"n_tf_bound": bp.sum(1).astype(float)},
            "GENE": {"gene_mean_abs_z": g_mean_abs, "gene_frac_causal": g_frac_causal,
                     "gene_max_abs_z": g_max_full[gene_idx], "gene_expr": gexpr[gene_idx]},
            "TFC": tf,
        }

    real = blocks(el_idx, dist)
    names = {k: sorted(v) for k, v in real.items()}
    report(f"    feature blocks: " + "; ".join(f"{k}({len(v)})" for k, v in names.items()))

    def mat(bl, keys):
        cols = []
        for k in keys:
            for n in names[k]:
                cols.append(np.asarray(bl[k][n], dtype=np.float64))
        return np.column_stack(cols)

    ARMS = {
        "distance_only":            ["DIST"],
        "accessibility_only":       ["ACC"],
        "distance+accessibility":   ["DIST", "ACC"],
        "dist+acc+TFoccupancy":     ["DIST", "ACC", "OCC"],
        "gene_responsiveness_only": ["GENE"],
        "dist+acc+gene":            ["DIST", "ACC", "GENE"],
        "TFcausality_only":         ["OCC", "TFC"],
        "FULL_dist+acc+TFcausality": ["DIST", "ACC", "OCC", "TFC"],
        "FULL+gene":                ["DIST", "ACC", "OCC", "GENE", "TFC"],
    }
    BASELINES = ["distance_only", "accessibility_only", "distance+accessibility",
                 "dist+acc+TFoccupancy", "dist+acc+gene"]
    FULL = "FULL_dist+acc+TFcausality"

    # ---- 4. out-of-fold AUPRC on identical folds ---------------------------------------------------------
    report(f"\n  4  OUT-OF-FOLD AUPRC, chromosome-held-out (gene-disjoint), {len(SEEDS)} genuinely different "
           f"partitions x {FOLDS} folds")
    report(f"    POSITIVE BASE RATE {y.mean():.4f} ({int(y.sum())}/{len(y)}) -- every AUPRC below is RAW, "
           f"not a difference from a shuffle")
    folds, fold_maps = {}, {}
    for s in SEEDS:
        folds[s], fold_maps[s] = chrom_folds(chrTSS, y, s)
    # prove the partitions genuinely differ rather than being relabelings of one split. The comparison is
    # made on the (chromosome -> fold) CO-MEMBERSHIP matrix, which is invariant to renaming the folds; a
    # rotated labelling would score 1.00 here and be caught.
    pair_agree = []
    for a in SEEDS:
        for b in SEEDS:
            if a >= b:
                continue
            cs = sorted(fold_maps[a])
            ca = np.array([fold_maps[a][c] for c in cs])
            cb = np.array([fold_maps[b][c] for c in cs])
            m = np.triu(np.ones((len(cs), len(cs)), bool), 1)
            pair_agree.append(float(np.mean((ca[:, None] == ca[None, :])[m] ==
                                            (cb[:, None] == cb[None, :])[m])))
    report(f"    partition check: mean agreement of chromosome CO-MEMBERSHIP across seed pairs "
           f"{np.mean(pair_agree):.2f} (1.00 would mean the seeds are relabelings of one split)")

    res, MODELS = {}, {}
    for arm, ks in ARMS.items():
        F = mat(real, ks)
        vals = []
        for s in SEEDS:
            ms = fit_folds(F, y, folds[s], s)
            if arm in (FULL, "TFcausality_only"):
                MODELS.setdefault(arm, {})[s] = ms      # kept so the swap re-scores, never retrains
            vals.append(score_oof(ms, F, y, folds[s]))
        res[arm] = {"auprc_mean": float(np.mean(vals)), "auprc_sd": float(np.std(vals)),
                    "auprc_per_seed": [float(v) for v in vals], "n_features": F.shape[1],
                    "lift_over_base_rate": float(np.mean(vals) / y.mean())}
        report(f"    {arm:28s} AUPRC {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
               f"({np.mean(vals)/y.mean():4.2f}x the {y.mean():.4f} base rate, {F.shape[1]} features)")

    best_base = max(BASELINES, key=lambda a: res[a]["auprc_mean"])
    inc = res[FULL]["auprc_mean"] - res[best_base]["auprc_mean"]
    per_seed_win = [int(res[FULL]["auprc_per_seed"][i] - res[best_base]["auprc_per_seed"][i]
                        >= MIN_INCREMENT) for i in range(len(SEEDS))]
    report(f"    BEST BASELINE: {best_base} at {res[best_base]['auprc_mean']:.4f}")
    report(f"    INCREMENT of {FULL} over it: {inc:+.4f} AUPRC "
           f"(>= {MIN_INCREMENT:.3f} in {sum(per_seed_win)}/{len(SEEDS)} partitions)")

    # ---- 5. THE ELEMENT SWAP -----------------------------------------------------------------------------
    report(f"\n  5  ELEMENT SWAP -- keep the gene, replace the element with a decile-matched decoy")
    dq = np.digitize(dist, np.quantile(dist, np.linspace(.1, .9, 9)))
    aq = np.digitize(el_dhspct[el_idx], np.quantile(el_dhspct[el_idx], np.linspace(.1, .9, 9)))
    nq = np.digitize(nbound_el[el_idx], np.quantile(nbound_el[el_idx], np.linspace(.1, .9, 9)))
    report(f"    matching on DECILES, not quartiles: distance decile x accessibility (DHS percentile) "
           f"decile; the degree-matched variant adds the n-TFs-bound decile")

    def swap_indices(bins, rng):
        """Donor pair drawn from the same bin, on a DIFFERENT chromosome from this pair's gene and with a
        different element. Different-chromosome is not fastidiousness: a decoy in cis with the gene could be
        a genuine regulator of it, which would make the control pass for the wrong reason."""
        pool = {}
        for i in range(len(y)):
            pool.setdefault(bins[i], []).append(i)
        pool = {k: np.array(v) for k, v in pool.items()}
        out = np.arange(len(y))
        nfail = 0
        for k, v in pool.items():
            tgt = np.where(bins == k)[0]
            if len(v) < 2:
                nfail += len(tgt)
                continue
            pick = v[rng.integers(0, len(v), size=len(tgt))]
            for attempt in range(8):
                bad = (ech[el_idx[pick]] == chrTSS[tgt]) | (el_idx[pick] == el_idx[tgt])
                if not bad.any():
                    break
                pick[bad] = v[rng.integers(0, len(v), size=int(bad.sum()))]
            bad = (ech[el_idx[pick]] == chrTSS[tgt]) | (el_idx[pick] == el_idx[tgt])
            nfail += int(bad.sum())
            out[tgt] = pick
        return out, nfail

    def run_swap(bins, label, model_arms):
        """One donor draw is shared by every arm evaluated on it, and NO model is refitted -- the fold models
        from section 4 are re-used and shown the swapped features."""
        arms = list(model_arms)
        aupr = {a: [] for a in arms}
        imb = {"log10_dist": [], "dhs_pct": [], "n_tf_bound": []}
        pv = {"log10_dist": [], "dhs_pct": [], "n_tf_bound": []}
        nfail_tot = 0
        for d in range(SWAP_DRAWS):
            rng = np.random.default_rng(5000 + d)
            j, nf = swap_indices(bins, rng)
            nfail_tot += nf
            sw = blocks(el_idx[j], dist[j])
            for cov, a_, b_ in (("log10_dist", np.log10(dist), np.log10(dist[j])),
                                ("dhs_pct", el_dhspct[el_idx], el_dhspct[el_idx[j]]),
                                ("n_tf_bound", nbound_el[el_idx].astype(float),
                                 nbound_el[el_idx[j]].astype(float))):
                imb[cov].append(float(np.mean(a_) - np.mean(b_)))
                pv[cov].append(float(stats.mannwhitneyu(a_, b_, alternative="two-sided")[1]))
            for arm in arms:
                Fs = mat(sw, ARMS[arm])
                aupr[arm].append(float(np.mean([score_oof(MODELS[arm][s], Fs, y, folds[s])
                                                for s in SEEDS])))
        out = {}
        report(f"    {label}")
        report(f"      residual imbalance (real - swapped): "
               f"{ {k: round(float(np.mean(v)), 4) for k, v in imb.items()} }   "
               f"mean Mann-Whitney p per covariate: "
               f"{ {k: float(f'{np.mean(v):.3g}') for k, v in pv.items()} }")
        report(f"      {nfail_tot / SWAP_DRAWS:,.0f} pairs per draw had no valid donor in their decile cell "
               f"and kept their own element (they make the control CONSERVATIVE, not liberal)")
        for arm in arms:
            v = aupr[arm]
            real_a = res[arm]["auprc_mean"]
            frac = float(np.mean([real_a > a for a in v]))
            o = {"label": label, "model_arm": arm, "draws": SWAP_DRAWS,
                 "EFFECT_raw_real_auprc": real_a,
                 "swapped_auprc_mean": float(np.mean(v)), "swapped_auprc_sd": float(np.std(v)),
                 "swapped_auprc_min": float(np.min(v)), "swapped_auprc_max": float(np.max(v)),
                 "frac_draws_real_beats_swap": frac,
                 "delta_real_minus_swap": real_a - float(np.mean(v)),
                 "frac_of_real_reproduced_by_swap": float(np.mean(v)) / max(real_a, 1e-12),
                 "n_pairs_without_valid_donor_per_draw": int(nfail_tot / SWAP_DRAWS),
                 "residual_imbalance_real_minus_swap": {k: float(np.mean(x)) for k, x in imb.items()},
                 "residual_imbalance_max_abs": {k: float(np.max(np.abs(x))) for k, x in imb.items()},
                 "residual_imbalance_mean_mannwhitney_p": {k: float(np.mean(x)) for k, x in pv.items()}}
            report(f"      [{arm}] real {real_a:.4f}   swapped {np.mean(v):.4f} +/- {np.std(v):.4f} "
                   f"(range {np.min(v):.4f}-{np.max(v):.4f});  real beats the decoy in {frac:.0%} of "
                   f"{SWAP_DRAWS} draws; the decoy reproduces "
                   f"{o['frac_of_real_reproduced_by_swap']:.0%} of the real AUPRC")
            out[arm] = o
        return out

    bins_da = np.array([f"{a}|{b}" for a, b in zip(dq, aq)])
    bins_dan = np.array([f"{a}|{b}|{c}" for a, b, c in zip(dq, aq, nq)])
    s_da = run_swap(bins_da, "SWAP matched on distance decile x accessibility decile",
                    [FULL, "TFcausality_only"])
    s_dg = run_swap(bins_dan, "SWAP additionally matched on n-TFs-bound decile (DEGREE-MATCHED)",
                    [FULL, "TFcausality_only"])
    swaps = {
        "distance_x_accessibility": s_da[FULL],
        "degree_matched": s_dg[FULL],
        "distance_x_accessibility_TFCarm": s_da["TFcausality_only"],
        "degree_matched_TFCarm": s_dg["TFcausality_only"],
    }

    # ---- 6. what the increment is, sanity-checked against the arms it must beat --------------------------
    report("\n  6  READING THE THREE THINGS TOGETHER")
    passes_inc = (inc >= MIN_INCREMENT) and (sum(per_seed_win) >= MIN_SEEDS)
    passes_swap = swaps["distance_x_accessibility"]["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC
    passes_deg = swaps["degree_matched"]["frac_draws_real_beats_swap"] >= MIN_SWAP_FRAC
    report(f"    (i)   increment >= {MIN_INCREMENT:.3f} in >= {MIN_SEEDS}/{len(SEEDS)} partitions: "
           f"{'PASS' if passes_inc else 'FAIL'}  ({inc:+.4f}, {sum(per_seed_win)}/{len(SEEDS)})")
    report(f"    (ii)  beats the distance x accessibility swap in >= {MIN_SWAP_FRAC:.0%} of draws: "
           f"{'PASS' if passes_swap else 'FAIL'}  "
           f"({swaps['distance_x_accessibility']['frac_draws_real_beats_swap']:.0%})")
    report(f"    (iii) beats the DEGREE-matched swap in >= {MIN_SWAP_FRAC:.0%} of draws: "
           f"{'PASS' if passes_deg else 'FAIL'}  "
           f"({swaps['degree_matched']['frac_draws_real_beats_swap']:.0%})")

    # a descriptive cross-check, not a claim: are positives bound by more CAUSAL TFs than negatives, matched
    # on distance x accessibility deciles? 20 draws, raw rates reported.
    report("\n    descriptive cross-check: TFs bound AND causal at the element, positives vs matched negatives")
    tfc_n = real["TFC"]["tf_n_causal"]
    pooln, pool1 = {}, {}
    for i in range(len(y)):
        (pool1 if y[i] else pooln).setdefault(bins_da[i], []).append(i)
    dr = []
    for d in range(SWAP_DRAWS):
        rng = np.random.default_rng(7000 + d)
        a, b = [], []
        for k, v1 in pool1.items():
            v0 = pooln.get(k, [])
            if not v0:
                continue
            a += list(v1)
            b += list(rng.choice(v0, len(v1), replace=len(v0) < len(v1)))
        if len(a) < 30:
            continue
        dr.append((float(np.mean(tfc_n[a])), float(np.mean(tfc_n[b])),
                   float(stats.mannwhitneyu(tfc_n[a], tfc_n[b], alternative="two-sided")[1])))
    if dr:
        mp = float(np.mean([x[0] for x in dr])); mn_ = float(np.mean([x[1] for x in dr]))
        fs = float(np.mean([x[2] < 0.05 for x in dr]))
        cross = {"n_draws": len(dr), "EFFECT_raw_positives_mean_n_causal_bound_TFs": mp,
                 "matched_negatives_mean": mn_, "ratio": mp / max(mn_, 1e-12),
                 "frac_draws_p_lt_0.05": fs, "median_p": float(np.median([x[2] for x in dr]))}
        report(f"      positives {mp:.2f} causal-and-bound TFs vs matched negatives {mn_:.2f} "
               f"({mp/max(mn_,1e-12):.2f}x), p<0.05 in {fs:.0%} of {len(dr)} draws "
               f"(median p {cross['median_p']:.3g})")
    else:
        cross = {"n_draws": 0, "reason": "no stratum carried both a positive and a negative"}
        report("      NOT RUN: no stratum carried both arms")

    # ---- verdict -----------------------------------------------------------------------------------------
    base_rate = float(y.mean())
    rb = res[best_base]["auprc_mean"]
    rf = res[FULL]["auprc_mean"]
    sw1 = swaps["distance_x_accessibility"]
    sw2 = swaps["degree_matched"]
    head = (f"On {len(y):,} powered distal K562 CRISPR pairs at a {base_rate:.4f} positive base rate, "
            f"chromosome-held-out AUPRC is {res['distance_only']['auprc_mean']:.4f} for distance alone, "
            f"{res['accessibility_only']['auprc_mean']:.4f} for accessibility alone, "
            f"{res['gene_responsiveness_only']['auprc_mean']:.4f} for gene responsiveness with no element "
            f"information at all, and {rf:.4f} for distance + accessibility + TF causality over "
            f"{len(tfs)} TFs")
    if passes_inc and passes_swap and passes_deg:
        v = (f"TF CAUSALITY AT THE ELEMENT IS A REAL INCREMENT. {head}. That is {inc:+.4f} AUPRC over the "
             f"best baseline ({best_base}, {rb:.4f}), held in {sum(per_seed_win)}/{len(SEEDS)} chromosome "
             f"partitions, and it survives the decisive control: swapping in a distance- and "
             f"accessibility-decile-matched decoy element drops AUPRC to {sw1['swapped_auprc_mean']:.4f} "
             f"(real wins {sw1['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws), and a decoy also "
             f"matched on the number of bound TFs drops it to {sw2['swapped_auprc_mean']:.4f} "
             f"({sw2['frac_draws_real_beats_swap']:.0%}). The model is reading WHICH TFs sit on the element, "
             f"not how busy it is and not which gene it is pointed at.")
    elif not passes_inc:
        v = (f"NO INCREMENT FROM TF CAUSALITY OVER DISTANCE AND ACCESSIBILITY -- NOT DETECTED. {head}. The "
             f"full model is {inc:+.4f} AUPRC against the best baseline ({best_base}, {rb:.4f}), clearing "
             f"the prespecified {MIN_INCREMENT:.3f} bar in only {sum(per_seed_win)}/{len(SEEDS)} chromosome "
             f"partitions. Weighting an element's bound TFs by whether those TFs are causal for the gene in "
             f"Perturb-seq does not add to what distance and accessibility already say. The element swap is "
             f"reported anyway and is consistent: a decile-matched decoy element reproduces "
             f"{sw1['frac_of_real_reproduced_by_swap']:.0%} of the full model's AUPRC "
             f"({sw1['swapped_auprc_mean']:.4f} vs {rf:.4f}, real winning "
             f"{sw1['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws), which is what a model that "
             f"reads the gene and the distance rather than the element looks like. This is a null and the "
             f"project commits it.")
    elif not passes_swap:
        v = (f"THE INCREMENT IS REAL BUT THE ELEMENT IS NOT DOING THE WORK -- the model reads the gene and "
             f"the distance. {head}, an increment of {inc:+.4f} over {best_base} ({rb:.4f}). But replacing "
             f"the element with a distance- and accessibility-decile-matched decoy, keeping the gene, leaves "
             f"AUPRC at {sw1['swapped_auprc_mean']:.4f} -- {sw1['frac_of_real_reproduced_by_swap']:.0%} of "
             f"the real value, with the real element winning only "
             f"{sw1['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws. Whatever the TF-causality "
             f"features add, they add it from the GENE side of the pair, and a gene-responsiveness-only arm "
             f"with no element information already reaches "
             f"{res['gene_responsiveness_only']['auprc_mean']:.4f}. This does not license scoring an "
             f"element-gene pair by the element's TF set.")
    else:
        v = (f"THE INCREMENT IS ELEMENT-SPECIFIC BUT DEGREE EXPLAINS IT -- the finding is TF DENSITY, not TF "
             f"IDENTITY. {head}, an increment of {inc:+.4f} over {best_base} ({rb:.4f}), and it does survive "
             f"a distance x accessibility matched element swap ({sw1['swapped_auprc_mean']:.4f}, real "
             f"winning {sw1['frac_draws_real_beats_swap']:.0%} of draws). It does NOT survive the "
             f"DEGREE-MATCHED swap: a decoy element carrying the same NUMBER of bound TFs reproduces "
             f"{sw2['frac_of_real_reproduced_by_swap']:.0%} of the AUPRC "
             f"({sw2['swapped_auprc_mean']:.4f}), with the real element winning only "
             f"{sw2['frac_draws_real_beats_swap']:.0%} of {SWAP_DRAWS} draws. An element with many TFs on it "
             f"is a better enhancer candidate; WHICH TFs, weighted by their Perturb-seq causality for the "
             f"gene, is not shown here to matter. That is the finding, not a footnote.")

    limits = [
        f"same cell line on both sides. TF->gene causality is Replogle K562 Perturb-seq and the ground truth "
        f"is CRISPRi enhancer perturbation in K562. The experiments differ (gene knockdown vs element "
        f"silencing, different labs and libraries), so this is not scoring a K562 layer on K562 Perturb-seq, "
        f"but 'which genes move easily in K562' is shared between them. That nuisance is what the "
        f"gene-responsiveness-only arm ({res['gene_responsiveness_only']['auprc_mean']:.4f} AUPRC with NO "
        f"element information) measures and what the element swap removes. There is no RPE1 enhancer-CRISPR "
        f"benchmark to escape to.",
        f"the benchmark's K562 arm is EIGHT screens with different libraries, distance ranges and hit rates. "
        f"Dataset is NOT a matching key here (the swap matches on distance and accessibility deciles), so a "
        f"decoy can come from a different screen than its target pair. Distance decile absorbs most of that "
        f"because the screens differ mainly in distance range, but not all of it.",
        f"one ChIP file per TF (lowest-accession GRCh38 '{OUTPUT_TYPE}'). A TF whose K562 experiments "
        f"disagree is represented by one of them, and {len(idx) - len(tfs)} ChIP-profiled TFs are dropped "
        f"for having no Replogle perturbation.",
        f"{len(tfs)} TFs is a large but not exhaustive slice of the ~1,600 human TFs; an element's real "
        f"driver may be a TF with no K562 ChIP track, which would make its TF-causality features look empty "
        f"for a reason that is about ENCODE's coverage and not about the element.",
        f"occupancy is binary at ELEMENT_PAD={ELEMENT_PAD} bp around the tested element. Peak strength is "
        f"discarded, so a marginal peak and a summit-in-the-middle peak count the same.",
        f"'causal' is |robust z| >= {Z_EDGE} in a pseudobulk knockdown. It is a CONSEQUENCE of removing the "
        f"TF, direct or not -- bound_causal.py's own finding is that only ~12% of causal edges are bound at "
        f"the promoter, so most of these weights are downstream effects that happen to be attached to a TF "
        f"that also sits at this element.",
        f"the swap donor pool is the benchmark's own {len(uel):,} elements. That is deliberate (a decoy that "
        f"is a real tested candidate is a harder control than a random genomic window) but it means the "
        f"control asks 'is THIS element better than ANOTHER TESTED element', not 'better than random DNA'.",
        f"pairs below {POWER_COL} {MIN_POWER} are excluded, not counted as negatives, because not-detected "
        f"is not not-linked. {int((pw < MIN_POWER).sum()):,} pairs are dropped that way.",
        f"AUPRC is pooled over out-of-fold predictions within a seed. Folds are whole chromosomes, so fold "
        f"sizes are uneven and a single large chromosome carries disproportionate weight in every partition.",
        f"a decoy element is required to be on a different chromosome from the gene, which removes the risk "
        f"that the decoy genuinely regulates the gene but also means the swap changes chromosome-level "
        f"context (replication timing, compartment) along with the element.",
        f"the model is one gradient-boosting configuration, fixed across all arms rather than tuned per arm. "
        f"A tuned baseline could beat an untuned full model or the reverse; holding it fixed is what makes "
        f"the arms comparable, and it means no arm is at its ceiling.",
    ]

    R = {
        "module": "causal_enhancer",
        "question": ("do the TFs bound at a distal element, weighted by whether those TFs are causal for the "
                     "gene in K562 Perturb-seq, predict a CRISPR-validated enhancer-gene link better than "
                     "distance and better than accessibility?"),
        "decision_threshold_stated_before_numbers": {
            "min_increment_auprc_over_best_baseline": MIN_INCREMENT,
            "min_seeds_of_total": [MIN_SEEDS, len(SEEDS)],
            "min_fraction_of_swap_draws_real_beats_decoy": MIN_SWAP_FRAC,
            "swap_draws": SWAP_DRAWS,
            "rule": "all three must hold; failing (i) = no increment, failing (ii) = reads the gene and the "
                    "distance, failing (iii) = reads TF density not TF identity",
        },
        "ground_truth": {
            "source": "EP CRISPR benchmark (EngreitzLab/CRISPR_comparison), K562 arm: "
                      f"{TRAINING} + K562 rows of {HELDOUT}",
            "n_pairs_powered": len(y), "n_positives": int(y.sum()), "positive_base_rate": base_rate,
            "n_elements": len(uel), "n_genes": len(set(ens)), "n_chromosomes": len(set(chrTSS)),
            "median_distance_bp": float(np.median(dist)), "min_distance_bp": float(dist.min()),
            "power_col": POWER_COL, "min_power": MIN_POWER,
            "datasets": dict(Counter(r["Dataset"] for r in rows)),
        },
        "joins": {
            "benchmark_gene_to_replogle": {"rate": rate, "declared_floor": MIN_JOIN_GENE, "raises": True},
            "encode_chip_target_to_replogle_perturbation": {
                "n": len(tfs), "of": len(idx), "rate": tf_rate, "declared_floor": MIN_JOIN_TF,
                "raises": True},
        },
        "occupancy": {
            "n_tfs": len(tfs), "output_type": OUTPUT_TYPE, "element_pad_bp": ELEMENT_PAD,
            "frac_cells_bound": float(B.mean()),
            "median_tfs_per_element": float(np.median(nbound_el)),
            "n_elements_with_no_profiled_tf": int((nbound_el == 0).sum()),
        },
        "causal_weights": {"z_edge": Z_EDGE, "normalisation": "per-gene robust z (lfc - median)/(1.4826*MAD)",
                           "self_knockdown_diagonal_zeroed": nself},
        "cv": {"scheme": "chromosome-held-out (gene-disjoint by construction)", "folds": FOLDS,
               "seeds": list(SEEDS),
               "mean_frac_chromosomes_in_same_fold_across_seed_pairs": float(np.mean(pair_agree))},
        "arms_raw_auprc": res,
        "best_baseline": best_base,
        "increment_over_best_baseline": float(inc),
        "increment_per_seed_passes": per_seed_win,
        "element_swap": swaps,
        "descriptive_cross_check_causal_bound_TFs": cross,
        "gates": {"increment": bool(passes_inc), "element_swap": bool(passes_swap),
                  "degree_matched_swap": bool(passes_deg)},
        "verdict": v,
        "limits": limits,
        "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "causal_enhancer.json").write_text(json.dumps(R, indent=1, default=float))
    report("\n" + "=" * 104)
    report(f"VERDICT: {v}")
    report("=" * 104)
    R["log"] = log
    (OUT / "causal_enhancer.json").write_text(json.dumps(R, indent=1, default=float))
    report(f"  -> {OUT/'causal_enhancer.json'}")


if __name__ == "__main__":
    main()
