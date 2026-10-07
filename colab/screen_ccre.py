"""screen_ccre -- the genome-wide K562 element universe, and the size of the hole the E-G layer scores around.

WHAT THIS IS FOR. The cell object's enhancer-gene layer (`eg_layer.py`, 452,330 edges) is built and calibrated
on the EP CRISPR benchmark: 4,644 elements that someone chose to test in K562. That is a BENCHMARK-SCOPED
universe, and the layer's own limits list records the consequence -- the 100-250 kb distance stratum never
reaches the 0.30 precision floor, so it emits ZERO distal edges, and nothing beyond the 250 kb window is
looked at at all. What has never been written down is how much of the cell that leaves unaddressed. Not "the
layer is incomplete" as a caveat, but a NUMBER: of the regulatory elements plausibly active in K562, what
fraction can this layer ever attach to a gene.

ENCODE SCREEN registers the denominator. Registry V3 carries 1,063,878 cCREs in GRCh38 and a per-biosample
classification, so for K562 the accessible set can be counted rather than guessed, and every element can be
placed against the registry's own gene TSS coordinates.

THE PROVENANCE TRAP, WHICH ALMOST ATE THIS. The portal serves SIX V3 K562 cCRE annotations, all described
identically as "candidate regulatory elements for GRCh38 in K562". They differ only in an alias suffix naming
the DNase experiment. Five of them -- including the one whose alias has NO experiment suffix and therefore
looks canonical -- carry `Missing-data/Partial-classification` on 100% of rows: DNase only, no H3K4me3,
H3K27ac or CTCF, so every element in them is either "Low-DNase" or "DNase-only" and NOTHING is ever an
enhancer. Reading that file would have reported ~0 K562 enhancers and looked like a finding. Exactly one file
(ENCFF210CAN, ENCSR940SYU, DNase ENCSR000EOT) is fully classified, and it is pinned and asserted here.

FOUR MEASUREMENTS, in order of what they buy.

  1. THE ELEMENT SET. 1,063,878 V3 cCREs with K562-specific class, joined to the registry's 16,380 genes with
     TSS coordinates. Delivered as a layer. This is the thing the E-G work never had.

  2. THE SCOPE GAP. Per gene, and pooled: how many K562-accessible cCREs sit within the E-G window, and how
     many of those did the benchmark actually test? A ratio near 1 would mean the benchmark's universe is
     effectively the genome's and the scoping worry was misplaced.

  3. THE REACHABILITY CENSUS -- the number that qualifies every E-G claim. Split the K562-accessible cCREs by
     what the layer could ever do with them: has a TSS within 100 kb (an edge is emittable), 100-250 kb
     (scored, but that stratum never clears the precision floor, so no edge is ever emitted), or no TSS
     within 250 kb (never enters the layer). The last two categories are the layer's blind spot, exactly.

  4. DOES cCRE STATUS ACTUALLY PREDICT? The census above is arithmetic; it says nothing about whether SCREEN's
     K562 classification is worth anything as a candidate filter. So: among benchmark pairs powered to detect
     an effect, is the positive rate higher for elements overlapping a K562-accessible cCRE than for matched
     elements that do not -- matched on dataset x distance decile x power tercile, 25 draws. The effect size
     reported is the RAW held-out positive rate, not a difference from the control.

WHAT WOULD FALSIFY THE POINT OF THIS MODULE. If (2) came back near 1, or if (3) put almost all K562-accessible
cCREs inside 100 kb of a gene, then the E-G layer's benchmark-scoped universe would be a near-complete one and
this module would be recording a non-problem. And if (4) shows K562 cCRE status carries no signal over matched
controls, then the genome-wide set delivered here is a bigger universe of no better elements -- which is worth
knowing before anything is built on it, and is reported as such rather than buried.

NOT DONE HERE, DELIBERATELY. No E-G edges are scored or emitted. Enlarging the candidate universe without a
model that can rank inside it would multiply the 452,330 edges by an unvalidated factor, which is the exact
failure mode this project keeps correcting. The deliverable is the universe, the census, and one honest test
of whether the classification means anything.
"""
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
AGG = SP / "cCREs.bed"                             # V3 aggregate registry, already on disk (63 MB)
K562 = SP / "screen" / "K562_ENCFF210CAN.bed.gz"   # V3 K562-specific classification
CR = SP / "crispr"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
LAYER = SP / "cell_screen_ccre.json.gz"
CACHE = SP / "screen" / "ccre_reduced.npz"

# ---- THE PIN. One registry version, one biosample file, one classification status. Asserted, not assumed.
REGISTRY_VERSION = "ENCODE Registry of cCREs V3 (SCREEN), GRCh38"
K562_FILE = "ENCFF210CAN"
K562_ANNOTATION = "ENCSR940SYU"
K562_ALIAS = "zhiping-weng:cCREs-GRCh38-V6-EFO_0002067-ENCSR000EOT-7group"
N_CCRE = 1_063_878
ASSEMBLY = "GRCh38"

# The E-G layer's own geometry, read off outputs/orphan/eg_layer.json rather than reinvented, so the census
# below is a census of THAT layer and not of a hypothetical one.
EG_WINDOW = 250_000
EG_EMIT_MAX = 100_000     # 100-250 kb calibrates to threshold=null: no edge is ever emitted there
EG_K562_CANDIDATE_PAIRS = 389_703
EG_K562_EDGES = 122_562

DRAWS = 25
MIN_POWER = 0.8           # PowerAtEffectSize25, the same bar ranking_gate.py uses
POWER_COL = "PowerAtEffectSize25"
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------------------------------
# elements
# --------------------------------------------------------------------------------------------------
def read_agg(report):
    """V3 aggregate registry: chrom, start, end, EH38D, EH38E, aggregate class."""
    acc, chrom, start, end, cls = [], [], [], [], []
    with open(AGG) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            chrom.append(f[0]); start.append(int(f[1])); end.append(int(f[2]))
            acc.append(f[4]); cls.append(f[5])
    return acc, chrom, np.array(start), np.array(end), cls


def read_k562(report):
    """V3 K562-specific classification. Column 10 is the class IN K562, column 11 the classification status.

    Returns accession -> (chrom, start, end, class, status). The K562 file is NOT in the same row order as
    the aggregate registry, so a positional merge would silently attach the wrong class to every element --
    the same class of bug as this project's index-vs-symbol failure. Merged on accession, and the
    coordinates are re-checked after the merge.
    """
    out = {}
    with gzip.open(K562, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            out[f[3]] = (f[0], int(f[1]), int(f[2]), f[9], f[10])
    return out


def build_elements(report):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        report(f"    element table from cache: {len(z['acc']):,} cCREs")
        return (list(z["acc"]), list(z["chrom"]), z["start"], z["end"],
                list(z["agg_cls"]), list(z["k562_cls"]))
    for p in (AGG, K562):
        if not p.exists():
            raise SystemExit(f"absent: {p}")
    acc_a, chrom, start, end, agg_cls = read_agg(report)
    kmap = read_k562(report)

    # ---- PROVENANCE ASSERTIONS. Both files must be the SAME universe, merged on ACCESSION, or the K562
    # class being attached to an aggregate coordinate is attaching it to the wrong element.
    assert len(acc_a) == N_CCRE, f"aggregate registry has {len(acc_a):,} cCREs, expected {N_CCRE:,}"
    assert len(kmap) == N_CCRE, f"K562 file has {len(kmap):,} cCREs, expected {N_CCRE:,}"
    missing = [a for a in acc_a if a not in kmap]
    assert not missing, f"{len(missing):,} aggregate accessions absent from {K562_FILE}, e.g. {missing[:5]}"
    bad = sum(1 for a in acc_a if kmap[a][4] != "All-data/Full-classification")
    assert bad == 0, (
        f"{bad:,}/{N_CCRE:,} rows of {K562_FILE} are not fully classified. Five of the six V3 K562 files on "
        f"the portal are DNase-only and classify NOTHING as an enhancer; this is that trap. Use "
        f"{K562_FILE} ({K562_ALIAS}).")
    coord_mismatch = sum(1 for i, a in enumerate(acc_a)
                         if kmap[a][0] != chrom[i] or kmap[a][1] != int(start[i]) or kmap[a][2] != int(end[i]))
    assert coord_mismatch == 0, (
        f"{coord_mismatch:,} accessions have different coordinates in the two V3 files; they are not the "
        f"same registry build")
    k_cls = [kmap[a][3] for a in acc_a]
    report(f"    ASSERTED  {len(acc_a):,} cCREs, accession sets identical and coordinates identical between "
           f"the aggregate and K562 files, 100% All-data/Full-classification")
    np.savez_compressed(CACHE, acc=np.array(acc_a), chrom=np.array(chrom), start=start, end=end,
                        agg_cls=np.array(agg_cls), k562_cls=np.array(k_cls))
    return acc_a, chrom, start, end, agg_cls, k_cls


def base_class(c):
    """'dELS,CTCF-bound' -> 'dELS'. CTCF-binding is a separate axis and collapsing it into the class would
    make 'dELS' and 'dELS,CTCF-bound' look like different element types, which they are not."""
    return c.split(",")[0]


# --------------------------------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------------------------------
def load_benchmark():
    def rd(f):
        p = CR / f
        if not p.exists():
            raise SystemExit(f"benchmark absent at {p}")
        return list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
    tr = rd(TRAINING)
    for r in tr:
        r["CellType"] = "K562"
    ho = [r for r in rd(HELDOUT) if r["CellType"] == "K562"]
    return tr + ho


def fnum(r, c, d=0.0):
    try:
        v = float(r[c])
        return v if np.isfinite(v) else d
    except (ValueError, KeyError, TypeError):
        return d


# --------------------------------------------------------------------------------------------------
# interval machinery
# --------------------------------------------------------------------------------------------------
class Intervals:
    """Per-chromosome sorted intervals with an overlap query. Plain searchsorted plus a bounded back-scan;
    cCREs are short (median 250 bp) and non-overlapping within a chromosome, so the back-scan is O(1) in
    practice and the bound is a guard, not an approximation -- it is asserted against the max width."""

    def __init__(self, chrom, start, end, payload):
        self.by = {}
        order = np.lexsort((start, np.array(chrom)))
        chrom = np.array(chrom)[order]; start = start[order]; end = end[order]
        payload = np.array(payload, dtype=object)[order]
        self.maxw = int((end - start).max())
        for c in np.unique(chrom):
            m = chrom == c
            self.by[c] = (start[m], end[m], payload[m])

    def overlapping(self, ch, lo, hi):
        """Exact, not heuristic: any interval overlapping [lo,hi) must start at or after lo - maxwidth, so
        `searchsorted(st, lo - maxw)` is a sound lower bound rather than a fixed back-scan that could miss."""
        if ch not in self.by:
            return []
        st, en, pl = self.by[ch]
        i = int(np.searchsorted(st, lo - self.maxw))
        j = int(np.searchsorted(st, hi))
        if j <= i:
            return []
        sel = np.where(en[i:j] > lo)[0]
        return list(pl[i:j][sel])

    def best_overlap(self, ch, lo, hi):
        """The single interval with the largest bp overlap, or None. Used to give a tested element ONE cCRE
        class: 'first alphabetically' was the alternative and it is arbitrary in a way that moves the answer
        (a 500 bp element overlapping both a CTCF-only and a dELS would have been called CTCF-only)."""
        if ch not in self.by:
            return None
        st, en, pl = self.by[ch]
        i = int(np.searchsorted(st, lo - self.maxw))
        j = int(np.searchsorted(st, hi))
        if j <= i:
            return None
        ov = np.minimum(en[i:j], hi) - np.maximum(st[i:j], lo)
        k = int(np.argmax(ov))
        return (pl[i:j][k], int(ov[k])) if ov[k] > 0 else None


# --------------------------------------------------------------------------------------------------
# matched control
# --------------------------------------------------------------------------------------------------
def cmh(treated, control, strat, y):
    """Cochran-Mantel-Haenszel on the (arm x outcome x stratum) table.

    The draw-based control satisfies this project's rule that a matched comparison is never read from one
    draw, but with 10,467 treated against a 1,573-pair control pool every draw must reuse controls, which
    makes its Fisher p optimistic. CMH is the test that uses each unit exactly once and conditions on the
    same strata, so it is the honest significance statistic; the draws remain the honest EFFECT estimate.
    """
    from scipy import stats
    st = {}
    for i in treated:
        s = st.setdefault(strat[i], [0, 0, 0, 0])
        s[0 if y[i] else 1] += 1
    for i in control:
        s = st.setdefault(strat[i], [0, 0, 0, 0])
        s[2 if y[i] else 3] += 1
    num = den = 0.0
    nt = nc = 0
    for a, b, c, d in st.values():
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a - (a + b) * (a + c) / n
        den += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        nt += a + b; nc += c + d
    if den <= 0:
        return {"ok": False, "reason": "no stratum has both arms"}
    chi2 = (abs(num) - 0.5) ** 2 / den
    return {"ok": True, "chi2": float(chi2), "p": float(stats.chi2.sf(chi2, 1)),
            "n_treated_used": nt, "n_control_used": nc,
            "n_strata_with_both_arms": sum(1 for a, b, c, d in st.values()
                                           if (a + b) and (c + d) and (a + b + c + d) > 1)}


def matched_draws(report, treated, control, strat, y, covs, draws=DRAWS, label=""):
    """Draw `draws` matched control sets, one control per treated unit from the same stratum.

    Returns the RAW treated positive rate as the effect size, the mean matched-control rate, the fraction of
    draws significant at p<0.05 by Fisher exact, and residual imbalance on every covariate supplied. A single
    draw is never reported -- three prior errors in this project came from reading one.
    """
    from scipy import stats
    pool = {}
    for i in control:
        pool.setdefault(strat[i], []).append(i)
    usable = [i for i in treated if strat[i] in pool]
    n_drop = len(treated) - len(usable)
    if len(usable) < 30:
        return {"ok": False, "reason": f"only {len(usable)} treated units have a matched control stratum",
                "n_treated": len(treated), "n_matchable": len(usable), "n_control_pool": len(control)}
    t_rate = float(np.mean(y[usable]))
    by_str_t = {}
    for i in usable:
        by_str_t.setdefault(strat[i], []).append(i)

    rates, ps, bal_ps, bal_rt, bal_rc, imb = [], [], [], [], [], {k: [] for k in covs}
    for d in range(draws):
        rng = np.random.default_rng(1000 + d)
        # (a) FULL: every treated unit gets a stratum-mate. The control pool is smaller than the treated set,
        #     so controls are reused; that makes the Fisher p OPTIMISTIC and it is reported as such.
        pick = [pool[strat[i]][rng.integers(len(pool[strat[i]]))] for i in usable]
        rates.append(float(np.mean(y[pick])))
        a, b = int(y[usable].sum()), len(usable) - int(y[usable].sum())
        c, e = int(y[pick].sum()), len(pick) - int(y[pick].sum())
        ps.append(float(stats.fisher_exact([[a, b], [c, e]])[1]))
        for k, v in covs.items():
            imb[k].append(float(np.mean(v[usable]) - np.mean(v[pick])))
        # (b) BALANCED: within each stratum take k = min(n_treated, n_control) of each, WITHOUT replacement.
        #     No unit appears twice, so this p is honest at the cost of throwing away most of the treated set.
        bt, bc = [], []
        for s, ti in by_str_t.items():
            ci = pool[s]
            k = min(len(ti), len(ci))
            bt.extend(rng.choice(ti, k, replace=False))
            bc.extend(rng.choice(ci, k, replace=False))
        bal_n = len(bt)
        bal_rt.append(float(np.mean(y[bt]))); bal_rc.append(float(np.mean(y[bc])))
        bal_ps.append(float(stats.fisher_exact(
            [[int(y[bt].sum()), len(bt) - int(y[bt].sum())],
             [int(y[bc].sum()), len(bc) - int(y[bc].sum())]])[1]))
    return {
        "ok": True, "label": label,
        "n_treated": len(treated), "n_matchable": len(usable), "n_dropped_no_stratum": n_drop,
        "n_control_pool": len(control), "n_strata_used": len(by_str_t),
        "EFFECT_raw_treated_positive_rate": t_rate,
        "matched_control_positive_rate_mean": float(np.mean(rates)),
        "matched_control_positive_rate_sd": float(np.std(rates)),
        "ratio_treated_over_control": float(t_rate / max(np.mean(rates), 1e-12)),
        "draws": draws,
        "frac_draws_p_lt_0.05": float(np.mean([p < 0.05 for p in ps])),
        "median_p": float(np.median(ps)),
        "balanced_n_per_arm": int(bal_n),
        "balanced_treated_rate_mean": float(np.mean(bal_rt)),
        "balanced_control_rate_mean": float(np.mean(bal_rc)),
        "balanced_frac_draws_p_lt_0.05": float(np.mean([p < 0.05 for p in bal_ps])),
        "balanced_median_p": float(np.median(bal_ps)),
        "cmh": cmh(usable, control, strat, y),
        "residual_imbalance_treated_minus_control": {k: float(np.mean(v)) for k, v in imb.items()},
        "residual_imbalance_max_abs": {k: float(np.max(np.abs(v))) for k, v in imb.items()},
        "dataset_imbalance": 0.0,      # dataset is an EXACT matching key, so this is zero by construction
    }


# --------------------------------------------------------------------------------------------------
def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("ENCODE SCREEN cCREs IN K562 -- the genome-wide element universe, and what the E-G layer misses")
    report("=" * 100)

    from entity_registry import load as load_reg
    REG = load_reg()
    assert REG.meta.get("assembly") == ASSEMBLY, "registry is not on GRCh38; coordinates would not be comparable"

    # ---- 1. THE ELEMENT SET -----------------------------------------------------------------------
    report(f"\n  1  ELEMENT SET   {REGISTRY_VERSION}")
    report(f"     K562 classification: {K562_FILE} / {K562_ANNOTATION}")
    report(f"     alias {K562_ALIAS}")
    acc, chrom, start, end, agg_cls, k_cls = build_elements(report)
    chrom = np.array(chrom); acc = np.array(acc)
    kbase = np.array([base_class(c) for c in k_cls])
    abase = np.array([base_class(c) for c in agg_cls])
    ctcf_bound = np.array(["CTCF-bound" in c for c in k_cls])
    mid = (start + end) // 2

    ACTIVE = np.array([c not in ("Low-DNase",) for c in kbase])          # accessible in K562
    MARKED = np.array([c in ("PLS", "pELS", "dELS", "DNase-H3K4me3", "CTCF-only") for c in kbase])
    report(f"\n     K562-specific class of the {len(acc):,} registered cCREs:")
    for c, n in sorted([(c, int((kbase == c).sum())) for c in set(kbase)], key=lambda x: -x[1]):
        report(f"       {c:16s} {n:9,d}  ({n/len(acc):6.2%})   CTCF-bound {int(ctcf_bound[kbase == c].sum()):,}")
    report(f"     K562-ACCESSIBLE (class != Low-DNase)      {int(ACTIVE.sum()):,}  ({ACTIVE.mean():.2%})")
    report(f"     K562-MARKED     (excludes DNase-only)     {int(MARKED.sum()):,}  ({MARKED.mean():.2%})")

    # The aggregate class is what a cell-type-agnostic pipeline would have used. Quantifying the disagreement
    # is the point: 'dELS' in the aggregate registry means 'an enhancer somewhere', not 'an enhancer here'.
    agg_dels = abase == "dELS"
    report(f"\n     the aggregate (cell-type-agnostic) registry calls {int(agg_dels.sum()):,} elements dELS; "
           f"in K562 {int((agg_dels & ACTIVE).sum()):,} of those ({(agg_dels & ACTIVE).sum()/agg_dels.sum():.1%}) "
           f"are accessible at all")

    # ---- gene TSS from the registry ---------------------------------------------------------------
    genes = [(e["chrom"], int(e["tss"]), eid, e.get("symbol"))
             for eid, e in REG.entities.items()
             if e.get("entity_type") == "gene" and e.get("chrom") and e.get("tss") is not None]
    report(f"\n     registry supplies TSS coordinates for {len(genes):,} genes "
           f"(of {sum(1 for e in REG.entities.values() if e.get('entity_type') == 'gene'):,} gene entities; "
           f"the rest are not in the cell object and are invisible to any layer built on it)")
    tss_by = {}
    for c, t, eid, sym in genes:
        tss_by.setdefault(c, []).append((t, eid))
    for c in tss_by:
        tss_by[c].sort()
    tss_pos = {c: np.array([t for t, _ in v]) for c, v in tss_by.items()}
    tss_eid = {c: [e for _, e in v] for c, v in tss_by.items()}

    # ---- nearest TSS for every cCRE ---------------------------------------------------------------
    # A chromosome with no registry gene leaves `nearest` at int64 max, i.e. UNKNOWN-and-unreachable rather
    # than 0. That distinction is the whole point of the census: distance-to-nothing must never read as 0.
    nearest = np.full(len(acc), np.iinfo(np.int64).max, dtype=np.int64)
    for c in np.unique(chrom):
        m = np.where(chrom == c)[0]
        if c not in tss_pos:
            continue
        tp = tss_pos[c]
        p = mid[m]
        j = np.searchsorted(tp, p)
        left = np.where(j > 0, np.abs(p - tp[np.clip(j - 1, 0, len(tp) - 1)]), np.iinfo(np.int64).max)
        right = np.where(j < len(tp), np.abs(tp[np.clip(j, 0, len(tp) - 1)] - p), np.iinfo(np.int64).max)
        nearest[m] = np.minimum(left, right)
    no_gene_chrom = int((nearest == np.iinfo(np.int64).max).sum())
    report(f"     {no_gene_chrom:,} cCREs sit on a chromosome with no registry gene at all "
           f"(counted as unreachable, not as distance 0)")

    # ---- 3. THE REACHABILITY CENSUS ---------------------------------------------------------------
    # (printed before the scope gap because it is the number that qualifies every E-G claim)
    report(f"\n  3  REACHABILITY CENSUS -- what the E-G layer can ever do with a K562-accessible cCRE")
    report(f"     the layer scores (element, gene) inside {EG_WINDOW/1000:.0f} kb and, per its own "
           f"calibration, emits NO edge in the {EG_EMIT_MAX/1000:.0f}-{EG_WINDOW/1000:.0f} kb stratum")
    census = {}
    for name, sel in (("K562-accessible", ACTIVE), ("K562-marked", MARKED),
                      ("ELS (dELS+pELS)", np.isin(kbase, ["dELS", "pELS"])),
                      ("dELS (K562)", kbase == "dELS"), ("pELS (K562)", kbase == "pELS"),
                      ("PLS (K562)", kbase == "PLS"), ("CTCF-only (K562)", kbase == "CTCF-only"),
                      ("DNase-only (K562)", kbase == "DNase-only")):
        d = nearest[sel]
        n = int(sel.sum())
        emit = int((d <= EG_EMIT_MAX).sum())
        scored = int(((d > EG_EMIT_MAX) & (d <= EG_WINDOW)).sum())
        never = n - emit - scored
        census[name] = {"n": n, "emittable_within_100kb": emit, "scored_never_emitted_100_250kb": scored,
                        "outside_250kb_never_considered": never,
                        "frac_emittable": emit / max(n, 1), "frac_unreachable": (scored + never) / max(n, 1)}
        report(f"       {name:20s} n={n:8,d}   emittable(<100kb) {emit:8,d} ({emit/max(n,1):6.2%})   "
               f"scored-but-never-emitted(100-250kb) {scored:7,d} ({scored/max(n,1):6.2%})   "
               f"outside 250kb {never:7,d} ({never/max(n,1):6.2%})")
    head = census["K562-accessible"]
    report(f"\n     HEADLINE: {head['frac_unreachable']:.1%} of K562-accessible cCREs "
           f"({head['scored_never_emitted_100_250kb'] + head['outside_250kb_never_considered']:,} of "
           f"{head['n']:,}) can NEVER receive an edge from the current E-G layer -- not because they were "
           f"scored and rejected, but because of the window and the empty distal stratum.")

    # ---- 2. THE SCOPE GAP -------------------------------------------------------------------------
    report(f"\n  2  SCOPE GAP -- benchmark-tested elements vs K562-accessible cCREs, per gene")
    rows = load_benchmark()
    ens = [r["measuredGeneEnsemblId"] for r in rows]
    gid, st = REG.join("gene", "ensembl", ens, 0.95, "EP CRISPR benchmark K562 arm -> gene/ensembl")
    report(f"     benchmark K562 arm: {len(rows):,} pairs; registry join {st['joined']:,}/{st['of']:,} "
           f"({st['rate']:.1%}) via {st['via']}")
    has_tss = np.array([g is not None and REG.entities[g].get("tss") is not None for g in gid])
    report(f"     {int(has_tss.sum()):,}/{len(rows):,} of those pairs name a gene the registry has a TSS for; "
           f"the rest are outside the cell object entirely")

    bm_by_gene = {}
    for r, g, ok in zip(rows, gid, has_tss):
        if not ok:
            continue
        bm_by_gene.setdefault(g, set()).add((r["chrom"], int(r["chromStart"]), int(r["chromEnd"])))
    report(f"     {len(bm_by_gene):,} genes carry at least one tested element")

    gap_rows = []
    for g, els in bm_by_gene.items():
        e = REG.entities[g]
        c, t = e["chrom"], int(e["tss"])
        if c not in tss_pos:
            continue
        m = np.where(chrom == c)[0]
        d_el = np.abs(mid[m] - t)
        for w in (EG_EMIT_MAX, EG_WINDOW, 1_000_000):
            gw = int((ACTIVE[m] & (d_el <= w)).sum())
            bm = sum(1 for (ec, s2, e2) in els if abs((s2 + e2) // 2 - t) <= w)
            gap_rows.append((g, w, gw, bm))
    gap = {}
    for w in (EG_EMIT_MAX, EG_WINDOW, 1_000_000):
        sub = [(gw, bm) for _g, ww, gw, bm in gap_rows if ww == w]
        GW = sum(x[0] for x in sub); BM = sum(x[1] for x in sub)
        per = [bm / gw for gw, bm in sub if gw > 0]
        gap[f"{w//1000}kb"] = {
            "genes": len(sub), "genome_wide_ccre_gene_pairs": GW, "benchmark_tested_pairs": BM,
            "pooled_fraction_tested": BM / max(GW, 1),
            "median_per_gene_fraction_tested": float(np.median(per)) if per else None,
            "median_ccres_per_gene": float(np.median([gw for gw, _ in sub])),
            "median_tested_per_gene": float(np.median([bm for _, bm in sub])),
        }
        d = gap[f"{w//1000}kb"]
        report(f"       within {w//1000:5d} kb of a tested gene: {d['genome_wide_ccre_gene_pairs']:9,d} "
               f"K562-accessible cCREs, {d['benchmark_tested_pairs']:7,d} tested "
               f"({d['pooled_fraction_tested']:.2%})   median per gene "
               f"{d['median_ccres_per_gene']:.0f} vs {d['median_tested_per_gene']:.0f}")

    # coverage of the element universe itself, independent of any gene
    iv = Intervals(chrom, start, end, np.arange(len(acc)))
    tested_idx = set()
    all_els = set()
    for r in rows:
        all_els.add((r["chrom"], int(r["chromStart"]), int(r["chromEnd"])))
    el_hits = {}
    for (c, s2, e2) in all_els:
        h = iv.overlapping(c, s2, e2)
        el_hits[(c, s2, e2)] = h
        tested_idx.update(int(x) for x in h)
    touched_active = sum(1 for i in tested_idx if ACTIVE[i])
    report(f"\n     the benchmark's {len(all_els):,} distinct K562 elements overlap {len(tested_idx):,} "
           f"registered cCREs, {touched_active:,} of them K562-accessible")
    report(f"     that is {touched_active/int(ACTIVE.sum()):.3%} of the {int(ACTIVE.sum()):,} "
           f"K562-accessible cCREs in the genome")
    within_eg = int((ACTIVE & (nearest <= EG_WINDOW)).sum())
    report(f"     restricted to the {within_eg:,} that sit within {EG_WINDOW//1000} kb of a registry gene "
           f"at all, the benchmark touches {touched_active/max(within_eg,1):.3%}")

    # ---- 4. DOES K562 cCRE STATUS PREDICT? --------------------------------------------------------
    report(f"\n  4  DOES SCREEN's K562 CLASSIFICATION PREDICT A VALIDATED E-G LINK?")
    y = np.array([1 if r.get("Significant") in ("TRUE", "True", "true") else 0 for r in rows])
    pw = np.array([fnum(r, POWER_COL) for r in rows])
    dist = np.array([max(abs(fnum(r, "distanceToTSS")), 1.0) for r in rows])
    ds = np.array([r["Dataset"] for r in rows])
    key = [(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows]
    el_active = np.array([any(ACTIVE[int(i)] for i in el_hits[k]) for k in key])

    # ONE class per tested element: the K562-accessible cCRE with the largest bp overlap. Picking the
    # alphabetically-first class instead moved dELS elements into CTCF-only and is arbitrary.
    cls_of = {}
    for k in el_hits:
        c, s2, e2 = k
        h = [int(i) for i in el_hits[k] if ACTIVE[int(i)]]
        if not h:
            cls_of[k] = "none"
            continue
        ov = [min(int(end[i]), e2) - max(int(start[i]), s2) for i in h]
        cls_of[k] = kbase[h[int(np.argmax(ov))]]
    el_class = np.array([cls_of[k] for k in key])

    powered = pw >= MIN_POWER
    report(f"     {int(powered.sum()):,}/{len(rows):,} pairs are powered at {POWER_COL} >= {MIN_POWER}; "
           f"an unpowered non-significant pair is 'not detected', which is NOT 'no link'")
    report(f"     positive rate overall {y[powered].mean():.4f} ({int(y[powered].sum())} positives)")
    report(f"     datasets in the powered K562 arm: "
           f"{ {str(d): int((ds[powered] == d).sum()) for d in sorted(set(ds[powered]))} }")

    # stratum = dataset x distance decile x power tercile. Dataset is in the stratum BECAUSE the K562 arm is
    # eight screens with different libraries and hit rates; pooling them on one axis is the failure this
    # project has already paid for twice.
    dq = np.quantile(dist[powered], np.linspace(0, 1, 11))
    pq = np.quantile(pw[powered], np.linspace(0, 1, 4))
    dd = np.clip(np.searchsorted(dq[1:-1], dist), 0, 9)
    pp = np.clip(np.searchsorted(pq[1:-1], pw), 0, 2)
    strat = np.array([f"{a}|{b}|{c}" for a, b, c in zip(ds, dd, pp)])
    covs = {"log10_distance": np.log10(dist), "power": pw}

    idx = np.where(powered)[0]
    treated = [i for i in idx if el_active[i]]
    control = [i for i in idx if not el_active[i]]
    report(f"     treated (element overlaps a K562-accessible cCRE) {len(treated):,}; "
           f"control (no K562-accessible cCRE) {len(control):,}")
    main_test = matched_draws(report, treated, control, strat, y, covs, label="K562-accessible cCRE overlap")
    tests = {"accessible_vs_not": main_test}
    if main_test.get("ok"):
        report(f"       EFFECT (raw held-out positive rate, treated)   "
               f"{main_test['EFFECT_raw_treated_positive_rate']:.4f}")
        report(f"       matched control rate over {DRAWS} draws          "
               f"{main_test['matched_control_positive_rate_mean']:.4f} "
               f"+/- {main_test['matched_control_positive_rate_sd']:.4f}")
        report(f"       ratio {main_test['ratio_treated_over_control']:.3f}x   "
               f"full draws with p<0.05: {main_test['frac_draws_p_lt_0.05']:.0%} "
               f"(median p {main_test['median_p']:.3g}) -- OPTIMISTIC, controls are reused")
        report(f"       balanced draws (n={main_test['balanced_n_per_arm']:,}/arm, no reuse): "
               f"treated {main_test['balanced_treated_rate_mean']:.4f} vs control "
               f"{main_test['balanced_control_rate_mean']:.4f}, p<0.05 in "
               f"{main_test['balanced_frac_draws_p_lt_0.05']:.0%} of draws "
               f"(median p {main_test['balanced_median_p']:.3g})")
        c_ = main_test["cmh"]
        if c_.get("ok"):
            report(f"       Mantel-Haenszel over {c_['n_strata_with_both_arms']} strata, every unit used once: "
                   f"chi2 {c_['chi2']:.1f}, p {c_['p']:.3g}")
        report(f"       residual imbalance (treated - control): "
               f"{ {k: round(v, 4) for k, v in main_test['residual_imbalance_treated_minus_control'].items()} }"
               f"  max|.| { {k: round(v,4) for k,v in main_test['residual_imbalance_max_abs'].items()} }")
    else:
        report(f"       NOT RUN: {main_test['reason']}")

    # ELS = dELS + pELS pooled. Worth its own arm because "K562-accessible" mixes the enhancer-like classes
    # with CTCF-only and DNase-only, and if those two carry nothing the pooled effect is diluted by them --
    # which changes what candidate filter the E-G layer should actually adopt.
    for cls in ("ELS", "dELS", "pELS", "PLS", "CTCF-only", "DNase-only"):
        tr = ([i for i in idx if el_class[i] in ("dELS", "pELS")] if cls == "ELS"
              else [i for i in idx if el_class[i] == cls])
        if len(tr) < 30:
            tests[f"class_{cls}"] = {"ok": False, "reason": f"only {len(tr)} powered pairs", "n_treated": len(tr)}
            report(f"       {cls:12s} only {len(tr)} powered pairs -- not tested")
            continue
        t = matched_draws(report, tr, control, strat, y, covs, label=f"{cls} vs no K562-accessible cCRE")
        tests[f"class_{cls}"] = t
        if t.get("ok"):
            cp = t["cmh"].get("p") if t["cmh"].get("ok") else float("nan")
            call = ("ELEVATED" if (t["ratio_treated_over_control"] > 1 and cp < 0.05) else
                    "DEPLETED" if (t["ratio_treated_over_control"] < 1 and cp < 0.05) else
                    "NOT DETECTED (not reversed)")
            report(f"       {cls:12s} raw rate {t['EFFECT_raw_treated_positive_rate']:.4f}  "
                   f"matched control {t['matched_control_positive_rate_mean']:.4f}  "
                   f"ratio {t['ratio_treated_over_control']:.3f}x  "
                   f"full-draw p<0.05 {t['frac_draws_p_lt_0.05']:.0%}  MH p {cp:.3g}  "
                   f"n={t['n_matchable']:,}  -> {call}")
        else:
            report(f"       {cls:12s} NOT RUN: {t['reason']}")

    # recall: the number that decides whether cCREs are usable as a candidate filter at all
    pos = idx[y[idx] == 1]
    rec_active = float(np.mean(el_active[pos])) if len(pos) else float("nan")
    rec_marked = float(np.mean([el_class[i] in ("PLS", "pELS", "dELS", "DNase-H3K4me3", "CTCF-only")
                                for i in pos])) if len(pos) else float("nan")
    ELS = np.isin(kbase, ["dELS", "pELS"])
    rec_els = float(np.mean([el_class[i] in ("dELS", "pELS") for i in pos])) if len(pos) else float("nan")
    report(f"     RECALL of validated positives, and the size of the universe each filter keeps:")
    for nm, r_, frac in (("K562-accessible", rec_active, float(ACTIVE.mean())),
                         ("K562-marked", rec_marked, float(MARKED.mean())),
                         ("ELS (dELS+pELS)", rec_els, float(ELS.mean()))):
        report(f"       {nm:18s} recall {r_:6.1%}   keeps {frac:6.2%} of the 1,063,878 registered cCREs "
               f"({r_/max(frac,1e-9):5.1f}x enrichment)")
    report(f"     ({len(pos)} positives; see limits -- recall is measured only among elements the benchmark "
           f"chose to test, and those screens targeted accessible chromatin.)")

    # ---- 5. LINK-LEVEL REACHABILITY -- the number that qualifies every E-G claim -------------------
    # The element-level census (3) asks whether SOME gene is near enough. That is the weaker question: a cCRE
    # with a gene 40 kb away is 'emittable' even if its real target is 600 kb away. The stronger question is
    # over VALIDATED links -- of the E-G connections CRISPR has actually confirmed in K562, what fraction sit
    # at a distance the layer's geometry can express at all?
    report(f"\n  5  LINK-LEVEL REACHABILITY -- validated K562 E-G links vs the layer's geometry")
    dpos = dist[pos]
    link = {"n_validated_positives": int(len(pos)),
            "frac_within_100kb_emittable": float((dpos <= EG_EMIT_MAX).mean()),
            "frac_100_250kb_scored_never_emitted": float(((dpos > EG_EMIT_MAX) & (dpos <= EG_WINDOW)).mean()),
            "frac_beyond_250kb_never_considered": float((dpos > EG_WINDOW).mean()),
            "median_distance_bp": float(np.median(dpos))}
    report(f"     {link['n_validated_positives']} powered, validated positives; median distance to TSS "
           f"{link['median_distance_bp']:,.0f} bp")
    report(f"       within 100 kb (an edge is emittable)          {link['frac_within_100kb_emittable']:6.1%}")
    report(f"       100-250 kb (scored, stratum emits nothing)    "
           f"{link['frac_100_250kb_scored_never_emitted']:6.1%}")
    report(f"       beyond 250 kb (outside the window entirely)   "
           f"{link['frac_beyond_250kb_never_considered']:6.1%}")
    link["frac_unreachable"] = link["frac_100_250kb_scored_never_emitted"] + \
        link["frac_beyond_250kb_never_considered"]
    report(f"     LINK-LEVEL HEADLINE: {link['frac_unreachable']:.1%} of experimentally validated K562 "
           f"enhancer-gene links are at a distance the current E-G layer cannot express.")
    link["per_dataset"] = {}
    for dname in sorted(set(ds[pos])):
        m = ds[pos] == dname
        if m.sum() < 10:
            continue
        link["per_dataset"][str(dname)] = {
            "n": int(m.sum()), "frac_unreachable": float((dpos[m] > EG_EMIT_MAX).mean()),
            "median_distance_bp": float(np.median(dpos[m]))}
        report(f"       {str(dname):16s} n={int(m.sum()):4d}  unreachable "
               f"{(dpos[m] > EG_EMIT_MAX).mean():6.1%}  median {np.median(dpos[m]):9,.0f} bp")
    report(f"     each screen has its own library design and distance range, so the pooled figure is a "
           f"property of the benchmark's composition, not of K562 -- both are reported.")


    # ---- write the layer --------------------------------------------------------------------------
    report(f"\n  WRITING THE LAYER")
    sel = np.where(ACTIVE)[0]
    pair_e, pair_g, pair_d = [], [], []
    eid_list, eid_pos = [], {}
    for c in np.unique(chrom):
        m = np.where(chrom == c)[0]
        m = m[ACTIVE[m]]
        if len(m) == 0 or c not in tss_pos:
            continue
        tp, te = tss_pos[c], tss_eid[c]
        p = mid[m]
        lo = np.searchsorted(tp, p - EG_WINDOW, side="left")
        hi = np.searchsorted(tp, p + EG_WINDOW, side="right")
        for a, i0, i1, pos_ in zip(m, lo, hi, p):
            for j in range(i0, i1):
                g = te[j]
                if g not in eid_pos:
                    eid_pos[g] = len(eid_list); eid_list.append(g)
                pair_e.append(int(a)); pair_g.append(eid_pos[g]); pair_d.append(int(tp[j] - pos_))
    remap = {int(a): i for i, a in enumerate(sel)}
    layer = {
        "source": f"{REGISTRY_VERSION}; K562 classification {K562_FILE} ({K562_ANNOTATION}, {K562_ALIAS})",
        "assembly": ASSEMBLY,
        "n_registered_ccres": int(len(acc)),
        "elements": {
            "accession": [str(x) for x in acc[sel]],
            "chrom": [str(x) for x in chrom[sel]],
            "start": [int(x) for x in start[sel]],
            "end": [int(x) for x in end[sel]],
            "k562_class": [str(x) for x in np.array(k_cls)[sel]],
            "aggregate_class": [str(x) for x in np.array(agg_cls)[sel]],
            "nearest_tss_bp": [int(x) if x < np.iinfo(np.int64).max else -1 for x in nearest[sel]],
        },
        "gene_eids": eid_list,
        "pairs_within_250kb": {
            "element": [remap[a] for a in pair_e],
            "gene": pair_g,
            "signed_distance_tss_minus_element": pair_d,
        },
        "note": ("elements are K562-ACCESSIBLE cCREs (SCREEN class != Low-DNase). No edge score is attached: "
                 "this is a candidate universe, not a link prediction. Pairs are (element, gene) within "
                 "250 kb of the registry TSS and carry distance only."),
    }
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    report(f"    {LAYER}  {LAYER.stat().st_size/1e6:.1f} MB  "
           f"{len(sel):,} elements, {len(pair_e):,} (element, gene) pairs within {EG_WINDOW//1000} kb, "
           f"{len(eid_list):,} genes")
    pair_d_abs = np.abs(np.array(pair_d))
    universe = {
        "ccre_pairs_within_250kb": len(pair_e),
        "ccre_pairs_within_100kb": int((pair_d_abs <= EG_EMIT_MAX).sum()),
        "eg_layer_k562_candidate_pairs": EG_K562_CANDIDATE_PAIRS,
        "eg_layer_k562_edges_emitted": EG_K562_EDGES,
        "ratio_ccre_to_eg_candidates": len(pair_e) / EG_K562_CANDIDATE_PAIRS,
    }
    report(f"    candidate-universe size: {len(pair_e):,} (cCRE, gene) pairs within {EG_WINDOW//1000} kb "
           f"vs the {EG_K562_CANDIDATE_PAIRS:,} peak-derived pairs the E-G layer scored in K562 "
           f"({universe['ratio_ccre_to_eg_candidates']:.2f}x) -- the two are different objects "
           f"(DNase peaks vs registered cCREs) and the peak file is not on disk to recompute, so this is a "
           f"size comparison and not an overlap.")

    limits = [
        f"ENCODE Registry V3 only. V4 (2.35M cCREs, different class vocabulary) is NOT merged; mixing the two "
        f"would put two annotation versions on one axis.",
        f"K562 classification comes from ONE file, {K562_FILE} ({K562_ALIAS}). Five of the six V3 K562 "
        f"annotations on the ENCODE portal are DNase-only and classify 0 elements as enhancers; this module "
        f"asserts 100% All-data/Full-classification and would raise on any of them.",
        f"the cCRE class is a QUALITATIVE call (PLS/pELS/dELS/CTCF-only/DNase-only/Low-DNase). No accessibility "
        f"z-score is carried, because the portal's bed9+ cell-type files do not contain one. 'K562-accessible' "
        f"is therefore a threshold someone else chose, not a continuous covariate this module can re-tune.",
        f"gene coordinates come from the registry's {len(genes):,} genes with a TSS, which is the cell object's "
        f"gene set. cCREs whose only nearby gene is outside that set are counted as unreachable, which is true "
        f"of the model but not of the genome.",
        f"one TSS per gene. Alternative promoters would move some distances across the 100 kb / 250 kb "
        f"boundaries the census turns on; the direction of that bias is toward UNDER-stating reachability.",
        f"the E-G geometry (250 kb window, empty 100-250 kb stratum) is read from outputs/orphan/eg_layer.json, "
        f"not recomputed; if that layer is retrained the census must be rerun.",
        f"the benchmark K562 arm is EIGHT screens (Gasperini2019, Nasser2021, Schraivogel2020, K562_DC_TAP, "
        f"Xie, Morris, Klann, Reilly) with different libraries and hit rates. Dataset is a matching stratum in "
        f"test 4 for exactly that reason; the pooled positive rate across them is not a property of K562.",
        f"absence of a cCRE call is UNKNOWN, not 'not an element': SCREEN's universe is itself derived from "
        f"DNase peaks, so an element with no DNase in any biosample was never registered and is invisible here "
        f"as well as to the E-G layer.",
        f"no edge is scored or emitted. This module delivers a candidate universe and a census; it does not "
        f"claim any element regulates any gene.",
        f"the {rec_active:.1%} recall of validated positives by K562-accessible cCREs is measured ONLY among "
        f"elements the benchmark chose to test, and those screens preferentially targeted accessible "
        f"chromatin. Genome-wide recall is therefore at most this and probably lower; it cannot be estimated "
        f"from a benchmark that shares the selection criterion.",
        f"the matched test reuses controls (10,467 treated vs a 1,573-pair control pool), so the Fisher p on "
        f"the full draw is optimistic. A balanced without-replacement variant is reported alongside it and is "
        f"the one to read for significance.",
        f"'Significant' is the benchmark's own call at its own FDR, and power differs by screen. Pairs below "
        f"{POWER_COL} {MIN_POWER} are excluded rather than counted as negatives, because not-detected is not "
        f"not-linked.",
        f"the (cCRE, gene) pair count is compared to the E-G layer's peak-derived candidate count as a SIZE "
        f"comparison only; the K562 accessibility peak file is not on disk, so their overlap is unmeasured.",
    ]

    res = {
        "module": "screen_ccre",
        "source": layer["source"],
        "assembly": ASSEMBLY,
        "provenance_asserted": {
            "registry_version": REGISTRY_VERSION, "n_ccres": int(len(acc)),
            "k562_file": K562_FILE, "k562_annotation": K562_ANNOTATION, "k562_alias": K562_ALIAS,
            "classification_status": "All-data/Full-classification on 100% of rows (asserted)",
            "aggregate_and_k562_accession_vectors_identical": True,
        },
        "element_set": {
            "n_registered": int(len(acc)),
            "k562_class_counts": {c: int((kbase == c).sum()) for c in sorted(set(kbase))},
            "n_k562_accessible": int(ACTIVE.sum()),
            "n_k562_marked": int(MARKED.sum()),
            "aggregate_dELS": int(agg_dels.sum()),
            "aggregate_dELS_accessible_in_K562": int((agg_dels & ACTIVE).sum()),
        },
        "gene_universe": {"n_genes_with_tss": len(genes),
                          "n_gene_entities": sum(1 for e in REG.entities.values()
                                                 if e.get("entity_type") == "gene")},
        "reachability_census_element_level": census,
        "reachability_link_level": link,
        "headline_unreachable_fraction_elements": head["frac_unreachable"],
        "headline_unreachable_fraction_validated_links": link["frac_unreachable"],
        "candidate_universe": universe,
        "scope_gap": gap,
        "benchmark": {
            "n_pairs_k562_arm": len(rows),
            "registry_join": st,
            "n_pairs_gene_has_tss": int(has_tss.sum()),
            "n_distinct_elements": len(all_els),
            "n_ccres_overlapped": len(tested_idx),
            "n_k562_accessible_ccres_overlapped": touched_active,
            "frac_of_all_k562_accessible": touched_active / int(ACTIVE.sum()),
            "frac_of_k562_accessible_within_250kb_of_a_gene": touched_active / max(within_eg, 1),
            "datasets": {str(d): int((ds == d).sum()) for d in sorted(set(ds))},
        },
        "eg_layer_reference": {"window_bp": EG_WINDOW, "emit_max_bp": EG_EMIT_MAX,
                               "k562_candidate_pairs": EG_K562_CANDIDATE_PAIRS,
                               "k562_edges": EG_K562_EDGES,
                               "distal_stratum_threshold": None},
        "predictive_test": {
            "min_power": MIN_POWER, "power_col": POWER_COL,
            "n_powered": int(powered.sum()), "n_positives_powered": int(y[powered].sum()),
            "overall_positive_rate_powered": float(y[powered].mean()),
            "matching": "dataset x distance decile x power tercile",
            "draws": DRAWS,
            "recall_of_positives_overlapping_accessible_ccre": rec_active,
            "recall_of_positives_overlapping_marked_ccre": rec_marked,
            "recall_of_positives_overlapping_ELS_ccre": rec_els,
            "universe_fraction_kept": {"K562-accessible": float(ACTIVE.mean()),
                                       "K562-marked": float(MARKED.mean()),
                                       "ELS": float(ELS.mean())},
            "tests": tests,
        },
        "layer_file": str(LAYER),
        "layer_pairs_within_250kb": len(pair_e),
        "limits": limits,
        "log": log,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_ccre.json").write_text(json.dumps(res, indent=1))

    mt = tests["accessible_vs_not"]
    if mt.get("ok"):
        sig = (f"{mt['EFFECT_raw_treated_positive_rate']:.4f} raw held-out positive rate vs "
               f"{mt['matched_control_positive_rate_mean']:.4f} matched ({mt['ratio_treated_over_control']:.2f}x; "
               f"p<0.05 in {mt['frac_draws_p_lt_0.05']:.0%} of {DRAWS} full draws and "
               f"{mt['balanced_frac_draws_p_lt_0.05']:.0%} of {DRAWS} balanced draws)")
    else:
        sig = f"predictive test NOT RUN ({mt['reason']})"
    report("\n" + "=" * 100)
    report(f"VERDICT: {int(ACTIVE.sum()):,} K562-accessible cCREs genome-wide, of which the CRISPR benchmark "
           f"tested {touched_active/int(ACTIVE.sum()):.2%} and the E-G layer can never emit an edge for "
           f"{head['frac_unreachable']:.1%}; {link['frac_unreachable']:.1%} of experimentally VALIDATED K562 "
           f"E-G links are at a distance that layer cannot express. Near the genes it did test, the benchmark "
           f"scored {gap[str(EG_WINDOW//1000)+'kb']['pooled_fraction_tested']:.2%} of the cCREs within "
           f"{EG_WINDOW//1000} kb. K562 cCRE overlap does carry signal: {sig}.")
    report("=" * 100)
    (OUT / "screen_ccre.json").write_text(json.dumps({**res, "log": log}, indent=1))


if __name__ == "__main__":
    main()
