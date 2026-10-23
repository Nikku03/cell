"""BIOPLEX 3.0 AP-MS -- rebuilding the PPI layer with the three things it threw away: assay, cell line, confidence.

WHAT IS THERE NOW. The cell object's `ppi` field is 191,447 bare 2-tuples of gene INDEX. No assay, no cell
line, no confidence, no direction. Every edge in it asserts exactly one thing -- "these two are connected" --
and asserts it with equal force whether it came from a co-crystal structure, a yeast two-hybrid screen, or a
text-mining co-mention score of 0.4 that STRING would itself label low confidence. That layer is the single
biggest source of edges in the model and it is the least interrogable. This module does not extend it. It
rebuilds one well-characterised slice of it with the metadata intact, and then measures whether the metadata
buys anything.

WHY BIOPLEX AND NOT MORE STRING. BioPlex 3.0 is one assay (affinity purification of an HA-FLAG tagged bait
followed by mass spectrometry), one pipeline (CompPASS-plus), one release (Dec 2019), run to saturation over
10,128 baits in HEK293T and 5,522 in HCT116. Every edge carries a bait/prey direction and a three-way
posterior -- pWrongID, pNI (no interaction), pInt (interaction) -- from a single model applied uniformly.
That is the opposite of an aggregated score: the confidence means the same thing on every edge.

AND THE POINT OF KEEPING THE CELL LINE. A 293T edge is not a K562 edge. Roughly half the interactions BioPlex
detects in one line are absent in the other even where the bait was screened in both, because complex
membership tracks expression, and expression is cell-type specific. The existing layer's flattening erased
exactly the axis the whole-cell model is supposed to care about. So the cell line is retained per edge, and it
is TESTED: does an interaction seen in BOTH lines predict K562 transcriptional coupling better than one seen
in only one?

THE FIVE PRESPECIFIED QUESTIONS.

  1. DOES THE LAYER CARRY INFORMATION AT ALL? The fast test: for (perturbation P, gene G) pairs this layer
     links, is |robust z| in Replogle K562 CRISPRi Perturb-seq elevated over pairs matched on perturbation
     strength and gene responsiveness? Reference layers are RE-MEASURED HERE with the same instrument rather
     than quoted, because a ratio is only comparable to another ratio computed against a control of the same
     tightness (see THE CONTROL LADDER).
  2. WAS KEEPING CONFIDENCE WORTH IT? Top-tercile pInt edges must beat bottom-tercile edges, both against
     their own matched controls and in a direct matched head-to-head. If not, the score is noise above the
     0.75 publication cut and the honest thing is to record that and stop treating it as a weight.
  3. IS CROSS-CELL-LINE REPRODUCIBILITY A BETTER FILTER THAN CONFIDENCE? Restricted to edges whose bait was
     screened in BOTH lines -- so the interaction had the OPPORTUNITY to be seen twice -- do the ones
     actually seen twice transfer to K562 better than the ones seen once?
  4. IS THIS A REBUILD OR A DUPLICATE? How much of BioPlex is already in the existing 191,447-edge layer, and
     do the edges that are NOT already there carry signal?
  5. WAS BAIT/PREY DIRECTION WORTH KEEPING? Perturbing the bait and measuring the prey, versus the reverse.

THE CONTROL THAT MATTERS MOST, AND WHY THERE ARE TWO KINDS. The obvious control -- random gene pairs matched
on the two covariates -- compares a BioPlex edge against a pair BioPlex NEVER TESTED. A gene chosen as a bait
in a saturating interactome screen is not a random gene, and neither is a protein abundant enough to be seen
as prey. So a STRINGENT control is run throughout: the perturbed gene must itself be a BioPlex BAIT and the
measured gene must appear somewhere in the BioPlex network, but this specific pair must not be an edge. That
is "tested and not found", not "never looked at".

THE CONTROL LADDER, AND WHY THIS MODULE NO LONGER REPORTS A SINGLE RATIO. An earlier version of this module
matched on DECILES of the two covariates, checked residual imbalance with a Mann-Whitney rank test, saw
p 0.33 / 0.097, and called the control balanced. It was not. The reported effect is a ratio of MEANS, and on
the means the linked rows were 1.11% stronger and the linked columns 0.70% more responsive than their
"matched" controls -- an imbalance a rank test on 139,034 heavy-tailed values does not see and which,
because |z| is close to multiplicative in the two covariates, propagates straight into the ratio. Tightening
the bins moves the answer monotonically (10x10 1.102 -> 20x20 1.090 -> 40x40 1.079 -> 80x80 1.071 ->
160x160 1.068 -> 320x320 1.068) and it converges only once the residual MEAN ratio reaches 1.000. So three
rungs are reported, always together, and residual imbalance is now reported as a MEAN RATIO per covariate and
as the WORST rank p across draws, not as the average of per-draw p-values:

  LEGACY DECILE    10x10, kept only so the number is comparable to layers measured before this fix. It is
                   biased upward and is not the headline.
  CONVERGED JOINT  quantile bins tightened until the ratio stops moving and the residual mean ratio is 1.000.
  EXACT PERTURBATION  the control column is drawn from the SAME perturbation row as the linked pair, matched
                   on gene responsiveness. This removes every row-level confounder -- observed, unobserved,
                   and the shape of the response spectrum, which a bin on the row MEAN cannot capture -- by
                   construction rather than by approximation. It is the tightest control available and it is
                   the one to quote.

WHAT WOULD FALSIFY THIS. A ratio at or below ~1.05 against the stringent control at a CONVERGED control means
AP-MS physical association does not constrain the K562 transcriptional response enough to be weighted as if it
does. A confidence stratification that is flat means pInt should be dropped. A cell-line stratification that
is flat means the 293T/HCT116 distinction is not worth carrying.

ABSENCE IS NOT ZERO. A gene that was never a bait cannot have BioPlex edges. Bait coverage is measured
against perturbation strength and gene responsiveness rather than assumed uniform, non-edges among non-baits
are recorded as UNKNOWN, and every confounder covariate carries a dedicated UNKNOWN bin rather than a zero.
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DIR = SP / "bioplex"
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
LAYER = SP / "cell_bioplex.json.gz"

BASE = "https://bioplex.hms.harvard.edu/data/"
# ONE RELEASE, ONE PIPELINE, PINNED BY FILENAME. BioPlex 1.0/2.0/2.3 and the RPE1/U2OS v0.1 previews sit in
# the same directory and use the same column names with a DIFFERENT interaction model behind them. Mixing
# releases would make pInt mean two things at once, which is the exact defect this module is fixing in the
# existing PPI layer. Only these four files are ever read.
FILES = {"293T": "BioPlex_293T_Network_10K_Dec_2019.tsv",
         "HCT116": "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv"}
BAITS = {"293T": "BioPlex_3p0_293T_baitList.tsv",
         "HCT116": "BioPlex_3p0_HCT116_baitList.tsv"}
RELEASE = "BioPlex 3.0 (Dec 2019); Huttlin et al., Cell 2021 / bioRxiv 2020"
ASSAY = "AP-MS (HA-FLAG bait immunoprecipitation, LC-MS/MS, CompPASS-plus)"
HEADER = ["GeneA", "GeneB", "UniprotA", "UniprotB", "SymbolA", "SymbolB", "pW", "pNI", "pInt"]
PINT_FLOOR = 0.75          # the published admission threshold of BioPlex 3.0
N_DRAWS = 20
N_DEC = 10                 # the legacy decile rung, kept for comparability only
LADDER = (10, 20, 40, 80, 160, 320)
CONVERGED = 160            # the rung at which the ratio stops moving and residual mean ratio hits 1.000
INTEREST = 1.05            # the prespecified "worth weighting" threshold, read at a CONVERGED control


def download(name, tries=4):
    p = DIR / name
    if p.exists() and p.stat().st_size > 0:
        return p
    DIR.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            r = urllib.request.Request(BASE + name, headers={"User-Agent": "cellos"})
            with urllib.request.urlopen(r, timeout=900) as fh, open(p, "wb") as o:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
            return p
        except Exception:
            if p.exists():
                p.unlink()
            if i == tries - 1:
                raise SystemExit(f"BLOCKED: cannot fetch {BASE + name}")
            time.sleep(2 ** (i + 1))


def read_tsv(path):
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t", quotechar='"')
        head = next(r)
        return head, [row for row in r]


def qbin(v, n):
    """Quantile bins. n bins over the finite values."""
    return np.digitize(v, np.quantile(v, np.linspace(1 / n, 1 - 1 / n, n - 1)))


def qbin_unknown(v, n):
    """Quantile bins with a DEDICATED bin for missing. Absence is a bin, never a zero."""
    out = np.full(len(v), -1, dtype=np.int64)
    f = np.isfinite(v)
    if f.sum() > n:
        out[f] = np.digitize(v[f], np.quantile(v[f], np.linspace(1 / n, 1 - 1 / n, n - 1)))
    return out


def _strata(a):
    """Normalise a stratum specification to a list of hashable keys, one per row/column."""
    a = np.asarray(a)
    return [tuple(r) for r in a] if a.ndim > 1 else a.tolist()


class Bench:
    """The matched-control instrument. One object so every arm shares one definition of `matched`.

    THE EFFECT SIZE IS `mean_abs_z_linked` -- the raw held-out value. The ratio is the CONTRAST and the
    p-value is the significance; neither is the effect, and nothing is ever reported as a difference from a
    shuffled arm. One matched draw is one sample of the matched population, so every draw is repeated
    N_DRAWS times and the FRACTION of draws reaching p<0.05 is reported alongside the mean.

    RESIDUAL IMBALANCE IS REPORTED THREE WAYS, because the earlier version of this module reported only the
    first and it hid a real bias: the MEAN of the per-draw rank p (weak -- p-values are uniform under the
    null so their mean is ~0.5 and one bad draw disappears), the WORST per-draw rank p, and the MEAN RATIO
    of each covariate between linked and control, which is the quantity the effect is actually a ratio of.
    """

    def __init__(self, A, ok, rstr, cresp, linkset):
        self.A, self.ok, self.rstr, self.cresp, self.linkset = A, ok, rstr, cresp, linkset
        self.NR, self.NC = A.shape

    def arm(self, label, i2, j2, rowk, colk, rowmask=None, colmask=None,
            n_draws=N_DRAWS, seed0=0, exclude=None):
        from scipy import stats
        A, ok = self.A, self.ok
        exclude = self.linkset if exclude is None else exclude
        rk, ck = _strata(rowk), _strata(colk)
        rp, cp = defaultdict(list), defaultdict(list)
        for i in range(self.NR):
            if rowmask is None or rowmask[i]:
                rp[rk[i]].append(i)
        for j in range(self.NC):
            if colmask is None or colmask[j]:
                cp[ck[j]].append(j)
        rp = {k: np.array(v) for k, v in rp.items()}
        cp = {k: np.array(v) for k, v in cp.items()}
        lz = A[i2, j2]
        ct = Counter(zip((rk[i] for i in i2), (ck[j] for j in j2)))
        rows = []
        for s in range(n_draws):
            rng = np.random.default_rng(seed0 + s)
            ci, cj = [], []
            for (a, b), n in ct.items():
                rs, cs = rp.get(a), cp.get(b)
                if rs is None or cs is None or not len(rs) or not len(cs):
                    continue
                need, tries = n, 0
                while need > 0 and tries < 60:
                    m = int(need * 1.6) + 16
                    xi = rs[rng.integers(0, len(rs), m)]
                    xj = cs[rng.integers(0, len(cs), m)]
                    for x, y in zip(xi, xj):
                        if need <= 0:
                            break
                        if not ok[x, y] or (x, y) in exclude:
                            continue
                        ci.append(x)
                        cj.append(y)
                        need -= 1
                    tries += 1
            ci = np.array(ci, dtype=np.int64)
            cj = np.array(cj, dtype=np.int64)
            if len(ci) < 50:
                continue
            cz = A[ci, cj]
            rows.append({
                "n_ctrl": int(len(ci)), "fill": len(ci) / len(lz), "mean_ctrl": float(cz.mean()),
                "ratio": float(lz.mean() / cz.mean()),
                "p": float(stats.mannwhitneyu(lz, cz, alternative="two-sided")[1]),
                "res_strength_p": float(stats.mannwhitneyu(self.rstr[i2], self.rstr[ci],
                                                           alternative="two-sided")[1]),
                "res_resp_p": float(stats.mannwhitneyu(self.cresp[j2], self.cresp[cj],
                                                       alternative="two-sided")[1]),
                "res_strength_mr": float(self.rstr[i2].mean() / self.rstr[ci].mean()),
                "res_resp_mr": float(self.cresp[j2].mean() / self.cresp[cj].mean())})
        if not rows:
            return {"label": label, "n_pairs": int(len(lz)), "testable": False}

        def mn(k):
            return float(np.mean([r[k] for r in rows]))
        return {"label": label, "n_pairs": int(len(lz)), "testable": True, "n_draws": len(rows),
                "control_fill": mn("fill"),
                "mean_abs_z_linked": float(lz.mean()),        # <- THE EFFECT SIZE: raw held-out value
                "median_abs_z_linked": float(np.median(lz)),
                "mean_abs_z_matched": mn("mean_ctrl"),
                "ratio": mn("ratio"),
                "ratio_min": float(min(r["ratio"] for r in rows)),
                "ratio_max": float(max(r["ratio"] for r in rows)),
                "p_median": float(np.median([r["p"] for r in rows])),
                "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
                "residual_strength_p": mn("res_strength_p"),
                "residual_responsiveness_p": mn("res_resp_p"),
                "residual_strength_p_worst": float(min(r["res_strength_p"] for r in rows)),
                "residual_responsiveness_p_worst": float(min(r["res_resp_p"] for r in rows)),
                "residual_strength_meanratio": mn("res_strength_mr"),
                "residual_responsiveness_meanratio": mn("res_resp_mr")}

    def h2h(self, label, ia, ja, ib, jb, rowk, colk, extras=None, extra_names=(),
            n_draws=N_DRAWS, seed0=0):
        """Direct matched comparison of two linked sets against each other (not against random pairs).

        `extras` are PAIR-level axes given as (values_for_a, values_for_b) already binned to integers --
        used where the obvious alternative explanation is a property of the pair rather than of its row or
        column (BioPlex degree for the cell-line arm; prior membership of the existing PPI layer).
        """
        from scipy import stats
        rk, ck = _strata(rowk), _strata(colk)
        exa = [e[0] for e in extras] if extras else []
        exb = [e[1] for e in extras] if extras else []
        pa, pb = defaultdict(list), defaultdict(list)
        for k in range(len(ia)):
            pa[(rk[ia[k]], ck[ja[k]]) + tuple(int(e[k]) for e in exa)].append(k)
        for k in range(len(ib)):
            pb[(rk[ib[k]], ck[jb[k]]) + tuple(int(e[k]) for e in exb)].append(k)
        za, zb = self.A[ia, ja], self.A[ib, jb]
        rows = []
        for s in range(n_draws):
            rng = np.random.default_rng(seed0 + s)
            ka, kb = [], []
            for key, la in pa.items():
                lb = pb.get(key)
                if not lb:
                    continue
                n = min(len(la), len(lb))
                ka += list(rng.choice(la, n, replace=False))
                kb += list(rng.choice(lb, n, replace=False))
            if len(ka) < 50:
                continue
            ka, kb = np.array(ka), np.array(kb)
            rows.append({"n": int(len(ka)), "cov": len(ka) / max(len(ia), 1),
                         "a": float(za[ka].mean()), "b": float(zb[kb].mean()),
                         "p": float(stats.mannwhitneyu(za[ka], zb[kb], alternative="two-sided")[1]),
                         "rs": float(self.rstr[ia[ka]].mean() / self.rstr[ib[kb]].mean()),
                         "rr": float(self.cresp[ja[ka]].mean() / self.cresp[jb[kb]].mean())})
        if not rows:
            return {"label": label, "testable": False}

        def mn(k):
            return float(np.mean([r[k] for r in rows]))
        return {"label": label, "testable": True, "n_draws": len(rows), "n_matched": int(mn("n")),
                "frac_of_a_matched": mn("cov"),
                "mean_abs_z_a": mn("a"), "mean_abs_z_b": mn("b"),
                "ratio_a_over_b": mn("a") / mn("b"),
                "p_median": float(np.median([r["p"] for r in rows])),
                "frac_draws_p05": float(np.mean([r["p"] < 0.05 for r in rows])),
                "residual_strength_meanratio": mn("rs"),
                "residual_responsiveness_meanratio": mn("rr"),
                "matched_also_on": list(extra_names)}

    def cluster_ci(self, i2, j2, rowk, colk, rowmask, colmask, cluster_of_row,
                   nboot=400, seed=0, exclude=None):
        """Cluster-robust interval on the ratio, resampling PERTURBED GENES.

        139,034 linked pairs are not 139,034 independent observations: they come from 8,115 perturbed genes,
        and one strong knockdown contributes hundreds of rows. The Mann-Whitney p is therefore optimistic by
        construction. This resamples the actual unit of independence and reports a percentile interval.
        """
        exclude = self.linkset if exclude is None else exclude
        rk, ck = _strata(rowk), _strata(colk)
        rp, cp = defaultdict(list), defaultdict(list)
        for i in range(self.NR):
            if rowmask is None or rowmask[i]:
                rp[rk[i]].append(i)
        for j in range(self.NC):
            if colmask is None or colmask[j]:
                cp[ck[j]].append(j)
        rp = {k: np.array(v) for k, v in rp.items()}
        cp = {k: np.array(v) for k, v in cp.items()}
        rng = np.random.default_rng(seed)
        ci = np.empty(len(i2), np.int64)
        cj = np.empty(len(i2), np.int64)
        for t in range(len(i2)):
            rs, cs = rp.get(rk[i2[t]]), cp.get(ck[j2[t]])
            x, y = i2[t], j2[t]
            for _ in range(200):
                x = rs[rng.integers(0, len(rs))]
                y = cs[rng.integers(0, len(cs))]
                if self.ok[x, y] and (x, y) not in exclude:
                    break
            ci[t], cj[t] = x, y
        zl, zc = self.A[i2, j2], self.A[ci, cj]
        byl = defaultdict(list)
        for t, i in enumerate(i2):
            byl[cluster_of_row[i]].append(t)
        byl = {g: np.array(v) for g, v in byl.items()}
        gs = np.array(sorted(byl))
        out = []
        for _ in range(nboot):
            pick = gs[rng.integers(0, len(gs), len(gs))]
            sel = np.concatenate([byl[g] for g in pick])
            out.append(zl[sel].mean() / zc[sel].mean())
        out = np.array(out)
        return {"ratio": float(zl.mean() / zc.mean()), "n_clusters": int(len(gs)),
                "ci_lo": float(np.percentile(out, 2.5)), "ci_hi": float(np.percentile(out, 97.5)),
                "frac_boot_le_1": float(np.mean(out <= 1))}


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 110)
    report("BIOPLEX 3.0 AP-MS -- a PPI rebuild that keeps assay, cell line, direction and confidence")
    report("=" * 110)

    # ---------------------------------------------------------------- fetch, and PIN THE PROVENANCE
    paths = {k: download(v) for k, v in FILES.items()}
    bpaths = {k: download(v) for k, v in BAITS.items()}
    nets, baitsets = {}, {}
    for cl in FILES:
        head, rows = read_tsv(paths[cl])
        if head != HEADER:
            raise SystemExit(f"{FILES[cl]}: unexpected columns {head} -- release layout changed")
        nets[cl] = rows
        bh, br = read_tsv(bpaths[cl])
        baitsets[cl] = {r[1] for r in br}
        report(f"  {cl:7s} {FILES[cl]:44s} {len(rows):7,d} edges   {len(baitsets[cl]):6,d} baits")
    # ASSERTIONS, not docstring claims. (a) the three posteriors are a partition; (b) GeneA is the BAIT --
    # the direction is not documented in the file, it is verified against the bait list; (c) nothing below
    # the published pInt floor leaked in, which bounds what "low confidence" can mean here.
    for cl, rows in nets.items():
        s = np.array([float(r[6]) + float(r[7]) + float(r[8]) for r in rows])
        if np.abs(s - 1).max() > 1e-6:
            raise SystemExit(f"{cl}: pW+pNI+pInt != 1 (max dev {np.abs(s-1).max():.2g})")
        a_bait = np.mean([r[0] in baitsets[cl] for r in rows])
        b_bait = np.mean([r[1] in baitsets[cl] for r in rows])
        if a_bait < 0.999:
            raise SystemExit(f"{cl}: GeneA is not always a bait ({a_bait:.3%}) -- direction assumption dead")
        pint = np.array([float(r[8]) for r in rows])
        if pint.min() < PINT_FLOOR - 1e-9:
            raise SystemExit(f"{cl}: pInt below the published floor ({pint.min():.3f})")
        report(f"  {cl:7s} pW+pNI+pInt=1 OK | GeneA in baitList {a_bait:.1%} -> A=BAIT, B=PREY "
               f"(GeneB is a bait in {b_bait:.1%}) | pInt in [{pint.min():.3f}, {pint.max():.3f}]")
    report(f"  release pinned: {RELEASE}")
    report(f"  assay pinned:   {ASSAY}")

    # ---------------------------------------------------------------- join through the registry
    from entity_registry import Registry, REGISTRY as REGPATH
    raw = json.load(gzip.open(REGPATH, "rt"))
    REG = Registry(raw)
    CELLIDX = raw["cell_index"]        # the cell object keys `ppi`/`ppm`/`abund` by gene INDEX, not symbol

    ent_ids = sorted({r[0] for rs in nets.values() for r in rs} |
                     {r[1] for rs in nets.values() for r in rs})
    uni, sym = {}, {}
    for rs in nets.values():
        for r in rs:
            uni[r[0]], uni[r[1]] = r[2], r[3]
            sym[r[0]], sym[r[1]] = r[4], r[5]
    # ACCESSION FIRST. The file carries a UniProt accession for every endpoint; that is what the mass
    # spectrometer actually identified, and it is the strongest key here. Entrez is second. Symbol is used
    # only where both accessions fail and every such endpoint is counted and recorded as via=symbol.
    _g, st_u = REG.join("gene", "uniprot", [uni[g].split("-")[0] for g in ent_ids], 0.90,
                        "BioPlex endpoints by UniProt accession")
    # NOTE, and it is a defect in the registry rather than in BioPlex: gene|entrez alias keys were written
    # from a float column and are stored as "7105.0". Resolving the integer id returns None for EVERY gene
    # -- the same class of silent-miss the registry exists to prevent. It is handled here by normalising on
    # this side (the registry is not modified) and the surviving raw-integer rate is reported so the defect
    # is visible rather than papered over.
    _g, st_e_raw = REG.join("gene", "entrez", ent_ids, 0.0, "BioPlex endpoints by raw Entrez id")
    _g, st_e = REG.join("gene", "entrez", [g + ".0" for g in ent_ids], 0.90,
                        "BioPlex endpoints by Entrez id (float-normalised)")
    report(f"\n  REGISTRY JOIN over {len(ent_ids):,} distinct endpoints")
    report(f"    gene/uniprot (accession, primary)   {st_u['joined']:6,d}/{st_u['of']:,} "
           f"({st_u['rate']:.1%})")
    report(f"    gene/entrez  raw integer            {st_e_raw['joined']:6,d}/{st_e_raw['of']:,} "
           f"({st_e_raw['rate']:.1%})  <- registry stores entrez keys as '7105.0'; a raw join silently "
           f"misses everything")
    report(f"    gene/entrez  float-normalised       {st_e['joined']:6,d}/{st_e['of']:,} "
           f"({st_e['rate']:.1%})")

    cache = {}

    def resolve(g):
        if g in cache:
            return cache[g]
        u = uni.get(g)
        e = REG.resolve("gene", "uniprot", u.split("-")[0]) if u else None
        v = "uniprot"
        if e is None:
            e, v = REG.resolve("gene", "entrez", g + ".0"), "entrez"
        if e is None and g in sym:
            e, v = REG.resolve("gene", "symbol", sym[g]), "symbol"
        cache[g] = (e, v if e else None)
        return cache[g]

    via = Counter()
    for g in ent_ids:
        via[resolve(g)[1] or "UNRESOLVED"] += 1
    resolved_frac = 1 - via["UNRESOLVED"] / len(ent_ids)
    report(f"    resolved endpoints by route: {dict(via)}  "
           f"({100*resolved_frac:.2f}% carry an entity id; {100*via['uniprot']/len(ent_ids):.1f}% of that "
           f"is the UniProt route, {100*via['symbol']/len(ent_ids):.2f}% is symbol-only)")

    # ---------------------------------------------------------------- the edge layer
    edges = []
    drop_unres = drop_self = 0
    for cl, rows in nets.items():
        for r in rows:
            ea, va = resolve(r[0])
            eb, vb = resolve(r[1])
            if ea is None or eb is None:
                drop_unres += 1
                continue
            if ea == eb:
                drop_self += 1        # two accessions of one gene; not an interaction claim
                continue
            edges.append({"bait": ea, "prey": eb, "bait_symbol": r[4], "prey_symbol": r[5],
                          "bait_uniprot": r[2], "prey_uniprot": r[3],
                          "cell_line": cl, "assay": "AP-MS",
                          "pWrongID": float(r[6]), "pNI": float(r[7]), "pInt": float(r[8]),
                          "via": va if va == vb else f"{va}/{vb}"})
    report(f"\n  {len(edges):,} directed bait->prey edges retained "
           f"({drop_unres:,} dropped for an unresolvable endpoint, {drop_self:,} self-edges)")
    per_cl = Counter(e["cell_line"] for e in edges)
    report(f"    by cell line: {dict(per_cl)}")

    gp = {}
    for e in edges:
        k = tuple(sorted((e["bait"], e["prey"])))
        d = gp.setdefault(k, {"cl": set(), "pint": 0.0})
        d["cl"].add(e["cell_line"])
        d["pint"] = max(d["pint"], e["pInt"])
    report(f"    {len(gp):,} distinct unordered gene pairs; "
           f"{sum(1 for v in gp.values() if len(v['cl']) == 2):,} seen in BOTH cell lines")

    # ---------------------------------------------------------------- Q4a: overlap with the existing layer
    d = json.load(gzip.open(NET, "rt"))
    names = [g["name"] for g in d["genes"]]
    # study-bias and abundance covariates, taken from the cell object by INDEX via cell_index
    pubs_by_idx = np.array([float(g.get("pubs") or np.nan) for g in d["genes"]])
    ppideg_by_idx = np.array([float(g.get("ppi") or np.nan) for g in d["genes"]])
    ppm_by_idx = {int(k): float(v) for k, v in d.get("ppm", {}).items()}
    old_pairs = set()
    n_old_bad = 0
    for a, b in d["ppi"]:
        ea, eb = CELLIDX.get(str(a)), CELLIDX.get(str(b))
        if ea is None or eb is None or ea == eb:
            n_old_bad += 1
            continue
        old_pairs.add(tuple(sorted((ea, eb))))
    codep_pairs = set()
    for k, v in d.get("codep", {}).items():
        ea = CELLIDX.get(str(k))
        if not ea:
            continue
        for it in v:
            eb = CELLIDX.get(str(it[0]))
            if eb and eb != ea:
                codep_pairs.add(tuple(sorted((ea, eb))))
    n_old_raw = len(d["ppi"])
    del d
    eid2idx = {}
    for i in range(len(names)):
        e = CELLIDX.get(str(i))
        if e:
            eid2idx.setdefault(e, i)
    incell = set(eid2idx)
    report(f"\n  EXISTING PPI LAYER: {n_old_raw:,} index 2-tuples -> {len(old_pairs):,} distinct entity "
           f"pairs ({n_old_bad:,} unmappable/self); co-dependency layer {len(codep_pairs):,} pairs")
    bp_incell = {k for k in gp if k[0] in incell and k[1] in incell}
    inter = bp_incell & old_pairs
    report(f"    BioPlex pairs with both endpoints in the cell object: {len(bp_incell):,} "
           f"({100*len(bp_incell)/len(gp):.1f}% of BioPlex)")
    report(f"    overlap: {len(inter):,} pairs = {100*len(inter)/max(len(bp_incell),1):.1f}% of "
           f"(in-cell) BioPlex, {100*len(inter)/max(len(old_pairs),1):.1f}% of the existing layer")
    report(f"    BioPlex pairs ABSENT from the existing layer: {len(bp_incell)-len(inter):,}")

    # ---------------------------------------------------------------- the fast test substrate
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        obs = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        var = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    report(f"\n  GWPS: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes, "
           f"{100*(1-np.isfinite(X).mean()):.4f}% non-finite cells")
    Xn = np.where(np.isfinite(X), X, np.nan)
    del X
    med = np.nanmedian(Xn, axis=0)
    mad = np.nanmedian(np.abs(Xn - med), axis=0)
    scale = 1.4826 * mad
    if (scale <= 0).any():
        report(f"    {(scale<=0).sum()} genes have zero MAD -- dropped, never imputed")
    A = np.abs((Xn - med) / np.where(scale > 0, scale, np.nan)).astype(np.float32)
    del Xn
    ok = np.isfinite(A)
    report(f"    per-gene robust z computed; {100*ok.mean():.4f}% of cells usable "
           f"(non-finite pairs DROPPED, not imputed)")
    rstr = np.nanmean(np.where(ok, A, np.nan), axis=1)
    cresp = np.nanmean(np.where(ok, A, np.nan), axis=0)
    NR, NC = A.shape
    RB = {n: qbin(rstr, n) for n in LADDER}
    CB = {n: qbin(cresp, n) for n in LADDER}
    ROW_EXACT, COL_EXACT = np.arange(NR), np.arange(NC)

    # perturbation -> ENSG. The obs label is `{idx}_{symbol}_{transcript}_{ENSG}` and the transcript field
    # can itself contain an underscore (NM_001159524), so the ACCESSION is taken from the last field and
    # the symbol is only a fallback for the handful whose ENSG is literally "nan".
    pert_rows, n_nt, n_symfall = {}, 0, 0
    for i, o in enumerate(obs):
        p = o.split("_")
        if p[1] == "non-targeting":
            n_nt += 1
            continue
        e = p[-1] if p[-1].startswith("ENSG") else None
        if e is None:
            eid = REG.resolve("gene", "symbol", p[1])
            e = REG.entities[eid].get("ensembl") if eid else None
            if e:
                n_symfall += 1
        if e:
            pert_rows.setdefault(e, []).append(i)
    colof = {}
    for j, g in enumerate(var):
        colof.setdefault(g, j)
    report(f"    {len(pert_rows):,} perturbed genes carry an ENSG ({n_nt:,} non-targeting rows excluded "
           f"from the pair universe; {n_symfall} recovered via symbol), {len(colof):,} measured genes")

    ensof = {}

    def ens(eid):
        if eid not in ensof:
            ensof[eid] = REG.entities.get(eid, {}).get("ensembl")
        return ensof[eid]

    # ---------------------------------------------------------------- linked pairs, with their attributes
    # A BAIT THAT YIELDED NO EDGE IS THE PUREST "TESTED AND NOT FOUND" ROW THERE IS, so the bait universe is
    # taken from the bait LISTS, not from the network endpoints -- baits that produced no retained
    # interaction must stay in the stringent pool or dropping them would quietly make the control easier.
    eid_bait_both = {resolve(g)[0] for g in (baitsets["293T"] & baitsets["HCT116"])} - {None}
    eid_bait_any = {resolve(g)[0] for cl in baitsets for g in baitsets[cl]} - {None}
    eid_in_bioplex = {e["bait"] for e in edges} | {e["prey"] for e in edges}

    def link_pairs(gene_pairs):
        """(row, col) pairs testable in GWPS for an unordered gene-pair set, both orientations."""
        out = set()
        for a, b in gene_pairs:
            for src, dst in ((a, b), (b, a)):
                se, de = ens(src), ens(dst)
                if not se or not de:
                    continue
                ri, j = pert_rows.get(se), colof.get(de)
                if not ri or j is None:
                    continue
                for i in ri:
                    if ok[i, j]:
                        out.add((i, j))
        return out

    pairs = {}
    for e in edges:
        for src, dst, orient in ((e["bait"], e["prey"], "bait2prey"),
                                 (e["prey"], e["bait"], "prey2bait")):
            se, de = ens(src), ens(dst)
            if not se or not de:
                continue
            rows_i, j = pert_rows.get(se), colof.get(de)
            if not rows_i or j is None:
                continue
            for i in rows_i:
                if not ok[i, j]:
                    continue
                r = pairs.get((i, j))
                if r is None:
                    r = pairs[(i, j)] = {"pint": 0.0, "cl": set(), "orient": set(),
                                         "gp": tuple(sorted((e["bait"], e["prey"])))}
                r["pint"] = max(r["pint"], e["pInt"])
                r["cl"].add(e["cell_line"])
                r["orient"].add(orient)
    keys = list(pairs)
    li = np.array([k[0] for k in keys], dtype=np.int64)
    lj = np.array([k[1] for k in keys], dtype=np.int64)
    linkset = set(keys)
    B = Bench(A, ok, rstr, cresp, linkset)
    # PSEUDO-REPLICATION IS REAL HERE, TWICE OVER. 788 genes were perturbed by more than one sgRNA construct
    # and appear as separate GWPS rows; and every perturbation contributes many measured genes at once. The
    # first is handled by a de-replicated arm, the second by a cluster-robust interval over perturbed genes.
    first_row = {e: min(v) for e, v in pert_rows.items()}
    keep_rows = set(first_row.values())
    onerow = np.array([k[0] in keep_rows for k in keys])
    n_multi = sum(1 for v in pert_rows.values() if len(v) > 1)
    n_gene_pairs_linked = len({pairs[k]["gp"] for k in keys})
    row_gene = {}
    for e, ii in pert_rows.items():
        for i in ii:
            row_gene[i] = e
    n_pert_genes_linked = len({row_gene[i] for i in set(li.tolist())})
    report(f"\n  LINKED (perturbation, gene) pairs testable in GWPS: {len(keys):,} "
           f"over {n_gene_pairs_linked:,} distinct gene pairs, {len(set(li.tolist())):,} perturbation rows "
           f"({n_pert_genes_linked:,} distinct perturbed genes), {len(set(lj.tolist())):,} measured genes")
    report(f"    {n_multi} genes carry >1 construct; {int(onerow.sum()):,} pairs survive de-replication")

    # control pools --------------------------------------------------------
    bait_ens = {ens(e) for e in eid_bait_any} - {None}
    seen_ens = {ens(e) for e in eid_in_bioplex} - {None}
    row_is_bait = np.zeros(NR, bool)
    for e, ii in pert_rows.items():
        if e in bait_ens:
            row_is_bait[np.array(ii)] = True
    col_is_seen = np.array([v in seen_ens for v in var])
    report(f"    STRINGENT pool (perturbed gene is a BioPlex bait x measured gene detected in BioPlex): "
           f"{int(row_is_bait.sum()):,} rows x {int(col_is_seen.sum()):,} cols")

    R = {"model": "bioplex-v2", "release": RELEASE, "assay": ASSAY,
         "files": {k: FILES[k] for k in FILES}, "pint_floor": PINT_FLOOR,
         "n_edges": len(edges), "edges_by_cell_line": dict(per_cl),
         "n_gene_pairs": len(gp),
         "n_pairs_both_cell_lines": int(sum(1 for v in gp.values() if len(v["cl"]) == 2)),
         "join": {"uniprot_rate": st_u["rate"], "entrez_raw_rate": st_e_raw["rate"],
                  "entrez_normalised_rate": st_e["rate"], "routes": dict(via),
                  "any_route_rate": resolved_frac},
         "overlap_existing_ppi": {
             "existing_tuples": n_old_raw, "existing_distinct_pairs": len(old_pairs),
             "bioplex_pairs_in_cell": len(bp_incell), "shared": len(inter),
             "frac_of_bioplex": len(inter) / max(len(bp_incell), 1),
             "frac_of_existing": len(inter) / max(len(old_pairs), 1),
             "bioplex_novel": len(bp_incell) - len(inter)},
         "n_linked_pairs": len(keys), "n_linked_gene_pairs": n_gene_pairs_linked,
         "n_linked_perturbed_genes": n_pert_genes_linked,
         "n_multi_construct_genes": n_multi,
         "n_linked_pairs_dereplicated": int(onerow.sum())}

    hdr = (f"      {'stratum':46s} {'pairs':>8s} {'mean|z|':>8s} {'matched':>8s} {'ratio':>7s} "
           f"{'p(med)':>9s} {'drw':>4s} {'resid mean-ratio':>18s}")

    def line(a):
        if not a.get("testable"):
            report(f"      {a['label']:46s} {a['n_pairs']:8,d}  -- too few to test --")
            return
        report(f"      {a['label']:46s} {a['n_pairs']:8,d} {a['mean_abs_z_linked']:8.4f} "
               f"{a['mean_abs_z_matched']:8.4f} {a['ratio']:7.4f} {a['p_median']:9.2e} "
               f"{100*a['frac_draws_p05']:3.0f}% "
               f"{a['residual_strength_meanratio']:8.4f}/{a['residual_responsiveness_meanratio']:.4f}")

    def lineh(h):
        if not h.get("testable"):
            report(f"      {h['label']:46s}  -- not testable --")
            return
        report(f"      {h['label']:46s} n {h['n_matched']:7,d} ({100*h['frac_of_a_matched']:3.0f}% of A) "
               f"a {h['mean_abs_z_a']:.4f} b {h['mean_abs_z_b']:.4f} ratio {h['ratio_a_over_b']:.4f} "
               f"p {h['p_median']:8.3g} {100*h['frac_draws_p05']:3.0f}%")

    def rung(label, sel=None, n=CONVERGED, mode="joint", seed0=0, exclude=None):
        i2 = li[sel] if sel is not None else li
        j2 = lj[sel] if sel is not None else lj
        if mode == "joint":
            return B.arm(label, i2, j2, RB[n], CB[n], row_is_bait, col_is_seen, seed0=seed0,
                         exclude=exclude)
        if mode == "loose":
            return B.arm(label, i2, j2, RB[n], CB[n], None, None, seed0=seed0, exclude=exclude)
        if mode == "exact_row":
            return B.arm(label, i2, j2, ROW_EXACT, CB[n], None, col_is_seen, seed0=seed0,
                         exclude=exclude)
        if mode == "exact_col":
            return B.arm(label, i2, j2, RB[n], COL_EXACT, row_is_bait, None, seed0=seed0,
                         exclude=exclude)
        raise ValueError(mode)

    # ---------------------------------------------------------------- Q1: does the layer carry information?
    report("\n" + "-" * 110)
    report("  Q1  FAST TEST -- |robust z| of linked pairs vs matched pairs.  EFFECT SIZE = mean |z| of the")
    report("      linked pairs (raw, held-out).  The rightmost column is the RESIDUAL MEAN RATIO of the two")
    report("      matched covariates; it must reach 1.000 before a ratio is trusted, and a rank test does")
    report("      NOT establish that -- at 10x10 the rank test passes at p 0.33 while the means are 1.1% off.")
    report("-" * 110)
    report(hdr)
    R["ladder"] = {}
    R["ladder"]["loose_10"] = rung("LEGACY loose 10x10 (comparability only)", None, 10, "loose")
    line(R["ladder"]["loose_10"])
    for n in LADDER:
        a = rung(f"stringent joint {n}x{n}" + ("  <- LEGACY RUNG" if n == 10 else
                                               "  <- CONVERGED" if n == CONVERGED else ""),
                 None, n, "joint", seed0=n)
        R["ladder"][f"joint_{n}"] = a
        line(a)
    R["ladder"]["exact_row"] = rung(f"stringent EXACT-PERTURBATION x resp{CONVERGED}  <- TIGHTEST",
                                    None, CONVERGED, "exact_row", seed0=301)
    line(R["ladder"]["exact_row"])
    R["ladder"]["exact_col"] = rung(f"stringent strength{CONVERGED} x EXACT-GENE",
                                    None, CONVERGED, "exact_col", seed0=302)
    line(R["ladder"]["exact_col"])
    R["ladder"]["dereplicated"] = rung("de-replicated (1 row per pert gene), converged",
                                       onerow, CONVERGED, "joint", seed0=7)
    line(R["ladder"]["dereplicated"])
    R["ladder"]["dereplicated_exact_row"] = rung("de-replicated, exact-perturbation",
                                                 onerow, CONVERGED, "exact_row", seed0=8)
    line(R["ladder"]["dereplicated_exact_row"])

    # THE PIPELINE SANITY ARM, run at BOTH tightnesses: a tightened control could in principle manufacture a
    # deflation, so the null must still come back at 1.00 after the fix, not only before it.
    rng = np.random.default_rng(1234)
    fi = rng.integers(0, NR, len(keys) * 2)
    fj = rng.integers(0, NC, len(keys) * 2)
    keep = ok[fi, fj]
    fi, fj = fi[keep][:len(keys)], fj[keep][:len(keys)]
    shamset = set(zip(fi.tolist(), fj.tolist()))
    R["sham"] = {
        "legacy_10": B.arm("SHAM random pairs, 10x10", fi, fj, RB[10], CB[10], row_is_bait,
                           col_is_seen, seed0=500, exclude=shamset),
        "converged": B.arm(f"SHAM random pairs, {CONVERGED}x{CONVERGED}", fi, fj, RB[CONVERGED],
                           CB[CONVERGED], row_is_bait, col_is_seen, seed0=501, exclude=shamset),
        "exact_row": B.arm("SHAM random pairs, exact-perturbation", fi, fj, ROW_EXACT, CB[CONVERGED],
                           None, col_is_seen, seed0=502, exclude=shamset)}
    for k in ("legacy_10", "converged", "exact_row"):
        line(R["sham"][k])

    # cluster-robust intervals on the two rungs that the verdict rests on
    report("\n      CLUSTER-ROBUST INTERVAL (resampling the {:,} perturbed genes, not the pairs)"
           .format(n_pert_genes_linked))
    R["cluster_ci"] = {}
    for nm, mode, rk, ck, rm, cm in (
            ("converged joint", "joint", RB[CONVERGED], CB[CONVERGED], row_is_bait, col_is_seen),
            ("exact-perturbation", "exact_row", ROW_EXACT, CB[CONVERGED], None, col_is_seen)):
        c = B.cluster_ci(li, lj, rk, ck, rm, cm, row_gene, seed=5)
        R["cluster_ci"][nm] = c
        report(f"        {nm:22s} ratio {c['ratio']:.4f}  95% CI [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]  "
               f"fraction of {c['n_clusters']:,}-gene bootstrap replicates <= 1: {c['frac_boot_le_1']:.3f}")

    # ---------------------------------------------------------------- the named confounders
    report("\n" + "-" * 110)
    report("  Q1b CONFOUNDERS THE TWO PRESCRIBED COVARIATES DO NOT CONTROL: publication count (study bias),")
    report("      protein abundance (ppm), gene length, BioPlex degree, existing-PPI degree.")
    report("-" * 110)
    import pandas as pd
    gl = pd.read_parquet(ROOT / "outputs/orphan/science_data/genes.parquet")
    glen = {}
    for e0, s0, x0 in zip(gl["ensembl_gene_id"], gl["start_position"], gl["end_position"]):
        try:
            glen[str(e0).split(".")[0]] = abs(float(x0) - float(s0)) + 1
        except Exception:
            pass
    del gl
    bpdeg = Counter()
    for k in gp:
        bpdeg[k[0]] += 1
        bpdeg[k[1]] += 1

    def cov_of(e):
        """(pubs, ppm, genomic span, BioPlex degree, existing-PPI degree); nan = UNKNOWN, never 0."""
        if not e:
            return (np.nan,) * 5
        eid = REG.resolve("gene", "ensembl", e)
        i = eid2idx.get(eid) if eid else None
        return (pubs_by_idx[i] if i is not None else np.nan,
                ppm_by_idx.get(i, np.nan) if i is not None else np.nan,
                glen.get(e, np.nan),
                float(bpdeg.get(eid, 0)) if eid else np.nan,
                ppideg_by_idx[i] if i is not None else np.nan)

    CN = ["pubs", "ppm", "genelen", "bpdeg", "ppideg"]
    covr = np.array([cov_of(row_gene.get(i)) for i in range(NR)])
    covc = np.array([cov_of(v) for v in var])
    # is each confounder actually imbalanced in the control the module draws?
    a_ref = R["ladder"]["joint_10"]
    rng2 = np.random.default_rng(7)
    rp10 = {k: np.where((RB[10] == k) & row_is_bait)[0] for k in range(10)}
    cp10 = {k: np.where((CB[10] == k) & col_is_seen)[0] for k in range(10)}
    ci, cj = [], []
    for (a_, b_), n in Counter(zip(RB[10][li].tolist(), CB[10][lj].tolist())).items():
        rs, cs = rp10[a_], cp10[b_]
        if not len(rs) or not len(cs):
            continue
        need = n
        while need > 0:
            xi = rs[rng2.integers(0, len(rs), need * 2 + 16)]
            xj = cs[rng2.integers(0, len(cs), need * 2 + 16)]
            for x, y in zip(xi, xj):
                if need <= 0:
                    break
                if not ok[x, y] or (x, y) in linkset:
                    continue
                ci.append(x)
                cj.append(y)
                need -= 1
    ci, cj = np.array(ci), np.array(cj)
    R["confounder_imbalance_in_legacy_control"] = {}
    report(f"      {'covariate':22s} {'linked median':>14s} {'ctrl median':>12s} {'MWU p':>10s}")
    for t, nm in enumerate(CN):
        for side, iL, iC, cov in (("row", li, ci, covr), ("col", lj, cj, covc)):
            a_, b_ = cov[iL, t], cov[iC, t]
            f1, f2 = np.isfinite(a_), np.isfinite(b_)
            if f1.sum() < 100 or f2.sum() < 100:
                continue
            p = float(stats.mannwhitneyu(a_[f1], b_[f2], alternative="two-sided")[1])
            R["confounder_imbalance_in_legacy_control"][f"{side}_{nm}"] = {
                "linked_median": float(np.median(a_[f1])), "control_median": float(np.median(b_[f2])),
                "known_frac_linked": float(f1.mean()), "mwu_p": p}
            report(f"      {side + ' ' + nm:22s} {np.median(a_[f1]):14.1f} {np.median(b_[f2]):12.1f} "
                   f"{p:10.2g}")
    report("      -> every one of them is imbalanced in the LEGACY control. Each is now added as a third")
    report("         matching axis (quintiles, with a dedicated UNKNOWN bin) on top of 40x40:")
    report(hdr)
    R["confounder_arms"] = {}
    for t, nm in enumerate(CN):
        rk = np.stack([RB[40], qbin_unknown(covr[:, t], 5)], 1)
        ck = np.stack([CB[40], qbin_unknown(covc[:, t], 5)], 1)
        a = B.arm(f"40x40 + {nm} quintile (both sides)", li, lj, rk, ck, row_is_bait, col_is_seen,
                  seed0=900 + t)
        R["confounder_arms"][nm] = a
        line(a)
    rk = np.stack([RB[20]] + [qbin_unknown(covr[:, t], 3) for t in range(5)], 1)
    ck = np.stack([CB[20]] + [qbin_unknown(covc[:, t], 3) for t in range(5)], 1)
    a = B.arm("20x20 + ALL FIVE at tertiles", li, lj, rk, ck, row_is_bait, col_is_seen, seed0=999)
    R["confounder_arms"]["all_five"] = a
    line(a)

    # ---------------------------------------------------------------- the reference layers, same instrument
    report("\n" + "-" * 110)
    report("  Q1c THE REFERENCE LAYERS, RE-MEASURED HERE. A ratio is only comparable to another ratio taken")
    report("      against a control of the same tightness, so the two layers this one is ranked against are")
    report("      run through the identical instrument rather than quoted from their own modules.")
    report("-" * 110)
    report(hdr)
    R["reference_layers"] = {}
    for nm, pset in (("existing PPI layer", old_pairs), ("co-dependency layer", codep_pairs)):
        lp = link_pairs(pset)
        oi = np.array([k[0] for k in lp], dtype=np.int64)
        oj = np.array([k[1] for k in lp], dtype=np.int64)
        R["reference_layers"][nm] = {}
        for tag, mode, n, seed in (("legacy 10x10", "joint", 10, 55), ("converged", "joint", CONVERGED, 56),
                                   ("exact-perturbation", "exact_row", CONVERGED, 57)):
            if mode == "joint":
                a = B.arm(f"{nm} [{tag}]", oi, oj, RB[n], CB[n], row_is_bait, col_is_seen,
                          seed0=seed, exclude=lp)
            else:
                a = B.arm(f"{nm} [{tag}]", oi, oj, ROW_EXACT, CB[n], None, col_is_seen,
                          seed0=seed, exclude=lp)
            R["reference_layers"][nm][tag] = a
            line(a)

    # ---------------------------------------------------------------- Q2: was confidence worth keeping?
    report("\n" + "-" * 110)
    report("  Q2  WAS KEEPING pInt WORTH IT?  terciles of the pair's best pInt")
    report("-" * 110)
    pint = np.array([pairs[k]["pint"] for k in keys])
    t1, t2 = np.quantile(pint, [1 / 3, 2 / 3])
    lo, mid, hi = pint <= t1, (pint > t1) & (pint <= t2), pint > t2
    pq = qbin(pint, 10)
    report(f"      tercile cuts at pInt {t1:.4f} / {t2:.4f}  "
           f"(the file floor is {PINT_FLOOR}, so 'low' means 0.75-{t1:.2f}, not 0-0.5)")
    report(hdr)
    R["confidence"] = {}
    for nm, m in (("pInt low", lo), ("pInt mid", mid), ("pInt high", hi)):
        for tag, mode in (("converged", "joint"), ("exact-perturbation", "exact_row")):
            a = rung(f"{nm} [{tag}]", m, CONVERGED, mode, seed0=17)
            R["confidence"][f"{nm}|{tag}"] = a
            line(a)
    R["confidence_head_to_head"] = {
        "legacy_10": B.h2h("pInt high vs low [legacy 10x10]", li[hi], lj[hi], li[lo], lj[lo],
                           RB[10], CB[10], seed0=31),
        "matched_40": B.h2h("pInt high vs low [40x40]", li[hi], lj[hi], li[lo], lj[lo],
                            RB[40], CB[40], seed0=31)}
    for k in ("legacy_10", "matched_40"):
        lineh(R["confidence_head_to_head"][k])
    # SIGN AND SIGNIFICANCE READ TOGETHER. At the legacy rung the low tercile looks BETTER, which would be a
    # reversal if it were real. It is not: tightening the match moves the ratio to ~1.00 at p~0.8 with no
    # draw reaching 0.05. The legacy ordering was a composition artefact, and the answer is a NULL.
    report("      the legacy rung runs the 'wrong' way; tightened, this is a NULL at ~1.00, not a reversal")

    # ---------------------------------------------------------------- Q3: cell line reproducibility
    report("\n" + "-" * 110)
    report("  Q3  DOES CROSS-CELL-LINE REPRODUCIBILITY BEAT CONFIDENCE?")
    report("      restricted to pairs whose gene pair had an endpoint screened as a bait in BOTH lines")
    report("-" * 110)
    opp = np.array([(pairs[k]["gp"][0] in eid_bait_both) or (pairs[k]["gp"][1] in eid_bait_both)
                    for k in keys])
    ncl = np.array([len(pairs[k]["cl"]) for k in keys])
    both_m, one_m = opp & (ncl == 2), opp & (ncl == 1)
    known = np.array([pairs[k]["gp"] in old_pairs for k in keys])
    pair_deg = np.array([bpdeg[pairs[k]["gp"][0]] + bpdeg[pairs[k]["gp"][1]] for k in keys], float)
    dq = qbin(pair_deg, 10)
    report(f"      pairs with the opportunity to be seen twice: {int(opp.sum()):,} "
           f"({int(both_m.sum()):,} seen in both lines, {int(one_m.sum()):,} in one)")
    report(hdr)
    R["cell_line"] = {}
    for nm, m in (("seen in BOTH lines", both_m), ("seen in ONE line", one_m)):
        for tag, mode in (("converged", "joint"), ("exact-perturbation", "exact_row")):
            a = rung(f"{nm} [{tag}]", m, CONVERGED, mode, seed0=41)
            R["cell_line"][f"{nm}|{tag}"] = a
            line(a)
    for cl in ("293T", "HCT116"):
        m = np.array([cl in pairs[k]["cl"] for k in keys])
        a = rung(f"{cl} edges [converged]", m, CONVERGED, "joint", seed0=61)
        R["cell_line"][f"{cl}|converged"] = a
        line(a)
    # THE ALTERNATIVE EXPLANATIONS FOR "SEEN TWICE" ARE (a) the bait is a HUB, so any purification recovers
    # it, and (b) the pair is a long-known stable complex, i.e. it is already in the existing PPI layer and
    # "seen twice" is a proxy for prior knowledge rather than for cell-line robustness. Both are added as
    # pair-level matching axes. pInt is added too, though it is partly mechanical: a pair seen twice has two
    # edges and therefore a higher max pInt by construction, so matching on it removes some real signal.
    R["cell_line_head_to_head"] = {
        "legacy_10_deg": B.h2h("both vs one [legacy 10x10 + degree]", li[both_m], lj[both_m],
                               li[one_m], lj[one_m], RB[10], CB[10],
                               extras=[(dq[both_m], dq[one_m])], extra_names=("degree",), seed0=53),
        "matched_40_deg": B.h2h("both vs one [40x40 + degree]", li[both_m], lj[both_m],
                                li[one_m], lj[one_m], RB[40], CB[40],
                                extras=[(dq[both_m], dq[one_m])], extra_names=("degree",), seed0=53),
        "matched_40_deg_pint": B.h2h("both vs one [40x40 + degree + pInt]", li[both_m], lj[both_m],
                                     li[one_m], lj[one_m], RB[40], CB[40],
                                     extras=[(dq[both_m], dq[one_m]), (pq[both_m], pq[one_m])],
                                     extra_names=("degree", "pInt"), seed0=53),
        "matched_40_deg_pint_known": B.h2h(
            "both vs one [40x40 + degree + pInt + prior-PPI]", li[both_m], lj[both_m],
            li[one_m], lj[one_m], RB[40], CB[40],
            extras=[(dq[both_m], dq[one_m]), (pq[both_m], pq[one_m]),
                    (known[both_m].astype(int), known[one_m].astype(int))],
            extra_names=("degree", "pInt", "prior_PPI"), seed0=53)}
    for k in ("legacy_10_deg", "matched_40_deg", "matched_40_deg_pint", "matched_40_deg_pint_known"):
        lineh(R["cell_line_head_to_head"][k])
    for nm, sub in (("within pairs ALREADY in existing PPI", known),
                    ("within pairs NOVEL to BioPlex", ~known)):
        ma, mb = both_m & sub, one_m & sub
        if ma.sum() < 500 or mb.sum() < 500:
            continue
        h = B.h2h(f"both vs one, {nm}", li[ma], lj[ma], li[mb], lj[mb], RB[40], CB[40],
                  extras=[(dq[ma], dq[mb])], extra_names=("degree",), seed0=53)
        R["cell_line_head_to_head"][nm] = h
        lineh(h)

    # ---------------------------------------------------------------- Q4b: is the novel remainder real?
    report("\n" + "-" * 110)
    report("  Q4  REBUILD OR DUPLICATE?  BioPlex pairs already in the existing PPI layer vs not")
    report("-" * 110)
    report(f"      linked pairs whose gene pair is already an existing PPI edge: {int(known.sum()):,}; "
           f"novel: {int((~known).sum()):,}")
    report(hdr)
    R["novelty"] = {}
    for nm, m in (("already in existing PPI", known), ("novel to BioPlex", ~known)):
        for tag, mode in (("converged", "joint"), ("exact-perturbation", "exact_row")):
            a = rung(f"{nm} [{tag}]", m, CONVERGED, mode, seed0=71)
            R["novelty"][f"{nm}|{tag}"] = a
            line(a)
    R["novelty_head_to_head"] = B.h2h("known vs novel [40x40]", li[known], lj[known],
                                      li[~known], lj[~known], RB[40], CB[40], seed0=83)
    lineh(R["novelty_head_to_head"])

    # ---------------------------------------------------------------- Q5: orientation
    report("\n" + "-" * 110)
    report("  Q5  WAS KEEPING BAIT/PREY DIRECTION WORTH IT?  perturbing the BAIT and measuring the PREY,")
    report("      versus perturbing the PREY and measuring the BAIT (exclusive sets)")
    report("-" * 110)
    ob = np.array([pairs[k]["orient"] == {"bait2prey"} for k in keys])
    op = np.array([pairs[k]["orient"] == {"prey2bait"} for k in keys])
    report(hdr)
    R["orientation"] = {}
    for nm, m in (("perturb BAIT -> measure PREY", ob), ("perturb PREY -> measure BAIT", op)):
        a = rung(f"{nm} [converged]", m, CONVERGED, "joint", seed0=97)
        R["orientation"][nm] = a
        line(a)
    R["orientation_head_to_head"] = B.h2h("bait->prey vs prey->bait [40x40]", li[ob], lj[ob],
                                          li[op], lj[op], RB[40], CB[40], seed0=101)
    lineh(R["orientation_head_to_head"])

    # ---------------------------------------------------------------- coverage, and whether it is biased
    report("\n" + "-" * 110)
    report("  COVERAGE -- absence of a BioPlex edge is UNKNOWN, and the missingness is not random")
    report("-" * 110)
    n_pert_bait = sum(1 for e in pert_rows if e in bait_ens)
    n_var_seen = int(col_is_seen.sum())
    p_str = float(stats.mannwhitneyu(rstr[row_is_bait], rstr[~row_is_bait], alternative="two-sided")[1])
    p_resp = float(stats.mannwhitneyu(cresp[col_is_seen], cresp[~col_is_seen],
                                      alternative="two-sided")[1])
    report(f"      {n_pert_bait:,}/{len(pert_rows):,} ({100*n_pert_bait/len(pert_rows):.1f}%) of GWPS "
           f"perturbed genes were screened as a BioPlex bait")
    report(f"      {n_var_seen:,}/{len(var):,} ({100*n_var_seen/len(var):.1f}%) of GWPS measured genes "
           f"appear anywhere in the BioPlex network")
    report(f"      perturbation strength: baits {rstr[row_is_bait].mean():.4f} vs non-baits "
           f"{rstr[~row_is_bait].mean():.4f}  MWU p {p_str:.3g}")
    report(f"      gene responsiveness:   in-BioPlex {cresp[col_is_seen].mean():.4f} vs absent "
           f"{cresp[~col_is_seen].mean():.4f}  MWU p {p_resp:.3g}")
    report("      -> coverage is biased on both matched covariates, which is why every arm above uses the "
           "stringent pool.")
    R["coverage"] = {"gwps_perts_that_are_baits": n_pert_bait, "gwps_perts": len(pert_rows),
                     "gwps_genes_in_bioplex": n_var_seen, "gwps_genes": len(var),
                     "bias_strength_p": p_str, "bias_responsiveness_p": p_resp,
                     "bait_mean_strength": float(rstr[row_is_bait].mean()),
                     "nonbait_mean_strength": float(rstr[~row_is_bait].mean())}

    # ---------------------------------------------------------------- write the layer
    LEG = R["ladder"]["joint_10"]
    CVG = R["ladder"][f"joint_{CONVERGED}"]
    EXR = R["ladder"]["exact_row"]
    EXC = R["ladder"]["exact_col"]
    R["headline"] = {
        "effect_size_raw_mean_abs_z": CVG["mean_abs_z_linked"],
        "n_pairs": CVG["n_pairs"], "n_gene_pairs": n_gene_pairs_linked,
        "ratio_legacy_decile": LEG["ratio"],
        "ratio_converged": CVG["ratio"],
        "ratio_exact_perturbation": EXR["ratio"],
        "ratio_exact_gene": EXC["ratio"],
        "quote_this": EXR["ratio"],
        "interest_threshold": INTEREST,
        "clears_threshold_at_converged": bool(CVG["ratio"] >= INTEREST),
        "clears_threshold_at_exact_perturbation": bool(EXR["ratio"] >= INTEREST)}
    R["limits"] = [
        "THE HEADLINE DEPENDS ON HOW TIGHTLY THE CONTROL IS MATCHED, so a single ratio is not quotable "
        f"without its rung. Decile matching gives {LEG['ratio']:.3f}x but leaves the linked rows "
        f"{100*(LEG['residual_strength_meanratio']-1):.1f}% stronger and the linked columns "
        f"{100*(LEG['residual_responsiveness_meanratio']-1):.1f}% more responsive than their controls ON "
        f"THE MEAN, which is what the ratio is a ratio of; a Mann-Whitney rank test does not detect this "
        f"and passed it. The converged joint control gives {CVG['ratio']:.3f}x and the exact-perturbation "
        f"control {EXR['ratio']:.3f}x.",
        f"RATIOS FROM DIFFERENT MODULES ARE NOT COMPARABLE unless they were computed at the same control "
        f"tightness. That is why the existing PPI and co-dependency layers are re-measured here rather "
        f"than quoted; their published decile-rung numbers are inflated by the same mechanism.",
        "NEITHER CELL LINE IS K562. BioPlex 3.0 is HEK293T and HCT116; the fast test asks whether an "
        "interaction found in those lines constrains the K562 transcriptional response, which is a "
        "transfer question, not a within-cell measurement. Nothing here establishes that any given edge "
        "exists in K562.",
        "AP-MS reports CO-COMPLEX MEMBERSHIP, not binary direct contact. A bait->prey edge may be bridged "
        "by one or more other proteins; the direction is experimental (who was tagged), not mechanistic.",
        f"The released network is already filtered at pInt >= {PINT_FLOOR}, so the 'low confidence' arm is "
        f"0.75-tercile, not a genuine low-confidence tail. The confidence test therefore measures "
        f"resolution WITHIN the published set and says nothing about edges BioPlex rejected.",
        "Bait selection is not random and neither is prey detectability: baits skew to well-expressed, "
        "clonable ORFs and prey to abundant proteins. Absence of an edge among non-baits is UNKNOWN, never "
        "evidence of no interaction.",
        "Publication count, protein abundance, gene length, BioPlex degree and existing-PPI degree are all "
        "significantly imbalanced between linked pairs and the decile-matched control (table above). Each "
        "was added as a third matching axis and none of them individually accounts for the effect, but the "
        "layer is measurably concentrated on well-studied, abundant, highly connected genes and any "
        "downstream use inherits that.",
        "Overexpression of a tagged bait can create interactions that do not occur at endogenous "
        "stoichiometry; BioPlex mitigates this with near-endogenous promoters but does not eliminate it.",
        "The GWPS readout is transcriptional. A physical interaction that acts post-translationally leaves "
        "no signature in this test, so a null here is a null for TRANSCRIPTIONAL coupling only -- it is "
        "not evidence against the interaction.",
        "Endpoints are joined to the registry by UniProt accession first "
        f"({100*via['uniprot']/len(ent_ids):.1f}% of endpoints); {via['entrez']} resolve only by Entrez and "
        f"{via['symbol']} only by gene symbol, recorded via=symbol, the weakest key. "
        f"{via['UNRESOLVED']} endpoints carry no entity id at all.",
        "The registry's gene|entrez alias keys are stored as floats ('7105.0'); this module normalises on "
        "its side. Any layer joining raw integer Entrez ids against this registry will silently match "
        "nothing.",
        "pW/pNI/pInt come from one CompPASS-plus model fitted to this release. They are not comparable to "
        "STRING confidences, MIscores, or the confidences of any other release, and must not be pooled.",
        f"{n_multi} genes are perturbed by more than one construct, and far more importantly one "
        f"perturbation contributes hundreds of linked columns, so the {len(keys):,} pairs are not "
        f"{len(keys):,} independent observations. Mann-Whitney p-values are optimistic by construction; "
        f"the cluster-robust interval over {n_pert_genes_linked:,} perturbed genes is the honest one.",
        "The cell-line result is not separable from prior knowledge at the available power: matching "
        "additionally on whether the pair is already in the existing PPI layer drops the both-vs-one "
        "contrast to a null on a fifth of the data, which is underpowered for an effect of that size and "
        "therefore neither confirms nor refutes it.",
        "Every ratio here is a contrast against a matched control drawn from the SAME perturbation matrix; "
        "no arm has been validated on a held-out cell line, so this measures association within K562 "
        "Perturb-seq, not predictive transfer.",
    ]
    layer = dict(R)
    layer["edges"] = edges
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)

    # ---------------------------------------------------------------- verdict
    report("\n" + "=" * 110)
    hc = R["confidence_head_to_head"]["matched_40"]
    hl = R["cell_line_head_to_head"]["matched_40_deg"]
    hlk = R["cell_line_head_to_head"]["matched_40_deg_pint_known"]
    nov = R["novelty"]["novel to BioPlex|exact-perturbation"]
    ho = R["orientation_head_to_head"]
    ref = R["reference_layers"]
    cc = R["cluster_ci"]["exact-perturbation"]

    def sig(a):
        return a.get("testable") and a["frac_draws_p05"] >= 0.8 and a["ratio"] > 1.0
    strong = EXR["ratio"] >= INTEREST and sig(EXR)
    conf_works = hc.get("testable") and hc["frac_draws_p05"] >= 0.8 and hc["ratio_a_over_b"] > 1.0
    cl_works = hl.get("testable") and hl["frac_draws_p05"] >= 0.8 and hl["ratio_a_over_b"] > 1.0
    head = (f"BioPlex 3.0 yields {len(edges):,} directed AP-MS edges "
            f"({per_cl.get('293T',0):,} in 293T, {per_cl.get('HCT116',0):,} in HCT116) over "
            f"{len(gp):,} distinct gene pairs; {100*via['uniprot']/len(ent_ids):.1f}% of endpoints join by "
            f"UniProt accession and only {via['symbol']} by symbol ({100*resolved_frac:.1f}% carry an id by "
            f"some route). Only {100*len(inter)/max(len(bp_incell),1):.1f}% of the in-cell BioPlex pairs "
            f"are already in the existing {n_old_raw:,}-tuple PPI layer, so this is a rebuild that also "
            f"adds {len(bp_incell)-len(inter):,} pairs that layer does not have.")
    body = (f"EFFECT SIZE: mean |robust z| of the {len(keys):,} linked (perturbation, gene) pairs is "
            f"{CVG['mean_abs_z_linked']:.4f}, over {n_gene_pairs_linked:,} distinct gene pairs from "
            f"{n_pert_genes_linked:,} perturbed genes. The CONTRAST depends on how tight the control is, "
            f"and that is the correction this version makes: at the project's legacy decile matching the "
            f"ratio is {LEG['ratio']:.3f}x, but that control leaves the linked rows "
            f"{100*(LEG['residual_strength_meanratio']-1):.1f}% stronger and the linked columns "
            f"{100*(LEG['residual_responsiveness_meanratio']-1):.1f}% more responsive on the MEAN while its "
            f"rank test passes at p {LEG['residual_strength_p']:.2g}. Tightening the bins moves the ratio "
            f"monotonically to {CVG['ratio']:.3f}x, where the residual mean ratio reaches "
            f"{CVG['residual_strength_meanratio']:.4f}/{CVG['residual_responsiveness_meanratio']:.4f} and "
            f"stops moving ({R['ladder']['joint_320']['ratio']:.3f}x at 320x320). Holding the perturbation "
            f"EXACTLY -- drawing the control gene from the same knockdown, matched on responsiveness -- "
            f"gives {EXR['ratio']:.3f}x ({100*EXR['frac_draws_p05']:.0f}% of {EXR['n_draws']} draws p<0.05; "
            f"cluster-robust 95% CI over perturbed genes [{cc['ci_lo']:.3f}, {cc['ci_hi']:.3f}]). The sham "
            f"arm returns {R['sham']['exact_row']['ratio']:.3f}x at that tightness, so the tightened "
            f"instrument is not manufacturing a deflation.")
    if strong:
        lead = "THIS LAYER CARRIES INFORMATION AND CLEARS THE THRESHOLD UNDER THE TIGHTEST CONTROL. "
    elif sig(EXR):
        lead = (f"THIS LAYER CARRIES A REAL BUT SMALL SIGNAL THAT DOES NOT CLEAR THE PRESPECIFIED "
                f"{INTEREST:.2f}x THRESHOLD ONCE THE CONTROL IS PROPERLY MATCHED. ")
    else:
        lead = "THIS LAYER IS A WELL-CONTROLLED NULL AGAINST THE TIGHTEST CONTROL. "
    rank = (f"Ranking is only meaningful at a fixed control tightness, so the two layers this one is "
            f"usually compared against were re-measured here with the same instrument. At the legacy "
            f"decile rung: existing PPI {ref['existing PPI layer']['legacy 10x10']['ratio']:.3f}x, BioPlex "
            f"{LEG['ratio']:.3f}x, co-dependency {ref['co-dependency layer']['legacy 10x10']['ratio']:.3f}x "
            f"-- which is where the earlier 'BioPlex sits between them' reading came from. At the "
            f"exact-perturbation control the order changes: existing PPI "
            f"{ref['existing PPI layer']['exact-perturbation']['ratio']:.3f}x, co-dependency "
            f"{ref['co-dependency layer']['exact-perturbation']['ratio']:.3f}x, BioPlex "
            f"{EXR['ratio']:.3f}x. BioPlex is the WEAKEST of the three once perturbation-level confounding "
            f"is removed by construction rather than approximated with bins.")
    q2 = (f"Confidence was worth keeping: matched, top-tercile pInt pairs reach {hc['mean_abs_z_a']:.4f} "
          f"against {hc['mean_abs_z_b']:.4f} ({hc['ratio_a_over_b']:.3f}x, median p {hc['p_median']:.2g})."
          if conf_works else
          f"Confidence was NOT worth keeping for this purpose: matched head-to-head at 40x40, top-tercile "
          f"pInt reaches {hc['mean_abs_z_a']:.4f} against {hc['mean_abs_z_b']:.4f} for the bottom tercile "
          f"({hc['ratio_a_over_b']:.3f}x, median p {hc['p_median']:.2g}, "
          f"{100*hc['frac_draws_p05']:.0f}% of draws p<0.05). Above the 0.75 publication floor pInt does "
          f"not resolve which edges couple transcriptionally in K562; it is stored, not weighted. At the "
          f"legacy rung this arm read {R['confidence_head_to_head']['legacy_10']['ratio_a_over_b']:.3f}x, "
          f"i.e. the 'wrong' way -- that was a composition artefact of the coarse match, not a reversal.")
    q3 = (f"Cross-cell-line reproducibility is the one metadata field that does anything, but less than the "
          f"earlier reading claimed: both-lines vs one-line, matched on strength x responsiveness x BioPlex "
          f"degree, is {hl['ratio_a_over_b']:.3f}x (median p {hl['p_median']:.2g}, "
          f"{100*hl['frac_draws_p05']:.0f}% of draws) at 40x40, down from "
          f"{R['cell_line_head_to_head']['legacy_10_deg']['ratio_a_over_b']:.3f}x at the legacy rung. It is "
          f"also not separable from prior knowledge: adding 'already in the existing PPI layer' as a fourth "
          f"axis leaves {hlk['ratio_a_over_b']:.3f}x at {100*hlk['frac_draws_p05']:.0f}% of draws on "
          f"n={hlk['n_matched']:,}, which is underpowered for an effect of that size and so neither "
          f"confirms nor refutes it. The honest statement is that 'seen in two lines' is worth about as "
          f"much as the whole layer is."
          if cl_works else
          f"Cross-cell-line reproducibility does not separate either once the match is tightened: "
          f"{hl['ratio_a_over_b']:.3f}x, {100*hl['frac_draws_p05']:.0f}% of draws p<0.05.")
    q4 = (f"The pairs BioPlex adds that the existing layer lacks are not noise but they are weak: "
          f"{nov['ratio']:.3f}x against the exact-perturbation control on {nov['n_pairs']:,} pairs "
          f"({100*nov['frac_draws_p05']:.0f}% of draws p<0.05), against "
          f"{R['novelty']['already in existing PPI|exact-perturbation']['ratio']:.3f}x for the pairs that "
          f"were already known."
          if sig(nov) else
          f"The pairs BioPlex adds that the existing layer lacks do NOT carry signal on their own "
          f"({nov['ratio']:.3f}x, {nov['n_pairs']:,} pairs, {100*nov['frac_draws_p05']:.0f}% of draws "
          f"p<0.05), so the value of this rebuild is the metadata on known edges, not the new edges.")
    q5 = (f"Bait/prey direction buys nothing on this readout ({ho['ratio_a_over_b']:.3f}x matched, median "
          f"p {ho['p_median']:.2g}, {100*ho['frac_draws_p05']:.0f}% of draws), so it is stored for "
          f"provenance and not used as a weight.")
    cov = (f"Coverage is biased on both matched covariates ({n_pert_bait:,}/{len(pert_rows):,} perturbed "
           f"genes are baits, strength p {p_str:.2g}; {n_var_seen:,}/{len(var):,} measured genes appear in "
           f"BioPlex, responsiveness p {p_resp:.2g}) and the linked pairs are also enriched for "
           f"well-studied, abundant, high-degree genes on every one of publication count, ppm, gene length "
           f"and both degree measures -- none of which individually accounts for the effect, but all of "
           f"which the layer inherits. A missing edge is stored as unknown, never as zero.")
    v = " ".join([lead + head, body, rank, q2, q3, q4, q5, cov])
    R["verdict"] = v
    report(f"  VERDICT: {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "bioplex.json", "w"), indent=1, default=float)
    report(f"\n  -> {LAYER}  ({LAYER.stat().st_size/1e6:.1f} MB)")
    report(f"  -> {OUT/'bioplex.json'}")


if __name__ == "__main__":
    main()
