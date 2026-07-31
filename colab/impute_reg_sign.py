"""IMPUTE THE SIGN OF THE 558,005 UNSIGNED CURATED REGULATORY EDGES -- and the answer is mostly no.

THE FINDING, PUT FIRST SO NOBODY READS THE MOTIVATION AS THE RESULT. Sign is imputable for the edges that
already have one and essentially NOT imputable for the ones that do not, and the reason is structural rather
than statistical. Three measurements, in the order they were made:

  1  THE CONVENTION IS REAL AND STRONG. On the 20,118 curated edges that DO carry a sign and are testable in
     Replogle K562, the curated sign predicts the measured perturbational sign at +13.6 points over a
     strength-matched swapped-source control (rising to +20.7 on the |z| >= 3 subset). Curated sign carries
     direction. So the layer's signed 8.8% is worth something and the question "can the other 91.2% be
     filled in" is worth asking.
  2  THE LOOKUP ROUTE IS CLOSED BY CONSTRUCTION, AT EXACTLY ZERO EDGES. CollecTRI covers 43,000 of the
     54,128 signed edges and 0 of the 558,005 unsigned ones. The signed subset IS CollecTRI/TRRUST. The
     unsigned subset came from somewhere else -- its top sources are CTCF, ETS1, TFAP2C, STAT1 with
     4,000-5,000 targets each, which is the shape of a ChIP/motif compendium, not of a curated
     activation/repression statement. So an imputation cannot be a join; it has to generalise across a
     provenance boundary.
  3  THERE IS A PER-SOURCE DIRECTION EFFECT TO IMPUTE, AND THE CURATED LAYER DOES NOT KNOW IT. Split each
     source's unsigned targets in half, estimate that source's measured direction bias on one half and
     predict the other: +7.6 points over the swapped-source control. That is the CEILING on any source-level
     imputation. But the curated activator fraction of a source correlates with that measured bias at
     Spearman +0.009 (p = 0.91, n = 149 sources). The signal exists; the annotation does not see it.

WHAT IS MEASURED, AND AGAINST WHAT CONTROL, BEFORE ANY NUMBER APPEARS.

  QUANTITY      For a curated edge (source S, target T), the sign of T's response to knocking S down:
                sign(robust z) where z = (lfc - median over perturbations) / (1.4826 * MAD), per gene column
                -- the identical construction used by colab/causal_reg.py, reused rather than re-derived.
  CONTROL       SWAPPED SOURCE. Keep the target T and the model's predicted sign; read the measured sign at
                a DIFFERENT perturbation S' drawn from the same perturbation-strength DECILE. 0.5 is the
                wrong null: a shared stress/proliferation axis gives every target a preferred direction under
                any perturbation, so sign agreement sits away from 0.5 with no edge at all. The number that
                means anything is agreement MINUS the swapped-source arm, over 20 draws, reported with the
                fraction of draws the edge arm won.
  SECOND        LABEL SHUFFLE. Measured signs permuted; the whole pipeline rerun. This establishes
  CONTROL       significance, never effect size -- the effect size reported is always the RAW held-out
                agreement, never a difference from a shuffle.
  THIRD         DEGREE-MATCHED REWIRING. Every graph a feature touches (`reg`, `ppi`, `coexpr`, complex
  CONTROL       membership) is rewired by a configuration model that preserves each node's degree EXACTLY
                -- for the directed `reg` graph by permuting the (target, sign) stubs, which leaves every
                source's out-degree and every target's in-degree untouched. Uniform rewiring is not used
                anywhere. If the degree-matched control reproduces the effect, that IS the result.

THE CIRCULARITY, WHICH IS THE WHOLE DIFFICULTY, AND HOW IT IS SOLVED.
  * Labels are K562 perturbational signs. Scoring on the same K562 rows that trained the model would report
    the training data. So the split is BY SOURCE GENE: every edge from one TF lands in one fold, and a
    held-out TF's own perturbation row is never seen at fit time.
  * A source's measured direction bias is therefore UNAVAILABLE as a feature for a held-out TF by
    construction. That is not a limitation to work around, it is the point: the ceiling in section 2 exists
    precisely because that quantity is informative, and the imputation problem is to reach it from
    annotations instead.
  * The target's global direction bias IS available for a held-out TF (targets recur across TFs), so it is
    computed LEAVE-FOLD-OUT (only training-fold perturbations enter it) and is scored as its OWN ARM. It is
    the confound the swapped-source control is built to remove, and it is reported separately so the removal
    can be seen happening.
  * Transfer is scored on Replogle RPE1 (rpe1_pseudobulk.npz), a cell line no layer in this project was
    built from, using the same held-out-source folds.
  * CV seeds draw genuinely different partitions: each seed re-randomises the source -> fold assignment, and
    the module reports the pairwise co-assignment rate between seeds (~1/K if the partitions differ, 1.0 if
    somebody rotated fold labels, which gives an sd of 0.0000 and means nothing).

THE DECISION THRESHOLD, FIXED BEFORE THE NUMBERS. The imputation is declared USABLE only if all four hold:
    (a) K562 held-out-source agreement exceeds its swapped-source control by >= 3.0 points, in >= 18 of 20
        control draws;
    (b) the same increment on RPE1 is > 0 with >= 18 of 20 draws won;
    (c) the source+edge arm -- which contains NO K562 measurement in its features -- carries the increment,
        not the target-direction arm;
    (d) a degree-matched rewiring of every graph the features touch destroys at least half the increment.
Anything short of all four is reported as a NULL, and the null is the deliverable.

SAMPLING, AND WHAT IT EXCLUDES. Nothing is subsampled on the edge side: all 612,133 curated triples are
carried, all 360,946 that join K562 are used. What IS excluded is genes: K562 measures 8,248 genes and RPE1
8,749, so 59.0% of curated edges are testable in K562 and 7.9% in RPE1. An edge whose target is unmeasured is
absent from every number here -- not signed and not counted as failed. The rewiring control runs at one CV
seed rather than three, and the RPE1 arm inherits the K562 fold assignment rather than re-partitioning.
"""
import collections
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
GWPS = SP / "gwps.h5ad"
NET = SP / "cell_complete.json.gz"
PB_CACHE = SP / "rpe1_pseudobulk.npz"
RPE1 = SP / "rpe1.h5ad"
EXPR = SP / "depmap" / "omics_expression.npz"
COLLECTRI = SP / "collectri.tsv"
TRRUST = SP / "trrust.tsv"
LAYER = SP / "cell_reg_sign_imputed.json.gz"

Z_LAB = 2.0          # |robust z| at which a measured sign is treated as a LABEL rather than as noise
K_FOLD = 5           # folds, grouped by SOURCE GENE
SEEDS = (0, 1, 2)    # each seed re-randomises the source -> fold map; NOT a rotation of one partition
N_CTRL = 20          # swapped-source draws (and label-shuffle draws)
N_REWIRE = 20        # degree-matched configuration-model rewirings
MIN_SIGNED = 3       # a source needs this many signed curated edges before its activator fraction is used
CHUNK = 20_000
# --- the decision threshold, fixed before any result is computed ---
MIN_INCREMENT = 0.03     # points of agreement over the swapped-source control, K562, held-out sources
MIN_WIN_FRAC = 0.90      # >= 18 of 20 control draws must be beaten
MAX_REWIRE_SURVIVAL = 0.50   # a degree-matched rewiring must destroy at least half the increment
# --- the confidence rule for how many edges may be signed ---
SIGN_MIN_ACC = 0.65      # held-out accuracy required inside a confidence bin before its edges may be signed
SIGN_MIN_INC = 0.05      # ...and its increment over the swapped-source control inside the same bin


# ----------------------------------------------------------------------------------------------- io
def robust_z(M):
    """Per-COLUMN robust z, identical to colab/causal_reg.py so the label is the same object it uses.

    Median and MAD rather than mean and sd because the column is exactly where the few enormous effects
    live, and letting them set the scale would shrink the signal toward the noise.
    """
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def load_k562(report):
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["var/__categories/gene_name"][:]]
        gsym = np.array([cats[c] for c in f["var/gene_name"][:]])
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/gene_transcript"][:]]
        cov = {k: f["obs"][k][:].astype(np.float64) for k in
               ("num_cells_filtered", "UMI_count_unfiltered", "mitopercent", "fold_expr", "pct_expr")}
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var/gene_id"][:]])
        gmean = f["var"]["mean"][:].astype(np.float64)
        gcv = f["var"]["cv"][:].astype(np.float64)
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    nb = int((~np.isfinite(X)).sum())
    if nb:
        report(f"    {nb:,} non-finite entries set to NaN (they are not zeros)")
        X = np.where(np.isfinite(X), X, np.nan)
    return X, psym, gsym, gid, cov, gmean, gcv


def _cat(f, grp, key):
    import h5py
    o = f[grp][key]
    if isinstance(o, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        return np.array(cats), o["codes"][:]
    return None, o[:]


def pseudobulk_rpe1(report):
    """Identical to colab/causal_reg.py's construction, and cached, so both modules read the same RPE1."""
    import h5py
    if PB_CACHE.exists():
        z = np.load(PB_CACHE, allow_pickle=True)
        report(f"    RPE1 pseudobulk from cache: {z['lfc'].shape[0]:,} perturbations")
        return z["lfc"], np.array([str(x) for x in z["perts"]]), \
            np.array([str(x) for x in z["genes"]]), z["ncells"]
    if not RPE1.exists():
        raise SystemExit(f"absent: {RPE1} and no pseudobulk cache at {PB_CACHE}")
    with h5py.File(RPE1, "r") as f:
        cats, codes = _cat(f, "obs", "gene")
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["ensembl_id"][:]])
        n, G = f["X"].shape
        S = np.zeros((len(cats), G), dtype=np.float64)
        Cn = np.zeros(len(cats), dtype=np.int64)
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            np.add.at(S, codes[i0:i1], f["X"][i0:i1])
            np.add.at(Cn, codes[i0:i1], 1)
    lg = np.log2(1.0 + np.divide(S, np.maximum(S.sum(1, keepdims=True), 1), where=True) * 1e6)
    ci = int(np.where(cats == "non-targeting")[0][0])
    lfc = lg - lg[ci]
    keep = (Cn >= 25) & (np.arange(len(cats)) != ci)
    np.savez_compressed(PB_CACHE, lfc=lfc[keep].astype(np.float32), perts=cats[keep], genes=gid,
                        ncells=Cn[keep])
    return lfc[keep].astype(np.float32), cats[keep], gid, Cn[keep]


def declared_join(a, b, expect, label, report):
    """A join that RAISES when it silently loses most of its rows.

    This project has twice reported a legitimate-looking zero that was a failed lookup. A join that matches
    nothing must never return an empty result that a control then passes.
    """
    bi = {x: i for i, x in enumerate(b)}
    hit = [i for i, x in enumerate(a) if x in bi]
    rate = len(hit) / max(len(a), 1)
    if rate < expect:
        raise SystemExit(f"{label}: joined {len(hit):,}/{len(a):,} ({rate:.1%}), expected >= {expect:.0%}. "
                         f"The identifier spaces do not match and the numbers that would follow are not "
                         f"about the genes they claim to be about.")
    report(f"    {label}: {len(hit):,}/{len(a):,} ({rate:.1%}, expected >= {expect:.0%})")
    return rate


# ------------------------------------------------------------------------------------- graph handling
class Graphs:
    """Everything a feature reads out of a graph, plus a degree-preserving rewiring of all of it.

    THE REWIRING IS A CONFIGURATION MODEL, NOT A UNIFORM SHUFFLE. For the directed `reg` graph the source
    column is left completely untouched and the (target, sign) column is permuted: every source keeps its
    out-degree exactly and every target keeps its in-degree exactly, because a permutation preserves the
    multiset of targets. For the undirected `ppi` graph the stub list is permuted and re-paired, which is
    the textbook configuration model and preserves every node's degree (it can create self-loops and
    multi-edges; those are counted and reported rather than removed, since removing them would break the
    degree guarantee the control exists to provide).
    """

    def __init__(self, reg, ppi, coexpr, gene2cplx, complexes, sig, nichenet, name2idx, n_genes):
        self.n = n_genes
        self.reg = reg                      # (E, 3) int: source, target, sign
        self.ppi = ppi                      # (P, 2) int
        self.coex_a, self.coex_b, self.coex_r = coexpr
        self.cplx_g, self.cplx_c = gene2cplx
        self.n_cplx = complexes
        self.sig = sig                      # (S, 3) int
        self.nn_a, self.nn_b = nichenet
        self.name2idx = name2idx
        self.build()

    def rewired(self, seed):
        r = np.random.default_rng(seed)
        reg = self.reg.copy()
        p = r.permutation(len(reg))
        reg[:, 1:] = self.reg[p, 1:]                     # target+sign stubs re-paired; degrees preserved
        stubs = np.concatenate([self.ppi[:, 0], self.ppi[:, 1]])
        stubs = r.permutation(stubs)
        ppi = stubs.reshape(2, -1).T                      # configuration model on the undirected graph
        coex_b = self.coex_b[r.permutation(len(self.coex_b))]
        cplx_g = self.cplx_g[r.permutation(len(self.cplx_g))]
        sig = self.sig.copy()
        sig[:, 1:] = self.sig[r.permutation(len(sig)), 1:]
        nn_b = self.nn_b[r.permutation(len(self.nn_b))]
        g = Graphs.__new__(Graphs)
        g.n, g.name2idx, g.n_cplx = self.n, self.name2idx, self.n_cplx
        g.reg, g.ppi = reg, ppi
        g.coex_a, g.coex_b, g.coex_r = self.coex_a, coex_b, self.coex_r
        g.cplx_g, g.cplx_c = cplx_g, self.cplx_c
        g.sig = sig
        g.nn_a, g.nn_b = self.nn_a, nn_b
        g.build()
        return g

    def build(self):
        n = self.n
        self.outdeg = np.bincount(self.reg[:, 0], minlength=n)
        self.indeg = np.bincount(self.reg[:, 1], minlength=n)
        s = self.reg[self.reg[:, 2] != 0]
        self.n_signed = np.bincount(s[:, 0], minlength=n)
        self.n_pos = np.bincount(s[s[:, 2] == 1][:, 0], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.act_frac = np.where(self.n_signed >= MIN_SIGNED, self.n_pos / np.maximum(self.n_signed, 1),
                                     np.nan)
        self.ppideg = np.bincount(self.ppi.ravel(), minlength=n)
        self.ppiset = set(map(tuple, self.ppi)) | {(b, a) for a, b in self.ppi}
        self.regset = set(zip(self.reg[:, 0].tolist(), self.reg[:, 1].tolist()))
        self.regsign = {(a, b): c for a, b, c in self.reg.tolist()}
        self.coexset = {}
        for a, b, r in zip(self.coex_a, self.coex_b, self.coex_r):
            self.coexset[(int(a), int(b))] = float(r)
        self.cplxof = collections.defaultdict(set)
        for g, c in zip(self.cplx_g, self.cplx_c):
            self.cplxof[int(g)].add(int(c))
        self.ncplx = np.array([len(self.cplxof.get(i, ())) for i in range(n)], dtype=float)
        ss = self.sig[self.sig[:, 2] != 0]
        sig_n = np.bincount(ss[:, 0], minlength=n)
        sig_p = np.bincount(ss[ss[:, 2] == 1][:, 0], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.sig_act = np.where(sig_n >= 2, sig_p / np.maximum(sig_n, 1), np.nan)
        self.nnset = set(zip(self.nn_a.tolist(), self.nn_b.tolist()))
        # regulon sets, for the Jaccard feature
        self.regulon = collections.defaultdict(set)
        for a, b in zip(self.reg[:, 0], self.reg[:, 1]):
            self.regulon[int(a)].add(int(b))
        # GUILT BY ASSOCIATION: the mean curated activator fraction of a source's PPI partners. A TF that
        # binds corepressors should look like a repressor even when its own edges are unsigned. This is a
        # graph feature and it is exactly the kind the degree-matched rewiring exists to test.
        nb_sum = np.zeros(n)
        nb_cnt = np.zeros(n)
        af = np.where(np.isfinite(self.act_frac), self.act_frac, 0.0)
        fin = np.isfinite(self.act_frac).astype(float)
        np.add.at(nb_sum, self.ppi[:, 0], af[self.ppi[:, 1]])
        np.add.at(nb_cnt, self.ppi[:, 0], fin[self.ppi[:, 1]])
        np.add.at(nb_sum, self.ppi[:, 1], af[self.ppi[:, 0]])
        np.add.at(nb_cnt, self.ppi[:, 1], fin[self.ppi[:, 0]])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.ppi_partner_act = np.where(nb_cnt >= 2, nb_sum / np.maximum(nb_cnt, 1), np.nan)


def edge_features(G, si, ti, ann, coexpr_corr, gene_stat, prefix_only=None):
    """Feature matrix for an edge list. Columns are named so an arm can select by prefix.

    src_*     source-level, from curation and annotation only -- no perturbation measurement anywhere
    edge_*    pair-level: DepMap cross-line co-expression, PPI, shared complex, reciprocal edge, regulon
              overlap, NicheNet
    tgtann_*  target ANNOTATION and K562 expression level -- no direction information
    tgtdir_*  the K562-derived target DIRECTION bias. This is the confound; it is filled in by the caller
              leave-fold-out and lives in its own arm.
    """
    F, names = [], []

    def add(nm, v):
        F.append(np.asarray(v, dtype=np.float64))
        names.append(nm)

    add("src_act_frac", G.act_frac[si])
    add("src_n_signed", np.log1p(G.n_signed[si]))
    add("src_outdeg", np.log1p(G.outdeg[si]))
    add("src_indeg", np.log1p(G.indeg[si]))
    add("src_ppideg", np.log1p(G.ppideg[si]))
    add("src_ncplx", G.ncplx[si])
    add("src_sig_act", G.sig_act[si])
    add("src_ppi_partner_act", G.ppi_partner_act[si])
    for k in ("tf", "ess", "loeuf", "dep_frac", "npath", "cpg", "enh", "master"):
        add(f"src_{k}", ann[k][si])
    add("edge_coexpr", coexpr_corr)
    add("edge_abs_coexpr", np.abs(coexpr_corr))
    add("edge_ppi", [1.0 if (a, b) in G.ppiset else 0.0 for a, b in zip(si, ti)])
    add("edge_cplx", [1.0 if (G.cplxof.get(a, set()) & G.cplxof.get(b, set())) else 0.0
                      for a, b in zip(si, ti)])
    add("edge_reverse", [1.0 if (b, a) in G.regset else 0.0 for a, b in zip(si, ti)])
    add("edge_reverse_sign", [float(G.regsign.get((b, a), 0)) for a, b in zip(si, ti)])
    add("edge_coex_topk", [G.coexset.get((a, b), G.coexset.get((b, a), 0.0)) for a, b in zip(si, ti)])
    add("edge_nichenet", [1.0 if (a, b) in G.nnset else 0.0 for a, b in zip(si, ti)])
    # Jaccard of the two regulons. The union is computed arithmetically rather than by building the union
    # SET: a hub source has 5,000 targets and materialising 340,000 unions of that size does not finish.
    jac = []
    empty = set()
    for a, b in zip(si, ti):
        A, B = G.regulon.get(a, empty), G.regulon.get(b, empty)
        if not A or not B:
            jac.append(0.0)
            continue
        inter = len(A & B)
        jac.append(inter / max(len(A) + len(B) - inter, 1))
    add("edge_regulon_jaccard", jac)
    add("tgtann_outdeg", np.log1p(G.outdeg[ti]))
    add("tgtann_indeg", np.log1p(G.indeg[ti]))
    add("tgtann_ppideg", np.log1p(G.ppideg[ti]))
    for k in ("tf", "ess", "loeuf", "dep_frac", "npath", "cpg", "enh"):
        add(f"tgtann_{k}", ann[k][ti])
    add("tgtann_k562_mean", gene_stat[0])
    add("tgtann_k562_cv", gene_stat[1])
    M = np.vstack(F).T
    return M, names


# -------------------------------------------------------------------------------------- the harness
class Swap:
    """The swapped-source control, and the residual-imbalance audit that says whether it matched.

    Keep the target and the PREDICTED sign; read the measured sign at a different perturbation of matched
    perturbation strength. Deciles, not quartiles.

    AND THE DECILE MATCH IS NOT TIGHT ENOUGH HERE, WHICH THE AUDIT CAUGHT. Curated sources are TFs, and TFs
    sit at the TOP of whatever strength decile they land in: the first run of this module matched on deciles
    and still left a standardised mean difference of +0.25 in perturbation strength between the real and the
    swapped sources. A control that is systematically weaker than the arm it is controlling would flatter
    the edge arm. So both are computed and both are reported:

        DECILE      the construction colab/causal_reg.py uses, kept so the numbers are comparable to it
        NEAREST     the replacement is drawn from the +/- 150 perturbations nearest in strength RANK,
                    which drives the residual imbalance on strength to ~0

    Reporting the loose one alone would have been the mistake; reporting only the tight one would have hidden
    that the incumbent construction has this defect.
    """
    NN_WINDOW = 150

    def __init__(self, Z, cov, report, label):
        self.Z = Z
        self.strength = np.nanmean(np.abs(Z), axis=1)
        q = np.nanquantile(self.strength, np.linspace(.1, .9, 9))
        self.dec = np.digitize(self.strength, q)
        self.order = np.argsort(np.where(np.isfinite(self.strength), self.strength, np.inf))
        self.rank = np.empty(len(self.strength), dtype=int)
        self.rank[self.order] = np.arange(len(self.strength))
        self.cov = dict(cov)
        self.cov["strength"] = self.strength
        self.label = label
        report(f"    {label}: {len(self.strength):,} perturbations; strength deciles sized "
               f"{np.bincount(self.dec, minlength=10).tolist()}")

    def draw(self, rows, seed, mode="decile"):
        r = np.random.default_rng(seed)
        if mode == "nn":
            n = len(self.strength)
            off = r.integers(-self.NN_WINDOW, self.NN_WINDOW + 1, size=len(rows))
            off = np.where(off == 0, 1, off)          # never return the source itself
            return self.order[np.clip(self.rank[rows] + off, 0, n - 1)]
        alt = np.empty(len(rows), dtype=int)
        for q in np.unique(self.dec):
            m = self.dec[rows] == q
            if m.any():
                alt[m] = r.choice(np.where(self.dec == q)[0], int(m.sum()))
        return alt

    def agree(self, rows, cols, pred, seeds, cond=None, mode="decile"):
        """Mean agreement of `pred` with the measured sign at a swapped source, one value per seed.

        `cond` conditions the control identically to the edge arm: selecting the edge arm on |z| >= t
        conditions on the outcome and inflates agreement, so the control must be selected the same way.
        """
        out = []
        for s in seeds:
            z2 = self.Z[self.draw(rows, s, mode), cols]
            f = np.isfinite(z2)
            if cond is not None:
                f &= np.abs(z2) >= cond
            out.append(float((np.sign(z2) == pred)[f].mean()) if f.sum() > 20 else float("nan"))
        return np.array(out)

    def imbalance(self, rows, seeds, mode="decile"):
        """Standardised mean difference per covariate between the real and the swapped sources.

        Rule 3 says match on deciles AND check for residual imbalance per covariate. This is that check,
        reported whatever it says -- it is what forced the nearest-neighbour arm above into existence.
        """
        rep = {}
        for k, v in self.cov.items():
            a = v[rows]
            b = np.concatenate([v[self.draw(rows, s, mode)] for s in seeds[:5]])
            sd = np.nanstd(np.concatenate([a, b]))
            rep[k] = {"real": float(np.nanmean(a)), "swapped": float(np.nanmean(b)),
                      "smd": float((np.nanmean(a) - np.nanmean(b)) / max(sd, 1e-9))}
        return rep


def fit_predict(Xtr, ytr, Xte, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       random_state=seed, early_stopping=False)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def auc(y, p):
    """Rank AUC. y in {0,1}."""
    y = np.asarray(y)
    o = np.argsort(p)
    r = np.empty(len(p), dtype=float)
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------- main
def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 104)
    report("IMPUTE THE SIGN OF THE 558,005 UNSIGNED CURATED REGULATORY EDGES")
    report("=" * 104)
    report(f"  DECISION THRESHOLD, FIXED BEFORE ANY NUMBER: usable only if the K562 held-out-source "
           f"increment\n  over the swapped-source control is >= {MIN_INCREMENT:+.3f} in >= "
           f"{MIN_WIN_FRAC:.0%} of {N_CTRL} draws, RPE1 increment > 0 on the same terms,\n  the increment "
           f"is carried by the arm with NO K562 measurement in its features, and a degree-matched\n  "
           f"rewiring destroys >= {MAX_REWIRE_SURVIVAL:.0%} of it.")
    for p in (GWPS, NET, EXPR):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    # ------------------------------------------------------------------ 0. load
    report(f"\n  0. SOURCES")
    X, psym, gsym, kens, cov, gmean, gcv = load_k562(report)
    report(f"    K562 pseudobulk: {X.shape[0]:,} perturbations x {X.shape[1]:,} measured genes")
    Zk = robust_z(X)
    C = json.load(gzip.open(NET))
    genes = C["genes"]
    gname = np.array([g["name"] for g in genes])
    n_genes = len(genes)
    reg = np.array(C["reg"], dtype=np.int64)
    sg = reg[:, 2]
    report(f"    cell object: {n_genes:,} genes, reg {len(reg):,} triples "
           f"({int((sg == 0).sum()):,} unsigned = {100*(sg == 0).mean():.1f}%, "
           f"{int((sg == 1).sum()):,} activating, {int((sg == -1).sum()):,} repressing)")

    ann = {}
    for k in ("tf", "ess", "npath", "cpg", "enh"):
        ann[k] = np.array([float(g.get(k) or 0) for g in genes])
    ann["loeuf"] = np.array([float(g["loeuf"]) if g.get("loeuf") is not None else np.nan for g in genes])
    ann["dep_frac"] = np.array([float(g.get("dep_frac") or 0) for g in genes])
    ann["master"] = np.array([1.0 if g.get("master") else 0.0 for g in genes])

    ppi = np.array(C["ppi"], dtype=np.int64)
    ca, cb, cr = [], [], []
    for k, v in C["coexpr"].items():
        for j, r in v:
            ca.append(int(k)); cb.append(int(j)); cr.append(float(r))
    coexpr_layer = (np.array(ca), np.array(cb), np.array(cr))
    cplx_names = {c: i for i, c in enumerate(C["complexes"])}
    cg, cc = [], []
    for k, v in C["gene2cplx"].items():
        for c in v:
            if c in cplx_names:
                cg.append(int(k)); cc.append(cplx_names[c])
    cplx = (np.array(cg), np.array(cc))
    sig = np.array(C["sig"], dtype=np.int64) if C.get("sig") else np.zeros((0, 3), dtype=np.int64)
    name2idx = {n: i for i, n in enumerate(gname)}
    na, nb = [], []
    for s, ts in C.get("nichenet", {}).items():
        if s in name2idx:
            for t in ts:
                if t in name2idx:
                    na.append(name2idx[s]); nb.append(name2idx[t])
    nichenet = (np.array(na, dtype=np.int64), np.array(nb, dtype=np.int64))
    G = Graphs(reg, ppi, coexpr_layer, cplx, len(cplx_names), sig, nichenet, name2idx, n_genes)
    report(f"    graphs: ppi {len(ppi):,} pairs, coexpr {len(ca):,} pairs, complex memberships "
           f"{len(cg):,}, sig {len(sig):,}, nichenet {len(na):,}")

    # ------------------------------------------------------------------ joins, with declared rates
    report(f"\n  JOINS (each declares an expected rate and raises below it)")
    pi, gi = {}, {}
    for i, s in enumerate(psym):
        pi.setdefault(s, i)
    for i, s in enumerate(gsym):
        gi.setdefault(s, i)
    src, tgt = gname[reg[:, 0]], gname[reg[:, 1]]
    jr_p = declared_join(sorted({s for s in src}), list(pi), 0.60, "reg sources -> K562 perturbations",
                         report)
    jr_g = declared_join(sorted({t for t in tgt}), list(gi), 0.30, "reg targets -> K562 measured genes",
                         report)
    ri = np.array([pi.get(a, -1) for a in src])
    ci = np.array([gi.get(b, -1) for b in tgt])
    testable = (ri >= 0) & (ci >= 0)
    report(f"    curated edges testable in K562: {int(testable.sum()):,}/{len(reg):,} "
           f"({100*testable.mean():.1f}%)")
    if testable.mean() < 0.40:
        raise SystemExit("edge-level join below 40%; refusing to report numbers on a broken join")

    # DepMap expression, for the one edge-level feature with near-complete coverage
    d = np.load(EXPR, allow_pickle=True)
    E = d["E"].astype(np.float32)
    egen = np.array([g.split(" (")[0] for g in d["genes"]])
    En = (E - E.mean(0)) / np.where(E.std(0) < 1e-9, np.nan, E.std(0))
    En = np.nan_to_num(En, nan=0.0)
    ei_ = {}
    for i, g in enumerate(egen):
        ei_.setdefault(g, i)
    declared_join(sorted(set(gname[reg[:, 0]]) | set(gname[reg[:, 1]])), list(ei_), 0.70,
                  "reg genes -> DepMap expression", report)
    eA = np.array([ei_.get(a, -1) for a in src])
    eB = np.array([ei_.get(b, -1) for b in tgt])

    def coexpr_corr(a, b):
        """Cross-line expression correlation, chunked -- 1,066 lines x 340k pairs will not fit at once."""
        out = np.full(len(a), np.nan)
        ok = (a >= 0) & (b >= 0)
        idx = np.where(ok)[0]
        N = En.shape[0]
        for i0 in range(0, len(idx), CHUNK):
            j = idx[i0:i0 + CHUNK]
            out[j] = (En[:, a[j]] * En[:, b[j]]).sum(0) / N
        return out

    # ------------------------------------------------------------------ 1. convention + lookup route
    report(f"\n  1. IS THERE A CONVENTION TO IMPUTE? (curated sign vs the measured perturbational sign)")
    report(f"     Under CRISPRi, knocking down an ACTIVATOR should LOWER its target, so the prediction is\n"
           f"     measured sign = -(curated sign). Read against the swapped-source control, not against 0.5.")
    SW = Swap(Zk, cov, report, "K562")
    sgn_mask = testable & (sg != 0)
    r_s, c_s = ri[sgn_mask], ci[sgn_mask]
    z_s = Zk[r_s, c_s]
    fin = np.isfinite(z_s)
    r_s, c_s, z_s, cur_s = r_s[fin], c_s[fin], z_s[fin], sg[sgn_mask][fin]
    conv = []
    report(f"     {'subset':>14s} {'n':>8s} {'agree':>8s} {'swap':>8s} {'incr':>9s} {'won':>5s} "
           f"{'swapNN':>8s} {'incrNN':>9s} {'wonNN':>6s}")
    for thr in (0.0, 2.0, 3.0):
        k = np.abs(z_s) >= thr
        pr = -cur_s[k]
        ag = float((np.sign(z_s[k]) == pr).mean())
        row = {"z_min": thr, "n": int(k.sum()), "agree": ag}
        for mode, sfx in (("decile", ""), ("nn", "_nn")):
            sw = SW.agree(r_s[k], c_s[k], pr, range(700, 700 + N_CTRL),
                          cond=(thr if thr > 0 else None), mode=mode)
            row[f"swap{sfx}"] = float(np.nanmean(sw))
            row[f"swap_sd{sfx}"] = float(np.nanstd(sw))
            row[f"increment{sfx}"] = ag - float(np.nanmean(sw))
            row[f"frac_draws_won{sfx}"] = float(np.mean(ag > sw))
        conv.append(row)
        report(f"     {'|z| >= ' + f'{thr:g}':>14s} {int(k.sum()):8,d} {ag:8.4f} {row['swap']:8.4f} "
               f"{row['increment']:+9.4f} {row['frac_draws_won']:5.2f} {row['swap_nn']:8.4f} "
               f"{row['increment_nn']:+9.4f} {row['frac_draws_won_nn']:6.2f}")

    report(f"\n     ...AND WHY THE OBVIOUS ROUTE IS CLOSED. If the unsigned edges were the same objects as "
           f"the signed\n     ones, the sign could simply be looked up in an external signed resource.")
    ext = {}
    if COLLECTRI.exists():
        with open(COLLECTRI) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                st, ih = row["consensus_stimulation"] == "True", row["consensus_inhibition"] == "True"
                if st != ih:
                    ext[(row["source_genesymbol"], row["target_genesymbol"])] = 1 if st else -1
    tr = {}
    if TRRUST.exists():
        with open(TRRUST) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and p[2] in ("Activation", "Repression"):
                    tr[(p[0], p[1])] = 1 if p[2] == "Activation" else -1
    key = list(zip(src.tolist(), tgt.tolist()))
    hit = np.array([(k in ext) or (k in tr) for k in key])
    lookup = {"collectri_pairs": len(ext), "trrust_pairs": len(tr),
              "signed_edges_covered": int((hit & (sg != 0)).sum()),
              "unsigned_edges_covered": int((hit & (sg == 0)).sum())}
    report(f"     CollecTRI {len(ext):,} signed pairs + TRRUST {len(tr):,}: they cover "
           f"{lookup['signed_edges_covered']:,} of the {int((sg != 0).sum()):,} SIGNED curated edges and "
           f"{lookup['unsigned_edges_covered']:,} of the {int((sg == 0).sum()):,} UNSIGNED ones.")
    report(f"     The signed subset IS those resources. The unsigned subset has a different provenance -- "
           f"its top\n     sources carry thousands of targets each, the shape of a ChIP/motif compendium -- "
           f"so imputation\n     has to generalise across that boundary rather than join across it.")

    # ------------------------------------------------------------------ 2. the ceiling
    report(f"\n  2. THE CEILING -- how much is there for ANY source-level feature to find?")
    report(f"     Split each source's unsigned targets in half; estimate the source's measured direction "
           f"bias on\n     one half, predict the other. No annotation can beat this, because this IS the "
           f"quantity an annotation\n     would have to recover. Its control is the same swapped-source arm.")
    uns = testable & (sg == 0)
    r_u, c_u, s_u = ri[uns], ci[uns], reg[uns, 0]
    z_u = Zk[r_u, c_u]
    fin = np.isfinite(z_u)
    r_u, c_u, s_u, z_u = r_u[fin], c_u[fin], s_u[fin], z_u[fin]
    idx_u = np.where(uns)[0][fin]
    report(f"     unsigned curated edges with a finite K562 measurement: {len(z_u):,} "
           f"over {len(set(s_u.tolist())):,} source genes")
    rng = np.random.default_rng(7)
    half = rng.random(len(z_u)) < 0.5
    ceil = []
    for thr in (0.0, 2.0):
        strong = np.abs(z_u) >= max(thr, 1e-9)
        bias = collections.defaultdict(list)
        for s, z in zip(s_u[half & (np.abs(z_u) >= Z_LAB)], z_u[half & (np.abs(z_u) >= Z_LAB)]):
            bias[int(s)].append(np.sign(z))
        bs = {k: float(np.mean(v)) for k, v in bias.items() if len(v) >= 10}
        has = np.array([int(s) in bs for s in s_u])
        sub = (~half) & strong & has
        pr = np.array([1 if bs[int(s)] > 0 else -1 for s in s_u[sub]])
        ag = float((np.sign(z_u[sub]) == pr).mean())
        row = {"z_min": thr, "n": int(sub.sum()), "n_sources": len(bs), "agree": ag}
        for mode, sfx in (("decile", ""), ("nn", "_nn")):
            sw = SW.agree(r_u[sub], c_u[sub], pr, range(800, 800 + N_CTRL),
                          cond=(thr if thr > 0 else None), mode=mode)
            row[f"swap{sfx}"] = float(np.nanmean(sw))
            row[f"increment{sfx}"] = ag - float(np.nanmean(sw))
            row[f"frac_draws_won{sfx}"] = float(np.mean(ag > sw))
        ceil.append(row)
        report(f"     |z| >= {thr:g}: n {int(sub.sum()):7,d} over {len(bs):4,d} sources  agree {ag:.4f}  "
               f"swap {row['swap']:.4f}  CEILING INCREMENT {row['increment']:+.4f}  "
               f"(nearest-neighbour control {row['swap_nn']:.4f}, {row['increment_nn']:+.4f}, "
               f"{row['frac_draws_won_nn']:.0%} of draws won)")

    # does any annotation see that bias? the crux, measured at source level
    strong_u = np.abs(z_u) >= Z_LAB
    per = collections.defaultdict(list)
    for s, z in zip(s_u[strong_u], z_u[strong_u]):
        per[int(s)].append(np.sign(z))
    rows = [(k, float(np.mean(v)), len(v)) for k, v in per.items() if len(v) >= 30]
    crux = {}
    if len(rows) >= 30:
        sidx = np.array([r[0] for r in rows])
        sbias = np.array([r[1] for r in rows])
        cand = {"curated_act_frac": G.act_frac[sidx], "log_outdeg": np.log1p(G.outdeg[sidx]),
                "ppi_partner_act_frac": G.ppi_partner_act[sidx], "tf_flag": ann["tf"][sidx],
                "loeuf": ann["loeuf"][sidx], "dep_frac": ann["dep_frac"][sidx],
                "log_ppideg": np.log1p(G.ppideg[sidx]), "n_complexes": G.ncplx[sidx]}
        report(f"\n     DOES ANY ANNOTATION SEE THAT BIAS? Spearman against the measured source direction "
               f"bias,\n     over the {len(rows):,} sources with >= 30 strongly-responding curated targets:")
        for nm, v in cand.items():
            f = np.isfinite(v)
            if f.sum() < 20:
                continue
            rho, p = stats.spearmanr(v[f], sbias[f])
            crux[nm] = {"spearman": float(rho), "p": float(p), "n": int(f.sum())}
            report(f"       {nm:24s} rho {rho:+.3f}  p {p:.2g}  n {int(f.sum()):,}")

    # ------------------------------------------------------------------ 3. the model
    report(f"\n  3. THE MODEL -- held out BY SOURCE GENE, so no TF is in both train and test")
    co_u = coexpr_corr(eA[idx_u], eB[idx_u])
    report(f"     DepMap cross-line co-expression computed for {int(np.isfinite(co_u).sum()):,}/{len(co_u):,}"
           f" edges (1,066 lines)")
    gs = (gmean[c_u], gcv[c_u])
    M, fnames = edge_features(G, reg[idx_u, 0], reg[idx_u, 1], ann, co_u, gs)
    report(f"     {M.shape[1]} features: "
           f"{sum(n.startswith('src_') for n in fnames)} src, "
           f"{sum(n.startswith('edge_') for n in fnames)} edge, "
           f"{sum(n.startswith('tgtann_') for n in fnames)} tgtann, +2 tgtdir filled leave-fold-out")
    lab = np.abs(z_u) >= Z_LAB
    y = (np.sign(z_u) > 0).astype(int)
    report(f"     labels: |z| >= {Z_LAB:g} on {int(lab.sum()):,}/{len(z_u):,} edges "
           f"({100*lab.mean():.1f}%); of those {100*y[lab].mean():.1f}% are POSITIVE "
           f"(knockdown raises the target)")
    report(f"     the majority-class baseline is therefore {max(y[lab].mean(), 1-y[lab].mean()):.4f} -- "
           f"any accuracy is read against\n     the swapped-source control, not against this and not "
           f"against 0.5")

    sources = np.unique(s_u)
    sidx_of = {int(s): i for i, s in enumerate(sources)}
    src_pos = np.array([sidx_of[int(s)] for s in s_u])

    def fold_map(seed):
        r = np.random.default_rng(1000 + seed)
        return r.integers(0, K_FOLD, size=len(sources))

    fmaps = [fold_map(s) for s in SEEDS]
    coassign = float(np.mean([(fmaps[i] == fmaps[j]).mean()
                              for i in range(len(fmaps)) for j in range(i + 1, len(fmaps))]))
    report(f"     {len(sources):,} source genes -> {K_FOLD} folds, re-randomised per seed. Pairwise "
           f"co-assignment across\n     seeds {coassign:.3f} (1/K = {1/K_FOLD:.3f} if the partitions "
           f"genuinely differ; 1.000 would mean fold labels were rotated)")
    if coassign > 0.5:
        raise SystemExit("CV seeds are not drawing different partitions")

    ARMS = {"A target-direction only (K562)": ("tgtdir_",),
            "B source+edge (no K562 in feats)": ("src_", "edge_", "tgtann_"),
            "C full": ("src_", "edge_", "tgtann_", "tgtdir_")}

    def run_cv(Mx, fnamesx, fmap, seed, shuffle=False, arms=None, predict_scope=None):
        """One CV pass. Returns an out-of-fold probability per arm.

        The target-direction features cannot be precomputed: they must be estimated from the TRAINING fold's
        perturbations only, or the held-out TF's own row leaks into the feature that is supposed to be a
        property of the target. They are therefore rebuilt inside the fold loop and only when an arm asks
        for them.
        """
        arms = arms or ARMS
        need_td = any("tgtdir_" in pref for pref in arms.values())
        oof = {a: np.full(len(z_u), np.nan) for a in arms}
        yy = y.copy()
        if shuffle:
            rr = np.random.default_rng(9000 + seed)
            keep = np.where(lab)[0]
            yy[keep] = yy[rr.permutation(keep)]
        for k in range(K_FOLD):
            te = np.isin(src_pos, np.where(fmap == k)[0])
            tr = (~te) & lab
            # predict on EVERY held-out edge, not only the labelled ones: the RPE1 arm scores a different
            # subset (|RPE1 z| >= 2) and conditioning its predictions on the K562 label would make the
            # transfer test a test of edges already known to respond in K562.
            pe = te if predict_scope is None else (te & predict_scope)
            if tr.sum() < 500 or (te & lab).sum() < 100:
                continue
            fn = list(fnamesx)
            Mfull = Mx
            if need_td:
                rows_tr = np.unique(r_u[tr])
                with np.errstate(invalid="ignore"):
                    tdir = np.nanmean(np.sign(Zk[rows_tr]), axis=0)
                    tresp = np.nanmean(np.abs(Zk[rows_tr]), axis=0)
                Mfull = np.hstack([Mx, np.vstack([tdir[c_u], tresp[c_u]]).T])
                fn = fn + ["tgtdir_bias", "tgtdir_resp"]
            itr, ite = np.where(tr)[0], np.where(pe)[0]
            for aname, pref in arms.items():
                cols = [i for i, n in enumerate(fn) if n.startswith(pref)]
                p, _m = fit_predict(Mfull[np.ix_(itr, cols)], yy[tr], Mfull[np.ix_(ite, cols)], seed)
                oof[aname][ite] = p
        return oof

    def score(oof_p, mask, sw_obj, rows, cols, zvec, tag, seeds0):
        """RAW held-out agreement -- never a difference from a shuffle -- plus both swapped-source arms."""
        m = mask & np.isfinite(oof_p)
        pr = np.where(oof_p[m] > 0.5, 1, -1)
        ag = float((np.sign(zvec[m]) == pr).mean())
        a = auc((np.sign(zvec[m]) > 0).astype(int), oof_p[m])
        out = {"tag": tag, "n": int(m.sum()), "accuracy": ag, "auc": a,
               "majority_baseline": float(max((np.sign(zvec[m]) > 0).mean(),
                                              (np.sign(zvec[m]) < 0).mean()))}
        for mode, sfx in (("decile", ""), ("nn", "_nn")):
            sw = sw_obj.agree(rows[m], cols[m], pr, seeds0, cond=Z_LAB, mode=mode)
            out[f"swap{sfx}"] = float(np.nanmean(sw))
            out[f"swap_sd{sfx}"] = float(np.nanstd(sw))
            out[f"increment{sfx}"] = ag - float(np.nanmean(sw))
            out[f"frac_draws_won{sfx}"] = float(np.mean(ag > sw))
        return out

    res_k562 = collections.defaultdict(list)
    oof_by_seed = {}
    for seed, fmap in zip(SEEDS, fmaps):
        oof = run_cv(M, fnames, fmap, seed)
        oof_by_seed[seed] = oof
        for aname, p in oof.items():
            res_k562[aname].append(score(p, lab, SW, r_u, c_u, z_u, aname, range(2000 + 50 * seed,
                                                                                 2000 + 50 * seed + N_CTRL)))
    report(f"\n     K562, held-out SOURCE GENES, labels |z| >= {Z_LAB:g} "
           f"(mean over {len(SEEDS)} genuinely different partitions)")
    report(f"     {'arm':34s} {'n':>8s} {'acc':>7s} {'AUC':>7s} {'major':>7s} {'swap':>7s} {'incr':>8s} "
           f"{'won':>5s} {'swapNN':>7s} {'incrNN':>8s} {'wonNN':>6s}")
    k562_summary = {}
    for aname, rs in res_k562.items():
        s = {k: float(np.mean([r[k] for r in rs])) for k in
             ("accuracy", "auc", "majority_baseline", "swap", "swap_sd", "increment", "frac_draws_won",
              "swap_nn", "swap_sd_nn", "increment_nn", "frac_draws_won_nn")}
        s["accuracy_sd_over_seeds"] = float(np.std([r["accuracy"] for r in rs]))
        s["n"] = int(np.mean([r["n"] for r in rs]))
        k562_summary[aname] = s
        report(f"     {aname:34s} {s['n']:8,d} {s['accuracy']:7.4f} {s['auc']:7.4f} "
               f"{s['majority_baseline']:7.4f} {s['swap']:7.4f} {s['increment']:+8.4f} "
               f"{s['frac_draws_won']:5.2f} {s['swap_nn']:7.4f} {s['increment_nn']:+8.4f} "
               f"{s['frac_draws_won_nn']:6.2f}")
    report(f"     seed-to-seed sd of accuracy (different partitions, not rotated fold labels): " +
           ", ".join(f"{a.split()[0]} {k562_summary[a]['accuracy_sd_over_seeds']:.4f}"
                     for a in k562_summary))

    # residual imbalance of both swapped-source matches
    imb = {m: SW.imbalance(r_u[lab], list(range(2000, 2000 + N_CTRL)), mode=m) for m in ("decile", "nn")}
    report(f"\n     RESIDUAL IMBALANCE per covariate (standardised mean difference, real vs swapped source)")
    report(f"       {'covariate':24s} {'real':>12s} {'swap decile':>13s} {'SMD':>8s} {'swap NN':>13s} "
           f"{'SMD':>8s}")
    for k in imb["decile"]:
        d, n_ = imb["decile"][k], imb["nn"][k]
        report(f"       {k:24s} {d['real']:12.4f} {d['swapped']:13.4f} {d['smd']:+8.4f} "
               f"{n_['swapped']:13.4f} {n_['smd']:+8.4f}")
    worst = max(abs(v["smd"]) for v in imb["decile"].values())
    worst_nn = max(abs(v["smd"]) for v in imb["nn"].values())
    report(f"       worst |SMD|: decile {worst:.4f} "
           f"({'ok' if worst < 0.10 else 'RESIDUAL IMBALANCE PRESENT'}), "
           f"nearest-neighbour {worst_nn:.4f} "
           f"({'ok' if worst_nn < 0.10 else 'RESIDUAL IMBALANCE PRESENT'})")
    report(f"       the decile match leaves the control WEAKER than the arm it controls, which would "
           f"flatter the edge\n       arm; the nearest-neighbour arm above is the one to read when the two "
           f"disagree")

    # ------------------------------------------------------------------ 4. label shuffle
    report(f"\n  4. LABEL-SHUFFLED ARM (significance only -- the effect size above is the RAW held-out "
           f"value)")
    shuf = []
    for seed, fmap in zip(SEEDS, fmaps):
        o = run_cv(M, fnames, fmap, seed, shuffle=True, arms={"C full": ARMS["C full"]})
        shuf.append(score(o["C full"], lab, SW, r_u, c_u, z_u, "shuffled",
                          range(3000 + 50 * seed, 3000 + 50 * seed + N_CTRL)))
    sh = {k: float(np.mean([r[k] for r in shuf])) for k in ("accuracy", "auc", "increment")}
    report(f"     labels permuted, full pipeline rerun: acc {sh['accuracy']:.4f}  AUC {sh['auc']:.4f}  "
           f"incr {sh['increment']:+.4f}")
    real_c = k562_summary["C full"]
    report(f"     real full arm:                       acc {real_c['accuracy']:.4f}  "
           f"AUC {real_c['auc']:.4f}  incr {real_c['increment']:+.4f}")

    # ------------------------------------------------------------------ 5. degree-matched rewiring
    report(f"\n  5. DEGREE-MATCHED REWIRING ({N_REWIRE} configuration-model draws, one CV seed)")
    report(f"     Source out-degree and target in-degree are preserved EXACTLY; only which source is wired "
           f"to which\n     target is destroyed. If this reproduces the increment, the increment is degree, "
           f"and that is the finding.")
    rew = []
    for j in range(N_REWIRE):
        Gr = G.rewired(500 + j)
        Mr, fnr = edge_features(Gr, reg[idx_u, 0], reg[idx_u, 1], ann, co_u, gs)
        o = run_cv(Mr, fnr, fmaps[0], SEEDS[0], predict_scope=lab,
                   arms={"B source+edge (no K562 in feats)": ARMS["B source+edge (no K562 in feats)"]})
        rew.append(score(o["B source+edge (no K562 in feats)"], lab, SW, r_u, c_u, z_u, f"rewire{j}",
                         range(4000 + 30 * j, 4000 + 30 * j + 5)))
        if j % 5 == 0:
            report(f"       rewiring {j+1}/{N_REWIRE}: acc {rew[-1]['accuracy']:.4f} "
                   f"incr {rew[-1]['increment']:+.4f}")
    rw_acc = float(np.mean([r["accuracy"] for r in rew]))
    rw_inc = float(np.mean([r["increment"] for r in rew]))
    rw_sd = float(np.std([r["increment"] for r in rew]))
    armB = k562_summary["B source+edge (no K562 in feats)"]
    beat = float(np.mean([armB["increment"] > r["increment"] for r in rew]))
    beat_acc = float(np.mean([armB["accuracy"] > r["accuracy"] for r in rew]))
    # A RATIO IS ONLY READABLE WHEN THERE IS AN EFFECT TO DIVIDE. If the real increment is <= 0 there is
    # nothing for the rewiring to destroy and "survival 2.21" would be an artefact of a negative
    # denominator, not a result.
    surv = rw_inc / armB["increment"] if armB["increment"] > 1e-9 else float("nan")
    report(f"     real arm B      : acc {armB['accuracy']:.4f}  incr {armB['increment']:+.4f}")
    report(f"     degree-matched  : acc {rw_acc:.4f}  incr {rw_inc:+.4f} (sd {rw_sd:.4f}) over "
           f"{N_REWIRE} rewirings")
    report(f"     the real wiring beats the degree-matched control on increment in {beat:.0%} of "
           f"rewirings, on accuracy in {beat_acc:.0%}")
    if np.isfinite(surv):
        report(f"     fraction of the real increment the degree-matched control already reproduces: "
               f"{surv:.2f}")
    else:
        report(f"     survival ratio NOT COMPUTED: the real increment is {armB['increment']:+.4f}, so "
               f"there is no effect for a\n     rewiring to destroy. Degree is not the explanation for a "
               f"result that does not exist.")

    # ------------------------------------------------------------------ 6. RPE1 transfer
    report(f"\n  6. TRANSFER TO RPE1 -- a cell line no layer in this project was built from")
    Lr, rperts, rens, rn = pseudobulk_rpe1(report)
    Zr = robust_z(Lr)
    e2s = dict(zip(kens, gsym))
    rsym = np.array([e2s.get(g, "") for g in rens])
    declared_join([g for g in rens], list(e2s), 0.70, "RPE1 ensembl -> symbol (via K562 var)", report)
    rpi, rgi = {}, {}
    for i, s in enumerate(rperts):
        rpi.setdefault(str(s), i)
    for i, s in enumerate(rsym):
        if s:
            rgi.setdefault(s, i)
    rr = np.array([rpi.get(a, -1) for a in gname[reg[idx_u, 0]]])
    rc = np.array([rgi.get(b, -1) for b in gname[reg[idx_u, 1]]])
    zr = np.full(len(idx_u), np.nan)
    okr = (rr >= 0) & (rc >= 0)
    zr[okr] = Zr[rr[okr], rc[okr]]
    rate_r = okr.mean()
    report(f"    curated unsigned edges testable in RPE1: {int(okr.sum()):,}/{len(idx_u):,} "
           f"({100*rate_r:.1f}%)")
    if okr.sum() < 2000:
        raise SystemExit("too few RPE1-testable edges to score transfer")
    SWr = Swap(Zr, {"rpe1_ncells": rn.astype(float)}, report, "RPE1")
    lab_r = okr & np.isfinite(zr) & (np.abs(zr) >= Z_LAB)
    report(f"    of those, {int(lab_r.sum()):,} have |RPE1 z| >= {Z_LAB:g}")
    rpe1_summary = {}
    report(f"     {'arm':34s} {'n':>8s} {'acc':>7s} {'AUC':>7s} {'swap':>7s} {'sd':>6s} {'incr':>8s} "
           f"{'won':>5s} {'incrNN':>8s} {'wonNN':>6s}")
    for aname in ARMS:
        rs = []
        for seed in SEEDS:
            p = oof_by_seed[seed][aname]
            m = lab_r & np.isfinite(p)
            pr = np.where(p[m] > 0.5, 1, -1)
            ag = float((np.sign(zr[m]) == pr).mean())
            row = {"n": int(m.sum()), "accuracy": ag,
                   "auc": auc((np.sign(zr[m]) > 0).astype(int), p[m]),
                   "majority_baseline": float(max((np.sign(zr[m]) > 0).mean(),
                                                  (np.sign(zr[m]) < 0).mean()))}
            for mode, sfx in (("decile", ""), ("nn", "_nn")):
                sw = SWr.agree(rr[m], rc[m], pr, range(6000 + 40 * seed, 6000 + 40 * seed + N_CTRL),
                               cond=Z_LAB, mode=mode)
                row[f"swap{sfx}"] = float(np.nanmean(sw))
                row[f"swap_sd{sfx}"] = float(np.nanstd(sw))
                row[f"increment{sfx}"] = ag - float(np.nanmean(sw))
                row[f"frac_draws_won{sfx}"] = float(np.mean(ag > sw))
            rs.append(row)
        s = {k: float(np.mean([r[k] for r in rs])) for k in
             ("accuracy", "auc", "majority_baseline", "swap", "swap_sd", "increment", "frac_draws_won",
              "swap_nn", "swap_sd_nn", "increment_nn", "frac_draws_won_nn")}
        s["n"] = int(np.mean([r["n"] for r in rs]))
        # SIGN AND SIGNIFICANCE ARE READ TOGETHER. An increment smaller than twice the draw-to-draw spread
        # of its own control is NOT DETECTED, whichever way it points -- it is never "reversed".
        s["detected"] = bool(abs(s["increment"]) > 2 * max(s["swap_sd"], 1e-9))
        rpe1_summary[aname] = s
        report(f"     {aname:34s} {s['n']:8,d} {s['accuracy']:7.4f} {s['auc']:7.4f} {s['swap']:7.4f} "
               f"{s['swap_sd']:6.4f} {s['increment']:+8.4f} {s['frac_draws_won']:5.2f} "
               f"{s['increment_nn']:+8.4f} {s['frac_draws_won_nn']:6.2f}")
    imb_r = SWr.imbalance(rr[lab_r], list(range(6000, 6000 + N_CTRL)), mode="decile")
    imb_rn = SWr.imbalance(rr[lab_r], list(range(6000, 6000 + N_CTRL)), mode="nn")
    report(f"     RPE1 swap residual imbalance, worst |SMD|: decile "
           f"{max(abs(v['smd']) for v in imb_r.values()):.4f}, nearest-neighbour "
           f"{max(abs(v['smd']) for v in imb_rn.values()):.4f}")

    # ------------------------------------------------------------------ 7. how many edges can be signed
    report(f"\n  7. HOW MANY OF THE {int((sg == 0).sum()):,} UNSIGNED EDGES CAN BE SIGNED?")
    report(f"     Rule stated in advance: an edge may be signed only if its out-of-fold confidence bin "
           f"reaches\n     held-out accuracy >= {SIGN_MIN_ACC:.2f} AND an increment over the swapped-source "
           f"control >= {SIGN_MIN_INC:+.2f} inside that\n     same bin. Accuracy alone is not enough: an "
           f"accuracy the swapped source also achieves is the target's\n     preferred direction, not the "
           f"edge's.")
    best_arm = max(k562_summary, key=lambda a: k562_summary[a]["increment"])
    p_best = np.nanmean(np.vstack([oof_by_seed[s][best_arm] for s in SEEDS]), axis=0)
    conf = np.abs(p_best - 0.5)
    bins = []
    m0 = lab & np.isfinite(p_best)
    qs = np.nanquantile(conf[m0], np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = m0 & (conf >= lo) & (conf <= hi if i == 9 else conf < hi)
        if m.sum() < 200:
            continue
        pr = np.where(p_best[m] > 0.5, 1, -1)
        ag = float((np.sign(z_u[m]) == pr).mean())
        sw = SW.agree(r_u[m], c_u[m], pr, range(7000, 7000 + N_CTRL), cond=Z_LAB)
        bins.append({"conf_lo": float(lo), "conf_hi": float(hi), "n": int(m.sum()), "accuracy": ag,
                     "swap": float(np.nanmean(sw)), "increment": ag - float(np.nanmean(sw))})
    report(f"     {'conf bin':>16s} {'n':>8s} {'held-out acc':>13s} {'swap':>8s} {'incr':>8s} {'usable':>7s}")
    for b in bins:
        use = b["accuracy"] >= SIGN_MIN_ACC and b["increment"] >= SIGN_MIN_INC
        b["usable"] = bool(use)
        report(f"     {b['conf_lo']:.3f}-{b['conf_hi']:.3f}    {b['n']:8,d} {b['accuracy']:13.4f} "
               f"{b['swap']:8.4f} {b['increment']:+8.4f} {str(use):>7s}")
    usable = [b for b in bins if b["usable"]]
    if usable:
        cut = min(b["conf_lo"] for b in usable)
        exp_err = 1 - float(np.average([b["accuracy"] for b in usable],
                                       weights=[b["n"] for b in usable]))
    else:
        cut, exp_err = float("nan"), float("nan")
    # what would be signable, if any bin qualified: score every unsigned edge with a model fit on all
    # testable unsigned edges. Reported even when the answer is zero, so the coverage is a measured zero.
    n_signable = 0
    if usable:
        rows_all = np.unique(r_u[lab])
        with np.errstate(invalid="ignore"):
            tdir = np.nanmean(np.sign(Zk[rows_all]), axis=0)
            tresp = np.nanmean(np.abs(Zk[rows_all]), axis=0)
        allu = np.where(sg == 0)[0]
        co_all = coexpr_corr(eA[allu], eB[allu])
        cc_all = np.array([gi.get(b, -1) for b in gname[reg[allu, 1]]])
        # an unsigned edge whose target K562 never measured has no gene statistic and no target-direction
        # feature; it enters as NaN, which the booster handles, and it is counted separately below
        gm_all = np.where(cc_all >= 0, gmean[np.maximum(cc_all, 0)], np.nan)
        gc_all = np.where(cc_all >= 0, gcv[np.maximum(cc_all, 0)], np.nan)
        Ma, fna = edge_features(G, reg[allu, 0], reg[allu, 1], ann, co_all, (gm_all, gc_all))
        td = np.where(cc_all >= 0, tdir[np.maximum(cc_all, 0)], np.nan)
        tr_ = np.where(cc_all >= 0, tresp[np.maximum(cc_all, 0)], np.nan)
        Ma = np.hstack([Ma, np.vstack([td, tr_]).T])
        fna = list(fna) + ["tgtdir_bias", "tgtdir_resp"]
        TD = np.vstack([tdir[c_u], tresp[c_u]]).T
        cols = [i for i, n in enumerate(fna) if n.startswith(ARMS[best_arm])]
        p_all, _ = fit_predict(np.hstack([M, TD])[np.ix_(np.where(lab)[0], cols)], y[lab],
                               Ma[:, cols], 0)
        n_signable = int((np.abs(p_all - 0.5) >= cut).sum())
        report(f"     confidence cut {cut:.3f} -> {n_signable:,} of {len(allu):,} unsigned edges signable "
               f"at an expected error rate of {exp_err:.3f}")
    else:
        report(f"     NO confidence bin reaches accuracy {SIGN_MIN_ACC:.2f} with an increment "
               f"{SIGN_MIN_INC:+.2f} over the swapped-source\n     control. 0 of "
               f"{int((sg == 0).sum()):,} edges may be signed under the rule set before the run.")

    # ------------------------------------------------------------------ 8. verdict
    rpB = rpe1_summary["B source+edge (no K562 in feats)"]
    armA = k562_summary["A target-direction only (K562)"]
    # Where the two control constructions disagree the STRICTER (lower) increment governs, so that adding
    # the tighter control after the imbalance audit can only make the bar harder, never easier.
    inc_k = min(armB["increment"], armB["increment_nn"])
    won_k = min(armB["frac_draws_won"], armB["frac_draws_won_nn"])
    inc_r = min(rpB["increment"], rpB["increment_nn"])
    won_r = min(rpB["frac_draws_won"], rpB["frac_draws_won_nn"])
    ok_a = bool(inc_k >= MIN_INCREMENT and won_k >= MIN_WIN_FRAC)
    ok_b = bool(inc_r > 0 and won_r >= MIN_WIN_FRAC)
    ok_c = bool(armB["increment"] >= armA["increment"])
    ok_d = bool(np.isfinite(surv) and surv <= MAX_REWIRE_SURVIVAL)
    passed = bool(ok_a and ok_b and ok_c and ok_d)
    report(f"\n  8. DECISION against the threshold fixed before the run "
           f"(stricter of the two control constructions)")
    report(f"     (a) K562 increment >= {MIN_INCREMENT:+.3f} in >= {MIN_WIN_FRAC:.0%} of draws: "
           f"{inc_k:+.4f}, {won_k:.0%} -> {ok_a}")
    report(f"     (b) RPE1 increment > 0 on the same terms: {inc_r:+.4f}, {won_r:.0%} -> {ok_b}")
    report(f"     (c) carried by the no-K562-features arm, not the target-direction arm: "
           f"{armB['increment']:+.4f} vs {armA['increment']:+.4f} -> {ok_c}")
    report(f"     (d) degree-matched rewiring destroys >= {MAX_REWIRE_SURVIVAL:.0%} of it: "
           f"survival {surv:.2f} -> {ok_d}")

    cbest = max(conv, key=lambda c: c["increment"])
    ceil2 = [c for c in ceil if c["z_min"] == 2.0][0]
    af_rho = crux.get("curated_act_frac", {}).get("spearman", float("nan"))
    af_p = crux.get("curated_act_frac", {}).get("p", float("nan"))
    core = (
        f"On the {conv[0]['n']:,} curated edges that ALREADY carry a sign, that sign predicts the measured "
        f"K562 perturbational sign at {100*cbest['agree']:.1f}% against {100*cbest['swap']:.1f}% for a "
        f"strength-matched swapped source ({100*cbest['increment']:+.1f} points at |z| >= "
        f"{cbest['z_min']:g}, {cbest['frac_draws_won']:.0%} of {N_CTRL} draws won), so direction is a real "
        f"property of a curated edge and the question was worth asking. It cannot be answered by lookup: "
        f"CollecTRI and TRRUST cover {lookup['signed_edges_covered']:,} of the {int((sg != 0).sum()):,} "
        f"SIGNED edges and {lookup['unsigned_edges_covered']:,} of the {int((sg == 0).sum()):,} unsigned "
        f"ones -- the signed subset IS those resources, and the unsigned subset is a different, "
        f"ChIP/motif-shaped compendium. And the thing that would have to be imputed does exist: estimating "
        f"a source's measured direction bias on half its unsigned targets predicts the other half at "
        f"{100*ceil2['increment']:+.1f} points over the same control, which is the ceiling for any "
        f"source-level feature. The curated layer simply does not know it -- a source's curated activator "
        f"fraction correlates with its measured direction bias at Spearman {af_rho:+.3f} (p {af_p:.2g}).")
    controls = (
        f"CONTROLS, ALL FOUR OF THEM. (i) The target-direction arm -- the confound -- reaches "
        f"{100*armA['accuracy']:.1f}% raw accuracy on held-out TFs while its increment over the swapped "
        f"source is {100*armA['increment']:+.1f} points, which is exactly why 0.5 is the wrong null and why "
        f"a raw accuracy near 60% here means nothing. (ii) Label-shuffled, whole pipeline rerun: acc "
        f"{sh['accuracy']:.4f}, incr {sh['increment']:+.4f}, against the real full arm's "
        f"{real_c['accuracy']:.4f} / {real_c['increment']:+.4f}. (iii) Degree-matched configuration-model "
        f"rewiring, {N_REWIRE} draws with source out-degree and target in-degree preserved EXACTLY, leaves "
        f"the source+edge arm at incr {rw_inc:+.4f} against the real {armB['increment']:+.4f}; the real "
        f"wiring beats the rewired control in {beat:.0%} of draws on increment and {beat_acc:.0%} on "
        f"accuracy, so the little the features do carry is degree and not connectivity. (iv) THE MATCHED "
        f"CONTROL ITSELF FAILED ITS OWN AUDIT AND HAD TO BE TIGHTENED: matching the swapped source on "
        f"perturbation-strength DECILE left a standardised mean difference of {worst:.2f} in strength "
        f"(curated sources are TFs and sit at the top of whatever decile they land in), so a "
        f"nearest-neighbour-in-strength arm was added, which brings the worst |SMD| to {worst_nn:.2f}. It "
        f"moves the K562 increment {armB['increment']:+.4f} -> {armB['increment_nn']:+.4f} and the RPE1 "
        f"increment {rpB['increment']:+.4f} -> {rpB['increment_nn']:+.4f}; the stricter of the two governs "
        f"every decision above. The same defect is present in the decile-matched control that "
        f"colab/causal_reg.py and colab/reliable_edges.py both use, and it is worth checking there.")
    if passed:
        verdict = (
            f"THE SIGN OF AN UNSIGNED CURATED EDGE IS IMPUTABLE, AT {n_signable:,} EDGES. Held out by "
            f"SOURCE GENE so no TF is in both train and test, a model on source, edge and target-annotation "
            f"features with no K562 measurement in it reaches {100*armB['accuracy']:.1f}% accuracy "
            f"(AUC {armB['auc']:.3f}) against {100*armB['swap']:.1f}% for a strength-matched swapped source "
            f"-- {100*armB['increment']:+.1f} points, {armB['frac_draws_won']:.0%} of {N_CTRL} draws won -- "
            f"and transfers to RPE1 at {100*rpe1_summary['B source+edge (no K562 in feats)']['increment']:+.1f}"
            f" points. {n_signable:,} of the {int((sg == 0).sum()):,} unsigned edges clear the "
            f"confidence cut set in advance, at an expected error rate of {exp_err:.3f}. " + core + " " +
            controls)
    else:
        verdict = (
            f"NULL: THE 558,005 UNSIGNED EDGES CANNOT BE SIGNED FROM WHAT THE CURATED LAYER KNOWS, AND THE "
            f"REASON IS PROVENANCE RATHER THAN STATISTICS. Held out by SOURCE GENE, the arm with no K562 "
            f"measurement in its features reaches {100*armB['accuracy']:.1f}% accuracy "
            f"(AUC {armB['auc']:.3f}) against {100*armB['swap']:.1f}% for a strength-matched swapped source "
            f"-- an increment of {100*armB['increment']:+.1f} points against the "
            f"{100*MIN_INCREMENT:+.1f} required, {armB['frac_draws_won']:.0%} of {N_CTRL} draws won -- and "
            f"on RPE1 {100*rpe1_summary['B source+edge (no K562 in feats)']['increment']:+.1f} points. "
            f"0 of the {int((sg == 0).sum()):,} unsigned edges reach a confidence bin that is both "
            f"{SIGN_MIN_ACC:.0%} accurate and {SIGN_MIN_INC:+.2f} above its own swapped-source control, so "
            f"none are signed. " + core + " " + controls + " "
            f"WHAT THIS IS GOOD FOR. The measured layer is the answer for these edges: "
            f"colab/causal_reg.py already carries a K562 perturbational sign for every "
            f"(perturbation, gene) pair it measures, at +15.0 points of RPE1 sign transfer, and "
            f"{int(testable.sum()):,} of the {len(reg):,} curated triples fall inside that matrix. Signing "
            f"a curated edge is a measurement problem, not an inference problem, and the honest coverage "
            f"statement is {100*testable.mean():.0f}% by measurement and 0% by imputation.")
    report("\n" + "=" * 104)
    report(f"  VERDICT: {verdict}")

    R = {"model": "impute-reg-sign-v1",
         "question": "can the sign of the 558,005 unsigned curated regulatory edges be imputed, validated "
                     "where it cannot be circular?",
         "decision_threshold_fixed_before_run": {
             "min_increment_over_swapped_source_k562": MIN_INCREMENT,
             "min_fraction_of_control_draws_won": MIN_WIN_FRAC,
             "rpe1_increment_must_be_positive": True,
             "must_be_carried_by_arm_without_k562_features": True,
             "degree_matched_rewiring_must_destroy_at_least": MAX_REWIRE_SURVIVAL,
             "sign_an_edge_only_if_bin_accuracy": SIGN_MIN_ACC, "and_bin_increment": SIGN_MIN_INC},
         "label": {"definition": "sign of per-gene robust z of K562 log fold change, as in causal_reg.py",
                   "z_min": Z_LAB},
         "joins": {"reg_sources_to_k562_perturbations": jr_p, "reg_targets_to_k562_genes": jr_g,
                   "edge_level_testable_fraction_k562": float(testable.mean()),
                   "edge_level_testable_fraction_rpe1": float(rate_r)},
         "curated_sign_convention": conv,
         "lookup_route": lookup,
         "ceiling_source_level": ceil,
         "annotation_vs_measured_source_bias": crux,
         "cv": {"k_fold": K_FOLD, "seeds": list(SEEDS), "grouped_by": "source gene",
                "pairwise_coassignment_across_seeds": coassign,
                "expected_if_partitions_differ": 1 / K_FOLD},
         "k562": k562_summary, "k562_per_seed": {k: v for k, v in res_k562.items()},
         "rpe1": rpe1_summary,
         "label_shuffled": sh,
         "degree_matched_rewiring": {"n_rewirings": N_REWIRE, "kind": "configuration model; reg target+sign "
                                     "stubs permuted (out-degree and in-degree exact), ppi stub list "
                                     "permuted and re-paired, coexpr and complex incidence permuted",
                                     "mean_accuracy": rw_acc, "mean_increment": rw_inc, "sd": rw_sd,
                                     "real_increment": armB["increment"],
                                     "fraction_of_effect_reproduced_by_degree_matched_control": surv,
                                     "fraction_of_rewirings_the_real_features_beat": beat},
         "swap_control_residual_imbalance": imb, "worst_abs_smd": worst,
         "confidence_bins": bins,
         "signable": {"n_edges_signed": n_signable, "confidence_cut": cut,
                      "expected_error_rate": exp_err,
                      "n_unsigned_total": int((sg == 0).sum())},
         "criteria": {"a_k562_increment": bool(ok_a), "b_rpe1_increment": bool(ok_b),
                      "c_carried_by_no_k562_arm": bool(ok_c), "d_rewiring_destroys_it": bool(ok_d)},
         "passed": passed,
         "limits": [
             "ONLY 59.0% OF CURATED EDGES ARE TESTABLE AT ALL. K562 measures 8,248 genes and RPE1 8,749, "
             "so an edge whose target is unmeasured contributes to no number here -- it is neither signed "
             "nor counted as a failure, and the null is a null about the measurable majority only",
             "the label is a K562 CRISPRi response, which is a causal consequence of knocking the source "
             "down and not evidence the source acts on the target. An indirect edge with a consistent sign "
             "is scored as a correct sign here",
             "the ceiling in section 2 is estimated from a random split of each source's targets, so it "
             "shares the source's perturbation row between the two halves; it bounds source-level "
             "imputation optimistically",
             "training labels are restricted to |z| >= 2, which conditions on the outcome magnitude. Every "
             "control arm is conditioned identically, but the population being predicted is 'edges whose "
             "target responds strongly', not all edges",
             "the source+edge arm's features are dominated by ONE informative quantity, the curated "
             "activator fraction, and that is available on only 86.4% of unsigned edges and rests on the "
             "signed 8.8% whose provenance (CollecTRI/TRRUST) differs from the unsigned 91.2%. A null here "
             "is a null about THIS feature set, not a proof that no feature set could work",
             "the DepMap cross-line co-expression feature is not a graph and therefore has no rewiring "
             "control of its own; its only control is the swapped-source measurement arm",
             "the rewiring control runs at one CV seed rather than three, and the configuration model can "
             "create self-loops and multi-edges, which are kept because removing them would break the "
             "exact-degree guarantee the control exists to provide",
             "the RPE1 arm inherits the K562 source->fold assignment rather than re-partitioning, and only "
             f"{100*rate_r:.1f}% of unsigned curated edges are testable there, so it is a thinner test than "
             "the K562 arm",
             "sources are held out but TARGETS are not: a target gene appears in both training and test "
             "folds. That is deliberate -- the target-direction confound is what the swapped-source "
             "control removes -- but it means a target-specific model could still leak, which is why arm A "
             "is reported separately rather than folded into arm C"],
         "verdict": verdict, "log": log}

    layer = {"model": "reg-sign-imputed-v1",
             "n_edges_signed": n_signable,
             "rule": f"signed only inside confidence bins with held-out accuracy >= {SIGN_MIN_ACC} and "
                     f"increment >= {SIGN_MIN_INC} over a strength-matched swapped-source control",
             "result": "no bin qualified" if not usable else f"cut {cut:.3f}",
             "does_not_replace": "`reg` is left intact; nothing is written back into it",
             "verdict": verdict}
    with gzip.open(LAYER, "wt") as fh:
        json.dump(layer, fh)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "impute_reg_sign.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'impute_reg_sign.json'}")
    report(f"  -> {LAYER}")


if __name__ == "__main__":
    main()
