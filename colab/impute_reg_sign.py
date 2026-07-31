"""IMPUTE THE SIGN OF THE 558,005 UNSIGNED CURATED REGULATORY EDGES -- and the answer is mostly no.

THE FINDING, PUT FIRST SO NOBODY READS THE MOTIVATION AS THE RESULT. Sign is imputable for the edges that
already have one and essentially NOT imputable for the ones that do not, and the reason is structural rather
than statistical. Three measurements, in the order they were made:

  1  THE CONVENTION IS REAL AND STRONG. On the 20,118 curated edges that DO carry a sign and are testable in
     Replogle K562, the curated sign predicts the measured perturbational sign at +17.6 points over a
     strength-matched swapped-source control, rising to +23.4 on the |z| >= 3 subset (70.9% against 47.5%,
     20 of 20 control draws won). Curated sign carries direction. So the layer's signed 8.8% is worth
     something and the question "can the other 91.2% be filled in" is worth asking.
  2  THE LOOKUP ROUTE IS CLOSED BY CONSTRUCTION, AT EXACTLY ZERO EDGES. CollecTRI and TRRUST cover
     43,348 of the 54,128 signed edges and 0 of the 558,005 unsigned ones. The signed subset IS CollecTRI/TRRUST. The
     unsigned subset came from somewhere else -- its top sources are CTCF, ETS1, TFAP2C, STAT1 with
     4,000-5,000 targets each, which is the shape of a ChIP/motif compendium, not of a curated
     activation/repression statement. So an imputation cannot be a join; it has to generalise across a
     provenance boundary.
  3  THE PER-SOURCE DIRECTION EFFECT IS REAL AND IS NOT REGULATORY. Split each source's unsigned targets
     in half, estimate that source's measured direction bias on one half and predict the other, and you
     beat the swapped-source control by several points. The first version of this module called that the
     CEILING on source-level imputation and pointed at TF sequence features as the way to reach it. It ran
     the wrong control. Estimate the same source's direction bias from genes that are NOT curated targets
     of it at all -- same perturbation row, every curated target of that source excluded -- and it predicts
     the held-out TRUE targets at least as well. The quantity being recovered is a global up/down shift of
     that one CRISPRi row (normalisation, stress, growth arrest), not the TF's regulatory logic. There is
     no ceiling for a sequence feature to reach, because there is nothing about the TF in it. That the
     curated activator fraction correlates with it at Spearman +0.010 (p = 0.9) is not a missed
     opportunity.

  4  SO THE MODEL LANDS ON THE NULL, AND ON A PAIRED CONTROL IT LANDS THERE CLEANLY. Held out by source
     gene, a gradient-boosted model on source/edge/target-annotation features reaches about 60% raw
     accuracy against a 59.6% majority-class baseline. The first version scored it against a control that
     lived on different rows and got -0.7 points (decile), +1.4 (nearest-neighbour) and -1.4 (joint), and
     concluded that the control's construction moved the answer by as much as the effect was worth. Paired,
     all three constructions are negative and they agree; the +1.4 was the row-set defect, not the match.
     A degree-matched rewiring of every graph the features touch reproduces whatever is left. 0 of 558,005
     edges are signed.

WHAT IS MEASURED, AND AGAINST WHAT CONTROL, BEFORE ANY NUMBER APPEARS.

  QUANTITY      For a curated edge (source S, target T), the sign of T's response to knocking S down:
                sign(robust z) where z = (lfc - median over perturbations) / (1.4826 * MAD), per gene column
                -- the identical construction used by colab/causal_reg.py, reused rather than re-derived.
  CONTROL       SWAPPED SOURCE, SCORED ON THE SAME ROWS. Keep the target T and the model's predicted sign;
                read the measured sign at a DIFFERENT perturbation S' drawn from the same
                perturbation-strength DECILE. 0.5 is the wrong null: a shared stress/proliferation axis
                gives every target a preferred direction under any perturbation, so sign agreement sits
                away from 0.5 with no edge at all. The number that means anything is agreement MINUS the
                swapped-source arm, over 20 draws, with the fraction of draws the edge arm won.
                AND THE CONTROL MUST BE READ ON THE SAME ROWS AS THE ARM, WHICH THE FIRST VERSION OF THIS
                MODULE DID NOT DO AND WHICH CHANGED ITS ANSWER. The control is conditioned on
                |z_swapped| >= 2 so that it is selected on outcome magnitude exactly as the edge arm is.
                That condition keeps only ~22% of the arm's rows, and the ones it keeps are targets that
                also respond strongly to unrelated perturbations -- stress and proliferation genes, whose
                direction is much more consistent than average. Scoring the arm on all 29,136 of its rows
                against a control living on 6,300 easier ones is not a comparison: the arm's own accuracy
                on the control's rows is 0.5741, not 0.5962. Every increment below is therefore reported
                twice, UNPAIRED (the incumbent construction, kept so the numbers stay comparable to
                colab/causal_reg.py and colab/reliable_edges.py, which have the same defect) and PAIRED
                (both arms on the same rows, draw by draw). The PAIRED number governs every decision.
                It reverses the sign of the one positive increment the first version reported.
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

THE DECISION THRESHOLD, FIXED BEFORE THE NUMBERS AND UNCHANGED BY THE REVISION. The imputation is
declared USABLE only if all four hold, all increments PAIRED and taking the STRICTEST of the three matched
constructions:
    (a) K562 held-out-source agreement exceeds its swapped-source control by >= 3.0 points, in >= 18 of 20
        control draws;
    (b) the same increment on RPE1 is > 0 with >= 18 of 20 draws won, AND the rank increment (AUC against a
        swapped-source AUC) is > 0 -- because the labelled class prior FLIPS between the cell lines (32%
        positive in RPE1 against 60% in K562), so an RPE1 accuracy thresholded at 0.5 in K562 sits below
        the RPE1 majority baseline however well the model ranks, and measures the flip rather than transfer;
    (c) arm B0 -- which contains NO K562 measurement of any kind in its features -- carries the increment,
        not the target-direction arm;
    (d) a degree-matched rewiring of every graph the features touch destroys at least half the increment.
An edge may be signed only inside a confidence bin whose held-out accuracy is >= 0.65 AND whose PAIRED
increment over the swapped-source control is >= +0.05. Anything short of all four is reported as a NULL,
and the null is the deliverable.

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
    tgtann_*  target ANNOTATION only -- curated degrees and gene-level flags, nothing measured
    tgtk562_* the target's K562 EXPRESSION LEVEL and CV. These carry no direction information, but they
              ARE K562 measurements and they are computed over the whole matrix INCLUDING the held-out
              fold's perturbations. The first version of this module put them inside the arm it then
              described as having "no K562 measurement anywhere in its features", which was false. They
              now have their own prefix so that arm B0 below can be exactly that claim and arm B (which
              keeps them) can be compared against it.
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
    add("tgtk562_mean", gene_stat[0])
    add("tgtk562_cv", gene_stat[1])
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

        DECILE      strength decile only -- the construction colab/causal_reg.py uses, kept so the numbers
                    here are comparable to it
        NEAREST     the replacement is drawn from the +/- 150 perturbations nearest in strength RANK, which
                    drives the residual imbalance ON STRENGTH to ~0
        JOINT       strength decile x library-size decile. The nearest-neighbour arm fixes strength (|SMD|
                    0.25 -> 0.07) and leaves UMI count imbalanced at 0.23, because a strong perturbation is
                    not the same thing as a deeply sequenced one. This arm matches both.

    All three are reported. Reporting the loose one alone would have been the mistake; reporting only a
    tight one would have hidden that the incumbent construction has this defect. Where they disagree the
    STRICTEST increment governs the decision.
    """
    NN_WINDOW = 150
    MIN_POOL = 20     # a matched cell smaller than this cannot supply a distinct replacement

    def __init__(self, Z, cov, report, label, joint_cov=None):
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
        jc = self.cov.get(joint_cov)
        if jc is None:
            self.joint = self.dec
        else:
            d2 = np.digitize(jc, np.nanquantile(jc, np.linspace(.1, .9, 9)))
            self.joint = self.dec * 10 + d2
        self.joint_cov = joint_cov
        sizes = np.bincount(self.joint)
        report(f"    {label}: {len(self.strength):,} perturbations; strength deciles sized "
               f"{np.bincount(self.dec, minlength=10).tolist()}; joint strength x {joint_cov} cells "
               f"{int((sizes > 0).sum())} (min occupied {int(sizes[sizes > 0].min())})")

    def draw(self, rows, seed, mode="decile"):
        r = np.random.default_rng(seed)
        if mode == "nn":
            n = len(self.strength)
            off = r.integers(-self.NN_WINDOW, self.NN_WINDOW + 1, size=len(rows))
            off = np.where(off == 0, 1, off)          # never return the source itself
            alt = self.order[np.clip(self.rank[rows] + off, 0, n - 1)]
            self.last_thin, self.last_self = 0, int((alt == rows).sum())
            return alt
        # TWO WAYS A MATCHED DRAW CAN QUIETLY BECOME THE ARM IT IS CONTROLLING, both closed here.
        #   THIN CELLS. The joint strength x library-size grid has 79 occupied cells and the smallest holds
        #   3 perturbations (17 of 11,258 sit in cells below 20). Drawing "a different source" from a cell
        #   of size 1 returns the source itself and the control becomes the edge arm. Rows in a thin cell
        #   fall back to the coarser strength decile, and the fallback count is recorded.
        #   SELF-DRAWS. Even a large pool can return the row itself. Those are redrawn.
        # Neither is a big number, and neither is allowed to be silent: a control that is accidentally the
        # treatment inflates the control and would make a real effect look like a null.
        key = self.joint if mode == "joint" else self.dec
        cnt = np.bincount(key, minlength=int(key.max()) + 1)
        alt = np.empty(len(rows), dtype=int)
        thin = cnt[key[rows]] < self.MIN_POOL
        for q in np.unique(key[rows][~thin]):
            m = (key[rows] == q) & ~thin
            alt[m] = r.choice(np.where(key == q)[0], int(m.sum()))
        for q in np.unique(self.dec[rows][thin]):
            m = thin & (self.dec[rows] == q)
            alt[m] = r.choice(np.where(self.dec == q)[0], int(m.sum()))
        for _ in range(3):
            same = alt == rows
            if not same.any():
                break
            for q in np.unique(self.dec[rows][same]):
                m = same & (self.dec[rows] == q)
                alt[m] = r.choice(np.where(self.dec == q)[0], int(m.sum()))
        self.last_thin = int(thin.sum())
        self.last_self = int((alt == rows).sum())
        return alt

    def agree(self, rows, cols, pred, seeds, zreal=None, cond=None, mode="decile"):
        """Agreement of `pred` with the measured sign at a swapped source -- UNPAIRED and PAIRED.

        `cond` conditions the control identically to the edge arm: selecting the edge arm on |z| >= t
        conditions on the outcome and inflates agreement, so the control must be selected the same way.

        AND CONDITIONING THE CONTROL THAT WAY MOVES IT ONTO DIFFERENT ROWS, WHICH IS A DEFECT THIS
        MODULE ORIGINALLY HAD AND AN ADVERSARIAL RE-READ CAUGHT. The edge arm is scored on every row
        with |z_real| >= t. Requiring |z_swapped| >= t on the same rows keeps only ~22% of them: the
        rows that survive are the ones whose TARGET also responds strongly to an unrelated
        perturbation, i.e. stress/proliferation-responsive genes, whose direction is far more
        consistent than average. The control was therefore being measured on an easier population than
        the arm it controls, and the two numbers were not comparable. Measured on the control's own
        rows the arm drops from 0.5962 to 0.5741 -- 2.2 points, which is THREE TIMES the effect being
        argued about.

        So both are returned:
            UNPAIRED  arm on all its rows vs control on the rows where the swapped source also
                      responds strongly. This is the construction colab/causal_reg.py and
                      colab/reliable_edges.py use, kept so the numbers stay comparable to them.
            PAIRED    arm and control on the SAME rows, draw by draw. This is the honest comparison
                      and it is the one that governs every decision below.
        `zreal` is the measured z at the real source; without it only the unpaired arm is available.
        """
        sw, armp = [], []
        for s in seeds:
            z2 = self.Z[self.draw(rows, s, mode), cols]
            f = np.isfinite(z2)
            if cond is not None:
                f &= np.abs(z2) >= cond
            ok = f.sum() > 20
            sw.append(float((np.sign(z2) == pred)[f].mean()) if ok else float("nan"))
            if zreal is None:
                armp.append(float("nan"))
            else:
                armp.append(float((np.sign(zreal) == pred)[f].mean()) if ok else float("nan"))
        return np.array(sw), np.array(armp)

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


MODES = (("decile", ""), ("nn", "_nn"), ("joint", "_joint"))
SFX = ("", "_nn", "_joint")
SWAP_KEYS = ([f"swap{s}" for s in SFX] + [f"swap_sd{s}" for s in SFX] +
             [f"increment{s}" for s in SFX] + [f"frac_draws_won{s}" for s in SFX] +
             [f"arm_on_control_rows{s}" for s in SFX] + [f"increment_paired{s}" for s in SFX] +
             [f"frac_draws_won_paired{s}" for s in SFX] +
             ["increment_paired_strictest", "frac_draws_won_paired_strictest",
              "increment_strictest", "frac_draws_won_strictest"])


def swap_block(sw_obj, rows, cols, pred, zreal, seeds, cond, ag_all):
    """Every swapped-source statistic, for all three matched constructions, UNPAIRED and PAIRED.

    One helper rather than four copies of the same loop, because the paired columns were added after
    the fact and a copy that was missed would silently report the old, non-comparable number.
    """
    out = {}
    for mode, sfx in MODES:
        sw, armp = sw_obj.agree(rows, cols, pred, seeds, zreal=zreal, cond=cond, mode=mode)
        out[f"swap{sfx}"] = float(np.nanmean(sw))
        out[f"swap_sd{sfx}"] = float(np.nanstd(sw))
        out[f"increment{sfx}"] = ag_all - float(np.nanmean(sw))
        out[f"frac_draws_won{sfx}"] = float(np.mean(ag_all > sw))
        out[f"arm_on_control_rows{sfx}"] = float(np.nanmean(armp))
        out[f"increment_paired{sfx}"] = float(np.nanmean(armp - sw))
        out[f"frac_draws_won_paired{sfx}"] = float(np.mean(armp > sw))
    out["increment_paired_strictest"] = min(out[f"increment_paired{s}"] for s in SFX)
    out["frac_draws_won_paired_strictest"] = min(out[f"frac_draws_won_paired{s}"] for s in SFX)
    out["increment_strictest"] = min(out[f"increment{s}"] for s in SFX)
    out["frac_draws_won_strictest"] = min(out[f"frac_draws_won{s}"] for s in SFX)
    return out


def fit_predict(Xtr, ytr, Xte, seed):
    """One gradient-boosted fit, IDENTICAL hyperparameters everywhere.

    Deliberately small trees. This machine has 4 cores and is shared, and 160 fits have to finish -- but
    the capacity is not what limits the answer: at a held-out AUC of 0.54 the model is signal-limited, not
    capacity-limited, and a larger booster only fits the training fold's noise harder. Every arm, every
    control and every rewiring uses these same settings, so no comparison here is between a big model and
    a small one.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.12, max_depth=4,
                                       min_samples_leaf=80, l2_regularization=1.0,
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
           f"PAIRED increment\n  over the swapped-source control is >= {MIN_INCREMENT:+.3f} in >= "
           f"{MIN_WIN_FRAC:.0%} of {N_CTRL} draws, RPE1 paired increment > 0 on the same terms AND its "
           f"rank\n  increment > 0, the increment is carried by the arm with NO K562 measurement of any "
           f"kind in its\n  features, and a degree-matched rewiring destroys >= "
           f"{MAX_REWIRE_SURVIVAL:.0%} of it. PAIRED means the arm and\n  its control are scored on the "
           f"SAME rows; an edge may be signed only inside a confidence bin at\n  accuracy >= "
           f"{SIGN_MIN_ACC:.2f} with a PAIRED increment >= {SIGN_MIN_INC:+.2f}.")
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
    SW = Swap(Zk, cov, report, "K562", joint_cov="UMI_count_unfiltered")
    sgn_mask = testable & (sg != 0)
    r_s, c_s = ri[sgn_mask], ci[sgn_mask]
    z_s = Zk[r_s, c_s]
    fin = np.isfinite(z_s)
    r_s, c_s, z_s, cur_s = r_s[fin], c_s[fin], z_s[fin], sg[sgn_mask][fin]
    conv = []
    report(f"     The HEADLINE row is |z| >= 0, i.e. every signed testable edge. The |z| >= 2 and |z| >= 3\n"
           f"     rows condition on the outcome magnitude and are reported as SELECTED SUBSETS with their "
           f"own n --\n     the first version of this module quoted the |z| >= 3 accuracy next to the "
           f"|z| >= 0 count, which\n     attached a 1,011-edge number to a 20,118-edge claim.")
    report(f"     {'subset':>14s} {'n':>8s} {'agree':>8s} | {'swap':>8s} {'incrUNP':>9s} {'won':>5s} | "
           f"{'armOnCtl':>9s} {'incrPAIR':>9s} {'won':>5s}")
    for thr in (0.0, 2.0, 3.0):
        k = np.abs(z_s) >= thr
        pr = -cur_s[k]
        ag = float((np.sign(z_s[k]) == pr).mean())
        row = {"z_min": thr, "n": int(k.sum()), "agree": ag, "selected_subset": bool(thr > 0)}
        row.update(swap_block(SW, r_s[k], c_s[k], pr, z_s[k], range(700, 700 + N_CTRL),
                              (thr if thr > 0 else None), ag))
        conv.append(row)
        report(f"     {'|z| >= ' + f'{thr:g}':>14s} {int(k.sum()):8,d} {ag:8.4f} | {row['swap']:8.4f} "
               f"{row['increment']:+9.4f} {row['frac_draws_won']:5.2f} | "
               f"{row['arm_on_control_rows']:9.4f} {row['increment_paired']:+9.4f} "
               f"{row['frac_draws_won_paired']:5.2f}")

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
           f"bias on\n     one half, predict the other. Its control is the same swapped-source arm.")
    report(f"     AND THE CONTROL THAT DECIDES WHAT THIS MEASURES IS A SECOND ARM THE FIRST VERSION OF THIS\n"
           f"     MODULE DID NOT RUN: estimate the same source's direction bias from genes that are NOT "
           f"curated\n     targets of it at all -- same perturbation row, every curated target of that "
           f"source excluded --\n     and use THAT to predict its held-out true targets. If the non-target "
           f"arm does as well, the\n     'source-level signal' is a global up/down shift of that one CRISPRi "
           f"row (normalisation, stress,\n     growth arrest), not regulatory specificity, and no sequence "
           f"feature of the TF could ever recover\n     it, because it is a property of the experiment "
           f"rather than of the TF.")
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
    # every curated target of a source, so the non-target arm can exclude all of them
    tset_all = collections.defaultdict(set)
    for a_, b_ in zip(reg[:, 0], ci):
        if b_ >= 0:
            tset_all[int(a_)].add(int(b_))
    row_of_src = {}
    for s_, r_ in zip(s_u, r_u):
        row_of_src.setdefault(int(s_), int(r_))
    ceil = []
    strong_lab = np.abs(z_u) >= Z_LAB
    bias = collections.defaultdict(list)
    for s, z in zip(s_u[half & strong_lab], z_u[half & strong_lab]):
        bias[int(s)].append(np.sign(z))
    bs = {k: float(np.mean(v)) for k, v in bias.items() if len(v) >= 10}
    # ...and the same quantity estimated off NON-targets of the same source, matched in count
    rgnt = np.random.default_rng(11)
    bs_nt = {}
    for s, rw in row_of_src.items():
        if s not in bs:
            continue
        zz = Zk[rw]
        ok = np.isfinite(zz) & (np.abs(zz) >= Z_LAB)
        ok[list(tset_all[s])] = False        # exclude EVERY curated target of this source, signed or not
        idx = np.where(ok)[0]
        if len(idx) < 10:
            continue
        pick = rgnt.choice(idx, min(len(idx), max(10, len(bias[s]))), replace=False)
        bs_nt[s] = float(np.mean(np.sign(zz[pick])))
    report(f"     per-source direction bias estimated from {len(bs):,} sources' true targets and from "
           f"{len(bs_nt):,}\n     sources' NON-targets (same row, all curated targets removed)")
    for thr in (0.0, 2.0):
        strong = np.abs(z_u) >= max(thr, 1e-9)
        for tname, table in (("TRUE-target half", bs), ("NON-target genes", bs_nt)):
            has = np.array([int(s) in table for s in s_u])
            sub = (~half) & strong & has
            pr = np.array([1 if table[int(s)] > 0 else -1 for s in s_u[sub]])
            ag = float((np.sign(z_u[sub]) == pr).mean())
            row = {"z_min": thr, "estimated_from": tname, "n": int(sub.sum()),
                   "n_sources": len(table), "agree": ag}
            row.update(swap_block(SW, r_u[sub], c_u[sub], pr, z_u[sub], range(800, 800 + N_CTRL),
                                  (thr if thr > 0 else None), ag))
            ceil.append(row)
            report(f"     |z| >= {thr:g}  {tname:17s} n {int(sub.sum()):7,d} over {len(table):4,d} sources"
                   f"  agree {ag:.4f}  swap {row['swap']:.4f}  incrUNPAIRED {row['increment']:+.4f}  "
                   f"incrPAIRED {row['increment_paired']:+.4f} ({row['frac_draws_won_paired']:.0%} won)")
    c2t = [c for c in ceil if c["z_min"] == 2.0 and c["estimated_from"] == "TRUE-target half"][0]
    c2n = [c for c in ceil if c["z_min"] == 2.0 and c["estimated_from"] == "NON-target genes"][0]
    ceil_reproduced = (c2n["increment_paired"] / c2t["increment_paired"]
                       if c2t["increment_paired"] > 1e-9 else float("nan"))
    ceiling_is_row_global = bool(ceil_reproduced >= 0.75)
    report(f"     -> the NON-TARGET arm reaches {c2n['increment_paired']:+.4f} against the true-target "
           f"arm's {c2t['increment_paired']:+.4f}: it reproduces {100*ceil_reproduced:.0f}% of the "
           f"'ceiling'\n        while knowing NOTHING about which genes the source regulates. At most "
           f"{100*(c2t['increment_paired']-c2n['increment_paired']):.1f} of the "
           f"{100*c2t['increment_paired']:.1f} points is target-specific.")
    report(f"     -> THE CEILING IS A GLOBAL PROPERTY OF THE PERTURBATION ROW, NOT SOURCE-LEVEL "
           f"REGULATORY SIGNAL."
           if ceiling_is_row_global else
           f"     -> the true-target arm exceeds the non-target arm, so some of the ceiling is "
           f"target-specific.")

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
    af_cov_edges = float(np.isfinite(G.act_frac[reg[idx_u, 0]]).mean())
    af_cov_src = float(np.isfinite(G.act_frac[np.unique(reg[idx_u, 0])]).mean())
    report(f"     src_act_frac -- the one feature with any prior claim to informativeness -- is FINITE ON "
           f"ONLY\n     {100*af_cov_edges:.1f}% of these edges and {100*af_cov_src:.1f}% of their source "
           f"genes (a source needs >= {MIN_SIGNED} signed\n     curated edges before it has one). The "
           f"first version of this module asserted 86.4% in its limits; it is not.")
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

    # B0 is the arm the verdict is allowed to describe as containing no K562 measurement of any kind.
    # B keeps the target's K562 expression level and CV -- no direction information, but a measurement,
    # and one computed over the held-out fold's rows as well. Both are run so the difference is visible
    # rather than asserted.
    ARMS = {"A target-direction only (K562)": ("tgtdir_",),
            "B0 source+edge, ZERO K562 feats": ("src_", "edge_", "tgtann_"),
            "B  +target K562 expr level": ("src_", "edge_", "tgtann_", "tgtk562_"),
            "C full": ("src_", "edge_", "tgtann_", "tgtk562_", "tgtdir_")}
    ARM_B = "B0 source+edge, ZERO K562 feats"

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
        out.update(swap_block(sw_obj, rows[m], cols[m], pr, zvec[m], seeds0, Z_LAB, ag))
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
    report(f"     UNPAIRED = arm on all its rows vs control on the ~22% of them where the swapped source "
           f"also\n     responds strongly (the incumbent construction). PAIRED = both on the SAME rows. "
           f"The paired\n     columns are the honest ones and they govern the decision.")
    report(f"     {'arm':34s} {'n':>7s} {'acc':>7s} {'AUC':>6s} {'major':>6s} | {'incrUdec':>8s} "
           f"{'incrUnn':>8s} {'incrUjnt':>8s} | {'armOnCtl':>8s} {'incrPdec':>8s} {'incrPnn':>8s} "
           f"{'incrPjnt':>8s} {'wonP':>5s}")
    k562_summary = {}
    for aname, rs in res_k562.items():
        s = {k: float(np.mean([r[k] for r in rs])) for k in
             ["accuracy", "auc", "majority_baseline"] + SWAP_KEYS}
        s["accuracy_sd_over_seeds"] = float(np.std([r["accuracy"] for r in rs]))
        s["n"] = int(np.mean([r["n"] for r in rs]))
        k562_summary[aname] = s
        report(f"     {aname:34s} {s['n']:7,d} {s['accuracy']:7.4f} {s['auc']:6.3f} "
               f"{s['majority_baseline']:6.3f} | {s['increment']:+8.4f} {s['increment_nn']:+8.4f} "
               f"{s['increment_joint']:+8.4f} | {s['arm_on_control_rows']:8.4f} "
               f"{s['increment_paired']:+8.4f} {s['increment_paired_nn']:+8.4f} "
               f"{s['increment_paired_joint']:+8.4f} {s['frac_draws_won_paired_strictest']:5.2f}")
    # how much of the arm's row set the control's own conditioning actually keeps -- the size of the
    # defect that pairing removes, stated as a number rather than asserted
    _z2 = SW.Z[SW.draw(r_u[lab], 2000, "decile"), c_u[lab]]
    frac_ctrl_rows = float((np.isfinite(_z2) & (np.abs(_z2) >= Z_LAB)).mean())
    report(f"     the |z| >= {Z_LAB:g} condition on the control keeps {100*frac_ctrl_rows:.1f}% of the "
           f"arm's {int(lab.sum()):,} rows; the arm and its\n     control were being scored on those "
           f"different populations, which is what the paired columns fix.")
    report(f"     the arm's own accuracy falls from {k562_summary[ARM_B]['accuracy']:.4f} on all its rows "
           f"to {k562_summary[ARM_B]['arm_on_control_rows']:.4f} on the control's rows:\n     that "
           f"{k562_summary[ARM_B]['accuracy']-k562_summary[ARM_B]['arm_on_control_rows']:.4f} gap is the "
           f"whole reason the unpaired nearest-neighbour increment looked positive.")
    report(f"     seed-to-seed sd of accuracy (different partitions, not rotated fold labels): " +
           ", ".join(f"{a.split()[0]} {k562_summary[a]['accuracy_sd_over_seeds']:.4f}"
                     for a in k562_summary))

    # residual imbalance of both swapped-source matches
    imb = {m: SW.imbalance(r_u[lab], list(range(2000, 2000 + N_CTRL)), mode=m)
           for m in ("decile", "nn", "joint")}
    report(f"\n     RESIDUAL IMBALANCE per covariate (standardised mean difference, real vs swapped "
           f"source) -- rule 3's check")
    report(f"       {'covariate':22s} {'real':>12s} | {'decile':>11s} {'SMD':>7s} | {'NN':>11s} "
           f"{'SMD':>7s} | {'joint':>11s} {'SMD':>7s}")
    for k in imb["decile"]:
        d, n_, j_ = imb["decile"][k], imb["nn"][k], imb["joint"][k]
        report(f"       {k:22s} {d['real']:12.4f} | {d['swapped']:11.4f} {d['smd']:+7.3f} | "
               f"{n_['swapped']:11.4f} {n_['smd']:+7.3f} | {j_['swapped']:11.4f} {j_['smd']:+7.3f}")
    worst = max(abs(v["smd"]) for v in imb["decile"].values())
    worst_nn = max(abs(v["smd"]) for v in imb["nn"].values())
    worst_j = max(abs(v["smd"]) for v in imb["joint"].values())
    draw_audit = {}
    for m_ in ("decile", "nn", "joint"):
        SW.draw(r_u[lab], 2000, m_)
        draw_audit[m_] = {"rows": int(lab.sum()), "fell_back_to_strength_decile": SW.last_thin,
                          "drew_the_source_itself": SW.last_self}
    report(f"       draw audit (one draw, {int(lab.sum()):,} rows): " +
           "; ".join(f"{k} thin-cell fallbacks {v['fell_back_to_strength_decile']:,}, self-draws "
                     f"{v['drew_the_source_itself']:,}" for k, v in draw_audit.items()))
    for nm, w in (("decile", worst), ("nearest-neighbour", worst_nn), ("joint", worst_j)):
        report(f"       worst |SMD| {nm:18s} {w:.4f} "
               f"({'ok, < 0.10' if w < 0.10 else 'RESIDUAL IMBALANCE PRESENT'})")
    report(f"       The decile match leaves the control WEAKER in strength than the arm it controls, which "
           f"would\n       flatter the edge arm. The nearest-neighbour match fixes strength and leaves "
           f"library size behind;\n       the joint match fixes both and leaves mitochondrial fraction. "
           f"The strictest increment governs. But note\n       which defect actually mattered: matching "
           f"moves the unpaired increment by ~2 points and PAIRING moves\n       it by ~2 points in the "
           f"other direction, and it is pairing that removes the only positive number here.")

    # ------------------------------------------------------------------ 4. label shuffle
    report(f"\n  4. LABEL-SHUFFLED ARM (significance only -- the effect size above is the RAW held-out "
           f"value)")
    shuf = []
    for seed, fmap in zip(SEEDS, fmaps):
        o = run_cv(M, fnames, fmap, seed, shuffle=True, arms={"C full": ARMS["C full"]})
        shuf.append(score(o["C full"], lab, SW, r_u, c_u, z_u, "shuffled",
                          range(3000 + 50 * seed, 3000 + 50 * seed + N_CTRL)))
    sh = {k: float(np.mean([r[k] for r in shuf])) for k in ["accuracy", "auc"] + SWAP_KEYS}
    report(f"     labels permuted, full pipeline rerun: acc {sh['accuracy']:.4f}  AUC {sh['auc']:.4f}  "
           f"incrUNPAIRED {sh['increment']:+.4f}  incrPAIRED {sh['increment_paired']:+.4f}")
    real_c = k562_summary["C full"]
    report(f"     real full arm:                       acc {real_c['accuracy']:.4f}  "
           f"AUC {real_c['auc']:.4f}  incrUNPAIRED {real_c['increment']:+.4f}  "
           f"incrPAIRED {real_c['increment_paired']:+.4f}")

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
                   arms={ARM_B: ARMS[ARM_B]})
        rew.append(score(o[ARM_B], lab, SW, r_u, c_u, z_u, f"rewire{j}",
                         range(4000 + 30 * j, 4000 + 30 * j + 5)))
        if j % 5 == 0:
            report(f"       rewiring {j+1}/{N_REWIRE}: acc {rew[-1]['accuracy']:.4f} "
                   f"incr {rew[-1]['increment']:+.4f}")
    rw_acc = float(np.mean([r["accuracy"] for r in rew]))
    rw_inc = float(np.mean([r["increment_paired"] for r in rew]))
    rw_sd = float(np.std([r["increment_paired"] for r in rew]))
    rw_inc_unp = float(np.mean([r["increment"] for r in rew]))
    armB = k562_summary[ARM_B]
    beat = float(np.mean([armB["increment_paired"] > r["increment_paired"] for r in rew]))
    beat_acc = float(np.mean([armB["accuracy"] > r["accuracy"] for r in rew]))
    # A RATIO IS ONLY READABLE WHEN THERE IS AN EFFECT TO DIVIDE. If the real increment is <= 0 there is
    # nothing for the rewiring to destroy and "survival 2.21" would be an artefact of a negative
    # denominator, not a result.
    surv = rw_inc / armB["increment_paired"] if armB["increment_paired"] > 1e-9 else float("nan")
    report(f"     real arm B0     : acc {armB['accuracy']:.4f}  incrPAIRED "
           f"{armB['increment_paired']:+.4f}")
    report(f"     degree-matched  : acc {rw_acc:.4f}  incrPAIRED {rw_inc:+.4f} (sd {rw_sd:.4f}) over "
           f"{N_REWIRE} rewirings  [unpaired {rw_inc_unp:+.4f}]")
    report(f"     the real wiring beats the degree-matched control on the paired increment in {beat:.0%} "
           f"of\n     rewirings, on accuracy in {beat_acc:.0%}")
    if np.isfinite(surv):
        report(f"     fraction of the real increment the degree-matched control already reproduces: "
               f"{surv:.2f}")
    else:
        report(f"     survival ratio NOT COMPUTED: the real paired increment is "
               f"{armB['increment_paired']:+.4f}, so there is no\n     effect for a rewiring to destroy. "
               f"Degree is not the explanation for a result that does not exist.")

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
    SWr = Swap(Zr, {"rpe1_ncells": rn.astype(float)}, report, "RPE1", joint_cov="rpe1_ncells")
    lab_r = okr & np.isfinite(zr) & (np.abs(zr) >= Z_LAB)
    report(f"    of those, {int(lab_r.sum()):,} have |RPE1 z| >= {Z_LAB:g}")
    # THE CLASS PRIOR FLIPS BETWEEN THE TWO CELL LINES AND THAT BREAKS AN ACCURACY READOUT. Among K562
    # rows with |z| >= 2, 59.6% are positive; among RPE1 rows with |z| >= 2 only 32.4% are, so the RPE1
    # majority class is the OPPOSITE one. A model thresholded at p = 0.5 in K562 therefore scores far
    # BELOW the RPE1 majority baseline no matter how good its ranking is, and the first version of this
    # module reported those sub-baseline accuracies (0.34-0.40) and their increments as if they measured
    # transfer. They do not: they measure the prior flip. The baseline is now printed next to them, and
    # a RANK readout that a prior shift cannot touch (AUC against a swapped-source AUC) is reported
    # alongside, because AUC is invariant to exactly the shift that destroys the accuracy.
    frac_pos_r = float((np.sign(zr[lab_r]) > 0).mean())
    report(f"    CLASS PRIOR: {100*frac_pos_r:.1f}% of RPE1 labelled rows are POSITIVE against "
           f"{100*y[lab].mean():.1f}% in K562 --\n    the majority class FLIPS, so an accuracy "
           f"thresholded at 0.5 in K562 is not a transfer measurement here.")
    rpe1_summary = {}
    report(f"     {'arm':34s} {'n':>7s} {'acc':>7s} {'major':>6s} {'AUC':>6s} {'swapAUC':>7s} "
           f"{'dAUC':>7s} | {'incrUdec':>8s} {'incrPdec':>8s} {'incrPnn':>8s} {'incrPjnt':>8s} "
           f"{'wonP':>5s}")
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
            # the rank readout and its OWN matched control: same probabilities, but the label is the
            # sign measured at a strength-matched DIFFERENT perturbation of the same target
            sa = []
            for sd_ in range(6000 + 40 * seed, 6000 + 40 * seed + N_CTRL):
                z2 = Zr[SWr.draw(rr[m], sd_, "decile"), rc[m]]
                f2 = np.isfinite(z2) & (np.abs(z2) >= Z_LAB)
                if f2.sum() > 20:
                    sa.append(auc((np.sign(z2[f2]) > 0).astype(int), p[m][f2]))
            row["swap_auc"] = float(np.nanmean(sa)) if sa else float("nan")
            row["swap_auc_sd"] = float(np.nanstd(sa)) if sa else float("nan")
            row["auc_increment"] = row["auc"] - row["swap_auc"]
            row["frac_auc_draws_won"] = float(np.mean(row["auc"] > np.array(sa))) if sa else float("nan")
            row.update(swap_block(SWr, rr[m], rc[m], pr, zr[m],
                                  range(6000 + 40 * seed, 6000 + 40 * seed + N_CTRL), Z_LAB, ag))
            rs.append(row)
        s = {k: float(np.mean([r[k] for r in rs])) for k in
             ["accuracy", "auc", "majority_baseline", "swap_auc", "swap_auc_sd", "auc_increment",
              "frac_auc_draws_won"] + SWAP_KEYS}
        s["n"] = int(np.mean([r["n"] for r in rs]))
        # SIGN AND SIGNIFICANCE ARE READ TOGETHER. An increment smaller than twice the draw-to-draw spread
        # of its own control is NOT DETECTED, whichever way it points -- it is never "reversed".
        s["detected"] = bool(abs(s["increment_paired"]) > 2 * max(s["swap_sd"], 1e-9))
        s["auc_detected"] = bool(abs(s["auc_increment"]) > 2 * max(s["swap_auc_sd"], 1e-9))
        rpe1_summary[aname] = s
        report(f"     {aname:34s} {s['n']:7,d} {s['accuracy']:7.4f} {s['majority_baseline']:6.3f} "
               f"{s['auc']:6.3f} {s['swap_auc']:7.3f} {s['auc_increment']:+7.3f} | "
               f"{s['increment']:+8.4f} {s['increment_paired']:+8.4f} "
               f"{s['increment_paired_nn']:+8.4f} {s['increment_paired_joint']:+8.4f} "
               f"{s['frac_draws_won_paired_strictest']:5.2f}")
    report(f"     every arm's accuracy sits BELOW the RPE1 majority baseline of "
           f"{rpe1_summary[ARM_B]['majority_baseline']:.3f}; read the AUC columns, not\n     the accuracy "
           f"columns, for whether anything transfers.")
    imb_r = SWr.imbalance(rr[lab_r], list(range(6000, 6000 + N_CTRL)), mode="decile")
    imb_rn = SWr.imbalance(rr[lab_r], list(range(6000, 6000 + N_CTRL)), mode="nn")
    imb_rj = SWr.imbalance(rr[lab_r], list(range(6000, 6000 + N_CTRL)), mode="joint")
    report(f"     RPE1 swap residual imbalance, worst |SMD|: decile "
           f"{max(abs(v['smd']) for v in imb_r.values()):.4f}, nearest-neighbour "
           f"{max(abs(v['smd']) for v in imb_rn.values()):.4f}, joint "
           f"{max(abs(v['smd']) for v in imb_rj.values()):.4f}")

    # ------------------------------------------------------------------ 7. how many edges can be signed
    report(f"\n  7. HOW MANY OF THE {int((sg == 0).sum()):,} UNSIGNED EDGES CAN BE SIGNED?")
    report(f"     Rule stated in advance: an edge may be signed only if its out-of-fold confidence bin "
           f"reaches\n     held-out accuracy >= {SIGN_MIN_ACC:.2f} AND a PAIRED increment over the "
           f"swapped-source control >= {SIGN_MIN_INC:+.2f} inside\n     that same bin. Accuracy alone is "
           f"not enough: an accuracy the swapped source also achieves is the\n     target's preferred "
           f"direction, not the edge's. The unpaired increment is printed beside it and is\n     the "
           f"LOOSER of the two here -- signing on it would have been the easier rule, and nothing passes "
           f"either.")
    best_arm = max(k562_summary, key=lambda a: k562_summary[a]["increment_paired"])
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
        b = {"conf_lo": float(lo), "conf_hi": float(hi), "n": int(m.sum()), "accuracy": ag}
        b.update(swap_block(SW, r_u[m], c_u[m], pr, z_u[m], range(7000, 7000 + N_CTRL), Z_LAB, ag))
        bins.append(b)
    report(f"     {'conf bin':>16s} {'n':>8s} {'held-out acc':>13s} {'swap':>8s} {'incrUNP':>8s} "
           f"{'incrPAIR':>9s} {'usable':>7s}")
    for b in bins:
        # the SIGNING rule reads the PAIRED increment, i.e. the strict one. Reading the unpaired
        # increment here would let a bin qualify because its control was measured on other rows.
        use = b["accuracy"] >= SIGN_MIN_ACC and b["increment_paired"] >= SIGN_MIN_INC
        b["usable"] = bool(use)
        report(f"     {b['conf_lo']:.3f}-{b['conf_hi']:.3f}    {b['n']:8,d} {b['accuracy']:13.4f} "
               f"{b['swap']:8.4f} {b['increment']:+8.4f} {b['increment_paired']:+9.4f} {str(use):>7s}")
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
        report(f"     NO confidence bin reaches accuracy {SIGN_MIN_ACC:.2f} with a paired increment "
               f"{SIGN_MIN_INC:+.2f} over the swapped-source\n     control. 0 of "
               f"{int((sg == 0).sum()):,} edges may be signed under the rule set before the run.")

    # ------------------------------------------------------------------ 8. verdict
    rpB = rpe1_summary[ARM_B]
    armA = k562_summary["A target-direction only (K562)"]
    # Where the two control constructions disagree the STRICTER (lower) increment governs, so that adding
    # the tighter control after the imbalance audit can only make the bar harder, never easier.
    # PAIRED and STRICTEST-OF-THREE. Adding a control construction can only make the bar harder.
    inc_k = armB["increment_paired_strictest"]
    won_k = armB["frac_draws_won_paired_strictest"]
    inc_r = rpB["increment_paired_strictest"]
    won_r = rpB["frac_draws_won_paired_strictest"]
    ok_a = bool(inc_k >= MIN_INCREMENT and won_k >= MIN_WIN_FRAC)
    # (b) is now read on the RANK readout as well, because the RPE1 accuracy is destroyed by the class
    # prior flipping between cell lines and an accuracy increment there measures the flip, not transfer.
    ok_b = bool(inc_r > 0 and won_r >= MIN_WIN_FRAC and rpB["auc_increment"] > 0)
    ok_c = bool(armB["increment_paired"] >= armA["increment_paired"])
    ok_d = bool(np.isfinite(surv) and surv <= MAX_REWIRE_SURVIVAL)
    passed = bool(ok_a and ok_b and ok_c and ok_d)
    report(f"\n  8. DECISION against the threshold fixed before the run "
           f"(PAIRED, STRICTEST of the three control constructions)")
    report(f"     (a) K562 paired increment >= {MIN_INCREMENT:+.3f} in >= {MIN_WIN_FRAC:.0%} of draws: "
           f"{inc_k:+.4f}, {won_k:.0%} -> {ok_a}    [unpaired, for comparison: "
           f"{armB['increment_strictest']:+.4f}]")
    report(f"     (b) RPE1 paired increment > 0 on the same terms AND rank transfer > 0: "
           f"{inc_r:+.4f}, {won_r:.0%}, dAUC {rpB['auc_increment']:+.3f} -> {ok_b}")
    report(f"     (c) carried by the zero-K562-features arm, not the target-direction arm: "
           f"{armB['increment_paired']:+.4f} vs {armA['increment_paired']:+.4f} -> {ok_c}")
    report(f"     (d) degree-matched rewiring destroys >= {MAX_REWIRE_SURVIVAL:.0%} of it: "
           f"survival {surv:.2f} -> {ok_d}")

    conv0 = conv[0]                                    # |z| >= 0: EVERY signed testable edge
    cbest = max([c for c in conv if c["z_min"] > 0], key=lambda c: c["increment_paired"])
    ceil2 = c2t
    af_rho = crux.get("curated_act_frac", {}).get("spearman", float("nan"))
    af_p = crux.get("curated_act_frac", {}).get("p", float("nan"))
    core = (
        f"On ALL {conv0['n']:,} curated edges that already carry a sign, that sign predicts the measured "
        f"K562 perturbational sign at {100*conv0['agree']:.1f}% against {100*conv0['swap']:.1f}% for a "
        f"strength-matched swapped source measured on the SAME rows -- {100*conv0['increment_paired']:+.1f} "
        f"points, {conv0['frac_draws_won_paired']:.0%} of {N_CTRL} draws won. On the "
        f"{cbest['n']:,}-edge subset selected post hoc at |z| >= {cbest['z_min']:g} it is "
        f"{100*cbest['agree']:.1f}% against {100*cbest['arm_on_control_rows']:.1f}% on the control's rows, "
        f"{100*cbest['increment_paired']:+.1f} points -- NOT the {100*cbest['increment']:+.1f} the unpaired "
        f"comparison reports, and that subset number belongs to {cbest['n']:,} edges, not to "
        f"{conv0['n']:,}. Direction is a real property of a curated edge; it is worth about half what an "
        f"unpaired control makes it look like. It cannot be answered by lookup: CollecTRI and TRRUST cover "
        f"{lookup['signed_edges_covered']:,} of the {int((sg != 0).sum()):,} SIGNED edges and "
        f"{lookup['unsigned_edges_covered']:,} of the {int((sg == 0).sum()):,} unsigned ones -- the signed "
        f"subset IS those resources, and the unsigned subset is a different, ChIP/motif-shaped compendium. "
        f"AND THE 'CEILING' THAT THE FIRST VERSION OF THIS MODULE OFFERED AS THE REASON TO TRY HARDER IS "
        f"NOT REGULATORY SIGNAL AT ALL. Estimating a source's measured direction bias on half its unsigned "
        f"targets predicts the other half at {100*c2t['increment_paired']:+.1f} points over the same "
        f"control -- but estimating that bias from genes that are NOT curated targets of the source, same "
        f"perturbation row, every curated target excluded, predicts those same held-out true targets at "
        f"{100*c2n['increment_paired']:+.1f} points -- {100*ceil_reproduced:.0f}% of it, from a control "
        f"that knows nothing about which genes the source regulates, leaving at most "
        f"{100*(c2t['increment_paired']-c2n['increment_paired']):.1f} points that is target-specific. What "
        f"looked like a source-level ceiling for a future sequence feature to reach is a global up/down "
        f"shift of one CRISPRi row -- normalisation, stress, growth arrest -- and no property of the TF "
        f"could predict it, because it is a property of the experiment. That the curated activator "
        f"fraction correlates with it at Spearman {af_rho:+.3f} (p {af_p:.2g}) is therefore not a missed "
        f"opportunity; there was nothing there to see.")
    controls = (
        f"CONTROLS, FIVE OF THEM, AND THE FIFTH REWROTE THE OTHER FOUR. (i) PAIRING. The swapped-source "
        f"control is conditioned on |z| >= {Z_LAB:g} exactly as the edge arm is -- correctly, because "
        f"selecting on outcome magnitude inflates agreement -- but that condition moves the control onto "
        f"DIFFERENT ROWS: only ~{100*frac_ctrl_rows:.0f}% of the arm's rows survive it, and the survivors "
        f"are targets that also respond strongly to unrelated perturbations, whose direction is far more "
        f"consistent. Measured on the control's own rows the arm scores "
        f"{armB['arm_on_control_rows']:.4f} rather than {armB['accuracy']:.4f}. Every increment is "
        f"therefore reported twice, and the paired one governs: arm B0 moves from "
        f"{armB['increment']:+.4f} (unpaired decile) / {armB['increment_nn']:+.4f} (unpaired nearest) to "
        f"{armB['increment_paired']:+.4f} / {armB['increment_paired_nn']:+.4f} paired. The nearest-neighbour "
        f"POSITIVE increment that made the three constructions look like they straddled zero was an "
        f"artefact of the unequal row sets; paired, all three are negative and the answer is the same "
        f"whichever match is used. (ii) The target-direction arm -- the confound -- reaches "
        f"{100*armA['accuracy']:.1f}% raw accuracy on held-out TFs at a paired increment of "
        f"{100*armA['increment_paired']:+.1f} points, which is why 0.5 is the wrong null and why a raw "
        f"accuracy near 60% here means nothing. (iii) Label-shuffled, whole pipeline rerun: acc "
        f"{sh['accuracy']:.4f}, AUC {sh['auc']:.4f}, paired incr {sh['increment_paired']:+.4f}, against the "
        f"real full arm's {real_c['accuracy']:.4f} / {real_c['auc']:.4f} / "
        f"{real_c['increment_paired']:+.4f}. (iv) Degree-matched configuration-model rewiring, "
        f"{N_REWIRE} draws with source out-degree and target in-degree preserved EXACTLY, leaves the "
        f"source+edge arm at paired incr {rw_inc:+.4f} against the real {armB['increment_paired']:+.4f}; "
        f"the real wiring beats the rewired control in {beat:.0%} of draws on increment and "
        f"{beat_acc:.0%} on accuracy, so the little the features do carry is degree, not connectivity. "
        f"(v) The matched control still fails its own imbalance audit: strength |SMD| {worst:.2f} under a "
        f"decile match (curated sources are TFs and sit at the top of whatever decile they land in), "
        f"{worst_nn:.2f} under nearest-neighbour-in-strength, {worst_j:.2f} under a joint strength x "
        f"library-size match, with mitochondrial fraction stuck near 0.19 under all three. Those "
        f"tightenings are why three constructions are reported; PAIRING, not matching, turned out to be "
        f"what moved the answer. colab/causal_reg.py and colab/reliable_edges.py use the unpaired "
        f"decile-matched construction and should be re-audited for the same row-set defect -- their "
        f"increments are large enough that a 2-point shift would not overturn them, but it should be "
        f"stated.")
    if passed:
        verdict = (
            f"THE SIGN OF AN UNSIGNED CURATED EDGE IS IMPUTABLE, AT {n_signable:,} EDGES. Held out by "
            f"SOURCE GENE so no TF is in both train and test, an arm containing no K562 measurement of any "
            f"kind reaches {100*armB['accuracy']:.1f}% raw accuracy (AUC {armB['auc']:.3f}) against "
            f"{100*armB['arm_on_control_rows']:.1f}% for a strength-matched swapped source on the same "
            f"rows -- {100*armB['increment_paired']:+.1f} points, "
            f"{armB['frac_draws_won_paired']:.0%} of {N_CTRL} draws won -- and transfers to RPE1 at "
            f"dAUC {rpB['auc_increment']:+.3f}. {n_signable:,} of the {int((sg == 0).sum()):,} unsigned "
            f"edges clear the confidence cut set in advance, at an expected error rate of {exp_err:.3f}. "
            + core + " " + controls)
    else:
        verdict = (
            f"NULL: THE {int((sg == 0).sum()):,} UNSIGNED EDGES CANNOT BE SIGNED FROM WHAT THE CURATED "
            f"LAYER KNOWS, AND ON A CORRECTED CONTROL THE NULL IS CLEANER THAN THE FIRST VERSION OF THIS "
            f"MODULE REPORTED. Held out by SOURCE GENE so no TF is in both train and test, the arm "
            f"containing NO K562 measurement of any kind (arm B0) reaches {100*armB['accuracy']:.1f}% raw "
            f"held-out accuracy (AUC {armB['auc']:.3f}) against a majority-class baseline of "
            f"{100*armB['majority_baseline']:.1f}%. Against a strength-matched swapped source measured on "
            f"THE SAME ROWS it is {100*armB['increment_paired']:+.1f} points (decile), "
            f"{100*armB['increment_paired_nn']:+.1f} (nearest-neighbour), "
            f"{100*armB['increment_paired_joint']:+.1f} (joint strength x library size), winning "
            f"{armB['frac_draws_won_paired_strictest']:.0%} of {N_CTRL} draws at worst, against the "
            f"{100*MIN_INCREMENT:+.1f} required. ALL THREE ARE NEGATIVE. The first version reported "
            f"{100*armB['increment_nn']:+.1f} for the nearest-neighbour arm and concluded that 'the "
            f"control's construction moves the answer by as much as the effect is worth'; that positive "
            f"number was an artefact of scoring the arm and its control on different rows, and once they "
            f"are scored on the same rows the three constructions agree. On RPE1 the rank transfer is "
            f"dAUC {rpB['auc_increment']:+.3f} against a swapped-source AUC of {rpB['swap_auc']:.3f} "
            f"({'detected' if rpB['auc_detected'] else 'not detected'}); the RPE1 ACCURACY of "
            f"{100*rpB['accuracy']:.1f}% is not a transfer measurement at all, because the labelled class "
            f"prior flips between the lines ({100*frac_pos_r:.1f}% positive in RPE1 against "
            f"{100*y[lab].mean():.1f}% in K562) and every arm sits below the RPE1 majority baseline of "
            f"{100*rpB['majority_baseline']:.1f}%. 0 of the {int((sg == 0).sum()):,} unsigned edges reach a "
            f"confidence bin that is both {SIGN_MIN_ACC:.0%} accurate and {SIGN_MIN_INC:+.2f} above its own "
            f"swapped-source control on the same rows, so none are signed and the layer written is empty "
            f"by construction. " + core + " " + controls +
            " WHAT TO DO INSTEAD, AND IT IS NARROWER THAN THE FIRST VERSION CLAIMED. The measured layer "
            f"already answers this for the edges it can reach: colab/causal_reg.py carries a K562 "
            f"perturbational sign for every (perturbation, gene) pair it measures, and "
            f"{int(testable.sum()):,} of the {len(reg):,} curated triples ({100*testable.mean():.0f}%) fall "
            f"inside that matrix -- {int(uns.sum()):,} of them currently unsigned. Signing a curated edge "
            f"is a measurement problem, not an inference problem, and the honest coverage statement for "
            f"this project is {100*testable.mean():.0f}% by measurement and 0% by imputation. The first "
            f"version pointed at sequence-level features and a '+7.6 point source-level ceiling' as the "
            f"obvious next attempt; the non-target control above removes that motivation, because the "
            f"ceiling is a property of the perturbation row and not of the TF. The route that would extend "
            f"coverage is more perturbation atlases: the unsigned edges whose source is never perturbed "
            f"anywhere are the ones that stay dark.")
    report("\n" + "=" * 104)
    report(f"  VERDICT: {verdict}")

    R = {"model": "impute-reg-sign-v1",
         "question": "can the sign of the 558,005 unsigned curated regulatory edges be imputed, validated "
                     "where it cannot be circular?",
         "decision_threshold_fixed_before_run": {
             "increments_are_PAIRED": "the arm and its swapped-source control are scored on the SAME "
                                      "rows; the unpaired construction is reported alongside for "
                                      "comparability with colab/causal_reg.py but governs nothing",
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
                                     "mean_accuracy": rw_acc, "mean_increment_paired": rw_inc,
                                     "mean_increment_unpaired": rw_inc_unp, "sd": rw_sd,
                                     "real_increment_paired": armB["increment_paired"],
                                     "real_increment_unpaired": armB["increment"],
                                     "real_accuracy": armB["accuracy"],
                                     "fraction_of_effect_reproduced_by_degree_matched_control": surv,
                                     "fraction_of_rewirings_the_real_features_beat_on_increment": beat,
                                     "fraction_of_rewirings_the_real_features_beat_on_accuracy": beat_acc},
         "swap_control": {"constructions": ["decile: strength decile, as colab/causal_reg.py",
                                            "nn: +/-150 nearest in strength rank, added after the decile "
                                            "arm failed its own imbalance audit",
                                            "joint: strength decile x library-size decile, added after the "
                                            "nn arm still left UMI count imbalanced"],
                          "residual_imbalance_k562": imb, "worst_abs_smd_decile": worst,
                          "worst_abs_smd_nn": worst_nn,
                          "worst_abs_smd_joint": worst_j, "draw_audit": draw_audit,
                          "worst_abs_smd_rpe1_decile": max(abs(v["smd"]) for v in imb_r.values()),
                          "worst_abs_smd_rpe1_nn": max(abs(v["smd"]) for v in imb_rn.values()),
                          "worst_abs_smd_rpe1_joint": max(abs(v["smd"]) for v in imb_rj.values())},
         "worst_abs_smd": worst,
         "paired_vs_unpaired": {
             "fraction_of_arm_rows_the_control_condition_keeps": frac_ctrl_rows,
             "arm_accuracy_on_its_own_rows": armB["accuracy"],
             "arm_accuracy_on_the_controls_rows": armB["arm_on_control_rows"],
             "why": "conditioning the control on |z|>=2 moves it onto stress-responsive targets whose "
                    "direction is more consistent; scoring the arm on all its rows against a control on "
                    "that subset is not a comparison. The paired columns fix it and reverse the sign of "
                    "the nearest-neighbour increment."},
         "ceiling_is_a_row_global_bias_not_source_signal": ceiling_is_row_global,
         "ceiling_fraction_reproduced_by_non_target_control": ceil_reproduced,
         "src_act_frac_coverage": {"fraction_of_scored_edges": af_cov_edges,
                                   "fraction_of_source_genes": af_cov_src},
         "rpe1_class_prior": {"frac_positive_rpe1": frac_pos_r, "frac_positive_k562": float(y[lab].mean()),
                              "note": "the majority class FLIPS between the lines, so the RPE1 accuracy "
                                      "readout measures the flip and not transfer; read the AUC columns"},
         "confidence_bins": bins,
         "signable": {"n_edges_signed": n_signable, "confidence_cut": cut,
                      "expected_error_rate": exp_err,
                      "n_unsigned_total": int((sg == 0).sum())},
         "criteria": {"a_k562_increment_paired": bool(ok_a), "b_rpe1_increment_and_rank": bool(ok_b),
                      "c_carried_by_no_k562_arm": bool(ok_c), "d_rewiring_destroys_it": bool(ok_d)},
         "passed": passed,
         "limits": [
             "ONLY 59.0% OF CURATED EDGES ARE TESTABLE AT ALL. K562 measures 8,248 genes and RPE1 8,749, "
             "so an edge whose target is unmeasured contributes to no number here -- it is neither signed "
             "nor counted as a failure, and the null is a null about the measurable majority only",
             "the label is a K562 CRISPRi response, which is a causal consequence of knocking the source "
             "down and not evidence the source acts on the target. An indirect edge with a consistent sign "
             "is scored as a correct sign here",
             "THE 'CEILING' IN SECTION 2 IS NOT SOURCE-LEVEL REGULATORY SIGNAL. Both halves come from the "
             "SAME perturbation row, so any global up/down shift of that row transfers between them "
             "perfectly. The non-target control added here settles it: a source's direction bias estimated "
             "from genes that are NOT its curated targets predicts its held-out true targets at least as "
             "well as the bias estimated from its own targets. The quantity is a property of the "
             "experiment, not of the TF, and no sequence feature could recover it",
             "training labels are restricted to |z| >= 2, which conditions on the outcome magnitude. Every "
             "control arm is conditioned identically, but the population being predicted is 'edges whose "
             "target responds strongly', not all edges",
             f"the source+edge arm rests on ONE quantity with any prior claim to informativeness, the "
             f"curated activator fraction, and it is finite on only {100*af_cov_edges:.1f}% of the scored "
             f"edges and {100*af_cov_src:.1f}% of their source genes -- NOT the 86.4% an earlier version of "
             f"this module asserted -- and it is derived from the signed 8.8% whose provenance "
             f"(CollecTRI/TRRUST) differs from the unsigned 91.2%. Its values are also nearly degenerate "
             f"(interquartile range about 0.78-0.92). A null here is a null about THIS feature set on THIS "
             f"coverage, not a proof that no feature set could work",
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
             "is reported separately rather than folded into arm C",
             "the JOINT swapped-source arm is worse balanced on RPE1 than the plain decile arm, because "
             "RPE1 has only 2,122 perturbations and the strength x cell-count grid leaves thin cells that "
             "fall back to the coarser match. On RPE1 the decile and nearest-neighbour arms are the "
             "trustworthy ones; the joint arm is reported because the decision takes the strictest of the "
             "three and a badly-matched arm can only make the bar harder",
             "mitochondrial fraction stays imbalanced at |SMD| ~0.19 under ALL THREE matches. It is not "
             "matched on, and a perturbation that stresses cells is both a strong perturbation and a "
             "high-mito one, so some of what the swapped-source control removes is stress rather than "
             "source identity",
             "THE PAIRED SCORING, THE NON-TARGET CEILING CONTROL, ARM B0, THE RPE1 RANK READOUT AND THE "
             "NEAREST-NEIGHBOUR AND JOINT SWAPPED-SOURCE ARMS WERE ALL ADDED AFTER THE FIRST RUN, in "
             "response to an adversarial re-read of this module. Every one of them is a tightening rather "
             "than a loosening and the strictest number governs every decision, but none was pre-registered "
             "and they are reported as post-hoc controls. The decision threshold itself was not changed",
             "THE UNPAIRED INCREMENT IS STILL REPORTED because colab/causal_reg.py and "
             "colab/reliable_edges.py use that construction and the numbers have to stay comparable to "
             "theirs. It should not be read as an effect size here, and those two modules carry the same "
             "row-set defect and should be re-audited",
             "the paired comparison restricts both arms to the rows where the swapped source ALSO responds "
             "at |z| >= 2, which is a stress-responsive subset of targets. It is the right comparison "
             "because both arms are measured there, but it is a narrower population than the arm's full "
             "row set, and the arm's raw accuracy on its own full row set is reported separately as the "
             "effect size",
             "the RPE1 ACCURACY numbers cannot measure transfer, because the labelled class prior flips "
             "between the lines (32% positive in RPE1 against 60% in K562) and a model thresholded at 0.5 "
             "in K562 lands below the RPE1 majority baseline regardless of how well it ranks. Criterion "
             "(b) therefore also requires a positive rank increment, and the AUC columns are the ones to "
             "read. Only 13.4% of unsigned curated edges are RPE1-testable, so it stays a thin test",
             "a null on this feature set is not a proof that sign is unpredictable. Sequence-level "
             "features that were not available here -- the source's DNA-binding domain family, its "
             "cofactor complexes from a proteomics resource, promoter motif orientation at the target -- "
             "are the obvious next attempt, and the ceiling measurement says there is up to +7 points of "
             "source-level signal for them to reach",
             "the imputation is scored against a PERTURBATIONAL sign, which is what a whole-cell model "
             "needs but is not the same object as the curated activation/repression statement. An edge "
             "whose curated sign is right and whose K562 response is buffered scores as an error here"],
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
