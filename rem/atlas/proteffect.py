"""The effect of each protein on every other gene and pathway: what is real, what is shared, and
what belongs to the protein itself.

WHY THIS IS NEW. effectmap.py measured protein -> pathway effects in ONE experiment and found them
dominated by a few cell-wide axes; its only replicate was 102 genes knocked down at two promoters.
It never asked which effects on individual GENES are real. Replogle et al. 2022 ran three separate
Perturb-seq experiments with the same normalisation (figshare 20029387, CC BY 4.0):
   A  K562 genome-wide   9,865 knockdowns x 8,248 genes   day 8
   B  K562 essential     2,057 knockdowns x 8,563 genes   day 6   (separate cells, sequencing, time)
   C  RPE1 essential     2,393 knockdowns x 8,749 genes   day 7   (a different, non-cancer, TP53-wild-type line)
2,053 proteins were knocked down in both A and B, 2,052 in all three. THE SAME sgRNA LIBRARY ELEMENTS
were used in all three (the pre-run review checked: identical guide-pair IDs for 2,048 of 2,053), so
"reproduced in A and B" means reproduced in new cells, on a new day, sequenced separately -- NOT with
an independent reagent: a guide's off-target effect would reproduce too. (Before this was written
only shapes and overlaps were inspected; no effect value was looked at.)

NAMES. Every knockdown and measured gene is mapped to its CURRENT HGNC symbol through its Ensembl id
(obs/gene_transcript's last field, var/gene_id; hgnc_complete_set.txt), else an unambiguous previous
symbol, else kept. All lookups (TSS positions, HGNC groups, Reactome, literature controls) use these.

UNITS AND NOISE. Every value is the normalised mean expression change of a gene in cells carrying the
knockdown. Each screen's non-targeting control rows (A 585, B 109, C 130) give each gene's noise:
sd_g = 1.4826 x MAD over control rows, floored at the 10th percentile across genes. Z = value / sd_g;
non-finite -> 0 (no effect).

CRISPRi ARTEFACTS, REMOVED OR FLAGGED BEFORE ANY ANALYSIS.
  SELF and CIS: the knocked-down gene and every gene with a refGene hg38 TSS within 100 kb of one of the
    target's TSSs are set to 0 in that knockdown's row, and kept at 0 after every later transformation
    (dCas9-KRAB silences neighbouring promoters -- a property of the tool, not of the protein). Shared
    knockdowns and measured genes with no TSS after name mapping are dropped (their zones are unknown).
  PARALOG FLAG (heuristic): a DOWN effect on a gene sharing an HGNC gene group with the target, the
    group not being a domain/repeat/complex family, and the two symbols sharing their first 3
    characters, is flagged POSSIBLE GUIDE CROSS-REACTIVITY (kept, not removed).
  READOUT FLAGS: mitochondrially encoded (MT-) and replication-dependent histone genes are kept but
    flagged, and left out of the hub rankings in R6.

SHARED vs SPECIFIC. A knockdown's effect = the tide (what every knockdown does) + a few cell-wide axes
+ the rest. The shared subspace is learnt ONLY from screen-A knockdowns of proteins NOT knocked down in
B, with an expression phenotype (energy test p < 0.05): their mean (the tide) and top 5 principal axes
(effectmap: 5 axes = 83% of specific pathway variance). Each row's coefficients on that 6-dimensional
span are fitted by Huber-weighted least squares (k = 3, 5 iterations; cis cells excluded), so a sparse
strong effect is not smeared over the axes; the SPECIFIC row is the remainder, cis cells re-zeroed.
SPECIFIC therefore means "beyond the tide and 5 cell-wide axes": it still contains programmes shared by
the members of a complex or pathway (losing any subunit of the proteasome moves the proteasome genes),
and it REMOVES the defining effects of regulators that drive a shared axis.

=================================================================================================
GATES, PREDECLARED (revised after the blind pre-run review, before any number existed -- see below)
=================================================================================================
H0 HARNESS (blocking). (a) For A, the loader returns celldiscover.load_gwps's matrix and knockdown list
   exactly, apart from knockdowns whose symbol starts with "non" (NONO), which load_gwps drops by
   mistake. (b) In each screen the knocked-down gene's own Z has median <= -3. (c) Name mapping: the
   counts of shared knockdowns and measured genes still without a TSS are printed (they are dropped).
H1 THE INSTRUMENT SEES THE KNOWN CRISPRi ARTEFACT (reported, not blocking). In A, before cis zeroing:
   P(Z <= -3) by distance from the target's TSS (0-1, 1-10, 10-50, 50-100, 100-500 kb; same chromosome
   > 1 Mb) vs other chromosomes. PASS if the < 10 kb odds ratio vs other chromosomes is >= 2, p < 0.001.
H2 THE INSTRUMENT SEES KNOWN REGULATION (blocking for interpretation: if it fails, every verdict below
   is printed NOT INTERPRETABLE and nothing is written to the memory bank). Literature sets, expected
   DOWN on knockdown; raw Z; cis-zone targets excluded; >= 5 measured targets; z = (score - mean) / sd
   of the same set score over the screen's control rows; a member passes at z <= -3.
     GATA1 -> HBG1, HBG2, HBZ, HBA1, HBA2, ALAS2, GYPA        (Zon 1991 PMID 1660143; Woon Kim 2011 21609963)
     SREBP pathway: SREBF2, SCAP, MBTPS1, MBTPS2 -> 21 sterol genes  (Horton 2002 11994399; Rawson 1997 9659902)
     ATF4 -> 24 amino-acid-response genes                     (Harding 2003 12667446; Torrence 2021 33646118)
     HSF1 -> HSPA8, HSP90AA1, HSPD1, HSPE1, CHORDC1, DNAJA1 (basal targets; Mendillo 2012 22863008)
     MYC -> 19 ribosome-biogenesis / nucleolar genes          (Muhar 2018 29622725, in K562)
     TFAM -> 10 mitochondrially encoded genes                 (Larsson 1998 9500544)
     NRF1 -> TFAM, TFB2M, TFB1M, COX6C, CYCS                  (Gleyzer 2005 15684387)
   The gate counts FAMILIES per screen (the four SREBP-pathway regulators are one family, passing in a
   screen if most of its evaluable members pass): >= 2/3 of evaluable (family, screen) pairs must pass.
   REPORTED, NOT COUNTED: NFE2L1 -> proteasome subunits (largely inactive in unstressed cells; Radhakrishnan
   2010 20385086); NEGATIVE CONTROL TP53 in K562, a TP53-null line (Law 1993 8246608) -> p53 targets,
   PASS if |z| < 3 (weak: it shows noise rarely reaches |z| >= 3). A pass shows the instrument is
   SENSITIVE to known regulation; growth-linked sets (MYC) also fall in many other knockdowns.
R1 DO WHOLE EFFECT PROFILES REPEAT ACROSS EXPERIMENTS? Per shared protein, Pearson r of its A and B rows
   (raw Z); the median vs the medians of 200 derangements (A_p paired with B_q, q != p): REPEATS if
   above their 95th percentile.
R2 DO THE SPECIFIC PARTS REPEAT? Same on specific rows. Per protein, exact empirical p against ALL 2,052
   mismatched partners, (1 + #{q != p: r_pq >= r_pp}) / n; Benjamini-Hochberg FDR < 0.05 -> the protein
   has a REPRODUCIBLE SPECIFIC PROFILE. Reported: how many, their median r, and how many belong to an
   HGNC group with >= 2 other tested proteins (complex- or family-level programmes).
R3 GENE-LEVEL CATALOGUE. p -> g is REPRODUCED if |Z| >= t in A and in B with the same sign. For SPECIFIC,
   on specific Z, also requiring the raw and specific signs to agree in each screen (no sign created by
   the projection). FDR = SIGN-FLIP estimate: co-hits with OPPOSITE signs / co-hits with the same sign
   (noise has random sign; this keeps each row's own noise and strength, which a derangement does not).
   t from the ladder 3, 4, 5: the first t with FDR <= 0.10 AND >= 20 effects; if none, NO RELIABLE
   CATALOGUE. Reported beside it, not as an FDR: the derangement ratio (mean co-hits with a mismatched
   partner / real), i.e. the share NOT attributable to the protein's identity.
R4 DO THEY HOLD IN ANOTHER CELL TYPE? For reproduced K562 effects with the protein knocked down and the
   gene measured in RPE1, among those with |Z_C| >= 3 (raw for ALL; specific, fitted the same way with
   the subspace re-orthonormalised on the genes all three measure, for SPECIFIC): the share with the
   SAME sign as in K562. GENERAL BEYOND CHANCE if it is above 1/2, exceeds the same share for all 20
   derangements of the RPE1 rows, AND a binomial test against 1/2 gives p < 0.05. (Sign agreement, because a strong
   knockdown is strong in both lines whether or not its gene-level effects are conserved.) The share
   reaching |Z_C| >= 3 at all is printed as description only.
R5 PROTEIN -> PATHWAY CATALOGUE. Specific Z averaged over each Reactome process's genes (15-300 measured
   genes; cis genes out of that knockdown's numerator and denominator); z vs the same score over control
   rows treated identically. REPRODUCED if |z| >= t in A and B, same sign; sign-flip FDR; ladder 3, 4, 5,
   7, 10 with the R3 rule; derangement ratio reported beside it.
R6 THE NETWORK'S SHAPE (descriptive): effects per protein and per gene; proteins with the most
   reproduced specific effects; genes most often hit (flagged readouts left out); paralog-flag share.
X  WHAT THIS IS NOT. Steady state (day 6-8): direct and indirect effects are mixed. A reproduced effect
   is an effect of the KNOCKDOWN with this reagent -- usually of losing the protein, but guide
   off-targets and co-silenced promoter partners (e.g. head-to-head genes beyond the cis zone's reach
   of their trans effects) are not excluded. Two K562 time points (day 6, 8): agreement is a lower
   bound. Pathway effects are averages over their genes.

PRE-RUN REVIEW (2026-10-13; 3 reviewers, 21 verifiers, a PubMed check; before any number existed).
Changes from the committed draft (8bc6dc8): cis cells re-zeroed after residualisation (the projection
had refilled them, so artefacts and p -> p would have 'reproduced' by construction); names mapped via
Ensembl -> HGNC (old symbols had left 166 measured cis-neighbour pairs unzeroed and lost 6 ATF4
targets); robust (Huber) axis fit and a raw/specific sign-agreement rule (the plain projection could
create opposite-sign 'specific' effects); exact all-partner p in R2 (200 derangements could not
declare fewer than 205 proteins); sign-flip FDR in R3/R5 (derangements count real shared effects as
false and are fooled by strength-linked noise); R4 tests sign agreement (the coverage test passed with
no conserved effect in simulation); H2 families, NFE2L1 reported-not-counted, targets revised per the
literature check, and H2 now blocks; R6 directions from the field that defines each class; NONO kept;
paralog flag restricted; MT-/histone readouts flagged; ladder needs >= 20 effects; JSON strict.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import collections
import csv
import gzip
import importlib.util
import json
import math
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CA = os.path.join(HERE, "_cache")
OUT = os.path.join(HERE, "RESULTS_proteffect.txt")
ART = R("outputs", "proteffect.json")
BANK = R("memory_bank", "cell_v3")
HGNC = os.path.join(HERE, "hgnc_complete_set.txt")
RULE = "=" * 97
FILES = {"A": os.path.join(CA, "perturbseq", "K562_gwps_normalized_bulk_01.h5ad"),
         "B": os.path.join(CA, "perturbseq_ess", "K562_essential_normalized_bulk_01.h5ad"),
         "C": os.path.join(CA, "perturbseq_rpe1", "rpe1_normalized_bulk_01.h5ad")}
CITE = "Replogle et al., Cell 2022, doi 10.1016/j.cell.2022.05.013; figshare 20029387; CC BY 4.0"


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


cd, wc, rc = _load("celldiscover"), _load("wholecell"), _load("rawchip")

K_AX, CIS, NEAR = 5, 100_000, 10_000
LADDER, LADDER_P, FDR_MAX, MIN_N, ZC_T = (3.0, 4.0, 5.0), (3.0, 4.0, 5.0, 7.0, 10.0), 0.10, 20, 3.0
N_DER, SEED, HUBER_K, IRLS = 20, 20261013, 3.0, 5
BINS = [(0, 1_000), (1_000, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 500_000)]
SREBP = ["HMGCR", "HMGCS1", "LDLR", "INSIG1", "SQLE", "MVD", "FDFT1", "IDI1", "LSS", "DHCR7", "MVK", "FDPS",
         "SC5D", "CYP51A1", "MSMO1", "NSDHL", "HSD17B7", "EBP", "TM7SF2", "DHCR24", "STARD4"]
FAMILIES = {
    "GATA1": (["GATA1"], ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "ALAS2", "GYPA"]),
    "SREBP": (["SREBF2", "SCAP", "MBTPS1", "MBTPS2"], SREBP),
    "ATF4": (["ATF4"], ["ASNS", "PSAT1", "PHGDH", "PSPH", "CHAC1", "DDIT3", "TRIB3", "SESN2", "SLC7A11", "VEGFA",
                        "ATF3", "CEBPB", "SLC3A2", "SLC7A5", "MTHFD2", "ALDH18A1", "CARS1", "WARS1", "GARS1",
                        "SARS1", "YARS1", "AARS1", "EIF4EBP1", "XPOT"]),
    "HSF1": (["HSF1"], ["HSPA8", "HSP90AA1", "HSPD1", "HSPE1", "CHORDC1", "DNAJA1"]),
    "MYC": (["MYC"], ["NPM1", "NCL", "NOP56", "NOP58", "FBL", "DKC1", "NOP16", "RRP9", "RRS1", "PA2G4", "CAD", "ODC1",
                      "SRM", "POLR1B", "BOP1", "PES1", "WDR12", "GNL3", "NOLC1"]),
    "TFAM": (["TFAM"], ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP6", "MT-CYB", "MT-ND4", "MT-ND5", "MT-CO3",
                        "MT-ND3"]),
    "NRF1": (["NRF1"], ["TFAM", "TFB2M", "TFB1M", "COX6C", "CYCS"]),
}
REPORTED = {
    "NFE2L1": (["NFE2L1"], ["PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7", "PSMB1", "PSMB2", "PSMB3",
                            "PSMB4", "PSMB5", "PSMB6", "PSMB7", "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
                            "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD11", "PSMD12", "PSMD14"], "down"),
    "TP53 (negative)": (["TP53"], ["CDKN1A", "MDM2", "BAX", "BBC3", "GADD45A", "RRM2B", "SESN1", "TP53I3", "FAS",
                                   "ZMAT3", "TNFRSF10B", "PLK3"], "none"),
}
NOT_PARALOG = ("domain containing", "zinc finger", "ring finger", "rna binding motif", "wd repeat", "armadillo",
               "ribosomal", "proteasome", "nadh", "spliceosom", "nucleoporin", "subunit", "complex", "repeat",
               "host gene", "kinases", "phosphatase", "motif", "binding", "family members", "proteins")


# =================================================================================================
# LOADING, NAMES, NOISE, ARTEFACT ZONES
# =================================================================================================
def load_screen(path):
    """One row per knocked-down gene (its strongest-knockdown row, as celldiscover.load_gwps), plus every
    non-targeting control row. Symbols as in the file, with their Ensembl ids."""
    import h5py
    f = h5py.File(path, "r")
    X = f["X"][()]
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]
    fold = f["obs/fold_expr"][()]; core = f["obs/core_control"][()]; ep = f["obs/energy_test_p_value"][()]
    cats = [s.decode() for s in f["var/__categories/gene_name"][()]]
    genes = [cats[c] for c in f["var/gene_name"][()]]
    gens = [s.decode() if isinstance(s, bytes) else str(s) for s in f["var/gene_id"][()]]
    best, ctrl, ens = {}, [], {}
    for i, g in enumerate(gt):
        parts = g.split("_")
        sym = parts[1] if len(parts) > 1 else g
        if "non-targeting" in g.lower():
            ctrl.append(i)
            continue
        if core[i]:
            continue
        fv = fold[i] if np.isfinite(fold[i]) else 1.0
        if sym not in best or fv < best[sym][1]:
            best[sym] = (i, fv)
        if parts[-1].startswith("ENSG"):
            ens[sym] = parts[-1]
    kos = sorted(best)
    rows = [best[k][0] for k in kos]
    Xs = X[rows].astype(np.float32); Xs[~np.isfinite(Xs)] = np.nan
    Xc = X[sorted(ctrl)].astype(np.float32); Xc[~np.isfinite(Xc)] = np.nan
    return dict(X=Xs, kos=kos, kos_ens=[ens.get(k) for k in kos], genes=genes, genes_ens=gens,
                ep=np.array([ep[r] for r in rows]), Xc=Xc)


def hgnc_tables():
    ens2sym, prev, groups = {}, collections.defaultdict(set), {}
    for r in csv.DictReader(open(HGNC), delimiter="\t"):
        s = r["symbol"]
        if r.get("ensembl_gene_id"):
            ens2sym[r["ensembl_gene_id"]] = s
        for q in (r.get("prev_symbol") or "").split("|"):
            if q:
                prev[q].add(s)
        groups[s] = {g for g in (r.get("gene_group") or "").split("|") if g}
    return ens2sym, prev, groups


def make_canon(ens2sym, prev, current):
    def canon(sym, ens):
        if ens and ens in ens2sym:
            return ens2sym[ens]
        if sym in current:
            return sym
        if len(prev.get(sym, ())) == 1:
            return next(iter(prev[sym]))
        return sym
    return canon


def canonicalise(s, canon):
    """Rename knockdowns and measured genes to current symbols; duplicates keep their first row/column."""
    kc = [canon(p, e) for p, e in zip(s["kos"], s["kos_ens"])]
    keep, seen = [], set()
    for i, k in enumerate(kc):
        if k not in seen:
            seen.add(k); keep.append(i)
    s["dup_kos"] = len(kc) - len(keep)
    s["X"], s["ep"], s["kos"] = s["X"][keep], s["ep"][keep], [kc[i] for i in keep]
    s["renamed_kos"] = sum(1 for i in keep if kc[i] != s["kos_raw"][i])
    gc = [canon(g, e) for g, e in zip(s["genes"], s["genes_ens"])]
    s["renamed_genes"] = sum(1 for a, b in zip(gc, s["genes"]) if a != b)
    s["genes"] = gc
    s["dup_genes"] = len(gc) - len(set(gc))


def noise_sd(Xc):
    med = np.nanmedian(Xc, 0)
    sd = 1.4826 * np.nanmedian(np.abs(Xc - med), 0)
    ok = np.isfinite(sd) & (sd > 0)
    floor = float(np.percentile(sd[ok], 10))
    return np.where(ok, np.maximum(sd, floor), floor).astype(np.float32)


def to_z(X, sd):
    Z = X / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z.astype(np.float32)


def first_index(names):
    out = {}
    for j, g in enumerate(names):
        out.setdefault(g, j)
    return out


def tss_positions(tss):
    """gene -> [(chrom, TSS)], primary-assembly chromosomes only."""
    gpos = collections.defaultdict(list)
    for c, (pos, names) in tss.items():
        if "_" in c:
            continue
        for x, n in zip(pos, names):
            gpos[n].append((c, int(x)))
    return gpos


def zone(p, window, tss, gpos, lo_bp=0):
    """Genes with a TSS within `window` bp (and >= lo_bp) of one of p's TSSs; p itself included when lo_bp == 0."""
    out = {p} if lo_bp == 0 else set()
    near = set()
    for c, x in gpos.get(p, ()):
        pos, names = tss[c]
        lo, hi = np.searchsorted(pos, x - window), np.searchsorted(pos, x + window, side="right")
        out.update(names[lo:hi])
        if lo_bp:
            a, b = np.searchsorted(pos, x - lo_bp + 1), np.searchsorted(pos, x + lo_bp - 1, side="right")
            near.update(names[a:b])
    return out - near if lo_bp else out


def zero_cis(Z, kos, gi, tss, gpos):
    """Zero each row's self + cis zone in place; returns row -> sorted list of zeroed columns."""
    cols = {}
    for r, p in enumerate(kos):
        c = sorted({gi[g] for g in zone(p, CIS, tss, gpos) if g in gi})
        Z[r, c] = 0.0
        cols[r] = c
    return cols


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def rownorm(M):
    C = M - M.mean(1, keepdims=True)
    n = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.where(n > 0, n, 1.0)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def resid_robust(Z, Q, excl):
    """Remove each row's span(Q) component, coefficients by Huber IRLS on non-excluded cells; excluded
    cells set to 0 afterwards."""
    out = np.empty(Z.shape, np.float32)
    k = Q.shape[1]
    for i in range(len(Z)):
        z = Z[i].astype(np.float64)
        w = (~excl[i]).astype(np.float64)
        c = np.zeros(k)
        for _ in range(IRLS):
            Qw = Q * w[:, None]
            c = np.linalg.solve(Q.T @ Qw + 1e-9 * np.eye(k), Qw.T @ z)
            r = np.abs(z - Q @ c)
            w = np.where(excl[i], 0.0, np.minimum(1.0, HUBER_K / np.maximum(r, 1e-12)))
        out[i] = z - Q @ c
    out[excl] = 0.0
    return out


def cohits(MA, MB, t, perm=None, agree=None):
    """(same-sign, opposite-sign) co-exceedances of |value| >= t; agree: extra boolean mask."""
    Mb = MB if perm is None else MB[perm]
    hit = (np.abs(MA) >= t) & (np.abs(Mb) >= t)
    if agree is not None:
        hit &= agree if perm is None else agree[1][perm] & agree[0]
    same = hit & (np.sign(MA) == np.sign(Mb))
    return int(same.sum()), int((hit & ~same).sum())


def ladder(rungs, real_fn, der_fn):
    """First rung with sign-flip FDR <= FDR_MAX and >= MIN_N effects; table rows (t, same, opposite,
    fdr, derangement ratio)."""
    table = []
    for t in rungs:
        same, opp = real_fn(t)
        fdr = opp / same if same else None
        dr = float(np.mean([der_fn(t, d) for d in range(N_DER)])) / same if same else None
        table.append((t, same, opp, fdr, dr))
        if fdr is not None and fdr <= FDR_MAX and same >= MIN_N:
            return t, table
    return None, table


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue)


def paralog(p, g, groups):
    if p[:3] != g[:3]:
        return False
    shared = groups.get(p, set()) & groups.get(g, set())
    return any(not any(w in s.lower() for w in NOT_PARALOG) for s in shared)


def clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None, tuples/sets to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def readout_flag(g):
    if g.startswith("MT-"):
        return "mtDNA"
    if g.startswith(("H1-", "H2AC", "H2BC", "H3C", "H4C", "HIST")):
        return "histone"
    return ""


# =================================================================================================
def main():
    from scipy.stats import fisher_exact
    out, res = [], {}

    def P_(s=""):
        print(s, flush=True); out.append(s)

    def finish():
        open(OUT, "w").write("\n".join(out) + "\n")
        os.makedirs(os.path.dirname(ART), exist_ok=True)
        json.dump(clean(res), open(ART, "w"), indent=1, allow_nan=False)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    P_(RULE); P_("THE EFFECT OF EACH PROTEIN ON EVERY OTHER GENE AND PATHWAY: REAL, SHARED, OR ITS OWN"); P_(RULE)
    S = {k: load_screen(v) for k, v in FILES.items()}

    # ---- H0 (a): loader identity, before renaming ----------------------------------------------------
    Xg, kg, _, _, _ = cd.load_gwps()
    A0 = S["A"]
    keepA = [i for i, k in enumerate(A0["kos"]) if not k.lower().startswith("non")]
    same = kg == [A0["kos"][i] for i in keepA] and Xg.shape == (len(keepA), A0["X"].shape[1])
    if same:
        Xm = A0["X"][keepA]
        same = bool(np.array_equal(np.isnan(Xg), np.isnan(Xm))) and bool(np.allclose(np.nan_to_num(Xg), np.nan_to_num(Xm)))
    keep_set = set(keepA)
    extra = [A0["kos"][i] for i in range(len(A0["kos"])) if i not in keep_set]
    del Xg

    ens2sym, prev, groups = hgnc_tables()
    canon = make_canon(ens2sym, prev, set(groups))
    for k in "ABC":
        s = S[k]
        s["kos_raw"] = list(s["kos"])
        canonicalise(s, canon)
        s["gi"] = first_index(s["genes"]); s["ki"] = {p: i for i, p in enumerate(s["kos"])}
        s["sd"] = noise_sd(s["Xc"]); s["Z"] = to_z(s["X"], s["sd"]); s["Zc"] = to_z(s["Xc"], s["sd"])
        del s["X"], s["Xc"]
        P_(f"  screen {k}: {len(s['kos']):,} knockdowns x {len(s['genes']):,} genes; control rows {len(s['Zc'])}; median noise sd "
           f"{np.median(s['sd']):.3f}; renamed to current symbols: {s['renamed_kos']} knockdowns, {s['renamed_genes']} genes; "
           f"duplicates dropped: {s['dup_kos']} knockdowns, {s['dup_genes']} gene columns")
    own = {}
    for k in "ABC":
        s = S[k]
        own[k] = float(np.median([s["Z"][s["ki"][p], s["gi"][p]] for p in s["kos"] if p in s["gi"]]))
    tss = rc.tss_table(); gpos = tss_positions(tss)
    A, B, C = S["A"], S["B"], S["C"]
    shared_all = sorted(set(A["kos"]) & set(B["kos"]))
    no_tss_k = [p for p in shared_all if not gpos.get(p)]
    no_tss_g = sorted(g for g in set(A["gi"]) & set(B["gi"]) if not gpos.get(g))
    h0 = same and all(v <= -3 for v in own.values())
    P_(f"  H0 (a) loader identical to celldiscover.load_gwps (apart from {extra}): {same}; (b) own-gene median Z "
       + ", ".join(f"{k} {v:+.1f}" for k, v in own.items()) + f"; (c) without a TSS after name mapping: "
       f"{len(no_tss_k)} shared knockdowns {no_tss_k[:8]}, {len(no_tss_g)} measured genes (dropped) -> {'PASS' if h0 else 'FAIL'}")
    res["H0"] = dict(identical=same, extra_kept=extra, own_median_z=own, no_tss_knockdowns=no_tss_k,
                     no_tss_genes=len(no_tss_g), passed=bool(h0))
    if not h0:
        P_("  H0 FAILED: nothing below is computed."); finish(); return

    # ---- H1: the CRISPRi neighbour artefact by distance, before cis zeroing ------------------------
    chroms = sorted({c for v in gpos.values() for c, _ in v}); code = {c: k for k, c in enumerate(chroms)}
    gA = A["genes"]
    chrom_of = np.array([code[gpos[g][0][0]] if gpos.get(g) else -1 for g in gA])
    cnt = collections.defaultdict(lambda: [0, 0])
    for r, p in enumerate(A["kos"]):
        if not gpos.get(p):
            continue
        row = A["Z"][r]
        used = set()
        for lo_bp, hi_bp in BINS:
            cols = [A["gi"][g] for g in zone(p, hi_bp, tss, gpos, lo_bp=lo_bp) if g in A["gi"] and g != p]
            cols = [c for c in cols if c not in used]; used.update(cols)
            cnt[(lo_bp, hi_bp)][0] += int((row[cols] <= -3).sum()); cnt[(lo_bp, hi_bp)][1] += len(cols)
        pch = [code[c] for c, _ in gpos[p]]
        far_same = [A["gi"][g] for g in zone(p, 10**9, tss, gpos, lo_bp=1_000_000) if g in A["gi"]]
        cnt["same chr > 1 Mb"][0] += int((row[far_same] <= -3).sum()); cnt["same chr > 1 Mb"][1] += len(far_same)
        other = (chrom_of >= 0) & ~np.isin(chrom_of, pch)
        cnt["other chr"][0] += int((row[other] <= -3).sum()); cnt["other chr"][1] += int(other.sum())
    od, on = cnt["other chr"]
    nd = cnt[(0, 1_000)][0] + cnt[(1_000, 10_000)][0]; nn = cnt[(0, 1_000)][1] + cnt[(1_000, 10_000)][1]
    orr, p1 = fisher_exact([[nd, nn - nd], [od, on - od]])
    h1 = bool(orr >= 2 and p1 < 1e-3)
    P_("  H1 CRISPRi neighbour artefact in A, P(Z <= -3) by distance from the target's TSS:")
    lab = lambda b: b if isinstance(b, str) else f"{b[0] // 1000}-{b[1] // 1000} kb"
    P_("     " + "   ".join(f"{lab(b)} {d / max(n, 1):.4f} ({n:,})" for b, (d, n) in cnt.items()))
    P_(f"     < 10 kb vs other chromosomes: OR {orr:.1f}, p {p1:.2g} -> {'PASS' if h1 else 'FAIL'}")
    res["H1"] = dict(bins={lab(b): v for b, v in cnt.items()}, odds_ratio=float(orr), p=float(p1), passed=h1)

    cis = {k: zero_cis(S[k]["Z"], S[k]["kos"], S[k]["gi"], tss, gpos) for k in "ABC"}
    P_(f"  cis zones zeroed: median {np.median([len(v) for v in cis['A'].values()]):.0f} genes per knockdown in A "
       f"(self + TSS within {CIS // 1000} kb)")

    # ---- H2: known regulation --------------------------------------------------------------------
    P_(); P_("  H2 KNOWN REGULATION (raw Z, expected DOWN; z vs the same set score over control rows)")

    def set_z(reg, targets, k):
        s = S[k]
        if reg not in s["ki"]:
            return None
        zc = zone(reg, CIS, tss, gpos)
        meas = [t for t in targets if t in s["gi"]]
        cols = [s["gi"][t] for t in meas if t not in zc]
        if len(cols) < 5:
            return dict(n=len(cols), missing=[t for t in targets if t not in s["gi"]])
        sc = float(s["Z"][s["ki"][reg], cols].mean()); ctl = s["Zc"][:, cols].mean(1)
        return dict(n=len(cols), z=float((sc - ctl.mean()) / ctl.std(ddof=1)),
                    missing=[t for t in targets if t not in s["gi"]])

    fam_rows, h2rows = [], []
    for fam, (regs, tg) in FAMILIES.items():
        for k in "ABC":
            ms = []
            for reg in regs:
                r_ = set_z(reg, tg, k)
                if r_ is None:
                    continue
                if "z" not in r_:
                    P_(f"     {reg:7s} {k}  only {r_['n']} targets measured outside the cis zone -- not evaluable; missing {r_['missing']}")
                    continue
                ok = r_["z"] <= -3
                ms.append(ok)
                h2rows.append(dict(family=fam, regulator=reg, screen=k, targets=r_["n"], z=r_["z"], passed=bool(ok)))
                P_(f"     {reg:7s} {k}  {r_['n']:2d} targets  z {r_['z']:+7.1f}  {'pass' if ok else 'fail'}"
                   + (f"   (not measured: {', '.join(r_['missing'])})" if r_["missing"] else ""))
            if ms:
                fam_rows.append((fam, k, sum(ms) > len(ms) / 2))
    for fam, (regs, tg, expect) in REPORTED.items():
        for k in "ABC":
            for reg in regs:
                r_ = set_z(reg, tg, k)
                if r_ is None or "z" not in r_:
                    continue
                ok = abs(r_["z"]) < 3 if expect == "none" else r_["z"] <= -3
                h2rows.append(dict(family=fam, regulator=reg, screen=k, targets=r_["n"], z=r_["z"], passed=bool(ok), counted=False))
                P_(f"     {reg:7s} {k}  {r_['n']:2d} targets  z {r_['z']:+7.1f}  {'pass' if ok else 'fail'}   (reported, not counted: {fam})")
    passed = sum(1 for *_, ok in fam_rows if ok); evaluable = len(fam_rows)
    h2 = evaluable > 0 and passed >= math.ceil(2 * evaluable / 3)
    P_(f"  H2 {passed}/{evaluable} (family, screen) pairs pass: " + ", ".join(f"{f}/{k} {'+' if ok else '-'}" for f, k, ok in fam_rows)
       + f" -> {'PASS' if h2 else 'FAIL'}")
    NI = "" if h2 else "   [NOT INTERPRETABLE: H2 FAILED]"
    res["H2"] = dict(rows=h2rows, families=[list(x) for x in fam_rows], passed_n=passed, evaluable=evaluable, passed=bool(h2))
    res["interpretable"] = bool(h2)

    # ---- shared subspace, learnt from A only, proteins not in B ----------------------------------
    shared = [p for p in shared_all if gpos.get(p)]
    g2 = sorted(g for g in set(A["gi"]) & set(B["gi"]) if gpos.get(g))
    cA = [A["gi"][g] for g in g2]; cB = [B["gi"][g] for g in g2]
    pos2A = {j: k for k, j in enumerate(cA)}; pos2B = {j: k for k, j in enumerate(cB)}
    learn = [A["ki"][p] for p, e in zip(A["kos"], A["ep"]) if p not in B["ki"] and np.isfinite(e) and e < 0.05]
    Zl = A["Z"][learn][:, cA].astype(np.float64)
    tide = Zl.mean(0); Cl = Zl - tide
    _, sv, Vt = np.linalg.svd(Cl, full_matrices=False)
    Q, _ = np.linalg.qr(np.vstack([tide, Vt[:K_AX]]).T)
    var_ax = float((sv[:K_AX] ** 2).sum() / (sv ** 2).sum())
    del Zl, Cl
    n = len(shared)
    rowA = [A["ki"][p] for p in shared]; rowB = [B["ki"][p] for p in shared]
    ZA2 = A["Z"][rowA][:, cA]; ZB2 = B["Z"][rowB][:, cB]

    def mask(rows, cism, pos2, ncol):
        m = np.zeros((len(rows), ncol), bool)
        for i, r_ in enumerate(rows):
            cc = [pos2[c] for c in cism[r_] if c in pos2]
            m[i, cc] = True
        return m

    mA = mask(rowA, cis["A"], pos2A, len(g2)); mB = mask(rowB, cis["B"], pos2B, len(g2))
    RA2 = resid_robust(ZA2, Q, mA); RB2 = resid_robust(ZB2, Q, mB)
    tot = lambda M: float((M.astype(np.float64) ** 2).sum())
    P_(); P_(f"  SHARED SUBSPACE from {len(learn):,} screen-A knockdowns not in B (phenotype), {len(g2):,} genes: "
         f"top {K_AX} axes = {var_ax:.1%} of their centred variance")
    P_(f"  shared proteins {n:,} (with a TSS): share of their energy in the tide+axes span  A {1 - tot(RA2) / tot(ZA2):.1%}  "
       f"B {1 - tot(RB2) / tot(ZB2):.1%}")
    pheno2 = sum(1 for p in shared if np.isfinite(A["ep"][A["ki"][p]]) and A["ep"][A["ki"][p]] < 0.05 and
                 np.isfinite(B["ep"][B["ki"][p]]) and B["ep"][B["ki"][p]] < 0.05)
    P_(f"  shared proteins with an expression phenotype in both screens: {pheno2:,} of {n:,}")
    res["subspace"] = dict(n_learn=len(learn), genes=len(g2), var_axes=var_ax, energy_shared_A=1 - tot(RA2) / tot(ZA2),
                           energy_shared_B=1 - tot(RB2) / tot(ZB2), shared=n, pheno_both=pheno2)
    ders = [derangement(n, rng) for _ in range(200)]

    # ---- R1 / R2 ---------------------------------------------------------------------------------
    P_(); P_("  R1/R2 DO EFFECT PROFILES REPEAT ACROSS EXPERIMENTS? (per protein, A vs B)")
    grp_size = collections.Counter(gname for p in shared for gname in groups.get(p, ()))
    sig_spec = np.zeros(n, bool)
    for tag, MA, MB in (("raw", ZA2, ZB2), ("specific", RA2, RB2)):
        An, Bn = rownorm(MA.astype(np.float64)), rownorm(MB.astype(np.float64))
        Rall = An @ Bn.T
        r = np.diag(Rall).copy()
        med_null = np.array([np.median(Rall[np.arange(n), d]) for d in ders])
        rep = bool(np.median(r) > np.percentile(med_null, 95))
        pv = (Rall >= r[:, None]).sum(1) / n
        q = bh(pv); sig = q < 0.05
        in_grp = int(sum(1 for i in np.nonzero(sig)[0] if any(grp_size[gname] >= 3 for gname in groups.get(shared[i], ()))))
        P_(f"     {tag:8s} median r {np.median(r):+.3f} (IQR {np.percentile(r, 25):+.3f} to {np.percentile(r, 75):+.3f}); deranged "
           f"medians 95th pct {np.percentile(med_null, 95):+.3f} -> {'REPEATS' if rep else 'DOES NOT REPEAT'}{NI}")
        P_(f"              proteins individually reproducible (exact p over all partners, BH < 0.05): {int(sig.sum()):,} of {n:,}; "
           f"their median r {np.median(r[sig]) if sig.any() else float('nan'):+.3f}; in an HGNC group with >= 2 other tested proteins: {in_grp}")
        res[f"R_{tag}"] = dict(median_r=float(np.median(r)), null95=float(np.percentile(med_null, 95)), repeats=rep,
                               n_sig=int(sig.sum()), median_r_sig=float(np.median(r[sig])) if sig.any() else None,
                               sig_in_group=in_grp, interpretable=bool(h2))
        if tag == "specific":
            sig_spec = sig
        del Rall

    # ---- R3: gene-level catalogues ---------------------------------------------------------------
    P_(); P_(f"  R3 GENE-LEVEL CATALOGUE: |Z| >= t in A and B, same sign; sign-flip FDR; ladder {LADDER}; >= {MIN_N} effects")
    agreeS = ((np.sign(RA2) == np.sign(ZA2)), (np.sign(RB2) == np.sign(ZB2)))
    cats = {}
    for tag, MA, MB, ag in (("ALL", ZA2, ZB2, None), ("SPECIFIC", RA2, RB2, agreeS)):
        agm = None if ag is None else ag[0] & ag[1]
        t, table = ladder(LADDER, lambda t: cohits(MA, MB, t, agree=agm),
                          lambda t, d: cohits(MA, MB, t, perm=ders[d], agree=ag)[0])
        for tt, sm, op, f, dr in table:
            P_(f"     {tag:8s} t {tt:.0f}: reproduced {sm:,}, opposite-sign {op:,}, FDR {f if f is not None else float('nan'):.3f}; "
               f"mismatched-partner ratio {dr if dr is not None else float('nan'):.3f}")
        P_(f"     {tag:8s} -> " + (f"CATALOGUE AT t = {t:.0f}" if t else "NO RELIABLE GENE-LEVEL CATALOGUE") + NI)
        cats[tag] = t
        res[f"R3_{tag}"] = dict(t=t, table=[list(x) for x in table], interpretable=bool(h2))

    entries = {}
    for tag, MA, MB in (("ALL", ZA2, ZB2), ("SPECIFIC", RA2, RB2)):
        t = cats[tag]
        if t is None:
            continue
        hit = (np.abs(MA) >= t) & (np.abs(MB) >= t) & (np.sign(MA) == np.sign(MB))
        if tag == "SPECIFIC":
            hit &= agreeS[0] & agreeS[1]
        ii, jj = np.nonzero(hit)
        for i, j in zip(ii, jj):
            if g2[j] == shared[i]:
                continue
            e = entries.setdefault((shared[i], g2[j]), dict(protein=shared[i], gene=g2[j], i=int(i), j=int(j), classes=set()))
            e["classes"].add(tag)
    for e in entries.values():
        i, j = e["i"], e["j"]
        e.update(zA=float(ZA2[i, j]), zB=float(ZB2[i, j]), sA=float(RA2[i, j]), sB=float(RB2[i, j]))
        e["paralog_flag"] = paralog(e["protein"], e["gene"], groups) and (e["zA"] < 0 or e["sA"] < 0)
        e["readout_flag"] = readout_flag(e["gene"])

    # ---- R4: another cell type -------------------------------------------------------------------
    P_(); P_("  R4 DO REPRODUCED K562 EFFECTS HOLD IN RPE1? (among |Z_C| >= 3: same sign as K562?)")
    at2 = {g: k for k, g in enumerate(g2)}
    g3 = [g for g in g2 if g in C["gi"]]; j3 = {g: k for k, g in enumerate(g3)}
    Q3, _ = np.linalg.qr(Q[[at2[g] for g in g3]])
    sh3 = [p for p in shared if p in C["ki"]]; i3 = {p: k for k, p in enumerate(sh3)}
    rowC = [C["ki"][p] for p in sh3]; cC = [C["gi"][g] for g in g3]; pos3C = {j: k for k, j in enumerate(cC)}
    ZC3 = C["Z"][rowC][:, cC]
    RC3 = resid_robust(ZC3, Q3, mask(rowC, cis["C"], pos3C, len(g3)))
    ders3 = [derangement(len(sh3), rng) for _ in range(N_DER)]
    for e in entries.values():
        if e["protein"] in i3 and e["gene"] in j3:
            a, b = i3[e["protein"]], j3[e["gene"]]
            e["zC"], e["sC"] = float(ZC3[a, b]), float(RC3[a, b])
    for tag, MC, key, kref in (("ALL", ZC3, "zC", "zA"), ("SPECIFIC", RC3, "sC", "sA")):
        ev = [e for e in entries.values() if tag in e["classes"] and key in e]
        if not ev:
            P_(f"     {tag:8s} no evaluable effects"); continue
        a = np.array([i3[e["protein"]] for e in ev]); b = np.array([j3[e["gene"]] for e in ev])
        sgn = np.sign([e[kref] for e in ev])

        def agree_share(rows):
            v = MC[rows, b]; hit = np.abs(v) >= ZC_T
            return int((hit & (np.sign(v) == sgn)).sum()), int(hit.sum())

        k_, m_ = agree_share(a)
        nul = [agree_share(d[a]) for d in ders3]
        nul_sh = [x / y if y else 0.0 for x, y in nul]
        share = k_ / m_ if m_ else 0.0
        pb = binom_two_sided(k_, m_)
        gen = bool(m_ > 0 and share > max(nul_sh) and pb < 0.05 and share > 0.5)
        cov = m_ / len(ev); cov0 = float(np.mean([y for _, y in nul])) / len(ev)
        P_(f"     {tag:8s} {len(ev):,} evaluable; reach |Z_C| >= 3: {m_:,} ({cov:.1%}; deranged {cov0:.1%}, description only); "
           f"same sign {share:.1%} (binomial p {pb:.2g}) vs deranged {np.mean(nul_sh):.1%} (max {max(nul_sh):.1%}) -> "
           f"{'GENERAL BEYOND CHANCE' if gen else 'NOT BEYOND CHANCE'}{NI}")
        res[f"R4_{tag}"] = dict(n=len(ev), reach=m_, same_sign=k_, share=share, binom_p=pb, null_mean=float(np.mean(nul_sh)),
                                null_max=float(max(nul_sh)), coverage=cov, coverage_null=cov0, general=gen, interpretable=bool(h2))

    # ---- R5: protein -> pathway ------------------------------------------------------------------
    P_(); P_(f"  R5 PROTEIN -> PATHWAY CATALOGUE (specific Z averaged over each process; z vs control rows; ladder {LADDER_P})")
    mods, pnames, ptop = wc.reactome(set(g2)); mid = sorted(mods)
    M = np.zeros((len(g2), len(mid)), np.float32)
    for c_, pid in enumerate(mid):
        for g in mods[pid]:
            M[at2[g], c_] = 1
    base = M.sum(0)

    def pscore(Rm, msk):
        return (Rm @ M) / np.maximum(base[None, :] - (msk.astype(np.float32) @ M), 1)

    PA = pscore(RA2, mA); PB = pscore(RB2, mB)
    for Pm, s, cc in ((PA, A, cA), (PB, B, cB)):
        Zc = s["Zc"][:, cc]
        Rc = resid_robust(Zc, Q, np.zeros(Zc.shape, bool))
        ctl = (Rc @ M) / np.maximum(base, 1)
        Pm -= ctl.mean(0); Pm /= np.maximum(ctl.std(0, ddof=1), 1e-9)
    tP, tableP = ladder(LADDER_P, lambda t: cohits(PA, PB, t), lambda t, d: cohits(PA, PB, t, perm=ders[d])[0])
    for tt, sm, op, f, dr in tableP:
        P_(f"     t {tt:.0f}: reproduced {sm:,}, opposite-sign {op:,}, FDR {f if f is not None else float('nan'):.3f}; "
           f"mismatched-partner ratio {dr if dr is not None else float('nan'):.3f}")
    P_("     -> " + (f"CATALOGUE AT t = {tP:.0f}" if tP else "NO RELIABLE PATHWAY CATALOGUE") + NI)
    res["R5"] = dict(t=tP, table=[list(x) for x in tableP], n_pathways=len(mid), interpretable=bool(h2))
    pw = []
    if tP:
        ii, jj = np.nonzero((np.abs(PA) >= tP) & (np.abs(PB) >= tP) & (np.sign(PA) == np.sign(PB)))
        pw = [dict(protein=shared[i], pathway=mid[j], name=pnames[mid[j]], top="; ".join(ptop[mid[j]]),
                   zA=float(PA[i, j]), zB=float(PB[i, j])) for i, j in zip(ii, jj)]

    # ---- R6: shape -------------------------------------------------------------------------------
    P_(); P_("  R6 THE NETWORK'S SHAPE" + NI)
    spec = [e for e in entries.values() if "SPECIFIC" in e["classes"]]
    allc = [e for e in entries.values() if "ALL" in e["classes"]]
    for tag, ev, key in (("ALL", allc, "zA"), ("SPECIFIC", spec, "sA")):
        if not ev:
            continue
        outd = collections.Counter(e["protein"] for e in ev)
        ind = collections.Counter(e["gene"] for e in ev if not e["readout_flag"])
        up = sum(e[key] > 0 for e in ev)
        P_(f"     {tag:8s} {len(ev):,} effects from {len(outd):,} proteins onto {len({e['gene'] for e in ev}):,} genes; up {up:,} / "
           f"down {len(ev) - up:,}; paralog-flagged {sum(e['paralog_flag'] for e in ev):,}; MT-/histone readouts "
           f"{sum(bool(e['readout_flag']) for e in ev):,}")
        P_("        most effects: " + ", ".join(f"{p} {c}" for p, c in outd.most_common(12)))
        P_("        most often hit (flagged readouts left out): " + ", ".join(f"{g} {c}" for g, c in ind.most_common(12)))
        res[f"R6_{tag}"] = dict(n=len(ev), proteins=len(outd), up=up, top_proteins=outd.most_common(30),
                                top_genes=ind.most_common(30))
    if spec:
        top = sorted([e for e in spec if not e["paralog_flag"] and not e["readout_flag"]],
                     key=lambda e: -min(abs(e["sA"]), abs(e["sB"])))[:40]
        P_("     strongest reproduced SPECIFIC protein -> gene effects (no flags):")
        for e in top[:25]:
            rp = f"{e['sC']:+6.1f}" if "sC" in e else "   n/a"
            P_(f"        {e['protein']:>10s} -> {e['gene']:<10s} {'UP  ' if e['sA'] > 0 else 'DOWN'}  specific z A {e['sA']:+6.1f}  "
               f"B {e['sB']:+6.1f}  RPE1 {rp}")
        res["top_specific_gene_effects"] = [{k: v for k, v in e.items() if k not in ("i", "j", "classes")} for e in top]
    if pw:
        seen, toppw = [], []
        for e in sorted(pw, key=lambda e: -min(abs(e["zA"]), abs(e["zB"]))):
            gs = mods[e["pathway"]]
            if any(e["protein"] == q and len(gs & mods[w]) / len(gs | mods[w]) > 0.5 for q, w in seen):
                continue
            seen.append((e["protein"], e["pathway"])); toppw.append(e)
            if len(toppw) >= 40:
                break
        P_("     strongest reproduced protein -> pathway effects (non-redundant per protein):")
        for e in toppw[:25]:
            P_(f"        {e['protein']:>10s} -> {e['name'][:58]:<58s} {'UP  ' if e['zA'] > 0 else 'DOWN'} z A {e['zA']:+6.1f} B {e['zB']:+6.1f}")
        res["top_pathway_effects"] = toppw
        po = collections.Counter(e["protein"] for e in pw)
        P_(f"     {len(pw):,} reproduced pathway effects from {len(po):,} proteins; most: "
           + ", ".join(f"{p} {c}" for p, c in po.most_common(10)))

    # ---- write ----------------------------------------------------------------------------------
    dest = BANK if h2 else R("outputs")
    pre = "" if h2 else "NOT INTERPRETABLE (H2 FAILED: the instrument did not detect known regulation). "
    os.makedirs(dest, exist_ok=True)
    fg = os.path.join(dest, "protein_gene_effects_reproduced.tsv.gz")
    with gzip.open(fg, "wt") as fh:
        fh.write(f"# {pre}{CITE}. Reproduced in two K562 Perturb-seq experiments (same guides, new cells and day), "
                 f"rem/atlas/proteffect.py. z = noise-standardised; s = specific (beyond tide + {K_AX} shared axes). "
                 f"Read protein_effects_CAVEATS.json.\n")
        fh.write("protein\tgene\tclass\tdirection_all\tdirection_specific\tzA_gw\tzB_ess\tzC_rpe1\tsA_gw\tsB_ess\tsC_rpe1\t"
                 "paralog_flag\treadout_flag\n")
        for e in sorted(entries.values(), key=lambda e: (e["protein"], e["gene"])):
            fh.write("\t".join([e["protein"], e["gene"], "+".join(sorted(e["classes"])),
                                ("up" if e["zA"] > 0 else "down") if "ALL" in e["classes"] else "",
                                ("up" if e["sA"] > 0 else "down") if "SPECIFIC" in e["classes"] else "",
                                f"{e['zA']:.2f}", f"{e['zB']:.2f}", f"{e['zC']:.2f}" if "zC" in e else "",
                                f"{e['sA']:.2f}", f"{e['sB']:.2f}", f"{e['sC']:.2f}" if "sC" in e else "",
                                "possible" if e["paralog_flag"] else "", e["readout_flag"]]) + "\n")
    fp = os.path.join(dest, "protein_pathway_effects_reproduced.tsv.gz")
    with gzip.open(fp, "wt") as fh:
        fh.write(f"# {pre}{CITE}; Reactome (CC0). Specific effects reproduced in two K562 experiments.\n")
        fh.write("protein\tpathway\tname\ttop_level\tdirection\tzA_gw\tzB_ess\n")
        for e in sorted(pw, key=lambda e: (e["protein"], e["pathway"])):
            fh.write("\t".join([e["protein"], e["pathway"], e["name"], e["top"], "up" if e["zA"] > 0 else "down",
                                f"{e['zA']:.2f}", f"{e['zB']:.2f}"]) + "\n")
    cav = dict(applies_to=[os.path.basename(fg), os.path.basename(fp)], source="rem/atlas/proteffect.py", data=CITE,
               interpretable=bool(h2),
               caveats=["Same sgRNA library elements in all three screens: reproduction is across cells, days and sequencing, "
                        "not across reagents; guide off-target effects would reproduce.",
                        "Steady state (day 6 and day 8): direct and indirect effects are mixed.",
                        f"SPECIFIC = beyond the tide and {K_AX} cell-wide axes learnt from other knockdowns. It still contains "
                        "programmes shared by members of a complex or pathway, and removes the defining effects of regulators "
                        "that drive a shared axis.",
                        "Cis zone: self + genes with a TSS within 100 kb were zeroed; trans effects of a co-silenced promoter "
                        "partner (e.g. head-to-head genes) are attributed to the targeted protein.",
                        "paralog_flag is a heuristic (shared HGNC family + symbol prefix), not sequence paralogy.",
                        "FDR is a sign-flip estimate (opposite-sign co-hits / same-sign co-hits)."],
               gates=dict(H0=res["H0"]["passed"], H1=res["H1"]["passed"], H2=res["H2"]["passed"]))
    json.dump(cav, open(os.path.join(dest, "protein_effects_CAVEATS.json"), "w"), indent=1)
    res["files"] = [os.path.relpath(x, R()) for x in (fg, fp, os.path.join(dest, "protein_effects_CAVEATS.json"))]
    res["runtime_s"] = time.time() - t0
    P_(); P_(f"  written: {', '.join(res['files'])}; artifact outputs/proteffect.json; runtime {time.time() - t0:.0f}s")
    P_(f"  data: {CITE}")
    finish()


if __name__ == "__main__":
    main()
