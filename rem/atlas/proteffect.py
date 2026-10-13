"""The effect of each protein on every other gene and pathway: what is real, what is shared, and
what belongs to the protein itself.

WHY THIS IS NEW. effectmap.py measured protein -> pathway effects in ONE experiment and found them
dominated by a few cell-wide axes; its only replicate was 102 genes knocked down at two promoters.
It never asked which effects on individual GENES are real. Replogle et al. 2022 ran three separate
Perturb-seq experiments with the same normalisation (figshare 20029387, CC BY 4.0):
   A  K562 genome-wide   9,865 knockdowns x 8,248 genes   day 8
   B  K562 essential     2,057 knockdowns x 8,563 genes   day 6   (separate library, cells, sequencing)
   C  RPE1 essential     2,393 knockdowns x 8,749 genes   day 7   (a different, non-cancer cell line)
2,053 proteins were knocked down in both A and B, 2,052 in all three. An effect seen in A AND in B
survived an independent experiment; one also seen in C holds in another cell type. (Before this was
written only shapes and overlaps were inspected; no effect value was looked at.)

UNITS AND NOISE. Every value is the normalised mean expression change of a gene in cells carrying the
knockdown. Each screen's non-targeting control guides (A 585, B 109, C 130 rows) give each gene's
noise: sd_g = 1.4826 x MAD over control rows, floored at the 10th percentile across genes.
Z = value / sd_g; non-finite -> 0 (no effect). The control rows include the core controls that
defined the normalisation, which slightly understates noise; the derangement nulls do not depend
on it.

CRISPRi ARTEFACTS, REMOVED OR FLAGGED BEFORE ANY ANALYSIS.
  SELF and CIS: the knocked-down gene and every gene with a refGene hg38 TSS within 100 kb of one of
    the target's TSSs are set to 0 in that knockdown's row (dCas9-KRAB silences neighbouring
    promoters -- a property of the tool, not of the protein).
  PARALOG FLAG: a DOWN effect on a gene in the same HGNC gene group as the target is flagged
    POSSIBLE GUIDE CROSS-REACTIVITY (kept, not removed; UP effects there are compensation candidates).

SHARED vs SPECIFIC. A knockdown's effect = the tide (what every knockdown does) + a few cell-wide
axes + what is specific to the protein. The shared subspace is learnt ONLY from screen-A knockdowns
of proteins NOT knocked down in B (so never from a protein tested below) that have an expression
phenotype (energy test p < 0.05): their mean (the tide) and their top 5 principal axes (effectmap:
5 axes = 83% of specific pathway variance). SPECIFIC effect = the component orthogonal to that
6-dimensional span. The same projection is applied to every row of every screen, control rows
included; for comparisons involving C it is re-orthonormalised on the genes all three measure.

NULL. DERANGEMENT: pair protein p's row in A with ANOTHER protein's row in B (or C). It keeps every
row's distribution, sparsity and shared programmes, and breaks only identity. Cis zones of both rows
are zeroed.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
H0 HARNESS (blocking). (a) For A, the loader returns exactly celldiscover.load_gwps's matrix and
   knockdown list. (b) In each screen the knocked-down gene's own Z has median <= -3.
H1 THE INSTRUMENT SEES THE KNOWN CRISPRi ARTEFACT (reported, not blocking). In A, before cis zeroing:
   P(Z <= -3) for genes with a TSS < 10 kb from the target's (self excluded) vs genes on other
   chromosomes. PASS if odds ratio >= 2 and Fisher p < 0.001.
H2 THE INSTRUMENT SEES KNOWN REGULATION (blocking for interpretation). Literature regulator -> target
   sets, expected DOWN on knockdown; raw Z (not specific), cis-zone targets excluded:
     GATA1 -> erythroid genes          NFE2L1 -> proteasome subunits
     SREBF2, SCAP, MBTPS1, MBTPS2 -> sterol synthesis and uptake
     ATF4 -> amino-acid-response genes  HSF1 -> heat-shock genes
     MYC -> ribosome-biogenesis / nucleolar genes   TFAM -> mitochondrially encoded genes
     NRF1 -> TFAM, TFB2M, POLRMT, COX6C, CYCS
   Set score = mean Z over measured targets (>= 5 needed); z = (score - mean) / sd of the same score
   over the screen's control rows. A (regulator, screen) pair PASSES at z <= -3. Gate: >= 2/3 of
   evaluable pairs pass. NEGATIVE CONTROL: TP53 knockdown in K562, a TP53-null line -> p53 targets;
   PASS if |z| < 3. Reported.
R1 DO WHOLE EFFECT PROFILES REPEAT IN AN INDEPENDENT EXPERIMENT? Per shared protein, Pearson r of its
   A and B rows (raw Z); the median vs the medians of 200 derangements: REPEATS if above their 95th
   percentile.
R2 DO THE PROTEIN-SPECIFIC PARTS REPEAT? Same on specific rows. Per protein, empirical p of its r
   against its 200 deranged partners; Benjamini-Hochberg FDR < 0.05 -> the protein has a
   REPRODUCIBLE SPECIFIC EFFECT. Reported: how many, and their median r.
R3 GENE-LEVEL CATALOGUE. p -> g is REPRODUCED if |Z| >= t in A and in B with the same sign. FDR =
   mean count over 20 derangements / real count. t from the ladder 3, 4, 5: the first t with FDR <=
   0.10 is used; if none, NO RELIABLE GENE-LEVEL CATALOGUE. Done on raw Z (ALL effects) and on
   specific Z (SPECIFIC effects).
R4 DO THEY HOLD IN ANOTHER CELL TYPE? For reproduced K562 effects with p knocked down and g measured
   in RPE1: share with |Z_C| >= 3 and the same sign (raw for ALL, specific for SPECIFIC), vs the
   share when C rows are deranged (20 times). GENERAL BEYOND CHANCE if above all 20.
R5 PROTEIN -> PATHWAY CATALOGUE. Specific Z averaged over each Reactome process's genes (15-300
   measured genes; cis genes left out of that knockdown's average); z against the same score over
   residualised control rows. REPRODUCED if |z| >= t in A and B, same sign; ladder and FDR rule as R3.
R6 THE NETWORK'S SHAPE (descriptive): effects per protein and per gene; proteins with the most
   reproduced specific effects; genes most often hit specifically; share carrying the paralog flag.
X  WHAT THIS IS NOT. Steady state (day 6-8): direct and indirect effects are mixed, and a reproduced
   effect is an effect of the KNOCKDOWN -- usually of losing the protein, but CRISPRi at a paralog
   promoter cannot be excluded without guide sequences (flagged). Two K562 time points (day 6 and
   8): agreement is a lower bound on reproducibility. A pathway effect is an average over its genes.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import collections
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
LADDER, FDR_MAX, ZC_T = (3.0, 4.0, 5.0), 0.10, 3.0
N_DER, N_NULL, SEED = 20, 200, 20261013
SREBP = ["HMGCR", "HMGCS1", "LDLR", "INSIG1", "SQLE", "MVD", "FDFT1", "IDI1", "LSS", "DHCR7", "MVK", "FDPS",
         "SC5D", "CYP51A1", "MSMO1", "NSDHL", "HSD17B7", "EBP", "TM7SF2", "DHCR24", "STARD4", "PCSK9"]
CONTROLS = {
    "GATA1": ["HBG1", "HBG2", "HBE1", "HBZ", "ALAS2", "GYPA", "KLF1", "TAL1", "SLC4A1", "EPOR"],
    "NFE2L1": ["PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7", "PSMB1", "PSMB2", "PSMB3", "PSMB4",
               "PSMB5", "PSMB6", "PSMB7", "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6", "PSMD1", "PSMD2",
               "PSMD3", "PSMD4", "PSMD11", "PSMD12", "PSMD14"],
    "SREBF2": SREBP, "SCAP": SREBP, "MBTPS1": SREBP, "MBTPS2": SREBP,
    "ATF4": ["ASNS", "PSAT1", "PHGDH", "PSPH", "CHAC1", "DDIT3", "TRIB3", "SESN2", "SLC7A11", "VEGFA", "ATF3",
             "CEBPB", "SLC3A2", "SLC7A5", "MTHFD2", "ALDH18A1", "CARS1", "WARS1", "GARS1", "SARS1", "YARS1",
             "AARS1", "EIF4EBP1", "XPOT"],
    "HSF1": ["HSPA1A", "HSPA1B", "DNAJB1", "HSPH1", "HSPA8", "HSP90AA1", "HSPD1", "HSPE1", "BAG3", "HSPB1",
             "DNAJA1", "CHORDC1"],
    "MYC": ["NPM1", "NCL", "NOP56", "NOP58", "FBL", "DKC1", "NOP16", "RRP9", "RRS1", "PA2G4", "CAD", "ODC1", "LDHA",
            "SRM", "POLR1B", "BOP1", "PES1", "WDR12", "GNL3", "NOLC1"],
    "TFAM": ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP6", "MT-CYB", "MT-ND4", "MT-ND5", "MT-CO3", "MT-ND3"],
    "NRF1": ["TFAM", "TFB2M", "POLRMT", "COX6C", "CYCS"],
}
NEGATIVE = ("TP53", ["CDKN1A", "MDM2", "BAX", "BBC3", "GADD45A", "RRM2B", "SESN1", "TP53I3", "FAS", "ZMAT3",
                     "TNFRSF10B", "PLK3"])


# =================================================================================================
# LOADING, NOISE, ARTEFACT ZONES
# =================================================================================================
def load_screen(path):
    """One row per knocked-down gene (its strongest-knockdown row, as celldiscover.load_gwps), plus
    every non-targeting control row."""
    import h5py
    f = h5py.File(path, "r")
    X = f["X"][()]
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]
    fold = f["obs/fold_expr"][()]; core = f["obs/core_control"][()]; ep = f["obs/energy_test_p_value"][()]
    cats = [s.decode() for s in f["var/__categories/gene_name"][()]]
    genes = [cats[c] for c in f["var/gene_name"][()]]
    best, ctrl = {}, []
    for i, g in enumerate(gt):
        parts = g.split("_")
        sym = parts[1] if len(parts) > 1 else g
        if "non-targeting" in g.lower():
            ctrl.append(i)
            continue
        if core[i] or sym.lower().startswith("non"):
            continue
        fv = fold[i] if np.isfinite(fold[i]) else 1.0
        if sym not in best or fv < best[sym][1]:
            best[sym] = (i, fv)
    kos = sorted(best)
    rows = [best[k][0] for k in kos]
    Xs = X[rows].astype(np.float32); Xs[~np.isfinite(Xs)] = np.nan
    Xc = X[sorted(ctrl)].astype(np.float32); Xc[~np.isfinite(Xc)] = np.nan
    return dict(X=Xs, kos=kos, genes=genes, ep=np.array([ep[r] for r in rows]), Xc=Xc)


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


def zone(p, window, tss, gpos):
    """Genes with a TSS within `window` bp of one of p's TSSs (p itself included)."""
    out = {p}
    for c, x in gpos.get(p, ()):
        pos, names = tss[c]
        lo, hi = np.searchsorted(pos, x - window), np.searchsorted(pos, x + window, side="right")
        out.update(names[lo:hi])
    return out


def zero_cis(Z, kos, gi, tss, gpos):
    """Zero each row's self + cis zone in place; returns row -> list of zeroed columns."""
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


def ladder(real_fn, null_fn):
    """First t in LADDER whose derangement FDR <= FDR_MAX; returns (t or None, table)."""
    table = []
    for t in LADDER:
        n_real = real_fn(t); n_null = float(np.mean([null_fn(t, d) for d in range(N_DER)]))
        fdr = n_null / n_real if n_real else float("inf")
        table.append((t, n_real, n_null, fdr))
        if fdr <= FDR_MAX:
            return t, table
    return None, table


# =================================================================================================
def main():
    from scipy.stats import fisher_exact
    out, res = [], {}

    def P_(s=""):
        print(s, flush=True); out.append(s)

    def finish():
        open(OUT, "w").write("\n".join(out) + "\n")
        os.makedirs(os.path.dirname(ART), exist_ok=True)
        json.dump(res, open(ART, "w"), indent=1, default=float)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    P_(RULE); P_("THE EFFECT OF EACH PROTEIN ON EVERY OTHER GENE AND PATHWAY: REAL, SHARED, OR ITS OWN"); P_(RULE)
    S = {k: load_screen(v) for k, v in FILES.items()}
    for k in "ABC":
        s = S[k]
        s["gi"] = first_index(s["genes"]); s["ki"] = {p: i for i, p in enumerate(s["kos"])}
        s["sd"] = noise_sd(s["Xc"]); s["Z"] = to_z(s["X"], s["sd"]); s["Zc"] = to_z(s["Xc"], s["sd"])
        P_(f"  screen {k}: {len(s['kos']):,} knockdowns x {len(s['genes']):,} genes; control rows {len(s['Xc'])}; "
           f"median noise sd {np.median(s['sd']):.3f}")

    # ---- H0 --------------------------------------------------------------------------------------
    Xg, kg, _, _, _ = cd.load_gwps()
    same = kg == S["A"]["kos"] and Xg.shape == S["A"]["X"].shape and \
        bool(np.array_equal(np.isnan(Xg), np.isnan(S["A"]["X"]))) and \
        bool(np.allclose(np.nan_to_num(Xg), np.nan_to_num(S["A"]["X"])))
    del Xg
    own = {}
    for k in "ABC":
        s = S[k]
        own[k] = float(np.median([s["Z"][s["ki"][p], s["gi"][p]] for p in s["kos"] if p in s["gi"]]))
    h0 = same and all(v <= -3 for v in own.values())
    P_(f"  H0 (a) loader identical to celldiscover.load_gwps: {same}; (b) own-gene median Z "
       + ", ".join(f"{k} {v:+.1f}" for k, v in own.items()) + f" -> {'PASS' if h0 else 'FAIL'}")
    res["H0"] = dict(identical=same, own_median_z=own, passed=h0)
    if not h0:
        P_("  H0 FAILED: nothing below is computed."); finish(); return

    # ---- H1: the CRISPRi neighbour artefact, before cis zeroing ----------------------------------
    tss = rc.tss_table(); gpos = tss_positions(tss)
    A = S["A"]; gA = A["genes"]
    chroms = sorted({c for v in gpos.values() for c, _ in v}); code = {c: k for k, c in enumerate(chroms)}
    chrom_of = np.array([code[gpos[g][0][0]] if gpos.get(g) else -1 for g in gA])
    nn = nd = on = od = 0; no_tss = 0
    for r, p in enumerate(A["kos"]):
        if not gpos.get(p):
            no_tss += 1
            continue
        near = [A["gi"][g] for g in zone(p, NEAR, tss, gpos) if g in A["gi"] and g != p]
        pch = [code[c] for c, _ in gpos[p]]
        other = (chrom_of >= 0) & ~np.isin(chrom_of, pch)
        row = A["Z"][r]
        nn += len(near); nd += int((row[near] <= -3).sum())
        on += int(other.sum()); od += int((row[other] <= -3).sum())
    orr, p1 = fisher_exact([[nd, nn - nd], [od, on - od]])
    h1 = orr >= 2 and p1 < 1e-3
    P_(f"  H1 CRISPRi neighbour artefact: P(down) within 10 kb {nd}/{nn} = {nd / max(nn, 1):.4f} vs other chromosomes "
       f"{od:,}/{on:,} = {od / max(on, 1):.4f}; OR {orr:.1f}, p {p1:.2g} -> {'PASS' if h1 else 'FAIL'}")
    res["H1"] = dict(near=[nd, nn], other=[od, on], odds_ratio=orr, p=p1, passed=h1, knockdowns_without_tss=no_tss)
    P_(f"     {no_tss:,} of {len(A['kos']):,} screen-A knockdowns have no refGene TSS under their name: their cis zone is self only")

    cis = {k: zero_cis(S[k]["Z"], S[k]["kos"], S[k]["gi"], tss, gpos) for k in "ABC"}
    P_(f"  cis zones zeroed: median {np.median([len(v) for v in cis['A'].values()]):.0f} genes per knockdown in A "
       f"(self + TSS within {CIS // 1000} kb)")

    # ---- H2: known regulation --------------------------------------------------------------------
    P_(); P_("  H2 KNOWN REGULATION (raw Z, expected DOWN; z vs the same set score over control rows)")
    passed = evaluable = 0; h2rows = []
    for reg, tg in list(CONTROLS.items()) + [NEGATIVE]:
        for k in "ABC":
            s = S[k]
            if reg not in s["ki"]:
                continue
            zc = set(zone(reg, CIS, tss, gpos))
            cols = [s["gi"][t] for t in tg if t in s["gi"] and t not in zc]
            if len(cols) < 5:
                continue
            sc = float(s["Z"][s["ki"][reg], cols].mean()); ctl = s["Zc"][:, cols].mean(1)
            z = (sc - ctl.mean()) / ctl.std(ddof=1)
            neg = reg == NEGATIVE[0]
            ok = abs(z) < 3 if neg else z <= -3
            if not neg:
                evaluable += 1; passed += ok
            h2rows.append(dict(regulator=reg, screen=k, targets=len(cols), z=float(z), passed=bool(ok), negative=neg))
            P_(f"     {reg:7s} {k}  {len(cols):2d} targets  z {z:+7.1f}  {'PASS' if ok else 'FAIL'}"
               + ("   (negative control: TP53-null line)" if neg else ""))
    h2 = evaluable > 0 and passed >= math.ceil(2 * evaluable / 3)
    P_(f"  H2 {passed}/{evaluable} positive (regulator, screen) pairs pass -> {'PASS' if h2 else 'FAIL'}")
    res["H2"] = dict(rows=h2rows, passed_n=passed, evaluable=evaluable, passed=h2)

    # ---- shared subspace, learnt from A only, proteins not in B ----------------------------------
    B, C = S["B"], S["C"]
    shared = sorted(set(A["kos"]) & set(B["kos"]))
    g2 = sorted(set(A["gi"]) & set(B["gi"])); cA = [A["gi"][g] for g in g2]; cB = [B["gi"][g] for g in g2]
    learn = [A["ki"][p] for p, e in zip(A["kos"], A["ep"]) if p not in B["ki"] and np.isfinite(e) and e < 0.05]
    Zl = A["Z"][learn][:, cA].astype(np.float64)
    tide = Zl.mean(0); Cl = Zl - tide
    _, sv, Vt = np.linalg.svd(Cl, full_matrices=False)
    Q, _ = np.linalg.qr(np.vstack([tide, Vt[:K_AX]]).T)
    var_ax = float((sv[:K_AX] ** 2).sum() / (sv ** 2).sum())
    resid = lambda M, Q_: (M - (M @ Q_) @ Q_.T).astype(np.float32)
    ZA2 = A["Z"][[A["ki"][p] for p in shared]][:, cA]; ZB2 = B["Z"][[B["ki"][p] for p in shared]][:, cB]
    RA2, RB2 = resid(ZA2, Q), resid(ZB2, Q)
    tot = lambda M: float((M.astype(np.float64) ** 2).sum())
    P_(); P_(f"  SHARED SUBSPACE from {len(learn):,} screen-A knockdowns not in B (phenotype), {len(g2):,} genes: "
         f"top {K_AX} axes = {var_ax:.1%} of their centred variance")
    P_(f"  shared proteins {len(shared):,}: share of their energy in the tide+axes span  A {1 - tot(RA2) / tot(ZA2):.1%}  "
       f"B {1 - tot(RB2) / tot(ZB2):.1%}")
    res["subspace"] = dict(n_learn=len(learn), genes=len(g2), var_axes=var_ax,
                           energy_shared_A=1 - tot(RA2) / tot(ZA2), energy_shared_B=1 - tot(RB2) / tot(ZB2))
    n = len(shared)
    pheno2 = sum(1 for p in shared if np.isfinite(A["ep"][A["ki"][p]]) and A["ep"][A["ki"][p]] < 0.05 and
                 np.isfinite(B["ep"][B["ki"][p]]) and B["ep"][B["ki"][p]] < 0.05)
    P_(f"  shared proteins with an expression phenotype in both screens: {pheno2:,} of {n:,}")

    # cis zones on the paired matrices (rows of B deranged must keep their own zeroing: already zero)
    ders = [derangement(n, rng) for _ in range(N_NULL)]

    # ---- R1 / R2 ---------------------------------------------------------------------------------
    P_(); P_("  R1/R2 DO EFFECT PROFILES REPEAT IN AN INDEPENDENT EXPERIMENT? (per protein, A vs B)")
    for tag, MA, MB in (("raw", ZA2, ZB2), ("specific", RA2, RB2)):
        An, Bn = rownorm(MA.astype(np.float64)), rownorm(MB.astype(np.float64))
        r = (An * Bn).sum(1)
        nulls = np.stack([(An * Bn[d]).sum(1) for d in ders], 1)
        med_null = np.median(nulls, 0)
        rep = np.median(r) > np.percentile(med_null, 95)
        pv = (1 + (nulls >= r[:, None]).sum(1)) / (N_NULL + 1)
        q = bh(pv); sig = q < 0.05
        P_(f"     {tag:8s} median r {np.median(r):+.3f} (IQR {np.percentile(r, 25):+.3f} to {np.percentile(r, 75):+.3f}); "
           f"deranged medians 95th pct {np.percentile(med_null, 95):+.3f} -> {'REPEATS' if rep else 'DOES NOT REPEAT'}; "
           f"proteins individually reproducible (FDR < 0.05) {int(sig.sum()):,} of {n:,}, their median r "
           f"{np.median(r[sig]) if sig.any() else float('nan'):+.3f}")
        res[f"R_{tag}"] = dict(median_r=float(np.median(r)), null95=float(np.percentile(med_null, 95)), repeats=bool(rep),
                               n_sig=int(sig.sum()), median_r_sig=float(np.median(r[sig])) if sig.any() else None)

    # ---- R3: gene-level catalogues ---------------------------------------------------------------
    P_(); P_(f"  R3 GENE-LEVEL CATALOGUE: |Z| >= t in A and B, same sign; FDR from {N_DER} derangements; ladder {LADDER}")

    def count(MA, MB, t, perm=None):
        Mb = MB if perm is None else MB[perm]
        return int(((np.abs(MA) >= t) & (np.abs(Mb) >= t) & (np.sign(MA) == np.sign(Mb))).sum())

    cats = {}
    for tag, MA, MB in (("ALL", ZA2, ZB2), ("SPECIFIC", RA2, RB2)):
        t, table = ladder(lambda t: count(MA, MB, t), lambda t, d: count(MA, MB, t, ders[d]))
        for tt, nr, nl, f in table:
            P_(f"     {tag:8s} t {tt:.0f}: reproduced {nr:,}, deranged mean {nl:,.0f}, FDR {f:.3f}")
        P_(f"     {tag:8s} -> " + (f"CATALOGUE AT t = {t:.0f}" if t else "NO RELIABLE GENE-LEVEL CATALOGUE"))
        cats[tag] = t
        res[f"R3_{tag}"] = dict(t=t, table=table)

    groups, hnames = cd.load_hgnc()
    entries = {}
    for tag, MA, MB in (("ALL", ZA2, ZB2), ("SPECIFIC", RA2, RB2)):
        t = cats[tag]
        if t is None:
            continue
        ii, jj = np.nonzero((np.abs(MA) >= t) & (np.abs(MB) >= t) & (np.sign(MA) == np.sign(MB)))
        for i, j in zip(ii, jj):
            e = entries.setdefault((shared[i], g2[j]), dict(protein=shared[i], gene=g2[j], i=int(i), j=int(j), classes=set()))
            e["classes"].add(tag)
    for e in entries.values():
        i, j = e["i"], e["j"]
        e.update(zA=float(ZA2[i, j]), zB=float(ZB2[i, j]), sA=float(RA2[i, j]), sB=float(RB2[i, j]))
        para = bool(groups.get(e["protein"], set()) & groups.get(e["gene"], set()))
        e["paralog_flag"] = para and e["zA"] < 0

    # ---- R4: another cell type -------------------------------------------------------------------
    P_(); P_("  R4 DO REPRODUCED K562 EFFECTS HOLD IN RPE1? (|Z_C| >= 3, same sign)")
    g3 = [g for g in g2 if g in C["gi"]]; j3 = {g: k for k, g in enumerate(g3)}; at2 = {g: k for k, g in enumerate(g2)}
    Q3, _ = np.linalg.qr(Q[[at2[g] for g in g3]])
    sh3 = [p for p in shared if p in C["ki"]]; i3 = {p: k for k, p in enumerate(sh3)}
    ZC3 = C["Z"][[C["ki"][p] for p in sh3]][:, [C["gi"][g] for g in g3]]
    RC3 = resid(ZC3, Q3)
    ZA3 = A["Z"][[A["ki"][p] for p in sh3]][:, [A["gi"][g] for g in g3]]; RA3 = resid(ZA3, Q3)
    ders3 = [derangement(len(sh3), rng) for _ in range(N_DER)]
    for e in entries.values():
        if e["protein"] in i3 and e["gene"] in j3:
            a, b = i3[e["protein"]], j3[e["gene"]]
            e["zC"], e["sC"] = float(ZC3[a, b]), float(RC3[a, b])
    for tag, MC, MA_, key in (("ALL", ZC3, ZA3, "zC"), ("SPECIFIC", RC3, RA3, "sC")):
        ev = [e for e in entries.values() if tag in e["classes"] and key in e]
        if not ev:
            P_(f"     {tag:8s} no evaluable effects"); continue
        a = np.array([i3[e["protein"]] for e in ev]); b = np.array([j3[e["gene"]] for e in ev])
        sgn = np.sign(MA_[a, b])
        share = float(((np.abs(MC[a, b]) >= ZC_T) & (np.sign(MC[a, b]) == sgn)).mean())
        nul = [float(((np.abs(MC[d[a], b]) >= ZC_T) & (np.sign(MC[d[a], b]) == sgn)).mean()) for d in ders3]
        gen = share > max(nul)
        P_(f"     {tag:8s} {len(ev):,} evaluable: also in RPE1 {share:.1%} vs deranged {np.mean(nul):.1%} (max {max(nul):.1%}) "
           f"-> {'GENERAL BEYOND CHANCE' if gen else 'NOT BEYOND CHANCE'}")
        res[f"R4_{tag}"] = dict(n=len(ev), share=share, null_mean=float(np.mean(nul)), null_max=max(nul), general=bool(gen))

    # ---- R5: protein -> pathway ------------------------------------------------------------------
    P_(); P_("  R5 PROTEIN -> PATHWAY CATALOGUE (specific Z averaged over each process; z vs control rows)")
    mods, pnames, ptop = wc.reactome(set(g2)); mid = sorted(mods); g2i = {g: j for j, g in enumerate(g2)}
    M = np.zeros((len(g2), len(mid)), np.float32)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            M[g2i[g], c] = 1
    base = M.sum(0)
    rowA = [A["ki"][p] for p in shared]; rowB = [B["ki"][p] for p in shared]
    pos2A = {j: k for k, j in enumerate(cA)}; pos2B = {j: k for k, j in enumerate(cB)}

    def pscore(Rm, cisrows, pos2):
        Sx = Rm @ M
        cnt = np.tile(base, (len(Rm), 1))
        for r_, cols in enumerate(cisrows):
            cc = [pos2[c] for c in cols if c in pos2]
            if cc:
                cnt[r_] -= M[cc].sum(0)
        return Sx / np.maximum(cnt, 1)

    PA = pscore(RA2, [cis["A"][r] for r in rowA], pos2A); PB = pscore(RB2, [cis["B"][r] for r in rowB], pos2B)
    for Pm, s, cc in ((PA, A, cA), (PB, B, cB)):
        ctl = (resid(s["Zc"][:, cc], Q) @ M) / np.maximum(base, 1)
        Pm -= ctl.mean(0); Pm /= ctl.std(0, ddof=1)
    tP, tableP = ladder(lambda t: count(PA, PB, t), lambda t, d: count(PA, PB, t, ders[d]))
    for tt, nr, nl, f in tableP:
        P_(f"     t {tt:.0f}: reproduced {nr:,}, deranged mean {nl:,.0f}, FDR {f:.3f}")
    P_("     -> " + (f"CATALOGUE AT t = {tP:.0f}" if tP else "NO RELIABLE PATHWAY CATALOGUE"))
    res["R5"] = dict(t=tP, table=tableP, n_pathways=len(mid))
    pw = []
    if tP:
        ii, jj = np.nonzero((np.abs(PA) >= tP) & (np.abs(PB) >= tP) & (np.sign(PA) == np.sign(PB)))
        pw = [dict(protein=shared[i], pathway=mid[j], name=pnames[mid[j]], top="; ".join(ptop[mid[j]]),
                   zA=float(PA[i, j]), zB=float(PB[i, j])) for i, j in zip(ii, jj)]

    # ---- R6: shape -------------------------------------------------------------------------------
    P_(); P_("  R6 THE NETWORK'S SHAPE")
    spec = [e for e in entries.values() if "SPECIFIC" in e["classes"]]
    allc = [e for e in entries.values() if "ALL" in e["classes"]]
    for tag, ev in (("ALL", allc), ("SPECIFIC", spec)):
        if not ev:
            continue
        outd = collections.Counter(e["protein"] for e in ev); ind = collections.Counter(e["gene"] for e in ev)
        up = sum(e["zA"] > 0 for e in ev)
        P_(f"     {tag:8s} {len(ev):,} effects from {len(outd):,} proteins onto {len(ind):,} genes; up {up:,} / down {len(ev) - up:,}; "
           f"paralog-flagged {sum(e['paralog_flag'] for e in ev):,}")
        P_(f"        most effects: " + ", ".join(f"{p} {c}" for p, c in outd.most_common(12)))
        P_(f"        most often hit: " + ", ".join(f"{g} {c}" for g, c in ind.most_common(12)))
        res[f"R6_{tag}"] = dict(n=len(ev), proteins=len(outd), genes=len(ind), up=up,
                                top_proteins=outd.most_common(30), top_genes=ind.most_common(30))
    if spec:
        top = sorted([e for e in spec if not e["paralog_flag"]], key=lambda e: -min(abs(e["sA"]), abs(e["sB"])))[:40]
        P_("     strongest reproduced SPECIFIC protein -> gene effects (no paralog flag):")
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

    # ---- write to the memory bank ----------------------------------------------------------------
    os.makedirs(BANK, exist_ok=True)
    with gzip.open(os.path.join(BANK, "protein_gene_effects_reproduced.tsv.gz"), "wt") as fh:
        fh.write(f"# {CITE}. Reproduced in two independent K562 screens (rem/atlas/proteffect.py). "
                 f"z = noise-standardised; s = specific (tide + {K_AX} shared axes removed). Read protein_effects_CAVEATS.json.\n")
        fh.write("protein\tgene\tclass\tdirection\tzA_gw\tzB_ess\tzC_rpe1\tsA_gw\tsB_ess\tsC_rpe1\tparalog_flag\n")
        for e in sorted(entries.values(), key=lambda e: (e["protein"], e["gene"])):
            fh.write("\t".join([e["protein"], e["gene"], "+".join(sorted(e["classes"])), "up" if e["zA"] > 0 else "down",
                                f"{e['zA']:.2f}", f"{e['zB']:.2f}", f"{e['zC']:.2f}" if "zC" in e else "",
                                f"{e['sA']:.2f}", f"{e['sB']:.2f}", f"{e['sC']:.2f}" if "sC" in e else "",
                                "yes" if e["paralog_flag"] else ""]) + "\n")
    with gzip.open(os.path.join(BANK, "protein_pathway_effects_reproduced.tsv.gz"), "wt") as fh:
        fh.write(f"# {CITE}; Reactome (CC0). Specific effects, reproduced in two independent K562 screens.\n")
        fh.write("protein\tpathway\tname\ttop_level\tdirection\tzA_gw\tzB_ess\n")
        for e in sorted(pw, key=lambda e: (e["protein"], e["pathway"])):
            fh.write("\t".join([e["protein"], e["pathway"], e["name"], e["top"], "up" if e["zA"] > 0 else "down",
                                f"{e['zA']:.2f}", f"{e['zB']:.2f}"]) + "\n")
    res["files"] = ["memory_bank/cell_v3/protein_gene_effects_reproduced.tsv.gz",
                    "memory_bank/cell_v3/protein_pathway_effects_reproduced.tsv.gz"]
    res["runtime_s"] = time.time() - t0
    P_(); P_(f"  written: {', '.join(res['files'])}; artifact outputs/proteffect.json; runtime {time.time() - t0:.0f}s")
    P_(f"  data: {CITE}")
    finish()


if __name__ == "__main__":
    main()
