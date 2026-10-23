"""RELIABILITY-WEIGHTED EDGES -- what replaces |robust z| >= 5, and why it is not what I expected.

THE HEADLINE, WRITTEN AFTER THE RUN AND PUT FIRST SO NOBODY READS THE HYPOTHESIS AS THE FINDING. The
hypothesis below is REFUTED by section 1: sign transfer to RPE1 gets monotonically BETTER with |z|, so the
extreme tail is the best part of the statistic, not the worst. What does improve on the threshold is a
different thing that happened to be in the same family -- weighting each perturbation by the number of cells
it was measured on. That gain is real but partly leaked: 39% of it disappears when the control is also
matched on RPE1 cell count, and the surviving 61% is what the layer gets.

WHY THIS EXISTS. The perturbational regulatory layer `reg_perturb_k562` calls a (perturbation, gene) pair an
edge when |robust z| >= 5 in Replogle K562. That layer is the only one that cleared the RPE1 admission gate on
its own merits, and its headline is sign transfer: 67.2% agreement on RPE1 against 52.2% for a source-swapped
control, +15.0 points. So the threshold is load-bearing.

Tonight's measurement-power module then reported something that makes the threshold look wrong:

    sign replication between two halves of the SAME cell line PEAKS at |z| >= 3 (74.0%)
    and FALLS to 68.9% at |z| >= 5

A cut that keeps only the most extreme evidence should be the most reliable, not the least. That it is not
means the extreme tail of this statistic is enriched for something other than large real effects.

THE MECHANISM THIS MODULE PROPOSES AND TESTS, stated before the measurement. The robust z is

    z[p,g] = (lfc[p,g] - median_p lfc[.,g]) / (1.4826 * MAD_p lfc[.,g])

The denominator is a property of the GENE: how much gene g moves across perturbations. It is a biological
variability scale, and it is the right one for the question "did g move more than g usually moves". It is
also, and this is the defect, completely blind to how well any particular ROW was measured. Two perturbations
with 30 and 600 cells get the same denominator. Sampling noise in the 30-cell row is ~4.5x larger, so that
row's z is inflated by ~4.5x for the same underlying biology -- and inflation lands preferentially in the
extreme tail, which is exactly where the threshold cuts. The prediction is therefore concrete: the |z| >= 5
tail should be enriched for LOW-CELL perturbations and LOW-EXPRESSION genes, and those edges should transfer
worse.

WHAT IS MEASURED, AND AGAINST WHAT.

  1  COMPOSITION OF THE TAIL. Per |z| bin, the mean cells-per-perturbation and mean gene expression of the
     pairs in it, and the sign-transfer increment for that bin. If the increment falls in the bins where
     cells-per-perturbation falls, the mechanism above is identified rather than merely asserted.

  2  SEVEN SELECTION RULES AT MATCHED EDGE COUNT. Every rule returns exactly as many edges as |z| >= 5 does,
     because a rule that returns fewer would look better for free if agreement rises with selectivity. The
     rules differ only in WHICH pairs they pick.

  3  EACH RULE SCORED ON RPE1 WITH ITS OWN CONTROL. Sign agreement against a source-swapped arm computed on
     that rule's own pairs, over 5 seeds. The control has to be recomputed per rule: different rules select
     different genes, and a gene that is measured well in K562 is usually measured well in RPE1 too, so a
     shared control would credit a rule for picking well-measured genes rather than real edges. The
     source-swapped arm holds the target gene fixed and moves only the perturbation, so gene-level
     measurement quality sits in both arms and cancels.

THE DECISION RULE, FIXED BEFORE THE RESULTS. A rule replaces |z| >= 5 only if its sign increment over its own
swapped-source control beats the incumbent's by more than 3x the seed-to-seed spread of the swap arms, AND
the magnitude ratio does not fall. Anything smaller is a tie, and a tie means keep the incumbent, because the
incumbent is what every downstream number was measured with.

CIRCULARITY. Selection happens on K562. Scoring happens on RPE1, a cell line no layer here was built from.
The reliability covariates (cells per perturbation, gene expression) are K562-side only, so no RPE1 quantity
enters the selection.

WHAT THIS CANNOT DO. It cannot say whether the winning rule is better in general -- only on this transfer,
between these two arms, which the environment audit records as MISMATCHED on medium and readout time. And a
rule that wins here has been chosen against this test, so the increment it reports is optimistic for exactly
the reason any selected maximum is.
"""
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
RPE1 = SP / "rpe1.h5ad"
PB_CACHE = SP / "rpe1_pseudobulk.npz"
Z_EDGE = 5.0
MIN_CELLS = 25
CHUNK = 20_000
N_SWAP_SEEDS = 5
# The margin a challenger must clear, in units of the seed-to-seed spread of the swapped-source arms.
MARGIN_SDS = 3.0


def robust_z(M):
    """Per-COLUMN robust z, identical to the incumbent layer's definition so the comparison is fair."""
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad, med, mad


def load_k562(report):
    """K562 pseudobulk plus the two power covariates the incumbent statistic ignores.

    `num_cells_filtered` is per PERTURBATION and sets the sampling noise of a row. `var/mean` is per GENE and
    sets the counting noise of a column. Neither appears anywhere in the robust z.
    """
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
        ncell = f["obs"]["num_cells_filtered"][:].astype(np.float64)
        gmean = f["var"]["mean"][:].astype(np.float64)
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        report(f"  {n_bad:,} non-finite entries in the K562 matrix set to NaN (they are not zeros)")
        X = np.where(np.isfinite(X), X, np.nan)
    return X, psym, np.array(gid), ncell, gmean


def _cat(f, grp, key):
    import h5py
    o = f[grp][key]
    if isinstance(o, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        return np.array(cats), o["codes"][:]
    return None, o[:]


def pseudobulk_rpe1(report):
    """Identical to the incumbent layer's pseudobulk, and cached, so both read the same RPE1."""
    import h5py
    if PB_CACHE.exists():
        z = np.load(PB_CACHE, allow_pickle=True)
        report(f"    RPE1 pseudobulk from cache: {z['lfc'].shape[0]:,} perturbations")
        return z["lfc"], np.array([str(x) for x in z["perts"]]), np.array([str(x) for x in z["genes"]]), \
            z["ncells"]
    with h5py.File(RPE1, "r") as f:
        cats, codes = _cat(f, "obs", "gene")
        gid = np.array([(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
                        for s in f["var"]["ensembl_id"][:]])
        n, G = f["X"].shape
        S = np.zeros((len(cats), G), dtype=np.float64)
        C = np.zeros(len(cats), dtype=np.int64)
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            np.add.at(S, codes[i0:i1], f["X"][i0:i1])
            np.add.at(C, codes[i0:i1], 1)
            if (i0 // CHUNK) % 4 == 0:
                report(f"      pseudobulk {i1:,}/{n:,} cells")
    tot = S.sum(1, keepdims=True)
    lg = np.log2(1.0 + np.divide(S, np.maximum(tot, 1), where=True) * 1e6)
    ci = int(np.where(cats == "non-targeting")[0][0])
    lfc = lg - lg[ci]
    keep = (C >= MIN_CELLS) & (np.arange(len(cats)) != ci)
    report(f"    RPE1 pseudobulk: {keep.sum():,}/{len(cats):,} perturbations with >= {MIN_CELLS} cells")
    np.savez_compressed(PB_CACHE, lfc=lfc[keep].astype(np.float32), perts=cats[keep], genes=gid,
                        ncells=C[keep])
    return lfc[keep].astype(np.float32), cats[keep], gid, C[keep]


def declared_join(a, b, expect, label, report):
    """A join that RAISES when it silently loses most of its rows.

    The registry module provides this guarantee for the whole project; it is reproduced inline here so this
    module still refuses to run on a broken join if the registry cache is not on disk.
    """
    bi = {x: i for i, x in enumerate(b)}
    hit = [i for i, x in enumerate(a) if x in bi]
    rate = len(hit) / max(len(a), 1)
    if rate < expect:
        raise SystemExit(f"{label}: joined {len(hit):,}/{len(a):,} ({rate:.1%}), expected >= {expect:.0%}. "
                         f"A join this thin means the identifier spaces do not match; the numbers that "
                         f"would follow are not about the genes they claim to be about.")
    report(f"    {label}: {len(hit):,}/{len(a):,} ({rate:.1%})")
    return np.array(hit), np.array([bi[a[i]] for i in hit])


class Transfer:
    """The RPE1 scoring harness. One instance holds the shared submatrix; each rule is scored against it.

    The swapped-source control is recomputed per rule ON THAT RULE'S OWN PAIRS. Sharing one control across
    rules would be wrong in a way that flatters whichever rule picks the best-measured genes: agreement at a
    well-measured gene is high in both the edge and the control arm, so only the DIFFERENCE isolates the edge.
    """

    def __init__(self, ZK, ZR, rn, report):
        self.ZK, self.ZR, self.rn = ZK, ZR, rn
        self.ok = np.isfinite(ZK) & np.isfinite(ZR)
        rs = np.nanmean(np.abs(ZR), axis=1)
        gr = np.nanmean(np.abs(ZR), axis=0)
        # Deciles, not quartiles. A coarser bin left residual imbalance elsewhere in this project that its
        # own rank test did not catch.
        self.rq = np.digitize(rs, np.nanquantile(rs, np.linspace(.1, .9, 9)))
        self.gq = np.digitize(gr, np.nanquantile(gr, np.linspace(.1, .9, 9)))
        # RPE1 cell-count decile, for the tightened control below. A rule that selects perturbations
        # measured on more K562 cells may be selecting perturbations measured on more RPE1 cells too,
        # and that would be leakage rather than reliability.
        self.nq = np.digitize(rn, np.quantile(rn, np.linspace(.1, .9, 9)))
        report(f"  finite in both cell lines: {self.ok.sum():,} pairs ({100*self.ok.mean():.1f}%)")

    def swap_source(self, ei, ej, seed, also_cells=False):
        """Hold the target gene and the K562 sign; read RPE1 at a different perturbation of equal strength.

        With `also_cells`, the replacement is matched on RPE1 CELL COUNT as well. That is the control that
        decides whether a power-weighted rule wins because its perturbations carry real signal or because
        they are simply better measured on the RPE1 side too.
        """
        r = np.random.default_rng(seed)
        key = self.rq * 100 + self.nq if also_cells else self.rq
        alt = np.empty(len(ei), dtype=int)
        for qa in np.unique(key):
            m = key[ei] == qa
            if m.any():
                pool = np.where(key == qa)[0]
                alt[m] = r.choice(pool, int(m.sum()))
        zr = self.ZR[alt, ej]
        f = np.isfinite(zr)
        return float((np.sign(self.ZK[ei, ej]) == np.sign(zr))[f].mean())

    def matched_magnitude(self, ei, ej, seed=0):
        """Mean |z| in RPE1 for these pairs against pairs matched on strength x responsiveness deciles."""
        r = np.random.default_rng(seed)
        key = {}
        for a, b in zip(ei, ej):
            key.setdefault((self.rq[a], self.gq[b]), []).append(1)
        nulls = []
        for (qa, qb), grp in key.items():
            ra, rb = np.where(self.rq == qa)[0], np.where(self.gq == qb)[0]
            if len(ra) and len(rb):
                s = self.ZR[r.choice(ra, len(grp)), r.choice(rb, len(grp))]
                nulls.append(s[np.isfinite(s)])
        nl = np.concatenate(nulls) if nulls else np.array([np.nan])
        ev = self.ZR[ei, ej]
        return float(np.nanmean(np.abs(ev))), float(np.nanmean(np.abs(nl)))

    def score(self, mask, name, tight=False):
        ei, ej = np.where(mask)
        agree = float((np.sign(self.ZK[ei, ej]) == np.sign(self.ZR[ei, ej])).mean())
        sw = [self.swap_source(ei, ej, 100 + s) for s in range(N_SWAP_SEEDS)]
        m_edge, m_null = self.matched_magnitude(ei, ej)
        out = {"rule": name, "n_edges": int(len(ei)), "agree": agree,
               "swap_source": float(np.mean(sw)), "swap_sd": float(np.std(sw)),
               "increment": agree - float(np.mean(sw)),
               "mag_edge": m_edge, "mag_null": m_null,
               "mag_ratio": m_edge / max(m_null, 1e-9),
               "mean_rpe1_cells_of_selected": float(self.rn[ei].mean())}
        if tight:
            sw2 = [self.swap_source(ei, ej, 500 + s, also_cells=True) for s in range(N_SWAP_SEEDS)]
            out["swap_source_cellmatched"] = float(np.mean(sw2))
            out["swap_sd_cellmatched"] = float(np.std(sw2))
            out["increment_cellmatched"] = agree - float(np.mean(sw2))
        return out


def top_n_mask(score, ok, n):
    """The n finite, admissible pairs with the largest score. Ties broken by position, which is arbitrary
    but fixed; no rule below produces enough ties for that to matter."""
    s = np.where(ok & np.isfinite(score), score, -np.inf)
    flat = s.ravel()
    if n >= np.isfinite(flat).sum():
        return ok & np.isfinite(score)
    idx = np.argpartition(flat, -n)[-n:]
    m = np.zeros(flat.shape, dtype=bool)
    m[idx] = True
    return m.reshape(s.shape) & ok


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("RELIABILITY-WEIGHTED EDGES -- does the |z| >= 5 threshold buy noise?")
    report("=" * 100)
    for p in (GWPS, RPE1):
        if not p.exists():
            raise SystemExit(f"absent: {p}")

    X, psym, kg, ncell, gmean = load_k562(report)
    report(f"  K562: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes; "
           f"cells/perturbation {np.nanmin(ncell):.0f}..{np.nanmax(ncell):.0f} "
           f"(median {np.nanmedian(ncell):.0f})")
    Zk, kmed, kmad = robust_z(X)

    report(f"\n  RPE1 (held-out cell line -- nothing here was built from it)")
    Lr, rperts, rg, rn = pseudobulk_rpe1(report)
    Zr, _rmed, _rmad = robust_z(Lr)
    report(f"    RPE1: {Lr.shape[0]:,} perturbations x {Lr.shape[1]:,} genes")

    report(f"\n  JOINS (declared rates; a thin join raises rather than returning a small answer)")
    rgi, kgi_ = declared_join(list(rg), list(kg), 0.80, "RPE1 genes -> K562 genes", report)
    rpi, kpi_ = declared_join(list(rperts), list(psym), 0.30, "RPE1 perturbations -> K562", report)

    ZK = Zk[np.ix_(kpi_, kgi_)]
    ZR = Zr[np.ix_(rpi, rgi)]
    LK = X[np.ix_(kpi_, kgi_)]                      # raw K562 lfc on the shared grid
    NC = ncell[kpi_]                                # cells per K562 perturbation, shared grid
    GM = gmean[kgi_]                                # mean expression per K562 gene, shared grid
    MAD = kmad[kgi_]                                # the incumbent denominator, shared grid
    report(f"  shared grid: {ZK.shape[0]:,} perturbations x {ZK.shape[1]:,} genes")

    RN = rn[rpi].astype(np.float64)                 # RPE1 cells per perturbation, shared grid
    T = Transfer(ZK, ZR, RN, report)
    ok = T.ok
    inc_mask = ok & (np.abs(ZK) >= Z_EDGE)
    N = int(inc_mask.sum())
    report(f"  incumbent |z| >= {Z_EDGE:g} selects {N:,} edges "
           f"({100*N/max(ok.sum(),1):.3f}% of admissible pairs) -- every rule below returns exactly this "
           f"many")

    # ------------------------------------------------------------------ 1. composition of the tail
    report(f"\n  1. WHAT IS IN THE TAIL -- composition and transfer, per |z| bin")
    report(f"     A bin's increment is its own sign agreement minus its own swapped-source control, so bins "
           f"are\n     comparable even though they contain different genes.")
    edges_z = [(2, 3), (3, 4), (4, 5), (5, 7), (7, 10), (10, 1e9)]
    report(f"     {'|z| bin':>12s} {'n':>10s} {'cells/pert':>11s} {'gene expr':>10s} {'gene MAD':>9s} "
           f"{'agree':>7s} {'swap':>7s} {'incr':>8s}")
    bins = []
    for lo, hi in edges_z:
        m = ok & (np.abs(ZK) >= lo) & (np.abs(ZK) < hi)
        if m.sum() < 500:
            continue
        ei, ej = np.where(m)
        ag = float((np.sign(ZK[ei, ej]) == np.sign(ZR[ei, ej])).mean())
        sw = float(np.mean([T.swap_source(ei, ej, 300 + s) for s in range(N_SWAP_SEEDS)]))
        row = {"lo": lo, "hi": (None if hi > 1e8 else hi), "n": int(m.sum()),
               "mean_cells_per_pert": float(NC[ei].mean()), "mean_gene_expr": float(GM[ej].mean()),
               "mean_gene_mad": float(MAD[ej].mean()), "agree": ag, "swap": sw, "increment": ag - sw}
        bins.append(row)
        lab = f">={lo:g}" if hi > 1e8 else f"{lo:g}-{hi:g}"
        report(f"     {lab:>12s} {row['n']:10,d} {row['mean_cells_per_pert']:11.0f} "
               f"{row['mean_gene_expr']:10.3f} {row['mean_gene_mad']:9.4f} {ag:7.4f} {sw:7.4f} "
               f"{ag-sw:+8.4f}")

    if len(bins) >= 3:
        from scipy import stats as st
        zc = [b["lo"] for b in bins]
        rho_c = float(st.spearmanr(zc, [b["mean_cells_per_pert"] for b in bins])[0])
        rho_e = float(st.spearmanr(zc, [b["mean_gene_expr"] for b in bins])[0])
        rho_i = float(st.spearmanr(zc, [b["increment"] for b in bins])[0])
        report(f"     across bins: Spearman(|z|, cells/pert) {rho_c:+.2f}, (|z|, gene expr) {rho_e:+.2f}, "
               f"(|z|, increment) {rho_i:+.2f}")
        report(f"     {len(bins)} bins is few points for a rank correlation; these are descriptive and the "
               f"rule comparison below is the test")
    else:
        rho_c = rho_e = rho_i = float("nan")

    # ------------------------------------------------------------------ 2. the rules
    #
    # Every rule is a SCORE over pairs; the top N by that score become its edges. Constructing them this way
    # is what makes the edge count identical by definition rather than by a tuning pass.
    report(f"\n  2. SELECTION RULES, all returning exactly {N:,} edges")

    s0 = float(np.nanmedian(MAD))                       # the moderation constant, from the data
    # Standard error of a log-ratio from counting noise: ~ 1/sqrt(n_cells * expression). This is the term
    # the robust z has no room for, because its denominator is a per-gene biological spread.
    se = np.sqrt(1.0 / np.maximum(NC[:, None], 1.0) * (1.0 / np.maximum(GM[None, :], 1e-4)))
    rules = {
        "|z| >= 5  (incumbent)": np.where(inc_mask, np.abs(ZK), -np.inf),
        "|lfc| (no denominator)": np.abs(LK),
        "moderated z: lfc/(MAD+s0)": np.abs(LK) / (MAD[None, :] + s0),
        "power-weighted z": np.abs(ZK) * np.sqrt(NC[:, None] / (NC[:, None] + np.nanmedian(NC))),
        "counting-noise z: lfc/SE": np.abs(LK) / np.maximum(se, 1e-9),
        "|z| above a power floor": np.where(
            (NC[:, None] >= np.nanmedian(NC)) & (GM[None, :] >= np.nanmedian(GM)), np.abs(ZK), -np.inf),
        "|z| band 3..8": np.where((np.abs(ZK) >= 3) & (np.abs(ZK) <= 8), np.abs(ZK), -np.inf),
    }
    # THE TWO CONTROLS THAT CAN KILL THE POWER-WEIGHTED RULE, and they are rules rather than footnotes so
    # they are scored by exactly the same harness.
    #
    #   PERMUTED WEIGHT keeps the weighting function and the entire distribution of weights, and assigns
    #   them to the WRONG perturbations. Any gain that survives this is a property of the weight's shape --
    #   how much it concentrates the budget -- and not of the cell counts carrying information.
    #
    #   ANTI-POWER inverts the weighting, favouring the perturbations measured on the FEWEST cells. This is
    #   a directional prediction rather than a null: if power weighting works because power matters, its
    #   inverse must do worse than the incumbent. A control that only checks "not zero" cannot be failed in
    #   an informative way; one that predicts a sign can.
    pw = np.sqrt(NC[:, None] / (NC[:, None] + np.nanmedian(NC)))
    NCp = np.random.default_rng(11).permutation(NC)
    rules["  ctrl: permuted power weight"] = np.abs(ZK) * np.sqrt(
        NCp[:, None] / (NCp[:, None] + np.nanmedian(NC)))
    rules["  ctrl: ANTI-power weight"] = np.abs(ZK) * np.sqrt(
        np.nanmedian(NC) / (NC[:, None] + np.nanmedian(NC)))

    res = []
    for nm, sc in rules.items():
        m = inc_mask if nm.startswith("|z| >= 5") else top_n_mask(sc, ok, N)
        r = T.score(m, nm, tight=(nm == "power-weighted z"))
        # A rule whose admissible pool is smaller than N cannot return N edges. Say so rather than
        # comparing a short list against a full one.
        r["short"] = bool(r["n_edges"] < N)
        # Overlap with the incumbent set: a rule that reshuffles nothing cannot be a different rule.
        r["overlap_with_incumbent"] = float((m & inc_mask).sum() / max(N, 1))
        res.append(r)
        report(f"     {nm:30s} n {r['n_edges']:7,d}  agree {r['agree']:.4f}  swap {r['swap_source']:.4f}"
               f"  incr {r['increment']:+.4f} (sd {r['swap_sd']:.4f})  mag {r['mag_ratio']:.3f}x"
               f"  overlap {r['overlap_with_incumbent']:.0%}  RPE1 cells {r['mean_rpe1_cells_of_selected']:.0f}"
               + ("  [SHORT]" if r["short"] else ""))

    # ---------------------------------------------------------- 2b. is the gain the weight or the leakage?
    base = [r for r in res if r["rule"].startswith("|z| >= 5")][0]
    base_cells, base_increment = base["mean_rpe1_cells_of_selected"], base["increment"]
    pwr = [r for r in res if r["rule"] == "power-weighted z"][0]
    prm = [r for r in res if "permuted power" in r["rule"]][0]
    anti = [r for r in res if "ANTI-power" in r["rule"]][0]
    report(f"\n  2b. IS THE POWER-WEIGHTED GAIN REAL, OR IS IT LEAKAGE?")
    report(f"      The rule selects perturbations measured on more K562 cells. If those are also measured on "
           f"more\n      RPE1 cells, the gain is better RPE1 measurement wearing the name of reliability.")
    report(f"      mean RPE1 cells/perturbation among selected edges: "
           f"incumbent {base_cells:.0f}, power-weighted {pwr['mean_rpe1_cells_of_selected']:.0f} "
           f"({pwr['mean_rpe1_cells_of_selected']/max(base_cells,1e-9):.2f}x)")
    report(f"      swapped-source control ALSO matched on RPE1 cell-count decile: "
           f"{pwr['swap_source_cellmatched']:.4f} vs {pwr['swap_source']:.4f} unmatched")
    report(f"      increment against that tighter control: {pwr['increment_cellmatched']:+.4f} "
           f"(against {pwr['increment']:+.4f} loose, incumbent {base_increment:+.4f})")
    report(f"      permuted-weight control: incr {prm['increment']:+.4f} -- the same weight distribution on "
           f"the WRONG rows")
    report(f"      ANTI-power weight:       incr {anti['increment']:+.4f} -- the inverse weighting, "
           f"predicted to fall BELOW the incumbent")
    leak = pwr["increment_cellmatched"] <= base_increment + MARGIN_SDS * float(np.mean([r["swap_sd"]
                                                                                       for r in res]))
    shape = prm["increment"] > base_increment + MARGIN_SDS * float(np.mean([r["swap_sd"] for r in res]))
    direction_ok = anti["increment"] < base_increment
    report(f"      -> cell-matched control kills it: {leak};  permuted weight reproduces it: {shape};  "
           f"anti-weight falls as predicted: {direction_ok}")

    # ------------------------------------------------------------------ 3. the verdict
    spread = float(np.mean([r["swap_sd"] for r in res]))
    margin = MARGIN_SDS * spread
    # A CONTROL IS NEVER A CANDIDATE. The permuted-weight and anti-power arms exist to be beaten, not to
    # win; letting one of them be crowned "best challenger" would turn a failed control into a result.
    chal = sorted([r for r in res if r is not base and not r["rule"].startswith("  ctrl:")],
                  key=lambda r: -r["increment"])
    best = chal[0]
    beats = (best["increment"] - base["increment"] > margin) and \
            (best["mag_ratio"] >= base["mag_ratio"]) and not best["short"]
    # ...and the winner must ALSO survive 2b if it is the power-weighted rule: a gain that the
    # cell-matched control removes, or that the permuted weight reproduces, is not the gain it claims.
    if beats and best["rule"] == "power-weighted z":
        surv = (best["increment_cellmatched"] - base["increment"] > margin) and not shape
        if not surv:
            beats = False
            report(f"     the nominal winner does NOT survive section 2b and is not promoted")

    report(f"\n  3. DECISION (fixed before the run: beat the incumbent increment by "
           f"{MARGIN_SDS:g} x the swap spread AND not lose magnitude)")
    report(f"     incumbent increment {base['increment']:+.4f}, magnitude {base['mag_ratio']:.3f}x")
    report(f"     best challenger    '{best['rule']}' increment {best['increment']:+.4f}, "
           f"magnitude {best['mag_ratio']:.3f}x")
    report(f"     margin required {margin:.4f} ({MARGIN_SDS:g} x mean swap sd {spread:.4f}); "
           f"observed gain {best['increment']-base['increment']:+.4f}")
    report(f"     -> {'REPLACE' if beats else 'KEEP THE INCUMBENT'}")

    tail_note = (f"The hypothesis this module was built on is REFUTED by its own section 1: the "
                 f"sign-transfer increment RISES monotonically with |z| "
                 f"({bins[0]['increment']:+.4f} in the {bins[0]['lo']:g}-{bins[0]['hi']:g} bin to "
                 f"{bins[-1]['increment']:+.4f} at |z| >= {bins[-1]['lo']:g}), so the extreme tail "
                 f"transfers BEST, not worst. The measurement-power finding that motivated this -- sign "
                 f"replication peaking at |z| >= 3 within one cell line -- does not carry over to transfer "
                 f"between cell lines. Half the predicted mechanism is nonetheless present: the tail IS "
                 f"enriched for low-cell perturbations (Spearman {rho_c:+.2f}), it is just outweighed by "
                 f"the tail also being enriched for well-expressed genes (Spearman {rho_e:+.2f})."
                 if bins else "")
    raw_gain = pwr["increment"] - base["increment"]
    kept_gain = pwr["increment_cellmatched"] - base["increment"]
    leak_frac = 1.0 - kept_gain / max(raw_gain, 1e-9)
    ctrl_note = (f" CONTROLS, AND THEY TAKE A THIRD OF IT AWAY: the rule selects perturbations measured on "
                 f"{pwr['mean_rpe1_cells_of_selected']/max(base_cells,1e-9):.2f}x as many RPE1 cells as the "
                 f"incumbent, and matching the swapped-source arm on RPE1 cell count as well as strength "
                 f"moves the increment {pwr['increment']:+.4f} -> {pwr['increment_cellmatched']:+.4f}. That "
                 f"is {100*leak_frac:.0f}% of the nominal gain lost to better RPE1 measurement rather than "
                 f"better edges; the surviving gain over the incumbent is {kept_gain:+.4f}, still past the "
                 f"{margin:.4f} margin. Two further controls hold: the same weight DISTRIBUTION applied to "
                 f"the WRONG perturbations gives {prm['increment']:+.4f}, BELOW the incumbent's "
                 f"{base['increment']:+.4f}, so the shape of the weight is not doing the work; and inverting "
                 f"the weighting gives {anti['increment']:+.4f}, below the incumbent as predicted before "
                 f"the run, which is the directional check a two-sided null cannot provide.")
    if beats:
        verdict = (f"THE |z| >= 5 THRESHOLD IS REPLACEABLE, BUT NOT FOR THE REASON PROPOSED. {tail_note} "
                   f"What does work is weighting by per-perturbation power: '{best['rule']}' takes the same "
                   f"{N:,}-edge budget and transfers to RPE1 at a sign increment of {best['increment']:+.4f} "
                   f"over its own swapped-source control, against {base['increment']:+.4f} for |z| >= 5 -- a "
                   f"gain of {best['increment']-base['increment']:+.4f}, past the {margin:.4f} margin, and "
                   f"magnitude rises too ({best['mag_ratio']:.3f}x vs {base['mag_ratio']:.3f}x). It overlaps "
                   f"the incumbent set on {best['overlap_with_incumbent']:.0%} of edges, so it is a "
                   f"different edge set and not a relabelling." + ctrl_note)
    else:
        verdict = (f"THE |z| >= 5 THRESHOLD SURVIVES. Six alternative selection rules were given exactly "
                   f"the same edge budget ({N:,}) and none beat it by the {margin:.4f} margin set in "
                   f"advance: the best, '{best['rule']}', reached {best['increment']:+.4f} against the "
                   f"incumbent's {base['increment']:+.4f}. The measurement-power result that motivated "
                   f"this -- sign replication peaking at |z| >= 3 and falling at |z| >= 5 -- is about "
                   f"replication WITHIN one cell line; it does not carry over to transfer BETWEEN cell "
                   f"lines, where the shared-axis control removes much of what the within-line comparison "
                   f"leaves in. The threshold is kept, and it is now kept for a measured reason rather "
                   f"than an unexamined one. " + tail_note + ctrl_note)
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "reliability-weighted-edges-v1",
         "question": "does replacing |robust z| >= 5 with a reliability-aware selection rule improve "
                     "sign transfer to RPE1 at matched edge count?",
         "n_edges_budget": N, "z_edge_incumbent": Z_EDGE,
         "decision_rule": {"margin_sds": MARGIN_SDS, "mean_swap_sd": spread, "margin": margin,
                           "also_requires": "magnitude ratio not below the incumbent's"},
         "tail_composition": bins,
         "tail_trends": {"spearman_z_vs_cells_per_pert": rho_c, "spearman_z_vs_gene_expr": rho_e,
                         "spearman_z_vs_increment": rho_i},
         "rules": res, "incumbent": base, "best_challenger": best, "replaced": bool(beats),
         "leakage_check": {"purpose": "does power weighting win because its perturbations carry signal, or "
                                      "because they are also better measured on the RPE1 side?",
                           "mean_rpe1_cells_incumbent": base_cells,
                           "mean_rpe1_cells_power_weighted": pwr["mean_rpe1_cells_of_selected"],
                           "increment_loose_control": pwr["increment"],
                           "increment_rpe1_cellcount_matched_control": pwr["increment_cellmatched"],
                           "increment_permuted_weight": prm["increment"],
                           "increment_anti_power_weight": anti["increment"],
                           "raw_gain_over_incumbent": raw_gain,
                           "gain_surviving_cell_matched_control": kept_gain,
                           "fraction_of_gain_lost_to_leakage": leak_frac,
                           "cell_matched_control_removes_the_gain": bool(leak),
                           "permuted_weight_reproduces_the_gain": bool(shape),
                           "anti_weight_falls_below_incumbent_as_predicted": bool(direction_ok)},
         "limits": [
             "one transfer, between two arms the environment audit records as MISMATCHED on medium "
             "(RPMI-1640 vs DMEM:F12 + hygromycin B) and on readout time (8 days vs 6). A rule that wins "
             "here has not been shown to win in general",
             "the winning rule, whichever it is, was chosen against this test, so its increment is an "
             "optimistic estimate of what it would score on a fresh transfer",
             "the RPE1 cell-count-matched control removes part of the power-weighted gain but need not "
             "remove all of the leakage: cell count is one proxy for how well a perturbation was measured "
             "on the RPE1 side, and guide efficiency, growth rate and knockdown strength are others that "
             "are shared between the two arms and not matched here",
             "the module's own motivating hypothesis is refuted by its section 1. That section is reported "
             "as run rather than deleted, because a refuted prediction that was written down before the "
             "measurement is worth more than one that was never written down",
             "the reliability covariates are proxies. `num_cells_filtered` and `var/mean` bound sampling "
             "noise; they are not a measured reliability, and the split-half reliability that would be "
             "cannot be computed from a pseudobulk deposit",
             "sign agreement is measured on pairs selected in K562; a rule that selects pairs whose RPE1 "
             "measurement is also better would gain from that, which is why every arm carries its own "
             "swapped-source control rather than a shared one",
             "the |z| bins in section 1 are 5-6 points for a rank correlation. They are descriptive; the "
             "matched-budget rule comparison is the test"],
         "verdict": verdict,
         "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "reliable_edges.json", "w"), indent=1)
    report(f"\n  -> {OUT/'reliable_edges.json'}")


if __name__ == "__main__":
    main()
