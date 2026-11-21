"""IS "BLAST RADIUS" A BIOLOGICAL QUANTITY OR A PROPERTY OF THE EXPERIMENT? Measured: mostly the experiment.

WHY THIS EXISTS. ulam_branching.py predicts each knockout's number of specific movers -- the "blast radius", the
neutron-count analogue in the Ulam mapping -- and compares a branching cascade against a ladder of NETWORK baselines:
PPI degree 0.3135, essentiality 0.4286, a GBM on node features 0.4325, the mean-field cascade 0.3171. The cascade
loses. But that whole ladder is made of biology, and it never asked the prior question: how much of the target is
TECHNICAL? An adversarial review claimed the answer is "most of it". This module checks that claim directly, because
it decides whether the blast-radius result means anything at all.

THE SOURCE. scratchpad/k562.h5ad carries per-perturbation covariates in obs. Nine of them are genuine assay
properties: num_cells_filtered, num_cells_unfiltered, UMI_count_unfiltered, mitopercent, z_gemgroup_UMI, fold_expr,
control_expr, pct_expr, cnv_score_z. The obs index is of the form "10023_ZC3H18_P1P2_ENSG00000158545", so the gene
symbol is field 1 -- joining on the raw index matches 0/1400 knockouts and joining on the parsed symbol matches
1400/1400, which is worth stating because the first attempt here silently matched nothing.

SIX OTHER obs COLUMNS ARE THE LABEL IN DISGUISE AND ARE EXCLUDED, NOT USED AS BASELINES. anderson_darling_counts,
mann_whitney_counts, mean_leverage_score, std_leverage_score, energy_test_p_value and TE_ratio are Replogle's own
perturbation-EFFECT statistics, computed from the same expression matrix the target is computed from. Measured here
for the record: they correlate with response size at +0.674, +0.651, +0.759 and +0.619. Those are not baselines,
they are restatements of the answer, and a ladder that included them would be measuring nothing.

WHAT WAS MEASURED, 5-fold CV over all 1400 knockouts, identical target and folds to ulam_branching:

    ASSAY-ONLY GBM (9 covariates)   Spearman 0.6881
    ---- everything below is from ulam_branching on the same target ----
    NODE-GBM (degree, essentiality, LOEUF, TF, complexes)    0.4325
    essentiality alone                                       0.4286
    mean-field branching cascade                             0.3171
    PPI degree alone                                         0.3135

    strongest single covariate: UMI_count_unfiltered   -0.4272
    second:                     z_gemgroup_UMI         -0.3923

THE READING. A model that knows nothing except how deeply each perturbation was sequenced predicts blast radius
BETTER than any biological model built here, and sequencing depth ALONE (-0.4272) beats PPI degree (0.3135) and
nearly matches the full node-GBM. Response size is therefore substantially a statement about detection power in the
assay, not about how far a perturbation propagates through the cell. That does not make the branching cascade's
failure less real -- it lost to biological baselines too -- but it removes most of the meaning from the target it
lost on, and any future work on blast radius must control for the assay block or it is modelling the sequencing run.

THREE HONEST CAVEATS, because this cuts both ways.
  1. THE COVARIATES ARE POST-TREATMENT. UMI depth and mitochondrial percentage are measured from the SAME cells that
     produce the readout. A perturbation with a large biological response may itself depress UMI counts (sick cells,
     lower complexity), in which case conditioning on them over-controls and throws away real signal. This is not
     resolvable from this data. The correct statement is that blast radius is NOT IDENTIFIED as a network quantity,
     not that it is definitely an artefact.
  2. The negative sign is worth noticing rather than explaining away: MORE sequencing goes with FEWER called movers,
     which is not naive detection power. It is consistent with depth entering the z-score denominator, and it is
     consistent with caveat 1. This module does not settle which.
  3. fold_expr, pct_expr and cnv_score_z are all-NaN on these rows, so the "nine covariates" are effectively six.
"""
import pickle

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, TIDE_FRAC, NFOLD = 1.0, 0.05, 5
ASSAY = ["num_cells_filtered", "num_cells_unfiltered", "UMI_count_unfiltered", "mitopercent",
         "z_gemgroup_UMI", "fold_expr", "control_expr", "pct_expr", "cnv_score_z"]
# NOT baselines -- these are perturbation-effect statistics computed from the same matrix as the target.
LEAKY = ["anderson_darling_counts", "mann_whitney_counts", "mean_leverage_score", "std_leverage_score",
         "energy_test_p_value", "TE_ratio"]


def main():
    f = h5py.File(f"{SP}/k562.h5ad", "r")
    raw = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"]["gene_transcript"][:]]
    sym = [r.split("_")[1] if len(r.split("_")) > 1 else r for r in raw]

    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(kos), len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    y = ((A >= TAU) & (~tide)).sum(1).astype(float)

    row = {}
    for i, s in enumerate(sym):
        row.setdefault(s, i)
    hit = [(ki[k], row[k]) for k in kos if k in row]
    print(f"matched {len(hit)}/{len(kos)} knockouts to h5ad obs rows (join on parsed gene symbol, not the raw index)")
    kidx = np.array([a for a, b in hit]); oidx = np.array([b for a, b in hit])
    yt = y[kidx]; yy = np.log1p(yt)

    X = np.stack([np.asarray(f["obs"][c][:], float)[oidx] for c in ASSAY], 1)
    rng = np.random.RandomState(0)
    fold = rng.permutation(len(kidx)) % NFOLD
    pred = np.zeros(len(kidx))
    for fo in range(NFOLD):
        tr, te = fold != fo, fold == fo
        pred[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4,
                                                 min_samples_leaf=20, random_state=0).fit(X[tr], yy[tr]).predict(X[te])
    s_assay = float(spearmanr(pred, yt).statistic)

    print(f"\nASSAY-ONLY GBM ({len(ASSAY)} covariates, {NFOLD}-fold CV, n={len(kidx)}): Spearman = {s_assay:.4f}")
    print("   from ulam_branching, same target: NODE-GBM 0.4325  essentiality 0.4286  cascade 0.3171  degree 0.3135")
    print("\nunivariate Spearman with response size:")
    uni = sorted(((c, float(spearmanr(x, yt).statistic)) for c, x in zip(ASSAY, X.T)),
                 key=lambda t: -abs(t[1]) if t[1] == t[1] else 0)
    for c, v in uni:
        print(f"   {c:24s} {v:+.4f}")
    print("\nLABEL-IN-DISGUISE columns, measured only to justify excluding them:")
    for c in LEAKY:
        try:
            print(f"   {c:24s} {float(spearmanr(np.asarray(f['obs'][c][:], float)[oidx], yt).statistic):+.4f}")
        except Exception:
            pass

    best = max((v for _, v in uni if v == v), key=abs)
    print(f"\nVERDICT: an assay-only model scores {s_assay:.4f} against 0.4325 for the best BIOLOGICAL model built in "
          f"ulam_branching, and the single strongest covariate ({uni[0][0]}, {uni[0][1]:+.4f}) on its own beats PPI "
          f"degree (0.3135) and nearly matches the full node-GBM. BLAST RADIUS IS SUBSTANTIALLY A PROPERTY OF THE "
          f"EXPERIMENT, not of how far a perturbation travels through the cell. This does not rescue the branching "
          f"cascade, which lost to the biological baselines as well -- it removes most of the meaning from the target "
          f"the cascade lost on. BUT THE COVARIATES ARE POST-TREATMENT, measured from the same cells as the readout, "
          f"so a large biological response could itself depress UMI counts and conditioning on them would "
          f"over-control. The defensible claim is that response size is NOT IDENTIFIED as a network quantity, not "
          f"that it is definitely an artefact. Strongest |rho| among assay covariates: {best:+.4f}.")


if __name__ == "__main__":
    main()
