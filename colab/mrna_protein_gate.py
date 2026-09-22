"""IS THE mRNA->PROTEIN GAP PREDICTABLE? The gate that decides whether protein can be laid on the network.

THE ARCHITECTURE THIS TESTS. The proposal is: for each knockout, put protein counts on the network, see which
part of the network moves and by how much, trace TF -> transcription -> protein -> reaction, and cross-check
against Perturb-seq mRNA. Every step is sound except the first, and the first has no data layer:

    CRISPR-paired proteomics does not exist at proteome scale. Perturb-CITE-seq gives 4-200 targeted
    proteins; single-cell mass spec gives 2,000-6,000 proteins but is not paired with pooled CRISPR at
    scale. There is no "protein count per knockout, proteome-wide" to lay on anything.

THE SUBSTITUTE the architecture implies is to use Perturb-seq mRNA -- which we DO have genome-wide across
six cell lines -- and correct it toward protein. That only works if the correction is PREDICTABLE.

`mrna_vs_protein.py` found the gap is molecule-specific and swings hard: PD-L1's mRNA response predicts its
protein at R^2 = 0.81 while its own paralog PD-L2, same cells and same guides, sits at 0.025. Four proteins
cannot tell us whether that swing is learnable. This does it at proteome scale.

THE DATA. CPTAC, matched per-tumour transcriptomics and proteomics: 11,510 genes measured both ways across
121 samples in BRCA alone, with more cohorts available. For each gene, the mRNA-protein correlation across
samples is a per-gene number, and the question is whether it can be predicted from properties of the gene.

WHAT THIS IS AND IS NOT. CPTAC measures ABUNDANCE correlation across samples. Papalexi measured RESPONSE
correlation under perturbation. Those are different quantities and neither substitutes for the other. What
CPTAC can answer -- and Papalexi's four points cannot -- is whether the gap has structure: if abundance
correlation is predictable from gene properties, there is a genome-wide prior on where mRNA is trustworthy,
which is exactly the per-node confidence the network propagation needs. If it is unpredictable, no amount of
mRNA correction will do, and the architecture needs measured protein.

THE PREDICTORS, all things we already have or can compute:
    protein abundance and its variance     highly expressed proteins are measured better
    mRNA abundance and variance            a gene with no mRNA variation cannot show correlation
    PPI degree                             hub proteins are complex members, buffered by stoichiometry
    ubiquitous expression                  housekeeping genes are constitutively regulated
    predicted mRNA half-life               fast-turnover transcripts track protein more closely
Held out by chromosome-free random split with a shuffled-label control, because "predictable" must mean
predictable out-of-sample, not fitted.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
COHORTS = os.environ.get("MPG_COHORTS", "Brca,Ccrcc,Luad").split(",")
MIN_SAMPLES = 40          # a per-gene correlation on fewer paired samples is not a measurement
SEEDS = (0, 1, 2, 3, 4)


def flat(cols):
    return [c[0] if isinstance(c, tuple) else c for c in cols]


def per_gene_corr(cohort):
    """Per-gene Spearman between mRNA and protein across samples, for one cohort."""
    import cptac
    from scipy import stats
    c = getattr(cptac, cohort)()
    srcs = c.list_data_sources()
    tsrc = [s for s in srcs.loc[srcs["Data type"] == "transcriptomics", "Available sources"].iloc[0]]
    psrc = [s for s in srcs.loc[srcs["Data type"] == "proteomics", "Available sources"].iloc[0]]
    rna = c.get_transcriptomics(tsrc[-1])
    pro = c.get_proteomics(psrc[-1])
    rna.columns = flat(rna.columns)
    pro.columns = flat(pro.columns)
    rna = rna.loc[:, ~rna.columns.duplicated()]
    pro = pro.loc[:, ~pro.columns.duplicated()]
    si = rna.index.intersection(pro.index)
    shared = sorted(set(rna.columns) & set(pro.columns))
    out = {}
    R, P = rna.loc[si], pro.loc[si]
    for g in shared:
        a, b = R[g].values.astype(float), P[g].values.astype(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < MIN_SAMPLES or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            continue
        rho = stats.spearmanr(a[m], b[m])[0]
        if np.isfinite(rho):
            out[g] = {"rho": float(rho), "n": int(m.sum()),
                      "prot_mean": float(np.nanmean(b[m])), "prot_sd": float(np.nanstd(b[m])),
                      "rna_mean": float(np.nanmean(a[m])), "rna_sd": float(np.nanstd(a[m]))}
    return out, len(si)


def ppi_degree():
    """PPI degree per gene, if the project's interaction network is on disk."""
    import gzip
    import collections
    for cand in (OUT / "interactions.tsv.gz", OUT / "ppi.tsv.gz",
                 OUT.parent / "interactions.tsv.gz"):
        if cand.exists():
            d = collections.Counter()
            op = gzip.open if str(cand).endswith(".gz") else open
            with op(cand, "rt") as fh:
                for line in fh:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 2:
                        d[p[0]] += 1
                        d[p[1]] += 1
            return dict(d)
    return {}


def main():
    from scipy import stats
    from sklearn.metrics import r2_score
    import xgboost as xgb

    print("=" * 100)
    print("IS THE mRNA->PROTEIN GAP PREDICTABLE? -- CPTAC, matched per-tumour RNA and protein")
    print("=" * 100)

    per_cohort, merged = {}, {}
    for co in COHORTS:
        try:
            g, ns = per_gene_corr(co)
        except Exception as e:
            print(f"  {co:7s} FAILED ({type(e).__name__}: {e})")
            continue
        rhos = np.array([v["rho"] for v in g.values()])
        print(f"  {co:7s} {len(g):6,d} genes over {ns:3d} paired samples   "
              f"median rho {np.median(rhos):+.3f}   "
              f"frac rho>0.5 {np.mean(rhos > 0.5):.3f}   frac rho<0 {np.mean(rhos < 0):.3f}")
        per_cohort[co] = {"n_genes": len(g), "n_samples": ns,
                          "median_rho": float(np.median(rhos)),
                          "frac_above_0.5": float(np.mean(rhos > 0.5)),
                          "frac_negative": float(np.mean(rhos < 0))}
        for k, v in g.items():
            merged.setdefault(k, []).append(v)
    if not merged:
        raise SystemExit("no cohort loaded -- nothing to gate")

    # A gene's rho averaged over cohorts, and how much it VARIES between them. If rho is not even
    # reproducible across cohorts it cannot be predicted from gene properties, and that check comes first.
    genes = sorted(merged)
    rho = np.array([np.mean([v["rho"] for v in merged[g]]) for g in genes])
    nco = np.array([len(merged[g]) for g in genes])
    multi = nco >= 2
    if multi.sum() > 50:
        spread = np.array([np.std([v["rho"] for v in merged[g]]) for g in genes])[multi]
        pair = [(np.mean([v["rho"] for v in merged[g][:1]]),
                 np.mean([v["rho"] for v in merged[g][1:2]])) for g in np.array(genes)[multi]]
        a, b = np.array([p[0] for p in pair]), np.array([p[1] for p in pair])
        cross = stats.spearmanr(a, b)[0]
        print(f"\n  REPRODUCIBILITY across cohorts ({multi.sum():,} genes in >=2): "
              f"cohort-to-cohort rho of rho = {cross:+.3f}, mean within-gene sd {spread.mean():.3f}")
        print("    ^ if this is low, per-gene rho is mostly noise and cannot be predicted by anything")

    print(f"\n  pooled: {len(genes):,} genes   median rho {np.median(rho):+.3f}   "
          f"IQR {np.percentile(rho,25):+.3f}..{np.percentile(rho,75):+.3f}")

    # ---- features ----
    deg = ppi_degree()
    ubi = set()
    up = SP / "ubiquitous_genes.txt"
    if up.exists():
        ubi = {x.strip() for x in open(up) if x.strip()}
    F, names = [], ["prot_mean", "prot_sd", "rna_mean", "rna_sd", "n_samples",
                    "log_ppi_degree", "ubiquitous"]
    for g in genes:
        v = merged[g][0]
        F.append([v["prot_mean"], v["prot_sd"], v["rna_mean"], v["rna_sd"], v["n"],
                  np.log1p(deg.get(g, 0)), 1.0 if g in ubi else 0.0])
    F = np.array(F)
    print(f"  features: {names}   PPI degrees found for "
          f"{int((F[:,5]>0).sum()):,}/{len(genes):,} genes; ubiquitous flag on {int(F[:,6].sum()):,}")

    # gene symbols are cached too: without them no external per-gene annotation can be joined on,
    # and the first cache written here omitted them, which is exactly why the biology arm was empty.
    np.savez(OUT / "_mpg_cache.npz", F=F, rho=rho, names=np.array(names),
             genes=np.array(genes), rho_sd=np.array([np.std([v["rho"] for v in merged[g]])
                                                     for g in genes]), n_cohorts=nco)

    # ---- the gate: predict rho out-of-sample ----
    print("\n  PREDICTING per-gene rho, held-out (5 random 80/20 splits)")
    real, ctrl = [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(genes))
        cut = int(0.8 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        m = xgb.XGBRegressor(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=2.0, n_jobs=4)
        m.fit(F[tr], rho[tr])
        real.append(r2_score(rho[te], m.predict(F[te])))
        ys = rho[rng.permutation(len(rho))]           # shuffled target: the null
        m2 = xgb.XGBRegressor(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=2.0, n_jobs=4)
        m2.fit(F[tr], ys[tr])
        ctrl.append(r2_score(ys[te], m2.predict(F[te])))
    real, ctrl = np.array(real), np.array(ctrl)
    print(f"    held-out R2 predicting rho : {real.mean():+.4f}  "
          f"[{' '.join(f'{x:+.3f}' for x in real)}]")
    print(f"    shuffled-target control    : {ctrl.mean():+.4f}")
    net = real.mean() - ctrl.mean()
    print(f"    NET                        : {net:+.4f}")

    R = {"cohorts": per_cohort, "n_genes_pooled": len(genes),
         "median_rho": float(np.median(rho)),
         "iqr": [float(np.percentile(rho, 25)), float(np.percentile(rho, 75))],
         "frac_rho_above_0.5": float(np.mean(rho > 0.5)),
         "frac_rho_negative": float(np.mean(rho < 0)),
         "predict_r2": float(real.mean()), "predict_r2_control": float(ctrl.mean()),
         "predict_net": float(net), "features": names}
    R["verdict"] = (
        "PREDICTABLE -- per-gene mRNA-protein agreement can be estimated from gene properties, so mRNA can "
        "be given a per-node confidence and propagated on the network."
        if net >= 0.10 else
        "NOT PREDICTABLE -- per-gene mRNA-protein agreement cannot be estimated from these properties "
        f"(held-out R2 {real.mean():+.4f}). Laying protein on the network needs MEASURED protein per "
        "knockout, which does not exist at proteome scale; mRNA cannot be corrected into it.")
    print(f"\n  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "mrna_protein_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'mrna_protein_gate.json'}")


if __name__ == "__main__":
    main()


def attribution():
    """WHICH FEATURE CARRIES THE +0.27? The caveat that decides how to read the verdict.

    The full model used five usable features -- protein mean/sd, mRNA mean/sd, n_samples -- because PPI
    degree and the ubiquitous flag came back empty (0 of 11,636 genes matched). All five are abundance or
    variance quantities, and highly abundant proteins are simply MEASURED better by mass spectrometry. So
    the +0.27 could be "we can predict which genes are well quantified" rather than "we can predict which
    genes have biologically tight mRNA-protein coupling". Those imply very different things for the
    network: the first gives a per-node CONFIDENCE, the second a per-node CORRECTION.

    Single-feature and leave-one-out held-out R2 separates them.
    """
    import numpy as np
    from sklearn.metrics import r2_score
    import xgboost as xgb
    import json as _json

    print("\n" + "=" * 100)
    print("ATTRIBUTION -- is the +0.27 biology or assay quality?")
    print("=" * 100)
    cache = OUT / "_mpg_cache.npz"
    if not cache.exists():
        print("  (no cached matrix; run main() first)")
        return
    z = np.load(cache, allow_pickle=True)
    F, rho, names = z["F"], z["rho"], list(z["names"])

    def held(cols):
        sc = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(rho))
            cut = int(0.8 * len(idx))
            tr, te = idx[:cut], idx[cut:]
            m = xgb.XGBRegressor(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=2.0, n_jobs=4)
            m.fit(F[np.ix_(tr, cols)], rho[tr])
            sc.append(r2_score(rho[te], m.predict(F[np.ix_(te, cols)])))
        return float(np.mean(sc))

    allc = list(range(F.shape[1]))
    full = held(allc)
    print(f"  all features            R2 {full:+.4f}")
    print("\n  single feature alone:")
    singles = {}
    for i, n in enumerate(names):
        if F[:, i].std() == 0:
            print(f"    {n:16s} constant, skipped")
            continue
        singles[n] = held([i])
        print(f"    {n:16s} R2 {singles[n]:+.4f}")
    print("\n  leave-one-out:")
    for i, n in enumerate(names):
        if F[:, i].std() == 0:
            continue
        cols = [j for j in allc if j != i]
        print(f"    without {n:16s} R2 {held(cols):+.4f}")
    best = max(singles, key=singles.get) if singles else None
    if best:
        frac = singles[best] / full if full > 0 else float("nan")
        print(f"\n  best single feature: {best} at {singles[best]:+.4f} = {100*frac:.0f}% of the full "
              f"{full:+.4f}")
        verdict = ("ASSAY QUALITY -- abundance alone reproduces most of the signal, so this is a per-node "
                   "CONFIDENCE (where mRNA is trustworthy), not a per-node CORRECTION."
                   if best.startswith("prot") and frac > 0.6 else
                   "MIXED -- no single abundance feature reproduces the fit, so some of it is structure "
                   "beyond measurement depth.")
        print(f"  READING: {verdict}")
        p = OUT / "mrna_protein_gate.json"
        d = _json.load(open(p)) if p.exists() else {}
        d["attribution"] = {"full_r2": full, "singles": singles, "best": best,
                            "best_frac_of_full": float(frac), "reading": verdict}
        _json.dump(d, open(p, "w"), indent=1)
        print(f"  -> updated {p}")


if __name__ == "__main__" and os.environ.get("MPG_ATTRIB") == "1":
    attribution()
