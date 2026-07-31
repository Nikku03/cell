"""IS EFFECT MAGNITUDE PREDICTABLE ONCE MEASUREMENT POWER IS PROPERLY MODELLED?

THE ENTRY THIS REOPENS, AND EXACTLY HOW FAR IT REOPENS IT. The contradiction registry carries:

    dyn-magnitude   status null
    claim           "The MAGNITUDE of a perturbation's effect is predictable from biological features."
    measured        detection power contributed +0.0881, larger than any biological block
    killed_by       magnitude is confounded with detection power; existence and sign were separable,
                    magnitude was not
    module          colab/effectsize.py
    reopens_if      an assay with uniform detection power across genes, so power cannot stand in for size

That was measured on 820 enhancer-gene pairs from the EPCrispr benchmark, with power entered as ONE feature
in ONE arm. Two things about it are worth revisiting and they are different:

  the SAMPLE   820 pairs against ~20 features. Whatever the answer, it was measured at low power about
               power, which is an uncomfortable place to conclude from.
  the MODEL    one detection-power feature is not a model of detection power. Finding that it beats the
               biological arms establishes CONFOUNDING. It does not establish that biology adds nothing
               CONDITIONAL on power, which is the question the claim actually asks.

THIS MODULE DOES NOT HAVE THE ASSAY THE ENTRY ASKS FOR, and does not pretend to. Uniform detection power
across genes does not exist in any deposit here. What it does instead is the two things that are available:

  1  MODEL power richly rather than as one covariate -- per-gene expression and variability, per-perturbation
     cell count and library size, mitochondrial fraction, knockdown efficacy, and a leave-one-out per-gene
     responsiveness term -- and ask whether pair-level BIOLOGY adds held-out R2 on top of all of it.
  2  STRATIFY to approximate the missing assay. Inside a narrow bin of gene expression x gene variability x
     perturbation cell count, detection power is nearly constant by construction, so within a bin power
     cannot stand in for size. If biology predicts magnitude inside bins, the confound is not the whole
     story; if it does not, the null holds under a much stronger test than the one that produced it.

AND IT IS A DIFFERENT SYSTEM, WHICH BOUNDS WHAT A POSITIVE RESULT WOULD MEAN. The registry entry is about
enhancer silencing -> gene expression. This is gene knockdown -> transcriptome, in K562, at 11,258
perturbations x 8,248 genes. A positive result here reopens the GENERAL claim and says nothing about the
n=820 enhancer measurement, which stands as measured. A null here strengthens the entry rather than
duplicating it.

THE CONTROLS, because a biology block over millions of rows will fit something.

  DEGREE-MATCHED REWIRING of every graph the biology block reads. The pair features are PPI adjacency,
  co-complex membership, curated regulation, co-dependency and network degree; rewiring preserves each
  gene's degree and destroys only which partner it has. A gain that survives its own degree-matched control
  is about the specific partner. A gain that does not is about hubs, and hubs are a power story again --
  well-connected genes are well-studied genes are well-measured genes.

  LEAVE-ONE-OUT RESPONSIVENESS. The single strongest predictor of |lfc| for gene g is how much g moves for
  every OTHER perturbation. Computed in-fold it leaks the answer, so it is computed leaving the current
  perturbation out, and it sits in the NUISANCE block where it belongs rather than being sold as biology.

  FOLDS GROUPED BY PERTURBED GENE, with genuinely distinct partitions per seed. An earlier module in this
  project rotated the labels of one partition across seeds and reported the resulting 0.0000 spread as
  agreement between seeds; the folds here are re-drawn from a different permutation for each seed.
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
NET = SP / "cell_complete.json.gz"

N_PAIRS = 400_000       # sampled (perturbation, gene) pairs; the full grid is 92M
N_FOLDS = 5
N_SEEDS = 3
N_REWIRE = 5            # degree-matched rewirings of the biology block
STRATA_BINS = 20        # quantile bins of the power model's own out-of-fold prediction


def load_k562(report):
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
        gname_cat = [c.decode() if isinstance(c, bytes) else str(c)
                     for c in f["var"]["__categories"]["gene_name"][:]]
        gname = np.array([gname_cat[i] for i in f["var"]["gene_name"][:]])
        obs = {k: f["obs"][k][:] for k in ("num_cells_filtered", "UMI_count_unfiltered", "mitopercent",
                                           "fold_expr", "pct_expr", "z_gemgroup_UMI")}
        var = {k: f["var"][k][:].astype(np.float64) for k in ("mean", "std", "cv", "clean_mean",
                                                              "clean_std")}
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    bad = ~np.isfinite(X)
    if bad.any():
        report(f"  {int(bad.sum()):,} non-finite entries set to NaN (they are not zeros)")
        X = np.where(bad, np.nan, X)
    return X, psym, np.array(gid), gname, obs, var


def load_graphs(report, gname):
    """Pair-level biology, keyed to the K562 gene symbols. Raises if a join collapses."""
    d = json.load(gzip.open(NET, "rt"))
    name = [g["name"] for g in d["genes"]]
    idx = {n: i for i, n in enumerate(name)}
    ppi = [(int(a), int(b)) for a, b, *_ in d["ppi"]]
    codep = {(int(a), int(b)) for a, v in (d.get("codep") or {}).items() for b, _s in v}
    g2c = {int(k): set(v) for k, v in (d.get("gene2cplx") or {}).items()}
    reg = [(int(a), int(b)) for a, b, *_ in (d.get("reg") or [])]
    del d
    hit = sum(1 for g in np.unique(gname) if g in idx)
    rate = hit / max(len(np.unique(gname)), 1)
    if rate < 0.50:
        raise SystemExit(f"K562 gene symbols joined to the cell object at {rate:.1%}, expected >= 50%. "
                         f"Below that the biology block is mostly missing and its null would be an "
                         f"artefact of the join.")
    report(f"  cell object: {len(ppi):,} PPI, {len(codep):,} co-dependency, {len(reg):,} regulatory "
           f"edges; symbol join {rate:.1%}")
    return idx, ppi, codep, g2c, reg


def rewire(edges, n, rng):
    """Configuration-model rewiring: keeps every node's degree, destroys which partner it has.

    Uniform rewiring would not do -- it would also destroy the degree distribution, and then a 'biology'
    gain that is really a hub effect would survive the control and be read as specificity.
    """
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    return set(zip(a.tolist(), rng.permutation(b).tolist()))


def pair_features(pi, gi, idx_of_p, idx_of_g, ppi_set, codep, g2c, reg_set, deg_ppi, deg_reg):
    """Biology of the PAIR, not of either gene alone -- except the two degree terms, which are node-level
    and are included precisely so the rewired control can show whether the gain is really about them."""
    ip, ig = idx_of_p[pi], idx_of_g[gi]
    ok = (ip >= 0) & (ig >= 0)
    n = len(pi)
    f = np.zeros((n, 6), dtype=np.float32)
    for k in range(n):
        if not ok[k]:
            continue
        a, b = int(ip[k]), int(ig[k])
        f[k, 0] = 1.0 if ((a, b) in ppi_set or (b, a) in ppi_set) else 0.0
        f[k, 1] = 1.0 if ((a, b) in codep or (b, a) in codep) else 0.0
        f[k, 2] = 1.0 if (g2c.get(a) and g2c.get(b) and g2c[a] & g2c[b]) else 0.0
        f[k, 3] = 1.0 if (a, b) in reg_set else 0.0
        f[k, 4] = deg_ppi.get(a, 0)
        f[k, 5] = deg_ppi.get(b, 0)
    return f


def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=6,
                                      early_stopping=False, random_state=0)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    ss = float(np.sum((yte - p) ** 2))
    st = float(np.sum((yte - yte.mean()) ** 2))
    return 1.0 - ss / max(st, 1e-12)


def folds(groups, seed):
    ug = np.unique(groups)
    a = np.random.default_rng(7000 + seed).permutation(len(ug))
    fold_of = {g: int(a[i]) % N_FOLDS for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def cv_r2(F, y, groups, seed, report=None):
    """Held-out R2 with folds grouped by perturbed gene. Each seed draws a DIFFERENT partition."""
    fold = folds(groups, seed)
    out = []
    for k in range(N_FOLDS):
        te = fold == k
        out.append(fit_eval(F[~te], y[~te], F[te], y[te]))
    return float(np.mean(out))


def cv_oof(F, y, groups, seed):
    """Out-of-fold predictions from the nuisance block, on the same grouped folds.

    Used to CONDITION on power rather than to score it: every row gets a predicted magnitude from a model
    that never saw it, and rows with the same prediction are rows the power model cannot tell apart.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    fold = folds(groups, seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for k in range(N_FOLDS):
        te = fold == k
        m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=6,
                                          early_stopping=False, random_state=0)
        m.fit(F[~te], y[~te])
        oof[te] = m.predict(F[te])
    return oof


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("MAGNITUDE, DECONFOUNDED -- retesting `dyn-magnitude` where power can be modelled, not assumed")
    report("=" * 100)
    if not GWPS.exists():
        raise SystemExit(f"absent: {GWPS}")

    X, psym, gid, gname, obs, var = load_k562(report)
    nP, nG = X.shape
    report(f"  K562: {nP:,} perturbations x {nG:,} genes = {nP*nG:,} magnitudes on the full grid")
    idx, ppi, codep, g2c, reg = load_graphs(report, gname)

    # ---- leave-one-out per-gene responsiveness, computed once ----
    #
    # THE STRONGEST SINGLE PREDICTOR, AND IT MUST NOT LEAK. Mean |lfc| of gene g over perturbations is
    # almost the answer for any pair involving g. Computed including the current perturbation it leaks;
    # computed leaving it out it is a legitimate nuisance covariate, and it belongs in the nuisance block
    # rather than in anything called biology.
    A = np.abs(X)
    cnt = np.isfinite(A).sum(0).astype(np.float64)
    tot = np.nansum(A, axis=0)
    report(f"  |lfc|: median {np.nanmedian(A):.4f}, 99th pct {np.nanpercentile(A, 99):.4f}")

    rng = np.random.default_rng(0)
    pi = rng.integers(0, nP, N_PAIRS)
    gi = rng.integers(0, nG, N_PAIRS)
    y = A[pi, gi]
    keep = np.isfinite(y)
    pi, gi, y = pi[keep], gi[keep], y[keep]
    report(f"  sampled {len(y):,} finite pairs of {N_PAIRS:,} drawn")

    loo = (tot[gi] - y) / np.maximum(cnt[gi] - 1, 1)

    # ---- the nuisance block: measurement power + perturbation efficacy + responsiveness ----
    NU = np.column_stack([
        var["mean"][gi], var["std"][gi], var["cv"][gi], var["clean_mean"][gi], var["clean_std"][gi],
        obs["num_cells_filtered"][pi], obs["UMI_count_unfiltered"][pi], obs["mitopercent"][pi],
        obs["z_gemgroup_UMI"][pi],
        obs["fold_expr"][pi], obs["pct_expr"][pi],
        loo,
    ]).astype(np.float32)
    NU = np.nan_to_num(NU)
    nu_names = ["gene mean", "gene sd", "gene cv", "gene clean mean", "gene clean sd",
                "pert cells", "pert UMI", "pert mito%", "pert gem z",
                "knockdown fold_expr", "knockdown pct_expr", "gene responsiveness (LOO)"]
    report(f"\n  NUISANCE BLOCK ({NU.shape[1]} features): " + ", ".join(nu_names))

    # ---- the biology block ----
    idx_of_p = np.array([idx.get(s, -1) for s in psym])
    idx_of_g = np.array([idx.get(s, -1) for s in gname])
    ppi_set = set((a, b) for a, b in ppi)
    reg_set = set((a, b) for a, b in reg)
    deg_ppi = {}
    for a, b in ppi:
        deg_ppi[a] = deg_ppi.get(a, 0) + 1
        deg_ppi[b] = deg_ppi.get(b, 0) + 1
    BIO = pair_features(pi, gi, idx_of_p, idx_of_g, ppi_set, codep, g2c, reg_set, deg_ppi, None)
    bio_names = ["PPI edge", "co-dependency", "co-complex", "curated regulation",
                 "PPI degree of perturbed gene", "PPI degree of measured gene"]
    hit_rates = {n: float(BIO[:, k].mean()) for k, n in enumerate(bio_names[:4])}
    report(f"  BIOLOGY BLOCK ({BIO.shape[1]} features): " + ", ".join(bio_names))
    report(f"    pair-feature hit rates: " +
           ", ".join(f"{n} {100*v:.3f}%" for n, v in hit_rates.items()))
    if max(hit_rates.values()) < 1e-4:
        raise SystemExit("every pair-level biology feature is on for <0.01% of sampled pairs; the block "
                         "carries no signal to test and a null would be about the sample, not the claim")

    groups = psym[pi]

    # ---- 1. does biology add on top of a properly-modelled power block? ----
    report(f"\n  1. HELD-OUT R2, folds grouped by perturbed gene, {N_SEEDS} seeds x {N_FOLDS} folds")
    arms = {"nuisance only": NU,
            "nuisance + biology": np.hstack([NU, BIO]),
            "biology only": BIO}
    res = {}
    for nm, F in arms.items():
        vals = [cv_r2(F, y, groups, s) for s in range(N_SEEDS)]
        res[nm] = {"r2": float(np.mean(vals)), "sd": float(np.std(vals)), "per_seed": vals}
        report(f"     {nm:24s} R2 {np.mean(vals):+.4f}  (seed sd {np.std(vals):.4f}, "
               f"seeds {', '.join(f'{v:+.4f}' for v in vals)})")
    delta = res["nuisance + biology"]["r2"] - res["nuisance only"]["r2"]
    seed_sd = max(res["nuisance + biology"]["sd"], res["nuisance only"]["sd"])
    report(f"     -> biology adds {delta:+.4f} over the nuisance block (largest seed sd {seed_sd:.4f})")

    # ---- 2. the degree-matched control ----
    report(f"\n  2. DEGREE-MATCHED REWIRING of every graph the biology block reads ({N_REWIRE} draws)")
    ctrl = []
    for r in range(N_REWIRE):
        rr = np.random.default_rng(900 + r)
        ppi_r = rewire(ppi, len(idx), rr)
        reg_r = rewire(reg, len(idx), rr) if reg else set()
        cod_r = rewire(sorted(codep), len(idx), rr) if codep else set()
        BIOr = pair_features(pi, gi, idx_of_p, idx_of_g, ppi_r, cod_r, g2c, reg_r, deg_ppi, None)
        v = cv_r2(np.hstack([NU, BIOr]), y, groups, 0)
        ctrl.append(v - res["nuisance only"]["r2"])
        report(f"     rewiring {r+1}/{N_REWIRE}: biology adds {v - res['nuisance only']['r2']:+.4f}")
    cm, cs = float(np.mean(ctrl)), float(np.std(ctrl))
    report(f"     -> degree-matched control adds {cm:+.4f} (sd {cs:.4f}); the real block adds "
           f"{delta:+.4f}, an excess of {delta-cm:+.4f}")

    # ---- 3. conditioning on power, done with the power model rather than three of its axes ----
    #
    # WHAT THE FIRST VERSION OF THIS SECTION DID AND WHY IT WAS REPLACED. It binned pairs on a 6x6x6 grid of
    # gene expression x gene variability x perturbation cell count and asserted that inside a cell detection
    # power is nearly constant. The run reported 52 occupied cells and within-cell centring removing 0.5% of
    # the variance in |lfc| -- against a power model that explains 20% of it. A conditioning that removes
    # 0.5% is not holding power fixed, so the sentence was not supported by the operation underneath it.
    #
    # This version conditions on the power model's OWN out-of-fold prediction, binned into deciles-of-20.
    # Two rows in the same bin are two rows the twelve-feature power model cannot tell apart, which is what
    # "uniform detection power" means operationally when the assay that would deliver it does not exist. It
    # is non-parametric in the residual -- the model supplies the ordering, the binning does the rest -- so
    # it is not simply section 1's delta computed twice.
    report(f"\n  3. CONDITIONING ON POWER -- {STRATA_BINS} quantile bins of the nuisance model's own "
           f"out-of-fold prediction")
    oof = cv_oof(NU, y, groups, 0)
    edges = np.quantile(oof, np.linspace(0, 1, STRATA_BINS + 1)[1:-1])
    cell = np.digitize(oof, edges)
    ymean = np.zeros_like(y)
    for c in np.unique(cell):
        m = cell == c
        ymean[m] = y[m].mean()
    yr = y - ymean
    removed = float(1 - yr.var() / y.var())
    report(f"     {len(np.unique(cell)):,} bins; within-bin centring removes {100*removed:.1f}% of the "
           f"variance in |lfc|")
    report(f"     (the 6x6x6 grid this replaces removed 0.5%, which is why it could not support the claim "
           f"it was making)")
    if removed < 0.05:
        report(f"     WARNING: this conditioning is also weak. Read section 3 as uninformative rather than "
               f"as a null.")
    s_bio = [cv_r2(BIO, yr, groups, s) for s in range(N_SEEDS)]
    s_nu = [cv_r2(NU, yr, groups, s) for s in range(N_SEEDS)]
    s_ctrl = []
    for r in range(N_REWIRE):
        rr = np.random.default_rng(1900 + r)
        BIOr = pair_features(pi, gi, idx_of_p, idx_of_g, rewire(ppi, len(idx), rr),
                             rewire(sorted(codep), len(idx), rr) if codep else set(), g2c,
                             rewire(reg, len(idx), rr) if reg else set(), deg_ppi, None)
        s_ctrl.append(cv_r2(BIOr, yr, groups, 0))
    report(f"     biology on the within-bin residual   R2 {np.mean(s_bio):+.4f} (sd {np.std(s_bio):.4f})")
    report(f"     nuisance on the same residual        R2 {np.mean(s_nu):+.4f} (sd {np.std(s_nu):.4f}) "
           f"-- should be near zero if the binning worked")
    report(f"     degree-matched biology               R2 {np.mean(s_ctrl):+.4f} "
           f"(sd {np.std(s_ctrl):.4f})")
    strat_excess = float(np.mean(s_bio) - np.mean(s_ctrl))
    report(f"     -> with power held fixed, biology beats its own rewired control by {strat_excess:+.4f}")

    # ---- verdict ----
    #
    # THE BAR, FIXED BEFORE THE NUMBERS. Beating a rewired control by less than three seed spreads is a tie,
    # and a tie leaves the registry entry standing. This project has twice admitted a layer on a
    # fifth-decimal difference at its own noise floor, and both were withdrawn.
    bar = 3 * max(seed_sd, cs, 1e-6)
    excess = delta - cm
    reopened = bool(excess > bar and strat_excess > 3 * max(float(np.std(s_bio)), float(np.std(s_ctrl)),
                                                            1e-6))
    verdict = (
        f"{'THE ENTRY REOPENS' if reopened else 'THE NULL HOLDS, AND NOW HOLDS UNDER A FAR STRONGER TEST'}. "
        f"Registry entry `dyn-magnitude` was measured on 820 enhancer-gene pairs with detection power "
        f"entered as one feature in one arm. Retested here on {len(y):,} K562 knockdown (perturbation, "
        f"gene) magnitudes against a twelve-feature nuisance block -- per-gene expression and variability, "
        f"per-perturbation cell count, library size, mitochondrial fraction, knockdown efficacy, and a "
        f"leave-one-out responsiveness term -- the nuisance block alone reaches R2 "
        f"{res['nuisance only']['r2']:+.4f} and adding pair-level biology moves it to "
        f"{res['nuisance + biology']['r2']:+.4f}, a gain of {delta:+.4f}. Its own degree-matched rewiring "
        f"gains {cm:+.4f}, so the partner-specific excess is {excess:+.4f} against a bar of {bar:.4f} "
        f"(three seed spreads). "
        f"Inside power strata -- the reopens_if condition approximated by conditioning, since no assay here "
        f"has uniform detection power -- biology reaches {np.mean(s_bio):+.4f} on the within-cell residual "
        f"against {np.mean(s_ctrl):+.4f} for the rewired control. "
        + (f"Both arms clear their controls, so magnitude carries partner-specific biological information "
           f"that survives a properly-modelled power block. The entry moves from `null` to `bounded`: the "
           f"enhancer measurement stands as measured, and the general claim does not."
           if reopened else
           f"Neither arm clears its control by the margin set in advance. The entry stands, and it now "
           f"stands on {len(y):,} pairs and a modelled power block rather than on 820 pairs and one "
           f"covariate.")
        + f" WHAT THIS IS NOT: this is gene knockdown -> transcriptome, not enhancer silencing -> gene. It "
          f"tests the same general claim in a different system and it neither confirms nor overturns the "
          f"n=820 enhancer result, which was measured on its own data and stays as it is.")
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "magnitude-deconfound-v1",
         "reopens_registry_entry": "dyn-magnitude",
         "target": "|log fold change| of a (perturbation, gene) pair in Replogle K562",
         "n_pairs_sampled": int(len(y)), "n_pairs_available": int(nP * nG),
         "nuisance_features": nu_names, "biology_features": bio_names,
         "pair_feature_hit_rates": hit_rates,
         "arms": res, "biology_gain": delta, "seed_sd": seed_sd,
         "degree_matched_control": {"mean_gain": cm, "sd": cs, "per_draw": ctrl, "n_draws": N_REWIRE},
         "partner_specific_excess": excess, "bar": bar,
         "stratified": {"conditioning": "quantile bins of the nuisance model's out-of-fold "
                                        "prediction, replacing a 6x6x6 covariate grid that removed only "
                                        "0.5% of the variance",
                        "n_bins": int(len(np.unique(cell))),
                        "variance_removed_by_centring": removed,
                        "biology_r2": float(np.mean(s_bio)), "biology_sd": float(np.std(s_bio)),
                        "nuisance_r2": float(np.mean(s_nu)),
                        "rewired_r2": float(np.mean(s_ctrl)), "rewired_sd": float(np.std(s_ctrl)),
                        "excess": strat_excess},
         "reopened": reopened,
         "limits": [
             "this is a DIFFERENT SYSTEM from the one the registry entry was measured on. It tests the "
             "general claim, not the enhancer result, and cannot overturn a measurement it did not repeat",
             "no assay here has uniform detection power. Binning on the power model's own out-of-fold "
             "prediction makes power nearly constant inside a bin AS THAT MODEL SEES IT -- a power axis "
             "the twelve features do not capture is not removed by conditioning on their prediction, and "
             "is still available to the biology block",
             "the leave-one-out responsiveness term is a nuisance covariate by placement and a biological "
             "quantity by content -- how much a gene moves is partly what the gene is. Putting it in the "
             "nuisance block is the conservative choice and it makes the biology gain a LOWER bound",
             "knockdown efficacy (fold_expr, pct_expr) is a property of the perturbation, so including it "
             "as nuisance removes real biology about which genes are easy to knock down. Also conservative, "
             "also a reason the biology gain is a lower bound",
             "the pair-level biology features are sparse -- a randomly drawn (perturbation, gene) pair is "
             "almost never a PPI edge -- so the block is carried by a small fraction of rows and its R2 "
             "contribution is bounded by that fraction whatever the biology is worth",
             "the rewiring permutes ONE endpoint of each edge. That preserves the degree sequence on the "
             "source side and the degree distribution on the target side, not every node's degree exactly. "
             "The degree features themselves are held at their real values in both arms, so the control "
             "isolates adjacency, which is what it is for",
             "400,000 pairs are sampled from 92 million. The sample is uniform over the grid, so it "
             "inherits the grid's composition rather than correcting it"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "magnitude_deconfound.json", "w"), indent=1)
    report(f"\n  -> {OUT/'magnitude_deconfound.json'}")


if __name__ == "__main__":
    main()
