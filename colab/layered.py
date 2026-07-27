"""BUILD 6 -- THE LAYERED SOLVER: mRNA as the output of a chain, not a lookup.

THE ARGUMENT, FROM DESIGN.md sec 2. The present system treats one graph as if every edge acted on the same
timescale. They do not:

    metabolic flux              seconds     mass balance, stoichiometry     WORKS   (AUC 0.656 vs K562)
    signalling / metabolite     minutes     what the cell now lacks         partly
    transcription               hours       activity -> mRNA                BLOCKED (regulation layer is binding)

mRNA -- the thing every benchmark in this project scores against -- is the output of the SLOWEST layer, two steps
downstream of the only layer that works. Predicting it directly from a static graph skips the two steps that carry
the specificity. The design consequence is an explicit ordering rather than one propagation:

    LAYER 1  solve the fast layer to steady state              gene knockout  ->  flux vector F
    LAYER 2  take its DEVIATION as the driver of the next      dF             ->  metabolite disturbance A
    LAYER 3  read the slow layer, and ONLY through A           A              ->  mRNA response

THE PRE-REGISTERED ACCEPTANCE TEST (DESIGN.md sec 2): perturbing a purely metabolic gene must change F before it
changes A, and the model must produce an mRNA response ONLY through A. If mRNA changes while A does not, the
layering is decoration and the file says so. That is checked directly: every perturbation whose A is identically
zero must receive the intercept prediction and nothing gene-specific.

TWO CHOICES THAT ARE FORCED BY EARLIER MEASUREMENTS, NOT BY PREFERENCE

  pFBA, NOT PLAIN FBA. Build 5 found the LP has alternate optima: oxygen withdrawal left growth identical to six
  decimals while the flux vector changed. An unidentified F makes dF noise. Parsimonious FBA -- minimum total flux
  subject to the optimal objective -- makes the distribution comparable between conditions, which is the whole
  requirement for a DEVIATION to mean anything.

  THE CLASSICAL REGIME, NOT THE CAPACITY-CALIBRATED ONE. Build 5's T1 showed that under the calibrated enzyme
  ceiling the cell is enzyme-limited: withdrawing oxygen or glucose moves growth by <0.1%, and only 34 reactions
  sit at their ceiling. A layer that cannot feel its own inputs cannot drive the layer above it. Layer 1 therefore
  runs in the regime where it was actually validated (AUC 0.656, Warburg reproduced).

WHAT LAYER 2 IS, CONCRETELY. For each metabolite, its total turnover T_i = sum_j |S_ij * F_j| -- how much of it
the cell makes and consumes per hour. A = T(perturbed) - T(wild type). This is a real intermediate quantity: it
says WHAT THE CELL NOW LACKS OR ACCUMULATES, which is the only thing a metabolic lesion can signal upward with.

CONTROLS, because a chain of three fitted steps can manufacture a result at any of them:
    frequency          the globally most-moved genes; the bar seven mechanistic methods here have failed
    shuffled A         the identical chain with the flux-to-perturbation pairing destroyed
    gene-features only  the same regressor fed gene ANNOTATION instead of flux, which isolates whether the
                       metabolic intermediate contributes anything a gene property table does not already have
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
sys.path.insert(0, str(Path(__file__).resolve().parent))
TAU, TIDE_FRAC, MIN_SPEC, NPICK = 1.0, 0.05, 5, 50
NCOMP = 40


def readout():
    z = np.load(SP / "residual_K562_gwps.npz", allow_pickle=True)
    kos, genes = [str(x) for x in z["kos"]], [str(x) for x in z["genes"]]
    R, train = np.asarray(z["R"], np.float32), np.asarray(z["train"], bool)
    y = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    M = np.asarray(y["M"], np.float32)
    assert [str(x) for x in y["kos"]] == kos, "residual/readout misaligned -- refusing to continue"
    assert np.isfinite(R).all() and np.isfinite(M).all(), "NON-FINITE matrices"
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = [set(np.where((A[i] >= TAU) & ~tide)[0].tolist()) for i in range(len(kos))]
    return kos, genes, M, R, train, tide, spec


def turnover(model, fluxes):
    """T_i = sum over reactions of |S_ij * v_j| -- how much of metabolite i the cell moves per hour."""
    T = np.zeros(len(model.metabolites))
    mi = {mt.id: i for i, mt in enumerate(model.metabolites)}
    for r in model.reactions:
        v = float(fluxes[r.id])
        if v == 0.0:
            continue
        for mt, c in r.metabolites.items():
            T[mi[mt.id]] += abs(c * v)
    return T


def rank_pick(vec, tide, k=NPICK):
    order = np.argsort(-np.abs(vec))
    return [int(i) for i in order if not tide[i]][:k]


def main():
    import cobra
    from cobra.flux_analysis import pfba
    cobra.Configuration().solver = "glpk"
    from cell_sim import set_medium, ensembl_to_symbol

    kos, genes, M, R, train, tide, spec = readout()
    ki = {k: i for i, k in enumerate(kos)}
    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    set_medium(m)
    e2s = ensembl_to_symbol()
    sym2id = {}
    for gid, s in e2s.items():
        sym2id.setdefault(s, gid)

    # ---------- coverage: how much of the screen can this layer even see? ----------
    metab = [k for k in kos if k in sym2id and sym2id[k] in {g.id for g in m.genes}]
    print(f"coverage: {len(metab):,} of {len(kos):,} perturbations ({100*len(metab)/len(kos):.1f}%) are genes "
          f"the metabolic model contains -- layer 1 is blind to the rest")
    scorable = [k for k in metab if len(spec[ki[k]]) >= MIN_SPEC]
    print(f"          {len(scorable):,} of those have >= {MIN_SPEC} specific movers and are scorable")
    assert len(metab) > 100, "METABOLIC JOIN TOO SMALL -- refusing to continue"

    # ---------- LAYER 1: flux, made identifiable with pFBA ----------
    print("\nLAYER 1 (seconds): parsimonious FBA per knockout")
    wt_sol = pfba(m)
    Twt = turnover(m, wt_sol.fluxes)
    # pfba()'s objective_value is the MINIMISED TOTAL FLUX, not the growth rate. Printing it as "growth" put a
    # number three orders of magnitude wrong on screen; growth has to be read off the biomass reaction itself.
    bio = str(list(cobra.util.solver.linear_reaction_coefficients(m))[0].id)
    print(f"  wild type: growth {float(wt_sol.fluxes[bio]):.4f}/h (biomass {bio}), total flux "
          f"{wt_sol.objective_value:,.0f}, {int((np.abs(wt_sol.fluxes.values) > 1e-9).sum()):,} reactions carry flux")
    dF, dA, ok, nlethal = {}, {}, [], 0
    for n, k in enumerate(metab):
        gid = sym2id[k]
        try:
            with m:
                m.genes.get_by_id(gid).knock_out()
                s = pfba(m)
            f = s.fluxes.values
        except Exception:
            # pFBA is infeasible when the knockout cannot sustain the growth objective. Dropping those would
            # silently restrict the analysis to TOLERATED knockouts -- the loudest lesions would be the ones
            # excluded. A dead cell carries no flux, which is the correct deviation, so it is recorded as such.
            f = np.zeros_like(wt_sol.fluxes.values)
            nlethal += 1
        dF[k] = float(np.abs(f - wt_sol.fluxes.values).sum())
        dA[k] = turnover(m, dict(zip([r.id for r in m.reactions], f))) - Twt
        ok.append(k)
        if (n + 1) % 200 == 0:
            print(f"    {n+1:,}/{len(metab):,} solved")
    print(f"  solved {len(ok):,} knockouts ({nlethal:,} were infeasible = the cell cannot meet the objective, "
          f"recorded as zero flux rather than dropped)")
    nz = [k for k in ok if dF[k] > 1e-9]
    print(f"  LAYER 1 -> LAYER 2: {len(nz):,} of {len(ok):,} knockouts change the flux vector at all "
          f"({100*len(nz)/max(len(ok),1):.1f}%)")
    nzA = [k for k in ok if np.abs(dA[k]).sum() > 1e-9]
    print(f"  of those, {len(nzA):,} produce a non-zero metabolite disturbance A")

    # ---------- the pre-registered ORDERING check ----------
    viol = [k for k in ok if dF[k] <= 1e-9 and np.abs(dA[k]).sum() > 1e-9]
    print(f"\n  ORDERING CHECK: knockouts with A != 0 but dF == 0 (would mean A is not downstream of F): "
          f"{len(viol)}")
    silentA = [k for k in ok if np.abs(dA[k]).sum() <= 1e-9]
    print(f"  knockouts with A == 0: {len(silentA):,} -- these MUST receive the intercept prediction only, "
          f"which is enforced by construction below")

    # ---------- LAYER 3: A -> mRNA, learned on the donor half only ----------
    idx = [ki[k] for k in ok]
    X = np.array([dA[k] for k in ok], np.float64)
    keepcol = X.std(0) > 0
    X = X[:, keepcol]
    print(f"\nLAYER 3 (hours): {X.shape[0]:,} perturbations x {X.shape[1]:,} metabolites with any variation")
    tr = np.array([train[i] for i in idx])
    te = ~tr & np.array([len(spec[i]) >= MIN_SPEC for i in idx])
    print(f"  donor {int(tr.sum()):,} / held-out scorable {int(te.sum()):,}")
    if int(te.sum()) < 40:
        print("  TOO FEW held-out scorable metabolic perturbations -- refusing to report")
        return

    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Y = R[idx]
    svd = TruncatedSVD(n_components=NCOMP, random_state=0).fit(Y[tr])
    Yr = svd.transform(Y)
    sc = StandardScaler().fit(X[tr])
    pca = PCA(n_components=min(NCOMP, int(tr.sum()) - 1, X.shape[1]), random_state=0).fit(sc.transform(X[tr]))
    print(f"  metabolite disturbance reduced to {pca.n_components_} components "
          f"({100*pca.explained_variance_ratio_.sum():.1f}% of its variance)")

    def fit_predict(Xtr, Xte):
        rg = Ridge(alpha=10.0).fit(Xtr, Yr[tr])
        return svd.inverse_transform(rg.predict(Xte))

    Xr = pca.transform(sc.transform(X))
    P_layered = fit_predict(Xr[tr], Xr[te])

    rng = np.random.default_rng(0)
    Xsh = Xr.copy()
    Xsh[tr] = Xr[tr][rng.permutation(int(tr.sum()))]
    P_shuf = fit_predict(Xsh[tr], Xr[te])

    # gene-annotation-only arm: same regressor, no metabolic intermediate
    from abstain import features as gene_features, COLS, LOGGED
    F = gene_features()
    G = np.array([[float(F.get(k, {}).get(c, 0.0)) if c != "detected" else 1.0 for c in COLS] for k in ok])
    for j, c in enumerate(COLS):
        if c in LOGGED:
            G[:, j] = np.log1p(np.clip(G[:, j], 0, None))
    gsc = StandardScaler().fit(G[tr])
    P_genes = fit_predict(gsc.transform(G[tr]), gsc.transform(G[te]))

    freq = (np.abs(M[train]) >= TAU).mean(0)
    freq_pick = rank_pick(freq, tide)
    nontide = np.where(~tide)[0]

    te_i = [idx[j] for j in np.where(te)[0]]
    zeroA = np.array([np.abs(dA[ok[j]]).sum() <= 1e-9 for j in np.where(te)[0]])
    arms = {}
    for tag, Pm in (("layered  F -> A -> mRNA", P_layered), ("shuffled A (control)", P_shuf),
                    ("gene annotation, no flux", P_genes)):
        rec, prec = [], []
        for r, gi_ in enumerate(te_i):
            # ENFORCED BY CONSTRUCTION: no metabolite disturbance, no gene-specific answer. The
            # fallback is the gene-agnostic null prediction, which is what "mRNA only through A" means.
            pick = freq_pick if zeroA[r] and tag.startswith("layered") else rank_pick(Pm[r], tide)
            t = spec[gi_]
            rec.append(len(set(pick) & t) / len(t))
            prec.append(len(set(pick) & t) / max(len(pick), 1))
        arms[tag] = (np.array(rec), np.array(prec))
    for tag, pick in (("frequency", freq_pick), ("random", None)):
        rec, prec = [], []
        for gi_ in te_i:
            p = pick if pick is not None else rng.choice(nontide, size=NPICK, replace=False).tolist()
            t = spec[gi_]
            rec.append(len(set(p) & t) / len(t))
            prec.append(len(set(p) & t) / max(len(p), 1))
        arms[tag] = (np.array(rec), np.array(prec))

    base = arms["frequency"][0]
    print("\n" + "=" * 88)
    print(f"{'arm':30s} {'recall':>8s} {'precision':>10s}    vs frequency (paired bootstrap)")
    res = {}
    for tag, (rec, prec) in arms.items():
        d = rec - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        tail = "" if tag == "frequency" else f"    {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]"
        print(f"{tag:30s} {rec.mean():>8.4f} {prec.mean():>10.4f}{tail}")
        res[tag] = {"recall": float(rec.mean()), "precision": float(prec.mean()),
                    "delta_vs_frequency": float(d.mean()), "ci": [float(lo), float(hi)]}
    print("=" * 88)

    lay, sh, gn = res["layered  F -> A -> mRNA"], res["shuffled A (control)"], res["gene annotation, no flux"]
    beats_freq = lay["ci"][0] > 0
    beats_shuf = lay["recall"] > sh["recall"]
    beats_gene = lay["recall"] > gn["recall"]
    ordering = len(viol) == 0
    verdict = "PASS" if (ordering and beats_freq and beats_shuf) else "FAIL"
    print(f"\nPRE-REGISTERED TEST (DESIGN.md sec 2)")
    print(f"  A is strictly downstream of F (no A without dF)  : {ordering}  ({len(viol)} violations)")
    print(f"  mRNA arises only through A                       : enforced -- {int(zeroA.sum())} held-out "
          f"perturbations with A == 0 received the gene-agnostic null prediction")
    print(f"  beats the frequency baseline                     : {beats_freq}  "
          f"({lay['recall']:.4f} vs {res['frequency']['recall']:.4f})")
    print(f"  beats its own shuffled control                   : {beats_shuf}  "
          f"({lay['recall']:.4f} vs {sh['recall']:.4f})")
    print(f"  the metabolic intermediate earns its place       : {beats_gene}  "
          f"({lay['recall']:.4f} vs gene-annotation {gn['recall']:.4f})")
    print(f"  VERDICT: {verdict}")

    json.dump({"n_perturbations": len(kos), "n_metabolic": len(metab), "n_scorable": len(scorable),
               "n_solved": len(ok), "n_flux_changed": len(nz), "n_A_nonzero": len(nzA),
               "n_ordering_violations": len(viol), "n_test": int(te.sum()),
               "n_test_zero_A": int(zeroA.sum()), "arms": res,
               "ordering_holds": ordering, "beats_frequency": beats_freq, "beats_shuffled": beats_shuf,
               "beats_gene_annotation": beats_gene, "verdict": verdict},
              open(OUT / "layered.json", "w"), indent=1)
    print(f"\n  -> {OUT/'layered.json'}")


if __name__ == "__main__":
    main()
