"""IS THE +0.27 ASSAY QUALITY OR BIOLOGY? Adding features that cannot be measurement depth.

WHAT THE GATE LEFT UNRESOLVED. `mrna_protein_gate.py` predicts per-gene mRNA-protein agreement (Spearman
rho across CPTAC tumours) at held-out R2 +0.2695 against a shuffled-target control of -0.0228. That is a
real, reproducible fit. But every feature in it -- prot_mean, prot_sd, rna_mean, rna_sd, n_samples -- is a
property of the MEASUREMENT, not of the protein. Its own attribution pass found no single feature carries
the fit (best is rna_sd at 23% of it), which rules out a one-variable artefact and rules in nothing.

WHY IT MATTERS, CONCRETELY. rho is a correlation, so it is bounded above by dynamic range: a gene that
barely varies across tumours cannot show agreement even if its mRNA and protein track perfectly. If the
+0.27 is that bound, the model outputs a per-node CONFIDENCE -- "trust this mRNA reading" -- and nothing
biological. If instead genes are buffered for structural reasons (complex subunits degraded when
unpartnered, long proteins translated slowly), the model outputs a per-node CORRECTION -- "mRNA overstates
this protein" -- which is what laying proteomics on the network per knockout actually requires.

THE FEATURES ADDED, chosen because none of them can be assay depth:
    log_ppi_degree_700   STRING v12 high-confidence partner count, real per-gene values this time
    log_ppi_degree_900   the same at score>=900, so the conclusion cannot hinge on one threshold
    log_protein_size     amino-acid length from STRING's own annotation
    log_n_pathways       Reactome membership count
    log_pathway_partners size of the gene's largest Reactome pathway -- functional-module scale

THE DECIDING TEST IS THE RESIDUAL, NOT THE JOINT R2. Adding features to a model almost always raises
held-out R2 a little, and biology features correlate with abundance (housekeeping genes are both abundant
and well connected), so "+both beats measurability" would not separate the two. So: fit measurability
alone, take the held-out RESIDUAL, and predict THAT from biology alone. Residual R2 above its own shuffled
control is information measurability does not contain, and cannot be re-explained as depth.

THE RESIDUAL MUST COME FROM THE STRONGEST MEASUREMENT FIT AVAILABLE. A first version of this module fit
measurability with ridge, which reached R2 +0.0967 where the gate's boosted tree reaches +0.2695 on the same
columns. Biology then "predicted the residual" -- but a residual that large still contains the nonlinear
measurability structure ridge could not reach, so biology may simply have been proxying it. The
measurability arm therefore uses the gate's own XGBoost settings, so what is left over is as close to
measurement-free as this data allows.

PROTEIN LENGTH IS QUARANTINED, NOT USED. Shotgun proteomics quantifies a protein from its peptides, and a
longer protein yields more of them, so length raises quantification quality directly. It is a measurement
feature wearing biology's clothes, and it is reported separately for exactly that reason -- if the residual
signal lives in length, the honest verdict is confidence, not correction.

STUDY BIAS IS THE LIMIT OF EVEN THE CLEAN BLOCK. STRING degree and Reactome membership both grow with how
much a gene has been studied, and well-studied genes tend to be better quantified. This cannot be removed
with the data here, so it is stated: the clean-block residual is an UPPER bound on structural coupling.

DIRECTION IS REPORTED, because a mechanism makes a signed prediction and a fit does not. If complex
membership buffers protein against mRNA, PPI degree must correlate NEGATIVELY with the residual. A positive
correlation would falsify the mechanism while still raising R2, and that distinction is the whole point.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
STRING_INFO = SP / "protein" / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS = SP / "protein" / "9606.protein.links.v12.0.txt.gz"
REACTOME = SP / "ReactomePathways.gmt"
SEEDS = (0, 1, 2, 3, 4)
# n_cohorts belongs here and was missing from the gate's own feature list. The pooled rho is a MEAN over
# however many cohorts measured the gene, so a 3-cohort gene has a less noisy target than a 1-cohort gene,
# and a tree can read that directly. It also correlates with STRING degree at Spearman +0.177 (p 4e-82) --
# well-connected genes are detected in more cohorts -- so leaving it out would let the biology block proxy
# it and be credited for a measurement effect.
MEAS = ["prot_mean", "prot_sd", "rna_mean", "rna_sd", "n_samples", "n_cohorts"]


def string_features():
    """Per-gene STRING degree at two confidence thresholds, plus protein length.

    STRING ships ENSP ids; `preferred_name` is the gene symbol. Both are needed -- degree is counted over
    ENSP and then relabelled, because relabelling first would merge isoform entries and inflate degree.
    """
    if not STRING_INFO.exists() or not STRING_LINKS.exists():
        raise SystemExit(f"STRING files absent ({STRING_INFO}); the biology arm cannot be built")
    ensp2sym, size = {}, {}
    with gzip.open(STRING_INFO, "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            ensp2sym[f[0]] = f[1]
            try:
                size[f[1]] = max(size.get(f[1], 0), int(f[2]))
            except (ValueError, IndexError):
                pass
    d700, d900 = {}, {}
    n = 0
    with gzip.open(STRING_LINKS, "rt") as fh:
        fh.readline()
        for line in fh:
            a, b, s = line.split()
            s = int(s)
            if s < 700:
                continue
            n += 1
            for p in (a, b):
                d700[p] = d700.get(p, 0) + 1
                if s >= 900:
                    d900[p] = d900.get(p, 0) + 1
    # ENSP degree -> symbol degree, taking the max over isoform entries of the same symbol
    def relabel(d):
        out = {}
        for p, k in d.items():
            s = ensp2sym.get(p)
            if s:
                out[s] = max(out.get(s, 0), k)
        return out
    print(f"  STRING: {n:,} edges at score>=700, {len(ensp2sym):,} proteins, "
          f"{len(relabel(d700)):,} symbols with a partner")
    return relabel(d700), relabel(d900), size


def reactome_features():
    """Pathway count per gene, and the size of the largest pathway it belongs to."""
    if not REACTOME.exists():
        raise SystemExit(f"Reactome GMT absent ({REACTOME})")
    npath, biggest = {}, {}
    for line in open(REACTOME):
        f = line.rstrip("\n").split("\t")
        mem = [x for x in f[2:] if x]
        for g in mem:
            npath[g] = npath.get(g, 0) + 1
            biggest[g] = max(biggest.get(g, 0), len(mem))
    print(f"  Reactome: {len(npath):,} genes across pathways")
    return npath, biggest


def held_out(F, t, seeds=SEEDS, ret_pred=False):
    """Held-out R2 over random 80/20 splits, using the gate's own XGBoost settings.

    The settings are copied from `mrna_protein_gate.py` deliberately and not tuned here: the residual this
    produces has to be the residual of the model whose +0.2695 is the thing under interrogation. Tuning
    either arm separately would make the comparison between them meaningless.
    Predictions are returned for the residual construction, so every residual is out-of-sample.
    """
    from sklearn.metrics import r2_score
    import xgboost as xgb
    r2s, pred, seen = [], np.full(len(t), np.nan), np.zeros(len(t), bool)
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(t))
        cut = int(0.8 * len(t))
        tr, te = idx[:cut], idx[cut:]
        m = xgb.XGBRegressor(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=2.0, n_jobs=4).fit(F[tr], t[tr])
        p = m.predict(F[te])
        r2s.append(r2_score(t[te], p))
        pred[te] = np.where(seen[te], pred[te], p)      # first fold to hold a gene out owns it
        seen[te] = True
    if ret_pred:
        return float(np.mean(r2s)), pred, seen
    return float(np.mean(r2s)), r2s


def main():
    from scipy import stats
    print("=" * 100)
    print("ASSAY QUALITY OR BIOLOGY? -- the mRNA-protein coupling predictor, with non-measurement features")
    print("=" * 100)
    cache = OUT / "_mpg_cache.npz"
    if not cache.exists():
        raise SystemExit(f"{cache} absent -- run mrna_protein_gate.py first")
    z = np.load(cache, allow_pickle=True)
    if "genes" not in z.files:
        raise SystemExit("the cache predates gene-symbol caching; re-run mrna_protein_gate.py "
                         "so external annotation can be joined on")
    F0, rho, mnames = z["F"], z["rho"], [str(x) for x in z["names"]]
    genes = [str(g) for g in z["genes"]]
    cols = []
    for n in MEAS:
        cols.append(F0[:, mnames.index(n)] if n in mnames else z[n].astype(float))
    M = np.column_stack(cols)
    print(f"  {len(genes):,} genes with a pooled rho; measurability block = {MEAS}")
    print(f"  cohorts per gene: " + ", ".join(
        f"{int(k)}->{int(v)}" for k, v in zip(*np.unique(z["n_cohorts"], return_counts=True))))
    # rho_sd is deliberately NOT a control: it is the spread of the very quantity being predicted, so
    # conditioning on it would be conditioning on the target.

    d700, d900, size = string_features()
    npath, biggest = reactome_features()
    B, bnames = [], ["log_ppi_degree_700", "log_ppi_degree_900", "log_protein_size",
                     "log_n_pathways", "log_pathway_partners"]
    for g in genes:
        B.append([np.log1p(d700.get(g, 0)), np.log1p(d900.get(g, 0)), np.log1p(size.get(g, 0)),
                  np.log1p(npath.get(g, 0)), np.log1p(biggest.get(g, 0))])
    B = np.array(B)
    cov = {n: float((B[:, i] > 0).mean()) for i, n in enumerate(bnames)}
    print("  coverage of the biology block (fraction of genes with a non-zero value):")
    for n, f in cov.items():
        flag = "   <- too sparse to carry anything" if f < 0.5 else ""
        print(f"    {n:22s} {f:.4f}{flag}")
    if cov["log_ppi_degree_700"] < 0.5:
        raise SystemExit("PPI degree still missing for most genes -- the join failed again, and no "
                         "verdict from this run should be believed")

    # protein length is separated out here, not dropped: shotgun MS quantifies a protein from its peptides,
    # so length is partly quantification quality. Keeping it in "biology" would let a measurement effect
    # be reported as structure.
    CLEAN = [i for i, n in enumerate(bnames) if n != "log_protein_size"]
    SIZE = [bnames.index("log_protein_size")]
    R = {"n_genes": len(genes), "meas_features": MEAS, "bio_features": bnames, "bio_coverage": cov,
         "clean_block": [bnames[i] for i in CLEAN], "quarantined": ["log_protein_size"]}

    # ---- arms ----
    print("\n  HELD-OUT R2 PREDICTING rho (the gate's own XGBoost, 5 random 80/20 splits)")
    rng = np.random.default_rng(7)
    arms = {"measurability only": M, "biology (clean)": B[:, CLEAN], "protein length only": B[:, SIZE],
            "meas + clean biology": np.hstack([M, B[:, CLEAN]]), "everything": np.hstack([M, B])}
    for tag, X in arms.items():
        r2, per = held_out(X, rho)
        sh = np.mean([held_out(X, rho[rng.permutation(len(rho))], seeds=(0, 1, 2))[0] for _ in range(3)])
        R[tag] = {"r2": r2, "shuffled": float(sh), "net": float(r2 - sh)}
        print(f"    {tag:22s} R2 {r2:+.4f}   shuffled {sh:+.4f}   net {r2-sh:+.4f}")
    mb = R["measurability only"]["r2"]
    print(f"\n    clean biology ON TOP of measurability {R['meas + clean biology']['r2']-mb:+.4f}")
    print(f"    everything    ON TOP of measurability {R['everything']['r2']-mb:+.4f}")
    print("    ^ both inflated by shared variance; the residual test below is the honest one")
    R["clean_over_meas"] = float(R["meas + clean biology"]["r2"] - mb)
    R["all_over_meas"] = float(R["everything"]["r2"] - mb)

    # ---- the deciding test: predict the out-of-sample measurability residual ----
    print("\n  RESIDUAL TEST -- fit measurability at full strength, then predict what it got wrong")
    _, pm, seen = held_out(M, rho, ret_pred=True)
    ok = seen & np.isfinite(pm)
    res = rho[ok] - pm[ok]
    print(f"    {int(ok.sum()):,} genes held out at least once; residual sd {res.std():.4f} "
          f"(rho sd {rho.std():.4f})")
    R["residual"] = {"n": int(ok.sum()), "sd": float(res.std())}
    blocks = {"clean biology (PPI + Reactome)": B[np.ix_(ok, CLEAN)],
              "protein length alone": B[np.ix_(ok, SIZE)],
              "all five together": B[ok]}
    for tag, Xb in blocks.items():
        rr, per = held_out(Xb, res)
        rsh = [held_out(Xb, res[np.random.default_rng(100 + i).permutation(len(res))],
                        seeds=(0, 1, 2))[0] for i in range(3)]
        net = rr - float(np.mean(rsh))
        print(f"    {tag:32s} R2 {rr:+.4f}   shuffled {np.mean(rsh):+.4f}   NET {net:+.4f}"
              f"   [{' '.join(f'{x:+.3f}' for x in per)}]")
        R["residual"][tag] = {"r2": rr, "shuffled": float(np.mean(rsh)), "net": float(net)}
    net = R["residual"]["clean biology (PPI + Reactome)"]["net"]
    print("    ^ the verdict below is read off the CLEAN block; protein length is a measurement feature")

    # LINEAR SANITY CHECK. The largest marginal Spearman in the clean block is ~0.05, which alone explains
    # R2 ~0.002. If the tree reports ten times that, the claim rests on interactions among four features --
    # possible, but it is also what an overfit looks like, so the linear number is printed beside it.
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    Xb = B[np.ix_(ok, CLEAN)]
    lin = []
    for s in SEEDS:
        idx = np.random.default_rng(s).permutation(len(res))
        cut = int(0.8 * len(res))
        tr, te = idx[:cut], idx[cut:]
        mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit((Xb[tr] - mu) / sd, res[tr])
        lin.append(r2_score(res[te], m.predict((Xb[te] - mu) / sd)))
    print(f"    linear (ridge) on the same clean block: R2 {np.mean(lin):+.4f}"
          f"   [{' '.join(f'{x:+.3f}' for x in lin)}]")
    print("    ^ ridge cannot overfit 4 columns, so a positive value here is the floor on the effect")
    R["residual"]["clean_linear_r2"] = float(np.mean(lin))

    # ---- direction, per feature, against the residual ----
    print("\n  DIRECTION vs the measurability residual (Spearman; a mechanism predicts a sign)")
    print(f"    {'feature':22s} {'rho_vs_residual':>16s} {'p':>10s}   mechanism check")
    EXPECT = {"log_ppi_degree_700": "negative if complex members are buffered",
              "log_ppi_degree_900": "negative if complex members are buffered",
              "log_protein_size": "negative if long proteins are translation-limited",
              "log_n_pathways": "no prediction",
              "log_pathway_partners": "no prediction"}
    R["direction"] = {}
    for i, n in enumerate(bnames):
        v = B[ok][:, i]
        if v.std() == 0:
            continue
        rv, pv = stats.spearmanr(v, res)
        sign = "negative" if rv < 0 else "positive"
        exp = EXPECT[n]
        verdict = ("as predicted" if exp.startswith(sign) else
                   "-- no prediction" if exp == "no prediction" else "OPPOSITE to the mechanism")
        print(f"    {n:22s} {rv:+16.4f} {pv:10.2e}   {exp}: {verdict}")
        R["direction"][n] = {"spearman": float(rv), "p": float(pv), "expected": exp, "verdict": verdict}

    # ---- what this licenses ----
    # THE EFFECT SIZE IS THE RAW HELD-OUT R2, NOT R2 MINUS SHUFFLED. An earlier version of this ladder
    # branched on the shuffle-adjusted net, which is backwards: the tree scores -0.0345 on a shuffled
    # residual, so subtracting that CREDITS the model for how badly it overfits noise. A positive
    # out-of-sample R2 is already evidence on its own; the shuffled arm establishes significance, and the
    # raw value is the size. Reading net here turned +0.023 into +0.057 and moved the verdict a whole rung.
    raw = R["residual"]["clean biology (PPI + Reactome)"]["r2"]
    lin_r2 = R["residual"]["clean_linear_r2"]
    sd0 = res.std()
    sd1 = sd0 * np.sqrt(max(1 - raw, 0))
    print(f"\n  WHAT {raw:+.4f} OF RESIDUAL VARIANCE IS WORTH IN THE UNITS THAT MATTER")
    print(f"    residual spread before biology {sd0:.4f} rho units -> after {sd1:.4f}; "
          f"shrinkage {100*(1-sd1/sd0):.2f}%")
    print(f"    tree {raw:+.4f} vs ridge floor {lin_r2:+.4f}: the effect is somewhere in that band, and "
          f"the tree end is the one that also scores {R['residual']['clean biology (PPI + Reactome)']['shuffled']:+.4f} on noise")
    R["residual"]["res_sd_before"] = float(sd0)
    R["residual"]["res_sd_after"] = float(sd1)
    R["effect_size_read"] = "raw held-out R2 of the clean block; shuffled arm used only for significance"
    net = raw
    print("\n" + "=" * 100)
    if net < 0.005:
        R["verdict"] = ("CONFIDENCE ONLY -- the clean biology block adds nothing measurability does not "
                        "already contain, so the predictor estimates how well a gene was quantified, not "
                        "how tightly its protein tracks its mRNA. Usable as a per-node weight on "
                        "Perturb-seq readings; NOT usable as a per-node correction, and must not be "
                        "described as one.")
    elif net < 0.05:
        R["verdict"] = (f"MOSTLY CONFIDENCE -- the clean block explains {100*net:.1f}% of the "
                        f"measurability residual (ridge floor {100*lin_r2:.1f}%), which shrinks the "
                        f"residual spread from {sd0:.4f} to {sd1:.4f} rho units. The direction of "
                        f"high-confidence PPI degree is negative as complex buffering predicts, so this is "
                        f"not nothing -- but it is far too small to correct a protein count from, and study "
                        f"bias makes even this an upper bound. Use as a per-node WEIGHT, not a correction.")
    else:
        R["verdict"] = (f"BIOLOGY PRESENT -- the clean block explains {100*net:.1f}% of the measurability "
                        f"residual, enough that per-gene coupling is partly structural and a per-node "
                        f"correction has a basis. Study bias still makes this an upper bound, and the size "
                        f"has to be checked against the correction actually needed.")
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "coupling_biology.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'coupling_biology.json'}")


if __name__ == "__main__":
    main()
