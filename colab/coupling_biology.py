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
MEAS = ["prot_mean", "prot_sd", "rna_mean", "rna_sd", "n_samples"]


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
    """Ridge held-out R2 over random 80/20 splits, standardised inside each fold.

    Ridge and not a tree: the point is to ask whether a linear combination of these features carries
    information, and a boosted tree on 5-8 columns can fit interactions that make direction unreadable.
    Predictions are returned for the residual construction, so every residual is out-of-sample.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    r2s, pred, seen = [], np.full(len(t), np.nan), np.zeros(len(t), bool)
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(t))
        cut = int(0.8 * len(t))
        tr, te = idx[:cut], idx[cut:]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit((F[tr] - mu) / sd, t[tr])
        p = m.predict((F[te] - mu) / sd)
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
    keep = [mnames.index(n) for n in MEAS]
    M = F0[:, keep]
    print(f"  {len(genes):,} genes with a pooled rho; measurability block = {MEAS}")

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

    R = {"n_genes": len(genes), "meas_features": MEAS, "bio_features": bnames, "bio_coverage": cov}

    # ---- arms ----
    print("\n  HELD-OUT R2 PREDICTING rho (ridge, 5 random 80/20 splits)")
    rng = np.random.default_rng(7)
    arms = {"measurability only": M, "biology only": B, "both": np.hstack([M, B])}
    for tag, X in arms.items():
        r2, per = held_out(X, rho)
        sh = np.mean([held_out(X, rho[rng.permutation(len(rho))], seeds=(0, 1, 2))[0] for _ in range(3)])
        R[tag] = {"r2": r2, "shuffled": float(sh), "net": float(r2 - sh)}
        print(f"    {tag:20s} R2 {r2:+.4f}   shuffled {sh:+.4f}   net {r2-sh:+.4f}")
    gain = R["both"]["r2"] - R["measurability only"]["r2"]
    R["bio_over_meas"] = float(gain)
    print(f"    biology ON TOP of measurability {gain:+.4f}"
          f"   <- inflated by shared variance; the residual test below is the honest one")

    # ---- the deciding test: predict the out-of-sample measurability residual from biology alone ----
    print("\n  RESIDUAL TEST -- fit measurability, then predict what it got wrong, from biology alone")
    _, pm, seen = held_out(M, rho, ret_pred=True)
    ok = seen & np.isfinite(pm)
    res = rho[ok] - pm[ok]
    print(f"    {int(ok.sum()):,} genes held out at least once; residual sd {res.std():.4f}")
    rr, per = held_out(B[ok], res)
    rsh = [held_out(B[ok], res[np.random.default_rng(100 + i).permutation(len(res))],
                    seeds=(0, 1, 2))[0] for i in range(5)]
    net = rr - float(np.mean(rsh))
    print(f"    residual R2 from biology alone : {rr:+.4f}  [{' '.join(f'{x:+.3f}' for x in per)}]")
    print(f"    shuffled-residual control      : {np.mean(rsh):+.4f} +/- {np.std(rsh):.4f}")
    print(f"    NET                            : {net:+.4f}")
    R["residual"] = {"n": int(ok.sum()), "r2": rr, "shuffled": float(np.mean(rsh)),
                     "shuffled_sd": float(np.std(rsh)), "net": float(net)}

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
    print("\n" + "=" * 100)
    if net < 0.005:
        R["verdict"] = ("CONFIDENCE ONLY -- biology adds nothing measurability does not already contain, "
                        "so the predictor estimates how well a gene was quantified, not how tightly its "
                        "protein tracks its mRNA. Usable as a per-node weight on Perturb-seq readings; "
                        "NOT usable as a per-node correction, and must not be described as one.")
    elif net < 0.02:
        R["verdict"] = (f"MOSTLY CONFIDENCE -- biology carries a real but small residual signal "
                        f"({net:+.4f}). Enough to say coupling is not purely an assay property, far too "
                        f"little to correct a protein count from. Per-node weight, with a caveat.")
    else:
        R["verdict"] = (f"BIOLOGY PRESENT -- biology predicts {net:+.4f} of the measurability residual, "
                        f"so per-gene coupling is partly structural and a per-node correction has a basis. "
                        f"Its size still has to be checked against the correction actually needed.")
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "coupling_biology.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'coupling_biology.json'}")


if __name__ == "__main__":
    main()
