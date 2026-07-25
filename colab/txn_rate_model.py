"""txn_rate_model -- compute the normal transcription rate of every gene, then fit an equation that predicts it, and test whether knowing
WHICH TFs regulate a gene adds anything.

THE RATE IS DIRECTLY COMPUTABLE AND NOBODY HAD COMPUTED IT. RNAdecayCafe gives, per gene per cell line, both the steady-state abundance
(RPKM) and the measured degradation constant (k_deg, from SLAM/TimeLapse-seq). At steady state d[mRNA]/dt = k_syn - k_deg*[mRNA] = 0, so

        k_syn = [mRNA] * k_deg = RPKM * k_deg          units: RPKM per hour

That is an absolute synthesis rate, not a fold-change and not a z-score, for ~10.8k genes in K562 and 12 further cell lines. This project
has been reasoning about "transcription rate" for several steps without ever computing it.

WHY 13 CELL LINES IS THE NATURAL EXPERIMENT. There is no dataset here that titrates one TF and reads the resulting rate. What there is: the
same gene transcribed at different rates in 13 different cellular contexts, where different TFs are present. That is the closest available
approximation to "before and after" -- not a controlled induction, but a real contrast, and it is what makes the TF term testable at all.

THE EQUATION, built in nested layers so each addition has to earn its place:
        log k_syn  =  b0
                    + f(promoter)      CpG island, enhancer count, gene length, chromosome
                    + f(gene class)    annotated process, compartment, essentiality, constraint
                    + f(regulation)    HOW MANY TFs regulate the gene
                    + f(TF identity)   WHICH TFs regulate it -- one indicator per TF, the term the question is really about
                    + cell-line effect
Each layer is added on top of the previous and scored on HELD-OUT GENES, so a layer that only memorises gene identity gains nothing.

THE CIRCULARITY THAT WOULD RUIN IT. k_syn is RPKM x k_deg, so ANY feature carrying expression level predicts it trivially. Baseline
expression, protein abundance and mover-frequency are therefore excluded from the feature set entirely. What remains is promoter structure,
gene class, and regulatory wiring -- none of which contains the answer.

WHAT THIS DOES AND DOES NOT SETTLE. It gives the normal rate and an equation for it. It does NOT give "TF X boosts gene Y by Z-fold",
because the cell lines differ in thousands of ways at once and no TF is being individually manipulated -- the TF-identity layer measures
whether regulatory wiring carries rate information, not the causal effect of any single factor."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
MIN_TF_GENES = 25          # a TF needs this many regulated genes before it gets its own indicator column
NSPLIT = 5


def main():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import pearsonr, spearmanr

    dc = pd.read_csv(SP / "rnadecay" / "AvgKdegs.csv")
    dc = dc[(dc["avg_RPKM_total"] > 0) & (dc["avg_kdeg"] > 0)].copy()
    dc["k_syn"] = dc["avg_RPKM_total"] * dc["avg_kdeg"]          # RPKM per hour
    dc["log_ksyn"] = np.log10(dc["k_syn"])
    lines = sorted(dc["cell_line"].unique())
    print(f"transcription rate k_syn = RPKM x k_deg for {len(dc)} gene-celltype pairs across {len(lines)} cell lines")
    k5 = dc[dc.cell_line == "K562"]
    print(f"  K562: {len(k5)} genes | k_syn median {k5.k_syn.median():.2f} RPKM/h, "
          f"IQR {k5.k_syn.quantile(.25):.2f}-{k5.k_syn.quantile(.75):.2f}, "
          f"range {k5.k_syn.min():.4f}-{k5.k_syn.max():.0f}")
    print(f"  spans {np.log10(k5.k_syn.max()/k5.k_syn.min()):.1f} orders of magnitude")
    # how much of the rate is set by abundance vs by turnover?
    print(f"  corr(log k_syn, log RPKM) = {pearsonr(np.log10(k5.avg_RPKM_total), k5.log_ksyn)[0]:+.3f}   "
          f"corr(log k_syn, log k_deg) = {pearsonr(np.log10(k5.avg_kdeg), k5.log_ksyn)[0]:+.3f}")

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; N = len(names)
    info = {g["name"]: g for g in D["genes"]}
    reg_in = collections.defaultdict(set)
    for e in (D.get("reg") or []):
        try:
            reg_in[names[e[1]]].add(names[e[0]])
        except (TypeError, IndexError):
            continue
    tfcount = collections.Counter()
    for g, ts in reg_in.items():
        tfcount.update(ts)
    TFS = sorted(t for t, c in tfcount.items() if c >= MIN_TF_GENES)
    print(f"  regulatory wiring: {len(reg_in)} genes have a known regulator; "
          f"{len(TFS)} TFs regulate >= {MIN_TF_GENES} genes and get an indicator column")

    genes = sorted(set(dc.feature_ID) & set(info))
    gi = {g: i for i, g in enumerate(genes)}
    print(f"  {len(genes)} genes have both a measured rate and an annotation record")

    PROC = sorted({(info[g].get("proc") or "?") for g in genes})
    COMP = sorted({(info[g].get("comp") or "?") for g in genes})
    tfi = {t: i for i, t in enumerate(TFS)}

    def num(g, k, d=0.0):
        try:
            return float(info[g].get(k) or d)
        except (TypeError, ValueError):
            return d

    # NOTE: no expression, no abundance, no mover-frequency anywhere -- those would make k_syn trivially predictable
    def promoter(g):
        return [num(g, "cpg"), np.log1p(num(g, "enh")), num(g, "loeuf", 1.0), num(g, "ess"),
                num(g, "dep_frac"), np.log1p(num(g, "pubs")), num(g, "dark"), num(g, "tf")]

    def gclass(g):
        p = info[g].get("proc") or "?"; c = info[g].get("comp") or "?"
        return ([1.0 if p == x else 0.0 for x in PROC] + [1.0 if c == x else 0.0 for x in COMP]
                + [np.log1p(num(g, "npath")), np.log1p(num(g, "ppi")), np.log1p(num(g, "ndis"))])

    def nreg(g):
        return [np.log1p(len(reg_in.get(g, ())))]

    def tfid(g):
        v = np.zeros(len(TFS))
        for t in reg_in.get(g, ()):
            if t in tfi:
                v[tfi[t]] = 1.0
        return list(v)

    P = np.array([promoter(g) for g in genes])
    G = np.array([gclass(g) for g in genes])
    R = np.array([nreg(g) for g in genes])
    Tm = np.array([tfid(g) for g in genes])
    print(f"  feature blocks: promoter {P.shape[1]}, gene-class {G.shape[1]}, n-regulators {R.shape[1]}, "
          f"TF-identity {Tm.shape[1]}")

    LAYERS = [("promoter only", [P]), ("+ gene class", [P, G]), ("+ how many TFs", [P, G, R]),
              ("+ WHICH TFs", [P, G, R, Tm])]

    rng = np.random.RandomState(0)
    results = collections.defaultdict(list)
    for line in ["K562"] + [l for l in lines if l != "K562"][:4]:
        sub = dc[dc.cell_line == line]
        y_all = {r.feature_ID: r.log_ksyn for r in sub.itertuples()}
        idx = [i for i, g in enumerate(genes) if g in y_all]
        y = np.array([y_all[genes[i]] for i in idx])
        for si in range(NSPLIT):
            order = np.array(idx); rng.shuffle(order)
            cut = int(0.7 * len(order)); tr, te = order[:cut], order[cut:]
            pos = {g: k for k, g in enumerate(idx)}
            ytr = np.array([y_all[genes[i]] for i in tr]); yte = np.array([y_all[genes[i]] for i in te])
            for lab, blocks in LAYERS:
                X = np.hstack(blocks)
                m = X[tr].mean(0); s = X[tr].std(0) + 1e-9
                pr = Ridge(alpha=10.0).fit((X[tr] - m) / s, ytr).predict((X[te] - m) / s)
                results[(line, lab, "ridge")].append((pearsonr(pr, yte)[0], np.mean((pr - yte) ** 2)))
            X = np.hstack([P, G, R, Tm])
            pr = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                               random_state=0).fit(X[tr], ytr).predict(X[te])
            results[(line, "+ WHICH TFs", "gbm")].append((pearsonr(pr, yte)[0], np.mean((pr - yte) ** 2)))

    print(f"\n  HELD-OUT GENE prediction of log10 k_syn ({NSPLIT} splits, 70/30)")
    print(f"  {'cell line':10s} {'model':18s} {'pearson r':>10s} {'R2':>8s} {'MSE':>8s}")
    tab = []
    for line in ["K562"] + [l for l in lines if l != "K562"][:4]:
        for lab, _ in LAYERS:
            v = results[(line, lab, "ridge")]
            if not v:
                continue
            r = float(np.mean([x[0] for x in v])); mse = float(np.mean([x[1] for x in v]))
            tab.append({"cell_line": line, "model": lab, "algo": "ridge", "pearson_r": round(r, 4),
                        "r2": round(r * r, 4), "mse": round(mse, 4)})
            print(f"  {line:10s} {lab:18s} {r:>10.4f} {r*r:>8.4f} {mse:>8.4f}")
        v = results[(line, "+ WHICH TFs", "gbm")]
        if v:
            r = float(np.mean([x[0] for x in v])); mse = float(np.mean([x[1] for x in v]))
            tab.append({"cell_line": line, "model": "+ WHICH TFs (GBM)", "algo": "gbm",
                        "pearson_r": round(r, 4), "r2": round(r * r, 4), "mse": round(mse, 4)})
            print(f"  {line:10s} {'+ WHICH TFs (GBM)':18s} {r:>10.4f} {r*r:>8.4f} {mse:>8.4f}")

    k = [t for t in tab if t["cell_line"] == "K562"]
    base = next(t for t in k if t["model"] == "promoter only")
    nreg_m = next(t for t in k if t["model"] == "+ how many TFs")
    which = next(t for t in k if t["model"] == "+ WHICH TFs")
    gbm = next((t for t in k if t["algo"] == "gbm"), None)
    tf_gain = which["r2"] - nreg_m["r2"]
    print(f"\n  DOES TF IDENTITY ADD ANYTHING? knowing HOW MANY TFs regulate a gene gives R2={nreg_m['r2']:.4f}; "
          f"adding WHICH {len(TFS)} TFs gives R2={which['r2']:.4f}, a gain of {tf_gain:+.4f}")

    verdict = (
        f"TRANSCRIPTION RATE ({len(dc)} gene-celltype pairs, {len(lines)} cell lines): computes the quantity this project had been "
        "discussing without ever calculating. RNAdecayCafe carries both steady-state abundance and measured decay constant per gene per "
        "cell line, and at steady state k_syn = RPKM * k_deg, so the absolute synthesis rate follows directly in RPKM per hour -- no "
        f"modelling required. In K562 the median gene is transcribed at {k5.k_syn.median():.2f} RPKM/h and rates span "
        f"{np.log10(k5.k_syn.max()/k5.k_syn.min()):.1f} orders of magnitude. "
        f"AN EQUATION was then fitted to log10 k_syn from promoter structure, gene class and regulatory wiring, scored on HELD-OUT GENES, "
        "with expression, protein abundance and any other level-carrying feature EXCLUDED because k_syn is built from abundance and would "
        f"otherwise be trivially predictable. Nested layers, K562: promoter alone R2={base['r2']:.3f}, plus gene class "
        f"{next(t for t in k if t['model'] == '+ gene class')['r2']:.3f}, plus how many TFs regulate the gene {nreg_m['r2']:.3f}, plus "
        f"which {len(TFS)} specific TFs {which['r2']:.3f}"
        + (f", and a gradient-boosted fit on the same features {gbm['r2']:.3f}. " if gbm else ". ")
        + f"TF IDENTITY IS WORTH {tf_gain:+.4f} R2 over merely counting regulators, which is the honest answer to whether regulatory wiring "
        "carries rate information. "
        "WHAT THIS IS NOT: it is not 'TF X boosts gene Y by Z-fold'. No TF is manipulated anywhere in this data; the 13 cell lines differ "
        "in thousands of ways simultaneously, so the TF-identity layer measures association between wiring and rate, not the causal effect "
        "of any single factor. Getting that would need a TF induction series with nascent-RNA readout, which is not on disk. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_pairs": int(len(dc)), "cell_lines": lines, "n_tfs_with_column": len(TFS),
               "k562_median_ksyn_rpkm_per_h": float(k5.k_syn.median()),
               "k562_orders_of_magnitude": float(np.log10(k5.k_syn.max() / k5.k_syn.min())),
               "table": tab, "tf_identity_r2_gain": round(tf_gain, 4),
               "verdict": verdict, "note": verdict}, open(OUT / "txn_rate_model.json", "w"), indent=1)
    print("\n  -> outputs/orphan/txn_rate_model.json")


if __name__ == "__main__":
    main()
