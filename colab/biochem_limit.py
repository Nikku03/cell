"""biochem_limit — test the limit of the biochemical layer: can the software DECODE enzyme kinetics (kcat, the
catalytic turnover number) from the parts we DO have — domains, substrates, partners, abundance — for the enzymes
we DON'T have it measured for? kcat is the missing quantity that would let us RUN the other 83% of the proteome the
way we run metabolism.

HONESTY DISCIPLINE (the whole test hinges on it): the enzyme_records mix ~420 EXPERIMENTALLY-MEASURED kcats with
~2,100 that are themselves model-predicted (CatPred / EC-class priors). We validate ONLY against the measured ones
— predicting a prediction would be circular and inflate the number. Cross-validated, so no leakage.

Features (the parts we have): InterPro domain family (sequence-derived → sets the enzyme's kcat ballpark),
substrate/partner counts, log-Km, subcellular compartment, protein abundance. Target: log10(kcat) [~7 orders of
magnitude]. Metric: Spearman(pred, measured) + RMSE in log10 units (how many FOLD off), vs a mean baseline and a
domain-only ablation. Honest verdict: how far can coarse parts decode a quantity that really needs the active-site
chemistry.

-> outputs/orphan/biochem_limit.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def main():
    from complete_cell import CompleteCell
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr
    C = CompleteCell()
    recs = json.load(open(f"{OUT}/enzyme_records.json"))
    dom = json.load(open(f"{OUT}/domains.json")).get("domains", {})     # {gene_idx_str: [IPR...]}
    ppm = C.D.get("ppm", {}) or {}

    # ground truth: MEASURED kcats only
    meas = {g: v for g, v in recs.items()
            if v.get("kcat_tier") in ("measured", "EC-measured")
            and isinstance(v.get("kcat_invitro_per_s"), (int, float)) and v["kcat_invitro_per_s"] > 0}
    print(f"measured kcat ground truth: {len(meas)} enzymes (of {len(recs)} records; the rest are model-predicted)")

    # top domains as one-hot features
    from collections import Counter
    dc = Counter()
    for g in meas:
        i = C.idx.get(g)
        if i is not None:
            dc.update(dom.get(str(i), []))
    top_dom = [d for d, n in dc.most_common(120) if n >= 3]
    di = {d: k for k, d in enumerate(top_dom)}

    def feats(g):
        i = C.idx.get(g); r = recs[g]
        f = np.zeros(len(top_dom) + 6, dtype=np.float32)
        for d in dom.get(str(i), []) if i is not None else []:
            if d in di:
                f[di[d]] = 1.0
        f[len(top_dom) + 0] = len(r.get("reaction_metabolites") or [])
        f[len(top_dom) + 1] = r.get("n_ppi") or 0
        km = r.get("km_uM")
        f[len(top_dom) + 2] = np.log10(km) if isinstance(km, (int, float)) and km > 0 else -99
        f[len(top_dom) + 3] = float(isinstance(km, (int, float)) and km > 0)     # km-available flag
        ab = ppm.get(str(i)) or ppm.get(g) or 0 if i is not None else 0
        try:
            f[len(top_dom) + 4] = np.log10(float(ab) + 1)
        except (TypeError, ValueError):
            f[len(top_dom) + 4] = 0
        cmp = (C.genes[i].get("comp") if i is not None else "") or ""
        f[len(top_dom) + 5] = hash(cmp) % 20                                     # crude compartment code
        return f

    genes = sorted(meas)
    X = np.vstack([feats(g) for g in genes])
    y = np.array([np.log10(meas[g]["kcat_invitro_per_s"]) for g in genes])
    print(f"  target log10(kcat): range [{y.min():.1f}, {y.max():.1f}], std {y.std():.2f} (≈{10**y.std():.0f}-fold spread)")

    def cv_pred(Xu):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xu):
            m = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                      random_state=0, n_jobs=-1).fit(Xu[tr], y[tr])
            pred[te] = m.predict(Xu[te])
        return pred

    ndom = len(top_dom)
    variants = {
        "domain family only": X[:, :ndom],
        "+ substrate/partner counts": X[:, :ndom + 2],
        "+ Km + abundance (all parts)": X,
    }
    print(f"\n  {'features':<34}{'Spearman':>10}{'RMSE(log10)':>13}{'fold-err':>10}")
    print("  " + "-" * 67)
    res = {}
    for name, Xu in variants.items():
        p = cv_pred(Xu)
        rho = float(spearmanr(p, y).statistic)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        res[name] = {"spearman": rho, "rmse_log10": rmse, "fold_error": float(10 ** rmse)}
        print(f"  {name:<34}{rho:>10.3f}{rmse:>13.2f}{10**rmse:>9.1f}x")
    base_rmse = float(y.std())
    print("  " + "-" * 67)
    print(f"  {'baseline (predict mean)':<34}{0.0:>10.3f}{base_rmse:>13.2f}{10**base_rmse:>9.1f}x")

    best = res["+ Km + abundance (all parts)"]
    rho, fold = best["spearman"], best["fold_error"]
    if rho > 0.55:
        v = (f"DECODABLE: coarse parts predict kcat at Spearman {rho:.2f} (within {fold:.1f}-fold), approaching the "
             f"deep sequence+substrate state of the art. The kinetic layer is largely recoverable from the parts.")
    elif rho > 0.3:
        v = (f"PARTIAL: the parts decode kcat weakly (Spearman {rho:.2f}, within {fold:.1f}-fold vs {10**base_rmse:.0f}x "
             f"baseline) — enzyme FAMILY (domains) sets the ballpark, but the precise turnover needs the active-site "
             f"chemistry we don't encode (protein-LM/substrate features). Real signal, honest ceiling.")
    else:
        v = (f"NEAR-LIMIT: kcat barely decodes from the parts we have (Spearman {rho:.2f}); the missing kinetic layer "
             f"genuinely requires measurement or sequence-level modelling we don't hold.")
    print("\n" + "=" * 72 + f"\nVERDICT: {v}\n" + "=" * 72)
    json.dump({"n_measured": len(meas), "n_records_total": len(recs), "log10_kcat_std": base_rmse,
               "variants": res, "verdict": v}, open(f"{OUT}/biochem_limit.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
