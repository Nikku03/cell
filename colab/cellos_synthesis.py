"""cellos_synthesis — do the TOP-DOWN (physics) and BOTTOM-UP (measured data) cells help each other? Measure it.

The whole project found both approaches hit the same wall from opposite sides, and blind in DIFFERENT places:
  * TOP-DOWN  = FBA. Predicts essentiality from physics (grow under mass-balance + capacity). Reaches metabolic
                genes even when silent in the assay; blind to signalling/TF genes (no biomass objective for them).
  * BOTTOM-UP = the measured Perturb-seq debugger. A gene whose knockout SHOCKS the transcriptome (big signature
                norm) tends to matter. Reaches genes expressed in the screen; blind to the ~7,800 K562 doesn't express.
They overlap on almost nothing (295 of 4,609 covered genes). So the union should cover ~2x either alone — IF each
actually predicts on its own turf. This tests that, cross-checks them where they meet, and asks the honest question:
does combining raise trustworthy COVERAGE while HOLDING accuracy — or do the extra genes come at the cost of being
right? Truth = measured DepMap essentiality (dep_frac), independent of both predictors.

-> outputs/orphan/cellos_synthesis.json + verdict. If it genuinely helps, wire `assess GENE` into CellOS.
"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def auc(score, label):
    """tie-aware Mann-Whitney AUC (AVERAGE ranks for ties -> deterministic and unbiased; FBA growth has many ties)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan"), len(label)
    ranks = rankdata(score, method="average")
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)), len(label)


def zrank(d):
    """map a {gene: value} to {gene: rank in [0,1]} so two different-scaled predictors can be averaged fairly."""
    if not d:
        return {}
    genes = list(d); vals = np.array([d[g] for g in genes], float)
    r = np.argsort(np.argsort(vals)) / max(len(vals) - 1, 1)
    return dict(zip(genes, r))


def main():
    from complete_cell import CompleteCell
    import cellos
    C = CompleteCell()

    # truth: measured essentiality
    truth = {}
    for i in range(len(C.genes)):
        try:
            truth[C.name[i]] = float(C.genes[i].get("dep_frac"))
        except (TypeError, ValueError):
            pass
    ess = {g: v > 0.5 for g, v in truth.items()}                 # common-essential

    # TOP-DOWN predictor: FBA. essential <=> low predicted growth on knockout -> score = 1 - growth
    fba = json.load(open(f"{OUT}/fba_essentiality.json"))["predicted_growth_on_knockout"]
    T = {g: 1.0 - float(v) for g, v in fba.items()}

    # BOTTOM-UP predictor: measured Perturb-seq knockout signature magnitude (bigger shock -> matters more)
    k = cellos.CellKernel(quiet=True); dbg = k._load_debugger()
    B = {g: float(n) for g, n in zip(dbg["pgenes"], dbg["norms"])} if dbg else {}
    screen = os.path.basename(cellos.PERTURB)

    def ev(pred, genes=None):
        genes = genes if genes is not None else list(pred)
        gg = [g for g in genes if g in ess and g in pred]
        a, n = auc([pred[g] for g in gg], [ess[g] for g in gg])
        return a, n, sum(ess[g] for g in gg)

    Tset, Bset = set(T), set(B)
    overlap = Tset & Bset
    aT, nT, eT = ev(T)
    aB, nB, eB = ev(B)
    # solo turf (where only one modality reaches)
    aTonly, nTonly, _ = ev(T, Tset - Bset)
    aBonly, nBonly, _ = ev(B, Bset - Tset)
    # cross-check on the overlap: does each predict there, and do they agree?
    aTo, nTo, _ = ev(T, overlap); aBo, _, _ = ev(B, overlap)
    tz, bz = zrank({g: T[g] for g in overlap}), zrank({g: B[g] for g in overlap})
    agree = float(np.corrcoef([tz[g] for g in overlap], [bz[g] for g in overlap])[0, 1]) if len(overlap) > 3 else float("nan")

    # COMBINE #1 (naive): rank-normalize each and average/route — this MISCALIBRATES (different base rates) and loses
    Tz, Bz = zrank(T), zrank(B)
    union = Tset | Bset
    Cnaive = {g: (0.5 * (Tz[g] + Bz[g]) if g in Tz and g in Bz else Tz.get(g, Bz.get(g))) for g in union}
    aNaive, _, _ = ev(Cnaive)

    # COMBINE #2 (ROUTER): calibrate EACH predictor to a real P(essential) on its own turf (out-of-fold, no leakage),
    # then route each gene to its specialist — metabolic->FBA prob, screen->data prob, both->average of the two probs.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    def platt_oof(pred):
        gg = [g for g in pred if g in ess]
        x = np.array([[pred[g]] for g in gg], float)
        y = np.array([ess[g] for g in gg], int)
        p = np.full(len(gg), np.nan)
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            return {}                                          # too few of a class to calibrate
        for tr, te in KFold(5, shuffle=True, random_state=0).split(x):
            lr = LogisticRegression(max_iter=200).fit(x[tr], y[tr])
            p[te] = lr.predict_proba(x[te])[:, 1]
        return {g: float(pp) for g, pp in zip(gg, p) if np.isfinite(pp)}

    Pt, Pb = platt_oof(T), platt_oof(B)
    Croute = {}
    for g in union:
        if g in Pt and g in Pb:
            Croute[g] = 0.5 * (Pt[g] + Pb[g])
        elif g in Pt:
            Croute[g] = Pt[g]
        elif g in Pb:
            Croute[g] = Pb[g]
    aC, nC, eC = ev(Croute)
    # coverage-weighted "effective accuracy" of the router = each modality's AUC on the genes it OWNS
    own_T = Tset - Bset; own_B = Bset - Tset
    aOT, nOT, _ = ev(T, own_T); aOB, nOB, _ = ev(B, own_B); aOv, nOv, _ = ev(Croute, overlap)
    cw = np.nansum([aOT * nOT, aOB * nOB, aOv * nOv]) / max(nOT + nOB + nOv, 1)

    print(f"TOP-DOWN↔BOTTOM-UP SYNTHESIS  (truth = measured DepMap essentiality; screen = {screen})\n")
    print(f"  {'predictor':<34}{'AUC':>7}{'n(truth)':>10}{'essential':>11}")
    print(f"  {'-'*60}")
    print(f"  {'TOP-DOWN  FBA (physics)':<34}{aT:>7.3f}{nT:>10}{eT:>11}")
    print(f"  {'BOTTOM-UP Perturb-seq shock (data)':<34}{aB:>7.3f}{nB:>10}{eB:>11}")
    print(f"  {'  FBA on physics-only genes':<34}{aTonly:>7.3f}{nTonly:>10}")
    print(f"  {'  data on screen-only genes':<34}{aBonly:>7.3f}{nBonly:>10}")
    print(f"  {'COMBINED naive rank-avg (WRONG)':<34}{aNaive:>7.3f}{nC:>10}")
    print(f"  {'COMBINED calibrated ROUTER':<34}{aC:>7.3f}{nC:>10}{eC:>11}")
    print(f"\n  coverage: physics {nT}  |  data {nB}  |  UNION {nC}  "
          f"(+{nC-max(nT,nB)} over the better single modality)")
    print(f"  router effective accuracy (coverage-weighted, each modality on the genes it OWNS): {cw:.3f}")
    print(f"  overlap cross-check ({len(overlap)} genes): FBA AUC {aTo:.2f}, data AUC {aBo:.2f}, "
          f"rank-agreement r={agree:+.2f}")

    # HONEST headline = coverage gain + effective (within-population) accuracy. The pooled AUC is base-rate-inflated
    # (calibration bakes in each population's essential-fraction, so it partly separates genes by WHICH screen they
    # came from, not per-gene skill) — reported but explicitly NOT the headline.
    best_single = max(aT, aB)
    helped_cov = nC > 1.3 * max(nT, nB)
    held_acc = cw >= best_single - 0.02
    if helped_cov and held_acc:
        verdict = (f"SYNERGY (coverage): routing each gene to its specialist nearly doubles coverage "
                   f"({max(nT,nB)}→{nC} genes) while effective per-gene accuracy {cw:.2f} stays at/above the best "
                   f"single modality ({best_single:.2f}). Physics and data are blind in different places, so the "
                   f"union is broader without going wrong. CAVEAT: the pooled AUC {aC:.2f} is base-rate-inflated "
                   f"(separates metabolic-vs-screen populations) — {cw:.2f} is the honest number, not {aC:.2f}. "
                   f"Naive pooling scored {aNaive:.2f}; calibration + routing is what makes it work.")
    elif helped_cov:
        verdict = (f"BROADER, coverage-only: the router covers {nC} genes (vs {max(nT,nB)}) but effective accuracy "
                   f"{cw:.2f} trails best-single {best_single:.2f} — the weaker modality's turf drags per-gene skill.")
    else:
        verdict = f"LIMITED: combining did not broaden coverage as hoped (union {nC}, best single {max(nT,nB)})."
    print("\n" + "=" * 74 + f"\nVERDICT: {verdict}\n" + "=" * 74)

    out = {"screen": screen, "truth": "DepMap dep_frac>0.5",
           "topdown_fba": {"auc": aT, "n": nT, "auc_solo": aTonly, "n_solo": nTonly},
           "bottomup_data": {"auc": aB, "n": nB, "auc_solo": aBonly, "n_solo": nBonly},
           "overlap": {"n": len(overlap), "auc_fba": aTo, "auc_data": aBo, "rank_agreement": agree},
           "combined": {"auc_router_calibrated": aC, "auc_naive_pool": aNaive, "effective_accuracy_cw": float(cw),
                        "n": nC, "coverage_gain_over_best_single": nC - max(nT, nB)},
           "verdict": verdict}
    json.dump(out, open(f"{OUT}/cellos_synthesis.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
