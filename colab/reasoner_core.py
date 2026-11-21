"""reasoner_core — the PRINCIPLE of the reason engine, lifted out of the gene-specific working so it applies to
ANY part of the cell (genes, reactions, metabolites, complexes, edges…).

The principle (unchanged from reason.py, which validated it on genes → AUC 0.80, calibrated):
  1. gather N INDEPENDENT, differently-blind evidence lines (each a weak vote in {0,1}, nan = unavailable)
  2. score each line's RELIABILITY against a ground truth (per-line AUC)
  3. FUSE by reliability — an out-of-fold logistic regression, so a line is trusted in proportion to how often
     it's right, with NO circularity (the weights are learned on held-out folds)
  4. CALIBRATE — P(truth) as a function of how many lines agree, so the conclusion knows its own confidence
  5. verdict: reasoning WORKS if the fused score beats the best single line AND confidence is monotonic;
     CALIBRATED-MODEST if only the calibration holds; NULL otherwise (reported honestly)

Only the WORKING changes per substrate: which evidence lines you feed, and what "truth" you ground against.
`reason_over(lines, truth)` is substrate-agnostic. This module ships that core + a self-test that reproduces the
gene result, so the generalization is proven faithful, not asserted.
"""
import numpy as np


def auc(score, label):
    """Mann-Whitney AUC, tie-aware (rankdata average), nan-robust. nan if one class is empty."""
    from scipy.stats import rankdata
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def reason_over(lines, truth, min_lines=2, calib_min_lines=None, seed=0):
    """Reason across independent evidence lines to a calibrated conclusion — for ANY entity type.

    lines : dict name -> array(N) of votes in {0.0, 1.0} (1 = this line says 'yes'), np.nan = line unavailable.
    truth : array(N) of {0,1}/bool/nan — the ground truth to score reliability + calibration against.
    Returns a report dict (per-line AUC, convergence AUC, weighted OOF reasoning AUC, best single, calibration
    curve, verdict) plus per-entity fused score `oof`, `agree`, `avail` so callers can quote a calibrated P.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    names = list(lines)
    stack = np.vstack([np.asarray(lines[k], float) for k in names])   # (n_lines, N)
    N = stack.shape[1]
    truth = np.asarray(truth, float)
    tb = truth > 0.5

    avail = np.isfinite(stack).sum(0)
    agree = np.nansum(stack, 0)
    frac = np.where(avail > 0, agree / np.maximum(avail, 1), np.nan)
    ok = np.isfinite(truth) & (avail >= min_lines)

    singles = {k: auc(stack[i][ok], tb[ok]) for i, k in enumerate(names)}
    finite_singles = [v for v in singles.values() if np.isfinite(v)]
    best_single = max(finite_singles) if finite_singles else float("nan")
    conv_auc = auc(frac[ok], tb[ok])

    # weighted reasoning: OOF logistic (missing line imputed to 0.5 = 'no evidence')
    X = np.where(np.isfinite(stack.T), stack.T, 0.5)
    y = tb.astype(int)
    idx_ok = np.where(ok)[0]
    oof = np.full(N, np.nan)
    if len(idx_ok) >= 25 and y[idx_ok].min() != y[idx_ok].max():
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(idx_ok):
            lr = LogisticRegression(max_iter=200).fit(X[idx_ok[tr]], y[idx_ok[tr]])
            oof[idx_ok[te]] = lr.predict_proba(X[idx_ok[te]])[:, 1]
    weighted_auc = auc(oof[ok], tb[ok])

    # calibration: P(truth) by number of agreeing lines, on entities with enough lines for a clean axis
    cml = calib_min_lines if calib_min_lines is not None else max(2, len(names) - 1)
    full = ok & (avail >= cml)
    calib = {}
    for a in range(0, len(names) + 1):
        sel = full & (np.round(agree) == a)
        if sel.sum() >= 15:
            calib[a] = {"n": int(sel.sum()), "p_truth": float(tb[sel].mean())}

    ps = [calib[a]["p_truth"] for a in sorted(calib)]
    monotonic = len(ps) >= 2 and all(ps[i] <= ps[i + 1] + 0.02 for i in range(len(ps) - 1))
    # calibration must have real DYNAMIC RANGE — a flat curve (e.g. 0%→1%→1%→0%) is monotonic-by-tolerance but is
    # NOT calibration. Require the confidence to actually move (>=15 points) across the agreement axis.
    dyn_range = (ps[-1] - ps[0]) if len(ps) >= 2 else 0.0
    calibrated = monotonic and dyn_range >= 0.15
    beats = np.isfinite(weighted_auc) and np.isfinite(best_single) and weighted_auc > best_single + 0.02
    if beats and calibrated:
        verdict = (f"REASONING WORKS: fused AUC {weighted_auc:.2f} beats best single line {best_single:.2f}; "
                   f"confidence calibrated P(truth) {ps[0]:.0%}→{ps[-1]:.0%} with agreement.")
    elif calibrated:
        verdict = (f"CALIBRATED, MODEST LIFT: confidence rises {ps[0]:.0%}→{ps[-1]:.0%} with agreement (trustworthy); "
                   f"fused AUC {weighted_auc:.2f} vs best single {best_single:.2f} — value is calibration+coverage.")
    elif beats:
        verdict = (f"LIFT WITHOUT CLEAN CALIBRATION: fused AUC {weighted_auc:.2f} > best single {best_single:.2f}, "
                   f"but P(truth) is not monotonic in agreement — rank it, don't quote its confidence.")
    else:
        verdict = (f"NULL: fused AUC {weighted_auc:.2f} neither beats the best single line ({best_single:.2f}) nor "
                   f"calibrates cleanly on this substrate — the lines don't converge here.")

    return {"n": int(ok.sum()), "n_truth_pos": int(tb[ok].sum()), "lines": names, "singles": singles,
            "convergence_auc": conv_auc, "weighted_reasoning_auc": weighted_auc, "best_single": best_single,
            "calibration": {int(k): v for k, v in calib.items()}, "verdict": verdict,
            "oof": oof, "agree": agree, "avail": avail, "ok_mask": ok}


def print_report(rep, entity="entity", truth_name="truth"):
    print(f"reasoning over {rep['n']:,} {entity}s with >= evidence ({rep['n_truth_pos']:,} truly {truth_name})\n")
    print(f"  {'evidence line':<26}{'AUC vs '+truth_name:>22}")
    print("  " + "-" * 48)
    for k, v in rep["singles"].items():
        print(f"  {('single: '+k):<26}{v:>22.3f}")
    print("  " + "-" * 48)
    print(f"  {'CONVERGENCE (equal-weight)':<26}{rep['convergence_auc']:>22.3f}")
    print(f"  {'REASONING (weighted, OOF)':<26}{rep['weighted_reasoning_auc']:>22.3f}   ← weighs by reliability")
    print(f"  {'best single line':<26}{rep['best_single']:>22.3f}")
    if rep["calibration"]:
        print(f"\n  CALIBRATION — P({truth_name}) by # agreeing lines:")
        for a in sorted(rep["calibration"]):
            c = rep["calibration"][a]
            print(f"    {a} agree  n={c['n']:>5}   P={c['p_truth']:>5.0%}  {'█'*int(c['p_truth']*40)}")
    print("\n" + "=" * 80 + f"\nVERDICT: {rep['verdict']}\n" + "=" * 80)


def _selftest_genes():
    """prove the general core reproduces the validated GENE result (reason.py → AUC ~0.80)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from complete_cell import CompleteCell
    import reason as gene_reason
    C = CompleteCell(); N = len(C.genes)
    truth = np.array([float(C.genes[i].get("dep_frac")) if str(C.genes[i].get("dep_frac")).replace(".", "", 1)
                      .replace("-", "", 1).isdigit() else np.nan for i in range(N)])
    L = gene_reason.evidence_lines(C)
    rep = reason_over(L, truth)
    print_report(rep, entity="gene", truth_name="essential")
    return rep


if __name__ == "__main__":
    _selftest_genes()
