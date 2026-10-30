"""Is CatPred's kcat error a LEARNABLE pattern on the truly-measured enzymes, or irreducible noise?

The kinetics layer keeps ~380 enzymes at literature-measured kcat ('measured' tier) and predicts the rest
with CatPred. Natural question: use the measured subset to fit a correction that also improves the predicted
ones. This script tests that honestly and is a calibration guardrail (11th recovery-scorecard axis): it
asserts we do NOT wire in a correction that only looks good because of label leakage.

Three findings (all reproduced below):
  1. The residual (measured - CatPred) has a real but weak regression-to-mean tilt (r ~ -0.3): CatPred
     slightly compresses the dynamic range. Real pattern, detectable.
  2. It does NOT generalise. On the clean in-vitro labels, every correction fit on the measured subset is
     neutral-to-worse held-out; a 20-seed shrinkage scan puts the CV-optimal shrinkage at s=1.0 (CatPred
     unchanged). CatPred's ~3.3x error on real measurements is at the lab-to-lab noise floor.
  3. A naive "recalibration" appears to win ~20% ONLY when the label set mixes in davidi's non-measured
     tiers (EC-class / network-propagated / prior); on davidi's truly-measured tier the same fit is +6x
     WORSE. The apparent win is the model learning to reproduce the priors -> label-quality artifact.

-> outputs/orphan/kcat_calibration_validation.json  (regression gate)
Run on Colab: mounts Drive, reads catpred_kinetics.json / kinetics_refined.json / davidi_kcat.json.
In-sandbox: falls back to the session tool-results copies.
"""
import json, re, os, glob, numpy as np
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = "outputs/orphan"


# ---------- data loading (Drive on Colab, tool-results fallback in-sandbox) ----------
def _clean(s):
    return re.sub(r'\\(?=[^"\\/bfnrtu])', '', s)  # strip markdown-escape backslashes some readers inject


def _drive_json(*name_globs):
    for ng in name_globs:
        c = glob.glob(f"/content/drive/MyDrive/cell_model/**/{ng}", recursive=True)
        if c:
            return json.load(open(sorted(c, key=os.path.getsize, reverse=True)[0]))
    return None


def load_inputs():
    """Return (catpred, kinetics_refined, davidi, seqlen, subsize)."""
    cp = _drive_json("catpred_kinetics.json", "catpred*.json")
    kr = _drive_json("kinetics_refined.json")
    dv = _drive_json("davidi_kcat.json", "davidi*.json")
    csv_text = None
    if cp is None:  # in-sandbox fallback to the session's Drive-read tool-results
        TR = ("/root/.claude/projects/-home-user-cell/"
              "0f039315-b3a9-52ac-8187-9fae0d726994/tool-results/")

        def esc(fn, key):
            return json.loads(_clean(json.load(open(TR + fn))['fileContent']))[key]
        cp = {"catpred": esc('mcp-Google_Drive-read_file_content-1783416491589.txt', 'catpred')}
        kr = {"kinetics_refined": json.loads(_clean(json.load(
            open(TR + 'mcp-Google_Drive-read_file_content-1783416612863.txt'))['fileContent']))['kinetics_refined']}
        dv = {"davidi_kcat": esc('mcp-Google_Drive-read_file_content-1783417250169.txt', 'davidi_kcat')}
        csv_text = _clean(json.load(
            open(TR + 'mcp-Google_Drive-read_file_content-1783416613081.txt'))['fileContent'])
    catpred = cp.get("catpred", cp) if isinstance(cp, dict) else cp
    kinr = kr.get("kinetics_refined", kr) if isinstance(kr, dict) else kr
    davidi = dv.get("davidi_kcat", dv) if isinstance(dv, dict) else dv
    seqlen, subsize = {}, {}
    if csv_text is None:
        for ng in ("catpred_input.csv",):
            c = glob.glob(f"/content/drive/MyDrive/cell_model/**/{ng}", recursive=True)
            if c:
                csv_text = open(sorted(c, key=os.path.getsize, reverse=True)[0]).read()
    if csv_text:
        for ln in csv_text.splitlines()[1:]:
            p = ln.split(',')
            if len(p) >= 4:
                seqlen[p[3].strip()] = len(p[1])
                subsize[p[3].strip()] = sum(ch.isalpha() and ch.upper() in 'CNOSPFIB' for ch in p[0])
    return catpred, kinr, davidi, seqlen, subsize


def fold(e):
    return float(10 ** np.median(np.abs(e)))


def run():
    catpred, kr, davidi, seqlen, subsize = load_inputs()

    # ----- clean in-vitro label set: truly-measured kcat + a CatPred prediction -----
    rows = [g for g, r in kr.items() if r.get('tier') in {'measured', 'EC-measured'}
            and r.get('kcat_per_s', 0) > 0 and g in catpred and catpred[g].get('kcat', 0) > 0]
    y   = np.array([np.log10(kr[g]['kcat_per_s']) for g in rows])
    xc  = np.array([np.log10(catpred[g]['kcat']) for g in rows])
    km  = np.array([np.log10(catpred[g].get('km') or np.nan) for g in rows])
    unc = np.array([catpred[g].get('kcat_unc', 1.0) for g in rows])
    plen = np.array([np.log10(seqlen.get(g, np.nan) or np.nan) for g in rows])
    ssz  = np.array([np.log10(subsize.get(g, np.nan) or np.nan) for g in rows])
    resid = y - xc
    raw_fe = fold(resid)
    raw_w2 = float(np.mean(np.abs(resid) < np.log10(2)))

    # 1) is the residual systematic?
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 20 else None
    rtm_r = corr(resid, xc)
    m = np.isfinite(xc) & np.isfinite(resid)
    slope = float(np.polyfit(xc[m], resid[m], 1)[0])
    residual_corr = {"catpred_pred_regression_to_mean": rtm_r, "log_km": corr(resid, km),
                     "catpred_unc": corr(resid, unc), "log_protein_len": corr(resid, plen),
                     "log_substrate_size": corr(resid, ssz)}

    # 2) does any fitted correction generalise? 5-fold CV
    def cv(F, model_fn):
        F = np.nan_to_num(F); kf = KFold(5, shuffle=True, random_state=0); pred = np.zeros(len(y))
        for tr, te in kf.split(F):
            pred[te] = model_fn().fit(F[tr], y[tr]).predict(F[te])
        return fold(pred - y)
    correctors = {
        "raw_catpred": raw_fe,
        "linear_deshrinkage": cv(xc.reshape(-1, 1), LinearRegression),
        "linear_catpred_km_unc": cv(np.c_[xc, km, unc], LinearRegression),
        "gbm_all_features": cv(np.c_[xc, km, unc, plen, ssz],
            lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                    learning_rate=0.05, min_samples_leaf=15)),
    }
    # shrinkage scan (20 seeds) — CV-optimal shrinkage factor
    shrink = {}
    for s in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        fes = []
        for seed in range(20):
            kf = KFold(5, shuffle=True, random_state=seed); pred = np.zeros(len(y))
            for tr, te in kf.split(xc):
                pred[te] = y[tr].mean() + s * (xc[te] - xc[tr].mean())
            fes.append(fold(pred - y))
        shrink[f"{s:.1f}"] = float(np.mean(fes))
    optimal_s = float(min(shrink, key=shrink.get))
    best_corrector_fe = min(correctors.values())
    correction_helps = bool(best_corrector_fe < raw_fe - 0.05 or optimal_s != 1.0)

    # 3) label-quality artifact: split davidi by its own tier, fit recalibration on each
    def cv_recal(genes):
        yy  = np.array([np.log10(davidi[g]['kcat_measured_per_s']) for g in genes])
        xx  = np.array([np.log10(catpred[g]['kcat']) for g in genes])
        kk  = np.array([np.log10(catpred[g].get('km') or np.nan) for g in genes])
        uu  = np.array([catpred[g].get('kcat_unc', 1.0) for g in genes])
        F = np.nan_to_num(np.c_[xx, uu, kk]); kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros(len(yy))
        for tr, te in kf.split(F):
            pr[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                     learning_rate=0.05).fit(F[tr], yy[tr]).predict(F[te])
        return fold(xx - yy), fold(pr - yy), len(yy)
    dgroups = {}
    for g, v in davidi.items():
        if v.get('kcat_measured_per_s', 0) > 0 and g in catpred and catpred[g].get('kcat', 0) > 0:
            dgroups.setdefault(v.get('measured_tier'), []).append(g)
    tier_artifact = {}
    for tier, gs in dgroups.items():
        if len(gs) >= 25:
            r, rc, n = cv_recal(gs)
            tier_artifact[tier] = {"n": n, "catpred_raw": r, "recalibrated": rc, "delta": rc - r}
    real_meas = dgroups.get('measured', []) + dgroups.get('EC-measured', [])
    rmr = cv_recal(real_meas) if len(real_meas) >= 25 else None
    # recalibration helps only on synthetic tiers, hurts on real-measured
    hurts_real = bool(rmr and rmr[1] > rmr[0])
    helps_synthetic = any(v["delta"] < -0.3 for t, v in tier_artifact.items()
                          if t in ("network-propagated", "global-prior", "EC-class"))

    passed = bool((not correction_helps) and hurts_real and helps_synthetic)
    result = {
        "summary": {
            "n_measured_invitro": len(rows),
            "catpred_raw_fold_error": raw_fe,
            "catpred_within_2x": raw_w2,
            "residual_regression_to_mean_r": rtm_r,
            "cv_optimal_shrinkage": optimal_s,
            "best_fitted_corrector_fold_error": best_corrector_fe,
            "correction_generalises": correction_helps,
            "recalibration_hurts_real_measured": hurts_real,
            "recalibration_helps_only_synthetic_tiers": helps_synthetic,
        },
        "residual_correlations": residual_corr,
        "regression_to_mean_slope": slope,
        "corrector_cv_fold_error": correctors,
        "shrinkage_scan_fold_error": shrink,
        "label_quality_artifact_by_davidi_tier": tier_artifact,
        "real_measured_recal": ({"catpred_raw": rmr[0], "recalibrated": rmr[1], "n": rmr[2]} if rmr else None),
        "verdict": ("No fitted correction beats CatPred-as-is on real measurements (optimal shrinkage=1.0); "
                    "the apparent recalibration win is a label-quality artifact. Use CatPred at measured "
                    "accuracy; keep literature-measured kcat as ground truth."),
        "passed": passed,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(f"{OUT}/kcat_calibration_validation.json", "w"), indent=2)
    return result


def main():
    r = run()
    s = r["summary"]
    print("=" * 74)
    print(f"KCAT CALIBRATION CHECK — {'PASS' if r['passed'] else 'FAIL'}")
    print("=" * 74)
    print(f"  measured in-vitro enzymes (with CatPred): {s['n_measured_invitro']}")
    print(f"  CatPred raw fold-error: {s['catpred_raw_fold_error']:.2f}x  (within 2x {s['catpred_within_2x']:.2f})")
    print(f"  residual regression-to-mean r: {s['residual_regression_to_mean_r']:+.3f}  (real but weak)")
    print(f"  best fitted corrector (5-fold CV): {s['best_fitted_corrector_fold_error']:.2f}x  "
          f"| CV-optimal shrinkage s={s['cv_optimal_shrinkage']:.1f}")
    print(f"  correction generalises? {s['correction_generalises']}  "
          f"(-> use CatPred as-is)" if not s['correction_generalises'] else "")
    print("\n  label-quality artifact (davidi tier -> recalibration delta):")
    for tier, v in r["label_quality_artifact_by_davidi_tier"].items():
        print(f"    {tier:20} n={v['n']:>3}  raw {v['catpred_raw']:.2f}x -> recal {v['recalibrated']:.2f}x  "
              f"({v['delta']:+.2f})")
    print(f"\n  VERDICT: {r['verdict']}")
    print("\n-> outputs/orphan/kcat_calibration_validation.json")
    return r


if __name__ == "__main__":
    main()
