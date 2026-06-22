"""Per-segment mechanistic analysis -- the process for concluding *how* the
model arrives at each segment's score, and *what* would fix the failures.

Five-step framework, applied to CONSERVED CORE / CONDITIONAL / NEAR-ORPHAN /
TRUE ORPHAN, all on the 3-seed averaged v2 nosc + GBM predictions:

  1. INPUT CHARACTERIZATION
     - focal feature stats per segment (means, stds)
     - MSA depth + profile presence per segment
     - distributional shift between segments (the why behind segment behavior)

  2. OUTPUT CHARACTERIZATION
     - score distribution split by truth (essential vs non-essential)
     - brightness distribution
     - calibration table (binned p vs actual accuracy)

  3. ERROR DECOMPOSITION
     - confusion matrix at p>0.5
     - false-negative rate on essentials (the conditional-residual signature)
     - direction of the error: FP-skew vs FN-skew

  4. MECHANISM TEST (where applicable)
     - TRUE ORPHAN: per-focal-feature pointwise pearson correlation with
       essentiality on CORE vs on ORPHAN, side by side. If signs flip, the
       model trained on core has learned an inverted mapping when applied to
       orphans -- mechanistically explaining AUC < 0.5.
     - NEAR-ORPHAN: AUC binned by exact MSA depth 1..5, to see if the
       gradient is monotonic (would justify a hard threshold).

  5. WRITTEN CONCLUSIONS (one paragraph per segment)

CPU only. Reads outputs/orphan/af_torch_preds_v2nosc_full_s{0,1,2}.npz.
"""
import csv, json, argparse
from pathlib import Path
import numpy as np

from af_common import OUT, auc, auprc


FOC_NAMES = ["log_len", "leading", "oriC", "gc", "log_npar",
             "dnds", "has_dnds",
             "is_regulator", "is_transporter", "is_signaling", "is_conditional"]


def load_avg_preds(stem, seeds):
    P, BR = [], []
    for s in seeds:
        z = np.load(OUT / f"{stem}{s}.npz", allow_pickle=True)
        P.append(z["p"]); BR.append(z["brightness"])
    z = np.load(OUT / f"{stem}{seeds[0]}.npz", allow_pickle=True)
    return (np.mean(P, 0), np.mean(BR, 0),
            z["y"], z["clade"], z["meta_org"], z["lt"])


def load_conditional(labels_dir):
    is_conditional_map = {}
    for d in (Path(labels_dir), Path("data/drive_import/labels")):
        p = d / "regulator_features.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                try:
                    is_conditional_map[(r["organism"], r["locus_tag"])] = \
                        int(float(r.get("is_conditional", 0) or 0))
                except Exception:
                    pass
            break
    return is_conditional_map


def calibration_table(p, y, bins=(0, .2, .4, .6, .8, 1.01)):
    rows = []
    for lo, hi in zip(bins, bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 10:
            continue
        rows.append((f"p∈[{lo:.1f},{hi:.1f})",
                     int(m.sum()),
                     round(float(p[m].mean()), 3),
                     round(float(y[m].mean()), 3)))
    return rows


def step1_inputs(name, m, foc, msa_depth, is_orphan):
    print(f"\n  [{name}]  n={int(m.sum()):,}")
    print(f"    focal feature means (z-normalised foc cols):")
    for i, fn in enumerate(FOC_NAMES[:foc.shape[1]]):
        v = foc[m, i]
        print(f"      {fn:<14}  mean {v.mean():+.3f}  std {v.std():.3f}")
    print(f"    MSA depth     mean {msa_depth[m].mean():.1f}  median {int(np.median(msa_depth[m]))}  zero_frac {(msa_depth[m]==0).mean():.3f}")
    print(f"    is_orphan     {is_orphan[m].mean():.3f}")


def step2_outputs(name, m, p, br, y):
    pm, bm, ym = p[m], br[m], y[m]
    print(f"\n  [{name}] OUTPUT")
    print(f"    score: mean {pm.mean():.3f}, std {pm.std():.3f}")
    print(f"      median by truth -- nonEss: {np.median(pm[ym==0]):.3f}  Ess: {np.median(pm[ym==1]):.3f}")
    print(f"    brightness: mean {bm.mean():.3f}  cov>=0.85: {(bm>=0.85).mean():.3f}")
    print(f"    calibration (binned):  bin           n      mean_p   actual_ess_rate")
    for label, n, mp, ae in calibration_table(pm, ym):
        print(f"                          {label:<13} {n:>6}  {mp:>6.3f}  {ae:>6.3f}")


def step3_errors(name, m, p, y):
    pm, ym = p[m], y[m]
    call = (pm > 0.5).astype(int)
    tp = int(((call == 1) & (ym == 1)).sum())
    fp = int(((call == 1) & (ym == 0)).sum())
    tn = int(((call == 0) & (ym == 0)).sum())
    fn = int(((call == 0) & (ym == 1)).sum())
    n_ess = int(ym.sum()); n_non = int((1 - ym).sum())
    print(f"\n  [{name}] ERRORS (threshold p>0.5)")
    print(f"    truth\\call    non-ess       ess")
    print(f"    non-ess       {tn:>7}   {fp:>7}    (FP rate = {fp/max(1,n_non):.3f})")
    print(f"    ess           {fn:>7}   {tp:>7}    (FN rate = {fn/max(1,n_ess):.3f}; recall = {tp/max(1,n_ess):.3f})")
    print(f"    accuracy: {(tp+tn)/max(1,len(pm)):.3f}")
    skew = "FN-heavy" if fn > fp else ("FP-heavy" if fp > fn else "balanced")
    print(f"    error skew: {skew}  ({fp} FP vs {fn} FN)")


def step4_orphan_mechanism(foc, y, is_orphan, msa_depth):
    """Are focal features used differently on core vs orphan?"""
    core = (msa_depth >= 6) & ~is_orphan
    orph = is_orphan
    print("\n  TRUE ORPHAN mechanism test: per-feature corr(focal, essentiality)")
    print("                              positive = feature -> essential more likely")
    print(f"    {'feature':<14} {'core':>9} {'orphan':>9} {'sign_flip?':>12}")
    flips = 0
    for i, fn in enumerate(FOC_NAMES[:foc.shape[1]]):
        c_core = np.corrcoef(foc[core, i], y[core])[0, 1]
        c_orph = np.corrcoef(foc[orph, i], y[orph])[0, 1]
        flip = (c_core * c_orph < 0) and abs(c_core) > 0.05 and abs(c_orph) > 0.05
        if flip: flips += 1
        print(f"    {fn:<14} {c_core:>+9.3f} {c_orph:>+9.3f} {('YES' if flip else '-'):>12}")
    print(f"    -> {flips} focal features have SIGN-FLIPPED relationships between core and orphan.")
    if flips >= 3:
        print("       Hypothesis CONFIRMED: orphan AUC<0.5 because the focal mapping learned on")
        print("       conserved genes is anti-aligned with orphan biology. Separate head needed.")
    else:
        print("       Hypothesis NOT clearly supported -- something else explains AUC<0.5.")


def step4_near_orphan_mechanism(p, y, msa_depth, is_orphan):
    """AUC binned by exact MSA depth -- is it monotonic?"""
    print("\n  NEAR-ORPHAN mechanism test: AUC by exact MSA depth")
    print(f"    {'depth':<7} {'n':>8} {'ess%':>6} {'AUC':>7} {'AUPRC':>7}")
    for d in range(1, 7):
        m = (msa_depth == d) & ~is_orphan
        if m.sum() < 50: continue
        a = auc(p[m], y[m]); ap = auprc(p[m], y[m])
        print(f"    {d:<7} {int(m.sum()):>8} {y[m].mean()*100:>5.1f}  {a:>7.3f} {ap:>7.3f}")
    deep = msa_depth >= 20
    if deep.sum() > 100:
        print(f"    {'>=20':<7} {int(deep.sum()):>8} {y[deep].mean()*100:>5.1f}  "
              f"{auc(p[deep], y[deep]):>7.3f} {auprc(p[deep], y[deep]):>7.3f}")


def main(preds_stem, seeds, cache_path, labels_dir):
    print("loading predictions and cache...")
    p, br, y, clade, meta_org, lt = load_avg_preds(preds_stem, seeds)
    cache = np.load(cache_path, allow_pickle=True)
    foc = cache["foc"]
    msa_depth = cache["msa_mask"].sum(1).astype(int)
    is_orphan = cache["is_orphan"].astype(bool) if "is_orphan" in cache.files else \
                np.zeros(len(y), bool)
    orgs = list(cache["orgs"])
    ev = ~np.isnan(p)
    p, br, y = p[ev], br[ev], y[ev]
    foc = foc[ev]; msa_depth = msa_depth[ev]; is_orphan = is_orphan[ev]
    meta_org = meta_org[ev]; lt = lt[ev]; clade = clade[ev]
    org_name = np.array([str(orgs[int(o)]) for o in meta_org])

    is_cond_map = load_conditional(labels_dir)
    cond = np.array([is_cond_map.get((str(org_name[i]), str(lt[i])), 0)
                     for i in range(len(p))], dtype=int)

    segments = [
        ("CONSERVED CORE", (msa_depth >= 6) & ~is_orphan & (cond == 0)),
        ("CONDITIONAL",    cond.astype(bool)),
        ("NEAR-ORPHAN",    (msa_depth >= 1) & (msa_depth < 6) & ~is_orphan),
        ("TRUE ORPHAN",    is_orphan),
    ]

    print("=" * 78); print("STEP 1 -- INPUT CHARACTERIZATION"); print("=" * 78)
    for name, m in segments:
        step1_inputs(name, m, foc, msa_depth, is_orphan)

    print("\n" + "=" * 78); print("STEP 2 -- OUTPUT CHARACTERIZATION"); print("=" * 78)
    for name, m in segments:
        step2_outputs(name, m, p, br, y)

    print("\n" + "=" * 78); print("STEP 3 -- ERROR DECOMPOSITION"); print("=" * 78)
    for name, m in segments:
        step3_errors(name, m, p, y)

    print("\n" + "=" * 78); print("STEP 4 -- MECHANISM TESTS"); print("=" * 78)
    step4_orphan_mechanism(foc, y, is_orphan, msa_depth)
    step4_near_orphan_mechanism(p, y, msa_depth, is_orphan)

    print("\n" + "=" * 78); print("STEP 5 -- CONCLUSIONS"); print("=" * 78)
    print("""
  Read off the segment numbers above. The pattern to look for in each:

  CORE      : high AUC + low confW_ess + good calibration on conserved genes
              => model is working as designed within scope.
  CONDITIONAL: AUC close to core + higher abstention (lower brCov) + same
              brAcc on what IS called => model has correctly learned to abstain
              on conditional uncertainty rather than mispredict.
  NEAR-ORPHAN: monotonic AUC rise with MSA depth (step 4) + brAcc still high
              when it does call => sparse-MSA degradation is GRACEFUL, not
              catastrophic. There IS a depth threshold for useful prediction.
  TRUE ORPHAN: if focal feature sign-flips between core and orphan (step 4),
              the AUC<0.5 is mechanistically explained -- the model trained on
              core learned features in a direction opposite to orphan biology.
              Implies a SEPARATE-HEAD architecture, not a feature change.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_stem", default="af_torch_preds_v2nosc_full_s")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default=str(OUT / "af_msa_cache_aug_full.npz"))
    ap.add_argument("--labels_dir", default="data/drive_import/labels_aug")
    a = ap.parse_args()
    main(a.preds_stem, a.seeds, a.cache, a.labels_dir)
