"""PATH-ORPHAN, VALIDATION -- leave-one-ORGANISM-out ESM test (the decisive one).

The within-organism CV result (rogue-zone R@P30=0.843, ESM-driven) is real and
not compositional, but within-org CV is OPTIMISTIC: it lets the model exploit
organism-specific proteome structure. The project's gold standard -- and the
deployment scenario -- is leave-one-ORGANISM-out: train on OTHER organisms,
predict a held-out organism's rogue essentials cold. This script runs that.

If LOO-org R@P30 holds in the rogue zone, ESM genuinely detects rogue essentials
that conservation cannot, and that transfers to novel organisms -> a result.
If it collapses toward 0, ESM was memorizing org structure -> same lesson the
rogue specialist taught, caught honestly.

INPUT (real):
  outputs/orphan/esm_all.parquet   (org, locus_tag, esm_0..esm_k)  -- ALL orgs
                                   embedded together with a GLOBAL PCA so the
                                   PCs are comparable across organisms
  labels.csv, orthology_features.csv (conservation leak-free per org)
OUTPUT: outputs/orphan/loo_<org>.json

--smoke : synthetic multi-org data. Verifies the test PASSES on a
          cross-org-transferable feature AND FAILS on an org-specific-only
          feature (i.e. the test actually catches non-generalization).
--real  : run LOO holding out --org
"""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "drive_import"
OUT = REPO / "outputs" / "orphan"
ROGUE_CONS = 0.10

# reuse the verified model primitives
sys.path.insert(0, str(REPO / "scripts"))
from orphan_model import _logreg_cv_oof, recall_at_precision  # noqa: E402


def _fit_predict(Xtr, ytr, Xte, iters=400, lr=0.5, l2=1.0):
    """Train logistic regression on train org(s), predict held-out org."""
    import numpy as np
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6
    cw = np.where(ytr == 1, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xtr @ w))
        g = Xtr.T @ ((p - ytr) * cw) / len(Xtr)
        g[1:] += l2 * w[1:] / len(Xtr)
        w -= lr * g
    return 1.0 / (1.0 + np.exp(-Xte @ w))


def loo_eval(df, feat_cols, held_org, clade_regex=None):
    """df: rows with columns organism, essential, conservation, feat_cols.
    Train on all orgs not in the held-out set, predict held_org.
    If clade_regex is given, EXCLUDE every org matching it from training
    (leave-one-CLADE-out, the strict test -- no sister organisms leak in)."""
    import numpy as np, re
    te = df[df.organism == held_org]
    if clade_regex:
        tr = df[~df.organism.str.contains(clade_regex, case=False, regex=True)]
    else:
        tr = df[df.organism != held_org]
    Xtr = tr[feat_cols].to_numpy(float); ytr = tr.essential.to_numpy(int)
    Xte = te[feat_cols].to_numpy(float); yte = te.essential.to_numpy(int)
    pred = _fit_predict(Xtr, ytr, Xte)
    rogue = te.conservation.to_numpy() < ROGUE_CONS
    out = {
        "held_org": held_org, "clade_regex": clade_regex,
        "n_train": len(tr), "n_train_orgs": int(tr.organism.nunique()),
        "n_test": len(te), "test_rogue_n": int(rogue.sum()),
        "test_rogue_essential": int(yte[rogue].sum()),
        "rogue_RatP30": round(recall_at_precision(pred[rogue], yte[rogue]), 4),
        "whole_RatP30": round(recall_at_precision(pred, yte), 4),
    }
    return out, pred, rogue, yte


def run_real(args):
    import pandas as pd, numpy as np
    esm = OUT / "esm_all.parquet"
    if not esm.exists():
        print(f"ERROR: {esm} missing -- embed ALL orgs first (orphan_esm "
              f"--multi on GPU). Needs the global-PCA combined embedding."); return 2
    E = pd.read_parquet(esm)
    esm_cols = [c for c in E.columns if c.startswith("esm_")]
    # labels + conservation
    lab = pd.read_csv(DATA / "labels" / "labels.csv")[
        ["organism", "locus_tag", "essential"]]
    orth = pd.read_csv(DATA / "labels" / "orthology_features.csv")[
        ["organism", "locus_tag", "family_frac_essential_leakfree"]].rename(
        columns={"family_frac_essential_leakfree": "conservation"})
    df = (E.merge(lab, on=["organism", "locus_tag"], how="inner")
            .merge(orth, on=["organism", "locus_tag"], how="inner"))
    df = df.dropna(subset=["conservation"])
    feat_cols = esm_cols + ["conservation"]
    print("=" * 70)
    print(f"LEAVE-ONE-ORGANISM-OUT ESM -- hold out {args.org}")
    print(f"  orgs: {df.organism.nunique()}  genes: {len(df):,}  "
          f"esm PCs: {len(esm_cols)}")
    print("=" * 70)
    cr = args.clade_regex or None
    res, pred, rogue, yte = loo_eval(df, feat_cols, args.org, cr)
    res_esm, *_ = loo_eval(df, esm_cols, args.org, cr)
    mode = (f"leave-one-CLADE-out (exclude /{cr}/ from training)"
            if cr else "leave-one-ORGANISM-out (sisters stay in training)")
    print(f"  mode: {mode}")
    print(f"  train: {res['n_train_orgs']} orgs / {res['n_train']:,} genes  ->  "
          f"held-out {args.org}: {res['n_test']} genes, "
          f"{res['test_rogue_essential']} rogue essentials")
    print(f"  ROGUE-ZONE R@P30 (ESM+conservation): {res['rogue_RatP30']:.3f}")
    print(f"  ROGUE-ZONE R@P30 (ESM only)        : {res_esm['rogue_RatP30']:.3f}")
    print(f"  whole-genome R@P30                 : {res['whole_RatP30']:.3f}")
    print(f"\n  within-org CV was 0.843; this is the HONEST cross-org number.")
    verdict = ("HOLDS -- ESM detects rogue essentials in a NOVEL organism"
               if res["rogue_RatP30"] >= 0.10 else
               "COLLAPSES -- within-org signal did NOT transfer (memorization)")
    print(f"  VERDICT: {verdict}")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "clade" if cr else "org"
    (OUT / f"loo_{tag}_{args.org}.json").write_text(json.dumps(
        {**res, "mode": mode, "rogue_RatP30_esm_only": res_esm["rogue_RatP30"],
         "within_org_cv_reference": 0.843}, indent=2))
    print(f"\n  wrote loo_{args.org}.json")
    return 0


def run_smoke():
    import numpy as np, pandas as pd
    print("=" * 70)
    print("LOO-ORG SMOKE -- the test catches transferable vs memorized signal")
    print("=" * 70)
    rng = np.random.RandomState(0)
    orgs = [f"org{i}" for i in range(5)]
    rows = []
    for o in orgs:
        oi = int(o[-1])
        for g in range(300):
            ess = int(rng.rand() < 0.12)
            cons = rng.rand() * 0.09          # all rogue zone
            # transferable feature: same essential->signal mapping in every org
            f_transfer = ess * 1.8 + rng.normal(0, 1)
            # org-specific feature: essential->signal but with a per-org SHIFT
            # and sign flip, so a model trained on other orgs can't use it
            f_memorize = (ess * 1.8 * (1 if oi % 2 == 0 else -1)
                          + oi * 5 + rng.normal(0, 1))
            rows.append((o, ess, cons, f_transfer, f_memorize))
    df = pd.DataFrame(rows, columns=["organism", "essential", "conservation",
                                     "f_transfer", "f_memorize"])
    rt, *_ = loo_eval(df, ["f_transfer"], "org0")
    rm, *_ = loo_eval(df, ["f_memorize"], "org0")
    print(f"  transferable feature -> LOO rogue R@P30 = {rt['rogue_RatP30']:.3f}")
    print(f"  org-specific feature -> LOO rogue R@P30 = {rm['rogue_RatP30']:.3f}")
    assert rt["rogue_RatP30"] > 0.3, "transferable signal must survive LOO"
    assert rm["rogue_RatP30"] < rt["rogue_RatP30"], \
        "org-specific signal must NOT transfer as well (test catches memorization)"
    print("\n=== LOO-ORG SMOKE PASS ===")
    print("  LOO machinery validated: HOLDS on transferable signal, drops on")
    print("  org-specific signal. This is what makes the real verdict trustworthy.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--clade_regex", default="",
                    help="exclude all orgs matching this from training "
                         "(leave-one-clade-out; e.g. 'Ralstonia')")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
