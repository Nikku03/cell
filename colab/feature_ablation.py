"""Feature attribution for the GBM rescue stacker.

The new clean features (is_regulator/is_transporter/codon/fba) lifted v1+GBM
combined coverage ~0.725 -> ~0.804 @ 90% precision. This isolates WHICH of them
did it: same preds, same cross-clade GBM protocol, same precision floor, only
the feature subset changes (ablate by zeroing columns).

Reports, for each config, combined coverage @0.90 and deltas vs the full set and
vs the base-8 (original channels only).

  python colab/feature_ablation.py --cache outputs/orphan/af_msa_cache_aug.npz \\
      --preds outputs/orphan/af_torch_preds_aug.npz \\
      --labels_dir data/drive_import/labels_aug --model gbm
"""
import json, argparse
import numpy as np

from af_common import Cache, CACHE, OUT
from smart_cooccur import load_orthology, channel_features, evaluate, FEATURE_COLS

I = {c: k for k, c in enumerate(FEATURE_COLS)}
GROUPS = {
    "is_regulator":   [I["is_regulator"]],
    "is_transporter": [I["is_transporter"]],
    "codon":          [I["cai"], I["gc3"], I["has_codon"]],
    "fba":            [I["fba_essential"], I["has_fba"]],
}
NEW = sum(GROUPS.values(), [])


def get_model(name):
    if name == "gbm":
        from rescue_models import fit_gbm, pred_gbm
        return fit_gbm, pred_gbm
    from smart_cooccur import fit_logreg, pred_logreg
    return fit_logreg, pred_logreg


def run(X, y, meta, preds, ff, pf, zero_cols, label):
    Xa = X.copy()
    for c in zero_cols:
        Xa[:, c] = 0.0
    r = evaluate(preds, Xa, y, meta, label=label, fit_fn=ff, pred_fn=pf)
    return r["smart_combined"]["coverage"], r["smart_combined"]["precision"]


def main(cache_path, preds_path, labels_dir, model):
    preds = dict(np.load(preds_path or (OUT / "af_torch_preds.npz"), allow_pickle=True))
    c = Cache(cache_path or CACHE)
    preds["foc"] = c.foc; preds["orgs"] = np.array(c.orgs, dtype=object)
    orth = load_orthology(labels_dir=labels_dir)
    X, y, meta = channel_features(preds, orth, use_dnds=True)
    ff, pf = get_model(model)
    print(f"rescue model: {model}; abstained genes: {len(y)}; features: {len(FEATURE_COLS)}\n")

    configs = [("FULL (all features)", [])]
    configs += [("base-8 only (no new)", NEW)]
    configs += [(f"FULL - {g}", cols) for g, cols in GROUPS.items()]
    configs += [(f"base-8 + {g} only", [c for c in NEW if c not in cols])
                for g, cols in GROUPS.items()]

    res = {}
    full_cov = base_cov = None
    print(f"  {'config':<26} {'cov':>7} {'prec':>7} {'vsFULL':>8} {'vsBASE8':>8}")
    print("  " + "-" * 60)
    for label, zc in configs:
        cov, prec = run(X, y, meta, preds, ff, pf, zc, label)
        if label.startswith("FULL ("):
            full_cov = cov
        if label.startswith("base-8 only"):
            base_cov = cov
        res[label] = dict(coverage=cov, precision=prec)
        vf = "" if full_cov is None else f"{(cov-full_cov)*100:+.2f}"
        vb = "" if base_cov is None else f"{(cov-base_cov)*100:+.2f}"
        print(f"  {label:<26} {cov:>7.4f} {prec:>7.4f} {vf:>8} {vb:>8}")

    if full_cov and base_cov:
        print(f"\n  total new-feature lift: {(full_cov-base_cov)*100:+.2f}pp "
              f"({base_cov:.4f} -> {full_cov:.4f})")
    json.dump(res, open(OUT / "feature_ablation_results.json", "w"), indent=2)
    print("  wrote feature_ablation_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--labels_dir", default=None)
    ap.add_argument("--model", default="gbm", choices=["gbm", "logistic"])
    a = ap.parse_args()
    main(a.cache, a.preds, a.labels_dir, a.model)
