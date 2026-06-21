"""Headline deployment number with a multi-seed confidence interval.

Runs the locked config -- v1 transformer (--big) on the DEG-augmented cache,
all clades, then the GBM rescue stacker on the base-8 + fba_essential feature
subset -- for N seeds, and reports mean +/- 1.96*SE on combined coverage @ the
0.90 precision floor. This is the number the paper should quote (with its CI),
addressing the single-seed point-estimate critique.

  python colab/multiseed_headline.py --cache outputs/orphan/af_msa_cache_aug.npz \\
      --labels_dir data/drive_import/labels_aug --seeds 0 1 2
"""
import argparse, json
import numpy as np

from af_common import Cache, OUT
import af_torch
from smart_cooccur import (load_orthology, channel_features, evaluate, FEATURE_COLS)
from rescue_models import fit_gbm, pred_gbm

KEEP = set(["p", "brightness", "backup", "presence", "coess", "phyl", "dnds", "has_dnds",
            "fba_essential", "has_fba"])
ZERO_COLS = [i for i, c in enumerate(FEATURE_COLS) if c not in KEEP]   # drop reg/transp/codon


def one_seed(cache_path, labels_dir, seed, orth):
    cfg = af_torch.AFConfig(heads=4, head_dim=16, hid=64, blocks=2, epochs=12, bs=8192)
    cfg.seed = seed
    all_p, all_c, res = af_torch.run_loco(cfg, save=False, cache_path=cache_path,
                                          eval_clades="all")
    c = Cache(cache_path)
    preds = dict(p=all_p, brightness=all_c, y=c.y, clade=c.clade,
                 meta_org=c.meta_org, lt=c.lt, foc=c.foc,
                 orgs=np.array(c.orgs, dtype=object))
    X, y, meta = channel_features(preds, orth, use_dnds=True)
    for j in ZERO_COLS:
        X[:, j] = 0.0
    r = evaluate(preds, X, y, meta, fit_fn=fit_gbm, pred_fn=pred_gbm)
    return res["pooled"]["auc"], r["smart_combined"]["coverage"], r["smart_combined"]["precision"]


def main(cache_path, labels_dir, seeds):
    orth = load_orthology(labels_dir=labels_dir)
    rows = []
    for s in seeds:
        auc, cov, prec = one_seed(cache_path, labels_dir, s, orth)
        rows.append((s, auc, cov, prec))
        print(f"  seed {s}: AUC={auc:.4f}  combined coverage={cov:.4f} @ {prec:.4f}")
    covs = np.array([r[2] for r in rows]); aucs = np.array([r[1] for r in rows])
    n = len(covs)
    def ci(a):
        m = a.mean(); se = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return m, 1.96 * se
    mc, hc = ci(covs); ma, ha = ci(aucs)
    print(f"\n  n={n} seeds")
    print(f"  pooled AUC          : {ma:.4f} +/- {ha:.4f}")
    print(f"  combined coverage   : {mc:.4f} +/- {hc:.4f}   (95% CI {mc-hc:.4f}-{mc+hc:.4f})  @ 0.90 precision")
    out = dict(seeds=list(seeds), per_seed=[dict(seed=r[0], auc=r[1], coverage=r[2], precision=r[3]) for r in rows],
               coverage_mean=round(float(mc), 4), coverage_ci95=round(float(hc), 4),
               auc_mean=round(float(ma), 4), auc_ci95=round(float(ha), 4),
               feature_set="base-8 + fba_essential")
    json.dump(out, open(OUT / "multiseed_headline.json", "w"), indent=2)
    print("  wrote multiseed_headline.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(OUT / "af_msa_cache_aug.npz"))
    ap.add_argument("--labels_dir", default="data/drive_import/labels_aug")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.cache, a.labels_dir, a.seeds)
