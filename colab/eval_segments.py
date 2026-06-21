"""Per-segment evaluation for the extended cache (orphans + conditionals).

Splits the evaluated set into four scientifically distinct segments and reports
AUC + neCov@P90 + brightness coverage on each:

  1. CONSERVED CORE  -- MSA depth >= 6, is_orphan=0, is_conditional=0
  2. CONDITIONAL     -- is_conditional=1 (in cache; intermediate signal)
  3. NEAR-ORPHAN     -- MSA depth 1-5, is_orphan=0 (sparse family)
  4. TRUE ORPHAN     -- is_orphan=1 (no OG, empty MSA)

Reads outputs/orphan/af_torch_preds_<tag>.npz (multi-seed-averaged if a stem is
given) and the full cache to recover is_orphan, msa_mask, and the per-gene
is_conditional looked up from regulator_features.csv.
"""
import csv, json, argparse
from pathlib import Path
import numpy as np

from af_common import OUT, auc, auprc, cov_at_p


def load_preds(stem, seeds):
    if seeds is None:
        z = np.load(OUT / f"{stem}.npz", allow_pickle=True)
        return z["p"], z["brightness"], z["y"], z["clade"], z["meta_org"], z["lt"]
    P = []; BR = []
    for s in seeds:
        z = np.load(OUT / f"{stem}{s}.npz", allow_pickle=True)
        P.append(z["p"]); BR.append(z["brightness"])
    z = np.load(OUT / f"{stem}{seeds[0]}.npz", allow_pickle=True)
    return (np.mean(P, axis=0), np.mean(BR, axis=0),
            z["y"], z["clade"], z["meta_org"], z["lt"])


def segment_metrics(name, m, p, br, y):
    n = int(m.sum())
    if n < 50:
        return dict(segment=name, n=n)
    pm, bm, ym = p[m], br[m], y[m]
    hi = bm >= 0.85
    call = (pm > 0.5).astype(int)
    bright_n = int(hi.sum())
    bright_acc = float((call[hi] == ym[hi].astype(int)).mean()) if bright_n else float("nan")
    confidently_wrong = (float((pm[ym == 1] < 0.2).mean())
                         if (ym == 1).any() else float("nan"))
    return dict(segment=name, n=n, ess_rate=round(float(ym.mean()), 3),
                auc=round(auc(pm, ym), 3), auprc=round(auprc(pm, ym), 3),
                neCov_P90=round(cov_at_p(1 - pm, 1 - ym, 0.9), 3),
                bright_cov=round(bright_n / n, 3),
                bright_acc=round(bright_acc, 3) if bright_acc == bright_acc else None,
                confidently_wrong_ess=round(confidently_wrong, 3)
                if confidently_wrong == confidently_wrong else None)


def main(preds_stem, seeds, cache_path, labels_dir):
    cache = np.load(cache_path, allow_pickle=True)
    is_orphan = cache["is_orphan"].astype(bool) if "is_orphan" in cache.files else \
                np.zeros(len(cache["y"]), bool)
    msa_depth = cache["msa_mask"].sum(1).astype(int)
    orgs = list(cache["orgs"])

    # per-gene is_conditional lookup
    is_conditional_map = {}
    for d in (Path(labels_dir), Path("data/drive_import/labels")):
        p = d / "regulator_features.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                try:
                    is_conditional_map[(r["organism"], r["locus_tag"])] = \
                        int(float(r.get("is_conditional", 0) or 0))
                except Exception: pass
            break

    p, br, y, clade, meta_org, lt = load_preds(preds_stem, seeds)
    ev = ~np.isnan(p)
    p, br, y, clade = p[ev], br[ev], y[ev], clade[ev]
    is_orphan = is_orphan[ev]; msa_depth = msa_depth[ev]
    meta_org = meta_org[ev]; lt = lt[ev]
    org_name = np.array([str(orgs[int(o)]) for o in meta_org])
    cond = np.array([is_conditional_map.get((str(org_name[i]), str(lt[i])), 0)
                     for i in range(len(p))], dtype=int)

    print(f"evaluated genes: {len(p):,}")
    print(f"  orphans     : {int(is_orphan.sum()):,}")
    print(f"  conditional : {int(cond.sum()):,}")
    print(f"  near-orphan (depth 1-5, not orphan): "
          f"{int(((msa_depth >= 1) & (msa_depth < 6) & ~is_orphan).sum()):,}")
    print(f"  conserved   : {int(((msa_depth >= 6) & ~is_orphan & (cond == 0)).sum()):,}\n")

    segments = [
        ("ALL",            np.ones_like(p, bool)),
        ("CONSERVED CORE", (msa_depth >= 6) & ~is_orphan & (cond == 0)),
        ("CONDITIONAL",    cond.astype(bool)),
        ("NEAR-ORPHAN",    (msa_depth >= 1) & (msa_depth < 6) & ~is_orphan),
        ("TRUE ORPHAN",    is_orphan),
    ]
    print(f"{'segment':<18} {'n':>8} {'ess%':>6} {'AUC':>6} {'AUPRC':>7} {'neCov':>7} "
          f"{'brCov':>6} {'brAcc':>6} {'confW_ess':>10}")
    print("-" * 84)
    rows = []
    for name, m in segments:
        r = segment_metrics(name, m, p, br, y)
        rows.append(r)
        if r.get("n", 0) >= 50:
            print(f"{name:<18} {r['n']:>8} {r['ess_rate']*100:>5.1f}  "
                  f"{r['auc']:>6.3f} {r['auprc']:>7.3f} {r['neCov_P90']:>7.3f} "
                  f"{r['bright_cov']:>6.3f} "
                  f"{r['bright_acc'] if r['bright_acc'] is not None else 0:>6.3f} "
                  f"{r['confidently_wrong_ess'] if r['confidently_wrong_ess'] is not None else 0:>10.3f}")
        else:
            print(f"{name:<18} {r.get('n', 0):>8}  (too small)")
    json.dump(dict(segments=rows), open(OUT / "eval_segments.json", "w"), indent=2)
    print("\nwrote eval_segments.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_stem", default="af_torch_preds_v2nosc_full_s",
                    help="prediction file stem (sX.npz appended if --seeds set)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--cache", default=str(OUT / "af_msa_cache_aug_full.npz"))
    ap.add_argument("--labels_dir", default="data/drive_import/labels_aug")
    a = ap.parse_args()
    main(a.preds_stem, a.seeds, a.cache, a.labels_dir)
