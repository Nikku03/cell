"""Diagnose how the deployment model handles ORPHAN and CONDITIONAL essentials.

Two distinct hard cases the writeup must address honestly:

  ORPHANS (no orthologs or very few): excluded from the cache at build time
  (build_cache.py:98 -- `if og is None: continue`). So the 0.834 headline does
  NOT cover them. We report (a) how many labelled genes were excluded as true
  orphans, (b) within the cache, how the model treats low-MSA-depth genes
  (the closest in-cache approximation to orphan-ness).

  CONDITIONAL (essential only under certain conditions): IN the coverage but
  unevenly handled. The binary essentiality label averages over assay
  conditions, so the model trained against it learns 'average essentiality'
  and under-predicts condition-specific essentials. Tagged by the prebuilt
  is_conditional flag in regulator_features.csv (lift only 1.04 on its own,
  confirming the signal is weak when collapsed to binary).

For each segment we report: n, essential rate, transformer-confident coverage
(brightness>=0.85) and its accuracy, abstention rate (= the GBM rescue's
universe), and -- crucially -- the fraction of essentials confidently called
non-essential (the conditional-residual pollution).

CPU only, reads outputs/orphan/af_torch_preds_v2nosc_s*.npz (3 seeds averaged).
"""
import csv, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

from af_common import OUT, DATA


def load_3seed_preds(stem="af_torch_preds_v2nosc_s"):
    P = []; BR = []
    for s in (0, 1, 2):
        z = np.load(OUT / f"{stem}{s}.npz", allow_pickle=True)
        P.append(z["p"]); BR.append(z["brightness"])
    z = np.load(OUT / f"{stem}0.npz", allow_pickle=True)
    return (np.mean(P, axis=0), np.mean(BR, axis=0),
            z["y"], z["clade"], z["meta_org"], z["lt"])


def load_labels_and_orth(labels_dir):
    labels = {}
    for r in csv.DictReader(open(labels_dir / "labels.csv")):
        labels[(r["organism"], r["locus_tag"])] = int(r["essential"])
    gene_og = {}; family_size = {}; n_paralogs = {}
    for r in csv.DictReader(open(labels_dir / "orthology_features.csv")):
        if r["og_id"]:
            k = (r["organism"], r["locus_tag"])
            gene_og[k] = r["og_id"]
            try: family_size[k] = int(float(r.get("family_n_organisms") or 1))
            except Exception: family_size[k] = 1
            try: n_paralogs[k] = int(float(r.get("n_paralogs_in_genome") or 0))
            except Exception: n_paralogs[k] = 0
    is_conditional = {}
    rf = labels_dir / "regulator_features.csv"
    if rf.exists():
        for r in csv.DictReader(open(rf)):
            try: is_conditional[(r["organism"], r["locus_tag"])] = int(float(r.get("is_conditional", 0) or 0))
            except Exception: pass
    return labels, gene_og, family_size, n_paralogs, is_conditional


def summarize(name, mask, p, br, y):
    n = int(mask.sum())
    if n == 0:
        return dict(n=0)
    ess = float(y[mask].mean())
    hi = (br[mask] >= 0.85)
    n_hi = int(hi.sum())
    call = (p[mask] > 0.5).astype(int)
    acc_hi = float((call[hi] == y[mask][hi].astype(int)).mean()) if n_hi else float("nan")
    abstain = float((~hi).mean())
    # confidently-wrong essentials: p < 0.2 AND truly essential
    truly_ess = y[mask] == 1
    confidently_wrong = float((p[mask][truly_ess] < 0.2).mean()) if truly_ess.any() else float("nan")
    return dict(n=n, ess_rate=round(ess, 3), bright_cov=round(n_hi / n, 3),
                bright_acc=round(acc_hi, 3), abstain=round(abstain, 3),
                ess_called_confident_NE=round(confidently_wrong, 3))


def main(labels_dir, preds_stem):
    labels_dir = Path(labels_dir)
    print(f"loading labels + orthology from {labels_dir} ...")
    labels, gene_og, family_size, n_paralogs, is_conditional = load_labels_and_orth(labels_dir)
    n_lab = len(labels); n_in_og = sum(1 for k in labels if k in gene_og)
    n_orphan = n_lab - n_in_og
    print(f"\n== ORPHAN GENES (true orphans excluded from the cache) ==")
    print(f"  total labelled genes  : {n_lab:,}")
    print(f"  with orthology (in cache): {n_in_og:,}")
    print(f"  TRUE ORPHANS excluded : {n_orphan:,}  ({n_orphan/n_lab*100:.1f}% of labels)")
    if n_orphan:
        orphan_ess = sum(labels[k] for k in labels if k not in gene_og) / n_orphan
        print(f"  orphan ess rate       : {orphan_ess:.3f}  (vs {sum(labels.values())/n_lab:.3f} overall)")
    print(f"  -> these are NOT in the headline 0.834 coverage.")

    print(f"\nloading 3-seed averaged v2 nosc predictions ...")
    p, br, y, clade, meta_org, lt = load_3seed_preds(preds_stem)
    cache = np.load(OUT / "af_msa_cache_aug.npz", allow_pickle=True)
    orgs = list(cache["orgs"]); msa_mask = cache["msa_mask"]
    msa_depth = msa_mask.sum(1).astype(int)         # # real ortholog rows per gene

    ev = ~np.isnan(p)
    p, br, y, clade = p[ev], br[ev], y[ev], clade[ev]
    msa_depth = msa_depth[ev]; meta_org = meta_org[ev]; lt = lt[ev]
    # per-gene is_conditional + family size
    org_name = np.array([str(orgs[int(o)]) for o in meta_org])
    cond = np.array([is_conditional.get((str(org_name[i]), str(lt[i])), 0)
                     for i in range(len(p))], dtype=int)
    fam = np.array([family_size.get((str(org_name[i]), str(lt[i])), 1)
                    for i in range(len(p))], dtype=int)
    print(f"  evaluated genes in cache: {len(p):,}")

    print(f"\n== COVERAGE BY MSA DEPTH (in-cache 'orphan-ness' proxy) ==")
    print(f"{'depth bucket':<20} {'n':>8} {'ess%':>6} {'bright_cov':>10} {'bright_acc':>10} {'abst':>6} {'confWrong_ess':>13}")
    print("-" * 80)
    for lo, hi, lab in [(1, 2, "1 (near-orphan)"), (2, 6, "2-5"), (6, 16, "6-15"), (16, 25, "16-24 (full)")]:
        m = (msa_depth >= lo) & (msa_depth < hi)
        r = summarize(lab, m, p, br, y)
        if r["n"]:
            print(f"{lab:<20} {r['n']:>8} {r['ess_rate']*100:>5.1f}  {r['bright_cov']:>10.3f} {r['bright_acc']:>10.3f} "
                  f"{r['abstain']:>6.3f} {r['ess_called_confident_NE']:>13.3f}")

    print(f"\n== COVERAGE BY FAMILY SIZE (# organisms with the OG) ==")
    print(f"{'family size':<20} {'n':>8} {'ess%':>6} {'bright_cov':>10} {'bright_acc':>10} {'confWrong_ess':>13}")
    for lo, hi, lab in [(1, 2, "1 (singleton-ish)"), (2, 5, "2-4"), (5, 15, "5-14"), (15, 999, "15+ (broad)")]:
        m = (fam >= lo) & (fam < hi)
        r = summarize(lab, m, p, br, y)
        if r["n"]:
            print(f"{lab:<20} {r['n']:>8} {r['ess_rate']*100:>5.1f}  {r['bright_cov']:>10.3f} {r['bright_acc']:>10.3f} "
                  f"{r['ess_called_confident_NE']:>13.3f}")

    print(f"\n== CONDITIONAL ESSENTIALS (regulator_features.is_conditional) ==")
    print(f"  total with is_conditional=1 (in cache): {int(cond.sum()):,}  "
          f"({cond.mean()*100:.1f}%)")
    print(f"{'segment':<28} {'n':>8} {'ess%':>6} {'bright_cov':>10} {'bright_acc':>10} {'confWrong_ess':>13}")
    print("-" * 84)
    for label, m in [("all genes", np.ones_like(cond, bool)),
                     ("is_conditional = 0", cond == 0),
                     ("is_conditional = 1", cond == 1)]:
        r = summarize(label, m, p, br, y)
        if r["n"]:
            print(f"{label:<28} {r['n']:>8} {r['ess_rate']*100:>5.1f}  {r['bright_cov']:>10.3f} {r['bright_acc']:>10.3f} "
                  f"{r['ess_called_confident_NE']:>13.3f}")

    # Cross: conditional essentials in burkholderia/magnetospirillum (the residual hot spots)
    print(f"\n== CONDITIONAL ESSENTIALS IN THE WEAK CLADES ==")
    print(f"{'clade':<22} {'n_cond':>8} {'ess|cond':>9} {'ess|other':>10} {'cond->NE':>9}")
    for cl in ("burkholderia", "magnetospirillum", "ralstonia", "pseudomonas"):
        m = (clade == cl)
        if m.sum() < 100: continue
        nc = int((m & (cond == 1)).sum())
        if nc == 0: continue
        e_cond = float(y[m & (cond == 1)].mean())
        e_other = float(y[m & (cond == 0)].mean()) if (m & (cond == 0)).sum() else float("nan")
        cond_to_ne = float((p[m & (cond == 1) & (y == 1)] < 0.2).mean()) if (m & (cond == 1) & (y == 1)).sum() else float("nan")
        print(f"{cl:<22} {nc:>8} {e_cond:>9.3f} {e_other:>10.3f} {cond_to_ne:>9.3f}")

    # save
    out = dict(n_orphans_excluded=int(n_orphan),
               orphan_pct=round(n_orphan / n_lab * 100, 2),
               cache_size=int(len(p)),
               n_conditional_in_cache=int(cond.sum()))
    json.dump(out, open(OUT / "diagnose_orphan_conditional.json", "w"), indent=2)
    print(f"\nwrote diagnose_orphan_conditional.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="data/drive_import/labels_aug")
    ap.add_argument("--preds_stem", default="af_torch_preds_v2nosc_s")
    a = ap.parse_args()
    main(a.labels_dir, a.preds_stem)
