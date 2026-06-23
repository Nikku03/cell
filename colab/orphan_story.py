"""The orphan-gene story: can essentiality be predicted WITHOUT orthologs?

True orphans (no OG) cannot use the conservation signal the whole project rests
on. The earlier "orphan AUC 0.35" was an ARTIFACT: build_cache gave orphans
DEFAULTED (constant) focal features, so the model had no varying input -- it was
not a real test of orphan predictability.

This is the real test. For the 9,082 orphans that have codon features (mtub,
ecoli, saur, spne), use only ORTHOLOGY-FREE signals:
  cai      -- codon adaptation index (expression-level proxy; highly-expressed
              genes are more often essential)
  gc, gc3  -- composition (HGT / origin proxy)
  cds_length
  intergenic_prev/next + same_strand_prev/next -- local operon structure
Then LEAVE-ONE-ORGANISM-OUT: train on 3 orgs, predict the 4th's orphan
essentiality. If AUC >> 0.5 this refutes "orphans are unpredictable".

Output: outputs/orphan/orphan_story.json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path("outputs/orphan")


def auc(s, y):
    s = np.asarray(s); y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cov_at_p(s, y, t, mn=20):
    o = np.argsort(-s); ys = np.asarray(y, float)[o]
    k = np.arange(1, len(ys) + 1); pr = np.cumsum(ys) / k
    ok = (pr >= t) & (k >= mn)
    return float(k[np.where(ok)[0].max()] / len(ys)) if ok.any() else 0.0


def fit_logit_bal(X, y, l2=1.0, iters=500, lr=0.5):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Xs.shape[1])
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    sw = cw[y.astype(int)]; sw = sw / sw.mean()
    n = len(X)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((p - y) * sw) / n
        g[1:] += l2 * w[1:] / n
        w -= lr * g
    return w, mu, sd


def pred_logit(model, X):
    w, mu, sd = model
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))


# --- load labels + identify orphans ---
labels = {(r["organism"], r["locus_tag"]): int(r["essential"])
          for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))}
og = set((r["organism"], r["locus_tag"])
         for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")))
orphans = set(k for k in labels if k not in og)
print(f"orphans: {len(orphans):,}  ({sum(labels[k] for k in orphans)/len(orphans)*100:.1f}% essential)")

# --- load codon features for orphans ---
FEAT = ["cai", "gc", "gc3", "log_len",
        "intergenic_prev", "intergenic_next",
        "same_strand_prev", "same_strand_next"]
rows = []
for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv")):
    k = (r["organism"], r["locus_tag"])
    if k not in orphans:
        continue
    def fv(x, d=np.nan):
        try: return float(x)
        except Exception: return d
    cds_len = fv(r.get("cds_length"), 0)
    rows.append(dict(
        org=k[0], y=labels[k],
        cai=fv(r.get("cai"), 0.5), gc=fv(r.get("gc"), 0.5), gc3=fv(r.get("gc3"), 0.5),
        log_len=np.log1p(cds_len if cds_len == cds_len else 0),
        intergenic_prev=fv(r.get("intergenic_prev"), 0),
        intergenic_next=fv(r.get("intergenic_next"), 0),
        same_strand_prev=fv(r.get("same_strand_prev"), 0),
        same_strand_next=fv(r.get("same_strand_next"), 0)))
print(f"orphans with codon features: {len(rows):,}")

orgs = sorted(set(r["org"] for r in rows))
X = np.array([[r[f] for f in FEAT] for r in rows], np.float32)
X = np.nan_to_num(X, nan=0.0)
y = np.array([r["y"] for r in rows], np.int8)
org_arr = np.array([r["org"] for r in rows])
print(f"organisms: {orgs}")

# =============================================================================
# 1. univariate AUC of each orthology-free feature
# =============================================================================
print("\n== univariate AUC (orphan essentiality, pooled) ==")
uni = {}
for fi, fn in enumerate(FEAT):
    a = auc(X[:, fi], y); a2 = auc(-X[:, fi], y)
    best = max(a, a2); sign = "+" if a >= a2 else "-"
    uni[fn] = round(best, 3)
    print(f"  {fn:<18} AUC {best:.3f}  ({sign})")

# =============================================================================
# 2. LEAVE-ONE-ORGANISM-OUT: the honest cross-organism orphan test
# =============================================================================
print("\n== leave-one-organism-out orphan essentiality prediction ==")
print(f"  {'held org':<26} {'n':>6} {'ess%':>6} {'AUC':>7} {'neCov@P90':>10} {'essCov@P70':>11}")
print("  " + "-" * 70)
per_org = {}
all_p = np.full(len(y), np.nan)
for held in orgs:
    te = org_arr == held; tr = ~te
    if te.sum() < 50 or y[tr].sum() < 10:
        continue
    M = fit_logit_bal(X[tr], y[tr])
    p = pred_logit(M, X[te])
    all_p[te] = p
    a = auc(p, y[te])
    ne = cov_at_p(1 - p, 1 - y[te], 0.9)
    es = cov_at_p(p, y[te], 0.7)
    per_org[held] = dict(n=int(te.sum()), ess_rate=round(float(y[te].mean()), 3),
                          auc=round(a, 3), neCov_p90=round(ne, 3), essCov_p70=round(es, 3))
    print(f"  {held:<26} {int(te.sum()):>6} {y[te].mean()*100:>5.1f} {a:>7.3f} "
          f"{ne:>10.3f} {es:>11.3f}")

# pooled
ev = ~np.isnan(all_p)
pooled_auc = auc(all_p[ev], y[ev])
pooled_ne = cov_at_p(1 - all_p[ev], 1 - y[ev], 0.9)
print(f"\n  POOLED (LOO-org): AUC {pooled_auc:.3f}  neCov@P90 {pooled_ne:.3f}")
print(f"  baseline (random): AUC 0.500   |  prior 'artifact' run: AUC 0.350")

# =============================================================================
# 3. CAI-only vs full model (is it all just expression?)
# =============================================================================
print("\n== CAI-only vs full feature set (LOO-org pooled AUC) ==")
cai_only = np.full(len(y), np.nan)
for held in orgs:
    te = org_arr == held; tr = ~te
    if te.sum() < 50 or y[tr].sum() < 10:
        continue
    M = fit_logit_bal(X[tr][:, [0]], y[tr])   # cai is col 0
    cai_only[te] = pred_logit(M, X[te][:, [0]])
ev2 = ~np.isnan(cai_only)
cai_auc = auc(cai_only[ev2], y[ev2])
print(f"  CAI alone : AUC {cai_auc:.3f}")
print(f"  full set  : AUC {pooled_auc:.3f}  (Δ {pooled_auc-cai_auc:+.3f} from genome structure)")

json.dump(dict(
    n_orphans=len(orphans),
    orphan_ess_rate=round(sum(labels[k] for k in orphans)/len(orphans), 3),
    n_with_codon=len(rows),
    organisms=orgs,
    univariate_auc=uni,
    per_org_loo=per_org,
    pooled_loo_auc=round(pooled_auc, 3),
    pooled_loo_neCov_p90=round(pooled_ne, 3),
    cai_only_auc=round(cai_auc, 3),
    note="orthology-free orphan essentiality prediction; LOO-org over 4 orgs with codon features",
), open(OUT / "orphan_story.json", "w"), indent=2)
print("\nwrote outputs/orphan/orphan_story.json")
