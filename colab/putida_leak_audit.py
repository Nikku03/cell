"""Fast leak-clean precision audit on Putida — vectorised version of stage 5.

The earlier full-test stuck on 200k row Python list-comp dict lookups.
This rebuilds the same matrices in <30s using a single pass + np.array.

Disentangles family_essrate_fold_mean (suspected leakage) from
family_essrate_leakfree (legitimate) on the pseudomonas held-out clade.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path("outputs/orphan")
HELD = "pseudomonas"

# ---- predictions + clades ----
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))
C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs = [str(x) for x in C["orgs"]]
o2c = {corgs[int(C["meta_org"][i])]: str(C["clade"][i]) for i in range(len(C["meta_org"]))}

# ---- single-pass orthology load: build dict THEN one vectorised lookup ----
# key = (org, lt)
orth_key = {}; orth_vals = []
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    orth_key[(r["organism"], r["locus_tag"])] = len(orth_vals)
    fm = float(np.mean([float(r.get(f"family_frac_essential_fold{i}") or 0)
                         for i in range(5)]))
    orth_vals.append((float(r.get("n_paralogs_in_genome") or 0),
                      float(r.get("family_n_organisms") or 0),
                      float(r.get("is_orphan") or 0),
                      float(r.get("family_size_total") or 0),
                      float(r.get("family_frac_essential_leakfree") or 0),
                      fm,
                      r.get("og_id") or ""))
orth_arr = np.array([(a, b, c, d, e, f) for a, b, c, d, e, f, g in orth_vals])
orth_og = [g for a, b, c, d, e, f, g in orth_vals]
print(f"orthology loaded: {len(orth_arr)} rows")

# obylt for org resolution
obylt = defaultdict(list)
for (o, ltt), i in orth_key.items():
    obylt[ltt].append(o)

# resolve org per prediction
res_idx = np.full(len(p), -1)
for i in range(len(p)):
    cands = obylt.get(lt[i], [])
    match = [o for o in cands if o2c.get(o) == clade[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    if ch is not None:
        res_idx[i] = orth_key.get((ch, lt[i]), -1)
print(f"resolved orthology: {int((res_idx >= 0).sum())}")

# vectorised columns
have = res_idx >= 0
par = np.zeros(len(p)); brd = np.zeros(len(p)); orph = np.zeros(len(p))
size = np.zeros(len(p)); lf = np.zeros(len(p)); fm = np.zeros(len(p))
par[have] = orth_arr[res_idx[have], 0]
brd[have] = orth_arr[res_idx[have], 1]
orph[have] = orth_arr[res_idx[have], 2]
size[have] = orth_arr[res_idx[have], 3]
lf[have] = orth_arr[res_idx[have], 4]
fm[have] = orth_arr[res_idx[have], 5]

# OG-universality LOCO
og_arr_str = np.array([orth_og[res_idx[i]] if res_idx[i] >= 0 else "" for i in range(len(p))])
gs2 = defaultdict(float); gn2 = defaultdict(float)
csum = defaultdict(float); cn = defaultdict(float)
for i in np.where(valid)[0]:
    og = og_arr_str[i]
    if not og: continue
    gs2[og] += y[i]; gn2[og] += 1
    csum[(og, clade[i])] += y[i]; cn[(og, clade[i])] += 1
ou = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    og = og_arr_str[i]
    if not og: continue
    n_ = gn2[og] - cn[(og, clade[i])]
    if n_ > 0: ou[i] = (gs2[og] - csum[(og, clade[i])]) / n_

mask = valid & have & ~np.isnan(ou)
ou_safe = np.where(np.isnan(ou), 0, ou)

# 4 feature configs
F_BASE = np.column_stack([p, br, ou_safe, np.log1p(par), np.log1p(brd), orph])
F_LF = np.column_stack([F_BASE, np.log1p(size), lf])
F_FM = np.column_stack([F_BASE, np.log1p(size), fm])
F_ALL = np.column_stack([F_BASE, np.log1p(size), lf, fm])

def fit(X, yy, l2=1.0, it=300, lr=0.4):
    X = np.nan_to_num(X)
    mu_ = X.mean(0); sd_ = np.maximum(X.std(0), 1e-3)
    Xs = np.c_[np.ones(len(X)), (X - mu_) / sd_]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** 0.5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(it):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X)
        g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu_, sd_
def predL(m, X):
    X = np.nan_to_num(X); w, mu_, sd_ = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu_) / sd_] @ w, -30, 30)))
def cov(s, yt, pos=True, t=0.9):
    sc = s if pos else -s; ys = yt if pos else 1 - yt
    o = np.argsort(-sc); o_ys = ys[o]; tp = np.cumsum(o_ys); fp = np.cumsum(1 - o_ys)
    prec = tp / np.maximum(tp + fp, 1); ok = prec >= t
    return float(tp[ok].max() / max(ys.sum(), 1)) if ok.any() else 0.0

ab = mask & (br < 0.85)
tr = ab & (clade != HELD); te = ab & (clade == HELD)
print(f"train: {int(tr.sum())}  test (pseudomonas held-out): {int(te.sum())}")

def runF(F, label):
    m_ = fit(F[tr], y[tr])
    pred = predL(m_, F[te])
    return dict(label=label, n=int(te.sum()),
                essCov=round(cov(pred, y[te].astype(int), True), 3),
                neCov=round(cov(pred, y[te].astype(int), False), 3))

r_base = runF(F_BASE, "baseline (6)")
r_lf = runF(F_LF, "+size + leakfree")
r_fm = runF(F_FM, "+size + fold_mean")
r_all = runF(F_ALL, "+size + leakfree + fold_mean")
print()
for r in (r_base, r_lf, r_fm, r_all):
    print(f"  {r['label']:<38} essCov={r['essCov']:.3f}  neCov={r['neCov']:.3f}")
print(f"\nLIFTS over baseline (pseudomonas held-out):")
print(f"  leakfree alone   : essCov +{(r_lf['essCov']-r_base['essCov'])*100:.1f}pp"
      f"   neCov +{(r_lf['neCov']-r_base['neCov'])*100:.1f}pp   [legitimate]")
print(f"  fold_mean alone  : essCov +{(r_fm['essCov']-r_base['essCov'])*100:.1f}pp"
      f"   neCov +{(r_fm['neCov']-r_base['neCov'])*100:.1f}pp   [suspect]")
print(f"  both             : essCov +{(r_all['essCov']-r_base['essCov'])*100:.1f}pp"
      f"   neCov +{(r_all['neCov']-r_base['neCov'])*100:.1f}pp")

json.dump(dict(held=HELD,
               baseline=r_base, leakfree=r_lf, fold_mean=r_fm, both=r_all,
               delta_leakfree_essCov_pp=round((r_lf["essCov"] - r_base["essCov"])*100, 1),
               delta_fold_mean_essCov_pp=round((r_fm["essCov"] - r_base["essCov"])*100, 1),
               delta_both_essCov_pp=round((r_all["essCov"] - r_base["essCov"])*100, 1)),
          open(OUT / "putida_leak_audit.json", "w"), indent=2)
print("\nwrote outputs/orphan/putida_leak_audit.json")
