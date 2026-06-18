"""Second round trip: can the conductor's NOTES alone predict the platforms?

The first ride proved individual notes are enriched/depleted for essentials.
Now we make it adversarial: hand a model ONLY the 16 binary architecture notes
(no sequence, no AA composition, no conservation, no gene identity) and ask it
to predict essentiality LEAVE-ONE-ORGANISM-OUT across the 10 organisms.

This isolates the question: how far does PURE GENOME ARCHITECTURE travel?

We compare three rungs on the SAME LOO split:
  (1) architecture-only  -- the 16 notes (this ride)
  (2) conservation-only  -- the known strong $0 baseline (family essentiality frac)
  (3) architecture + conservation -- does layout ADD over conservation?

Reports AUC + AUPRC per organism and pooled, plus precision-at-coverage so it
slots into the earlier risk-coverage framing. Also prints standardized note
coefficients so we see WHICH notes the model leans on cross-organism.

Outputs:
  outputs/orphan/train_ride_roundtrip.png
  outputs/orphan/train_ride_roundtrip_summary.json
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/cell/data/drive_import")
OUT  = Path("/home/user/cell/outputs/orphan")

NOTES = ["DEJA_VU","WRONG_SIDE","EMPTY_PLATFORM","EXPRESS","FOREIGN_SIGN",
         "TINY","GRAND_CENTRAL","HEADWIND","SCRAPYARD","LONE_STAR",
         "NEW_LINE","HOMETOWN","REMOTE","DEEP_OPERON","GC_NORMAL",
         "SAME_STRAND_RUN"]

df = pd.read_parquet(OUT / "train_ride_stations.parquet")
df = df[df.essential.notna()].copy()
df["essential"] = df["essential"].astype(int)

# attach conservation (family essentiality fraction, leak-free) from orthology
cons = {}
for r in csv.DictReader(open(DATA / "labels" / "orthology_features.csv")):
    v = r.get("family_frac_essential_leakfree", "")
    cons[(r["organism"], r["locus_tag"])] = (
        float(v) if v not in ("", "nan", None) else np.nan)
df["conservation"] = [cons.get((o, l), np.nan)
                      for o, l in zip(df.organism, df.locus_tag)]
df["conservation"] = df["conservation"].fillna(0.0)

ORGS = list(dict.fromkeys(df.organism))
print(f"{len(df):,} labeled stations across {len(ORGS)} organisms\n")


def fit_predict(Xtr, ytr, Xte, iters=600, lr=0.5, l2=2.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    Xte = np.c_[np.ones(len(Xte)), (Xte-mu)/sd]
    w = np.zeros(Xtr.shape[1]); ytr = ytr.astype(float)
    pos = ytr.mean() or 1e-6
    cw = np.where(ytr == 1, 0.5/pos, 0.5/(1-pos))
    for _ in range(iters):
        p = 1/(1+np.exp(-Xtr@w))
        g = Xtr.T@((p-ytr)*cw)/len(Xtr); g[1:] += l2*w[1:]/len(Xtr)
        w -= lr*g
    return 1/(1+np.exp(-Xte@w)), w


def auprc(score, y):
    o = np.argsort(-score); ys = np.asarray(y)[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys)+1)
    prec = tp/k; rec = tp/max(1, ys.sum())
    return float(np.sum(prec*np.diff(np.concatenate([[0], rec]))))


def auc(score, y):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[y==1].sum() - n1*(n1+1)/2) / (n1*n0))


FS = {
    "architecture_only": NOTES,
    "conservation_only": ["conservation"],
    "arch_plus_cons":    NOTES + ["conservation"],
}

print("LEAVE-ONE-ORGANISM-OUT (AUC / AUPRC):")
print(f"  {'organism':<26} base | arch        cons        arch+cons")
results = []
score_arch = np.zeros(len(df)); df = df.reset_index(drop=True)
for held in ORGS:
    m = (df.organism == held).to_numpy()
    tr = df[~m]; te = df[m]; yte = te.essential.to_numpy()
    row = {"organism": held, "n": int(m.sum()), "base": round(float(yte.mean()), 3)}
    line = f"  {held.replace('beril_',''):<26} {yte.mean():.2f} |"
    for fs, cols in FS.items():
        pred, _ = fit_predict(tr[cols].to_numpy(float), tr.essential.to_numpy(),
                              te[cols].to_numpy(float))
        if fs == "architecture_only":
            score_arch[m] = pred
        ac = auc(pred, yte); ap = auprc(pred, yte)
        row[f"{fs}_auc"] = round(ac, 3); row[f"{fs}_auprc"] = round(ap, 3)
        line += f"  AUC {ac:.2f}/AP {ap:.2f}"
    results.append(row); print(line)
res = pd.DataFrame(results)
df["score_arch"] = score_arch

print("\n=== MEAN across 10 organisms ===")
for fs in FS:
    print(f"  {fs:<20} AUC {res[f'{fs}_auc'].mean():.3f}   "
          f"AUPRC {res[f'{fs}_auprc'].mean():.3f}")

# coefficients: train architecture-only on ALL orgs, standardized
X = df[NOTES].to_numpy(float); y = df.essential.to_numpy()
_, w = fit_predict(X, y, X)
coef = sorted(zip(NOTES, w[1:]), key=lambda t: -t[1])
print("\nstandardized note coefficients (architecture-only, all orgs):")
for n, v in coef:
    bar = ("+" if v >= 0 else "-") * int(min(30, abs(v)*40))
    print(f"  {n:<16} {v:+.3f} {bar}")

# precision @ coverage for architecture-only, bacteria pool (exclude archaea)
ARCH_ORGS = {"beril_Methanococcus_JJ", "beril_Methanococcus_S2"}
def cov_at_prec(score, yv, t, mc=20):
    o = np.argsort(-score); ys = np.asarray(yv)[o].astype(float)
    k = np.arange(1, len(ys)+1); prec = np.cumsum(ys)/k; cov = k/len(ys)
    rec = np.cumsum(ys)/max(1, ys.sum())
    ok = (prec >= t) & (k >= mc)
    if not ok.any(): return None
    i = np.where(ok)[0].max()
    return float(cov[i]), float(rec[i])

bac = df[~df.organism.isin(ARCH_ORGS)]
yv = bac.essential.to_numpy(); sv = bac.score_arch.to_numpy()
o = np.argsort(-sv); ys = yv[o]
for c in (0.10, 0.25, 0.50):
    k = int(c*len(ys)); print(f"  arch-only bacteria precision@top{int(c*100)}% = {ys[:k].mean():.2f}")
for t in (0.5, 0.7):
    r = cov_at_prec(sv, yv, t)
    print(f"  arch-only bacteria coverage@P>={t}: " +
          (f"{r[0]*100:.1f}% (recall {r[1]:.2f})" if r else "never"))

(OUT / "train_ride_roundtrip_summary.json").write_text(json.dumps({
    "n_labeled": int(len(df)), "n_orgs": len(ORGS),
    "mean_auc":   {fs: round(float(res[f"{fs}_auc"].mean()), 3) for fs in FS},
    "mean_auprc": {fs: round(float(res[f"{fs}_auprc"].mean()), 3) for fs in FS},
    "note_coefficients": {n: round(float(v), 3) for n, v in coef},
    "per_org": results,
}, indent=2))

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
ax = axes[0]
orgs_s = [o.replace("beril_","") for o in res.organism]
x = np.arange(len(res)); wbar = 0.26
for i,(fs,c) in enumerate([("architecture_only","#16A085"),
                           ("conservation_only","#3498DB"),
                           ("arch_plus_cons","#C0392B")]):
    ax.bar(x+(i-1)*wbar, res[f"{fs}_auc"], wbar,
           label=fs.replace("_"," "), color=c, alpha=0.88)
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(orgs_s, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("AUC (held-out organism)"); ax.set_ylim(0,1)
ax.set_title("Round trip: architecture-only vs conservation, LOO over 10 orgs")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
names = [n for n,_ in coef]; vals = [v for _,v in coef]
cols = ["#27AE60" if v>=0 else "#E74C3C" for v in vals]
yb = np.arange(len(names))[::-1]
ax.barh(yb, vals, color=cols, alpha=0.85)
ax.set_yticks(yb); ax.set_yticklabels(names, fontsize=9)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("standardized logistic weight")
mean_arch = res["architecture_only_auc"].mean()
mean_cons = res["conservation_only_auc"].mean()
mean_both = res["arch_plus_cons_auc"].mean()
ax.set_title(f"Which notes carry the signal\n"
             f"mean AUC: arch {mean_arch:.2f} | cons {mean_cons:.2f} | both {mean_both:.2f}")
ax.grid(alpha=0.3, axis="x")

fig.suptitle("How far does pure genome architecture travel? (conductor's 16 notes as the only features)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT / "train_ride_roundtrip.png", dpi=140, bbox_inches="tight")
print("\nwrote train_ride_roundtrip.png + summary")
