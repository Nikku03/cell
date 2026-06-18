"""Iterative refinement: eliminate high-confidence non-essentials, then chase essentials.

The loop:
  R1) Train a NON-ESSENTIAL classifier on the 5 training organisms (residual
      pool, full at start). Calibrate a score threshold on TRAINING data to
      hit precision >= 0.90 for the non-essential class. Eliminate held-out
      genes that score above that threshold. Re-fit an ESSENTIAL classifier
      on the surviving training residual; predict on surviving held-out.
      Report base rate, AUC, AUPRC, precision@top-k, coverage@P_target.
  R2) On the genes that SURVIVED R1, repeat. Re-train non-ess + ess on the
      new (smaller, harder) residual training pool. Threshold non-ess again.
  R3) Same again.

LOO across 6 bacteria. All thresholds are set on TRAINING data only (no
leakage). At the end, where did it converge -- what stayed unclassified?

Outputs:
  outputs/orphan/iterative_refinement.png
  outputs/orphan/iterative_refinement_summary.json
  outputs/orphan/iterative_refinement_residual_GMI1000.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
ARCHAEA = {"beril_Methanococcus_JJ","beril_Methanococcus_S2"}
df0 = pd.read_csv(OUT / "cross_org_coverage_scores.csv")
df0 = df0[~df0.organism.isin(ARCHAEA)].reset_index(drop=True)
BACT = list(dict.fromkeys(df0.organism))
FEATS = ["leading","oriC_frac","mobile_dist_kb","n_paralogs","gene_len_aa","gc","conservation"]

def fit_predict(Xtr,ytr,Xte,it=600,lr=0.5,l2=2.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=np.c_[np.ones(len(Xtr)),(Xtr-mu)/sd]; Xte=np.c_[np.ones(len(Xte)),(Xte-mu)/sd]
    w=np.zeros(Xtr.shape[1]); ytr=ytr.astype(float)
    pos=ytr.mean() or 1e-6; cw=np.where(ytr==1,0.5/pos,0.5/(1-pos))
    for _ in range(it):
        p=1/(1+np.exp(-Xtr@w)); g=Xtr.T@((p-ytr)*cw)/len(Xtr)
        g[1:]+=l2*w[1:]/len(Xtr); w-=lr*g
    return 1/(1+np.exp(-Xtr@w)), 1/(1+np.exp(-Xte@w))

def auc(s,y):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    ranks=np.argsort(np.argsort(s))+1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def auprc(s,y):
    o=np.argsort(-s); ys=np.asarray(y)[o]
    tp=np.cumsum(ys); k=np.arange(1,len(ys)+1)
    prec=tp/k; rec=tp/max(1,ys.sum())
    return float(np.sum(prec*np.diff(np.concatenate([[0],rec]))))

def thr_at_prec(s, y, target=0.9, min_n=20):
    """Threshold on the training-fit predictions for the positive class such
    that PRECISION >= target at maximum coverage. Returns the score threshold."""
    o=np.argsort(-s); ys=np.asarray(y)[o].astype(float); ss=np.asarray(s)[o]
    k=np.arange(1,len(ys)+1); prec=np.cumsum(ys)/k
    ok=(prec>=target) & (k>=min_n)
    if not ok.any(): return np.inf
    return float(ss[np.where(ok)[0].max()])

# track per-organism, per-round metrics
results=[]
df = df0.copy()
df["round_killed"] = 0          # round in which gene was eliminated (0=alive)
df["round_ess_score"] = np.nan  # last essential score it got

N_ROUNDS = 3
PREC_TARGET = 0.90

for rnd in range(1, N_ROUNDS+1):
    alive = df.round_killed == 0
    print(f"\n========== ROUND {rnd}  (alive genes: {alive.sum()}) ==========")
    for held in BACT:
        te_mask = (df.organism == held) & alive
        tr_mask = (df.organism != held) & alive
        te = df[te_mask]; tr = df[tr_mask]
        if len(te) < 20 or tr.essential.nunique() < 2:
            continue
        # ---- non-essential classifier ----
        ytr_ne = (1 - tr.essential).to_numpy()
        ptr_ne, pte_ne = fit_predict(tr[FEATS].to_numpy(float), ytr_ne, te[FEATS].to_numpy(float))
        thr = thr_at_prec(ptr_ne, ytr_ne, target=PREC_TARGET)
        kill_mask_te = pte_ne >= thr   # eliminate these as confident non-ess
        # ---- essential classifier on the SURVIVING training pool ----
        survive_tr = ptr_ne < thr
        if survive_tr.sum() < 50 or tr.essential[survive_tr].nunique() < 2:
            continue
        _, pte_e = fit_predict(tr[FEATS].to_numpy(float)[survive_tr],
                               tr.essential.to_numpy()[survive_tr],
                               te[FEATS].to_numpy(float))
        # metrics on the SURVIVING held-out
        survive_te = ~kill_mask_te
        te_idx = te.index.to_numpy()
        # write back ess scores for surviving genes
        df.loc[te_idx[survive_te], "round_ess_score"] = pte_e[survive_te]
        # mark killed genes
        df.loc[te_idx[kill_mask_te], "round_killed"] = rnd
        # compute metrics on the held-out SURVIVING genes
        yte_surv = te.essential.to_numpy()[survive_te]
        sc_surv  = pte_e[survive_te]
        if len(yte_surv) >= 20 and yte_surv.sum() >= 5:
            base = float(yte_surv.mean())
            a = auc(sc_surv, yte_surv); ap = auprc(sc_surv, yte_surv)
            o = np.argsort(-sc_surv); ys = yte_surv[o]
            top10 = ys[:max(1,int(0.10*len(ys)))].mean()
            top25 = ys[:max(1,int(0.25*len(ys)))].mean()
            # coverage@P>=0.7
            k = np.arange(1,len(ys)+1); prec_cum=np.cumsum(ys)/k
            ok7 = (prec_cum >= 0.7) & (k >= 20)
            cov7 = float(k[np.where(ok7)[0].max()])/len(ys) if ok7.any() else 0.0
            results.append(dict(
                round=rnd, organism=held,
                n_alive_te=int(survive_te.sum()),
                n_killed_te=int(kill_mask_te.sum()),
                base_ess=round(base,3),
                ess_auc=round(a,3), ess_auprc=round(ap,3),
                top10_prec=round(float(top10),3),
                top25_prec=round(float(top25),3),
                cov_at_p70=round(cov7,3)))
            print(f"  {held.replace('beril_',''):<24} alive={survive_te.sum():<5} killed={kill_mask_te.sum():<5} "
                  f"base={base:.3f}  AUC={a:.2f}  AP={ap:.2f}  "
                  f"P@10%={top10:.2f}  P@25%={top25:.2f}  cov@P0.7={cov7:.2f}")

res = pd.DataFrame(results)
print("\n=== ROUND MEANS (across 6 bacteria) ===")
print(f"{'round':<6}{'alive':>8}{'killed':>8}{'base':>8}{'AUC':>7}{'AUPRC':>8}{'P@10':>7}{'P@25':>7}{'cov@P0.7':>10}")
round_summary = []
for r in range(1, N_ROUNDS+1):
    sub = res[res["round"]==r]
    if len(sub)==0: continue
    row = dict(round=r,
        alive=int(sub.n_alive_te.mean()), killed=int(sub.n_killed_te.mean()),
        base=round(float(sub.base_ess.mean()),3),
        auc=round(float(sub.ess_auc.mean()),3),
        auprc=round(float(sub.ess_auprc.mean()),3),
        p10=round(float(sub.top10_prec.mean()),3),
        p25=round(float(sub.top25_prec.mean()),3),
        cov_p70=round(float(sub.cov_at_p70.mean()),3))
    round_summary.append(row)
    print(f"{r:<6}{row['alive']:>8}{row['killed']:>8}{row['base']:>8}{row['auc']:>7}"
          f"{row['auprc']:>8}{row['p10']:>7}{row['p25']:>7}{row['cov_p70']:>10}")

# what stayed at the end
final_alive = df[df.round_killed == 0]
final_killed = df[df.round_killed > 0]
ess_in_alive = final_alive[final_alive.essential==1]
ess_in_killed = final_killed[final_killed.essential==1]
print(f"\nFINAL STATE (across 6 bacteria pool):")
print(f"  total genes: {len(df)}")
print(f"  killed as non-ess across rounds: {len(final_killed)} "
      f"({len(final_killed)/len(df)*100:.0f}%)")
print(f"  remaining alive: {len(final_alive)}")
print(f"  TRUE essentials in alive set: {len(ess_in_alive)}  (of {df.essential.sum()} total ess; "
      f"recall {len(ess_in_alive)/df.essential.sum():.2f})")
print(f"  TRUE essentials wrongly killed: {len(ess_in_killed)}  "
      f"({len(ess_in_killed)/df.essential.sum()*100:.1f}% of all essentials)")
print(f"  alive base essential rate: {final_alive.essential.mean():.3f} "
      f"(vs {df.essential.mean():.3f} starting)")

# rogue genes outcome
rogue = df[(df.essential==1) & (df.conservation<0.1)]
rogue_alive = rogue[rogue.round_killed==0]
rogue_killed = rogue[rogue.round_killed>0]
print(f"\n  rogue essentials (cons<0.1): {len(rogue)} total, "
      f"{len(rogue_alive)} survived to final ({len(rogue_alive)/max(1,len(rogue))*100:.0f}%), "
      f"{len(rogue_killed)} wrongly killed.")

# dump GMI1000 residual
gmi_final = df[df.organism=="beril_RalstoniaGMI1000"].copy()
atlas = pd.read_csv(OUT/"gene_dot_atlas_GMI1000.csv")[["locus_tag","product"]]
gmi_final = gmi_final.merge(atlas, on="locus_tag", how="left")
gmi_final[["locus_tag","product","essential","conservation","n_paralogs",
           "gene_len_aa","round_killed","round_ess_score"]].to_csv(
    OUT/"iterative_refinement_residual_GMI1000.csv", index=False)

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# A: pool shrinkage + base rate per round
ax = axes[0,0]
rs = list(range(0, N_ROUNDS+1))
alives = [len(df)/6] + [round_summary[i-1]["alive"] for i in range(1, N_ROUNDS+1)]
bases  = [df.essential.mean()] + [round_summary[i-1]["base"] for i in range(1, N_ROUNDS+1)]
ax.bar(rs, alives, color="#7F8C8D", alpha=0.85)
for i,a in enumerate(alives): ax.text(i, a, f" {int(a)}", ha="center", va="bottom", fontsize=9)
ax2 = ax.twinx()
ax2.plot(rs, bases, "o-", color="#C0392B", lw=2, label="essential rate (alive)")
for i,b in enumerate(bases): ax2.text(i, b, f" {b:.2f}", color="#C0392B", fontsize=9)
ax.set_xlabel("round"); ax.set_ylabel("mean alive genes per organism")
ax2.set_ylabel("essential rate in alive pool", color="#C0392B")
ax.set_title("A. Pool shrinks; essential rate concentrates")
ax.set_xticks(rs)

# B: AUC / AUPRC / coverage@P0.7 per round
ax = axes[0,1]
xs = list(range(1, N_ROUNDS+1))
ax.plot(xs, [r["auc"] for r in round_summary], "o-", label="AUC", color="#2980B9", lw=2)
ax.plot(xs, [r["auprc"] for r in round_summary], "o-", label="AUPRC", color="#16A085", lw=2)
ax.plot(xs, [r["cov_p70"] for r in round_summary], "o-", label="coverage@P≥0.7", color="#E67E22", lw=2)
for r in round_summary:
    ax.text(r["round"], r["auc"], f"  {r['auc']:.2f}", fontsize=9, color="#2980B9")
    ax.text(r["round"], r["auprc"], f"  {r['auprc']:.2f}", fontsize=9, color="#16A085")
    ax.text(r["round"], r["cov_p70"], f"  {r['cov_p70']:.2f}", fontsize=9, color="#E67E22")
ax.set_xlabel("round"); ax.set_xticks(xs); ax.set_ylim(0,1)
ax.set_title("B. Predictor quality on the surviving pool, by round")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# C: top-decile precision on shrinking pool
ax = axes[1,0]
ax.plot(xs, [r["p10"] for r in round_summary], "o-", color="#C0392B", lw=2, label="precision@top10%")
ax.plot(xs, [r["p25"] for r in round_summary], "o-", color="#8E44AD", lw=2, label="precision@top25%")
for r in round_summary:
    ax.text(r["round"], r["p10"], f"  {r['p10']:.2f}", fontsize=9, color="#C0392B")
    ax.text(r["round"], r["p25"], f"  {r['p25']:.2f}", fontsize=9, color="#8E44AD")
ax.set_xlabel("round"); ax.set_xticks(xs); ax.set_ylim(0,1)
ax.set_title("C. Top-of-list precision on the residual")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# D: where did the genes end up?  killed-by-round vs essential status
ax = axes[1,1]
cats = ["killed R1","killed R2","killed R3","still alive"]
ess_counts = [
    int(((df.round_killed==1)&(df.essential==1)).sum()),
    int(((df.round_killed==2)&(df.essential==1)).sum()),
    int(((df.round_killed==3)&(df.essential==1)).sum()),
    int(((df.round_killed==0)&(df.essential==1)).sum())]
ne_counts = [
    int(((df.round_killed==1)&(df.essential==0)).sum()),
    int(((df.round_killed==2)&(df.essential==0)).sum()),
    int(((df.round_killed==3)&(df.essential==0)).sum()),
    int(((df.round_killed==0)&(df.essential==0)).sum())]
xb = np.arange(4); w_=0.4
ax.bar(xb-w_/2, ne_counts, w_, color="#BDC3C7", label="true non-ess")
ax.bar(xb+w_/2, ess_counts, w_, color="#C0392B", label="true ess")
for i,(n,e) in enumerate(zip(ne_counts, ess_counts)):
    ax.text(i-w_/2, n, f"{n}", ha="center", va="bottom", fontsize=8)
    ax.text(i+w_/2, e, f"{e}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xb); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("number of genes (6 bacteria pooled)")
ax.set_title("D. Final disposition: who got killed when?")
ax.legend(fontsize=9)

fig.suptitle("Iterative refinement: eliminate confident non-essentials, then chase essentials, 3 rounds",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"iterative_refinement.png", dpi=140, bbox_inches="tight")

(OUT/"iterative_refinement_summary.json").write_text(json.dumps({
    "n_rounds": N_ROUNDS, "precision_target_for_kill": PREC_TARGET,
    "round_summary": round_summary,
    "final_pool": {
        "total": int(len(df)),
        "killed": int(len(final_killed)),
        "alive": int(len(final_alive)),
        "alive_base_ess": round(float(final_alive.essential.mean()),3),
        "essentials_alive": int(len(ess_in_alive)),
        "essentials_killed": int(len(ess_in_killed)),
        "essential_recall_alive": round(float(len(ess_in_alive)/df.essential.sum()),3),
    },
    "rogue_zone": {
        "total_rogue": int(len(rogue)),
        "rogue_alive": int(len(rogue_alive)),
        "rogue_killed": int(len(rogue_killed)),
    },
    "per_round_per_org": results,
},indent=2))
print("\nwrote iterative_refinement.png + summary + GMI1000 residual csv")
