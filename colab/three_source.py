"""Three-source essentiality consensus on E. coli Keio.

Wires the b-number bridge -> FBA flux essentiality as a THIRD independent
source alongside the transformer (Wheel 1) and cross-organism conservation
(Wheel 2). Each source is mechanistically different:

  W1 model     : sequence + ortholog patterns       (what's conserved)
  W2 conserv.  : leave-one-clade-out OG essentiality (what's essential elsewhere)
  W3 FBA flux  : metabolic-network solvability      (what's needed for biomass)

If they AGREE -> very high precision. If they DISAGREE -> the genes are
mechanistically interpretable (auxotrophic rescue, non-metabolic essential,
or model-only signal). On the 494 strict-bridged Keio genes with all three
signals + wet-lab truth, we measure: per-source accuracy, pairwise +
3-source consensus precision, and ablation lift.

Output: outputs/orphan/three_source.png / .json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
ORG = "beril_Keio"

# ---------- load ----------
truth = {r["locus_tag"]: int(r["essential"])
         for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
         if r["organism"] == ORG}
ann = {r["locus_tag"]: r["product"]
       for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv"))
       if r["organism"] == ORG and r.get("product")}
bridge = {}; sources = {}
for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv")):
    bridge[r["locus_tag"]] = r["b_number"]; sources[r["locus_tag"]] = r["source"]
b2gene = {}; b2prod = {}
for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv")):
    b2gene[r["b_number"]] = r["gene_name"]; b2prod[r["b_number"]] = r["product"]
fba = {r["locus_tag"]: int(r["fba_essential"])
       for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv"))
       if r["organism"] == ORG}
STRICT = {"anchor_product", "anchor_product_round2", "synteny_walk"}

# transformer (W1) + brightness
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
clade = P["clade"].astype(str); lt_arr = P["lt"].astype(str)
p_arr = P["p"].astype(float); br_arr = P["brightness"].astype(float)
ec_idx = (clade == "escherichia")
lt2pred = {lt_arr[i]: (p_arr[i], br_arr[i])
           for i in np.where(ec_idx)[0]
           if not np.isnan(p_arr[i])}

# conservation (W2) leak-clean OG-universality (mtub is not in beril; for Keio
# we exclude beril_Keio itself from the rate)
og_of = {}; og_present = defaultdict(set)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r.get("og_id"):
        og_of[(r["organism"], r["locus_tag"])] = r["og_id"]
        og_present[r["og_id"]].add(r["organism"])
og_essn = defaultdict(int); og_essd = defaultdict(int)
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"] == ORG: continue
    og = og_of.get((r["organism"], r["locus_tag"]))
    if og: og_essd[og] += 1; og_essn[og] += int(r["essential"])
og_rate = {og: og_essn[og] / og_essd[og] for og in og_essd if og_essd[og] >= 5}

# ---------- assemble the 3-source table ----------
keys = [lt for lt in truth
        if lt in bridge and sources[lt] in STRICT
        and bridge[lt] in fba and lt in lt2pred
        and og_of.get((ORG, lt)) in og_rate]
print(f"Joined 3-source + truth: {len(keys)} genes")

rows = []
for lt in keys:
    p, br = lt2pred[lt]
    rows.append(dict(lt=lt, b=bridge[lt], gene=b2gene[bridge[lt]],
                     product=b2prod[bridge[lt]],
                     truth=truth[lt],
                     w1_model=int(p >= 0.5),
                     w2_cons=int(og_rate[og_of[(ORG, lt)]] >= 0.5),
                     w3_fba=fba[bridge[lt]],
                     score_w1=float(p), score_w2=float(og_rate[og_of[(ORG, lt)]])))

# ---------- per-source accuracy ----------
def metrics(preds, ys):
    tp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, ys) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, ys) if p == 0 and y == 0)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(P=round(p, 3), R=round(r, 3),
                F1=round(2 * p * r / max(p + r, 1e-9), 3),
                TP=tp, FP=fp, FN=fn, TN=tn, n=len(ys))

ys = [r["truth"] for r in rows]
print(f"\n=== per-source accuracy on the joined set (n={len(rows)}) ===")
for src in ("w1_model", "w2_cons", "w3_fba"):
    m = metrics([r[src] for r in rows], ys)
    print(f"  {src:<10}  P={m['P']:.3f}  R={m['R']:.3f}  F1={m['F1']:.3f}  "
          f"(TP {m['TP']} FP {m['FP']} FN {m['FN']} TN {m['TN']})")

# ---------- pairwise & 3-source consensus ----------
print(f"\n=== consensus calls ===")
# All three agree on essential
ttt = [r for r in rows if r["w1_model"] == r["w2_cons"] == r["w3_fba"] == 1]
nnn = [r for r in rows if r["w1_model"] == r["w2_cons"] == r["w3_fba"] == 0]
maj2_ess = [r for r in rows
            if r["w1_model"] + r["w2_cons"] + r["w3_fba"] >= 2]
maj2_non = [r for r in rows
            if r["w1_model"] + r["w2_cons"] + r["w3_fba"] <= 1]
def prec(group, target):
    if not group: return None
    return sum(1 for r in group if r["truth"] == target) / len(group)

print(f"  3-way ESSENTIAL agreement (TTT): n={len(ttt)}, "
      f"precision={prec(ttt, 1):.3f}  (essential by truth)")
print(f"  3-way NON-ESS agreement (NNN):  n={len(nnn)}, "
      f"precision={prec(nnn, 0):.3f}  (non-essential by truth)")
print(f"  2-of-3 ESSENTIAL majority:      n={len(maj2_ess)}, "
      f"precision={prec(maj2_ess, 1):.3f}")
print(f"  2-of-3 NON-ESS majority:        n={len(maj2_non)}, "
      f"precision={prec(maj2_non, 0):.3f}")

# pairwise (model+conserv only, vs all 3)
mc_ess = [r for r in rows if r["w1_model"] == 1 and r["w2_cons"] == 1]
mcf_ess = [r for r in rows
           if r["w1_model"] == 1 and r["w2_cons"] == 1 and r["w3_fba"] == 1]
print(f"\n  model+conservation agree on essential: n={len(mc_ess)}, "
      f"precision={prec(mc_ess, 1):.3f}")
print(f"  + FBA also agrees:                       n={len(mcf_ess)}, "
      f"precision={prec(mcf_ess, 1):.3f}  "
      f"(+{(prec(mcf_ess, 1) - prec(mc_ess, 1))*100:+.1f}pp)")

# ---------- ablation: does adding W3 (FBA) improve over W1+W2? ----------
def best_logit_score(features, ys):
    """Quick AUC of a min-feature logistic combo on (n, k) features."""
    X = np.array(features); y = np.array(ys)
    # standardise + bias
    Xs = np.c_[np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)]
    w = np.zeros(Xs.shape[1])
    for _ in range(500):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        w -= 0.3 * Xs.T @ (pr - y) / len(X)
    sc = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
    o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(len(sc))
    pos = y == 1
    auc = (r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())
    return auc, sc

ys_arr = np.array([r["truth"] for r in rows])
auc_w1, _ = best_logit_score([[r["score_w1"]] for r in rows], ys_arr)
auc_w2, _ = best_logit_score([[r["score_w2"]] for r in rows], ys_arr)
auc_w3, _ = best_logit_score([[r["w3_fba"]]  for r in rows], ys_arr)
auc_w12, _ = best_logit_score([[r["score_w1"], r["score_w2"]] for r in rows], ys_arr)
auc_w13, _ = best_logit_score([[r["score_w1"], r["w3_fba"]] for r in rows], ys_arr)
auc_w23, _ = best_logit_score([[r["score_w2"], r["w3_fba"]] for r in rows], ys_arr)
auc_all, _ = best_logit_score([[r["score_w1"], r["score_w2"], r["w3_fba"]]
                                for r in rows], ys_arr)
print(f"\n=== AUC ablation (logistic combos, in-sample on n={len(rows)}) ===")
print(f"  W1 model only    : {auc_w1:.3f}")
print(f"  W2 conservation  : {auc_w2:.3f}")
print(f"  W3 FBA only      : {auc_w3:.3f}")
print(f"  W1+W2            : {auc_w12:.3f}")
print(f"  W1+W3            : {auc_w13:.3f}")
print(f"  W2+W3            : {auc_w23:.3f}")
print(f"  W1+W2+W3 (all 3) : {auc_all:.3f}  <- FBA adds {auc_all-auc_w12:+.3f} vs W1+W2")

# ---------- discordance highlights ----------
disc_fba_only = [r for r in rows
                 if r["w3_fba"] == 1 and r["w1_model"] == 0 and r["w2_cons"] == 0]
disc_no_fba = [r for r in rows
               if r["w3_fba"] == 0 and r["w1_model"] == 1 and r["w2_cons"] == 1]
print(f"\n=== unique-contribution discordances ===")
print(f"  FBA-only essential calls (W3 says yes, W1+W2 say no): {len(disc_fba_only)}")
print(f"    of these, TRUTH says essential: "
      f"{sum(1 for r in disc_fba_only if r['truth']==1)}  "
      f"(FBA catching what sequence+conservation miss)")
for r in sorted(disc_fba_only, key=lambda x: -x["truth"])[:5]:
    print(f"    {r['b']:>6} {r['gene']:<6} truth={r['truth']}  -- {r['product'][:55]}")
print(f"  Model+conservation say essential but FBA says no: {len(disc_no_fba)}")
print(f"    of these, TRUTH says essential: "
      f"{sum(1 for r in disc_no_fba if r['truth']==1)}  "
      f"(FBA blind to non-metabolic essentials)")
for r in sorted(disc_no_fba, key=lambda x: -x["truth"])[:5]:
    print(f"    {r['b']:>6} {r['gene']:<6} truth={r['truth']}  -- {r['product'][:55]}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# (a) per-source bar + consensus
labels = ["W1 model", "W2 conserv", "W3 FBA", "2-of-3", "3-of-3"]
precs_ess = [
    metrics([r["w1_model"] for r in rows], ys)["P"],
    metrics([r["w2_cons"] for r in rows], ys)["P"],
    metrics([r["w3_fba"]  for r in rows], ys)["P"],
    prec(maj2_ess, 1),
    prec(ttt, 1),
]
recs_ess = [
    metrics([r["w1_model"] for r in rows], ys)["R"],
    metrics([r["w2_cons"] for r in rows], ys)["R"],
    metrics([r["w3_fba"]  for r in rows], ys)["R"],
    len([r for r in maj2_ess if r["truth"]==1]) / max(sum(ys), 1),
    len([r for r in ttt if r["truth"]==1]) / max(sum(ys), 1),
]
x = np.arange(len(labels))
axes[0].bar(x - 0.2, precs_ess, 0.4, color="#3182bd", label="precision")
axes[0].bar(x + 0.2, recs_ess, 0.4, color="#e31a1c", label="recall")
axes[0].axhline(0.9, color="k", ls="--", lw=1, alpha=0.5)
for i, (p, r) in enumerate(zip(precs_ess, recs_ess)):
    axes[0].text(i - 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
    axes[0].text(i + 0.2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9, rotation=10)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Essential-call precision & recall per source\n"
                  f"(joined set, n={len(rows)})", fontweight="bold", fontsize=11)
axes[0].legend(fontsize=9)

# (b) 3-way Venn-like agreement matrix
patterns = [(1,1,1), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1), (0,0,0)]
labels2 = ["W1∩W2∩W3", "W1∩W2 only", "W1∩W3 only", "W2∩W3 only",
           "W1 only", "W2 only", "W3 only", "none"]
counts = []; pos_frac = []
for w1, w2, w3 in patterns:
    g = [r for r in rows
         if r["w1_model"]==w1 and r["w2_cons"]==w2 and r["w3_fba"]==w3]
    counts.append(len(g))
    pos_frac.append(sum(1 for r in g if r["truth"]==1) / max(len(g), 1))
yp = np.arange(len(labels2))
axes[1].barh(yp, counts, color="#9ecae1")
for i, (c, f) in enumerate(zip(counts, pos_frac)):
    if c > 0:
        col = "#e31a1c" if f >= 0.5 else "#888"
        axes[1].text(c + 3, i, f"n={c}, P_ess={f:.2f}", va="center",
                     fontsize=9, color=col,
                     fontweight="bold" if f >= 0.9 else "normal")
axes[1].set_yticks(yp); axes[1].set_yticklabels(labels2, fontsize=9)
axes[1].invert_yaxis(); axes[1].set_xlabel("# genes in that agreement pattern")
axes[1].set_title("3-source agreement matrix\n"
                  "(red label = >50% truly essential in that bucket)",
                  fontweight="bold", fontsize=11)

# (c) AUC ablation
ablation = [("W1", auc_w1), ("W2", auc_w2), ("W3", auc_w3),
            ("W1+W2", auc_w12), ("W1+W3", auc_w13), ("W2+W3", auc_w23),
            ("W1+W2+W3", auc_all)]
x = np.arange(len(ablation))
cols = ["#888"]*3 + ["#9ecae1"]*3 + ["#e31a1c"]
axes[2].bar(x, [a for _, a in ablation], color=cols)
for i, (lbl, a) in enumerate(ablation):
    axes[2].text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
axes[2].set_xticks(x); axes[2].set_xticklabels([l for l, _ in ablation],
                                                fontsize=9, rotation=15)
axes[2].set_ylabel("AUC (in-sample)"); axes[2].set_ylim(0.5, 1.0)
axes[2].set_title(f"AUC ablation: does FBA add signal?\n"
                  f"3-source AUC {auc_all:.3f} vs W1+W2 {auc_w12:.3f}  "
                  f"({(auc_all-auc_w12)*100:+.1f}pp)",
                  fontweight="bold", fontsize=11)

fig.suptitle("Three-source essentiality consensus on E. coli "
             "(model + conservation + FBA flux)", fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "three_source.png", dpi=120, bbox_inches="tight")
print(f"\nwrote outputs/orphan/three_source.png")

json.dump(dict(n_joined=len(rows), n_essential_truth=int(sum(ys)),
               per_source={"w1": metrics([r["w1_model"] for r in rows], ys),
                            "w2": metrics([r["w2_cons"] for r in rows], ys),
                            "w3": metrics([r["w3_fba"] for r in rows], ys)},
               consensus_3way_ess=dict(n=len(ttt), precision=round(prec(ttt, 1), 3)),
               consensus_2of3_ess=dict(n=len(maj2_ess), precision=round(prec(maj2_ess, 1), 3)),
               auc=dict(w1=round(float(auc_w1), 3), w2=round(float(auc_w2), 3),
                        w3=round(float(auc_w3), 3),
                        w1w2=round(float(auc_w12), 3),
                        w1w3=round(float(auc_w13), 3),
                        w2w3=round(float(auc_w23), 3),
                        all3=round(float(auc_all), 3),
                        fba_lift_over_w1w2=round(float(auc_all - auc_w12), 3))),
          open(OUT / "three_source.json", "w"), indent=2)
print("wrote outputs/orphan/three_source.json")
