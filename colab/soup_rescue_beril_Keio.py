"""
Family-based rescue test for SOUP genes of beril_Keio (E. coli Keio).
Vectorized — no per-row dict.get loops.
"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

T0 = time.time()
def tlog(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

ORG = "beril_Keio"
CLADE = "escherichia"
LABELS = "/home/user/cell/data/drive_import/labels/labels.csv"
ORTHO  = "/home/user/cell/data/drive_import/labels/orthology_features.csv"
ANNOT  = "/home/user/cell/data/drive_import/labels/gene_annotations.csv"
CLADES = "/home/user/cell/data/drive_import/labels/clade_splits.csv"
PREDS  = "/home/user/cell/outputs/orphan/af_torch_preds_aug.npz"
OUT    = "/home/user/cell/outputs/orphan/soup_rescue_beril_Keio.json"

# ---------------------------------------------------------------------------
# 1) Load predictions and define SOUP
# ---------------------------------------------------------------------------
tlog("loading predictions npz")
z = np.load(PREDS, allow_pickle=True)
p = z["p"].astype(np.float32)
br = z["brightness"].astype(np.float32)
clade = z["clade"].astype(str)
lt = z["lt"].astype(str)

# Only escherichia-clade rows
mask_clade = clade == CLADE
tlog(f"  escherichia-clade preds: {mask_clade.sum()}")

# SOUP filter: 0.30 < p < 0.70 AND brightness < 0.85
soup_mask = mask_clade & (p > 0.30) & (p < 0.70) & (br < 0.85)
soup_lt = lt[soup_mask]
tlog(f"  raw soup rows: {soup_mask.sum()}, unique lt: {len(np.unique(soup_lt))}")

# We want beril_Keio loci only. Map locus_tags -> truth via labels.csv
tlog("loading labels.csv")
labels = pd.read_csv(LABELS, usecols=["organism","locus_tag","gene_name","essential"])
labels_keio = labels[labels["organism"] == ORG].copy()
keio_truth = dict(zip(labels_keio["locus_tag"].values, labels_keio["essential"].astype(int).values))
keio_gene  = dict(zip(labels_keio["locus_tag"].values, labels_keio["gene_name"].astype(str).values))
tlog(f"  beril_Keio labeled loci: {len(keio_truth)}")

# Restrict soup to those whose locus_tag is in beril_Keio labels
# (af_torch_preds_aug only stores lt+clade — multiple orgs may share clade)
soup_lt_unique = np.unique(soup_lt)
in_keio = np.array([l in keio_truth for l in soup_lt_unique])
soup_lt_keio = soup_lt_unique[in_keio]
truth_arr = np.array([keio_truth[l] for l in soup_lt_keio], dtype=np.int8)
soup_size = int(len(soup_lt_keio))
soup_essential_count = int(truth_arr.sum())
print(f"soup_size = {soup_size}")
print(f"soup_essential_count = {soup_essential_count}")
baseline_precision = soup_essential_count / max(soup_size, 1)
print(f"baseline_precision = {baseline_precision:.4f}")

# ---------------------------------------------------------------------------
# 2) Build leak-clean OG essential rate (exclude escherichia clade rows)
# ---------------------------------------------------------------------------
tlog("loading orthology + clade splits")
ortho = pd.read_csv(ORTHO, usecols=["organism","locus_tag","og_id","family_n_organisms"])
clades_df = pd.read_csv(CLADES, usecols=["organism","clade"])
org_to_clade = dict(zip(clades_df["organism"].values, clades_df["clade"].values))

# vectorized clade lookup
ortho["clade"] = ortho["organism"].map(org_to_clade)

# Merge labels (organism, locus_tag) -> essential. Vectorized merge.
tlog("merging labels into orthology")
merged = ortho.merge(
    labels[["organism","locus_tag","essential"]],
    on=["organism","locus_tag"],
    how="left",
)
# leak-clean: essential value only counted if clade != escherichia
mask_loco = (merged["clade"].values != CLADE)
ess_vals = merged["essential"].values  # may be NaN
has_label = ~pd.isna(ess_vals)
loco_count_mask = mask_loco & has_label

# group by og_id
tlog("groupby og for leak-clean rate")
og_ids = merged["og_id"].values
loco_df = pd.DataFrame({
    "og_id": og_ids[loco_count_mask],
    "ess":   ess_vals[loco_count_mask].astype(np.float32),
})
agg = loco_df.groupby("og_id", sort=False)["ess"].agg(["sum","count"])
agg = agg[agg["count"] >= 5]
agg["rate"] = agg["sum"] / agg["count"]
og_loco_rate = dict(zip(agg.index.values, agg["rate"].values.astype(np.float32)))
tlog(f"  OGs with >=5 leak-clean members: {len(og_loco_rate)}")

# family_breadth: distinct organisms carrying each OG (any clade)
tlog("family_breadth")
breadth = ortho.groupby("og_id", sort=False)["organism"].nunique()
og_breadth = dict(zip(breadth.index.values, breadth.values.astype(np.int32)))

# beril_Keio lt -> og_id
keio_ortho = ortho[ortho["organism"] == ORG]
lt_to_og = dict(zip(keio_ortho["locus_tag"].values, keio_ortho["og_id"].values))
tlog(f"  beril_Keio lt->og mappings: {len(lt_to_og)}")

# ---------------------------------------------------------------------------
# 3) Annotation consensus product per OG
#    only build for OGs that appear among soup genes (saves time)
# ---------------------------------------------------------------------------
tlog("loading annotations")
ann = pd.read_csv(ANNOT, usecols=["organism","locus_tag","product"])
# attach og
ann_m = ann.merge(ortho[["organism","locus_tag","og_id"]], on=["organism","locus_tag"], how="inner")
ann_m["product"] = ann_m["product"].fillna("").astype(str).str.strip()
pl = ann_m["product"].str.lower()
good_mask = (
    (pl != "")
    & (pl != "nan")
    & (~pl.str.contains("hypothetical", na=False))
    & (~pl.str.contains("uncharacterized", na=False))
)
ann_good = ann_m[good_mask]
tlog(f"  good annotations: {len(ann_good)}")

# soup OGs only
soup_ogs = set()
for l in soup_lt_keio:
    og = lt_to_og.get(l)
    if og is not None and not (isinstance(og, float) and np.isnan(og)):
        soup_ogs.add(og)
tlog(f"  unique soup OGs: {len(soup_ogs)}")

ann_soup = ann_good[ann_good["og_id"].isin(soup_ogs)]
# vectorized "most common product per og"
tlog("consensus_product per soup OG")
# use groupby + value_counts via mode-trick
def mode_first(s):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else ""
consensus = ann_soup.groupby("og_id")["product"].agg(mode_first)
og_consensus = dict(zip(consensus.index.values, consensus.values.astype(str)))
tlog(f"  consensus products: {len(og_consensus)}")

# ---------------------------------------------------------------------------
# 4) Score soup genes
# ---------------------------------------------------------------------------
tlog("scoring soup genes")
rows = []
for l, t in zip(soup_lt_keio, truth_arr):
    og = lt_to_og.get(l)
    if og is None or (isinstance(og, float) and np.isnan(og)):
        u = 0.0
        b = 0
        prod = ""
        og_str = ""
    else:
        u = float(og_loco_rate.get(og, 0.0))
        b = int(og_breadth.get(og, 0))
        prod = og_consensus.get(og, "")
        og_str = str(og)
    score = 0.7 * u + 0.3 * min(b/30.0, 1.0)
    rows.append((l, og_str, u, b, prod, int(t), score))

scored = pd.DataFrame(rows, columns=["lt","og","loco_rate","breadth","product","truth","score"])
scored = scored.sort_values("score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5) Metrics
# ---------------------------------------------------------------------------
K = soup_essential_count
topK = scored.head(K)
precision_topK = float(topK["truth"].mean()) if K > 0 else 0.0
lift_pp = (precision_topK - baseline_precision) * 100.0

# cov_at_p80: largest K such that cumulative precision >= 0.80
truth_sorted = scored["truth"].values.astype(np.float32)
cum_tp = np.cumsum(truth_sorted)
ks = np.arange(1, len(truth_sorted)+1, dtype=np.float32)
cum_prec = cum_tp / ks
valid = cum_prec >= 0.80
if valid.any():
    K_max = int(np.where(valid)[0].max()) + 1
    coverage_at_p80 = float(cum_tp[K_max-1] / max(soup_essential_count, 1))
else:
    K_max = 0
    coverage_at_p80 = 0.0

# top-15 candidates
top15 = []
for _, r in scored.head(15).iterrows():
    top15.append({
        "lt": r["lt"],
        "gene": keio_gene.get(r["lt"], ""),
        "og": r["og"],
        "score": round(float(r["score"]), 4),
        "consensus_product": r["product"],
        "true_essential": int(r["truth"]),
    })

result = {
    "organism": ORG,
    "clade": CLADE,
    "soup_size": soup_size,
    "soup_essentials": soup_essential_count,
    "baseline_precision": round(baseline_precision, 4),
    "topK_precision": round(precision_topK, 4),
    "lift_pp": round(lift_pp, 2),
    "K_at_p80": K_max,
    "coverage_at_p80": round(coverage_at_p80, 4),
    "top_15": top15,
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
tlog(f"wrote {OUT}")

# concise final summary
top5 = [{
    "lt": x["lt"],
    "gene": x["gene"],
    "product": x["consensus_product"],
    "score": x["score"],
    "truth": x["true_essential"],
} for x in top15[:5]]

summary = {
    "organism": ORG,
    "soup_size": soup_size,
    "soup_essentials": soup_essential_count,
    "baseline_precision": round(baseline_precision, 4),
    "topK_precision": round(precision_topK, 4),
    "lift_pp": round(lift_pp, 2),
    "cov_at_p80": round(coverage_at_p80, 4),
    "top_5": top5,
}
print("FINAL_SUMMARY " + json.dumps(summary))
