"""SOUP rescue test for beril_Putida via orthogroup family essentiality.

Vectorized: build all dicts once, then numpy-index. No per-row loops.
"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

ORG = "beril_Putida"
CLADE = "pseudomonas"
P_LO, P_HI = 0.30, 0.70
BRIGHTNESS_MAX = 0.85
MIN_OG_MEMBERS = 5
W_UNIV = 0.7
W_BREADTH = 0.3
BREADTH_NORM = 30.0
P_FLOOR = 0.80

OUT_JSON = f"/home/user/cell/outputs/orphan/soup_rescue_{ORG}.json"

t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ---------- load ----------
log("Loading clade_splits...")
clade_df = pd.read_csv("/home/user/cell/data/drive_import/labels/clade_splits.csv")
org_to_clade = dict(zip(clade_df.organism.values, clade_df.clade.values))
pseudomonas_orgs = set(clade_df.loc[clade_df.clade == CLADE, "organism"].values)

log("Loading labels...")
labels = pd.read_csv(
    "/home/user/cell/data/drive_import/labels/labels.csv",
    usecols=["organism", "locus_tag", "gene_name", "essential"],
)
# essentiality as int
labels["essential"] = pd.to_numeric(labels["essential"], errors="coerce").fillna(0).astype(np.int8)

log("Loading orthology features...")
ortho = pd.read_csv(
    "/home/user/cell/data/drive_import/labels/orthology_features.csv",
    usecols=["organism", "locus_tag", "og_id", "family_n_organisms", "is_orphan"],
)

log("Loading gene annotations...")
ann = pd.read_csv(
    "/home/user/cell/data/drive_import/labels/gene_annotations.csv",
    usecols=["organism", "locus_tag", "product"],
)

log("Loading predictions npz...")
pred = np.load("/home/user/cell/outputs/orphan/af_torch_preds_aug.npz", allow_pickle=True)
p_arr = pred["p"].astype(np.float32)
b_arr = pred["brightness"].astype(np.float32)
y_arr = pred["y"].astype(np.float32)
clade_arr = pred["clade"].astype(str)
lt_arr = pred["lt"].astype(str)

# ---------- SOUP definition ----------
# Predictions don't carry organism directly; we have clade and locus_tag.
# Strategy: filter predictions to clade==pseudomonas first, then map (lt -> organism) via labels.csv
# restricted to organism==beril_Putida. We trust locus_tag is org-unique for beril_Putida.

log("Building beril_Putida label map...")
bp_labels = labels[labels.organism == ORG].copy()
bp_lt_to_ess = dict(zip(bp_labels.locus_tag.values, bp_labels.essential.values.astype(np.int8)))
bp_lt_to_gene = dict(zip(bp_labels.locus_tag.values, bp_labels.gene_name.fillna("").values))
log(f"beril_Putida labels: {len(bp_lt_to_ess)} genes")

# Mask predictions: clade pseudomonas AND lt in beril_Putida labels
pseudo_mask = (clade_arr == CLADE)
log(f"pseudomonas pred rows: {pseudo_mask.sum()}")

bp_lt_set = set(bp_lt_to_ess.keys())
# Vectorized membership via pandas
pred_df = pd.DataFrame({
    "locus_tag": lt_arr[pseudo_mask],
    "p": p_arr[pseudo_mask],
    "b": b_arr[pseudo_mask],
})
pred_df = pred_df[pred_df["locus_tag"].isin(bp_lt_set)].copy()
log(f"pred rows matching beril_Putida locus tags: {len(pred_df)}")

# Many beril_Putida locus tags can appear multiple times if predictions are stored per-org? Drop dup keeping first
pred_df = pred_df.drop_duplicates(subset=["locus_tag"], keep="first")
log(f"after dedup: {len(pred_df)} unique locus tags with predictions")

# SOUP filter
soup_mask = (pred_df.p > P_LO) & (pred_df.p < P_HI) & (pred_df.b < BRIGHTNESS_MAX)
soup = pred_df[soup_mask].copy()
soup["truth"] = soup["locus_tag"].map(bp_lt_to_ess).fillna(0).astype(np.int8)
soup_size = int(len(soup))
soup_essential_count = int(soup.truth.sum())
log(f"soup_size={soup_size}  soup_essential_count={soup_essential_count}")
print(f"soup_size: {soup_size}")
print(f"soup_essential_count: {soup_essential_count}")

if soup_size == 0:
    raise SystemExit("No soup genes; aborting.")

# ---------- leak-clean OG essential rate ----------
log("Building leak-clean OG stats...")
# Mark each ortho row as 'in pseudomonas clade' or not
ortho["is_pseudo"] = ortho.organism.isin(pseudomonas_orgs)

# Merge ortho with labels on (organism, locus_tag) to get essentiality per gene-in-OG
log("Merging ortho with labels...")
labeled_ortho = ortho.merge(
    labels[["organism", "locus_tag", "essential"]],
    on=["organism", "locus_tag"],
    how="left",
)
labeled_ortho["has_label"] = labeled_ortho["essential"].notna()
labeled_ortho["essential"] = labeled_ortho["essential"].fillna(0).astype(np.int8)
log(f"labeled_ortho rows: {len(labeled_ortho)}")

# leak-clean: exclude pseudomonas-clade organisms AND require label present
clean = labeled_ortho[(~labeled_ortho.is_pseudo) & (labeled_ortho.has_label)].copy()
log(f"leak-clean labeled rows: {len(clean)}")

og_stats = clean.groupby("og_id").agg(
    n_clean=("essential", "size"),
    e_clean=("essential", "sum"),
).reset_index()
og_stats["og_rate"] = og_stats.e_clean / og_stats.n_clean
og_stats = og_stats[og_stats.n_clean >= MIN_OG_MEMBERS].copy()
log(f"OGs passing MIN_OG_MEMBERS={MIN_OG_MEMBERS}: {len(og_stats)}")
og_to_rate = dict(zip(og_stats.og_id.values, og_stats.og_rate.values.astype(np.float32)))

# ---------- family breadth (across all clades, including pseudomonas) ----------
log("Computing family breadth (#distinct organisms per OG)...")
og_breadth = ortho.groupby("og_id")["organism"].nunique().to_dict()

# ---------- consensus product per OG ----------
log("Computing consensus product per OG...")
# Merge ann with ortho on (organism, locus_tag)
ann["product"] = ann["product"].fillna("").astype(str)
ortho_ann = ortho[["organism", "locus_tag", "og_id"]].merge(
    ann[["organism", "locus_tag", "product"]],
    on=["organism", "locus_tag"],
    how="left",
)
ortho_ann["product"] = ortho_ann["product"].fillna("").astype(str)

def is_real_product(s: str) -> bool:
    if not s:
        return False
    sl = s.lower()
    if "hypothetical" in sl or "uncharacter" in sl or "unknown" in sl or sl == "putative protein":
        return False
    return True

# Vectorized filter then group
ortho_ann["good"] = ortho_ann["product"].map(is_real_product)
good_rows = ortho_ann[ortho_ann.good].copy()
log(f"good product rows: {len(good_rows)}")

og_consensus_product = {}
# groupby + value_counts idxmax
if len(good_rows):
    grp = good_rows.groupby("og_id")["product"]
    # Use mode-like fastest: value_counts per group is slow; use Counter via iter
    # But this is over ~150k rows max — fine.
    og_consensus_product = grp.agg(lambda s: s.value_counts().index[0]).to_dict()
log(f"OGs with consensus product: {len(og_consensus_product)}")

# ---------- map soup genes -> OG ----------
log("Mapping soup genes -> OG...")
bp_ortho = ortho[ortho.organism == ORG][["locus_tag", "og_id"]].copy()
bp_lt_to_og = dict(zip(bp_ortho.locus_tag.values, bp_ortho.og_id.values))

soup["og"] = soup["locus_tag"].map(bp_lt_to_og).fillna("")
soup["og_rate"] = soup.og.map(og_to_rate).astype(float)  # NaN if missing
soup["breadth"] = soup.og.map(og_breadth).fillna(0).astype(float)
soup["product"] = soup.og.map(og_consensus_product).fillna("")
soup["gene"] = soup["locus_tag"].map(bp_lt_to_gene).fillna("")

n_with_og = int(soup.og.ne("").sum())
n_with_clean_rate = int(soup.og_rate.notna().sum())
log(f"soup w/ OG: {n_with_og}  w/ leak-clean rate: {n_with_clean_rate}")

# rescue score: for genes with no OG/rate, score is 0 (cannot be rescued)
og_rate_filled = soup.og_rate.fillna(0).values.astype(np.float32)
breadth_norm = np.minimum(soup.breadth.values / BREADTH_NORM, 1.0).astype(np.float32)
soup["rescue_score"] = W_UNIV * og_rate_filled + W_BREADTH * breadth_norm
# Mark unscoreable as -inf so they sink in ranking
no_signal = (soup.og_rate.isna()) & (soup.breadth == 0)
soup.loc[no_signal, "rescue_score"] = -1.0

# ---------- validate ----------
K = soup_essential_count
soup_sorted = soup.sort_values("rescue_score", ascending=False).reset_index(drop=True)

baseline_precision = soup_essential_count / soup_size if soup_size else 0.0

if K > 0 and K <= soup_size:
    topK = soup_sorted.iloc[:K]
    precision_topK = float(topK.truth.sum()) / K
else:
    precision_topK = float("nan")
lift_pp = (precision_topK - baseline_precision) * 100.0 if not np.isnan(precision_topK) else float("nan")

# precision-at-fixed-precision-floor
truth_sorted = soup_sorted.truth.values.astype(np.int32)
cum_tp = np.cumsum(truth_sorted)
k_arr = np.arange(1, len(truth_sorted) + 1)
prec_at_k = cum_tp / k_arr
# largest K with prec_at_k >= P_FLOOR
mask_above = prec_at_k >= P_FLOOR
if mask_above.any():
    K_max = int(np.max(np.where(mask_above)[0])) + 1
else:
    K_max = 0
coverage_at_p80 = K_max / soup_essential_count if soup_essential_count else 0.0

log(f"baseline={baseline_precision:.4f}  topK={precision_topK:.4f}  lift_pp={lift_pp:.2f}  cov@p80={coverage_at_p80:.3f}")

# ---------- top candidates ----------
top15 = soup_sorted.head(15)
top15_rec = [
    {
        "lt": str(r["locus_tag"]),
        "gene": str(r["gene"]),
        "og": str(r["og"]),
        "score": float(r["rescue_score"]),
        "consensus_product": str(r["product"]),
        "true_essential": int(r["truth"]),
        "p": float(r["p"]),
        "brightness": float(r["b"]),
        "og_rate_loco": float(r["og_rate"]) if not np.isnan(r["og_rate"]) else None,
        "family_breadth": int(r["breadth"]),
    }
    for _, r in top15.iterrows()
]

top5 = [
    {
        "lt": rec["lt"],
        "gene": rec["gene"],
        "product": rec["consensus_product"],
        "score": rec["score"],
        "truth": rec["true_essential"],
    }
    for rec in top15_rec[:5]
]

result = {
    "organism": ORG,
    "clade": CLADE,
    "soup_size": soup_size,
    "soup_essentials": soup_essential_count,
    "baseline_precision": float(baseline_precision),
    "topK_precision": float(precision_topK) if not np.isnan(precision_topK) else None,
    "lift_pp": float(lift_pp) if not np.isnan(lift_pp) else None,
    "K_max_p80": K_max,
    "cov_at_p80": float(coverage_at_p80),
    "soup_with_og": n_with_og,
    "soup_with_leak_clean_rate": n_with_clean_rate,
    "min_og_members": MIN_OG_MEMBERS,
    "weights": {"universality": W_UNIV, "breadth": W_BREADTH, "breadth_norm": BREADTH_NORM},
    "top_15": top15_rec,
    "top_5": top5,
}

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
log(f"wrote {OUT_JSON}")

summary = {
    "organism": ORG,
    "soup_size": soup_size,
    "soup_essentials": soup_essential_count,
    "baseline_precision": round(float(baseline_precision), 4),
    "topK_precision": round(float(precision_topK), 4) if not np.isnan(precision_topK) else None,
    "lift_pp": round(float(lift_pp), 2) if not np.isnan(lift_pp) else None,
    "cov_at_p80": round(float(coverage_at_p80), 4),
    "top_5": top5,
}
print(json.dumps(summary, indent=2))
log("DONE")
