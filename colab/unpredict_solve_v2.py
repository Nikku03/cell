"""Unpredictability solver v2 -- richer per-ortholog and product features.

The v1 router achieved 0.96 recall on EASY but only 0.13 on CONDITIONAL and
0.01 on NOISY because it used family SUMMARIES (mean ess rate, mean variance)
that destroy the within-family signal. This version uses:

  1. PER-ORTHOLOG evolutionary features (variance/spread of dN/dS across the
     MSA, same-clade-vs-cross-clade essentiality differential, # clades the
     family spans -- distinguishes evolutionarily-anomalous genes).
  2. GENE PRODUCT / FUNCTIONAL CLASS from regulator_features.csv
     (is_regulator, is_transporter, is_signaling).
  3. CODON-USAGE features from codon_features.csv (CAI, GC3 -- proxies for
     expression level / horizontal-transfer origin).
  4. ORTHOLOGY CONTEXT from orthology_features.csv (family_n_organisms,
     n_paralogs_in_genome).
  5. CLASS-BALANCED training so the router stops collapsing to the majority.

Comparison to v1 reported per-mode precision/recall on the same modes; the
question is whether functional + per-ortholog signal lets us identify
conditional/noisy genes from leak-clean features.
"""
import csv, json
from pathlib import Path
import numpy as np

CACHE = "outputs/orphan/af_msa_cache.npz"
PREDS = "outputs/orphan/af_torch_preds_aug.npz"
REG_CSV = "data/drive_import/labels/regulator_features.csv"
CODON_CSV = "data/drive_import/labels/codon_features.csv"
ORTH_CSV = "data/drive_import/labels/orthology_features.csv"
OUT = Path("outputs/orphan")


# =============================================================================
# Load cache + align predictions (same as v1)
# =============================================================================
print("Loading cache + predictions ...")
cache = np.load(CACHE, allow_pickle=True)
foc, msa, msa_mask = cache["foc"], cache["msa"], cache["msa_mask"]
y_cache = cache["y"].astype(np.int8)
clade_cache = cache["clade"]; meta_org_cache = cache["meta_org"]
lt_cache = cache["lt"]; orgs = list(cache["orgs"])
N = len(y_cache)

preds = np.load(PREDS, allow_pickle=True)
p_preds, br_preds = preds["p"], preds["brightness"]
clade_preds = preds["clade"]; lt_preds = preds["lt"]
preds_key = {(str(clade_preds[i]), str(lt_preds[i])): i for i in range(len(p_preds))}

p_aligned = np.full(N, np.nan, np.float32)
br_aligned = np.full(N, np.nan, np.float32)
has_pred = np.zeros(N, bool)
for i in range(N):
    j = preds_key.get((str(clade_cache[i]), str(lt_cache[i])))
    if j is not None:
        has_pred[i] = True
        p_aligned[i] = p_preds[j]; br_aligned[i] = br_preds[j]
print(f"  cache={N:,}  aligned predictions={int(has_pred.sum()):,}")


# =============================================================================
# RICHER feature extraction
# =============================================================================
print("\nBuilding richer features (per-ortholog + functional + codon) ...")

# MSA channels: [label, label_known, dnds, dn, ds, same_clade]
msa_label = msa[..., 0]; msa_known = msa[..., 1]
msa_dnds  = msa[..., 2]; msa_same_clade = msa[..., 5]
msa_depth = msa_mask.sum(1).astype(np.float32)

# --- ortholog-level evolutionary signal ---
# valid mask: real ortholog row that has labels OR dN/dS info
known = msa_known * msa_mask                                 # [N,K]
known_count = known.sum(1)

# dN/dS spread across orthologs (capture evolutionary heterogeneity)
# we compute these only over valid rows
with np.errstate(invalid="ignore", divide="ignore"):
    dnds_sum = (msa_dnds * msa_mask).sum(1)
    dnds_n = msa_mask.sum(1)
    dnds_mean = np.where(dnds_n > 0, dnds_sum / np.maximum(dnds_n, 1), 0)
    # variance via E[X^2] - E[X]^2
    dnds_sq = ((msa_dnds ** 2) * msa_mask).sum(1) / np.maximum(dnds_n, 1)
    dnds_var = np.clip(dnds_sq - dnds_mean ** 2, 0, None)
    dnds_max = np.where(msa_mask > 0, msa_dnds, -np.inf).max(1)
    dnds_min = np.where(msa_mask > 0, msa_dnds,  np.inf).min(1)
    dnds_max = np.where(np.isfinite(dnds_max), dnds_max, 0)
    dnds_min = np.where(np.isfinite(dnds_min), dnds_min, 0)
    dnds_spread = dnds_max - dnds_min

# --- same-clade vs cross-clade essentiality differential ---
sc_known = known * msa_same_clade
cc_known = known * (1 - msa_same_clade)
sc_ess = (msa_label * sc_known).sum(1)
cc_ess = (msa_label * cc_known).sum(1)
sc_n = sc_known.sum(1); cc_n = cc_known.sum(1)
sc_ess_rate = np.where(sc_n > 0, sc_ess / np.maximum(sc_n, 1), 0.5)
cc_ess_rate = np.where(cc_n > 0, cc_ess / np.maximum(cc_n, 1), 0.5)
# the differential is the key new feature: if a gene is essential in same-clade
# orthologs but not cross-clade (or vice versa), it suggests clade-specific
# (i.e., conditional) essentiality
same_minus_cross = sc_ess_rate - cc_ess_rate

# how many DIFFERENT clades the family spans
# (proxy via the cache's clade list, indexed via msa_org)
msa_org = cache["msa_org"]
# map cache org index -> clade
o2c = {}
for oi, c in zip(meta_org_cache, clade_cache):
    o2c[int(oi)] = str(c)
n_clades_in_family = np.zeros(N, np.float32)
for i in range(N):
    cs = set()
    for k in range(msa_org.shape[1]):
        if msa_mask[i, k] > 0:
            oi = int(msa_org[i, k])
            if oi in o2c:
                cs.add(o2c[oi])
    n_clades_in_family[i] = len(cs)

# family essentiality (overall, leak-clean since MSA excludes own org)
family_ess_rate = np.where(known_count > 0,
                           (msa_label * known).sum(1) / np.maximum(known_count, 1),
                           0.5)
family_ess_var = family_ess_rate * (1 - family_ess_rate)

# --- functional class (gene product) features ---
print("  loading regulator_features.csv ...")
funct = {}                                       # (org, lt) -> dict
for r in csv.DictReader(open(REG_CSV)):
    funct[(r["organism"], r["locus_tag"])] = dict(
        is_regulator=float(r.get("is_regulator", 0) or 0),
        is_transporter=float(r.get("is_transporter", 0) or 0),
        is_signaling=float(r.get("is_signaling", 0) or 0))
is_regulator = np.zeros(N, np.float32)
is_transporter = np.zeros(N, np.float32)
is_signaling = np.zeros(N, np.float32)
for i in range(N):
    nm = str(orgs[int(meta_org_cache[i])]); lt = str(lt_cache[i])
    f = funct.get((nm, lt))
    if f:
        is_regulator[i] = f["is_regulator"]
        is_transporter[i] = f["is_transporter"]
        is_signaling[i] = f["is_signaling"]
print(f"    is_regulator   set on {int(is_regulator.sum()):,} genes "
      f"({is_regulator.mean()*100:.1f}%)")
print(f"    is_transporter set on {int(is_transporter.sum()):,} genes "
      f"({is_transporter.mean()*100:.1f}%)")
print(f"    is_signaling   set on {int(is_signaling.sum()):,} genes "
      f"({is_signaling.mean()*100:.1f}%)")

# --- codon usage (sparse but informative) ---
print("  loading codon_features.csv ...")
codon = {}
for r in csv.DictReader(open(CODON_CSV)):
    try:
        codon[(r["organism"], r["locus_tag"])] = dict(
            cai=float(r.get("cai", 0) or 0),
            gc3=float(r.get("gc3", 0) or 0))
    except Exception: pass
cai = np.zeros(N, np.float32); has_cai = np.zeros(N, np.float32)
gc3 = np.zeros(N, np.float32)
for i in range(N):
    nm = str(orgs[int(meta_org_cache[i])]); lt = str(lt_cache[i])
    c = codon.get((nm, lt))
    if c:
        cai[i] = c["cai"]; gc3[i] = c["gc3"]; has_cai[i] = 1.0
print(f"    codon features set on {int(has_cai.sum()):,} genes "
      f"({has_cai.mean()*100:.1f}%)")

# --- orthology context ---
print("  loading orthology_features.csv ...")
fam = {}
for r in csv.DictReader(open(ORTH_CSV)):
    try:
        fam[(r["organism"], r["locus_tag"])] = dict(
            family_n_organisms=float(r.get("family_n_organisms", 1) or 1),
            n_paralogs=float(r.get("n_paralogs_in_genome", 0) or 0),
            family_size=float(r.get("family_size_total", 1) or 1))
    except Exception: pass
family_n_orgs = np.zeros(N, np.float32)
n_paralogs = np.zeros(N, np.float32)
family_size = np.zeros(N, np.float32)
for i in range(N):
    nm = str(orgs[int(meta_org_cache[i])]); lt = str(lt_cache[i])
    f = fam.get((nm, lt))
    if f:
        family_n_orgs[i] = f["family_n_organisms"]
        n_paralogs[i] = f["n_paralogs"]
        family_size[i] = f["family_size"]


# =============================================================================
# Build feature matrix (NO own-label leak)
# =============================================================================
FEAT_NAMES = [
    # focal (intrinsic)
    "foc_loglen", "foc_lead", "foc_oric", "foc_gc", "foc_npar",
    "foc_dnds", "foc_has_dnds",
    # family summary (v1 had these)
    "msa_depth", "known_count", "family_ess_rate", "family_ess_var",
    # NEW: per-ortholog evolutionary spread
    "dnds_mean", "dnds_var", "dnds_max", "dnds_spread",
    # NEW: same-vs-cross clade essentiality differential
    "sc_ess_rate", "cc_ess_rate", "same_minus_cross",
    "n_clades_in_family",
    # NEW: gene product / functional class
    "is_regulator", "is_transporter", "is_signaling",
    # NEW: codon usage
    "cai", "gc3", "has_cai",
    # NEW: orthology context
    "family_n_orgs", "n_paralogs", "family_size",
]
F = np.column_stack([
    foc[:, 0], foc[:, 1], foc[:, 2], foc[:, 3], foc[:, 4], foc[:, 5], foc[:, 6],
    msa_depth, known_count, family_ess_rate, family_ess_var,
    dnds_mean, dnds_var, dnds_max, dnds_spread,
    sc_ess_rate, cc_ess_rate, same_minus_cross, n_clades_in_family,
    is_regulator, is_transporter, is_signaling,
    cai, gc3, has_cai,
    family_n_orgs, n_paralogs, family_size,
]).astype(np.float32)
print(f"\n  feature matrix: {F.shape}  ({len(FEAT_NAMES)} features)")


# =============================================================================
# Define modes (same as v1, but using own_ess from cache)
# =============================================================================
own_ess = y_cache.astype(int)
call = (p_aligned > 0.5).astype(int)
correct = (call == own_ess) & has_pred
confident = (br_aligned >= 0.85) & has_pred

family_majority = (family_ess_rate >= 0.5).astype(int)
family_disagree = (own_ess != family_majority).astype(int)

mode = np.full(N, -1, np.int8)
MODE_EASY, MODE_CONDITIONAL, MODE_SPARSE, MODE_DOMAIN, MODE_NOISY = 0, 1, 2, 3, 4
MODE_NAMES = ["EASY", "CONDITIONAL", "SPARSE", "DOMAIN", "NOISY"]
is_domain = np.isin(clade_cache, ["archaea_methanococcus"])
is_sparse = (known_count < 3) & ~is_domain
is_conditional = (known_count >= 3) & (family_disagree == 1) & ~is_domain & ~is_sparse
mode[is_domain] = MODE_DOMAIN
mode[is_sparse & (mode == -1)] = MODE_SPARSE
mode[is_conditional & (mode == -1)] = MODE_CONDITIONAL
remaining = (mode == -1)
mode[remaining & has_pred & correct] = MODE_EASY
mode[remaining & has_pred & ~correct] = MODE_NOISY
mode[remaining & ~has_pred] = MODE_EASY

print(f"\n  mode populations:")
for mi, mn in enumerate(MODE_NAMES):
    n = int((mode == mi).sum())
    print(f"    {mn:<14} {n:>8} ({n/N*100:.1f}%)")


# =============================================================================
# CLASS-BALANCED multinomial logistic regression (LOCO)
# =============================================================================
print("\nTraining class-balanced router (LOCO over clades >= 2000 genes) ...")

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_balanced_multilogit(X, y, n_classes, l2=0.1, iters=600, lr=0.5):
    """Class-balanced multinomial logistic regression."""
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    W = np.zeros((Xs.shape[1], n_classes))
    Y = np.eye(n_classes)[y]
    # class weights: inverse class frequency
    counts = np.bincount(y, minlength=n_classes).astype(float)
    cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    sw = cw[y]; sw = sw / sw.mean()
    n = len(X)
    for _ in range(iters):
        P = softmax(Xs @ W)
        G = Xs.T @ ((P - Y) * sw[:, None]) / n
        G[1:] += l2 * W[1:] / n
        W -= lr * G
    return W, mu, sd

def predict_multilogit(model, X):
    W, mu, sd = model
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    return softmax(Xs @ W)

uniq_clades = sorted(set(clade_cache.tolist()))
big_clades = [c for c in uniq_clades if (clade_cache == c).sum() >= 2000]
mode_loco = np.full(N, -1, np.int8)
for held in big_clades:
    test_mask = (clade_cache == held); train_mask = ~test_mask
    M = fit_balanced_multilogit(F[train_mask], mode[train_mask], 5)
    probs = predict_multilogit(M, F[test_mask])
    mode_loco[test_mask] = probs.argmax(1)

valid = mode_loco >= 0
acc = float((mode_loco[valid] == mode[valid]).mean())
print(f"\n  router LOCO accuracy: {acc:.3f}  ({int(valid.sum()):,} genes)")

# Per-class metrics + comparison vs v1
v1 = json.load(open(OUT / "unpredict_router.json"))["router"]
print(f"\n  per-mode precision/recall (v1 vs v2):")
print(f"  {'mode':<14} {'v1_prec':>9} {'v1_rec':>8} {'v2_prec':>9} {'v2_rec':>8} {'Δrec':>7}")
print("  " + "-" * 60)
router_v2 = {"loco_accuracy": round(acc, 3)}
for mi, mn in enumerate(MODE_NAMES):
    pred_mi = (mode_loco == mi); true_mi = (mode == mi) & valid
    tp = int((pred_mi & true_mi).sum())
    prec = tp / max(1, int(pred_mi.sum()))
    rec = tp / max(1, int(true_mi.sum()))
    v1m = v1.get(mn, {})
    v1_prec = v1m.get("precision", 0); v1_rec = v1m.get("recall", 0)
    drec = (rec - v1_rec)
    print(f"  {mn:<14} {v1_prec:>9.3f} {v1_rec:>8.3f} {prec:>9.3f} {rec:>8.3f} {drec:>+7.3f}")
    router_v2[mn] = dict(precision=round(prec, 3), recall=round(rec, 3),
                          predicted_n=int(pred_mi.sum()),
                          true_n=int(true_mi.sum()),
                          delta_recall_vs_v1=round(drec, 3))


# =============================================================================
# Feature importance (mean absolute coefficient per feature per mode)
# =============================================================================
print("\nFeature importance (|coef| averaged across LOCO folds):")
# pool importance from a model trained on full data (informative ranking)
M_full = fit_balanced_multilogit(F, mode, 5)
W_full = M_full[0][1:]   # drop bias row
# per-feature L2 norm across the 5 mode columns
imp = np.linalg.norm(W_full, axis=1)
ranked = sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])
print(f"  {'rank':>4} {'feature':<22} {'importance':>10}")
for r, (fn, w) in enumerate(ranked[:15], 1):
    print(f"  {r:>4} {fn:<22} {w:>10.3f}")


# =============================================================================
# Re-evaluate deployment with v2 router
# =============================================================================
predict_modes = {MODE_EASY, MODE_CONDITIONAL}
use = valid & np.isin(mode_loco, list(predict_modes)) & has_pred & (br_aligned >= 0.85)
n_use = int(use.sum()); n_eval = int(has_pred.sum())
if n_use > 100:
    call_use = (p_aligned[use] > 0.5).astype(int)
    prec = float((call_use == own_ess[use]).mean())
    cov = n_use / max(1, n_eval)
    print(f"\n  v2 routed deployment:")
    print(f"    coverage : {cov:.4f}  ({n_use:,} / {n_eval:,})")
    print(f"    precision: {prec:.4f}")

json.dump(dict(
    router=router_v2,
    feature_importance={fn: round(float(w), 3) for fn, w in ranked},
    n_features=len(FEAT_NAMES),
    deployment=dict(coverage=round(cov, 4) if n_use else None,
                     precision=round(prec, 4) if n_use else None,
                     n_called=n_use, n_evaluable=n_eval),
), open(OUT / "unpredict_router_v2.json", "w"), indent=2)
print(f"\nwrote outputs/orphan/unpredict_router_v2.json")
