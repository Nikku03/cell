"""Solve the unpredictable-gene problem by identifying failure modes from
data and building a leak-clean ROUTER that abstains with a typed reason.

This is the missing modular piece of the project: a per-gene scope classifier
that, given only deployment-time features, predicts which of five
mechanistically-distinct outcomes a gene will fall into:

  MODE_EASY        : conserved + family agrees + model already handles it
  MODE_CONDITIONAL : family is informative but DISAGREES with the gene's label
                     (the conditional-residual signature)
  MODE_SPARSE      : MSA depth too low; insufficient comparative evidence
  MODE_DOMAIN      : archaea or other very-distant clade; conservation
                     transfer breaks
  MODE_NOISY       : everything looks fine but the model is confidently wrong
                     (the irreducible residual)

For each mode we report (a) what features identify it, (b) the per-mode
ground-truth AUC of the existing predictor, (c) the recommended action
(predict / abstain / route to another model).

Output: outputs/orphan/unpredict_modes.json (the typology)
        outputs/orphan/unpredict_router.json (the trained router metrics)
        outputs/orphan/per_gene_mode.npz     (per-gene mode + features)

Pure numpy; CPU only; uses what's already on disk.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

CACHE = "outputs/orphan/af_msa_cache.npz"
PREDS = "outputs/orphan/af_torch_preds_aug.npz"
LABELS_CSV = "data/drive_import/labels/labels.csv"
ORTH_CSV   = "data/drive_import/labels/orthology_features.csv"
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


# =============================================================================
# STEP 1 -- pull cache + predictions; align them by (organism, locus_tag)
# =============================================================================

print("=" * 72)
print("STEP 1 -- Loading cache and predictions; aligning by (org, locus_tag)")
print("=" * 72)
cache = np.load(CACHE, allow_pickle=True)
foc, msa, msa_mask = cache["foc"], cache["msa"], cache["msa_mask"]
y_cache = cache["y"].astype(np.int8)
clade_cache = cache["clade"]; meta_org_cache = cache["meta_org"]
lt_cache = cache["lt"]; orgs = list(cache["orgs"])
print(f"  cache: {len(y_cache):,} genes")

preds = np.load(PREDS, allow_pickle=True)
p_preds, br_preds = preds["p"], preds["brightness"]
y_preds = preds["y"].astype(np.int8); clade_preds = preds["clade"]
lt_preds = preds["lt"]; meta_org_preds = preds["meta_org"].astype(int)
# the preds file was produced from the augmented cache; we need org NAMES, not
# augmented org indices. Try to recover names via the preds file's clade+lt.
# Simpler: align by (clade, locus_tag) which is unique enough for diagnosis.
preds_key = {(str(clade_preds[i]), str(lt_preds[i])): i
             for i in range(len(p_preds))}
print(f"  preds: {len(p_preds):,} predictions (augmented run)")

# For each cache gene, look up its prediction (if any)
N = len(y_cache)
has_pred = np.zeros(N, bool)
p_aligned = np.full(N, np.nan, np.float32)
br_aligned = np.full(N, np.nan, np.float32)
for i in range(N):
    k = (str(clade_cache[i]), str(lt_cache[i]))
    j = preds_key.get(k)
    if j is not None:
        has_pred[i] = True
        p_aligned[i] = p_preds[j]
        br_aligned[i] = br_preds[j]
print(f"  aligned: {has_pred.sum():,} / {N:,} cache genes have a prediction")


# =============================================================================
# STEP 2 -- per-gene leak-clean family statistics
# =============================================================================

print("\n" + "=" * 72)
print("STEP 2 -- Leak-clean per-gene family statistics (LOO over org)")
print("=" * 72)
# msa channels (from af_build_msa.py): [label, label_known, dnds, dn, ds, same_clade]
msa_label = msa[..., 0]
msa_known = msa[..., 1]
msa_same_clade = msa[..., 5]
msa_depth = msa_mask.sum(1)

# leak-clean family stats: how does this gene's label compare to its FAMILY?
# All msa rows are FROM OTHER ORGANISMS (af_build_msa skips own org) -- so they
# are already leak-safe wrt the focal gene.
known_count = (msa_known * msa_mask).sum(1)               # # family with labels
ess_in_family = (msa_label * msa_known * msa_mask).sum(1)
family_ess_rate = np.where(known_count > 0,
                           ess_in_family / np.maximum(known_count, 1), 0.5)
family_ess_var = np.where(
    known_count > 1,
    family_ess_rate * (1 - family_ess_rate),    # bernoulli variance proxy
    0.5)
same_clade_count = (msa_same_clade * msa_mask).sum(1)
cross_clade_count = msa_depth - same_clade_count

# the gene's own label
own_ess = y_cache.astype(int)

# family-disagreement: how far is the gene's label from the family majority?
family_majority = (family_ess_rate >= 0.5).astype(int)
family_disagree = (own_ess != family_majority).astype(int)

print(f"  MSA depth: median {int(np.median(msa_depth))} / max {int(msa_depth.max())}")
print(f"  family informative (known≥3): "
      f"{int((known_count >= 3).sum()):,} ({(known_count >= 3).mean()*100:.1f}%)")
print(f"  family-disagreement rate: {family_disagree.mean()*100:.1f}%")
print(f"  archaea clade genes: {int((clade_cache == 'archaea_methanococcus').sum()):,}")


# =============================================================================
# STEP 3 -- failure-mode taxonomy from observed outcomes
# =============================================================================

print("\n" + "=" * 72)
print("STEP 3 -- Failure-mode taxonomy from observed predictions")
print("=" * 72)

# For genes with predictions, classify the actual outcome
ev = has_pred
call = (p_aligned > 0.5).astype(int)
correct = (call == own_ess) & ev
confident = (br_aligned >= 0.85) & ev
conf_wrong = confident & ~correct
conf_right = confident & correct
unconfident = ev & ~confident

# the modes (a gene gets exactly one)
mode = np.full(N, -1, np.int8)
MODE_EASY, MODE_CONDITIONAL, MODE_SPARSE, MODE_DOMAIN, MODE_NOISY = 0, 1, 2, 3, 4
MODE_NAMES = ["EASY", "CONDITIONAL", "SPARSE", "DOMAIN", "NOISY"]

# DOMAIN: archaea (or any clade where bacterial signal famously fails)
is_domain = np.isin(clade_cache, ["archaea_methanococcus"])
# SPARSE: too little family information
is_sparse = (known_count < 3) & ~is_domain
# CONDITIONAL: family is informative AND disagrees with this gene's label
is_conditional = (known_count >= 3) & (family_disagree == 1) & ~is_domain & ~is_sparse
# NOISY: the rest of failed predictions (high known_count + family agrees but model still wrong)
# EASY: the rest

# Assign modes (priority: DOMAIN > SPARSE > CONDITIONAL > observed outcome)
mode[is_domain] = MODE_DOMAIN
mode[is_sparse & (mode == -1)] = MODE_SPARSE
mode[is_conditional & (mode == -1)] = MODE_CONDITIONAL
# remaining unassigned go to EASY (if model gets them right) or NOISY (if wrong)
remaining = (mode == -1)
mode[remaining & ev & correct] = MODE_EASY
mode[remaining & ev & ~correct] = MODE_NOISY
# genes without a prediction default to EASY if they look conserved + family-agree
mode[remaining & ~ev] = MODE_EASY

# -- per-mode stats --
print(f"\n  {'mode':<14} {'n':>8} {'%total':>7} {'ess_rate':>9} {'model_AUC':>10} "
      f"{'br>=0.85_acc':>13}")
print("  " + "-" * 70)
report = {}
for mi, mn in enumerate(MODE_NAMES):
    sel = (mode == mi)
    n = int(sel.sum())
    if n == 0:
        continue
    sel_ev = sel & ev
    ess_rate = float(own_ess[sel].mean())
    sel_auc = auc(p_aligned[sel_ev], own_ess[sel_ev]) if sel_ev.sum() > 50 else float("nan")
    sel_conf = sel_ev & confident
    bright_acc = float((call[sel_conf] == own_ess[sel_conf]).mean()) if sel_conf.sum() > 20 else float("nan")
    bright_cov = float(sel_conf.sum() / max(1, sel_ev.sum()))
    print(f"  {mn:<14} {n:>8} {n/N*100:>6.1f}% {ess_rate:>9.3f} "
          f"{sel_auc:>10.3f} {bright_acc:>13.3f}")
    report[mn] = dict(
        n=n, pct=round(n / N * 100, 2), ess_rate=round(ess_rate, 3),
        model_auc=round(sel_auc, 3) if sel_auc == sel_auc else None,
        bright_acc=round(bright_acc, 3) if bright_acc == bright_acc else None,
        bright_cov=round(bright_cov, 3))


# =============================================================================
# STEP 4 -- per-mode mechanism analysis (what features identify each mode?)
# =============================================================================

print("\n" + "=" * 72)
print("STEP 4 -- Per-mode mechanism: feature means by mode")
print("=" * 72)
# build a feature row per gene (leak-clean: NO OWN LABEL)
FEAT_NAMES = ["msa_depth", "known_count", "family_ess_rate", "family_ess_var",
              "same_clade_count", "cross_clade_count",
              "foc_loglen", "foc_lead", "foc_oric", "foc_gc",
              "foc_npar", "foc_dnds", "foc_has_dnds"]
F = np.column_stack([
    msa_depth, known_count, family_ess_rate, family_ess_var,
    same_clade_count, cross_clade_count,
    foc[:, 0], foc[:, 1], foc[:, 2], foc[:, 3], foc[:, 4], foc[:, 5], foc[:, 6],
]).astype(np.float32)

print(f"\n  {'feature':<18}", *[f"{n:>10}" for n in MODE_NAMES])
print("  " + "-" * 72)
for fi, fn in enumerate(FEAT_NAMES):
    means = []
    for mi in range(5):
        sel = (mode == mi)
        means.append(float(F[sel, fi].mean()) if sel.any() else float("nan"))
    print(f"  {fn:<18}", *[f"{m:>10.3f}" for m in means])


# =============================================================================
# STEP 5 -- train the leak-clean ROUTER (multinomial logistic regression)
# =============================================================================

print("\n" + "=" * 72)
print("STEP 5 -- Train the per-gene mode classifier (leak-clean LOCO)")
print("=" * 72)
# leave-one-CLADE-out: train on N-1 clades, predict mode for the held clade
uniq_clades = sorted(set(clade_cache.tolist()))
big_clades = [c for c in uniq_clades if (clade_cache == c).sum() >= 2000]
print(f"  {len(big_clades)} clades with >= 2000 genes")

# pure-numpy multinomial logistic regression
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_multilogit(X, y, n_classes, l2=1.0, iters=300, lr=0.5):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    W = np.zeros((Xs.shape[1], n_classes))
    Y = np.eye(n_classes)[y]
    n = len(X)
    for _ in range(iters):
        P = softmax(Xs @ W)
        G = Xs.T @ (P - Y) / n
        G[1:] += l2 * W[1:] / n
        W -= lr * G
    return W, mu, sd

def predict_multilogit(model, X):
    W, mu, sd = model
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    return softmax(Xs @ W)

router_pred = np.full((N, 5), np.nan, np.float32)
mode_loco = np.full(N, -1, np.int8)
for held in big_clades:
    test_mask = (clade_cache == held)
    train_mask = ~test_mask
    M_router = fit_multilogit(F[train_mask], mode[train_mask], 5)
    probs = predict_multilogit(M_router, F[test_mask])
    router_pred[test_mask] = probs
    mode_loco[test_mask] = probs.argmax(1)

# evaluate the router
valid = mode_loco >= 0
acc = float((mode_loco[valid] == mode[valid]).mean())
print(f"\n  router LOCO accuracy: {acc:.3f}  (over {int(valid.sum()):,} genes)")
# per-mode precision/recall
print(f"  {'mode':<14} {'precision':>10} {'recall':>10} {'predicted_n':>12}")
print("  " + "-" * 50)
router_report = {"loco_accuracy": round(acc, 3)}
for mi, mn in enumerate(MODE_NAMES):
    pred_mi = (mode_loco == mi)
    true_mi = (mode == mi) & valid
    n_pred = int(pred_mi.sum())
    n_true = int(true_mi.sum())
    tp = int((pred_mi & true_mi).sum())
    prec = tp / max(1, n_pred); rec = tp / max(1, n_true)
    print(f"  {mn:<14} {prec:>10.3f} {rec:>10.3f} {n_pred:>12}")
    router_report[mn] = dict(precision=round(prec, 3), recall=round(rec, 3),
                              predicted_n=n_pred, true_n=n_true)


# =============================================================================
# STEP 6 -- the DEPLOYMENT ROUTING decision: per-mode action + coverage @P>=0.90
# =============================================================================

print("\n" + "=" * 72)
print("STEP 6 -- Routed deployment: per-mode action and combined coverage @ P>=0.90")
print("=" * 72)
# policy: if the ROUTER predicts EASY or CONDITIONAL with high probability AND
# the model brightness is high -> use the model prediction. Otherwise abstain.
# CONDITIONAL: we still predict but flag.
# SPARSE/DOMAIN/NOISY: abstain.
ACTION = {"EASY": "predict", "CONDITIONAL": "predict_with_flag",
          "SPARSE": "abstain_insufficient_data",
          "DOMAIN": "abstain_domain_barrier",
          "NOISY": "abstain_irreducible_residual"}

# combined coverage @ 0.90 precision -- only count "predict*" modes
predict_modes = {MODE_EASY, MODE_CONDITIONAL}
use = valid & np.isin(mode_loco, list(predict_modes)) & has_pred & (br_aligned >= 0.85)
n_use = int(use.sum())
n_eval = int(has_pred.sum())
if n_use > 100:
    call_use = (p_aligned[use] > 0.5).astype(int)
    correct_use = (call_use == own_ess[use])
    prec = float(correct_use.mean())
    cov = n_use / max(1, n_eval)
    print(f"\n  routed deployment (modes EASY+CONDITIONAL, brightness>=0.85):")
    print(f"    coverage : {cov:.4f}  ({n_use:,} / {n_eval:,} evaluable genes)")
    print(f"    precision: {prec:.4f}")
    deploy = dict(coverage=round(cov, 4), precision=round(prec, 4),
                  n_called=n_use, n_evaluable=n_eval,
                  policy={k: v for k, v in ACTION.items()})
else:
    deploy = dict(error="too few predictions to evaluate router-routed deployment")

# abstention typology breakdown
print("\n  abstention breakdown (router-predicted mode | router policy):")
for mi, mn in enumerate(MODE_NAMES):
    sel = (mode_loco == mi) & valid
    n = int(sel.sum())
    print(f"    {mn:<14}  {n:>8}  ({n/max(1,int(valid.sum()))*100:>5.1f}%)  -> {ACTION[mn]}")


# =============================================================================
# OUTPUTS
# =============================================================================
json.dump(dict(
    mode_definitions=dict(
        EASY="conserved + family-agreement + model correct",
        CONDITIONAL="family informative but disagrees with focal label",
        SPARSE="MSA depth < 3; insufficient comparative evidence",
        DOMAIN="archaea / domain-distant clade",
        NOISY="model confidently wrong with no other structural reason",
    ),
    per_mode_report=report,
    feature_means_by_mode={
        MODE_NAMES[mi]: {FEAT_NAMES[fi]: round(float(F[mode == mi, fi].mean()), 3)
                         for fi in range(len(FEAT_NAMES))}
        for mi in range(5) if (mode == mi).any()
    },
), open(OUT / "unpredict_modes.json", "w"), indent=2)

json.dump(dict(router=router_report, deployment=deploy, actions=ACTION),
          open(OUT / "unpredict_router.json", "w"), indent=2)

np.savez_compressed(OUT / "per_gene_mode.npz",
                    mode_true=mode, mode_router=mode_loco,
                    router_probs=router_pred,
                    msa_depth=msa_depth.astype(np.float32),
                    known_count=known_count.astype(np.float32),
                    family_ess_rate=family_ess_rate.astype(np.float32),
                    family_disagree=family_disagree.astype(np.int8),
                    clade=clade_cache)

print("\nwrote:")
print("  outputs/orphan/unpredict_modes.json    (failure-mode typology)")
print("  outputs/orphan/unpredict_router.json   (router metrics + deployment policy)")
print("  outputs/orphan/per_gene_mode.npz       (per-gene mode + probs)")
