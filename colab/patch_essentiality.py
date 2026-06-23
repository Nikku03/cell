"""Patch the systems/network view into the essentiality model -- mutual benefit.

The per-gene model emits p (essential prob) + brightness (confidence). It is
already solved on the confident bucket (brightness>=0.85: 89% non-essential).
All the difficulty is the ABSTENTION bucket (brightness<0.85: ~81k genes, 44%
essential).

What the systems view adds that the per-gene model can't see:
  - og_universality  : leave-one-CLADE-out essential-rate of the gene's
                       orthogroup (the bimodal signal -- universal core vs
                       dispensable shell vs conditional middle). LEAK-CLEAN.
  - paralog_buffer   : # paralogs in genome (duplication softens the AND gate;
                       73% of paralog groups are fully dispensable). Structural.
  - family_breadth   : # organisms the OG spans (conservation). Structural.
  - is_orphan        : no orthogroup at all.

Mutual benefit:
  1 network -> model : stack these priors onto (p, brightness), retrain a
                       leak-clean cross-clade logistic on the abstention bucket,
                       measure coverage@precision lift.
  2 model -> network : the same p fills network nodes for organisms with NO
                       essentiality labels (reported as deployable coverage).
  3 neither -> hunt  : genes still uncallable at P>=0.90 after the patch are
                       categorized by WHY (conditional / paralog-buffered /
                       orphan-sparse / noisy) -> an experimental target list.

Output: outputs/orphan/patch_report.json
        outputs/orphan/patch_essentiality.png
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")

# ---------- model predictions ----------
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str)
lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))
print(f"genes={len(p)} valid={valid.sum()} ess_rate={y[valid].mean():.3f}")

# ---------- org->clade map (preds meta_org index space != cache; join by lt) ----
C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
cache_orgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {}
for i in range(len(cm)):
    org2clade[cache_orgs[int(cm[i])]] = cc[i]

# ---------- structural + OG features (keyed by organism, locus_tag) ----------
feat = {}; orgs_by_lt = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    feat[k] = (r.get("og_id") or "", float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_n_organisms") or 0), float(r.get("is_orphan") or 0))
    orgs_by_lt[r["locus_tag"]].append(r["organism"])

# resolve each prediction's organism by matching its clade among lt candidates
og = np.empty(len(p), object); paralog = np.zeros(len(p))
breadth = np.zeros(len(p)); is_orph = np.zeros(len(p))
resolved = 0
for i in range(len(p)):
    cands = orgs_by_lt.get(lt[i], [])
    chosen = None
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    if len(match) == 1:
        chosen = match[0]
    elif len(cands) == 1:
        chosen = cands[0]
    if chosen is not None:
        o_, np_, br_, or_ = feat[(chosen, lt[i])]
        og[i] = o_; paralog[i] = np_; breadth[i] = br_; is_orph[i] = or_; resolved += 1
    else:
        og[i] = ""
print(f"resolved org+features for {resolved}/{len(p)} predictions")

# ---------- leave-one-CLADE-out OG essential rate (leak-clean) ----------
# global per-og sum/count over valid-label genes, and per-(og,clade) sum/count;
# loco rate = (global - own_clade) / (global_n - own_clade_n)
g_sum = defaultdict(float); g_n = defaultdict(float)
c_sum = defaultdict(float); c_n = defaultdict(float)
for i in np.where(valid)[0]:
    o = og[i]
    if not o:
        continue
    g_sum[o] += y[i]; g_n[o] += 1
    c_sum[(o, clade[i])] += y[i]; c_n[(o, clade[i])] += 1
og_univ = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    o = og[i]
    if not o:
        continue
    n = g_n[o] - c_n[(o, clade[i])]
    if n > 0:
        og_univ[i] = (g_sum[o] - c_sum[(o, clade[i])]) / n
has_univ = ~np.isnan(og_univ)
print(f"genes with leak-clean OG-universality: {has_univ.sum()}")

# ---------- coverage@precision metric ----------
def cov_at_prec(score, ytrue, target=0.90):
    """max fraction of POSITIVES recalled while precision>=target (sweep thr)."""
    o = np.argsort(-score)
    ys = ytrue[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    npos = ytrue.sum()
    ok = prec >= target
    if not ok.any():
        return 0.0
    return float(tp[ok].max() / max(npos, 1))

def both_cov(score, ytrue, target=0.90):
    ess = cov_at_prec(score, ytrue, target)            # call essential
    ne = cov_at_prec(-score, 1 - ytrue, target)        # call non-essential
    npos = ytrue.sum(); nneg = (1 - ytrue).sum(); n = npos + nneg
    combined = (ess * npos + ne * nneg) / max(n, 1)
    return dict(essCov=round(ess, 3), neCov=round(ne, 3),
                combined=round(combined, 3))

# ---------- leak-clean cross-clade stacker on the ABSTENTION bucket ----------
def fit_logit(X, yy, l2=1.0, iters=500, lr=0.4):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** 0.5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(X); g[1:] += l2 * w[1:] / len(X); w -= lr * g
    return w, mu, sd
def pred_logit(m, X):
    w, mu, sd = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))

# work only where we have everything to compare fairly
base_mask = valid & has_univ
ab = base_mask & (br < 0.85)          # abstention bucket
print(f"abstention bucket (with features): {ab.sum()}  ess={y[ab].mean():.3f}")

# feature matrix for the stacker
def feats(mask):
    return np.column_stack([
        p[mask], br[mask], og_univ[mask],
        np.log1p(paralog[mask]), np.log1p(breadth[mask]), is_orph[mask],
    ])

clades = sorted(set(clade[ab]))
patched = np.full(len(p), np.nan)
for H in clades:
    tr = ab & (clade != H); te = ab & (clade == H)
    if tr.sum() < 200 or te.sum() < 20 or y[tr].sum() < 20:
        continue
    m = fit_logit(feats(tr), y[tr])
    patched[te] = pred_logit(m, feats(te))
done = ab & ~np.isnan(patched)
print(f"patched predictions: {done.sum()}")

# ---------- evaluate: model-alone vs patched, on the abstention bucket ----------
yb = y[done]
base = both_cov(p[done], yb)
new = both_cov(patched[done], yb)
# also the prior-alone (systems only, no model) for reference
prior = both_cov(og_univ[done], yb)
print("\n=== ABSTENTION bucket coverage@P90 ===")
print(f"  model alone     : {base}")
print(f"  systems prior   : {prior}")
print(f"  PATCHED (stack) : {new}")

# whole-genome deployable coverage (confident bucket auto-called + patched bucket)
conf = base_mask & (br >= 0.85)
# confident: model p is already calibrated; measure its coverage too
whole_score = np.where(br >= 0.85, p, patched)
whole_mask = base_mask & ~np.isnan(whole_score)
whole_base = both_cov(p[whole_mask], y[whole_mask])
whole_new = both_cov(whole_score[whole_mask], y[whole_mask])
print("\n=== WHOLE genome (confident + uncertain) coverage@P90 ===")
print(f"  model alone     : {whole_base}")
print(f"  PATCHED         : {whole_new}")

# ---------- categorize the residual hunt-list ----------
# genes still uncallable at P90: prediction in the ambiguous mid-band on BOTH
# directions (neither confidently essential nor non-essential after patch).
sc = patched[done]
ambiguous = (sc > 0.30) & (sc < 0.70)
hunt = np.where(done)[0][ambiguous]
def category(i):
    if is_orph[i] or not og[i]:
        return "orphan/sparse"
    if 0.2 <= og_univ[i] < 0.8:
        return "conditional (bimodal middle)"
    if paralog[i] >= 1:
        return "paralog-buffered"
    return "noisy/unresolved"
from collections import Counter
hunt_cats = Counter(category(i) for i in hunt)
print(f"\n=== HUNT LIST (still uncallable at P90): {len(hunt)} genes ===")
for k, v in hunt_cats.most_common():
    print(f"  {k:<32} {v}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
# (a) bucket sizes/ess-rate
labels_b = ["confident\n(br>=0.85)", "abstention\n(br<0.85)"]
sizes = [conf.sum(), ab.sum()]; ess = [y[conf].mean(), y[ab].mean()]
axb = ax[0]; x = np.arange(2)
axb.bar(x - 0.2, sizes, 0.4, color="#3182bd", label="# genes")
axb2 = axb.twinx()
axb2.bar(x + 0.2, ess, 0.4, color="#e31a1c", label="essential rate")
axb.set_xticks(x); axb.set_xticklabels(labels_b); axb.set_ylabel("# genes")
axb2.set_ylabel("essential rate", color="#e31a1c"); axb2.set_ylim(0, 1)
axb.set_title("Where the difficulty is\n(abstention bucket = the patch target)",
              fontweight="bold", fontsize=11)

# (b) coverage lift on abstention bucket
metrics = ["essCov", "neCov", "combined"]
bv = [base[m] for m in metrics]; pv = [new[m] for m in metrics]
prv = [prior[m] for m in metrics]
x = np.arange(3)
ax[1].bar(x - 0.27, bv, 0.27, color="#9ecae1", label="model alone")
ax[1].bar(x, prv, 0.27, color="#fdae6b", label="systems prior")
ax[1].bar(x + 0.27, pv, 0.27, color="#e31a1c", label="PATCHED")
ax[1].set_xticks(x); ax[1].set_xticklabels(metrics)
ax[1].set_ylabel("coverage @ precision>=0.90")
ax[1].set_title("Network -> model: coverage lift\n(abstention bucket)",
                fontweight="bold", fontsize=11)
ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
for xi, (b_, p_) in enumerate(zip(bv, pv)):
    ax[1].text(xi + 0.27, p_ + 0.02, f"+{(p_-b_)*100:.0f}pp", ha="center",
               fontsize=8, color="#e31a1c", fontweight="bold")

# (c) hunt list
items = hunt_cats.most_common()
ax[2].barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
           color="#54278f")
ax[2].set_xlabel("# genes")
ax[2].set_title(f"Neither -> hunt list ({len(hunt)} genes)\n"
                "categorized by why they're hard", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "patch_essentiality.png", dpi=130, bbox_inches="tight")
print("\nwrote outputs/orphan/patch_essentiality.png")

json.dump(dict(
    buckets=dict(confident=int(conf.sum()), confident_ess_rate=round(float(y[conf].mean()),3),
                 abstention=int(ab.sum()), abstention_ess_rate=round(float(y[ab].mean()),3)),
    abstention_coverage=dict(model_alone=base, systems_prior=prior, patched=new),
    whole_genome_coverage=dict(model_alone=whole_base, patched=whole_new),
    hunt_list=dict(total=int(len(hunt)), by_category=dict(hunt_cats)),
), open(OUT / "patch_report.json", "w"), indent=2)
print("wrote outputs/orphan/patch_report.json")
