"""Why did CONDITIONAL become partially recoverable in v2?

For each feature, measure how well it distinguishes CONDITIONAL from EASY on
its own (univariate AUC), then add features one at a time to a binary
classifier and watch the recall curve to identify the load-bearing inputs.

Inputs: per_gene_mode.npz (modes already labeled by v1), the cache, and the
same auxiliary CSVs the v2 router used.
"""
import csv, json
from pathlib import Path
import numpy as np

CACHE = "outputs/orphan/af_msa_cache.npz"
PREDS = "outputs/orphan/af_torch_preds_aug.npz"
REG_CSV = "data/drive_import/labels/regulator_features.csv"
ORTH_CSV = "data/drive_import/labels/orthology_features.csv"
CODON_CSV = "data/drive_import/labels/codon_features.csv"
OUT = Path("outputs/orphan")


def auc(s, y):
    s = np.asarray(s); y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_logit_bal(X, y, l2=0.1, iters=400, lr=0.5):
    """class-balanced binary logistic regression."""
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


# =============================================================================
# Load everything (same features as v2)
# =============================================================================
print("Loading cache, modes, and features ...")
cache = np.load(CACHE, allow_pickle=True)
foc, msa, msa_mask = cache["foc"], cache["msa"], cache["msa_mask"]
y_cache = cache["y"].astype(np.int8)
clade_cache = cache["clade"]; meta_org_cache = cache["meta_org"]
lt_cache = cache["lt"]; orgs = list(cache["orgs"])
N = len(y_cache)

m = np.load(OUT / "per_gene_mode.npz", allow_pickle=True)
mode_true = m["mode_true"]

# rebuild features (same as v2)
msa_label = msa[..., 0]; msa_known = msa[..., 1]
msa_dnds  = msa[..., 2]; msa_same_clade = msa[..., 5]
msa_depth = msa_mask.sum(1).astype(np.float32)
known = msa_known * msa_mask
known_count = known.sum(1)
with np.errstate(invalid="ignore", divide="ignore"):
    dnds_sum = (msa_dnds * msa_mask).sum(1)
    dnds_n = msa_mask.sum(1)
    dnds_mean = np.where(dnds_n > 0, dnds_sum / np.maximum(dnds_n, 1), 0)
    dnds_sq = ((msa_dnds ** 2) * msa_mask).sum(1) / np.maximum(dnds_n, 1)
    dnds_var = np.clip(dnds_sq - dnds_mean ** 2, 0, None)
    dnds_max = np.where(msa_mask > 0, msa_dnds, -np.inf).max(1)
    dnds_min = np.where(msa_mask > 0, msa_dnds,  np.inf).min(1)
    dnds_max = np.where(np.isfinite(dnds_max), dnds_max, 0)
    dnds_min = np.where(np.isfinite(dnds_min), dnds_min, 0)
    dnds_spread = dnds_max - dnds_min
sc_known = known * msa_same_clade
cc_known = known * (1 - msa_same_clade)
sc_ess = (msa_label * sc_known).sum(1); cc_ess = (msa_label * cc_known).sum(1)
sc_n = sc_known.sum(1); cc_n = cc_known.sum(1)
sc_ess_rate = np.where(sc_n > 0, sc_ess / np.maximum(sc_n, 1), 0.5)
cc_ess_rate = np.where(cc_n > 0, cc_ess / np.maximum(cc_n, 1), 0.5)
same_minus_cross = sc_ess_rate - cc_ess_rate
o2c = {int(oi): str(c) for oi, c in zip(meta_org_cache, clade_cache)}
msa_org = cache["msa_org"]
n_clades_in_family = np.array([
    len({o2c[int(msa_org[i, k])] for k in range(msa_org.shape[1])
         if msa_mask[i, k] > 0 and int(msa_org[i, k]) in o2c})
    for i in range(N)], dtype=np.float32)
family_ess_rate = np.where(known_count > 0,
                           (msa_label * known).sum(1) / np.maximum(known_count, 1), 0.5)
family_ess_var = family_ess_rate * (1 - family_ess_rate)

# functional + codon + orthology aux (same as v2)
def load_dict(path, fields):
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["organism"], r["locus_tag"])] = {f: float(r.get(f, 0) or 0) for f in fields}
    return d
reg = load_dict(REG_CSV, ["is_regulator", "is_transporter", "is_signaling"])
cod = load_dict(CODON_CSV, ["cai", "gc3"])
fam = load_dict(ORTH_CSV, ["family_n_organisms", "n_paralogs_in_genome", "family_size_total"])
def col(d, k, default=0.0):
    out = np.full(N, default, np.float32)
    for i in range(N):
        nm = str(orgs[int(meta_org_cache[i])]); lt = str(lt_cache[i])
        v = d.get((nm, lt))
        if v: out[i] = v[k]
    return out
is_regulator = col(reg, "is_regulator")
is_transporter = col(reg, "is_transporter")
is_signaling = col(reg, "is_signaling")
cai_v = col(cod, "cai"); gc3_v = col(cod, "gc3")
has_cai = (cai_v != 0).astype(np.float32)
family_n_orgs = col(fam, "family_n_organisms", 1.0)
n_paralogs = col(fam, "n_paralogs_in_genome", 0.0)
family_size = col(fam, "family_size_total", 1.0)

# assemble feature matrix and names (must match v2)
FEAT_NAMES = [
    "foc_loglen","foc_lead","foc_oric","foc_gc","foc_npar","foc_dnds","foc_has_dnds",
    "msa_depth","known_count","family_ess_rate","family_ess_var",
    "dnds_mean","dnds_var","dnds_max","dnds_spread",
    "sc_ess_rate","cc_ess_rate","same_minus_cross","n_clades_in_family",
    "is_regulator","is_transporter","is_signaling",
    "cai","gc3","has_cai",
    "family_n_orgs","n_paralogs","family_size",
]
F = np.column_stack([
    foc[:, 0], foc[:, 1], foc[:, 2], foc[:, 3], foc[:, 4], foc[:, 5], foc[:, 6],
    msa_depth, known_count, family_ess_rate, family_ess_var,
    dnds_mean, dnds_var, dnds_max, dnds_spread,
    sc_ess_rate, cc_ess_rate, same_minus_cross, n_clades_in_family,
    is_regulator, is_transporter, is_signaling,
    cai_v, gc3_v, has_cai,
    family_n_orgs, n_paralogs, family_size,
]).astype(np.float32)

# binary target: CONDITIONAL vs EASY (drop SPARSE/DOMAIN/NOISY so the question
# is clean: "what distinguishes a conditional from a structurally-easy gene?")
MODE_EASY, MODE_CONDITIONAL = 0, 1
y_binary = np.where(mode_true == MODE_EASY, 0,
                    np.where(mode_true == MODE_CONDITIONAL, 1, -1))
keep = y_binary >= 0
Xb = F[keep]; yb = y_binary[keep].astype(int)
print(f"\nbinary CONDITIONAL vs EASY: {len(yb):,} genes "
      f"({yb.mean()*100:.1f}% conditional, "
      f"{(1-yb.mean())*100:.1f}% easy)")


# =============================================================================
# STEP 1 -- univariate signal: AUC of each feature alone for CONDITIONAL
# =============================================================================
print("\n" + "=" * 72)
print("STEP 1 -- univariate distinguishability (AUC of feature -> CONDITIONAL)")
print("=" * 72)
print("  rank | feature                | univariate AUC | sign")
print("  -----+------------------------+----------------+------")
ranking = []
for fi, fn in enumerate(FEAT_NAMES):
    fv = Xb[:, fi]
    a_pos = auc(fv, yb)
    a_neg = auc(-fv, yb)
    if a_pos > a_neg:
        ranking.append((fn, a_pos, "+"))
    else:
        ranking.append((fn, a_neg, "-"))
ranking.sort(key=lambda x: -x[1])
for r, (fn, a, sgn) in enumerate(ranking, 1):
    print(f"  {r:>4} | {fn:<22} | {a:>14.3f} | {sgn:>4}")
top_features = [fn for fn, _, _ in ranking[:10]]


# =============================================================================
# STEP 2 -- forward feature-addition: at each step, add the feature that most
# improves recall (with leave-one-clade-out)
# =============================================================================
print("\n" + "=" * 72)
print("STEP 2 -- forward selection: recall@precision-0.50 as features add up")
print("=" * 72)
# uses a simple held-out test split of CONDITIONAL clades, since LOCO with
# leak-clean training on 28 clades * 28 features * 5 model fits would burn
# minutes; this gives us a relative ranking.
rng = np.random.default_rng(0)
mode_clade = clade_cache[keep]
uniq = sorted(set(mode_clade.tolist()))
held = set(rng.choice(uniq, len(uniq) // 5, replace=False).tolist())
te = np.array([c in held for c in mode_clade])
tr = ~te

selected = []
remaining = list(range(len(FEAT_NAMES)))
print(f"  step | features so far                                 |  prec | recall")
print("  -----+--------------------------------------------------+-------+-------")
for step in range(1, 11):
    best_fi, best_rec, best_prec = None, -1, 0
    for fi in remaining:
        cols = selected + [fi]
        M = fit_logit_bal(Xb[tr][:, cols], yb[tr])
        p = pred_logit(M, Xb[te][:, cols])
        call = (p >= 0.5).astype(int)
        # use F1 to balance precision and recall on this minority class
        tp = int(((call == 1) & (yb[te] == 1)).sum())
        fp = int(((call == 1) & (yb[te] == 0)).sum())
        fn = int(((call == 0) & (yb[te] == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_rec:
            best_fi, best_rec, best_prec = fi, f1, (prec, rec)
    selected.append(best_fi); remaining.remove(best_fi)
    names = ", ".join(FEAT_NAMES[i] for i in selected)
    p, r = best_prec
    print(f"  {step:>4} | {names[:48]:<48} | {p:>5.2f} | {r:>5.2f}")


# =============================================================================
# STEP 3 -- feature-group ablation: drop each group, measure recall drop
# =============================================================================
print("\n" + "=" * 72)
print("STEP 3 -- group ablation: which feature class carries the signal?")
print("=" * 72)
GROUPS = {
    "FOCAL (intrinsic)":     ["foc_loglen","foc_lead","foc_oric","foc_gc","foc_npar","foc_dnds","foc_has_dnds"],
    "FAMILY SUMMARY":        ["msa_depth","known_count","family_ess_rate","family_ess_var"],
    "PER-ORTHOLOG dN/dS":    ["dnds_mean","dnds_var","dnds_max","dnds_spread"],
    "SAME-vs-CROSS CLADE":   ["sc_ess_rate","cc_ess_rate","same_minus_cross","n_clades_in_family"],
    "FUNCTIONAL (product)":  ["is_regulator","is_transporter","is_signaling"],
    "CODON USAGE":           ["cai","gc3","has_cai"],
    "ORTHOLOGY CONTEXT":     ["family_n_orgs","n_paralogs","family_size"],
}
# full model first
M_full = fit_logit_bal(Xb[tr], yb[tr])
p_full = pred_logit(M_full, Xb[te])
call_full = (p_full >= 0.5).astype(int)
tp = int(((call_full == 1) & (yb[te] == 1)).sum())
fp = int(((call_full == 1) & (yb[te] == 0)).sum())
fn = int(((call_full == 0) & (yb[te] == 1)).sum())
full_p = tp / max(1, tp + fp); full_r = tp / max(1, tp + fn)
print(f"  FULL model (28 features):  prec {full_p:.3f}  recall {full_r:.3f}\n")
print(f"  {'group dropped':<26} {'prec':>6} {'recall':>7} {'Δprec':>7} {'Δrec':>7}")
print("  " + "-" * 56)
ranking_groups = []
for gn, members in GROUPS.items():
    idx = [FEAT_NAMES.index(m) for m in members]
    keep_idx = [i for i in range(len(FEAT_NAMES)) if i not in idx]
    M = fit_logit_bal(Xb[tr][:, keep_idx], yb[tr])
    p = pred_logit(M, Xb[te][:, keep_idx])
    call = (p >= 0.5).astype(int)
    tp = int(((call == 1) & (yb[te] == 1)).sum())
    fp = int(((call == 1) & (yb[te] == 0)).sum())
    fn = int(((call == 0) & (yb[te] == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    dprec = prec - full_p; drec = rec - full_r
    print(f"  -{gn:<25} {prec:>6.3f} {rec:>7.3f} {dprec:>+7.3f} {drec:>+7.3f}")
    ranking_groups.append((gn, abs(drec)))


# =============================================================================
# STEP 4 -- mechanistic statement
# =============================================================================
print("\n" + "=" * 72)
print("STEP 4 -- which is the LOAD-BEARING signal?")
print("=" * 72)
ranking_groups.sort(key=lambda x: -x[1])
print("  feature groups ranked by recall-drop when removed (= what's carrying CONDITIONAL):")
for gn, drop in ranking_groups:
    print(f"    {gn:<26} drops recall by {drop:.3f} when removed")

# write everything to disk
json.dump(dict(
    univariate_ranking=[dict(feature=fn, auc=round(a, 3), sign=s) for fn, a, s in ranking],
    forward_selection_top=[FEAT_NAMES[i] for i in selected],
    full_model=dict(precision=round(full_p, 3), recall=round(full_r, 3)),
    group_ablation_ranking=[dict(group=gn, recall_drop_when_removed=round(d, 3))
                              for gn, d in ranking_groups],
), open(OUT / "conditional_mechanism.json", "w"), indent=2)
print(f"\nwrote outputs/orphan/conditional_mechanism.json")
