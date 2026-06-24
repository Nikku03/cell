"""Everything-bagel test: ALL features in, see what survives.

Load every per-gene feature we've computed across the session, build a
~50-column feature matrix, and run LOCO precision-coverage on the abstention
bucket with feature blocks added incrementally. Final config uses every feature.

Feature blocks (cumulative):
  B0  baseline (6): p, brightness, og_univ_LOCO, log_paralogs, log_breadth, is_orphan
  B1  + orthology  : family_size, family_essrate_leakfree, fold-mean ess-rate
  B2  + co-occurr  : jaccard, log_n50, log_n80, ess_neighbor_frac
  B3  + regulator  : is_regulator/transporter/signaling/conditional
  B4  + codon      : cai, gc, gc3, log_cds_len, intergenic+strand x4
  B5  + slot       : 13-way functional-slot one-hot (from product keywords)
  B6  + universal  : universal-core flag
  B7  + FBA        : fba_essential (3 orgs; Keio via the b-number bridge)
  B8  + JEPA       : residual + 8 embedding dims (from jepa.py)
  ALL: every feature above

For each config: essCov @ P>=0.90, neCov @ P>=0.90, combined.
Plus per-clade lift on the final ALL config and feature-importance (logistic
coefficients).

Output: outputs/orphan/all_features.png / .json
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
RNG = np.random.default_rng(0)

# ============== LOAD EVERYTHING ==============
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
p = P["p"].astype(float); br = P["brightness"].astype(float)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
valid = ~(np.isnan(p) | np.isnan(br))
print(f"predictions: {len(p):,} genes, {int(valid.sum()):,} valid")

C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {corgs[int(cm[i])]: str(cc[i]) for i in range(len(cm))}

# orthology
orth = {}; obylt = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    fold_rates = [float(r.get(f"family_frac_essential_fold{i}") or 0) for i in range(5)]
    orth[k] = (r.get("og_id") or "",
               float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_size_total") or 0),
               float(r.get("family_n_organisms") or 0),
               float(r.get("is_orphan") or 0),
               float(r.get("family_frac_essential_leakfree") or 0),
               float(np.mean(fold_rates)))
    obylt[r["locus_tag"]].append(r["organism"])

# resolve org per prediction
resolved = np.empty(len(p), object)
for i in range(len(p)):
    cands = obylt.get(lt[i], [])
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    resolved[i] = ch or ""

# co-occurrence
co = {}
for r in csv.DictReader(open("data/drive_import/labels/cooccurrence_features.csv")):
    try:
        co[(r["organism"], r["locus_tag"])] = (
            float(r.get("cooccur_max_jaccard") or 0),
            float(r.get("cooccur_n_neighbors_50") or 0),
            float(r.get("cooccur_n_neighbors_80") or 0),
            float(r.get("cooccur_essential_neighbor_frac") or 0))
    except (ValueError, KeyError): pass

# regulator
reg = {}
for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv")):
    reg[(r["organism"], r["locus_tag"])] = (
        float(r.get("is_regulator") or 0),
        float(r.get("is_transporter") or 0),
        float(r.get("is_signaling") or 0),
        float(r.get("is_conditional") or 0))

# codon (limited orgs)
codon = {}
for r in csv.DictReader(open("data/drive_import/labels/codon_features.csv")):
    try:
        codon[(r["organism"], r["locus_tag"])] = (
            float(r.get("cai") or 0), float(r.get("gc") or 0),
            float(r.get("gc3") or 0), float(r.get("cds_length") or 0),
            float(r.get("intergenic_prev") or 0), float(r.get("intergenic_next") or 0),
            float(r.get("same_strand_prev") or 0), float(r.get("same_strand_next") or 0))
    except (ValueError, KeyError): pass

# FBA (3 orgs) + Keio b-number bridge
fba = {}
for r in csv.DictReader(open("data/drive_import/labels/fba_features.csv")):
    try: fba[(r["organism"], r["locus_tag"])] = float(r["fba_essential"])
    except (ValueError, KeyError): pass
bridge = {}
for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv")):
    if r["source"] in {"anchor_product", "anchor_product_round2", "synteny_walk"}:
        bridge[r["locus_tag"]] = r["b_number"]

# annotations -> slot
annot = {}
for r in csv.DictReader(open("data/drive_import/labels/gene_annotations.csv")):
    if r.get("product"):
        annot[(r["organism"], r["locus_tag"])] = r["product"]
SLOTS = [
    ("ribosomal", [r"ribosom"]),
    ("synthetase", [r"--trna lig", r"trna ligase", r"synthetase", r"aminoacyl"]),
    ("trna", [r"\btrna\b"]),
    ("transcription", [r"rna polymerase", r"transcription", r"sigma"]),
    ("replication", [r"dna polymerase", r"helicase", r"gyrase", r"primase",
                     r"topoisomer", r"replicat", r"dna ligase"]),
    ("translation", [r"elongation factor", r"initiation factor", r"release factor",
                     r"chaperon"]),
    ("cell_division", [r"cell division", r"septation", r"ftsz", r"divisome"]),
    ("membrane", [r"membrane", r"lipoprotein", r"porin"]),
    ("energy", [r"atp synthase", r"atpase", r"cytochrome", r"nadh"]),
    ("nucleotide", [r"purine", r"pyrimidine", r"nucleotide"]),
    ("amino_acid", [r"amino acid", r"glutam", r"aspart"]),
    ("cofactor", [r"folate", r"thiamine", r"riboflavin", r"coenzyme"]),
    ("metabolism", [r"dehydrogenase", r"reductase", r"transferase", r"kinase"]),
]
SP = [(nm, [re.compile(x, re.I) for x in pats]) for nm, pats in SLOTS]
def slot_idx(prod):
    if not prod: return -1
    pl = prod.lower()
    for i, (_, rx) in enumerate(SP):
        if any(r.search(pl) for r in rx): return i
    return -1

# OG presence (for universal-core flag)
og_present = defaultdict(set)
for (o, ltt), info in orth.items():
    if info[0]: og_present[info[0]].add(o)
n_orgs = len({o for s in og_present.values() for o in s})

# ============== LEAK-CLEAN OG-universality (LOCO) ==============
gs = defaultdict(float); gn = defaultdict(float)
csum = defaultdict(float); cn = defaultdict(float)
for i in np.where(valid)[0]:
    if not resolved[i]: continue
    og = orth.get((resolved[i], lt[i]), ("",))[0]
    if not og: continue
    gs[og] += y[i]; gn[og] += 1
    csum[(og, clade[i])] += y[i]; cn[(og, clade[i])] += 1
og_univ = np.full(len(p), np.nan)
for i in np.where(valid)[0]:
    if not resolved[i]: continue
    og = orth.get((resolved[i], lt[i]), ("",))[0]
    if not og: continue
    n = gn[og] - cn[(og, clade[i])]
    if n > 0: og_univ[i] = (gs[og] - csum[(og, clade[i])]) / n

mask = valid & (resolved != "") & ~np.isnan(og_univ)
print(f"workable (valid + resolved + og_univ): {int(mask.sum()):,}")

# ============== JEPA: load embeddings or train ==============
JEPA_CACHE = OUT / "jepa_embeddings.npz"
if JEPA_CACHE.exists():
    jc = np.load(JEPA_CACHE)
    Z_self = jc["Z_self"]; resid = jc["resid"]
    print("loaded cached JEPA embeddings")
else:
    print("training JEPA (8 feat -> 32 emb, 25 epochs)...")
    # 8 leak-clean structural features (same as jepa.py)
    X8 = np.zeros((len(p), 8))
    for i in range(len(p)):
        if not resolved[i]: continue
        ent = orth.get((resolved[i], lt[i]))
        if ent is None: continue
        _, par, _, brd, orph, _, _ = ent
        jac, n50, n80, _ = co.get((resolved[i], lt[i]), (0, 0, 0, 0))
        isr, ist, iss, _ = reg.get((resolved[i], lt[i]), (0, 0, 0, 0))
        X8[i] = [np.log1p(par), np.log1p(brd), orph, jac, np.log1p(n80),
                 isr, ist, iss]
    mu8 = X8[mask].mean(0); sd8 = X8[mask].std(0) + 1e-9
    Xn = (X8 - mu8) / sd8
    # context = mean of operon + phyletic neighbors
    synt = {}
    for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
        synt[(r["organism"], r["locus_tag"])] = (r.get("prev_locus_tag", ""),
                                                  r.get("next_locus_tag", ""))
    idx_by_key = {(resolved[i], lt[i]): i for i in range(len(p)) if resolved[i]}
    og_members = defaultdict(list)
    for i in range(len(p)):
        og = orth.get((resolved[i], lt[i]), ("",))[0] if resolved[i] else ""
        if og: og_members[og].append(i)
    neigh = [[] for _ in range(len(p))]
    for i in range(len(p)):
        if not resolved[i]: continue
        k = (resolved[i], lt[i])
        pi, ni = synt.get(k, ("", ""))
        for nb in (pi, ni):
            if nb and (resolved[i], nb) in idx_by_key:
                neigh[i].append(idx_by_key[(resolved[i], nb)])
        og = orth.get(k, ("",))[0]
        if og and len(og_members[og]) > 1:
            mems = [m for m in og_members[og] if m != i]
            if len(mems) > 6: mems = list(RNG.choice(mems, 6, replace=False))
            neigh[i] += list(mems)
    Xctx = np.zeros_like(X8)
    for i in range(len(p)):
        if neigh[i]:
            Xctx[i] = Xn[neigh[i]].mean(0)
    # JEPA
    def init_mlp(di, dh, do):
        return (RNG.standard_normal((di, dh)).astype(np.float32) * np.sqrt(2/di),
                np.zeros(dh, np.float32),
                RNG.standard_normal((dh, do)).astype(np.float32) * np.sqrt(2/dh),
                np.zeros(do, np.float32))
    def mlp_fwd(W1, b1, W2, b2, x):
        h = np.maximum(0, x @ W1 + b1); return h @ W2 + b2, h
    enc = init_mlp(8, 64, 32); tgt = tuple(q.copy() for q in enc)
    prd = init_mlp(32, 64, 32)
    LR = 0.01; TAU = 0.99
    has_n = np.array([len(neigh[i]) > 0 for i in range(len(p))])
    tr_idx = np.where(mask & has_n)[0]
    for ep in range(25):
        perm = RNG.permutation(tr_idx)
        for s in range(0, len(perm), 4096):
            b = perm[s:s+4096]
            if len(b) < 16: continue
            ctx = Xctx[b].astype(np.float32); tgx = Xn[b].astype(np.float32)
            z_t, _ = mlp_fwd(*tgt, tgx)
            z_c, h_e = mlp_fwd(*enc, ctx); z_p, h_p = mlp_fwd(*prd, z_c)
            err = z_p - z_t; d_zp = 2 * err / err.size
            W1p, b1p, W2p, b2p = prd
            dW2p = h_p.T @ d_zp; db2p = d_zp.sum(0)
            d_hp = d_zp @ W2p.T * (h_p > 0)
            dW1p = z_c.T @ d_hp; db1p = d_hp.sum(0)
            d_zc = d_hp @ W1p.T
            W1e, b1e, W2e, b2e = enc
            dW2e = h_e.T @ d_zc; db2e = d_zc.sum(0)
            d_he = d_zc @ W2e.T * (h_e > 0)
            dW1e = ctx.T @ d_he; db1e = d_he.sum(0)
            prd = (W1p-LR*dW1p, b1p-LR*db1p, W2p-LR*dW2p, b2p-LR*db2p)
            enc = (W1e-LR*dW1e, b1e-LR*db1e, W2e-LR*dW2e, b2e-LR*db2e)
            tgt = tuple(TAU*t + (1-TAU)*o_ for t, o_ in zip(tgt, enc))
    Z_self = np.zeros((len(p), 32), np.float32); Z_pred = np.zeros_like(Z_self)
    for s in range(0, len(p), 8192):
        e = min(s+8192, len(p))
        Z_self[s:e], _ = mlp_fwd(*tgt, Xn[s:e].astype(np.float32))
        z_c, _ = mlp_fwd(*enc, Xctx[s:e].astype(np.float32))
        Z_pred[s:e], _ = mlp_fwd(*prd, z_c)
    resid = np.full(len(p), np.nan)
    valid_jepa = mask & has_n
    resid[valid_jepa] = ((Z_pred[valid_jepa] - Z_self[valid_jepa]) ** 2).sum(1)
    np.savez_compressed(JEPA_CACHE, Z_self=Z_self, resid=resid)
    print(f"JEPA cached: {(~np.isnan(resid)).sum():,} genes with residual")

# ============== BUILD FEATURE BLOCKS ==============
print("\nbuilding feature blocks...")
N = len(p)
# Block B0: baseline (6)
F_BASE = np.column_stack([
    p, br, np.where(np.isnan(og_univ), 0, og_univ),
    np.log1p([orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[1] for i in range(N)]),
    np.log1p([orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[3] for i in range(N)]),
    np.array([orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[4] for i in range(N)]),
])
# Block B1: more orthology (3 cols)
F_ORTH = np.column_stack([
    np.log1p([orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[2] for i in range(N)]),
    [orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[5] for i in range(N)],
    [orth.get((resolved[i], lt[i]), ("", 0, 0, 0, 0, 0, 0))[6] for i in range(N)],
])
# Block B2: co-occurrence (4)
F_CO = np.array([co.get((resolved[i], lt[i]), (0., 0., 0., 0.)) for i in range(N)])
F_CO[:, 1:3] = np.log1p(F_CO[:, 1:3])
# Block B3: regulator (4)
F_REG = np.array([reg.get((resolved[i], lt[i]), (0., 0., 0., 0.)) for i in range(N)])
# Block B4: codon (8)
F_COD = np.array([codon.get((resolved[i], lt[i]), (0.,)*8) for i in range(N)])
F_COD[:, 3] = np.log1p(F_COD[:, 3])     # log cds_length
# Block B5: slot one-hot (13)
S_idx = np.array([slot_idx(annot.get((resolved[i], lt[i]), "")) for i in range(N)])
F_SLOT = np.zeros((N, len(SLOTS)))
for i in range(N):
    if S_idx[i] >= 0: F_SLOT[i, S_idx[i]] = 1
# Block B6: universal-core (1)
F_UNI = np.array([(1. if orth.get((resolved[i], lt[i]), ("",))[0]
                   and len(og_present[orth[(resolved[i], lt[i])][0]]) / n_orgs >= 0.9
                   else 0.) for i in range(N)]).reshape(-1, 1)
# Block B7: FBA (1 + availability flag = 2)
F_FBA = np.zeros((N, 2))
for i in range(N):
    if not resolved[i]: continue
    f_ = fba.get((resolved[i], lt[i]))
    if f_ is None and resolved[i] == "beril_Keio":
        b = bridge.get(lt[i])
        if b: f_ = fba.get(("beril_Keio", b))
    if f_ is not None: F_FBA[i, 0] = f_; F_FBA[i, 1] = 1
# Block B8: JEPA (1 residual + 8 emb)
F_JEPA = np.column_stack([np.where(np.isnan(resid), 0, -resid),     # flip
                          Z_self[:, :8]])

# names per block
BLOCK_NAMES = {
    "B0_baseline": ["p", "brightness", "og_univ_loco", "log_paralogs",
                    "log_breadth", "is_orphan"],
    "B1_orth+": ["log_family_size", "family_essrate_lf", "family_essrate_fold_mean"],
    "B2_cooccur": ["co_jac", "log_co_n50", "log_co_n80", "co_ess_nb_frac"],
    "B3_regulator": ["is_regulator", "is_transporter", "is_signaling", "is_conditional"],
    "B4_codon": ["cai", "gc", "gc3", "log_cds_len", "intgn_prev", "intgn_next",
                 "strand_prev", "strand_next"],
    "B5_slot": [f"slot_{s[0]}" for s in SLOTS],
    "B6_universal": ["universal_core"],
    "B7_FBA": ["fba_essential", "fba_available"],
    "B8_JEPA": ["jepa_neg_resid"] + [f"jepa_emb_{i}" for i in range(8)],
}
BLOCKS = [F_BASE, F_ORTH, F_CO, F_REG, F_COD, F_SLOT, F_UNI, F_FBA, F_JEPA]
for j, B in enumerate(BLOCKS):
    BLOCKS[j] = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    nan_count = int(np.isnan(B).sum())
    if nan_count:
        print(f"  cleaned {nan_count} NaN in block {list(BLOCK_NAMES)[j]}")
F_BASE, F_ORTH, F_CO, F_REG, F_COD, F_SLOT, F_UNI, F_FBA, F_JEPA = BLOCKS
print(f"feature blocks built: " +
      ", ".join(f"{k}={v.shape[1]}d" for k, v in
                zip(BLOCK_NAMES, BLOCKS)) +
      f"  | total={sum(b.shape[1] for b in BLOCKS)}d")

# ============== STACKER + LOCO ==============
def fit(X, yy, l2=1.0, it=500, lr=0.4):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    w, mu_, sd_ = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu_) / sd_] @ w, -30, 30)))
def cov(score, yt, pos=True, t=0.9):
    sc = score if pos else -score; ys = yt if pos else 1 - yt
    o = np.argsort(-sc); o_ys = ys[o]; tp = np.cumsum(o_ys); fp = np.cumsum(1 - o_ys)
    prec = tp / np.maximum(tp + fp, 1); ok = prec >= t
    return float(tp[ok].max() / max(ys.sum(), 1)) if ok.any() else 0.0

def loco_eval(F):
    ab = mask & (br < 0.85)
    pr = np.full(len(p), np.nan)
    for H in sorted(set(clade[ab])):
        tr = ab & (clade != H); te = ab & (clade == H)
        if tr.sum() < 200 or te.sum() < 5 or y[tr].sum() < 20: continue
        pr[te] = predL(fit(F[tr], y[tr]), F[te])
    done = ab & ~np.isnan(pr)
    s = pr[done]; yy = y[done].astype(int)
    return dict(essCov=round(cov(s, yy, True), 3),
                neCov=round(cov(s, yy, False), 3),
                combined=round((cov(s, yy, True) * yy.sum()
                                + cov(s, yy, False) * (1 - yy).sum())
                               / max(len(yy), 1), 3),
                n=int(done.sum()), score=pr, mask=done)

# cumulative block additions
results = []; coef_data = []
names_so_far = list(BLOCK_NAMES["B0_baseline"])
F_cum = F_BASE.copy()
r = loco_eval(F_cum); r["label"] = "B0 baseline (6)"
results.append(r); print(f"  {r['label']:<40} essCov={r['essCov']:.3f} "
                          f"neCov={r['neCov']:.3f} combined={r['combined']:.3f}")
for idx, (key, fcols) in enumerate(zip(list(BLOCK_NAMES)[1:], BLOCKS[1:]), 1):
    F_cum = np.column_stack([F_cum, fcols])
    names_so_far += BLOCK_NAMES[key]
    r = loco_eval(F_cum)
    r["label"] = f"+ {key.split('_', 1)[1]} ({F_cum.shape[1]})"
    results.append(r)
    print(f"  {r['label']:<40} essCov={r['essCov']:.3f} "
          f"neCov={r['neCov']:.3f} combined={r['combined']:.3f}")

# Final coefficient analysis: fit on all data, take normalized weights
print("\nfeature importance (final fit, all valid + abstention):")
ab = mask & (br < 0.85)
w_final, mu_f, sd_f = fit(F_cum[ab], y[ab])
coefs = w_final[1:]                   # drop intercept
order = np.argsort(-np.abs(coefs))
for i in order[:25]:
    print(f"  {names_so_far[i]:<32}  w = {coefs[i]:+.3f}")

# per-clade lift on FINAL config vs baseline
def per_clade(F):
    ab = mask & (br < 0.85)
    out = {}
    for H in sorted(set(clade[ab])):
        tr = ab & (clade != H); te = ab & (clade == H)
        if tr.sum() < 200 or te.sum() < 5 or y[tr].sum() < 20: continue
        pr = predL(fit(F[tr], y[tr]), F[te])
        s = pr; yy = y[te].astype(int)
        out[H] = round(cov(s, yy, True), 3)
    return out
pc_base = per_clade(F_BASE)
pc_all = per_clade(F_cum)
print(f"\nper-clade essCov@P90 lift (top gains/losses):")
diffs = sorted([(c, pc_all.get(c, 0) - pc_base.get(c, 0))
                for c in pc_base], key=lambda x: -x[1])
for c, d in diffs[:6]: print(f"  +{d:.3f}  {c}")
for c, d in diffs[-6:][::-1]: print(f"  {d:+.3f}  {c}")

# ============== figure ==============
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
labels = [r["label"] for r in results]
ess = [r["essCov"] for r in results]
combined = [r["combined"] for r in results]
xs = np.arange(len(results))
ax[0].plot(xs, ess, "o-", color="#e31a1c", lw=2, label="essCov @ P>=0.90")
ax[0].plot(xs, combined, "s-", color="#3182bd", lw=2, label="combined coverage")
for i, r in enumerate(results):
    ax[0].text(i, r["essCov"] + 0.005, f"{r['essCov']:.3f}", ha="center",
               fontsize=8.5, color="#e31a1c")
ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax[0].set_ylabel("coverage @ precision >= 0.90 (abstention bucket)")
ax[0].set_title("Add every feature block, cumulative LOCO\n(red = essential coverage, blue = combined)",
                fontweight="bold", fontsize=11)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

# top-N coefficient
top = order[:18]
ax[1].barh(np.arange(len(top)), coefs[top],
           color=["#e31a1c" if c > 0 else "#3182bd" for c in coefs[top]])
ax[1].set_yticks(np.arange(len(top)))
ax[1].set_yticklabels([names_so_far[i] for i in top], fontsize=8)
ax[1].invert_yaxis(); ax[1].axvline(0, color="k", lw=0.5)
ax[1].set_xlabel("logistic weight (standardized)")
ax[1].set_title("Top features by absolute weight\n(red = predicts essential, blue = predicts non-essential)",
                fontweight="bold", fontsize=11)

# per-clade lift
items = [(c, pc_all.get(c, 0) - pc_base.get(c, 0)) for c in sorted(pc_base)]
items.sort(key=lambda x: x[1])
ax[2].barh(range(len(items)), [d for _, d in items],
           color=["#e31a1c" if d > 0 else "#3182bd" for _, d in items])
ax[2].axvline(0, color="k", lw=0.5)
ax[2].set_yticks(range(len(items)))
ax[2].set_yticklabels([c[:24] for c, _ in items], fontsize=7.5)
ax[2].set_xlabel("Δ essCov@P90 (all features - baseline)")
ax[2].set_title("Per-clade lift: ALL features vs baseline\n(positive = improvement)",
                fontweight="bold", fontsize=11)

fig.suptitle("Everything-bagel test: cumulative feature addition through the patch stacker",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT / "all_features.png", dpi=120, bbox_inches="tight")
print(f"\nwrote outputs/orphan/all_features.png")

json.dump(dict(n_features_per_config=[F_BASE.shape[1]] +
                                      [F_BASE.shape[1] + sum(BLOCKS[1:j+1].__sizeof__()*0
                                                              + BLOCKS[k].shape[1]
                                                              for k in range(1, j+1))
                                       for j in range(1, 9)],
               results=[{k: v for k, v in r.items() if k not in ("score", "mask")}
                         for r in results],
               top_features=[(names_so_far[i], round(float(coefs[i]), 4))
                             for i in order[:25]],
               per_clade_lift={c: round(pc_all.get(c, 0) - pc_base.get(c, 0), 3)
                               for c in pc_base}),
          open(OUT / "all_features.json", "w"), indent=2)
print("wrote outputs/orphan/all_features.json")
