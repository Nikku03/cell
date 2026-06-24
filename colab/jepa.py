"""JEPA-lite: self-supervised gene-context prediction in latent space.

Predict each gene's embedding from its (operon + phyletic) context embeddings.
Asymmetric design (online encoder + EMA target encoder + predictor + stop-grad)
prevents collapse without contrastive negatives -- the canonical JEPA recipe,
reduced to a pure-numpy MLP for sandbox runtime.

  online encoder f_theta:    8-d gene features -> 32-d embedding
  target encoder f_theta':   same shape, EMA-updated copy (NEVER trained)
  predictor      g_phi:      32 -> 32
  loss:                      ||g(f(context)) - sg(f'(target))||^2

Features per gene (8, all leak-clean -- no essentiality label leakage):
  log_paralogs, log_breadth, is_orphan, cooccur_max_jaccard,
  log_n_neighbors_80, is_regulator, is_transporter, is_signaling

After training (UNSUPERVISED -- no essentiality labels touched), evaluate:
  A) Residual norm vs essentiality: per-clade AUC (zero-shot question)
  B) Add JEPA embedding to baseline stacker, measure precision lift.

Output: outputs/orphan/jepa.png / .json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/orphan")
RNG = np.random.default_rng(0)

# ============== load features (LEAK-CLEAN structural only) ==============
P = np.load(OUT / "af_torch_preds_aug.npz", allow_pickle=True)
y = P["y"].astype(float); clade = P["clade"].astype(str); lt = P["lt"].astype(str)
br = P["brightness"].astype(float); p_arr = P["p"].astype(float)
valid_pred = ~(np.isnan(p_arr) | np.isnan(br))

C = np.load(OUT / "af_msa_cache.npz", allow_pickle=True)
corgs = [str(x) for x in C["orgs"]]; cm = C["meta_org"]; cc = C["clade"].astype(str)
org2clade = {corgs[int(cm[i])]: str(cc[i]) for i in range(len(cm))}

feat = {}; obylt = defaultdict(list)
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    k = (r["organism"], r["locus_tag"])
    feat[k] = (float(r.get("n_paralogs_in_genome") or 0),
               float(r.get("family_n_organisms") or 0),
               float(r.get("is_orphan") or 0),
               r.get("og_id") or "")
    obylt[r["locus_tag"]].append(r["organism"])
co = {}
for r in csv.DictReader(open("data/drive_import/labels/cooccurrence_features.csv")):
    try:
        co[(r["organism"], r["locus_tag"])] = (
            float(r.get("cooccur_max_jaccard", 0) or 0),
            float(r.get("cooccur_n_neighbors_80", 0) or 0))
    except (ValueError, KeyError): pass
reg = {}
for r in csv.DictReader(open("data/drive_import/labels/regulator_features.csv")):
    reg[(r["organism"], r["locus_tag"])] = (
        float(r.get("is_regulator", 0) or 0),
        float(r.get("is_transporter", 0) or 0),
        float(r.get("is_signaling", 0) or 0))
synt = {}
for r in csv.DictReader(open("data/drive_import/labels/synteny_features.csv")):
    synt[(r["organism"], r["locus_tag"])] = (r.get("prev_locus_tag", ""),
                                              r.get("next_locus_tag", ""))

# resolve organism per prediction by clade match on locus_tag
resolved = np.empty(len(p_arr), object)
X = np.zeros((len(p_arr), 8))
for i in range(len(p_arr)):
    cands = obylt.get(lt[i], [])
    match = [o for o in cands if org2clade.get(o) == clade[i]]
    ch = match[0] if len(match) == 1 else (cands[0] if len(cands) == 1 else None)
    resolved[i] = ch or ""
    if ch is None: continue
    par, brd, orph, _og = feat[(ch, lt[i])]
    jac, nb80 = co.get((ch, lt[i]), (0.0, 0.0))
    isr, ist, iss = reg.get((ch, lt[i]), (0.0, 0.0, 0.0))
    X[i] = [np.log1p(par), np.log1p(brd), orph, jac, np.log1p(nb80),
            isr, ist, iss]
have_feat = (resolved != "") & valid_pred
print(f"valid gene-feature rows: {have_feat.sum():,}")

# normalize features (z-score over the dataset)
mu = X[have_feat].mean(0); sd = X[have_feat].std(0) + 1e-9
Xn = (X - mu) / sd

# ============== build (operon + phyletic) context neighbours ==============
# operon neighbours via synteny prev/next
# phyletic neighbours via shared OG (use family members at the same OG, exclude self)
# index by (org, locus_tag)
idx_by_key = {}
for i in range(len(p_arr)):
    if resolved[i] != "":
        idx_by_key[(resolved[i], lt[i])] = i
# operon
op_neighbours = [[] for _ in range(len(p_arr))]
for i, k in enumerate(idx_by_key):
    pi, ni = synt.get(k, ("", ""))
    org = k[0]
    for nb in (pi, ni):
        if nb and (org, nb) in idx_by_key:
            op_neighbours[idx_by_key[k]].append(idx_by_key[(org, nb)])
# phyletic: for each OG, the genes in OTHER organisms (cap to keep memory ok)
og_by_idx = np.empty(len(p_arr), object)
og_members = defaultdict(list)
for i, k in enumerate(idx_by_key):
    og = feat.get(k, ("", "", "", ""))[3]
    og_by_idx[idx_by_key[k]] = og
    if og: og_members[og].append(idx_by_key[k])
# combine into one neighbour list per gene (operon + a few phyletic samples)
neigh = [[] for _ in range(len(p_arr))]
for i in range(len(p_arr)):
    nb = list(op_neighbours[i])
    og = og_by_idx[i]
    if og and len(og_members[og]) > 1:
        mems = [m for m in og_members[og] if m != i]
        if len(mems) > 6:
            mems = list(RNG.choice(mems, 6, replace=False))
        nb += mems
    neigh[i] = nb
has_neigh = np.array([len(neigh[i]) > 0 for i in range(len(p_arr))])
train_mask = have_feat & has_neigh
print(f"genes with context (operon+phyletic): {train_mask.sum():,}")

# Per-gene context features = mean of neighbour features
Xctx = np.zeros_like(X)
for i in np.where(train_mask)[0]:
    Xctx[i] = Xn[neigh[i]].mean(0)

# ============== JEPA-lite (numpy MLPs + EMA + stop-grad) ==============
D_IN, D_HID, D_EMB = 8, 64, 32
def init_mlp(d_in, d_hid, d_out, rng):
    return (rng.standard_normal((d_in, d_hid)).astype(np.float32) * np.sqrt(2/d_in),
            np.zeros(d_hid, np.float32),
            rng.standard_normal((d_hid, d_out)).astype(np.float32) * np.sqrt(2/d_hid),
            np.zeros(d_out, np.float32))
def mlp_fwd(W1, b1, W2, b2, x):
    h = np.maximum(0, x @ W1 + b1)
    return h @ W2 + b2, h

# online encoder f, target encoder f' (EMA), predictor g
enc = init_mlp(D_IN, D_HID, D_EMB, RNG)
tgt = tuple(p.copy() for p in enc)             # target encoder (EMA copy)
prd = init_mlp(D_EMB, D_HID, D_EMB, RNG)

LR = 0.01; TAU = 0.99; EPOCHS = 25; BS = 4096
n_train = int(train_mask.sum())
train_idx = np.where(train_mask)[0]
losses = []
print(f"\nJEPA training: {EPOCHS} epochs, {n_train:,} genes, batch {BS}")
for ep in range(EPOCHS):
    perm = RNG.permutation(train_idx)
    epoch_loss = 0; n_b = 0
    for start in range(0, len(perm), BS):
        batch = perm[start:start + BS]
        if len(batch) < 16: continue
        ctx = Xctx[batch].astype(np.float32)
        tgt_x = Xn[batch].astype(np.float32)
        # target embedding (no grad)
        z_t, _ = mlp_fwd(*tgt, tgt_x)
        # online embedding of context, then predict target
        z_c, h_enc = mlp_fwd(*enc, ctx)
        z_p, h_prd = mlp_fwd(*prd, z_c)
        # MSE loss
        err = z_p - z_t                                       # [B, D_EMB]
        loss = (err ** 2).mean()
        epoch_loss += loss; n_b += 1
        # backprop through predictor + encoder (NOT through target)
        d_z_p = 2 * err / err.size
        # predictor backward
        W1p, b1p, W2p, b2p = prd
        dW2p = h_prd.T @ d_z_p; db2p = d_z_p.sum(0)
        d_h_prd = d_z_p @ W2p.T * (h_prd > 0)
        dW1p = z_c.T @ d_h_prd; db1p = d_h_prd.sum(0)
        d_z_c = d_h_prd @ W1p.T
        # encoder backward
        W1e, b1e, W2e, b2e = enc
        dW2e = h_enc.T @ d_z_c; db2e = d_z_c.sum(0)
        d_h_enc = d_z_c @ W2e.T * (h_enc > 0)
        dW1e = ctx.T @ d_h_enc; db1e = d_h_enc.sum(0)
        # update
        prd = (W1p - LR*dW1p, b1p - LR*db1p, W2p - LR*dW2p, b2p - LR*db2p)
        enc = (W1e - LR*dW1e, b1e - LR*db1e, W2e - LR*dW2e, b2e - LR*db2e)
        # EMA target encoder
        tgt = tuple(TAU * t + (1 - TAU) * o for t, o in zip(tgt, enc))
    avg = epoch_loss / max(n_b, 1); losses.append(float(avg))
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep+1:>2}  loss = {avg:.4f}")

# ============== embeddings + residual norm for ALL genes ==============
Z_self = np.zeros((len(p_arr), D_EMB), np.float32)
Z_pred = np.zeros((len(p_arr), D_EMB), np.float32)
resid = np.full(len(p_arr), np.nan)
for start in range(0, len(p_arr), 8192):
    end = min(start + 8192, len(p_arr))
    Z_self[start:end], _ = mlp_fwd(*tgt, Xn[start:end].astype(np.float32))
    z_c, _ = mlp_fwd(*enc, Xctx[start:end].astype(np.float32))
    Z_pred[start:end], _ = mlp_fwd(*prd, z_c)
resid[train_mask] = ((Z_pred[train_mask] - Z_self[train_mask]) ** 2).sum(1)
print(f"residual computed for {(~np.isnan(resid)).sum():,} genes")

# ============== EVALUATE (A): residual vs essentiality, per clade ==============
def auc_of(scores, ys):
    o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(len(scores))
    pos = ys == 1
    if pos.sum() == 0 or (~pos).sum() == 0: return 0.5
    return float((r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) /
                  (pos.sum() * (~pos).sum()))
clades = sorted(set(clade[train_mask]))
per_clade_auc = {}                # directional AUC: -residual as the score
                                  # (low residual -> essential -- this is the
                                  # learned direction: essentials are MORE
                                  # predictable from their conserved context)
for H in clades:
    m = train_mask & (clade == H) & ~np.isnan(resid)
    if m.sum() < 100: continue
    per_clade_auc[H] = auc_of(-resid[m], y[m].astype(int))
mean_auc = float(np.mean(list(per_clade_auc.values())))
pos_above_055 = sum(1 for a in per_clade_auc.values() if a > 0.55)
pos_above_065 = sum(1 for a in per_clade_auc.values() if a > 0.65)
print(f"\n=== EVAL A: residual norm vs essentiality (zero-shot, no labels in training) ===")
print(f"  signal direction: LOW residual -> ESSENTIAL "
      f"(essentials are MORE predictable from their conserved context)")
print(f"  {len(per_clade_auc)} clades evaluated, mean directional AUC = {mean_auc:.3f}")
print(f"  clades with AUC > 0.55: {pos_above_055} / {len(per_clade_auc)}")
print(f"  clades with AUC > 0.65: {pos_above_065} / {len(per_clade_auc)}")
top = sorted(per_clade_auc.items(), key=lambda kv: -kv[1])[:6]
bot = sorted(per_clade_auc.items(), key=lambda kv: kv[1])[:6]
print("  top 6 clades:", [(c, round(a,3)) for c,a in top])
print("  bot 6 clades:", [(c, round(a,3)) for c,a in bot])

# ============== EVALUATE (B): add JEPA embedding to baseline stacker ==============
# Baseline = (p, brightness, og_universality, paralogs, breadth, orphan).
# Compute LOCO og_universality (leak-clean) so we don't double-leak.
og_essn = defaultdict(float); og_essd = defaultdict(float)
c_essn = defaultdict(float); c_essd = defaultdict(float)
for i in np.where(have_feat)[0]:
    og = og_by_idx[i]
    if not og: continue
    og_essn[og] += y[i]; og_essd[og] += 1
    c_essn[(og, clade[i])] += y[i]; c_essd[(og, clade[i])] += 1
ou = np.full(len(p_arr), np.nan)
for i in np.where(have_feat)[0]:
    og = og_by_idx[i]
    if not og: continue
    n = og_essd[og] - c_essd[(og, clade[i])]
    if n > 0:
        ou[i] = (og_essn[og] - c_essn[(og, clade[i])]) / n

def fit(Xf, yy, l2=1.0, it=400, lr=.4):
    mu_ = Xf.mean(0); sd_ = Xf.std(0) + 1e-9
    Xs = np.c_[np.ones(len(Xf)), (Xf - mu_) / sd_]; w = np.zeros(Xs.shape[1])
    cb = np.bincount(yy.astype(int), minlength=2).astype(float)
    cw = (cb.sum() / np.maximum(cb, 1)) ** .5; sw = cw[yy.astype(int)]; sw /= sw.mean()
    for _ in range(it):
        pr = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        g = Xs.T @ ((pr - yy) * sw) / len(Xf); g[1:] += l2 * w[1:] / len(Xf); w -= lr * g
    return w, mu_, sd_
def predL(m, Xf):
    w, mu_, sd_ = m
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(Xf)), (Xf - mu_) / sd_] @ w, -30, 30)))

mask = have_feat & ~np.isnan(ou)
ab = mask & (br < 0.85) & ~np.isnan(resid)
def feats(mk, with_jepa):
    cols = [p_arr[mk], br[mk], ou[mk],
            np.log1p(X[mk, 0]), np.log1p(X[mk, 1]), X[mk, 2]]
    if with_jepa:
        cols.append(-resid[mk])         # flip: low residual -> essential
        # also include the residual *vector* norm bits via a few PCs of Z_self
        # use first 4 dims of the target embedding for compactness
        for d in range(4):
            cols.append(Z_self[mk, d])
    return np.column_stack(cols)
def run(with_jepa):
    pred = np.full(len(p_arr), np.nan)
    for H in sorted(set(clade[ab])):
        tr = ab & (clade != H); te = ab & (clade == H)
        if tr.sum() < 200 or te.sum() < 5 or y[tr].sum() < 20: continue
        pred[te] = predL(fit(feats(tr, with_jepa), y[tr]), feats(te, with_jepa))
    done = ab & ~np.isnan(pred)
    s = pred[done]; yy = y[done].astype(int)
    o = np.argsort(-s); ys = yy[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); ok = prec >= 0.9
    ess_cov = float(tp[ok].max() / max(yy.sum(), 1)) if ok.any() else 0.0
    o2 = np.argsort(s); ys2 = (1-yy)[o2]; tn = np.cumsum(ys2); fn = np.cumsum(1-ys2)
    pre2 = tn / np.maximum(tn + fn, 1); ok2 = pre2 >= 0.9
    ne_cov = float(tn[ok2].max() / max((1-yy).sum(), 1)) if ok2.any() else 0.0
    return ess_cov, ne_cov, int(done.sum())

base_ess, base_ne, n_b = run(False)
jepa_ess, jepa_ne, n_j = run(True)
print(f"\n=== EVAL B: precision lift from JEPA features ===")
print(f"  baseline    essCov@P90 = {base_ess:.3f}   neCov@P90 = {base_ne:.3f}  (n={n_b})")
print(f"  +JEPA       essCov@P90 = {jepa_ess:.3f}   neCov@P90 = {jepa_ne:.3f}  (n={n_j})")
print(f"  delta       essCov     = {jepa_ess-base_ess:+.3f}   neCov     = {jepa_ne-base_ne:+.3f}")

# ============== figure ==============
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
# (a) loss curve
ax[0].plot(losses, "o-", color="#3182bd")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (latent space)")
ax[0].set_title("JEPA training curve\n(self-supervised; no essentiality labels)",
                fontweight="bold", fontsize=11)
ax[0].grid(alpha=0.3)
# (b) per-clade residual AUC
items = sorted(per_clade_auc.items(), key=lambda kv: -kv[1])
ax[1].barh(range(len(items)), [a for _, a in items],
            color=["#e31a1c" if a > 0.55 else "#888" for _, a in items])
ax[1].axvline(0.5, color="k", ls="--", lw=1)
ax[1].axvline(0.55, color="#e31a1c", ls=":", lw=1)
ax[1].set_yticks(range(len(items)))
ax[1].set_yticklabels([c[:24] for c, _ in items], fontsize=7.5)
ax[1].invert_yaxis(); ax[1].set_xlim(0.4, 0.85)
ax[1].set_xlabel("directional AUC: -residual vs essentiality (per clade)")
ax[1].set_title(f"EVAL A: zero-shot essentiality signal\n"
                f"mean AUC = {mean_auc:.3f}  | >0.55: {pos_above_055}/{len(per_clade_auc)}  "
                f"| >0.65: {pos_above_065}/{len(per_clade_auc)}",
                fontweight="bold", fontsize=11)
# (c) precision boost
labels = ["essCov@P90", "neCov@P90"]
xp = np.arange(2)
ax[2].bar(xp - 0.2, [base_ess, base_ne], 0.4, color="#9ecae1", label="baseline (6 feat)")
ax[2].bar(xp + 0.2, [jepa_ess, jepa_ne], 0.4, color="#e31a1c", label="+ JEPA (residual + 4 emb dims)")
for i, (b_, j_) in enumerate(zip([base_ess, base_ne], [jepa_ess, jepa_ne])):
    ax[2].text(i - 0.2, b_ + 0.01, f"{b_:.3f}", ha="center", fontsize=9)
    ax[2].text(i + 0.2, j_ + 0.01, f"{j_:.3f}", ha="center", fontsize=9, fontweight="bold")
ax[2].set_xticks(xp); ax[2].set_xticklabels(labels)
ax[2].set_ylim(0, max(base_ess, jepa_ess, base_ne, jepa_ne) + 0.1)
ax[2].set_title(f"EVAL B: precision lift on abstention bucket\n"
                f"essCov {base_ess:.3f} -> {jepa_ess:.3f}  ({(jepa_ess-base_ess)*100:+.1f}pp)",
                fontweight="bold", fontsize=11)
ax[2].legend(fontsize=9); ax[2].grid(axis="y", alpha=0.3)

fig.suptitle("JEPA-lite on the cell: self-supervised gene-context prediction "
             "in latent space", fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT / "jepa.png", dpi=120, bbox_inches="tight")
print(f"\nwrote outputs/orphan/jepa.png")

json.dump(dict(arch=dict(d_in=D_IN, d_hid=D_HID, d_emb=D_EMB,
                          epochs=EPOCHS, lr=LR, tau=TAU, bs=BS),
               n_train=int(train_mask.sum()),
               training_loss=losses,
               eval_A_residual_AUC=dict(mean=round(mean_auc, 3),
                                        per_clade={c: round(a, 3) for c, a in per_clade_auc.items()},
                                        n_clades_above_055=pos_above_055,
                                        n_clades_above_065=pos_above_065),
               eval_B_precision=dict(baseline_essCov=round(base_ess, 3),
                                     jepa_essCov=round(jepa_ess, 3),
                                     baseline_neCov=round(base_ne, 3),
                                     jepa_neCov=round(jepa_ne, 3),
                                     ess_delta_pp=round((jepa_ess-base_ess)*100, 1))),
          open(OUT / "jepa.json", "w"), indent=2)
print("wrote outputs/orphan/jepa.json")
