"""af_torch2 -- an improved essentiality model, not a port.

Two changes over af_torch:

  1. SECOND TRACK over the phyletic profile. The MSA track only contains
     orthologs that are PRESENT, so the original model cannot see absence. We add
     a track that attends over the full per-organism profile of the focal gene's
     OG -- [present, ess_known, ess_label(masked), same_clade] for EVERY organism
     -- so backup / co-occurrence logic is learned end-to-end instead of being
     collapsed into 4 hand-built scalars. Held-clade essentiality labels are
     masked in BOTH tracks (presence is genome content, not a label, so it stays).

  2. A real training algorithm: focal loss (essential-tail asymmetry), AdamW +
     cosine schedule with warmup, dropout, early stopping on a held-out val
     clade, and optional seed-ensembling.

Leave-one-clade-out, --all_clades to score every clade with enough genes.
Reads the (DEG-augmented) cache + the matching labels dir for the profile track.

  python colab/af_torch2.py --cache outputs/orphan/af_msa_cache_aug.npz \\
      --labels_dir data/drive_import/labels_aug --tag _v2 --all_clades --seeds 3
"""
import json, time, argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from af_common import Cache, CACHE, OUT, MAJOR_CLADES, auc, auprc, cov_at_p
from smart_cooccur import load_orthology


@dataclass
class Cfg2:
    heads: int = 4
    head_dim: int = 16
    hid: int = 64
    blocks: int = 2
    dropout: float = 0.1
    epochs: int = 15
    bs: int = 8192
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    focal_gamma: float = 1.5
    patience: int = 3            # early-stop patience (epochs)
    val_frac: float = 0.08       # fraction of train genes held for val
    seeds: int = 1
    simple: bool = False         # use v1's proven recipe (isolates the 2nd track)
    loss: str = "bce"            # "bce" or "rank" (#4: pairwise ranking surrogate)
    cal_frac: float = 0.05       # held-out split to calibrate the confidence head
    profile_mask: str = "own_clade"   # "own_clade" (strict) or "own_org" (match v1's budget)
    no_same_clade: bool = False  # zero the same_clade profile column (identity-signal ablation)
    prof_rows: int = 0           # set at runtime = n organisms
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Attn(nn.Module):
    """Focal-query attention over a set of rows (used for both tracks)."""
    def __init__(self, df, dm, heads, hd):
        super().__init__()
        self.h, self.hd = heads, hd
        self.Wq = nn.Linear(df, heads * hd, bias=False)
        self.Wk = nn.Linear(dm, heads * hd, bias=False)
        self.Wv = nn.Linear(dm, heads * hd, bias=False)

    def forward(self, foc, rows, mask):
        B = foc.shape[0]
        q = self.Wq(foc).view(B, self.h, self.hd)
        k = self.Wk(rows).view(B, -1, self.h, self.hd)
        v = self.Wv(rows).view(B, -1, self.h, self.hd)
        s = torch.einsum("bkhd,bhd->bhk", k, q) / (self.hd ** 0.5)
        s = s.masked_fill(mask[:, None, :] <= 0, -1e9)
        a = torch.softmax(s, dim=-1)
        return torch.einsum("bhk,bkhd->bhd", a, v).reshape(B, -1)


class Model2(nn.Module):
    def __init__(self, df, dm, dp, cfg: Cfg2):
        super().__init__()
        self.msa_blocks = nn.ModuleList(
            [Attn(df, dm, cfg.heads, cfg.head_dim) for _ in range(cfg.blocks)])
        self.prof_block = Attn(df, dp, cfg.heads, cfg.head_dim)   # the new track
        ctx = cfg.heads * cfg.head_dim
        self.W1 = nn.Linear(df + 2 * ctx, cfg.hid)
        self.drop = nn.Dropout(cfg.dropout)
        self.head_ess = nn.Linear(cfg.hid, 1)
        self.head_conf = nn.Linear(cfg.hid, 1)

    def forward(self, foc, msa, msa_mask, prof, prof_mask):
        c1 = self.msa_blocks[0](foc, msa, msa_mask)
        for blk in self.msa_blocks[1:]:
            c1 = c1 + blk(foc, msa, msa_mask)
        c2 = self.prof_block(foc, prof, prof_mask)
        z = torch.cat([foc, c1, c2], dim=1)
        h = self.drop(F.relu(self.W1(z)))
        return self.head_ess(h).squeeze(-1), self.head_conf(h).squeeze(-1)


def focal_bce(logit, y, w, gamma):
    p = torch.sigmoid(logit)
    pt = torch.where(y == 1, p, 1 - p)
    return (w * (1 - pt).clamp_min(1e-6) ** gamma *
            F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()


def rank_loss(logit, y, max_pairs=200000):
    """Sampled pairwise ranking surrogate (differentiable AUC): push essential
    logits above non-essential ones. Optimises the ranking/coverage objective
    directly instead of calibration. Combined with a little BCE so probabilities
    stay usable. Item #4 -- enabled with --loss rank."""
    pos = logit[y == 1]; neg = logit[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return F.binary_cross_entropy_with_logits(logit, y)
    npair = min(max_pairs, len(pos) * len(neg))
    pi = torch.randint(len(pos), (npair,), device=logit.device)
    ni = torch.randint(len(neg), (npair,), device=logit.device)
    margin = pos[pi] - neg[ni]
    return F.softplus(-margin).mean()


class Trainer2:
    def __init__(self, cache: Cache, labels_dir, cfg: Cfg2):
        self.c, self.cfg = cache, cfg
        d = cfg.device
        self.foc = torch.tensor(cache.foc, device=d)
        self.msa = torch.tensor(cache.msa, device=d)
        self.msa_mask = torch.tensor(cache.msa_mask, device=d)
        self.msa_org = cache.msa_org
        self.y = torch.tensor(cache.y, device=d)

        # ---- build the phyletic-profile inputs from orthology ----
        orth = load_orthology(labels_dir=labels_dir)
        Y, P = orth["Y"], orth["P"]; O = orth["O"]; org_idx = orth["org_idx"]; og_idx = orth["og_idx"]
        cfg.prof_rows = O
        # cache org name -> orthology org index; cache gene -> (og_j, org_i)
        c2o = np.full(len(cache.orgs), -1, np.int64)
        for ci, nm in enumerate(cache.orgs):
            c2o[ci] = org_idx.get(str(nm), -1)
        gene_j = np.full(cache.N, -1, np.int64)
        for i in range(cache.N):
            og = orth["gene_og"].get((str(cache.orgs[cache.meta_org[i]]), str(cache.lt[i])))
            gene_j[i] = og_idx.get(og, -1) if og is not None else -1
        # orthology org -> clade (via cache org->clade), for same-clade + leak mask
        orth_clade = np.array(["?"] * O, dtype=object)
        for ci, nm in enumerate(cache.orgs):
            oi = c2o[ci]
            if oi >= 0:
                orth_clade[oi] = cache.org_clade.get(ci, "?")
        self.Y = Y; self.P = P; self.O = O
        self.gene_j = gene_j
        self.gene_orthorg = c2o[cache.meta_org]      # focal org in orthology space (-1 if absent)
        self.gene_focal_clade = cache.clade
        self.orth_clade = orth_clade
        # precompute essentiality-known mask once
        self.Yknown = (~np.isnan(Y)).astype(np.float32)
        self.Yfill = np.nan_to_num(Y, nan=0.0).astype(np.float32)

    def build_profile(self, idx, held_clade):
        """[len(idx), O, 4] = present, ess_known, ess_label, same_clade.
        LEAK CONTROL: mask essentiality labels for (a) the held clade AND
        (b) each focal gene's OWN clade -- so a training gene never sees its own
        label or its same-clade neighbours' labels (mirrors the LOCO test
        condition, where the whole focal clade is masked). Presence is genome
        content, not a label, so it stays."""
        j = self.gene_j[idx]
        valid = j >= 0
        jj = np.where(valid, j, 0)
        present = self.P[:, jj].T                       # [B,O]
        known = self.Yknown[:, jj].T.copy()
        label = self.Yfill[:, jj].T.copy()
        same = (self.orth_clade[None, :] == self.gene_focal_clade[idx][:, None])  # [B,O] bool
        held = (self.orth_clade == held_clade)          # [O]
        if self.cfg.profile_mask == "own_org":
            # match v1's budget: hide only the focal org's own label (+ held clade),
            # leaving same-clade siblings visible during training as the MSA does.
            own = np.zeros_like(same)
            oo = self.gene_orthorg[idx]
            valid_o = oo >= 0
            own[np.where(valid_o)[0], oo[valid_o]] = True
            maskcol = own | held[None, :]
        else:                                           # "own_clade" (strict, default)
            maskcol = same | held[None, :]
        known[maskcol] = 0.0
        label[maskcol] = 0.0
        same_feat = np.zeros_like(same, np.float32) if self.cfg.no_same_clade else same.astype(np.float32)
        prof = np.stack([present, known, label, same_feat], axis=-1).astype(np.float32)
        prof[~valid] = 0.0
        mask = np.ones((len(idx), self.O), np.float32)
        mask[~valid] = 0.0
        return prof, mask

    def selftest(self, n=200, seed=0):
        """Leak assertion: for sampled TRAINING genes (held_clade = some OTHER
        clade), the gene's own-clade essentiality labels must be fully masked in
        its profile. Catches the v2 self-leak class of bug. Returns True/raises."""
        rng = np.random.default_rng(seed)
        clades = sorted(set(map(str, self.gene_focal_clade)))
        strict = self.cfg.profile_mask == "own_clade"
        ok = 0
        for i in rng.choice(self.c.N, min(n, self.c.N), replace=False):
            i = int(i)
            focal = str(self.gene_focal_clade[i])
            other = next(c for c in clades if c != focal)   # simulate a TRAIN fold
            prof, _ = self.build_profile(np.array([i]), other)
            # universal guard: the focal gene's OWN organism label must be masked
            oo = int(self.gene_orthorg[i])
            if oo >= 0 and (prof[0, oo, 1] > 0 or prof[0, oo, 2] > 0):
                raise AssertionError(f"SELF-LEAK: gene {i} ({focal}) sees its own org label "
                                     f"when held={other}")
            # strict mode also masks the whole own clade
            if strict:
                own = (self.orth_clade == focal)
                if prof[0, own, 1].max() > 0 or prof[0, own, 2].max() > 0:
                    raise AssertionError(f"CLADE-LEAK: gene {i} ({focal}) sees own-clade "
                                         f"labels when held={other}")
            ok += 1
        mode = "own-clade" if strict else "own-org"
        print(f"  leak selftest PASSED on {ok} training genes ({mode} masking)")
        return True

    def _fit_one(self, train_idx, held_clade, seed):
        cfg = self.cfg; d = cfg.device
        torch.manual_seed(seed); np.random.seed(seed)
        model = Model2(self.c.Df, self.c.Dm, 4, cfg).to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # val split (early stopping) from within the training set.
        # --simple = v1's recipe: no focal, no early stop, constant lr, train all.
        do_val = (cfg.val_frac > 0) and not cfg.simple
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx)
        # cal split: held out from ess-head training, used to calibrate the conf
        # head on HONEST (not memorised) correctness.
        n_cal = int(len(perm) * cfg.cal_frac)
        cal_idx = perm[:n_cal]; rest = perm[n_cal:]
        n_val = int(len(rest) * cfg.val_frac) if do_val else 0
        val_idx, tr_idx = rest[:n_val], rest[n_val:]
        pos = float(self.y[tr_idx].mean().clamp_min(1e-6))
        wpos, wneg = 0.5 / pos, 0.5 / (1 - pos)
        steps = cfg.epochs * max(1, len(tr_idx) // cfg.bs)
        warm = max(1, int(steps * cfg.warmup_frac)); t = 0
        gamma = 0.0 if cfg.simple else cfg.focal_gamma

        def lr_at(step):
            if cfg.simple:
                return cfg.lr
            if step < warm:
                return cfg.lr * step / warm
            pr = (step - warm) / max(1, steps - warm)
            return 0.5 * cfg.lr * (1 + np.cos(np.pi * pr))

        # cache masked MSA for this fold once
        msa_m = self.c.masked_msa(held_clade)
        msa_m = torch.tensor(msa_m, device=d)
        best_val = -1; best_state = None; bad = 0
        for ep in range(cfg.epochs):
            model.train()
            order = tr_idx[rng.permutation(len(tr_idx))]
            for i in range(0, len(order), cfg.bs):
                bi = order[i:i + cfg.bs]
                for g in opt.param_groups:
                    g["lr"] = lr_at(t)
                t += 1
                prof, pmask = self.build_profile(bi, held_clade)
                prof = torch.tensor(prof, device=d); pmask = torch.tensor(pmask, device=d)
                bt = torch.as_tensor(bi, device=d)
                lo, clo = model(self.foc[bt], msa_m[bt], self.msa_mask[bt], prof, pmask)
                yb = self.y[bt]
                if cfg.loss == "rank":
                    loss = rank_loss(lo, yb) + 0.2 * F.binary_cross_entropy_with_logits(lo, yb)
                else:
                    w = torch.where(yb == 1, torch.tensor(wpos, device=d), torch.tensor(wneg, device=d))
                    loss = focal_bce(lo, yb, w, gamma)
                with torch.no_grad():
                    corr = ((torch.sigmoid(lo) > 0.5).float() == yb).float()
                loss = loss + F.binary_cross_entropy_with_logits(clo, corr)
                opt.zero_grad(); loss.backward(); opt.step()
            if not do_val:
                continue                       # simple recipe: train all epochs, keep final
            vp, _ = self._predict(model, val_idx, msa_m, held_clade)
            va = auc(vp, self.c.y[val_idx])
            if va > best_val + 1e-4:
                best_val = va; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- recalibrate the confidence head on HELD-OUT correctness ----
        # Freeze everything but head_conf; train it to predict whether the (now
        # frozen) essentiality head is right on cal_idx -- honest correctness the
        # model did not train its ess head on. Fixes the brightness saturation.
        if len(cal_idx) > 200:
            with torch.no_grad():
                cp, _ = self._predict(model, cal_idx, msa_m, held_clade)
            corr_cal = torch.tensor(((cp > 0.5).astype(np.float32) ==
                                     self.c.y[cal_idx]).astype(np.float32), device=d)
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.head_conf.parameters():
                p.requires_grad_(True)
            optc = torch.optim.Adam(model.head_conf.parameters(), lr=1e-2)
            for _ in range(60):
                order = rng.permutation(len(cal_idx))
                for i in range(0, len(order), cfg.bs):
                    sub = cal_idx[order[i:i + cfg.bs]]
                    prof, pmask = self.build_profile(sub, held_clade)
                    prof = torch.tensor(prof, device=d); pmask = torch.tensor(pmask, device=d)
                    bt = torch.as_tensor(sub, device=d)
                    model.eval()
                    _, clo = model(self.foc[bt], msa_m[bt], self.msa_mask[bt], prof, pmask)
                    tgt = corr_cal[order[i:i + cfg.bs]]
                    lc = F.binary_cross_entropy_with_logits(clo, tgt)
                    optc.zero_grad(); lc.backward(); optc.step()
            for p in model.parameters():
                p.requires_grad_(True)
        return model, msa_m

    @torch.no_grad()
    def _predict(self, model, idx, msa_m, held_clade):
        model.eval(); d = self.cfg.device
        out_p = np.empty(len(idx), np.float32); out_c = np.empty(len(idx), np.float32)
        for i in range(0, len(idx), 16384):
            bi = idx[i:i + 16384]
            prof, pmask = self.build_profile(bi, held_clade)
            prof = torch.tensor(prof, device=d); pmask = torch.tensor(pmask, device=d)
            bt = torch.as_tensor(bi, device=d)
            lo, clo = model(self.foc[bt], msa_m[bt], self.msa_mask[bt], prof, pmask)
            out_p[i:i + len(bi)] = torch.sigmoid(lo).cpu().numpy()
            out_c[i:i + len(bi)] = torch.sigmoid(clo).cpu().numpy()
        return out_p, out_c

    def fit_predict(self, held_clade, test_idx):
        train_idx = np.where(self.c.clade != held_clade)[0]
        ps, cs = [], []
        for s in range(self.cfg.seeds):
            model, msa_m = self._fit_one(train_idx, held_clade, seed=s)
            p, c = self._predict(model, test_idx, msa_m, held_clade)
            ps.append(p); cs.append(c)
        return np.mean(ps, 0), np.mean(cs, 0)


def run(cfg: Cfg2, cache_path, labels_dir, tag="", eval_clades=None, min_clade_genes=2000):
    c = Cache(cache_path or CACHE)
    tr = Trainer2(c, labels_dir, cfg)
    tr.selftest()                      # fail fast if the profile track leaks
    if eval_clades == "all":
        uniq, cnt = np.unique(c.clade, return_counts=True)
        eval_clades = sorted(str(u) for u, n in zip(uniq, cnt) if n >= min_clade_genes)
        print(f"auto-selected {len(eval_clades)} clades >= {min_clade_genes} genes")
    elif eval_clades is None:
        eval_clades = MAJOR_CLADES
    all_p = np.full(c.N, np.nan, np.float32); all_c = np.full(c.N, np.nan, np.float32)
    per = {}; t0 = time.time()
    for cl in eval_clades:
        test_idx = np.where(c.clade == cl)[0]
        if len(test_idx) < 100:
            continue
        p, br = tr.fit_predict(cl, test_idx)
        all_p[test_idx] = p; all_c[test_idx] = br
        yt = c.y[test_idx]
        per[cl] = dict(n=int(len(test_idx)), auc=round(auc(p, yt), 3),
                       auprc=round(auprc(p, yt), 3),
                       ne_cov_p90=round(cov_at_p(1 - p, 1 - yt, 0.9), 3))
        print(f"{cl:<22} n={len(test_idx):<6} AUC={per[cl]['auc']:.3f} "
              f"neCov@P90={per[cl]['ne_cov_p90']:.2f}")
    ev = ~np.isnan(all_p); pe = all_p[ev]; ye = c.y[ev]; ce = all_c[ev]
    pooled = dict(n=int(ev.sum()), auc=round(auc(pe, ye), 3), auprc=round(auprc(pe, ye), 3),
                  ne_cov_p90=round(cov_at_p(1 - pe, 1 - ye, 0.9), 3),
                  brightness85_coverage=round(float((ce >= 0.85).mean()), 3),
                  brightness85_accuracy=round(float(((pe[ce >= 0.85] > 0.5).astype(int)
                                                     == ye[ce >= 0.85]).mean()), 3) if (ce >= 0.85).any() else 0.0)
    dt = time.time() - t0
    print(f"\nPOOLED AUC={pooled['auc']} neCov@P90={pooled['ne_cov_p90']} | "
          f"brightness>=0.85 cov={pooled['brightness85_coverage']} @ {pooled['brightness85_accuracy']}  ({dt:.0f}s)")
    res = dict(config=cfg.__dict__, per_clade=per, pooled=pooled, seconds=round(dt, 1))
    json.dump(res, open(OUT / f"af_torch2_results{tag}.json", "w"), indent=2)
    np.savez(OUT / f"af_torch_preds{tag}.npz", p=all_p, brightness=all_c,
             y=c.y, clade=c.clade, meta_org=c.meta_org, lt=c.lt)
    print(f"wrote af_torch2_results{tag}.json + af_torch_preds{tag}.npz")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--labels_dir", default="data/drive_import/labels")
    ap.add_argument("--tag", default="_v2")
    ap.add_argument("--all_clades", action="store_true")
    ap.add_argument("--min_clade_genes", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--simple", action="store_true",
                    help="v1's recipe (no focal/early-stop/cosine, dropout=0) to isolate the 2nd track")
    ap.add_argument("--loss", default="bce", choices=["bce", "rank"],
                    help="rank = pairwise ranking surrogate (#4)")
    ap.add_argument("--profile_mask", default="own_clade", choices=["own_clade", "own_org"],
                    help="own_org matches v1's information budget for a fair v1-vs-v2 test")
    ap.add_argument("--no_same_clade", action="store_true",
                    help="zero the same_clade profile column (identity-signal ablation)")
    ap.add_argument("--selftest", action="store_true", help="run the leak assertion and exit")
    a = ap.parse_args()
    cfg = Cfg2(seeds=a.seeds, epochs=a.epochs, simple=a.simple, loss=a.loss,
               profile_mask=a.profile_mask, no_same_clade=a.no_same_clade)
    if a.simple:
        cfg.dropout = 0.0          # match v1
    if a.selftest:
        c = Cache(a.cache or CACHE); Trainer2(c, a.labels_dir, cfg).selftest(); raise SystemExit
    print(f"device={cfg.device} cfg={cfg}")
    run(cfg, a.cache, a.labels_dir, tag=a.tag,
        eval_clades="all" if a.all_clades else None, min_clade_genes=a.min_clade_genes)
