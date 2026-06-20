"""PyTorch (A100-ready) port of the AlphaFold-style essentiality model.

Faithful to scripts/af_train.py (single-head MSA row-attention + pLDDT-style
confidence head), but: (a) runs on GPU, (b) optionally scales to multi-head /
multi-block for the A100, (c) exposes fit / predict / masked-MSA helpers that
the active-learning loop and the cooccur stacker reuse.

Run standalone:
    python colab/af_torch.py            # parity config, leave-one-clade-out
    python colab/af_torch.py --big      # multi-head, deeper (uses the A100)
"""
import json, time, argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from af_common import Cache, OUT, MAJOR_CLADES, auc, auprc, cov_at_p


@dataclass
class AFConfig:
    heads: int = 1          # parity = 1; --big bumps this
    head_dim: int = 12      # H in the numpy model
    hid: int = 24           # readout MLP width
    blocks: int = 1         # stacked attention blocks
    epochs: int = 8
    bs: int = 4096          # larger batch -> better A100 utilisation
    lr: float = 3e-3
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AttnBlock(nn.Module):
    """One MSA row-attention block: focal-query attends over ortholog rows."""
    def __init__(self, df, dm, heads, head_dim):
        super().__init__()
        self.h, self.hd = heads, head_dim
        self.Wq = nn.Linear(df, heads * head_dim, bias=False)
        self.Wk = nn.Linear(dm, heads * head_dim, bias=False)
        self.Wv = nn.Linear(dm, heads * head_dim, bias=False)

    def forward(self, foc, msa, mask):
        B = foc.shape[0]
        q = self.Wq(foc).view(B, self.h, self.hd)               # B,h,hd
        k = self.Wk(msa).view(B, -1, self.h, self.hd)           # B,K,h,hd
        v = self.Wv(msa).view(B, -1, self.h, self.hd)           # B,K,h,hd
        s = torch.einsum("bkhd,bhd->bhk", k, q) / (self.hd ** 0.5)  # B,h,K
        s = s.masked_fill(mask[:, None, :] <= 0, -1e9)
        a = torch.softmax(s, dim=-1)                            # B,h,K
        ctx = torch.einsum("bhk,bkhd->bhd", a, v).reshape(B, -1)  # B,h*hd
        return ctx


class AFModel(nn.Module):
    def __init__(self, df, dm, cfg: AFConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttnBlock(df, dm, cfg.heads, cfg.head_dim) for _ in range(cfg.blocks)])
        ctx_dim = cfg.heads * cfg.head_dim
        self.W1 = nn.Linear(df + ctx_dim, cfg.hid)
        self.head_ess  = nn.Linear(cfg.hid, 1)   # essentiality logit
        self.head_conf = nn.Linear(cfg.hid, 1)   # brightness logit

    def forward(self, foc, msa, mask):
        ctx = self.blocks[0](foc, msa, mask)
        for blk in self.blocks[1:]:
            ctx = ctx + blk(foc, msa, mask)
        z = torch.cat([foc, ctx], dim=1)
        hid = F.relu(self.W1(z))
        return self.head_ess(hid).squeeze(-1), self.head_conf(hid).squeeze(-1)


class Trainer:
    """Holds cache tensors on device; provides fit / predict / masked-MSA."""
    def __init__(self, cache: Cache, cfg: AFConfig):
        self.c, self.cfg = cache, cfg
        d = cfg.device
        self.foc  = torch.tensor(cache.foc, device=d)
        self.mask = torch.tensor(cache.msa_mask, device=d)
        self.y    = torch.tensor(cache.y, device=d)
        self.msa_org = cache.msa_org  # keep on cpu for masking logic

    def masked_msa(self, held_clade, reveal_idx=None):
        """MSA with held-clade label/known zeroed, EXCEPT rows belonging to
        revealed genes (active learning), which stay visible."""
        m = self.c.masked_msa(held_clade)
        if reveal_idx is not None and len(reveal_idx):
            # un-mask ortholog rows that correspond to the revealed focal genes
            reveal_orgs = set(int(self.c.meta_org[i]) for i in reveal_idx)
            reveal_lts  = set(str(self.c.lt[i]) for i in reveal_idx)
            # cheap heuristic: un-mask rows whose org is a revealed org AND row label_known
            # was originally set; we restore from the unmasked cache for those orgs.
            orig = self.c.msa
            restore = np.isin(self.msa_org, list(reveal_orgs))
            m[..., 0] = np.where(restore, orig[..., 0], m[..., 0])
            m[..., 1] = np.where(restore, orig[..., 1], m[..., 1])
        return torch.tensor(m, device=self.cfg.device)

    def fit(self, train_idx, msa_t, epochs=None, warm=None, verbose=False):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        model = warm if warm is not None else AFModel(self.c.Df, self.c.Dm, cfg).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        train_idx = np.asarray(train_idx)
        pos = float(self.y[train_idx].mean().clamp_min(1e-6))
        wpos, wneg = 0.5 / pos, 0.5 / (1 - pos)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        E = epochs or cfg.epochs
        for ep in range(E):
            perm = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
            for i in range(0, len(perm), cfg.bs):
                bi = perm[i:i + cfg.bs]
                bt = torch.as_tensor(bi, device=cfg.device)
                lo, clo = model(self.foc[bt], msa_t[bt], self.mask[bt])
                yb = self.y[bt]
                w = torch.where(yb == 1, torch.tensor(wpos, device=cfg.device),
                                torch.tensor(wneg, device=cfg.device))
                loss_e = F.binary_cross_entropy_with_logits(lo, yb, weight=w)
                with torch.no_grad():
                    correct = ((torch.sigmoid(lo) > 0.5).float() == yb).float()
                loss_c = F.binary_cross_entropy_with_logits(clo, correct)
                opt.zero_grad(); (loss_e + loss_c).backward(); opt.step()
            if verbose:
                print(f"    epoch {ep+1}/{E} loss_e={loss_e.item():.3f}")
        return model

    @torch.no_grad()
    def predict(self, model, idx, msa_t):
        idx = np.asarray(idx)
        out_p = np.empty(len(idx), np.float32); out_c = np.empty(len(idx), np.float32)
        for i in range(0, len(idx), 16384):
            bi = idx[i:i + 16384]
            bt = torch.as_tensor(bi, device=self.cfg.device)
            lo, clo = model(self.foc[bt], msa_t[bt], self.mask[bt])
            out_p[i:i + len(bi)] = torch.sigmoid(lo).cpu().numpy()
            out_c[i:i + len(bi)] = torch.sigmoid(clo).cpu().numpy()
        return out_p, out_c


def run_loco(cfg: AFConfig, save=True, cache_path=None, tag=""):
    """Leave-one-clade-out over the 5 major clades. Returns per-gene p,
    brightness (nan where unevaluated) and the metrics dict.

    cache_path: alternate af_msa_cache npz (e.g. the DEG-augmented one).
    tag:        suffix for the output files (af_torch_results{tag}.json etc.)."""
    from af_common import CACHE
    c = Cache(cache_path or CACHE)
    tr = Trainer(c, cfg)
    N = c.N
    all_p = np.full(N, np.nan, np.float32); all_c = np.full(N, np.nan, np.float32)
    per_clade = {}
    t0 = time.time()
    for cl in MAJOR_CLADES:
        test_idx = np.where(c.clade == cl)[0]
        if len(test_idx) < 100:
            continue
        train_idx = np.where(c.clade != cl)[0]
        msa_t = tr.masked_msa(cl)
        model = tr.fit(train_idx, msa_t)
        p, br = tr.predict(model, test_idx, msa_t)
        all_p[test_idx] = p; all_c[test_idx] = br
        yt = c.y[test_idx]
        per_clade[cl] = dict(
            n=int(len(test_idx)), base_ess=round(float(yt.mean()), 3),
            auc=round(auc(p, yt), 3), auprc=round(auprc(p, yt), 3),
            ne_cov_p90=round(cov_at_p(1 - p, 1 - yt, 0.9), 3),
            ess_cov_p90=round(cov_at_p(p, yt, 0.9), 3),
            ess_cov_p70=round(cov_at_p(p, yt, 0.7), 3))
        print(f"{cl:<12} n={len(test_idx):<6} AUC={per_clade[cl]['auc']:.3f} "
              f"AUPRC={per_clade[cl]['auprc']:.3f} neCov@P90={per_clade[cl]['ne_cov_p90']:.2f}")
    ev = ~np.isnan(all_p); pe = all_p[ev]; ye = c.y[ev]; ce = all_c[ev]
    pooled = dict(n=int(ev.sum()), auc=round(auc(pe, ye), 3), auprc=round(auprc(pe, ye), 3),
                  ne_cov_p90=round(cov_at_p(1 - pe, 1 - ye, 0.9), 3),
                  ess_cov_p90=round(cov_at_p(pe, ye, 0.9), 3),
                  ess_cov_p70=round(cov_at_p(pe, ye, 0.7), 3))
    # brightness-thresholded two-sided coverage (the deployment number)
    hi = ce >= 0.85
    bright_cov = float(hi.mean())
    bright_acc = float(((pe[hi] > 0.5).astype(int) == ye[hi]).mean()) if hi.any() else 0.0
    pooled["brightness85_coverage"] = round(bright_cov, 3)
    pooled["brightness85_accuracy"] = round(bright_acc, 3)
    dt = time.time() - t0
    print(f"\nPOOLED AUC={pooled['auc']} | brightness>=0.85 coverage={bright_cov:.3f} @ {bright_acc:.3f}  ({dt:.0f}s)")
    res = dict(config=cfg.__dict__, per_clade=per_clade, pooled=pooled, seconds=round(dt, 1))
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT / f"af_torch_results{tag}.json", "w"), indent=2)
        np.savez(OUT / f"af_torch_preds{tag}.npz", p=all_p, brightness=all_c,
                 y=c.y, clade=c.clade, meta_org=c.meta_org, lt=c.lt)
        print(f"wrote af_torch_results{tag}.json + af_torch_preds{tag}.npz")
    return all_p, all_c, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="multi-head, deeper (uses the A100)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cache", default=None, help="alternate af_msa_cache npz")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _aug)")
    a = ap.parse_args()
    cfg = AFConfig()
    if a.big:
        cfg = AFConfig(heads=4, head_dim=16, hid=64, blocks=2, epochs=12, bs=8192)
    if a.epochs:
        cfg.epochs = a.epochs
    print(f"device={cfg.device} config={cfg}")
    run_loco(cfg, cache_path=a.cache, tag=a.tag)
