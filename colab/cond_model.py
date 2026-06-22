"""Stage 2b -- the conditional essentiality model.

Predicts P(essential | gene, condition) and continuous fitness, by combining:
  GENE branch     : focal features + MSA row-attention (same as the v2 core) -> gene_vec
  CONDITION branch: compositional multi-hot condition features -> cond_vec
  HEAD            : [gene_vec, cond_vec, gene_vec*cond_proj] -> fit (regression)
                    + essential (classification). Multi-task: MSE(fit)+BCE(ess).

Three evaluation protocols (the scientific contribution is doing all three):
  - leave-CONDITION-out : hold out a condition CLUSTER entirely (cross-condition
                          generalisation: predict a stress never seen in train)
  - leave-CLADE-out     : hold out a phylogenetic clade (cross-organism)
  - leave-BOTH-out      : hold out (clade x condition) -- the hardest, the real
                          deployment ask

Reported per fold: Spearman(fit_pred, fit_true), AUC(essential), and rho as a
FRACTION OF THE PER-ORG NOISE FLOOR (the honest ceiling-relative metric).

  python colab/cond_model.py --epochs 12 --eval condition
"""
import json, argparse, time
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from af_common import Cache, OUT, auc


@dataclass
class CondCfg:
    H: int = 32
    cond_dim: int = 24
    hid: int = 64
    epochs: int = 12
    bs: int = 16384
    lr: float = 2e-3
    wd: float = 1e-4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CondModel(nn.Module):
    def __init__(self, df, dm, n_comp, cfg):
        super().__init__()
        self.Wq = nn.Linear(df, cfg.H, bias=False)
        self.Wk = nn.Linear(dm, cfg.H, bias=False)
        self.Wv = nn.Linear(dm, cfg.H, bias=False)
        self.gene_proj = nn.Linear(df + cfg.H, cfg.H)
        self.cond_enc = nn.Sequential(nn.Linear(n_comp, cfg.cond_dim), nn.ReLU(),
                                      nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.cond_gate = nn.Linear(cfg.cond_dim, cfg.H)   # for the interaction term
        din = cfg.H + cfg.cond_dim + cfg.H
        self.mlp = nn.Sequential(nn.Linear(din, cfg.hid), nn.ReLU(), nn.Dropout(0.1))
        self.head_fit = nn.Linear(cfg.hid, 1)
        self.head_ess = nn.Linear(cfg.hid, 1)

    def gene_vec(self, foc, msa, mask):
        q = self.Wq(foc); k = self.Wk(msa); v = self.Wv(msa)
        s = torch.einsum("bkh,bh->bk", k, q) / (k.shape[-1] ** 0.5)
        s = s.masked_fill(mask <= 0, -1e9)
        a = torch.softmax(s, -1)
        ctx = torch.einsum("bk,bkh->bh", a, v)
        return F.relu(self.gene_proj(torch.cat([foc, ctx], -1)))

    def forward(self, foc, msa, mask, cond):
        g = self.gene_vec(foc, msa, mask)
        c = self.cond_enc(cond)
        inter = g * torch.tanh(self.cond_gate(c))          # gene x condition interaction
        z = self.mlp(torch.cat([g, c, inter], -1))
        return self.head_fit(z).squeeze(-1), self.head_ess(z).squeeze(-1)


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = (ra.std() * rb.std())
    return float((ra * rb).mean() / d) if d > 0 else float("nan")


def main(cache_path, cond_dir, eval_mode, cfg):
    d = cfg.device
    cache = Cache(cache_path)
    Z = np.load(f"{cond_dir}/cond_pairs.npz", allow_pickle=True)
    meta = json.load(open(f"{cond_dir}/cond_meta.json"))
    gene_idx = Z["gene_idx"]; cond_feat = Z["cond_feat"]
    y_fit = Z["y_fit"]; y_ess = Z["y_ess"].astype(np.float32)
    clade = Z["clade"]; org = Z["org"]; cluster_id = Z["cluster_id"]
    noise = meta["noise_floor"]
    print(f"pairs={len(gene_idx):,}  genes={len(set(gene_idx.tolist())):,}  "
          f"clusters={meta['n_clusters']}  orgs={meta['n_orgs']}  ess={meta['ess_rate']}")

    foc = torch.tensor(cache.foc, device=d)
    msa = torch.tensor(cache.msa, device=d)
    mask = torch.tensor(cache.msa_mask, device=d)
    cf = torch.tensor(cond_feat, device=d)
    gi = torch.tensor(gene_idx.astype(np.int64), device=d)
    yf = torch.tensor(y_fit, device=d); ye = torch.tensor(y_ess, device=d)

    # ---- folds ----
    if eval_mode == "condition":
        keys = cluster_id; uniq = np.unique(keys)
        rng = np.random.default_rng(0); rng.shuffle(uniq)
        folds = [("cond_fold%d" % i, np.isin(keys, uniq[i::5])) for i in range(5)]
    elif eval_mode == "clade":
        uniq = sorted(set(clade.tolist()))
        folds = [(c, clade == c) for c in uniq]
    else:  # both
        uniq_c = sorted(set(clade.tolist()))
        rng = np.random.default_rng(0); cl_ids = np.unique(cluster_id)
        held_clu = set(rng.choice(cl_ids, max(1, len(cl_ids) // 5), replace=False).tolist())
        folds = [("clade=%s+condhold" % c,
                  (clade == c) & np.isin(cluster_id, list(held_clu))) for c in uniq_c]

    results = []
    t0 = time.time()
    for name, test_mask in folds:
        tr = np.where(~test_mask)[0]; te = np.where(test_mask)[0]
        if len(te) < 50 or len(tr) < 1000:
            continue
        torch.manual_seed(cfg.seed)
        model = CondModel(cache.Df, cache.Dm, cond_feat.shape[1], cfg).to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        g = torch.Generator().manual_seed(cfg.seed)
        for ep in range(cfg.epochs):
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            for i in range(0, len(perm), cfg.bs):
                b = perm[i:i + cfg.bs]; bt = torch.as_tensor(b, device=d)
                gb = gi[bt]
                fp, ep_ = model(foc[gb], msa[gb], mask[gb], cf[bt])
                loss = F.mse_loss(fp, yf[bt]) + F.binary_cross_entropy_with_logits(ep_, ye[bt])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            fp_all, ep_all = [], []
            for i in range(0, len(te), 65536):
                b = te[i:i + 65536]; bt = torch.as_tensor(b, device=d)
                gb = gi[bt]
                fp, ep_ = model(foc[gb], msa[gb], mask[gb], cf[bt])
                fp_all.append(fp.cpu().numpy()); ep_all.append(torch.sigmoid(ep_).cpu().numpy())
        fp_all = np.concatenate(fp_all); ep_all = np.concatenate(ep_all)
        rho = spearman(fp_all, y_fit[te])
        a = auc(ep_all, y_ess[te]) if 0 < y_ess[te].sum() < len(te) else float("nan")
        # noise floor: median over the test orgs
        test_orgs = set(org[te].tolist())
        nf = np.nanmedian([noise.get(o, np.nan) for o in test_orgs])
        frac = rho / nf if nf == nf and nf > 0 else float("nan")
        results.append(dict(fold=name, n_test=int(len(te)), rho=round(rho, 4),
                            auc=round(a, 4) if a == a else None,
                            noise_floor=round(float(nf), 4) if nf == nf else None,
                            frac_of_floor=round(float(frac), 4) if frac == frac else None))
        print(f"  {name:<26} n={len(te):>8}  rho={rho:.3f}  AUC={a:.3f}  "
              f"floor={nf:.3f}  %floor={frac*100 if frac==frac else float('nan'):.0f}")

    rhos = [r["rho"] for r in results if r["rho"] == r["rho"]]
    aucs = [r["auc"] for r in results if r["auc"]]
    fracs = [r["frac_of_floor"] for r in results if r["frac_of_floor"]]
    summary = dict(eval_mode=eval_mode, n_folds=len(results),
                   median_rho=round(float(np.median(rhos)), 4) if rhos else None,
                   median_auc=round(float(np.median(aucs)), 4) if aucs else None,
                   median_frac_of_floor=round(float(np.median(fracs)), 4) if fracs else None,
                   seconds=round(time.time() - t0, 1), folds=results)
    print(f"\n=== {eval_mode} | median rho {summary['median_rho']} "
          f"| median AUC {summary['median_auc']} "
          f"| median %floor {summary['median_frac_of_floor']} ===")
    json.dump(summary, open(OUT / f"cond_model_{eval_mode}.json", "w"), indent=2)
    print(f"wrote cond_model_{eval_mode}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/orphan/af_msa_cache.npz")
    ap.add_argument("--cond_dir", default="data/drive_import/cond")
    ap.add_argument("--eval", default="condition", choices=["condition", "clade", "both"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = CondCfg(epochs=a.epochs, seed=a.seed)
    print(f"device={cfg.device} eval={a.eval}")
    main(a.cache, a.cond_dir, a.eval, cfg)
