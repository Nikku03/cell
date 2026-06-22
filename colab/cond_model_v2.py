"""Stage 4 -- conditional essentiality model v2.

Five changes from v1 that make condition-awareness actually trainable:

  1. PROTEIN BRANCH    : ESM-2 embedding [P, 480] projected to the gene rep.
                         Carries biochemistry the MSA can't see.
  2. BILINEAR g x c    : low-rank bilinear interaction
                         score = sum_r (gene_vec @ U_r) * (cond_vec @ V_r)
                         instead of a multiplicative gate the model can ignore.
  3. PROTEIN x COND    : same bilinear, separately for protein x condition.
                         This is *the* term where biochemistry-environment
                         interaction lives.
  4. RANKING LOSS      : within-condition pairwise margin. The right question
                         is "which gene is more affected by THIS condition",
                         not "what's the absolute fit_z". Eliminates the
                         experiment-scale problem v1 had.
  5. NO BINARY HEAD    : the 1.9% ess rate had the BCE head dominating with a
                         degenerate predictor. Pure regression; threshold at
                         eval time to report AUC.

CLI: --use_protein {0,1} --use_gene_emb {0,1}  for ablation.
  python colab/cond_model_v2.py --eval condition --use_protein 1 --use_gene_emb 0
"""
import json, argparse, time
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from af_common import Cache, OUT, auc


@dataclass
class V2Cfg:
    H: int = 64
    cond_dim: int = 32
    prot_dim: int = 64
    rank: int = 8                # bilinear rank
    gene_emb_dim: int = 16       # for the memorization ablation
    hid: int = 128
    epochs: int = 10
    bs: int = 8192
    lr: float = 1e-3
    wd: float = 1e-4
    margin_pairs: int = 64       # ranking pairs per condition per batch
    seed: int = 0
    use_protein: int = 1
    use_gene_emb: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class V2(nn.Module):
    def __init__(self, df, dm, n_comp, n_prot_dim, n_genes, cfg):
        super().__init__()
        self.cfg = cfg
        # gene branch (focal + MSA attention -> gene_vec)
        self.Wq = nn.Linear(df, cfg.H, bias=False)
        self.Wk = nn.Linear(dm, cfg.H, bias=False)
        self.Wv = nn.Linear(dm, cfg.H, bias=False)
        self.gene_proj = nn.Linear(df + cfg.H, cfg.H)
        # condition branch
        self.cond_enc = nn.Sequential(nn.Linear(n_comp, cfg.cond_dim), nn.ReLU(),
                                      nn.Linear(cfg.cond_dim, cfg.cond_dim))
        # protein branch (optional)
        if cfg.use_protein:
            self.prot_enc = nn.Sequential(nn.Linear(n_prot_dim, cfg.prot_dim), nn.ReLU(),
                                          nn.Linear(cfg.prot_dim, cfg.prot_dim))
        # gene-id embedding (memorization ablation -- the consultant's test)
        if cfg.use_gene_emb:
            self.gene_emb = nn.Embedding(n_genes, cfg.gene_emb_dim)
        # bilinear interactions: low rank
        self.U_gc = nn.Linear(cfg.H, cfg.rank, bias=False)
        self.V_gc = nn.Linear(cfg.cond_dim, cfg.rank, bias=False)
        if cfg.use_protein:
            self.U_pc = nn.Linear(cfg.prot_dim, cfg.rank, bias=False)
            self.V_pc = nn.Linear(cfg.cond_dim, cfg.rank, bias=False)
        # final head: concatenate everything; MLP -> scalar fitness regression
        din = cfg.H + cfg.cond_dim + cfg.rank
        if cfg.use_protein: din += cfg.prot_dim + cfg.rank
        if cfg.use_gene_emb: din += cfg.gene_emb_dim
        self.mlp = nn.Sequential(nn.Linear(din, cfg.hid), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(cfg.hid, cfg.hid // 2), nn.ReLU())
        self.head = nn.Linear(cfg.hid // 2, 1)

    def gene_vec(self, foc, msa, mask):
        q = self.Wq(foc); k = self.Wk(msa); v = self.Wv(msa)
        s = torch.einsum("bkh,bh->bk", k, q) / (k.shape[-1] ** 0.5)
        s = s.masked_fill(mask <= 0, -1e9)
        a = torch.softmax(s, -1)
        ctx = torch.einsum("bk,bkh->bh", a, v)
        return F.relu(self.gene_proj(torch.cat([foc, ctx], -1)))

    def forward(self, foc, msa, mask, cond, prot=None, gid=None):
        g = self.gene_vec(foc, msa, mask)
        c = self.cond_enc(cond)
        parts = [g, c, self.U_gc(g) * self.V_gc(c)]
        if self.cfg.use_protein and prot is not None:
            p = self.prot_enc(prot)
            parts += [p, self.U_pc(p) * self.V_pc(c)]
        if self.cfg.use_gene_emb and gid is not None:
            parts.append(self.gene_emb(gid))
        z = self.mlp(torch.cat(parts, -1))
        return self.head(z).squeeze(-1)


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = ra.std() * rb.std()
    return float((ra * rb).mean() / d) if d > 0 else float("nan")


def within_condition_pairs(cluster_id, batch_idx, n_pairs):
    """Sample positive/negative pairs sharing the SAME condition cluster --
    so the loss rewards within-condition gene ranking, not absolute scale."""
    by_cluster = {}
    for bi, ci in enumerate(cluster_id[batch_idx]):
        by_cluster.setdefault(int(ci), []).append(bi)
    pairs_a, pairs_b = [], []
    rng = np.random.default_rng()
    pool = [lst for lst in by_cluster.values() if len(lst) >= 2]
    if not pool:
        return None, None
    for _ in range(n_pairs):
        lst = pool[rng.integers(len(pool))]
        a, b = rng.choice(lst, 2, replace=False)
        pairs_a.append(a); pairs_b.append(b)
    return np.array(pairs_a), np.array(pairs_b)


def main(cache_path, cond_dir, eval_mode, cfg):
    d = cfg.device
    cache = Cache(cache_path)
    Z = np.load(f"{cond_dir}/cond_pairs.npz", allow_pickle=True)
    meta = json.load(open(f"{cond_dir}/cond_meta.json"))
    gene_idx = Z["gene_idx"]; cond_feat = Z["cond_feat"]
    y_fit = Z["y_fit"]; y_ess = Z["y_ess"].astype(np.float32)
    clade = Z["clade"]; org = Z["org"]; cluster_id = Z["cluster_id"]
    noise = meta["noise_floor"]
    n_pairs_total = len(gene_idx)
    n_unique_genes = int(cache.foc.shape[0])
    print(f"pairs={n_pairs_total:,} unique genes={len(set(gene_idx.tolist())):,} "
          f"clusters={meta['n_clusters']} orgs={meta['n_orgs']}")

    # ---- protein embeddings (optional) ----
    prot = None
    if cfg.use_protein:
        emb_path = f"{cond_dir}/protein_emb.npy"
        try:
            prot_np = np.load(emb_path).astype(np.float32)
            print(f"protein embeddings: {prot_np.shape}")
            prot = torch.tensor(prot_np, device=d)
        except FileNotFoundError:
            print(f"!! {emb_path} not found -- run embed_proteins_esm.py; "
                  f"falling back to no-protein.")
            cfg.use_protein = 0

    foc = torch.tensor(cache.foc, device=d)
    msa = torch.tensor(cache.msa, device=d)
    mask = torch.tensor(cache.msa_mask, device=d)
    cf = torch.tensor(cond_feat, device=d)
    gi = torch.tensor(gene_idx.astype(np.int64), device=d)
    yf = torch.tensor(y_fit, device=d)

    # ---- folds ----
    if eval_mode == "condition":
        uniq = np.unique(cluster_id); rng = np.random.default_rng(0); rng.shuffle(uniq)
        folds = [("cond_fold%d" % i, np.isin(cluster_id, uniq[i::5])) for i in range(5)]
    elif eval_mode == "clade":
        folds = [(c, clade == c) for c in sorted(set(clade.tolist()))]
    else:
        uniq_c = sorted(set(clade.tolist()))
        rng = np.random.default_rng(0); cl_ids = np.unique(cluster_id)
        held = set(rng.choice(cl_ids, max(1, len(cl_ids) // 5), replace=False).tolist())
        folds = [(f"clade={c}+condhold",
                  (clade == c) & np.isin(cluster_id, list(held))) for c in uniq_c]

    results = []; t0 = time.time()
    n_comp = cond_feat.shape[1]
    n_prot_dim = prot.shape[1] if prot is not None else 0

    for name, test_mask in folds:
        tr = np.where(~test_mask)[0]; te = np.where(test_mask)[0]
        if len(te) < 50 or len(tr) < 1000:
            continue
        torch.manual_seed(cfg.seed)
        model = V2(cache.Df, cache.Dm, n_comp, n_prot_dim, n_unique_genes, cfg).to(d)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        torch_gen = torch.Generator().manual_seed(cfg.seed)
        for ep in range(cfg.epochs):
            perm = tr[torch.randperm(len(tr), generator=torch_gen).numpy()]
            ep_loss = 0.0; nb = 0
            for i in range(0, len(perm), cfg.bs):
                b = perm[i:i + cfg.bs]; bt = torch.as_tensor(b, device=d)
                gb = gi[bt]
                pb = prot[gb] if prot is not None else None
                fp = model(foc[gb], msa[gb], mask[gb], cf[bt], pb, gb)
                # MSE keeps the absolute calibration; ranking gives within-condition signal
                loss_mse = F.mse_loss(fp, yf[bt])
                a_idx, b_idx = within_condition_pairs(cluster_id, b, cfg.margin_pairs)
                loss_rk = torch.tensor(0.0, device=d)
                if a_idx is not None:
                    fa = fp[a_idx]; fb = fp[b_idx]
                    truth = (yf[bt][a_idx] - yf[bt][b_idx])    # >0 means a > b
                    pred = fa - fb
                    # margin loss: pred should have same sign as truth
                    loss_rk = F.relu(0.1 - pred * torch.sign(truth)).mean()
                loss = loss_mse + 0.3 * loss_rk
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            fp_all = []
            for i in range(0, len(te), 65536):
                b = te[i:i + 65536]; bt = torch.as_tensor(b, device=d)
                gb = gi[bt]
                pb = prot[gb] if prot is not None else None
                fp = model(foc[gb], msa[gb], mask[gb], cf[bt], pb, gb)
                fp_all.append(fp.cpu().numpy())
        fp_all = np.concatenate(fp_all)
        rho = spearman(fp_all, y_fit[te])
        # AUC from regression: threshold predictions at median, compare to is_ess
        ye_test = y_ess[te]
        a_val = auc(-fp_all, ye_test) if 0 < ye_test.sum() < len(ye_test) else float("nan")
        test_orgs = set(org[te].tolist())
        nf = np.nanmedian([noise.get(o, np.nan) for o in test_orgs])
        frac = rho / nf if nf == nf and nf > 0 else float("nan")
        results.append(dict(fold=name, n_test=int(len(te)), rho=round(rho, 4),
                            auc=round(a_val, 4) if a_val == a_val else None,
                            noise_floor=round(float(nf), 4) if nf == nf else None,
                            frac_of_floor=round(float(frac), 4) if frac == frac else None))
        print(f"  {name:<26} n={len(te):>8}  rho={rho:.3f}  AUC={a_val:.3f}  "
              f"floor={nf:.3f}  %floor={frac*100 if frac==frac else float('nan'):.0f}")

    rhos = [r["rho"] for r in results if r["rho"] == r["rho"]]
    aucs = [r["auc"] for r in results if r["auc"]]
    fracs = [r["frac_of_floor"] for r in results if r["frac_of_floor"]]
    summary = dict(eval_mode=eval_mode, config=cfg.__dict__, n_folds=len(results),
                   median_rho=round(float(np.median(rhos)), 4) if rhos else None,
                   median_auc=round(float(np.median(aucs)), 4) if aucs else None,
                   median_frac_of_floor=round(float(np.median(fracs)), 4) if fracs else None,
                   seconds=round(time.time() - t0, 1), folds=results)
    tag = f"_v2_p{cfg.use_protein}_e{cfg.use_gene_emb}"
    print(f"\n=== {eval_mode}{tag} | rho {summary['median_rho']} "
          f"| AUC {summary['median_auc']} | %floor {summary['median_frac_of_floor']} ===")
    json.dump(summary, open(OUT / f"cond_model{tag}_{eval_mode}.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/orphan/af_msa_cache.npz")
    ap.add_argument("--cond_dir", default="data/drive_import/cond")
    ap.add_argument("--eval", default="condition", choices=["condition", "clade", "both"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_protein", type=int, default=1)
    ap.add_argument("--use_gene_emb", type=int, default=0)
    a = ap.parse_args()
    cfg = V2Cfg(epochs=a.epochs, seed=a.seed,
                use_protein=a.use_protein, use_gene_emb=a.use_gene_emb)
    print(f"device={cfg.device} eval={a.eval} use_protein={cfg.use_protein} "
          f"use_gene_emb={cfg.use_gene_emb}")
    main(a.cache, a.cond_dir, a.eval, cfg)
