"""Shared utilities for the A100 Colab bundle: path resolution, cache loading,
leak-clean MSA masking, and ranking metrics.

Works both on Colab (repo cloned to /content/cell) and locally
(/home/user/cell) by resolving the repo root from this file's location.
"""
from pathlib import Path
import numpy as np

# ---- path resolution -------------------------------------------------------
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent                       # colab/ -> repo root
OUT  = REPO / "outputs" / "orphan"
DATA = REPO / "data" / "drive_import"
CACHE = OUT / "af_msa_cache.npz"

MAJOR_CLADES = ["pseudomonas", "ralstonia", "shewanella", "dickeya", "burkholderia"]


# ---- cache loading ---------------------------------------------------------
class Cache:
    """Thin holder for the af_msa_cache.npz tensors + org->clade map."""
    def __init__(self, path=CACHE):
        Z = np.load(path, allow_pickle=True)
        self.foc      = Z["foc"].astype(np.float32)        # [N,7] focal features (incl dN/dS at col5, has_dnds col6)
        self.msa      = Z["msa"].astype(np.float32)        # [N,24,6] ortholog rows: label,known,dnds,dn,ds,same_clade
        self.msa_org  = Z["msa_org"].astype(np.int64)      # [N,24] org index per ortholog row
        self.msa_mask = Z["msa_mask"].astype(np.float32)   # [N,24] 1 real / 0 pad
        self.y        = Z["y"].astype(np.float32)          # [N]
        self.clade    = Z["clade"]                         # [N] str
        self.meta_org = Z["meta_org"].astype(np.int64)     # [N] focal org index
        self.lt       = Z["lt"]                            # [N] locus tag
        self.orgs     = list(Z["orgs"])                    # [n_org] names
        self.N, self.K, self.Dm = self.msa.shape
        self.Df = self.foc.shape[1]
        # org index -> clade string
        o2c = {}
        for oi, c in zip(self.meta_org, self.clade):
            o2c[int(oi)] = str(c)
        self.org_clade = {i: o2c.get(i, "?") for i in range(len(self.orgs))}

    def held_cache_orgs(self, held_clade):
        """Cache-space org indices belonging to the held clade."""
        return [i for i, c in self.org_clade.items() if c == held_clade]

    def masked_msa(self, held_clade):
        """Return a copy of msa with label(col0)+known(col1) zeroed for
        ortholog rows from the held clade -- the core leakage control."""
        bad_orgs = self.held_cache_orgs(held_clade)
        out = self.msa.copy()
        bad = np.isin(self.msa_org, bad_orgs)
        out[..., 0] = np.where(bad, 0.0, out[..., 0])
        out[..., 1] = np.where(bad, 0.0, out[..., 1])
        return out


# ---- metrics ---------------------------------------------------------------
def auc(s, yy):
    yy = np.asarray(yy); n1 = yy.sum(); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(s, yy):
    o = np.argsort(-s); ys = np.asarray(yy)[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    return float(np.sum((tp / k) * np.diff(np.concatenate([[0], tp / max(1, ys.sum())]))))


def cov_at_p(s, yy, t, mn=20):
    """Largest coverage fraction whose running precision (sorted by score
    descending) stays >= t."""
    o = np.argsort(-s); ys = np.asarray(yy)[o].astype(float); k = np.arange(1, len(ys) + 1)
    pr = np.cumsum(ys) / k; ok = (pr >= t) & (k >= mn)
    return float(k[np.where(ok)[0].max()] / len(ys)) if ok.any() else 0.0
