"""crosscell — use the second cell type (Xaira X-Atlas/Orion HCT116) to tell CELL-TYPE-ROBUST knockout effects from
CELL-TYPE-SPECIFIC ones. The K562 (Replogle) debugger says what a knockout does in leukemia cells; HCT116 says what
it does in colorectal cells. An effect seen in BOTH is a property of the gene, not the cell line — the trustworthy
core of a cascade; an effect in only one is cell-type-specific (real, but conditional). Validated: on strongly-moved
genes the two screens agree at Pearson ~0.5-0.6 (vs ~0.04 shuffled), so agreement is meaningful.

  consistency(GENE) : the K562-vs-HCT116 correlation of a knockout's signature (how reproducible it is)
  compare(GENE)     : responders split into ROBUST (both, same direction) / K562-only / HCT116-only
  robust(GENE)      : just the cell-type-robust responders (the effects safe to build a cascade on)

Needs {SCRATCH}/hct116.h5ad staged (build it with `python3 colab/fetch_xaira.py`).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import cellos
SCRATCH = cellos.SCRATCH


class CrossCell:
    def __init__(self, kernel=None, cell="hct116", thr=0.5):
        self.k = kernel if kernel is not None else cellos.CellKernel(quiet=True)
        self.cell = cell; self.thr = thr
        dbg = self.k._load_debugger()
        if not dbg:
            raise RuntimeError("K562 debugger not mounted")
        self.kM = np.asarray(dbg["M"], "float32")
        self.kpidx = {g: i for i, g in enumerate(dbg["pgenes"])}
        self.ksidx = {g: i for i, g in enumerate(dbg["syms"])}
        path = f"{SCRATCH}/{cell}.h5ad"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not staged — run `python3 colab/fetch_xaira.py`")
        self.hM, hpg, hsy = self.k._read_pert(path)
        self.hpidx = {g: i for i, g in enumerate(hpg)}
        self.hsidx = {g: i for i, g in enumerate(hsy)}
        # shared measured-gene columns (align the two readouts)
        self.shared = [g for g in hsy if g in self.ksidx]
        self.gk = np.array([self.ksidx[g] for g in self.shared])
        self.gh = np.array([self.hsidx[g] for g in self.shared])

    def measured_in_both(self, gene):
        return gene in self.kpidx and gene in self.hpidx

    def _vecs(self, gene):
        return self.kM[self.kpidx[gene]][self.gk], self.hM[self.hpidx[gene]][self.gh]

    def consistency(self, gene):
        """Pearson of the two cell types' signatures for KO(gene), on the strongly-moved genes."""
        if not self.measured_in_both(gene):
            return {"status": "not-in-both", "gene": gene,
                    "in_K562": gene in self.kpidx, "in_" + self.cell: gene in self.hpidx}
        from scipy.stats import pearsonr
        a, b = self._vecs(gene)
        strong = np.where((np.abs(a) > self.thr) | (np.abs(b) > self.thr))[0]
        if len(strong) < 5 or np.std(a[strong]) < 1e-6 or np.std(b[strong]) < 1e-6:
            return {"status": "too-weak", "gene": gene, "n_strong": int(len(strong))}
        r = float(pearsonr(a[strong], b[strong])[0])
        return {"status": "ok", "gene": gene, "pearson": round(r, 2), "n_strong": int(len(strong))}

    def compare(self, gene, topn=12):
        """responders split into ROBUST (moved in both, same direction), K562-only, and HCT116-only."""
        if not self.measured_in_both(gene):
            c = "K562" if gene in self.kpidx else (self.cell if gene in self.hpidx else "neither")
            return {"status": "not-in-both", "gene": gene, "measured_in": c}
        a, b = self._vecs(gene)
        ka = np.abs(a) > self.thr; hb = np.abs(b) > self.thr
        both = np.where(ka & hb)[0]
        robust = [(self.shared[j], round(float(a[j]), 2), round(float(b[j]), 2))
                  for j in both if np.sign(a[j]) == np.sign(b[j])]
        robust.sort(key=lambda x: -abs(x[1] + x[2]))
        k_only = sorted((self.shared[j] for j in np.where(ka & ~hb)[0]),
                        key=lambda g: -abs(a[self.shared.index(g)]))[:topn] if ka.sum() else []
        h_only = [self.shared[j] for j in np.where(hb & ~ka)[0]]
        cons = self.consistency(gene)
        return {"status": "ok", "gene": gene, "consistency": cons.get("pearson"),
                "n_robust": len(robust), "robust": robust[:topn],
                "n_K562_only": int((ka & ~hb).sum()), "K562_only": list(k_only)[:topn],
                "n_" + self.cell + "_only": int((hb & ~ka).sum()), self.cell + "_only": h_only[:topn]}

    def robust(self, gene):
        c = self.compare(gene)
        return [] if c["status"] != "ok" else [g for g, _, _ in c["robust"]]
