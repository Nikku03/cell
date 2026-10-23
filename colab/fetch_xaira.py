"""fetch_xaira — stage the Xaira X-Atlas/Orion genome-wide Perturb-seq screen (a SECOND cell type) as a pseudobulk
[perturbation x gene] signature h5ad the CellOS debugger can mount, and measure the coverage it adds + its cross-cell
consistency with the K562 (Replogle) screen we already have.

The source is per-cell sparse expression stored as ~330 parquet batches on Hugging Face
(Xaira-Therapeutics/X-Atlas-Orion): 8M HCT116 + HEK293T cells, 18,903 protein-coding targets, ~16k UMIs/cell, median
~140 cells/perturbation, CRISPRi knockdown (75% HCT116 / 51% HEK293T). We do NOT keep the 8M cells: we STREAM each
batch (download -> fold into a running sum(log1p(CP10k)) & cell-count per gene_target -> delete the parquet), so peak
disk stays ~one batch and the output is a compact control-relative signature matrix in the exact format
cellos._read_pert already reads (obs index 'g_<GENE>_<cell>', var/gene_name, X = mean-log-norm minus non-targeting).

  build_pseudobulk(cell, max_batches)  — stream + accumulate (resumable checkpoint)
  assemble(cell, min_cells)            — checkpoint -> {SCRATCH}/<cell>.h5ad
  validate(cell)                       — coverage vs the cell model + cross-cell-type consistency vs K562

Run:  python3 colab/fetch_xaira.py            (HCT116, all batches, then assemble + validate)
      python3 colab/fetch_xaira.py HEK293T    (the other cell line)
-> outputs/orphan/xaira_coverage.json
"""
import os, sys, json, gc, subprocess, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import cellos
SCRATCH = cellos.SCRATCH
XDIR = f"{SCRATCH}/xaira"
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
BASE = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main"
API = "https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/main?recursive=true"


def _vocab(C):
    """column vocabulary = Xaira genes ∩ our cell model, plus token->column map (deterministic)."""
    import pandas as pd
    meta = pd.read_parquet(f"{XDIR}/gene_metadata.parquet")
    tok2sym = dict(zip(meta["gene_token_id"].astype(int), meta["gene_name"].astype(str)))
    cellset = set(C.name)
    cols = sorted({s for s in tok2sym.values() if s in cellset})
    col_idx = {s: i for i, s in enumerate(cols)}
    tok2col = np.full(max(tok2sym) + 1, -1, np.int64)
    for t, s in tok2sym.items():
        if s in col_idx:
            tok2col[t] = col_idx[s]
    return cols, tok2col


def build_pseudobulk(cell="HCT116", max_batches=None):
    """stream every <cell> batch into a checkpointed [pert x gene] sum + cell-count. Resumable; peak disk ~1 batch."""
    from scipy import sparse
    import pyarrow.parquet as pq
    os.makedirs(XDIR, exist_ok=True)
    if not os.path.exists(f"{XDIR}/gene_metadata.parquet"):
        subprocess.run(["curl", "-sS", "-L", "-o", f"{XDIR}/gene_metadata.parquet",
                        f"{BASE}/metadata/gene_metadata.parquet"], check=True)
    C = cellos.CellKernel(quiet=True).C
    cols, tok2col = _vocab(C); NCOL = len(cols)
    ckpt = f"{XDIR}/pseudobulk_ckpt_{cell}.npz"
    SUM = np.zeros((0, NCOL), np.float32); CNT = np.zeros((0,), np.int64); rows, done = [], []
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        SUM, CNT, rows, done = z["SUM"], z["CNT"], list(z["rows"]), list(z["done"])
    pert2row = {p: i for i, p in enumerate(rows)}
    tree = json.load(urllib.request.urlopen(API))
    batches = sorted(x["path"] for x in tree if x.get("type") == "file"
                     and x["path"].startswith(f"data/{cell}_Batch") and x["path"].endswith(".parquet"))
    todo = [b for b in batches if os.path.basename(b) not in done]
    if max_batches:
        todo = todo[:max_batches]
    print(f"[{cell}] {len(batches)} batches; {len(done)} folded; folding {len(todo)} now ({NCOL} gene cols)", flush=True)

    def grow(n):
        nonlocal SUM, CNT
        if n > SUM.shape[0]:
            SUM = np.vstack([SUM, np.zeros((n - SUM.shape[0], NCOL), np.float32)])
            CNT = np.concatenate([CNT, np.zeros(n - CNT.shape[0], np.int64)])

    for b in todo:
        fn = os.path.basename(b); lp = f"{XDIR}/{fn}"; t0 = time.time()
        if subprocess.run(["curl", "-sS", "-L", "--max-time", "900", "-o", lp, f"{BASE}/{b}"]).returncode or \
           not os.path.exists(lp):
            print(f"[skip] {fn}", flush=True); continue
        try:
            df = pq.ParquetFile(lp).read(columns=["gene_token_id", "gene_expression", "gene_target",
                                                  "total_counts", "pass_guide_filter"]).to_pandas()
            df = df[df["pass_guide_filter"] == 1]
            tgts = df["gene_target"].astype(str).values
            for t in np.unique(tgts):
                if t not in pert2row:
                    pert2row[t] = len(rows); rows.append(t)
            grow(len(rows))
            prow = np.array([pert2row[t] for t in tgts])
            tok, expr = df["gene_token_id"].values, df["gene_expression"].values
            tot = df["total_counts"].values.astype(np.float32)
            data_l, col_l, indptr = [], [], [0]
            for k in range(len(df)):
                tk = np.asarray(tok[k], np.int64); c = tok2col[tk]; m = c >= 0
                v = np.log1p(np.asarray(expr[k], np.float32)[m] * (1e4 / tot[k])) if tot[k] > 0 \
                    else np.zeros(int(m.sum()), np.float32)
                col_l.append(c[m]); data_l.append(v); indptr.append(indptr[-1] + int(m.sum()))
            X = sparse.csr_matrix((np.concatenate(data_l), np.concatenate(col_l), np.array(indptr)),
                                  shape=(len(df), NCOL), dtype=np.float32)
            ind = sparse.csr_matrix((np.ones(len(prow), np.float32), (prow, np.arange(len(prow)))),
                                    shape=(len(rows), len(prow)))
            SUM[:len(rows)] += np.asarray((ind @ X).todense(), np.float32)
            CNT[:len(rows)] += np.bincount(prow, minlength=len(rows))
            done.append(fn)
            del df, X, ind; gc.collect()
        except Exception as e:
            print(f"[err] {fn}: {str(e)[:120]}", flush=True)
        finally:
            if os.path.exists(lp):
                os.remove(lp)
        np.savez(ckpt, SUM=SUM, CNT=CNT, rows=np.array(rows, dtype=object), done=np.array(done, dtype=object))
        cov = int((CNT > 0).sum()); med = int(np.median(CNT[CNT > 0])) if cov else 0
        print(f"[{len(done):3}/{len(batches)}] {fn} ({time.time()-t0:.0f}s) perts={cov} med_cells/pert={med}", flush=True)
    return ckpt


def assemble(cell="HCT116", min_cells=20):
    """checkpoint -> control-relative signature h5ad ({SCRATCH}/<cell>.h5ad) in cellos._read_pert format."""
    import h5py
    C = cellos.CellKernel(quiet=True).C
    cols, _ = _vocab(C)
    z = np.load(f"{XDIR}/pseudobulk_ckpt_{cell}.npz", allow_pickle=True)
    SUM, CNT, rows = z["SUM"], z["CNT"], [str(r) for r in z["rows"]]
    pert2row = {p: i for i, p in enumerate(rows)}
    mean = np.zeros_like(SUM); nz = CNT > 0; mean[nz] = SUM[nz] / CNT[nz][:, None]
    ctrl = next((k for k in ("Non-Targeting", "non-targeting", "NTC") if k in pert2row), None)
    ntc = mean[pert2row[ctrl]] if ctrl else mean[nz].mean(0)
    sig = (mean - ntc).astype(np.float32)
    keep = [i for i in range(len(rows)) if CNT[i] >= min_cells and rows[i] != ctrl]
    labels = [f"g_{rows[i].replace('_', '-')}_{cell.lower()}" for i in keep]
    out = f"{SCRATCH}/{cell.lower()}.h5ad"; vlen = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as f:
        f.create_dataset("X", data=sig[keep], dtype="float32")
        g = f.create_group("obs"); g.attrs["_index"] = "idx"
        g.create_dataset("idx", data=np.array(labels, dtype=object), dtype=vlen)
        f.create_group("var").create_dataset("gene_name", data=np.array(cols, dtype=object), dtype=vlen)
    print(f"[assemble] {out}  ({os.path.getsize(out)/1e6:.0f} MB)  perts={len(keep)} "
          f"median_cells/pert={int(np.median(CNT[keep])) if keep else 0}", flush=True)
    return out


def validate(cell="HCT116"):
    """coverage vs the cell model + cross-cell-type consistency vs the K562 debugger."""
    import h5py
    from scipy.stats import pearsonr
    K = cellos.CellKernel(quiet=True); C = K.C
    cellset = set(C.name)
    f = h5py.File(f"{SCRATCH}/{cell.lower()}.h5ad", "r")
    labels = [x.decode() if isinstance(x, bytes) else x for x in f["obs"]["idx"][:]]
    hgenes = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_name"][:]]
    X = f["X"][:]; f.close()
    hperts = [l.split("_")[1] for l in labels]; hrow = {p: i for i, p in enumerate(hperts)}
    hcol = {g: i for i, g in enumerate(hgenes)}
    dbg = K._load_debugger(); k_perts = set(dbg["pgenes"]); k_genes = set(dbg["syms"])
    kM, kpg, ksy = K._read_pert(f"{SCRATCH}/k562.h5ad")           # essential screen for the paired test
    kpidx = {g: i for i, g in enumerate(kpg)}; ksidx = {g: i for i, g in enumerate(ksy)}
    kc = k_perts & cellset; hc = set(hperts) & cellset
    shared_g = [g for g in hgenes if g in ksidx]
    gh = np.array([hcol[g] for g in shared_g]); gk = np.array([ksidx[g] for g in shared_g])
    rng = np.random.default_rng(0); se_real, se_shuf = [], []
    shared_p = [p for p in hperts if p in kpidx]
    for p in shared_p:
        a = X[hrow[p]][gh]; b = kM[kpidx[p]][gk]
        strong = np.where(np.abs(kM[kpidx[p]][gk]) > 0.5)[0]
        if len(strong) >= 8 and np.std(a[strong]) > 1e-6 and np.std(b[strong]) > 1e-6:
            se_real.append(pearsonr(a[strong], b[strong])[0])
            q = shared_p[rng.integers(len(shared_p))]; bq = kM[kpidx[q]][gk][strong]
            if np.std(bq) > 1e-6:
                se_shuf.append(pearsonr(a[strong], bq)[0])
    rep = {"cell": cell, "perturbations": len(hperts), "genes_measured": len(hgenes),
           "knockouts_in_cell_model": len(hc),
           "K562_knockouts_in_model": len(kc), "in_both_celltypes": len(kc & hc),
           "union_coverage": len(kc | hc), "cell_model_size": len(cellset),
           "union_coverage_frac": round(len(kc | hc) / len(cellset), 3),
           "new_knockouts_not_in_K562": len(hc - kc),
           "genes_readout_K562": len(k_genes & cellset), "genes_readout_union": len((k_genes | set(hgenes)) & cellset),
           "cross_celltype_strong_pearson": round(float(np.median(se_real)), 3) if se_real else None,
           "cross_celltype_shuffled": round(float(np.median(se_shuf)), 3) if se_shuf else None,
           "n_paired_perturbations": len(se_real)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(f"{OUT}/xaira_coverage.json", "w"), indent=1)
    return rep


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "HCT116"
    build_pseudobulk(cell)
    assemble(cell)
    rep = validate(cell)
    print("\n" + "=" * 92)
    print(f"XAIRA X-Atlas/Orion {cell} — coverage this SECOND cell type adds to the cell:")
    print(f"  knockouts (cell-model genes): {rep['knockouts_in_cell_model']:,}  "
          f"(K562 had {rep['K562_knockouts_in_model']:,}); +{rep['new_knockouts_not_in_K562']:,} new")
    print(f"  UNION knockout coverage: {rep['union_coverage']:,}/{rep['cell_model_size']:,} "
          f"= {rep['union_coverage_frac']:.0%} of the cell perturbed in >=1 cell type")
    print(f"  cross-cell-type validatable (both K562 & {cell}): {rep['in_both_celltypes']:,}")
    print(f"  genes measured (readout): {rep['genes_measured']:,}  (K562 {rep['genes_readout_K562']:,})")
    print(f"  CROSS-CELL-TYPE consistency (strong effects): r={rep['cross_celltype_strong_pearson']} "
          f"(shuffled {rep['cross_celltype_shuffled']}, n={rep['n_paired_perturbations']})")
    print("=" * 92)
    return rep


if __name__ == "__main__":
    main()
