"""CELLBANK9 -- engine v2 + the datasets already collected on the user's Drive.

The Drive inventory (2026-11-14) found four large datasets that the engine had never used: LINCS L1000 knockdown
signatures, the Tahoe-100M drug-screen statistics, Geneformer gene embeddings, and the ARCHS4 expression compendium.
The Drive connector refuses files over 10 MB, and the Drive copies are themselves downloads of public releases, so
each is read here from its public origin (only the parts needed; HTTP range requests where the file allows it).
Raw files and derived vectors stay in the git-ignored cache (rem/atlas/_cache/cellbank9, _cache/cell_v9); only code
and summary statistics are committed.

THE FOUR SOURCES AND HOW EACH ENTERS THE MODEL
 L  LINCS 2020 (Level 5, beta release; clue.io S3 public bucket). Knockdown signatures in 9-18 cell lines measured on
    978 landmark genes plus best-inferred genes (z-scores). Two new screens for the bank and the translator:
      LINCS-sh   the exemplar consensus shRNA signatures (trt_sh.cgs: hairpins collapsed, fewer seed off-targets)
      LINCS-xpr  the exemplar CRISPR knockout signatures (trt_xpr)
    One record per gene per screen = the mean of its exemplar signatures over cell lines, times and doses, mapped by
    symbol (HPA synonyms allowed) onto screen A's genes, then exactly as cellbank8 built the other screens (TRAIN-gene
    mean removed, projected on the bank's 128 components, strength = norm / median norm, int8). Bank v9 = bank v8 +
    these records. The translator is retrained on 13 screens.
 T  Tahoe-100M DE statistics (tahoebio/tahoe-de-rhaister, cell_eval table, CC0): pseudobulk deltas for ~2,000 genes in
    every (cell line, drug, dose) condition. A gene's vector = its standardised deltas over all conditions
    (drug co-response). Only ~141 of screen A's knocked-down genes are among the 2,000: a small source by design.
 F  Geneformer V2-104M (ctheodoris/Geneformer, Apache-2.0): the input token embedding of each gene (768-d), read
    straight from the model file. (Not the contextual embedding the Drive file may hold; a disclosed approximation.)
 X  ARCHS4 v2 (human_gene_v2.latest.h5, 1.13M GEO samples): co-expression over 10,000 samples, five contiguous
    blocks of 2,000 at fixed offsets (50k + 188k*i, i = 0..4; contiguous because the file stores one sample per
    chunk column), bulk only (single-cell probability < 0.5, library > 1M reads), log2(1 + CPM), standardised per gene.
 T, F, X enter as gene-neighbour inputs, the same way as cellbank8's analog input: for each gene, its 20 nearest
 genes by cosine in that source (vectors centred over genes, unit length) among genes with a K562 record (A if TRAIN,
 else the K562-essential replicate B; never the gene itself), weights exp(10 cos); input = the weighted mean of
 their K562 records (128) + top cosine + has-vector flag.

ARMS (seeds 0-2 averaged; cellbank8's recipe unchanged; everything retrained in this run so arms are matched)
  M2    cellbank8's model, rerun: 11-screen translator + [T, flags, analog, partners, co-essential].
  M9L   + LINCS only: 13-screen translator (and analog over 13 screens).
  M9N   + neighbours only: M2's inputs + the T, F, X neighbour inputs.
  M9    both.

GATES, PREDECLARED (paired sign test over genes; BETTER = p < 0.05 and median gain >= +0.01; WORSE mirrored)
  Scored on the EXAM half of cellbank8. This is the EXAM half's SECOND read: cellbank8 scored its own gates on it
  once (outputs/cellbank8/exam_lock.json); nothing in this script was chosen by looking at it. Development uses DEV
  only. `exam` scores once, writes outputs/cellbank9/exam_lock.json, and refuses to run again.
  Y1  PRIMARY: M9 vs M2, r, all EXAM genes.
  Y2  M9 vs M2, r, EXAM genes whose strongest record in another cell type (bank v8) is weaker than the median over
      all TEST genes (the weak-record half: where new data should matter most).
      [Revision 0, after `build`, before any training or scoring: Y2 first read "EXAM genes with no record in any
      other cell type in bank v8"; the build report showed only 13 of 2,525 TEST genes have none (~6 on EXAM), too
      few for a test, so the subgroup was replaced as above. The no-record genes are still reported, descriptively.]
  Y3  M9 vs M2, specific r (top 3 TRAIN programmes removed).
  Descriptive (no verdicts claimed): M9L and M9N vs M2; coverage of each source; fraction of ceiling.
NOT: one way to pool LINCS signatures, one neighbour rule (k 20, temperature 10), one ARCHS4 sample draw, input
     embeddings rather than contextual ones; Tabula Sapiens, DepMap expression and ESM (also on Drive) are not used
     here (cell-atlas API / overlap with ARCHS4 co-expression / would need computing).

Run: python3 -P rem/atlas/cellbank9.py pull | build | train | dev | exam
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import pickle
import struct
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbank8", os.path.join(HERE, "cellbank8.py"))
cb8 = importlib.util.module_from_spec(_s); _s.loader.exec_module(cb8)
cb4 = cb8.cb4
import remotezip  # noqa: E402

CACHE9 = os.path.join(HERE, "_cache", "cellbank9")
RAW = os.path.join(CACHE9, "raw")
BANK9 = os.path.join(HERE, "_cache", "cell_v9")
OUTD9 = R("outputs", "cellbank9")
OUT9 = os.path.join(HERE, "RESULTS_cellbank9.txt")
LINCS = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020"
LINCS_FILES = {"LINCS-sh": (LINCS + "/level5/level5_beta_trt_sh_n238351x12328.gctx", "trt_sh.cgs"),
               "LINCS-xpr": (LINCS + "/level5/level5_beta_trt_xpr_n142901x12328.gctx", "trt_xpr")}
NEW = list(LINCS_FILES)
SRC8 = list(cb8.SRC)
SRC9 = SRC8 + NEW
TAHOE = "https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister/resolve/main/cell_eval/plate_plate{}.parquet"
GF = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/"
ARCHS4 = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5"
ARCHS4_STARTS, ARCHS4_N = [50_000 + 188_000 * i for i in range(5)], 2000
NB = ("Tahoe", "Geneformer", "ARCHS4")
NB_K, NB_TEMP = 20, 10.0
SEEDS, RULE = cb8.SEEDS, cb8.RULE


def _get(url, path):
    if os.path.exists(path):
        return
    for k in range(5):
        try:
            tmp = path + ".part"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cell-cellbank9"}), timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, path); return
        except Exception:
            if k == 4:
                raise
            time.sleep(2 ** k * 3)


class _NoCode(pickle.Unpickler):
    """Dictionaries of strings and ints only: any class lookup in a downloaded pickle is refused."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


# ---------------------------------------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------------------------------------
def pull_lincs(G, to_kn, to_col):
    import h5py
    import numpy as np
    import pandas as pd
    n_dst = len(G["A"]["genes"])
    sig = pd.read_csv(os.path.join(RAW, "siginfo_beta.txt"), sep="\t", low_memory=False,
                      usecols=["sig_id", "pert_type", "cmap_name", "cell_iname", "is_exemplar_sig"])
    gi = pd.read_csv(os.path.join(RAW, "geneinfo_beta.txt"), sep="\t", dtype=str)
    gi = gi[gi.feature_space.isin(["landmark", "best inferred"])]
    sym_of = dict(zip(gi.gene_id, gi.gene_symbol))
    for name, (url, ptype) in LINCS_FILES.items():
        path = os.path.join(CACHE9, f"src_{name}.npz")
        if os.path.exists(path):
            print(f"{name}: cached"); continue
        t0 = time.time()
        f = remotezip.RangeFile(url, cache_blocks=128); h = h5py.File(f, "r")
        cols = [x.decode() for x in h["0/META/COL/id"][()]]
        genes_m = [x.decode() for x in h["0/META/ROW/id"][()]]
        col_src, col_dst, seen = [], [], set()
        for j, gid in enumerate(genes_m):
            d = to_col.get(str(sym_of.get(gid, "")).upper())
            if d is not None and d not in seen:
                seen.add(d); col_src.append(j); col_dst.append(d)
        w = sig[(sig.pert_type == ptype) & (sig.is_exemplar_sig == 1)]
        pos = {s: i for i, s in enumerate(cols)}
        row_gene, lines = {}, {}
        for s, c, cl in zip(w.sig_id, w.cmap_name, w.cell_iname):
            g = to_kn.get(str(c).upper())
            if g is not None and s in pos:
                row_gene[pos[s]] = g; lines.setdefault(g, set()).add(cl)
        sel = np.array(sorted(row_gene))
        X = h["0/DATA/0/matrix"]
        step = (X.chunks[0] * 10) if X.chunks else 2000
        a0 = (int(sel.min()) // step) * step
        acc, cnt = {}, {}
        for a in range(a0, int(sel.max()) + 1, step):
            b = min(a + step, X.shape[0]); rr = sel[(sel >= a) & (sel < b)]
            if not len(rr):
                continue
            blk = X[a:b, :]
            for r in rr:
                g = row_gene[int(r)]; v = blk[int(r) - a, col_src].astype(np.float64)
                acc[g] = acc.get(g, 0.0) + v; cnt[g] = cnt.get(g, 0) + 1
            print(f"   {name}: rows {b:,}/{int(sel.max()) + 1:,}, {f.fetched / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
        gl = sorted(acc)
        M = np.full((len(gl), n_dst), np.nan, np.float32)
        for k, g in enumerate(gl):
            M[k, col_dst] = acc[g] / cnt[g]
        np.savez_compressed(path, genes=np.array(gl), M=M.astype(np.float16),
                            meta=json.dumps(dict(name=name, cell="LINCS lines", design=ptype + " exemplar, mean over lines/times/doses",
                                                 signatures=int(len(sel)), kept=len(gl), genes_mapped=len(col_dst),
                                                 median_signatures_per_gene=float(np.median([cnt[g] for g in gl])),
                                                 median_lines_per_gene=float(np.median([len(lines[g]) for g in gl])),
                                                 gb_fetched=round(f.fetched / 1e9, 2), seconds=round(time.time() - t0))))
        print(f"{name}: {len(gl):,} genes from {len(sel):,} signatures, {len(col_dst):,} genes mapped, {f.fetched / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)


def _save_vecs(name, genes, V, info):
    import numpy as np
    V = V.astype(np.float64)
    V = V - V.mean(0, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    np.savez_compressed(os.path.join(CACHE9, f"nb_{name}.npz"), genes=np.array(genes), V=V.astype(np.float16), meta=json.dumps(info))
    print(f"{name}: {len(genes):,} genes x {V.shape[1]:,} -> {info}", flush=True)


def pull_tahoe(G, to_kn):
    import numpy as np
    import pyarrow.parquet as pq
    path = os.path.join(CACHE9, "nb_Tahoe.npz")
    if os.path.exists(path):
        print("Tahoe: cached"); return
    t0 = time.time(); blocks, genes = [], None
    for p in range(1, 15):
        fp = os.path.join(RAW, f"tahoe_plate{p}.parquet"); _get(TAHOE.format(p), fp)
        names = pq.read_schema(fp).names
        if genes is None:
            pairs, seen = [], set()
            for c in names:
                g = to_kn.get(c.upper())
                if g is not None and g not in seen:
                    seen.add(g); pairs.append((c, g))
            cols = [c for c, _ in pairs]; genes = [g for _, g in pairs]
        M = pq.read_table(fp, columns=cols).to_pandas().to_numpy(dtype=np.float32)   # conditions x genes
        blocks.append(M)
        print(f"   Tahoe plate {p}: {M.shape[0]:,} conditions ({time.time() - t0:.0f}s)", flush=True)
    M = np.vstack(blocks).T; M = np.nan_to_num(M)
    sd = M.std(1, keepdims=True); sd[sd == 0] = 1
    V = (M - M.mean(1, keepdims=True)) / sd
    _save_vecs("Tahoe", genes, V, dict(conditions=int(M.shape[1]), genes=len(genes), seconds=round(time.time() - t0)))


def pull_geneformer(G, to_kn):
    import numpy as np
    path = os.path.join(CACHE9, "nb_Geneformer.npz")
    if os.path.exists(path):
        print("Geneformer: cached"); return
    t0 = time.time()
    tok = os.path.join(RAW, "gf_token_dictionary_gc104M.pkl"); _get(GF + "geneformer/token_dictionary_gc104M.pkl", tok)
    nid = os.path.join(RAW, "gf_gene_name_id_dict_gc104M.pkl"); _get(GF + "geneformer/gene_name_id_dict_gc104M.pkl", nid)
    token = _NoCode(open(tok, "rb")).load(); name_id = _NoCode(open(nid, "rb")).load()
    f = remotezip.RangeFile(GF + "Geneformer-V2-104M/model.safetensors")
    f.seek(0); n = struct.unpack("<Q", f.read(8))[0]; head = json.loads(f.read(n))
    e = head["bert.embeddings.word_embeddings.weight"]; a, b = e["data_offsets"]
    assert e["dtype"] == "F32"
    f.seek(8 + n + a); E = np.frombuffer(f.read(b - a), dtype="<f4").reshape(e["shape"])
    genes, rows, seen = [], [], set()
    for sym, ens in name_id.items():
        g = to_kn.get(str(sym).upper()); t = token.get(ens)
        if g is not None and t is not None and g not in seen:
            seen.add(g); genes.append(g); rows.append(t)
    _save_vecs("Geneformer", genes, E[rows], dict(model="Geneformer-V2-104M", dim=int(E.shape[1]), vocabulary=int(E.shape[0]),
                                                  genes=len(genes), mb_fetched=round(f.fetched / 1e6), seconds=round(time.time() - t0)))


def pull_archs4(G, to_kn):
    import h5py
    import numpy as np
    path = os.path.join(CACHE9, "nb_ARCHS4.npz")
    if os.path.exists(path):
        print("ARCHS4: cached"); return
    t0 = time.time()
    f = remotezip.RangeFile(ARCHS4, cache_blocks=128); h = h5py.File(f, "r")
    sym = [x.decode() if isinstance(x, bytes) else str(x) for x in h["meta/genes/symbol"][()]]
    scp = h["meta/samples/singlecellprobability"][()]
    rows, genes, seen = [], [], set()
    for i, s in enumerate(sym):
        g = to_kn.get(s.upper())
        if g is not None and g not in seen:
            seen.add(g); rows.append(i); genes.append(g)
    X = h["data/expression"]; parts, n_seen = [], 0
    for a in ARCHS4_STARTS:
        E = X[:, a:a + ARCHS4_N].astype(np.float64); lib = E.sum(0); n_seen += E.shape[1]
        ok = (scp[a:a + ARCHS4_N] < 0.5) & (lib > 1e6)
        parts.append(np.log2(1 + E[rows][:, ok] / lib[ok] * 1e6).astype(np.float32))
        print(f"   ARCHS4 block {a:,}: {int(ok.sum())} bulk samples kept of {E.shape[1]}, {f.fetched / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
        del E
    M = np.hstack(parts)
    sd = M.std(1, keepdims=True); sd[sd == 0] = 1
    V = (M - M.mean(1, keepdims=True)) / sd
    _save_vecs("ARCHS4", genes, V, dict(samples_read=n_seen, samples_kept=int(M.shape[1]), genes=len(genes),
                                        gb_fetched=round(f.fetched / 1e9, 2), seconds=round(time.time() - t0)))


def pull():
    G = cb8.nb(); os.makedirs(RAW, exist_ok=True)
    to_kn, to_col = cb8.aliases(G)
    for fn in ("siginfo_beta.txt", "geneinfo_beta.txt", "cellinfo_beta.txt"):
        _get(f"{LINCS}/{fn}", os.path.join(RAW, fn))
    pull_geneformer(G, to_kn)
    pull_tahoe(G, to_kn)
    pull_lincs(G, to_kn, to_col)
    pull_archs4(G, to_kn)


# ---------------------------------------------------------------------------------------------------------
# bank v9 = bank v8 + the two LINCS screens (local only)
# ---------------------------------------------------------------------------------------------------------
def build():
    import numpy as np
    G = nb9(); split = G["split"]; Vy = G["Vy"].cpu().numpy()
    z8 = np.load(os.path.join(cb8.BANK8, "records.npz")); meta = json.load(open(os.path.join(cb8.BANK8, "records_meta.json")))
    codes, scale, rep = [z8["codes"]], [z8["scale"]], {}
    for name in NEW:
        z = np.load(os.path.join(CACHE9, f"src_{name}.npz")); info = json.loads(str(z["meta"]))
        genes = list(z["genes"]); M = z["M"].astype(np.float32)
        tr = [i for i, g in enumerate(genes) if split(g) == "TRAIN"]
        Z = np.nan_to_num(M - np.nanmean(M[tr], 0)) @ Vy
        nrm = np.linalg.norm(Z, axis=1); strength = nrm / np.median(nrm)
        s = np.maximum(np.abs(Z).max(1), 1e-12)
        codes.append(np.round(Z / s[:, None] * 127).astype(np.int8)); scale.append(s.astype(np.float32))
        for g, st in zip(genes, strength):
            meta["strength"][str(len(meta["gene"]))] = float(st)
            meta["gene"].append(g); meta["ens"].append(""); meta["screen"].append(f"S:{name}"); meta["split"].append(split(g)); meta["ep"].append(float("nan"))
        rep[name] = dict(info, records=len(genes))
    os.makedirs(BANK9, exist_ok=True)
    np.savez_compressed(os.path.join(BANK9, "records.npz"), codes=np.concatenate(codes), scale=np.concatenate(scale), basis=z8["basis"])
    json.dump(meta, open(os.path.join(BANK9, "records_meta.json"), "w"))
    json.dump(json.load(open(os.path.join(cb8.BANK8, "genes.json"))), open(os.path.join(BANK9, "genes.json"), "w"))
    b9 = cb4.Bank(BANK9); te = {G["KN"][int(i)] for i in G["rte"]}
    cov8 = sum(any(f"S:{n}" in b9.by_gene.get(g, {}) for n in SRC8) for g in te)
    cov9 = sum(any(f"S:{n}" in b9.by_gene.get(g, {}) for n in SRC9) for g in te)
    nb_cov = {}
    for n in NB:
        p = os.path.join(CACHE9, f"nb_{n}.npz")
        if os.path.exists(p):
            z = np.load(p); nb_cov[n] = dict(json.loads(str(z["meta"])), test_genes=len(te & set(z["genes"].tolist())))
    rep = dict(screens=rep, neighbour_sources=nb_cov, records=len(b9.gene), test_genes=len(te),
               test_genes_with_record_elsewhere_v8=cov8, test_genes_with_record_elsewhere_v9=cov9,
               bytes=int(b9.codes.nbytes + b9.scale.nbytes + b9.basis.nbytes))
    os.makedirs(OUTD9, exist_ok=True); json.dump(rep, open(os.path.join(OUTD9, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------------------
def nb9():
    return cb8.nb()


def tensors(G, bank_dir, src):
    """cellbank8.source_tensor on another bank / screen list (cellbank8 reads its module globals)."""
    old = (cb8.SRC, cb8.BANK8)
    cb8.SRC, cb8.BANK8 = list(src), bank_dir
    try:
        return cb8.source_tensor(G, cb4.Bank(bank_dir))
    finally:
        cb8.SRC, cb8.BANK8 = old


def translator_inputs(G, Z, Mk, Lg, St, t0, tag):
    """cellbank8's translator block: 5 out-of-fold translators for TRAIN rows + a final one for TEST rows."""
    torch = G["torch"]; KN, rtr, rte, split = G["KN"], G["rtr"], G["rte"], G["split"]
    S = Z.shape[0] - 1
    train_genes = [j for j, g in enumerate(KN) if split(g) == "TRAIN"]
    fold = lambda g: int(hashlib.sha256(("fold:" + g).encode()).hexdigest(), 16) % 5
    Tz = torch.zeros(G["NPH"], 128); Tn = torch.zeros(G["NPH"])
    tr_rows = [int(i) for i in rtr]
    for k in range(5):
        tr = cb8.train_translator(G, Z, Mk, Lg, [j for j in train_genes if fold(KN[j]) != k], seed=k)
        rows = [j for j in tr_rows if fold(KN[j]) == k]
        p, nin = cb8.translate(G, tr, Z, Mk, Lg, rows, S); Tz[rows] = p; Tn[rows] = nin
    final = cb8.train_translator(G, Z, Mk, Lg, train_genes, seed=99)
    te_rows = [int(i) for i in rte]
    p, nin = cb8.translate(G, final, Z, Mk, Lg, te_rows, S); Tz[te_rows] = p; Tn[te_rows] = nin
    has_T = Tn > 0
    Tu = torch.where(has_T[:, None], Tz / Tz.norm(dim=1, keepdim=True).clamp(min=1e-12), torch.zeros_like(Tz))
    flags = torch.cat([Mk[:S].T.float(), torch.log1p(Mk[:S].sum(0).float())[:, None], has_T.float()[:, None], Lg[:S].T], 1)
    print(f"  translator {tag} ({S} screens) done: {time.time() - t0:.0f}s", flush=True)
    return Tu, flags, has_T


def neighbours(G, name, K, hasK):
    import numpy as np
    torch = G["torch"]; NPH = G["NPH"]
    z = np.load(os.path.join(CACHE9, f"nb_{name}.npz")); pos = {g: j for j, g in enumerate(G["KN"])}
    V = torch.zeros(NPH, z["V"].shape[1]); has = torch.zeros(NPH, dtype=torch.bool)
    for g, v in zip(z["genes"].tolist(), z["V"].astype(np.float32)):
        if g in pos:
            V[pos[g]] = torch.tensor(v); has[pos[g]] = True
    C = V @ V.T
    C[:, ~(has & hasK)] = -2.0; C[~has] = -2.0; C.fill_diagonal_(-2.0)
    v, i = C.topk(NB_K, dim=1)
    w = torch.where(v > -1.5, torch.exp(NB_TEMP * v), torch.zeros_like(v))
    out = torch.zeros(NPH, 130); tot = w.sum(1, keepdim=True); ok = tot.squeeze(1) > 0
    out[ok, :128] = (w[ok, :, None] * K[i[ok]]).sum(1) / tot[ok]
    out[ok, 128] = v[ok, 0]; out[ok, 129] = 1.0
    return out, has


def train():
    t0 = time.time()
    G = nb9(); torch = G["torch"]
    os.makedirs(CACHE9, exist_ok=True); os.makedirs(OUTD9, exist_ok=True)
    bank6 = cb4.Bank(cb8.V6)
    rte = G["rte"]; te_rows = [int(i) for i in rte]
    Z8, Mk8, St8, Lg8 = tensors(G, cb8.BANK8, SRC8)
    Z9, Mk9, St9, Lg9 = tensors(G, BANK9, SRC9)
    Tu8, fl8, hT8 = translator_inputs(G, Z8, Mk8, Lg8, St8, t0, "v8")
    Tu9, fl9, hT9 = translator_inputs(G, Z9, Mk9, Lg9, St9, t0, "v9")
    K, hasK = cb8.k562_records(G, bank6)
    An8 = cb8.analog(G, Z8, Mk8, K, hasK); An9 = cb8.analog(G, Z9, Mk9, K, hasK)
    Pt, Ct = cb4.inputs(G, bank6, False)
    NBI, NBH = {}, {}
    for n in NB:
        NBI[n], NBH[n] = neighbours(G, n, K, hasK)
    print(f"  inputs ready ({time.time() - t0:.0f}s)", flush=True)
    arms = {"M2": [Tu8, fl8, An8, Pt, Ct], "M9L": [Tu9, fl9, An9, Pt, Ct],
            "M9N": [Tu8, fl8, An8, Pt, Ct] + [NBI[n] for n in NB], "M9": [Tu9, fl9, An9, Pt, Ct] + [NBI[n] for n in NB]}
    tz = {"M2": Tu8, "M9L": Tu9, "M9N": Tu8, "M9": Tu9}
    preds, info = {}, {}
    for arm, blocks in arms.items():
        X = cb4.scaled(G, blocks); outs = []
        for s in SEEDS:
            pz, best = cb8.train_residual(G, X, tz[arm].to(G["DEV"]), "mlp", s)
            outs.append(pz); info.setdefault(arm, []).append(round(best, 4))
            print(f"  {arm} seed {s}: held-out TRAIN r {best:+.3f} ({X.shape[1]} features, {time.time() - t0:.0f}s)", flush=True)
        preds[arm] = torch.stack(outs).mean(0)
    cov = dict(v8=Mk8[:len(SRC8), te_rows].any(0).tolist(), v9=Mk9[:len(SRC9), te_rows].any(0).tolist(),
               max_strength_v8=St8[:, te_rows].max(0).values.tolist(),
               **{f"screen_{n}": Mk9[s, te_rows].tolist() for s, n in enumerate(SRC9) if n in NEW},
               **{f"nb_{n}": NBH[n][te_rows].tolist() for n in NB})
    torch.save(dict(preds=preds, cov=cov, te_rows=te_rows), os.path.join(CACHE9, "train_out.pt"))
    json.dump(dict(heldout_train_r=info, seconds=round(time.time() - t0)), open(os.path.join(OUTD9, "train_info.json"), "w"), indent=1)
    print(f"train done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------------
# DEV (open) and EXAM (once)
# ---------------------------------------------------------------------------------------------------------
ARMS = ("M2", "M9L", "M9N", "M9")
LABEL = {"M2": "M2  cellbank8 model, rerun", "M9L": "M9L + LINCS screens", "M9N": "M9N + Tahoe/Geneformer/ARCHS4 neighbours",
         "M9": "M9  + both"}


def _report(half):
    import numpy as np
    G = nb9(); torch = G["torch"]
    o = torch.load(os.path.join(CACHE9, "train_out.pt")); preds, cov = o["preds"], o["cov"]
    score, ceil, ceil_s = cb8.scorer(G)
    genes, dv, ex, n_seen = cb8.sets(G)
    m = dv if half == "DEV" else ex
    S = {a: score(preds[a]) for a in ARMS}
    c8 = np.array(cov["v8"]); c9 = np.array(cov["v9"])
    ms = np.array(cov["max_strength_v8"]); weak = ms < np.median(ms)      # median over all TEST genes, both halves
    o8 = torch.load(os.path.join(cb8.CACHE, "train_out.pt"))["preds"]["M2"]
    r_old = score(o8)[0]
    lines = []
    P_ = lambda s="": (print(s, flush=True), lines.append(s))
    P_(f"  {half} genes {m.sum()}; ceiling r {np.median(ceil[m]):.3f}, specific {np.median(ceil_s[m]):.3f}")
    P_("  coverage on these genes: " + ", ".join([f"record elsewhere v8 {int(c8[m].sum())}, v9 {int(c9[m].sum())}"] +
                                             [f"{k[3:] if k.startswith('nb_') else k[7:]} {int(np.array(cov[k])[m].sum())}" for k in cov if k.startswith(("nb_", "screen_"))]))
    P_(f"     {'arm':42s} median r | specific r | fraction of ceiling")
    for a in ARMS:
        r, rs = S[a]
        P_(f"     {LABEL[a]:42s} {np.median(r[m]):+.3f} | {np.median(rs[m]):+.3f} | {np.median(r[m]) / np.median(ceil[m]):.0%}")
    P_(f"     {'(cellbank8 M2 as scored there)':42s} {np.median(r_old[m]):+.3f}")
    for lab, mm in (("weak-record half (v8)", m & weak), ("strong-record half (v8)", m & ~weak),
                    ("with no record elsewhere (v8)", m & ~c8), ("with none (v8) but a LINCS record", m & ~c8 & c9)):
        P_(f"     genes {lab:34s} n {mm.sum():4d}: " + ", ".join(f"{a} {np.median(S[a][0][mm]):+.3f}" for a in ARMS))
    return G, S, m, weak, lines


def dev():
    out = []
    P_ = lambda s="": (print(s, flush=True), out.append(s))
    P_(RULE); P_("CELLBANK9 -- engine v2 + the Drive datasets: DEVELOPMENT SET (open)"); P_(RULE)
    G, S, m, weak, lines = _report("DEV"); out += lines
    for a in ("M9", "M9L", "M9N"):
        for lab, mm in (("all", m), ("weak half", m & weak)):
            v = cb8.verdict(S[a][0][mm], S["M2"][0][mm])
            P_(f"     DEV {a:4s} vs M2, {lab:9s}: better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g}")
    open(os.path.join(HERE, "RESULTS_cellbank9_dev.txt"), "w").write("\n".join(out) + "\n")


def exam():
    lock = os.path.join(OUTD9, "exam_lock.json")
    if os.path.exists(lock):
        print("EXAM already scored for cellbank9 (exam_lock.json exists); refusing to score it again."); return
    import numpy as np
    out = []
    P_ = lambda s="": (print(s, flush=True), out.append(s))
    P_(RULE); P_("CELLBANK9 -- THE EXAM HALF, SECOND READ (scored once for these gates)"); P_(RULE)
    G, S, m, weak, lines = _report("EXAM"); out += lines
    torch = G["torch"]
    preds = torch.load(os.path.join(CACHE9, "train_out.pt"))["preds"]
    digest = hashlib.sha256(b"".join(preds[a].numpy().tobytes() for a in ARMS)).hexdigest()
    V = {"Y1": cb8.verdict(S["M9"][0][m], S["M2"][0][m]), "Y2": cb8.verdict(S["M9"][0][m & weak], S["M2"][0][m & weak]),
         "Y3": cb8.verdict(S["M9"][1][m], S["M2"][1][m])}
    for tag, lab in (("Y1", "PRIMARY M9 vs M2, r"), ("Y2", "M9 vs M2, r, weak-record half"), ("Y3", "M9 vs M2, specific r")):
        v = V[tag]
        P_(f"  {tag} {lab:34s} better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g} -> {v['verdict']}")
    for a in ("M9L", "M9N"):
        v = cb8.verdict(S[a][0][m], S["M2"][0][m])
        P_(f"  (descriptive) {a} vs M2, r: better {v['better']} / worse {v['worse']}, median {v['median_diff']:+.4f}, p {v['p']:.2g}")
    open(OUT9, "w").write("\n".join(out) + "\n")
    json.dump(dict(verdicts=V, predictions_sha256=digest, genes=int(m.sum()), exam_read="second (first: cellbank8)",
                   scored=time.strftime("%Y-%m-%d %H:%M")), open(lock, "w"), indent=1)


if __name__ == "__main__":
    {"pull": pull, "build": build, "train": train, "dev": dev, "exam": exam}[sys.argv[1]]()
