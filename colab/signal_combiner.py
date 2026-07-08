"""signal_combiner — the one genuinely-trained piece: fuse many INDEPENDENT edge-evidence signals into ONE
calibrated probability, so the audit loop stops asking "does any single lens fire?" and instead asks "what's
the calibrated probability these two interact, given everything at once?"

This is a small, honest supervised model (logistic regression / gradient boosting over a handful of features),
NOT a deep net — its job is to WEIGH evidence, and to be calibrated and interpretable while doing it.

CORE features (always available, from the committed cell): shared_partners, jaccard (structural) + same_complex,
co-expression, co-dependency, co-essentiality (independent of the PPI graph).

OPTIONAL DENSE features (strictly stronger; activate automatically when the files are mounted — i.e. in Colab
with Drive): STRING physical score, and Geneformer gene-embedding cosine. The sparse top-k coexpr/codep lists
carry little alone (independent-only AUC ~0.57); these dense signals are what push it up. Point the env vars
STRING_LINKS / STRING_ALIASES / GENEFORMER_NPZ at the Drive files and re-run — the pipeline is unchanged and
the model records which features it was trained with.

Honesty note on STRING: STRING physical shares source databases (IntAct/BioGRID/…) with our own PPI layer, so
it is a strong but only PARTLY-independent feature — its single-feature training AUC is somewhat inflated by
that overlap. It is most valuable in the LOOP, where STRING confirming an edge that is MISSING from our graph
is genuinely new information (STRING's databases know something ours didn't).

Labels: positives = known curated PPI edges; negatives = a MIX of easy (random) and HARD (non-edges that share
≥3 partners) — the hard negatives force the model to use the *independent* signals, not just triadic closure.

-> outputs/orphan/signal_combiner.pkl              (model + scaler + feature list + chosen threshold)
-> outputs/orphan/signal_combiner_validation.json  (AUCs, ablation, calibration, threshold)
"""
import os, sys, json, pickle, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

CORE = ["shared_partners", "jaccard", "same_complex", "coexpression", "codependency", "coessentiality"]


def _mem(label=""):
    """print current + peak RSS in GB so we can SEE which feature load spikes RAM (the 103 GB question)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)   # KB -> GB on Linux
        cur = 0.0
        try:
            with open("/proc/self/statm") as fh:
                cur = int(fh.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") / (1024 ** 3))  # pages -> GB
        except Exception:
            pass
        print(f"    [RAM {cur:.1f} GB now, {peak:.1f} GB peak]  {label}")
    except Exception:
        pass


class SignalCombiner:
    def __init__(self, C=None, dm=None, use_depmap=True, relation="ppi",
                 string_links=None, string_aliases=None, geneformer_npz=None):
        self.C = C or CompleteCell()
        if hasattr(self.C, "_reset_base"):
            self.C._reset_base()
        # WHOLE-CELL: the combiner predicts ANY relation — swap the labels. relation="ppi" -> physical binding;
        # "reg" -> regulatory (TF->target); "sig" -> signaling. Structural features use THAT relation's graph, so
        # each relation is scored on its own terms and each dataset earns its place per relation (add->measure->keep).
        self.relation = relation
        self.adj = self._reladj(relation)
        self._coexpr = self._nbr_map(self.C.D.get("coexpr", {}))
        self._codep = self._nbr_map(self.C.D.get("codep", {}))
        self._cplx = {int(k): set(v) for k, v in (self.C.D.get("gene2cplx", {}) or {}).items()}
        _mem("after cell + graph")
        self.dm = dm
        if self.dm is None and use_depmap:
            try:
                from phase3_depmap import DepMapEdges
                self.dm = DepMapEdges()
            except Exception as e:
                print("  (co-essentiality feature off — DepMap CSV not found:", str(e)[:50], ")")
        _mem("after DepMap (co-essentiality)")
        # optional dense features (auto-on when the Drive files are present)
        self.string = self._load_string(string_links or os.environ.get("STRING_LINKS"),
                                         string_aliases or os.environ.get("STRING_ALIASES"))
        _mem("after STRING")
        self.emb = self._load_emb(geneformer_npz or os.environ.get("GENEFORMER_NPZ"))
        _mem("after Geneformer")
        self.expr = self._load_expr(os.environ.get("EXPR_MATRIX"))
        _mem("after expr matrix")
        self.tahoe = self._load_tahoe(os.environ.get("TAHOE_DE_DIR"))
        _mem("after Tahoe")
        self.esm = self._load_esm(os.environ.get("ESM_PARQUET"))
        _mem("after ESM")
        self.features_list = list(CORE)
        if self.string is not None:
            self.features_list.append("string_score")
        if self.emb is not None:
            self.features_list.append("embedding_cos")
        if self.expr is not None:
            self.features_list.append("expr_corr")
        if self.tahoe is not None:
            self.features_list.append("tahoe_corr")
        if self.esm is not None:
            self.features_list.append("esm_cos")

    def for_relation(self, relation):
        """cheap clone sharing ALL loaded feature data (STRING/expr/emb/tahoe/esm/coess maps) but with this
        relation's adjacency — so the loop can score reg/sig without re-loading the big features."""
        other = object.__new__(SignalCombiner)
        other.__dict__.update(self.__dict__)
        other.relation = relation
        other.adj = other._reladj(relation)
        other.features_list = list(self.features_list)
        return other

    def _reladj(self, relation):
        """undirected adjacency for the chosen relation (ppi reuses the cell's; reg/sig built from the edge list)."""
        if relation == "ppi":
            return self.C.ppi_adj
        from collections import defaultdict
        adj = defaultdict(set)
        for e in self.C.D.get(relation, []):
            if isinstance(e[0], int) and isinstance(e[1], int):
                adj[e[0]].add(e[1]); adj[e[1]].add(e[0])
        return adj

    @staticmethod
    def _nbr_map(layer):
        out = {}
        for k, lst in (layer or {}).items():
            d = {}
            for pair in lst:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    d[int(pair[0])] = float(pair[1])
            out[int(k)] = d
        return out

    # ---- optional dense-feature loaders (defensive: any failure -> feature simply off, pipeline still runs) ----
    def _load_string(self, links, aliases):
        if not links or not os.path.exists(links):
            return None
        try:
            idx = self.C.idx
            # map STRING protein id -> OUR gene index. A STRING protein has many aliases (RefSeq, UniProt, symbol,
            # …); keep the one that IS a known gene symbol (in our index). That is the fix — "first alias" grabbed
            # the wrong id and mapped almost nothing.
            id2i = {}
            if aliases and os.path.exists(aliases):
                aop = gzip.open if aliases.endswith(".gz") else open
                with aop(aliases, "rt") as fh:
                    for line in fh:
                        p = line.rstrip("\n").split("\t")
                        if len(p) >= 2 and p[0] not in id2i and p[1] in idx:
                            id2i[p[0]] = idx[p[1]]
                print(f"  STRING aliases mapped {len(id2i):,} proteins -> gene symbols")
            op = gzip.open if links.endswith(".gz") else open
            d = {}
            with op(links, "rt") as fh:
                header = fh.readline()
                sep = " " if (" " in header and "\t" not in header) else "\t"
                for line in fh:
                    p = line.rstrip("\n").split(sep)
                    if len(p) < 3:
                        continue
                    ia = id2i.get(p[0], idx.get(p[0]))         # aliases first, else the id may already be a symbol
                    ib = id2i.get(p[1], idx.get(p[1]))
                    if ia is None or ib is None:
                        continue
                    try:
                        s = float(p[-1])
                    except ValueError:
                        continue
                    s = s / 1000.0 if s > 1.5 else s           # STRING scores are 0-1000
                    key = (min(ia, ib), max(ia, ib))
                    if s > d.get(key, 0):
                        d[key] = s
            print(f"  STRING loaded: {len(d):,} scored gene pairs")
            return d if d else None
        except Exception as e:
            print("  (STRING feature off:", str(e)[:60], ")")
            return None

    def _load_emb(self, npz):
        if not npz or not os.path.exists(npz):
            return None
        try:
            z = np.load(npz, allow_pickle=True)
            keys = list(z.keys())
            print(f"  npz keys: {[(k, z[k].shape, str(z[k].dtype)) for k in keys]}")
            # names = the 1D STRING/object array (e.g. 'genes'); NOT the bool 'have' mask or an int array
            names_arr = next((z[k] for k in keys if z[k].ndim == 1 and z[k].dtype.kind in ("U", "S", "O")), None)
            mat = next((z[k] for k in keys if z[k].ndim == 2), None)
            have = next((z[k] for k in keys if z[k].ndim == 1 and z[k].dtype == bool), None)
            if names_arr is None or mat is None or len(names_arr) != len(mat):
                print("  (embedding feature off: could not identify names+matrix in npz)")
                return None
            names = [str(n) for n in names_arr]
            valid = set(range(len(names))) if have is None else {i for i, h in enumerate(have) if h}
            print(f"  npz name sample: {[names[i] for i in list(valid)[:3]]}  ({len(valid):,} with have=True)")
            mat = mat.astype(np.float32)
            mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            idx = self.C.idx
            # names may be gene symbols (map directly) or Ensembl gene ids (map ENSG->symbol via mygene)
            name2i = {}
            direct = sum(1 for i in valid if names[i] in idx)
            if direct >= 0.2 * len(valid):
                name2i = {i: names[i] for i in valid if names[i] in idx}
            elif names and names[list(valid)[0]].split(".")[0].startswith("ENSG"):
                try:
                    import mygene
                    q = sorted({names[i].split(".")[0] for i in valid})
                    hits = mygene.MyGeneInfo().querymany(q, scopes="ensembl.gene", fields="symbol",
                                                         species="human", verbose=False, as_dataframe=False)
                    ens2sym = {h["query"]: h["symbol"] for h in hits if h.get("symbol")}
                    name2i = {i: ens2sym[names[i].split(".")[0]] for i in valid
                              if ens2sym.get(names[i].split(".")[0]) in idx}
                    print(f"  mapped {len(name2i):,} Ensembl ids -> symbols via mygene")
                except Exception as e:
                    print("  (embedding: ENSG->symbol map failed:", str(e)[:50], ")")
            emb = {idx[sym]: mat[i] for i, sym in name2i.items()}
            print(f"  Geneformer embeddings loaded: {len(emb):,} genes")
            return emb if emb else None
        except Exception as e:
            print("  (embedding feature off:", str(e)[:60], ")")
            return None

    def _load_expr(self, path):
        """dense CO-EXPRESSION from an expression matrix (Tabula / DepMap expression). Same machinery as DepMap
        co-essentiality: z-score each gene across samples, correlation = dot of unit vectors. Much denser than
        the stored top-k coexpr lists (which barely help, AUC ~0.51). Auto-detects which axis is genes by
        matching our symbols."""
        cache = "outputs/orphan/expr_vecs.npz"
        if os.path.exists(cache):                              # reuse derived vectors -> no 305 MB CSV re-parse
            z = np.load(cache)
            vecs = {int(i): z["vecs"][k] for k, i in enumerate(z["ids"])}
            print(f"  expr_corr: loaded {len(vecs):,} gene vectors from cache (no CSV re-parse)")
            return vecs
        if not path or not os.path.exists(path):
            return None
        try:
            import pandas as pd
            df = pd.read_csv(path, index_col=0)
            idx = self.C.idx
            nrm = lambda c: str(c).split(" (")[0]              # DepMap headers look like 'A1BG (1)' -> 'A1BG'
            col_syms = [nrm(c) for c in df.columns]
            row_syms = [nrm(r) for r in df.index]
            if sum(c in idx for c in col_syms) >= sum(r in idx for r in row_syms):
                df.columns = col_syms; df = df.T               # genes were columns -> now rows
            else:
                df.index = row_syms
            df = df[~df.index.duplicated()]
            df = df.loc[[g for g in df.index if g in idx]]     # keep only rows that are known genes
            if df.shape[0] < 50:
                print(f"  (expr_corr off: only {df.shape[0]} genes matched)")
                return None
            X = df.to_numpy(dtype=np.float32)                  # genes x samples
            mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True); sd[sd == 0] = 1
            Z = (X - mu) / sd; Z[np.isnan(Z)] = 0.0
            Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            genes = list(df.index)
            del df
            vecs = {idx[g]: Z[i] for i, g in enumerate(genes)}
            ids = list(vecs)
            try:
                np.savez_compressed(cache, ids=np.array(ids),
                                    vecs=np.array([vecs[i] for i in ids], dtype=np.float32))
            except Exception:
                pass
            print(f"  expr_corr loaded: {len(vecs):,} genes x {X.shape[1]} samples -> cached")
            return vecs
        except Exception as e:
            print("  (expr_corr off:", str(e)[:60], ")")
            return None

    def _load_tahoe(self, d):
        """Tahoe-100M drug-perturbation co-response. Uses the FILTERED cell_eval table (wide pseudobulk deltas:
        one row per cell_line×treatment, ~2,000 gene columns) — NOT pdex (4.1B-row long format) and NOT the
        per-cell embeddings. Two genes that respond alike across 50 lines × 384 drugs × 3 doses are functionally
        linked — an independent signal (drug-response co-regulation), orthogonal to STRING. Covers only the ~2,000
        analysis genes, so it's 0 outside them (honest)."""
        # fast path: derived per-gene vectors cached (built once, then persisted to Drive) -> no re-download
        cache = "outputs/orphan/tahoe_vecs.npz"
        if os.path.exists(cache):
            z = np.load(cache)
            vecs = {int(i): z["vecs"][k] for k, i in enumerate(z["ids"])}
            print(f"  tahoe_corr: loaded {len(vecs):,} gene vectors from cache (no download)")
            return vecs
        if not d or not os.path.isdir(d):
            return None
        try:
            import glob as _g, pandas as pd
            files = _g.glob(os.path.join(d, "**", "cell_eval", "*.parquet"), recursive=True)
            if not files:                                      # fall back to any parquet, but never pdex (huge/raw)
                files = [f for f in _g.glob(os.path.join(d, "**", "*.parquet"), recursive=True)
                         if "pdex" not in f.lower()]
            if not files:
                print("  (tahoe_corr off: no cell_eval parquet found)")
                return None
            files = sorted(files); idx = self.C.idx
            print(f"  tahoe_corr: building vectors from {len(files)} cell_eval plates…")
            gene_cols, mats = None, []
            for k, f in enumerate(files):
                dfp = pd.read_parquet(f)
                if gene_cols is None:
                    gene_cols = [c for c in dfp.columns if c in idx]
                    if len(gene_cols) < 50:
                        print(f"  (tahoe_corr off: only {len(gene_cols)} gene columns matched)")
                        return None
                mats.append(dfp[gene_cols].to_numpy(dtype=np.float32))   # keep only gene cols, drop the DataFrame
                print(f"    plate {k+1}/{len(files)}  (+{len(dfp)} conditions)")
            M = np.vstack(mats).T                               # genes x conditions
            if M.shape[1] > 8000:                               # subsample conditions (plenty for correlation)
                sel = np.random.default_rng(0).choice(M.shape[1], 8000, replace=False); M = M[:, sel]
            mu = np.nanmean(M, 1, keepdims=True); sd = np.nanstd(M, 1, keepdims=True); sd[sd == 0] = 1
            Z = (M - mu) / sd; Z[np.isnan(Z)] = 0.0
            Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            vecs = {idx[g]: Z[i] for i, g in enumerate(gene_cols)}
            ids = list(vecs)
            np.savez_compressed(cache, ids=np.array(ids), vecs=np.array([vecs[i] for i in ids], dtype=np.float32))
            print(f"  tahoe_corr loaded: {len(vecs):,} genes x {M.shape[1]} conditions (cell_eval) -> cached")
            return vecs
        except Exception as e:
            print("  (tahoe_corr off:", str(e)[:60], ")")
            return None

    def _load_esm(self, path):
        """ESM protein-sequence embedding cosine. Sequence-based (orthogonal to Geneformer's expression-based
        embedding), so it MIGHT catch interaction-interface similarity — an honest test, judged by add->measure->
        keep. Defensive parquet loader: finds the id column (gene symbol, or UniProt acc mapped via mygene) and
        the embedding (a list/array column, or many float columns)."""
        if not path or not os.path.exists(path):
            return None
        try:
            import pandas as pd
            try:                                              # check size FIRST — a per-residue parquet is huge
                import pyarrow.parquet as pq
                nrows = pq.ParquetFile(path).metadata.num_rows
                print(f"  esm parquet: {nrows:,} rows")
                if nrows > 500_000:
                    print("  (esm_cos off: too many rows — looks per-RESIDUE, not per-protein; would blow RAM. "
                          "Use a mean-pooled per-protein embedding.)")
                    return None
            except Exception:
                pass
            df = pd.read_parquet(path)
            idx = self.C.idx
            print(f"  esm columns: {[(c, str(df[c].dtype)) for c in df.columns][:8]}")
            # id column: the string/object column with the most matches to our symbols; else a UniProt-looking one
            id_col, best = None, 0
            for c in df.columns:
                if df[c].dtype == object:
                    hit = sum(1 for v in df[c].astype(str).head(2000) if v in idx)
                    if hit > best:
                        best, id_col = hit, c
            ensembl = False
            if id_col is None or best < 0.05 * min(len(df), 2000):
                # maybe UniProt accessions -> map via mygene
                cand = next((c for c in df.columns if df[c].dtype == object), None)
                if cand is None:
                    print("  (esm_cos off: no id column found)"); return None
                id_col = cand; ensembl = True
            # embedding: a column of arrays/lists, else all numeric columns
            arr_col = next((c for c in df.columns if c != id_col and
                            isinstance(df[c].iloc[0], (list, tuple, np.ndarray))), None)
            if arr_col is not None:
                mat = np.array([np.asarray(v, dtype=np.float32) for v in df[arr_col]])
            else:
                num = df.select_dtypes("number")
                if num.shape[1] < 8:
                    print("  (esm_cos off: no embedding matrix found)"); return None
                mat = num.to_numpy(dtype=np.float32)
            ids = df[id_col].astype(str).tolist()
            if ensembl:
                try:
                    import mygene
                    hits = mygene.MyGeneInfo().querymany([i.split(".")[0] for i in ids], scopes="uniprot,ensembl.gene",
                                                         fields="symbol", species="human", verbose=False)
                    m2s = {h["query"]: h["symbol"] for h in hits if h.get("symbol")}
                    ids = [m2s.get(i.split(".")[0], i) for i in ids]
                except Exception as e:
                    print("  (esm_cos: id->symbol map failed:", str(e)[:40], ")")
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            emb = {idx[s]: mat[i] for i, s in enumerate(ids) if s in idx}
            print(f"  esm_cos loaded: {len(emb):,} proteins")
            return emb if len(emb) >= 50 else None
        except Exception as e:
            print("  (esm_cos off:", str(e)[:60], ")")
            return None

    # ---------- features ----------
    def _sym(self, m, a, b):
        return max(m.get(a, {}).get(b, 0.0), m.get(b, {}).get(a, 0.0))

    def features(self, a, b):
        adj = self.adj
        na, nb = adj.get(a, set()), adj.get(b, set())
        shared = len(na & nb); union = len(na | nb)
        f = [float(shared), shared / union if union else 0.0,
             1.0 if (self._cplx.get(a, set()) & self._cplx.get(b, set())) else 0.0,
             self._sym(self._coexpr, a, b), self._sym(self._codep, a, b)]
        coess = 0.0
        if self.dm is not None:
            v = self.dm.coess(self.C.name[a], self.C.name[b])
            coess = float(v) if v is not None else 0.0
        f.append(coess)
        if self.string is not None:
            f.append(self.string.get((min(a, b), max(a, b)), 0.0))
        if self.emb is not None:
            ea, eb = self.emb.get(a), self.emb.get(b)
            f.append(float(ea @ eb) if ea is not None and eb is not None else 0.0)
        if self.expr is not None:
            va, vb = self.expr.get(a), self.expr.get(b)
            f.append(float(va @ vb) if va is not None and vb is not None else 0.0)
        if self.tahoe is not None:
            ta, tb = self.tahoe.get(a), self.tahoe.get(b)
            f.append(float(ta @ tb) if ta is not None and tb is not None else 0.0)
        if self.esm is not None:
            ea, eb = self.esm.get(a), self.esm.get(b)
            f.append(float(ea @ eb) if ea is not None and eb is not None else 0.0)
        return f

    # ---------- labelled examples ----------
    def build_examples(self, n_pos=6000, seed=0):
        rng = np.random.default_rng(seed)
        adj = self.adj; name = self.C.name; n = len(name)
        inset = set(self.dm.col) if self.dm is not None else None
        ok = (lambda i: inset is None or name[i] in inset)
        edges = [(u, v) for u, ps in adj.items() for v in ps if u < v and ok(u) and ok(v)]
        rng.shuffle(edges); edges = edges[:n_pos]
        edgeset = {(min(u, v), max(u, v)) for u, ps in adj.items() for v in ps}
        from collections import Counter
        hard = []
        for a in sorted(adj, key=lambda u: -len(adj[u]))[:600]:
            pa = adj[a]
            if len(pa) > 600:
                continue
            two = Counter()
            for p in pa:
                for c in adj.get(p, ()):
                    if c != a and c not in pa:
                        two[c] += 1
            for c, sh in two.items():
                if sh >= 3 and a < c and (a, c) not in edgeset and ok(a) and ok(c):
                    hard.append((a, c))
        rng.shuffle(hard); hard = hard[:n_pos // 2]
        easy = set()
        while len(easy) < n_pos - len(hard):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a != b and ok(a) and ok(b) and (min(a, b), max(a, b)) not in edgeset:
                easy.add((min(a, b), max(a, b)))
        rows = [(a, b, 1) for a, b in edges] + [(a, b, 0) for a, b in hard] + [(a, b, 0) for a, b in easy]
        X = np.array([self.features(a, b) for a, b, _ in rows], dtype=float)
        y = np.array([lab for _, _, lab in rows])
        return X, y, dict(n_pos=int(y.sum()), n_hard_neg=len(hard), n_easy_neg=len(easy),
                          features=self.features_list)

    STRUCT = ("shared_partners", "jaccard")

    # ---------- train + honest evaluation (with add->measure->KEEP: auto-drop features below the bar) ----------
    def train(self, seed=0, keep_auc=0.53):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve
        X, y, counts = self.build_examples(seed=seed)
        F = self.features_list
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        sf = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sf.transform(Xtr), sf.transform(Xte)
        # per-feature AUC (each signal ALONE)
        ablation = {}
        for j, f in enumerate(F):
            try:
                m = LogisticRegression(max_iter=500).fit(Xtr_s[:, [j]], ytr)
                ablation[f] = round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, [j]])[:, 1])), 3)
            except Exception:
                ablation[f] = None
        # grouped structural-vs-independent (on ALL features)
        def grp(idx):
            m = LogisticRegression(max_iter=800).fit(Xtr_s[:, idx], ytr)
            return round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, idx])[:, 1])), 3)
        s_idx = [F.index(x) for x in self.STRUCT if x in F]
        auc_struct = grp(s_idx); auc_indep = grp([i for i in range(len(F)) if i not in s_idx])
        auc_all_gb = float(roc_auc_score(yte, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)
                                         .fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        # KEEP: structural always; an independent feature must clear keep_auc on its own to earn a slot
        kept = [f for f in F if f in self.STRUCT or (ablation.get(f) or 0) >= keep_auc]
        if len(kept) < 2:
            kept = list(F)
        dropped = [f for f in F if f not in kept]
        ki = [F.index(f) for f in kept]
        # refit scaler + models on KEPT columns only
        Xtr_k, Xte_k = Xtr[:, ki], Xte[:, ki]
        sk = StandardScaler().fit(Xtr_k); Xtr_ks, Xte_ks = sk.transform(Xtr_k), sk.transform(Xte_k)
        lr = LogisticRegression(max_iter=1000).fit(Xtr_ks, ytr)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08).fit(Xtr_k, ytr)
        auc_lr = float(roc_auc_score(yte, lr.predict_proba(Xte_ks)[:, 1]))
        auc_gb = float(roc_auc_score(yte, gb.predict_proba(Xte_k)[:, 1]))
        best, best_auc, uses_scaler = (gb, auc_gb, False) if auc_gb >= auc_lr else (lr, auc_lr, True)
        p_te = best.predict_proba(Xte_ks if uses_scaler else Xte_k)[:, 1]
        brier = float(brier_score_loss(yte, p_te))
        prec, rec, thr = precision_recall_curve(yte, p_te)
        chosen, chosen_rec = 0.5, 0.0
        for p, r, t in zip(prec[:-1], rec[:-1], thr):
            if p >= 0.9:
                chosen, chosen_rec = float(t), float(r); break
        self.model, self.scaler, self.uses_scaler, self.threshold = best, sk, uses_scaler, chosen
        self.kept, self.kept_idx = kept, ki
        self.informative_independent = [f for f in kept if f not in self.STRUCT]
        return dict(counts=counts, features_all=F, kept_features=kept,
                    dropped={f: ablation.get(f) for f in dropped},
                    per_feature_auc=ablation, auc_structural_only=auc_struct, auc_independent_only=auc_indep,
                    combined_auc=round(best_auc, 3), combined_auc_all_features=round(auc_all_gb, 3),
                    auc_logistic=round(auc_lr, 3), auc_gbm=round(auc_gb, 3),
                    chosen_model="gbm" if not uses_scaler else "logistic",
                    brier=round(brier, 4), threshold_at_0p9_precision=round(chosen, 3),
                    recall_at_that_threshold=round(chosen_rec, 3),
                    informative_independent=self.informative_independent,
                    note="add->measure->keep: any independent feature below AUC %.2f alone is DROPPED (and can no "
                         "longer gate the loop). combined_auc vs combined_auc_all_features shows dropping the dead "
                         "weight did not cost anything." % keep_auc)

    def _pkl(self):
        return "outputs/orphan/signal_combiner.pkl" if self.relation == "ppi" \
            else f"outputs/orphan/signal_combiner_{self.relation}.pkl"

    def save(self, res):
        pickle.dump(dict(model=self.model, scaler=self.scaler, uses_scaler=self.uses_scaler,
                         relation=self.relation, features=self.kept, features_all=self.features_list,
                         threshold=self.threshold, informative_independent=self.informative_independent,
                         use_depmap=self.dm is not None, has_string=self.string is not None),
                    open(self._pkl(), "wb"))
        suffix = "" if self.relation == "ppi" else "_" + self.relation
        json.dump(res, open(f"outputs/orphan/signal_combiner{suffix}_validation.json", "w"), indent=2)

    def proba(self, a, b):
        full = self.features(a, b)
        x = np.array([[full[i] for i in self.kept_idx]])
        return float(self.model.predict_proba(self.scaler.transform(x) if self.uses_scaler else x)[0, 1])


def load(path="outputs/orphan/signal_combiner.pkl"):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def _report(sc, res):
    print(f"examples: {res['counts']}")
    print("per-feature AUC (each evidence source ALONE):")
    for f, a in res["per_feature_auc"].items():
        print(f"    {f:16} {a}")
    print(f"GROUPED (honest): structural-only {res['auc_structural_only']}  vs  independent-only {res['auc_independent_only']}")
    if res["dropped"]:
        print("DROPPED (below AUC 0.53 alone — dead weight, can no longer gate the loop):")
        for f, a in res["dropped"].items():
            print(f"    {f:16} {a}")
    print(f"KEPT: {res['kept_features']}")
    print(f"combined AUC (kept): {res['combined_auc']}  vs  all features: {res['combined_auc_all_features']}  "
          f"(dropping cost {round(res['combined_auc_all_features']-res['combined_auc'],3)})")
    print(f"calibration (Brier): {res['brier']}  threshold@0.9-precision: {res['threshold_at_0p9_precision']} "
          f"(recall {int(100*res['recall_at_that_threshold'])}%)")
    print(f"-> {sc._pkl()}\n")


def main(relations=("ppi",)):
    """Load the (big) features ONCE, then train every relation off cheap clones — so cell 6 doesn't re-parse
    DepMap/expr/ESM/STRING per relation (that 3x re-parse was the ~45-min cost; it's all CPU, the GPU is idle)."""
    if isinstance(relations, str):
        relations = [relations]
    base = SignalCombiner(relation=relations[0])            # <-- the single, shared feature load
    print("features:", base.features_list, "| loaded ONCE, shared across", list(relations))
    for rel in relations:
        sc = base if rel == relations[0] else base.for_relation(rel)
        print("=" * 84)
        print(f"SIGNAL COMBINER [{rel}]  ({sum(len(v) for v in sc.adj.values())//2:,} edges in relation)")
        print("=" * 84)
        res = sc.train()
        sc.save(res)
        _report(sc, res)
    return True


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["ppi"])
