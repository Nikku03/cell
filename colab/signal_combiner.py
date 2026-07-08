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


class SignalCombiner:
    def __init__(self, C=None, dm=None, use_depmap=True,
                 string_links=None, string_aliases=None, geneformer_npz=None):
        self.C = C or CompleteCell()
        if hasattr(self.C, "_reset_base"):
            self.C._reset_base()
        self._coexpr = self._nbr_map(self.C.D.get("coexpr", {}))
        self._codep = self._nbr_map(self.C.D.get("codep", {}))
        self._cplx = {int(k): set(v) for k, v in (self.C.D.get("gene2cplx", {}) or {}).items()}
        self.dm = dm
        if self.dm is None and use_depmap:
            try:
                from phase3_depmap import DepMapEdges
                self.dm = DepMapEdges()
            except Exception as e:
                print("  (co-essentiality feature off — DepMap CSV not found:", str(e)[:50], ")")
        # optional dense features (auto-on when the Drive files are present)
        self.string = self._load_string(string_links or os.environ.get("STRING_LINKS"),
                                         string_aliases or os.environ.get("STRING_ALIASES"))
        self.emb = self._load_emb(geneformer_npz or os.environ.get("GENEFORMER_NPZ"))
        self.expr = self._load_expr(os.environ.get("EXPR_MATRIX"))
        self.features_list = list(CORE)
        if self.string is not None:
            self.features_list.append("string_score")
        if self.emb is not None:
            self.features_list.append("embedding_cos")
        if self.expr is not None:
            self.features_list.append("expr_corr")

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
        if not path or not os.path.exists(path):
            return None
        try:
            import pandas as pd
            df = pd.read_csv(path, index_col=0)
            idx = self.C.idx
            col_hit = sum(1 for c in df.columns if c in idx)
            row_hit = sum(1 for r in df.index if r in idx)
            if col_hit >= row_hit:                             # columns are genes -> want genes as rows
                df = df.T
            df = df.loc[[g for g in df.index if g in idx]]     # keep only rows that are known genes
            if df.shape[0] < 50:
                print("  (expr_corr off: too few genes matched)")
                return None
            X = df.to_numpy(dtype=np.float32)                  # genes x samples
            mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True); sd[sd == 0] = 1
            Z = (X - mu) / sd; Z[np.isnan(Z)] = 0.0
            Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            vecs = {idx[g]: Z[i] for i, g in enumerate(df.index)}
            print(f"  expr_corr loaded: {len(vecs):,} genes x {X.shape[1]} samples")
            return vecs
        except Exception as e:
            print("  (expr_corr off:", str(e)[:60], ")")
            return None

    # ---------- features ----------
    def _sym(self, m, a, b):
        return max(m.get(a, {}).get(b, 0.0), m.get(b, {}).get(a, 0.0))

    def features(self, a, b):
        adj = self.C.ppi_adj
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
        return f

    # ---------- labelled examples ----------
    def build_examples(self, n_pos=6000, seed=0):
        rng = np.random.default_rng(seed)
        adj = self.C.ppi_adj; name = self.C.name; n = len(name)
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

    # ---------- train + honest evaluation ----------
    def train(self, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve
        X, y, counts = self.build_examples(seed=seed)
        F = self.features_list
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        ablation = {}
        for j, f in enumerate(F):
            try:
                m = LogisticRegression(max_iter=500).fit(Xtr_s[:, [j]], ytr)
                ablation[f] = round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, [j]])[:, 1])), 3)
            except Exception:
                ablation[f] = None
        struct_idx = [F.index(x) for x in ("shared_partners", "jaccard") if x in F]
        indep_idx = [i for i in range(len(F)) if i not in struct_idx]
        def grp(idx):
            m = LogisticRegression(max_iter=800).fit(Xtr_s[:, idx], ytr)
            return round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, idx])[:, 1])), 3)
        auc_struct, auc_indep = grp(struct_idx), grp(indep_idx)
        lr = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08).fit(Xtr, ytr)
        auc_lr = float(roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1]))
        auc_gb = float(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]))
        best, best_auc, uses_scaler = (gb, auc_gb, False) if auc_gb >= auc_lr else (lr, auc_lr, True)
        p_te = best.predict_proba(Xte_s if uses_scaler else Xte)[:, 1]
        brier = float(brier_score_loss(yte, p_te))
        prec, rec, thr = precision_recall_curve(yte, p_te)
        chosen, chosen_rec = 0.5, 0.0
        for p, r, t in zip(prec[:-1], rec[:-1], thr):
            if p >= 0.9:
                chosen, chosen_rec = float(t), float(r); break
        self.model, self.scaler, self.uses_scaler, self.threshold = best, scaler, uses_scaler, chosen
        return dict(counts=counts, features=F, auc_logistic=round(auc_lr, 3), auc_gbm=round(auc_gb, 3),
                    chosen_model="gbm" if not uses_scaler else "logistic", combined_auc=round(best_auc, 3),
                    per_feature_auc=ablation, auc_structural_only=auc_struct, auc_independent_only=auc_indep,
                    logistic_weights=dict(zip(F, [round(float(c), 3) for c in lr.coef_[0]])),
                    brier=round(brier, 4), threshold_at_0p9_precision=round(chosen, 3),
                    recall_at_that_threshold=round(chosen_rec, 3),
                    note="hard negatives force use of INDEPENDENT signals; grouped structural-vs-independent AUC "
                         "shows how much is the graph confirming itself vs genuinely independent evidence")

    def save(self, res):
        pickle.dump(dict(model=self.model, scaler=self.scaler, uses_scaler=self.uses_scaler,
                         features=self.features_list, threshold=self.threshold,
                         use_depmap=self.dm is not None, has_string=self.string is not None,
                         has_emb=self.emb is not None),
                    open("outputs/orphan/signal_combiner.pkl", "wb"))
        json.dump(res, open("outputs/orphan/signal_combiner_validation.json", "w"), indent=2)

    def proba(self, a, b):
        x = np.array([self.features(a, b)])
        return float(self.model.predict_proba(self.scaler.transform(x) if self.uses_scaler else x)[0, 1])


def load(path="outputs/orphan/signal_combiner.pkl"):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def main():
    print("=" * 84)
    print("SIGNAL COMBINER — train one calibrated edge-probability from many independent signals")
    print("=" * 84)
    sc = SignalCombiner()
    print("features:", sc.features_list)
    res = sc.train()
    sc.save(res)
    print(f"examples: {res['counts']}")
    print("\nper-feature AUC (each evidence source ALONE):")
    for f, a in res["per_feature_auc"].items():
        print(f"    {f:16} {a}")
    print(f"\nGROUPED (honest): structural-only {res['auc_structural_only']}  vs  independent-only {res['auc_independent_only']}")
    print(f"combined AUC: logistic {res['auc_logistic']}  |  gbm {res['auc_gbm']}  -> using {res['chosen_model']}")
    print(f"calibration (Brier): {res['brier']}   threshold@0.9-precision: {res['threshold_at_0p9_precision']} "
          f"(recall {int(100*res['recall_at_that_threshold'])}%)")
    print("\n-> outputs/orphan/signal_combiner.pkl  +  signal_combiner_validation.json")
    return res


if __name__ == "__main__":
    main()
