"""signal_combiner — the one genuinely-trained piece: fuse many INDEPENDENT edge-evidence signals into ONE
calibrated probability, so the audit loop stops asking "does any single lens fire?" and instead asks "what's
the calibrated probability these two interact, given everything at once?"

This is a small, honest supervised model (logistic regression / gradient boosting over a handful of features),
NOT a deep net — its job is to WEIGH evidence, and to be calibrated and interpretable while doing it.

Features per candidate gene pair (a,b):
  shared_partners  |N(a) ∩ N(b)|                         (structural — from the PPI graph)
  jaccard          shared / |N(a) ∪ N(b)|                (structural, degree-normalised)
  same_complex     1 if a,b share a curated complex      (curated, independent of the graph edges)
  coexpression     co-expression score if present        (independent)
  codependency     DepMap co-dependency score if present (independent)
  coessentiality   Pearson r of gene-effect profiles     (independent — needs the DepMap CSV)

Labels: positives = known curated PPI edges; negatives = a MIX of easy (random) and HARD (non-edges that share
≥3 partners) — the hard negatives force the model to use the *independent* signals, not just triadic closure
(otherwise it would trivially rate every triadic candidate as positive, which is useless inside the loop).

Honest reporting: per-feature single-signal AUC (ablation — shows which evidence actually carries), a grouped
STRUCTURAL-only vs INDEPENDENT-only AUC (shows how much is the graph confirming itself vs genuinely independent
evidence), combined AUC, calibration (Brier), and the precision/threshold trade-off.

Extending it: STRING physical score and Geneformer/ESM embedding cosine are strictly stronger, DENSE features
(the sparse top-k coexpr/codep lists carry little alone). They are on disk in Drive but too large to stream
here; add them as extra columns in features()/FEATURES when the files are mounted (e.g. in Colab) and retrain —
the pipeline is otherwise unchanged.

-> outputs/orphan/signal_combiner.pkl              (model + scaler + feature list + chosen threshold)
-> outputs/orphan/signal_combiner_validation.json  (AUCs, ablation, calibration, threshold)
"""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell


class SignalCombiner:
    FEATURES = ["shared_partners", "jaccard", "same_complex", "coexpression", "codependency", "coessentiality"]

    def __init__(self, use_depmap=True):
        self.C = CompleteCell()
        self.C._reset_base() if hasattr(self.C, "_reset_base") else None
        # build fast lookups for the independent signals
        self._coexpr = self._nbr_map(self.C.D.get("coexpr", {}))
        self._codep = self._nbr_map(self.C.D.get("codep", {}))
        self._cplx = {int(k): set(v) for k, v in (self.C.D.get("gene2cplx", {}) or {}).items()}
        self.dm = None
        if use_depmap:
            try:
                from phase3_depmap import DepMapEdges
                self.dm = DepMapEdges()
            except Exception as e:
                print("  (co-essentiality feature off — DepMap CSV not found:", str(e)[:50], ")")

    @staticmethod
    def _nbr_map(layer):
        """idx -> {partner_idx: score} for a top-k neighbour layer (coexpr/codep)."""
        out = {}
        for k, lst in (layer or {}).items():
            d = {}
            for pair in lst:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    d[int(pair[0])] = float(pair[1])
            out[int(k)] = d
        return out

    # ---------- features ----------
    def _sym(self, m, a, b):
        return max(m.get(a, {}).get(b, 0.0), m.get(b, {}).get(a, 0.0))

    def features(self, a, b):
        adj = self.C.ppi_adj
        na, nb = adj.get(a, set()), adj.get(b, set())
        shared = len(na & nb)
        union = len(na | nb)
        jac = shared / union if union else 0.0
        same_cplx = 1.0 if (self._cplx.get(a, set()) & self._cplx.get(b, set())) else 0.0
        coex = self._sym(self._coexpr, a, b)
        cod = self._sym(self._codep, a, b)
        coess = 0.0
        if self.dm is not None:
            v = self.dm.coess(self.C.name[a], self.C.name[b])
            coess = float(v) if v is not None else 0.0
        return [float(shared), float(jac), same_cplx, coex, cod, coess]

    # ---------- labelled examples ----------
    def build_examples(self, n_pos=6000, seed=0):
        rng = np.random.default_rng(seed)
        adj = self.C.ppi_adj; name = self.C.name; n = len(name)
        inset = set(self.dm.col) if self.dm is not None else None      # keep pairs scorable by co-essentiality

        def ok(i):
            return inset is None or name[i] in inset
        # positives: known PPI edges
        edges = [(u, v) for u, ps in adj.items() for v in ps if u < v and ok(u) and ok(v)]
        rng.shuffle(edges); edges = edges[:n_pos]
        pos = set(edges)
        edgeset = {(min(u, v), max(u, v)) for u, ps in adj.items() for v in ps}
        # HARD negatives: non-edges that share >=3 partners (structurally look like edges)
        hard = []
        hubs = sorted(adj, key=lambda u: -len(adj[u]))[:600]
        from collections import Counter
        for a in hubs:
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
        # EASY negatives: random non-edges
        easy = set()
        while len(easy) < n_pos - len(hard):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a == b or not ok(a) or not ok(b):
                continue
            key = (min(a, b), max(a, b))
            if key in edgeset:
                continue
            easy.add(key)
        rows = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in hard] + [(a, b, 0) for a, b in easy]
        X = np.array([self.features(a, b) for a, b, _ in rows], dtype=float)
        y = np.array([lab for _, _, lab in rows])
        return X, y, dict(n_pos=int(y.sum()), n_hard_neg=len(hard), n_easy_neg=len(easy))

    # ---------- train + honest evaluation ----------
    def train(self, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve
        X, y, counts = self.build_examples(seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        # per-feature ablation: single-signal AUC (how much each evidence source carries ALONE)
        ablation = {}
        for j, f in enumerate(self.FEATURES):
            try:
                m = LogisticRegression(max_iter=500).fit(Xtr_s[:, [j]], ytr)
                ablation[f] = round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, [j]])[:, 1])), 3)
            except Exception:
                ablation[f] = None
        # grouped ablation: STRUCTURAL-only vs INDEPENDENT-only — shows honestly how much of the signal is the
        # graph confirming itself vs genuinely independent evidence
        struct_idx = [0, 1]                                   # shared_partners, jaccard
        indep_idx = [2, 3, 4, 5]                              # same_complex, coexpr, codep, coessentiality
        def grouped_auc(idx):
            m = LogisticRegression(max_iter=800).fit(Xtr_s[:, idx], ytr)
            return round(float(roc_auc_score(yte, m.predict_proba(Xte_s[:, idx])[:, 1])), 3)
        auc_struct = grouped_auc(struct_idx)
        auc_indep = grouped_auc(indep_idx)
        # combined models
        lr = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08).fit(Xtr, ytr)
        auc_lr = float(roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1]))
        auc_gb = float(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]))
        best, best_auc, uses_scaler = (gb, auc_gb, False) if auc_gb >= auc_lr else (lr, auc_lr, True)
        p_te = best.predict_proba(Xte_s if uses_scaler else Xte)[:, 1]
        brier = float(brier_score_loss(yte, p_te))
        # choose the threshold at ~0.9 precision (so a "verified" edge is really verified)
        prec, rec, thr = precision_recall_curve(yte, p_te)
        target = 0.9; chosen = 0.5; chosen_rec = 0.0
        for p, r, t in zip(prec[:-1], rec[:-1], thr):
            if p >= target:
                chosen, chosen_rec = float(t), float(r); break
        coef = dict(zip(self.FEATURES, [round(float(c), 3) for c in lr.coef_[0]]))
        self.model = best; self.scaler = scaler; self.uses_scaler = uses_scaler; self.threshold = chosen
        res = dict(counts=counts,
                   auc_logistic=round(auc_lr, 3), auc_gbm=round(auc_gb, 3),
                   chosen_model="gbm" if not uses_scaler else "logistic", combined_auc=round(best_auc, 3),
                   per_feature_auc=ablation, auc_structural_only=auc_struct, auc_independent_only=auc_indep,
                   logistic_weights=coef, brier=round(brier, 4),
                   threshold_at_0p9_precision=round(chosen, 3), recall_at_that_threshold=round(chosen_rec, 3),
                   features=self.FEATURES,
                   note="hard negatives (non-edges sharing >=3 partners) force the model to use INDEPENDENT signals, "
                        "not just triadic closure; single-feature AUCs show which evidence actually carries")
        return res

    def save(self, res):
        pickle.dump(dict(model=self.model, scaler=self.scaler, uses_scaler=self.uses_scaler,
                         features=self.FEATURES, threshold=self.threshold, use_depmap=self.dm is not None),
                    open("outputs/orphan/signal_combiner.pkl", "wb"))
        json.dump(res, open("outputs/orphan/signal_combiner_validation.json", "w"), indent=2)

    # ---------- inference (used by the loop) ----------
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
    res = sc.train()
    sc.save(res)
    print(f"examples: {res['counts']}")
    print(f"\nper-feature AUC (how much each evidence source carries ALONE):")
    for f, a in res["per_feature_auc"].items():
        print(f"    {f:16} {a}")
    print(f"\nGROUPED (honest): structural-only AUC {res['auc_structural_only']}  vs  "
          f"independent-only AUC {res['auc_independent_only']}")
    print(f"logistic weights: {res['logistic_weights']}")
    print(f"\ncombined AUC: logistic {res['auc_logistic']}  |  gbm {res['auc_gbm']}  -> using {res['chosen_model']}")
    print(f"calibration (Brier, lower=better): {res['brier']}")
    print(f"threshold at 0.9 precision: {res['threshold_at_0p9_precision']}  "
          f"(catches {int(100*res['recall_at_that_threshold'])}% of true edges at that precision)")
    print("\n-> outputs/orphan/signal_combiner.pkl  +  signal_combiner_validation.json")
    return res


if __name__ == "__main__":
    main()
