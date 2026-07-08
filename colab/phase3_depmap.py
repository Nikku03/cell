"""Phase 3 — train the edge model on DepMap co-essentiality, then re-run the Phase-2 audit with it.

DepMap CRISPRGeneEffect is the Achilles gene-effect matrix (~1,150 cancer cell lines x ~18,000 genes): how much
knocking out each gene hurts each line. Two genes whose knockout-effect profiles track across lines are
CO-ESSENTIAL — a strong, independent signal of functional/physical interaction (orthogonal to the PPI databases
the model already has). This module:

  1. TRAIN  a logistic edge predictor on [co-essentiality corr, triadic shared-partners] with known PPI as
            positives and random non-edges as negatives; report held-out AUC and, crucially, whether DepMap
            ADDS over the triadic-closure signal alone (honest ablation).
  2. CORROBORATE  the Phase-2 cell-v2 additions (the 978 co-complex PPI edges) against DepMap co-essentiality —
            independently. Reported with the known caveat: co-essentiality sees CONTEXT-VARIABLE dependencies,
            and is blind to PAN-ESSENTIAL housekeeping complexes (ribosome/proteasome/ATP-synthase), whose
            members are essential everywhere so their profiles barely vary. Those stay verified by complex
            membership; co-essentiality corroborates the context-variable ones.
  3. EXTEND  score candidate pairs with the trained model; the high-precision, DepMap-supported ones that are
            not yet edges become cell-v3 additions (predicted, confidence-tagged; facts untouched).

Anti-trap held throughout: DepMap is an independent validator/α trainer; it never overwrites a measured value.
-> outputs/orphan/phase3_depmap_validation.json  +  cell_v3_additions.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

SCRATCH = os.environ.get("DEPMAP_DIR",
    "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GENE_EFFECT = os.path.join(SCRATCH, "CRISPRGeneEffect.csv")


def _json_np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def load_gene_effect(path=GENE_EFFECT):
    """-> (symbols list, Z matrix [genes x lines] z-scored per gene with NaN->0). Column headers look like
    'A1BG (1)'; rows are ModelIDs."""
    # cache the derived z-scored matrix: after the first parse, reuse the small .npz -> no 419 MB CSV re-read
    # (much faster AND much lower peak RAM on every subsequent run / reconnect). Delete the npz to refresh.
    cache = "outputs/orphan/depmap_vecs.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  DepMap: loaded z-scored matrix from cache ({z['Z'].shape[0]:,} genes x {z['Z'].shape[1]} lines)")
        return list(z["syms"]), z["Z"], z["valid"], z["mu"], z["sd"]
    import pandas as pd
    df = pd.read_csv(path, index_col=0)                       # lines x genes
    syms = [c.split(" (")[0] for c in df.columns]
    X = df.to_numpy(dtype=np.float32).T                       # genes x lines (float32)
    del df                                                    # free the DataFrame promptly (lower peak RAM)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True); sd[sd == 0] = 1.0
    valid = (~np.isnan(X)).sum(1)
    X -= mu; X /= sd                                          # z-score IN PLACE (no extra full-size copy)
    np.nan_to_num(X, copy=False)                             # NaN -> 0 in place; X is now Z
    try:
        np.savez_compressed(cache, syms=np.array(syms), Z=X, valid=valid, mu=mu.ravel(), sd=sd.ravel())
        print(f"  DepMap: cached z-scored matrix -> {cache}")
    except Exception as e:
        print("  (DepMap cache write failed:", str(e)[:50], ")")
    return syms, X, valid, mu.ravel(), sd.ravel()


class DepMapEdges:
    def __init__(self):
        from complete_cell import CompleteCell
        self.C = CompleteCell()
        self.syms, self.Z, self.valid, self.raw_mean, self.raw_std = load_gene_effect()
        self.col = {s: i for i, s in enumerate(self.syms)}
        self.nlines = self.Z.shape[1]

    def _pan_essential(self, i):
        """essential across most lines: strongly negative effect everywhere (low variance in RAW gene-effect).
        Such genes' profiles barely vary, so co-essentiality is uninformative for them."""
        return bool(self.raw_mean[i] < -0.5 and self.raw_std[i] < 0.3)

    def coess(self, a, b):
        """co-essentiality = Pearson corr of two genes' gene-effect profiles across lines (z-scored dot / n)."""
        ia, ib = self.col.get(a), self.col.get(b)
        if ia is None or ib is None:
            return None
        return float(np.dot(self.Z[ia], self.Z[ib]) / self.nlines)

    def shared_partners(self, a, b):
        ia, ib = self.C.idx.get(a), self.C.idx.get(b)
        if ia is None or ib is None:
            return 0
        return len(self.C.ppi_adj.get(ia, set()) & self.C.ppi_adj.get(ib, set()))

    # ---------- 1. TRAIN ----------
    def train(self, n_pos=4000, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(seed)
        name, idx, adj = self.C.name, self.C.idx, self.C.ppi_adj
        inset = set(self.col)                                  # genes present in DepMap
        # positives: known PPI edges whose BOTH ends are in DepMap
        edges = [(name[u], name[v]) for u, ps in adj.items() for v in ps
                 if u < v and name[u] in inset and name[v] in inset]
        rng.shuffle(edges); edges = edges[:n_pos]
        # negatives: random gene pairs that are NOT PPI edges (same count)
        pool = [g for g in name if g in inset]
        negs = set()
        while len(negs) < len(edges):
            a, b = name[rng.integers(len(name))], pool[rng.integers(len(pool))]
            ia, ib = idx.get(a), idx.get(b)
            if a == b or ia is None or ib is None or a not in inset:
                continue
            if ib in adj.get(ia, set()):
                continue
            negs.add((a, b))
        pairs = [(a, b, 1) for a, b in edges] + [(a, b, 0) for a, b in negs]
        X, y = [], []
        for a, b, lab in pairs:
            ce = self.coess(a, b)
            if ce is None:
                continue
            X.append([ce, abs(ce), self.shared_partners(a, b)]); y.append(lab)
        X = np.array(X); y = np.array(y)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        # ablation: coess-only vs triadic-only vs both — does DepMap ADD signal?
        def auc(cols):
            m = LogisticRegression(max_iter=1000).fit(Xtr[:, cols], ytr)
            return round(float(roc_auc_score(yte, m.predict_proba(Xte[:, cols])[:, 1])), 3)
        res = dict(n_pairs=len(y), n_pos=int(y.sum()),
                   auc_coessentiality_only=auc([0, 1]),
                   auc_triadic_only=auc([2]),
                   auc_combined=auc([0, 1, 2]),
                   mean_coess_pos=round(float(X[y == 1, 0].mean()), 3),
                   mean_coess_neg=round(float(X[y == 0, 0].mean()), 3))
        self.model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        # HONEST framing: on KNOWN edges triadic closure dominates and co-essentiality is largely redundant
        # (known edges favour well-studied genes that already have shared partners). Co-essentiality's value is
        # COMPLEMENTARY — finding edges the DBs miss — measured separately in extend() (triadic-blind fraction).
        res["redundant_on_known_edges"] = (res["auc_combined"] - res["auc_triadic_only"]) < 0.02
        res["note"] = ("triadic closure already predicts KNOWN edges well; DepMap adds only ~%.3f AUC there "
                       "(redundant). DepMap is NOT redundant for MISSING edges — see extend.triadic_blind_frac"
                       % (res["auc_combined"] - res["auc_triadic_only"]))
        return res

    @staticmethod
    def _same_family(a, b):
        import re
        pa = re.match(r"[A-Z]+[0-9]*", a or "X").group()[:4]
        pb = re.match(r"[A-Z]+[0-9]*", b or "Y").group()[:4]
        return len(pa) >= 3 and pa == pb

    # ---------- 2. CORROBORATE the cell-v2 additions ----------
    def corroborate_v2(self, thresh=0.1):
        add = json.load(open("outputs/orphan/cell_v2_additions.json"))
        pairs = add.get("ppi_edges_added", [])
        vals, paness = [], 0
        rows = []
        for e in pairs:
            ce = self.coess(e["a"], e["b"])
            if ce is None:
                continue
            # pan-essential? both genes essential across most lines -> low-variance profile (coess uninformative)
            ia, ib = self.col.get(e["a"]), self.col.get(e["b"])
            pe = self._pan_essential(ia) or self._pan_essential(ib)
            paness += pe
            vals.append(ce)
            rows.append((e["a"], e["b"], round(ce, 3), bool(pe)))
        vals = np.array(vals)
        # baseline: co-essentiality of random pairs
        rng = np.random.default_rng(1)
        base = []
        for _ in range(3000):
            a, b = self.syms[rng.integers(len(self.syms))], self.syms[rng.integers(len(self.syms))]
            if a != b:
                base.append(self.coess(a, b))
        base = np.array([b for b in base if b is not None])
        corod = int((vals > thresh).sum())
        top = sorted(rows, key=lambda r: -r[2])[:10]
        # split by pan-essential: co-essentiality should corroborate the CONTEXT-VARIABLE pairs, not the
        # pan-essential ones (honest — shows exactly where the DepMap signal applies)
        ctxvar = np.array([c for _, _, c, pe in rows if not pe])
        panes = np.array([c for _, _, c, pe in rows if pe])
        return dict(n_scored=len(vals),
                    mean_coess_additions=round(float(vals.mean()), 3),
                    mean_coess_random=round(float(base.mean()), 3),
                    frac_above_thresh=round(corod / max(1, len(vals)), 3),
                    random_frac_above_thresh=round(float((base > thresh).mean()), 3),
                    pan_essential_pairs=int(paness),
                    frac_above_thresh_context_variable=round(float((ctxvar > thresh).mean()), 3) if len(ctxvar) else None,
                    frac_above_thresh_pan_essential=round(float((panes > thresh).mean()), 3) if len(panes) else None,
                    enrichment_vs_random=round(float((vals > thresh).mean() / max(1e-6, (base > thresh).mean())), 1),
                    enrichment_context_variable=round(float((ctxvar > thresh).mean() / max(1e-6, (base > thresh).mean())), 1) if len(ctxvar) else None,
                    top_corroborated=[dict(a=a, b=b, coess=c, pan_essential=pe) for a, b, c, pe in top],
                    note="co-essentiality corroborates the CONTEXT-VARIABLE additions; PAN-ESSENTIAL complex "
                         "members (low-variance profiles) stay verified by complex membership, not co-essentiality")

    # ---------- 3. EXTEND: new DepMap-supported edges (cell v3) ----------
    def extend(self, add_coess=0.4, scan_coess=0.3, max_new=2000):
        """high co-essentiality pairs that are NOT current PPI edges -> predicted edges for cell v3.
        Confidence is anchored on the CO-ESSENTIALITY itself (the validated selection signal), NOT the trained
        model p — because the triadic-dominated model under-values pure co-essentiality edges (e.g. understudied
        mito genes with no shared PPI partners are exactly the real edges the PPI DB is missing). The trained
        model_p is reported as a secondary corroborator. Bounded; the cap is LOGGED."""
        name, idx, adj = self.C.name, self.C.idx, self.C.ppi_adj
        cands = {}
        ctx_genes = [g for g in name if g in self.col and len(adj.get(idx[g], ())) >= 5]
        Zc = self.Z; colmap = self.col
        for g in ctx_genes:
            ig = colmap[g]
            cors = Zc @ Zc[ig] / self.nlines                  # coess of g vs all genes
            order = np.argsort(-cors)[:15]
            for j in order:
                partner = self.syms[j]
                if partner == g or cors[j] < scan_coess:
                    continue
                ia, ib = idx.get(g), idx.get(partner)
                if ia is None or ib is None or ib in adj.get(ia, set()):
                    continue                                   # already an edge
                if self._same_family(g, partner):
                    continue                                   # conservative: skip paralog pairs (co-essential but
                                                               # that's family redundancy, not a PPI to add)
                key = (min(ia, ib), max(ia, ib))
                cands[key] = max(cands.get(key, 0.0), float(cors[j]))
        # ADD only the strong ones (coess >= add_coess); the 0.3-0.4 band stays candidates, not applied
        strong = {k: v for k, v in cands.items() if v >= add_coess}
        ranked = sorted(strong.items(), key=lambda kv: -kv[1])
        capped = len(ranked) > max_new
        ranked = ranked[:max_new]
        adds, triadic_blind = [], 0
        for (ia, ib), ce in ranked:
            a, b = name[ia], name[ib]
            sp = len(adj.get(ia, set()) & adj.get(ib, set()))
            triadic_blind += (sp == 0)
            p = float(self.model.predict_proba([[ce, abs(ce), sp]])[0, 1]) if hasattr(self, "model") else None
            adds.append(dict(a=a, b=b, coess=round(ce, 3), shared_partners=sp,
                             model_p=round(p, 3) if p is not None else None, relation="coessential",
                             evidence="DepMap co-essentiality (Pearson r across %d lines)" % self.nlines,
                             confidence=round(min(0.95, ce), 3)))       # confidence = the co-essentiality itself
        # honest: report the module concentration (co-essential additions cluster into coherent modules)
        from collections import Counter
        gc = Counter()
        for x in adds:
            gc[x["a"]] += 1; gc[x["b"]] += 1
        return dict(n_candidates_scanned=len(cands), n_strong=len(strong), n_added=len(adds), capped=capped,
                    add_coess=add_coess, scan_coess=scan_coess, distinct_genes=len(gc),
                    top_module_genes=[g for g, _ in gc.most_common(12)],
                    triadic_blind_frac=round(triadic_blind / max(1, len(adds)), 3),
                    triadic_blind_note=("%d/%d additions have ZERO shared PPI partners — triadic closure is "
                                        "structurally blind to them; DepMap co-essentiality is the ONLY signal "
                                        "that finds these (the real complementarity)" % (triadic_blind, len(adds))),
                    cap_note=(f"{len(strong)} pairs >= coess {add_coess}; kept top {max_new} (LOGGED, not silent)"
                              if capped else f"all {len(strong)} pairs >= coess {add_coess} kept"),
                    additions=adds)


def main():
    if not os.path.exists(GENE_EFFECT):
        print("DepMap CRISPRGeneEffect.csv not found at", GENE_EFFECT, "- download it first.")
        return
    dm = DepMapEdges()
    print("=" * 84)
    print(f"PHASE 3 — DepMap co-essentiality  ({len(dm.syms):,} genes x {dm.nlines} cell lines)")
    print("=" * 84)
    tr = dm.train()
    print("\n[1] TRAIN edge predictor (known PPI vs random non-edges):")
    print(f"    AUC  co-essentiality-only : {tr['auc_coessentiality_only']}")
    print(f"    AUC  triadic-only         : {tr['auc_triadic_only']}")
    print(f"    AUC  combined             : {tr['auc_combined']}   (redundant on KNOWN edges: {tr['redundant_on_known_edges']})")
    print(f"    mean co-ess  edges {tr['mean_coess_pos']}  vs random {tr['mean_coess_neg']}  ({round(tr['mean_coess_pos']/max(1e-6,tr['mean_coess_neg']))}x)")
    print(f"    -> {tr['note']}")
    cor = dm.corroborate_v2()
    print("\n[2] CORROBORATE the 978 cell-v2 co-complex PPI additions (independent DepMap signal):")
    print(f"    mean co-ess additions {cor['mean_coess_additions']}  vs random {cor['mean_coess_random']}")
    print(f"    fraction co-essential (> .1): {cor['frac_above_thresh']}  vs random {cor['random_frac_above_thresh']}"
          f"   ({cor['enrichment_vs_random']}x enrichment)")
    print(f"    context-variable pairs: {cor['frac_above_thresh_context_variable']} above thresh "
          f"({cor['enrichment_context_variable']}x)  |  pan-essential pairs (coess blind): "
          f"{cor['pan_essential_pairs']} (verified by complex membership instead)")
    ext = dm.extend()
    print("\n[3] EXTEND -> cell v3: new DepMap-supported edges not currently present:")
    print(f"    scanned {ext['n_candidates_scanned']} candidate pairs  ->  {ext['n_added']} added   ({ext['cap_note']})")
    print(f"    COMPLEMENTARITY: {ext['triadic_blind_note']}")
    for a in ext["additions"][:6]:
        print(f"      {a['a']:8}—{a['b']:8}  coess={a['coess']}  shared={a['shared_partners']}  model_p={a['model_p']}  conf={a['confidence']}")

    res = dict(train=tr, corroborate_v2=cor,
               extend=dict(n_candidates_scanned=ext["n_candidates_scanned"], n_strong=ext["n_strong"],
                           n_added=ext["n_added"], capped=ext["capped"], add_coess=ext["add_coess"],
                           triadic_blind_frac=ext["triadic_blind_frac"], triadic_blind_note=ext["triadic_blind_note"]),
               source="DepMap 24Q2 CRISPRGeneEffect (Achilles gene-effect)",
               hierarchy="DepMap is an independent validator/trainer; never overwrites a measured value")
    json.dump(res, open("outputs/orphan/phase3_depmap_validation.json", "w"), indent=2, default=_json_np)
    json.dump(dict(meta=dict(source="DepMap co-essentiality-trained edge model, cell v3",
                             rule="predicted edges only; facts untouched"),
                   ppi_edges_added=ext["additions"]),
              open("outputs/orphan/cell_v3_additions.json", "w"))
    print("\n-> outputs/orphan/phase3_depmap_validation.json  +  cell_v3_additions.json")
    return res


if __name__ == "__main__":
    main()
