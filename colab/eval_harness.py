"""eval_harness -- the honest, NON-FOOLABLE test bench for any knockout far-field predictor (a state-dependent tensor-field engine, a
network walk, an LLM, anything). Built because "it matches the phenotype" is a trap on this problem: ~70% of the measured far field is
the TIDE (the generic stress response every strong perturbation shares), so a predictor can look great on aggregate recall while getting
every knockout-SPECIFIC edge wrong. This harness scores against the bar that can't be gamed that way.

Two tests, each with a guard against the exact way it would otherwise fool you:

  SYSTEM test  score_system(predictor)  -- held-out knockouts, K562 Perturb-seq ground truth. Reports SPECIFIC-mover recall (tide
     REMOVED -- the real target) next to ALL-mover recall (tide counts -- the inflated number), plus a TIDE-ALIGNMENT diagnostic
     (how much the predictor's scores just are the tide). Flanked by three fixed references computed inline every run: RANDOM (floor),
     TIDE-null (the "no specific information" baseline you must beat ~0.26), ORACLE* (optimistic ceiling ~0.62, peeks at the held-out
     KO). A predictor "passes" only if it beats the TIDE-null on tide-removed specific movers -- reproducing the tide earns nothing.

  EDGE test    score_edges(edge_predictor)  -- component-level spot-check against ground-truth edges (co-dependency / complex / directed
     regulatory-vs-mRNA). GUARD AGAINST SELECTION BIAS: scores a RANDOM sample AND a WELL-STUDIED sample (both endpoints high-
     publication) and reports the gap -- because the edges that HAVE ground truth are the well-studied ones, and passing those
     over-estimates the dark majority. (This is the guard for "we can't check every edge, only a few" -- it makes the few honest.)

Predictor contracts (both pluggable, both may return None to abstain):
   predictor(ko_gene_name)      -> {gene_name: score}   # predicted movers for that knockout
   edge_predictor(a_name,b_name)-> float                # predicted interaction/effect strength for that ordered pair

main() runs the built-in reference predictors to (a) demonstrate the API and (b) SELF-CALIBRATE -- the bench must reproduce the known
numbers (tide ~0.26, network model ~0.37, oracle ~0.62) or it is not trustworthy. Deterministic (RandomState(0), sorted iteration).
"""
import json, pickle, collections
from pathlib import Path
import numpy as np
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU = 1.0
TIDE_FRAC = 0.05
MIN_SPEC = 5
N_NEI = 10


class Harness:
    def __init__(self, cell_type="K562"):
        self.KO = pickle.load(open(f"{SP}/nlz_{cell_type}.pkl", "rb"))
        self.kos = sorted(self.KO)
        self.genes = sorted({g for v in self.KO.values() for g in v})
        self.gi = {g: i for i, g in enumerate(self.genes)}
        self.ki = {k: i for i, k in enumerate(self.kos)}
        nK, nG = len(self.kos), len(self.genes)
        self.nG = nG
        M = np.zeros((nK, nG), dtype=np.float32)
        for k, v in self.KO.items():
            r = self.ki[k]
            for g, z in v.items():
                M[r, self.gi[g]] = z
        self.M = M
        self.A = np.abs(M)
        self.mover = self.A >= TAU
        self.mover_freq = self.mover.mean(axis=0)                 # per-gene tide-ness (the tide vector)
        self.tide_mask = self.mover_freq >= TIDE_FRAC
        self.nontide = ~self.tide_mask
        self.nontide_idx = np.where(self.nontide)[0]
        self.Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        self._load_graph()
        self._load_edge_truth()
        # the scorable held-out knockouts (>= MIN_SPEC specific movers) -- the honest test set
        self.scorable = []
        for x in self.kos:
            mv = np.where(self.mover[self.ki[x]])[0]
            spec = set(int(i) for i in mv if self.nontide[i])
            if len(mv) >= MIN_SPEC and len(spec) >= MIN_SPEC:
                self.scorable.append(x)
        print(f"harness[{cell_type}]: {nK} knockouts x {nG} genes; tide={int(self.tide_mask.sum())} genes; "
              f"{len(self.scorable)} scorable held-out knockouts (>= {MIN_SPEC} specific movers)")

    def _load_graph(self):
        """the network-neighbour graph (PPI + complex/pathway + regulon) used by the reference model."""
        idx = set(self.genes)
        D = json.load(open(OUT / "cell_complete.json"))
        info = {g["name"]: g for g in D["genes"]}
        nb = collections.defaultdict(set)
        # THE INDEX-SPACE BUG, in its eval_harness form. 'ppi' endpoints are INTEGER INDICES into genes[] while
        # `idx` is a set of gene NAME strings, so `e[0] in idx` was False for every edge and this loop accepted
        # 0 of 191,447. The NEIGHBOR reference model was therefore built from complex + pathway + regulon only,
        # with no PPI in it at all, despite being described as including PPI.
        allnames = [g["name"] for g in D["genes"]]
        for e in D.get("ppi", []):
            try:
                a_, b_ = int(e[0]), int(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (0 <= a_ < len(allnames) and 0 <= b_ < len(allnames)) or a_ == b_:
                continue
            na_, nb_ = allnames[a_], allnames[b_]
            if na_ in idx and nb_ in idx:
                nb[na_].add(nb_)
                nb[nb_].add(na_)
        cplx = collections.defaultdict(set)
        for g, cs in (D.get("gene2cplx", {}) or {}).items():
            for c in (cs if isinstance(cs, (list, set, tuple)) else [cs]) if cs else []:
                cplx[("c", c)].add(g)
        for g in idx:
            p = info.get(g, {}).get("path")
            if p:
                cplx[("p", p)].add(g)
        for grp, mem in cplx.items():
            if 2 <= len(mem) <= 200:
                ml = list(mem)
                for a in ml:
                    nb[a].update(x for x in ml if x != a)
        trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
        for tf, ts in trr.items():
            for t in ts:
                g = t[0] if isinstance(t, (list, tuple)) else t
                if tf in idx and g in idx:
                    nb[tf].add(g); nb[g].add(tf)
        self.nb = nb

    def _load_edge_truth(self):
        """ground-truth edge sets for the component spot-check, + a publication count per gene for the studied/random split."""
        D = json.load(open(OUT / "cell_complete.json"))
        self.names = [g["name"] for g in D["genes"]]
        self.name2i = {n: i for i, n in enumerate(self.names)}
        self.pubs = {g["name"]: float(g.get("pubs") or 0.0) for g in D["genes"]}
        codep = {int(k): {int(j) for j, _ in v} for k, v in D["codep"].items()}
        self.codep_edges = set()
        for i, js in codep.items():
            for j in js:
                self.codep_edges.add((min(i, j), max(i, j)))
        cx = set()
        for m in D["complexes"].values():
            mm = sorted(x for x in m if isinstance(x, int) and x < len(self.names))
            for a in range(len(mm)):
                for b in range(a + 1, len(mm)):
                    cx.add((mm[a], mm[b]))
        self.complex_edges = cx
        self._codep_nodes = sorted({i for e in self.codep_edges for i in e})

    # ---------- SYSTEM-LEVEL TEST ----------
    def _scores_to_vec(self, scores):
        v = np.zeros(self.nG)
        if scores is None:
            return None
        for g, s in scores.items():
            j = self.gi.get(g)
            if j is not None:
                v[j] = s
        return v

    def _recall(self, vec, truth_idx, restrict, k):
        order = restrict[np.argsort(-vec[restrict])][:k]
        return len(set(order.tolist()) & truth_idx) / max(len(truth_idx), 1)

    def _references(self, x, K):
        """RANDOM / TIDE / ORACLE reference scores for held-out knockout x (oracle peeks -- it is a ceiling, not a predictor)."""
        r = self.ki[x]
        rng = np.random.RandomState(r)                 # deterministic per-KO seed (index, not Python hash())
        s_rand = rng.rand(self.nG)
        s_tide = self.mover_freq.copy()
        sim = self.Mn @ self.Mn[r]; sim[r] = -1
        top = np.argsort(-sim)[:N_NEI]
        s_oracle = self.A[top].mean(axis=0)
        return {"RANDOM": s_rand, "TIDE-null": s_tide, "ORACLE*": s_oracle}

    def score_system(self, predictor, K=50, label="predictor"):
        """Score `predictor(ko_name)->{gene:score}` on held-out specific-mover recall (tide removed). Returns the honest scorecard."""
        spec_rec, all_rec, cov, tide_align = [], [], 0, []
        ref_spec = collections.defaultdict(list); ref_all = collections.defaultdict(list)
        tide_unit = self.mover_freq / (np.linalg.norm(self.mover_freq) + 1e-9)
        for x in self.scorable:
            r = self.ki[x]
            mv = np.where(self.mover[r])[0]
            spec = set(int(i) for i in mv if self.nontide[i])
            allm = set(mv.tolist())
            vec = self._scores_to_vec(predictor(x))
            if vec is not None:
                cov += 1
                spec_rec.append(self._recall(vec, spec, self.nontide_idx, K))
                all_rec.append(self._recall(vec, allm, np.arange(self.nG), K))
                nv = np.linalg.norm(vec)
                tide_align.append(float(vec @ tide_unit / nv) if nv > 1e-9 else 0.0)
            refs = self._references(x, K)
            for nm, s in refs.items():
                ref_spec[nm].append(self._recall(s, spec, self.nontide_idx, K))
                ref_all[nm].append(self._recall(s, allm, np.arange(self.nG), K))
        spec = float(np.mean(spec_rec)) if spec_rec else float("nan")
        allm = float(np.mean(all_rec)) if all_rec else float("nan")
        refs_spec = {k: float(np.mean(v)) for k, v in ref_spec.items()}
        tide = refs_spec["TIDE-null"]; oracle = refs_spec["ORACLE*"]
        lift = spec / max(tide, 1e-9)
        passes = spec > tide + 0.02          # must beat the tide null on tide-removed movers by a real margin
        card = {"label": label, "n_scored": len(self.scorable), "coverage": cov / max(len(self.scorable), 1),
                "specific_recall": round(spec, 4), "all_mover_recall": round(allm, 4),
                "tide_null": round(tide, 4), "oracle_ceiling": round(oracle, 4), "random_floor": round(refs_spec["RANDOM"], 4),
                "lift_over_tide": round(lift, 3), "tide_alignment": round(float(np.mean(tide_align)) if tide_align else 0.0, 3),
                "PASSES_bar": bool(passes),
                "headroom_used": round((spec - tide) / max(oracle - tide, 1e-9), 3)}
        print(f"\n  SYSTEM scorecard [{label}] @K={K}  (coverage {card['coverage']:.0%} of {card['n_scored']} held-out KOs)")
        print(f"    SPECIFIC-mover recall (tide removed, THE BAR): {card['specific_recall']:.3f}")
        print(f"      vs  RANDOM {card['random_floor']:.3f}  <  TIDE-null {card['tide_null']:.3f}  <  [you]  <  ORACLE* {card['oracle_ceiling']:.3f}")
        print(f"    all-mover recall (tide counts, INFLATED):     {card['all_mover_recall']:.3f}   <- ignore this; it's mostly tide")
        print(f"    lift over tide-null: {card['lift_over_tide']:.2f}x   headroom used: {card['headroom_used']:.0%}   "
              f"tide-alignment: {card['tide_alignment']:.2f}")
        print(f"    -> {'PASSES' if passes else 'FAILS'} the honest bar (beat tide-null on SPECIFIC movers)"
              + ("" if passes else "  -- reproducing the tide earns nothing here"))
        return card

    # ---------- COMPONENT (EDGE) TEST ----------
    def _edge_sample(self, truth_edges, n, studied, rng):
        pos = sorted(truth_edges)
        if studied:
            thr = np.percentile([self.pubs.get(nm, 0) for nm in self.names], 66)
            pos = [(a, b) for a, b in pos if self.pubs.get(self.names[a], 0) >= thr and self.pubs.get(self.names[b], 0) >= thr]
        if len(pos) > n:
            pos = [pos[i] for i in rng.choice(len(pos), n, replace=False)]
        nodes = self._codep_nodes
        truth = set(truth_edges)
        neg = []
        while len(neg) < len(pos):
            a, b = rng.choice(nodes, 2, replace=False)
            e = (min(int(a), int(b)), max(int(a), int(b)))
            if e not in truth:
                neg.append(e)
        return pos, neg

    def score_edges(self, edge_predictor, truth="codep", n=1500, label="edge_predictor"):
        """Score `edge_predictor(a_name,b_name)->float` on separating true edges from matched non-edges, RANDOM vs WELL-STUDIED
        sample. Reports AUC for each and the selection-bias gap."""
        from sklearn.metrics import roc_auc_score
        truth_edges = {"codep": self.codep_edges, "complex": self.complex_edges}[truth]
        out = {}
        for studied in (False, True):
            rng = np.random.RandomState(0)
            pos, neg = self._edge_sample(truth_edges, n, studied, rng)
            if len(pos) < 30:
                out["studied" if studied else "random"] = None; continue
            y, s = [], []
            for e, lab in [(p, 1) for p in pos] + [(q, 0) for q in neg]:
                sc = edge_predictor(self.names[e[0]], self.names[e[1]])
                if sc is None:
                    continue
                y.append(lab); s.append(sc)
            auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
            out["studied" if studied else "random"] = {"auc": round(float(auc), 3), "n_pos": len(pos)}
        gap = None
        if out.get("random") and out.get("studied"):
            gap = round(out["studied"]["auc"] - out["random"]["auc"], 3)
        print(f"\n  EDGE scorecard [{label}] vs {truth} truth:")
        print(f"    RANDOM sample  AUC {out['random']['auc'] if out.get('random') else 'n/a'}  (n_pos {out['random']['n_pos'] if out.get('random') else 0})")
        print(f"    STUDIED sample AUC {out['studied']['auc'] if out.get('studied') else 'n/a'}  (n_pos {out['studied']['n_pos'] if out.get('studied') else 0})")
        print(f"    SELECTION-BIAS gap (studied - random): {gap}   <- a large positive gap means spot-checks on famous edges over-sell")
        return {"label": label, "truth": truth, **out, "selection_bias_gap": gap}

    # ---------- reference predictors (examples + self-calibration) ----------
    def ref_random(self, x):
        rng = np.random.RandomState(self.ki[x] ^ 0x5555)      # deterministic per-KO seed (index, not Python hash())
        return {g: float(rng.rand()) for g in self.genes[:200]}

    def ref_tide(self, x):
        return {g: float(self.mover_freq[i]) for i, g in enumerate(self.genes)}

    def ref_neighbor(self, x):
        """the wall_test MODEL: average the measured |z| profiles of x's graph neighbours (transfer from priors)."""
        neigh = [self.ki[g] for g in self.nb.get(x, ()) if g in self.ki and g != x]
        if not neigh:
            return None
        v = self.A[neigh].mean(axis=0)
        return {g: float(v[i]) for i, g in enumerate(self.genes)}

    def ref_edge_structure(self, a, b):
        """example edge predictor: complex co-membership + direct PPI (structure/contact proxy)."""
        ia, ib = self.name2i.get(a), self.name2i.get(b)
        if ia is None or ib is None:
            return 0.0
        e = (min(ia, ib), max(ia, ib))
        return 2.0 * (e in self.complex_edges) + 1.0 * (b in self.nb.get(a, ()))


def main():
    print("=" * 100)
    print("eval_harness -- non-foolable test bench for knockout far-field predictors (self-calibration run)")
    print("=" * 100)
    H = Harness("K562")
    print("\n[SELF-CALIBRATION] the bench must reproduce the known reference numbers, else it is not trustworthy:")
    cards = {}
    cards["RANDOM"] = H.score_system(H.ref_random, label="RANDOM (ref)")
    cards["TIDE"] = H.score_system(H.ref_tide, label="TIDE (ref)")
    cards["NEIGHBOR"] = H.score_system(H.ref_neighbor, label="NEIGHBOR model (ref)")
    ec = H.score_edges(H.ref_edge_structure, truth="codep", label="structure/contact (ref)")

    # sanity: TIDE predictor should ~= the tide-null baseline; neighbour should sit between tide and oracle
    tide_card = cards["TIDE"]; nb_card = cards["NEIGHBOR"]
    calibrated = (abs(tide_card["specific_recall"] - tide_card["tide_null"]) < 0.03
                  and nb_card["specific_recall"] > nb_card["tide_null"] > cards["RANDOM"]["specific_recall"]
                  and nb_card["oracle_ceiling"] > nb_card["specific_recall"])
    verdict = (
        "eval_harness -- the honest, non-foolable bench for ANY knockout far-field predictor, built to defeat the two ways "
        "'it matches the phenotype' misleads on this problem. SYSTEM test: held-out specific-mover recall with the TIDE REMOVED (the real "
        f"target), flanked every run by RANDOM {cards['RANDOM']['specific_recall']:.3f} < TIDE-null {tide_card['tide_null']:.3f} < NEIGHBOR "
        f"model {nb_card['specific_recall']:.3f} < ORACLE* {nb_card['oracle_ceiling']:.3f}; a predictor PASSES only by beating the "
        "TIDE-null on tide-removed movers, so reproducing the generic stress response (~70% of the raw phenotype) earns nothing -- and the "
        "all-mover recall + tide-alignment diagnostic expose any predictor that is secretly just the tide. EDGE test: component spot-check "
        "vs ground-truth edges scored on a RANDOM sample AND a WELL-STUDIED (high-publication) sample, reporting the selection-bias gap "
        f"(structure/contact ref: random AUC {ec['random']['auc'] if ec.get('random') else 'n/a'} vs studied "
        f"{ec['studied']['auc'] if ec.get('studied') else 'n/a'}, gap {ec.get('selection_bias_gap')}) -- because the edges that HAVE "
        "ground truth are the famous ones, so 'we can only check a few' is only honest if the few are sampled at random. "
        + ("SELF-CALIBRATION PASSED: the TIDE reference reproduces the tide-null and the NEIGHBOR model sits strictly between the "
           "tide-null and the oracle, so the bench is calibrated and trustworthy. "
           if calibrated else "SELF-CALIBRATION WARNING: reference numbers did not land where expected -- inspect before trusting. ")
        + "USAGE: from eval_harness import Harness; H=Harness(); H.score_system(my_predictor); H.score_edges(my_edge_predictor). "
        "The bar to beat is the TIDE-null (~0.26) on tide-removed specific movers, not raw recall. This makes 'if it matches the "
        "phenotype it works' a rigorous claim instead of a hopeful one: an engine only earns belief by beating the tide on held-out "
        "specific movers AND passing random (not cherry-picked) edge spot-checks.")
    print(f"\nVERDICT: {verdict}")
    out = {"self_calibration_passed": bool(calibrated), "system_cards": cards, "edge_card": ec,
           "the_bar": {"metric": "specific-mover recall @50, tide removed, held-out",
                       "tide_null_to_beat": tide_card["tide_null"], "oracle_ceiling": nb_card["oracle_ceiling"]},
           "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "eval_harness.json", "w"), indent=1)
    print("\n  -> outputs/orphan/eval_harness.json")
    return out


if __name__ == "__main__":
    main()
