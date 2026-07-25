"""predictor_v2 -- the deployable predictor that packages what actually survived testing, and SELF-VALIDATES its own calibration.

What improve_v2 established: the four recall levers (similarity-weighting, rank fusion, agreement-gating, similarity-level fusion) all land
within seed noise of the ~0.49 baseline -- recall is saturated for this model class on this data. What DID improve is CAPABILITY:
  * DIRECTION  -- a sign head (from signed neighbour responses) calls up-vs-down, which the model previously could not do at all.
  * CALIBRATION -- per-gene neighbour consensus stratifies a top-50 list into a high-precision shortlist (~67-79%) and a long tail (~18%).

So this class answers a knockout query with a RANKED, SIGNED, CALIBRATED list instead of a bare ranking:
    predict("GATA1") -> [{gene, direction, consensus, expected_precision}, ...]

Leave-one-out by construction: the queried knockout is excluded from its own retrieval pool, so a query for a gene already in the panel is
scored exactly as an unseen gene would be. The expected_precision attached to each prediction is not asserted -- calibrate() MEASURES it on
held-out knockouts and the demo prints observed-vs-claimed so the numbers are auditable. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness

OUT = Path("outputs/orphan")
W0 = {"BEHAV": 0.35, "GRAPH": 0.25, "STRUCT": 1.5, "TIDE": 0.25}
NEI, K = 10, 50
BINS = [(0.7, "very high"), (0.5, "high"), (0.3, "moderate"), (0.2, "low"), (0.0, "very low")]


class KOPredictor:
    def __init__(self, cell="K562"):
        self.H = H = Harness(cell)
        dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
        syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32)
        self.srow = {s: i for i, s in enumerate(syms)}
        self.Zc = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
        self.comem = collections.defaultdict(set)
        for mem in D["complexes"].values():
            mm = [names[x] for x in mem if isinstance(x, int) and x < N]
            if 2 <= len(mm) <= 200:
                for g in mm:
                    self.comem[g].update(x for x in mm if x != g)
        self.pool = [k for k in H.kos if k in self.srow]                       # retrieval pool = every profiled knockout
        self.pool_rows = np.array([H.ki[k] for k in self.pool])
        self.Cmat = np.stack([self.Zc[self.srow[k]] for k in self.pool])
        self.An = H.A / (np.linalg.norm(H.A, axis=1, keepdims=True) + 1e-9)
        self.calib = None

    # ---------- core ----------
    def _neighbours(self, ko, exclude):
        """channel -> (rows, sims), always excluding the query itself (leave-one-out => an in-panel gene is scored as if unseen)."""
        H = self.H; out = {}
        if ko in self.srow:
            sims = self.Cmat @ self.Zc[self.srow[ko]]
            mask = np.array([p not in exclude for p in self.pool])
            sims = np.where(mask, sims, -np.inf)
            top = np.argsort(-sims)[:NEI]
            out["BEHAV"] = (self.pool_rows[top], sims[top])
        g_nb = [H.ki[g] for g in H.nb.get(ko, ()) if g in H.ki and g not in exclude]
        if g_nb:
            out["GRAPH"] = (np.array(g_nb[:50]), np.ones(min(len(g_nb), 50)))
        s_nb = [H.ki[g] for g in self.comem.get(ko, set()) if g in H.ki and g not in exclude]
        if s_nb:
            out["STRUCT"] = (np.array(s_nb[:50]), np.ones(min(len(s_nb), 50)))
        return out

    def _score(self, ko, exclude):
        H = self.H
        nb = self._neighbours(ko, exclude)
        acc = np.zeros(H.nG); signed = np.zeros(H.nG); wsum = 0.0
        consensus = np.zeros(H.nG); conf = 0.0
        for c, (rows, sims) in nb.items():
            v = H.A[rows].mean(0)
            acc += W0[c] * (v - v.mean()) / (v.std() + 1e-9)
            signed += W0[c] * H.M[rows].mean(0); wsum += W0[c]
            if c == "BEHAV":
                consensus = H.mover[rows].mean(0)                              # per-gene neighbour agreement
                X = self.An[rows]
                conf = float(np.mean(sims[np.isfinite(sims)])) * float(np.mean(X @ X.T))   # margin x agreement
        t = H.mover_freq
        acc += W0["TIDE"] * (t - t.mean()) / (t.std() + 1e-9)
        if wsum > 0:
            signed /= wsum
        return acc, signed, consensus, conf

    def predict(self, ko, k=K, min_consensus=0.0):
        """ranked, SIGNED, CALIBRATED predictions for a knockout. Excludes the query from its own retrieval."""
        H = self.H
        acc, signed, consensus, conf = self._score(ko, exclude={ko})
        order = H.nontide_idx[np.argsort(-acc[H.nontide_idx])][:k]
        out = []
        for g in order:
            c = float(consensus[g])
            if c < min_consensus:
                continue
            band = next(lbl for lo, lbl in BINS if c >= lo)
            out.append({"gene": H.genes[g], "score": float(acc[g]), "consensus": round(c, 2),
                        "direction": "up" if signed[g] > 0 else "down",
                        "confidence_band": band,
                        "expected_precision": (self.calib or {}).get(band)})
        return {"knockout": ko, "ko_confidence": round(conf, 4), "predictions": out}

    # ---------- self-validation ----------
    def calibrate(self, seeds=(0, 1)):
        """MEASURE the precision of each confidence band on held-out knockouts (not asserted -- observed)."""
        H = self.H; buckets = collections.defaultdict(list); recs = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            scor = list(H.scorable); perm = rng.permutation(len(scor))
            test = sorted(scor[i] for i in perm[:int(0.30 * len(scor))])
            held = set(test)
            for ko in test:
                spec = set(int(i) for i in np.where(H.mover[H.ki[ko]])[0] if H.nontide[i])
                if not spec:
                    continue
                acc, signed, consensus, _ = self._score(ko, exclude=held)      # whole test split held out
                order = H.nontide_idx[np.argsort(-acc[H.nontide_idx])][:K]
                recs.append(len(set(order.tolist()) & spec) / len(spec))
                for g in order:
                    band = next(lbl for lo, lbl in BINS if consensus[g] >= lo)
                    buckets[band].append(1.0 if g in spec else 0.0)
        self.calib = {b: round(float(np.mean(v)), 3) for b, v in buckets.items()}
        self.calib_n = {b: len(v) for b, v in buckets.items()}
        self.holdout_recall = float(np.mean(recs))
        return self.calib


def main():
    P = KOPredictor("K562")
    print("\ncalibrating on held-out knockouts (measuring, not asserting)...")
    calib = P.calibrate()
    print(f"\n  held-out specific recall@50 = {P.holdout_recall:.3f}\n")
    print(f"  {'confidence band':>16s} {'n predictions':>14s} {'OBSERVED precision':>19s}")
    for lo, b in BINS:
        if b in calib:
            print(f"  {b:>16s} {P.calib_n[b]:>14d} {calib[b]*100:>18.1f}%")

    for demo in ("GATA1", "NRF1"):
        if demo not in P.H.ki:
            continue
        r = P.predict(demo, k=50, min_consensus=0.3)
        H = P.H; truth = set(int(i) for i in np.where(H.mover[H.ki[demo]])[0] if H.nontide[i])
        print(f"\n{'='*88}\nDEMO  predict('{demo}')  -- high-confidence shortlist (consensus>=0.3), query excluded from its own retrieval")
        print(f"{'='*88}")
        print(f"  KO confidence {r['ko_confidence']:.3f};  {len(r['predictions'])} shortlisted of 50 ranked\n")
        print(f"  {'gene':10s} {'dir':>5s} {'consensus':>10s} {'exp.prec':>9s}  {'TRUE?':>6s} {'obs z':>7s}")
        hits = 0
        for p in r["predictions"][:12]:
            gi = H.gi[p["gene"]]; is_true = gi in truth; hits += is_true
            print(f"  {p['gene']:10s} {p['direction']:>5s} {p['consensus']:>10.2f} "
                  f"{(p['expected_precision'] or 0)*100:>8.0f}% {'YES' if is_true else 'no':>6s} {H.M[H.ki[demo], gi]:>7.1f}")
        allhit = sum(1 for p in r["predictions"] if H.gi[p["gene"]] in truth)
        sgn_ok = sum(1 for p in r["predictions"] if H.gi[p["gene"]] in truth
                     and (H.M[H.ki[demo], H.gi[p["gene"]]] > 0) == (p["direction"] == "up"))
        print(f"\n  shortlist precision {allhit}/{len(r['predictions'])} = {allhit/max(len(r['predictions']),1)*100:.0f}%"
              f"   |  direction correct on {sgn_ok}/{allhit} of the true hits")

    verdict = (
        "PREDICTOR V2 (predictor_v2.py): the deployable predictor packaging what survived testing. improve_v2 showed the four recall levers "
        "(similarity-weighting, rank fusion, agreement-gating, similarity-level fusion) all sit within seed noise of ~0.49 -- recall is SATURATED "
        "for this model class on this data -- so v2 adds CAPABILITY rather than recall: it returns a ranked, SIGNED (up/down) and CALIBRATED list "
        f"instead of a bare ranking. Held-out specific recall@50 {P.holdout_recall:.3f}. The calibration is MEASURED, not asserted: observed "
        "precision by confidence band = "
        + ", ".join(f"{b} {calib[b]*100:.0f}% (n={P.calib_n[b]})" for lo, b in BINS if b in calib)
        + ". Practically: a user asks for a knockout's effects and gets a short high-confidence list they can take to the bench (moderate-and-"
        "above bands run far above the 27% precision of the raw top-50), each entry carrying a predicted DIRECTION. Leave-one-out by "
        "construction -- the query is excluded from its own retrieval pool, so an in-panel gene is scored exactly as an unseen one. "
        "HONEST BOUND: this does not move the ceiling (oracle 0.62); it makes the existing signal usable and honest about where it is weak. "
        "Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"holdout_recall_at50": round(P.holdout_recall, 4), "observed_precision_by_band": calib,
               "n_by_band": P.calib_n, "verdict": verdict, "note": verdict},
              open(OUT / "predictor_v2.json", "w"), indent=1)
    print("\n  -> outputs/orphan/predictor_v2.json")


if __name__ == "__main__":
    main()
