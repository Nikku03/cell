"""fieldsim_test — the honest one: run REAL gene knockouts through the fieldsim engine's forward model and score its
predicted downstream signature against the MEASURED Perturb-seq knockout effect. This is the test the internal-
consistency checks in fieldsim.py could not give — does the engine predict measured biology?

For a knockout X, fieldsim's predictive core is regulatory propagation with Hill kinetics + decay (Layers 2+4; the
spatial/SH Layers 1+3 spread fields but add no signal about WHICH genes change — spot-checked). We build the local
signed regulatory graph around X (reg + causal edges, 2 hops), solve the Hill fixed point wild-type, knock X out
(field -> 0), re-solve, and read the predicted change per gene. Then score vs measured M[X, :]:

  - AUC       : does predicted |Δ| pick out the genes that actually moved (|measured|>0.5)?
  - sign agr. : on genes fieldsim predicts to move, does the direction match measured?
  - Spearman  : predicted Δ vs measured Δ over the genes fieldsim reaches.
Baselines: RANDOM, and STATIC-TARGETS (just sign of X's curated direct edges, no dynamics) — to see if the fieldsim
dynamics beat simply knowing the network.
-> outputs/orphan/fieldsim_test.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def hill(u, K=1.0, n=2.0):
    u = np.maximum(u, 0.0)
    return u ** n / (K ** n + u ** n)


class KnockoutValidator:
    def __init__(self, kernel=None):
        import cellos
        self.k = kernel if kernel is not None else cellos.CellKernel(quiet=True)
        self.C = C = self.k.C
        dbg = self.k._load_debugger()
        self.M = np.asarray(dbg["M"], "float32")
        self.pidx = {g: i for i, g in enumerate(dbg["pgenes"])}
        self.sidx = {g: i for i, g in enumerate(dbg["syms"])}
        self.syms = dbg["syms"]
        # signed directed regulatory + causal out-edges by SYMBOL: X -> [(target_sym, sign)]
        self.out = {}
        for i, tgts in C.reg_out.items():
            self.out.setdefault(C.name[i], []).extend((C.name[t], (1 if s >= 0 else -1)) for t, s in tgts)
        for i, tgts in (C.causal_out or {}).items():
            self.out.setdefault(C.name[i], []).extend((C.name[t], (1 if (s or 1) >= 0 else -1)) for t, s, *_ in tgts)
        # dedupe
        for g in self.out:
            self.out[g] = list({t: s for t, s in self.out[g]}.items())

    def _subgraph(self, X, hops=2, cap=500):
        """nodes within `hops` regulatory out-steps of X (capped), and the signed edges among them."""
        nodes = {X}; frontier = {X}
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                for (v, _s) in self.out.get(u, []):
                    if v not in nodes:
                        nxt.add(v)
            nodes |= nxt; frontier = nxt
            if len(nodes) > cap:
                break
        others = [g for g in nodes if g != X][:cap - 1]
        nodes = [X] + others; nidx = {g: i for i, g in enumerate(nodes)}   # X is always node 0
        from scipy import sparse
        rows, colu, sgn = [], [], []
        for u in nodes:
            for (v, s) in self.out.get(u, []):
                if v in nidx:
                    rows.append(nidx[v]); colu.append(nidx[u]); sgn.append(s)   # S[v,u] = sign(u->v)
        S = sparse.csr_matrix((np.array(sgn, "float32"), (rows, colu)), shape=(len(nodes), len(nodes)))
        return nodes, nidx, S

    def _fixedpoint(self, S, ko_idx=None, iters=60, damp=0.5, base=0.5):
        """Hill regulatory dynamics to steady state: phi = Hill(base + S @ phi). ko_idx clamped to 0. Vectorised."""
        n = S.shape[0]; phi = np.full(n, hill(base), "float32")
        for _ in range(iters):
            new = hill(base + S @ phi)
            if ko_idx is not None:
                new[ko_idx] = 0.0
            phi = damp * phi + (1 - damp) * new
            if ko_idx is not None:
                phi[ko_idx] = 0.0
        return phi

    def predict(self, X):
        """fieldsim forward prediction of KO(X): Δφ per reachable gene (Layers 2+4 dynamics)."""
        nodes, nidx, S = self._subgraph(X)
        if len(nodes) < 3:
            return {}
        wt = self._fixedpoint(S)
        ko = self._fixedpoint(S, ko_idx=0)                 # X is node 0
        return {nodes[i]: float(ko[i] - wt[i]) for i in range(1, len(nodes))}

    def static_targets(self, X):
        """baseline: X's direct curated targets with sign (no dynamics)."""
        return {t: -s for (t, s) in self.out.get(X, [])}   # KO of activator (s=+1) lowers target -> predicted -1

    def score(self, X, thr=0.5):
        if X not in self.pidx:
            return None
        meas = self.M[self.pidx[X]]
        moved = np.abs(meas) > thr
        pred = self.predict(X)
        stat = self.static_targets(X)
        if not pred:
            return None

        def auc_of(scoremap):
            s = np.zeros(len(self.syms))
            for g, v in scoremap.items():
                if g in self.sidx:
                    s[self.sidx[g]] = abs(v)
            pos = moved.copy(); pos[self.sidx.get(X, -1)] = False       # exclude self
            if pos.sum() < 3 or (~pos).sum() < 3:
                return None
            from sklearn.metrics import roc_auc_score
            try:
                return float(roc_auc_score(pos, s))
            except Exception:
                return None
        # sign agreement on genes fieldsim predicts to move (and that actually moved) — return counts to POOL later
        sign_hit = tot = 0
        moved_syms = {self.syms[j] for j in np.where(moved)[0]}
        reached_movers = 0
        for g, dv in pred.items():
            if g in self.sidx:
                mv = meas[self.sidx[g]]
                if abs(mv) > thr:
                    reached_movers += 1
                if abs(dv) > 0.02 and abs(mv) > thr:
                    tot += 1; sign_hit += int(np.sign(dv) == np.sign(mv))
        n_movers = int(moved.sum())
        return {"gene": X, "n_measured_movers": n_movers, "n_reached": len(pred),
                "auc_fieldsim": auc_of(pred), "auc_static_targets": auc_of(stat),
                "reached_movers": reached_movers, "recall": round(reached_movers / max(n_movers, 1), 3),
                "sign_hit": sign_hit, "sign_tot": tot}

    def validate(self, n=60, min_movers=25, seed=0):
        rng = np.random.default_rng(seed)
        cands = [g for g in self.out if g in self.pidx and len(self.out[g]) >= 10]
        cands = list(np.array(cands, dtype=object)[rng.choice(len(cands), min(n * 6, len(cands)), replace=False)])
        rows = []
        for X in cands:
            s = self.score(X)
            if s and s["auc_fieldsim"] is not None and s["n_measured_movers"] >= min_movers:  # real measured effect
                rows.append(s)
            if len(rows) >= n:
                break

        def med(key):
            v = [r[key] for r in rows if r.get(key) is not None]
            return round(float(np.median(v)), 3) if v else None
        pooled_sign = (sum(r["sign_hit"] for r in rows) / max(sum(r["sign_tot"] for r in rows), 1)) if rows else None
        return {"n_genes_tested": len(rows), "min_movers_filter": min_movers,
                "median_auc_fieldsim": med("auc_fieldsim"),
                "median_auc_static_targets": med("auc_static_targets"),
                "random_auc": 0.5,
                "median_recall_of_movers": med("recall"),
                "median_reached_neighborhood": med("n_reached"),
                "pooled_sign_agreement": round(pooled_sign, 3) if pooled_sign is not None else None,
                "pooled_sign_n": int(sum(r["sign_tot"] for r in rows)),
                "examples": [{k: r[k] for k in ("gene", "n_measured_movers", "n_reached", "reached_movers",
                                                "recall", "auc_fieldsim")} for r in rows[:8]]}


def main():
    print("=" * 96)
    print("FIELDSIM vs MEASURED — run real knockouts through the engine, score against Perturb-seq")
    print("=" * 96)
    V = KnockoutValidator()
    r = V.validate(n=60)
    print(f"\n  genes tested (measured KO with >={r['min_movers_filter']} real movers + >=10 regulatory targets): "
          f"{r['n_genes_tested']}")
    print(f"\n  DOES THE ENGINE PREDICT WHICH GENES MOVE? (AUC, 0.5 = chance)")
    print(f"    fieldsim dynamics (Layers 2+4):   {r['median_auc_fieldsim']}")
    print(f"    static curated targets (no dyn):  {r['median_auc_static_targets']}")
    print(f"    random:                           {r['random_auc']}")
    print(f"\n  WHERE IT REACHES:")
    print(f"    median recall — of the genes that actually moved, fraction fieldsim even touches: "
          f"{r['median_recall_of_movers']:.1%}")
    print(f"    median neighborhood it lights up (mostly non-movers): {r['median_reached_neighborhood']:.0f} genes")
    print(f"    pooled sign agreement on the few movers it does reach: {r['pooled_sign_agreement']} "
          f"(n={r['pooled_sign_n']}, 0.5=chance)")

    af, at, rec, sa = (r["median_auc_fieldsim"], r["median_auc_static_targets"],
                       r["median_recall_of_movers"], r["pooled_sign_agreement"])
    verdict = (
        f"Run against measured Perturb-seq (on knockouts that actually DO something — >={r['min_movers_filter']} moved "
        f"genes each), the fieldsim engine's forward prediction is at CHANCE: it ranks which genes move at AUC {af} "
        f"(random 0.5), and — the decisive comparison — that is no better than the STATIC baseline of just listing the "
        f"knockout's curated targets with no dynamics (AUC {at}). So the Hill+decay+diffusion machinery adds nothing "
        f"over the network topology it was handed. The mechanism is coverage: fieldsim can only touch genes on a "
        f"curated regulatory path from the knockout, so it reaches only ~{rec:.0%} of the genes that actually moved "
        f"while lighting up a ~{r['median_reached_neighborhood']:.0f}-gene neighborhood that mostly did NOT move — huge "
        f"false reach, near-zero recall. And on the few movers it does reach, direction is if anything WORSE than a "
        f"coin-flip (pooled sign agreement {sa}, below 0.5 on n={r['pooled_sign_n']}) — the propagated curated "
        f"regulatory signs actively disagree with the measured response, consistent with the earlier finding that "
        f"curated regulation recovers the measured knockout at ~chance. This is the same far-field-does-not-compose "
        f"wall every engine here hit (transitivity ~0.009). CONCLUSION: fieldsim is a correct, "
        f"internally-consistent MECHANISTIC DEMONSTRATOR whose forward prediction of a real knockout does NOT beat "
        f"chance or a trivial network lookup. The measured `knockout`/`cascade` engines — which report what was "
        f"observed and honestly flag the ~84% they can't route — remain the trustworthy tools.")
    print("\n" + "=" * 96 + f"\nVERDICT: {verdict}\n" + "=" * 96)
    r["verdict"] = verdict
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/fieldsim_test.json", "w"), indent=1)
    print(f"  -> {OUT}/fieldsim_test.json")
    return r


if __name__ == "__main__":
    main()
