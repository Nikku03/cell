"""mechanistic_hybrid -- the user's refinement: ROUTE by the knocked-out gene's type, then COMBINE with retrieval.

Idea: a TF knockout acts through TF->target regulation (the transcriptional graph we DO have -- TRRUST + SIGNOR/CollecTRI), so for TFs
propagate the regulatory network; a plain pathway protein has no TF->DNA edges, so fall back to retrieval. Then blend the mechanism into the
retrieval prediction, ONLY where it applies (TF knockouts).

Tests, held-out K562, 3 splits:
  - is the specific response of TF knockouts more predictable than non-TF? (does routing by type even carve a better-predicted class?)
  - on TF knockouts, does the regulatory-propagation add anything to retrieval (val-tuned blend), or does retrieval already dominate?
  - does the routed hybrid beat retrieval overall?
Deterministic; no training (DepMap-retrieval + fixed regulatory propagation)."""
import os
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy
from transformer_causal import build_causal

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
K = 50


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    reg, _, sources = build_causal(H)                 # directed signed TF->target graph; sources = the TFs/regulators

    def propagate(g, iters=3, alpha=0.6):             # near-field signed propagation (3 hops; convergence over-diffuses)
        infl = collections.defaultdict(float); frontier = {g: 1.0}
        for _ in range(iters):
            nxt = collections.defaultdict(float)
            for node, w in frontier.items():
                for tgt, s in reg.get(node, []):
                    c = w * (s if s != 0 else 0.5) * alpha
                    infl[tgt] += c; nxt[tgt] += c
            if not nxt:
                break
            frontier = nxt
        v = np.zeros(H.nG)
        for nm, x in infl.items():
            j = H.gi.get(nm)
            if j is not None:
                v[j] += abs(x)
        return v

    prop_cache = {ko: propagate(ko) for ko in H.scorable if ko in sources}

    def spec_recall(vec, ko):
        r = H.ki[ko]; sp = set(int(i) for i in np.where(H.mover[r])[0] if H.nontide[i])
        return H._recall(vec, sp, H.nontide_idx, K) if sp else np.nan

    def zt(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    seeds = [0, 1, 2]
    agg = collections.defaultdict(list)
    for s in seeds:
        rng = np.random.RandomState(s)
        scor = [k for k in H.scorable if k in hidx]
        perm = rng.permutation(len(scor)); nt = int(0.30 * len(scor))
        test = sorted(scor[i] for i in perm[:nt]); test_set = set(test)
        rest = [k for k in scor if k not in test_set]
        vperm = rng.permutation(len(rest)); nv = int(0.4 * len(rest))
        val = [rest[i] for i in vperm[:nv]]
        tr = np.array([hidx[k] for k in have if k not in test_set and k not in set(val)])

        def retr(ko):
            i = hidx[ko]; sim = Xn[tr] @ Xn[i]; return Y[tr[np.argsort(-sim)[:10]]].mean(0)

        # tune blend beta on VAL TF-knockouts only
        val_tf = [k for k in val if k in prop_cache]
        best_b, best_v = 0.0, -1
        for b in (0.0, 0.15, 0.3, 0.5, 0.7):
            rs = []
            for ko in val_tf:
                pv = prop_cache[ko]
                comb = zt(retr(ko)) if b == 0 else (1 - b) * zt(retr(ko)) + b * zt(pv)
                rs.append(spec_recall(comb, ko))
            v = np.nanmean(rs) if rs else -1
            if v > best_v:
                best_v, best_b = v, b

        # evaluate on TEST, split by TF vs non-TF
        r_all, r_tf, r_ntf, prop_tf, hyb_all, hyb_tf = [], [], [], [], [], []
        for ko in test:
            rv = retr(ko); rr = spec_recall(rv, ko)
            is_tf = ko in prop_cache
            r_all.append(rr)
            if is_tf:
                r_tf.append(rr)
                pv = prop_cache[ko]
                prop_tf.append(spec_recall(pv, ko))
                comb = (1 - best_b) * zt(rv) + best_b * zt(pv) if best_b > 0 else rv
                hr = spec_recall(comb, ko); hyb_tf.append(hr); hyb_all.append(hr)
            else:
                r_ntf.append(rr); hyb_all.append(rr)
        m = lambda a: float(np.nanmean(a)) if len(a) else float("nan")
        agg["retr_all"].append(m(r_all)); agg["retr_tf"].append(m(r_tf)); agg["retr_ntf"].append(m(r_ntf))
        agg["prop_tf"].append(m(prop_tf)); agg["hybrid_all"].append(m(hyb_all)); agg["hybrid_tf"].append(m(hyb_tf))
        agg["beta"].append(best_b); agg["n_tf"].append(len(r_tf)); agg["n_test"].append(len(test))

    A = {k: (float(np.mean(v)), float(np.std(v))) for k, v in agg.items()}
    tide = float(np.mean([spec_recall(H.mover_freq, k) for k in H.scorable]))
    print(f"held-out K562, 3 splits. TF knockouts ~{A['n_tf'][0]:.0f} of {A['n_test'][0]:.0f} test KOs. tide-null {tide:.3f}\n")
    print(f"  retrieval, ALL KOs            {A['retr_all'][0]:.3f} +/- {A['retr_all'][1]:.3f}")
    print(f"  retrieval, TF knockouts       {A['retr_tf'][0]:.3f} +/- {A['retr_tf'][1]:.3f}")
    print(f"  retrieval, non-TF knockouts   {A['retr_ntf'][0]:.3f} +/- {A['retr_ntf'][1]:.3f}")
    print(f"  regulatory PROPAGATION, TF    {A['prop_tf'][0]:.3f} +/- {A['prop_tf'][1]:.3f}   (mechanism alone, on TFs)")
    print(f"  HYBRID (route TF->blend), TF  {A['hybrid_tf'][0]:.3f}   (tuned beta {A['beta'][0]:.2f})")
    print(f"  HYBRID overall                {A['hybrid_all'][0]:.3f}   vs retrieval {A['retr_all'][0]:.3f}  "
          f"({A['hybrid_all'][0]-A['retr_all'][0]:+.3f})")

    d_hyb = A['hybrid_all'][0] - A['retr_all'][0]
    tf_more_pred = A['retr_tf'][0] > A['retr_ntf'][0] + 0.02
    verdict = (
        "MECHANISTIC HYBRID (mechanistic_hybrid.py): route by knocked-out gene type -- TF knockouts propagated through the regulatory "
        f"(TF->target) network, everything else via retrieval, then blended. 3 splits, held-out K562. "
        f"TF-knockout specific responses are {'MORE' if tf_more_pred else 'NOT more'} predictable than non-TF "
        f"(retrieval {A['retr_tf'][0]:.3f} vs {A['retr_ntf'][0]:.3f}). On TF knockouts, the regulatory PROPAGATION alone scores "
        f"{A['prop_tf'][0]:.3f} vs retrieval {A['retr_tf'][0]:.3f} (tide {tide:.3f}) -- "
        + ("propagation is competitive." if A['prop_tf'][0] > A['retr_tf'][0] else
           "retrieval already dominates the mechanism even where the mechanism should apply, and the val-tuned blend picks "
           f"beta={A['beta'][0]:.2f}. ")
        + f"The routed hybrid scores {A['hybrid_all'][0]:.3f} vs retrieval-alone {A['retr_all'][0]:.3f} ({d_hyb:+.3f}) -- "
        + ("the type-routed mechanism HELPS." if d_hyb > 0.01 else
           "no gain. WHY: even for TFs -- where the regulatory graph is the RIGHT graph -- its targets are the CANONICAL (generic/tide) targets, "
           "which retrieval already captures and which the tide-removed metric discards; the TF-knockout-SPECIFIC residue is the non-canonical "
           "part the static regulatory network does not name. So combining + type-routing is sound in principle but adds nothing on THIS data: "
           "the regulatory network encodes generic regulation, and the specific response is context-specific biology missing from it.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({k: [round(A[k][0], 4), round(A[k][1], 4)] for k in A} | {"tide": round(tide, 4),
              "hybrid_minus_retrieval": round(d_hyb, 4), "verdict": verdict, "note": verdict},
              open(OUT / "mechanistic_hybrid.json", "w"), indent=1)
    print("\n  -> outputs/orphan/mechanistic_hybrid.json")


if __name__ == "__main__":
    main()
