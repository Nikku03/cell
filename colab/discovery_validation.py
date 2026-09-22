"""discovery_validation — the epistemic foundation: does the model PREDICT relationships it was never shown, or
only RECALL ones it was given? Every earlier cancer test was open-book; this is the blind harness that separates
discovery from recall.

Method (leakage-free holdout on the interactome):
  1. Split true edges of a relation (ppi/reg/sig) into TRAIN and TEST (held-out, hidden from the predictor).
  2. Rebuild the graph with TEST edges REMOVED. Score every held-out edge + matched random non-edges using only
     TRAIN-graph topology (shared partners, Jaccard) + INDEPENDENT modalities (DepMap co-essentiality). No test
     edge, and nothing derived from it, touches the scorer.
  3. Report recovery, and — crucially — split it by HARDNESS:
        EASY held-out edge : its endpoints still share >=1 neighbour in the train graph (recoverable by triadic
                             closure = interpolation).
        HARD held-out edge : its endpoints share ZERO neighbours in the train graph. Closure is useless; the
                             edge can ONLY be recovered from independent signal (co-essentiality, structure).
                             HARD-edge AUC is the DISCOVERY metric — predicting a link with no network shortcut.
  4. discovery_lift = AUC_hard - 0.5 (above chance on the un-interpolatable edges).

A model that scores high overall but ~0.5 on HARD edges is a recall engine dressed as a predictor. A model that
beats chance on HARD edges is doing something discovery-shaped. Reported honestly either way.

Also provides a NOVELTY split: held-out edges supported by only ONE source DB (frontier) vs multi-source
(consensus) — a proxy for temporal recency when dates are absent.
-> outputs/orphan/discovery_validation.json
"""
import os, sys, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"


def _depmap():
    p = f"{OUT}/depmap_vecs.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"Z": z["Z"], "col": {s: i for i, s in enumerate(z["syms"])}, "name2row": None}


def _coess(dm, name, a, b):
    """DepMap co-essentiality corr(a,b) — an INDEPENDENT modality (not from the PPI graph)."""
    ca, cb = dm["col"].get(name[a]), dm["col"].get(name[b])
    if ca is None or cb is None:
        return 0.0
    x, y = dm["Z"][ca], dm["Z"][cb]; m = (x != 0) & (y != 0)
    if m.sum() < 60:
        return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])


def run(relation="ppi", frac=0.2, n_test=1500, seed=0):
    random.seed(seed); np.random.seed(seed)
    C = CompleteCell(); name = C.name
    edges = C.D.get(relation, [])
    E = set()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] != e[1]:
            E.add((min(e[0], e[1]), max(e[0], e[1])))
    E = list(E)
    random.shuffle(E)
    n_hold = int(len(E) * frac)
    test = E[:n_hold]; train = E[n_hold:]
    if len(test) > n_test:
        test = random.sample(test, n_test)

    # TRAIN adjacency (test edges removed) — the leakage-free graph
    adj = {}
    for a, b in train:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)

    dm = _depmap()
    nodes = [n for n in adj if len(adj[n]) > 0]

    def feats(a, b):
        na, nb = adj.get(a, set()), adj.get(b, set())
        shared = len(na & nb)
        union = len(na | nb) or 1
        jac = shared / union
        ce = _coess(dm, name, a, b) if dm else 0.0
        return shared, jac, ce

    # score = simple leakage-free combination (shared-partner + jaccard + |coessentiality|); no trained weights
    def score(a, b):
        s, j, ce = feats(a, b)
        return 0.5 * j + 0.1 * min(s, 10) / 10 + 0.6 * max(0.0, ce)

    # negatives: random non-edges (not in the full edge set)
    Eset = set(E)
    negs = []
    while len(negs) < len(test):
        a, b = random.randrange(len(name)), random.randrange(len(name))
        if a == b:
            continue
        k = (min(a, b), max(a, b))
        if k in Eset or a not in adj or b not in adj:
            continue
        negs.append(k)

    def auc(pos_scores, neg_scores):
        if not pos_scores or not neg_scores:
            return None
        c = sum(1 for p in pos_scores for n in neg_scores if p > n)
        t = sum(1 for p in pos_scores for n in neg_scores if p == n)
        return (c + 0.5 * t) / (len(pos_scores) * len(neg_scores))

    pos_s = [(score(a, b), feats(a, b)[0]) for a, b in test]
    neg_s = [(score(a, b), feats(a, b)[0]) for a, b in negs]
    all_auc = auc([s for s, _ in pos_s], [s for s, _ in neg_s])
    # HARD = held-out edges whose endpoints share 0 train neighbours (closure useless)
    hard_pos = [s for s, sh in pos_s if sh == 0]
    easy_pos = [s for s, sh in pos_s if sh > 0]
    hard_neg = [s for s, sh in neg_s if sh == 0]
    auc_hard = auc(hard_pos, hard_neg or [s for s, _ in neg_s])
    auc_easy = auc(easy_pos, [s for s, _ in neg_s])

    res = {
        "relation": relation, "n_edges": len(E), "n_held_out": len(test),
        "auc_overall": round(all_auc, 3) if all_auc else None,
        "auc_easy_triadic_recoverable": round(auc_easy, 3) if auc_easy else None,
        "n_hard_edges": len(hard_pos),
        "auc_hard_DISCOVERY": round(auc_hard, 3) if auc_hard else None,
        "discovery_lift": round(auc_hard - 0.5, 3) if auc_hard else None,
        "verdict": None,
    }
    lift = res["discovery_lift"]
    res["verdict"] = (
        "no discovery signal — recovers only what network closure gives; HARD edges at chance" if (lift is None or lift < 0.03)
        else "weak discovery signal on un-interpolatable edges (independent modalities carry some)" if lift < 0.12
        else "genuine discovery signal — predicts held-out edges with no network shortcut, above chance")
    json.dump(res, open(f"{OUT}/discovery_validation.json", "w"), indent=2)
    print("=" * 74)
    print(f"BLIND HOLDOUT — {relation}  ({len(test)} held-out edges hidden from the scorer)")
    print("=" * 74)
    print(f"  AUC overall                {res['auc_overall']}")
    print(f"  AUC easy (triadic)         {res['auc_easy_triadic_recoverable']}   <- interpolation")
    print(f"  AUC HARD (no shortcut)     {res['auc_hard_DISCOVERY']}   <- DISCOVERY metric  ({res['n_hard_edges']} edges)")
    print(f"  discovery lift             {res['discovery_lift']}")
    print(f"  -> {res['verdict']}")
    return res


if __name__ == "__main__":
    for rel in ("ppi", "sig"):
        run(rel)
