"""CAN THE NETWORK COMPLETE ITSELF? Feed the Markov chain's own predictions back into the graph and iterate.

THE QUESTION. link_completion.py showed the 2-step K562 chain recovers held-out PPI edges at AUC 0.938 under a joint
degree+pubs control, beating plain common-neighbours. So run it in a loop: predict edges, ADD them to the graph,
recompute the chain on the augmented graph, predict again. Does the network bootstrap itself toward completeness?

THE FAILURE MODE THIS IS BUILT TO DETECT, and the reason a loop is not obviously a good idea. Adding an edge between
two proteins that already share many neighbours CREATES new shared neighbours for their neighbours, which justifies
more edges in the same dense region, which creates more shared neighbours. A self-completing graph can therefore
collapse INWARD -- getting denser exactly where it was already dense, adding no information and looking more
confident with every round. link_completion already measured the ingredient that makes this likely: 79.7% of
held-out edges have >10 shared neighbours, and in the CN=0 stratum every graph method sits at chance.

So the loop is scored on three things per round, not one:
  PRECISION of that round's additions -- what fraction are genuinely held-out true edges
  WHERE they land               -- mean shared-neighbour count of the additions (the self-confirmation tell)
  WHAT IT DOES TO RECOVERY      -- AUC on the still-unrecovered held-out edges

THE CONTROL THAT DECIDES IT. NO-FEEDBACK: rank once on the ORIGINAL graph and take the top M*R pairs in a single
shot, never recomputing. If iterating is no better than one-shot, the feedback loop is decoration -- the chain is
just re-reading the structure it already had. This is the same question as "does the extra machinery earn its cost"
that the ensemble and Monte Carlo work kept answering no to, asked of a loop.

WHAT "TRUE" MEANS HERE, stated plainly because it bounds every number below. 20% of the KO-KO PPI edges are held out.
An addition counts as TRUE only if it is one of those. An addition that is not is scored false -- but the whole
premise of network completion is that the database is incomplete, so some of those "false" additions may be real
undiscovered edges. The precision reported here is therefore a LOWER BOUND on correctness, and it is valid for
COMPARING rounds and arms, which is what it is used for. It is not a discovery rate.
"""
import collections
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
HOLDOUT, SEED = 0.20, 0
ROUNDS, PER_ROUND = 6, 500


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); nK = len(kos)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    konode = np.array([nidx[k] for k in kos])

    f = h5py.File(f"{SP}/k562.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["__categories"]["gene_name"][:]]
    gname = np.array(cats)[f["var"]["gene_name"][:]]
    expressed = np.zeros(N, bool)
    for n_ in set(gname.tolist()):
        if n_ in nidx:
            expressed[nidx[n_]] = True

    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)

    kmap = {int(v): i for i, v in enumerate(konode)}
    ko_edges = sorted(set((min(kmap[a], kmap[b]), max(kmap[a], kmap[b]))
                          for a, b in pe if a in kmap and b in kmap))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ko_edges))
    nho = int(len(ko_edges) * HOLDOUT)
    held = set(ko_edges[i] for i in perm[:nho])
    trainE = set(ko_edges[i] for i in perm[nho:])
    print(f"{len(ko_edges)} KO-KO edges: {len(trainE)} kept, {len(held)} held out; "
          f"{nK*(nK-1)//2:,} candidate pairs", flush=True)

    # the non-KO part of the graph never changes; only the KO-KO block is edited by the loop
    other_a, other_b = [], []
    for a, b in pe:
        if a in kmap and b in kmap:
            continue
        other_a.append(a); other_b.append(b)
    other_a = np.array(other_a); other_b = np.array(other_b)

    def chain2(edges):
        """2-step K562-expressed Markov reachability over the KO-KO block, given the current KO-KO edge set."""
        ea = np.array([konode[a] for a, b in edges]); eb = np.array([konode[b] for a, b in edges])
        ra = np.r_[other_a, other_b, ea, eb]; rb = np.r_[other_b, other_a, eb, ea]
        m = expressed[ra] & expressed[rb]
        G = sparse.coo_matrix((np.ones(int(m.sum()), np.float32), (ra[m], rb[m])), shape=(N, N)).tocsr()
        mag = np.asarray(G.sum(1)).ravel(); inv = np.zeros(N, np.float32)
        nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        P = (sparse.diags(inv) @ G).tocsr()
        return np.asarray((P[konode] @ P[:, konode]).todense(), np.float32)

    def shared_nb(edges):
        ea = np.array([konode[a] for a, b in edges]); eb = np.array([konode[b] for a, b in edges])
        ra = np.r_[other_a, other_b, ea, eb]; rb = np.r_[other_b, other_a, eb, ea]
        A = sparse.coo_matrix((np.ones(len(ra), np.float32), (ra, rb)), shape=(N, N)).tocsr()
        A.data[:] = 1.0
        return np.asarray((A[konode] @ A[:, konode]).todense(), np.float32)

    iu = np.triu_indices(nK, 1)
    code = iu[0].astype(np.int64) * nK + iu[1].astype(np.int64)
    code_held = np.array([a * nK + b for a, b in held], np.int64)
    is_held = np.isin(code, code_held)

    def run(feedback, label):
        cur = set(trainE); added = set()
        rows = []
        S0 = None
        for r in range(ROUNDS):
            if feedback or S0 is None:
                S = chain2(cur)
                CNm = shared_nb(cur)
                if S0 is None:
                    S0, CN0 = S, CNm
            else:
                S, CNm = S0, CN0
            sc = S[iu].copy()
            known = np.isin(code, np.array([a * nK + b for a, b in cur], np.int64)) if cur else np.zeros(len(code), bool)
            sc[known] = -np.inf
            take = np.argsort(-sc)[:PER_ROUND * (1 if feedback else ROUNDS)] if not feedback else \
                np.argsort(-sc)[:PER_ROUND]
            newp = [(int(iu[0][t]), int(iu[1][t])) for t in take]
            hits = sum(1 for p in newp if p in held)
            cnv = float(np.mean([CNm[a, b] for a, b in newp])) if newp else 0.0
            added |= set(newp); cur |= set(newp)
            # AUC on the held-out edges NOT yet added, against all still-unknown pairs
            rem = ~np.isin(code, np.array([a * nK + b for a, b in cur], np.int64))
            y = is_held[rem]
            auc = float(roc_auc_score(y, S[iu][rem])) if y.sum() > 20 and (~y).sum() > 20 else float("nan")
            rows.append({"round": r + 1, "added": len(newp), "true": hits,
                         "precision": hits / max(len(newp), 1), "mean_shared_nb": cnv,
                         "cum_true": sum(x["true"] for x in rows) + hits, "auc_remaining": auc})
            print(f"    {label} round {r+1}: added {len(newp):4d}  true {hits:4d}  "
                  f"precision {hits/max(len(newp),1):.3f}  mean sharedNb {cnv:6.1f}  "
                  f"AUC(remaining) {auc:.4f}", flush=True)
            if not feedback:
                break
        return rows, added

    print("\n  ITERATIVE SELF-COMPLETION (chain recomputed on the augmented graph each round)")
    fb_rows, fb_added = run(True, "FEEDBACK ")
    print("\n  CONTROL -- NO FEEDBACK (rank once on the original graph, take the same total in one shot)")
    nf_rows, nf_added = run(False, "ONE-SHOT ")

    fb_true = sum(r["true"] for r in fb_rows); fb_n = sum(r["added"] for r in fb_rows)
    nf_true, nf_n = nf_rows[0]["true"], nf_rows[0]["added"]
    print(f"\n  TOTALS")
    print(f"    FEEDBACK  {fb_true}/{fb_n} additions are real held-out edges  ({fb_true/max(fb_n,1):.3f})")
    print(f"    ONE-SHOT  {nf_true}/{nf_n} additions are real held-out edges  ({nf_true/max(nf_n,1):.3f})")
    print(f"    held-out edges recovered: FEEDBACK {fb_true}/{len(held)} = {fb_true/len(held):.1%}, "
          f"ONE-SHOT {nf_true}/{len(held)} = {nf_true/len(held):.1%}")

    p1, plast = fb_rows[0]["precision"], fb_rows[-1]["precision"]
    cn1, cnlast = fb_rows[0]["mean_shared_nb"], fb_rows[-1]["mean_shared_nb"]
    helps = fb_true > nf_true * 1.05
    decays = plast < p1 * 0.8

    verdict = (
        f"CAN THE NETWORK COMPLETE ITSELF? Feeding the 2-step K562 chain's own predictions back into the graph and "
        f"re-running it for {ROUNDS} rounds of {PER_ROUND} edges adds {fb_n} edges, of which {fb_true} "
        f"({fb_true/max(fb_n,1):.1%}) are genuinely held-out true edges -- recovering {fb_true/len(held):.1%} of the "
        f"{len(held)} hidden edges. "
        + (f"BUT THE FEEDBACK LOOP IS NOT WHAT DOES IT. Ranking ONCE on the original graph and taking the same "
           f"{nf_n} pairs in a single shot recovers {nf_true} ({nf_true/max(nf_n,1):.1%}), which is "
           f"{'no worse' if nf_true >= fb_true else 'close'}. Recomputing the chain on the augmented graph buys "
           f"{fb_true - nf_true:+d} edges, so the loop is re-reading structure it already had. "
           if not helps else
           f"AND THE FEEDBACK EARNS ITS COST: the one-shot control taking the same {nf_n} pairs recovers only "
           f"{nf_true} ({nf_true/max(nf_n,1):.1%}), so iterating adds {fb_true - nf_true:+d} true edges. ")
        + (f"AND PRECISION DECAYS AS IT GOES: round 1 is {p1:.3f} and round {ROUNDS} is {plast:.3f}. "
           if decays else f"Precision holds up across rounds ({p1:.3f} -> {plast:.3f}). ")
        + f"THE SELF-CONFIRMATION TELL is where the additions land: the mean shared-neighbour count of the edges "
        f"added goes {cn1:.1f} -> {cnlast:.1f} from the first round to the last. "
        + (f"It RISES, which is the collapse-inward signature -- each round adds edges in denser territory than the "
           f"last, because the previous round's additions manufactured the shared neighbours that justify them. The "
           f"graph is getting more confident about the part it already knew. "
           if cnlast > cn1 else
           f"It does not rise, so the loop is at least not feeding on its own additions. ")
        + f"WHAT THIS CANNOT SAY. An addition is scored TRUE only if it is one of the {len(held)} held-out edges; the "
        f"premise of completion is that the database is incomplete, so some additions scored false may be real "
        f"undiscovered edges. Every precision here is therefore a LOWER BOUND, valid for comparing rounds and arms "
        f"-- which is all it is used for -- and NOT a discovery rate. The universe is knockout-knockout pairs only.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"feedback": fb_rows, "one_shot": nf_rows, "held": len(held),
               "feedback_true": fb_true, "one_shot_true": nf_true,
               "feedback_helps": bool(helps), "precision_decays": bool(decays),
               "verdict": verdict}, open(OUT / "self_completion.json", "w"), indent=2)
    print(f"\n  -> {OUT/'self_completion.json'}")


if __name__ == "__main__":
    main()
