"""THE CHAIN, SCORED AGAINST PERTURB-SEQ: does propagating a knockout through the typed network predict what moves?

WHAT THE CHAIN IS. Everything built today assembled into one propagation. Knock out gene X, then follow, in typed
steps, what that breaks:

    X  ->  reactions X catalyses or participates in   ->  their other participants and products
       ->  PPI partners of X
       ->  genes X transcriptionally regulates
       ->  pathways downstream of X's pathways

and score the resulting gene ranking against what actually moved in K562.

TWO SCORING MODES, BECAUSE THEY ASK DIFFERENT QUESTIONS.

  DIRECT    the chain alone ranks genes; score recall@50 of the knockout's specific movers. This is the pure
            mechanistic test -- nothing is borrowed from measured data, so it is the honest question "does the
            network know what this knockout does". It is also much harder, and a low number here is expected.
  TRANSFER  use the chain as a SIMILARITY between knockouts, rank TRAIN knockouts by it, and average their MEASURED
            profiles. This is exactly how the incumbent (0.4736) works, so it is the apples-to-apples comparison.
            A gain here means the chain identifies better neighbours than the raw PPI graph does.

PRE-REGISTERED EXPECTATION, WRITTEN BEFORE RUNNING. Perturb-seq measures mRNA. Most of this network is protein and
metabolite level, and an earlier measurement in this project found that of 3,164 unambiguously directed PPI edges,
only 2 had a non-trivial measured effect in both directions -- protein interaction and transcriptional response are
different layers. So DIRECT should be poor. The transcriptional arm is the one layer that speaks the readout's
language, and it is also the one with a prior warning against it (below). If anything beats the baseline it should
be the signed regulation.

THE TF LAYER IS SPLIT IN TWO, ON THE STRENGTH OF AN EARLIER MEASUREMENT RATHER THAN A GUESS. `reg` holds 612,133
edges, but 558,005 of them are UNSIGNED -- ChIP-derived binding, not demonstrated regulation. bind_vs_reg.json
measured GATA1 binding against Perturb-seq regulation in this same cell line and found the overlap AT CHANCE: 4,621
bound genes, 56 regulated, 29 overlapping against 30.2 expected, fold 0.96, p=0.68, verified by permutation and four
adversarial lenses. So REG-ALL and REG-SIGNED (54,128 edges) are separate arms and the harness decides, rather than
folding a layer in that has already failed once.

CONTROLS. Every layer gets a degree-preserving edge shuffle, because a layer that helps only by adding connectivity
should be caught. The harness itself is unchanged: tide-removed specific-mover recall@50, TAU=1.0, TIDE_FRAC=0.05,
MIN_SPEC=5, N_NEI=10, 5 splits, paired bootstrap, hyperparameters chosen on TRAIN only.
"""
import collections
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI, NSPLIT, NBOOT = 1.0, 50, 5, 0.05, 10, 5, 2000
DAMP = [(1, 1.0), (2, 0.5), (3, 0.5)]


def main():
    KO = pickle.load(open(SP / "nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v})
    gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide = np.where(~tide)[0]
    spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}
    N = len(names)

    def edges_ppi():
        e = set()
        for x in D.get("ppi", []):
            try:
                a, b = int(x[0]), int(x[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= a < N and 0 <= b < N and a != b:
                e.add((a, b))
                e.add((b, a))
        return e

    def edges_reg(signed_only):
        e = set()
        for x in D.get("reg", []):
            try:
                s, t = int(x[0]), int(x[1])
                w = int(x[2]) if len(x) > 2 else 0
            except (TypeError, ValueError, IndexError):
                continue
            if signed_only and w == 0:
                continue
            if 0 <= s < N and 0 <= t < N and s != t:
                e.add((s, t))                      # directed: TF -> target
        return e

    def edges_rxn():
        """Two genes are linked when they participate in the same reaction. Currency species are excluded by
        construction because they carry no gene, and steps with more than 60 gene participants are skipped -- the
        ribosome would otherwise contribute a 90-clique and swamp everything."""
        e = set()
        net = json.load(open(OUT / "reaction_network.json"))
        for s in net["steps"]:
            gs = set(s["catalysts"])
            for p in s["in"] + s["out"]:
                gs |= set(p["genes"])
            ids = [nidx[g] for g in gs if g in nidx]
            if not (2 <= len(ids) <= 60):
                continue
            for a in ids:
                for b in ids:
                    if a != b:
                        e.add((a, b))
        return e

    LAYERS = {"PPI": edges_ppi(), "RXN": edges_rxn(),
              "REG-ALL": edges_reg(False), "REG-SIGNED": edges_reg(True)}
    for k, v in LAYERS.items():
        print(f"  layer {k:<11} {len(v):>9,} directed arcs", flush=True)

    rng = np.random.RandomState(0)

    def shuffled(e):
        """Degree-preserving-ish shuffle: keep each source, resample targets from the same global pool."""
        srcs = np.array([a for a, _ in e])
        tgts = np.array([b for _, b in e])
        return set(zip(srcs.tolist(), rng.permutation(tgts).tolist()))

    def op(edgesets):
        e = set()
        for s in edgesets:
            e |= s
        if not e:
            return None
        a = np.array([x for x, _ in e])
        b = np.array([y for _, y in e])
        G = sparse.coo_matrix((np.ones(len(a), np.float32), (a, b)), shape=(N, N)).tocsr()
        mag = np.asarray(G.sum(1)).ravel()
        inv = np.zeros(N, np.float32)
        nz = mag > 0
        inv[nz] = 1.0 / mag[nz]
        return (sparse.diags(inv) @ G).T.tocsr()

    konode = np.array([nidx.get(k, -1) for k in kos])
    have = konode >= 0
    gnode = np.array([nidx.get(g, -1) for g in genes])
    ghave = gnode >= 0
    print(f"  {int(have.sum()):,}/{nK:,} knockouts and {int(ghave.sum()):,}/{nG:,} readout genes are in the network")

    def spread(PT, nstep=3):
        """s-step distributions from every knockout node, returned as (knockout x network-node) matrices."""
        idx = np.where(have)[0]
        L = np.zeros((N, len(idx)), np.float32)
        L[konode[idx], np.arange(len(idx))] = 1.0
        cur, out = L, []
        for _ in range(nstep):
            cur = PT @ cur
            out.append(cur.copy())
        return idx, out

    ARMS = {
        "PPI (baseline)": ["PPI"],
        "PPI+RXN": ["PPI", "RXN"],
        "PPI+REG-ALL": ["PPI", "REG-ALL"],
        "PPI+REG-SIGNED": ["PPI", "REG-SIGNED"],
        "PPI+RXN+REG-SIGNED": ["PPI", "RXN", "REG-SIGNED"],
        "ALL": ["PPI", "RXN", "REG-ALL", "REG-SIGNED"],
        "PPI+RXN+REG-SIGNED-SHUF": ["PPI", "RXN", "SHUF"],
    }
    LAYERS["SHUF"] = shuffled(LAYERS["REG-SIGNED"])

    DIRECT, SIM = {}, {}
    for arm, ls in ARMS.items():
        PT = op([LAYERS[x] for x in ls])
        idx, steps = spread(PT)
        # DIRECT: gene score = accumulated damped mass on that gene's network node
        acc = sum((0.5 ** (s + 1)) * steps[s] for s in range(3))
        gmat = np.zeros((nK, nG), np.float32)
        gmat[np.ix_(idx, np.where(ghave)[0])] = acc[gnode[ghave]].T
        DIRECT[arm] = gmat
        # TRANSFER: similarity between knockouts = mass landing on the other knockout's node
        sm = np.zeros((nK, nK), np.float32)
        sm[np.ix_(idx, idx)] = acc[konode[idx]].T
        SIM[arm] = sm
        print(f"  built {arm}", flush=True)

    def rec(sc, truth):
        o = nontide[np.argsort(-sc[nontide])][:K]
        return len(set(o.tolist()) & truth) / max(len(truth), 1)

    def prof(ix):
        ix = np.asarray(list(ix), int)
        return np.abs(M[ix]).mean(0) if len(ix) else np.zeros(nG, np.float32)

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC])
    perD = collections.defaultdict(list)
    perT = collections.defaultdict(list)
    for sp in range(NSPLIT):
        srng = np.random.RandomState(100 + sp)
        perm = srng.permutation(scor)
        test = np.array(sorted(perm[:len(scor) // 3].tolist()))
        tset = set(test.tolist())
        train = np.array([i for i in range(nK) if i not in tset])
        SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}
        for arm in ARMS:
            for xi in test:
                perD[arm].append(rec(DIRECT[arm][int(xi)], SPEC[int(xi)]))
                v = SIM[arm][int(xi)].astype(np.float64).copy()
                msk = np.full(nK, -np.inf)
                msk[train] = 0.0
                v = v + msk
                v[int(xi)] = -np.inf
                perT[arm].append(rec(prof(np.argsort(-v)[:N_NEI]), SPEC[int(xi)]))
        for xi in test:
            perD["TIDE"].append(rec((A >= TAU).mean(0), SPEC[int(xi)]))
        print(f"  split {sp} done", flush=True)

    def paired(per, a, b):
        x = np.array(per[a]) - np.array(per[b])
        br = np.random.RandomState(7)
        bs = np.array([x[br.randint(0, len(x), len(x))].mean() for _ in range(NBOOT)])
        m, se = float(x.mean()), float(bs.std())
        return m, se, ("BETTER" if m > 2 * se else "WORSE" if m < -2 * se else "indistinguishable")

    res = {}
    for mode, per in (("DIRECT (chain alone, no measured data)", perD),
                      ("TRANSFER (chain as neighbour similarity)", perT)):
        print(f"\n  {mode}   recall@{K}, {NSPLIT} splits")
        print(f"    {'arm':<26} {'score':>8}   {'vs PPI':>9} {'+/-':>7}  verdict")
        for arm in ARMS:
            s = float(np.mean(per[arm]))
            if arm == "PPI (baseline)":
                print(f"    {arm:<26} {s:8.4f}   {'--':>9} {'':>7}  reference")
                res[f"{mode}|{arm}"] = s
                continue
            d, se, v = paired(per, arm, "PPI (baseline)")
            print(f"    {arm:<26} {s:8.4f}   {d:+9.4f} {se:7.4f}  {v}")
            res[f"{mode}|{arm}"] = s
    print(f"\n    TIDE null {np.mean(perD['TIDE']):.4f} | incumbent neighbour-transfer ~0.4736")

    json.dump({"layers": {k: len(v) for k, v in LAYERS.items()}, "scores": res,
               "tide": float(np.mean(perD["TIDE"]))}, open(OUT / "chain_test.json", "w"))
    print(f"\n  -> {OUT/'chain_test.json'}")


if __name__ == "__main__":
    main()
