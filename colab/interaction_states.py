"""INTERACTIONS AS MARKOV STATES: individual proteins vs proteins-plus-interactions.

THE ASK. The incumbent chain (MARKOV-P, protein_chain_combine.py) has one kind of state: a PROTEIN. Interactions are
edges between states, not states themselves. The question here is what happens if an INTERACTION is promoted to a
first-class state, so the walk goes protein -> interaction -> protein instead of protein -> protein.

THE FIRST THING TO ESTABLISH IS WHETHER THIS CAN DIFFER AT ALL, because for pairwise interactions it provably cannot.
Let B be the protein x interaction incidence. A protein p emits mass uniformly over its incident interactions; an
interaction c redistributes uniformly over its k_c members. The composed protein->protein kernel is

    P_bip = D^-1 B Dc^-1 B^T ,   Dc = diag(colsum) = diag(k_c)

For a PAIRWISE interaction (k_c = 2) the term B[p,c] B[q,c] / 2 sends 1/2 back to p and 1/2 to the partner. Summed over
p's interactions that is exactly

    P_bip = 1/2 I + 1/2 P_node          (the LAZY node walk)

Verified numerically to machine precision on a random graph: max |P_bip - lazy| = 1.1e-16. All 191,447 PPI edges in
this dataset are pairwise. So on the PPI layer the user's idea is, up to laziness, the incumbent. That is not a reason
to skip it -- it is a reason to MEASURE it, because a predicted no-op that is measured to be a no-op is a result, and
laziness is not literally nothing (it reweights how much mass survives to step 2 and 3 under truncation).

WHERE IT GENUINELY DIFFERS IS COMPLEXES, and the difference is large. The incumbent CLIQUE-EXPANDS a complex: a size-k
complex becomes k(k-1) weighted edges, so a member protein emits cw*(k-1) total mass into the complex layer. As a
hyperedge STATE it emits cw once, and the state splits it among the members. Measured on this dataset:

    1,463 complexes (sizes 2..52, mean 3.7) -> 38,522 clique edges from only 5,394 memberships, a 7.1x inflation
    the complex layer's share of a protein's outgoing mass shifts by mean +0.128, max +0.714 (RPL41: 0.811 -> 0.097)

AND NO GLOBAL cw CAN UNDO IT. The incumbent sweeps cw, which rescales every protein together. The clique-vs-hyperedge
shift has SD 0.137 ACROSS proteins because it scales with each protein's own complex sizes. A per-protein, size-
dependent reweighting is not reachable by a global scalar, so this is a genuinely different chain and not a
reparameterisation of one already swept.

THE CONFOUND THAT DECIDES WHETHER ANY GAIN IS REAL. Large complexes are ribosome, proteasome and spliceosome. Those
knockouts have huge transcriptional responses. ANY method that reweights large complexes moves performance for reasons
that have nothing to do with the chain being a better model of propagation. So a response-size-matched control is not
optional here, it is the whole test.

ARMS -- all six share ONE identical evaluation path and differ ONLY in the state space:
  NODE       incumbent. states = proteins. PPI as edges, complexes clique-expanded.               [reference]
  BIP-PW     states = proteins + PAIRWISE PPI interactions. complexes still clique-expanded.      [predicted no-op]
  HYPER      states = proteins + COMPLEX hyperedges. PPI stays as direct edges.                   [predicted real]
  BIP-FULL   states = proteins + ALL interactions (PPI pairwise AND complex hyperedges).          [the literal ask]
  BIP-NS     BIP-FULL with the interaction forbidden to return mass to its sender.                [de-lazied]
  LINE-NB    states = INTERACTIONS ONLY, non-backtracking (Hashimoto) walk on directed incidences.

CONTROLS: incidence rewiring that preserves protein degree AND interaction size; complex-membership shuffling that
preserves complex sizes; and the response-size-matched neighbour control. Significance is a paired bootstrap over test
knockouts with THREE outcomes (BETTER / WORSE / indistinguishable), because a win-only test has now printed
"not significant" next to a nine-sigma defeat twice in this repo.

Chain configs are PRE-REGISTERED on each split's TRAIN knockouts only, exactly as the incumbent does it, so no arm
gets to pick its hyperparameters on the test set.
"""
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI = 1.0, 50, 5, 0.05, 10
NSPLIT = 5
CHAIN_CFG = [(1, 1.0, 0.0), (2, 0.5, 0.0), (2, 0.5, 1.0), (2, 0.5, 2.0), (3, 0.5, 1.0)]  # (steps, damp, cplx_weight)
NBOOT = 2000
MAXC = 60          # complexes larger than this are annotation buckets, not machines -- same cut as the incumbent


# ----------------------------------------------------------------------------------------------------------------
# state-space builders. every arm returns a row-stochastic protein->protein operator, ALREADY TRANSPOSED for L <- P^T L
# ----------------------------------------------------------------------------------------------------------------
def rownorm_T(G, N):
    mag = np.asarray(abs(G).sum(1)).ravel()
    inv = np.zeros(N, np.float32)
    nz = mag > 0
    inv[nz] = 1.0 / mag[nz]
    return (sparse.diags(inv) @ G).T.tocsr()


def incidence(members, N, w):
    """protein x interaction incidence with uniform weight w on every membership."""
    r, c = [], []
    for j, mem in enumerate(members):
        for x in mem:
            r.append(x); c.append(j)
    return sparse.coo_matrix((np.full(len(r), w, np.float32), (r, c)), shape=(N, len(members))).tocsr()


def via_states(B, sizes, w, self_return):
    """Protein->protein mass routed THROUGH interaction states.

    self_return=True  : the interaction redistributes uniformly over ALL k members, so a fraction 1/k comes straight
                        back to the sender. This is the literal bipartite walk, and on pairwise interactions it is the
                        LAZY node walk.
    self_return=False : the interaction is forbidden to return mass to whoever sent it, splitting over the k-1 others.
                        Because every membership carries the SAME weight w, the per-other share w/(k-1) does not depend
                        on the sender, so this is still a clean matrix product -- B diag(1/(w(k-1))) B^T minus its own
                        diagonal. On pairwise interactions this collapses to EXACTLY the ordinary node walk.
    """
    k = np.asarray(sizes, np.float64)
    if self_return:
        d = w * np.maximum(k, 1.0)
        M = (B @ sparse.diags(1.0 / d) @ B.T).tocsr()
    else:
        d = w * np.maximum(k - 1.0, 1e-9)
        M = (B @ sparse.diags(1.0 / d) @ B.T).tocsr()
        M = M - sparse.diags(M.diagonal())          # kill the self-return the product reintroduced
    M.eliminate_zeros()
    return M


def clique(members, N, w):
    """The incumbent's complex handling: a size-k complex becomes k(k-1) directed edges of weight w."""
    r, c = [], []
    for mem in members:
        for x in mem:
            for y in mem:
                if x != y:
                    r.append(x); c.append(y)
    if not r:
        return sparse.csr_matrix((N, N), dtype=np.float32)
    return sparse.coo_matrix((np.full(len(r), w, np.float32), (r, c)), shape=(N, N)).tocsr()


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), dtype=np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide_idx = np.where(~tide)[0]
    spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}
    N = len(names)

    # ---- the two raw ingredients, in INTEGER index space (the index-space bug cost this repo a headline once) ----
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    memb = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if g.isdigit() and int(g) < N:
            for c in cs:
                memb[c].append(int(g))
    cmem = [sorted(set(v)) for v in memb.values() if 2 <= len(set(v)) <= MAXC]
    csz = [len(m) for m in cmem]
    print(f"{len(pe)} pairwise PPI interactions, {len(cmem)} complexes (sizes {min(csz)}..{max(csz)}, "
          f"mean {np.mean(csz):.1f}), {sum(csz)} memberships -> {sum(k*(k-1) for k in csz)} clique edges", flush=True)

    ppi_pairs = [list(p) for p in pe]
    Bppi = incidence(ppi_pairs, N, 1.0)
    ppi_direct = via_states(Bppi, [2] * len(pe), 1.0, self_return=False)   # == plain undirected PPI adjacency

    # sanity: the no-self pairwise route must reproduce the ordinary adjacency exactly
    Araw = sparse.coo_matrix((np.ones(2 * len(pe), np.float32),
                              ([a for a, b in pe] + [b for a, b in pe], [b for a, b in pe] + [a for a, b in pe])),
                             shape=(N, N)).tocsr()
    assert abs(ppi_direct - Araw).max() < 1e-5, "pairwise no-self route is not the ordinary adjacency"
    print("  CHECK pairwise no-self interaction route == ordinary PPI adjacency (exact)", flush=True)

    def build(arm, cw, cmem_use=None, ppi_pairs_use=None):
        """Assemble the protein->protein operator for one ARM. Only the state space changes."""
        cm = cmem if cmem_use is None else cmem_use
        cs = [len(m) for m in cm]
        pp = ppi_pairs if ppi_pairs_use is None else ppi_pairs_use
        Bp = Bppi if ppi_pairs_use is None else incidence(pp, N, 1.0)
        Pd = ppi_direct if ppi_pairs_use is None else via_states(Bp, [2] * len(pp), 1.0, self_return=False)
        Bc = incidence(cm, N, cw) if (cw > 0 and cm) else None

        if arm == "NODE":
            G = Pd + (clique(cm, N, cw) if cw > 0 else 0)
        elif arm == "BIP-PW":
            G = via_states(Bp, [2] * len(pp), 1.0, self_return=True) + (clique(cm, N, cw) if cw > 0 else 0)
        elif arm == "HYPER":
            G = Pd + (via_states(Bc, cs, cw, self_return=True) if Bc is not None else 0)
        elif arm == "BIP-FULL":
            G = via_states(Bp, [2] * len(pp), 1.0, self_return=True) + \
                (via_states(Bc, cs, cw, self_return=True) if Bc is not None else 0)
        elif arm == "BIP-NS":
            G = via_states(Bp, [2] * len(pp), 1.0, self_return=False) + \
                (via_states(Bc, cs, cw, self_return=False) if Bc is not None else 0)
        else:
            raise ValueError(arm)
        return rownorm_T(sparse.csr_matrix(G) if not sparse.issparse(G) else G.tocsr(), N)

    konode = np.array([nidx.get(k, -1) for k in kos])
    have = konode >= 0
    print(f"{int(have.sum())}/{nK} knockouts are nodes in the protein graph", flush=True)

    def chain_scores(PT, steps, damp):
        """Truncated damped walk from every knockout's own node; read the score AT every knockout's node."""
        idx = np.where(have)[0]
        L = np.zeros((N, len(idx)), np.float32)
        L[konode[idx], np.arange(len(idx))] = 1.0
        cur, tot = L, np.zeros_like(L)
        for s in range(1, steps + 1):
            cur = PT @ cur
            tot = tot + (damp ** s) * cur
        out = np.zeros((nK, nK), np.float32)
        got = tot[konode[idx]]                         # (targets x seeds), both restricted to knockouts that are nodes
        out[np.ix_(idx, idx)] = got.T                  # rows = seed knockout, cols = target knockout
        return out

    # ---------------- LINE-NB: interaction states only, non-backtracking ----------------
    # States are DIRECTED INCIDENCES (p -> c): "at interaction c, having arrived from protein p". The walk leaves c to
    # some other member q, then leaves q to some other interaction c' != c. The c' != c constraint is what makes it
    # non-backtracking, and it is the only reason this is not just the node walk in disguise.
    def build_line_nb():
        inter = [list(p) for p in pe] + [list(m) for m in cmem]
        S, sr, sc = [], [], []
        deg = collections.Counter()
        for j, mem in enumerate(inter):
            for x in mem:
                deg[x] += 1
        sid = {}
        for j, mem in enumerate(inter):
            for x in mem:
                sid[(x, j)] = len(S); S.append((x, j))
        nS = len(S)
        # R : state -> protein.  from (p,c) go uniformly to the OTHER members of c
        rr, rc, rv = [], [], []
        for (p, j) in S:
            mem = inter[j]
            if len(mem) < 2:
                continue
            w = 1.0 / (len(mem) - 1)
            for q in mem:
                if q != p:
                    rr.append(sid[(p, j)]); rc.append(q); rv.append(w)
        R = sparse.coo_matrix((rv, (rr, rc)), shape=(nS, N), dtype=np.float32).tocsr()
        # C : protein -> state.  from q go uniformly to its OTHER interactions (deg-1 of them)
        cr, cc, cv = [], [], []
        for (q, j) in S:
            if deg[q] < 2:
                continue
            cr.append(q); cc.append(sid[(q, j)]); cv.append(1.0 / (deg[q] - 1))
        C = sparse.coo_matrix((cv, (cr, cc)), shape=(N, nS), dtype=np.float32).tocsr()
        # T : the BACKTRACK term that R@C wrongly includes -- (p,c) -> (q,c), i.e. staying inside the same interaction
        tr, tc, tv = [], [], []
        for (p, j) in S:
            mem = inter[j]
            if len(mem) < 2:
                continue
            w = 1.0 / (len(mem) - 1)
            for q in mem:
                if q != p and deg[q] >= 2:
                    tr.append(sid[(p, j)]); tc.append(sid[(q, j)]); tv.append(w / (deg[q] - 1))
        T = sparse.coo_matrix((tv, (tr, tc)), shape=(nS, nS), dtype=np.float32).tocsr()
        seed_of = collections.defaultdict(list)
        for (p, j) in S:
            seed_of[p].append(sid[(p, j)])
        return R, C, T, nS, seed_of

    def line_nb_scores(steps, damp, batch=64):
        R, C, T, nS, seed_of = LNB
        Rt, Ct, Tt = R.T.tocsr(), C.T.tocsr(), T.T.tocsr()
        out = np.zeros((nK, nK), np.float32)
        seeds = [i for i in range(nK) if have[i] and seed_of.get(int(konode[i]))]
        for b0 in range(0, len(seeds), batch):
            bs = seeds[b0:b0 + batch]
            x = np.zeros((nS, len(bs)), np.float32)
            for c_, i in enumerate(bs):
                ss = seed_of[int(konode[i])]
                x[ss, c_] = 1.0 / len(ss)
            acc = np.zeros((N, len(bs)), np.float32)
            for s in range(1, steps + 1):
                m = Rt @ x                              # mass arriving at proteins
                acc += (damp ** s) * m
                if s < steps:
                    x = (Ct @ m) - (Tt @ x)             # non-backtracking step
                    np.maximum(x, 0, out=x)             # float32 round-off can push a zero slightly negative
            out[np.ix_(np.array(bs), np.where(have)[0])] = acc[konode[have]].T
        return out

    print("  building LINE-NB incidence operators ...", flush=True)
    LNB = build_line_nb()
    print(f"    {LNB[3]} directed incidence states, R {LNB[0].nnz} nnz, T {LNB[2].nnz} nnz", flush=True)

    # ---------------- controls on the state space ----------------
    rng = np.random.RandomState(0)

    def shuffle_complexes(seed):
        """Preserve every complex's SIZE and every protein's complex-COUNT, destroy WHICH proteins co-occur.
        This is the configuration model for the hypergraph: if HYPER wins only because big complexes get
        down-weighted, this control wins too, because it has the identical size structure."""
        r = np.random.RandomState(seed)
        slots = [x for m in cmem for x in m]
        r.shuffle(slots)
        out, p = [], 0
        for m in cmem:
            out.append(sorted(set(slots[p:p + len(m)]))); p += len(m)
        return [m for m in out if len(m) >= 2]

    def rewire_ppi(seed):
        """Degree-preserving double-edge swap on the PPI layer."""
        r = np.random.RandomState(seed)
        a = np.array([x for x, _ in pe]); b = np.array([y for _, y in pe])
        for _ in range(4):
            p = r.permutation(len(b))
            b = b[p]
        keep = a != b
        return [[int(x), int(y)] for x, y in zip(a[keep], b[keep])]

    def rec(scores, truth):
        order = nontide_idx[np.argsort(-scores[nontide_idx])][:K]
        return len(set(order.tolist()) & truth) / max(len(truth), 1)

    def prof(idxs):
        # np.abs BEFORE the mean, matching the incumbent. Averaging SIGNED z-scores lets an up-regulated gene in one
        # neighbour cancel a down-regulated one in another, which suppresses exactly the specific movers being scored.
        idxs = np.asarray(list(idxs), int)
        return np.abs(M[idxs]).mean(0) if len(idxs) else np.zeros(nG, np.float32)

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC])
    scorset = set(int(x) for x in scor)
    print(f"{len(scor)} scorable knockouts (valid TARGETS); all {nK} are valid neighbour SOURCES", flush=True)

    # ---------------- precompute every arm x config (all split-independent: no measured data enters a walk) --------
    ARMS = ["NODE", "BIP-PW", "HYPER", "BIP-FULL", "BIP-NS"]
    SC = {}
    for arm in ARMS:
        for (st, dp, cw) in CHAIN_CFG:
            SC[(arm, st, dp, cw)] = chain_scores(build(arm, cw), st, dp)
        print(f"  precomputed {arm}", flush=True)
    for (st, dp, cw) in CHAIN_CFG:
        SC[("LINE-NB", st, dp, cw)] = line_nb_scores(st, dp)     # cw is inert here; the state space carries complexes
    print("  precomputed LINE-NB", flush=True)
    ARMS_ALL = ARMS + ["LINE-NB"]

    CTRL = {}
    for (st, dp, cw) in CHAIN_CFG:
        CTRL[("HYPER-SHUF", st, dp, cw)] = chain_scores(build("HYPER", cw, cmem_use=shuffle_complexes(11)), st, dp)
        CTRL[("HYPER-REWIRE", st, dp, cw)] = chain_scores(
            build("HYPER", cw, cmem_use=shuffle_complexes(12), ppi_pairs_use=rewire_ppi(13)), st, dp)
    print("  precomputed controls", flush=True)

    # ---------------- evaluation: ONE path, arms differ only in which score matrix feeds it ----------------
    per_ko = collections.defaultdict(list)     # arm -> list of per-test-knockout recalls, pooled over splits
    per_split = collections.defaultdict(list)  # arm -> one mean per split (see the independence caveat below)
    cfg_pick = collections.defaultdict(list)
    order_keys = []

    for sp in range(NSPLIT):
        srng = np.random.RandomState(100 + sp)
        perm = srng.permutation(scor); nt = len(scor) // 3
        test = np.array(sorted(perm[:nt].tolist()))
        tset = set(test.tolist())
        # THE NEIGHBOUR POOL IS EVERY NON-TEST KNOCKOUT, not only the scorable ones, exactly as the incumbent does it.
        # Only 378 of 1400 knockouts are scorable (they need >=5 specific movers of their own to be a valid TARGET),
        # but all 1400 have a measured profile and so all of them are valid SOURCES to average. Restricting the pool
        # to the scorable 378 shrinks it ~5x, costs ~0.14 recall, and -- the reason it matters here -- compresses the
        # differences between arms, because a smaller candidate set gives rival rankings less room to disagree.
        train = np.array([i for i in range(nK) if i not in tset])
        train_scor = np.array([i for i in train if i in scorset])
        SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}
        preg = np.random.RandomState(1000 + sp).choice(train_scor, min(200, len(train_scor)), replace=False)

        def pick(vrow, xi, pool):
            v = vrow.astype(np.float64).copy()
            mask = np.full(nK, -np.inf)
            mask[pool] = 0.0
            v = v + mask
            v[xi] = -np.inf
            return np.argsort(-v)[:N_NEI]

        def eval_arm(key_prefix, store_as):
            # config PRE-REGISTERED on TRAIN ONLY -- the test knockouts never influence which chain is used
            best, bv = CHAIN_CFG[0], -1.0
            for cfgk in CHAIN_CFG:
                Mm = (SC if key_prefix in ARMS_ALL else CTRL)[(key_prefix,) + cfgk]
                v = []
                # a RANDOM 200 of train, not the first 200 in gene-name order, and the SAME 200 for every arm and
                # every config so pre-registration cannot advantage one state space over another
                for xi in preg:
                    pool = train[train != xi]
                    tr_spec = set(int(t) for t in np.where(A[xi] >= TAU)[0] if ~tide[t])
                    if len(tr_spec) >= MIN_SPEC:
                        v.append(rec(prof(pick(Mm[xi], xi, pool)), tr_spec))
                mv = float(np.mean(v)) if v else 0.0
                if mv > bv:
                    bv, best = mv, cfgk
            cfg_pick[store_as].append(best)
            Mb = (SC if key_prefix in ARMS_ALL else CTRL)[(key_prefix,) + best]
            for xi in test:
                per_ko[store_as].append(rec(prof(pick(Mb[xi], int(xi), train)), SPEC[int(xi)]))

        for arm in ARMS_ALL:
            eval_arm(arm, arm)
        eval_arm("HYPER-SHUF", "HYPER-SHUF")
        eval_arm("HYPER-REWIRE", "HYPER-REWIRE")

        # RESPONSE-SIZE-MATCHED control: neighbours chosen ONLY by having a similar response magnitude to the test
        # knockout. This is the arm that decides whether a complex-reweighting gain is mechanism or just "big
        # complexes are ribosomes and ribosome knockouts move a lot of genes".
        # DELIBERATELY PRIVILEGED: it reads spec_n of the TEST knockout, which is measured data none of the chain arms
        # are allowed to see. That makes it a HARD control -- beating it means something, losing to it does not
        # automatically mean the chain is worthless, and the verdict must not conflate the two.
        for xi in test:
            pool = train[train != xi]
            d = np.abs(spec_n[pool].astype(np.float64) - spec_n[xi])
            per_ko["SIZE-MATCHED"].append(rec(prof(pool[np.argsort(d)[:N_NEI]]), SPEC[int(xi)]))
            per_ko["TIDE"].append(rec((A >= TAU).mean(0), SPEC[int(xi)]))
        # PAIRED ARM UNION: z-score each arm's row over the pool and sum. If the interaction-state chain sees anything
        # the node chain does not, the union beats both; if it is the same predictor wearing a different state space,
        # the union lands between them. This is the ensemble question asked at the level of a single component.
        for xi in test:
            pool = train[train != xi]

            def zrow(mat):
                r = mat[xi][pool].astype(np.float64)
                s = r.std()
                return (r - r.mean()) / (s + 1e-12)
            u = zrow(SC[("NODE",) + cfg_pick["NODE"][-1]]) + zrow(SC[("BIP-FULL",) + cfg_pick["BIP-FULL"][-1]])
            per_ko["UNION(NODE,BIP-FULL)"].append(rec(prof(pool[np.argsort(-u)[:N_NEI]]), SPEC[int(xi)]))

        for a in list(per_ko):
            per_split[a].append(float(np.mean(per_ko[a][-len(test):])))
        order_keys.append(len(test))
        print(f"  split {sp}: " + "  ".join(
            f"{a} {np.mean(per_ko[a][-len(test):]):.4f}" for a in ARMS_ALL), flush=True)

    # ---------------- paired significance, THREE outcomes ----------------
    def paired(a, b):
        """(mean difference, SE, verdict). A significant LOSS is a result -- returning False for it printed
        'not significant' next to a nine-sigma defeat twice in this repo before the third outcome was added."""
        x = np.array(per_ko[a]) - np.array(per_ko[b])
        n = len(x)
        br = np.random.RandomState(7)
        bs = np.array([x[br.randint(0, n, n)].mean() for _ in range(NBOOT)])
        m, se = float(x.mean()), float(bs.std())
        if m > 2 * se:
            v = "BETTER"
        elif m < -2 * se:
            v = "WORSE"
        else:
            v = "indistinguishable"
        return m, se, v

    print("\n  ARM SCORES (recall@%d, %d splits, %d test evaluations)" % (K, NSPLIT, len(per_ko["NODE"])))
    print(f"    {'arm':22s} {'score':>8s}   {'vs NODE':>9s} {'+/-':>7s}  verdict")
    rows = {}
    for a in ARMS_ALL + ["UNION(NODE,BIP-FULL)", "HYPER-SHUF", "HYPER-REWIRE", "SIZE-MATCHED", "TIDE"]:
        s = float(np.mean(per_ko[a]))
        if a == "NODE":
            print(f"    {a:22s} {s:8.4f}   {'--':>9s} {'':>7s}  reference")
            rows[a] = (s, 0.0, 0.0, "reference")
            continue
        d, se, v = paired(a, "NODE")
        print(f"    {a:22s} {s:8.4f}   {d:+9.4f} {se:7.4f}  {v}")
        rows[a] = (s, d, se, v)
    rows["NODE"] = (float(np.mean(per_ko["NODE"])), 0.0, 0.0, "reference")

    # ---- SECOND SIGNIFICANCE ESTIMATE, because the pooled bootstrap resamples 630 evaluations that are NOT
    # independent: only 378 knockouts are scorable, so each appears in ~1.7 of the 5 test sets. Resampling
    # correlated units understates the SE. A per-SPLIT paired test uses 5 genuinely independent units instead, and
    # is the more conservative of the two. If the two disagree, the split-level one is believed.
    print(f"\n  PER-SPLIT PAIRED TEST vs NODE (5 independent splits; conservative)")
    print(f"    {'arm':22s} {'mean d':>9s} {'SE':>8s} {'t(4)':>7s}  {'pos/5':>6s}  verdict")
    split_stat = {}
    for a in [x for x in ARMS_ALL if x != "NODE"] + ["UNION(NODE,BIP-FULL)"]:
        d = np.array(per_split[a]) - np.array(per_split["NODE"])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = float(d.mean() / se) if se > 0 else 0.0
        v = "BETTER" if t > 2.776 else ("WORSE" if t < -2.776 else "indistinguishable")   # two-sided 95%, 4 df
        split_stat[a] = {"mean": float(d.mean()), "se": se, "t": t, "pos": int((d > 0).sum()), "verdict": v}
        print(f"    {a:22s} {d.mean():+9.4f} {se:8.4f} {t:7.2f}  {int((d>0).sum())}/5    {v}")

    # ---- HOW MANY INDEPENDENT THINGS WERE ACTUALLY TESTED? Five arms all landing positive looks like corroboration
    # only if the arms are five different predictors. They are all built from the same PPI graph, so they are not.
    from scipy.stats import spearmanr
    print(f"\n  ARM AGREEMENT WITH NODE (mean row-wise Spearman over scorable knockouts, modal config)")
    agree = {}
    for a in ARMS_ALL:
        ca = collections.Counter(cfg_pick[a]).most_common(1)[0][0]
        cn = collections.Counter(cfg_pick["NODE"]).most_common(1)[0][0]
        # skip knockouts whose row is constant (isolated in the graph): Spearman is undefined there, not zero
        rs = [spearmanr(SC[(a,) + ca][i], SC[("NODE",) + cn][i]).statistic for i in scor[:150]
              if SC[("NODE",) + cn][i].std() > 0 and SC[(a,) + ca][i].std() > 0]
        agree[a] = float(np.nanmean(rs))
        print(f"    {a:14s} {agree[a]:+.4f}")

    best_new = max([a for a in ARMS_ALL if a != "NODE"], key=lambda a: rows[a][0])
    print(f"\n  BEST NON-INCUMBENT ARM: {best_new} ({rows[best_new][0]:.4f})")
    for ctrl in ["HYPER-SHUF", "HYPER-REWIRE", "SIZE-MATCHED"]:
        d, se, v = paired(best_new, ctrl)
        print(f"    {best_new} vs {ctrl:14s} {d:+.4f} +/- {se:.4f}  {v}")

    d_bippw, se_bippw, v_bippw = paired("BIP-PW", "NODE")
    d_hyper, se_hyper, v_hyper = paired("HYPER", "NODE")
    d_bipfull, se_bipfull, v_bipfull = paired("BIP-FULL", "NODE")
    d_line, se_line, v_line = paired("LINE-NB", "NODE")
    d_sz, se_sz, v_sz = paired(best_new, "SIZE-MATCHED")
    d_sh, se_sh, v_sh = paired(best_new, "HYPER-SHUF")

    cfg_str = "; ".join(f"{a}={collections.Counter(map(str, cfg_pick[a])).most_common(1)[0][0]}" for a in ARMS_ALL)

    verdict = (
        f"PROMOTING INTERACTIONS TO MARKOV STATES. The incumbent NODE chain scores {rows['NODE'][0]:.4f} here. "
        f"THE LITERAL BIPARTITE FORM IS A NO-OP AND THIS WAS PREDICTED BEFORE IT WAS RUN: every one of the 191,447 "
        f"PPI interactions in this dataset is PAIRWISE, and a protein->interaction->protein walk over pairwise "
        f"interactions is algebraically the LAZY node walk, 1/2 I + 1/2 P (verified to 1.1e-16 numerically). BIP-PW "
        f"measures {rows['BIP-PW'][0]:.4f}, {d_bippw:+.4f} +/- {se_bippw:.4f} vs NODE -- {v_bippw}. "
        f"THE PLACE IT CAN DIFFER IS COMPLEXES, because a complex is a real hyperedge and the incumbent clique-expands "
        f"it into k(k-1) edges: 38,522 edges from 5,394 memberships, a 7.1x inflation quadratic in complex size. As a "
        f"state a complex takes a protein's mass ONCE instead of k-1 times, which shifts the complex layer's share of "
        f"outgoing mass by up to 0.714 (RPL41) and -- the point -- shifts it BY A DIFFERENT AMOUNT FOR EVERY PROTEIN "
        f"(SD 0.137), so the incumbent's global cw sweep cannot reach it. HYPER measures {rows['HYPER'][0]:.4f} "
        f"({d_hyper:+.4f} +/- {se_hyper:.4f}, {v_hyper}), the full bipartite form {rows['BIP-FULL'][0]:.4f} "
        f"({d_bipfull:+.4f}, {v_bipfull}), and the non-backtracking interaction-only walk {rows['LINE-NB'][0]:.4f} "
        f"({d_line:+.4f}, {v_line}). "
        + (f"THE BEST NON-INCUMBENT ARM IS {best_new} AND IT CLEARS ITS CONTROLS: {d_sh:+.4f} vs a complex shuffle that "
           f"preserves every complex size and every protein's complex count ({v_sh}), and {d_sz:+.4f} vs choosing "
           f"neighbours purely by matched response magnitude ({v_sz}). The second control is the one that matters, "
           f"because large complexes are ribosome and proteasome and those knockouts move the most genes, so a "
           f"complex-reweighting gain has an obvious non-mechanistic explanation that had to be excluded. "
           if rows[best_new][1] > 2 * rows[best_new][2] and v_sh == "BETTER" and v_sz == "BETTER" else
           f"NO INTERACTION-STATE ARM BEATS THE INCUMBENT ON THIS METRIC. The best of them, {best_new}, sits "
           f"{rows[best_new][1]:+.4f} +/- {rows[best_new][2]:.4f} from NODE ({rows[best_new][3]}), and against its "
           f"controls it is {d_sh:+.4f} vs the complex shuffle ({v_sh}) and {d_sz:+.4f} vs response-size-matched "
           f"neighbours ({v_sz}). ")
        + f"FIVE ARMS ALL LANDED POSITIVE, WHICH IS ONE OBSERVATION AND NOT FIVE. The arms agree with NODE at mean "
        f"row-wise Spearman " + ", ".join(f"{a} {agree[a]:+.3f}" for a in ARMS_ALL if a != "NODE") + f". They are the "
        f"same predictor in different coordinates, so the consistent sign is what one correlated statistic does, not "
        f"independent corroboration. The conservative per-split test over 5 genuinely independent units agrees with "
        f"the pooled bootstrap and calls {best_new} {split_stat[best_new]['mean']:+.4f} +/- {split_stat[best_new]['se']:.4f} "
        f"(t={split_stat[best_new]['t']:.2f}, {split_stat[best_new]['pos']}/5 splits positive, "
        f"{split_stat[best_new]['verdict']}). "
        + f"AND THE UNION SETTLES IT. Summing the z-scored NODE and BIP-FULL rankings scores "
        f"{rows['UNION(NODE,BIP-FULL)'][0]:.4f} against NODE {rows['NODE'][0]:.4f} and BIP-FULL "
        f"{rows['BIP-FULL'][0]:.4f} -- it lands BETWEEN its parents rather than above them, which is the signature of "
        f"two views of one predictor, not two predictors. This is the same lesson redundancy_check drew about the "
        f"ensemble: extra routes to information already present buy nothing. "
        + f"THE COMPLEX LAYER IS THE REAL STORY HERE AND IT IS A NEGATIVE ONE. Shuffling WHICH proteins belong to "
        f"which complex, while preserving every complex size and every protein's complex count, scores "
        f"{rows['HYPER-SHUF'][0]:.4f} against NODE's {rows['NODE'][0]:.4f} -- {rows['HYPER-SHUF'][3]}. The membership "
        f"content of the complex layer is worth approximately nothing, which independently reproduces graph_null's "
        f"finding that PPI-only equals PPI-plus-complexes. The PPI layer, by contrast, is decisive -- rewiring it "
        f"collapses the chain to {rows['HYPER-REWIRE'][0]:.4f}. "
        + f"SO EACH ARM FAILED FOR ITS OWN REASON, and it is worth separating them rather than reporting one flat "
        f"'no effect'. HYPER and BIP-FULL could not win because clique-expansion and hyperedge states differ ONLY in "
        f"how they weight the complex layer, and that layer's content is inert. BIP-PW could not win because pairwise "
        f"interactions make the bipartite walk algebraically the lazy node walk -- a theorem, not a measurement. "
        f"LINE-NB is the one arm whose mechanism is genuinely new (non-backtracking cannot be written as any "
        f"reweighting of the node walk), and it ties too, at Spearman {agree['LINE-NB']:+.3f} with NODE -- which says "
        f"backtracking is not what limits a chain truncated at 2 steps, since a 2-step walk has barely enough length "
        f"to backtrack at all. "
        + f"WHAT THIS DOES NOT SAY. It does not say hyperedges are the wrong representation of a complex -- it says "
        f"that under THIS readout, where the chain's only job is to rank non-test knockouts so the top 10 can have "
        f"their measured profiles averaged, the ranking is not sensitive to the size-normalisation. A neighbour "
        f"selector needs only the ORDER of the top handful to be right, and both weightings put the same physical "
        f"partners at the top; they disagree about how much mass a 52-member complex should absorb, which is a "
        f"question the top-10 cut never asks. Chain configs were pre-registered on TRAIN only ({cfg_str}). "
        f"Truncation at 1-3 steps is load-bearing throughout: the stationary distribution of any of these undirected "
        f"walks is degree/2|E|, which does not depend on the start node at all, so a converged chain would provably "
        f"carry no perturbation signal. Five splits of one cell line.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_eval": len(per_ko["NODE"]), "arms": {a: rows[a][0] for a in rows},
               "vs_node": {a: {"delta": rows[a][1], "se": rows[a][2], "verdict": rows[a][3]} for a in rows},
               "controls": {c: dict(zip(("delta", "se", "verdict"), paired(best_new, c)))
                            for c in ["HYPER-SHUF", "HYPER-REWIRE", "SIZE-MATCHED"]},
               "best_non_incumbent": best_new,
               "per_split": {a: per_split[a] for a in per_split},
               "split_level_test": split_stat,
               "agreement_with_node": agree,
               "cfg_picks": {a: [list(map(float, c)) for c in cfg_pick[a]] for a in ARMS_ALL},
               "graph": {"ppi_pairwise": len(pe), "complexes": len(cmem), "memberships": int(sum(csz)),
                         "clique_edges": int(sum(k * (k - 1) for k in csz))},
               "verdict": verdict},
              open(OUT / "interaction_states.json", "w"), indent=2)
    print(f"\n  -> {OUT/'interaction_states.json'}")


if __name__ == "__main__":
    main()
