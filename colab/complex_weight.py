"""IS THE INTERACTION-STATE EFFECT JUST A MIS-TUNED SCALAR? A finer sweep of the complex weight says: yes, mostly.

WHY THIS EXISTS. interaction_states.py measured six Markov state spaces and found all of them indistinguishable from
the incumbent node walk, with every interaction-state arm landing slightly POSITIVE (+0.0030 to +0.0046). An
adversarial review of that module raised a specific, checkable objection: every one of those arms moves in the same
direction, namely REDUCING how much of a protein's outgoing mass is routed through the complex layer -- and the
incumbent already has a scalar for exactly that, the complex weight cw. The incumbent's grid samples cw at only
{0.0, 1.0, 2.0}. If the optimum sits between 0 and 1, that grid straddles it, and the whole interaction-state line
is a reparameterisation of a mis-tuned float.

That objection is worth measuring rather than arguing about, because it is the difference between "a new state space
helps a little" and "the old state space was tuned wrong". Same harness, same splits, same pre-registration
procedure, same evaluation path; ONLY the cw grid is finer.

WHAT WAS FOUND -- the objection is half right, and the half that is right is the important half.

  THE CURVE HAS A REAL INTERIOR OPTIMUM the incumbent grid straddles. Read off the test set at (steps=2, damp=0.5):
      cw=0.0  0.4736     cw=0.1  0.4822  <- peak     cw=1.0  0.4768     cw=2.0  0.4757
  cw=0.1 beats cw=2.0 by +0.0065 +/- 0.0024, which is BETTER, and it beats every interaction-state arm measured in
  interaction_states.py (best of those: BIP-FULL 0.4814). So the complex layer IS over-weighted in the incumbent.

  BUT THAT NUMBER IS READ OFF THE TEST SET AND IS NOT A GAIN ANYONE CAN COLLECT. Pre-registering the config on TRAIN,
  which is the only honest procedure and the one the incumbent uses, the fine grid scores 0.4799 against the
  incumbent grid's 0.4768: +0.0031 +/- 0.0020, INDISTINGUISHABLE. Pre-registration on 200 train knockouts cannot
  reliably locate the optimum -- it picks cw=0.1 on one split and cw=0.35 on another. The peak is real; our ability
  to find it blind is not.

THE CONCLUSION THAT MATTERS FOR THE INTERACTION-STATE RESULT. Down-weighting the complex layer with a single scalar
buys +0.0031 +/- 0.0020 pre-registered. Promoting interactions to first-class Markov states buys +0.0046 +/- 0.0023
pre-registered. These are the same size and neither is significant. So the small positive tilt shared by all six
interaction-state arms has a simpler available explanation than "hyperedges are a better representation of a
complex": every one of those state spaces incidentally reduces the complex layer's effective weight, and doing that
directly with one number gets you the same place.

This SHARPENS rather than overturns interaction_states.py, whose headline finding -- all state spaces tie -- stands.
What it removes is the temptation to read the consistent positive sign as evidence for hyperedge structure. It is
evidence that the complex layer is over-weighted, which is a statement about one scalar, not about state spaces.

It also puts a bound on the earlier claim that a global cw "cannot reach" the hyperedge reweighting because the
per-protein shift has SD 0.137. That remains true as algebra -- a scalar cannot reproduce a per-protein,
size-dependent shift exactly. What is now measured is that the part it cannot reach is worth nothing: the global
component captures the entire effect, and the per-protein residual buys no more than noise.
"""

import collections, json, pickle
import numpy as np
from scipy import sparse

SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI, NSPLIT = 1.0, 50, 5, 0.05, 10, 5

KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
nK, nG = len(kos), len(genes)
M = np.zeros((nK, nG), np.float32)
for k, v in KO.items():
    for g, z in v.items(): M[ki[k], gi[g]] = z
A = np.abs(M); tide = (A >= TAU).mean(0) >= TIDE_FRAC
nontide_idx = np.where(~tide)[0]
spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])

D = json.load(open("outputs/orphan/cell_complete.json"))
names = [g["name"] for g in D["genes"]]; nidx = {n: i for i, n in enumerate(names)}; N = len(names)
pe = set()
for e in D.get("ppi", []):
    try: a, b = int(e[0]), int(e[1])
    except Exception: continue
    if 0 <= a < N and 0 <= b < N and a != b: pe.add((min(a, b), max(a, b)))
pe = sorted(pe)
pa = np.r_[[a for a, b in pe], [b for a, b in pe]]; pb = np.r_[[b for a, b in pe], [a for a, b in pe]]
memb = collections.defaultdict(list)
for g, cs in (D.get("gene2cplx", {}) or {}).items():
    if g.isdigit() and int(g) < N:
        for c in cs: memb[c].append(int(g))
ca, cb = [], []
for c, mem in memb.items():
    m = sorted(set(mem))
    if 2 <= len(m) <= 60:
        for x in m:
            for y in m:
                if x != y: ca.append(x); cb.append(y)
ca = np.array(ca); cb = np.array(cb)
konode = np.array([nidx.get(k, -1) for k in kos]); have = konode >= 0


def build(cw):
    ra, rb = pa, pb; v = np.ones(len(pa), np.float32)
    if cw > 0:
        ra = np.r_[ra, ca]; rb = np.r_[rb, cb]; v = np.r_[v, np.full(len(ca), cw, np.float32)]
    G = sparse.coo_matrix((v, (ra, rb)), shape=(N, N)).tocsr()
    mag = np.asarray(abs(G).sum(1)).ravel()
    inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
    return (sparse.diags(inv) @ G).T.tocsr()


def chain(PT, steps, damp):
    idx = np.where(have)[0]
    L = np.zeros((N, len(idx)), np.float32); L[konode[idx], np.arange(len(idx))] = 1.0
    cur, tot = L, np.zeros_like(L)
    for s in range(1, steps + 1):
        cur = PT @ cur; tot = tot + (damp ** s) * cur
    out = np.zeros((nK, nK), np.float32); out[np.ix_(idx, idx)] = tot[konode[idx]].T
    return out


def rec(sc, truth):
    o = nontide_idx[np.argsort(-sc[nontide_idx])][:K]
    return len(set(o.tolist()) & truth) / max(len(truth), 1)


def prof(idxs):
    idxs = np.asarray(list(idxs), int)
    return np.abs(M[idxs]).mean(0) if len(idxs) else np.zeros(nG, np.float32)


CW = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0]
SD = [(1, 1.0), (2, 0.5), (3, 0.5)]
SC = {}
for cw in CW:
    PT = build(cw)
    for (st, dp) in SD:
        SC[(cw, st, dp)] = chain(PT, st, dp)
print("precomputed", len(SC), "configs", flush=True)

scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC]); scorset = set(int(x) for x in scor)
fixed_grid = [(0.0, 1, 1.0), (0.0, 2, 0.5), (1.0, 2, 0.5), (2.0, 2, 0.5), (1.0, 3, 0.5)]   # the incumbent grid
full_grid = [(cw, st, dp) for cw in CW for (st, dp) in SD]
per = collections.defaultdict(list); percw = collections.defaultdict(list); picks = collections.defaultdict(list)

for sp in range(NSPLIT):
    srng = np.random.RandomState(100 + sp)
    perm = srng.permutation(scor); nt = len(scor) // 3
    test = np.array(sorted(perm[:nt].tolist())); tset = set(test.tolist())
    train = np.array([i for i in range(nK) if i not in tset])
    train_scor = np.array([i for i in train if i in scorset])
    preg = np.random.RandomState(1000 + sp).choice(train_scor, min(200, len(train_scor)), replace=False)
    SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}

    def pick(vrow, xi, pool):
        v = vrow.astype(np.float64).copy(); m = np.full(nK, -np.inf); m[pool] = 0.0
        v = v + m; v[xi] = -np.inf
        return np.argsort(-v)[:N_NEI]

    def preg_score(cfg):
        v = []
        for xi in preg:
            tr = set(int(t) for t in np.where(A[xi] >= TAU)[0] if ~tide[t])
            if len(tr) >= MIN_SPEC:
                v.append(rec(prof(pick(SC[cfg][xi], xi, train[train != xi])), tr))
        return float(np.mean(v)) if v else 0.0

    for label, grid in [("INCUMBENT-GRID {0,1,2}", fixed_grid), ("FINE-GRID", full_grid)]:
        best = max(grid, key=preg_score); picks[label].append(best)
        for xi in test:
            per[label].append(rec(prof(pick(SC[best][xi], int(xi), train)), SPEC[int(xi)]))
    # and the full cw curve at the incumbent's own (2,0.5), no pre-registration -- just the shape
    for cw in CW:
        for xi in test:
            percw[cw].append(rec(prof(pick(SC[(cw, 2, 0.5)][xi], int(xi), train)), SPEC[int(xi)]))
    print(f"  split {sp} done", flush=True)


def paired(a, b):
    x = np.array(per[a] if a in per else percw[a]) - np.array(per[b] if b in per else percw[b])
    br = np.random.RandomState(7); n = len(x)
    bs = np.array([x[br.randint(0, n, n)].mean() for _ in range(2000)])
    m, se = float(x.mean()), float(bs.std())
    return m, se, ("BETTER" if m > 2 * se else "WORSE" if m < -2 * se else "indistinguishable")


print("\nTHE cw CURVE at (steps=2, damp=0.5), no pre-registration -- just the shape of the dependence")
for cw in CW:
    m, se, v = paired(cw, 2.0)
    print(f"   cw={cw:<5} {np.mean(percw[cw]):.4f}   vs cw=2.0: {m:+.4f} +/- {se:.4f}  {v}")
print("\nPRE-REGISTERED, the only fair comparison")
for lab in ["INCUMBENT-GRID {0,1,2}", "FINE-GRID"]:
    print(f"   {lab:24s} {np.mean(per[lab]):.4f}   picks={collections.Counter(map(str,picks[lab])).most_common(2)}")
m, se, v = paired("FINE-GRID", "INCUMBENT-GRID {0,1,2}")
print(f"   FINE-GRID - INCUMBENT-GRID = {m:+.4f} +/- {se:.4f}  {v}")
print("\n   (interaction-state arms measured earlier, same harness: NODE 0.4768, best arm BIP-FULL 0.4814)")
