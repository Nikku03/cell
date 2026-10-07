"""HP-lattice protein folding -- EXACT ground states by branch and bound.

THE GOVERNING LAW, cost = d ** treewidth, is what tells us NOT to eliminate here.
Write the HP model as a factor graph over residue positions: a variable per residue with
d = number of lattice sites in the reachable box, and a self-avoidance constraint on
EVERY pair of residues. That constraint graph is the complete graph K_n, whose treewidth
is n - 1, so cost = d ** (n-1) = ((2n+1)**dim) ** (n-1). For the 20-mer on the square
lattice that is 41**2 raised to the 19th power, about 1e61. Bucket elimination is not the
tool. structure_info() prints those numbers for any instance so the choice is auditable
rather than asserted; the measured node counts of the search are reported next to them.

MODEL. A sequence over {H, P} is embedded as a self-avoiding walk on the square (2D) or
cubic (3D) lattice: consecutive residues sit at unit distance, no two residues share a
site. The energy is

    E = -(number of pairs i, j with |i - j| >= 2, seq[i] = seq[j] = H, |x_i - x_j| = 1)

i.e. -1 per non-bonded H-H contact at unit distance. Both lattices are bipartite, so a
contact forces i + j to be odd; that parity fact is the backbone of the bound below.

THE BOUND (admissible, and audited numerically -- see verify_bound_admissible).
State: residues 0..k-1 placed, c contacts already realised. Classify every contact that
is still to be made by which endpoints are already placed. Because every contact joins an
even chain index to an odd chain index:

    a  = (placed even H, future odd H)      b = (placed odd H, future even H)
    c' = (future even H, future odd H)

Let capE = sum over PLACED even-index H residues of the number of EMPTY lattice
neighbours of that residue's site (capO likewise for odd), with one slot removed from
residue k-1 because its next free neighbour is consumed by the chain bond to residue k.
Let De = sum over FUTURE even-index H residues j of deg_j, and Do likewise for odd, where
deg_j = z - 1 for a chain terminus and z - 2 otherwise (z = 2*dim): a residue's chain
neighbours occupy 1 or 2 of its z lattice neighbours, so deg_j caps its non-bonded
contacts. Then

    a <= capE,  b <= capO,  a + c' <= Do,  b + c' <= De,  c' <= min(De, Do),
    and every remaining contact spends at least one endpoint of a future H,

so the number of remaining contacts is at most

    B = min( De + Do,  capO + Do,  capE + De,  capE + capO + min(De, Do) ).

Prune the node when c + B <= best_so_far. At the root this collapses to the textbook
bound min(De_all, Do_all) (2 * min(#even H, #odd H) up to the two chain-end corrections).

TIER 2 -- REACHABILITY. Only R = n - k residues are left, so every future residue lies
within L1 distance R of the head. A placed H at L1 distance d from the head can therefore
only be touched by a future residue of index >= k + d - 2, and only by one of the opposite
chain parity. Dropping the placed H residues that fail either test shrinks capE and capO
and never grows them, so tier 2 <= tier 1 and it is admissible whenever tier 1 is. Tier 1
is incremental and nearly free, so it is tried first and tier 2 only runs on survivors.
Measured effect on the 20-mer benchmark: 23,664,764 nodes / 50.2 s with tier 1 alone,
183,215 nodes / 0.5 s with both tiers and the De + Do term.

SYMMETRY. Translation is fixed by placing residue 0 at the origin; the point group is
fixed by forcing step 0 to be +x, the first step off the x axis to be +y, and (3D) the
first step with a z component to be +z. That is exactly one representative per orbit of
the 8-element (2D) / 48-element (3D) point group, so no conformation class is lost.

VERIFICATION. verify() checks the branch-and-bound optimum against enumerate_bruteforce,
which is a naive DFS over every self-avoiding walk scored by hp_energy, an O(n^2)
all-pairs distance scan. It shares no bound, no incremental counter and no symmetry
reduction with the search; the only thing in common is the definition of the energy.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

LATTICE_DIM = {"square": 2, "cubic": 3}


# --------------------------------------------------------------------------- benchmarks
# The classic 2D HP benchmark set of Unger & Moult (1993), reused by essentially every
# later HP paper (Yue-Dill CHCC 1995, Beutler-Dill 1996, Toma-Toma, Lesh et al. 2003,
# Cebrian et al. 2008, ...). Published optimal energies for the square lattice.
HP_BENCHMARKS_2D: List[Tuple[str, str, int]] = [
    # (name, sequence, published optimal energy)
    ("20-mer  (HP)2PH(HP)2(PH)2HP(PH)2",
     "HPHPPHHPHPPHPHHPPHPH", -9),
    ("24-mer  H2P2(HP2)6H2",
     "HHPPHPPHPPHPPHPPHPPHPPHH", -9),
    ("25-mer  P2HP2(H2P4)3H2",
     "PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
    ("36-mer  P3H2P2H2P5H7P2H2P4H2P2HP2",
     "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
]


# ------------------------------------------------------------------------------- basics
def _check_lattice(lattice: str) -> int:
    if lattice not in LATTICE_DIM:
        raise ValueError(f"lattice must be one of {sorted(LATTICE_DIM)}, got {lattice!r}")
    return LATTICE_DIM[lattice]


def _clean_sequence(sequence) -> str:
    s = "".join(str(c) for c in sequence).upper().strip()
    if not s:
        raise ValueError("empty sequence")
    bad = sorted(set(s) - {"H", "P"})
    if bad:
        raise ValueError(f"sequence must be over H/P only; saw {bad}")
    return s


def coord_deltas(dim: int) -> List[Tuple[int, ...]]:
    """Unit lattice steps, ordered (+x, -x, +y, -y, +z, -z) to match the flat site
    strides used by the search."""
    out = []
    for axis in range(dim):
        for sgn in (1, -1):
            v = [0] * dim
            v[axis] = sgn
            out.append(tuple(v))
    return out


def hp_energy(sequence, conformation) -> int:
    """Energy of a conformation, from scratch: -1 per non-bonded H-H pair at squared
    distance 1. Naive O(n^2) all-pairs scan -- this is the DEFINITION, used by the brute
    force and by conformation validation, and never by the search's incremental counter."""
    seq = _clean_sequence(sequence)
    pts = [tuple(int(v) for v in p) for p in conformation]
    if len(pts) != len(seq):
        raise ValueError(f"conformation has {len(pts)} sites, sequence has {len(seq)}")
    e = 0
    n = len(seq)
    for i in range(n):
        if seq[i] != "H":
            continue
        pi = pts[i]
        for j in range(i + 2, n):
            if seq[j] != "H":
                continue
            pj = pts[j]
            d2 = 0
            for a, b in zip(pi, pj):
                d2 += (a - b) * (a - b)
            if d2 == 1:
                e -= 1
    return e


def validate_conformation(sequence, conformation, lattice: str = "square",
                          energy: Optional[int] = None) -> Dict:
    """Structural audit of a returned fold. A correct energy on an invalid walk is a bug,
    so every reported conformation goes through this."""
    dim = _check_lattice(lattice)
    seq = _clean_sequence(sequence)
    pts = [tuple(int(v) for v in p) for p in conformation]
    problems: List[str] = []
    if len(pts) != len(seq):
        problems.append(f"length {len(pts)} != sequence length {len(seq)}")
    if any(len(p) != dim for p in pts):
        problems.append(f"not all sites are {dim}-dimensional")
    self_avoiding = len(set(pts)) == len(pts)
    if not self_avoiding:
        problems.append("walk is not self-avoiding")
    unit_steps = True
    for i in range(len(pts) - 1):
        d2 = sum((a - b) * (a - b) for a, b in zip(pts[i], pts[i + 1]))
        if d2 != 1:
            unit_steps = False
            problems.append(f"step {i}->{i+1} has squared length {d2}, not 1")
            break
    e = hp_energy(seq, pts) if not problems else None
    matches = None
    if energy is not None:
        matches = (e == energy)
        if not matches:
            problems.append(f"recomputed energy {e} != reported {energy}")
    return {"ok": not problems, "self_avoiding": self_avoiding, "unit_steps": unit_steps,
            "energy": e, "energy_matches": matches, "problems": problems}


def structure_info(sequence, lattice: str = "square") -> Dict:
    """Why this module is a search and not an elimination: cost = d ** treewidth for the
    factor-graph formulation of the same problem."""
    import math
    seq = _clean_sequence(sequence)
    dim = _check_lattice(lattice)
    n = len(seq)
    d = (2 * n + 1) ** dim
    tw = n - 1                       # self-avoidance couples every pair -> K_n
    return {"n": n, "dim": dim, "lattice": lattice, "d_states_per_variable": d,
            "treewidth": tw, "cost_exponent": tw,
            "log10_elimination_cost": tw * math.log10(d) if d > 0 else 0.0,
            "n_H": seq.count("H")}


def max_contacts_upper_bound(sequence, lattice: str = "square") -> int:
    """The bound above evaluated at the root: min(De_all, Do_all). An upper bound on the
    number of H-H contacts of ANY conformation, hence -bound is a lower bound on E."""
    seq = _clean_sequence(sequence)
    dim = _check_lattice(lattice)
    z = 2 * dim
    n = len(seq)
    de = do = 0
    for j, c in enumerate(seq):
        if c != "H":
            continue
        deg = (z - 1) if (j == 0 or j == n - 1) else (z - 2)
        if j & 1:
            do += deg
        else:
            de += deg
    return min(de, do)


# ------------------------------------------------------------- brute force (independent)
def _walks(n: int, dim: int, fix_first_step: bool = False):
    """Every self-avoiding walk on n sites starting at the origin, by naive DFS.
    No bound, no scoring, no incremental bookkeeping. `fix_first_step` uses only the
    lattice symmetry that energy is invariant under a global rotation."""
    if n <= 0:
        return
    deltas = coord_deltas(dim)
    start = tuple([0] * dim)
    if n == 1:
        yield [start]
        return
    path = [start]
    seen = {start}

    def rec(k):
        if k == n:
            yield list(path)
            return
        p = path[-1]
        ds = deltas[:1] if (k == 1 and fix_first_step) else deltas
        for d in ds:
            q = tuple(p[a] + d[a] for a in range(dim))
            if q in seen:
                continue
            seen.add(q)
            path.append(q)
            yield from rec(k + 1)
            path.pop()
            seen.remove(q)

    yield from rec(1)


def enumerate_bruteforce_full(sequence, lattice: str = "square",
                              fix_first_step: bool = False) -> Dict:
    seq = _clean_sequence(sequence)
    dim = _check_lattice(lattice)
    t0 = time.perf_counter()
    best = None
    best_walk = None
    count = 0
    for w in _walks(len(seq), dim, fix_first_step):
        count += 1
        e = hp_energy(seq, w)
        if best is None or e < best:
            best = e
            best_walk = w
    return {"energy": best, "conformation": best_walk, "n_walks": count,
            "seconds": time.perf_counter() - t0, "sequence": seq, "lattice": lattice}


def enumerate_bruteforce(sequence, lattice: str = "square", fix_first_step: bool = False
                         ) -> Tuple[int, List[Tuple[int, ...]]]:
    """Exact optimum by full enumeration of self-avoiding walks. Reference implementation."""
    r = enumerate_bruteforce_full(sequence, lattice, fix_first_step)
    return r["energy"], r["conformation"]


# --------------------------------------------------------------------- branch and bound
class _Budget(Exception):
    """Raised to unwind the DFS when a node or time budget is exhausted."""


def _site_to_coord(s: int, L: int, dim: int) -> Tuple[int, ...]:
    out = []
    for _ in range(dim):
        out.append(s % L)
        s //= L
    return tuple(out)


def _search(seq: str, dim: int, best_contacts: int, best_sites: Optional[List[int]],
            node_limit: Optional[int], deadline: Optional[float],
            audit: bool = False, probe=None) -> Dict:
    """One depth-first branch-and-bound pass. Works in CONTACT COUNTS (energy = -contacts)
    and in flat site indices. `best_contacts` seeds the incumbent."""
    n = len(seq)
    z = 2 * dim
    isH = [c == "H" for c in seq]
    L = 2 * n + 1
    strides = [L ** a for a in range(dim)]
    deltas: List[int] = []
    for a in range(dim):
        deltas.append(strides[a])
        deltas.append(-strides[a])
    n_dirs = len(deltas)
    all_dirs = tuple(range(n_dirs))
    occ = [-1] * (L ** dim)
    center = sum(n * s for s in strides)

    # suffix sums of the per-residue non-bonded degree cap, split by chain parity
    degE = [0] * (n + 1)
    degO = [0] * (n + 1)
    for j in range(n - 1, -1, -1):
        deg = (z - 1) if (j == 0 or j == n - 1) else (z - 2)
        add = deg if isH[j] else 0
        degE[j] = degE[j + 1] + (add if (j & 1) == 0 else 0)
        degO[j] = degO[j + 1] + (add if (j & 1) == 1 else 0)
    lastH = max([j for j in range(n) if isH[j]], default=-1)

    # H indices already placed, per depth; and the next H of each chain parity at or
    # after an index -- both feed the reachability bound (tier 2).
    Hpre: List[Tuple[int, ...]] = []
    acc: List[int] = []
    for k in range(n + 1):
        Hpre.append(tuple(acc))
        if k < n and isH[k]:
            acc.append(k)
    nextH = [[n] * (n + 2) for _ in range(2)]
    for par in (0, 1):
        nxt = n
        for j in range(n - 1, -1, -1):
            if isH[j] and (j & 1) == par:
                nxt = j
            nextH[par][j] = nxt

    axis_of = [di >> 1 for di in range(n_dirs)]
    sign_of = [1 if (di & 1) == 0 else -1 for di in range(n_dirs)]
    cco = [[0] * n for _ in range(dim)]

    pos = [0] * n
    pos[0] = center
    occ[center] = 0
    cap = [0, 0]
    if isH[0]:
        cap[0] = z

    best = best_contacts
    bestc = list(best_sites) if best_sites is not None else None
    contacts = 0
    nodes = 0
    complete = True
    reason = ""

    def _cap_scratch(k):
        e = o = 0
        for i in range(k):
            if not isH[i]:
                continue
            cnt = 0
            for d in deltas:
                if occ[pos[i] + d] < 0:
                    cnt += 1
            if i & 1:
                o += cnt
            else:
                e += cnt
        return e, o

    def _complete(k):
        """Any valid completion of the walk from residue k on. Used only once the whole
        remaining suffix is P, where no further contact is possible."""
        if k == n:
            return pos[:]
        head = pos[k - 1]
        for d in deltas:
            s = head + d
            if occ[s] < 0:
                occ[s] = k
                pos[k] = s
                r = _complete(k + 1)
                occ[s] = -1
                if r is not None:
                    return r
        return None

    def rec(k, seen_perp, seen_z):
        nonlocal nodes, contacts, best, bestc
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _Budget("nodes")
        if deadline is not None and (nodes & 2047) == 0 and time.perf_counter() > deadline:
            raise _Budget("time")

        head = pos[k - 1]
        if k == 1:
            dirs = (0,)
        elif not seen_perp:
            dirs = (0, 1, 2)
        elif dim == 3 and not seen_z:
            dirs = (0, 1, 2, 3, 4)
        else:
            dirs = all_dirs
        succ = []
        for di in dirs:
            s = head + deltas[di]
            if occ[s] < 0:
                succ.append((di, s))
        if not succ:
            return

        if k > lastH:
            # every remaining residue is P: no further contact is possible, bound is 0
            if contacts > best:
                comp = _complete(k)
                if comp is not None:
                    best = contacts
                    bestc = comp
            return

        # ---- tier 1: the incremental bound, nearly free
        cE = cap[0]
        cO = cap[1]
        if audit:
            aE, aO = _cap_scratch(k)
            if (aE, aO) != (cE, cO):
                raise AssertionError(f"incremental cap {(cE, cO)} != scratch {(aE, aO)} "
                                     f"at k={k}")
        if isH[k - 1]:                       # one free neighbour goes to the chain bond
            if (k - 1) & 1:
                cO -= 1
            else:
                cE -= 1
        De = degE[k]
        Do = degO[k]
        m = De if De < Do else Do
        # every remaining contact spends at least one contact-endpoint of a FUTURE H,
        # and future H j has at most deg_j of them: remaining <= De + Do.
        b = De + Do
        t = cO + Do
        if t < b:
            b = t
        t = cE + De
        if t < b:
            b = t
        t = cE + cO + m
        if t < b:
            b = t
        if probe is not None:
            probe(k, contacts, b, [pos[i] for i in range(k)], 1)
        if contacts + b <= best:
            return

        # ---- tier 2: the same bound with a REACHABILITY filter on the placed H's.
        # Only R = n - k residues are left, so a future residue sits within L1 distance
        # R of the head; a placed H at L1 distance d from the head can only be touched
        # by a future residue of index >= k + d - 2, and only by one of the opposite
        # chain parity. Both filters only ever REMOVE capacity, so tier 2 <= tier 1.
        R = n - k
        cE2 = 0
        cO2 = 0
        hco = [cco[a][k - 1] for a in range(dim)]
        for i in Hpre[k]:
            d = 0
            for a in range(dim):
                v = cco[a][i] - hco[a]
                d += v if v >= 0 else -v
            if d - 1 > R:
                continue
            lo = k + d - 2
            if lo < k:
                lo = k
            if lo >= n or nextH[1 - (i & 1)][lo] >= n:
                continue
            si = pos[i]
            c = 0
            for d2 in deltas:
                if occ[si + d2] < 0:
                    c += 1
            if i == k - 1:
                c -= 1
                if c < 0:
                    c = 0
            if i & 1:
                cO2 += c
            else:
                cE2 += c
        b = De + Do
        t = cO2 + Do
        if t < b:
            b = t
        t = cE2 + De
        if t < b:
            b = t
        t = cE2 + cO2 + m
        if t < b:
            b = t
        if probe is not None:
            probe(k, contacts, b, [pos[i] for i in range(k)], 2)
        if contacts + b <= best:
            return

        hk = isH[k]
        if hk:
            # new contacts made by residue k at each candidate site; the search then
            # dives at the most contact-forming neighbour first (move ordering).
            scored = []
            for di, s in succ:
                c = 0
                for d2 in deltas:
                    i = occ[s + d2]
                    if i >= 0 and i != k - 1 and isH[i]:
                        c += 1
                scored.append((-c, di, s))
            if len(scored) > 1:
                scored.sort()
            order = [(di, s, -mc) for mc, di, s in scored]
        else:
            order = [(di, s, 0) for di, s in succ]     # a P residue makes no contact

        klast = (k + 1 == n)
        for di, s, cnew in order:
            occ[s] = k
            pos[k] = s
            for a in range(dim):
                cco[a][k] = cco[a][k - 1]
            cco[axis_of[di]][k] += sign_of[di]
            dE = 0
            dO = 0
            empt = 0
            for d2 in deltas:
                i = occ[s + d2]
                if i < 0:
                    empt += 1
                elif isH[i]:
                    if i & 1:
                        dO -= 1
                    else:
                        dE -= 1
            if hk:
                if k & 1:
                    dO += empt
                else:
                    dE += empt
            cap[0] += dE
            cap[1] += dO
            contacts += cnew
            if klast:
                if contacts > best:
                    best = contacts
                    bestc = pos[:]
            else:
                rec(k + 1, seen_perp or di >= 2, seen_z or di >= 4)
            contacts -= cnew
            cap[0] -= dE
            cap[1] -= dO
            occ[s] = -1

    t0 = time.perf_counter()
    if n == 1:
        bestc = [center]
        best = max(best, 0)
    else:
        try:
            rec(1, False, False)
        except _Budget as exc:
            complete = False
            reason = str(exc)
    return {"best": best, "sites": bestc, "nodes": nodes, "complete": complete,
            "reason": reason, "seconds": time.perf_counter() - t0, "L": L}


def fold_hp_full(sequence, lattice: str = "square", node_limit: Optional[int] = None,
                 time_limit: Optional[float] = None, warmup_nodes: int = 200_000) -> Dict:
    """Branch and bound with a two-phase incumbent: a short depth-first warm-up (the move
    ordering dives at the most contact-forming neighbour first) supplies a good incumbent,
    then the full proof pass runs with that incumbent so the bound prunes from the start.
    Phase 2 only ever accepts a STRICTLY better conformation, so phase 1's fold is kept."""
    seq = _clean_sequence(sequence)
    dim = _check_lattice(lattice)
    n = len(seq)
    t0 = time.perf_counter()
    deadline = (t0 + time_limit) if time_limit is not None else None

    total_nodes = 0
    best = -1
    sites = None
    phases = []

    if warmup_nodes and n > 2:
        wl = warmup_nodes if node_limit is None else min(warmup_nodes, node_limit)
        r = _search(seq, dim, -1, None, wl, deadline)
        total_nodes += r["nodes"]
        best, sites = r["best"], r["sites"]
        phases.append(("warmup", r["nodes"], r["complete"]))
        if r["complete"]:
            return _package(seq, lattice, dim, best, sites, r["L"], total_nodes, t0,
                            True, phases)
    nl = None if node_limit is None else max(0, node_limit - total_nodes)
    r2 = _search(seq, dim, best, sites, nl, deadline)
    total_nodes += r2["nodes"]
    if r2["best"] > best:
        best, sites = r2["best"], r2["sites"]
    phases.append(("proof", r2["nodes"], r2["complete"]))
    return _package(seq, lattice, dim, best, sites, r2["L"], total_nodes, t0,
                    r2["complete"], phases)


def _package(seq, lattice, dim, best, sites, L, nodes, t0, proved, phases) -> Dict:
    if sites is None:
        conf = None
    else:
        raw = [_site_to_coord(s, L, dim) for s in sites]
        o = raw[0]
        conf = [tuple(c - b for c, b in zip(p, o)) for p in raw]
        # guard: the incremental contact counter must agree with the O(n^2) definition
        chk = validate_conformation(seq, conf, lattice, -best)
        if not chk["ok"]:
            raise RuntimeError("branch-and-bound returned an inconsistent fold: "
                               + "; ".join(chk["problems"]))
    return {"energy": -best if best >= 0 else None, "conformation": conf,
            "nodes": nodes, "seconds": time.perf_counter() - t0,
            "proved_optimal": proved, "phases": phases,
            "root_bound_energy": -max_contacts_upper_bound(seq, lattice),
            "sequence": seq, "lattice": lattice,
            "structure": structure_info(seq, lattice)}


def fold_hp(sequence, lattice: str = "square", **kwargs
            ) -> Tuple[int, List[Tuple[int, ...]]]:
    """Exact HP ground state. Returns (energy, conformation).

    With no node_limit/time_limit the search always runs to completion and the energy is
    the proven optimum. If a budget is supplied and exhausted, a UserWarning is raised and
    the returned energy is only an upper bound (the best conformation actually found)."""
    r = fold_hp_full(sequence, lattice, **kwargs)
    if not r["proved_optimal"]:
        import warnings
        warnings.warn(f"budget exhausted after {r['nodes']} nodes / {r['seconds']:.1f}s; "
                      f"energy {r['energy']} is NOT proven optimal "
                      f"(lower bound on E is {r['root_bound_energy']})", stacklevel=2)
    return r["energy"], r["conformation"]


# ------------------------------------------------------------------ bound admissibility
def _bound_scratch(seq: str, dim: int, prefix: Sequence[Tuple[int, ...]]) -> Tuple[int, int]:
    """Independent re-implementation of the bound stated in this module's docstring,
    written from the formula and working on coordinate tuples and a set. It shares no
    code with the search's incremental counters. Returns (tier1, tier2)."""
    n = len(seq)
    z = 2 * dim
    k = len(prefix)
    R = n - k
    deltas = coord_deltas(dim)
    occupied = set(prefix)
    head = prefix[-1]
    cE1 = cO1 = cE2 = cO2 = 0
    for i in range(k):
        if seq[i] != "H":
            continue
        p = prefix[i]
        free = 0
        for d in deltas:
            t = tuple(p[a] + d[a] for a in range(dim))
            if t not in occupied:
                free += 1
        if i == k - 1:
            free = max(0, free - 1)          # one free neighbour goes to the chain bond
        dd = sum(abs(p[a] - head[a]) for a in range(dim))
        lo = max(k, k + dd - 2)
        reach = (dd - 1 <= R) and any(seq[j] == "H" and (j & 1) != (i & 1)
                                      for j in range(lo, n))
        if i & 1:
            cO1 += free
            cO2 += free if reach else 0
        else:
            cE1 += free
            cE2 += free if reach else 0
    De = Do = 0
    for j in range(k, n):
        if seq[j] != "H":
            continue
        deg = (z - 1) if (j == 0 or j == n - 1) else (z - 2)
        if j & 1:
            Do += deg
        else:
            De += deg
    m = min(De, Do)
    b1 = min(De + Do, cO1 + Do, cE1 + De, cE1 + cO1 + m)
    b2 = min(De + Do, cO2 + Do, cE2 + De, cE2 + cO2 + m)
    return b1, b2


def verify_bound_admissible(sequences: Optional[Sequence[str]] = None,
                            lattice: str = "square", verbose: bool = True) -> Dict:
    """EXHAUSTIVE admissibility audit. For EVERY self-avoiding prefix of every listed
    sequence the true best completion is computed by enumerating all completions, and the
    bound is required to be at least the number of contacts that completion still gains.
    A single positive deficit would mean the search can prune away the optimum."""
    dim = _check_lattice(lattice)
    if sequences is None:
        sequences = ["HHHHHHHH", "HPHPHPHP", "HHPPHHPP", "PHHPPHHP", "HPPHPHHP",
                     "HHHPPPHH", "PPHHPPHH"]
    deltas = coord_deltas(dim)
    worst1 = worst2 = -10 ** 9
    slack_sum = 0
    checked = 0
    tier_order_ok = True
    for seq in sequences:
        seq = _clean_sequence(seq)
        n = len(seq)
        lastH = max([j for j in range(n) if seq[j] == "H"], default=-1)
        origin = tuple([0] * dim)
        prefix = [origin]
        occupied = {origin}

        def rec():
            nonlocal worst1, worst2, slack_sum, checked, tier_order_ok
            k = len(prefix)
            if k == n:
                return -hp_energy(seq, prefix)
            best_total = -1
            p = prefix[-1]
            for d in deltas:
                t = tuple(p[a] + d[a] for a in range(dim))
                if t in occupied:
                    continue
                occupied.add(t)
                prefix.append(t)
                r = rec()
                prefix.pop()
                occupied.remove(t)
                if r > best_total:
                    best_total = r
            if best_total < 0:
                return -1                    # dead end: no completion exists
            cp = -hp_energy(seq[:k], prefix)
            b1, b2 = _bound_scratch(seq, dim, prefix)
            if k > lastH:
                b1 = b2 = 0                  # the shortcut the search takes
            if b2 > b1:
                tier_order_ok = False
            need = best_total - cp
            worst1 = max(worst1, need - b1)
            worst2 = max(worst2, need - b2)
            slack_sum += b2 - need
            checked += 1
            return best_total

        rec()
    res = {"prefixes_checked": checked, "max_deficit_tier1": worst1,
           "max_deficit_tier2": worst2, "mean_slack_tier2": slack_sum / max(checked, 1),
           "tier2_never_looser_than_tier1": tier_order_ok,
           "admissible": worst1 <= 0 and worst2 <= 0}
    if verbose:
        print(f"    bound admissibility, {len(sequences)} sequences, "
              f"{checked:,} self-avoiding prefixes, every completion enumerated")
        print(f"        max (true remaining contacts - bound):  tier1 {worst1:+d}   "
              f"tier2 {worst2:+d}    (must be <= 0)")
        print(f"        mean slack of the bound actually used:  "
              f"{res['mean_slack_tier2']:.2f} contacts;  "
              f"tier2 <= tier1 everywhere: {tier_order_ok}")
    return res


def verify_bound_matches_search(sequences: Optional[Sequence[str]] = None,
                                lattice: str = "square", verbose: bool = True) -> Dict:
    """The audited formula above and the bound the SEARCH actually evaluates must agree
    at every node, and the search's incrementally maintained capacities must equal the
    capacities recomputed from scratch."""
    dim = _check_lattice(lattice)
    if sequences is None:
        sequences = ["HPHPPHHPHP", "HHPPHPPHHH", "PHHPPHHPPH"]
    evals = 0
    mismatch = 0
    for seq in sequences:
        seq = _clean_sequence(seq)
        n = len(seq)
        L = 2 * n + 1
        state = {"n": 0, "bad": 0}

        def probe(k, contacts, b, sites, tier, seq=seq, L=L):
            pref = [_site_to_coord(x, L, dim) for x in sites]
            b1, b2 = _bound_scratch(seq, dim, pref)
            state["n"] += 1
            if (b1 if tier == 1 else b2) != b:
                state["bad"] += 1

        _search(seq, dim, -1, None, None, None, audit=True, probe=probe)
        evals += state["n"]
        mismatch += state["bad"]
    res = {"bound_evaluations": evals, "mismatches": mismatch}
    if verbose:
        print(f"    search-vs-formula: {evals:,} bound evaluations audited "
              f"(incremental capacities also recomputed from scratch at every node), "
              f"{mismatch} mismatches")
    return res


# ------------------------------------------------------------------------- benchmarking
def benchmark_2d(entries=None, time_limit: Optional[float] = None,
                 verbose: bool = True) -> List[Dict]:
    """The published 2D HP benchmarks: branch-and-bound energy against the literature."""
    if entries is None:
        entries = HP_BENCHMARKS_2D[:3]
    rows = []
    for name, seq, published in entries:
        r = fold_hp_full(seq, "square", time_limit=time_limit)
        chk = validate_conformation(seq, r["conformation"], "square", r["energy"])
        rows.append({"name": name, "sequence": seq, "n": len(seq),
                     "published": published, "energy": r["energy"],
                     "gap": r["energy"] - published, "nodes": r["nodes"],
                     "seconds": r["seconds"], "proved_optimal": r["proved_optimal"],
                     "root_bound_energy": r["root_bound_energy"],
                     "conformation_valid": chk["ok"], "problems": chk["problems"],
                     "conformation": r["conformation"]})
    if verbose:
        _print_benchmark_rows(rows)
    return rows


def _print_benchmark_rows(rows) -> None:
    print(f"    {'benchmark':<40} {'n':>3} {'ours':>5} {'published':>10} {'gap':>4} "
          f"{'proved':>7} {'nodes':>12} {'sec':>7}  conf")
    for r in rows:
        print(f"    {r['name']:<40} {r['n']:>3} {r['energy']:>5} {r['published']:>10} "
              f"{r['gap']:>+4d} {str(r['proved_optimal']):>7} {r['nodes']:>12,} "
              f"{r['seconds']:>7.2f}  "
              f"{'valid' if r['conformation_valid'] else 'INVALID ' + str(r['problems'])}")
        if r["gap"] != 0:
            if r["proved_optimal"]:
                print("        !! PROVEN OPTIMAL but disagrees with the published value "
                      "-- one of the two is wrong")
            else:
                print(f"        !! NOT proven optimal: the budget ran out after "
                      f"{r['seconds']:.1f}s and {r['nodes']:,} nodes; the reported energy "
                      f"is only an upper bound, gap to published = {r['gap']:+d}")


# -------------------------------------------------------------------------------- verify
def verify(verbose: bool = True, seed: int = 20250829, n_random: int = 16,
           benchmark_time_limit: Optional[float] = None) -> Dict:
    """(a) branch and bound == full enumeration, exact integer equality;
    (b) the published 2D HP benchmarks;
    (c) every conformation returned is self-avoiding, unit-step and re-scores to the
        energy that was reported."""
    import random
    rng = random.Random(seed)

    cases: List[Tuple[str, str]] = []
    for _ in range(n_random):
        n = rng.randint(5, 11)
        cases.append(("square", "".join(rng.choice("HP") for _ in range(n))))
    cases += [("square", "HHHHHHHHHH"), ("square", "PPPPPPP"),
              ("square", "HPHPHPHPHPH"), ("square", "HHPPHHPPHHP"),
              ("square", "PPPHHHPPPHHH")]
    for _ in range(6):
        n = rng.randint(4, 8)
        cases.append(("cubic", "".join(rng.choice("HP") for _ in range(n))))
    cases += [("cubic", "HHHHHHH"), ("cubic", "HPHPHPHP")]

    t0 = time.perf_counter()
    max_err = 0
    n_walks_total = 0
    bad_conf: List[str] = []
    per_lattice: Dict[str, Dict] = {"square": {"n": 0, "max_err": 0, "max_len": 0},
                                    "cubic": {"n": 0, "max_err": 0, "max_len": 0}}
    for lat, seq in cases:
        e_bb, conf_bb = fold_hp(seq, lat)
        full = enumerate_bruteforce_full(seq, lat)
        e_bf, conf_bf = full["energy"], full["conformation"]
        n_walks_total += full["n_walks"]
        err = abs(e_bb - e_bf)
        max_err = max(max_err, err)
        d = per_lattice[lat]
        d["n"] += 1
        d["max_err"] = max(d["max_err"], err)
        d["max_len"] = max(d["max_len"], len(seq))
        for tag, c, e in (("bb", conf_bb, e_bb), ("enum", conf_bf, e_bf)):
            chk = validate_conformation(seq, c, lat, e)
            if not chk["ok"]:
                bad_conf.append(f"{lat} {seq} {tag}: {chk['problems']}")
    t_enum = time.perf_counter() - t0

    adm = verify_bound_admissible(verbose=False)
    match = verify_bound_matches_search(verbose=False)

    t1 = time.perf_counter()
    rows = benchmark_2d(time_limit=benchmark_time_limit, verbose=False)
    t_bench = time.perf_counter() - t1
    bench_ok = all(r["gap"] == 0 and r["conformation_valid"] for r in rows)

    res = {"max_err_vs_enumeration": max_err, "n_cases": len(cases),
           "n_walks_enumerated": n_walks_total, "seconds_enumeration": t_enum,
           "per_lattice": per_lattice, "invalid_conformations": bad_conf,
           "bound_audit": adm, "bound_match": match, "benchmarks": rows,
           "benchmarks_all_match_published": bench_ok, "seconds_benchmarks": t_bench}

    if verbose:
        print("  rem.hp.verify")
        print(f"    (a) branch-and-bound optimum vs FULL ENUMERATION of self-avoiding "
              f"walks, {len(cases)} sequences")
        for lat in ("square", "cubic"):
            d = per_lattice[lat]
            print(f"        {lat:<7} {d['n']:>2} sequences, length <= {d['max_len']}, "
                  f"max |E_branch-and-bound - E_enumeration| = {d['max_err']}")
        print(f"        {n_walks_total:,} self-avoiding walks enumerated and scored by "
              f"the O(n^2) all-pairs definition in {t_enum:.1f}s")
        print(f"        max error over everything: {max_err}   (exact integer equality "
              f"required)")
        verify_bound_admissible(verbose=True)
        verify_bound_matches_search(verbose=True)
        print("    (b) published 2D HP benchmarks (Unger & Moult 1993 and successors)")
        _print_benchmark_rows(rows)
        print(f"    (c) conformations: {2 * len(cases) + len(rows)} folds validated "
              f"(self-avoiding, unit steps, energy recomputed from scratch); "
              f"{len(bad_conf)} invalid")
        si = structure_info(rows[0]["sequence"], "square")
        print(f"    cost: the factor-graph form of the {si['n']}-mer has "
              f"d = {si['d_states_per_variable']:,} lattice sites per residue and "
              f"treewidth {si['treewidth']} (self-avoidance couples every pair), so "
              f"d**treewidth = 1e{si['log10_elimination_cost']:.0f}.")
        print(f"          branch and bound instead visited {rows[0]['nodes']:,} nodes "
              f"({rows[0]['seconds']:.2f}s) for that instance.")
    return res


if __name__ == "__main__":
    verify()
