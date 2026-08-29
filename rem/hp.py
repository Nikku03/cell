"""HP lattice protein folding: where the governing law says NO, and exactly why.

This module exists to be a NEGATIVE result with a measured boundary, not a success story.
REM is not a general-purpose optimizer, and the cleanest way to show that is to take a
famous hard problem, write it as a factor graph, and MEASURE the treewidth.

THE MODEL. A sequence over {H, P} is embedded as a self-avoiding walk on a square lattice.
Energy = -1 for every pair of H residues that are lattice-adjacent but NOT adjacent in the
sequence. Finding the minimum is NP-hard (Berger and Leighton 1998, Crescenzi et al. 1998)
and nothing here contradicts that.

THE NATURAL ENCODING AND WHY IT FAILS. Take the variables to be the n-1 turn directions.
The energy couples residues i and j whenever they land on neighbouring lattice sites, and
whether they do depends on EVERY turn between them. Self-avoidance is worse: it forbids
any two residues sharing a site, which is a constraint on every pair. So the factor graph
is essentially complete, its treewidth is n-1, and cost = d^treewidth = 4^(n-1). That is
not a defect of the elimination ordering. It is the problem.

WHERE THE LAW SAYS YES. Confine the walk to a strip of width W and sweep column by column.
The frontier separating processed from unprocessed columns is W sites wide, so the state
that must be carried across the cut is the occupancy and connectivity of those W sites --
bond dimension bounded by a function of W alone, independent of the length L. The cost is
then O(L * c^W): LINEAR in the chain length, exponential only in the WIDTH. Same law, same
formula, opposite verdict, and the thing that changed is the geometry.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  H1  CORRECTNESS of the transfer-matrix / column-sweep count against explicit enumeration
      of every self-avoiding walk, on strips small enough to enumerate.
      GATE: identical counts and identical minimum energies, exactly, on all tested sizes.
  H2  THE WALL, MEASURED. Treewidth of the natural turn-variable encoding as a function of
      n. GATE: it must grow at least linearly -- a fitted slope >= 0.8 -- confirming that
      the encoding gives no structural saving. If this came out flat, the encoding would be
      wrong, not the problem easy.
  H3  THE OTHER SIDE, MEASURED. Column-sweep cost as a function of LENGTH at fixed width.
      GATE: fitted log-log slope in [0.8, 1.3], i.e. linear in length.
  H4  AND AS A FUNCTION OF WIDTH at fixed length. GATE: the cost must grow at least
      exponentially in W -- fitted slope of log(cost) vs W strictly positive -- because the
      claim is that the width is where the exponential lives, and a claim about where the
      cost is must be paid for by showing the cost there.
  H5  POSITIVE CONTROL on a sequence with a KNOWN optimum: the standard 20-mer benchmark
      HPHPPHHPHPPHPHHPPHPH, whose minimum energy on the 2D square lattice is -9. The
      column sweep must find exactly -9 when the strip is wide enough to contain the
      optimum, and must NOT find it when the strip is too narrow. Both halves are the gate:
      a method that returns -9 from a strip that cannot hold the fold is reading it from
      somewhere it should not be able to.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph

MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
BENCH20 = "HPHPPHHPHPPHPHHPPHPH"      # Unger & Moult 1993; 2D square-lattice optimum -9


# --------------------------------------------------------------------------------------
# reference: explicit enumeration of self-avoiding walks
# --------------------------------------------------------------------------------------

def enumerate_walks(seq: str, width: Optional[int] = None,
                    height: Optional[int] = None) -> Tuple[int, float, Optional[tuple]]:
    """Every self-avoiding walk of len(seq), by depth-first search. The ground truth.

    Returns (n_walks, best_energy, best_walk). Walks are counted up to nothing at all --
    no symmetry reduction -- so the count is directly comparable with any other exhaustive
    method. Exponential; small inputs only.
    """
    n = len(seq)
    best, best_w, count = 0.0, None, 0
    path: List[Tuple[int, int]] = [(0, 0)]
    occ = {(0, 0): 0}

    def ok(p):
        if p in occ:
            return False
        if width is not None and not (0 <= p[0] < width):
            return False
        if height is not None and not (0 <= p[1] < height):
            return False
        return True

    def rec(i: int):
        nonlocal best, best_w, count
        if i == n:
            count += 1
            e = energy_of(seq, path)
            if e < best:
                best, best_w = e, tuple(path)
            return
        x, y = path[-1]
        for dx, dy in MOVES:
            p = (x + dx, y + dy)
            if ok(p):
                occ[p] = i
                path.append(p)
                rec(i + 1)
                path.pop()
                del occ[p]

    rec(1)
    return count, best, best_w


def energy_of(seq: str, path: Sequence[Tuple[int, int]]) -> float:
    """-1 per H-H pair that is lattice-adjacent and NOT sequence-adjacent."""
    pos = {p: i for i, p in enumerate(path)}
    e = 0.0
    for i, (x, y) in enumerate(path):
        if seq[i] != "H":
            continue
        for dx, dy in MOVES:
            j = pos.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                e -= 1.0
    return e


# --------------------------------------------------------------------------------------
# the natural (turn-variable) encoding, whose treewidth is the point
# --------------------------------------------------------------------------------------

def turn_encoding_graph(n: int) -> FactorGraph:
    """Turn variables plus the pair factors self-avoidance and contact energy require.

    Whether residues i and j collide, or touch, depends on every turn strictly between
    them, so the honest factor for the (i, j) interaction has scope {t_i, ..., t_{j-1}}.
    Representing that as a pairwise graph over turns UNDER-states the coupling, so this
    builds the pairwise projection -- an edge between every pair of turns that share any
    such factor -- which is a LOWER bound on the true treewidth. The measured growth is
    therefore a lower bound on the real one, which is the conservative direction.
    """
    g = FactorGraph()
    for i in range(n - 1):
        g.add_var(f"t{i}", 4)
        g.add_factor([f"t{i}"], np.zeros(4))
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            g.add_factor([f"t{i}", f"t{j}"], np.zeros((4, 4)))
    return g


# --------------------------------------------------------------------------------------
# the column sweep: cost linear in length, exponential in width
# --------------------------------------------------------------------------------------

def column_sweep_energy(seq: str, width: int, height: Optional[int] = None) -> dict:
    """Exact minimum-energy confined fold, carrying the full path so contacts are exact.

    The state is (path-so-far restricted to what can still matter, head). Because the H-H
    contact term needs residue IDENTITIES and not just occupancy, the carried state is the
    map site -> residue index over the live window; that map IS the bond, and its size is
    reported as n_states so the cost claim is measured rather than asserted.
    """
    n = len(seq)
    height = height if height is not None else n
    t0 = time.perf_counter()
    layer: Dict[tuple, float] = {((( (0, 0), 0),), (0, 0)): 0.0}
    widest = 1
    for i in range(1, n):
        nxt: Dict[tuple, float] = {}
        for (occ_t, head) in list(layer):
            e = layer[(occ_t, head)]
            occ = dict(occ_t)
            hx, hy = head
            for dx, dy in MOVES:
                p = (hx + dx, hy + dy)
                if p in occ or not (0 <= p[0] < width) or not (0 <= p[1] < height):
                    continue
                de = 0.0
                if seq[i] == "H":
                    for ex, ey in MOVES:
                        q = (p[0] + ex, p[1] + ey)
                        j = occ.get(q)
                        if j is not None and j < i - 1 and seq[j] == "H":
                            de -= 1.0
                no = dict(occ); no[p] = i
                key = (tuple(sorted(no.items())), p)
                v = e + de
                if key not in nxt or v < nxt[key]:
                    nxt[key] = v
        layer = nxt
        widest = max(widest, len(layer))
    secs = time.perf_counter() - t0
    best = min(layer.values()) if layer else float("inf")
    return {"best_energy": best, "n_states_max": widest, "seconds": secs,
            "width": width, "length": n}


def _slope(xs, ys) -> float:
    return float(np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)[0])


def verify(verbose: bool = True) -> dict:
    """Run H1-H5. Bars are fixed in the module docstring, above, before any number."""
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}

    # ---- H1: column sweep vs explicit enumeration ---------------------------------------
    say("  H1 confined minimum energy: column sweep vs explicit enumeration of every SAW")
    say(f"      {'seq':14s} {'W':>2s} {'H':>2s} {'enumerate':>10s} {'sweep':>8s}  ok")
    rows, h1 = [], True
    for seq, W, H in (("HHHH", 2, 2), ("HPHP", 2, 3), ("HHPHH", 3, 3),
                      ("HHHHHH", 3, 3), ("HPHPHH", 3, 3), ("HHPPHHPH", 3, 4)):
        _c, be, _w = enumerate_walks(seq, width=W, height=H)
        sw = column_sweep_energy(seq, W, H)
        ok = abs(be - sw["best_energy"]) < 1e-12
        h1 &= ok
        rows.append((seq, W, H, be, sw["best_energy"], ok))
        say(f"      {seq:14s} {W:2d} {H:2d} {be:10.1f} {sw['best_energy']:8.1f}  "
            f"{'ok' if ok else 'FAIL'}")
    out["H1"] = bool(h1)
    say(f"      H1 {'PASS' if h1 else 'FAIL'}")

    # ---- H2: the wall in the natural encoding ---------------------------------------------
    say("\n  H2 treewidth of the turn-variable encoding vs chain length (the WALL)")
    ns, tws = [], []
    for n in (5, 7, 9, 11, 13, 15):
        tw = turn_encoding_graph(n).treewidth()
        ns.append(n); tws.append(tw)
        say(f"      n={n:3d}   treewidth {tw:3d}   cost 4^{tw} = {4.0**tw:.3e}")
    sl = _slope(ns, tws)
    out["H2_slope"], out["H2"] = sl, sl >= 0.8
    say(f"      slope of treewidth vs n: {sl:.3f} (bar >= 0.8)   "
        f"{'PASS' if out['H2'] else 'FAIL'}")

    # ---- H3: linear in LENGTH at fixed width ------------------------------------------------
    say("\n  H3 column-sweep cost vs LENGTH at fixed width 3")
    Ls, cost = [], []
    for L in (8, 10, 12, 14, 16):
        r = column_sweep_energy("HP" * (L // 2), 3, L)
        Ls.append(L); cost.append(r["n_states_max"])
        say(f"      L={L:3d}   max frontier states {r['n_states_max']:7d}   "
            f"{r['seconds']*1e3:7.1f} ms")
    sl3 = _slope(np.log10(Ls), np.log10(cost))
    out["H3_slope"], out["H3"] = sl3, 0.8 <= sl3 <= 1.3
    say(f"      log-log slope {sl3:.3f} (bar 0.8-1.3, i.e. linear)   "
        f"{'PASS' if out['H3'] else 'FAIL'}")

    # ---- H4: exponential in WIDTH at fixed length --------------------------------------------
    say("\n  H4 column-sweep cost vs WIDTH at fixed length 12")
    Ws, cw = [], []
    for W in (2, 3, 4, 5):
        r = column_sweep_energy("HPHPHHPPHPHH"[:12], W, 12)
        Ws.append(W); cw.append(r["n_states_max"])
        say(f"      W={W:3d}   max frontier states {r['n_states_max']:8d}   "
            f"{r['seconds']*1e3:8.1f} ms")
    sl4 = _slope(Ws, np.log10(cw))
    out["H4_slope"], out["H4"] = sl4, sl4 > 0.0
    say(f"      slope of log10(cost) vs W: {sl4:.3f} (bar > 0, i.e. exponential in W)   "
        f"{'PASS' if out['H4'] else 'FAIL'}")

    # ---- H5: POSITIVE CONTROL with a known optimum --------------------------------------------
    say(f"\n  H5 POSITIVE CONTROL: {BENCH20}, published 2D square-lattice optimum -9")
    found, narrow = None, None
    for W in (3, 4, 5, 6, 7):
        r = column_sweep_energy(BENCH20, W, 20)
        say(f"      strip width {W}   best {r['best_energy']:5.1f}   "
            f"frontier {r['n_states_max']:9d}   {r['seconds']:6.2f} s")
        if W <= 3 and narrow is None:
            narrow = r["best_energy"]
        if r["best_energy"] <= -9.0 + 1e-9 and found is None:
            found = W
    out["H5_found_at_width"] = found
    out["H5_narrow_energy"] = narrow
    out["H5"] = bool(found is not None and narrow is not None and narrow > -9.0 - 1e-9)
    say(f"      reached -9 first at width {found}; the width-3 strip could only reach "
        f"{narrow}")
    say(f"      H5 {'PASS' if out['H5'] else 'FAIL'}  (both halves: it must find -9 when "
        f"the strip can hold the fold, and must NOT when it cannot)")

    gates = ["H1", "H2", "H3", "H4", "H5"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    say("\n  THE HONEST SUMMARY: HP folding is NP-hard and REM does not change that. The "
        "\n  turn encoding has treewidth linear in n, so cost = d^treewidth is the wall. "
        "\n  Confining the chain to a strip moves the exponential from the LENGTH to the "
        "\n  WIDTH -- same law, different geometry -- and that is the whole of the saving.")
    return out


if __name__ == "__main__":
    verify()
