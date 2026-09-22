"""Is a LOCAL metabolic neighbourhood low-treewidth? Measured on HumanGEM.

THE QUESTION. REM cannot do whole-network metabolism -- E. coli core came out at 10^36 and
that is settled. But a weaker and more useful claim was proposed: that REM could model the
LOCAL ENVIRONMENT around a single reaction, to turn in-vitro kinetic constants into in-vivo
ones. That needs only the neighbourhood to be low-treewidth, not the network. This measures
whether it is.

THE OBVIOUS ESCAPE, TESTED. Metabolic networks are famously dense because of currency
metabolites -- ATP, ADP, NAD, H+, water -- which appear in thousands of reactions and connect
everything to everything. Standard practice is to exclude them. So the sweep is run three
ways: all species, top-30 species removed, top-100 removed. If the density is a currency-
metabolite artifact, removing them should collapse the treewidth.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  N1  Treewidth of an n-reaction neighbourhood, as a fraction of n. A neighbourhood REM can
      use needs treewidth that stays roughly constant as n grows, the way a chain or a
      hub-plus-chain does. GATE: is treewidth/n falling with n? If it is flat and near 1 the
      neighbourhood is a near-clique and cost is d^n, which is the wall at local scale.
  N2  Does removing currency metabolites rescue it? GATE: does treewidth at n = 40 drop by
      more than half when the top 100 species are removed?
"""
from __future__ import annotations

import collections, random, sys
sys.path.insert(0, ".")
import numpy as np
import xml.etree.ElementTree as ET
from rem.factorgraph import FactorGraph

NS = {'s': 'http://www.sbml.org/sbml/level3/version1/core'}
SBML = "HumanGEM.xml"


def parse(path=SBML):
    out = {}
    for _ev, el in ET.iterparse(path, events=("end",)):
        if el.tag.endswith('}reaction'):
            sp = set()
            for side in ('listOfReactants', 'listOfProducts'):
                L = el.find(f's:{side}', NS)
                if L is not None:
                    for r in L:
                        s = r.get('species')
                        if s:
                            sp.add(s)
            out[el.get('id')] = sp
            el.clear()
    return out


def adjacency(rxn_species, deg, exclude_top, hub_cap=200):
    banned = set(s for s, _ in deg.most_common(exclude_top))
    sp2r = collections.defaultdict(list)
    for r, sp in rxn_species.items():
        for s in sp:
            if s not in banned:
                sp2r[s].append(r)
    adj = collections.defaultdict(set)
    for s, rs in sp2r.items():
        if len(rs) > hub_cap:
            continue
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                adj[rs[i]].add(rs[j]); adj[rs[j]].add(rs[i])
    return adj


def neighbourhood_treewidth(adj, seed, nmax):
    seen, frontier, S = [seed], [seed], {seed}
    while len(seen) < nmax and frontier:
        nxt = []
        for r in frontier:
            for q in adj.get(r, ()):
                if q not in S:
                    S.add(q); seen.append(q); nxt.append(q)
                    if len(seen) >= nmax:
                        break
            if len(seen) >= nmax:
                break
        frontier = nxt
    sub = seen[:nmax]
    idx = {r: i for i, r in enumerate(sub)}
    g = FactorGraph()
    for r in sub:
        g.add_var(r, 2)
    for r in sub:
        for q in adj.get(r, ()):
            if q in idx and idx[q] > idx[r]:
                g.add_factor([r, q], np.zeros((2, 2)))
    return g.treewidth(), len(sub)


def main():
    rxn = parse()
    print(f"  HumanGEM: {len(rxn):,} reactions")
    deg = collections.Counter()
    for sp in rxn.values():
        for s in sp:
            deg[s] += 1
    print(f"  most-connected species: "
          + ", ".join(f"{s.replace('M_','')}:{c}" for s, c in deg.most_common(6)))
    random.seed(0)
    sizes = (10, 20, 40, 80)
    table = {}
    for exc, label in ((0, "all species"), (30, "top-30 removed"),
                       (100, "top-100 removed")):
        adj = adjacency(rxn, deg, exc)
        seeds = [r for r in rxn if len(adj.get(r, ())) > 2]
        random.shuffle(seeds); seeds = seeds[:6]
        row = {}
        print(f"\n  {label}")
        print(f"      {'n':>4s} {'treewidth':>10s} {'tw/n':>6s}")
        for n in sizes:
            tws = [neighbourhood_treewidth(adj, s, n)[0] for s in seeds]
            m = int(np.median(tws)); row[n] = m
            print(f"      {n:4d} {m:10d} {m/n:6.2f}   (min {min(tws)} max {max(tws)})")
        table[label] = row
    print("\n  N1 does treewidth/n fall as n grows?")
    for label, row in table.items():
        fr = [row[n] / n for n in sizes if n in row]
        print(f"      {label:18s} tw/n = " + " ".join(f"{x:.2f}" for x in fr)
              + ("   FLAT -> near-clique" if fr[-1] > 0.5 * fr[0] else "   falling"))
    a, b = table["all species"].get(40), table["top-100 removed"].get(40)
    if a and b:
        print(f"\n  N2 currency removal at n=40: {a} -> {b} "
              f"({'rescued' if b < a / 2 else 'NOT rescued'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
