"""Parse Human-GEM into the bipartite compartmentalised graph. A cache, not a test -- no gates.

WHY BIPARTITE AND NOT SPECIES-TO-SPECIES. Only 2,431 of 12,931 Human-GEM reactions are 1->1. The
other 81% converge or diverge, and an edge between two species cannot represent a 3->4 reaction
without either inventing 12 fictitious edges or inventing an intermediate that does not exist. A
bipartite graph -- species nodes and reaction nodes, edges only between the two kinds -- represents
every reaction exactly, and convergence and divergence become in-degree and out-degree.

WHAT IS KEPT. Direction (reactant -> reaction -> product), reversibility (both directions added
when the SBML says reversible), and the compartment of every species. Stoichiometric COEFFICIENTS
are dropped: this is a topology, not a flux model, and that limit is stated here rather than
discovered downstream.

-> colab/data/rem_bipartite.npz
"""
import re
import sys
import time
from pathlib import Path

import numpy as np

GEM = Path("HumanGEM.xml")
OUT = Path("colab/data/rem_bipartite.npz")


def build(say=print):
    t0 = time.time()
    s = GEM.read_text(errors="replace")
    say(f"  read {len(s) / 1e6:.1f} MB [{time.time() - t0:.0f}s]")

    sp_comp, sp_name = {}, {}
    for m in re.finditer(r'<species\b[^>]*?\bid="([^"]+)"[^>]*?>', s):
        tag = m.group(0)
        sid = m.group(1)
        c = re.search(r'compartment="([^"]+)"', tag)
        n = re.search(r'\bname="([^"]*)"', tag)
        if c:
            sp_comp[sid] = c.group(1)
            sp_name[sid] = n.group(1) if n else sid
    species = sorted(sp_comp)
    si = {g: i for i, g in enumerate(species)}
    say(f"  {len(species):,} species in {len(set(sp_comp.values()))} compartments")

    rxn, rev = [], []
    er_r, er_s, ep_r, ep_s = [], [], [], []

    def refs(block, tag):
        seg = re.search(r"<listOf%s>(.*?)</listOf%s>" % (tag, tag), block, re.S)
        return re.findall(r'species="([^"]+)"', seg.group(1)) if seg else []

    for m in re.finditer(r"<reaction\b.*?</reaction>", s, re.S):
        b = m.group(0)
        rid = re.search(r'\bid="([^"]+)"', b).group(1)
        j = len(rxn)
        rxn.append(rid)
        rev.append(1 if re.search(r'reversible="true"', b) else 0)
        for x in refs(b, "Reactants"):
            if x in si:
                er_r.append(j)
                er_s.append(si[x])
        for x in refs(b, "Products"):
            if x in si:
                ep_r.append(j)
                ep_s.append(si[x])
    say(f"  {len(rxn):,} reactions, {sum(rev):,} reversible "
        f"({sum(rev) / len(rxn):.1%}) [{time.time() - t0:.0f}s]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        species=np.array(species), sp_comp=np.array([sp_comp[g] for g in species]),
        sp_name=np.array([sp_name[g] for g in species]),
        reactions=np.array(rxn), reversible=np.array(rev, np.int8),
        react_rx=np.array(er_r, np.int32), react_sp=np.array(er_s, np.int32),
        prod_rx=np.array(ep_r, np.int32), prod_sp=np.array(ep_s, np.int32))
    say(f"  -> {OUT}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    build()
