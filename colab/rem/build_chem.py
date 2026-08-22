"""Add the chemistry layer to the bipartite graph: coefficients, formulas, charges. A cache, no gates.

Loop 160's R7 said outright that stoichiometric coefficients were dropped and that the graph was
therefore a topology and not a chemistry. This restores them, plus the two things coefficients are
only useful alongside: every species' elemental formula and its formal charge, both of which
Human-GEM carries in the SBML fbc package and neither of which the topology used.

WHAT THAT BUYS. With coefficients and formulas, a reaction has a MASS BALANCE -- the elements on
the left must equal the elements on the right -- and with charges it has an ELECTRON balance. Those
are constraints no amount of graph structure can see, and they do not care which metabolites are
popular, so a predictor built on them cannot be the degree confound that R5 lost to.

ALIGNMENT. The edge arrays here are produced by the same left-to-right scan as colab/rem/build.py,
so index k of react_coef corresponds to index k of that file's react_rx/react_sp. Loop 161's S1
checks that rather than trusting it.

GENERIC FORMULAS. 830 species carry an R group and 297 an X -- placeholders for an unspecified
side chain. A reaction containing one cannot be balance-checked, and those are flagged rather than
silently counted as balanced.

-> colab/data/rem_chem.npz
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np

GEM = Path("HumanGEM.xml")
OUT = Path("colab/data/rem_chem.npz")
GENERIC = {"R", "X", "Y", "Z"}


def parse_formula(f):
    d = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f or ""):
        if el:
            d[el] += int(n) if n else 1
    return d


def build(say=print):
    s = GEM.read_text(errors="replace")
    form, chg = {}, {}
    for m in re.finditer(r'<species\b[^>]*?\bid="([^"]+)"[^>]*?>', s):
        tag, sid = m.group(0), m.group(1)
        f = re.search(r'fbc:chemicalFormula="([^"]*)"', tag)
        c = re.search(r'fbc:charge="(-?\d+)"', tag)
        form[sid] = f.group(1) if f else ""
        chg[sid] = int(c.group(1)) if c else 0
    species = sorted(form)
    si = {g: i for i, g in enumerate(species)}
    parsed = {g: parse_formula(form[g]) for g in species}
    elements = sorted({e for d in parsed.values() for e in d} - GENERIC)
    ei = {e: i for i, e in enumerate(elements)}
    E = np.zeros((len(species), len(elements)), np.float32)
    generic = np.zeros(len(species), bool)
    for g, i in si.items():
        for e, n in parsed[g].items():
            if e in GENERIC:
                generic[i] = True
            else:
                E[i, ei[e]] = n
    say(f"  {len(species):,} species | {len(elements)} elements {elements[:12]}...")
    say(f"  {int(generic.sum()):,} species carry a generic R/X group and cannot be balanced")

    rxn = []
    rc_rx, rc_sp, rc_co = [], [], []
    pc_rx, pc_sp, pc_co = [], [], []

    def srefs(block, tag):
        seg = re.search(r"<listOf%s>(.*?)</listOf%s>" % (tag, tag), block, re.S)
        if not seg:
            return []
        out = []
        for m in re.finditer(r"<speciesReference\b[^>]*?/?>", seg.group(1)):
            t = m.group(0)
            sp = re.search(r'species="([^"]+)"', t)
            st = re.search(r'stoichiometry="([^"]+)"', t)
            if sp:
                out.append((sp.group(1), float(st.group(1)) if st else 1.0))
        return out

    for m in re.finditer(r"<reaction\b.*?</reaction>", s, re.S):
        b = m.group(0)
        j = len(rxn)
        rxn.append(re.search(r'\bid="([^"]+)"', b).group(1))
        for sid, co in srefs(b, "Reactants"):
            if sid in si:
                rc_rx.append(j)
                rc_sp.append(si[sid])
                rc_co.append(co)
        for sid, co in srefs(b, "Products"):
            if sid in si:
                pc_rx.append(j)
                pc_sp.append(si[sid])
                pc_co.append(co)
    say(f"  {len(rxn):,} reactions | {len(rc_rx):,} reactant refs, {len(pc_rx):,} product refs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, species=np.array(species), elements=np.array(elements), E=E,
        charge=np.array([chg[g] for g in species], np.float32), generic=generic,
        formula=np.array([form[g] for g in species]),
        reactions=np.array(rxn),
        react_rx=np.array(rc_rx, np.int32), react_sp=np.array(rc_sp, np.int32),
        react_coef=np.array(rc_co, np.float32),
        prod_rx=np.array(pc_rx, np.int32), prod_sp=np.array(pc_sp, np.int32),
        prod_coef=np.array(pc_co, np.float32))
    say(f"  -> {OUT}")


if __name__ == "__main__":
    build()
