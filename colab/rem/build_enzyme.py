"""The enzyme, direction, ion and proton layer of the REM graph. A cache, not a test -- no gates.

Loops 160 and 161 built a metabolite<->reaction graph and never asked who CATALYSES anything. This
adds the four channels that turn it into a directed protein-level object:

  ENZYMES AND WHICH PROTEINS WORK TOGETHER. Human-GEM carries 2,848 gene products and 7,782
  gene-product associations as an AND/OR tree. The distinction is not decoration: an AND group is a
  protein COMPLEX -- those subunits must all be present, so they are proteins that literally
  interact -- while an OR group is a set of ISOZYMES, alternatives that never need to meet. The
  tree is converted to disjunctive normal form, so every reaction becomes a list of complexes, any
  one of which suffices. Isozyme count and complex size then mean what they say.

  DIRECTION, FROM THE FLUX BOUNDS RATHER THAN THE FLAG. The SBML's reversible= attribute and the
  fbc flux bounds are two separate declarations and are not required to agree. The bounds are the
  ones a flux model obeys: lower bound 0 means the reaction only runs forward, lower bound -1000
  means both ways, upper bound 0 means reverse only. All three classes are recorded and their
  disagreement with the reversible= flag is counted rather than assumed away.

  IONS. 28 species are a bare metal or halide ion. Which reactions consume or produce one is a
  cofactor requirement, and it is recorded per ion and per compartment.

  PROTONS AND pH. Net H+ stoichiometry per reaction, and separately the H+ that CROSSES a membrane,
  which is the only kind that changes a compartment's pH relative to another. A reaction that emits
  a proton into the cytosol and one that pumps a proton from cytosol to lysosome are different
  events and are counted separately.

-> colab/data/rem_enzyme.npz
"""
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GEM = Path("HumanGEM.xml")
OUT = Path("colab/data/rem_enzyme.npz")
ION_RE = re.compile(r"^(Fe|Zn|Mg|Mn|Cu|Ca|Co|Na|K|Cl|Ni|Mo|Se|I)\d*$")


def tag(e):
    """Local tag name. The GPR fragment is reparsed with `fbc:` rewritten to `fbc_` because the
    fragment carries no namespace declaration of its own, so the prefix has to be stripped here as
    well as the `{uri}` form -- getting this wrong silently yields an empty GPR for every reaction."""
    return e.tag.rsplit("}", 1)[-1].replace("fbc_", "")


def dnf(node):
    """GPR tree -> list of complexes, each a frozenset of gene ids. Any one complex suffices."""
    t = tag(node)
    if t == "geneProductRef":
        gp = [v for k, v in node.attrib.items() if k.endswith("geneProduct")]
        return [frozenset(gp)] if gp else []
    kids = [c for c in node if tag(c) in ("and", "or", "geneProductRef")]
    if t == "or":
        out = []
        for c in kids:
            out.extend(dnf(c))
        return out
    if t == "and":
        cur = [frozenset()]
        for c in kids:
            sub = dnf(c)
            if not sub:
                continue
            cur = [a | b for a in cur for b in sub]
        return cur
    out = []
    for c in kids:
        out.extend(dnf(c))
    return out


def build(say=print):
    s = GEM.read_text(errors="replace")

    genes = re.findall(r'<fbc:geneProduct [^>]*fbc:id="([^"]+)"[^>]*fbc:label="([^"]*)"', s)
    gid = {g: i for i, (g, _) in enumerate(genes)}
    say(f"  {len(genes):,} gene products")

    params = {m.group(1): float(m.group(2))
              for m in re.finditer(r'<parameter[^>]*id="([^"]+)"[^>]*value="(-?[\d.]+)"', s)}

    sp_name, sp_form, sp_comp = {}, {}, {}
    for m in re.finditer(r"<species\b[^>]*?>", s):
        t = m.group(0)
        sid = re.search(r'\bid="([^"]+)"', t)
        if not sid:
            continue
        sid = sid.group(1)
        n = re.search(r'\bname="([^"]*)"', t)
        f = re.search(r'fbc:chemicalFormula="([^"]*)"', t)
        c = re.search(r'compartment="([^"]+)"', t)
        sp_name[sid] = n.group(1) if n else sid
        sp_form[sid] = f.group(1) if f else ""
        sp_comp[sid] = c.group(1) if c else "?"
    species = sorted(sp_comp)
    si = {g: i for i, g in enumerate(species)}
    is_ion = np.array([bool(ION_RE.match(sp_form[g] or "")) for g in species])
    is_h = np.array([sp_name[g] == "H+" for g in species])
    say(f"  {int(is_ion.sum())} bare ion species, {int(is_h.sum())} explicit H+ species")

    rxn, lo, hi, revflag = [], [], [], []
    g_rx, g_gene, g_cplx = [], [], []
    ncplx, maxsize = [], []
    hnet, hcross = [], []
    ion_rx, ion_sp = [], []

    for m in re.finditer(r"<reaction\b.*?</reaction>", s, re.S):
        b = m.group(0)
        head = b[:b.index(">") + 1]
        j = len(rxn)
        rxn.append(re.search(r'\bid="([^"]+)"', head).group(1))
        lb = re.search(r'fbc:lowerFluxBound="([^"]+)"', head)
        ub = re.search(r'fbc:upperFluxBound="([^"]+)"', head)
        lo.append(params.get(lb.group(1), 0.0) if lb else 0.0)
        hi.append(params.get(ub.group(1), 0.0) if ub else 0.0)
        revflag.append(1 if 'reversible="true"' in head else 0)

        gpa = re.search(r"<fbc:geneProductAssociation>.*?</fbc:geneProductAssociation>", b, re.S)
        comps = []
        if gpa:
            try:
                root = ET.fromstring(
                    gpa.group(0).replace("fbc:", "fbc_").replace("xmlns:", "xmlnsx:"))
                comps = [c for c in dnf(root) if c]
            except ET.ParseError:
                comps = [frozenset(re.findall(r'fbc:geneProduct="([^"]+)"', gpa.group(0)))]
        ncplx.append(len(comps))
        maxsize.append(max((len(c) for c in comps), default=0))
        for ci, c in enumerate(comps):
            for g in c:
                if g in gid:
                    g_rx.append(j)
                    g_gene.append(gid[g])
                    g_cplx.append(ci)

        h = 0.0
        hc = Counter()
        ions_here = set()
        for tagname, sign in (("Reactants", -1.0), ("Products", 1.0)):
            seg = re.search(r"<listOf%s>(.*?)</listOf%s>" % (tagname, tagname), b, re.S)
            if not seg:
                continue
            for sm in re.finditer(r"<speciesReference\b[^>]*?/?>", seg.group(1)):
                st = sm.group(0)
                sp = re.search(r'species="([^"]+)"', st)
                co = re.search(r'stoichiometry="([^"]+)"', st)
                if not sp or sp.group(1) not in si:
                    continue
                i = si[sp.group(1)]
                c = float(co.group(1)) if co else 1.0
                if is_h[i]:
                    h += sign * c
                    hc[(sp_comp[species[i]], sign)] += c
                if is_ion[i]:
                    ions_here.add(i)
        hnet.append(h)
        src = {c for (c, sg) in hc if sg < 0}
        dstc = {c for (c, sg) in hc if sg > 0}
        hcross.append(1 if (src - dstc) and (dstc - src) else 0)
        for i in ions_here:
            ion_rx.append(j)
            ion_sp.append(i)

    lo, hi = np.array(lo), np.array(hi)
    revflag = np.array(revflag, np.int8)
    direction = np.where(lo < 0, 1, np.where(hi > 0, 0, 2)).astype(np.int8)  # 0 fwd,1 rev,2 back
    disagree = int(((direction == 1) != (revflag == 1)).sum())
    say(f"  {len(rxn):,} reactions | direction from bounds: "
        f"forward-only {int((direction == 0).sum()):,}, reversible {int((direction == 1).sum()):,}, "
        f"reverse-only {int((direction == 2).sum()):,}")
    say(f"  flux bounds disagree with the reversible= flag on {disagree} reactions")
    say(f"  GPR: {len(set(g_rx)):,} reactions have an enzyme | {len(set(g_gene)):,} genes used | "
        f"{len(g_rx):,} gene-complex-reaction memberships")
    say(f"  complexes: max subunits {max(maxsize)}, reactions with a multi-subunit complex "
        f"{int((np.array(maxsize) > 1).sum()):,}, mean isozyme count "
        f"{np.mean([c for c in ncplx if c]):.2f}")
    say(f"  protons: net nonzero in {int((np.array(hnet) != 0).sum()):,} reactions; "
        f"H+ crosses a membrane in {int(np.sum(hcross)):,}")
    say(f"  ion participation: {len(ion_rx):,} reaction-ion pairs over "
        f"{len(set(ion_rx)):,} reactions")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        genes=np.array([g for g, _ in genes]), symbols=np.array([l for _, l in genes]),
        reactions=np.array(rxn), species=np.array(species),
        lower=lo, upper=hi, reversible_flag=revflag, direction=direction,
        gpr_rx=np.array(g_rx, np.int32), gpr_gene=np.array(g_gene, np.int32),
        gpr_complex=np.array(g_cplx, np.int32),
        n_complexes=np.array(ncplx, np.int32), max_subunits=np.array(maxsize, np.int32),
        h_net=np.array(hnet, np.float32), h_crosses=np.array(hcross, np.int8),
        is_ion=is_ion, is_hplus=is_h,
        ion_rx=np.array(ion_rx, np.int32), ion_sp=np.array(ion_sp, np.int32))
    say(f"  -> {OUT}")


if __name__ == "__main__":
    build()
