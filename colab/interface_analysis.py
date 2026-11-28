"""interface_analysis — step 6c: pin WHICH interaction a mutation breaks, from a COMPLEX structure.

The monomer scaffold can't see interfaces (a residue buried in the monomer can sit at a dimer interface — the
BRAF V600E / KRAS G12 pattern). The fix is a complex structure. Two sources, same analysis:
  - EXPERIMENTAL complex from the PDB (best — reachable now, no GPU); or
  - AlphaFold-Multimer prediction (Colab/GPU) for complexes with no solved structure.

Given a complex and a mutated residue (chain + position), report which OTHER chains that residue contacts (heavy
atoms within a cutoff). If it contacts partner X, the mutation is AT the X interface -> that interaction is the
one at risk. This turns 'the mutation matters, somewhere' into 'the mutation is at the VHL-ElonginC contact ->
VHL can't recruit ElonginC -> HIF is not degraded'.
-> interface_partners(pdb, chain, resnum) ; demo on VHL R167
"""
import os, sys, urllib.request, ssl, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

_CA = "/root/.ccr/ca-bundle.crt"
CHAIN_NAME = {}   # optional label map per PDB


def _ctx():
    return ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None


def fetch_pdb(pdb_id, cache="pdbs"):
    os.makedirs(cache, exist_ok=True)
    dst = f"{cache}/{pdb_id}.pdb"
    if not os.path.exists(dst):
        data = urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                                      timeout=60, context=_ctx()).read()
        open(dst, "wb").write(data)
    return dst


def parse_complex(path):
    """-> {chain: {resnum: [(x,y,z), ...heavy atoms...]}}  (ATOM records, heavy atoms only)."""
    chains = defaultdict(lambda: defaultdict(list))
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        atom = ln[12:16].strip()
        if atom.startswith("H"):
            continue
        ch = ln[21]
        try:
            res = int(ln[22:26]); x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        chains[ch][res].append((x, y, z))
    return {k: dict(v) for k, v in chains.items()}


def interface_partners(cx, chain, resnum, cutoff=5.0):
    """which OTHER chains does (chain,resnum) contact (any heavy atom within cutoff)? -> {chain: min_distance}."""
    if chain not in cx or resnum not in cx[chain]:
        return {}
    target = cx[chain][resnum]
    out = {}
    for oc, residues in cx.items():
        if oc == chain:
            continue
        best = 1e9
        for _, atoms in residues.items():
            for (x, y, z) in atoms:
                for (ax, ay, az) in target:
                    d = math.sqrt((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2)
                    if d < best:
                        best = d
        if best <= cutoff:
            out[oc] = round(best, 2)
    return out


def analyze(pdb_id, chain, resnum, chain_labels=None, cutoff=5.0):
    cx = parse_complex(fetch_pdb(pdb_id))
    part = interface_partners(cx, chain, resnum, cutoff)
    labels = chain_labels or {}
    present = resnum in cx.get(chain, {})
    return {"pdb": pdb_id, "chain": chain, "resnum": resnum, "residue_present": present,
            "contacts": {labels.get(c, c): d for c, d in part.items()},
            "at_interface": bool(part),
            "verdict": (f"AT the {'/'.join(labels.get(c, c) for c in part)} interface -> that interaction is the "
                        f"one at risk" if part else "NOT at any inter-chain interface in this structure -> the "
                        f"mutation's effect is intramolecular (stability/allostery), not a specific partner break")}


if __name__ == "__main__":
    # VHL R167W (Type-2 VHL disease): does 167 sit at the ElonginC interface or the HIF interface?  1LM8 has
    # ElonginB(B), ElonginC(C), VHL(V), HIF1a(H).
    labels = {"B": "ElonginB", "C": "ElonginC", "V": "VHL", "H": "HIF1a"}
    for pos in (167, 158, 111):   # 167 (disease), 158, 111 (a known HIF-contact residue) for contrast
        r = analyze("1LM8", "V", pos, labels)
        print(f"VHL {pos}: present={r['residue_present']} contacts={r['contacts']}")
        print(f"   -> {r['verdict']}")
