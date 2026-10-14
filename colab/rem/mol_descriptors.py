"""RDKit descriptors for the resolved metabolites: the electronics, sterics and bonding a formula omits.

WHAT A FORMULA CANNOT SAY AND THESE CAN. Two molecules with the identical elemental formula differ
in where the atoms are, and that difference IS the chemistry: whether an oxygen is a donor or an
acceptor, whether a ring is aromatic, how much surface is polar, how bulky the molecule is, how
freely it bends. Every descriptor here is a property of connectivity, which is exactly the layer
Human-GEM does not store.

  bonding      real H-bond donors and acceptors by Lipinski definition, not a count of N and O;
               rotatable bonds; ring count and aromatic ring count; fraction of sp3 carbon
  electronics  topological polar surface area, formal charge, and Crippen logP as the
               hydrophobicity that loop 165 found was the one complementarity term that worked
  sterics      molar refractivity as polarisable bulk, Labute accessible surface area, heavy-atom
               count, and the Bertz topological complexity index

-> colab/data/ml/mol_descriptors.npz
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
STRUCT = Path("colab/data/ml/metabolite_structures.json")
OUT = Path("colab/data/ml/mol_descriptors.npz")
NAMES = ["hbd", "hba", "rotb", "rings", "arom_rings", "fsp3", "tpsa", "logp", "mr",
         "labute_asa", "heavy", "bertz", "formal_charge", "n_stereo"]


def main():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    st = json.load(open(STRUCT))
    accs, rows, nfail = [], [], 0
    for k, (sp, d) in enumerate(st.items()):
        m = None
        try:
            if "inchi" in d:
                m = Chem.MolFromInchi(d["inchi"])
            if m is None and "smiles" in d:
                m = Chem.MolFromSmiles(d["smiles"])
        except Exception:
            m = None
        if m is None:
            nfail += 1
            continue
        try:
            v = [
                Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m),
                rdMolDescriptors.CalcTPSA(m), Crippen.MolLogP(m), Crippen.MolMR(m),
                rdMolDescriptors.CalcLabuteASA(m), m.GetNumHeavyAtoms(),
                Descriptors.BertzCT(m), Chem.GetFormalCharge(m),
                len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
            ]
        except Exception:
            nfail += 1
            continue
        if not all(np.isfinite(v)):
            nfail += 1
            continue
        accs.append(sp)
        rows.append(v)
        if (k + 1) % 1500 == 0:
            print(f"  {k+1:,}/{len(st):,} [{time.time()-t0:.0f}s]", flush=True)
    X = np.array(rows, np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, accs=np.array(accs), X=X, names=np.array(NAMES))
    print(f"  {X.shape[0]:,} metabolites x {X.shape[1]} descriptors "
          f"({nfail:,} unparseable) -> {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
