"""
Layer 1 — atomic / ML physics engine.

Purpose
-------
When Layer 3 encounters a substrate that isn't in the measured kinetic
database (a drug, a mutant metabolite, an unnatural amino acid, a novel
substrate for a promiscuous enzyme), we need a k_cat estimate from
first principles. This module provides that.

Two backends, both implement the same AtomicEngine interface:

  SimilarityBackend (always available)
    Given (substrate SMILES, enzyme class), finds the most similar
    measured substrate via Morgan-fingerprint Tanimoto distance, returns
    its k_cat scaled by structural similarity. Fast (<10 ms), coarse.

  MACEBackend (needs torch + mace-off + ideally a GPU)
    Uses the MACE-OFF pretrained ML force field to compute the bond-
    dissociation energy of the scissile bond of the substrate. Converts
    that to an activation energy estimate and k_cat via Eyring. Slower
    (~seconds per query), more principled — actually atomic physics.

The architecture is pluggable: Layer 3 asks for an estimate, the engine
returns one; which backend answered is noted in the rate_source field
for auditability.

Honest caveats
--------------
1. Similarity -> k_cat is a correlation, not a derivation. Two molecules
   with 90% Tanimoto similarity can have very different k_cats if one
   lacks a critical functional group (e.g., 2-deoxyglucose vs glucose
   at PGI). That's exactly the failure mode where MACE would help,
   because MACE sees bonds directly.

2. The MACE backend as shipped computes a proxy (BDE of the weakest
   O-H bond in the substrate), not a full transition-state calculation.
   Real k_cat prediction from atomic physics needs either a
   transition-state search (expensive) or a learned barrier-height
   regressor trained on enzyme data (not done here).

3. For truly novel chemistry (the enzyme has no known substrates at all
   in the database), this module cannot help. It handles "substrate
   variant" cases, not "unknown enzyme" cases.

Status: both backends work for the cases tested in tests/demo_priority3.py.
For production atomic-accuracy k_cat estimates you want a TS-search
wrapper on top of MACE, which is future work.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import warnings


# ============================================================================
# Physical constants
# ============================================================================
K_BOLTZMANN = 1.380649e-23   # J/K
H_PLANCK = 6.62607015e-34    # J s
R_GAS = 8.314462618e-3       # kJ/(mol K)
T_PHYSIOLOGICAL = 310.15     # K (37 C)


def eyring_kcat(Ea_kJ_per_mol: float, T: float = T_PHYSIOLOGICAL) -> float:
    """Eyring: k_cat = (kB T / h) exp(-Ea / RT). Returns 1/s."""
    prefactor = K_BOLTZMANN * T / H_PLANCK   # ~6.46e12 /s at 310 K
    return prefactor * math.exp(-Ea_kJ_per_mol / (R_GAS * T))


def kcat_to_Ea(kcat_per_s: float, T: float = T_PHYSIOLOGICAL) -> float:
    """Inverse Eyring: back out an effective Ea from an observed k_cat."""
    if kcat_per_s <= 0:
        return float('inf')
    prefactor = K_BOLTZMANN * T / H_PLANCK
    return -R_GAS * T * math.log(kcat_per_s / prefactor)


# ============================================================================
# Interface
# ============================================================================
@dataclass
class KcatEstimate:
    kcat_per_s: float
    confidence: float                    # 0.0-1.0; 1.0 = exact match
    source: str                          # 'similarity', 'mace_bde', ...
    nearest_known_substrate: Optional[str] = None
    similarity: Optional[float] = None
    activation_energy_kJ_per_mol: Optional[float] = None
    notes: str = ''


@dataclass
class EnzymeProfile:
    name: str
    reaction_class: str                  # 'phospho_transfer', 'isomerase', 'hydrolase'
    known_substrate_smiles: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Similarity backend — always available
# ============================================================================
class SimilarityBackend:
    def __init__(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
            self._Chem = Chem
            self._AllChem = AllChem
            self._DataStructs = DataStructs
            self.available = True
        except ImportError:
            self.available = False

    def estimate_kcat(self, substrate_smiles: str,
                       enzyme: EnzymeProfile) -> KcatEstimate:
        if not self.available:
            return KcatEstimate(1.0, 0.0, 'unavailable',
                                 notes='RDKit not installed')
        if not enzyme.known_substrate_smiles:
            return KcatEstimate(1.0, 0.0, 'no_reference',
                                 notes=f'No known substrates for {enzyme.name}')

        query = self._Chem.MolFromSmiles(substrate_smiles)
        if query is None:
            return KcatEstimate(0.0, 0.0, 'invalid_smiles',
                                 notes=f'Could not parse: {substrate_smiles}')
        qfp = self._AllChem.GetMorganFingerprintAsBitVect(query, 2, nBits=2048)

        best_sim, best_kcat, best_smi = 0.0, 0.0, None
        for smi, kcat in enzyme.known_substrate_smiles.items():
            mol = self._Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = self._AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = self._DataStructs.TanimotoSimilarity(qfp, fp)
            if sim > best_sim:
                best_sim, best_kcat, best_smi = sim, kcat, smi

        # Scale k_cat by similarity^2. An exact match (sim=1) returns the full
        # measured k_cat; a weak analog (sim=0.3) returns ~9% of it. This is a
        # coarse but honest choice that doesn't claim more precision than
        # Tanimoto warrants.
        estimate = best_kcat * (best_sim ** 2)
        return KcatEstimate(
            kcat_per_s=estimate,
            confidence=best_sim,
            source='similarity_morgan_tanimoto',
            nearest_known_substrate=best_smi,
            similarity=best_sim,
            notes=f'Nearest: k_cat={best_kcat:.2f}/s at Tanimoto={best_sim:.3f}',
        )


# ============================================================================
# MACE backend — real atomic physics
# ============================================================================
class MACEBackend:
    """
    Uses MACE-OFF to compute a bond-dissociation-energy proxy for the
    scissile bond, then scales the reference enzyme's measured k_cat by
    the delta-BDE via Hammond-postulate-style transfer (shift in
    activation energy tracks shift in bond strength).
    """

    def __init__(self, model: str = 'small', device: str = 'auto'):
        self.model_name = model
        self.device = device
        self._calc = None
        self.available = False

    def _ensure_loaded(self):
        if self._calc is not None:
            return
        try:
            from mace.calculators import mace_off
            import torch
            device = self.device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Downloads ~100 MB on first call
            self._calc = mace_off(model=self.model_name, device=device)
            self.available = True
        except Exception as e:
            warnings.warn(f'MACE unavailable: {type(e).__name__}: {e}')
            self.available = False

    def estimate_kcat(self, substrate_smiles: str,
                       enzyme: EnzymeProfile) -> KcatEstimate:
        self._ensure_loaded()
        if not self.available:
            return KcatEstimate(0.0, 0.0, 'mace_unavailable',
                                 notes='Fall back to similarity')

        if not enzyme.known_substrate_smiles:
            return KcatEstimate(0.0, 0.0, 'no_reference',
                                 notes='MACE needs >=1 known substrate to calibrate')

        try:
            ref_smi, ref_kcat = next(iter(enzyme.known_substrate_smiles.items()))
            ref_bde = self._min_OH_bde(ref_smi)
            query_bde = self._min_OH_bde(substrate_smiles)
        except Exception as e:
            return KcatEstimate(0.0, 0.0, 'mace_bde_failed',
                                 notes=f'BDE calculation error: {e}')

        ref_Ea = kcat_to_Ea(ref_kcat)
        delta_BDE = query_bde - ref_bde
        query_Ea = ref_Ea + delta_BDE
        query_kcat = eyring_kcat(query_Ea)

        return KcatEstimate(
            kcat_per_s=query_kcat,
            confidence=0.6,
            source='mace_bde',
            nearest_known_substrate=ref_smi,
            activation_energy_kJ_per_mol=query_Ea,
            notes=(f'ref Ea={ref_Ea:.1f} kJ/mol -> query Ea={query_Ea:.1f} kJ/mol '
                    f'(BDE shift {delta_BDE:+.1f})'),
        )

    def _min_OH_bde(self, smiles: str) -> float:
        """Compute the lowest O-H bond dissociation energy in kJ/mol."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from ase import Atoms

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

        conf = mol.GetConformer()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        positions = [[conf.GetAtomPosition(i).x,
                       conf.GetAtomPosition(i).y,
                       conf.GetAtomPosition(i).z]
                      for i in range(mol.GetNumAtoms())]
        full = Atoms(symbols=symbols, positions=positions)
        full.calc = self._calc
        E_full = full.get_potential_energy()

        best = float('inf')
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                continue
            nbrs = atom.GetNeighbors()
            if not nbrs or nbrs[0].GetSymbol() != 'O':
                continue
            keep = [j for j in range(len(full)) if j != i]
            rad = full[keep]
            rad.calc = self._calc
            try:
                E_rad = rad.get_potential_energy()
            except Exception:
                continue
            bde_eV = E_rad - E_full
            bde_kJ = bde_eV * 96.485        # eV -> kJ/mol
            if bde_kJ < best:
                best = bde_kJ
        if not math.isfinite(best):
            return 400.0                     # generic C-H fallback
        return best


# ============================================================================
# Top-level engine
# ============================================================================
class AtomicEngine:
    """
    Routes k_cat queries to MACE (if available and requested) or falls
    back to similarity.
    """

    def __init__(self, backend_name: str = 'auto'):
        self.similarity = SimilarityBackend()
        self._mace: Optional[MACEBackend] = None
        self._backend_name = backend_name
        if backend_name == 'mace':
            self._mace = MACEBackend()
        elif backend_name == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    self._mace = MACEBackend()
            except ImportError:
                pass

    @property
    def active_backend(self) -> str:
        if self._mace is not None:
            return 'mace'
        if self.similarity.available:
            return 'similarity'
        return 'none'

    def estimate_kcat(self, substrate_smiles: str,
                       enzyme: EnzymeProfile) -> KcatEstimate:
        if self._mace is not None:
            result = self._mace.estimate_kcat(substrate_smiles, enzyme)
            if result.source != 'mace_unavailable':
                return result
        return self.similarity.estimate_kcat(substrate_smiles, enzyme)


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    engine = AtomicEngine(backend_name='similarity')
    print(f'Active backend: {engine.active_backend}')

    # alpha-D-glucose, D-mannose as hexokinase references
    hex_enzyme = EnzymeProfile(
        name='HEX1',
        reaction_class='phospho_transfer',
        known_substrate_smiles={
            'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O': 190.0,
            'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O':   45.0,
        },
    )

    cases = {
        '2-deoxyglucose':  'OC[C@H]1OC(O)C[C@@H](O)[C@@H]1O',
        'xylose':          'OC[C@H]1OC(O)[C@H](O)[C@@H]1O',
        '6-deoxyglucose':  'C[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
        'methanol':        'CO',
    }
    print('\nHEX k_cat estimates:')
    for name, smi in cases.items():
        est = engine.estimate_kcat(smi, hex_enzyme)
        print(f'  {name:20s}  k_cat={est.kcat_per_s:>9.2f}/s  '
              f'(conf={est.confidence:.2f}, {est.source})')
        print(f'    {est.notes}')
