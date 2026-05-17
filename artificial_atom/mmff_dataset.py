"""Build a real-chemistry training dataset using RDKit + MMFF94.

For each SMILES we:
  1. parse, add H, embed N starting conformers (random seeds)
  2. MMFF94-minimise each (this gives us a low-energy "near-native" pose)
  3. perturb each minimum by random per-atom Gaussian displacements at
     several noise scales to span near-native -> decoy regions
  4. record MMFF94 energy and gradient (-> forces) at the perturbed geometry
  5. label by RMSD from the corresponding minimised conformer:
        near_native = RMSD < 0.4 A
        bad         = RMSD > 1.2 A
        intermediate samples are kept but unlabelled (pose_quality = NaN)

The output of `generate_chem_dataset` is a list of `ChemConfiguration`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField, MMFFGetMoleculeProperties,
)

from .chem_features import featurise


# A small but chemically diverse list of drug-like fragments.
DEFAULT_SMILES = (
    # alkanes / alcohols / amines
    "CC", "CCC", "CCO", "CCN", "CC(C)O", "CCCN",
    # carbonyls
    "CC=O", "CC(=O)C", "CC(=O)O", "CC(=O)N",
    # ethers / esters
    "COC", "CCOC", "CC(=O)OC",
    # aromatic
    "c1ccccc1", "c1ccncc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
    # heterocycles
    "c1cc[nH]c1", "c1ccoc1",
    # halogens
    "CCCl", "CC(F)F",
    # small drug-like fragments
    "CC(C)Cc1ccccc1", "OCC1OCCCC1",
)


@dataclass
class ChemConfiguration:
    # RDKit-derived per-atom and per-bond features (see chem_features.featurise)
    positions: torch.Tensor
    Z: torch.Tensor
    hybridisation: torch.Tensor
    formal_charge: torch.Tensor
    aromatic: torch.Tensor
    in_ring: torch.Tensor
    h_donor: torch.Tensor
    h_acceptor: torch.Tensor
    partial_charge: torch.Tensor
    vdw_radius: torch.Tensor

    bonds: torch.Tensor
    bond_order: torch.Tensor
    bond_in_ring: torch.Tensor
    bond_conjugated: torch.Tensor
    bond_rotatable: torch.Tensor

    # MMFF94 labels
    energy: torch.Tensor          # scalar (kcal/mol)
    forces: torch.Tensor          # (N, 3)
    rmsd_to_min: float            # A; smaller -> more native-like

    # Pose-quality label: 1 = near-native, 0 = bad, -1 = unlabelled (in between)
    pose_label: int

    # Reference to the source for debugging
    smiles: str = ""

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]


def _ff_for(mol) -> object | None:
    mp = MMFFGetMoleculeProperties(mol)
    if mp is None:
        return None
    return MMFFGetMoleculeForceField(mol, mp)


def _energy_and_forces(mol) -> tuple[float, torch.Tensor]:
    """MMFF energy and forces in (kcal/mol, kcal/mol/A) at the current conformer."""
    ff = _ff_for(mol)
    if ff is None:
        return float("nan"), torch.zeros(mol.GetNumAtoms(), 3)
    E = ff.CalcEnergy()
    g = ff.CalcGrad()                       # flat list of 3N floats: gradient (not force)
    g_t = torch.tensor(g, dtype=torch.float32).reshape(-1, 3)
    return E, -g_t                           # forces = -gradient


def _coords_to_conformer(mol, coords: torch.Tensor) -> None:
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(((a - b) ** 2).sum(dim=-1).mean()).item()


def _label_for(rmsd_val: float, near_th: float, bad_th: float) -> int:
    if rmsd_val < near_th:
        return 1
    if rmsd_val > bad_th:
        return 0
    return -1


def _make_one_record(
    mol, smiles: str, min_coords: torch.Tensor, ref_features: dict,
    energy: float, forces: torch.Tensor, pose_label: int, rmsd_val: float,
) -> ChemConfiguration:
    feats = dict(ref_features)
    # positions overridden by current conformer; everything else is geometry-independent
    feats["positions"] = torch.tensor(
        [[mol.GetConformer().GetAtomPosition(i).x,
          mol.GetConformer().GetAtomPosition(i).y,
          mol.GetConformer().GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float32,
    )
    return ChemConfiguration(
        positions=feats["positions"],
        Z=feats["Z"],
        hybridisation=feats["hybridisation"],
        formal_charge=feats["formal_charge"],
        aromatic=feats["aromatic"],
        in_ring=feats["in_ring"],
        h_donor=feats["h_donor"],
        h_acceptor=feats["h_acceptor"],
        partial_charge=feats["partial_charge"],
        vdw_radius=feats["vdw_radius"],
        bonds=feats["bonds"],
        bond_order=feats["bond_order"],
        bond_in_ring=feats["bond_in_ring"],
        bond_conjugated=feats["bond_conjugated"],
        bond_rotatable=feats["bond_rotatable"],
        energy=torch.tensor(float(energy), dtype=torch.float32),
        forces=forces,
        rmsd_to_min=rmsd_val,
        pose_label=pose_label,
        smiles=smiles,
    )


def _displace_atoms(
    coords: torch.Tensor, atom_idxs: list[int], dists: list[float], rng: torch.Generator
) -> torch.Tensor:
    """Move each listed atom by its distance, in independent random directions."""
    new = coords.clone()
    for ai, d in zip(atom_idxs, dists):
        direction = torch.randn(3, generator=rng)
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-8)
        new[ai] = new[ai] + direction * d
    return new


def generate_chem_dataset(
    smiles_list: Iterable[str] = DEFAULT_SMILES,
    n_conformers_per_smi: int = 3,
    n_perturbations_per_conf: int = 6,
    noise_scales: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
    n_displacements: int = 12,
    n_atoms_to_displace: tuple[int, ...] = (1, 2, 3),
    displace_dists: tuple[float, ...] = (1.2, 1.8, 2.5, 3.0),
    # RMSD thresholds scaled to small-molecule fluctuations.
    # bad_th lowered relative to v3-initial so the test set has enough
    # bad-pose samples for a statistically meaningful rejection rate.
    near_th: float = 0.25,
    bad_th: float = 0.5,
    energy_cap: float = 1500.0,  # bad poses are above ~300; cap so outliers don't dominate loss
    seed: int = 0,
) -> list[ChemConfiguration]:
    rng = torch.Generator().manual_seed(seed)
    configs: list[ChemConfiguration] = []
    smiles_list = list(smiles_list)

    for smi in smiles_list:
        mol_template = Chem.MolFromSmiles(smi)
        if mol_template is None:
            continue
        mol_template = Chem.AddHs(mol_template)
        # Embed multiple starting conformers
        seeds = [
            int(torch.randint(0, 10**6, (1,), generator=rng).item())
            for _ in range(n_conformers_per_smi)
        ]
        for cid_seed in seeds:
            mol = Chem.Mol(mol_template)
            ok = AllChem.EmbedMolecule(mol, randomSeed=cid_seed)
            if ok < 0:
                continue
            # MMFF minimise to get a "native" pose
            ff = _ff_for(mol)
            if ff is None:
                continue
            ff.Minimize(maxIts=200)
            min_E, min_F = _energy_and_forces(mol)
            if not (min_E == min_E):  # NaN
                continue
            # Cache geometry-independent features once (Z, bonds, hyb, charges, ...)
            ref_feats = featurise(mol)
            min_coords = ref_feats["positions"].clone()

            # Include the minimum itself as a sample (near-native by construction)
            configs.append(_make_one_record(
                mol, smi, min_coords, ref_feats,
                energy=min_E, forces=min_F, pose_label=1, rmsd_val=0.0,
            ))

            # 1) Per-atom Gaussian perturbations (mostly produce near-native
            #    or intermediate poses)
            for sigma in noise_scales:
                for _ in range(n_perturbations_per_conf):
                    noise = torch.randn(min_coords.shape, generator=rng) * sigma
                    new_coords = min_coords + noise
                    _coords_to_conformer(mol, new_coords)
                    E, F = _energy_and_forces(mol)
                    if not (E == E) or E > energy_cap:
                        continue
                    rmsd_val = _rmsd(new_coords, min_coords)
                    label = _label_for(rmsd_val, near_th, bad_th)
                    configs.append(_make_one_record(
                        mol, smi, min_coords, ref_feats,
                        energy=E, forces=F, pose_label=label, rmsd_val=rmsd_val,
                    ))

            # 2) Multi-atom displacements: pick k atoms (k in [1..3]), move
            #    each by a chosen distance. Produces a range of clear
            #    "bad pose" samples whose energies stay finite.
            n_atoms = min_coords.shape[0]
            for _ in range(n_displacements):
                k = int(n_atoms_to_displace[
                    int(torch.randint(0, len(n_atoms_to_displace), (1,), generator=rng).item())
                ])
                k = min(k, n_atoms)
                idxs = torch.randperm(n_atoms, generator=rng)[:k].tolist()
                dists = [
                    float(displace_dists[
                        int(torch.randint(0, len(displace_dists), (1,), generator=rng).item())
                    ])
                    for _ in range(k)
                ]
                new_coords = _displace_atoms(min_coords, idxs, dists, rng)
                _coords_to_conformer(mol, new_coords)
                E, F = _energy_and_forces(mol)
                if not (E == E) or E > energy_cap:
                    continue
                rmsd_val = _rmsd(new_coords, min_coords)
                label = _label_for(rmsd_val, near_th, bad_th)
                configs.append(_make_one_record(
                    mol, smi, min_coords, ref_feats,
                    energy=E, forces=F, pose_label=label, rmsd_val=rmsd_val,
                ))

    return configs


def chem_dataset_stats(configs: list[ChemConfiguration]) -> dict:
    Es = torch.stack([c.energy for c in configs])
    F_all = torch.cat([c.forces.reshape(-1) for c in configs])
    near = sum(1 for c in configs if c.pose_label == 1)
    bad = sum(1 for c in configs if c.pose_label == 0)
    mid = sum(1 for c in configs if c.pose_label == -1)
    return {
        "n_configs": len(configs),
        "energy_mean": float(Es.mean()),
        "energy_std": float(Es.std()),
        "energy_min": float(Es.min()),
        "energy_max": float(Es.max()),
        "force_rms": float(F_all.pow(2).mean().sqrt()),
        "n_atoms_total": sum(c.n_atoms for c in configs),
        "n_near_native": near,
        "n_bad": bad,
        "n_intermediate": mid,
        "unique_smiles": len({c.smiles for c in configs}),
    }
