"""
Priority 3 blinded benchmark — leave-one-out k_cat prediction.

For each reaction with a measured k_cat in the Syn3A kinetic database:
  1. Hold out that reaction's k_cat
  2. Find all other reactions in the SAME reaction class that have
     a SMILES for their primary substrate
  3. Use Morgan-2 Tanimoto similarity to find the nearest substrate
  4. Predict k_cat = nearest_kcat * similarity^2
  5. Compare to the held-out ground truth

This is the test that decides whether the Priority 3 architecture
is useful for organisms with NO measured kinetic data, or only
useful as a tool to extrapolate between closely related substrates.

Outputs
-------
  stdout                                : summary statistics
  data/priority3_benchmark.csv          : per-reaction results
  data/priority3_benchmark_scatter.png  : log-log predicted vs measured
"""

from __future__ import annotations
import sys, csv, math
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from layer3_reactions.sbml_parser import parse_sbml
from layer3_reactions.kinetics import load_all_kinetics
from layer3_reactions.novel_substrates import _infer_reaction_class
from layer3_reactions.metabolite_smiles import BIGG_TO_SMILES


# ============================================================================
# Config
# ============================================================================
COFACTORS = {
    'M_atp_c', 'M_adp_c', 'M_amp_c', 'M_gtp_c', 'M_gdp_c',
    'M_nad_c', 'M_nadh_c', 'M_nadp_c', 'M_nadph_c',
    'M_h_c', 'M_h_e', 'M_h2o_c', 'M_h2o_e',
    'M_pi_c', 'M_ppi_c', 'M_coa_c',
}


@dataclass
class ReactionEntry:
    short_name: str
    primary_substrate: str
    reaction_class: str
    kcat_measured: float
    smiles: str = ''
    fingerprint: object = None


@dataclass
class BenchmarkResult:
    short_name: str
    reaction_class: str
    kcat_measured: float
    kcat_predicted: Optional[float]
    nearest_neighbor: Optional[str]
    nearest_kcat: Optional[float]
    tanimoto: Optional[float]
    reason_skipped: str = ''

    @property
    def fold_error(self) -> Optional[float]:
        if self.kcat_predicted is None or self.kcat_predicted <= 0:
            return None
        if self.kcat_measured <= 0:
            return None
        return max(self.kcat_measured / self.kcat_predicted,
                   self.kcat_predicted / self.kcat_measured)


# ============================================================================
# Assemble the reaction set
# ============================================================================
def build_reaction_entries() -> List[ReactionEntry]:
    sbml_path = (Path(__file__).resolve().parent.parent
                 / 'data' / 'Minimal_Cell_ComplexFormation'
                 / 'input_data' / 'Syn3A_updated.xml')
    sbml = parse_sbml(sbml_path)
    kinetics = load_all_kinetics()
    rxns_by_short = sbml.reactions_by_short_name()

    entries: List[ReactionEntry] = []
    for short_name, k in kinetics.items():
        if short_name not in rxns_by_short:
            continue
        if k.kcat_forward <= 0:
            continue
        rxn = rxns_by_short[short_name]
        primary = [r for r in rxn.reactants if r not in COFACTORS]
        if not primary:
            continue
        sub = primary[0]
        smiles = BIGG_TO_SMILES.get(sub)
        fp = None
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048)
        entries.append(ReactionEntry(
            short_name=short_name,
            primary_substrate=sub,
            reaction_class=_infer_reaction_class(rxn),
            kcat_measured=k.kcat_forward,
            smiles=smiles or '',
            fingerprint=fp,
        ))
    return entries


# ============================================================================
# Leave-one-out prediction
# ============================================================================
def predict_for(held_out: ReactionEntry,
                all_entries: List[ReactionEntry]) -> BenchmarkResult:
    result = BenchmarkResult(
        short_name=held_out.short_name,
        reaction_class=held_out.reaction_class,
        kcat_measured=held_out.kcat_measured,
        kcat_predicted=None,
        nearest_neighbor=None,
        nearest_kcat=None,
        tanimoto=None,
    )

    if held_out.fingerprint is None:
        result.reason_skipped = 'no_smiles'
        return result

    # Find all OTHER reactions in the same class with a fingerprint
    candidates = [e for e in all_entries
                  if e.short_name != held_out.short_name
                  and e.reaction_class == held_out.reaction_class
                  and e.fingerprint is not None]
    if not candidates:
        result.reason_skipped = 'no_same_class_reference'
        return result

    # Find the nearest (by Tanimoto) substrate in the reference pool.
    # Tie-break by lexicographic order of short_name for determinism.
    best_sim = -1.0
    best = None
    for c in candidates:
        sim = DataStructs.TanimotoSimilarity(held_out.fingerprint, c.fingerprint)
        if sim > best_sim or (sim == best_sim and c.short_name < best.short_name):
            best_sim = sim
            best = c

    # k_cat = best reference's k_cat, scaled by similarity^2.
    # This is the SimilarityBackend formula from layer1_atomic/engine.py.
    predicted = best.kcat_measured * (best_sim ** 2)

    result.kcat_predicted = predicted
    result.nearest_neighbor = best.short_name
    result.nearest_kcat = best.kcat_measured
    result.tanimoto = best_sim
    return result


# ============================================================================
# Summary statistics
# ============================================================================
def summarize(results: List[BenchmarkResult]) -> None:
    predicted = [r for r in results if r.kcat_predicted is not None]
    skipped = [r for r in results if r.kcat_predicted is None]

    print(f'\nTotal reactions with measured k_cat: {len(results)}')
    print(f'  Predicted:  {len(predicted)}')
    print(f'  Skipped:    {len(skipped)}')
    if skipped:
        reasons = Counter(r.reason_skipped for r in skipped)
        for reason, n in reasons.most_common():
            print(f'    - {reason}: {n}')

    if not predicted:
        return

    # Fold errors (geometric distance from truth)
    fold_errors = np.array([r.fold_error for r in predicted
                             if r.fold_error is not None])
    log10_errors = np.log10(fold_errors)

    print(f'\nFold-error distribution across {len(fold_errors)} predictions:')
    print(f'  median:       {np.median(fold_errors):.2f}x')
    print(f'  geometric mean: {10**np.mean(log10_errors):.2f}x')
    print(f'  90th percentile: {10**np.percentile(log10_errors, 90):.2f}x')
    print()
    print(f'  within  2x:  {np.sum(fold_errors <= 2.0):>3d}/{len(fold_errors)}  '
          f'({100*np.mean(fold_errors <= 2.0):.0f}%)')
    print(f'  within  5x:  {np.sum(fold_errors <= 5.0):>3d}/{len(fold_errors)}  '
          f'({100*np.mean(fold_errors <= 5.0):.0f}%)')
    print(f'  within 10x:  {np.sum(fold_errors <= 10.0):>3d}/{len(fold_errors)}  '
          f'({100*np.mean(fold_errors <= 10.0):.0f}%)')

    # Per-class breakdown
    print(f'\nPer-class performance (median fold-error, % within 5x):')
    print(f'  {"class":<20s} {"n":>4s}  {"median":>8s}  {"<=5x":>6s}  {"<=10x":>6s}')
    by_class = defaultdict(list)
    for r in predicted:
        if r.fold_error is not None:
            by_class[r.reaction_class].append(r.fold_error)
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        errs = np.array(by_class[cls])
        print(f'  {cls:<20s} {len(errs):>4d}  {np.median(errs):>7.2f}x  '
              f'{100*np.mean(errs <= 5.0):>5.0f}%  '
              f'{100*np.mean(errs <= 10.0):>5.0f}%')

    # Similarity-banded performance
    print(f'\nFold-error as a function of Tanimoto similarity to nearest ref:')
    print(f'  {"Tanimoto":<15s} {"n":>4s}  {"median":>8s}')
    bins = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]
    for lo, hi in bins:
        band = [r.fold_error for r in predicted
                if r.tanimoto is not None
                and lo <= r.tanimoto < hi
                and r.fold_error is not None]
        if band:
            print(f'  [{lo:.2f}, {hi:.2f})     {len(band):>4d}  '
                  f'{np.median(band):>7.2f}x')


# ============================================================================
# Save outputs
# ============================================================================
def save_csv(results: List[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['short_name', 'class', 'kcat_measured',
                     'kcat_predicted', 'fold_error',
                     'nearest_neighbor', 'nearest_kcat', 'tanimoto',
                     'reason_skipped'])
        for r in sorted(results, key=lambda r: r.short_name):
            w.writerow([
                r.short_name,
                r.reaction_class,
                f'{r.kcat_measured:.4f}',
                f'{r.kcat_predicted:.4f}' if r.kcat_predicted is not None else '',
                f'{r.fold_error:.3f}' if r.fold_error is not None else '',
                r.nearest_neighbor or '',
                f'{r.nearest_kcat:.4f}' if r.nearest_kcat is not None else '',
                f'{r.tanimoto:.4f}' if r.tanimoto is not None else '',
                r.reason_skipped,
            ])
    print(f'\nWrote CSV: {path}')


def save_scatter(results: List[BenchmarkResult], path: Path) -> None:
    predicted = [r for r in results if r.kcat_predicted is not None
                 and r.kcat_predicted > 0 and r.kcat_measured > 0]
    if not predicted:
        return
    measured = np.array([r.kcat_measured for r in predicted])
    predicted_arr = np.array([r.kcat_predicted for r in predicted])
    classes = [r.reaction_class for r in predicted]

    # Colour by class
    unique_classes = sorted(set(classes))
    cmap = plt.get_cmap('tab10')
    class_colors = {c: cmap(i) for i, c in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for cls in unique_classes:
        idx = [i for i, c in enumerate(classes) if c == cls]
        ax.scatter(measured[idx], predicted_arr[idx],
                    c=[class_colors[cls]], label=f'{cls} (n={len(idx)})',
                    alpha=0.6, s=40, edgecolors='black', linewidth=0.5)

    # Perfect-prediction line and fold bands
    mn = min(measured.min(), predicted_arr.min()) * 0.5
    mx = max(measured.max(), predicted_arr.max()) * 2
    diag = np.array([mn, mx])
    ax.plot(diag, diag,      'k-',  lw=1.5, label='perfect')
    ax.plot(diag, diag * 2,   'k--', lw=0.6, alpha=0.6)
    ax.plot(diag, diag / 2,   'k--', lw=0.6, alpha=0.6, label='±2x')
    ax.plot(diag, diag * 10,  'k:',  lw=0.6, alpha=0.6)
    ax.plot(diag, diag / 10,  'k:',  lw=0.6, alpha=0.6, label='±10x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('measured k_cat (1/s)')
    ax.set_ylabel('predicted k_cat (1/s)')
    ax.set_title(f'Priority 3 leave-one-out benchmark '
                  f'(n={len(predicted)}, Syn3A)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Wrote scatter plot: {path}')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print('=' * 64)
    print('Priority 3 leave-one-out benchmark on Syn3A k_cat database')
    print('=' * 64)

    entries = build_reaction_entries()
    with_smiles = sum(1 for e in entries if e.fingerprint is not None)
    print(f'\nLoaded {len(entries)} reactions with measured k_cat')
    print(f'  {with_smiles} have SMILES for their primary substrate')
    print(f'  {len(entries) - with_smiles} do not (protein substrates, ions)')

    print(f'\nRunning leave-one-out prediction...')
    results = [predict_for(e, entries) for e in entries]

    summarize(results)

    out_dir = Path(__file__).resolve().parent.parent / 'data'
    save_csv(results,     out_dir / 'priority3_benchmark.csv')
    save_scatter(results, out_dir / 'priority3_benchmark_scatter.png')

    print()
