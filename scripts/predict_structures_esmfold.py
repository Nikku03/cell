"""Structure prediction + structural feature extraction for the PPI predictor.

STATUS: scaffold. Implementation deferred until the regularized LNN+PPI
held-out result tells us whether structural features have a chance of
helping cross-org generalization. If the regularized run on mtb stays
< 0.20 MCC, structure features are unlikely to fix a fundamentally
non-generalizing architecture and we should pivot.

Plan (to be filled in once Stage 1 validates direction)
=======================================================

1. **Structure prediction pipeline** — ESMFold on Colab (~30 min for 455
   Syn3A proteins; ~10 hours for the full 8-org panel).

   from transformers import AutoTokenizer, EsmForProteinFolding
   tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
   model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')
   model.cuda(); model.eval()

   For each protein sequence:
     tokens = tokenizer([aa_seq], return_tensors='pt', padding=True)
     out = model(tokens.input_ids.cuda())
     # out.positions: [N_residues, 37, 3] atom coords
     # out.plddt: [N_residues] confidence scores
     # save PDB + features

2. **Per-protein structural features** — to be appended to the existing
   per-gene scalar features:

   - mean_plddt: confidence in the predicted fold (well-folded vs disordered)
   - frac_disordered: fraction of residues with pLDDT < 50
   - n_helix, n_sheet, n_coil: secondary structure composition
   - radius_of_gyration: compactness
   - n_pockets: count of cavities deep enough to be ligand-binding
                (computed via P2Rank or FPocket on the predicted PDB)
   - max_pocket_volume: largest pocket size

3. **Structure-aware PPI predictor** — extend `PPIPredictor` with a
   structure encoder branch that takes per-protein structure features +
   pairwise structural compatibility (e.g., predicted-pocket-size
   compatibility, surface charge complementarity).

4. **Re-train PPI predictor** with structure as input → re-score per-gene
   PPI features → re-train LNN.

5. **KO ripple simulation** (separate script:
   scripts/ko_ripple_simulator.py) — for each gene, traverse the
   predicted PPI graph from the KO origin, mark partners as broken if
   their structural compatibility implies the missing partner is
   load-bearing, propagate until reaching terminal nodes, classify gene
   as essential if the ripple breaks any essential pathway.

Deferred until we know structure features have a chance of helping.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(
        description=('Stub. Will run ESMFold on a list of proteins. Pending '
                       'validation that LNN+PPI architecture has cross-org '
                       'signal worth augmenting.'))
    ap.add_argument('--organisms', default='syn3a',
                     help='comma-separated organism list (default: syn3a only)')
    ap.add_argument('--out', type=Path,
                     default=Path('outputs/structure_features_per_gene.csv'))
    args = ap.parse_args()
    print('=' * 60)
    print('STUB — structure prediction not yet implemented.')
    print('See module docstring for the planned pipeline.')
    print('=' * 60)
    print(f'Args parsed: organisms={args.organisms}, out={args.out}')
    print('\nNext step: run the regularized held-out experiment first')
    print('  notebooks/holdout_mtb_lnn_regularized.ipynb')
    print('If mtb MCC > 0.30, build out this script.')
    print('If mtb MCC < 0.20, pivot away from PPI features for cross-org.')


if __name__ == '__main__':
    main()
