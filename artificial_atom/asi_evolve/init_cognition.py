"""Seed the ASI-Evolve cognition store with artificial_atom v3 priors."""
from __future__ import annotations

import json
import sys


COGNITION_SEED = [
    {
        'title': 'Verification contract is sacred',
        'content': (
            'Each failed verification check costs 10 points in composite. '
            'Wall-time savings cost 0.1 points per minute. So saving 10 '
            'minutes by speeding up training is worth 1 point, but failing '
            'one check costs 10 points. Always prioritize keeping 8/8 '
            'passing over speed gains.'
        ),
    },
    {
        'title': 'Energy R² has the most slack to lose',
        'content': (
            'Energy R² baseline is 0.974 with threshold 0.85. There is 0.12 '
            'of slack. Reducing EPOCHS_PRIMARY below 12 risks dropping R² '
            'below 0.85 quickly. Aggressive epoch reductions almost always '
            'break this check first.'
        ),
    },
    {
        'title': 'Force MAE is overprovisioned',
        'content': (
            'Force MAE baseline is 32.85 with threshold 200. 6x margin. '
            'FORCE_WEIGHT=0.02 is low because MMFF forces are O(100); '
            'increasing to 0.05-0.1 helps Force MAE without major risk to '
            'other checks. This is a safe knob.'
        ),
    },
    {
        'title': 'Pose rejection and retention are imbalanced',
        'content': (
            'Bad-pose rejection 0.9792 (threshold 0.95, 0.03 margin) and '
            'near-native retention 0.8962 (threshold 0.85, 0.05 margin) '
            'both depend on the pose classifier head. BAD_POSE_WEIGHT=4.0 '
            'biases toward rejecting bad poses. Lowering it would help '
            'retention but hurt rejection.'
        ),
    },
    {
        'title': 'Ensemble OOD ratio is huge',
        'content': (
            'OOD std / in-dist std baseline is 7.35 with threshold 2.0. '
            'Lots of slack. Reducing EPOCHS_ENSEMBLE (the 2nd member) '
            'mainly affects this check. Going from 12 -> 6 epochs probably '
            'still keeps it above 2.0.'
        ),
    },
    {
        'title': 'Translation/rotation/Newton invariances are architectural',
        'content': (
            'These three checks are not affected by training hyperparameters '
            '— they depend on the model architecture being equivariant by '
            'construction. Changing D_H, EPOCHS, LR, etc. will not affect '
            'them. Only changes to the core forward pass would.'
        ),
    },
    {
        'title': 'LR safe range',
        'content': (
            'LR > 1e-3 causes loss divergence on MMFF data (forces are '
            'O(100)). LR < 1e-4 trains too slowly to converge in 18 epochs. '
            'Sweet spot is 3e-4 (default). 5e-4 is the upper safe bound.'
        ),
    },
    {
        'title': 'Batch size and gradient signal',
        'content': (
            'BATCH_SIZE=8 (default) is small because each sample has '
            'variable atom count (1421 samples avg ~10 atoms each). '
            'Increasing to 16 or 32 reduces gradient noise but may slow '
            'wall-time if the per-sample autograd graph dominates. '
            'Diminishing returns above 32.'
        ),
    },
    {
        'title': 'N_LAYERS halving',
        'content': (
            'N_LAYERS=1 (down from 2) roughly halves training time per '
            'epoch but reduces effective receptive field. For small drug-'
            'like molecules (~10 atoms), one layer can still capture most '
            'interactions, but rotation/translation invariance is unaffected '
            '(those are architectural). Bigger risk: Energy R² may drop.'
        ),
    },
    {
        'title': 'Cutoff radii physics',
        'content': (
            'R_CUTOFF=6.0Å covers van der Waals interactions; '
            'R_CUTOFF_ELECTRO=8.0Å is for longer-range coulomb. Reducing '
            'either reduces compute per message-pass but loses physics; '
            '6.0/8.0 are well-tuned defaults from the demo.'
        ),
    },
    {
        'title': 'Total params vs accuracy tradeoff',
        'content': (
            'D_H=40 D_RBF=20 N_LAYERS=2 gives ~75k params. Doubling D_H '
            'to 80 is 4x compute and 4x params with marginal accuracy '
            'gain on this small dataset (1421 samples). Direction of '
            'profitable mutation is usually DOWN (less compute, similar '
            'accuracy).'
        ),
    },
    {
        'title': 'Ensemble training cost',
        'content': (
            'The 2nd ensemble member is trained ONLY for the OOD check. '
            'It costs ~5 min of wall time. Reducing its EPOCHS_ENSEMBLE '
            'from 12 down to 6-8 typically still passes the OOD check '
            '(threshold 2.0; we have 7.35x in baseline).'
        ),
    },
    {
        'title': 'Composite scoring rewards small wins',
        'content': (
            'composite = 10*n_failed + 0.1*wall_minutes. Cutting train '
            'from 8 min to 4 min saves 0.04 composite. Failing one check '
            'costs 10 composite. So speed mutations that risk a check '
            'are net-negative; safe speed mutations (e.g. fewer ensemble '
            'epochs) compound nicely.'
        ),
    },
]


def write_seed(path: str = 'cognition_seed.json') -> None:
    with open(path, 'w') as f:
        json.dump(COGNITION_SEED, f, indent=2)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'cognition_seed.json'
    write_seed(out)
    print(f'wrote {len(COGNITION_SEED)} cognition seeds to {out}',
          file=sys.stderr)
