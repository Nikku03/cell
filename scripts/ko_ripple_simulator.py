"""Knockout-ripple simulator: trace a gene KO through the predicted PPI graph.

STATUS: scaffold. Implementation deferred until we know the LNN+PPI
architecture generalizes (regularized held-out test pending).

Idea
====

For each gene G in the held-out organism:
1. KO gene G → its protein product is gone.
2. Find G's predicted PPI partners (using PPIPredictor, threshold ≥ 0.7).
3. Those partners are now potentially broken — they can't form their
   complex without G. Mark them as "compromised".
4. Repeat: find each compromised partner's partners, mark them too.
5. Continue BFS until either:
   - the ripple reaches a "terminal" node (a protein with no further
     high-confidence PPI partners)
   - the ripple plateaus
6. Examine the terminal compounds: if any of them is essential for cell
   viability (e.g., ribosomal proteins, ATP synthase subunits — known
   from training-organism essentiality patterns + GO terms), gene G
   is essential.

Why this might work
===================

This is structurally analogous to v15's mechanistic detector but operates
on the predicted PPI graph instead of the curated metabolic network. v15
catches cases where KO disrupts a specific pathway (ATP synthesis, DNA
replication, etc.). The KO ripple catches cases where KO disrupts
complex assembly that v15's curated complex list misses.

The cross-org PPI predictor has MCC ~0.48 with hard negatives — noisy
but not random. Aggregating predictions over a graph traversal should
amplify signal vs noise via averaging.

Why this might NOT work
=======================

If the PPI predictor is mostly noise (which the regularized held-out
result will tell us), then the ripple will just be noise-of-noise. The
terminal compound classification will be unreliable.

This is the kind of architecture that MUST be validated against
held-out data. In-domain metrics will be inflated by the same
memorization mechanism we already saw with PPI features.

Plan (to be filled in once Stage 1 validates direction)
=======================================================

1. PPI graph builder
   - For each organism, score all upper-triangular pairs with
     PPIPredictor (already done in scripts/score_ppi_features.py).
   - Build adjacency map {gene: [(partner, score), ...]} keeping pairs
     with score ≥ 0.7.

2. KO ripple BFS
   - Initialize: queue = [(G, weight=1.0)]
   - For each (gene, weight) popped:
     - For each partner with edge_score >= threshold:
       - new_weight = weight * edge_score (compounded confidence)
       - if new_weight > eps: add (partner, new_weight) to queue
     - Stop when queue empty or visited >= max_nodes
   - Output: dict {visited_gene: max_compounded_weight}

3. Terminal compound essentiality
   - For each visited gene, look up:
     - is it in a known-essential GO category? (ribosome, ATPase,
       polymerase, etc.)
     - did training-organism orthologs come up essential?
     - does its sequence match a known-essential family?
   - Aggregate: max_terminal_essentiality = max(scores over visited)

4. Final per-gene call
   - cascade(G) = essential if max_terminal_essentiality * propagation
     > threshold

Deferred until structure pipeline lands and PPI predictor's
generalization is established.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(
        description=('Stub. Will simulate KO ripple through predicted PPI '
                       'graph. Pending validation that PPI predictor has '
                       'cross-org signal worth propagating.'))
    ap.add_argument('--organism', default='mtuberculosis',
                     help='organism to KO-ripple test on')
    ap.add_argument('--out', type=Path,
                     default=Path('outputs/ko_ripple_predictions.csv'))
    args = ap.parse_args()
    print('=' * 60)
    print('STUB — KO ripple simulator not yet implemented.')
    print('See module docstring for the planned pipeline.')
    print('=' * 60)
    print(f'Args parsed: organism={args.organism}, out={args.out}')
    print('\nNext step: validate the foundation first')
    print('  1. Run notebooks/holdout_mtb_lnn_regularized.ipynb')
    print('  2. If mtb MCC > 0.30, structure + ripple work is justified')
    print('  3. If mtb MCC < 0.20, the architecture is not generalizing')
    print('     and adding more components on top will not fix it')


if __name__ == '__main__':
    main()
