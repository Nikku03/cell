"""Microscope tools: post-training analyses on trained dynamics models.

Currently exposes:
  knockout_sweep — gene-by-gene perturbation sweep with rollout
                    divergence as the essentiality proxy
"""
from cell_sim.lgnn.microscope.knockout_sweep import (
    knockout_sweep,
    gene_row_mapping_from_loci,
    evaluate_against_essentiality,
)

__all__ = [
    'knockout_sweep',
    'gene_row_mapping_from_loci',
    'evaluate_against_essentiality',
]
