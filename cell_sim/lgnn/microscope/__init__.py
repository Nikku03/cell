"""Microscope tools: post-training analyses on trained dynamics models.

Three microscope deliverables, all post-hoc analyses:

  pattern_atlas   -- cluster trajectory data into discrete cell states
                      (whole-cell tier) and per-species dynamic states
                      (atomic tier). Maps each (species, time) → label.
  edge_attribution -- dump M1+axis2's per-edge attention α across many
                      timesteps. Identifies the load-bearing connections.
  knockout_sweep  -- gene-by-gene perturbation sweep with rollout
                      divergence as the essentiality proxy.
"""
from cell_sim.lgnn.microscope.knockout_sweep import (
    knockout_sweep,
    gene_row_mapping_from_loci,
    evaluate_against_essentiality,
)
from cell_sim.lgnn.microscope.pattern_atlas import (
    whole_cell_atlas,
    per_species_atlas,
    characterize_whole_cell_clusters,
    save_atlas,
    load_atlas,
)
from cell_sim.lgnn.microscope.edge_attribution import (
    capture_attention,
    aggregate_attention,
    top_attended_per_node,
)

__all__ = [
    'knockout_sweep',
    'gene_row_mapping_from_loci',
    'evaluate_against_essentiality',
    'whole_cell_atlas',
    'per_species_atlas',
    'characterize_whole_cell_clusters',
    'save_atlas',
    'load_atlas',
    'capture_attention',
    'aggregate_attention',
    'top_attended_per_node',
]
