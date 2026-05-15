"""Analysis utilities for the count-dynamics surrogate.

Each tool here is INDEPENDENT - you can use one without the others, and
a failure in one doesn't affect the rest. All tools work on the trained
M7 model + the trajectory data without requiring retraining.

  narrate_window          - "What reactions fired in this 1-second window?
                            Which species changed and why?"
  audit_species_economy   - "How many molecules of X were produced and
                            consumed in this time window, by which reactions?"
  find_coupled_species    - "Which gene pairs change together in this
                            replicate? (Hypothesis generation for shared
                            biology.)"
  run_knockout_sweep      - "Rank all gene loci by predicted knockout
                            impact (the main application of M7)."
  decompose_counts_by_location - "Break the aggregate species counts by
                            subcellular location using spatial fingerprints."
"""

from cell_sim.lgnn.analysis.narrate_window import (
    narrate_window,
    format_narrative,
    find_interesting_seconds,
    NarrationResult,
    load_genbank,
    load_stoich,
)
from cell_sim.lgnn.analysis.species_economy import (
    audit_species_economy,
    EconomyAudit,
)
from cell_sim.lgnn.analysis.functional_coupling import (
    find_coupled_species,
    hidden_state_similarity,
)
from cell_sim.lgnn.analysis.knockout_sweep import (
    run_knockout_sweep,
    score_against_breuer,
)
from cell_sim.lgnn.analysis.spatial_decompose import (
    decompose_counts_by_location,
)

__all__ = [
    # narrate_window
    'narrate_window', 'format_narrative', 'find_interesting_seconds',
    'NarrationResult', 'load_genbank', 'load_stoich',
    # species_economy
    'audit_species_economy', 'EconomyAudit',
    # functional_coupling
    'find_coupled_species', 'hidden_state_similarity',
    # knockout_sweep
    'run_knockout_sweep', 'score_against_breuer',
    # spatial_decompose
    'decompose_counts_by_location',
]
