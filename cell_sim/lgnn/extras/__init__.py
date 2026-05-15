"""Optional per-row feature builders from Tier 2/3 data files.

Each module here is INDEPENDENT - load failures in one don't affect
the others. All return torch tensors of shape (N, F_i) that get
appended to the M7 static feature stack.

  kinetic_priors.py     - per-reaction k_cat, K_M as features on F_* rows
  complex_constraints.py - one-hot complex membership per row
  medium_features.py     - external medium concentrations (broadcast global)
"""

from cell_sim.lgnn.extras.kinetic_priors import (
    build_kinetic_prior_features,
    KINETIC_PRIOR_COLS,
)
from cell_sim.lgnn.extras.complex_constraints import (
    build_complex_membership_features,
    COMPLEX_FEATURE_PREFIX,
)
from cell_sim.lgnn.extras.medium_features import (
    build_medium_features,
    MEDIUM_FEATURE_PREFIX,
)

__all__ = [
    'build_kinetic_prior_features', 'KINETIC_PRIOR_COLS',
    'build_complex_membership_features', 'COMPLEX_FEATURE_PREFIX',
    'build_medium_features', 'MEDIUM_FEATURE_PREFIX',
]
