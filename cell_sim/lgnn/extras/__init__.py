"""Optional per-row feature builders and integration prep.

Each module here is INDEPENDENT - load failures in one don't affect
the others. All return tensors or other objects that get wired into
M7 only when their corresponding config flag is set to True.

Active (wired into combined static features when flag is True):
  kinetic_priors.py        - per-reaction k_cat, K_M
  complex_constraints.py    - complex membership one-hots
  medium_features.py        - external medium summary
  regulatory_features.py    - protein-metabolite binding indicators
  ribosome_subunit.py       - 50S subunit assembly markers
  thermodynamics.py         - per-reaction Gibbs-derived log_K_eq + sign

Reference / analysis (not wired into training, used by analysis tools):
  alternate_sbml.py         - path to the iMB155 variant + describer
  wcm_naming.py             - WCM internal -> standard naming map
  sim_metadata.py           - sim_properties.pkl unpickler
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
from cell_sim.lgnn.extras.regulatory_features import (
    build_regulatory_features,
    REGULATORY_FEATURE_COLS,
)
from cell_sim.lgnn.extras.ribosome_subunit import (
    build_ribosome_subunit_features,
    RIBOSOME_SUBUNIT_COLS,
)
from cell_sim.lgnn.extras.thermodynamics import (
    build_thermodynamics_features,
    THERMO_FEATURE_COLS,
)
from cell_sim.lgnn.extras.alternate_sbml import (
    find_alternate_sbml_path,
    describe_alternate_sbml,
    ALTERNATE_SBML_FILENAME,
)
from cell_sim.lgnn.extras.wcm_naming import (
    load_wcm_naming_map,
    apply_name_map,
)
from cell_sim.lgnn.extras.sim_metadata import (
    load_sim_metadata,
    summarize_sim_metadata,
)

__all__ = [
    # active feature builders
    'build_kinetic_prior_features', 'KINETIC_PRIOR_COLS',
    'build_complex_membership_features', 'COMPLEX_FEATURE_PREFIX',
    'build_medium_features', 'MEDIUM_FEATURE_PREFIX',
    'build_regulatory_features', 'REGULATORY_FEATURE_COLS',
    'build_ribosome_subunit_features', 'RIBOSOME_SUBUNIT_COLS',
    'build_thermodynamics_features', 'THERMO_FEATURE_COLS',
    # reference / analysis
    'find_alternate_sbml_path', 'describe_alternate_sbml', 'ALTERNATE_SBML_FILENAME',
    'load_wcm_naming_map', 'apply_name_map',
    'load_sim_metadata', 'summarize_sim_metadata',
]
