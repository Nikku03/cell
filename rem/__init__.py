"""REM -- exact inference and optimization over structured problems.

    cost = d ** treewidth        d = states per variable

Entanglement across a cut, bond dimension, edges crossing the cut and treewidth are the
same number. When it is small REM is exact and fast. When the dependency graph is an
expander (treewidth proportional to n) nothing is efficient, and that is a property of the
problem rather than a limit of this implementation.

REM is NOT a quantum computer. It cannot factor RSA or break strong cryptography.
"""
__version__ = "0.1.0"

from rem import factorgraph, circulant          # noqa: F401
from rem.factorgraph import FactorGraph, Factor  # noqa: F401

__all__ = ["FactorGraph", "Factor", "factorgraph", "circulant", "__version__"]
