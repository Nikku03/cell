"""ArtificialAtom interatomic potential: a small SchNet-flavoured MLIP.

What this package is
--------------------

A compact, inspectable implementation of a *learned* interatomic potential
in the SchNet (Schuett et al., 2017) family. The model is trained to imitate
an analytic reference potential (Lennard-Jones with two element types) so
that we can verify, end-to-end, that:

  - the network's per-atom energy contributions sum to a correct total energy
  - forces obtained by autograd through the energy match the analytic ones
  - the network is invariant to global translations, rotations, and
    permutations of like atoms (the structural symmetries it must respect)
  - the predicted forces drive a random configuration toward the
    energy-minimising structure

What this is *not*
------------------

  - E(3)-equivariant (no vector outputs, no irreducible representations).
    This is the *invariant* simplification - same idea as SchNet, not MACE.
  - Trained on real QM data. We train on a Lennard-Jones target so we
    have analytic ground truth for every verification step.
  - A docking engine. The ingredients are here (atom encoding, message
    passing, force relaxation), but real docking needs an order of
    magnitude more on top - protein graphs, pose generators, scoring.

If you want a production MLIP, use MACE or NequIP. This module exists to
make the *mechanism* legible.
"""

from .atom import ArtificialAtom, Element
from .network import ArtificialAtomPotential
from .physics import LennardJonesReference

__all__ = [
    "ArtificialAtom",
    "Element",
    "ArtificialAtomPotential",
    "LennardJonesReference",
]
