"""Brain-inspired scaling tricks for spiking neural networks.

Seven techniques that let a small network punch above its weight, each
implemented as a separate module with its own verification contract:

  population.py         tensor-vectorised LIF + sparse synapses (the foundation)
  sparse.py             top-K activation, controlled sparsity
  deep_recurrence.py    recurrent depth: same neurons, more timesteps -> more compute
  reservoir.py          echo-state / liquid-state-machine reservoir + linear readout
  modular.py            multi-module architecture with task routing
  memory.py             external vector store for retrieval-augmented compute
  distill.py            teacher -> student knowledge distillation
  dendritic.py          multi-compartment neurons (per-neuron sub-network)
  surrogate.py          surrogate-gradient BPTT training of the SNN itself

Each module:
  - Exposes a clean PyTorch interface
  - Has a verify_phase{N}.py companion with gates
  - Has a demo_phase{N}.py runnable end-to-end
  - Builds on prior phases (regression-tested)

The numerical primitives match bio_neural_net's LIF model bit-for-bit
at small N, so the vectorised version is a correctness-preserving
rewrite, not a different model.
"""

from .population import LIFPopulation, NeuronParams, SparseSynapses, SNNCluster

__all__ = ["LIFPopulation", "NeuronParams", "SparseSynapses", "SNNCluster"]
