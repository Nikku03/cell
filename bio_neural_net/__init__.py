"""Bio-inspired neural network: spiking neurons + chemical synapses + Hebbian plasticity.

Unlike a "weighted-sum" artificial neural network, the building block here is a
biological neuron: it accumulates ionic current on a leaky membrane, emits a
discrete action potential when its membrane potential crosses threshold, then
becomes briefly refractory. Communication between neurons goes through chemical
synapses with conductance dynamics and reversal potentials - inhibition is not
"a negative weight", it is a conductance change that pulls the post-synaptic
membrane toward the chloride reversal potential.

A single neuron in isolation is just a leaky threshold detector. Intelligence
(working memory, winner-take-all, pattern recognition) is a property of how
many neurons are wired together. See `demo.py`.
"""

from .neuron import LIFNeuron, NeuronParams
from .synapse import Synapse, SynapseKind
from .network import NeuralCluster
from .plasticity import STDP

__all__ = [
    "LIFNeuron",
    "NeuronParams",
    "Synapse",
    "SynapseKind",
    "NeuralCluster",
    "STDP",
]
