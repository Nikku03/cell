"""cell_sim.layer_ml — Liquid Neural Net (LNN) for essentiality prediction
with multi-modal inputs (scalar / regulatory / kinetic / sequence), an
operon-graph passing layer, hyperbolic memory bank, and Elastic Weight
Consolidation (EWC) for continual learning across organisms.

Architecture inspired by enzyme_Software/liquid_nn_v2 (Nikku03's chemistry
LNN with ContextAwareTauPredictor + EdgeAwareMessagePassing) but tuned for
gene essentiality.

Importing this module imports nothing eagerly; pull individual classes via
their submodules to avoid import-cycle pain during development:

    from cell_sim.layer_ml.liquid_lnn import LiquidStep
    from cell_sim.layer_ml.hyperbolic_memory import HyperbolicMemoryBank
    from cell_sim.layer_ml.multimodal_encoder import MultimodalEncoder
    from cell_sim.layer_ml.continual_learning import EWCRegularizer
    from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig

Each submodule has a `__main__` self-test:

    python -m cell_sim.layer_ml.liquid_lnn
    python -m cell_sim.layer_ml.hyperbolic_memory
    python -m cell_sim.layer_ml.multimodal_encoder
    python -m cell_sim.layer_ml.continual_learning
    python -m cell_sim.layer_ml.essentiality_lnn
"""
