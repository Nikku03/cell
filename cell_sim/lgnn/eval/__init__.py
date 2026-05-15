"""Evaluation utilities for the LGNN dynamics surrogates."""

from cell_sim.lgnn.eval.full_rollout import (
    RolloutResult,
    full_rollout_evaluation,
)

__all__ = ['RolloutResult', 'full_rollout_evaluation']
