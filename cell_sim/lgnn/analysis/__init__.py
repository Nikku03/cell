"""Analysis utilities for the count-dynamics surrogate."""

from cell_sim.lgnn.analysis.narrate_window import (
    narrate_window,
    format_narrative,
    find_interesting_seconds,
    NarrationResult,
)

__all__ = [
    'narrate_window',
    'format_narrative',
    'find_interesting_seconds',
    'NarrationResult',
]
