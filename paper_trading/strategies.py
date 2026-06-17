"""Strategy framework.

A Strategy maps the price history *visible so far* to a target exposure in
[0, 1] (fraction of equity to hold in the asset). Keeping the contract to
"only data up to today" is what makes the backtester leak-free by construction.

Add your own by subclassing Strategy. That's the "evolve" part.
"""
from __future__ import annotations

import numpy as np


class Strategy:
    name = "base"

    def target_weight(self, prices: np.ndarray) -> float:
        """prices = all closes up to and INCLUDING today. Return weight in [0,1]."""
        raise NotImplementedError


class BuyAndHold(Strategy):
    name = "buyhold"

    def target_weight(self, prices):
        return 1.0


class TrendFollow(Strategy):
    """Long when price is above its N-day moving average, else flat.

    The textbook trend rule. N is fixed a priori (not tuned on the test data),
    which is the discipline that stops the backtest from lying to you.
    """
    name = "trend"

    def __init__(self, window=50):
        self.window = window

    def target_weight(self, prices):
        if len(prices) < self.window:
            return 0.0
        ma = prices[-self.window:].mean()
        return 1.0 if prices[-1] > ma else 0.0


class MeanRevert(Strategy):
    """Contrarian: buy when price is stretched below its band, trim when above.

    Included as a foil -- it tends to look great until a trend runs it over.
    A good lesson to feel rather than be told.
    """
    name = "meanrevert"

    def __init__(self, window=20, z=1.0):
        self.window = window
        self.z = z

    def target_weight(self, prices):
        if len(prices) < self.window:
            return 0.0
        w = prices[-self.window:]
        mu, sd = w.mean(), w.std()
        if sd == 0:
            return 0.5
        zscore = (prices[-1] - mu) / sd
        # more exposure when cheap (negative z), less when expensive
        return float(np.clip(0.5 - 0.5 * zscore / self.z, 0.0, 1.0))


REGISTRY = {s.name: s for s in [BuyAndHold, TrendFollow, MeanRevert]}


def make(name, **kw) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kw)
