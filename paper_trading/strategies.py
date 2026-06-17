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


class AbsMomentum(Strategy):
    """Absolute (time-series) momentum: long if trailing N-day return > 0.

    Antonacci's dual-momentum building block. Fixed lookback, no tuning.
    """
    name = "absmom"

    def __init__(self, window=126):  # ~6 trading months
        self.window = window

    def target_weight(self, prices):
        if len(prices) <= self.window:
            return 0.0
        return 1.0 if prices[-1] / prices[-self.window - 1] - 1.0 > 0 else 0.0


class Breakout(Strategy):
    """Donchian channel breakout (the Turtle rule): go long on a new `entry`-day
    high, exit to flat on an `exit`-day low. The position is a pure function of
    history -- recomputed by replaying the channel forward -- so it stays
    leak-free and stateless.
    """
    name = "breakout"

    def __init__(self, entry=55, exit=20):
        self.entry = entry
        self.exit = exit

    def target_weight(self, prices):
        if len(prices) < self.entry + 1:
            return 0.0
        state = 0.0
        for i in range(self.entry, len(prices)):
            hi = prices[i - self.entry:i].max()
            lo = prices[i - self.exit:i].min()
            if prices[i] >= hi:
                state = 1.0
            elif prices[i] <= lo:
                state = 0.0
        return state


class VolTargetTrend(Strategy):
    """Trend filter, but size to a target volatility rather than going all-in.
    A documented Sharpe improver. Exposure capped at 1.0 (no leverage here).
    """
    name = "voltrend"

    def __init__(self, window=50, vol_window=20, target_vol=0.5):
        self.window = window
        self.vol_window = vol_window
        self.target_vol = target_vol  # annualized

    def target_weight(self, prices):
        if len(prices) < max(self.window, self.vol_window) + 1:
            return 0.0
        if prices[-1] <= prices[-self.window:].mean():
            return 0.0
        w = prices[-self.vol_window - 1:]
        rets = w[1:] / w[:-1] - 1.0
        vol = rets.std() * np.sqrt(365)
        return 1.0 if vol <= 0 else float(np.clip(self.target_vol / vol, 0.0, 1.0))


class RegimeTrend(Strategy):
    """Trend-following, but only 'armed' when the market is cleanly trending.

    Regime gauge = Kaufman Efficiency Ratio over `er_window`: |net move| divided
    by total path length, in [0,1]. ~1 means a clean directional move; ~0 means
    choppy noise where trend rules get whipsawed. We go long only when price is
    above its MA AND efficiency clears `er_thresh`. All causal (trailing only).
    """
    name = "regime"

    def __init__(self, window=50, er_window=30, er_thresh=0.30):
        self.window = window
        self.er_window = er_window
        self.er_thresh = er_thresh

    def target_weight(self, prices):
        if len(prices) < max(self.window, self.er_window) + 1:
            return 0.0
        if prices[-1] <= prices[-self.window:].mean():
            return 0.0  # not in an uptrend -> flat regardless of regime
        seg = prices[-self.er_window - 1:]
        net = abs(seg[-1] - seg[0])
        path = np.abs(np.diff(seg)).sum()
        er = net / path if path > 0 else 0.0
        return 1.0 if er >= self.er_thresh else 0.0


class RegimeSwitch(Strategy):
    """Switch tools by volatility regime.

    When recent volatility is ABOVE the asset's longer-run volatility (a
    turbulent regime), use trend-following for its downside protection. When
    things are calm (recent vol below long-run), just hold -- calm markets tend
    to drift up and trend rules only cost whipsaw there. Causal throughout.
    """
    name = "regimeswitch"

    def __init__(self, window=50, vol_window=20, regime_window=200):
        self.window = window
        self.vol_window = vol_window
        self.regime_window = regime_window

    def target_weight(self, prices):
        if len(prices) < self.window + 1:
            return 0.0
        if len(prices) < self.regime_window + 1:
            return 1.0  # not enough history to judge regime -> hold
        w = prices[-self.regime_window - 1:]
        rets = w[1:] / w[:-1] - 1.0
        recent_vol = rets[-self.vol_window:].std()
        long_vol = rets.std()
        turbulent = recent_vol > long_vol
        if turbulent:
            return 1.0 if prices[-1] > prices[-self.window:].mean() else 0.0
        return 1.0  # calm regime -> hold


REGISTRY = {s.name: s for s in
            [BuyAndHold, TrendFollow, MeanRevert, AbsMomentum, Breakout,
             VolTargetTrend, RegimeTrend, RegimeSwitch]}


def make(name, **kw) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kw)
