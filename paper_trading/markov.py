"""Markov-regime timing, tested honestly.

The faithful version of the "Markov chain" idea: fit a 2-state Gaussian Hidden
Markov Model to returns, let it infer a hidden 'calm-bull' vs 'turbulent-bear'
state, and hold the asset only while the model thinks we're in the bull state.

Leak-free by construction:
  - The HMM is refit only on a ROLLING window of PAST returns (refit_every days).
  - The state inferred at day t uses returns up to day t only.
  - The resulting position is applied to day t+1's return, with costs.

Run: python -m paper_trading.markov
"""
from __future__ import annotations

import warnings
import logging

import numpy as np

from . import data

warnings.filterwarnings("ignore")  # hmmlearn convergence chatter
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

CRYPTO = {"BTC", "ETH", "LTC", "DOGE"}


def _metrics(equity, ppy):
    equity = np.asarray(equity, float)
    yrs = (len(equity) - 1) / ppy
    cagr = equity[-1] ** (1 / yrs) - 1 if yrs > 0 and equity[-1] > 0 else float("nan")
    rets = np.diff(equity) / equity[:-1]
    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    maxdd = ((equity - peak) / peak).min()
    return cagr, sharpe, maxdd


def hmm_weights(prices, window=750, refit_every=63, min_obs=300, seed=0):
    """Return a leak-free daily weight array in {0,1} from a 2-state HMM."""
    from hmmlearn.hmm import GaussianHMM

    logret = np.diff(np.log(prices))
    n = len(prices)
    w = np.zeros(n)
    model = None
    bull = None
    last_fit = -10 ** 9
    for t in range(min_obs, n):
        # returns strictly before today's close decision: logret[:t] aligns to prices[1..t]
        hist = logret[max(0, t - window):t]
        if model is None or (t - last_fit) >= refit_every:
            try:
                m = GaussianHMM(n_components=2, covariance_type="diag",
                                n_iter=25, random_state=seed)
                m.fit(hist.reshape(-1, 1))
                model = m
                bull = int(np.argmax(m.means_.ravel()))
                last_fit = t
            except Exception:
                pass
        if model is not None:
            try:
                state = model.predict(hist.reshape(-1, 1))[-1]
                w[t] = 1.0 if state == bull else 0.0
            except Exception:
                w[t] = 0.0
    return w


def backtest_weights(prices, w, cost=0.0015):
    eq = [1.0]
    w_prev = 0.0
    for t in range(len(prices) - 1):
        r = prices[t + 1] / prices[t] - 1.0
        eq.append(eq[-1] * (1 + w[t] * r - abs(w[t] - w_prev) * cost))
        w_prev = w[t]
    return eq


def run(symbols=("BTC", "ETH", "AAPL", "MSFT", "AMZN")):
    print("=" * 74)
    print("HIDDEN MARKOV REGIME TIMING vs buy & hold (out-of-sample, costs, no leaks)")
    print("2-state Gaussian HMM on returns; hold only in the inferred bull state.")
    print("=" * 74)
    print(f"{'asset':6s}{'HMM CAGR':>11s}{'HMM Shp':>9s}{'HMM DD':>9s}"
          f"{'B&H CAGR':>11s}{'B&H Shp':>9s}{'verdict':>10s}")
    hmm_sh, bh_sh = [], []
    for sym in symbols:
        prices = data.load_prices(sym)["price"].values.astype(float)
        ppy = 365 if sym.upper() in CRYPTO else 252
        w = hmm_weights(prices)
        eq = backtest_weights(prices, w)
        bh = np.cumprod(np.r_[1.0, prices[1:] / prices[:-1]])
        c, s, dd = _metrics(eq, ppy)
        bc, bs, bdd = _metrics(bh, ppy)
        hmm_sh.append(s); bh_sh.append(bs)
        verdict = "BEATS" if s > bs else "loses"
        print(f"{sym:6s}{c*100:10.1f}%{s:9.2f}{dd*100:8.1f}%"
              f"{bc*100:10.1f}%{bs:9.2f}{verdict:>10s}")
    print("-" * 74)
    print(f"mean Sharpe: HMM {np.mean(hmm_sh):.2f}  vs  buy & hold {np.mean(bh_sh):.2f}")


if __name__ == "__main__":
    run()
