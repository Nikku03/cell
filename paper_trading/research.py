"""Autonomous research harness for the paper-trading sandbox.

Two honest experiments:

1. ROBUSTNESS MATRIX -- run each fixed (a-priori-parameter) strategy across the
   whole cross-section of available assets, out-of-sample, and count how often
   it actually beats buy & hold on Sharpe. A real edge shows up on MOST assets,
   not one cherry-picked symbol.

2. THE OPTIMIZATION TRAP -- pick the "best" trend window by fitting the first
   half of an asset's history, then judge that choice on the held-out second
   half. This is the single most important lesson in quant trading: in-sample
   greatness rarely survives contact with out-of-sample reality.

Run: python -m paper_trading.research
"""
from __future__ import annotations

import numpy as np

from . import data, engine, strategies

CRYPTO = ["BTC", "ETH", "LTC", "DOGE"]
STOCKS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "KO", "GE", "WMT"]
ASSETS = CRYPTO + STOCKS


def robustness_matrix(strategy_names, oos_split=0.5):
    print("=" * 78)
    print(f"EXPERIMENT 1 - ROBUSTNESS MATRIX (out-of-sample tail, "
          f"{int((1-oos_split)*100)}% of history)")
    print("Cell = strategy Sharpe - buy&hold Sharpe. Positive (+) means it beat the market.")
    print("=" * 78)
    header = f"{'asset':6s}" + "".join(f"{n:>11s}" for n in strategy_names)
    print(header)
    wins = {n: 0 for n in strategy_names}
    total = {n: 0 for n in strategy_names}
    for sym in ASSETS:
        try:
            row = f"{sym:6s}"
            for n in strategy_names:
                res = engine.backtest(n, sym, oos_split=oos_split)
                d = res["strategy_metrics"]["sharpe"] - res["benchmark_metrics"]["sharpe"]
                if np.isfinite(d):
                    total[n] += 1
                    if d > 0:
                        wins[n] += 1
                    mark = "+" if d > 0 else " "
                    row += f"{mark}{d:>9.2f} "
                else:
                    row += f"{'n/a':>11s}"
            print(row)
        except Exception as e:
            print(f"{sym:6s} (skipped: {e})")
    print("-" * 78)
    print("BEAT-THE-MARKET RATE (how often each strategy won on Sharpe, OOS):")
    for n in strategy_names:
        if total[n]:
            print(f"  {n:10s} {wins[n]}/{total[n]} assets  ({wins[n]/total[n]*100:.0f}%)")
    return wins, total


def optimization_trap(symbol="BTC", windows=range(10, 210, 10)):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 2 - THE OPTIMIZATION TRAP (trend window on {symbol})")
    print("Pick the best window on the FIRST half, then see how it does on the SECOND.")
    print("=" * 78)
    prices = data.load_prices(symbol)["price"].values.astype(float)
    n = len(prices)
    mid = n // 2
    ppy = 365 if symbol.upper() in CRYPTO else 252

    def sharpe_on(prices_slice, window):
        strat = strategies.TrendFollow(window=window)
        eq = [1.0]
        w_prev = 0.0
        for t in range(window, len(prices_slice) - 1):
            w = strat.target_weight(prices_slice[: t + 1])
            r = prices_slice[t + 1] / prices_slice[t] - 1.0
            eq.append(eq[-1] * (1 + w * r - abs(w - w_prev) * 0.0015))
            w_prev = w
        rets = np.diff(eq) / np.array(eq[:-1])
        return (rets.mean() / rets.std() * np.sqrt(ppy)) if rets.std() > 0 else np.nan

    in_sample = {w: sharpe_on(prices[:mid], w) for w in windows}
    best_w = max(in_sample, key=lambda w: (in_sample[w] if np.isfinite(in_sample[w]) else -9))
    print(f"In-sample (first half) best window = {best_w}  "
          f"(Sharpe {in_sample[best_w]:.2f})")

    # how that 'optimized' choice does out-of-sample vs a few fixed alternatives
    oos_best = sharpe_on(prices[mid:], best_w)
    oos_default = sharpe_on(prices[mid:], 50)
    # what window would have actually been best OOS (hindsight, for contrast)
    oos_all = {w: sharpe_on(prices[mid:], w) for w in windows}
    truly_best_w = max(oos_all, key=lambda w: (oos_all[w] if np.isfinite(oos_all[w]) else -9))

    print(f"Out-of-sample (second half) Sharpe of that 'optimized' window {best_w}: {oos_best:.2f}")
    print(f"Out-of-sample Sharpe of the untuned default window 50:          {oos_default:.2f}")
    print(f"(Hindsight: the window that was ACTUALLY best OOS was {truly_best_w} "
          f"at Sharpe {oos_all[truly_best_w]:.2f} -- unknowable in advance.)")
    if np.isfinite(oos_best) and np.isfinite(oos_default):
        verdict = ("Optimizing helped." if oos_best > oos_default + 0.05
                   else "Optimizing did NOT beat just using the textbook default -- the trap.")
        print(f"-> {verdict}")


if __name__ == "__main__":
    robustness_matrix(["trend", "absmom", "breakout", "voltrend", "meanrevert"])
    for s in ["BTC", "ETH", "AAPL"]:
        optimization_trap(s)
