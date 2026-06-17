"""Backtest engine and live-paper auto-runner.

The backtest is deliberately leak-free: at each day the strategy sees only the
prices up to that day, the chosen weight is applied to the NEXT day's return,
and trading incurs cost on turnover. It always reports the buy & hold benchmark
next to the strategy, because "beat the benchmark" is the only result that
counts.
"""
from __future__ import annotations

import numpy as np

from . import data, strategies
from .broker import COMMISSION, SLIPPAGE


def _metrics(equity, periods_per_year):
    equity = np.asarray(equity, float)
    yrs = (len(equity) - 1) / periods_per_year
    cagr = equity[-1] ** (1 / yrs) - 1 if yrs > 0 and equity[-1] > 0 else float("nan")
    rets = np.diff(equity) / equity[:-1]
    sharpe = (rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    maxdd = ((equity - peak) / peak).min()
    return {"final_x": equity[-1], "cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def backtest(strategy_name, symbol, oos_split=0.0, cost=COMMISSION + SLIPPAGE, **kw):
    """Run a leak-free backtest.

    oos_split: fraction of history reserved as out-of-sample. The strategies
    here don't fit parameters, so the whole sample is honest; the split is for
    when you start tuning -- tune on the first part, judge on the held-out tail.
    Returns dict with strategy + benchmark metrics.
    """
    strat = strategies.make(strategy_name, **kw)
    prices = data.load_prices(symbol)["price"].values.astype(float)
    n = len(prices)
    start = max(int(n * oos_split), 1)

    strat_eq, bench_eq = [1.0], [1.0]
    w_prev = 0.0
    for t in range(start, n - 1):
        w = float(np.clip(strat.target_weight(prices[: t + 1]), 0.0, 1.0))
        r_next = prices[t + 1] / prices[t] - 1.0
        turn = abs(w - w_prev)
        strat_eq.append(strat_eq[-1] * (1 + w * r_next - turn * cost))
        bench_eq.append(bench_eq[-1] * (1 + r_next))
        w_prev = w

    ppy = 365 if symbol.upper() in ("BTC", "ETH", "LTC", "DOGE") else 252
    return {
        "symbol": symbol.upper(),
        "strategy": strat.name,
        "days": n - start,
        "oos_split": oos_split,
        "strategy_metrics": _metrics(strat_eq, ppy),
        "benchmark_metrics": _metrics(bench_eq, ppy),
    }


def auto_run(broker, strategy_name, symbol, days=None, rebalance_every=1, **kw):
    """Drive the live paper account with a strategy, advancing the market.

    Trades the real paper broker (with its costs) so the resulting account is a
    genuine track record you can inspect afterwards.
    """
    strat = strategies.make(strategy_name, **kw)
    symbol = symbol.upper()
    series = broker.market.frames[symbol]
    steps = 0
    while not broker.market.at_end() and (days is None or steps < days):
        visible = series.loc[: broker.market.date].values.astype(float)
        target_w = float(np.clip(strat.target_weight(visible), 0.0, 1.0))
        if steps % rebalance_every == 0:
            _rebalance_to(broker, symbol, target_w)
        broker.tick(1)
        steps += 1
    broker.save()
    return steps


def _rebalance_to(broker, symbol, target_w):
    equity = broker.equity()
    px = broker.market.price(symbol)
    if px <= 0 or np.isnan(px):
        return
    target_value = target_w * equity
    current_value = broker.position_value(symbol)
    delta_value = target_value - current_value
    qty = delta_value / px
    try:
        if qty > 1e-9:
            # cap buy by available cash
            max_qty = broker.account.cash / (px * (1 + 0.01)) - 1e-9
            broker.buy(symbol, min(qty, max(max_qty, 0.0)))
        elif qty < -1e-9:
            broker.sell(symbol, min(-qty, broker.account.positions.get(symbol, 0.0)))
    except ValueError:
        pass  # skip infeasible rebalance rather than crash the run
