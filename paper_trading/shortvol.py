"""The "20% a week" loophole, built and detonated on real data.

There IS a strategy that reliably prints 20% weekly returns: SELLING options
(a.k.a. selling volatility / theta harvesting). You collect premium every week.
Most weeks nothing bad happens and you keep it -> steady, huge-looking returns.

The catch is structural, not bad luck: you are selling insurance. The premiums
aren't profit, they're the counterparty paying you to carry a risk that has not
shown up YET. Your return stream has tiny positive numbers almost every week and
a rare, enormous negative one -- negative skew. Over a full cycle the rare week
pays back every premium you ever collected, with interest. It is picking up
pennies in front of a steamroller.

This module writes a weekly short strangle on the real BTC price path, prices it
with Black-Scholes on trailing realized vol, and shows the equity curve: a
seductive winning streak, then ruin. Run: python -m paper_trading.shortvol
"""
from __future__ import annotations

from math import log, sqrt, exp, erf

import numpy as np
import pandas as pd

from . import data


def _n(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _bs_put(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(K - S, 0)
    d1 = (log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    return K * exp(-r * T) * _n(-(d1 - sig * sqrt(T))) - S * _n(-d1)


def _bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0)
    d1 = (log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    return S * _n(d1) - K * exp(-r * T) * _n(d1 - sig * sqrt(T))


def run(start="2020-06-01", target_weekly=0.12, otm=0.08):
    """Sell a 1-week strangle each week, sized so a quiet week nets
    `target_weekly` of equity. Strikes are `otm` out of the money each side."""
    df = data.load_prices("BTC")
    df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    P = df["price"].values.astype(float)
    dates = df["date"].values
    logret = np.diff(np.log(P))

    equity = 100.0
    curve = [equity]
    weekly = []
    blown_week = None
    streak = best_streak = 0
    i = 70
    while i + 7 < len(P):
        S = P[i]
        sig = np.std(logret[i - 60:i]) * sqrt(365)
        T = 7 / 365
        Kp, Kc = S * (1 - otm), S * (1 + otm)
        prem = _bs_put(S, Kp, T, 0.0, sig) + _bs_call(S, Kc, T, 0.0, sig)
        if prem <= 0:
            i += 7
            continue
        N = (target_weekly * equity) / prem            # contracts sold
        S_exp = P[i + 7]
        pnl = N * prem - N * (max(Kp - S_exp, 0) + max(S_exp - Kc, 0))
        equity += pnl
        weekly.append(pnl / curve[-1])
        curve.append(equity)
        streak = streak + 1 if pnl > 0 else 0
        best_streak = max(best_streak, streak)
        if equity <= 0:
            blown_week = len(weekly)
            break
        i += 7

    curve = np.array(curve)
    weekly = np.array(weekly)
    print("=" * 68)
    print("THE '20% A WEEK' STRATEGY: weekly option selling on real BTC")
    print("=" * 68)
    print(f"Median weekly return:   {np.median(weekly) * 100:+.1f}%")
    print(f"Weeks green:            {100 * np.mean(weekly > 0):.0f}%")
    print(f"Longest winning streak: {best_streak} weeks")
    print(f"Peak equity from $100:  ${np.max(curve):,.0f}")
    print(f"Worst single week:      {np.min(weekly) * 100:+.0f}%")
    if blown_week:
        print(f">>> ACCOUNT TO $0 at week {blown_week}. "
              f"The one bad week erased every gain and the principal.")
    print("\nThe returns were never yours -- they were premiums held in trust "
          "for the crash.")


if __name__ == "__main__":
    run()
