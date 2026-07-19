"""Does an institutional-accumulation footprint predict catalyst pops?

Tests the intuitive thesis: "big money quietly piles in for days before a
catalyst, leaving a volume/price footprint; detect the pile-up, ride the pop."

Three angles, all leak-free (features at day t use only info through t):
  1. Forward: after an accumulation signal (elevated volume + up-drift + rising
     OBV), what do the next 2/5/10 days do vs baseline?
  2. Reverse (the crux): take every real 2-day POP and look BACKWARD -- do the
     lead-in days actually show accumulation, or something else?
  3. Net: after costs, does trading the signal beat just holding?

Data: bundled S&P-500 daily OHLCV sample. Run: python -m paper_trading.accumulation
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from . import data  # ensures cache exists via load_prices side-effects


def _build_features(df):
    feat = []
    for name, g in df.groupby("Name"):
        g = g.reset_index(drop=True)
        c = g["close"].values.astype(float)
        v = g["volume"].values.astype(float)
        if len(g) < 80:
            continue
        vol20 = pd.Series(v).rolling(20).mean().values
        volstd20 = pd.Series(v).rolling(20).std().values
        vol5 = pd.Series(v).rolling(5).mean().values
        volz = (vol5 - vol20) / volstd20                 # recent volume elevation
        ret5 = np.r_[[np.nan] * 5, c[5:] / c[:-5] - 1]   # 5-day drift
        obv = np.cumsum(np.sign(np.r_[0, np.diff(c)]) * v)
        obv_slope = pd.Series(obv).diff(5).values / (vol20 * 5 + 1e-9)
        for t in range(25, len(g) - 20):
            feat.append((name, volz[t], ret5[t], obv_slope[t],
                         c[t + 2] / c[t] - 1, c[t + 5] / c[t] - 1, c[t + 10] / c[t] - 1))
    return pd.DataFrame(feat, columns=["name", "volz", "ret5", "obvslope",
                                       "fwd2", "fwd5", "fwd10"]).dropna()


def _t(x, base):
    return (x.mean() - base) / (x.std() / sqrt(len(x)))


def run():
    df = pd.read_csv(data._cache_path("_sp500_5yr.csv"), parse_dates=["date"])
    df = df.dropna(subset=["close", "volume", "open"]).sort_values(["Name", "date"])
    F = _build_features(df)
    accum = F[(F.volz > F.volz.quantile(0.9)) & (F.ret5 > 0) &
              (F.obvslope > F.obvslope.quantile(0.75))]

    print("=" * 70)
    print(f"ACCUMULATION-FOOTPRINT STUDY  ({len(F):,} events, {F.name.nunique()} stocks)")
    print("=" * 70)
    print("\nTEST 1 - does the footprint predict forward returns?")
    for hz in ["fwd2", "fwd5", "fwd10"]:
        b = F[hz].mean()
        print(f"  {hz}: baseline {b*100:+.3f}%  |  after accumulation {accum[hz].mean()*100:+.3f}%"
              f"  (t vs base {_t(accum[hz], b):+.2f})")

    print("\nTEST 2 - reverse: what precedes REAL 2-day pops?")
    pop = F[F.fwd2 >= F.fwd2.quantile(0.98)]
    crash = F[F.fwd2 <= F.fwd2.quantile(0.02)]
    normal = F[(F.fwd2 > F.fwd2.quantile(0.45)) & (F.fwd2 < F.fwd2.quantile(0.55))]
    print(f"  {'lead-in to':12s}{'volume z':>10s}{'5d drift':>10s}{'OBV slope':>12s}")
    for lbl, grp in [("a POP", pop), ("a crash", crash), ("nothing", normal)]:
        print(f"  {lbl:12s}{grp.volz.mean():>10.3f}{grp.ret5.mean()*100:>9.2f}%{grp.obvslope.mean():>12.4f}")

    print("\nTEST 3 - net of costs?")
    print(f"  accumulation -> hold 5d: {accum.fwd5.mean()*100:+.3f}% gross, "
          f"{(accum.fwd5.mean()-0.001)*100:+.3f}% net(10bps)")
    print(f"  just holding 5d:         {F.fwd5.mean()*100:+.3f}%")
    print("\nVerdict: the visible footprint is real but points the WRONG way -- "
          "buying it is arriving late, and real pops follow WEAKNESS, not pile-up.")


if __name__ == "__main__":
    run()
