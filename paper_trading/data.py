"""Real (non-synthetic) market data loaders with on-disk caching.

Data sources reachable from this environment:
  - Crypto: Coin Metrics community data (daily PriceUSD) on GitHub.
  - Stocks: S&P-500 daily OHLCV sample dataset on GitHub.

Everything is cached under data_cache/ so the sandbox keeps working offline
after the first fetch. Nothing here is synthetic.
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# symbol -> (source_kind, url-or-key)
_CRYPTO = {
    "BTC": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
    "ETH": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv",
    "LTC": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/ltc.csv",
    "DOGE": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/doge.csv",
}
_STOCKS_BUNDLE = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"


def _download(url: str, timeout: int = 60) -> str:
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def load_prices(symbol: str) -> pd.DataFrame:
    """Return a DataFrame with columns ['date', 'price'] (daily, ascending).

    Raises ValueError for unknown symbols. Real data only.
    """
    symbol = symbol.upper()
    cache = _cache_path(f"{symbol}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"])
        return df

    if symbol in _CRYPTO:
        raw = _download(_CRYPTO[symbol])
        src = pd.read_csv(io.StringIO(raw), usecols=["time", "PriceUSD"]).dropna()
        df = pd.DataFrame(
            {"date": pd.to_datetime(src["time"]), "price": src["PriceUSD"].astype(float)}
        )
    else:
        # treat as a stock ticker in the bundled S&P-500 sample
        bundle_cache = _cache_path("_sp500_5yr.csv")
        if os.path.exists(bundle_cache):
            alldf = pd.read_csv(bundle_cache, parse_dates=["date"])
        else:
            raw = _download(_STOCKS_BUNDLE)
            alldf = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
            alldf.to_csv(bundle_cache, index=False)
        sub = alldf[alldf["Name"] == symbol]
        if sub.empty:
            avail = ", ".join(sorted(alldf["Name"].unique())[:15])
            raise ValueError(
                f"Unknown symbol '{symbol}'. Crypto: {list(_CRYPTO)}. "
                f"Stocks include e.g.: {avail} ... (S&P-500 sample, 2013-2018)."
            )
        df = pd.DataFrame(
            {"date": pd.to_datetime(sub["date"]), "price": sub["close"].astype(float)}
        )

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def available_symbols() -> dict:
    """List what can be loaded without guessing."""
    out = {"crypto": list(_CRYPTO)}
    bundle_cache = _cache_path("_sp500_5yr.csv")
    if os.path.exists(bundle_cache):
        alldf = pd.read_csv(bundle_cache)
        out["stocks_sample"] = sorted(alldf["Name"].unique().tolist())
    else:
        out["stocks_sample"] = "fetched on first stock request (S&P-500, 2013-2018)"
    return out


def returns(symbol: str) -> np.ndarray:
    p = load_prices(symbol)["price"].values.astype(float)
    return p[1:] / p[:-1] - 1.0
