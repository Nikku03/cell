#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_finance_emulator.py  —  Phase 1 finance prototype
========================================================

A Liquid Graph Neural Network (LGNN) with closed-form continuous-time (CfC)
cells, finance-specific equation cores, a multi-lens pattern-discovery suite,
and two-tier rule enforcement — adapted from the JCVI-Syn3A whole-cell
emulator (`colab_cell_emulator.py`, branch claude/bio-inspired-neural-network).

WHAT THIS IS
------------
A *minimum-viable backtest* to test whether the biology architecture has any
edge in equities.  We predict next-day cross-sectional stock returns on an
S&P-500 subset, build a cost-aware long/short portfolio, and judge it against
a hard kill-criterion.  If the edge isn't there, we say so and stop.

ARCHITECTURE MAP  (biology  ->  finance)
----------------------------------------
  DynamicsModel (LGNN+CfC)      KEPT AS-IS              backbone unchanged
  _CfCGraphLayer                KEPT AS-IS              Liquid CfC message passing
  MetabolismCore (bi-bi)    ->  BetaCAPMCore            r_i = a_i + b_i * r_mkt
  CentralDogmaCore          ->  MomentumCore +          12-1 cross-sec momentum
                                MeanReversionCore        short-horizon reversal
  VolumeCore                ->  DROPPED                 no clean analog
  PINNHead (mass balance)   ->  DROPPED (Phase 2:       portfolio-balance constraint)
  dG clamps                 ->  risk-limit projection   pos 5%, sector 25%, 2:1
  Gaussian StochasticHead   ->  Student-t head (df=4)   fat tails
  knockout augmentation     ->  halt augmentation       zero a ticker -> robustness
  6 lenses                  ->  ported + retuned
  (new)                     ->  lens_lead_lag_multi, lens_correlation_regime,
                                lens_mean_reversion
  Tier-1 RuleSet (hard)     ->  cointegration / vol-band projection
  Tier-2 Hypotheses (soft)  ->  aux loss on pair / momentum constraints

EVAL: Sharpe (net of 5 bps/side), Sortino, max drawdown, information
coefficient, hit rate, tail returns — NOT R^2.

KILL CRITERION (Phase 1):  Sharpe > 0.5 net of costs AND max DD < 25% on the
out-of-sample test year.  Sharpe < 0.3  =>  kill the project.

DATA
----
Primary path is yfinance (works in Colab; daily adjusted OHLCV).  When yfinance
is unavailable (e.g. a sandbox whose network blocks Yahoo), we fall back to a
real, committed GitHub dataset of S&P-500 daily bars (CNuge/kaggle-code,
2013-02..2018-02) so the pipeline runs end-to-end on REAL prices.  A synthetic
factor-model generator is the last resort (clearly flagged; NOT a valid
kill-criterion test).

Run:   python colab_finance_emulator.py
Smoke: SMOKE=1 python colab_finance_emulator.py     # tiny/fast sanity run
"""

from __future__ import annotations
import os, sys, io, math, time, zipfile, urllib.request, warnings
from contextlib import nullcontext
import numpy as np

warnings.filterwarnings("ignore")

# torch is the model; everything else (data, metrics) may use numpy/pandas/scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

try:
    import pandas as pd
except Exception:                                    # pragma: no cover
    pd = None

# ============================================================================
#  CONFIG
# ============================================================================
SMOKE = os.environ.get("SMOKE", "0") == "1"

SEED                = 0
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# ---- universe & dates ------------------------------------------------------
# Default universe: ~80 large, liquid S&P-500 names spanning all 11 GICS
# sectors plus thematic supply-chain clusters.  Override via FINANCE_TICKERS.
UNIVERSE = [
    # tech / semis (Apple supply chain + mega-cap)
    "AAPL","MSFT","GOOGL","AMZN","FB","NVDA","INTC","QCOM","MU","AVGO",
    "TXN","CSCO","ORCL","IBM",
    # energy (oil majors + services + refiners)
    "XOM","CVX","COP","SLB","HAL","OXY","VLO","MPC","PSX",
    # financials (banks + brokers)
    "JPM","BAC","C","WFC","GS","MS","USB","PNC","AXP","BLK",
    # consumer discretionary / retail
    "WMT","TGT","COST","HD","LOW","F","GM",
    # airlines
    "DAL","UAL","AAL","LUV",
    # health care / pharma
    "JNJ","PFE","MRK","UNH","ABBV","ABT","TMO","LLY","BMY","AMGN",
    # consumer staples
    "PG","KO","PEP","MO","CL","MDLZ",
    # communication services / media
    "DIS","NFLX","CMCSA","T","VZ",
    # industrials
    "BA","CAT","GE","HON","MMM","UPS","FDX","LMT",
    # utilities + REITs
    "NEE","DUK","SO","SPG","AMT","PLD",
]
_env_t = os.environ.get("FINANCE_TICKERS")
if _env_t:
    UNIVERSE = [t.strip().upper() for t in _env_t.split(",") if t.strip()]

# Spec target window (used when yfinance is reachable, e.g. in Colab):
YF_START, YF_END = "2019-01-01", "2024-12-31"
# Train/Val/Test are allocated by calendar fraction over whatever real data we
# actually obtain (4y / 1y / 1y in the spec).  See split_by_fraction().
SPLIT_FRACS = (0.66, 0.17, 0.17)             # train, val, test

# GICS sector for each ticker (embedded so we never depend on a flaky lookup;
# matches 2013-2018 membership).  Used for the type-embedding + sector edges.
SECTOR_MAP = {
    "AAPL":"InfoTech","MSFT":"InfoTech","GOOGL":"CommSvc","AMZN":"ConsDisc",
    "FB":"CommSvc","NVDA":"InfoTech","INTC":"InfoTech","QCOM":"InfoTech",
    "MU":"InfoTech","AVGO":"InfoTech","TXN":"InfoTech","CSCO":"InfoTech",
    "ORCL":"InfoTech","IBM":"InfoTech",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","HAL":"Energy",
    "OXY":"Energy","VLO":"Energy","MPC":"Energy","PSX":"Energy",
    "JPM":"Financials","BAC":"Financials","C":"Financials","WFC":"Financials",
    "GS":"Financials","MS":"Financials","USB":"Financials","PNC":"Financials",
    "AXP":"Financials","BLK":"Financials",
    "WMT":"ConsStaples","TGT":"ConsDisc","COST":"ConsStaples","HD":"ConsDisc",
    "LOW":"ConsDisc","F":"ConsDisc","GM":"ConsDisc",
    "DAL":"Industrials","UAL":"Industrials","AAL":"Industrials","LUV":"Industrials",
    "JNJ":"HealthCare","PFE":"HealthCare","MRK":"HealthCare","UNH":"HealthCare",
    "ABBV":"HealthCare","ABT":"HealthCare","TMO":"HealthCare","LLY":"HealthCare",
    "BMY":"HealthCare","AMGN":"HealthCare",
    "PG":"ConsStaples","KO":"ConsStaples","PEP":"ConsStaples","MO":"ConsStaples",
    "CL":"ConsStaples","MDLZ":"ConsStaples",
    "DIS":"CommSvc","NFLX":"CommSvc","CMCSA":"CommSvc","T":"CommSvc","VZ":"CommSvc",
    "BA":"Industrials","CAT":"Industrials","GE":"Industrials","HON":"Industrials",
    "MMM":"Industrials","UPS":"Industrials","FDX":"Industrials","LMT":"Industrials",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "SPG":"RealEstate","AMT":"RealEstate","PLD":"RealEstate",
}
SECTORS = ["InfoTech","Energy","Financials","ConsDisc","ConsStaples","CommSvc",
           "HealthCare","Industrials","Utilities","RealEstate"]
SECTOR_ID = {s: i for i, s in enumerate(SECTORS)}
N_SECTORS = len(SECTORS)

# Manual supply-chain / thematic clusters -> strong graph edges (intra-cluster).
SUPPLY_CHAIN_CLUSTERS = {
    "apple_semis": ["AAPL","QCOM","AVGO","MU","INTC","TXN","CSCO"],
    "oil":         ["XOM","CVX","COP","SLB","HAL","OXY","VLO","MPC","PSX"],
    "banks":       ["JPM","BAC","C","WFC","GS","MS","USB","PNC","AXP","BLK"],
    "airlines":    ["DAL","UAL","AAL","LUV"],
    "big_box":     ["WMT","TGT","COST","HD","LOW"],
    "autos":       ["F","GM"],
    "pharma":      ["JNJ","PFE","MRK","ABBV","ABT","BMY","AMGN","LLY","UNH","TMO"],
    "staples":     ["PG","KO","PEP","MO","CL","MDLZ"],
    "industrials": ["BA","CAT","GE","HON","MMM","UPS","FDX","LMT"],
    "telecom_media":["DIS","NFLX","CMCSA","T","VZ"],
    "utes":        ["NEE","DUK","SO"],
    "reits":       ["SPG","AMT","PLD"],
}

# ---- features --------------------------------------------------------------
AR_WINDOW   = 20            # last-N daily returns fed as the autoregressive node state
WARMUP      = 252           # trading days of history required before a row is usable
RET_SCALE   = 100.0         # predict returns in "percent" units for numeric conditioning

# ---- graph -----------------------------------------------------------------
CORR_K      = 6             # top-k correlation neighbours per node (train-period corr)
CORR_MIN    = 0.30          # minimum |corr| to draw a correlation edge

# ---- lenses ----------------------------------------------------------------
PAIRWISE_TOP_K       = 50
PAIRWISE_THRESHOLD   = 0.85          # |r| for a candidate pair (spec)
CONSERVATION_K       = 6             # smallest singular directions = mean-reverting baskets
PERIODICITY_TOP_K    = 12
LEADLAG_MAX          = 5
LEADLAG_HORIZONS     = (1, 5, 30)    # lens_lead_lag_multi
FDR_ALPHA            = 0.10          # Benjamini-Hochberg false-discovery rate
RULE_COMPLIANCE      = 0.99          # held-out compliance for a Tier-1 monotone/bound rule
MONO_EPS             = 1e-5
VOL_BAND_K           = 6.0           # Tier-1 return clamp: |pred| <= K * per-name vol

# ---- model -----------------------------------------------------------------
LGNN_HIDDEN     = 48
LGNN_N_LAYERS   = 2
LGNN_CFC_TAU_MIN= 0.1
N_TYPE_EMBED    = 6
STUDENT_T_DF    = 4.0                # Student-t degrees of freedom (fat tails)

# ---- training --------------------------------------------------------------
STEPS           = 8 if SMOKE else 900
BATCH           = 16 if SMOKE else 64        # batch of trading days
K_MAX           = 3                          # max rollout length (short: daily returns ~ iid)
TBPTT_CHUNK     = 3
LR              = 2e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
LAMBDA_1STEP    = 1.0
LAMBDA_NLL      = 1.0
LAMBDA_IC       = 1.0                         # cross-sectional rank-IC surrogate
LAMBDA_PNL      = 1.0                         # long/short factor-return surrogate
LAMBDA_HYP      = 0.01                        # Tier-2 soft hypothesis aux loss
LAMBDA_SIGMA_ANCHOR = 0.15                    # anchor log-sigma to empirical vol
LAMBDA_REFINE   = 0.15                        # v15.1 refinement pass (last 10%)
REFINE_START_FRAC = 0.9
USE_HALT_AUG    = True                        # finance analog of knockout augmentation
HALT_PROB       = 0.30

# ---- backtest --------------------------------------------------------------
COST_BPS        = 5.0          # per side, basis points (round-trip 10 bps)
SELECT_FRAC     = 0.20         # long top X%, short bottom X% (quintile; see note in build_weights)
MAX_POS         = 0.05         # max 5% gross in any one name
MAX_SECTOR      = 0.25         # max 25% gross in any one sector
GROSS_LEVERAGE  = 2.0          # gross long+short cap (2:1)
SIGMA_WEIGHT    = False        # equal-weight first; flip to size by 1/sigma
TRADING_DAYS    = 252

# kill criterion
KILL_SHARPE_GO  = 0.50
KILL_SHARPE_DIE = 0.30
KILL_MAXDD      = 0.25

OUT_DIR = os.environ.get("FINANCE_OUT", "outputs/finance")

np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


# ============================================================================
#  DATA LOADING   (yfinance  ->  GitHub real-data fallback  ->  synthetic)
# ============================================================================
CACHE_DIR = os.environ.get("FINANCE_CACHE", "finance_data")
CNUGE_ZIP = ("https://raw.githubusercontent.com/CNuge/kaggle-code/master/"
             "stock_data/individual_stocks_5yr.zip")
# Drop-in path for a user-supplied price file (e.g. Kaggle's sp500_stocks.csv,
# 2010-present, columns Date,Symbol,Adj Close,Close,High,Low,Open,Volume).
# Set FINANCE_PRICES_CSV, or just place the file at one of the auto-detected
# names below.  This is THE way to run the real 2019-2024 test offline: download
# sp500_stocks.csv from kaggle.com/datasets/andrewmvd/sp-500-stocks, commit it,
# and this loader consumes it directly (filtered to UNIVERSE + date window).
LOCAL_CSV_CANDIDATES = [
    os.environ.get("FINANCE_PRICES_CSV", ""),
    "finance_data/sp500_stocks.csv", "finance_data/prices.csv",
    "sp500_stocks.csv", "prices.csv",
]


def _try_local_csv(tickers, start, end):
    """Load a user-supplied LONG-format price CSV (Date,Symbol,<price>).

    Accepts the Kaggle `sp500_stocks.csv` schema directly (prefers 'Adj Close',
    falls back to 'Close').  Pivots to a (dates x tickers) panel filtered to the
    requested universe and [start,end] window.  Returns None if no file found."""
    if pd is None:
        return None
    path = next((p for p in LOCAL_CSV_CANDIDATES if p and os.path.exists(p)), None)
    if path is None:
        return None
    print(f"[data] local price file: {path}")
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date"); scol = cols.get("symbol", cols.get("ticker"))
    pcol = cols.get("adj close", cols.get("adjclose", cols.get("close")))
    if not (dcol and scol and pcol):
        print(f"[data] local CSV missing Date/Symbol/(Adj)Close cols: {list(df.columns)}")
        return None
    df = df[[dcol, scol, pcol]].copy()
    df[dcol] = pd.to_datetime(df[dcol])
    df = df[(df[dcol] >= pd.Timestamp(start)) & (df[dcol] <= pd.Timestamp(end))]
    df = df[df[scol].isin(set(tickers))]
    close = df.pivot_table(index=dcol, columns=scol, values=pcol).sort_index()
    close = close.dropna(axis=1, thresh=int(0.99 * len(close))).ffill().dropna(how="any")
    if close.shape[1] < 10 or close.shape[0] < WARMUP + 60:
        print(f"[data] local CSV too sparse after filter ({close.shape}); falling back")
        return None
    return close


def _try_yfinance(tickers, start, end):
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        print(f"[data] yfinance: downloading {len(tickers)} tickers {start}..{end}")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
        # auto_adjust=True -> 'Close' is split+dividend adjusted
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        close = close.dropna(axis=1, how="all").ffill().dropna(axis=0, how="any")
        if close.shape[1] < 10 or close.shape[0] < WARMUP + 60:
            print("[data] yfinance returned too little data; falling back")
            return None
        return close
    except Exception as e:                            # pragma: no cover
        print(f"[data] yfinance failed ({e}); falling back")
        return None


def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _try_github_real(tickers):
    """Real S&P-500 daily bars (2013-02..2018-02) committed to GitHub.

    Split-adjusted close (NOT dividend-adjusted) — a documented Phase-1
    limitation.  Returns a (dates x tickers) close-price DataFrame.
    """
    if pd is None:
        return None
    # 1) honour a pre-extracted local dir (sandbox convenience)
    local = "/tmp/sp5yr/individual_stocks_5yr"
    base = None
    if os.path.isdir(local):
        base = local
    else:
        try:
            zp = _download(CNUGE_ZIP, os.path.join(CACHE_DIR, "individual_stocks_5yr.zip"))
            ext = os.path.join(CACHE_DIR, "extracted")
            if not os.path.isdir(os.path.join(ext, "individual_stocks_5yr")):
                with zipfile.ZipFile(zp) as z:
                    z.extractall(ext)
            base = os.path.join(ext, "individual_stocks_5yr")
        except Exception as e:
            print(f"[data] github fallback download failed ({e})")
            return None
    print(f"[data] github real-data fallback: {base}")
    series = {}
    for t in tickers:
        fp = os.path.join(base, f"{t}_data.csv")
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp, parse_dates=["date"]).set_index("date")["close"]
        series[t] = d
    if len(series) < 10:
        return None
    close = pd.DataFrame(series).sort_index()
    close = close.dropna(axis=1, thresh=int(0.99 * len(close))).ffill().dropna(how="any")
    return close


def _synthetic(tickers, n_days=1500):
    """Last-resort synthetic prices: 3-factor + sector blocks + fat tails.

    *** NOT a valid kill-criterion test *** — the generator has whatever
    structure we bake in.  Exists only to guarantee the pipeline runs.
    """
    print("[data] *** SYNTHETIC data (NOT a valid backtest) ***")
    rng = np.random.default_rng(SEED)
    n = len(tickers)
    secid = np.array([SECTOR_ID.get(SECTOR_MAP.get(t, "InfoTech"), 0) for t in tickers])
    beta = rng.uniform(0.6, 1.5, n)
    mkt = rng.standard_t(4, n_days) * 0.01
    sec_f = rng.standard_t(5, (n_days, N_SECTORS)) * 0.006
    idio = rng.standard_t(4, (n_days, n)) * 0.012
    rets = (beta[None, :] * mkt[:, None]
            + sec_f[:, secid]
            + idio
            + 0.05 * np.r_[np.zeros((1, n)), idio[:-1]])     # weak reversal
    px = 50 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.bdate_range("2015-01-01", periods=n_days) if pd is not None else np.arange(n_days)
    return pd.DataFrame(px, index=dates, columns=tickers) if pd is not None else px


def load_prices(tickers, start=YF_START, end=YF_END):
    """Return (close_df, source_tag).  close_df: DataFrame[dates x tickers].

    Order: user-supplied CSV (real, any window) -> yfinance (Colab, 2019-2024)
    -> committed GitHub real data (2013-2018 proxy) -> synthetic (flagged)."""
    close = _try_local_csv(tickers, start, end)
    if close is not None:
        return close, "local-csv(user-supplied, real)"
    close = _try_yfinance(tickers, start, end)
    if close is not None:
        return close, "yfinance(adjusted)"
    close = _try_github_real(tickers)
    if close is not None:
        return close, "github-real(CNuge 2013-2018, split-adj)"
    return _synthetic(tickers), "SYNTHETIC"


# ---- feature engineering (no look-ahead) -----------------------------------
def _rsi(prices, window=14):
    delta = np.diff(prices, axis=0, prepend=prices[:1])
    up = np.clip(delta, 0, None)
    dn = np.clip(-delta, 0, None)
    # simple rolling mean RSI
    def roll(x):
        c = np.cumsum(x, axis=0)
        c[window:] = c[window:] - c[:-window]
        out = c / window
        out[:window] = c[:window] / np.arange(1, window + 1)[:, None]
        return out
    rs = roll(up) / (roll(dn) + 1e-9)
    return 100 - 100 / (1 + rs)


def build_dataset(close):
    """Build per-(day, stock) feature tensors + next-day return targets.

    Returns dict with:
      dates (T,), tickers (N,), rets (T,N) log-returns,
      feats (T,N,F) features computed from data <= t (NO look-ahead),
      target (T,N) = rets[t+1] (next-day),  feat_names list.
    Row t is usable for t in [WARMUP, T-2] (need t+1 for the target).
    """
    px = close.values.astype(np.float64)             # (T, N)
    T, N = px.shape
    logpx = np.log(px)
    rets = np.zeros_like(logpx)
    rets[1:] = logpx[1:] - logpx[:-1]                # r[t] realised at close t

    def roll_sum(x, w):
        c = np.cumsum(x, axis=0)
        out = np.empty_like(x)
        out[:w] = c[:w]
        out[w:] = c[w:] - c[:-w]
        return out

    def roll_std(x, w):
        s1 = roll_sum(x, w)
        s2 = roll_sum(x * x, w)
        n = np.minimum(np.arange(1, T + 1), w)[:, None]
        var = s2 / n - (s1 / n) ** 2
        return np.sqrt(np.clip(var, 1e-12, None))

    feats, names = [], []
    # autoregressive window: last AR_WINDOW returns (shifted so col 0 = r[t])
    for lag in range(AR_WINDOW):
        col = np.zeros_like(rets)
        if lag == 0:
            col = rets.copy()
        else:
            col[lag:] = rets[:-lag]
        feats.append(col); names.append(f"r_lag{lag}")
    # slow features
    mom_12_1 = roll_sum(rets, 252) - roll_sum(rets, 21)         # 12-1 momentum
    vol_20   = roll_std(rets, 20)
    vol_60   = roll_std(rets, 60)
    rsi_14   = (_rsi(px, 14) - 50.0) / 50.0
    dn = np.clip(-rets, 0, None)
    dvol_60  = roll_std(dn, 60)                                  # downside vol
    rev_5    = -roll_sum(rets, 5)                                # short reversal
    roll_max = np.maximum.accumulate(px, axis=0)                 # crude 52w-ish high
    rmax_252 = np.array([px[max(0, t - 251):t + 1].max(0) for t in range(T)])
    dist_high= (px - rmax_252) / (rmax_252 + 1e-9)
    for arr, nm in [(mom_12_1,"mom_12_1"),(vol_20,"vol_20"),(vol_60,"vol_60"),
                    (rsi_14,"rsi_14"),(dvol_60,"dvol_60"),(rev_5,"rev_5"),
                    (dist_high,"dist_52w_high")]:
        feats.append(arr); names.append(nm)

    feats = np.stack(feats, axis=-1).astype(np.float32)          # (T, N, F)
    target = np.zeros((T, N), np.float32)
    target[:-1] = rets[1:]                                       # next-day return
    return dict(dates=np.asarray(close.index), tickers=list(close.columns),
                rets=rets.astype(np.float32), feats=feats,
                target=target * 1.0, feat_names=names, n_ar=AR_WINDOW)


def split_by_fraction(T, fracs=SPLIT_FRACS, warmup=WARMUP):
    usable = np.arange(warmup, T - 1)                # need t+1 for target
    n = len(usable)
    a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return usable[:a], usable[a:b], usable[b:]


# ============================================================================
#  GRAPH CONSTRUCTION   (GICS sectors + rolling correlation + supply chain)
# ============================================================================
def build_full_graph(tickers, rets_train):
    """Build a directed, weighted graph over stocks from three sources.

    Mirrors the biology `build_full_graph`: union multiple edge sources, dedup,
    sum weights.  Correlation edges use TRAIN-period returns only (no leakage).
    Returns edge_index (2,E) long, edge_weight (E,) float, type_ids (N,) long.
    """
    N = len(tickers)
    tix = {t: i for i, t in enumerate(tickers)}
    edges = {}                                       # (src,dst) -> weight

    def add(i, j, w):
        if i == j:
            return
        edges[(i, j)] = max(edges.get((i, j), 0.0), w)
        edges[(j, i)] = max(edges.get((j, i), 0.0), w)

    # 1) GICS sector edges (same-sector coupling, modest weight)
    by_sec = {}
    for t in tickers:
        by_sec.setdefault(SECTOR_MAP.get(t, "InfoTech"), []).append(tix[t])
    n_sec = 0
    for members in by_sec.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                add(members[a], members[b], 0.4); n_sec += 1

    # 2) rolling-correlation edges (top-k peers by |corr| on train returns)
    R = rets_train - rets_train.mean(0, keepdims=True)
    sd = R.std(0) + 1e-9
    C = (R.T @ R) / R.shape[0] / np.outer(sd, sd)
    np.fill_diagonal(C, 0.0)
    n_corr = 0
    for i in range(N):
        order = np.argsort(-np.abs(C[i]))[:CORR_K]
        for j in order:
            if abs(C[i, j]) >= CORR_MIN:
                add(i, int(j), float(abs(C[i, j]))); n_corr += 1

    # 3) manual supply-chain / thematic clusters (strong weight)
    n_sc = 0
    for members in SUPPLY_CHAIN_CLUSTERS.values():
        idx = [tix[t] for t in members if t in tix]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], 0.8); n_sc += 1

    src = [s for (s, _) in edges]; dst = [d for (_, d) in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    ew = torch.tensor([edges[k] for k in edges], dtype=torch.float32)
    type_ids = [SECTOR_ID.get(SECTOR_MAP.get(t, "InfoTech"), 0) for t in tickers]
    print(f"[graph] nodes={N}  edges={ei.shape[1]} "
          f"(sector~{n_sec*2}, corr~{n_corr*2}, supply-chain~{n_sc*2}, deduped)")
    return ei, ew, torch.tensor(type_ids, dtype=torch.long)


# ============================================================================
#  LENSES  (6 ported + 3 new)   +   two-tier knowledge framework
# ============================================================================
def benjamini_hochberg(pvals, alpha=FDR_ALPHA):
    """Benjamini-Hochberg FDR.  Returns boolean mask of discoveries kept.

    Multiple-comparison correction (spec caution #5): with 9 lenses we will
    find patterns by chance; BH controls the expected false-discovery rate.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.where(passed)[0]
    keep = np.zeros(m, bool)
    if len(k):
        cutoff = p[order][k.max()]
        keep = p <= cutoff
    return keep


def _corr_pval(r, n):
    """Two-sided p-value for a Pearson correlation r with n samples."""
    if n < 4 or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else 1.0
    from scipy import stats
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


# ----- 6 ported lenses ------------------------------------------------------
def lens_monotone(rets_tr, rets_va, eps=MONO_EPS):
    """1D trend-regime lens (biology: per-species monotonicity).

    Per-stock fraction of up/down days; held-out compliance for a persistent
    drift.  In equities almost nothing is monotone -> mostly Tier-2.
    """
    up_tr = (rets_tr > eps).mean(0); up_va = (rets_va > eps).mean(0)
    emp_up   = set(np.where(up_tr   >= RULE_COMPLIANCE)[0].tolist())
    emp_down = set(np.where((1 - up_tr) >= RULE_COMPLIANCE)[0].tolist())
    return up_va, 1 - up_va, emp_up, emp_down


def lens_bounds(rets_tr, rets_va, vol):
    """1D bounded-range lens (biology: per-species bounds) -> vol bands.

    Tier-1 candidate band [-K*vol, +K*vol]; validated if all held-out daily
    returns fall inside.  These become the return-clamp projection.
    """
    lo = -VOL_BAND_K * vol; hi = VOL_BAND_K * vol
    ok = ((rets_va >= lo) & (rets_va <= hi)).all(0)
    return lo.astype(np.float32), hi.astype(np.float32), ok


def lens_pairwise(rets_tr, rets_va, tickers, top_k=PAIRWISE_TOP_K,
                  threshold=PAIRWISE_THRESHOLD):
    """2D pairwise-correlation lens (biology: pairwise on deltas) -> pairs trading.

    Top |r| pairs on train returns with |r|>threshold; report held-out r and a
    BH-corrected significance flag.  Returns list of dicts.
    """
    N = rets_tr.shape[1]
    Z = (rets_tr - rets_tr.mean(0)) / (rets_tr.std(0) + 1e-9)
    Zv = (rets_va - rets_va.mean(0)) / (rets_va.std(0) + 1e-9)
    C = (Z.T @ Z) / Z.shape[0]; np.fill_diagonal(C, 0.0)
    cand = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(C[i, j]) >= threshold:
                cand.append((i, j, float(C[i, j])))
    cand.sort(key=lambda x: -abs(x[2]))
    cand = cand[:top_k]
    pvals = [_corr_pval(c, rets_va.shape[0]) for *_, c in
             [(i, j, float((Zv[:, i] * Zv[:, j]).mean())) for i, j, _ in cand]]
    keep = benjamini_hochberg(pvals) if cand else np.zeros(0, bool)
    out = []
    for k, (i, j, r_tr) in enumerate(cand):
        r_va = float((Zv[:, i] * Zv[:, j]).mean())
        out.append(dict(i=i, j=j, ti=tickers[i], tj=tickers[j],
                        r_tr=r_tr, r_va=r_va, p=pvals[k], fdr_ok=bool(keep[k])))
    return out


def lens_conservation(rets_tr, rets_va, tickers, n_candidates=CONSERVATION_K):
    """Low-d SVD lens (biology: conservation laws) -> factor structure.

    LARGEST singular directions = dominant risk factors (market/sector).
    SMALLEST = low-variance linear combos = near-cointegrating / mean-reverting
    baskets (the finance analog of an approximately-conserved quantity).
    """
    Z = (rets_tr - rets_tr.mean(0)) / (rets_tr.std(0) + 1e-9)
    Zv = (rets_va - rets_va.mean(0)) / (rets_va.std(0) + 1e-9)
    try:
        _, s, Vh = np.linalg.svd(Z, full_matrices=False)
    except Exception:
        return dict(factors=[], baskets=[])
    var = s ** 2 / (s ** 2).sum()
    factors = []
    for idx in range(min(3, len(s))):
        v = Vh[idx]; top = np.argsort(-np.abs(v))[:4]
        factors.append(dict(var_explained=float(var[idx]),
                            top=[(tickers[t], float(v[t])) for t in top]))
    baskets = []
    for idx in np.argsort(s)[:n_candidates]:
        v = Vh[idx]; top = np.argsort(-np.abs(v))[:3]
        baskets.append(dict(std_train=float((Z @ v).std()),
                            std_val=float((Zv @ v).std()),
                            top=[(tickers[t], float(v[t])) for t in top]))
    return dict(factors=factors, baskets=baskets)


def lens_periodicity(rets_tr, dates_tr, tickers, top_k=PERIODICITY_TOP_K):
    """Time/FFT lens (biology: periodicity) -> calendar effects.

    FFT peak per stock + day-of-week / turn-of-month mean-return seasonality
    across the cross-section (BH-corrected).
    """
    x = rets_tr - rets_tr.mean(0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    power = (np.abs(fft) ** 2).mean(1)
    peak = int(power[1:].argmax() + 1) if len(power) > 1 else 0
    cal = {}
    if pd is not None and np.issubdtype(np.asarray(dates_tr).dtype, np.datetime64):
        d = pd.DatetimeIndex(dates_tr)
        xbar = rets_tr.mean(1)                        # equal-weight market
        from scipy import stats
        dow = {}
        for k, lab in enumerate(["Mon","Tue","Wed","Thu","Fri"]):
            m = d.dayofweek == k
            if m.sum() > 5:
                t, p = stats.ttest_1samp(xbar[m], 0.0)
                dow[lab] = (float(xbar[m].mean()), float(p))
        tom = d.day >= 25
        if tom.sum() > 5:
            t, p = stats.ttest_ind(xbar[tom], xbar[~tom], equal_var=False)
            cal["turn_of_month"] = (float(xbar[tom].mean() - xbar[~tom].mean()), float(p))
        cal["day_of_week"] = dow
    return dict(peak_bin=peak, calendar=cal)


def lens_lead_lag(rets_tr, tickers, max_lag=LEADLAG_MAX):
    """Chain lag lens (biology: gene-chain lag) -> single-horizon lead-lag.

    For sector aggregates, find the lag maximising cross-correlation (who leads
    whom within the cross-section).
    """
    secs = {}
    for i, t in enumerate(tickers):
        secs.setdefault(SECTOR_MAP.get(t, "InfoTech"), []).append(i)
    agg = {s: rets_tr[:, idx].mean(1) for s, idx in secs.items() if idx}
    names = list(agg); out = []
    for a in range(len(names)):
        for b in range(len(names)):
            if a == b:
                continue
            x = agg[names[a]] - agg[names[a]].mean()
            y = agg[names[b]] - agg[names[b]].mean()
            best_c, best_l = 0.0, 0
            for lag in range(1, max_lag + 1):
                c = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
                if abs(c) > abs(best_c):
                    best_c, best_l = c, lag
            if abs(best_c) > 0.15:
                out.append(dict(leader=names[a], follower=names[b],
                                lag=best_l, corr=best_c))
    out.sort(key=lambda d: -abs(d["corr"]))
    return out[:10]


# ----- 3 NEW finance lenses -------------------------------------------------
def lens_lead_lag_multi(rets_tr, tickers, horizons=LEADLAG_HORIZONS):
    """NEW: multi-horizon lead-lag.  Cross-correlate stock pairs at 1d/5d/30d.

    A leads B at horizon h if corr(r_A[t], r_B[t+h]) is large.  BH-corrected.
    """
    N = rets_tr.shape[1]; results = {}
    for h in horizons:
        if rets_tr.shape[0] <= h + 5:
            continue
        A = rets_tr[:-h]; B = rets_tr[h:]
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        B = (B - B.mean(0)) / (B.std(0) + 1e-9)
        C = (A.T @ B) / A.shape[0]                    # C[i,j] = corr(r_i[t], r_j[t+h])
        np.fill_diagonal(C, 0.0)
        pairs = []
        flat = np.argsort(-np.abs(C).ravel())[:30]
        for f in flat:
            i, j = int(f // N), int(f % N)
            r = float(C[i, j])
            pairs.append(dict(leader=tickers[i], follower=tickers[j],
                              corr=r, p=_corr_pval(r, A.shape[0])))
        keep = benjamini_hochberg([p["p"] for p in pairs])
        for k, pr in enumerate(pairs):
            pr["fdr_ok"] = bool(keep[k])
        results[h] = [p for p in pairs if p["fdr_ok"]][:8]
    return results


def lens_correlation_regime(rets_tr, dates_tr, window=42):
    """NEW: correlation-regime lens.  Rolling mean pairwise correlation, then a
    2-state split (high-corr 'risk-off' vs low-corr 'normal').

    Uses an HMM if hmmlearn is available, else a Gaussian-mixture / median
    threshold fallback.  Returns regime stats (mean corr per regime, frequency,
    persistence).
    """
    T = rets_tr.shape[0]
    if T < window + 10:
        return dict(available=False)
    avg_corr = np.full(T, np.nan)
    for t in range(window, T):
        w = rets_tr[t - window:t]
        w = (w - w.mean(0)) / (w.std(0) + 1e-9)
        C = (w.T @ w) / w.shape[0]
        iu = np.triu_indices(C.shape[0], 1)
        avg_corr[t] = C[iu].mean()
    series = avg_corr[window:]
    labels = None
    try:
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(n_components=2, n_iter=50, random_state=SEED)
        hmm.fit(series.reshape(-1, 1)); labels = hmm.predict(series.reshape(-1, 1))
        method = "HMM"
    except Exception:
        thr = np.median(series); labels = (series > thr).astype(int); method = "median-threshold"
    hi = int(np.argmax([series[labels == k].mean() if (labels == k).any() else -9
                        for k in (0, 1)]))
    frac_hi = float((labels == hi).mean())
    switches = int((np.diff(labels) != 0).sum())
    return dict(available=True, method=method,
                mean_corr_high=float(series[labels == hi].mean()),
                mean_corr_low=float(series[labels != hi].mean()),
                frac_high_corr=frac_hi,
                persistence=1 - switches / max(1, len(labels) - 1),
                series=series, labels=labels)


def lens_mean_reversion(close_tr, tickers, pairs, max_lag=1):
    """NEW: mean-reversion lens.  For candidate pairs, fit a cointegrating
    hedge ratio (OLS log-price), test the spread for stationarity (ADF), and
    estimate the reversion half-life via an AR(1) on the spread.

    Returns pairs that look cointegrated with finite half-life (BH-corrected on
    the ADF p-values).
    """
    out = []
    try:
        from statsmodels.tsa.stattools import adfuller
        import statsmodels.api as sm
        have_sm = True
    except Exception:
        have_sm = False
    lp = np.log(close_tr)
    cand = []
    for pr in pairs:
        i, j = pr["i"], pr["j"]
        x, y = lp[:, j], lp[:, i]
        if have_sm:
            beta = np.polyfit(x, y, 1)[0]
        else:
            beta = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
        spread = y - beta * x
        # AR(1): spread[t] = a + phi*spread[t-1]; half-life = -ln2/ln(phi)
        s0, s1 = spread[:-1], spread[1:]
        phi = np.polyfit(s0, s1, 1)[0]
        hl = -math.log(2) / math.log(phi) if 0 < phi < 1 else np.inf
        if have_sm:
            try:
                p_adf = float(adfuller(spread, maxlag=max_lag, autolag=None)[1])
            except Exception:
                p_adf = 1.0
        else:
            p_adf = 1.0
        cand.append(dict(ti=pr["ti"], tj=pr["tj"], i=i, j=j, beta=float(beta),
                         half_life=float(hl), adf_p=p_adf))
    keep = benjamini_hochberg([c["adf_p"] for c in cand]) if cand else np.zeros(0, bool)
    for k, c in enumerate(cand):
        c["coint_ok"] = bool(keep[k]) and np.isfinite(c["half_life"]) and 1 < c["half_life"] < 120
        if c["coint_ok"]:
            out.append(c)
    return out


# ----- discovery container + two tiers --------------------------------------
class DiscoveredPatterns:
    """Empirical regularities mined from training data via the 9 lenses.

    Ports `class DiscoveredPatterns` from the biology repo almost verbatim
    (different lenses, same container/summary discipline).
    """
    def __init__(self):
        self.up_va = self.down_va = None
        self.emp_up = set(); self.emp_down = set()
        self.lo = self.hi = self.bound_ok = None
        self.pairwise = []; self.conservation = {}; self.periodicity = {}
        self.lead_lag = []; self.lead_lag_multi = {}; self.corr_regime = {}
        self.mean_reversion = []

    @classmethod
    def discover(cls, close_tr, rets_tr, rets_va, dates_tr, tickers):
        p = cls()
        vol = rets_tr.std(0) + 1e-9
        print("[lens 1/9] monotone (trend regime) ...")
        p.up_va, p.down_va, p.emp_up, p.emp_down = lens_monotone(rets_tr, rets_va)
        print("[lens 2/9] bounds (vol bands) ...")
        p.lo, p.hi, p.bound_ok = lens_bounds(rets_tr, rets_va, vol)
        print("[lens 3/9] pairwise (pairs trading) ...")
        p.pairwise = lens_pairwise(rets_tr, rets_va, tickers)
        print("[lens 4/9] conservation (factor structure / cointegration baskets) ...")
        p.conservation = lens_conservation(rets_tr, rets_va, tickers)
        print("[lens 5/9] periodicity (calendar effects) ...")
        p.periodicity = lens_periodicity(rets_tr, dates_tr, tickers)
        print("[lens 6/9] lead-lag (sector) ...")
        p.lead_lag = lens_lead_lag(rets_tr, tickers)
        print("[lens 7/9] lead-lag-multi (NEW, 1d/5d/30d) ...")
        p.lead_lag_multi = lens_lead_lag_multi(rets_tr, tickers)
        print("[lens 8/9] correlation-regime (NEW, HMM) ...")
        p.corr_regime = lens_correlation_regime(rets_tr, dates_tr)
        print("[lens 9/9] mean-reversion (NEW, cointegration half-life) ...")
        p.mean_reversion = lens_mean_reversion(close_tr, tickers, p.pairwise)
        return p

    def summary(self):
        npair = sum(1 for d in self.pairwise if d["fdr_ok"])
        nll = sum(len(v) for v in self.lead_lag_multi.values())
        cr = self.corr_regime
        lines = [
            "DiscoveredPatterns (9-lens analysis of TRAIN data, FDR-corrected):",
            f"    monotone trend candidates : {len(self.emp_up)} up / {len(self.emp_down)} down",
            f"    vol-band bounds validated : {int(self.bound_ok.sum())}/{len(self.bound_ok)}",
            f"    pairwise |r|>{PAIRWISE_THRESHOLD} pairs   : {len(self.pairwise)} ({npair} survive FDR)",
            f"    factor structure (SVD)    : {len(self.conservation.get('factors',[]))} factors, "
            f"{len(self.conservation.get('baskets',[]))} low-var baskets",
            f"    calendar effects          : peak FFT bin {self.periodicity.get('peak_bin')}",
            f"    lead-lag (sector)         : {len(self.lead_lag)}",
            f"    lead-lag-multi (NEW)      : {nll} significant lead-lag pairs across {LEADLAG_HORIZONS}",
            f"    correlation regime (NEW)  : " + (
                f"{cr.get('method')}: hi-corr {cr.get('mean_corr_high',0):.2f} "
                f"({cr.get('frac_high_corr',0)*100:.0f}% of days) vs "
                f"lo-corr {cr.get('mean_corr_low',0):.2f}" if cr.get("available") else "n/a"),
            f"    mean-reversion (NEW)      : {len(self.mean_reversion)} cointegrated pairs "
            f"(median half-life "
            f"{np.median([m['half_life'] for m in self.mean_reversion]):.0f}d)"
            if self.mean_reversion else
            "    mean-reversion (NEW)      : 0 cointegrated pairs",
        ]
        return "\n".join(lines)


class RuleSet:
    """Tier 1: validated, HARD-projected guardrails (biology RuleSet port).

    Finance: clamp predicted next-day returns to per-name vol bands (lens_bounds)
    and enforce any validated directional prior.  This is the prediction-space
    analog of the dG / monotone projection.  (Portfolio-level risk limits —
    pos 5% / sector 25% / 2:1 — are enforced separately in the backtester.)
    """
    def __init__(self):
        self.lo = self.hi = None
        self.coint_pairs = []                          # Tier-1 cointegration (i,j,beta,hl)

    def to(self, dev):
        for a in ("lo", "hi"):
            t = getattr(self, a)
            if t is not None:
                setattr(self, a, t.to(dev))
        return self

    def project(self, pred):
        """pred: (B,N) predicted returns in RET_SCALE units."""
        if self.lo is not None:
            pred = torch.clamp(pred, self.lo.unsqueeze(0), self.hi.unsqueeze(0))
        return pred

    def summary(self):
        nb = int((self.lo is not None) and self.lo.numel())
        return (f"Tier 1 RuleSet: vol-band return clamp on {nb} names "
                f"(|r|<= {VOL_BAND_K} sigma), "
                f"{len(self.coint_pairs)} enforced cointegration pairs")


class Hypotheses:
    """Tier 2: candidates that did NOT pass validation, tracked + soft-loss.

    Ports the biology `Hypotheses` soft-loss machinery: pairwise constraints fed
    back to training as a small auxiliary loss (LAMBDA_HYP), so the model is
    nudged toward discovered structure without letting it override the data.
    """
    def __init__(self):
        self.items = []
        self._pair_i = self._pair_j = self._pair_r = None
        self._mu = self._sd = None

    def add(self, kind, detail, score, source, data=None):
        self.items.append(dict(kind=kind, detail=detail, score=score,
                               source=source, data=data))

    def build_tensors(self, rets_train, device):
        d = rets_train.astype(np.float32)
        self._mu = torch.tensor(d.mean(0), device=device)
        self._sd = torch.tensor(d.std(0) + 1e-6, device=device)
        pi, pj, pr = [], [], []
        for h in self.items:
            if h["kind"].startswith("pairwise") and h.get("data"):
                pi.append(h["data"]["i"]); pj.append(h["data"]["j"])
                pr.append(h["data"]["r_tr"])
        if pi:
            self._pair_i = torch.tensor(pi, device=device)
            self._pair_j = torch.tensor(pj, device=device)
            self._pair_r = torch.tensor(pr, dtype=torch.float32, device=device)
        print(f"[hyp] aux-loss tensors: {len(pi)} pairwise soft constraints "
              f"(lambda={LAMBDA_HYP})")
        return len(pi)

    def has_aux_loss(self):
        return self._pair_i is not None

    def soft_loss(self, pred):
        """Penalise predicted-return pairs that violate the discovered linear
        co-movement: (z_i - r*z_j)^2 in standardised return space."""
        if self._pair_i is None:
            return torch.zeros((), device=pred.device)
        z = (pred / RET_SCALE - self._mu) / self._sd
        zi = z[:, self._pair_i]; zj = z[:, self._pair_j]
        res = zi - self._pair_r.unsqueeze(0) * zj
        return (res * res).mean()

    def summary(self, max_show=12):
        if not self.items:
            return "Tier 2 Hypotheses: none"
        lines = [f"Tier 2 Hypotheses: {len(self.items)} (reported, soft-constrained)"]
        for h in sorted(self.items, key=lambda x: -x["score"])[:max_show]:
            lines.append(f"    [{h['source']:11s}] {h['kind']:18s} {h['detail']}  "
                         f"(score {h['score']:.3f})")
        return "\n".join(lines)


def build_enforcement(patterns, tickers, vol):
    """Sort validated patterns into Tier 1 (hard) and everything else into
    Tier 2 (soft).  Mirrors biology `build_enforcement`."""
    rs, hyp = RuleSet(), Hypotheses()
    rs.lo = torch.tensor(patterns.lo * RET_SCALE, dtype=torch.float32)
    rs.hi = torch.tensor(patterns.hi * RET_SCALE, dtype=torch.float32)
    # Tier 1 cointegration: pairs that pass FDR pairwise AND the ADF/half-life test
    coint = {(m["ti"], m["tj"]): m for m in patterns.mean_reversion}
    for m in patterns.mean_reversion:
        rs.coint_pairs.append((m["i"], m["j"], m["beta"], m["half_life"]))

    # Tier 2: pairwise (those not promoted), factor baskets, calendar, lead-lag
    for d in patterns.pairwise:
        promoted = (d["ti"], d["tj"]) in coint
        hyp.add("pairwise-coint" if promoted else "pairwise-corr",
                f"{d['ti']}<->{d['tj']} r_tr={d['r_tr']:+.2f} r_va={d['r_va']:+.2f}"
                + ("  [Tier-1 coint]" if promoted else ("  [FDR ok]" if d["fdr_ok"] else "")),
                abs(d["r_va"]), "pairwise", data=d if not promoted else None)
    for b in patterns.conservation.get("baskets", []):
        detail = " ".join(f"{w:+.2f}{t}" for t, w in b["top"])
        hyp.add("cointegration-basket", f"std_va={b['std_val']:.2f}: {detail}",
                1 / (1 + b["std_val"]), "svd")
    for f in patterns.conservation.get("factors", []):
        detail = " ".join(f"{w:+.2f}{t}" for t, w in f["top"])
        hyp.add("risk-factor", f"var={f['var_explained']*100:.0f}%: {detail}",
                f["var_explained"], "svd")
    for ll in patterns.lead_lag:
        hyp.add("lead-lag", f"{ll['leader']}->{ll['follower']} lag {ll['lag']}d "
                f"(r={ll['corr']:+.2f})", abs(ll["corr"]), "lead-lag")
    cal = patterns.periodicity.get("calendar", {})
    if "turn_of_month" in cal:
        eff, p = cal["turn_of_month"]
        hyp.add("calendar", f"turn-of-month edge {eff*1e4:+.0f}bp/day (p={p:.3f})",
                1 - p, "calendar")
    print(f"[rules] Tier 1: {int(patterns.bound_ok.sum())} vol-band clamps, "
          f"{len(rs.coint_pairs)} cointegration pairs")
    print(f"[rules] Tier 2: {len(hyp.items)} hypotheses (soft)")
    return rs, hyp


# ============================================================================
#  CORES   (finance equation cores — additive, interpretable contributions)
# ============================================================================
class BetaCAPMCore(nn.Module):
    """Replaces MetabolismCore.  r_i = alpha_i + beta_i * r_mkt.

    beta_i, alpha_i are FROZEN (OLS of stock returns on the equal-weight market
    over TRAIN).  The market return r_mkt is predicted online as a head over the
    pooled graph state.  Contribution is additive: the LGNN learns idiosyncratic
    + nonlinear residual on top of this systematic term.
    """
    def __init__(self, beta, alpha, hidden):
        super().__init__()
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32) * RET_SCALE)
        self.mkt_head = nn.Linear(hidden, 1)            # predict market return (scaled)
        nn.init.zeros_(self.mkt_head.bias); nn.init.normal_(self.mkt_head.weight, std=0.01)

    def forward(self, h_pool):
        r_mkt = self.mkt_head(h_pool)                   # (B,1) scaled market return
        return self.alpha.unsqueeze(0) + self.beta.unsqueeze(0) * r_mkt   # (B,N)


class MomentumCore(nn.Module):
    """Replaces half of CentralDogmaCore.  Cross-sectional 12-1 momentum:
    contribution = coef * z(mom_12_1).  coef initialised from a TRAIN regression
    of next-day return on the standardised momentum score, then fine-tuned."""
    def __init__(self, init_coef):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(float(init_coef) * RET_SCALE))

    def forward(self, mom_z):
        return self.coef * mom_z                         # (B,N)


class MeanReversionCore(nn.Module):
    """Replaces the other half of CentralDogmaCore.  Short-horizon reversal:
    contribution = coef * z(-r_5).  coef from a TRAIN regression, fine-tuned."""
    def __init__(self, init_coef):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(float(init_coef) * RET_SCALE))

    def forward(self, rev_z):
        return self.coef * rev_z                         # (B,N)


# ============================================================================
#  MODEL   (LGNN + CfC backbone — ported verbatim — + Student-t head)
# ============================================================================
class _CfCGraphLayer(nn.Module):
    """Message-passing graph layer with CfC (closed-form continuous-time) update.

    Ported verbatim from colab_cell_emulator.py:_CfCGraphLayer.  Each step:
    aggregate neighbour messages -> degree-normalise -> self-update -> CfC gating
    (h_new = sigma(-gate)*A + sigma(gate)*B with per-node A,B, bounded W via
    cfc_tau_min) -> residual + LayerNorm.  The canonical Liquid update.
    """
    def __init__(self, hidden, n_nodes, cfc_tau_min=0.1):
        super().__init__()
        self.hidden = hidden
        self.msg_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))
        self.self_lin = nn.Linear(hidden, hidden)
        self.W_proj = nn.Linear(hidden, hidden)
        self.b_proj = nn.Linear(hidden, hidden)
        self.cfc_A = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.cfc_B = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02)
        self.cfc_tau_min = cfc_tau_min
        self.norm = nn.LayerNorm(hidden)

    def _forward_impl(self, h, edge_index, edge_weight):
        B, N, H = h.shape
        src, dst = edge_index[0], edge_index[1]
        h_src = h.index_select(1, src); h_dst = h.index_select(1, dst)
        msg = self.msg_mlp(torch.cat([h_src, h_dst], dim=-1))
        msg = (msg * edge_weight.unsqueeze(0).unsqueeze(-1)).to(h.dtype)
        agg = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
        agg = agg.index_add(1, dst, msg)
        ones = torch.ones_like(dst, dtype=h.dtype)
        deg = torch.zeros(N, device=h.device, dtype=h.dtype).index_add(0, dst, ones).clamp(min=1.0)
        agg = agg / deg.unsqueeze(0).unsqueeze(-1)
        combined = agg + self.self_lin(h)
        W = self.W_proj(combined)
        W_clip = 1.0 / max(self.cfc_tau_min, 1e-6)
        gate = W.clamp(-W_clip, W_clip) + self.b_proj(combined)
        h_new = torch.sigmoid(-gate) * self.cfc_A.unsqueeze(0) + torch.sigmoid(gate) * self.cfc_B.unsqueeze(0)
        return self.norm(h + h_new)

    def forward(self, h, edge_index, edge_weight):
        if self.training and torch.is_grad_enabled() and h.device.type == "cuda":
            return ckpt.checkpoint(self._forward_impl, h, edge_index, edge_weight,
                                   use_reentrant=False)
        return self._forward_impl(h, edge_index, edge_weight)


class StudentTHead(nn.Module):
    """Replaces the Gaussian StochasticHead.  Outputs per-node log-sigma; the
    loss is a Student-t NLL (df=4) — finance has fat tails that a Gaussian
    under-prices by ~5-10x.  log-sigma init so sigma ~ 1 (in RET_SCALE units)."""
    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Linear(hidden, 1)
        nn.init.constant_(self.head.bias, 0.0)          # sigma ~ exp(0) = 1.0 (i.e. ~1% daily)
        nn.init.zeros_(self.head.weight)

    def forward(self, h):
        return self.head(h).squeeze(-1).clamp(-3.0, 3.0)


class DynamicsModel(nn.Module):
    """LGNN + CfC backbone (unchanged from biology) with finance cores + Student-t.

    Forward: features (B,N,F) -> per-node hidden via type-embed + in_proj ->
    N CfC graph layers -> idiosyncratic return head.  Then ADD the additive core
    contributions (CAPM systematic + momentum + reversal).  Tracks per-core
    contributions for attribution.  Returns (pred_return, log_sigma).
    """
    def __init__(self, N, F, hidden, n_layers, type_ids, edge_index, edge_weight,
                 mom_idx, rev_idx, beta, alpha, mom_coef, rev_coef,
                 cfc_tau_min=LGNN_CFC_TAU_MIN):
        super().__init__()
        self.N, self.F = N, F
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.register_buffer("type_ids", type_ids)
        self.mom_idx, self.rev_idx = mom_idx, rev_idx
        self.type_embed = nn.Embedding(N_SECTORS, N_TYPE_EMBED)
        self.in_proj = nn.Linear(F + N_TYPE_EMBED, hidden)
        self.layers = nn.ModuleList([_CfCGraphLayer(hidden, N, cfc_tau_min)
                                     for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out_head.bias); nn.init.normal_(self.out_head.weight, std=0.01)
        # cores
        self.capm = BetaCAPMCore(beta, alpha, hidden)
        self.momentum = MomentumCore(mom_coef)
        self.reversion = MeanReversionCore(rev_coef)
        self.stoch = StudentTHead(hidden)
        self.last_components = {}

    def forward(self, feats):                            # feats: (B,N,F)
        B = feats.shape[0]
        te = self.type_embed(self.type_ids).unsqueeze(0).expand(B, -1, -1)
        h = self.in_proj(torch.cat([feats, te], dim=-1))
        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_weight)
        h = self.out_norm(h)
        idio = self.out_head(h).squeeze(-1)              # (B,N) idiosyncratic
        h_pool = h.mean(dim=1)                           # (B,H) market state
        capm = self.capm(h_pool)
        mom = self.momentum(feats[:, :, self.mom_idx])
        rev = self.reversion(feats[:, :, self.rev_idx])
        pred = idio + capm + mom + rev
        self.last_components = dict(idio=idio, capm=capm, momentum=mom, reversion=rev)
        log_sigma = self.stoch(h)
        return pred, log_sigma


# ---- loss building blocks --------------------------------------------------
def student_t_nll(pred, true, log_sigma, df=STUDENT_T_DF):
    """Negative log-likelihood of a Student-t with `df` dof, location `pred`,
    scale exp(log_sigma).  Heavy-tailed -> realistic tail risk."""
    nu = df
    sigma = torch.exp(log_sigma)
    z = (true - pred) / sigma
    c = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
         - 0.5 * math.log(nu * math.pi))
    ll = c - log_sigma - 0.5 * (nu + 1) * torch.log1p(z * z / nu)
    return (-ll).mean()


def soft_ic(pred, true, eps=1e-8):
    """Differentiable cross-sectional IC surrogate (Pearson per day).  Maximise
    correlation between predicted and realised returns across the cross-section.
    (True rank-IC / Spearman is reported in eval; this is its smooth proxy.)"""
    pc = pred - pred.mean(1, keepdim=True)
    tc = true - true.mean(1, keepdim=True)
    num = (pc * tc).mean(1)
    den = pc.std(1) * tc.std(1) + eps
    return (num / den).mean()


def long_short_pnl(pred, true, eps=1e-8):
    """Differentiable long/short factor return: cross-sectionally demean the
    signal, gross-normalise to unit book, dot with realised returns.  A smooth
    stand-in for the decile portfolio (decile selection happens in backtest)."""
    s = pred - pred.mean(1, keepdim=True)
    w = s / (s.abs().sum(1, keepdim=True) + eps)
    return (w * true).sum(1).mean()


# ============================================================================
#  TRAINING   (Student-t NLL + rank-IC + P&L + Tier-2 aux + refinement)
# ============================================================================
def train_model(model, feats, target, rets, idx_train, ruleset, hyp,
                target_log_sigma):
    """K-step-rollout trainer with truncated BPTT (ported from biology
    train_model).  Composite finance loss:

        L = LAMBDA_1STEP * mse
          + LAMBDA_NLL   * student_t_nll
          - LAMBDA_IC    * soft_ic
          - LAMBDA_PNL   * long_short_pnl
          + LAMBDA_HYP   * hypothesis_aux_loss
          + LAMBDA_SIGMA_ANCHOR * (log_sigma - emp_vol)^2
          + LAMBDA_REFINE* refinement_mse   (last 10% of training)

    Rollout advances the autoregressive return-window features with the model's
    own predictions; the refinement pass re-feeds pred.detach() and checks it
    lands at t+2 (robustness to its own imperfect inputs).
    """
    dev = next(model.parameters()).device
    feats_t = torch.tensor(feats, device=dev)
    target_t = torch.tensor(target, device=dev) * RET_SCALE
    tls = torch.tensor(target_log_sigma, device=dev)
    halt_val = 0.0                                     # halted return-window value
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    gen = torch.Generator().manual_seed(SEED + 1)
    idx_train = np.asarray(idx_train)
    # only start days that leave room for the K-rollout
    max_start = idx_train[idx_train <= idx_train.max() - K_MAX]
    refine_start = int(STEPS * REFINE_START_FRAC)
    model.train()
    print(f"[train] steps={STEPS} batch={BATCH} K_MAX={K_MAX} lr={LR} "
          f"df={STUDENT_T_DF} halt_aug={'on' if USE_HALT_AUG else 'off'} dev={dev}")
    t0 = time.time()
    for step in range(STEPS):
        K = 1 + int((K_MAX - 1) * step / max(1, STEPS))
        sel = max_start[torch.randint(0, len(max_start), (BATCH,), generator=gen).numpy()]
        cur = feats_t[sel].clone()                     # (B,N,F)
        # halt augmentation: zero a random ticker's AR window for some batch rows
        if USE_HALT_AUG:
            m = torch.rand(BATCH, device=dev) < HALT_PROB
            if m.any():
                rows = m.nonzero(as_tuple=True)[0]
                cols = torch.randint(0, model.N, (rows.numel(),), device=dev)
                cur[rows, cols, :model.mom_idx] = halt_val
        opt.zero_grad()
        chunk, first_loss, all_means = [], None, []
        for k in range(K):
            pred, log_sigma = model(cur)
            true = target_t[sel + k]                   # next-day return at t+k
            mse = F.mse_loss(pred, true)
            nll = student_t_nll(pred, true, log_sigma)
            ic = soft_ic(pred, true)
            pnl = long_short_pnl(pred, true)
            loss_k = (LAMBDA_1STEP * mse + LAMBDA_NLL * nll
                      - LAMBDA_IC * ic - LAMBDA_PNL * pnl)
            loss_k = loss_k + LAMBDA_SIGMA_ANCHOR * ((log_sigma - tls) ** 2).mean()
            if hyp.has_aux_loss():
                loss_k = loss_k + LAMBDA_HYP * hyp.soft_loss(pred)
            if step >= refine_start and (k + 1) < K:
                pred2, _ = model(_roll_features(cur, pred, model))
                loss_k = loss_k + LAMBDA_REFINE * F.mse_loss(pred2, target_t[sel + k + 1])
            if first_loss is None:
                first_loss = loss_k
            chunk.append(loss_k)
            # advance features autoregressively for the rollout
            cur = _roll_features(cur, ruleset.project(pred).detach(), model)
            if (k + 1) % TBPTT_CHUNK == 0 or k == K - 1:
                cl = torch.stack(chunk).mean()
                if len(all_means) == 0:
                    cl = cl + LAMBDA_1STEP * first_loss
                cl.backward(); all_means.append(float(cl.detach()))
                chunk = []; cur = cur.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()
        if step == 0 or (step + 1) % max(1, STEPS // 8) == 0:
            with torch.no_grad():
                p, ls = model(feats_t[sel])
                tt = target_t[sel]
                ic_now = float(soft_ic(p, tt)); sg = float(torch.exp(ls).mean())
            print(f"  step {step+1:4d}  K={K}  loss {all_means[-1]:+.4f}  "
                  f"trainIC {ic_now:+.3f}  meanSigma {sg:.2f}", flush=True)
    print(f"[train] done in {time.time()-t0:.0f}s")


def _roll_features(cur, pred_scaled, model):
    """Autoregressive feature update for the K-step rollout: shift the AR return
    window forward one day and insert the predicted return; hold slow features.
    pred_scaled is in RET_SCALE units -> convert back to raw return for the window."""
    nxt = cur.clone()
    L = model.mom_idx                                   # AR window occupies cols [0, L)
    nxt[:, :, 1:L] = cur[:, :, 0:L - 1]
    nxt[:, :, 0] = pred_scaled / RET_SCALE
    return nxt


# ============================================================================
#  BACKTEST ENGINE   (long/short, cost-aware, risk-limited)
# ============================================================================
@torch.no_grad()
def predict_returns(model, feats, idx):
    """Predict next-day returns + sigma for the given day indices (1-step, no
    rollout — every signal uses only data available at close of day t)."""
    model.eval()
    dev = next(model.parameters()).device
    ft = torch.tensor(feats[idx], device=dev)
    preds, sigs, comps = [], [], {k: [] for k in ("idio","capm","momentum","reversion")}
    bs = 256
    for s in range(0, len(idx), bs):
        p, ls = model(ft[s:s + bs])
        preds.append(p.cpu().numpy()); sigs.append(torch.exp(ls).cpu().numpy())
        for k in comps:
            comps[k].append(model.last_components[k].cpu().numpy())
    pred = np.concatenate(preds) / RET_SCALE
    sig = np.concatenate(sigs) / RET_SCALE
    comp = {k: np.concatenate(v) / RET_SCALE for k, v in comps.items()}
    return pred, sig, comp


def build_weights(signal, sigma, type_ids):
    """Construct risk-limited long/short weights for one day.

    Long top SELECT_FRAC, short bottom SELECT_FRAC (cross-sectionally demeaned
    signal), equal- or sigma-weighted, then project onto the risk box:
      |w_i| <= MAX_POS,  sum|w| per sector <= MAX_SECTOR,  gross <= GROSS_LEVERAGE.
    Dollar-neutral (long gross == short gross).

    NOTE: with N~80 names a true 10% decile (8/side) cannot satisfy the 5%
    cap; we use SELECT_FRAC=0.20 (quintile) so the cap is feasible, then clip.
    """
    N = len(signal)
    s = signal - signal.mean()
    n_sel = max(2, int(round(SELECT_FRAC * N)))
    order = np.argsort(s)
    shorts = order[:n_sel]; longs = order[-n_sel:]
    w = np.zeros(N)
    base = (1.0 / np.maximum(sigma, 1e-4)) if SIGMA_WEIGHT else np.ones(N)
    w[longs] = base[longs]; w[shorts] = -base[shorts]
    # normalise each side to unit gross
    lg = w[longs].sum(); sg = -w[shorts].sum()
    if lg > 0: w[longs] /= lg
    if sg > 0: w[shorts] /= sg
    # per-name cap
    w = np.clip(w, -MAX_POS, MAX_POS)
    # sector cap: scale down any sector exceeding MAX_SECTOR (per side handled by sign)
    for sec in range(N_SECTORS):
        mask = type_ids == sec
        for sign_mask in (w > 0, w < 0):
            mm = mask & sign_mask
            tot = np.abs(w[mm]).sum()
            if tot > MAX_SECTOR:
                w[mm] *= MAX_SECTOR / tot
    # re-balance to dollar-neutral at the smaller side's gross, respect 2:1
    lg = w[w > 0].sum(); sg = -w[w < 0].sum()
    g = min(lg, sg)
    if lg > 0: w[w > 0] *= g / lg
    if sg > 0: w[w < 0] *= g / sg
    gross = np.abs(w).sum()
    if gross > GROSS_LEVERAGE:
        w *= GROSS_LEVERAGE / gross
    return w


def backtest(signal_panel, sigma_panel, fwd_returns, type_ids,
             cost_bps=COST_BPS, label="model"):
    """Daily-rebalanced long/short backtest.

    signal_panel/sigma_panel/fwd_returns: (Tdays, N) aligned so that row t uses
    signal at close t and earns fwd_returns[t] = return realised at close t+1.
    Charges cost_bps per side on turnover.  Returns a metrics dict + equity curve.
    """
    Tt, N = signal_panel.shape
    w_prev = np.zeros(N)
    pnl = np.zeros(Tt); turn = np.zeros(Tt)
    weights = np.zeros((Tt, N))
    for t in range(Tt):
        w = build_weights(signal_panel[t], sigma_panel[t], type_ids)
        weights[t] = w
        turn[t] = np.abs(w - w_prev).sum()
        gross_ret = float(np.dot(w, fwd_returns[t]))
        cost = turn[t] * (cost_bps * 1e-4)
        pnl[t] = gross_ret - cost
        w_prev = w
    equity = np.cumprod(1 + pnl)
    return _metrics(pnl, equity, turn, weights, label)


def _metrics(pnl, equity, turn, weights, label):
    mu, sd = pnl.mean(), pnl.std()
    ann = math.sqrt(TRADING_DAYS)
    sharpe = (mu / sd * ann) if sd > 1e-12 else 0.0
    downside = pnl[pnl < 0].std() if (pnl < 0).any() else 1e-9
    sortino = (mu / downside * ann) if downside > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = float(-dd.min())
    hit = float((pnl > 0).mean())
    cum = float(equity[-1] - 1)
    ann_ret = float(equity[-1] ** (TRADING_DAYS / max(1, len(pnl))) - 1)
    return dict(label=label, sharpe=float(sharpe), sortino=float(sortino),
                max_dd=maxdd, hit_rate=hit, ann_return=ann_ret, cum_return=cum,
                vol_ann=float(sd * ann), avg_turnover=float(turn.mean()),
                tail_1=float(np.percentile(pnl, 1)), tail_5=float(np.percentile(pnl, 5)),
                pnl=pnl, equity=equity, n_days=len(pnl))


def equal_weight_bah(fwd_returns, cost_bps=COST_BPS):
    """Buy-and-hold equal-weight long-only baseline (the sanity check the model
    must beat after costs).  One-time entry cost; then hold."""
    Tt, N = fwd_returns.shape
    pnl = fwd_returns.mean(1).copy()
    pnl[0] -= (cost_bps * 1e-4)                          # entry cost
    equity = np.cumprod(1 + pnl)
    return _metrics(pnl, equity, np.zeros(Tt), None, "buy&hold-EW")


def smooth_signal(panel, span=5):
    """EMA-smooth a (Tdays,N) signal panel along time to cut turnover.

    A slower signal trades less -> lower cost.  Used to test whether the daily
    strategy's losses are a COST problem (smoothing recovers it) or a SIGNAL
    problem (smoothing doesn't help)."""
    a = 2.0 / (span + 1.0)
    out = np.empty_like(panel)
    out[0] = panel[0]
    for t in range(1, len(panel)):
        out[t] = a * panel[t] + (1 - a) * out[t - 1]
    return out


# ============================================================================
#  EVAL & ATTRIBUTION
# ============================================================================
def spearman_ic_series(pred_panel, true_panel):
    """True cross-sectional rank IC per day (Spearman).  Reported, not trained."""
    from scipy import stats
    ics = []
    for t in range(pred_panel.shape[0]):
        ic, _ = stats.spearmanr(pred_panel[t], true_panel[t])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    return ics


def lens_attribution(patterns, feats, feat_names, fwd_returns, idx_test, type_ids):
    """Standalone P&L/IC of each lens-derived signal on the TEST set.

    Builds a simple tradable signal from each lens and backtests it alone, so we
    can see which lenses carried alpha and which decayed out-of-sample."""
    fi = {n: k for k, n in enumerate(feat_names)}
    F_test = feats[idx_test]                              # (Tt,N,F)
    fwd = fwd_returns
    out = {}

    def run(sig_panel, name):
        m = backtest(sig_panel, np.ones_like(sig_panel), fwd, type_ids, label=name)
        ic = spearman_ic_series(sig_panel, fwd)
        return dict(sharpe=m["sharpe"], ic_mean=float(ic.mean()),
                    ic_ir=float(ic.mean() / (ic.std() + 1e-9) * math.sqrt(TRADING_DAYS)))

    # momentum lens  (12-1)
    out["momentum(12-1)"] = run(F_test[:, :, fi["mom_12_1"]], "lens:momentum")
    # mean-reversion lens (5-day reversal)
    out["mean_reversion(rev5)"] = run(F_test[:, :, fi["rev_5"]], "lens:reversion")
    # bounds/vol lens (low-vol signal = -vol_20)
    out["bounds(low_vol)"] = run(-F_test[:, :, fi["vol_20"]], "lens:lowvol")
    # periodicity/RSI extreme (contrarian on rsi)
    out["periodicity(rsi_contra)"] = run(-F_test[:, :, fi["rsi_14"]], "lens:rsi")
    # pairwise/conservation: signal = own recent return minus correlated-peer mean
    #   (a stat-arb residual) — approximated by 1-day reversal vs cross-section
    resid = F_test[:, :, fi["r_lag0"]] - F_test[:, :, fi["r_lag0"]].mean(1, keepdims=True)
    out["pairwise(statarb_resid)"] = run(-resid, "lens:statarb")
    # trend/monotone: 20-day momentum
    out["monotone(trend20)"] = run(
        np.cumsum(F_test[:, :, [fi[f"r_lag{l}"] for l in range(20)]], -1)[:, :, -1],
        "lens:trend")
    return out


def core_attribution(comp, fwd_returns, type_ids):
    """P&L of each core's standalone contribution to the prediction (build the
    long/short book from that core alone)."""
    out = {}
    for k, sig in comp.items():
        m = backtest(sig, np.ones_like(sig), fwd_returns, type_ids, label=f"core:{k}")
        ic = spearman_ic_series(sig, fwd_returns)
        out[k] = dict(sharpe=m["sharpe"], ic_mean=float(ic.mean()))
    return out


def save_plots(model_m, bah_m, corr_regime, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e})"); return None
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios":[2,1]})
    ax[0].plot(model_m["equity"], label=f"LGNN+CfC L/S (Sharpe {model_m['sharpe']:.2f})", lw=2)
    ax[0].plot(bah_m["equity"], label=f"Buy&Hold EW (Sharpe {bah_m['sharpe']:.2f})",
               lw=1.5, alpha=0.8)
    ax[0].axhline(1.0, color="k", lw=0.5)
    ax[0].set_title("Out-of-sample equity curve (net of costs)")
    ax[0].set_ylabel("Equity (start=1.0)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    eq = model_m["equity"]; dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    ax[1].fill_between(range(len(dd)), dd * 100, color="crimson", alpha=0.4)
    ax[1].set_title(f"Drawdown (max {model_m['max_dd']*100:.1f}%)")
    ax[1].set_ylabel("DD %"); ax[1].set_xlabel("Trading day (test)"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "equity_curve.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"[plot] saved {path}")
    return path


# ============================================================================
#  MAIN
# ============================================================================
def estimate_factor_params(rets_train, feats_train, feat_names):
    """Frozen CAPM betas/alphas + initial momentum/reversal coefficients from
    TRAIN only (no leakage)."""
    fi = {n: k for k, n in enumerate(feat_names)}
    mkt = rets_train.mean(1)                              # equal-weight market
    N = rets_train.shape[1]
    beta = np.zeros(N); alpha = np.zeros(N)
    vm = mkt.var() + 1e-12
    for i in range(N):
        beta[i] = np.cov(rets_train[:, i], mkt)[0, 1] / vm
        alpha[i] = rets_train[:, i].mean() - beta[i] * mkt.mean()
    # next-day return regressed on standardised momentum / reversal (pooled)
    def coef(name):
        x = feats_train[:, :, fi[name]]
        xz = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
        y = np.zeros_like(x); y[:-1] = rets_train[1:]    # next-day
        xz, y = xz[:-1].ravel(), y[:-1].ravel()
        return float(np.cov(xz, y)[0, 1] / (xz.var() + 1e-12))
    return beta, alpha, coef("mom_12_1"), coef("rev_5")


def main():
    banner("FINANCE EMULATOR — Phase 1  (LGNN+CfC, biology-derived)")
    print(f"device={DEVICE}  smoke={SMOKE}  universe={len(UNIVERSE)} tickers")

    # ---- data -----
    banner("1. DATA")
    close, source = load_prices(UNIVERSE)
    tickers = list(close.columns)
    print(f"[data] source: {source}")
    print(f"[data] {close.shape[0]} trading days x {close.shape[1]} tickers; "
          f"{str(np.asarray(close.index)[0])[:10]} .. {str(np.asarray(close.index)[-1])[:10]}")
    if source == "SYNTHETIC":
        print("[data] *** WARNING: synthetic data — backtest numbers are NOT a "
              "valid kill-criterion test ***")
    ds = build_dataset(close)
    feats, target, rets = ds["feats"], ds["target"], ds["rets"]
    feat_names = ds["feat_names"]; T = feats.shape[0]
    fi = {n: k for k, n in enumerate(feat_names)}
    tr, va, te = split_by_fraction(T)
    d = np.asarray(close.index)
    def dr(idx):
        return f"{str(d[idx[0]])[:10]}..{str(d[idx[-1]])[:10]} ({len(idx)}d)"
    print(f"[split] train {dr(tr)} | val {dr(va)} | test {dr(te)}")

    # standardise features using TRAIN stats only
    mu = feats[tr].reshape(-1, feats.shape[-1]).mean(0)
    sd = feats[tr].reshape(-1, feats.shape[-1]).std(0) + 1e-8
    feats_n = (feats - mu) / sd
    # keep mom/rev as standardised-but-cross-sectional for the cores
    rets_tr = rets[tr]; rets_va = rets[va]

    # ---- graph -----
    banner("2. GRAPH")
    ei, ew, type_ids = build_full_graph(tickers, rets_tr)

    # ---- lenses + two-tier rules -----
    banner("3. LENSES (9) + TWO-TIER RULES")
    patterns = DiscoveredPatterns.discover(
        close.values[tr], rets_tr, rets_va, d[tr], tickers)
    print(patterns.summary())
    vol = rets_tr.std(0) + 1e-9
    ruleset, hyp = build_enforcement(patterns, tickers, vol)
    print(ruleset.summary())
    print(hyp.summary())
    hyp.build_tensors(rets_tr, DEVICE)

    # ---- model -----
    banner("4. MODEL (LGNN + CfC + finance cores + Student-t)")
    beta, alpha, mom_c, rev_c = estimate_factor_params(rets_tr, feats[tr], feat_names)
    print(f"[capm] mean|beta|={np.abs(beta).mean():.2f}  "
          f"momentum_coef0={mom_c:+.4f}  reversal_coef0={rev_c:+.4f}")
    # cores consume the (train-)standardised cross-sectional mom/rev features
    model = DynamicsModel(
        N=len(tickers), F=feats.shape[-1], hidden=LGNN_HIDDEN, n_layers=LGNN_N_LAYERS,
        type_ids=type_ids, edge_index=ei, edge_weight=ew,
        mom_idx=fi["mom_12_1"], rev_idx=fi["rev_5"],
        beta=beta, alpha=alpha, mom_coef=mom_c, rev_coef=rev_c).to(DEVICE)
    ruleset.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params, hidden={LGNN_HIDDEN}, layers={LGNN_N_LAYERS}, "
          f"df={STUDENT_T_DF}")

    # σ-anchor target: per-name empirical daily vol (in RET_SCALE units)
    target_log_sigma = np.log(np.clip(vol * RET_SCALE, 0.05, 20.0)).astype(np.float32)
    target_log_sigma = np.broadcast_to(target_log_sigma, (1, len(tickers))).copy()

    # ---- train -----
    banner("5. TRAINING")
    train_model(model, feats_n, target, rets, tr, ruleset, hyp,
                target_log_sigma)

    # ---- validation IC (quick generalisation check) -----
    banner("6. VALIDATION")
    p_va, s_va, _ = predict_returns(model, feats_n, va)
    ic_va = spearman_ic_series(p_va, target[va])
    print(f"[val] rank-IC mean {ic_va.mean():+.4f}  IC-IR "
          f"{ic_va.mean()/(ic_va.std()+1e-9)*math.sqrt(TRADING_DAYS):+.2f}  "
          f"(n={len(ic_va)} days)")

    # ---- backtest on TEST -----
    banner("7. BACKTEST (out-of-sample test period, net of costs)")
    pred_te, sig_te, comp_te = predict_returns(model, feats_n, te)
    fwd_te = target[te]                                  # realised next-day returns
    type_np = type_ids.cpu().numpy()
    model_m = backtest(pred_te, sig_te, fwd_te, type_np, label="LGNN+CfC L/S")
    bah_m = equal_weight_bah(fwd_te)
    ic_te = spearman_ic_series(pred_te, fwd_te)

    def show(m):
        print(f"  {m['label']:18s} Sharpe {m['sharpe']:+.2f}  Sortino {m['sortino']:+.2f}  "
              f"maxDD {m['max_dd']*100:5.1f}%  hit {m['hit_rate']*100:4.1f}%  "
              f"annRet {m['ann_return']*100:+5.1f}%  turn {m['avg_turnover']:.2f}")
    show(model_m); show(bah_m)
    print(f"  model rank-IC mean {ic_te.mean():+.4f}  IC-IR "
          f"{ic_te.mean()/(ic_te.std()+1e-9)*math.sqrt(TRADING_DAYS):+.2f}")
    print(f"  tail returns: 1%={model_m['tail_1']*100:.2f}%  5%={model_m['tail_5']*100:.2f}%")

    # --- cost / turnover diagnostics: is the loss a COST or SIGNAL problem? ---
    model_gross = backtest(pred_te, sig_te, fwd_te, type_np, cost_bps=0.0,
                           label="LGNN+CfC (GROSS, 0 cost)")
    pred_sm = smooth_signal(pred_te, span=10)
    model_smooth = backtest(pred_sm, sig_te, fwd_te, type_np,
                            label="LGNN+CfC (smoothed)")
    show(model_gross); show(model_smooth)
    # momentum factor traded directly (slow turnover) — the strongest lens
    fi_m = {n: k for k, n in enumerate(feat_names)}
    mom_sig = feats_n[te][:, :, fi_m["mom_12_1"]]
    mom_m = backtest(mom_sig, np.ones_like(mom_sig), fwd_te, type_np,
                     label="momentum(12-1) factor")
    show(mom_m)

    # σ-weighted variant
    global SIGMA_WEIGHT
    SIGMA_WEIGHT = True
    model_sw = backtest(pred_te, sig_te, fwd_te, type_np, label="LGNN+CfC L/S (σ-wt)")
    SIGMA_WEIGHT = False
    show(model_sw)

    # ---- attribution -----
    banner("8. ATTRIBUTION")
    print("Lens attribution (standalone test-set signal):")
    la = lens_attribution(patterns, feats_n, feat_names, fwd_te, te, type_np)
    for k, v in sorted(la.items(), key=lambda x: -x[1]["sharpe"]):
        print(f"    {k:26s} Sharpe {v['sharpe']:+.2f}  IC {v['ic_mean']:+.3f}  "
              f"IC-IR {v['ic_ir']:+.2f}")
    print("Core attribution (standalone contribution to model prediction):")
    ca = core_attribution(comp_te, fwd_te, type_np)
    for k, v in sorted(ca.items(), key=lambda x: -x[1]["sharpe"]):
        print(f"    {k:12s} Sharpe {v['sharpe']:+.2f}  IC {v['ic_mean']:+.3f}")

    # ---- plots + verdict -----
    banner("9. VERDICT vs KILL CRITERION")
    save_plots(model_m, bah_m, patterns.corr_regime, OUT_DIR)
    sh, dd = model_m["sharpe"], model_m["max_dd"]
    gross_sh = model_gross["sharpe"]; smooth_sh = model_smooth["sharpe"]
    print(f"Kill criterion (PRIMARY, net daily): Sharpe > {KILL_SHARPE_GO} AND "
          f"maxDD < {KILL_MAXDD*100:.0f}%  (kill if Sharpe < {KILL_SHARPE_DIE})")
    print(f"Result:  net-daily Sharpe = {sh:+.2f}  maxDD = {dd*100:.1f}%  | "
          f"GROSS Sharpe {gross_sh:+.2f} | smoothed {smooth_sh:+.2f} | "
          f"momentum-factor {mom_m['sharpe']:+.2f} | Buy&Hold {bah_m['sharpe']:+.2f}")
    cost_problem = (gross_sh > KILL_SHARPE_GO or smooth_sh > KILL_SHARPE_GO) and sh < KILL_SHARPE_DIE
    if source == "SYNTHETIC":
        verdict = ("INVALID — synthetic data. Re-run in Colab with yfinance for a "
                   "real verdict.")
    elif sh > KILL_SHARPE_GO and dd < KILL_MAXDD:
        verdict = "PASS — clears Phase-1 threshold. Candidate to proceed to Phase 2."
    elif cost_problem:
        verdict = ("KILL (as-built) but DIAGNOSED: net-daily fails, yet gross/"
                   "smoothed signal clears the bar — this is a TURNOVER/COST "
                   "problem, not a no-signal problem. Fix turnover (slower "
                   "rebalance, trade bands) before declaring dead. See README.")
    elif sh < KILL_SHARPE_DIE:
        verdict = ("KILL — net Sharpe below 0.3 and gross signal too weak to "
                   "rescue. Architecture lacks tradable edge for daily equities "
                   "on this universe/window.")
    else:
        verdict = ("BORDERLINE — between 0.3 and 0.5. Needs specific fixes before "
                   "Phase 2 (see README).")
    print(f"VERDICT: {verdict}")
    return dict(model=model_m, bah=bah_m, gross=model_gross, smooth=model_smooth,
                momentum=mom_m, sigma_wt=model_sw, ic_test=ic_te,
                lens=la, core=ca, source=source, verdict=verdict, patterns=patterns)


if __name__ == "__main__":
    main()
