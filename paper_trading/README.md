# Paper-trading sandbox

A zero-risk place to trade real-data markets, feel the wins and losses, keep a
real scoreboard — and risk **exactly $0**.

It exists because of a simple split: the desire to *grow money* and the desire
for *the rush* are different things, and chasing the rush with real money on
leveraged crypto / options / micro-caps is how accounts go to zero (we tested
it — at 3× leverage even **Bitcoin**, the best asset of the century, hit $0).
So: get the rush here, for free, and let real money compound somewhere boring.

## Principles baked in

- **Real prices, no synthetic data.** Crypto from Coin Metrics; stocks from an
  S&P-500 daily sample. Cached locally after first fetch.
- **Costs are charged.** Every fill pays commission + slippage, so paper
  results aren't rosier than reality.
- **No look-ahead.** The market advances one bar at a time; you only act on
  prices you can already see. The backtester applies today's decision to
  *tomorrow's* return.
- **The benchmark is always shown.** A strategy's only meaningful result is
  whether it beats buy & hold on risk-adjusted return (Sharpe). The tool prints
  the verdict for you, flattering or not.

## Quick start

```bash
# fresh $100 account, universe of BTC + ETH + AAPL, clock starting 2020
python -m paper_trading.cli reset --cash 100 --symbols BTC ETH AAPL --start 2020-01-01

python -m paper_trading.cli symbols          # what you can trade
python -m paper_trading.cli price BTC         # price at the current clock
python -m paper_trading.cli buy BTC 0.006     # place a paper trade
python -m paper_trading.cli tick 30           # advance the market 30 days
python -m paper_trading.cli status            # cash, positions, P&L
python -m paper_trading.cli history           # your trade log
```

## Test ideas honestly

```bash
# leak-free, cost-charged backtest vs buy & hold
python -m paper_trading.cli backtest trend BTC
python -m paper_trading.cli backtest trend AAPL     # note: trend LOSES here

# let a strategy drive the live paper account, then inspect the track record
python -m paper_trading.cli run trend BTC --days 200
python -m paper_trading.cli status
```

## Evolve it

Add a strategy by subclassing `Strategy` in `strategies.py` and returning a
target weight in `[0, 1]` from data **up to today only**:

```python
class MyEdge(Strategy):
    name = "myedge"
    def target_weight(self, prices):
        ...  # your idea here
        return weight
```

It's instantly available to `backtest` and `run`. The rule that keeps you
honest: **fix your parameters before you see the test period.** If an idea only
looks good after tuning it on the same data you judge it on, it's overfitting —
use `--oos 0.5` to tune on the first half and judge on the held-out tail.

## Files

| File | Role |
|---|---|
| `data.py` | real price loaders + cache |
| `broker.py` | persistent paper account, orders, costs, valuation |
| `strategies.py` | strategy framework + starters (buyhold, trend, meanrevert) |
| `engine.py` | leak-free backtester + live auto-runner |
| `cli.py` | the cockpit |

Account state (`state/`) and cached data (`data_cache/`) stay local and are
git-ignored.
