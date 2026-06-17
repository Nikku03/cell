"""Paper broker: a simulated trading account with persistent state.

NO real money is ever involved. The point is to feel the wins and losses on a
real-data market with a real scoreboard, while risking exactly $0.

Design choices that keep it honest:
  - Trades pay a commission AND slippage, so paper results aren't rosier than
    reality would be.
  - The market advances one bar at a time; you only ever act on prices you can
    already see (no look-ahead).
  - State persists to state/portfolio.json so the account evolves over sessions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import pandas as pd

from . import data

STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_PATH = os.path.join(STATE_DIR, "portfolio.json")

COMMISSION = 0.001   # 10 bps per trade
SLIPPAGE = 0.0005    # 5 bps adverse fill


class Market:
    """A multi-asset replayable market with one shared daily clock.

    The cursor is a calendar date; price(symbol) returns the most recent close
    at or before the cursor. advance(n) steps the clock forward n trading days.
    """

    def __init__(self, symbols, start=None):
        self.frames = {s.upper(): data.load_prices(s).set_index("date")["price"]
                       for s in symbols}
        # global union timeline
        idx = None
        for s in self.frames.values():
            idx = s.index if idx is None else idx.union(s.index)
        self.timeline = idx.sort_values()
        self.cursor = 0
        if start is not None:
            self.cursor = int(self.timeline.searchsorted(pd.Timestamp(start)))

    @property
    def date(self):
        return self.timeline[min(self.cursor, len(self.timeline) - 1)]

    def at_end(self) -> bool:
        return self.cursor >= len(self.timeline) - 1

    def price(self, symbol: str):
        symbol = symbol.upper()
        s = self.frames[symbol]
        upto = s.loc[:self.date]
        return float(upto.iloc[-1]) if len(upto) else float("nan")

    def advance(self, n: int = 1) -> int:
        before = self.cursor
        self.cursor = min(self.cursor + n, len(self.timeline) - 1)
        return self.cursor - before


@dataclass
class Account:
    starting_cash: float = 100.0
    cash: float = 100.0
    positions: dict = field(default_factory=dict)        # symbol -> qty
    trades: list = field(default_factory=list)           # trade log
    equity_curve: list = field(default_factory=list)     # [{date, value}]
    cursor: int = 0
    symbols: list = field(default_factory=lambda: ["BTC"])
    realized_pnl: float = 0.0


class PaperBroker:
    def __init__(self):
        self.account = self._load()
        self.market = Market(self.account.symbols)
        self.market.cursor = self.account.cursor

    # ---- persistence ----
    def _load(self) -> Account:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                return Account(**json.load(f))
        return Account()

    def save(self):
        self.account.cursor = self.market.cursor
        with open(STATE_PATH, "w") as f:
            json.dump(asdict(self.account), f, indent=2, default=str)

    def reset(self, starting_cash=100.0, symbols=None, start=None):
        symbols = symbols or ["BTC"]
        self.account = Account(starting_cash=starting_cash, cash=starting_cash,
                               symbols=[s.upper() for s in symbols])
        self.market = Market(self.account.symbols, start=start)
        self.account.cursor = self.market.cursor
        self.save()

    # ---- valuation ----
    def position_value(self, symbol):
        return self.account.positions.get(symbol.upper(), 0.0) * self.market.price(symbol)

    def equity(self) -> float:
        total = self.account.cash
        for sym in self.account.positions:
            total += self.position_value(sym)
        return total

    # ---- orders ----
    def buy(self, symbol, qty):
        symbol = symbol.upper()
        px = self.market.price(symbol) * (1 + SLIPPAGE)
        cost = px * qty * (1 + COMMISSION)
        if cost > self.account.cash + 1e-9:
            raise ValueError(f"Insufficient cash: need ${cost:,.2f}, have ${self.account.cash:,.2f}")
        self.account.cash -= cost
        self.account.positions[symbol] = self.account.positions.get(symbol, 0.0) + qty
        self._log("BUY", symbol, qty, px, cost)
        return px

    def sell(self, symbol, qty):
        symbol = symbol.upper()
        held = self.account.positions.get(symbol, 0.0)
        if qty > held + 1e-9:
            raise ValueError(f"Insufficient {symbol}: trying to sell {qty}, hold {held}")
        px = self.market.price(symbol) * (1 - SLIPPAGE)
        proceeds = px * qty * (1 - COMMISSION)
        self.account.cash += proceeds
        self.account.positions[symbol] = held - qty
        if abs(self.account.positions[symbol]) < 1e-12:
            del self.account.positions[symbol]
        self._log("SELL", symbol, qty, px, proceeds)
        return px

    def _log(self, side, symbol, qty, px, cash_flow):
        self.account.trades.append({
            "date": str(self.market.date.date()), "side": side, "symbol": symbol,
            "qty": round(qty, 8), "price": round(px, 6), "cash_flow": round(cash_flow, 2),
            "equity_after": round(self.equity(), 2),
        })

    # ---- time ----
    def tick(self, n=1):
        moved = self.market.advance(n)
        self.account.equity_curve.append(
            {"date": str(self.market.date.date()), "value": round(self.equity(), 2)})
        return moved
