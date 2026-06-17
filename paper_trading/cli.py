"""Command-line cockpit for the paper-trading sandbox.

Examples:
  python -m paper_trading.cli reset --cash 100 --symbols BTC ETH AAPL
  python -m paper_trading.cli symbols
  python -m paper_trading.cli price BTC
  python -m paper_trading.cli buy BTC 0.001
  python -m paper_trading.cli tick 30
  python -m paper_trading.cli status
  python -m paper_trading.cli backtest trend BTC
  python -m paper_trading.cli run trend BTC --days 200
  python -m paper_trading.cli history
"""
from __future__ import annotations

import argparse

from . import data, engine
from .broker import PaperBroker


def _fmt_money(x):
    return f"${x:,.2f}"


def cmd_reset(a):
    b = PaperBroker()
    b.reset(starting_cash=a.cash, symbols=a.symbols, start=a.start)
    print(f"Account reset: {_fmt_money(a.cash)} cash, universe = {b.account.symbols}")
    print(f"Market clock starts at {b.market.date.date()}.")


def cmd_symbols(a):
    s = data.available_symbols()
    print("Crypto:", ", ".join(s["crypto"]))
    stocks = s["stocks_sample"]
    if isinstance(stocks, list):
        print(f"Stocks ({len(stocks)} in S&P-500 sample, 2013-2018), e.g.:",
              ", ".join(stocks[:20]), "...")
    else:
        print("Stocks:", stocks)


def cmd_price(a):
    b = PaperBroker()
    print(f"{a.symbol.upper()} @ {b.market.date.date()}: {_fmt_money(b.market.price(a.symbol))}")


def cmd_buy(a):
    b = PaperBroker()
    px = b.buy(a.symbol, a.qty)
    b.save()
    print(f"BUY {a.qty} {a.symbol.upper()} @ {_fmt_money(px)}  ->  equity {_fmt_money(b.equity())}")


def cmd_sell(a):
    b = PaperBroker()
    px = b.sell(a.symbol, a.qty)
    b.save()
    print(f"SELL {a.qty} {a.symbol.upper()} @ {_fmt_money(px)}  ->  equity {_fmt_money(b.equity())}")


def cmd_tick(a):
    b = PaperBroker()
    moved = b.tick(a.n)
    b.save()
    print(f"Advanced {moved} day(s) to {b.market.date.date()}. Equity {_fmt_money(b.equity())}")


def cmd_status(a):
    b = PaperBroker()
    eq = b.equity()
    start = b.account.starting_cash
    pnl = eq - start
    pct = pnl / start * 100 if start else 0.0
    print(f"=== Paper account @ {b.market.date.date()} ===")
    print(f"Cash:    {_fmt_money(b.account.cash)}")
    if b.account.positions:
        print("Positions:")
        for sym, qty in b.account.positions.items():
            print(f"  {sym:6s} {qty:.6f}  worth {_fmt_money(b.position_value(sym))}")
    else:
        print("Positions: (none)")
    print(f"Equity:  {_fmt_money(eq)}")
    print(f"P&L:     {_fmt_money(pnl)}  ({pct:+.2f}% on {_fmt_money(start)} start)")
    print(f"Trades:  {len(b.account.trades)}")


def cmd_history(a):
    b = PaperBroker()
    rows = b.account.trades[-a.n:]
    if not rows:
        print("No trades yet.")
        return
    for t in rows:
        print(f"{t['date']}  {t['side']:4s} {t['qty']:>12.6f} {t['symbol']:6s} "
              f"@ {_fmt_money(t['price'])}  equity={_fmt_money(t['equity_after'])}")


def _print_bt(res):
    print(f"=== Backtest: {res['strategy']} on {res['symbol']} "
          f"({res['days']} days, leak-free, costs charged) ===")
    sm, bm = res["strategy_metrics"], res["benchmark_metrics"]
    hdr = f"{'':12s}{'CAGR':>10s}{'Sharpe':>9s}{'MaxDD':>9s}{'Final x':>12s}"
    print(hdr)
    for label, m in [("strategy", sm), ("buy & hold", bm)]:
        print(f"{label:12s}{m['cagr']*100:9.2f}%{m['sharpe']:9.2f}"
              f"{m['maxdd']*100:8.1f}%{m['final_x']:12.2f}")
    verdict = "BEATS" if sm["sharpe"] > bm["sharpe"] else "does NOT beat"
    print(f"-> On risk-adjusted return (Sharpe), strategy {verdict} buy & hold.")


def cmd_backtest(a):
    res = engine.backtest(a.strategy, a.symbol, oos_split=a.oos)
    _print_bt(res)


def cmd_run(a):
    b = PaperBroker()
    steps = engine.auto_run(b, a.strategy, a.symbol, days=a.days)
    print(f"Ran '{a.strategy}' on {a.symbol.upper()} for {steps} paper days.")
    cmd_status(a)


def build_parser():
    p = argparse.ArgumentParser(prog="paper_trading.cli",
                                description="Zero-risk paper-trading sandbox on real data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reset"); r.add_argument("--cash", type=float, default=100.0)
    r.add_argument("--symbols", nargs="+", default=["BTC"])
    r.add_argument("--start", default=None, help="start date, e.g. 2020-01-01")
    r.set_defaults(func=cmd_reset)

    sub.add_parser("symbols").set_defaults(func=cmd_symbols)

    for name, fn in [("price", cmd_price)]:
        sp = sub.add_parser(name); sp.add_argument("symbol"); sp.set_defaults(func=fn)

    for name, fn in [("buy", cmd_buy), ("sell", cmd_sell)]:
        sp = sub.add_parser(name); sp.add_argument("symbol"); sp.add_argument("qty", type=float)
        sp.set_defaults(func=fn)

    t = sub.add_parser("tick"); t.add_argument("n", type=int, nargs="?", default=1)
    t.set_defaults(func=cmd_tick)

    sub.add_parser("status").set_defaults(func=cmd_status)

    h = sub.add_parser("history"); h.add_argument("n", type=int, nargs="?", default=20)
    h.set_defaults(func=cmd_history)

    bt = sub.add_parser("backtest"); bt.add_argument("strategy"); bt.add_argument("symbol")
    bt.add_argument("--oos", type=float, default=0.0); bt.set_defaults(func=cmd_backtest)

    rn = sub.add_parser("run"); rn.add_argument("strategy"); rn.add_argument("symbol")
    rn.add_argument("--days", type=int, default=None); rn.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
