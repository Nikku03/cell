#!/usr/bin/env python3
"""Generate colab_finance_emulator.ipynb from colab_finance_emulator.py.

Same pattern as the biology repo: the notebook is a thin, self-contained Colab
wrapper that (1) installs deps, (2) writes the emulator source to disk via
%%writefile, (3) runs it.  No nbformat dependency — we emit the ipynb JSON
directly so this is reproducible anywhere.

Usage:  python scripts/build_finance_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "colab_finance_emulator.py")
OUT = os.path.join(HERE, "colab_finance_emulator.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.strip("\n").split("\n")]}


def code(lines):
    src = [l if l.endswith("\n") else l + "\n" for l in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def main():
    with open(SRC) as f:
        source_lines = f.read().split("\n")

    cells = []
    cells.append(md("""
# Finance Trading Prototype — LGNN + CfC + Lens Suite (Phase 1)

A Liquid Graph Neural Network with closed-form continuous-time (CfC) cells,
finance equation cores (Beta-CAPM / Momentum / Mean-Reversion), a 9-lens
pattern-discovery suite, two-tier rule enforcement, and a Student-t (fat-tailed)
uncertainty head — **adapted from a JCVI-Syn3A whole-cell emulator**.

**This is a minimum-viable backtest to test whether the biology architecture has
edge in equities.**  We predict next-day cross-sectional returns on ~80 S&P-500
names, build a cost-aware long/short book, and judge it against a hard kill
criterion.

### Kill criterion (Phase 1)
`Sharpe > 0.5` net of 5 bps/side **and** `max drawdown < 25%` on the
out-of-sample test period.  `Sharpe < 0.3` ⇒ kill the project.

### Data
- **In Colab:** uses `yfinance` (daily *adjusted* OHLCV, 2019-01-01 .. 2024-12-31).
- **Offline / blocked network:** falls back to a real, committed GitHub dataset
  (S&P-500 daily bars 2013-02 .. 2018-02) so the pipeline runs end-to-end on
  **real prices**.  Train/Val/Test are allocated by calendar fraction over
  whatever real data is obtained.
- Last resort: a flagged synthetic generator (NOT a valid backtest).

### Architecture map (biology → finance)
| Biology | Finance |
|---|---|
| `DynamicsModel` (LGNN+CfC), `_CfCGraphLayer` | **kept as-is** |
| `MetabolismCore` (bi-bi kinetics) | `BetaCAPMCore` |
| `CentralDogmaCore` | `MomentumCore` + `MeanReversionCore` |
| `VolumeCore`, `PINNHead` | dropped (Phase 2) |
| ΔG° clamps | risk-limit projection (5% / 25% / 2:1) |
| Gaussian `StochasticHead` | **Student-t head (df=4)** |
| knockout augmentation | halt augmentation |
| 6 lenses | ported + retuned; **+3 new** (lead-lag-multi, correlation-regime, mean-reversion) |
"""))

    cells.append(md("## 1. Install dependencies"))
    cells.append(code([
        "!pip install -q yfinance torch numpy pandas scipy scikit-learn "
        "statsmodels matplotlib hmmlearn",
    ]))

    cells.append(md("## 2. Write the emulator source to disk\n"
                    "(The full ~1,500-line module — self-contained for Colab.)"))
    cells.append(code(["%%writefile colab_finance_emulator.py"] + source_lines))

    cells.append(md("## 3. Run the full pipeline\n"
                    "Data → graph → 9 lenses → two-tier rules → LGNN+CfC training → "
                    "cost-aware backtest → attribution → **verdict vs kill criterion**.\n\n"
                    "In Colab this pulls 2019-2024 from yfinance. Set "
                    "`FINANCE_TICKERS` to customise the universe."))
    cells.append(code(["!python colab_finance_emulator.py"]))

    cells.append(md("## 4. (Optional) inspect / customise in-process\n"
                    "Import and call `main()` to get the results dict back, or "
                    "tweak config constants (e.g. `SELECT_FRAC`, `COST_BPS`, "
                    "`STEPS`) before running."))
    cells.append(code([
        "import colab_finance_emulator as fe",
        "# fe.STEPS = 1500          # train longer",
        "# fe.SIGMA_WEIGHT = True   # size positions by 1/sigma",
        "results = fe.main()",
        "print(results['verdict'])",
    ]))

    cells.append(md("""
## How to read the result

The backtest prints several variants on purpose:

- **net daily** — the primary kill-criterion strategy (after 5 bps/side costs).
- **GROSS (0 cost)** — isolates whether the model has *any* predictive signal.
- **smoothed** — slows turnover; tests whether losses are a cost problem.
- **momentum(12-1) factor** — a simple low-turnover benchmark the model must beat.
- **buy&hold EW** — the long-only sanity check.

If *net daily* fails but *gross* clears the bar, the edge is real but eaten by
turnover/costs — a different diagnosis (and different fix) than "no signal".
**Be honest about which one you got.**
"""))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.x"},
                       "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT}  ({len(cells)} cells, {len(source_lines)} source lines)")


if __name__ == "__main__":
    main()
