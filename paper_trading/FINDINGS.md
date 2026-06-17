# Research findings

Autonomous research session run via `python -m paper_trading.research`. Every
number below is **out-of-sample** (the strategy never saw the test period when
its parameters were set), **cost-charged**, and measured against buy & hold on
the **same** asset. Reproduce with the command above.

## Experiment 1 — Does any strategy reliably beat the market?

For each strategy × asset I held out the second half of history and measured
`strategy Sharpe − buy&hold Sharpe`. Positive = it beat the market.

| asset | trend | absmom | breakout | voltrend | meanrevert |
|---|--:|--:|--:|--:|--:|
| BTC | **+0.37** | **+0.21** | **+0.28** | **+0.40** | −0.76 |
| ETH | **+0.28** | **+0.05** | **+0.17** | **+0.28** | −0.55 |
| LTC | −0.30 | −0.01 | −0.43 | −0.38 | −0.25 |
| DOGE | **+0.01** | **+0.02** | **+0.03** | **+0.05** | −0.57 |
| AAPL | −0.16 | **+0.43** | −0.30 | −0.16 | −1.01 |
| MSFT | −0.41 | −0.14 | −0.24 | −0.41 | −0.08 |
| AMZN | **+0.08** | −0.08 | −0.58 | **+0.05** | −0.75 |
| JPM | **+0.04** | −0.07 | −0.45 | **+0.04** | −0.44 |
| XOM | −0.18 | −0.11 | **+0.17** | −0.18 | −0.16 |
| KO | −1.17 | −0.03 | −1.32 | −1.17 | −0.11 |
| GE | **+1.26** | **+0.63** | **+0.85** | **+1.26** | −0.42 |
| WMT | −0.19 | −0.05 | −0.04 | −0.19 | −0.33 |

**Beat-the-market rate (out-of-sample):**

| strategy | won on | rate |
|---|---|---|
| trend | 6/12 | 50% |
| voltrend | 6/12 | 50% |
| absmom | 5/12 | 42% |
| breakout | 5/12 | 42% |
| meanrevert | **0/12** | **0%** |

### What this honestly says

1. **No strategy beats the market more than ~half the time across a random
   cross-section of assets.** Anyone selling a system that "always wins" is
   lying. A real edge is a *tilt*, not a switch.
2. **The trend family's edge is real but concentrated**, not uniform. It wins
   decisively on the high-volatility, strongly-trending names (BTC +0.37,
   voltrend +0.40; GE +1.26) and loses on the placid, mean-reverting ones
   (KO −1.17, MSFT −0.41). The edge *is* "ride big persistent moves" — so it
   only shows up where big persistent moves exist. This matches why earlier the
   same rules cut Bitcoin's max drawdown from −93% to −70%: the value is largely
   risk control, not return-chasing.
3. **Mean-reversion is a clean, documented negative result: 0 for 12.** Buying
   dips and trimming strength underperformed buy & hold on *every* asset, badly
   on the trenders (AAPL −1.01). Catching falling knives is a losing game over a
   full cycle; it feels good (you "buy low") right up until a trend runs it over.
4. **`voltrend` ≈ `trend` on stocks** (identical cells) because a 50% annual
   vol target rarely binds for ~20-30%-vol equities; it only adds value on
   crypto, where it trimmed BTC's risk enough to nudge Sharpe up (+0.40 vs
   +0.37). Sizing matters only where volatility is extreme.

## Experiment 2 — The optimization trap

Pick the "best" trend window by fitting the **first** half of an asset's
history, then judge that choice on the **held-out second** half.

| asset | best window in-sample | in-sample Sharpe | that window's OOS Sharpe | untuned default (50) OOS | verdict |
|---|--:|--:|--:|--:|---|
| BTC | 10 | 2.29 | 0.66 | **1.22** | optimizing **hurt** |
| ETH | 30 | 2.26 | 0.62 | 0.61 | optimizing did nothing |
| AAPL | 160 | 1.42 | 1.34 | 0.47 | optimizing helped (1 of 3) |

### What this honestly says

- **In-sample Sharpes of 2.2–2.3 collapsed to ~0.6 out-of-sample.** The dazzling
  backtest number is mostly fitted noise. This is the single most expensive
  illusion in trading, shown here on real data.
- **The "optimized" window beat the dumb textbook default in only 1 of 3 cases**,
  and in the worst case (BTC) it nearly halved the real Sharpe. Tuning a knob to
  the past usually buys you a worse future.
- The window that was *actually* best out-of-sample (BTC: 120, ETH: 200) was
  unknowable in advance — which is the whole point.

## Experiment 3 — Does regime-awareness help?

Experiment 1 hinted that trend tools work *where the asset is volatile and
trending* and hurt elsewhere. So I built two regime filters and tested them the
same ruthless way (OOS, costs, vs the same assets). Cells are **OOS Sharpe**.

- **`regime`** — trend, but only "armed" when the Kaufman Efficiency Ratio says
  the market is moving cleanly (not chopping). Go long only if uptrend *and*
  efficiency clears a threshold.
- **`regimeswitch`** — switch tools by volatility: use trend in turbulent
  regimes (recent vol > long-run vol) for its downside protection; just hold in
  calm regimes, where trend only pays whipsaw.

| asset | buyhold | trend | regime | regimeswitch |
|---|--:|--:|--:|--:|
| BTC | 0.81 | **1.18** | 0.91 | 0.79 |
| ETH | 0.63 | **0.92** | 0.50 | 0.75 |
| LTC | **0.41** | 0.12 | −0.55 | 0.16 |
| DOGE | 0.92 | **0.94** | 0.84 | 0.90 |
| AAPL | 0.61 | 0.46 | 0.42 | **0.69** |
| MSFT | **1.29** | 0.89 | 0.60 | 1.18 |
| AMZN | 1.61 | 1.69 | 0.55 | **1.83** |
| JPM | 1.01 | **1.05** | 0.16 | 0.88 |
| XOM | 0.04 | −0.13 | −0.27 | **0.16** |
| KO | **0.27** | −0.90 | −0.47 | −0.54 |
| GE | −0.94 | 0.33 | **0.57** | −0.55 |
| WMT | **0.82** | 0.63 | 0.67 | 0.68 |
| **mean** | **0.62** | 0.60 | 0.33 | 0.58 |
| **median** | 0.72 | **0.76** | 0.52 | 0.72 |
| **worst** | −0.94 | −0.90 | −0.55 | **−0.55** |

### What this honestly says

1. **The clever filter (`regime`) was a bust — a clean negative result.** Mean
   Sharpe 0.33, dead last, *below buy & hold*. Gating on the efficiency ratio
   sat out so much of the move that it sacrificed more upside than the whipsaws
   it avoided. This is the over-engineering trap: a smarter-sounding rule that
   tests worse. Recorded, not buried.
2. **`regimeswitch` did not raise the ceiling — it raised the floor.** Its mean
   Sharpe (0.58) is a hair *below* both buy & hold (0.62) and plain trend (0.60),
   so it is **not** a free lunch. Its one real, narrow win is the **worst case**:
   −0.55, the least-bad of any approach (buy & hold and trend both crater to
   ≈ −0.9 on their worst asset). It also fixed plain trend's specific failures on
   the calm names — AAPL (0.69 vs trend 0.46), AMZN (1.83 vs 1.69), XOM
   (0.16 vs −0.13) — by holding instead of whipsawing there, while keeping most
   of trend's crypto edge.
3. **So regime-awareness bought robustness, not return.** That is the same lesson
   as the leverage and the 150-year tests, one more time: the durable gain is a
   higher floor (survive your worst asset / worst decade), not a higher peak.
   `regimeswitch` is the better *default* precisely because it is rarely the
   worst choice — never the best by much either.
4. **Honest limitation:** none of this beats just buying and holding a good asset
   on average. The edge across a blind cross-section remains a *tilt* toward
   better worst-case behavior, not a money pump.

## Takeaways for how we evolve

- Keep parameters fixed a priori; treat any in-sample optimization with deep
  suspicion and always confirm on held-out data (`--oos`).
- Judge strategies by their **beat rate across many assets**, not one flattering
  chart. Robustness > peak performance.
- **Regime-awareness was built and tested (Experiment 3).** The aggressive
  efficiency-ratio gate failed; the volatility-regime switch bought *robustness*
  (best worst-case, fixed trend's calm-asset failures) but not higher average
  return. Use `regimeswitch` as the better-behaved default, not as a money pump.
  If we push regime work further, the bar is: beat buy & hold's **mean** Sharpe
  across the cross-section, not just its worst case.
- Mean-reversion as implemented is dead on arrival; if we revisit it, it needs a
  trend filter so it stops catching knives.
