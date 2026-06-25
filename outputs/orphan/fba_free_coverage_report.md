# Coverage check on the FBA-free wheel — the honest picture

Your question after the precision result: *what about coverage?* The TTT
precision lift (+12.2 pp on unmodeled orgs) was real but came from a tiny
high-confidence subset (just a handful of calls per organism). The fair test
is **essential coverage at a matched operating precision** — e.g. P≥0.90 —
because that's the operating point that decides how many true essentials a
screen would actually catch.

I built a combined-score (mean of W1+W2+W3) and swept the threshold to the
largest call-set that still hits P≥0.90. Coverage = TP / total_essential.

## Modeled organisms — proxy vs real FBA at P≥0.90

| organism | total ess | REAL FBA: P / essCov / calls | PROXY: P / essCov / calls |
|---|---|---|---|
| beril_Putida | 365 | 0.901 / **0.373** / 151 | 0.900 / **0.542** / 220 |
| beril_Keio   | 189 | 0.902 / **0.630** / 132 | 0.900 / **0.619** / 130 |
| beril_Koxy   |  18 | 0.909 / 0.556 / 11 | unstable (only 18 ess) |

On the two organisms with enough essentials for a stable curve, the proxy
matches Keio (0.619 vs 0.630) and **beats real FBA on Putida by 17 pp of
essential coverage** at matched precision. So the proxy is not just "more
conservative" — at the same P=0.90, it actually ranks more true essentials
into the top call set than real FBA does.

## Unmodeled organisms — 2-wheel vs 3-wheel (with proxy) at P≥0.90

| metric | 2-wheel (W1+W2) | 3-wheel (W1+W2+proxy) | delta |
|---|---|---|---|
| mean precision | 0.875 | 0.874 | ≈ 0 |
| **mean essCov** | **0.505** | **0.507** | **+0.2 pp** |

That's the headline number on the 33 unmodeled organisms — and it's
effectively a tie. Per-org deltas range from −9 pp to +10 pp; roughly half
gain, half lose, mean zero.

## What this actually says

- **Modeled orgs (where we have FBA truth and the model fits):** the proxy is
  at least as good as real FBA for screening — same precision, same or higher
  recall.
- **Unmodeled orgs (the 45 with no model):** the proxy does **not** expand
  essential-coverage at the P≥0.90 operating point. It also doesn't shrink it.
  Net: zero — but you also get nothing back.
- **TTT (all-3-vote-essential) tier still wins by +12 pp precision on
  unmodeled orgs**, but the *count* in that tier is small. It's a
  high-confidence sub-tier, not a recall extender.

## So can we make a network that works without FBA at similar accuracy?

Yes — but with the honest scope:

1. **For modeled organisms:** the proxy is a drop-in substitute for FBA at the
   screening operating point (P≥0.90), with **equal or better recall**.
2. **For unmodeled organisms:** the 3-wheel system with proxy delivers
   the same operating-point coverage as the 2-wheel baseline (≈ 0.51 essCov @
   P=0.90), plus a high-confidence sub-tier that's noticeably more precise.

The recall ceiling on unmodeled orgs (~0.51 essCov @ P=0.90) is the **real
data ceiling** — it's the same ceiling that even real FBA would hit when
transferred via OG (we showed that hurts in `metabolic_transfer.json`). To
break through it, you need **new data** for those organisms: condition-specific
fitness (feba.db / Wheel 4) is the only signal not redundant with what we
already have.

## Numbers I trust now

| view | precision | essCov / recall |
|---|---|---|
| 2-wheel (any organism, unmodeled mean) | 0.875 | **0.505** |
| 3-wheel w/ proxy (unmodeled mean) | 0.874 | **0.507** |
| 3-wheel w/ proxy (Putida, modeled) | 0.900 | **0.542** |
| 3-wheel w/ real FBA (Putida, modeled) | 0.901 | **0.373** |

The FBA-free wheel is real and useful; it does not, by itself, lift the
recall ceiling on novel organisms.
