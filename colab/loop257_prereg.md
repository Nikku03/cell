# Loop 257 -- pre-registered reading of the outcome

Written and committed BEFORE the run produced a number. The point is that the
interpretation of every cell below is fixed in advance, so that whatever lands
cannot be talked into being a success after the fact.

Baseline to beat: the additive model, 0.4518 here / 0.4477 in loop 252.
Ceiling that no explanation can exceed: 0.2487 on the residual (construct split,
what two disjoint shRNA hairpins agree on).

Loop 256 already failed this target: two-tower 0.3001 / 0.3017 / 0.3058.

## The two gates that decide the reading

- **I2** -- gated operator vs additive baseline. Bar +0.02.
  Asks only *does the learned correction help at all*.
- **I3** -- K=8 gated vs K=1 context-blind, identical capacity otherwise. Bar +0.01.
  Asks *does knowing which cell line it is matter*. **This is the load-bearing gate.**

## The four cells, decided in advance

| I2 | I3 | What I will write |
|----|----|-------------------|
| PASS | PASS | The context-gated operator is a real context model. Then and only then does I6 (wrong-line control) decide whether it is real or memorisation of eight training lines. I6 FAIL demotes the whole thing to a leak. |
| PASS | FAIL | **A better GENE model, not a context model.** The correction is learning something about perturbation identity that the gene mean missed, and the gates are decoration. This is a NEGATIVE for the CG-HBN hypothesis and gets recorded as one, not as "loop 257 improved on the baseline". |
| FAIL | PASS | Incoherent -- gating beats no-gating while neither beats additive. Report as a defect to be diagnosed, not a finding. Most likely cause to check first: the correction is fitting noise that the K=1 arm fits worse. |
| FAIL | FAIL | The architecture change did not rescue loop 256. Combined with loop 256 that is two independent neural attempts losing to a linear additive model, and I will say plainly that the 68.7% interaction variance remains unexplained by anything I have built. |

## What no outcome here can establish

- Nine cell lines is nine points for the hypernetwork; it maps measured properties
  to K gates from **eight** training examples per fold. Any positive is a
  memorisation suspect until I6 clears it.
- A learned operator basis is dense but is not a physical network. A win names no
  mechanism and does not recover what loops 253 and 255 failed to find.
- The gene embedding is the gene's own mean response elsewhere, so this
  generalises across CONTEXTS, never across genes.
- 978 landmarks, shRNA knockdown, not a transcriptome and not a clean knockout.

## Pre-committed failure handling

If the run crashes or the 6h timeout kills it, that is reported as a crash with
the traceback, not as an inconclusive result, and the source is fixed and rerun.
No partial-seed numbers get quoted as if they were the result.
