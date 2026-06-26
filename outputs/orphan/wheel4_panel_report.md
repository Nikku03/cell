# Wheel 4 (feba fitness) across the panel + TF into Wheel 2

Built both as requested. The numbers split sharply into **trustworthy** and
**inflated**, and I'm labelling them so half-knowledge doesn't leak in.

## TRUSTWORTHY anchor — mtub, independent DeJesus truth (from prior verified test)

| | conservation | conservation + W4 fitness |
|---|---|---|
| AUC | 0.722 | **0.819** (+0.095, CI [0.070, 0.110]) |
| neCov @ P0.90 | 0.03 | **0.75** |

Verified by 4 adversarial checks. This is the honest measure of what Wheel 4 adds.

## INFLATED (upper bound) — panel of 46 organisms

| metric | W1+W2 | + W4 | delta |
|---|---|---|---|
| essCov @ P0.90 | 0.456 | 0.789 | **+33.3pp** |
| neCov @ P0.90 | 0.756 | 0.960 | **+20.3pp** |

**Do NOT take these magnitudes at face value.** The panel's truth labels are
`via essential_families` — *derived from feba.db itself*. So W4 (also feba) is
partly predicting its own source: circular. The +33pp is an **upper bound**, not
the deployable lift. The honest lift is the mtub anchor.

Two tells that confirm the circularity inflation:
- `Methanococcus_JJ/S2`: essCov 0.000 → 0.914/0.918. A jump from literally zero
  to >0.9 is the feba-essentiality call reappearing as both predictor and label.
- `Putida`, `Syringae`: W1+W2 essCov ~0.06–0.09 (anomalously low) → ~0.6–0.7.
  The low baseline + huge jump is the same artifact.

The *direction* (W4 helps everywhere, all 46 orgs positive on neCov; 45/46 on
essCov) is consistent with the clean mtub result. The *magnitude* is not
trustworthy from the panel.

## TF features into Wheel 2 — Keio, real RegulonDB (no proxy)

| | W1+W2 | +W4 | +W4+TF |
|---|---|---|---|
| essCov @ P0.90 | 0.554 | 0.817 | **0.817** |
| neCov @ P0.90 | 0.989 | 1.000 | **1.000** |
| AUC | 0.918 | 0.970 | **0.970** |

**TF features add exactly 0.000 on top of W1+W2+W4.** This confirms the earlier
ceiling test, now also on top of Wheel 4, and the TF null is *robust* (TF
features come from RegulonDB, independent of the feba-derived truth — so if they
carried essentiality signal it would show even against this truth; it doesn't).
Regulation tells you *when* a gene is on, not *whether* it's essential.

## Honest verdict

- **Wheel 4 is real and it's the breakthrough signal** — but its *measured*
  value is the mtub anchor (+0.095 AUC, neCov 0.03→0.75 on independent truth),
  not the +33pp panel headline (circular). To get a clean panel-wide lift we'd
  need non-feba-derived truth for more organisms.
- **TF into Wheel 2 adds nothing** (0.000), as predicted. Included as requested;
  the result is a clean, robust null.

## Durability

`data/external_data/feba_fitness_features.csv` (225k rows, 49 orgs) — the slim
per-gene Wheel-4 features (absent_from_feba, min_fit_sig). The 9 GB feba.db is
gitignored/ephemeral; this slim extract is committed so Wheel 4 is reproducible
without re-downloading.
