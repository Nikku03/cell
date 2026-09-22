# INVENTORY — Phase 0

Source: Fiandaca G, Péré M, Bonhomme K, Chaves M, Roux J. "Drug-tolerant persisters to TRAIL
emerge from a dose-dependent surface in a cell-state continuum of sensitivity." *npj Systems
Biology and Applications* (2026). PMID 42443213,
[DOI](https://doi.org/10.1038/s41540-026-00782-4).

## Acquisition

| file | bytes | SHA-256 |
|---|---|---|
| `41540_2026_782_MOESM1_ESM.pdf` | 2,278,946 | `495cbd346cc433a30007bb948b8cf3db4891226769cf1c34feb586d5bf8c847c` |
| `41540_2026_782_MOESM2_ESM.zip` | 316,041 | `9f320a56308775fcfea11593533a7401607895b9d47a56f05a76b67453f74bb5` |

Downloaded by `scripts/00_acquire.sh`, unpacked by `scripts/01_unpack.sh` into `data/raw/`.
Archive contains 98 files: **39 `.mat`**, 18 `.m`, plus 41 macOS `__MACOSX` resource-fork stubs
and 2 `.DS_Store` (no content; excluded from all analysis). Full per-variable dump:
`INVENTORY_raw.txt`, produced by `scripts/02_inventory.py`.

The main-text HTML is JavaScript-rendered; only the abstract was retrievable
(`data/dl/article.txt`, 11,878 chars). The supplementary PDF is 7 pages of figure captions
(`data/dl/supp.txt`).

## Archive structure

Organised as `Codes for Figures/Figure N/`. No README and no data dictionary; the variable
meanings below are transcribed from comments inside `Figure_2.m`.

### Figure 2 — 4 example cells per condition
| variable | shape | meaning |
|---|---|---|
| `FRET_{R,S}_{25,50,100}ng` | (4, 120) | FRET time series, 4 selected cells x 120 frames |
| `{R,S}_{dose}_ind_par` | (4, 5) | per-cell fitted `[pC8(0), FLIP(0), alpha_0, alpha_1, K_deg]` |
| `Tend_{R,S}_{dose}` | (1, 4) | death time in **minutes**; 600 for R (= end of window) |
| `common_parameters` | (1, 8) | population params `[rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET]` |

### Figure 3 — the full per-cell fitted dataset
| variable | shape |
|---|---|
| `FLIP0_values_R_{25,50,100}ng` | (150, 1), (114, 1), (65, 1) |
| `FLIP0_values_S_{25,50,100}ng` | (177, 1), (300, 1), (518, 1) |
| `pC80_values_*` | identical shapes; verified element-count match to `FLIP0_*` for all 6 groups |

### Figures 4-6 — simulation, not measurement
| variable | shape | meaning |
|---|---|---|
| `sampling` | (1000, 5) | 1000 **sampled** parameter sets |
| `data_classified_{dose}` | (1000, 8) | col1 = index 1..1000; cols 3-7 = a verified row-permutation of `sampling` (1000/1000 overlap); col8 = class label in {-1, +1} |

## Cells and doses

- **Distinct TRAIL doses: 3** — 25, 50, 100 ng/ml, entering the model as `TRAIL0` = 750, 1500,
  3000 molecules (`Figure_2.m` lines 108, 254, 400).
- **Distinct cells with fitted state: 1,324** — 327 (25 ng), 414 (50 ng), 583 (100 ng).
- Cells with a deposited **time series: 24** (4 per condition x 6 conditions).

| dose | tolerant (R) | sensitive (S) | total | tolerant fraction |
|---|---|---|---|---|
| 25 ng | 150 | 177 | 327 | 0.4587 |
| 50 ng | 114 | 300 | 414 | 0.2754 |
| 100 ng | 65 | 518 | 583 | 0.1115 |

## Time resolution

From `Figure_2.m` line 51: `t_obs = linspace(5,600,120)`, with the comment
`% T_window = [0 600] %unit measure = [min] - data collected every 5 minutes`.

- **Frame interval: 5 minutes.**
- **Total imaging duration: 600 minutes (10 h).**
- **Frames per cell: 120**, first frame at **t = +5 min relative to TRAIL addition**.
- Sensitive cells' traces terminate at death (`Tend`); observed finite-point counts range 17–71.

## The five critical questions

### (a) Lineage or tracking identifiers linking mother to daughter — **NO**
No variable in any of the 39 `.mat` files is a cell, lineage, track, or frame-to-cell
identifier. Of 39 variables, 5 are integer-typed and **all five are `Tend_*`** (death times in
minutes), shape (1,4). Verified by `scripts/04_phase0_structural.py` CHECK 4. A case-insensitive
grep of all 18 `.m` files for `lineage|track|mother|daughter|sister|sibling|famil|pedigree`
returns **zero matches**.

*Note on a false positive:* the supplementary PDF contains the word "lineage" once, in
"Cervix lineage" — a DepMap **tissue-of-origin** label for cell lines in Figure S4. That is not
cell-lineage tracking and is not counted.

### (b) Pre-treatment time series — **NO**
`Figure_2.m` line 51 fixes the observation window as `linspace(5,600,120)` minutes with the
window declared `[0 600]` and TRAIL present from t=0 (it is initial condition `y(1)=TRAIL0`).
The first observation is 5 minutes **after** dosing. There is no frame before treatment.

### (c) The same cell measured at two or more separated times — **NO (for the state variable)**
The FRET reporter *is* sampled 120 times per cell, but FRET is not the state variable. The state
is `(pC8(0), FLIP(0))` — fitted **initial conditions**, one scalar pair per cell. Proof it cannot
be re-read later: in the deposited ODE, `d(pC8)/dt` and `d(FLIP)/dt` contain **binding and
unbinding terms only** — no synthesis, no first-order decay (`scripts/04_phase0_structural.py`
CHECK 2). The state is consumed by the death cascade, not tracked through it.

### (d) Per-cell fitted parameters — **YES**
`FLIP0_values_*` and `pC80_values_*` (Figure 3), n = 1,324 cells; and the full five-parameter
vector `[pC8(0), FLIP(0), alpha_0, alpha_1, K_deg]` in `{R,S}_{dose}_ind_par` for the 24 example
cells.

### (e) Per-cell fate labels and death times — **YES (fates), PARTIAL (death times)**
Fate is encoded by which file a cell's value sits in — `_R_` (tolerant) vs `_S_` (sensitive) —
for all 1,324 cells. Death times exist only as `Tend_*`, i.e. for the **24 example cells**.

## Code in the archive

18 `.m` files. `Figure_2.m` (18 kB) is the substantive one: it declares the variable ordering,
the 10-state ODE, the time base, the rescaling constant `K1 = 0.007325300696406` min⁻¹, and the
initial conditions, then integrates with `ode15s` at `RelTol=AbsTol=1e-12` and overlays simulated
against experimental FRET. `Figure_3.m` plots the per-cell `FLIP0`/`pC80` distributions.
`Figure_4.m`/`Figure_5.m` operate on the 1000 sampled states and fit the classification
hyperplane (`mcfit3.m`, `supportingfile.m`). `Figure_6.m` re-runs the 50 ng R cells.
`plasma.m` is a colormap. Expected input is the co-located `.mat` files; no external data.

## Phase 0 verdict

**(a) NO, (b) NO, (c) NO.** All three routes to `tau` are absent. Per the predeclared decision
rule in `PREDECLARE.md`, Phase 2 is not run and no substitute quantity is promoted in its place.
