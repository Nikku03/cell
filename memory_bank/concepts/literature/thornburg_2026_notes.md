# Thornburg 2026 (Cell) — notes

_Paper registered as `sources/thornburg_2026_cell.json`. Full text was blocked from the sandbox; extracts below come from the bioRxiv preprint abstract + a dozen independent news summaries (Cell, bioRxiv, Illinois News Bureau, EurekAlert, phys.org, astrobiology.com, GEN, Singularity Hub, Short Box, Newswise, scienmag, C&EN). Anything marked "not reported" here means I did not find a number in those sources; the full Cell PDF will have more specifics._

## Title + authorship

- **Title:** Bringing the genetically minimal cell to life on a computer in 4D
- **Authors (17):** Thornburg ZR, Maytin A, Kwon J, Brier TA, Gilbert BR, Fu E, Gao Y-L, Quenneville J, Wu T, Li H, Long T, Pezeshkian W, Sun L, Glass JI, Mehta AP, Ha T, Luthey-Schulten Z
- **Journal / year:** Cell, March 9, 2026 (advance online)
- **DOI:** 10.1016/j.cell.2026.02.009
- **Preprint:** bioRxiv 10.1101/2025.06.10.658899 (June 2025)

## What's new vs Thornburg 2022

| Aspect | Thornburg 2022 | Thornburg 2026 |
|---|---|---|
| Dimensionality | 0D (well-mixed pools) | **4D (3D spatial + time)** |
| Simulated window | sub-second to minutes of bio-time | **full 105-min cell cycle** |
| Processes | gene expression + metabolism | + ribosome biogenesis, chromosome dynamics, DNA replication, morphological growth + division |
| Compute | single CPU core, minutes | **6 days on 2 GPUs** per replicate |

## Quantitative facts extracted

| Parameter | Value | Notes |
|---|---|---|
| Cell cycle / doubling time | **105 min = 6,300 s** | More precise than the ~2 h (7,200 s) we had from Thornburg 2022. |
| Gene count (simulated) | 493 | Does not match our CP016816.2 parse of 496 gene features (458 CDS + 38 RNA). Likely a definition / annotation-revision issue; recorded as `facts/uncertainty/syn3a_gene_count_thornburg2026_discrepancy.json`. |
| Chromosome | 543 kbp, single circular | Consistent with our `syn3a_chromosome_length` fact (543,379 bp). |
| Simulation software | Lattice Microbes (LM) | Same codebase that underlies the 2022 paper. Spatial extension is new. |
| GPUs | 2 (one for DNA replication, one for everything else) | Not reported: exact GPU models. |
| Wall time / cycle | 6 days | Per replicate. |
| Cell division | symmetrical | Confirmed from multiple replicates. |
| Voxel size | not reported (in the pieces I could read) | |
| Cell volume / diameter | not reported in the extracted digests | |

## Validation paradigm

Unlike this project (which targets MCC against Breuer 2019 essentiality labels), Thornburg 2026 validates against **experimental dynamic measurements**:

- Origin:terminus ratio measured by DNA sequencing.
- Measured **doubling time** of 105 min.
- **mRNA half-lives** (per-gene or lumped — not specified in the digests).
- **Protein distributions** (presumably per-gene steady-state copy numbers).
- **Ribosome counts**.

No MCC against Breuer / Hutchison essentiality labels is reported in the digests. If the Cell PDF does not contain one either, **our Layer 6 MCC work is still novel territory**: Thornburg 2026 validates *dynamics*, we target *essentiality prediction*.

## Implications for this project

1. **Update our `syn3a_doubling_time` fact** from the Thornburg-2022-era ~2 h (7,200 s) to the more-precisely-measured 105 min / 6,300 s from Thornburg 2026. The brief's 2 ± 0.5 h validation target comfortably covers 105 min.
2. **Reinforces CPU-first mandate**: their 6-day-on-2-GPUs budget per cell cycle confirms that a full-fidelity 4D simulation is NOT where our CPU-only project should go. Our Layer 6 essentiality work, with short bio-time windows, is the deliberately cheap-and-fast alternative. We stay there.
3. **New structural-validation targets** worth registering once we can reach the full text: per-gene mRNA half-lives (their values) and per-gene protein distributions. These would sharpen the Layers 1-3 validation targets currently described loosely in the brief as "Thornburg 2022 within 2x for 90% of genes".
4. **493-gene count discrepancy** with our 496 needs a one-line reconciliation when we get the PDF. Worth a `facts/uncertainty/` entry but not blocking.
5. **Our Layer 6 essentiality metric remains valuable**: the extracted digests do not mention Thornburg 2026 performing gene-knockout essentiality prediction with a Breuer-benchmarked MCC. Our pipeline is still the only effort I see that is pointed directly at that metric.

## Blockers (what I can't finish without user help)

- Cell fulltext at DOI `10.1016/j.cell.2026.02.009` — 403 on every channel from the sandbox.
- bioRxiv preprint (`10.1101/2025.06.10.658899v1`) — same.
- Illinois News Bureau, EurekAlert, phys.org, Singularity Hub — all blocked.
- **If the user can paste the bioRxiv PDF full text into a gist / message** I can extract specific numerical values I flagged as "not reported" and promote them from this notes file into proper `facts/` entries.

## Source list

Digests were assembled from these WebSearch hits (URLs in the raw logs):

- Cell fulltext article page (blocked for fetch, but title + authors / DOI visible from SERP).
- bioRxiv preprint page (same).
- University of Illinois News Bureau press release.
- EurekAlert release.
- phys.org, astrobiology.com, GEN, Singularity Hub, Short Box, Newswise, scienmag, C&EN news articles.
- ResearchGate figure metadata.
