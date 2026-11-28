# CellOS — Measured Capability Scorecard

Every figure below is **measured on held-out data with controls**, not asserted. Sources: K562 Perturb-seq (Replogle),
Norman 2019, Sci-Plex 3, RPE1, HCT116, ENCODE, S669. Interactive version:
https://claude.ai/code/artifact/e9f6a15b-9329-43b1-b06c-e113612d0ffa

The pattern is consistent: strong on **protein structure & mutation**, moderate on **how big** a response is, walled on
**which specific distal genes move** — and that wall is a missing-feature problem (the signal reproduces), not noise, and is
strongly cell-type-specific.

## ① Structure & mutation — reliable, ~proteome-wide
| Question | Metric (held-out) | Confidence |
|---|---|---|
| Will a missense mutation destabilize the protein? | **ΔΔG r = 0.47**, RMSE 1.45 kcal/mol (S669; beats DDGun/ThermoNet/FoldX) | Reliable |
| Will an interface mutation weaken binding / is it a hotspot? | **r = 0.45**, hotspot AUC 0.79 | Reliable |

## ② Direct targets (near field) — real where annotated
| Question | Metric | Confidence |
|---|---|---|
| Which genes does this TF directly regulate? | curated regulon **6.0×** enrichment — but only ~**10% of KOs** (annotated TFs) | Moderate |
| Which enhancer controls this gene? | **AUPRC 0.48** vs 0.41 distance (K562 CRISPRi) | Moderate |

## ③ Downstream response — how big
| Question | Metric | Confidence |
|---|---|---|
| How big will the response be (how many genes move)? | **Spearman 0.46** (up to 0.56 rich features), deep K562 | Moderate |
| Does response scale with drug dose? | **ρ 0.40**, magnitude rises with dose in 71% of drugs (Sci-Plex) | Moderate |

## ④ Downstream response — which genes (the hard part)
| Question | Metric | Confidence |
|---|---|---|
| Specific movers, **strong** KO (≥50 movers)? | **9.0/10** top-10 (deployment) — but generic stress genes | Strong-only |
| Specific movers, **weak** KO (<15 movers)? | **0.8/10** — majority of KOs; metric also capped at n_movers/10 | Walled ≈1/10 |
| Which generic stress program (the "tide")? | recall **48–63%** @top-100/200 | Moderate |
| How confident per gene (calibrated probability)? | **ECE 0.006** — honest probabilities, abstains where no signal | Reliable |

## ⑤ Combinations & transfer
| Question | Metric | Confidence |
|---|---|---|
| What does a double perturbation do? | additive-from-singles **ρ 0.40**, 37% of top movers; 15% strong genetic interaction (Norman) | Moderate |
| Does a K562 prediction transfer to another cell type? | **ρ 0.13** cross-line (RPE1 & HCT116 both), vs 0.32 within-line | Weak (cell-type-specific) |

## ⑥ The walls (one now partially breached)
- **Name the specific far-field genes of a weak KO** — chance from the graph (1.06×), **but reproducible within a cell line**
  (ρ 0.25, 7/20 top movers recur). It's a missing-feature problem, not noise. Paralogs, graph propagation, and promoter ATAC failed
  to close it — but the **analogy/transfer model does partially** (see below): borrowing from functionally-similar knockouts
  recovers **28.8%** of the specific (non-tide) movers vs 4.4% for random neighbors and 0% for the prior. The wall is bent, not
  broken.
- **Predict compensation from a single steady-state knockdown** — paralog upregulation is at chance (0/123). Buffering is real but
  only visible in combinatorial data (Norman), and even there it's the minority (15%).

## ⑦ New data this session
CELLxGENE Census was checked first — all 2,203 catalog datasets are cell **atlases**, not guide-labeled screens — so perturb-seq
came from **scPerturb/Zenodo**.

| Source | Status | Result |
|---|---|---|
| Norman 2019 (K562 double CRISPRa) | Tested | doubles largely additive (ρ 0.40); GI is the minority (15%), redundant pairs |
| Replogle RPE1 (3rd cell line) | Tested | cross-line ρ 0.13 even same-protocol → cell-type specificity is real biology |
| Sci-Plex 3 (dose titration) | Tested | magnitude scales with dose (ρ 0.40, 71%); no dose-invariant program shown |
| ENCODE K562 ATAC | Tested | promoter openness does not explain the tide (0.068, anti-correlated) — "open doors" wire falsified |
| Real K562 Hi-C TADs | Tested | 3D-TAD neighborhood is a weak wire for the tide (0.016, below linear proxy) — physical-genome hypothesis not supported |
| Sci-Plex 24h vs 72h (A549) | Tested | response reshapes massively over time (ρ 0.062, 77% transient movers) — time matters, but a 24/72h chemical proxy |
| Primary CD4⁺ T cells (Shifrut) | Tested | generic tide does **not** transfer to a primary cell type (ρ −0.065); magnitude intuition does (0.29–0.40) |
| Sub-24h genetic perturb-seq | Not available | the ideal data to break the transient-compensation wall — not a clean h5ad; needs GEO MTX assembly. **Real unmet need.** |
| Perturb-ATAC / multiome | Not available | no guide-labeled multiome h5ad in scPerturb; ENCODE has perturbation-snATAC but not ready-to-use |

## ⑧ What was missed → combinations (one tested and it worked)
The far field is **reproducible signal our graph can't read** (ρ 0.25 within-line), not noise — so the target is the right
*feature*, not a bigger model.

- ✅ **Analogy / transfer model — TESTED, WORKS.** Predict a KO's movers from its most functionally-similar *other* KOs. Recovers
  28.8% of specific movers vs 4.4% random / 0% prior; top-10 0.307 vs 0.168 prior; survives the random-neighbor control. First
  partial breach of the far-field wall. Worth wiring into the forecast. (`analogy.py`)
- **Time-resolved perturb-seq** — still the one wire that breaks the *compensation* wall (catch the hour-2–12 transient). A data acquisition, not an architecture. Highest remaining value.
- **Real 3D Hi-C TADs + metabolic messenger (Acetyl-CoA/FBA→chromatin)** — untested; the missing-wire test only ruled out *promoter* ATAC.
- **Fix the metric** — for weak KOs score recall at k = n_movers, not hits@10; part of the low average is a metric artifact.
