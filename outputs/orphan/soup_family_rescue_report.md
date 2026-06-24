# Soup Family Rescue — Final Report

## 1. The question
Can orthologous-family conservation rescue true essentials from the **SOUP** — the set of genes the per-organism model left ambiguous (predicted essentiality `p in [0.30, 0.70]`, brightness `< 0.85`)? The hope: a leak-clean cross-clade conservation signal will re-rank the soup so that the genes the base model was unsure about, but which are universally essential elsewhere, surface as high-confidence calls. This would convert the soup from a deployment liability into recoverable yield.

## 2. The method
For each gene in the soup we compute a `rescue_score = 0.7 * og_universality_loco + 0.3 * (family_breadth / 30)` (clamped to 1.0). `og_universality_loco` is the fraction of orgs in the gene's OG (min 5 members) where it is essential, computed with **the entire target clade held out** (`pseudomonas` for Putida, `escherichia` for Keio) — not just the target organism — so the signal cannot leak through sister strains. `family_breadth` is the number of distinct clades the OG spans. We rank the soup by `rescue_score` and evaluate against held-out experimental essentiality, reporting top-K precision at the K that maximizes precision subject to K being meaningful, plus coverage at a precision floor of 0.80.

## 3. Results

| Organism | Soup size | Soup essentials | Base prec. | Top-K prec. | Lift (pp) | Cov @ P=0.80 |
|---|---:|---:|---:|---:|---:|---:|
| beril_Putida | 1412 | 464 | 0.329 | 0.677 | +34.8 | 0.739 |
| beril_Keio | 824 | 79 | 0.096 | 0.203 | +10.7 | 0.000 |
| *third organism* | — | — | — | — | — | — |

The "across 3 organisms" framing is overstated: only two runs produced artifacts under `/home/user/cell/outputs/orphan/soup_rescue_*.json`. No script, no JSON exists for a third organism.

## 4. Adversarial verdicts

- **beril_Putida — TRIVIAL (not REAL).** Leak hygiene is clean, but **every top-15 candidate has `og_rate_loco == 1.0` and `family_breadth >= 32`**, so both score terms saturate at the ceiling and `rescue_score` is a constant 1.0 across the entire fully-conserved-elsewhere pool. The breadth term contributes only as a tiebreaker. ~60% of the top-15 are textbook trivial core (3x ribosomal proteins, 3x ATP synthase subunits, DNA Pol III delta, TsaD). These should never have entered the soup — admitting them is a **SOUP-definition leak**, not a rescue. The headline +34.8pp lift is real arithmetic but the wrong story.
- **beril_Keio — WEAK.** Leak-clean, breadth term is actually doing ranking work (scores 0.81–0.91, unsaturated), but it ranks the wrong things: top-5 includes SodB, PurA, DdlA — all conserved-essential elsewhere yet experimentally **non-essential in Keio** due to paralog redundancy. Lift is +10.7pp on a 9.6% base (~2.1x); `cov@P80 = 0`. Real signal, wrong working point.
- **Third organism — NO_ARTIFACT.** Cannot be verified.

## 5. Deployment implications
- **Putida:** at the P=0.80 floor we recover 343 candidates covering 74% of soup essentials. The catch: tighten the soup definition first (exclude OGs with loco-essentiality > 0.95) — then re-measure, because most of this yield is the model under-confidently re-discovering its own conservation prior. Expect the true *novel* yield after that fix to be substantially smaller than 343.
- **Keio:** no operating point clears 80% precision. **Do not promote any Keio soup gene to "essential" on family evidence alone.** Treat as a triage input for wet-lab follow-up only.
- **Third organism:** no promotion possible — artifact missing.

## 6. Honest limitations
1. **n = 2, not 3.** The third-organism claim is unsupported.
2. **Putida lift is partially a tautology.** Top-K is dominated by saturated-score trivial core; the breadth term is decorative there. A conservation-only baseline would catch the same hits.
3. **Soup definition is leaky upstream.** Genes with OG loco-essentiality > 0.95 should not be classified ambiguous; they are inflating apparent rescue performance.
4. **Conservation does not transfer through paralog redundancy** (Keio: DdlA/DdlB, SodB, PurA). The method has a hard ceiling in clades with strong functional backups.
5. **Recommended next steps:** (a) ablate the breadth term and re-report — Putida likely unchanged, Keio likely slightly worse; (b) tighten soup membership to exclude pre-saturated OGs; (c) run a true third organism before any generalization claim.

Artifacts: `/home/user/cell/outputs/orphan/soup_rescue_beril_Putida.json`, `/home/user/cell/outputs/orphan/soup_rescue_beril_Keio.json`, `/home/user/cell/colab/soup_rescue_beril_Putida.py`, `/home/user/cell/colab/soup_rescue_beril_Keio.py`.

## 7. Decisive ablation (post-verification)

The adversarial verdict claimed Putida's lift was "trivial" — entirely the
model under-confidently re-discovering obvious conserved core that leaked into
the soup. The direct ablation shows that's only **partly** true:

| test | result |
|---|---|
| Putida soup genes with og_rate>0.95 ("obvious") | **72 / 1412 = 5%** (71/72 truly essential) |
| Remove the obvious → genuinely-ambiguous soup | 1340 genes, 393 essential (base 0.293) |
| Rescue on ambiguous-only, full score | **topK precision 0.618 (+32.5pp over base)** |
| Rescue on ambiguous-only, og_rate ONLY (breadth ablated) | **0.636** |

Two conclusions:
1. **The rescue is REAL, not a soup-leak artifact.** Removing the obvious
   (og_rate>0.95) genes barely dents it — base rate 0.329→0.293, topK
   precision still +32.5pp. The verifier over-weighted the 5% trivial leak.
2. **The breadth/family-classification term is DEAD WEIGHT** — og_rate-alone
   (0.636) slightly BEATS the full score (0.618). The "family classification"
   add-on the experiment was built to test contributes nothing; the entire
   lift is cross-organism conservation (og_universality_loco), i.e. Wheel 2's
   existing signal applied to the soup.

**Net:** family/breadth scoring does NOT help. But conservation re-ranking of
the soup IS a real, deployable lift on a conservation-friendly organism
(Putida: recover ~74% of soup essentials at P≥0.80), and a near-useless one on
a paralog-redundant organism (Keio: no point clears P≥0.80). The honest
takeaway: the soup is recoverable exactly to the extent the organism's
essentiality is conserved — not via protein-family classification.
