# PHASE 1 BRIEF — Mine the conditional vulnerability atlas for one validated surprise

Status: locked. Phase 1 first, phase 2 (two-tower predictor) after one surprise lands.

## The artifact
`/content/drive/MyDrive/cell_count_dynamics/multiorg/conditional_atlas/`
- `conditional_vulnerability_atlas.csv` — 16,045 (organism, gene, killing-condition, fit, t, gene-name, desc) entries across 48 organisms.
- `atlas_summary_by_organism.csv` — per-organism counts.
- Median family_frac of these genes = 0.14 (confirmed: this IS the conditional-zone residual; binary models structurally can't see them).

## What "a validated surprise" means (the bar)
A conditional vulnerability that is:
1. **Mechanistically explainable** — the gene + the killing condition make biochemical sense (a transporter for that compound, a stress response, a metabolic dependency).
2. **Not obvious from sequence/conservation** — has low family_frac (already true: median 0.14), AND its function isn't a textbook case of "this gene rescues this stress."
3. **Checkable against literature the model never saw** — searchable in PubMed/UniProt; if confirmed as known, it validates the method (still useful); if it appears unstudied or under-studied, it's a discovery candidate.
4. **Druggable framing**: ideally a (gene, antibiotic) pair where the antibiotic is clinically used. That's the antibiotic-potentiation / adjuvant target class.

## Phase 1 workflow (in order)

### Step 1: Atlas cleanup
- Drop `expGroup == 'motility'` (measurement-context artifact: needed-to-swim-on-agar, not therapeutic).
- Drop entries with `condition_1` empty AND `expGroup` uninformative (~1,200 rows).
- Drop conditions that look like measurement artifacts (mixed community, magnetic pulldown, in planta — keep separately for ecology framing).
- Result: a cleaner atlas focused on (gene × pharmacologically/nutritionally interpretable condition).

### Step 2: Stratify by therapeutic relevance
Three buckets, ranked by drug-discovery interest:
- **A. Antibiotic-potentiation targets**: gene lethal under a clinically-used antibiotic (Polymyxin, Bacitracin, Fusidic acid, Sisomicin, D-Cycloserine, β-lactams, fluoroquinolones, ...). Knocking out the gene *sensitizes* the cell to the drug → adjuvant target. **This is the highest-value bucket.**
- **B. Nutrient-niche / host-environment targets**: gene lethal under a condition mimicking host environment (low iron, specific amino acid as N source, anaerobic, low pH).
- **C. Stress-response targets**: gene lethal under metal / oxidative / membrane stress (cisplatin, copper, benzalkonium).

### Step 3: Surprise mining (the search, not just ranking)
For each bucket, rank candidates by:
- magnitude (most-negative fit),
- specificity (lethal in few conditions, neutral in many — high `conditional_score`),
- across-organism consistency (same gene lethal under same condition in ≥3 organisms = the signal is reproducible, not a one-off).

Then for the top ~20 candidates per bucket: hand-check against UniProt + PubMed. The shortlist for "one surprise" is the candidate that is mechanistically sensible BUT has no/few hits on the specific (gene, condition) pair in the literature.

### Step 4: Validation framing
Document ONE surprise as: (gene, organism, condition, fitness, mechanistic hypothesis, literature gap). That's the deliverable that converts the project from rigor to discovery.

## What phase 1 success looks like
- A clean, deduplicated atlas with the three buckets defined.
- A ranked candidate list of ~50 conditional vulnerabilities per bucket, cross-org consistent.
- ONE documented surprise with the four properties above.

Only after that lands do we move to phase 2.

## Phase 2 preview (do not start until phase 1 ships)
Two-tower model: gene_repr (ESM + family + structure) × condition_repr (Experiment metadata: media, condition_1, concentration, units, temperature, pH, aerobic) → predicted fit. Trained on the atlas (the 16,045 entries + the negative/neutral background from GeneFitness). HONEST validation: leave-one-organism-out AND leave-one-condition-out separately, because the real test is "can you predict for a bacterium and stress combination you've never measured."

## Leak/discipline rules unchanged
Atlas-mining: no model risk. For phase 2: condition tower must NOT see the held-out condition's metadata at training; gene tower must NOT see the held-out organism's labels. Standard hygiene, but now with TWO axes of holdout, not one.
