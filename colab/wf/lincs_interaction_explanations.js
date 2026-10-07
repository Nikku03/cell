export const meta = {
  name: 'lincs-interaction-explanations',
  description: 'Test candidate explanations for the LINCS gene x line interaction against a shared verified harness',
  phases: [
    { title: 'Candidates', detail: 'each hypothesis implemented against the harness and scored' },
    { title: 'Verify', detail: 'adversarially check anything that clears the bar' },
  ],
}

const HARNESS = `
CONTEXT YOU MUST NOT RE-DERIVE. Working dir /home/user/cell. A shared harness exists at
colab/lincs_harness.py and it is VERIFIED. Read it first. Use it exactly as-is. Do not write your
own loader, your own residual, or your own split.

  import sys; sys.path.insert(0,'colab'); import lincs_harness as H
  D = H.load()                                   # ~90s, 33,197 (gene,line) profiles, 978 landmarks
  scores = H.evaluate(D, my_features)            # held out BY CELL LINE, returns per-pair corrs
  base   = H.evaluate(D, H.expression_baseline)  # the reference: 0.0029

Your features function has signature features(gene, line, D) -> numpy array (D['NL'], k) float32,
or None to skip. gene is a HUGO symbol string, line is one of D['LINES'].

WHAT IS BEING EXPLAINED. LINCS L1000 shRNA knockdowns across 9 cancer cell lines. The response to
a knockdown splits as 27.6% gene main effect, 3.8% cell-line main effect, 68.7% GENE x LINE
INTERACTION. The harness target is that interaction: the residual after gene mean and line mean.
It reproduces at 0.2487 across DISJOINT shRNA constructs, so 0.2487 is the CEILING - nothing can
explain more than two independent reagents agree on. Already tested and FAILED: expression context
0.0029, signed OmniPath edges -0.0000, ENCODE chromatin +0.0000.

RULES THAT MATTER MORE THAN YOUR RESULT.
- Report the number you get, including if it is zero or negative. A null result is the expected
  outcome and is valuable. Do NOT tune features until something passes.
- Always report your arm AND H.expression_baseline run in the same session, so the comparison is
  like-for-like.
- If your data source does not cover the 9 lines, say so with counts rather than silently
  imputing. The 9 lines are PC3 MCF7 VCAP A375 HA1E A549 HT29 HEPG2 HCC515.
- H.evaluate takes a few minutes. Use max_train_per_line=300 to keep it tractable. Run it at most
  3 times; you are testing a hypothesis, not searching.
- Do not modify colab/lincs_harness.py or anything in colab/. Write scratch code under
  /tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/wf/ .
- Do not git commit or push.

DATA AVAILABLE (scratchpad = /tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad):
  depmap/OmicsSomaticMutationsMatrixDamaging.csv   gene x cell-line damaging mutation matrix
  depmap/OmicsSomaticMutationsMatrixHotspot.csv    gene x cell-line hotspot mutation matrix
  depmap/OmicsAbsoluteCNGene.csv                   gene x cell-line absolute copy number
  depmap/gene_effect.npz                           CRISPR gene effect, E/genes/lines arrays
  depmap/Model.csv                                 cell line metadata incl OncotreeLineage
  lincs/line_map.json                               LINCS line name -> DepMap ModelID
  depmap_expr_aligned.npz                           XE/lines/genes expression
  biogrid_hs_edges.tsv.gz                           protein interaction edges, cols: A B type thr
  para/paralogs.tsv                                 symbol, paralogue symbol, percent identity
  reg/op_2022.tsv                                   OmniPath signed directed edges
DepMap files are keyed by ModelID; use lincs/line_map.json to map. Column headers in the Omics
matrices are usually "SYMBOL (ENTREZID)" - strip the parenthetical.
`

const SCHEMA = {
  type: 'object',
  required: ['hypothesis', 'score', 'baseline', 'delta', 'n_pairs', 'coverage', 'verdict', 'notes'],
  properties: {
    hypothesis:  { type: 'string', description: 'one sentence: the mechanism, not the feature list' },
    score:       { type: 'number', description: 'mean held-out correlation of YOUR arm' },
    baseline:    { type: 'number', description: 'H.expression_baseline in the same session' },
    delta:       { type: 'number', description: 'score minus baseline' },
    n_pairs:     { type: 'number', description: 'number of held-out (gene,line) pairs scored' },
    coverage:    { type: 'string', description: 'how much of the 9 lines / genes your data covers, with counts' },
    verdict:     { type: 'string', enum: ['explains', 'does_not_explain', 'could_not_test'] },
    notes:       { type: 'string', description: 'anything that would change how the number is read: confounds, defects found, why coverage is what it is' },
  },
}

const CANDIDATES = [
  { key: 'damaging-mutation',
    ask: `THE REQUESTED TEST. Hypothesis: a knockdown behaves differently when the pathway it feeds is already broken by a damaging mutation in that cell line. Build features from depmap/OmicsSomaticMutationsMatrixDamaging.csv. At minimum include: whether the KNOCKED-DOWN gene is damaged in this line, whether each LANDMARK is damaged in this line, the line's total damaged-gene burden, and the interaction between the knocked-down gene's damage status and each landmark's. Report how many of the 9 lines are present in the mutation matrix.` },
  { key: 'hotspot-mutation',
    ask: `Hypothesis: recurrent activating hotspot mutations (KRAS G12, BRAF V600, PIK3CA and so on) rewire the response, and unlike damaging mutations these are gain-of-function so their effect should differ in kind. Use depmap/OmicsSomaticMutationsMatrixHotspot.csv. Include the knocked-down gene's hotspot status, each landmark's, the line's hotspot burden, and their interaction. Also report WHICH hotspot genes actually differ across the 9 lines - if fewer than about 20 genes vary, say so, because the hypothesis is then being carried by a handful of features.` },
  { key: 'copy-number',
    ask: `Hypothesis: gene dosage sets the response - a landmark present at high copy number has more to lose, and a knocked-down gene at low copy number is already partly depleted. Use depmap/OmicsAbsoluteCNGene.csv. Include the knocked-down gene's copy number in this line, each landmark's, and their interaction, all z-scored ACROSS the 9 lines so you are measuring relative dosage rather than gene identity.` },
  { key: 'crispr-dependency',
    ask: `Hypothesis: what a knockdown does depends on whether that gene is ESSENTIAL in this particular line - knocking down a gene the line depends on perturbs it globally, knocking down a dispensable one does little. Use depmap/gene_effect.npz (keys E, genes, lines; E is lines x genes CRISPR gene effect). Include the knocked-down gene's gene-effect in THIS line, its gene-effect z-scored across the 9 lines, each landmark's gene-effect in this line, and the interaction. This is the strongest a priori candidate because it is a functional measurement in the same cell line rather than an annotation.` },
  { key: 'paralogue-buffering',
    ask: `Hypothesis: a knockdown is buffered where the gene has a paralogue that is well expressed in THAT line, so the same knockdown does less there. Use para/paralogs.tsv (columns: symbol, paralogue symbol, percent identity) with depmap_expr_aligned.npz. Include the max and mean expression of the knocked-down gene's paralogues in this line, z-scored across the 9 lines, the paralogue count, and interactions with landmark expression. Note that loops 239-240 tested a version of this on DepMap FITNESS and found the coefficient had the ANTI-buffering sign; report the sign you get and do not assume it.` },
  { key: 'lineage-and-line-identity',
    ask: `Hypothesis - and this one is deliberately close to a null. If the interaction is explained simply by WHICH line it is, rather than by anything measured about the line, then a one-hot of cell line identity crossed with coarse gene properties should capture it. Build features from a one-hot of OncotreeLineage (depmap/Model.csv) and from a one-hot of the cell line itself, crossed with each landmark's mean response magnitude across lines. This is important as a REFERENCE POINT: it tells us whether the interaction is learnable at all from line identity alone, which bounds how much any mechanistic explanation could ever add. Report it plainly whichever way it comes out.` },
  { key: 'landmark-responsiveness',
    ask: `Hypothesis: the interaction is mostly about WHICH LANDMARKS are volatile in which line - some genes are simply more reactive in some cell lines regardless of what was knocked down. Build features from the landmark's own response statistics computed on TRAINING lines only (its variance across knockdowns within each training line, its mean absolute response), plus the held-out line's expression of that landmark. CRITICAL: any statistic you compute for the held-out line must come from OTHER genes' knockdowns in that line, never from the pair being scored, or you leak. State clearly in your notes how you avoided leaking.` },
]

phase('Candidates')
const results = await pipeline(
  CANDIDATES,
  c => agent(`${HARNESS}\n\nYOUR HYPOTHESIS (${c.key}):\n${c.ask}\n\nImplement it, run H.evaluate, and report the structured result. Remember: a null is the expected outcome here and reporting it accurately is the job.`,
             { label: `test:${c.key}`, phase: 'Candidates', schema: SCHEMA }),
)

const scored = results.filter(Boolean)
log(`${scored.length}/${CANDIDATES.length} candidates returned`)
const passes = scored.filter(r => r.verdict === 'explains' && r.delta >= 0.01)
log(`${passes.length} clear the +0.01 bar and go to adversarial verification`)

phase('Verify')
const verdicts = passes.length ? await parallel(passes.map(r => () =>
  agent(`${HARNESS}\n\nAn agent reported this result on the LINCS interaction:\n\n${JSON.stringify(r, null, 2)}\n\nYour job is to REFUTE it. This project's history is that positives evaporate under scrutiny: loop 225 reported an MLP win that two later loops reversed; loop 228's H2 passed on a shared-denominator artefact; loop 253's F1 measured a different residual than the arms it was compared against. Look specifically for: leakage (any feature computed using the held-out pair's own data), a feature that encodes the target by construction, coverage so thin that a handful of cell lines carry the whole effect, and whether the effect survives a control where the CELL LINE IDENTITY is shuffled (H.evaluate accepts shuffle_line=True). Re-run what you need to. Default to refuted=true if you cannot confirm it. Report honestly if it survives.`,
        { label: `refute:${r.hypothesis.slice(0, 28)}`, phase: 'Verify',
          schema: { type: 'object', required: ['refuted', 'reason', 'shuffled_control', 'recomputed_delta'],
                    properties: { refuted: { type: 'boolean' },
                                  reason: { type: 'string' },
                                  shuffled_control: { type: 'string', description: 'what the shuffle_line=True control gave' },
                                  recomputed_delta: { type: 'number' } } } })
)) : []

return {
  ceiling: 0.2487,
  expression_reference: 0.0029,
  candidates: scored,
  verified: verdicts.filter(Boolean),
}
