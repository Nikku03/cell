# Phase R2b Curation Review Log

**Date:** 2026-04-30
**Reviewer:** Naresh Chhillar
**Scope:** Review the 4 candidates staged in Phase R2a; promote credible
entries to the production YAML at
`cell_sim/data/regulation_network_syn3a.yaml`; document deferrals.

## Decisions

### Promoted to production

| Candidate | Type | Confidence | Source basis | Rationale |
|---|---|---|---|---|
| `JCVISYN3A_0407` (`rpoD`) | Sigma factor | `inferred` | JCVI annotation + SwissProt sigma-70 homology | Canonical housekeeping sigma; unambiguous annotation; required for any transcription model |
| `JCVISYN3A_0525` (`mraZ`) | Transcription factor | `inferred` | JCVI annotation + SwissProt MraZ homology | Cell-division regulator; gene-name conservation; defensible inferred status |

The two promoted entries are now in `cell_sim/data/regulation_network_syn3a.yaml` with the loader's required provenance fields (`source`, `confidence: inferred`, `curated_on: 2026-04-30`) and a `parameters_status: not_specified` flag indicating that kinetic constants have not been set yet — wiring (Phase R3) will need to either supply defaults or treat absent kinetics as a no-op.

### Deferred (remain in staging)

| Candidate | Type | Reason for deferral |
|---|---|---|
| `JCVISYN3A_0042` | TF candidate (HTH motif) | Uncharacterized; no target genes mapped |
| `JCVISYN3A_0620` | TF candidate (HTH motif) | Uncharacterized; no target genes mapped |

Both deferred entries have a `deferral_decision` block appended in
`memory_bank/staging/regulation_curation/transcription_factor_candidates.yaml`
with `next_review_trigger: target_gene_mapping_session`. They were not
moved or deleted; a future R2b-style review can revisit them once
target inference data exists.

## Why no `measured` confidence

None of the promoted entries qualify for `confidence: measured`. Both
rest on cross-species sequence homology (sigma-70 family for `rpoD`;
MraZ family for `mraZ`) plus the JCVI annotator's gene name. There is
no direct ChIP-seq, biochemical pull-down, or in vitro binding
measurement performed on Syn3A specifically. `inferred` is the
honest ceiling for both.

## Notes on the regulation network

The promoted network has 2 entries: 1 sigma factor and 1 transcription
factor. Both target gene lists are either implicit (`rpoD`,
housekeeping = all genes by default) or empty (`mraZ`, target
inference deferred).

This is a small but credible regulation network for Syn3A. Padding the
YAML with weakly-supported entries to make the network "look complete"
would compromise the integrity of the curation. Syn3A is a minimal
cell with sparse regulation; sparse entries are the expected result.

## What this means for downstream work

Phase R3 (wiring regulation into production sweep) can proceed with
the current 2-entry network if desired. The expected MCC change at
flag-on is small or null, because:

- A single housekeeping sigma factor with no competition produces no
  differential transcription dynamics.
- A single TF with no defined targets has no regulatory effect.
- The simulator's existing predictions (v15 / v16) do not depend on
  regulatory dynamics; adding minimal regulation will not meaningfully
  change them.

This is consistent with what the field knows about Syn3A: it's a
minimally-regulated organism, and a faithful simulator should reflect
that.

## Two-component systems methodology gap

Phase R2a returned 0 two-component system candidates. This used a
keyword-based detection channel rather than strict Pfam HMM search
against PF07568 (HisKA) and PF00072 (Response_reg). The zero result
could mean either:

1. Syn3A genuinely has no two-component systems (consistent with its
   minimal regulatory architecture).
2. The keyword channel missed candidates that strict Pfam search
   would catch.

A future R2a-strict session, using direct Pfam HMM search from a
network-enabled environment (the R2a sandbox had EBI / Pfam blocked
at the proxy), would resolve this. Until then, the current "no
two-component systems" stance is provisional, not confirmed.

## Audit trail per promoted entry

### `JCVISYN3A_0407` — `rpoD`

- **GenBank accession:** `CP016816.2` (Syn3A reference assembly,
  12-APR-2018 submission).
- **Protein ID:** `AVX54771.1`.
- **`/inference=`:** `EXISTENCE: similar to AA
  sequence:RefSeq:WP_013447807.1`.
- **`/product=`:** "RNA polymerase subunit sigma".
- **Pfam family assignment:** PF00140 (Sigma70_r2 — DNA binding
  domain 2 of sigma-70 family). Assigned by family-class match in
  the curator's notes; not from a direct HMM run in this session.
- **Gene class assignment:** `housekeeping`. Justified by being the
  only sigma factor in the genome — the alternative-sigma case
  cannot apply when there is no alternative. The Phase R3 wiring
  step will need to decide what "housekeeping = all genes" means
  for the sigma_competition rule (likely: skip the rule when only
  one sigma is present; producing zero gain over baseline is the
  honest outcome).

### `JCVISYN3A_0525` — `mraZ`

- **GenBank accession:** `CP016816.2`.
- **Protein ID:** `AVX54841.1`.
- **`/inference=`:** `EXISTENCE: similar to AA
  sequence:SwissProt:Q6MT21.1`.
- **`/product=`:** "Cell division/cell wall cluster transcriptional
  repressor".
- **Pfam family assignment:** PF02381 (MraZ family DNA-binding
  domain). Assigned by family name in the curator's notes; not from
  a direct HMM run.
- **Target genes:** intentionally empty `[]`. In E. coli /
  B. subtilis MraZ regulates the dcw cluster (FtsZ, FtsA, FtsW,
  MurC-G); the analogous Syn3A genes are likely targets but the
  regulatory linkage is not directly established. Comparative-
  genomics target inference is deferred.

## Next session

**Phase R3** is optional: wire regulation behind
`enable_regulation: bool = False` flag, run bit-identity test against
v16, measure v17 MCC. Realistic expectation: v17 MCC ≈ v16 MCC ±
stochastic noise, because the 2-entry network has no defined
regulatory effect on the existing detector's metabolic-pool +
annotation-prior decision surface.

**Phase R2a-strict** is optional: re-run R2a candidate acquisition
with strict Pfam HMM search against the four regulator-class Pfams
from a network-enabled environment. Documented in `FUTURE_WORK.md`.
