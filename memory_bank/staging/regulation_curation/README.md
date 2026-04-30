# Regulation curation staging — Phase R2a

This directory holds **machine-acquired, unreviewed** regulator
candidates for Syn3A. Nothing here is biology; it is a list of things
worth a human looking at in a Phase R2b curation session.

## Contract

- Every entry below carries `confidence: candidate`.
- The production YAML at
  `cell_sim/data/regulation_network_syn3a.yaml` is **unchanged** by
  this phase. It still ships with all four lists empty.
- Candidates do not become biology until a Phase R2b reviewer reads
  the provenance, applies judgment, and either:
  1. Promotes the entry to the production YAML with `confidence:
     measured` or `inferred` and full provenance, **or**
  2. Rejects it (with the rejection reason recorded in the Phase R2b
     review log).

## Files

| File | Acquired |
|------|----------|
| `sigma_factor_candidates.yaml` | 1 |
| `transcription_factor_candidates.yaml` | 3 |
| `riboswitch_candidates.yaml` | 0 (channel limitation, see log) |
| `two_component_candidates.yaml` | 0 |
| `acquisition_log.md` | (run record) |

The candidate counts above match expectations for a minimal Mycoplasma
genome: very few sigma factors, a small TF complement, and rare-to-
absent riboswitches and two-component systems.

## Provenance contract per candidate

Every populated candidate dict carries:

| Field | Meaning |
|-------|---------|
| `provenance_channel` | `genbank_annotation` for this phase. |
| `genbank_accession` | `CP016816.2` (Syn3A reference assembly). |
| `protein_id` | GenBank `/protein_id` for the CDS. |
| `inference_xref` | GenBank `/inference` value (RefSeq or SwissProt). |
| `product_annotation` | Verbatim `/product` string. |
| `matched_keyword` | The keyword that selected this CDS into the candidate set. |
| `e_value` | `null` when acquired by annotation parsing rather than HMM/BLAST search. |
| `confidence` | Always `candidate`. |

When a Phase R2b promotion happens, the reviewer rewrites
`confidence` to `measured` or `inferred` according to the Phase R1
loader's allowed set
(`cell_sim/layer4_regulation/network_loader.py`) and adds the three
production-required provenance fields (`source`, `confidence`,
`curated_on`). Until then the entry stays here.

## Why only the GenBank annotation channel?

The session brief specified Pfam HMM searches (PF00140 sigma70, etc.)
and Rfam Infernal `cmscan`. Neither path was reachable from the
acquisition sandbox:

- **No tools installed.** `hmmscan`, `cmscan`, and `blastp` are
  absent.
- **No HMM/CM databases.** `Pfam-A.hmm` and `Rfam.cm` are not on
  disk.
- **No network route to fetch them.** EBI FTP, EBI REST, NCBI,
  UniProt, and rfam.org all return HTTP 403
  `host_not_allowed` from the proxy. PyPI and
  `raw.githubusercontent.com` are reachable but neither serves
  Pfam-A or Rfam directly. See `acquisition_log.md` for the
  raw probe transcript.

The fallback channel — local GenBank annotation parsing — has full
provenance for every candidate (RefSeq or SwissProt accession +
verbatim product string) and is fully reproducible (re-run
`python scripts/acquire_regulation_candidates.py`). It cannot
provide HMM E-values, and it cannot recover regulators with
ambiguous product strings ("hypothetical protein"). The Phase R2b
reviewer should treat this as a starting point, not a complete
list.

Riboswitches are a special case: they are RNA-structure features
in 5' UTRs, not CDS annotations. The GenBank channel cannot
recover them at all. The riboswitch file is intentionally empty
with the rationale documented in its header.

## What Phase R2b is for

For each candidate file, the reviewer must:

1. **Read the source provenance.** What is the JCVI annotator's
   product string? What does the RefSeq / SwissProt xref say about
   the closest characterized homolog? Is it a Mycoplasma sequence
   or a distant homolog?
2. **Apply biological judgment.** Genome reduction can leave
   pseudogenes that match Pfam HMMs but no longer function. Does
   the annotation feel credible? Does the gene name match? Is the
   product string a hedge ("uncharacterized") or specific?
3. **Decide.** Promote, reject, or defer. Promotions land in the
   production YAML with the three required Phase R1 provenance
   fields (`source`, `confidence`, `curated_on`). Rejections are
   logged with the rejection reason. Deferrals stay in this
   directory with a note.
4. **Document.** Each decision recorded with reasoning, not just
   yes/no. A reviewer six months from now must be able to
   reconstruct the call.

This is research work, not engineering. Don't rush it.

## Hard rules for the reviewer

- The production YAML is the contract. Never edit it directly with
  candidate data; promotions only happen after the per-entry review.
- Never rewrite `confidence: candidate` to `measured` without a
  direct experimental measurement on Syn3A or a very close
  homolog. Sequence similarity alone is `inferred` at best.
- If a candidate's provenance is just a JCVI annotator's hedge
  ("Uncharacterized transcriptional regulator"), the maximum
  defensible promotion is `confidence: inferred` with a caveat
  recording that the call rests on annotation alone.
- The strict path (Pfam HMM E-values, Rfam structure scores)
  remains the right way to do this. If a future session has
  network access to EBI / Pfam / Rfam, re-running with that
  channel will produce a richer candidate set with quantitative
  provenance. The annotation channel is the floor.
