# Acquisition log — Phase R2a regulation candidates

This log records every acquisition attempt: which tool / database,
when, with what query parameters, what came back. The intent is that a
Phase R2b reviewer (or a later replication attempt) can verify
provenance and re-run the queries. Entries are append-only.

## 2026-04-30 ~16:00 UTC — Tooling and network availability probe

The session brief specified HMM-based acquisition via `hmmscan` /
`cmscan` against Pfam-A and Rfam. Before doing anything else I probed
the acquisition sandbox for the tools and network routes that path
needs.

### Tools

```
which hmmscan hmmsearch jackhmmer  -> not found
which cmscan cmsearch              -> not found
which blastp psiblast rpsblast     -> not found
```

`pyhmmer` is pip-installable from PyPI but still requires Pfam HMM
model files to produce results. None are on disk:

```
find / -name "Pfam-A.hmm*"   -> (no matches)
find / -name "Rfam.cm*"      -> (no matches)
```

### Network

```
curl -I https://www.ebi.ac.uk/Tools/hmmer/api/
HTTP/1.1 403 Forbidden
x-deny-reason: host_not_allowed
```

| Host | HTTP status | Note |
|------|-------------|------|
| `www.ebi.ac.uk`            | 403 | proxy denial: `host_not_allowed` |
| `ftp.ebi.ac.uk`            | 403 | (Pfam-A.hmm.gz, Rfam.cm.gz both blocked) |
| `www.ebi.ac.uk/interpro/`  | 403 |
| `rfam.org`                 | 403 |
| `www.ncbi.nlm.nih.gov`     | 403 |
| `rest.uniprot.org`         | 403 |
| `rest.ensembl.org`         | 403 |
| `eddylab.org`              | 403 | (HMMER source distribution host) |
| `zenodo.org`               | 403 |
| `pypi.org/simple/`         | 200 | reachable |
| `api.github.com`           | 200 | reachable |
| `raw.githubusercontent.com`| 200 | reachable |

The proxy implements a host allowlist. Bio-database hosts are
excluded; PyPI and GitHub raw are included.

### Decision

Per the session brief's halt-or-fallback rule ("if a database is
genuinely unavailable, document the failure and skip that candidate
type rather than fabricate entries"), the strict Pfam HMM and Rfam
Infernal paths are unavailable in this sandbox. I used the local
GenBank annotation channel (Syn3A flat file
`cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb`,
GenBank accession CP016816.2, version 12-APR-2018) as a fallback. The
fallback gives full per-candidate provenance (RefSeq / SwissProt xref
+ verbatim `/product=` string + GenBank `/protein_id`) but cannot
provide HMM E-values; every E-value field is `null` and every entry
is tagged `provenance_channel: genbank_annotation`.

The same strict path will be the right way to do this in a future
session that has EBI / Pfam / Rfam network access.

## 2026-04-30 ~16:05 UTC — Sigma factor acquisition (fallback channel)

Source: `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb`
(CP016816.2). Total CDS features: 458.

Query: case-insensitive substring match of each CDS's `/product=`
qualifier against the keyword set
`{sigma factor, sigma-70, sigma 70, sigma subunit, RNA polymerase
sigma}`, plus `/gene=` match against
`{rpoD, sigA, sigB, sigH, sigE, rpoS}`. Anti-sigma hits (annotations
containing the substring "anti-sigma" or "anti sigma") explicitly
excluded.

| Step | Count |
|------|-------|
| CDS features scanned                                       | 458 |
| `/product=` keyword matches                                | 1   |
| `/gene=` matches not already in the product hit            | 0   |
| Anti-sigma exclusions                                      | 0   |
| Final candidates written to `sigma_factor_candidates.yaml` | 1   |

The single hit is `JCVISYN3A_0407` (`rpoD`, "RNA polymerase subunit
sigma"). This is the canonical housekeeping sigma 70 expected in any
bacterial genome and the most-credible sigma candidate for Phase R2b
promotion. Per the spec's expected range (1–3), 1 is at the low end;
this is consistent with the minimal-genome expectation.

## 2026-04-30 ~16:06 UTC — TF acquisition (fallback channel)

Source: same GenBank file. Total CDS features: 458.

Query: case-insensitive substring match of each CDS's `/product=` against
the keyword set `{transcriptional regulator, transcription factor,
transcriptional repressor, transcriptional activator, helix-turn-helix,
hth-type, MerR family, LysR family, TetR family, AraC family, Fur
family, Spx, DNA-binding response regulator}`. Hits whose product
also contained "histidine kinase", "sensor kinase", or "response
regulator" were excluded — those go to the two-component channel.

| Step | Count |
|------|-------|
| CDS features scanned                                          | 458 |
| `/product=` keyword matches                                   | 3   |
| Two-component-overlap exclusions                              | 0   |
| Response-regulator exclusions                                 | 0   |
| Final candidates written to `transcription_factor_candidates.yaml` | 3 |

The three candidates:

| Locus tag       | Gene  | Product (verbatim)                                        |
|-----------------|-------|-----------------------------------------------------------|
| JCVISYN3A_0042  | —     | "Uncharacterized transcriptional regulator"               |
| JCVISYN3A_0525  | mraZ  | "Cell division/cell wall cluster transcriptional repressor" |
| JCVISYN3A_0620  | —     | "Uncharacterized transcriptional regulator"               |

`mraZ` is a well-characterized cell-division-cluster regulator
conserved across bacteria; the two "Uncharacterized" entries rest on
JCVI's annotation alone. Per the spec's expected range (5–15), 3 is
below the low end — consistent with Mycoplasma genome reduction
having stripped most of the canonical TF families. A future strict-
HMM run would likely add a small number of additional candidates
(e.g. WhiA, YebC-family) that this annotation channel cannot find by
keyword.

Target gene prediction was NOT performed at this phase
(`target_genes_predicted: []` in every entry). ChIP-seq data does not
exist for Syn3A; comparative genomics inference is Phase R2b
curation work.

## 2026-04-30 ~16:07 UTC — Riboswitch acquisition (skipped)

The strict Rfam Infernal `cmscan` path is unavailable: `cmscan` is
not installed, the Rfam covariance model file is not on disk, and the
Rfam web endpoints (rfam.org, EBI Rfam FTP) are blocked at the proxy.

The GenBank annotation channel cannot substitute here. Riboswitches
are RNA-structure features in 5' UTRs, not CDS annotations; they do
not appear in `/product=` strings. Skipping the channel with zero
candidates is the honest outcome.

| Step | Count |
|------|-------|
| Candidates written to `riboswitch_candidates.yaml` | 0 |

The empty file is committed with a documented rationale in its
header. A future session with Infernal + Rfam access should re-run
this channel against the 5' UTRs of Syn3A CDSes.

## 2026-04-30 ~16:08 UTC — Two-component system acquisition (fallback channel)

Source: same GenBank file.

Query: case-insensitive substring match of each CDS's `/product=`:

* Sensor kinases: `{histidine kinase, sensor kinase, sensor histidine,
  two-component sensor, two component sensor}`
* Response regulators: `{response regulator, two-component response,
  two component response}`

Sensor / RR pairs identified by genomic adjacency (any sensor whose
genome position is within 10 kb of any RR — generous; real
two-component pairs are typically <2 kb apart, but the operon order
in Mycoplasma genomes is not always predictable).

| Step | Count |
|------|-------|
| CDS features scanned                                | 458 |
| Sensor histidine kinase keyword matches             | 0   |
| Response regulator keyword matches                  | 0   |
| Paired sensor + RR within 10 kb                     | 0   |
| Orphan sensors / regulators                         | 0   |
| Final entries in `two_component_candidates.yaml`    | 0   |

Zero candidates. This is consistent with the minimal-Mycoplasma
expectation (the spec's "Mycoplasma has limited two-component
signaling" caveat). Syn3A retained no obvious two-component system
in the JCVI minimization. A future strict-HMM run against the
PF07568 / PF00512 / PF02518 / PF00072 families would let us
distinguish "really absent" from "present but missed by annotation
keyword match" — that is Phase R2b homework.

## Summary

| Class         | Acquired | Spec expected range |
|---------------|----------|---------------------|
| Sigma factors        | 1 | 1–3 |
| Transcription factors | 3 | 5–15 |
| Riboswitches         | 0 | 0–3 |
| Two-component systems | 0 | 0–2 |

All counts are at or below the low end of the spec's expected ranges.
This is the expected pattern for the JCVI-minimized genome. The
production YAML at `cell_sim/data/regulation_network_syn3a.yaml`
remains unchanged (still all four lists empty).

## Reproducibility

Re-run from the repo root:

```
python scripts/acquire_regulation_candidates.py
```

The script is deterministic given the same GenBank input. It rewrites
the four candidate YAML files in place and prints a summary. No
network access required (and none possible for the strict path under
the current sandbox proxy).
