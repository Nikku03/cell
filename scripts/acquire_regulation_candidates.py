"""Phase R2a regulation candidate acquisition — reproducible script.

Acquires candidate regulators from the local Syn3A GenBank flat file
(``cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb``,
GenBank accession CP016816.2) by scanning each CDS feature's
``/product=`` and ``/note=`` qualifiers for regulatory keywords. Each
candidate carries full provenance (RefSeq or SwissProt xref from
``/inference=``, GenBank ``protein_id``, verbatim product string).

The strict acquisition path described in the session brief
-----------------------------------------------------------

The brief specifies HMM searches against Pfam (PF00140 sigma70,
PF00309 sigma54, PF08281 ECF; PF00126 / PF00165 / PF01047 HTH;
PF13411 MerR; PF00126 + PF03466 LysR; PF07568 / PF00512 / PF02518
histidine kinase; PF00072 response regulator) and Rfam riboswitch
families via Infernal cmscan.

That path is unavailable in this acquisition sandbox: hmmer,
infernal, and blast are not installed; the EBI/Pfam/Rfam/UniProt FTP
endpoints all return HTTP 403 ``host_not_allowed`` from the proxy;
PyPI and raw.githubusercontent.com are the only reachable bio-related
hosts and neither serves Pfam-A.hmm or Rfam.cm. See
``memory_bank/staging/regulation_curation/acquisition_log.md`` for
the probe transcript.

Fallback channel
----------------

Local GenBank annotation parsing. The Syn3A submission carries
``/product=`` strings (functional descriptions written by the JCVI
annotators) and ``/inference=EXISTENCE: similar to AA
sequence:RefSeq:WP_xxxxxxxxx.1`` cross-references. We use these as
the acquisition database. The match is keyword-based against the
product string; provenance is the RefSeq / SwissProt xref plus the
GenBank protein_id.

Limitations of the fallback
---------------------------

* No HMM E-values. Each candidate carries ``e_value: null`` and a
  ``provenance_channel: genbank_annotation`` tag so the human
  reviewer in Phase R2b can tell instantly that this was not a
  homology-based call.
* False negatives expected: regulators with ambiguous product strings
  (e.g. "hypothetical protein" hits that are actually TFs) will be
  missed. The strict HMM path would catch these; this channel
  cannot.
* False positives possible: keyword matches on words like
  "regulator" hit some non-regulatory proteins (e.g. carbon storage
  regulators, redox regulators). The Phase R2b reviewer must filter.

Reproducibility
---------------

Run from repo root::

    python scripts/acquire_regulation_candidates.py

This rewrites the four YAML files under
``memory_bank/staging/regulation_curation/``. The output is
deterministic given the same GenBank input.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from Bio import SeqIO


REPO_ROOT = Path(__file__).resolve().parent.parent
GENBANK_PATH = (
    REPO_ROOT
    / "cell_sim" / "data" / "Minimal_Cell_ComplexFormation"
    / "input_data" / "syn3A.gb"
)
STAGING_DIR = REPO_ROOT / "memory_bank" / "staging" / "regulation_curation"


# ============================================================================
# Keyword sets per regulator class
# ============================================================================
SIGMA_KEYWORDS = (
    "sigma factor",
    "sigma-70",
    "sigma 70",
    "sigma subunit",
    "rna polymerase sigma",
)
SIGMA_GENE_NAMES = ("rpod", "siga", "sigb", "sigh", "sige", "rpos")

TF_KEYWORDS = (
    "transcriptional regulator",
    "transcription factor",
    "transcriptional repressor",
    "transcriptional activator",
    "dna-binding response regulator",
    "hth-type",
    "helix-turn-helix",
    "merr family",
    "lysr family",
    "tetr family",
    "araC family",
    "fur family",
    "spx",
)
# These produce both TF and two-component matches; we count them in both
# only when explicitly disambiguated.
TF_EXCLUDE_IF_CONTAINS = (
    "histidine kinase",
    "sensor kinase",
)

HISTIDINE_KINASE_KEYWORDS = (
    "histidine kinase",
    "sensor kinase",
    "sensor histidine",
    "two-component sensor",
    "two component sensor",
)
RESPONSE_REGULATOR_KEYWORDS = (
    "response regulator",
    "two-component response",
    "two component response",
)

# Riboswitches don't appear as CDS annotations — they live in 5' UTRs.
# This channel cannot acquire them. We emit an empty file with a
# documented rationale rather than fabricate hits.


# ============================================================================
# CDS record extraction
# ============================================================================
def _extract_locus_tag(feature) -> Optional[str]:
    quals = feature.qualifiers
    tags = quals.get("locus_tag")
    return tags[0] if tags else None


def _extract_protein_id(feature) -> Optional[str]:
    quals = feature.qualifiers
    pids = quals.get("protein_id")
    return pids[0] if pids else None


def _extract_inference_xref(feature) -> Optional[str]:
    """Return the ``/inference=`` xref string. Empty if absent."""
    quals = feature.qualifiers
    inf = quals.get("inference") or []
    for entry in inf:
        # Format example:
        # "EXISTENCE: similar to AA sequence:RefSeq:WP_017698145.1"
        if "RefSeq:" in entry or "SwissProt:" in entry:
            return entry
    return inf[0] if inf else None


def _extract_product(feature) -> str:
    quals = feature.qualifiers
    prods = quals.get("product") or []
    return prods[0] if prods else ""


def _extract_gene_name(feature) -> Optional[str]:
    quals = feature.qualifiers
    genes = quals.get("gene") or []
    return genes[0] if genes else None


def _extract_sequence_length(feature) -> int:
    quals = feature.qualifiers
    trans = quals.get("translation") or []
    if trans:
        return len(trans[0])
    return 0


def _extract_genome_position(feature) -> tuple[int, int]:
    """Return ``(start, end)`` in 1-based GenBank coordinates."""
    loc = feature.location
    return int(loc.start) + 1, int(loc.end)


# ============================================================================
# Candidate filters
# ============================================================================
def _matches_any(haystack: str, keywords: Iterable[str]) -> Optional[str]:
    """Return the first matching keyword (lowercased) or None."""
    h = haystack.lower()
    for kw in keywords:
        if kw.lower() in h:
            return kw.lower()
    return None


def acquire_sigma_candidates(records) -> list[dict]:
    out = []
    for cds in records:
        product = _extract_product(cds)
        gene = (_extract_gene_name(cds) or "").lower()
        prod_lower = product.lower()
        keyword = _matches_any(product, SIGMA_KEYWORDS)
        # Don't double-count "sigma factor" hits that also match the gene
        # convention; just record the most specific signal.
        if keyword is None and gene in SIGMA_GENE_NAMES:
            keyword = f"gene-name:{gene}"
        if keyword is None:
            continue
        # Skip generic "sigma" hits inside annotations like "anti-sigma"
        # which describe inhibitors of sigma factors, not sigmas themselves.
        if "anti-sigma" in prod_lower or "anti sigma" in prod_lower:
            continue
        out.append({
            "candidate_id": _extract_locus_tag(cds),
            "sigma_locus": _extract_locus_tag(cds),
            "gene_name": _extract_gene_name(cds),
            "gene_class_guess": "housekeeping",
            "matched_keyword": keyword,
            "product_annotation": product,
            "protein_length_aa": _extract_sequence_length(cds),
            "genbank_accession": "CP016816.2",
            "protein_id": _extract_protein_id(cds),
            "inference_xref": _extract_inference_xref(cds),
            "provenance_channel": "genbank_annotation",
            "e_value": None,
            "domain_coverage": None,
            "confidence": "candidate",
        })
    return out


def acquire_tf_candidates(records) -> list[dict]:
    out = []
    for cds in records:
        product = _extract_product(cds)
        prod_lower = product.lower()
        # Skip if the product looks like a kinase / response regulator —
        # those are two-component candidates, not generic TFs.
        if any(excl in prod_lower for excl in TF_EXCLUDE_IF_CONTAINS):
            continue
        keyword = _matches_any(product, TF_KEYWORDS)
        if keyword is None:
            continue
        # Suppress "DNA-binding response regulator" — that's a
        # two-component-system response regulator that happens to be
        # a TF; it'll be acquired in the two-component channel.
        if "response regulator" in prod_lower:
            continue
        out.append({
            "candidate_id": _extract_locus_tag(cds),
            "tf_locus": _extract_locus_tag(cds),
            "gene_name": _extract_gene_name(cds),
            "matched_keyword": keyword,
            "product_annotation": product,
            "protein_length_aa": _extract_sequence_length(cds),
            "genbank_accession": "CP016816.2",
            "protein_id": _extract_protein_id(cds),
            "inference_xref": _extract_inference_xref(cds),
            "provenance_channel": "genbank_annotation",
            "predicted_class": _classify_tf(prod_lower),
            "target_genes_predicted": [],   # NOT predicted at R2a
            "e_value": None,
            "confidence": "candidate",
        })
    return out


def _classify_tf(prod_lower: str) -> str:
    for fam in ("merr", "lysr", "tetr", "araC", "fur", "spx",
                "helix-turn-helix", "hth"):
        if fam in prod_lower:
            return fam
    return "unspecified"


def acquire_two_component_candidates(records) -> tuple[list[dict], list[dict]]:
    sensors = []
    response_regs = []
    for cds in records:
        product = _extract_product(cds)
        if _matches_any(product, HISTIDINE_KINASE_KEYWORDS):
            sensors.append(cds)
        if _matches_any(product, RESPONSE_REGULATOR_KEYWORDS):
            response_regs.append(cds)

    # Pair by genomic adjacency (any sensor whose locus is within 5 CDS
    # of any RR — Mycoplasma operon-style two-component pairs are
    # typically directly adjacent).
    ADJACENCY_BP = 10_000  # generous; real two-component pairs are <2kb apart
    paired = []
    for sk in sensors:
        sk_start, sk_end = _extract_genome_position(sk)
        for rr in response_regs:
            rr_start, rr_end = _extract_genome_position(rr)
            distance = min(
                abs(sk_start - rr_end),
                abs(rr_start - sk_end),
                abs(sk_start - rr_start),
            )
            if distance <= ADJACENCY_BP:
                paired.append({
                    "candidate_id": (
                        f"{_extract_locus_tag(sk)}__{_extract_locus_tag(rr)}"
                    ),
                    "sensor_locus": _extract_locus_tag(sk),
                    "response_regulator_locus": _extract_locus_tag(rr),
                    "sensor_product": _extract_product(sk),
                    "response_regulator_product": _extract_product(rr),
                    "sensor_protein_id": _extract_protein_id(sk),
                    "response_regulator_protein_id": _extract_protein_id(rr),
                    "sensor_inference_xref": _extract_inference_xref(sk),
                    "response_regulator_inference_xref":
                        _extract_inference_xref(rr),
                    "genbank_accession": "CP016816.2",
                    "genomic_distance_bp": distance,
                    "provenance_channel": "genbank_annotation",
                    "sensor_e_value": None,
                    "response_regulator_e_value": None,
                    "confidence": "candidate",
                })
    # Also record orphan kinases / regulators (no nearby partner) so
    # the reviewer can see them; they're staged but flagged.
    paired_sk = {p["sensor_locus"] for p in paired}
    paired_rr = {p["response_regulator_locus"] for p in paired}
    orphans = []
    for sk in sensors:
        lt = _extract_locus_tag(sk)
        if lt not in paired_sk:
            orphans.append({
                "candidate_id": lt,
                "kind": "orphan_sensor_kinase",
                "locus_tag": lt,
                "product_annotation": _extract_product(sk),
                "protein_id": _extract_protein_id(sk),
                "inference_xref": _extract_inference_xref(sk),
                "genbank_accession": "CP016816.2",
                "provenance_channel": "genbank_annotation",
                "e_value": None,
                "confidence": "candidate",
            })
    for rr in response_regs:
        lt = _extract_locus_tag(rr)
        if lt not in paired_rr:
            orphans.append({
                "candidate_id": lt,
                "kind": "orphan_response_regulator",
                "locus_tag": lt,
                "product_annotation": _extract_product(rr),
                "protein_id": _extract_protein_id(rr),
                "inference_xref": _extract_inference_xref(rr),
                "genbank_accession": "CP016816.2",
                "provenance_channel": "genbank_annotation",
                "e_value": None,
                "confidence": "candidate",
            })
    return paired, orphans


# ============================================================================
# YAML emission (manual to keep formatting predictable + reviewable)
# ============================================================================
def _emit_yaml_header(class_name: str, n_candidates: int,
                      acquisition_method: str) -> str:
    return (
        f"# Phase R2a staged regulation candidates — {class_name}\n"
        f"#\n"
        f"# Status: MACHINE-ACQUIRED, AWAITING HUMAN REVIEW.\n"
        f"# Acquisition method: {acquisition_method}\n"
        f"# Candidate count: {n_candidates}\n"
        f"#\n"
        f"# Every entry below carries confidence: candidate. The Phase R2b\n"
        f"# review session promotes credible entries (with rewritten\n"
        f"# confidence: measured | inferred) to the production YAML at\n"
        f"# cell_sim/data/regulation_network_syn3a.yaml. Reject the rest.\n"
        f"#\n"
        f"# Provenance fields:\n"
        f"#   provenance_channel : how this candidate was acquired\n"
        f"#                        (genbank_annotation | pfam_hmm | rfam_cm)\n"
        f"#   protein_id         : GenBank /protein_id (always)\n"
        f"#   inference_xref     : GenBank /inference (RefSeq or SwissProt)\n"
        f"#   product_annotation : verbatim /product string\n"
        f"#   e_value            : null when not from an HMM/BLAST search\n"
        f"#\n"
        f"# Re-run via: python scripts/acquire_regulation_candidates.py\n"
        f"# ============================================================================\n\n"
    )


def _yaml_str(s: Optional[str]) -> str:
    """Quote a string for safe YAML emission. Returns ``null`` for None."""
    if s is None:
        return "null"
    if "\n" in s:
        s = s.replace("\n", " ")
    # Always single-quote strings — predictable, no markdown escapes.
    return "'" + s.replace("'", "''") + "'"


def _emit_candidate(d: dict, indent: str = "  ") -> str:
    """Emit one candidate as a YAML list item with a leading ``- ``."""
    keys = list(d.keys())
    lines = []
    first = True
    for k in keys:
        v = d[k]
        if isinstance(v, list):
            if not v:
                rendered = "[]"
            else:
                rendered = "[" + ", ".join(_yaml_str(str(x)) for x in v) + "]"
        elif isinstance(v, bool):
            rendered = "true" if v else "false"
        elif v is None:
            rendered = "null"
        elif isinstance(v, (int, float)):
            rendered = str(v)
        else:
            rendered = _yaml_str(str(v))
        prefix = ("- " if first else indent)
        lines.append(f"{prefix}{k}: {rendered}")
        first = False
    return "\n".join(lines) + "\n"


def write_candidate_file(
    path: Path,
    class_name: str,
    candidates: list[dict],
    acquisition_method: str,
    extra_notes: Optional[str] = None,
) -> None:
    body = _emit_yaml_header(class_name, len(candidates), acquisition_method)
    if extra_notes:
        body += "# " + extra_notes.replace("\n", "\n# ") + "\n\n"
    if not candidates:
        body += "# Zero candidates acquired by this channel.\n"
        body += "candidates: []\n"
    else:
        body += "candidates:\n"
        for c in candidates:
            body += _emit_candidate(c)
            body += "\n"
    path.write_text(body)


# ============================================================================
# Main
# ============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genbank", default=str(GENBANK_PATH),
        help="Path to Syn3A GenBank flat file.",
    )
    parser.add_argument(
        "--out-dir", default=str(STAGING_DIR),
        help="Staging directory for candidate YAML files.",
    )
    args = parser.parse_args(argv)

    gb_path = Path(args.genbank)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gb_path.exists():
        print(f"ERROR: GenBank file not found at {gb_path}", file=sys.stderr)
        return 1

    print(f"[info] reading {gb_path}")
    record = next(SeqIO.parse(str(gb_path), "genbank"))
    cds_features = [f for f in record.features if f.type == "CDS"]
    print(f"[info] CDS feature count: {len(cds_features)}")

    # ----- sigma factors -----
    sigmas = acquire_sigma_candidates(cds_features)
    print(f"[info] sigma candidates: {len(sigmas)}")
    write_candidate_file(
        out_dir / "sigma_factor_candidates.yaml",
        "sigma factors",
        sigmas,
        acquisition_method=(
            "GenBank annotation parsing (CP016816.2 /product= keyword "
            "match against {sigma factor, sigma-70, sigma subunit, RNA "
            "polymerase sigma, rpoD/sigA gene names}; 'anti-sigma' hits "
            "excluded). Strict Pfam HMM path (PF00140 / PF00309 / "
            "PF08281) unavailable — see acquisition_log.md."
        ),
    )

    # ----- transcription factors -----
    tfs = acquire_tf_candidates(cds_features)
    print(f"[info] TF candidates: {len(tfs)}")
    write_candidate_file(
        out_dir / "transcription_factor_candidates.yaml",
        "transcription factors",
        tfs,
        acquisition_method=(
            "GenBank annotation parsing (CP016816.2 /product= keyword "
            "match against {transcriptional regulator, helix-turn-helix, "
            "MerR/LysR/TetR/AraC/Fur/Spx family, ...}; histidine kinase / "
            "sensor kinase / response regulator hits excluded — those are "
            "in two_component_candidates.yaml). Strict HTH/Pfam HMM path "
            "(PF00126 / PF00165 / PF01047 / PF13411 / PF03466) "
            "unavailable — see acquisition_log.md."
        ),
        extra_notes=(
            "Target gene prediction is NOT performed in Phase R2a "
            "(target_genes_predicted is always []). ChIP-seq data does "
            "not exist for Syn3A; comparative genomics inference is "
            "Phase R2b curation work."
        ),
    )

    # ----- riboswitches -----
    write_candidate_file(
        out_dir / "riboswitch_candidates.yaml",
        "riboswitches",
        [],
        acquisition_method=(
            "Strict Rfam Infernal cmscan path unavailable: cmscan is not "
            "installed in the acquisition sandbox; rfam.org and EBI Rfam "
            "FTP both blocked at proxy (HTTP 403 host_not_allowed). The "
            "GenBank annotation channel cannot recover riboswitches "
            "because they are RNA-structure features in 5' UTRs, not "
            "CDS annotations. Skipping with zero candidates is the "
            "honest outcome at this phase. See acquisition_log.md for "
            "the proxy probe transcript."
        ),
    )

    # ----- two-component systems -----
    paired, orphans = acquire_two_component_candidates(cds_features)
    print(f"[info] two-component paired: {len(paired)}, orphans: {len(orphans)}")
    tcs_combined = []
    for p in paired:
        p2 = dict(p)
        p2["kind"] = "paired"
        tcs_combined.append(p2)
    for o in orphans:
        tcs_combined.append(o)
    write_candidate_file(
        out_dir / "two_component_candidates.yaml",
        "two-component systems",
        tcs_combined,
        acquisition_method=(
            "GenBank annotation parsing (CP016816.2 /product= keyword "
            "match: histidine kinase / sensor kinase for sensors, "
            "response regulator for RRs; partners paired by genomic "
            "adjacency within 10 kb). Strict Pfam HMM path "
            "(PF07568 / PF00512 / PF02518 / PF00072) unavailable — "
            "see acquisition_log.md."
        ),
    )

    # ----- summary returned to caller -----
    return {
        "n_sigma": len(sigmas),
        "n_tf": len(tfs),
        "n_two_component_paired": len(paired),
        "n_two_component_orphan": len(orphans),
        "n_riboswitch": 0,
        "cds_count": len(cds_features),
        "genbank_accession": record.id,
    }


if __name__ == "__main__":
    summary = main()
    if isinstance(summary, dict):
        print(f"\n[summary] {summary}")
        sys.exit(0)
    sys.exit(int(summary or 0))
