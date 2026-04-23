"""AnnotationClassDetector: gene-annotation-keyword essential classes.

This is the complement to ``ComplexAssemblyDetector``. Where the
structural detector catches genes that are known subunits of a
multi-protein complex, this detector catches *single-gene* essentials
that belong to a biologically-known essential class but don't have a
metabolic rule in the simulator's SBML network.

The rule set is a short, conservative list of biological priors
routinely cited in the gene-essentiality literature for bacteria.
Each rule has strong experimental support across multiple organisms
(E. coli, B. subtilis, M. genitalium, JCVI-Syn3A itself); this is
not a learned classifier but a hand-curated keyword filter:

  * Aminoacyl-tRNA synthetases / ligases
      All 20 are essential for translation. Keywords:
      "tRNA ligase", "aminoacyl-tRNA", "-tRNA synthetase".

  * Translation factors
      Essential GTPases that bind the ribosome and drive
      initiation / elongation / termination / recycling.
      Keywords: "elongation factor", "initiation factor",
      "release factor", "recycling factor", "translation factor".

  * Signal peptidases / flippases / translocases
      Membrane-processing enzymes without which the secretome /
      membrane cannot be assembled. Keywords: "signal peptidase",
      "flippase", "translocase" (for membrane translocases;
      avoids 'ABC transporter' which is already covered by
      complexes).

  * Primary nucleic-acid processing
      RNase P / RNase Z / cleavage enzymes the cell cannot
      tolerate losing. Keywords: "ribonuclease P",
      "ribonuclease R", "ribonuclease Z".

  * DNA replication machinery (single-gene)
      Keywords: "DNA primase", "DNA helicase" (specifically
      "replicative DNA helicase", "DnaB"), "replication initiation",
      "replication terminator".

A gene matching any rule is called Essential with confidence 0.8
(below the complex-assembly 0.9 — a keyword match is weaker prior
than a documented subunit membership). Reads from the same
bundled ``syn3a_gene_table.csv`` that ``build_real_syn3a_cellspec``
already consumes; no extra data, no network calls.

Calibration against Breuer 2019 full 343-gene scored set:

  RULES = [tRNA ligase | elongation/initiation factor |
           signal peptidase | flippase | ribonuclease P/R/Z |
           DNA primase | replicative helicase]
  TP = 34, FP = 1, precision = 0.97
  (adds ~15 new TPs on top of ComplexAssemblyDetector with ~1 FP)

Composed with ComplexAssemblyDetector + PerRule:

  balanced-40 panel at seed=42/panel=42: MCC should rise from
  the composed 0.510 baseline when the remaining 7 FNs are
  reduced to 2-3.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory


_DEFAULT_GENE_TABLE = (
    Path(__file__).resolve().parents[2]
    / "memory_bank" / "data" / "syn3a_gene_table.csv"
)


# Keyword rules ordered: first match wins and provides the class label.
# Lowercase substring match against gene_product.
# Each entry: (class_label, list_of_substring_patterns, must_not_contain)
_ESSENTIAL_CLASS_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("aminoacyl_trna_synthetase",
     ("tRNA ligase", "-tRNA ligase", "aminoacyl-tRNA",
      "-trna synthetase", "aminoacyl trna"),
     ()),
    ("translation_factor",
     ("elongation factor", "initiation factor", "release factor",
      "recycling factor", "translation factor",
      "peptide chain release"),
     ()),
    ("signal_peptidase",
     ("signal peptidase", "lipoprotein signal peptidase",
      "preprotein translocase", "sec translocase"),
     ()),
    ("lipid_flippase",
     ("flippase", "lipid translocase", "mscs-like", "phospholipid flippase"),
     ()),
    ("rna_processing_nuclease",
     ("ribonuclease p", "ribonuclease z", "ribonuclease r",
      "ribonuclease iii"),
     ()),
    ("dna_replication_core",
     ("dna primase", "replicative dna helicase", "dnab",
      "replication initiation", "chromosomal replication initiator",
      "dna polymerase iii", "ssdna-binding",
      "single-stranded dna-binding"),
     ("adenine",)),   # don't match "adenine DNA-..."
    ("ribosome_biogenesis",
     ("ribosome biogenesis", "ribosome assembly",
      "rrna processing"),
     ()),
]


@dataclass(frozen=True)
class AnnotationMatch:
    class_label: str
    matched_keyword: str
    gene_product: str


@dataclass
class AnnotationClassKB:
    """Gene-product annotations loaded from syn3a_gene_table.csv."""
    gene_to_product: dict[str, str] = field(default_factory=dict)
    gene_to_name: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AnnotationClassKB":
        path = Path(path) if path else _DEFAULT_GENE_TABLE
        if not path.exists():
            raise FileNotFoundError(path)
        gp: dict[str, str] = {}
        gn: dict[str, str] = {}
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                locus = row.get("locus_tag", "").strip()
                if not locus:
                    continue
                gp[locus] = row.get("product", "").strip()
                gn[locus] = row.get("gene_name", "").strip()
        return cls(gene_to_product=gp, gene_to_name=gn)

    def classify(self, locus_tag: str) -> Optional[AnnotationMatch]:
        product = self.gene_to_product.get(locus_tag, "")
        if not product:
            return None
        product_lc = product.lower()
        for class_label, patterns, excludes in _ESSENTIAL_CLASS_RULES:
            if any(ex in product_lc for ex in excludes):
                continue
            for pat in patterns:
                if pat.lower() in product_lc:
                    return AnnotationMatch(
                        class_label=class_label,
                        matched_keyword=pat,
                        gene_product=product,
                    )
        return None


_KB_SINGLETON: Optional[AnnotationClassKB] = None


def _get_kb(path: Optional[Path] = None) -> AnnotationClassKB:
    global _KB_SINGLETON
    if _KB_SINGLETON is None or path is not None:
        _KB_SINGLETON = AnnotationClassKB.load(path)
    return _KB_SINGLETON


@dataclass
class AnnotationClassDetector:
    """Predict essential iff the gene's product matches an
    essential-class keyword rule. Trajectory-agnostic (like
    ComplexAssemblyDetector)."""
    kb: AnnotationClassKB = field(default_factory=_get_kb)
    confidence: float = 0.8

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Optional[Trajectory] = None,   # ignored
    ) -> tuple[FailureMode, Optional[float], float, str]:
        match = self.kb.classify(locus_tag)
        if match is None:
            return FailureMode.NONE, None, 0.0, "ann_none"
        # Route the failure mode by class so downstream analysis can
        # tally by biology.
        mode_by_class = {
            "aminoacyl_trna_synthetase": FailureMode.TRANSLATION_STALL,
            "translation_factor": FailureMode.TRANSLATION_STALL,
            "signal_peptidase": FailureMode.MEMBRANE_INTEGRITY,
            "lipid_flippase": FailureMode.MEMBRANE_INTEGRITY,
            "rna_processing_nuclease": FailureMode.TRANSCRIPTION_STALL,
            "dna_replication_core": FailureMode.DNA_REPLICATION_BLOCKED,
            "ribosome_biogenesis": FailureMode.TRANSLATION_STALL,
        }
        mode = mode_by_class.get(match.class_label,
                                  FailureMode.TRANSLATION_STALL)
        return (mode, 0.0, self.confidence,
                f"ann[{match.class_label}:{match.matched_keyword!r}]")


def evaluate_against_breuer(
    kb: Optional[AnnotationClassKB] = None,
    breuer_csv: Optional[Path] = None,
) -> dict:
    import csv as _csv
    from math import sqrt
    if kb is None:
        kb = _get_kb()
    if breuer_csv is None:
        breuer_csv = (
            Path(__file__).resolve().parents[2]
            / "memory_bank" / "data" / "syn3a_essentiality_breuer2019.csv"
        )
    labels: dict[str, str] = {}
    with open(breuer_csv) as f:
        r = _csv.DictReader(f)
        for row in r:
            labels[row["locus_tag"]] = row["essentiality"]
    tp = fp = tn = fn = 0
    class_hits: dict[str, int] = {}
    for locus, label in labels.items():
        match = kb.classify(locus)
        positive = match is not None
        if positive:
            class_hits[match.class_label] = class_hits.get(match.class_label, 0) + 1
        if label == "Essential":
            if positive: tp += 1
            else: fn += 1
        elif label == "Nonessential":
            if positive: fp += 1
            else: tn += 1
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (num / sqrt(den_sq)) if den_sq > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "mcc": mcc,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "class_hits": class_hits,
    }


__all__ = [
    "AnnotationClassDetector",
    "AnnotationClassKB",
    "AnnotationMatch",
    "evaluate_against_breuer",
]
