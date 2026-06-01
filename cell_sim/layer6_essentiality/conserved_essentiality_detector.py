"""ConservedEssentialityDetector: cross-species essentiality transfer.

Complement to AnnotationClassDetector. Where the annotation detector
fires on gene-product keyword matches, this detector fires on
**transferred essentiality calls** from independent saturating Tn-seq
experiments in related Mycoplasma species:

  * M. genitalium G37 (Glass 2006 PNAS, Hutchison 1999 Science) —
    saturating transposon mutagenesis, 482 CDS, ~380 essential.
  * M. pneumoniae M129 (Lluch-Senar 2015 Mol Syst Biol, eLife
    4:e09943) — saturating transposon mutagenesis, 698 CDS,
    ~280 essential.

For each syn3A locus the cached CSV records the best homolog in each
reference and its essentiality call. If at least one reference homolog
is essential, the rule fires with confidence reflecting agreement:

  both ess agree         → 0.90   (matches structural-rule strength)
  single-reference essential → 0.78  (between annotation 0.80 and
                                       trajectory ~0.6)

The rule abstains (returns FailureMode.NONE) when:
  - syn3A locus has no homolog in either reference
  - both homologs are non-essential (informative negative — but the
    composed detector treats abstention as the same as "no signal";
    the rule does NOT mark genes as predicted-nonessential)

Why this is NOT label leakage against Breuer 2019:

  * M. genitalium and M. pneumoniae Tn-seq experiments are in different
    organisms with independent essentiality calls. They were done
    20 years ago (M.gen) and 10 years ago (M.pne) — long before syn3A.
  * Crucially: M. mycoides syn1.0 / syn3.0 Tn-seq (Hutchison 2016) is
    DELIBERATELY EXCLUDED. syn3A was constructed by iteratively
    deleting non-essentials from M. mycoides; using its essentiality
    as a feature would be circular.

Source data: `cell_sim/data/cross_species_essentiality.csv`. Built by
`scripts/build_cross_species_essentiality.py` from
`memory_bank/data/multiorg_essentiality/{labels.csv,sequences.fasta}`.
Schema documented in the build script.

Calibration: see test_conserved_essentiality_detector.py for the
synthetic tests; real-data calibration comes from running v16 vs v15
head-to-head on the full Breuer set (planned).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cell_sim.layer6_essentiality.harness import FailureMode, Trajectory


_DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "cross_species_essentiality.csv"
)


# Confidence settings.
# Both references agree essential is the strongest cross-species
# signal — higher than annotation keyword (0.80) because it is a
# transferred direct measurement, not a heuristic.
_CONF_BOTH_ESSENTIAL = 0.90
_CONF_ONE_ESSENTIAL  = 0.78


@dataclass(frozen=True)
class HomologRecord:
    """Per-syn3A-locus row from cross_species_essentiality.csv."""
    locus_tag:        str
    mg_homolog_locus: str    # M. genitalium locus tag (empty if no homolog)
    mg_homolog_method: str   # 'gene_name' | 'blast' | ''
    mg_pident:        Optional[float]   # null if gene_name match
    mg_essential:     Optional[int]     # 0/1, null if no homolog
    mp_homolog_locus: str
    mp_homolog_method: str
    mp_pident:        Optional[float]
    mp_essential:     Optional[int]


@dataclass
class ConservedEssentialityKB:
    """Per-locus cross-species essentiality calls."""
    records: dict[str, HomologRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ConservedEssentialityKB":
        path = Path(path) if path else _DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"cross-species essentiality CSV not found at {path}. "
                f"Run scripts/build_cross_species_essentiality.py to "
                f"populate it (or pass --cross-species-csv with an "
                f"alternative path)."
            )
        recs: dict[str, HomologRecord] = {}
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                lt = row.get("locus_tag", "").strip()
                if not lt:
                    continue

                def _opt_int(v: str) -> Optional[int]:
                    v = (v or "").strip()
                    return int(v) if v.isdigit() else None

                def _opt_float(v: str) -> Optional[float]:
                    v = (v or "").strip()
                    if not v:
                        return None
                    try:
                        return float(v)
                    except ValueError:
                        return None

                recs[lt] = HomologRecord(
                    locus_tag         = lt,
                    mg_homolog_locus  = (row.get("mg_homolog_locus") or "").strip(),
                    mg_homolog_method = (row.get("mg_homolog_method") or "").strip(),
                    mg_pident         = _opt_float(row.get("mg_pident", "")),
                    mg_essential      = _opt_int  (row.get("mg_essential", "")),
                    mp_homolog_locus  = (row.get("mp_homolog_locus") or "").strip(),
                    mp_homolog_method = (row.get("mp_homolog_method") or "").strip(),
                    mp_pident         = _opt_float(row.get("mp_pident", "")),
                    mp_essential      = _opt_int  (row.get("mp_essential", "")),
                )
        return cls(records=recs)


_KB_SINGLETON: Optional[ConservedEssentialityKB] = None


def _get_kb(path: Optional[Path] = None) -> ConservedEssentialityKB:
    global _KB_SINGLETON
    if _KB_SINGLETON is None or path is not None:
        _KB_SINGLETON = ConservedEssentialityKB.load(path)
    return _KB_SINGLETON


@dataclass
class ConservedEssentialityDetector:
    """Predict essential iff at least one Mycoplasma homolog is essential.

    Parameters
    ----------
    kb               : loaded KB; defaults to the bundled CSV.
    min_pident       : if a homolog is mapped by BLAST, require at least
                       this percent identity to count (default 30.0).
                       gene_name matches always count.
    require_both     : if True, fire only when BOTH mg and mp call essential.
                       More conservative; lower recall, fewer FPs.
                       Default False.
    """
    kb:           ConservedEssentialityKB = field(default_factory=_get_kb)
    min_pident:   float = 30.0
    require_both: bool  = False

    def _homolog_counts_as_essential(
        self, ess: Optional[int], method: str, pident: Optional[float]
    ) -> bool:
        if ess != 1:
            return False
        if method == "blast" and pident is not None and pident < self.min_pident:
            return False
        return True

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Optional[Trajectory] = None,   # ignored — trajectory-agnostic
    ) -> tuple[FailureMode, Optional[float], float, str]:
        rec = self.kb.records.get(locus_tag)
        if rec is None:
            return FailureMode.NONE, None, 0.0, "hom_none"

        mg_ess = self._homolog_counts_as_essential(
            rec.mg_essential, rec.mg_homolog_method, rec.mg_pident
        )
        mp_ess = self._homolog_counts_as_essential(
            rec.mp_essential, rec.mp_homolog_method, rec.mp_pident
        )

        if self.require_both:
            fire = mg_ess and mp_ess
        else:
            fire = mg_ess or mp_ess

        if not fire:
            # Build a short trace string so the composed detector's
            # "no_signal" evidence is interpretable.
            bits = []
            if rec.mg_homolog_locus:
                bits.append(
                    f"mg:{rec.mg_homolog_locus}"
                    f"={'ess' if rec.mg_essential == 1 else 'non' if rec.mg_essential == 0 else '?'}"
                )
            if rec.mp_homolog_locus:
                bits.append(
                    f"mp:{rec.mp_homolog_locus}"
                    f"={'ess' if rec.mp_essential == 1 else 'non' if rec.mp_essential == 0 else '?'}"
                )
            trace = "|".join(bits) if bits else "no_homologs"
            return FailureMode.NONE, None, 0.0, f"hom_no_fire[{trace}]"

        # Compose evidence string + confidence
        agree = mg_ess and mp_ess
        confidence = _CONF_BOTH_ESSENTIAL if agree else _CONF_ONE_ESSENTIAL
        bits = []
        if mg_ess:
            bits.append(
                f"mg:{rec.mg_homolog_locus}:{rec.mg_homolog_method}"
                + (f"/{rec.mg_pident:.0f}" if rec.mg_pident is not None else "")
            )
        if mp_ess:
            bits.append(
                f"mp:{rec.mp_homolog_locus}:{rec.mp_homolog_method}"
                + (f"/{rec.mp_pident:.0f}" if rec.mp_pident is not None else "")
            )
        evidence = "hom[" + "+".join(bits) + "]"
        return FailureMode.HOMOLOG_ESSENTIAL, 0.0, confidence, evidence


__all__ = [
    "ConservedEssentialityDetector",
    "ConservedEssentialityKB",
    "HomologRecord",
]
