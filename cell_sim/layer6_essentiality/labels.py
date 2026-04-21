"""Breuer 2019 essentiality labels — loader + binary mapping.

Labels come from memory_bank/data/syn3a_essentiality_breuer2019.csv
(pointed to by facts/structural/syn3a_essentiality_breuer2019.json).
Three classes:

    Essential        -> binary positive
    Quasiessential   -> binary positive (configurable)
    Nonessential     -> binary negative

The brief's MCC > 0.59 target corresponds to Breuer 2019's own FBA
performance under the default ``essential={Essential, Quasiessential}``
binarisation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABELS_CSV = _REPO_ROOT / "memory_bank/data/syn3a_essentiality_breuer2019.csv"


class EssentialityClass(str, Enum):
    ESSENTIAL = "Essential"
    QUASI = "Quasiessential"
    NONESSENTIAL = "Nonessential"


@dataclass(frozen=True, slots=True)
class Label:
    locus_tag: str
    gene_name: str
    essentiality: EssentialityClass
    primary_function: str

    def is_positive(self, quasi_as_positive: bool = True) -> bool:
        if self.essentiality == EssentialityClass.ESSENTIAL:
            return True
        if self.essentiality == EssentialityClass.QUASI:
            return quasi_as_positive
        return False


def load_breuer2019_labels(path: Path | None = None) -> dict[str, Label]:
    p = path or _LABELS_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"Labels CSV missing at {p}. Re-generate from "
            f"initial_concentrations.xlsx per memory_bank/data/STAGING.md."
        )
    out: dict[str, Label] = {}
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ess = (row.get("essentiality") or "").strip()
            if not ess:
                continue
            try:
                cls = EssentialityClass(ess)
            except ValueError:
                continue
            lab = Label(
                locus_tag=row["locus_tag"],
                gene_name=row.get("gene_name") or "",
                essentiality=cls,
                primary_function=row.get("primary_function") or "",
            )
            out[lab.locus_tag] = lab
    return out


def binary_labels(
    labels: dict[str, Label], *, quasi_as_positive: bool = True
) -> dict[str, int]:
    """Map the three-class labels into {0, 1} for MCC."""
    return {
        lt: int(lab.is_positive(quasi_as_positive))
        for lt, lab in labels.items()
    }


def select(
    labels: dict[str, Label], locus_tags: Iterable[str]
) -> dict[str, Label]:
    tags = set(locus_tags)
    return {lt: lab for lt, lab in labels.items() if lt in tags}
