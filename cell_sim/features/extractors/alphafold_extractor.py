"""Per-protein structural descriptors from AlphaFold Protein Structure
Database (AFDB) predictions.

For every Syn3A CDS with a UniProt accession, fetches
``https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v4.pdb``
and extracts a compact set of structural-summary features that the
detectors can use as priors: confidence (pLDDT mean / std / disorder
fraction), secondary-structure composition (helix / sheet / coil),
sequence length, and radius of gyration.

Heavy imports (``requests``, ``biopython``) are performed inside
``_ensure_loaded`` or inline in ``extract``. The module-level import
of this file is verified in ``test_module_import_cheap`` to not pull
those packages.

Missing-structure handling:

  * Genes without a ``uniprot_id`` in the input DataFrame, or whose
    PDB fetch returns 404, or whose PDB cannot be parsed, produce a
    row with every numeric column set to NaN and ``af_has_structure``
    set to 0.0. The detector sweep interprets NaN naturally (the
    ``FeatureRegistry.join_features`` contract).

Contract:

  * Input: DataFrame with at least ``locus_tag`` (required) and
    ``uniprot_id`` (optional — missing cells trigger the no-structure
    branch). Empty input short-circuits.
  * Output: DataFrame indexed by ``locus_tag`` with 9 float columns
    listed in :data:`_FEATURE_COLS`.
"""
from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

from cell_sim.features.batched_inference import (
    BatchedFeatureExtractor,
    BatchedInferenceConfig,
)


_FEATURE_COLS: list[str] = [
    "af_plddt_mean",
    "af_plddt_std",
    "af_disorder_fraction",
    "af_helix_fraction",
    "af_sheet_fraction",
    "af_coil_fraction",
    "af_sequence_length",
    "af_radius_of_gyration_angstrom",
    "af_has_structure",
]

_AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
_AFDB_REQUEST_TIMEOUT_S = 30.0
_AFDB_RETRIES = 3
_AFDB_RETRY_BACKOFF_S = 2.0


class AlphaFoldExtractor(BatchedFeatureExtractor):
    """Per-protein structural descriptors from AFDB.

    Unlike :class:`ESM2Extractor`, this extractor is network-bound
    (one HTTP GET per gene) rather than compute-bound. Expected
    wall-time on a warm connection: ~15-30 minutes for all 452
    Syn3A CDS.

    Subclasses or callers override :attr:`_fetch_pdb_bytes` for
    offline / cached-mirror runs.
    """

    name = "alphafold_db"
    version = "0.1.0"
    feature_cols = list(_FEATURE_COLS)

    def __init__(self) -> None:
        # Nothing to load up front — dependencies are pulled in on
        # the first ``extract`` call with non-empty input.
        self._biopython_loaded = False

    # ---- lazy imports ----

    def _ensure_loaded(self, config: BatchedInferenceConfig) -> None:
        """Import ``biopython`` on first non-empty extract call. The
        module is imported for its side effect (making ``Bio.PDB``
        available); we don't keep a reference."""
        if self._biopython_loaded:
            return
        import Bio.PDB  # noqa: F401, WPS433 — lazy heavy import
        self._biopython_loaded = True

    # ---- extraction ----

    def extract(
        self,
        inputs: pd.DataFrame,
        config: Optional[BatchedInferenceConfig] = None,
    ) -> pd.DataFrame:
        if config is None:
            config = BatchedInferenceConfig()
        if len(inputs) == 0:
            return _empty_frame(self.feature_cols)

        _validate_input_columns(inputs, required={"locus_tag"})

        self._ensure_loaded(config)

        locus_tags = inputs["locus_tag"].astype(str).tolist()
        uniprot_ids: list[str] = (
            inputs["uniprot_id"].astype(str).tolist()
            if "uniprot_id" in inputs.columns
            else [""] * len(inputs)
        )

        rows: list[dict[str, float]] = []
        for uid_raw in uniprot_ids:
            uid = uid_raw.strip()
            if not uid or uid.lower() in {"nan", "none", "-"}:
                rows.append(_no_structure_row())
                continue
            try:
                pdb_bytes = self._fetch_pdb_bytes(uid)
            except _AFDBMissing:
                rows.append(_no_structure_row())
                continue
            except _AFDBTransientError:
                # Network failure counts as missing for this run; the
                # user can rerun the notebook to fill in gaps later.
                rows.append(_no_structure_row())
                continue
            try:
                rows.append(_features_from_pdb(pdb_bytes))
            except Exception:  # noqa: BLE001 — parse failures get NaN
                rows.append(_no_structure_row())

        out = pd.DataFrame(rows, columns=self.feature_cols)
        out.index = pd.Index(locus_tags, name="locus_tag")
        return out

    # ---- network ----

    def _fetch_pdb_bytes(self, uniprot_id: str) -> bytes:
        """GET the AFDB model PDB with a small retry loop.

        Raises :class:`_AFDBMissing` on 404 (no AlphaFold prediction
        for that UniProt ID) and :class:`_AFDBTransientError` on
        any other network failure after exhausting retries.
        """
        url = _AFDB_URL.format(uid=uniprot_id)
        last_err: Optional[Exception] = None
        for attempt in range(_AFDB_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "cell_sim-AlphaFoldExtractor/0.1"},
                )
                with urllib.request.urlopen(
                    req, timeout=_AFDB_REQUEST_TIMEOUT_S,
                ) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise _AFDBMissing(
                        f"no AFDB prediction for UniProt {uniprot_id!r}"
                    ) from exc
                last_err = exc
            except Exception as exc:  # noqa: BLE001 — retry anything transient
                last_err = exc
            time.sleep(_AFDB_RETRY_BACKOFF_S * (attempt + 1))
        raise _AFDBTransientError(
            f"failed to fetch AFDB model for {uniprot_id!r}: "
            f"{type(last_err).__name__}: {last_err}"
        )


# ---- PDB parsing helpers ----


def _features_from_pdb(pdb_bytes: bytes) -> dict[str, float]:
    """Parse a PDB byte-blob and compute the 9 feature values.

    Imports biopython lazily; must only be called after
    ``_ensure_loaded``.
    """
    import io
    from Bio.PDB import PDBParser  # noqa: WPS433 — lazy

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        "afdb", io.StringIO(pdb_bytes.decode("utf-8", errors="ignore")),
    )

    ca_coords: list[tuple[float, float, float]] = []
    plddts: list[float] = []
    phi_psi: list[tuple[float, float]] = []

    for model in structure:
        for chain in model:
            residues = list(chain)
            for i, residue in enumerate(residues):
                if "CA" not in residue:
                    continue
                ca = residue["CA"]
                ca_coords.append(tuple(ca.get_coord()))
                # AFDB embeds pLDDT as the B-factor of each atom.
                plddts.append(float(ca.get_bfactor()))
                # Torsion-angle based SS call; simple proxy for DSSP.
                phi, psi = _phi_psi_for_residue(residues, i)
                phi_psi.append((phi, psi))
        break  # AFDB PDBs are single-model

    if not ca_coords:
        return _no_structure_row()

    seq_len = float(len(ca_coords))
    plddt_mean = float(sum(plddts) / seq_len)
    plddt_variance = sum((p - plddt_mean) ** 2 for p in plddts) / seq_len
    plddt_std = float(math.sqrt(plddt_variance))
    disorder_count = sum(1 for p in plddts if p < 50.0)
    disorder_fraction = float(disorder_count / seq_len)

    helix, sheet, coil = _secondary_structure_fractions(phi_psi)
    rg = _radius_of_gyration(ca_coords)

    return {
        "af_plddt_mean": plddt_mean,
        "af_plddt_std": plddt_std,
        "af_disorder_fraction": disorder_fraction,
        "af_helix_fraction": helix,
        "af_sheet_fraction": sheet,
        "af_coil_fraction": coil,
        "af_sequence_length": seq_len,
        "af_radius_of_gyration_angstrom": rg,
        "af_has_structure": 1.0,
    }


def _phi_psi_for_residue(
    residues, i: int,
) -> tuple[float, float]:
    """Torsion angles (phi, psi) in degrees; NaN if undefined (e.g.
    N-terminal phi, C-terminal psi, or missing atoms)."""
    # Lazy imports; only reached via extract() path.
    from Bio.PDB.vectors import calc_dihedral  # noqa: WPS433

    phi = math.nan
    psi = math.nan
    try:
        prev_c = residues[i - 1]["C"].get_vector() if i > 0 else None
        n = residues[i]["N"].get_vector()
        ca = residues[i]["CA"].get_vector()
        c = residues[i]["C"].get_vector()
        next_n = (
            residues[i + 1]["N"].get_vector()
            if i + 1 < len(residues)
            else None
        )
        if prev_c is not None:
            phi = math.degrees(calc_dihedral(prev_c, n, ca, c))
        if next_n is not None:
            psi = math.degrees(calc_dihedral(n, ca, c, next_n))
    except KeyError:
        pass
    return phi, psi


def _secondary_structure_fractions(
    phi_psi: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Classify each residue by its (phi, psi) into helix / sheet /
    coil using the standard Ramachandran-plot bins. Returns fractions
    summing to ~1.0 (residues with undefined torsions count as coil).

    Helix region (loosely):   phi in [-90, -30], psi in [-70, -15]
    Sheet region (loosely):   phi in [-180, -40], psi in [90, 180] or
                              psi in [-180, -170]
    Coil: everything else, including residues whose torsions are NaN.
    """
    n = max(len(phi_psi), 1)
    helix = sheet = 0
    for phi, psi in phi_psi:
        if not (math.isfinite(phi) and math.isfinite(psi)):
            continue
        if -90.0 <= phi <= -30.0 and -70.0 <= psi <= -15.0:
            helix += 1
            continue
        if (-180.0 <= phi <= -40.0) and (
            (90.0 <= psi <= 180.0) or (-180.0 <= psi <= -170.0)
        ):
            sheet += 1
    coil = n - helix - sheet
    return helix / n, sheet / n, coil / n


def _radius_of_gyration(
    coords: list[tuple[float, float, float]],
) -> float:
    """Mass-ignorant R_g over C-alpha coordinates, returned in Å."""
    n = len(coords)
    if n == 0:
        return math.nan
    cx = sum(x for x, _, _ in coords) / n
    cy = sum(y for _, y, _ in coords) / n
    cz = sum(z for _, _, z in coords) / n
    ss = 0.0
    for x, y, z in coords:
        ss += (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return math.sqrt(ss / n)


# ---- minor helpers ----


def _empty_frame(cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=cols)
    frame.index = pd.Index([], name="locus_tag")
    return frame


def _no_structure_row() -> dict[str, float]:
    row = {c: math.nan for c in _FEATURE_COLS}
    row["af_has_structure"] = 0.0
    return row


def _validate_input_columns(df: pd.DataFrame, *, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"AlphaFoldExtractor.extract: input DataFrame missing "
            f"required column(s) {sorted(missing)!r}. "
            f"Got: {list(df.columns)!r}"
        )


class _AFDBMissing(LookupError):
    """404 from AlphaFold-DB: the UniProt ID has no prediction."""


class _AFDBTransientError(RuntimeError):
    """Non-404 network failure; treated as missing for the current run."""


__all__ = ["AlphaFoldExtractor"]
