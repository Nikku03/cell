"""YAML loader for the regulation network config — Phase R1 infrastructure.

The on-disk schema (described in the comment block of
``cell_sim/data/regulation_network_syn3a.yaml``) is four named lists::

    transcription_factors: []
    sigma_factors: []
    riboswitches: []
    two_component_systems: []

Each entry, once curated, must carry ``source``, ``confidence``, and
``curated_on`` fields alongside its kinetic parameters. The loader
enforces those three fields when entries are present (raising
``RegulationConfigError`` if any are missing) but tolerates missing
keys gracefully when the section is empty.

Phase R1 ships an empty config. The loader is built and tested so
that a future curation session can drop populated entries into the
YAML and have them reach Phase R2 wiring without further changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_NETWORK_YAML_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "regulation_network_syn3a.yaml"
)


# ============================================================================
# Errors and config dataclass
# ============================================================================
class RegulationConfigError(ValueError):
    """Raised when the regulation network YAML fails schema validation.

    Phase R1 only validates entries that exist; an empty config is
    valid. The error fires when, e.g., a TF entry is missing its
    ``source`` field or a riboswitch entry is missing its
    ``ligand_metabolite``.
    """


@dataclass
class RegulationNetworkConfig:
    """Parsed regulation network. All four lists default to empty.

    Each list contains plain dicts with the schema fields documented
    in the YAML comment block. We keep dicts rather than nested
    dataclasses at this phase because (a) the schema is still in flux,
    and (b) it keeps the loader trivially compatible with Phase R2
    additions that don't require code changes.
    """

    transcription_factors: list[dict[str, Any]] = field(default_factory=list)
    sigma_factors: list[dict[str, Any]] = field(default_factory=list)
    riboswitches: list[dict[str, Any]] = field(default_factory=list)
    two_component_systems: list[dict[str, Any]] = field(default_factory=list)
    source_path: Optional[Path] = None

    def is_empty(self) -> bool:
        return not (
            self.transcription_factors
            or self.sigma_factors
            or self.riboswitches
            or self.two_component_systems
        )

    def total_entries(self) -> int:
        return (
            len(self.transcription_factors)
            + len(self.sigma_factors)
            + len(self.riboswitches)
            + len(self.two_component_systems)
        )


# ============================================================================
# Loader
# ============================================================================
_REQUIRED_PROVENANCE_FIELDS = ("source", "confidence", "curated_on")
_ALLOWED_CONFIDENCE = {"measured", "inferred", "placeholder"}


def load_regulation_network(
    path: Optional[Path] = None,
) -> RegulationNetworkConfig:
    """Parse the regulation network YAML and return a config object.

    Parameters
    ----------
    path
        Path to the YAML file. Defaults to
        :data:`DEFAULT_NETWORK_YAML_PATH`.

    Returns
    -------
    RegulationNetworkConfig
        Parsed config. Empty when the YAML has no entries (the Phase
        R1 default).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RegulationConfigError
        If any populated entry is missing a required provenance field
        (``source``, ``confidence``, ``curated_on``) or carries a
        confidence value outside {measured, inferred, placeholder}.
    yaml.YAMLError
        If the file is malformed YAML.
    """
    if path is None:
        path = DEFAULT_NETWORK_YAML_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"regulation network YAML not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # An entirely-comments file parses to None.
    if raw is None:
        return RegulationNetworkConfig(source_path=path)

    if not isinstance(raw, dict):
        raise RegulationConfigError(
            f"{path.name}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    cfg = RegulationNetworkConfig(
        transcription_factors=_extract_list(raw, "transcription_factors", path),
        sigma_factors=_extract_list(raw, "sigma_factors", path),
        riboswitches=_extract_list(raw, "riboswitches", path),
        two_component_systems=_extract_list(raw, "two_component_systems", path),
        source_path=path,
    )

    # Schema validation only fires for populated entries. Empty lists
    # pass through silently — that's the Phase R1 expected case.
    _validate_section(cfg.transcription_factors, "transcription_factors",
                      ("tf_locus", "target_genes"), path)
    _validate_section(cfg.sigma_factors, "sigma_factors",
                      ("sigma_locus", "gene_class"), path)
    _validate_section(cfg.riboswitches, "riboswitches",
                      ("gene_locus", "ligand_metabolite"), path)
    _validate_section(cfg.two_component_systems, "two_component_systems",
                      ("sensor_locus", "response_regulator_locus"), path)

    return cfg


def _extract_list(
    raw: dict, key: str, path: Path,
) -> list[dict[str, Any]]:
    val = raw.get(key)
    if val is None:
        return []
    if not isinstance(val, list):
        raise RegulationConfigError(
            f"{path.name}: section '{key}' must be a list, "
            f"got {type(val).__name__}"
        )
    return val


def _validate_section(
    entries: list[dict],
    section: str,
    required_kinetic_keys: tuple[str, ...],
    path: Path,
) -> None:
    """Validate every populated entry. Empty list -> no-op."""
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RegulationConfigError(
                f"{path.name}: {section}[{i}] must be a mapping"
            )
        missing_kinetic = [k for k in required_kinetic_keys if k not in entry]
        if missing_kinetic:
            raise RegulationConfigError(
                f"{path.name}: {section}[{i}] missing kinetic keys "
                f"{missing_kinetic}"
            )
        missing_prov = [k for k in _REQUIRED_PROVENANCE_FIELDS if k not in entry]
        if missing_prov:
            raise RegulationConfigError(
                f"{path.name}: {section}[{i}] missing provenance fields "
                f"{missing_prov}. Phase R2 curation MUST set source / "
                f"confidence / curated_on for every entry."
            )
        conf = entry.get("confidence")
        if conf not in _ALLOWED_CONFIDENCE:
            raise RegulationConfigError(
                f"{path.name}: {section}[{i}] confidence={conf!r} "
                f"not in {sorted(_ALLOWED_CONFIDENCE)}"
            )


__all__ = [
    "DEFAULT_NETWORK_YAML_PATH",
    "RegulationConfigError",
    "RegulationNetworkConfig",
    "load_regulation_network",
]
