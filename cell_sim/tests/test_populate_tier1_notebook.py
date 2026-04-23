"""Parse-time validation for notebooks/populate_tier1_cache.ipynb.

Uses plain ``json.loads`` so the test does not require ``nbformat``
as a runtime dependency. Verifies that the notebook is valid
JSON, conforms to the minimal nbformat-4 schema, has the 10 cells
listed in the Session-14 brief, and doesn't accidentally commit
model weights as base64-embedded outputs.
"""
from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks" / "populate_tier1_cache.ipynb"
)


def _load() -> dict:
    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        return json.load(f)


def test_notebook_file_exists():
    assert NOTEBOOK.exists(), f"{NOTEBOOK} missing"


def test_notebook_is_valid_json():
    # Raises on invalid JSON; no assertion needed beyond completion.
    _load()


def test_notebook_has_expected_cell_count():
    nb = _load()
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 10


def test_notebook_cell_order_and_types():
    """First cell is the markdown header; cells 2-10 are code."""
    nb = _load()
    assert nb["cells"][0]["cell_type"] == "markdown"
    for i, cell in enumerate(nb["cells"][1:], start=2):
        assert cell["cell_type"] == "code", (
            f"cell {i} should be code, got {cell['cell_type']}"
        )


def test_notebook_has_no_saved_outputs():
    """Every code cell's `outputs` should be empty at commit time —
    outputs can contain base64-embedded data (including model
    weights), which bloats git history and leaks environment
    artefacts."""
    nb = _load()
    for i, cell in enumerate(nb["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs", []) == [], (
            f"cell {i} has non-empty outputs at commit time; "
            f"clear them before committing."
        )
        assert cell.get("execution_count") is None, (
            f"cell {i} has execution_count set; clear it before commit."
        )


def test_notebook_references_three_extractors():
    """The notebook must import each of ESM2Extractor,
    AlphaFoldExtractor, MaceOffExtractor at least once."""
    nb = _load()
    text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"]
    )
    for name in ("ESM2Extractor", "AlphaFoldExtractor", "MaceOffExtractor"):
        assert name in text, f"notebook does not reference {name}"


def test_notebook_offers_three_output_modes():
    """Cell 9's OUTPUT_MODE switch must enumerate drive / download /
    github_pat."""
    nb = _load()
    text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"]
    )
    for mode in ("drive", "download", "github_pat"):
        assert f'"{mode}"' in text, (
            f'notebook does not expose OUTPUT_MODE={mode!r}'
        )


def test_notebook_does_not_embed_model_weights():
    """No base64-embedded binary blobs in any cell source or output."""
    raw = NOTEBOOK.read_text(encoding="utf-8")
    # Rough heuristic: model weights usually appear as
    # application/octet-stream base64 chunks. Look for common
    # weight-file extensions in the raw JSON.
    for needle in (".pt\"", ".bin\"", ".safetensors\"", ".onnx\""):
        assert needle not in raw, (
            f"notebook raw JSON contains {needle!r}; "
            "don't commit model weights."
        )
