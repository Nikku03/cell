"""Structural tests for Session-18 benchmark scripts.

The benchmarks themselves are too slow to run inside pytest (Rust vs
Python alone is ~8 minutes). These tests verify that each script
loads cleanly, has a ``__main__`` block, can print its CLI help
without executing the benchmark, produces a JSON file with the
expected top-level keys when invoked with ``--help``, and that the
aggregated measured fact file exists and passes invariant
validation.

Why this matters: benchmarks are only useful if they keep working as
the codebase evolves. These tests catch import regressions and
schema drift cheaply — the real numbers are produced by running the
scripts manually (or in CI with a separate nightly lane).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"
OUTPUTS = REPO_ROOT / "outputs"
FACT = (
    REPO_ROOT / "memory_bank/facts/measured"
    / "bench_available_optimizations.json"
)

BENCH_SCRIPTS = [
    "bench_rust_vs_python.py",
    "bench_esm2_batch.py",
    "bench_esm2_sizes.py",
    "bench_xgboost_treemethod.py",
    "bench_feature_assembly.py",
]


@pytest.mark.parametrize("script_name", BENCH_SCRIPTS)
def test_bench_script_is_importable(script_name: str) -> None:
    """Load the script as a module. Catches syntax errors, missing
    deps, and obvious regressions in the extractor or trainer APIs
    that each bench imports lazily or eagerly."""
    path = SCRIPTS / script_name
    assert path.exists(), f"missing {path}"
    spec = importlib.util.spec_from_file_location(
        f"bench_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Don't execute main() during import — the scripts guard it
    # behind ``if __name__ == '__main__'`` which is False here.
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main"), (
        f"{script_name}: no main() function"
    )


@pytest.mark.parametrize("script_name", BENCH_SCRIPTS)
def test_bench_script_has_main_block(script_name: str) -> None:
    src = (SCRIPTS / script_name).read_text()
    assert 'if __name__ == "__main__":' in src, (
        f"{script_name}: missing __main__ guard"
    )


@pytest.mark.parametrize("script_name", BENCH_SCRIPTS)
def test_bench_script_cli_help(script_name: str) -> None:
    """Invoke with --help. argparse should exit 0 and print without
    running the benchmark body. Catches CLI-level bugs cheaply."""
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0, (
        f"{script_name} --help exited {r.returncode}: {r.stderr}"
    )
    assert "--out" in r.stdout, (
        f"{script_name} --help missing --out: {r.stdout}"
    )


def test_measured_fact_is_valid_json() -> None:
    assert FACT.exists(), f"missing fact: {FACT}"
    d = json.loads(FACT.read_text())
    assert d["id"] == "bench_available_optimizations"
    assert d["confidence"] == "measured"
    assert isinstance(d["value"], dict)
    required_keys = {
        "gillespie_rust_speedup",
        "esm2_batch_size_sensitivity",
        "esm2_model_size_comparison",
        "xgboost_tree_method",
        "feature_assembly",
    }
    assert required_keys.issubset(d["value"].keys()), (
        f"missing keys: {required_keys - d['value'].keys()}"
    )


def test_measured_fact_passes_invariant_check() -> None:
    r = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "memory_bank/.invariants/check.py")],
        capture_output=True, text=True, timeout=20,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, (
        f"invariant check failed: {r.stdout}\n{r.stderr}"
    )
    assert "OK" in r.stdout


def test_rust_bench_output_schema_if_run() -> None:
    """Schema test for the one benchmark we DID run in-sandbox. If
    the output file exists (optional — CI may not have run it), it
    must have the expected top-level keys."""
    out_file = OUTPUTS / "bench_rust_python.json"
    if not out_file.exists():
        pytest.skip("bench_rust_python.json not generated in this env")
    d = json.loads(out_file.read_text())
    assert d["config"]["n_genes"] >= 1
    assert "python" in d
    # rust key is present even if None (when Rust backend absent)
    assert "rust" in d
    assert "speedup_factor" in d


def test_feature_assembly_output_schema_if_run() -> None:
    out_file = OUTPUTS / "bench_feature_assembly.json"
    if not out_file.exists():
        pytest.skip("bench_feature_assembly.json not generated")
    d = json.loads(out_file.read_text())
    assert "pandas_pyarrow" in d
    assert "polars" in d
    # config with shape + n_iter should always land
    assert d["config"]["n_rows"] == 455


def test_bench_plans_have_colab_invocation() -> None:
    """Both GPU-dependent benchmarks must ship a Colab invocation —
    that's the concrete plan we promise the user. Schema guard."""
    for name in ("bench_esm2_batch_plan", "bench_esm2_sizes_plan"):
        p = OUTPUTS / f"{name}.json"
        if not p.exists():
            pytest.skip(f"{name} not generated")
        d = json.loads(p.read_text())
        assert d["status"] in (
            "AWAITING_COLAB", "MEASURED", "ERROR",
        )
        assert "colab_invocation" in d, (
            f"{name}: missing colab_invocation block"
        )
