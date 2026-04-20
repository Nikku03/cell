# cell_sim bundle — files to add to github.com/Nikku03/cell

This bundle contains every new or updated file from our recent work
together. Everything is organized so you can drop each file directly
into its matching location in your repo.

## What's in this bundle

### `code/cell_sim/` — new/updated Python source
Drop these into the existing `cell_sim/` subfolder of your repo.

    layer2_field/rust_dynamics.py
        RustBackedFastEventSimulator — drop-in replacement for
        FastEventSimulator that uses the Rust extension for a 2.3×
        speedup. Inherits everything except _step.

    layer3_reactions/nutrient_uptake.py
        The nutrient-uptake patch. Adds 15 transport rules (GLCpts,
        O2t, GLYCt, FAt, CHOLt, TAGt, synthetic ADEt/GUAt/URAt/CYTDt)
        with literature-informed k_cats so the 100%-scale cell doesn't
        starve. This is what you need for the 100%-scale Colab run.

    tests/render_whole_cell.py
        100%-scale whole-cell runner with progress streaming, chunked
        simulation, trajectory CSV output. Now supports the uptake
        patch via WC_WITH_UPTAKE=1 (default on).

    tests/compare_uptake.py
        Side-by-side harness: runs baseline vs patched at a chosen
        scale, writes comparison plot. Good for quick (~5 min)
        verification at 25% scale before committing to a 30-min
        100% run.

    tests/test_rust_equivalence.py
        Bit-identity regression test: verifies RustBackedFastEventSimulator
        produces identical events to FastEventSimulator. 27,463 events
        tested, exact f64 match required.

    tests/demo_priority3.py
        Updated with the null-similarity fix for MACE backend
        (was crashing when formatting info.kcat_estimate.similarity).

### `code/cell_sim_rust/` — new Rust extension crate
This is a NEW sibling folder to cell_sim/ in your repo root. Add it
as-is.

    Cargo.toml, pyproject.toml      Build config
    src/lib.rs                      ~400 lines of Rust
    target/wheels/cell_sim_rust-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl
        Pre-built wheel for Colab (linux x86_64, Python 3.12).
        Install with: pip install target/wheels/cell_sim_rust-*.whl

### `notebook/cell_sim_colab.ipynb`
The Colab notebook, updated with Section 11 (whole-cell simulation
at 100% scale, fed vs starving comparison). Goes in cell_sim/ in
your repo (same path as before — replaces the existing file).

### `docs/` — writeups
Add these to a new docs/ folder in your repo root.

    PHASE1_RESULTS.md        — 11× vectorized speedup, bit-identical
    PHASE2_RESULTS.md        — next-reaction method (reference impl)
    PHASE3_RESULTS.md        — Rust hot-path, another 2.3× speedup
    PHASE4_UPTAKE_NOTES.md   — nutrient-uptake diagnosis + patch rationale

## Apply in one shell session

    # Unzip this bundle somewhere (e.g. ~/Downloads/cell_sim_bundle/)
    cd ~/path/to/your/cell/repo

    # 1. Python source
    cp -v ~/Downloads/cell_sim_bundle/code/cell_sim/layer2_field/rust_dynamics.py \
          cell_sim/layer2_field/
    cp -v ~/Downloads/cell_sim_bundle/code/cell_sim/layer3_reactions/nutrient_uptake.py \
          cell_sim/layer3_reactions/
    cp -v ~/Downloads/cell_sim_bundle/code/cell_sim/tests/*.py \
          cell_sim/tests/

    # 2. New Rust sibling folder
    mkdir -p cell_sim_rust
    cp -rv ~/Downloads/cell_sim_bundle/code/cell_sim_rust/* cell_sim_rust/

    # 3. Notebook
    cp -v ~/Downloads/cell_sim_bundle/notebook/cell_sim_colab.ipynb \
          cell_sim/

    # 4. Docs
    mkdir -p docs
    cp -v ~/Downloads/cell_sim_bundle/docs/*.md docs/

    # 5. Commit and push
    git add cell_sim/layer2_field/rust_dynamics.py \
            cell_sim/layer3_reactions/nutrient_uptake.py \
            cell_sim/tests/render_whole_cell.py \
            cell_sim/tests/compare_uptake.py \
            cell_sim/tests/test_rust_equivalence.py \
            cell_sim/tests/demo_priority3.py \
            cell_sim/cell_sim_colab.ipynb \
            cell_sim_rust/ \
            docs/
    git commit -m "Add Phase 3 Rust core + Phase 4 nutrient uptake + whole-cell runner"
    git push

## Running the 100% whole-cell simulation

After pushing, open the notebook on Colab:

    1. Runtime → Change runtime type → High-RAM CPU (important!)
    2. Section 1: clone repo — already wired for your username
    3. (Optional) Upload cell_sim_rust/target/wheels/*.whl to Colab for
       2.3× speedup — notebook Section 11.1 will auto-detect it, OR
       build from source in Colab:
           cd cell_sim_rust
           !pip install maturin
           !maturin build --release
           !pip install target/wheels/cell_sim_rust-*.whl
    4. Run all cells

Section 11 runs:
    - 11.2: fed cell (WC_WITH_UPTAKE=1) — ~20 min wall at 100% scale
    - 11.2b: starving cell (WC_WITH_UPTAKE=0) — another ~20 min
    - 11.3: side-by-side trajectory plots (ATP, AMP, G6P, PPi)
    - 11.4: top-20 reactions chart (uptake rules highlighted green)
    - 11.5: download all outputs as CSV/PNG

Expected total runtime: ~50-60 min for the fed+starving comparison.

## Quick verification before the long run

If you want to sanity-check the patch works before burning 60 min:

    cd cell_sim
    WC_SCALE=0.25 WC_T_END=1.0 python tests/compare_uptake.py

Takes ~5 min at 25% scale, produces data/whole_cell_compare/comparison.png.
If the nucleobase pools rise with uptake ON (they should), the patch
is functioning correctly.

## What you'll learn from the 100% run

Hypothesis: with the uptake patch, the cell stops decaying because
glucose can enter, adenines are salvaged, and glycerol is replenished.

If confirmed: ATP/ADP/AMP stabilize, PPi stops crashing, G6P pool
holds steady. This is a clean "cell can eat now" demonstration.

If NOT confirmed: something scale-dependent is happening beyond
just the food-in problem (enzyme-count clamp artifacts, substrate
saturation regime changes, etc.) — also a publishable finding, but
more diagnostic work needed.
