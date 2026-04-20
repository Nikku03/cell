# Priority 1 removal — patch

Priority 1 (naive forward-only stoichiometric coupling) is deprecated.
Priority 1.5 is a strict superset: same metabolite coupling, but with
reversibility, Michaelis-Menten saturation, and medium uptake added.
Keeping Priority 1 around as a runnable deliverable was misleading
because substrates drained to zero in ~200 ms.

## What this patch does

Modifies 5 files and deletes 1.

### Files to REPLACE (inside this tarball)

    cell_sim/layer3_reactions/coupled.py   (401 lines → 131 lines)
        Kept: metabolite utilities (mM↔count, initialize_metabolites,
              get_species_count, update_species_count,
              AVOGADRO, INFINITE_SPECIES, COUNTABLE_THRESHOLD)
        Removed: make_coupled_catalysis_rule, build_coupled_catalysis_rules,
                 the __main__ demo driver

    cell_sim/cell_sim_colab.ipynb          (25 cells → 22 cells)
        Removed: Section 5 "Priority 1 — stoichiometric coupling" (3 cells)
        Updated: intro list now reads "three progressively richer modes"
                 instead of four; downstream sections renumbered 5→6→7→8→9
        Fixed: download cell no longer references coupled_syn3a.mp4

    cell_sim/README.md
        Removed: Priority 1 status bullet, Priority 1 row from benchmark
                 table, render_coupled.py from render-script list
        Updated: note added explaining why render_coupled.py is gone;
                 directory-layout tree annotates coupled.py as utilities

    cell_sim/docs/DESIGN.md
        Updated: layer3 file descriptions rewritten so coupled.py is
                 described as "metabolite utilities" and reversible.py
                 is described as "the main simulator"

    cell_sim/docs/BUILD_LOG.md
        Kept intact: Phase 3 historical record of Priority 1 as it was built
        Added: Phase 6 section documenting the removal and what was kept

### File to DELETE

    cell_sim/tests/render_coupled.py    (no longer has anything to render)

## How to apply (one command each)

From the top of your local checkout of Nikku03/cell:

    tar -xzf priority1_removal_patch.tar.gz        # overwrites 5 files
    git rm cell_sim/tests/render_coupled.py
    git add cell_sim/
    git commit -m "Remove Priority 1; keep coupled.py as metabolite utilities"
    git push

## What doesn't change

- Priority 1.5 and Priority 2 work exactly as before — they never used
  the deleted rule builders, only the shared utilities (which stay).
- No other file in layer2_field/, layer0_genome/, or the remaining tests
  needs any change.

## Verified before shipping

- Slimmed coupled.py still exports all utilities that reversible.py and
  gene_expression.py import.
- Priority 1 rule builder imports correctly raise ImportError.
- End-to-end Priority 1.5 smoke test from the patched repo: 9,329 events
  in 17.9 s wall over 100 ms simulated, ATP 73,738 → 71,306.
- Notebook JSON parses cleanly; 22 cells; section numbering consistent.
