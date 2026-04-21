# Layer 0 decisions taken in Session 3

The user authorized "full system" autonomy, so the four Phase A open questions were resolved with the defaults I had recommended:

1. **Data staging location**: `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` (option **a**), matching the path the existing `cell_sim/layer0_genome/syn3a_real.py` loader already uses. Minimises changes to working code. Binary files live there and are kept out of git by the existing `data/` rule in `.gitignore`.
2. **Gene-table representation**: CSV + pointer fact (option **b**). `memory_bank/data/syn3a_gene_table.csv` with a single `facts/structural/syn3a_gene_table.json` pointer.
3. **Breuer 2019 essentiality labels**: source file to be resolved in Layer 6 Phase A (when we get there). For now, `breuer_2019_elife` is registered as a source with the caveat that the specific supplementary file / column layout is not yet pinned.
4. **Prototype `.py` files at the repo root**: leave as read-only history (option **a**). Not referenced by any layer code going forward.

These defaults can be revisited. If the user objects to any of them, the reversal is mechanical (rename a directory, edit a few paths, re-run the invariant checker).
