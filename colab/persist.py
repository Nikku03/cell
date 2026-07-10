"""persist — save/restore the trained artifacts + expensive derived features to Google Drive, so a Colab
disconnect does NOT throw away the trained model, the healed-cell ledger, or (crucially) the derived features
from big datasets that take real time to compute.

Two things it persists:
  1. ARTIFACTS  — the small trained outputs (combiner model, ledger, reports, additions). Cheap to save; means
                  a reconnect restores the exact trained state instead of retraining.
  2. CACHES     — expensive DERIVED features keyed by a content tag (e.g. a co-expression correlation matrix
                  built from Tahoe/DepMap). Computed once, reloaded forever. cache_feature/load_cached_feature.

Everything is a plain file copy to  <drive>/cell_model/artifacts|caches/ .  Missing files are skipped, never
an error, so it is safe to call unconditionally at the end of any run.
"""
import os, shutil, glob

OUT = "outputs/orphan"
ARTIFACTS = [
    "signal_combiner.pkl", "signal_combiner_validation.json",
    "signal_combiner_reg.pkl", "signal_combiner_reg_validation.json",       # whole-cell: per-relation combiners
    "signal_combiner_sig.pkl", "signal_combiner_sig_validation.json",
    "phase2_ledger.json", "phase2_loop_report.json",
    "cell_v2_additions.json", "cell_v3_additions.json",
    "phase3_depmap_validation.json", "kinetics_refined_corrected.json",
    "ghost_patch.json", "recovery_scorecard.json",
    "tahoe_vecs.npz",          # derived Tahoe per-gene vectors (58 MB) — cache so the 900 MB download is one-time
    "depmap_vecs.npz", "expr_vecs.npz",   # derived z-scored matrices — skip the 419/305 MB CSV re-parse (time+RAM)
    "extra_data_report.json", "causal_edges.json", "reactome_pathways.json",   # Reactome + SIGNOR/CollecTRI overlays
    "tf_motifs.json", "complexes_extra.json", "new_data_report.json",          # data-hunt overlays (motifs, complexes)
    "disorder.json", "darkness.json", "domains.json", "metabolites.json",      # data-hunt: disorder/Pharos/domains/HMDB
    "structure.json", "concentration.json", "translation.json",                # data-hunt: pLDDT/copies/translation
    "enhancers.json", "chip_reg.json", "ppi_extra.json",                       # data-hunt: enhancers/ChIP/extra-PPI
    "celltype_capacity.json", "saturation_kapp.json", "incell_rates_validation.json",  # in-cell rates from kcat
    "domain_ppi_validation.json", "metabolite_enzyme.json",                    # new inter-layer connections
    "rprom_validation.json", "blind_target_report.json", "blind_sweep_report.json",           # reg<->metabolic + cancer sweep
    "cancer_cell_map.json", "msih_cell_map.json", "full_cell_map.json", "context_dependency.json",
    "gene_additions.json", "gene_coverage.json", "discovery_validation.json", "discovery_engine.json", "variant_effect_cache.json", "blind_variant_test.json", "mutation_walk.json", "structural_context_validation.json", "full_test_scorecard.json", "interface_analysis.json", "gof_caller_validation.json",   # P0 + discovery-engine pieces   # perturbed-cell + MSI-H + full-stack + dependency attribution
]


def _dirs(drive):
    a = os.path.join(drive, "cell_model", "artifacts")
    c = os.path.join(drive, "cell_model", "caches")
    os.makedirs(a, exist_ok=True); os.makedirs(c, exist_ok=True)
    return a, c


def save_to_drive(drive="/content/drive/MyDrive"):
    """copy every existing artifact from outputs/orphan -> Drive. Returns the list saved."""
    adir, _ = _dirs(drive)
    saved = []
    for f in ARTIFACTS:
        src = os.path.join(OUT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(adir, f)); saved.append(f)
    print(f"saved {len(saved)} artifacts -> {adir}")
    for f in saved:
        print("   ", f)
    return saved


def restore_from_drive(drive="/content/drive/MyDrive"):
    """copy artifacts from Drive -> outputs/orphan if present. Returns the list restored (empty = fresh run)."""
    adir, _ = _dirs(drive)
    os.makedirs(OUT, exist_ok=True)
    restored = []
    for f in ARTIFACTS:
        src = os.path.join(adir, f)
        if os.path.exists(src):
            if f.endswith("_vecs.npz") and os.path.getsize(src) / 1e9 > 2.0:   # poisoned derived cache -> purge it
                print(f"  purging poisoned cache from Drive: {f} ({os.path.getsize(src)/1e9:.1f} GB)")
                try:
                    os.remove(src)
                except OSError:
                    pass
                continue
            shutil.copy(src, os.path.join(OUT, f)); restored.append(f)
    print(f"restored {len(restored)} artifacts from {adir}" if restored else
          f"no saved artifacts in {adir} (fresh run)")
    return restored


# ---- expensive derived features: compute once, reuse forever ----
def cache_path(tag, drive="/content/drive/MyDrive"):
    _, cdir = _dirs(drive)
    return os.path.join(cdir, tag)


def load_cached_feature(tag, drive="/content/drive/MyDrive"):
    """return the local path to a cached derived-feature file (restoring it from Drive), or None."""
    p = cache_path(tag, drive)
    if os.path.exists(p):
        local = os.path.join(OUT, tag)
        if not os.path.exists(local):
            shutil.copy(p, local)
        print(f"cache HIT  {tag}")
        return local
    print(f"cache MISS {tag} (will compute + save)")
    return None


def cache_feature(tag, local_path, drive="/content/drive/MyDrive"):
    """save a freshly-computed derived-feature file to Drive under `tag`."""
    if os.path.exists(local_path):
        shutil.copy(local_path, cache_path(tag, drive))
        print(f"cached {tag} -> Drive")


if __name__ == "__main__":
    import sys
    drive = sys.argv[2] if len(sys.argv) > 2 else "/content/drive/MyDrive"
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_to_drive(drive)
    else:
        restore_from_drive(drive)
