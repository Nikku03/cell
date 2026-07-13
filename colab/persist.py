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
    "cell_complete.json",   # 37MB core model data (restored/saved with the rest)
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
    "reasoning_chain.json", "reasoning_chain_test.json", "reasoning_chain_multitest.json",   # chain + blind ClinVar tests (VHL + 6-protein panel)
    "enzyme_records.json", "chip_reg_edges.json",   # joined kinetic+context records + measured HepG2 ChIP TF->target edges
    "struct_discovery_test.json",   # confound-controlled test: does a structural feature lift the HARD/discovery AUC
    "interface_discovery_test.json", "interface_pairs.json",   # fetched AF-Multimer interface vs discovery + Colab pair set
    "perturb_prioritizer.json", "cellformer.json",   # interventional prioritizer + transformer-style next-knockout prediction
    "epistasis.json",   # additive-vs-synergy test of combination perturbations (Norman 2019 doubles)
    "humap_validation.json", "humap_complexes.json",   # hu.MAP complexes: REJECTED as a coverage booster (r~0.09)
    "edge_predict_validation.json",   # fixed-model edge types tested for knockout-predictivity: only complexes work
    "propagation_test.json",   # software analogy test: does knockout perturb annotated connections? (no -- cell is robust)
    "fba_essentiality.json",   # objective-driven FBA (grow+reproduce under physics) predicts knockout survival AUC 0.70
    "perturb_screens_combined.json",   # tried all public Perturb-seq screens: +19 genes, 55% ceiling is data-gen-limited
    "litmine.json",   # PubMed literature extracted for dark genes (grounded + DOIs) -- descriptive layer
    "grn_validation.json", "grn_diag.json",   # the cell as RUNNING software (dynamical GRN): converges+robust, but knockout fragility does NOT predict essentiality (honest null, gain-robust) -- 'boot' syscall
    "grn_reprogram.json",   # TF reprogramming: forcing a master TF ON induces its OWN lineage program (8/9, AUC 0.99) -- the FORWARD win vs the backward essentiality null -- 'induce' syscall
    "cellos_synthesis.json",   # top-down(FBA)+bottom-up(Perturb-seq) fusion: blind in different places -> router covers ~2x (4,082 genes, effective AUC 0.77 vs 0.71 single) -- 'assess' syscall
    "cellsim.json",   # checkpoint-anchored simulation: data anchors reconstruct a held-out KO response (r 0.42->0.50), mechanistic sim adds ~nothing (r0.01) -- 'cellsim' syscall
    "coverage.json",   # honest whole-cell coverage map: response prediction 55% (deepest, data-limited), but >=1 trustworthy answer for 99.4% of the genome, only 0.6% truly dark -- 'coverage' syscall
    "cross_cell_line.json",
    "zhu_coverage.json",
    "zhu_targeted_genes.txt",
    "reason.json",
    "reason_mutation_demo.json",   # fuse variant-reasoning x cell-dependency on a real panel (TP53 R175H/P72R, MLH1, HBB sickle) -- 'mutate' syscall
    "reason_reactions.json",       # the reasoning PRINCIPLE (reasoner_core) applied to REACTIONS: transfers mechanically, but value is substrate-dependent (indep lines reason to only 0.66 vs FBA truth; genes converge, reactions don't)
    "pathway_position.json",       # NO-measured-layer property (a protein's LEVEL in a pathway): two independent textbook sources (KEGG topology + PubMed literature) cross-verify + fill gaps -> glycolysis 0.99/0.87, TCA 0.32/0.93, fused 0.99/0.90
    "pathway_tier.json",           # tier-2: LEVEL for the ~10k signaling/regulatory MEMBERSHIP genes via SIGNOR directed graph + SCC feedback handling; feed-forward tier trustworthy (+0.55, clean cascades 0.85-0.94), feedback modules flagged (abstain); whole-cell census 47% trustworthy / 3% feedback / 50% no-context
    "metabolic_levels.json",       # tier-1 scaled to all 92 KEGG metabolic maps: ~750 real enzymes get a step/cycle level (glycolysis 0.90); cyclic maps flagged
    "cell_levels.json",
    "needs.json",
    "feedback_order.json",
    "id_map.parquet", "id_map_summary.json",
    "localization.json",           # NEW LAYER from the science run: subcellular compartment (UniProt) for 100% of genes; predicts essentiality (nuclear/mito enriched, secreted depleted) and LIFTS reason 0.80->0.86 -- the loc syscall + 6th reason line  # canonical join backbone from the science reference tables (symbol/ensembl/uniprot/entrez + Human-GEM subsystem); 99% uniprot/entrez coverage         # answer to "can the HARD feedback-module level be predicted?" YES: topology fails (-0.39) but literature predicts the forward order (+0.85) -- reclassifies HARD->DATA                  # the software's own bill of materials: what it still needs (6/9 layers DATA-limited, 1 HARD) -- the 'needs' syscall            # the software's COMPLETED per-gene pathway-level table (tier-1 metabolic step + tier-2 signaling tier); 46% of membership genes get a trustworthy level, rest flagged (context-dependent/feedback) or abstained -- served by the 'level' syscall
    "biochem_limit.json",
    "kcat_flag.json",
    "kcat_verify.json",
    "davidi_kcat.json",            # Drive: 592-enzyme in-vivo kapp (max|v|/[E] over 13 NCI-60 conditions)
    "kcat_invivo_validate.json",   # HONEST non-circular test vs davidi_kcat: flag is WEAK (~70%), kapp does NOT predict kcat (0.08) -- supersedes the circular kcat_flag/kcat_verify
    "flag_base_vs_kcat.json",
    "fba_flux_coverage.json",
    "pathway_decode.json",
    "coverage_stack.json",
    "coverage_stack_genes.txt",
    "celltype_screen_gain.json",   # payoff meter for a fetched complementary-cell-line screen: NEW measured genes it adds vs K562 (self-checked: RPE1 -> 0)   # K562 vs RPE1: RPE1 adds 0 new genes (essential-scoped); responses transfer only weakly (r0.19 vs 0.06) -> 55% is PER-CELL-LINE, knockout effects are context-specific
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
