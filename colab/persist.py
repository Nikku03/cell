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
    "localization.json",
    "string_degree.json",
    "data_impact.json",
    "full_run.json",
    "connectivity.json",
    "biosim.json",                 # the cell as a RUNNING program: perturbation propagation on the physical network -- near-field (functional module) is real, whole-cell response is null (structure vs dynamics) -- the perturb syscall           # knockout/robustness test: the software is CONNECTED (removing a line drops the engine) AND ROBUST (missing artifact runs degraded, no crash) -- like the cell, and robustness hides damage the same way               # end-to-end whole-software run summary (the full picture dashboard)            # multi-metric evaluator: each new layer scored across ALL engines -- localization/STRING lift essentiality (+0.06/+0.05) but disease barely (+0.01/-0.00); a win on one metric is not a win on the cell          # STRING v12.0 physical-PPI layer (degree+partners); connectivity line LIFTS reason 0.86->0.89; the ppi syscall
    "disease_reason.json",         # SECOND reasoner_core engine: disease-gene association (a different axis from essentiality); modest AUC 0.65 but calibrated 22->62% -- the disease syscall           # NEW LAYER from the science run: subcellular compartment (UniProt) for 100% of genes; predicts essentiality (nuclear/mito enriched, secreted depleted) and LIFTS reason 0.80->0.86 -- the loc syscall + 6th reason line  # canonical join backbone from the science reference tables (symbol/ensembl/uniprot/entrez + Human-GEM subsystem); 99% uniprot/entrez coverage         # answer to "can the HARD feedback-module level be predicted?" YES: topology fails (-0.39) but literature predicts the forward order (+0.85) -- reclassifies HARD->DATA                  # the software's own bill of materials: what it still needs (6/9 layers DATA-limited, 1 HARD) -- the 'needs' syscall            # the software's COMPLETED per-gene pathway-level table (tier-1 metabolic step + tier-2 signaling tier); 46% of membership genes get a trustworthy level, rest flagged (context-dependent/feedback) or abstained -- served by the 'level' syscall
    "biochem_limit.json",
    "kcat_flag.json",
    "kcat_verify.json",
    "davidi_kcat.json",            # Drive: 592-enzyme in-vivo kapp (max|v|/[E] over 13 NCI-60 conditions)
    "kcat_invivo_validate.json",   # HONEST non-circular test vs davidi_kcat: flag is WEAK (~70%), kapp does NOT predict kcat (0.08) -- supersedes the circular kcat_flag/kcat_verify
    "flag_base_vs_kcat.json",
    "fba_flux_coverage.json",
    "pathway_decode.json",
    "discover.json",
    "investigate.json",
    "cell_map.json",               # the CELL NETWORK MAP data (cell_map.py): 3,200-gene functional core laid out by compartment (concentric cell) with the physical-PPI + co-dependency edges; rendered by viz/cell_network_map.html -> the interactive artifact
    "dependency.json",             # the MEASURED Perturb-seq surveillance turned into a directed dependency network (dependency.py, 'knockout' syscall): for the ~9,871 knocked-out genes, knockout GENE shows the real up/down blast radius, and impact ranks genes by blast-radius size (PSMD14/GATA1/POLR3A/RPL7A/SNRPD2 top). HONEST: the direct measured effect is real, but dependencies do NOT chain -- only ~0.5% of a knockout's responders are the gene's own STRING partners (mostly distal/regulatory) and propagating one step to predict the 2nd-order effect is Spearman ~0.009 (~chance); the DIRECT knockout is trustworthy, the PROPAGATED one is not
    "xaira_coverage.json",         # staged the Xaira X-Atlas/Orion genome-wide Perturb-seq screen as a SECOND cell type (fetch_xaira.py streams the 8M-cell HCT116 dataset from HuggingFace one parquet batch at a time -> pseudobulk [pert x gene] signature -> {SCRATCH}/hct116.h5ad; peak disk ~1 batch). COVERAGE this adds: knockouts of cell-model genes 8,696 (K562) -> 15,749 UNION = 95.4% of the cell perturbed in >=1 cell type (+7,053 genes K562 never knocked out); 8,288 knockouts now in BOTH cell types (cross-cell-validatable); genes-measured readout 6,977 -> 16,383 (2.3x, Xaira sequences ~16k UMIs/cell). VALIDATED cross-cell-type consistency: on strongly-moved genes a knockout's HCT116 signature matches its K562 signature at Pearson 0.611 (shuffled 0.064) -- the same knockout does the same thing across leukemia & colorectal cells. Powers the `xcell` syscall (crosscell.py): cell-type-ROBUST vs cell-type-specific responders (PSMB5 proteostasis = robust r0.55; GATA1 hematopoietic TF = 0 robust, K562-only -- correctly cell-type-specific). NOTE: hct116.h5ad (1.1GB) lives in scratchpad (ephemeral, regenerated by fetch_xaira.py), not committed
    "pertseq_apps.json",           # the FOUR canonical Perturb-seq applications attempted on this K562 data, each validated & honestly graded (pertseq_apps.py, 'pertseq' syscall): [1] GRN WIRING **does NOT work on this data** -- inferred TF->TF influence edges recover curated SIGNED direct regulation only 1.6x chance with 48% sign agreement (=chance) and FFLs are 52% coherent (=chance) = a CO-RESPONSE network, not causal wiring; a single bulk steady-state condition can't separate direct from indirect (needs time-resolved/nascent-RNA data). [2] GENE FUNCTION by transcriptomic fingerprint **WORKS** -- fingerprint-kNN recovers a held-out known gene's pathway 84% (4.1x chance), 128 confident dark-gene calls (TIMM23B->mito import, GPN3->RNA-pol assembly); a validated similarity HINT (can be fooled by shared growth-arrest phenotype), 'pertseq function GENE'. [3] DISEASE CONVERGENCE method validated (spliceosome positive control z=156) but thin (only 1 disease has enough knocked-out risk genes; K562 wrong context for neuro/immune disease). [4] IMMUNE brakes-vs-gas NOT feasible (needs +/- stimulus axis + immune cells; this is one unstimulated leukemia condition). Honest scorecard: one works, one validated hint, one method-awaiting-data, one out-of-scope
    "influence.json",              # perturbation used for ONE thing (influence.py, 'cascade'/'trace' syscall): a directed network of what a knockout AFFECTS (the measured ENDS -- we know the endpoints, not the mechanism), then each effect's ROUTE reconstructed WITHOUT predicting anything unmeasured -- DIRECT (X physically/causally contacts P), MEASURED-MEDIATED (a specific non-hub stepping-stone M that is itself a measured knockout: X->M->P, both hops real data -- this is where the OTHER knockouts are kept in context), or UNRESOLVED. VALIDATED honest on this screen: ~1.2% direct (6.5x chance), ~14% mediated (43x a placebo -- stepping-stones are non-random), ~85% unresolved (far-field that doesn't compose). The influence network is 100% real data; the route is reconstructable for a validated MINORITY. CONFOUND CONTROLLED: pan-hub mediators (whose KO moves everything) excluded, and PPI/pathway structure used only to name the DIRECT entry point (as a mediator filter it just re-selects hubs and doesn't help)
    "protein_knockout.json",       # knock out the PROTEIN, not the gene (protein_knockout.py, 'protein'/'degrade' syscall): degrading a protein (PROTAC-style) removes it from EVERY complex it's a subunit of at once -> the STRUCTURAL disassembly (SMARCA4 collapses all 4 SWI/SNF variants, PSMA1 all 4 proteasome assemblies), leaving co-subunits expressed but their assembly incomplete. VALIDATED honest: the STRONG signal is structural -- 46% of complex proteins have a complex-mate among their top co-dependencies (vs 0.04% chance, ~1000x), machine parts are co-essential; the measured TRANSCRIPTIONAL signal is deliberately WEAK (median 1.07x mate movement, only 18% of machines >1.5x) because a protein KO acts POST-transcriptionally and mRNA-seq is mostly blind to it (proteasome 3.1x / spliceosome are the feedback-wired exceptions) -- which is exactly why the structural view is NOT redundant with the measured gene `knockout`. LIMIT: one-protein-per-gene (no isoform/PTM/domain-resolved network -> no isoform-specific KO); names the machines that break, not the far-field cascade
    "network.json",                # the complete-cell database wired into ONE queryable multi-layer graph (network.py, 'network' syscall): 16,509 genes, INTERCONNECTED (physical: ppi/complex/ligand-receptor, 208,735 edges) + INTERDEPENDENT (functional: causal/regulatory/signaling/co-dependency/co-expression/synthetic-lethal, 922,265 edges). VALIDATED honest: the two families are COMPLEMENTARY (only ~6% of a functional edge is also physical -> who-NEEDS != who-TOUCHES) yet COHERENT (that overlap is 46-214x the 0.14% chance rate; PPI partners share a pathway 1.6x, complex 3.6x, causal 2.4x above chance); one giant component per family. CAVEAT: co-dependency/co-expression partners do NOT map to curated literature pathways (they track physical modules instead); edges are observed/curated, not predicted; and (per dependency.py/causal_reach) the edges do not let you PROPAGATE an unmeasured knockout's far-field cascade -- structure wired & real, long-range dynamics not
    "anomaly.json",                # patched connections + measured perturbation into one map and scanned for disagreement: COMPARTMENT-OUTLIER is a REAL annotation-error detector (FASTKD5 labeled nucleus but module mito 10/10 = mislabel; ACTR6/ELP5/DIS3/EXOSC10 corrected by their module), but the perturbation-based anomalies (decoupled/hidden-link) are CONFOUND-dominated -- weak K562 signal + stress axis read as anomalies; trust the cross-annotation one, not the perturbation one            # predicted investigative DOSSIERS for under-recorded genes (record/family/destination/path/timing/interactors/surveillance/required-support), closest-relatives-first; validated held-out: JOB ~55% (2x baseline), DESTINATION ~54% (2.4x); THE UNDERWORLD = dark genes embedded in characterized modules (EMC7->EMC complex, MTG2->mito translation, NCLN->ER Nicalin) -- the investigate syscall               # aim the reasoning machinery at the UNKNOWNS: SELECTIVE-dependency discovery is the WIN (skew of raw DepMap gene-effect + robustness floor recovers 20/22 known oncology targets in top-500, 33.5x enrichment; surfaces novel selective targets SHLD3/CD24/FGFR2/MAF); druggable = selective+surface (TNFSF10/CD24/FZD5, partial - no Pharos); disease_new coarse (0.65); dark-gene FUNCTION is an honest NULL (label transfer scatters on the dark proteome, margin 0.02) -- the discover syscall
    "coverage_stack.json",
    "causal_reach.json",
    "causal_patch.json",
    "causal_expand.json",         # PATCHED the genome-wide K562 Perturb-seq into the debugger: causal reach 2,058 -> 9,871 measured knockouts (+7,813); essential precision preserved (biology-recovery 16.5%->16.1%), new gwps-only genes weak (1.2% vs 0% random) and FLAGGED low-precision in strace/whodunit output. A measured-but-weak answer beats the inference null (causal_reach/causal_patch ~0). Merge default-ON when gwps.h5ad staged (PERTURB_MERGE=0 to disable)           # tested the on-disk shortcut to extend causal reach: does DepMap co-essentiality (genome-wide, 18443 genes) predict the measured Perturb-seq removal response? NULL (+0.009 vs structure +0.015 vs chance -0.001) -- fitness-coupling != transcriptional response; no cheap on-disk substitute, causal reach needs ACTUAL multi-context interventions (Replogle-Nadig 4-cell-line)           # honest limit for "remove any piece and calculate its effect": structure-alone propagation reproduces biosim's whole-field +0.02 ONLY as the distance envelope; WITHIN a fixed-distance shell it is NULL at every hop (1-hop -0.02, 2-hop -0.05, 3-hop -0.10) and the structure-only cause screen is near-chance -- the census cannot compute a removal's effect; genuine cause-and-effect lives in the intervention data (Perturb-seq), so the bottleneck is SURVEILLANCE (more measured interventions), not more census
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
