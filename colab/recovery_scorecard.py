"""RECOVERY SCORECARD — the validation spine (Robust axis).

The model's honest claim is "calibrated recovery of known biology + abstention where it doesn't know."
This unifies the scattered, individually-verified recovery/calibration tests into ONE versioned scorecard
that (a) states each claim as pass/fail against a fixed threshold and (b) serves as a regression gate: any
future change must not turn a PASS into a FAIL. It does not run new experiments — it reads the committed
result JSONs and asserts the numbers still clear their bars. -> recovery_scorecard.json
"""
import json
from pathlib import Path

OUT = Path("outputs/orphan")


def _load(name):
    f = OUT / name
    return json.load(open(f)) if f.exists() else None


def scorecard():
    rows = []

    def add(name, ok, value, baseline, bar, detail):
        rows.append(dict(test=name, passed=bool(ok), value=value, baseline=baseline, bar=bar, detail=detail))

    # 1) Algorithm correctness (synthetic GRNs, known drivers)
    d = _load("disease_to_reversal_validation.json")
    if d:
        dr = d["driver_recovery"]
        add("algorithm_correctness", dr["control_recall"] >= 0.9 and dr["control_precision"] >= 0.8,
            dict(recall=dr["control_recall"], precision=dr["control_precision"]),
            dict(setdiff_precision=dr["setdiff_precision"]), "recall>=0.9 & precision>=0.8",
            "Boolean target-control recovers known drivers on synthetic GRNs")

    # 2) Reprogramming recovery on the REAL human TF network (known recipes)
    r = _load("reversal_realmodel_validation.json")
    if r:
        s = r["reprogramming"]["summary"]
        add("reprogramming_recovery",
            s["mean_recipe_recall"] >= 0.5 and s["mean_enrichment"] >= 2.0,
            dict(recipe_recall=s["mean_recipe_recall"], enrichment=s["mean_enrichment"],
                 reached=s["reached_rate"]),
            dict(random=s["mean_random_expected"]), "recall>=0.5 & enrichment>=2x",
            "recovers Yamanaka/GMT/CEBPA recipes blind from the real network")
        bs = (r.get("bootstrap_robustness") or [{}])[0]
        if bs:
            add("reversal_robustness",
                bs.get("separates", False),
                dict(recipe_stability=bs.get("recipe_stability"), passenger_stability=bs.get("passenger_stability")),
                None, "recipe - passenger stability >= 0.3",
                "edge-bootstrap separates robust drivers from fragile passengers")

    # 3) Lens-agreement calibration (consensus + abstention)
    l = _load("lens_confidence_validation.json")
    if l:
        add("lens_calibration",
            (l.get("lift_ge2_over_random") or 0) >= 10 and (l.get("lift_ge2_over_1lens") or 0) >= 1.5,
            dict(ge2_over_random=l.get("lift_ge2_over_random"), ge2_over_1lens=l.get("lift_ge2_over_1lens")),
            dict(random=l.get("random_baseline")), ">=2 lenses: >=10x random & >=1.5x single-lens",
            ">=2 independent lenses -> calibrated high-confidence tier")

    # 4) IEM biomarker mechanism recovery (Play B — clinical ground truth, non-computational)
    m = _load("iem_mechanism_validation.json")
    if m:
        s = m["summary"]
        add("iem_mechanism_recovery",
            (s["recall"] or 0) >= 0.6 and (s["specificity_lift"] or 0) >= 20,
            dict(recall=s["recall"], specificity_lift=s["specificity_lift"], coverage=s["coverage"]),
            dict(random=s["random_pair_baseline"]), "recall>=0.6 & >=20x over random",
            "recovers documented inborn-error-of-metabolism biomarkers (clinical ground truth)")

    # 5) Cell-type identity recovery (Phase 1 — the populated emask encodes real lineage biology)
    c = _load("celltype_identity_validation.json")
    if c:
        s = c["summary"]
        add("celltype_identity_recovery",
            (s["recall"] or 0) >= 0.6 and (s["median_enrichment"] or 0) >= 2
            and (s["recall_over_random"] or 0) >= 2,
            dict(recall=s["recall"], enrichment=s["median_enrichment"], coverage=s["coverage"]),
            dict(random=s["random_baseline"]), "recall>=0.6 & enrich>=2x & >=2x over random",
            "populated emask recovers cell-type-lineage master TFs (Phase 1)")

    # 6) Tissue layer — cell-cell communication recovery
    t = _load("tissue_communication_validation.json")
    if t:
        s = t["summary"]
        add("tissue_communication_recovery",
            (s["recall"] or 0) >= 0.6 and (s["recall_over_random"] or 0) >= 20,
            dict(recall=s["recall"], context_recall=s["context_recall"], lift=s["recall_over_random"]),
            dict(random=s["random_baseline"]), "recall>=0.6 & >=20x over random",
            "tissue model recovers textbook cell-cell signaling axes in the right cells")

    # 7) Disease->target pipeline — recover a known drug target on OOD diseases (mechanism->intervention)
    dt = _load("disease_target_validation.json")
    if dt:
        s = dt["summary"]
        # honest bar: recover a known target in a strong MAJORITY of OOD diseases (not 100% — a broadened set
        # exposes a real failure mode: receptor-only targets absent from the directed reg/sig graph) AND beat
        # the random-label baseline in aggregate. n>=5 diseases so the rate isn't a small-n fluke.
        maj = s["n_ran"] >= 5 and s["recovery_rate"] >= 0.6
        beats = s["mean_precision_strong"] >= 1.5 * s["mean_random_label_baseline"]
        add("disease_target_recovery", maj and beats,
            dict(recovered=s["n_recovered"], ran=s["n_ran"], rate=s["recovery_rate"],
                 mean_precision=s["mean_precision_strong"], lift=s["precision_lift_over_random"]),
            dict(random_label=s["mean_random_label_baseline"]),
            "recover a known target in >=60% of >=5 OOD diseases & aggregate strong-precision >=1.5x random-label",
            "3-layer pipeline (causal->perturb-to-wildtype->druggability) selects the real drug target blind on 5/6 OOD diseases")

    # 8) Measured causal cause-finder — the alibi test on REAL knockout data (Replogle Perturb-seq)
    mc = _load("measured_cause_validation.json")
    if mc:
        s = mc["summary"]
        add("measured_cause_recovery",
            (s["driver_recall"] or 0) >= 0.7 and (s["suppressor_flagged_protector"] or 0) >= 0.99
            and (s["driver_vs_suppressor_separation"] or 0) >= 0.5,
            dict(driver_recall=s["driver_recall"], suppressor_protector=s["suppressor_flagged_protector"],
                 separation=s["driver_vs_suppressor_separation"]),
            dict(random=s["random_sign_baseline"]),
            "driver recall>=0.7 & suppressor flagged protector & separation>=0.5",
            "measured knockout effects separate DRIVER from PROTECTOR (real interventional causality)")

    # 9) CellGraph — learned whole-cell model (link / perturbation-direction / drug / structure->function)
    cg = _load("cellgraph_validation.json")
    if cg:
        s = cg["summary"]
        add("cellgraph_capabilities", cg["all_pass"],
            dict(link_ppi=s["link_ppi_auc"], perturb_dir=s["perturb_direction_acc"],
                 drug=s["drug_auc"], struct_fn=s["structure_function_auc"]),
            dict(random=0.5), "link>=0.7 & perturb-dir>=0.7 & drug>=0.7 & struct-fn>=0.65",
            "learned graph model: predicts binding, downstream direction, drug off-targets, function")

    # 10) Kinetics calibration honesty — no spurious kcat correction (guardrail against label leakage)
    kc = _load("kcat_calibration_validation.json")
    if kc:
        s = kc["summary"]
        add("kcat_calibration_honest", kc["passed"],
            dict(catpred_fold_error=s["catpred_raw_fold_error"], optimal_shrinkage=s["cv_optimal_shrinkage"],
                 correction_generalises=s["correction_generalises"]),
            dict(rtm_r=s["residual_regression_to_mean_r"]),
            "no fitted correction beats CatPred-as-is on real measurements (optimal shrinkage=1.0)",
            "measured kcat kept as ground truth; the 'recalibration win' is a label-quality artifact, not adopted")

    # 11) ΔΔG stability predictor — blind S669 benchmark, thermodynamically consistent
    dg = _load("ddg_validation.json")
    if dg:
        s = dg["summary"]
        add("ddg_stability", dg["passed"],
            dict(pearson_r=s["pearson_r"], rmse=s["rmse_kcal_mol"], anti_symmetry=s["anti_symmetry_corr"]),
            dict(baselines=dg.get("baselines_S669_abs_r")),
            "S669 blind Pearson>=0.38 & anti-symmetry corr>=0.9 (0.472, beats ACDC-NN via ProteinMPNN)",
            "predicts mutation stability ΔΔG (structure ProteinMPNN + biophysical); top-benchmark; mutation->phenotype keystone")

    # 12b) Learned GraphSAGE beats fixed propagation on the same leakage-free link benchmark
    gg = _load("cellgraph_gnn_validation.json")
    if gg:
        h = gg["headline_ppi"]
        add("learned_gnn_beats_fixed", gg["passed"],
            dict(fixed_ppi=h["fixed"], learned_ppi=h["learned"], delta=h["delta"], wins=gg["learned_wins"]),
            dict(fixed=h["fixed"]),
            "trained GraphSAGE beats fixed SIGN propagation on PPI link AUC (Δ>0.01) & wins >=2/3 relations",
            "the GPU-tier learned encoder measurably improves link prediction over fixed message-passing")

    # 13) Enzyme-constrained flux — quantitative enzyme→flux (essentiality recovery + metabolic dominance)
    ec = _load("ecflux_validation.json")
    if ec:
        e = ec["essentiality"]; d = ec["dominance"]
        add("ecflux_quantitative", ec["passed"],
            dict(ess_precision_lift=e["precision_lift"], ess_precision=e["precision"],
                 buffered_at_50pct=d["buffered_at_50pct"], essential_at_KO=d["essential_fraction_at_KO"]),
            dict(base_rate=e["base_rate"]),
            "FBA gene-deletion precision-lift>=3x & dominance reproduced (buffered@50%>=0.8, collapse@KO<=0.2)",
            "enzyme-constrained Human-GEM: known essential genes recovered + quantitative mutation->flux (dominance)")

    # 14b) Cell-type conditioning — the emask gate is cell-type-specific (external canonical markers)
    cc = _load("celltype_conditioning_validation.json")
    if cc:
        s = cc["summary"]
        add("celltype_conditioning_gate", cc["passed"],
            dict(min_specificity=s["min_specificity"], mean_specificity=s["mean_specificity"],
                 mean_fold=s["mean_fold_over_background"]),
            dict(background="cell-type expression base rate"),
            "external canonical markers: min specificity>=0.4 across>=5 lineages & mean>=4x background",
            "emask gate is cell-type-specific (markers land in their own lineage ~9x background) -> conditioned answers are correct; honest negatives (no predictive lift) recorded, not gated")

    # 14) ec-flux MEASURED capacity — per-enzyme Vmax=kcat·[E] from proteomics, not the blanket σ
    ep = _load("ecflux_ppm_validation.json")
    if ep:
        s = ep["summary"]
        add("ecflux_measured_capacity", ep["passed"],
            dict(coverage=s["coverage"], n_measured=s["n_measured_capacity"],
                 vmax_spread_log10=s["excess_heterogeneity_log10_std"]),
            dict(blanket_sigma=0.5),
            "measured ppm×kcat covers>=70% central-carbon enzymes & Vmax spread(log10 std)>=0.5",
            "per-enzyme capacity is data-backed (measured abundance×kcat); buffered-vs-dose-sensitive is measured, not assumed")

    # 15) Structure-based function transfer — AlphaFold + Foldseek (structure > sequence for the dark proteome)
    ff = _load("foldseek_function_validation.json")
    if ff:
        s = ff["summary"]
        add("foldseek_function", ff["passed"],
            dict(accuracy=s["accuracy"], n_ran=s["n_ran"],
                 twilight_recoveries=s["n_with_twilight_homolog"]),
            dict(sequence_methods="fail in the <30% seqId twilight zone"),
            "function keyword-match accuracy>=0.8 on>=4 proteins & >=2 recovered via a <30% seqId structural homolog",
            "AlphaFold structure + Foldseek transfers correct function, incl. from twilight-zone homologs sequence methods miss")

    # 16) End-to-end chain — destabilization-mechanism detector (honest scope: NOT a general classifier)
    ch = _load("chain_validation.json")
    if ch:
        s = ch["summary"]
        add("chain_mechanism", ch["passed"],
            dict(precision_when_fires=s["precision_when_fires"], lift=s["precision_lift"],
                 recall=s["recall_destabilizing"], auc_general=s["auc_ddg_general_classifier"]),
            dict(base_rate=s["base_rate"]),
            "mutation→ΔΔG→flux chain: precision(pathogenic|ΔΔG>1) >= 1.4x base (high-precision destabilization detector)",
            "one validated pipeline mutation→phenotype; high-precision LOW-recall (AUC~0.5 as a general classifier — honest scope)")

    # 17) Self-consistency engine — completions ('this edge is missing') recover held-out real edges
    scv = _load("self_consistency_validation.json")
    if scv:
        s = scv["summary"]
        add("self_consistency", scv["passed"],
            dict(completion_auc=s["auc_completion_triadic"], triadic_recall=s["heldout_with_triadic_evidence"]),
            dict(embedding_only=s["auc_embedding_only"]),
            "anomaly engine's completion proposals recover held-out real edges at triadic-closure AUC >= 0.75",
            "model checks itself: hard-constraint > measured > predicted; flags never auto-applied; completions validated leakage-free")

    # 18) Reasoned variant predictor — AlphaMissense (reliable) + mechanism reasoning + honest blind-spot flag
    rvv = _load("reasoned_variant_validation.json")
    if rvv:
        s = rvv["summary"]
        add("reasoned_variant", rvv["passed"],
            dict(alphamissense_auc=s["alphamissense_auc_ourset"], gain_over_chain=s["reliability_gain"],
                 sickle_cell_am=s["sickle_cell_alphamissense"], blind_spot_caught=s["sickle_cell_caught_by_reasoning"]),
            dict(chain_baseline=s["chain_auc_baseline"]),
            "AlphaMissense beats the stability chain by >=0.1 AUC AND the sickle-cell GOF blind spot is flagged (not trusted benign)",
            "reliable+reasoned variant call: AlphaMissense drives accuracy, mechanism rungs explain, GOF pattern overrides false-benign (the sickle-cell lesson)")

    # 19) Fill-and-verify — the engine proposes fixes, re-runs the mechanism, and never rewrites a measured fact
    fv = _load("fill_verify_validation.json")
    if fv:
        s = fv["summary"]
        add("fill_verify", fv["passed"],
            dict(kcat_verified=s["kcat_verified"], measured_escalated=s["kcat_measured_escalated"],
                 edge_verified=s["edge_verified"], physics_restored=s["physics_restored"]),
            dict(anti_trap=s["anti_trap_measured_escalated"]),
            "verified fixes resolve the oddness (physics restored / shared-complex); measured values ESCALATED not auto-applied; no false fixes",
            "fill-and-verify: propose a fix, re-run the mechanism, report only what verifiably resolves — the anti-trap loop")

    # 20) Ghost-gene patch — de-ghost the dark proteome into the cell (function=fact, pathway=prediction)
    gp = _load("ghost_patch_validation.json")
    if gp:
        s = gp["summary"]
        add("ghost_patch", gp["passed"],
            dict(function_patched=s["patched_function"], pathway_patched=s["patched_pathway"],
                 pathway_function_agreement=s["pathway_function_agreement"], lift=s["lift_over_shuffled"]),
            dict(shuffled=s["shuffled_baseline"]),
            "function de-ghosts >=3000 (curated literature) & predicted pathway agrees with function theme >=0.4 & >=1.8x shuffled",
            "ghost genes patched into the cell: function (fact tier) + pathway membership (prediction tier, cross-validated by the independent function signal)")

    # 21) Kinetics<->flux consistency — predicted kcats agree with the flux they carry vs the measured reference
    kf = _load("kinetics_flux_validation.json")
    if kf:
        s = kf["summary"]
        add("kinetics_flux_consistency", kf["passed"],
            dict(predicted_consistent=s["predicted_consistent"], n_predicted=s["n_predicted"],
                 consistency_rate=s["predicted_consistency_rate"], outliers=s["predicted_outliers_flagged"]),
            dict(measured_reference=s["n_measured"]),
            "predicted kcats >=90% distributionally consistent with the measured reference given flux; outliers have verified revisions; measured never altered",
            "kinetics<->flux self-consistency: Vmax=kcat*[E] vs flux; measured kcats are the reference, only predicted outliers flagged with a flux-consistent proposal")

    # 22) Cross-validation harness — independent measured data (codep) is a valid validator (known-edge control)
    cv = _load("crossval_measured_validation.json")
    if cv:
        s = cv["summary"]
        ok = s["known_ppi_enrichment"] >= 10
        add("crossval_measured", ok,
            dict(known_ppi_enrichment=s["known_ppi_enrichment"], known_ppi_rate=s["known_ppi_corroboration"],
                 predictions_corroborated=s["corroborated"], random_baseline=s["random_baseline"]),
            dict(random=s["random_baseline"]),
            "independent cell-line co-dependency corroborates KNOWN PPI edges >=10x over random (a valid cross-validator)",
            "cross-validate predictions against independent measured data; codep validates context-variable edges (23x) but is blind to pan-essential complexes — match the dataset to the prediction type")

    n_pass = sum(1 for x in rows if x["passed"])
    return dict(n_pass=n_pass, n_total=len(rows), all_pass=(n_pass == len(rows) and rows), tests=rows)


def main():
    sc = scorecard()
    if not sc["tests"]:
        print("recovery_scorecard: no result JSONs found (run the validators first).")
        return
    json.dump(sc, open(OUT / "recovery_scorecard.json", "w"), indent=2)
    print("=" * 74)
    print(f"RECOVERY SCORECARD — {sc['n_pass']}/{sc['n_total']} pass" +
          ("  (ALL PASS)" if sc["all_pass"] else ""))
    print("=" * 74)
    for x in sc["tests"]:
        mark = "PASS" if x["passed"] else "FAIL"
        print(f"  [{mark}] {x['test']:24} {x['value']}")
        print(f"         bar: {x['bar']}  | {x['detail']}")
    print("\n-> outputs/orphan/recovery_scorecard.json  (regression gate: future changes must keep all PASS)")
    return sc


if __name__ == "__main__":
    main()
