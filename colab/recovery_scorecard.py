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
        ran = [d for d in dt["diseases"] if d.get("ok")]
        all_rec = len(ran) >= 2 and all(d["recovered_known_target"] for d in ran)
        beats = all(d["precision_strong"] >= d["random_label_baseline"] for d in ran) if ran else False
        add("disease_target_recovery", all_rec and beats,
            dict(recovered=dt["summary"]["n_recovered"], ran=dt["summary"]["n_ran"],
                 rate=dt["summary"]["recovery_rate"]),
            dict(random_label=[d["random_label_baseline"] for d in ran]),
            "recover a known target as a top rescuer in every OOD disease & beat random-label baseline",
            "3-layer pipeline (causal->perturb-to-wildtype->druggability) selects the real drug target blind")

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
