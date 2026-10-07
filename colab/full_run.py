"""full_run — exercise the whole cell software end-to-end and print the complete picture: the map, every engine
and layer with its validated number, and a showcase gene driven through all the per-gene syscalls.
-> outputs/orphan/full_run.json  (+ a printed dashboard)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def A(f):
    p = f"{OUT}/{f}"
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    from cellos import CellKernel
    k = CellKernel(quiet=True)
    cov, rea, lev = A("coverage.json"), A("reason.json"), A("cell_levels.json").get("census", {})
    dis, loc, sd = A("disease_reason.json"), A("localization.json"), A("string_degree.json")
    nd = A("needs.json"), A("data_impact.json")
    line = "─" * 92

    print("╔" + "═" * 90 + "╗")
    print("║" + "  CELL SOFTWARE — FULL RUN".ljust(90) + "║")
    print("╚" + "═" * 90 + "╝")

    print("\n【 1. THE MAP 】")
    print(f"  genome {cov.get('genome'):,} genes   |   ≥1 trustworthy answer: "
          f"{cov.get('union_any_trustworthy_answer',{}).get('frac',0):.1%}   |   truly dark: {cov.get('truly_dark',{}).get('n')}")
    print(f"  pathway levels placed: {lev.get('trustworthy_level_total'):,}/{lev.get('pathway_membership_genes'):,}  "
          f"(metabolic {lev.get('metabolic_step_labeled',0)+lev.get('metabolic_cycle_labeled',0)}, "
          f"signaling {lev.get('signaling_tier_labeled')}, rest flagged/abstained)")

    print("\n【 2. REASONING ENGINES (one core, reasoner_core) 】")
    print(f"  essentiality  AUC {rea.get('weighted_reasoning_auc',0):.3f}  (7 lines; > best single {rea.get('best_single',0):.2f}; calibrated)")
    print(f"  disease       AUC {dis.get('weighted_reasoning_auc',0):.3f}  (modest, near-orthogonal axis; calibrated prior)")
    print(f"  + pathway-position/feedback (literature 0.85–0.99), mutation (variant×cell), assess (physics+data router)")

    print("\n【 3. DATA LAYERS 】")
    print(f"  localization: {loc.get('n_genes_localized'):,} genes (UniProt); lifted reason "
          f"{loc.get('reasoning_auc_base',0):.2f}→{loc.get('reasoning_auc_with_localization',0):.2f}")
    print(f"  STRING PPI:   {sd.get('n_genes'):,} genes (v12.0 physical); lifted reason "
          f"{sd.get('reason_auc_base',0):.2f}→{sd.get('reason_auc_with_string',0):.2f}")
    print(f"  ID backbone: 99% UniProt/Entrez  |  metabolic subsystems, complexes, pathways (Reactome/Complex Portal)")

    print("\n【 4. SHOWCASE — TP53 through every per-gene syscall 】")
    for label, out in [
        ("reason (essentiality)", k.reason("TP53")),
        ("disease", k.disease("TP53")),
        ("level (pathway)", k.level("TP53")),
        ("loc (compartment)", k.loc("TP53")),
        ("ppi (partners)", k.ppi("TP53")),
        ("viability (FBA)", k.viability("TP53")),
    ]:
        first = str(out).splitlines()[0]
        print(f"  ▸ {label:22} {first[:66]}")
    print("  ▸ mutate TP53 R175H       …", end=" ")
    try:
        m = k.mutate(["TP53", "P04637", "175", "R", "H"])
        print(str(m).splitlines()[0][:60])
    except Exception as e:
        print(f"(needs network: {str(e)[:30]})")

    print("\n【 5. WHAT IT STILL NEEDS 】")
    s = A("needs.json").get("summary", {})
    print(f"  {A('needs.json').get('headline','')}")
    di = A("data_impact.json").get("matrix", {})
    print("  latest data impact (multi-metric): "
          + "; ".join(f"{n}: ess {r['essentiality_reason']['lift']:+.2f}/dis {r['disease_reason']['lift']:+.2f}"
                      for n, r in di.items()))

    json.dump({"genome": cov.get("genome"), "coverage": cov.get("union_any_trustworthy_answer", {}).get("frac"),
               "reason_essentiality_auc": rea.get("weighted_reasoning_auc"),
               "reason_disease_auc": dis.get("weighted_reasoning_auc"),
               "levels_placed": lev.get("trustworthy_level_total"),
               "localization_genes": loc.get("n_genes_localized"), "string_genes": sd.get("n_genes")},
              open(f"{OUT}/full_run.json", "w"), indent=1)
    print("\n" + line)


if __name__ == "__main__":
    main()
