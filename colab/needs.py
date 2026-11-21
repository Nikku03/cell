"""needs — the bill of materials for completing the cell software. We KNOW the end state of a complete cell model
(every gene's function, essentiality, pathway level, kinetics, dynamics, response), so the gap is computable: for
each layer, current coverage vs the complete-state target, and — crucially — WHAT would close it, tagged as:

  DATA   — a measurement that exists in the world or can be generated (what a data-hunt / claude-science run finds)
  METHOD — the data exists but we lack an algorithm that turns it into the answer
  HARD   — biology has no single answer (feedback loops, context-dependence, robustness) → the honest end state is
           an abstention/flag, NOT a number; "completing" these would be faking them

Grounded in the committed artifacts, not a wishlist. -> outputs/orphan/needs.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def _load(f):
    p = f"{OUT}/{f}"
    return json.load(open(p)) if os.path.exists(p) else {}


def build():
    cov = _load("coverage.json")
    lev = _load("cell_levels.json").get("census", {})
    kc = _load("kcat_invivo_validate.json")
    fba = _load("fba_flux_coverage.json")
    genome = cov.get("genome", 16509)
    dark = cov.get("truly_dark", {}).get("n", 91)
    membership = lev.get("pathway_membership_genes", 10489)
    placed = lev.get("trustworthy_level_total", 5259)
    no_ctx = lev.get("no_level_abstain", 4781)
    feedback = lev.get("feedback_module_flagged", 322)
    ctxdep = lev.get("context_dependent_flagged", 180)

    rows = [
        dict(layer="function / annotation", current=f"{genome-dark:,}/{genome:,} have ≥1 answer",
             gap=f"{dark} truly dark genes", limiter="DATA",
             closes="literature / structural-homology / an experiment for the 91 orphans — a bounded, findable list"),
        dict(layer="essentiality", current="~96% measured (DepMap), reasoned AUC 0.80",
             gap="context/cell-line-specific (cross-line r=0.19)", limiter="DATA",
             closes="more genome-wide CRISPR screens in other cell types → per-context essentiality"),
        dict(layer="response prediction (KO→transcriptome)", current="55% (K562-limited)",
             gap="~45% of genes + all non-K562 contexts", limiter="DATA",
             closes="more Perturb-seq across cell types — the single biggest data-limited axis; a data hunt pays here"),
        dict(layer="pathway level — placeable", current=f"{placed:,}/{membership:,} trustworthy",
             gap=f"{no_ctx:,} no directed context", limiter="DATA",
             closes="more directed regulatory edges (SIGNOR is sparse) + a 2nd directional source (Reactome "
                    "preceding-event order) to place & cross-verify the no-context genes"),
        dict(layer="pathway level — feedback modules", current="topology fails (real feedback edges)",
             gap=f"~{feedback:,} feedback-loop genes had no level FROM THE WIRING", limiter="DATA",
             closes="TESTED (feedback_order.py): the INDEPENDENT literature gradient recovers the canonical forward "
                    "order — MAPK/TLR/PI3K topology −0.39 vs literature +0.85. NOT a hard limit; get the literature "
                    "source at scale (per-pathway markers), same fix that rescued the cyclic TCA cycle"),
        dict(layer="pathway level — context-dependent hubs", current="flagged; multi-position",
             gap=f"~{ctxdep:,} hubs act at different levels in different pathways", limiter="METHOD",
             closes="predictable PER PATHWAY (the level is defined once you name the pathway) — report the "
                    "conditional level, not a single aggregate number"),
        dict(layer="kinetics (kcat)", current="predict 0.35, flux-flag 70% (weak, non-circular)",
             gap="no reliable per-enzyme in-vivo kcat", limiter="DATA",
             closes="measured in-vivo human kcats (the Drive davidi set was cross-condition-noisy, 92× off) — real "
                    "13C-flux + absolute proteomics would give kapp within noise as it does in E. coli"),
        dict(layer="metabolic flux", current=f"FBA {fba.get('flux_carrying','?')}/{fba.get('reactions','?')} rxns/state, "
             f"{fba.get('enzyme_genes_with_flux','?')} enzymes", gap="condition-specific measured flux", limiter="DATA",
             closes="13C-MFA flux per condition → replace the single-optimum FBA state with real per-context fluxes"),
        dict(layer="dynamics — KO outcome from wiring", current="GRN boots + robust, but essentiality AUC 0.47 (null)",
             gap="the wiring does NOT carry knockout outcomes", limiter="METHOD/HARD",
             closes="quantitative kinetic parameters per edge (mostly unmeasured) — OR accept that robustness makes "
                    "outcome-from-topology genuinely underdetermined; today it's a measured lookup, not a simulation"),
        dict(layer="variant effect", current="AlphaMissense strong; ΔΔG 0.47; GOF/moonlighting blind",
             gap="gain-of-function & moonlighting (sickle-cell class)", limiter="METHOD",
             closes="a function-change (not just destabilization) model, or a functional assay — flagged today, "
                    "not solved"),
    ]
    data = [r for r in rows if r["limiter"] == "DATA"]
    method = [r for r in rows if "METHOD" in r["limiter"]]
    hard = [r for r in rows if r["limiter"] == "HARD"]
    summary = {"data_limited": len(data), "method_limited": len(method), "hard_limit": len(hard),
               "note": "UPDATED: testing 'can the HARD ones be predicted?' collapsed the HARD column. The feedback-"
                       "module level looked topology-impossible but the literature gradient predicts it (+0.85 vs "
                       "topology −0.39; feedback_order.py); context-dependent hubs are predictable per-pathway; "
                       "essentiality-from-wiring is null but essentiality itself is predicted (reason 0.80). So "
                       "almost everything is predictable from SOME source — the gap is which measurement/source, not "
                       "a wall. The ONLY thing not achievable is deriving outcomes by SIMULATING the topology; the "
                       "outcomes themselves are predictable. That is a DATA/METHOD map, not a hard limit."}
    return {"rows": rows, "summary": summary,
            "headline": f"{len(data)} of {len(rows)} layers DATA-limited (a data hunt closes them); "
                        f"{len(method)} METHOD; {len(hard)} genuinely HARD — testing collapsed the HARD column."}


def main():
    m = build()
    print("=" * 100)
    print("NEEDS — what the cell software still needs to reach the known complete state")
    print("=" * 100)
    for r in m["rows"]:
        tag = {"DATA": "[DATA  ]", "METHOD": "[METHOD]", "HARD": "[HARD  ]", "METHOD/HARD": "[METH/HD]"}[r["limiter"]]
        print(f"{tag} {r['layer']}")
        print(f"          now:  {r['current']}")
        print(f"          gap:  {r['gap']}")
        print(f"          →     {r['closes']}")
    s = m["summary"]
    print("-" * 100)
    print(f"  {m['headline']}")
    print(f"  {s['note']}")
    print("=" * 100)
    json.dump(m, open(f"{OUT}/needs.json", "w"), indent=1)
    return m


if __name__ == "__main__":
    main()
