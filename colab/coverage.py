"""coverage — the honest whole-cell coverage map. "55%" was never the whole story: it is the ceiling for one
question (predict a knockout's full transcriptional RESPONSE), and that one is data-generation-limited. But the
software answers MANY trustworthy questions, and this counts them all, per gene, without conflating them:

  ESSENTIALITY  measured DepMap dependency (dep_frac) — a real number for most genes            [C measured]
  RESPONSE      the knockout's transcriptional signature: measured in a Perturb-seq screen,     [C measured / modeled]
                or complex-predictable (r0.23) — THIS is the 55% axis
  VIABILITY(phys) FBA predicts grow/die for metabolic genes even when silent in the assay        [FBA modeled]
  REPROGRAM     forward master-regulator logic: a TF whose lineage program we can drive (0.99)   [modeled]
  DOCUMENTED    grounded literature / function annotation (litmine + curated)                    [LIT]
  DARK          none of the above — the gene we can say nothing trustworthy about

Reports each axis separately AND the union (fraction with >=1 trustworthy answer), so the honest headline is not a
single inflated number but the real shape: response prediction stuck at its data ceiling, total reach far wider,
truly-dark fraction small. -> outputs/orphan/coverage.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    from complete_cell import CompleteCell
    import cellos
    C = CompleteCell()
    N = len(C.genes)

    # ---- per-axis membership sets (by gene name) ----
    ess = set()          # measured essentiality (DepMap dep_frac present)
    reprog = set()       # a TF with outgoing regulatory edges -> induce can drive a program
    documented = set()   # literature or real function annotation
    for i, g in enumerate(C.genes):
        nm = C.name[i]
        if num(g.get("dep_frac")) is not None:
            ess.add(nm)
        if g.get("tf") and (C.causal_out.get(i)):
            reprog.add(nm)
        pubs = num(g.get("pubs")) or 0
        annotated = (g.get("path") or g.get("proc") not in (None, "", "other")) and not g.get("dark")
        if pubs >= 5 or annotated:
            documented.add(nm)

    # RESPONSE axis: measured in a mounted Perturb-seq screen; plus the validated complex-predictable count
    k = cellos.CellKernel(quiet=True); dbg = k._load_debugger()
    resp_measured = set(dbg["pgenes"]) if dbg else set()
    # the committed genome-wide accounting (gwps corpus) — use its counts so the 55% axis matches what we validated
    cf = json.load(open(f"{OUT}/cellformer.json")) if os.path.exists(f"{OUT}/cellformer.json") else {}
    comp = cf.get("completeness", {})
    resp_measured_n = comp.get("measured_debugger", len(resp_measured))
    complex_n = comp.get("predictable_complex_r0.23", 0)
    response_n = resp_measured_n + complex_n                      # the 55% numerator (genome-wide)

    # VIABILITY (physics): FBA metabolic genes
    fba = json.load(open(f"{OUT}/fba_essentiality.json"))["predicted_growth_on_knockout"]
    viability_phys = set(fba)

    # DOCUMENTED dark genes rescued by litmine
    if os.path.exists(f"{OUT}/litmine.json"):
        lm = json.load(open(f"{OUT}/litmine.json"))
        documented |= {g for g, v in lm.items() if v.get("n_papers", 0) > 0}

    # ---- union: at least one trustworthy answer (counted WITHIN the genome only; answer-sets may carry
    #      out-of-genome names from litmine/FBA, so the honest union is N - truly_dark) ----
    any_answer = ess | resp_measured | reprog | documented | viability_phys
    dark = [C.name[i] for i in range(N) if C.name[i] not in any_answer]
    union_n = N - len(dark)                                       # within-genome genes with >=1 answer

    def pct(n):
        return f"{n/N:.1%}"

    R = {
        "genome": N,
        "axes": {
            "essentiality_measured": {"n": len(ess), "frac": len(ess) / N, "grade": "C measured"},
            "response_predictable_55pct": {"n": response_n, "frac": response_n / N, "grade": "C measured/modeled",
                                           "note": "the data-generation-limited ceiling (measured Perturb-seq + complex r0.23)"},
            "viability_physics_FBA": {"n": len(viability_phys), "frac": len(viability_phys) / N, "grade": "FBA modeled"},
            "reprogrammable_TF": {"n": len(reprog), "frac": len(reprog) / N, "grade": "modeled (forward, 0.99)"},
            "documented": {"n": len(documented), "frac": len(documented) / N, "grade": "LIT"},
        },
        "union_any_trustworthy_answer": {"n": union_n, "frac": union_n / N},
        "truly_dark": {"n": len(dark), "frac": len(dark) / N, "examples": dark[:15]},
    }
    json.dump(R, open(f"{OUT}/coverage.json", "w"), indent=1)

    print(f"WHOLE-CELL COVERAGE MAP  —  {N:,} genes, every trustworthy answer counted honestly\n")
    print(f"  {'axis':<34}{'genes':>8}{'of genome':>11}   grade")
    print(f"  {'-'*70}")
    print(f"  {'ESSENTIALITY (measured DepMap)':<34}{len(ess):>8}{pct(len(ess)):>11}   [C]")
    print(f"  {'RESPONSE prediction  ← the 55%':<34}{response_n:>8}{pct(response_n):>11}   [C/modeled]  data-limited ceiling")
    print(f"  {'VIABILITY (FBA physics)':<34}{len(viability_phys):>8}{pct(len(viability_phys)):>11}   [FBA]")
    print(f"  {'REPROGRAMMABLE (TF, forward)':<34}{len(reprog):>8}{pct(len(reprog)):>11}   [modeled 0.99]")
    print(f"  {'DOCUMENTED (grounded literature)':<34}{len(documented):>8}{pct(len(documented)):>11}   [LIT]")
    print(f"  {'-'*70}")
    print(f"  {'≥1 TRUSTWORTHY ANSWER (union)':<34}{union_n:>8}{pct(union_n):>11}   ← the real reach")
    print(f"  {'TRULY DARK (nothing at all)':<34}{len(dark):>8}{pct(len(dark)):>11}")
    print(f"\n  honest headline: the DEEPEST answer (predict the full response) is stuck at {pct(response_n)} —")
    print(f"  data-limited, needs new wet-lab screens. But measured essentiality reaches {pct(len(ess))}, and the")
    print(f"  software gives at least one trustworthy answer for {pct(union_n)} of the genome (only {pct(len(dark))}")
    print(f"  truly dark). '55%' was the deepest axis, not the ceiling of the whole system.")
    return R


if __name__ == "__main__":
    main()
