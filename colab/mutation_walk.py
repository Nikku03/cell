"""mutation_walk — the step-by-step chain: don't ask 'GOF or LOF?' abstractly; take the mutant product and WALK
each wild-type function/interaction, checking whether the mutant still does it. Where it breaks is the pinned
consequence.

  1 mutation      : provided in the sequence (genome diff)
  2 gene product  : AlphaFold WT structure (the scaffold — folds WT ≈ mutant, so it's the board, not the effect)
  3 function      : Foldseek / ESMFold WT function (the WT baseline)
  4 pathway+stage : the cell model
  5 cut -> effect : perturbation propagation
  6 WALK          : for each WT interaction, can the mutant still make it?
        breaks a CATALYTIC/core contact      -> LOF
        breaks an AUTOINHIBITORY contact      -> GOF (the brake is gone)
        breaks NOTHING                        -> tolerated passenger

What we can resolve today, honestly:
  - does the mutation MATTER (ESM functional score + ΔΔG)                     -> yes
  - is it destabilising (global loss of product)                              -> yes (ΔΔG)
  - is it at the SURFACE (interface-prone) or BURIED (core)                    -> yes (WT structure burial)
  - WHICH SPECIFIC partner interaction it breaks                              -> NOT yet: we hold the PPI edges
        (who binds whom) but NOT the per-interface residues. Localising the break to one partner needs the
        COMPLEX structure — AlphaFold-Multimer per partner (or a PDB/docked complex). That is the one build the
        chain is missing, and it is exactly what separates a GOF (dimer-interface break, e.g. BRAF V600E) from a
        LOF (core break) — which a monomer scaffold cannot see.

-> honest per-mutation verdict + the list of WT interactions to walk + what each verdict needs to become certain.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import molecular_engine as me
import structural_context as sc
from complete_cell import CompleteCell

_C = {"cell": None}


def _cell():
    if _C["cell"] is None:
        _C["cell"] = CompleteCell()
    return _C["cell"]


def walk(gene, pos, wt, mut, acc, recurrent=False):
    C = _cell()
    ve = me.variant_effect(gene, pos, wt, mut, acc=acc, recurrent=recurrent)
    burial, hint = sc.gof_hint(acc, pos)
    func, ddg = ve["functional_score"], ve["ddg"]
    surface = burial is not None and burial < 135
    partners = [C.name[j] for j in list(C.ppi_adj.get(C.idx.get(gene, -1), ()))][:10] if gene in C.idx else []
    n_partners = len(C.ppi_adj.get(C.idx.get(gene, -1), ())) if gene in C.idx else 0

    # honest verdict: use ΔΔG for destabilisation (NOT burial alone — a buried, stability-neutral mutation is
    # NOT destabilising; it is an interface/allosteric change we cannot localise without the complex).
    if func is None or func < 0.35:
        verdict = "tolerated — low functional impact; WT interactions likely PRESERVED"
        needs = None
    elif ddg is not None and ddg > 1.5:
        verdict = "LOF by DESTABILISATION — product misfolds; loses most/all interactions at once"
        needs = None
    elif surface:
        verdict = "functional + SURFACE — at risk of BREAKING a specific interaction (LOF core-contact OR GOF autoinhibitory-contact)"
        needs = "AlphaFold-Multimer per partner to identify WHICH interface (and thus LOF vs GOF)"
    else:
        verdict = ("functional but STABILITY-NEUTRAL and buried in the MONOMER — effect is an interface/allosteric "
                   "change the monomer scaffold cannot see (this is the BRAF-V600E / KRAS-G12D pattern: buried in "
                   "the monomer, active at the DIMER/GAP interface)")
        needs = "AlphaFold-Multimer of the relevant complex — the effect is at an interface not present in the monomer"
    return {"gene": gene, "variant": f"{wt}{pos}{mut}", "matters": ve["call"], "functional_score": func,
            "ddg": ddg, "burial": burial, "surface": surface, "n_wt_interactions": n_partners,
            "interactions_to_walk": partners, "verdict": verdict, "needs_to_be_certain": needs}


def run(muts=None):
    muts = muts or [("BRAF", 600, "V", "E", "P15056", True), ("TP53", 175, "R", "H", "P04637", False),
                    ("KRAS", 12, "G", "D", "P01116", True), ("VHL", 167, "R", "W", "P40337", False)]
    import json
    out = []
    for g, p, w, m, a, rec in muts:
        r = walk(g, p, w, m, a, recurrent=rec)
        out.append(r)
        print("=" * 78)
        print(f"MUTATION WALK — {g} {w}{p}{m}   (matters: {r['matters']})")
        print("=" * 78)
        print(f"  functional {r['functional_score']} | ΔΔG {r['ddg']} | burial {r['burial']} "
              f"({'surface' if r['surface'] else 'buried'})")
        print(f"  WT interactions to walk: {r['n_wt_interactions']}  e.g. {r['interactions_to_walk'][:6]}")
        print(f"  -> {r['verdict']}")
        if r["needs_to_be_certain"]:
            print(f"     NEEDS: {r['needs_to_be_certain']}")
        print()
    json.dump(out, open("outputs/orphan/mutation_walk.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    run()
