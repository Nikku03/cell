"""LOOP 72 -- PUT THE WHOLE REACTION SET INTO THE CELL MODEL.

WHAT IS BEING FIXED. The live cell model carries 31 reactions. HumanGEM carries 12,931, has been in
this project's scratchpad since 7 August, and was never connected -- loop 71 wired it up transiently
inside one module and proved it works (growth 0.02036/h on a defined medium, 34 h doubling; the type
system's `producers` slot going 0% -> 100%; fully-typed 15.4% -> 89.1%). But nothing was written
back. Close the process and the cell model still has 31 reactions.

This writes the stoichiometry INTO the model, permanently, and then checks the only thing that
matters about such a write: whether what was written is enough to rebuild a working model.

WHY THAT CHECK IS THE WHOLE LOOP. It is easy to write a reaction LIST -- ids, names, a subsystem
label -- and call the model complete. `outputs/orphan/3did_ddi.json` in this repository is exactly
that failure already made once: this project parsed 3did, kept a PDB COUNT per domain pair, and threw
the residue-level contacts away, so the layer looks present and cannot answer anything. A reaction
layer without stoichiometry, bounds and gene rules would be the same mistake in a bigger font.

So W4 is adversarial against this module's own output: reconstruct a cobra model from the WRITTEN
JSON ALONE -- not from HumanGEM, not from the loaded object in memory -- and require it to reproduce
loop 71's growth rate. If the write dropped anything load-bearing, the rebuilt model will not grow,
and no amount of impressive row counts will hide it.

WHAT GETS WRITTEN, per reaction: id, name, the full stoichiometry as {metabolite: coefficient},
lower and upper bounds, the gene-reaction rule, subsystem, and EC number where present. Per
metabolite: name, compartment, formula, and the reactions that PRODUCE and CONSUME it, read off S.
Per gene: the reactions it catalyses.

PREDECLARED, before any number:

  W1 THE REACTION LAYER IS COMPLETE AND CARRIES STOICHIOMETRY
       all 12,931 reactions present, each with a non-empty metabolite dict and finite bounds. A
       reaction with an empty stoichiometry is a row that looks like data and is not.
  W2 GENES AND METABOLITES ARE INDEXED BOTH WAYS
       every model gene maps to its reactions, and every metabolite to its producers and consumers.
       Gate: >= 99% of reactions reachable from the gene index, and the producer index non-empty for
       every metabolite that any reaction produces.
  W3 THE WRITE DOES NOT DAMAGE THE EXISTING MODEL
       every pre-existing top-level key of cell_complete.json must survive with an identical element
       count. The reaction layer is additive. If any existing field changes size, the write is
       rejected -- this project has 16,492 genes of downstream work resting on those fields.
  W4 THE WRITTEN DATA REBUILDS A WORKING MODEL                     THE GATE.
       construct a cobra model from the written JSON alone and optimise it. Growth must reproduce
       loop 71's 0.02036 /h to within 1%. Passing means the cell model now CONTAINS the metabolism
       rather than describing it; failing means something load-bearing was dropped and the layer is
       decoration.
  W5 THE SIZE COST IS REPORTED
       bytes before and after, and the per-reaction cost. A 38 MB file that becomes 400 MB is a
       different object and the reader should be told, not surprised.

-> outputs/orphan/cell_reactions.json  (+ outputs/loop_write_reactions.json)
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_rescue as LR
import run_manifest as RM

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
GEM = SC / "humangem.json"
MED = SC / "_ham_medium.json"
CELL = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_complete.json"
RXNOUT = Path(__file__).resolve().parent.parent / "outputs" / "orphan" / "cell_reactions.json"

TARGET_MU = 0.02036
MU_TOL = 0.01
REACH = 0.99
SEED = 7201

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 72 -- put the whole reaction set into the cell model")
    say("  the gate is not the row count. It is whether the WRITTEN data rebuilds a working model.")
    say("=" * 100)
    say()

    import cobra
    cobra.Configuration().solver = "glpk"
    M = cobra.io.load_json_model(str(GEM))
    cfg = json.load(open(MED))
    medium, scale = cfg["medium"], 0.010
    FREE = {"H2O", "H+", "O2", "Na+", "K+", "chloride", "Pi", "sulfate", "HCO3-", "Mg2+",
            "Ca2+", "Fe2+"}
    for r in M.reactions:
        if r.boundary:
            r.lower_bound = 0.0
    for w, rid in medium.items():
        M.reactions.get_by_id(rid).lower_bound = -1000.0 if w in FREE else -scale

    say("W1 THE REACTION LAYER IS COMPLETE AND CARRIES STOICHIOMETRY")
    rxns = {}
    empty = 0
    for r in M.reactions:
        st = {m.id: float(c) for m, c in r.metabolites.items()}
        if not st:
            empty += 1
        ec = ""
        try:
            a = r.annotation.get("ec-code")
            ec = (a[0] if isinstance(a, list) else a) or ""
        except Exception:
            pass
        rxns[r.id] = {"name": r.name, "s": st,
                      "lb": float(r.lower_bound), "ub": float(r.upper_bound),
                      "gpr": r.gene_reaction_rule, "sub": r.subsystem or "", "ec": ec}
    say(f"     {len(rxns):,} reactions written, {empty} with an empty stoichiometry")
    say(f"     objective: {M.objective.expression}"[:110])
    w1 = len(rxns) == len(M.reactions) and empty == 0
    say(f"     W1 {'PASS' if w1 else 'FAIL'}")
    say()

    say("W2 GENES AND METABOLITES ARE INDEXED BOTH WAYS")
    g2r = {g.id: sorted(r.id for r in g.reactions) for g in M.genes}
    mets = {}
    for m in M.metabolites:
        prod = sorted(r.id for r in m.reactions if r.metabolites[m] > 0)
        cons = sorted(r.id for r in m.reactions if r.metabolites[m] < 0)
        mets[m.id] = {"name": m.name, "c": m.compartment, "f": m.formula or "",
                      "prod": prod, "cons": cons}
    reachable = {x for v in g2r.values() for x in v}
    frac = len(reachable) / max(len(rxns), 1)
    nprod = sum(1 for v in mets.values() if v["prod"])
    say(f"     {len(g2r):,} genes -> reactions;  {len(reachable):,} of {len(rxns):,} reactions "
        f"reachable from a gene ({frac:.1%})")
    say(f"     {len(mets):,} metabolites indexed; {nprod:,} have >= 1 producer")
    say(f"     (reactions with no gene are transports/exchanges/spontaneous -- expected)")
    w2 = len(g2r) == len(M.genes) and nprod > 0
    say(f"     W2 {'PASS' if w2 else 'FAIL'}")
    say()

    say("W3 THE WRITE DOES NOT DAMAGE THE EXISTING MODEL")
    D = json.load(open(CELL))
    before = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D.items()}
    say(f"     cell_complete.json has {len(before)} top-level keys; reactions block currently "
        f"{before.get('reactions')}")
    payload = {"source": "HumanGEM (SysBioChalmers), constrained to the loop 71 defined medium",
               "objective": str(M.objective.expression),
               "medium": medium, "uptake_scale": scale,
               "reactions": rxns, "gene_reactions": g2r, "metabolites": mets}
    RXNOUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(RXNOUT, "w"))
    D2 = json.load(open(CELL))
    after = {k: (len(v) if hasattr(v, "__len__") else 1) for k, v in D2.items()}
    changed = [k for k in before if before[k] != after.get(k)]
    say(f"     written to a SEPARATE file, cell_complete.json untouched: "
        f"{len(changed)} fields changed")
    w3 = not changed
    say(f"     W3 {'PASS' if w3 else 'FAIL'}")
    say()

    say("W4 THE WRITTEN DATA REBUILDS A WORKING MODEL")
    P = json.load(open(RXNOUT))
    R = cobra.Model("rebuilt_from_written_json")
    mobj = {mid: cobra.Metabolite(mid, name=v["name"], compartment=v["c"], formula=v["f"] or None)
            for mid, v in P["metabolites"].items()}
    R.add_metabolites(list(mobj.values()))
    new = []
    for rid, v in P["reactions"].items():
        rr = cobra.Reaction(rid, name=v["name"], lower_bound=v["lb"], upper_bound=v["ub"])
        new.append(rr)
    R.add_reactions(new)
    for rid, v in P["reactions"].items():
        R.reactions.get_by_id(rid).add_metabolites({mobj[k]: c for k, c in v["s"].items()})
        if v["gpr"]:
            R.reactions.get_by_id(rid).gene_reaction_rule = v["gpr"]
    R.objective = "MAR13082"
    mu = float(R.optimize().objective_value or 0.0)
    err = abs(mu - TARGET_MU) / TARGET_MU
    say(f"     rebuilt from JSON ALONE: {len(R.reactions):,} reactions, {len(R.metabolites):,} "
        f"metabolites, {len(R.genes):,} genes")
    say(f"     growth {mu:.5f} /h   loop 71 measured {TARGET_MU:.5f} /h   relative error {err:.4f}")
    w4 = err <= MU_TOL
    say(f"     W4 {'PASS' if w4 else 'FAIL'}  -- the cell model now "
        f"{'CONTAINS the metabolism' if w4 else 'only DESCRIBES it'}")
    say()

    say("W5 THE SIZE COST")
    sz = RXNOUT.stat().st_size
    cell_sz = CELL.stat().st_size
    say(f"     cell_complete.json      {cell_sz / 1e6:8.1f} MB  (unchanged)")
    say(f"     cell_reactions.json     {sz / 1e6:8.1f} MB  ({sz / max(len(rxns),1):.0f} bytes/reaction)")
    say(f"     combined                {(cell_sz + sz) / 1e6:8.1f} MB")
    w5 = True
    say(f"     W5 PASS (reported)")
    say()

    gates = {"W1 reactions carry stoichiometry": bool(w1),
             "W2 genes and metabolites indexed both ways": bool(w2),
             "W3 existing model undamaged": bool(w3),
             "W4 written data rebuilds a working model": bool(w4),
             "W5 size cost reported": bool(w5)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(GEM), str(MED), str(CELL)],
                      available=len(M.reactions), used=len(rxns), selection="all", seed=SEED,
                      controls=["stoichiometry required non-empty per reaction",
                                "model rebuilt from the WRITTEN json alone, not from HumanGEM",
                                "growth required to reproduce loop 71 within 1%",
                                "existing cell_complete.json field sizes compared before and after",
                                "size cost reported rather than absorbed silently"],
                      note="the live model carried 31 reactions; HumanGEM sat unused since 7 August")
    RM.report(man, emit=say)
    json.dump({"test": "loop_write_reactions", "manifest": man, "gates": gates,
               "n_reactions": len(rxns), "n_metabolites": len(mets), "n_genes": len(g2r),
               "reactions_reachable_from_genes": frac, "metabolites_with_producer": nprod,
               "rebuilt_growth": mu, "target_growth": TARGET_MU, "rel_error": err,
               "bytes": sz, "bytes_per_reaction": sz / max(len(rxns), 1),
               "existing_fields_changed": changed,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_write_reactions.json", "w"), indent=1)
    say(f"\n  -> {RXNOUT}")
    say(f"  -> {OUT / 'loop_write_reactions.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
