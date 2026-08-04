"""REVERSE-ENGINEER THE CELL AS A PARTS LIST: functions -> reactions -> parts -> genes -> products -> transport.

WHY THIS IS A DIFFERENT KIND OF TASK FROM EVERYTHING ELSE TODAY, and why it may actually work.

Every failure in this project has been a PREDICTION failure: seven representations could not say what an
unmeasured knockout does. But the thing that kept failing -- curated annotation -- failed at prediction, not at
enumeration. Curation IS a catalogue. Asking it to catalogue is asking it for the thing it is actually good at.

So this builds no model and predicts nothing. It walks the chain a cell physically implements and reports, at
every link, WHAT FRACTION WE CAN ACTUALLY FILL. That number is the honest answer to "can we reverse-engineer the
cell from what we have", and it is worth knowing before anything is built on top of it.

THE CHAIN, as asked, small to big and back down to the molecules:

    1  FUNCTION      what the cell must do. Taken from Reactome pathways and GO biological process, kept as a
                     hierarchy so "small" (a single catalytic step) and "big" (a whole process) both appear.
    2  REACTION      the concrete steps implementing each function, from the merged Reactome + HumanGEM network:
                     28,528 steps with typed participants, direction and reversibility.
    3  PARTS         every participant, split by what it IS -- protein, metabolite, cofactor/currency, complex,
                     nucleic acid, drug, other. The non-protein half is what a gene-only view misses entirely.
    4  MACHINE       the catalyst: which protein or complex performs it.
    5  GENE          which gene(s) encode that catalyst, including AND/OR logic where a complex needs several.
    6  PRODUCT       what the gene makes -- mRNA to protein, or a non-coding product (tRNA, rRNA, etc.).
    7  LOCATION      the compartment each participant sits in.
    8  TRANSPORT     reactions whose input and output compartments DIFFER. These are the steps a parts list
                     usually forgets, and they are recoverable exactly because compartments are annotated.

REPORTED AS COVERAGE, NOT AS A CLAIM.  For every link the output gives filled / total and names the largest
holes. Where a layer is absent from this disk it is reported as ABSENT rather than approximated -- the lesson
from today's circular depth control is that a fabricated stand-in is worse than a stated gap, because it looks
like completeness.

KNOWN GAPS EXPECTED BEFORE RUNNING, recorded here so the output cannot be read as a surprise:
    splice isoforms   the gene records are gene-level; transcript structures are not on this disk
    kinetics          kcat/Km exist only for the subset the earlier enzyme work covered
    temperature/pH    metabolic models carry compartment pH/charge at best; temperature is not modelled at all
    non-coding genes  tRNA/rRNA are largely absent from both Reactome and a metabolic model
"""
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("THE CELL AS A PARTS LIST -- function -> reaction -> parts -> gene -> product -> transport")
    report("=" * 100)
    report("  This enumerates. It does not predict. Every link reports COVERAGE, and absent layers are named")
    report("  as absent rather than approximated.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    cell = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    genes = {g["name"]: g for g in cell["genes"]}
    report(f"\n  reaction network {len(steps):,} steps | gene records {len(genes):,}")

    # ---------------- 3. PARTS: what is a cell actually made of, by participant type ----------------
    ptypes, part_names = Counter(), defaultdict(set)
    comp_of = Counter()
    for s in steps:
        for side in ("in", "out"):
            for p in s.get(side) or []:
                t = (p.get("type") or "UNKNOWN").upper()
                ptypes[t] += 1
                part_names[t].add(p.get("name") or p.get("id"))
                c = p.get("compartment")
                if c:
                    comp_of[c] += 1
    report("\n  [3] PARTS -- every participant across every reaction, by what it IS")
    tot = sum(ptypes.values())
    for t, n in ptypes.most_common():
        report(f"    {t:<16} {n:>9,} occurrences  {len(part_names[t]):>7,} distinct  ({100*n/tot:5.1f}%)")
    nonprot = sum(n for t, n in ptypes.items() if t not in ("PROTEIN", "COMPLEX"))
    report(f"    -> NON-PROTEIN share of all participants: {100*nonprot/tot:.1f}%  "
           f"(the half a gene-centric view cannot see)")
    if net.get("currency"):
        cur = net["currency"]
        report(f"    cofactors / currency metabolites flagged: {len(cur)}  e.g. {sorted(cur)[:12]}")

    # ---------------- 7. LOCATION ----------------
    report("\n  [7] LOCATION -- compartments carried by participants")
    if comp_of:
        for c, n in comp_of.most_common(12):
            report(f"    {str(c):<28} {n:>9,}")
    else:
        report("    ABSENT: participants in this merged network do not carry a compartment field.")
        report("    Compartments exist per-GENE in cell_complete and per-SPECIES in HumanGEM, but the merged")
        report("    reaction records dropped them, so per-reaction localisation must be rebuilt from HumanGEM.")

    # ---------------- 8. TRANSPORT ----------------
    report("\n  [8] TRANSPORT -- steps whose input and output compartments differ")
    moved = 0
    checked = 0
    for s in steps:
        ci = {p.get("compartment") for p in (s.get("in") or []) if p.get("compartment")}
        co = {p.get("compartment") for p in (s.get("out") or []) if p.get("compartment")}
        if ci and co:
            checked += 1
            if ci != co:
                moved += 1
    if checked:
        report(f"    {moved:,} of {checked:,} reactions cross a compartment boundary "
               f"({100*moved/checked:.1f}%)")
    else:
        report("    NOT COMPUTABLE from the merged network -- no per-participant compartments (see [7]).")
        report("    Recoverable from HumanGEM, where every species id carries a compartment suffix.")

    # ---------------- 4/5. MACHINE and GENE ----------------
    report("\n  [4,5] MACHINE and GENE -- which reactions have a named catalyst, and is it a gene we know")
    with_cat = [s for s in steps if s.get("catalysts")]
    allcat = Counter()
    for s in with_cat:
        for c in s["catalysts"]:
            allcat[c] += 1
    known = {c for c in allcat if c in genes}
    report(f"    reactions with >=1 catalyst : {len(with_cat):>7,} / {len(steps):,} "
           f"({100*len(with_cat)/len(steps):.1f}%)")
    report(f"    ORPHAN reactions, no catalyst: {len(steps)-len(with_cat):>7,} "
           f"({100*(len(steps)-len(with_cat))/len(steps):.1f}%)  <- the biggest hole in the chain")
    report(f"    distinct catalyst symbols    : {len(allcat):>7,}")
    report(f"    of those, in our gene records: {len(known):>7,} ({100*len(known)/max(len(allcat),1):.1f}%)")
    multi = sum(1 for s in with_cat if len(s["catalysts"]) > 1)
    report(f"    reactions needing >1 gene    : {multi:>7,} ({100*multi/max(len(with_cat),1):.1f}%) "
           f"-- complexes, i.e. AND logic")

    # ---------------- 1. FUNCTION ----------------
    report("\n  [1] FUNCTION -- what the cell has to do, and how much of it has reactions attached")
    pw = defaultdict(set)
    for name, g in genes.items():
        for p in (g.get("path") or []):
            pw[p].add(name)
    proc = defaultdict(set)
    for name, g in genes.items():
        for p in (g.get("proc") or []):
            proc[p].add(name)
    cat_genes = set(allcat)
    report(f"    Reactome pathways   {len(pw):>6,}   GO processes {len(proc):>6,}")
    sizes = sorted((len(v), k) for k, v in pw.items())
    report(f"    pathway size: median {sizes[len(sizes)//2][0]}, "
           f"smallest {sizes[0][0]} ({sizes[0][1][:40]}), largest {sizes[-1][0]} ({sizes[-1][1][:40]})")
    covered = sum(1 for k, v in pw.items() if v & cat_genes)
    report(f"    pathways with >=1 gene that catalyses a known reaction: {covered:,}/{len(pw):,} "
           f"({100*covered/max(len(pw),1):.1f}%)")
    coveredp = sum(1 for k, v in proc.items() if v & cat_genes)
    report(f"    GO processes likewise: {coveredp:,}/{len(proc):,} ({100*coveredp/max(len(proc),1):.1f}%)")

    # ---------------- 6. PRODUCT ----------------
    report("\n  [6] PRODUCT -- what each gene makes")
    fields = Counter()
    for g in genes.values():
        for f in ("tf", "comp", "path", "proc", "tss", "chrom"):
            if g.get(f):
                fields[f] += 1
    for f, n in fields.most_common():
        report(f"    genes with {f:<6} {n:>7,} / {len(genes):,} ({100*n/len(genes):.1f}%)")
    report("    ABSENT: transcript isoforms / splice variants. The records are GENE-level, so 'which isoform'")
    report("    cannot be answered from this disk. Needs GENCODE or Ensembl transcript models.")
    report("    ABSENT: gene biotype, so tRNA / rRNA / snoRNA genes cannot be separated from coding ones here.")

    # ---------------- CONDITIONS ----------------
    report("\n  [CONDITIONS] temperature, pH, concentrations")
    report("    ABSENT: no temperature anywhere in these sources. Metabolic models carry compartment pH and")
    report("    charge at best; Reactome carries neither. Kinetics (kcat/Km) exist only for the enzyme subset")
    report("    the earlier ecModel work covered, not genome-wide. Reported as a gap, not estimated.")

    # ---------------- THE SCORECARD ----------------
    report("\n" + "=" * 100)
    report("  COVERAGE SCORECARD -- how much of the chain can actually be filled from this disk")
    report("=" * 100)
    rows = [
        ("1 function", f"{len(pw):,} pathways + {len(proc):,} GO processes", "ENUMERABLE"),
        ("2 reaction", f"{len(steps):,} steps", "ENUMERABLE"),
        ("3 parts", f"{len(part_names['METABOLITE']):,} metabolites, {nonprot*100//tot}% non-protein",
         "ENUMERABLE"),
        ("4 machine", f"{100*len(with_cat)//len(steps)}% of reactions have a catalyst", "PARTIAL"),
        ("5 gene", f"{len(known):,} catalyst genes matched", "ENUMERABLE"),
        ("6 product", "gene-level only", "PARTIAL -- no isoforms, no biotype"),
        ("7 location", "per-gene and per-species, NOT per-reaction here", "PARTIAL"),
        ("8 transport", "needs HumanGEM species suffixes", "RECOVERABLE, not yet built"),
        ("conditions", "no temperature; kinetics for a subset", "ABSENT"),
    ]
    for a, b, c in rows:
        report(f"    {a:<12} {b:<52} {c}")

    json.dump({"test": "cell_partslist", "n_steps": len(steps), "n_genes": len(genes),
               "participant_types": dict(ptypes),
               "distinct_per_type": {k: len(v) for k, v in part_names.items()},
               "nonprotein_pct": 100.0 * nonprot / tot,
               "reactions_with_catalyst": len(with_cat), "orphan_reactions": len(steps) - len(with_cat),
               "distinct_catalysts": len(allcat), "catalysts_known": len(known),
               "multi_gene_reactions": multi, "n_pathways": len(pw), "n_processes": len(proc),
               "pathways_covered": covered, "processes_covered": coveredp,
               "transport_checked": checked, "transport_crossing": moved,
               "scorecard": [{"link": a, "detail": b, "status": c} for a, b, c in rows],
               "log": log}, open(OUT / "cell_partslist.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'cell_partslist.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
