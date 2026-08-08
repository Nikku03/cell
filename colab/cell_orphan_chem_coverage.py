"""CAN WE EVEN GET THE SUBSTRATES? -- the gate before any docking code is written.

WHY THIS COMES FIRST.  The one version of structure-based re-ranking not yet ruled out is pair-specific:
dock the actual substrate into the candidate's pocket, rather than asking whether the protein looks
enzyme-shaped. Three re-rankers have already failed as PER-PROTEIN priors -- ESM (0.122 against 0.675), a
rank-prior fusion (0.643), and sphere pockets with mode hinges (0.483 against 0.763) -- and the reason each
failed is the same: 68.5% of ranking errors are enzyme-against-enzyme, where a verdict on the protein alone
says nothing.

Making it pair-specific requires a 3D structure for the substrate, and the model stores metabolite NAMES.
So before writing a docking engine there is a prior question with a cheap answer:

    for how many reactions can every non-currency participant be resolved to a chemical structure at all?

If most reactions have an unresolvable participant, the method cannot reach them however well it works on
the rest, and that is decided here rather than after the engine exists.

WHY EXACT NAME MATCHING IS THE RIGHT TEST AND NOT A LAZY ONE.  Reactome draws its small-molecule names from
ChEBI, so ChEBI's own name table is the matching source rather than a third-party guess. Names are matched
case-folded against ChEBI NAMES, which includes synonyms and IUPAC forms, not only primary names -- a
metabolite called "pristanic acid" in one and "pristanate" in the other still resolves. What is NOT done is
fuzzy or substring matching: "acetyl-CoA" must not silently match "acetyl-CoA(4-)" of a different charge
state, because a wrong structure is worse than a missing one for docking.

WHAT IS REPORTED, and the distinction matters for what gets built
    per METABOLITE     how much of the vocabulary resolves
    per REACTION       how many reactions have EVERY non-currency participant resolved -- this is the one
                       that bounds the method, since a docking screen needs the substrate, not most of it
    per REACTION, main substrate only   a weaker requirement: only the largest non-currency input. If the
                       full requirement fails but this passes, a single-substrate screen is still viable
                       and the design changes rather than the project stopping
    stratified by whether the reaction is an orphan or a held-out test case, because the two populations
    differ in every other way measured so far and there is no reason to assume they match here

CURRENCY METABOLITES ARE EXCLUDED, and the model already flags them. ATP, H2O, NAD+ and the rest appear in
thousands of reactions and carry no specificity; requiring them to resolve would inflate the failure rate
with participants no docking screen would use as the discriminating ligand anyway.

PREDECLARED, before the numbers:
    most reactions fully resolvable
        -> build the pair-specific docking screen; the substrate side is not the blocker.
    few fully resolvable but most have a resolvable MAIN substrate
        -> build it as a single-substrate screen and say so; the second-substrate step was already ruled
           out on other grounds.
    most reactions have no resolvable substrate at all
        -> the last untested version of structure-based re-ranking cannot reach the population, and the
           honest conclusion is that the shortlist goes to a curator unmodified.

-> outputs/orphan/cell_orphan_chem_coverage.json
"""
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_fill import build_pool  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")


def load_chebi():
    """name -> ChEBI id, and ChEBI id -> SMILES. Names include synonyms; structures are the SMILES rows."""
    name2id = {}
    with gzip.open(SP / "chebi_names.tsv.gz", "rt", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci = hdr.index("COMPOUND_ID") if "COMPOUND_ID" in hdr else 1
        ni = hdr.index("NAME") if "NAME" in hdr else 4
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(ci, ni):
                continue
            name2id.setdefault(f[ni].strip().lower(), f[ci])
    # the primary ChEBI name lives in compounds.tsv, not names.tsv -- without it the common metabolites
    # miss, because their primary label is not repeated in the synonym table
    with gzip.open(SP / "chebi_compounds.tsv.gz", "rt", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(hdr)}
        ci, ni = idx.get("ID", 0), idx.get("NAME", 5)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(ci, ni):
                continue
            nm = f[ni].strip().lower()
            if nm and nm != "null":
                name2id.setdefault(nm, f[ci])
    id2smi = {}
    with gzip.open(SP / "chebi_structures.tsv.gz", "rt", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(hdr)}
        ci = idx.get("COMPOUND_ID", 1)
        si = idx.get("STRUCTURE", 2)
        ti = idx.get("TYPE", 3)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(ci, si, ti):
                continue
            if f[ti].strip().upper() == "SMILES":
                id2smi.setdefault(f[ci], f[si].strip())
    return name2id, id2smi


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("CAN WE EVEN GET THE SUBSTRATES? -- coverage gate before any docking is written")
    report("=" * 100)
    report("  Three re-rankers have failed as per-protein priors because 68.5% of ranking errors are")
    report("  enzyme-against-enzyme. The one untested version is PAIR-specific: dock the real substrate")
    report("  into the candidate's pocket. That needs a 3D structure for the substrate, and the model")
    report("  stores names. If most reactions have an unresolvable participant, the method cannot reach")
    report("  them however well it works elsewhere, and that is decided here rather than after the build.")

    name2id, id2smi = load_chebi()
    report(f"\n  ChEBI: {len(name2id)} names (incl. synonyms) -> id, {len(id2smi)} ids with SMILES")

    B, orphans, _ = build_pool()
    steps = B["steps"]
    meta = json.load(open(OUT / "holdout_meta.json"))
    holdout = [m["step"] for m in meta.values()]

    def resolve(nm):
        cid = name2id.get((nm or "").strip().lower())
        return id2smi.get(cid) if cid else None

    def parts(i):
        s = steps[i]
        return [q for q in (s.get("in") or []) + (s.get("out") or [])
                if q.get("type") == "METABOLITE" and not q.get("currency")]

    # ---- vocabulary-level ---------------------------------------------------------------------------
    vocab = Counter()
    for i in orphans:
        for q in parts(i):
            vocab[q.get("name")] += 1
    hit = {m for m in vocab if resolve(m)}
    cov_tok = sum(vocab[m] for m in hit) / max(sum(vocab.values()), 1)
    report(f"\n  METABOLITE VOCABULARY over the {len(orphans)} orphan reactions")
    report(f"    distinct non-currency metabolites {len(vocab):>6}")
    report(f"    resolved to a SMILES              {len(hit):>6}  ({len(hit)/max(len(vocab),1):.1%} of types)")
    report(f"    weighted by how often they occur                 {cov_tok:.1%} of mentions")
    miss = [m for m, _ in vocab.most_common() if m not in hit][:10]
    report(f"    most common unresolved: {', '.join(str(m)[:26] for m in miss)}")

    # ---- reaction-level -----------------------------------------------------------------------------
    report(f"\n  REACTION COVERAGE -- the number that bounds the method")
    report(f"    {'population':<22}{'n':>7}{'no metab':>10}{'ALL resolved':>14}{'MAIN resolved':>15}")
    res = {}
    for lab, idxs in (("orphans (the fill)", orphans), ("held-out (the test)", holdout)):
        nall = nmain = nnone = 0
        for i in idxs:
            p = parts(i)
            if not p:
                nnone += 1
                continue
            sm = [resolve(q.get("name")) for q in p]
            if all(sm):
                nall += 1
            # main substrate = the longest-named non-currency input, a crude but consistent proxy for the
            # largest species, chosen because it does not need a structure to evaluate
            ins = [q for q in (steps[i].get("in") or [])
                   if q.get("type") == "METABOLITE" and not q.get("currency")]
            if ins and resolve(max(ins, key=lambda q: len(q.get("name") or "")).get("name")):
                nmain += 1
        n = len(idxs)
        report(f"    {lab:<22}{n:>7}{nnone:>10}{nall:>7} ({nall/max(n,1):>5.1%}){nmain:>8} ({nmain/max(n,1):>5.1%})")
        res[lab] = {"n": n, "no_metabolite": nnone, "all_resolved": nall, "main_resolved": nmain}

    report("\n  READING")
    o = res["orphans (the fill)"]
    fa, fm, fn = (o["all_resolved"] / o["n"], o["main_resolved"] / o["n"], o["no_metabolite"] / o["n"])
    report(f"    {fn:.1%} of orphan reactions have NO non-currency metabolite at all -- protein-only steps,")
    report("    transport of complexes, modifications. A small-molecule docking screen has nothing to dock")
    report("    for these and they are outside the method by construction, not by failure.")
    if fa > 0.5:
        report(f"    {fa:.1%} have EVERY substrate resolvable, so the pair-specific screen can be built as")
        report("    designed and the substrate side is not the blocker.")
    elif fm > 0.5:
        report(f"    Only {fa:.1%} resolve fully, but {fm:.1%} have a resolvable MAIN substrate. So the")
        report("    screen is viable as a SINGLE-substrate design and should be described that way -- which")
        report("    costs nothing, since the second-substrate step was already ruled out for needing")
        report("    quantum mechanics to make and break bonds.")
    else:
        report(f"    Only {fm:.1%} have even a resolvable main substrate. The last untested version of")
        report("    structure-based re-ranking cannot reach most of the population, and the shortlist")
        report("    should go to a curator unmodified.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_orphan_chem_coverage", "n_names": len(name2id), "n_smiles": len(id2smi),
               "vocab": {"distinct": len(vocab), "resolved": len(hit), "token_coverage": cov_tok},
               "reactions": res, "log": log},
              open(OUT / "cell_orphan_chem_coverage.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_chem_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
