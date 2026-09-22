"""THE THREE BOTTLENECKS, FETCHED AGAINST -- and one of the three fetches did not help.

cell_run's coverage cascade put 87 genes at the end of it: full dynamics, a metabolic reaction, a
signed regulator and a trustworthy kcat, all at once. Three data gaps produce that number, and this
module fetches against each and MEASURES the gain rather than assuming one. Recorded as a module
because a fetch that failed is worth as much as a fetch that worked, and only one of these is
written down anywhere else.

GAP 1 -- DYNAMICS.  The state vector needs mRNA copies, protein copies and both half-lives from one
cell type. Only Schwanhausser 2011 supplies all four, in mouse fibroblasts, for 4,190 genes.

    FETCHED / ALREADY ON DISK
      Mathieson 2018 protein half-lives, HUMAN, 5 primary cell types      8,804 genes
      Itzhak 2016 protein copy numbers, HeLa                              8,469 genes
      AvgKdegs mRNA degradation rates, 13 cell lines INCLUDING HeLa       9,967 genes in HeLa
    MEASURED GAIN
      protein-side state (half-life + copies, human)   5,595  against the current 4,190
      genes gained outright                            2,607
      HeLa mRNA half-life AND HeLa protein copies      6,807  -- SAME CELL LINE, which is what
                                                       loop 92's rule demands
    STILL MISSING: absolute mRNA COPY NUMBERS in HeLa. Half-lives and protein copies are now human
    and same-cell-line; the mRNA count is the one remaining cross-species borrow.

GAP 2 -- SIGNED REGULATORY EDGES.  A rate law needs a sign; 91.2% of the model's 612,133 edges
carry none. Fetched OmniPath, which aggregates CollecTRI, DoRothEA, TRRUST, SIGNOR and others and
returns is_stimulation / is_inhibition per edge.

    FETCHED
      CollecTRI    64,515 edges     45,507 signed with both ends in the model
      DoRothEA     15,266 edges      6,649
      tf_target    69,654 edges     13,791
    MEASURED GAIN -- AND IT IS NEGATIVE
      union, signed, both ends in the model   47,613 edges, 1,336 regulators, 6,799 targets
      against the model's existing            54,128 edges
      ratio                                   0.88x -- FEWER, not more
      state genes reached                     38.1% against the current 42.9%

    THIS FETCH DID NOT HELP, by the measure that motivated it. The repository's existing signed
    network is already larger than CollecTRI, DoRothEA and tf_target combined, once both ends are
    required to be model genes. Two things that number does NOT settle, and neither is assumed here:
    whether CollecTRI's edges are BETTER (it is literature-curated, and loop 120 showed the existing
    signs carry no information -- real signs scored 0.5465 against shuffled signs at 0.5494, so
    "more edges" was never the binding problem), and whether the union would grow if the both-ends-
    in-model filter were relaxed. Testing CollecTRI's signs against loop 120's own protocol is the
    follow-up this points at, and it is a different question from coverage.

GAP 3 -- KINETICS VALIDATION.  Loop 127's filter could only be validated on 70 genes: every gene
with both a UniProt kcat and an EC carrying replicates. Fetched DLKcat's compilation, which the
repository had previously mined only at EC level.

    FETCHED
      DLKcat Kcat_combination_0918      17,010 records, 2,439 human
    THE MOVE THAT UNLOCKS IT: the file carries no UniProt ID but does carry the protein SEQUENCE,
    which maps to a gene exactly. Grouping by sequence rather than by EC gives PER-PROTEIN
    replicates instead of per-class ones.
      human records mapped to a gene by exact sequence   1,256 over 342 genes
      of those, in the model                             289
      validation set                                     70 -> 289, a 4.1x expansion

    AND A CORRECTION TO LOOP 127. That loop measured the experimental noise floor at 2.85x from
    leave-one-out WITHIN AN EC. But an EC class contains many enzymes and many substrates, so that
    figure is enzyme-to-enzyme variation, not experimental reproducibility. Grouping by protein AND
    substrate gives true replicates:

        same protein, same substrate, independent measurements    median 1.15x over 101 values

    The real experimental floor is 1.15x, not 2.85x. Loop 127's conclusion that 4x is a defensible
    operating point survives, but for the opposite reason to the one given: 4x is not close to the
    limit of what can be measured, it is roughly three times looser than it. A tighter threshold is
    available, and loop 127's filter should be re-run against it.

-> outputs/fetch_gaps.json
"""
import collections
import csv
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
MATH = LR.SC / "_mathieson2018.json"
ITZ = LR.SC / "itzhak_supp1.xlsx"
KDEG = LR.SC / "AvgKdegs_genes_v1.csv"
DLK = LR.SC / "dlkcat.json"
OPS = [LR.SC / f"omnipath_{k}.tsv" for k in ("collectri", "dorothea", "tf_target")]
PROT = LR.SC / "human_proteome.fasta.gz"

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def seq_to_gene():
    out, nm, buf = {}, None, []
    with gzip.open(PROT, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                if nm and buf:
                    out["".join(buf)] = nm
                buf, nm = [], None
                for p in ln.split():
                    if p.startswith("GN="):
                        nm = p[3:]
                        break
            else:
                buf.append(ln.strip())
    if nm and buf:
        out["".join(buf)] = nm
    return out


def main():
    t0 = time.time()
    say("=" * 100)
    say("  THE THREE BOTTLENECKS, FETCHED AGAINST")
    say("=" * 100)
    say()

    D = CA.load()
    names = set(D["names"])
    st = CA.state_vector(D)
    cur = set(st["genes"])
    res = {}

    # ---------------------------------------------------------------- gap 1
    say("GAP 1 -- DYNAMICS COVERAGE")
    import pandas as pd
    math = json.load(open(MATH))
    d = pd.read_excel(ITZ, sheet_name="Compact HeLa Spatial Proteome")
    itz = {g for g, c in zip(d["Lead Gene name"].astype(str),
                             pd.to_numeric(d["Estimated Copy number per cell"], errors="coerce"))
           if np.isfinite(c) and c > 0}
    rr = csv.reader(open(KDEG))
    next(rr)
    hela = {x[0] for x in rr if x[1] == "HeLa"}
    prot_new = (set(math) & itz) & names
    same_cell = (hela & itz) & names
    say(f"     current state vector (Schwanhausser, all four)  {len(cur):>7,}")
    say(f"     Mathieson 2018 protein half-lives, human        {len(math):>7,}")
    say(f"     Itzhak 2016 protein copies, HeLa                {len(itz):>7,}")
    say(f"     AvgKdegs mRNA half-lives, HeLa                  {len(hela):>7,}")
    say(f"     -> protein-side state, human                    {len(prot_new):>7,}   "
        f"({len(prot_new) - len(cur):+,} against the current)")
    say(f"     -> genes gained outright                        {len(prot_new - cur):>7,}")
    say(f"     -> HeLa mRNA half-life AND HeLa protein copies  {len(same_cell):>7,}   SAME CELL LINE")
    say(f"     STILL MISSING: absolute mRNA COPY numbers in HeLa -- the one cross-species borrow left")
    res["gap1"] = {"current": len(cur), "mathieson": len(math), "itzhak": len(itz),
                   "kdeg_hela": len(hela), "protein_state_human": len(prot_new),
                   "gained": len(prot_new - cur), "same_cell_line": len(same_cell)}
    say()

    # ---------------------------------------------------------------- gap 2
    say("GAP 2 -- SIGNED REGULATORY EDGES")
    cursig = [e for e in D["model"]["reg"] if e[2] != 0]
    tot, srcs, tgts = set(), set(), set()
    per = {}
    for f in OPS:
        r2 = csv.reader(open(f), delimiter="\t")
        hh = next(r2)
        iS, iT = hh.index("source_genesymbol"), hh.index("target_genesymbol")
        iSt, iIn = hh.index("is_stimulation"), hh.index("is_inhibition")
        n = 0
        for x in r2:
            s, t = x[iS], x[iT]
            sg = 1 if x[iSt] == "True" else (-1 if x[iIn] == "True" else 0)
            if sg and s in names and t in names:
                tot.add((s, t, sg))
                srcs.add(s)
                tgts.add(t)
                n += 1
        per[f.stem.split("_", 1)[1]] = n
        say(f"     {f.stem.split('_', 1)[1]:<12} {n:>7,} signed edges with both ends in the model")
    say(f"     UNION       {len(tot):>7,} edges, {len(srcs):,} regulators, {len(tgts):,} targets")
    say(f"     existing    {len(cursig):>7,} edges")
    say(f"     ratio       {len(tot) / len(cursig):>7.2f}x  -- "
        f"{'FEWER, not more' if len(tot) < len(cursig) else 'an expansion'}")
    say(f"     state genes reached {len(tgts & cur) / len(cur):.1%} against the current 42.9%")
    say(f"     THIS FETCH DID NOT HELP by the measure that motivated it. What it does not settle:")
    say(f"     whether CollecTRI's signs are BETTER. Loop 120 found the existing signs carry no")
    say(f"     information at all -- real 0.5465 against shuffled 0.5494 -- so edge COUNT was")
    say(f"     never the binding problem, and re-running loop 120's protocol on CollecTRI is the")
    say(f"     follow-up this points at.")
    res["gap2"] = {"per_source": per, "union": len(tot), "regulators": len(srcs),
                   "targets": len(tgts), "existing": len(cursig),
                   "ratio": len(tot) / len(cursig),
                   "state_reach": len(tgts & cur) / len(cur), "helped": False}
    say()

    # ---------------------------------------------------------------- gap 3
    say("GAP 3 -- KINETICS VALIDATION")
    dl = [x for x in json.load(open(DLK))
          if x.get("Organism") == "Homo sapiens" and x.get("Sequence")]
    s2g = seq_to_gene()
    byg = collections.defaultdict(list)
    bysub = collections.defaultdict(list)
    for x in dl:
        g = s2g.get(x["Sequence"])
        try:
            v = float(x["Value"])
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        if g:
            byg[g].append(v)
        bysub[(x["Sequence"], x.get("Substrate"))].append(v)
    rep = {k: v for k, v in bysub.items() if len(v) >= 2}
    folds = []
    for v in rep.values():
        m = float(np.median(v))
        folds += [max(q / m, m / q) for q in v]
    folds = np.array(folds)
    say(f"     DLKcat human records                          {len(dl):>7,}")
    say(f"     mapped to a gene by EXACT sequence match      {sum(len(v) for v in byg.values()):>7,}"
        f"   over {len(byg)} genes")
    say(f"     of those, in the model                        {len(set(byg) & names):>7,}")
    say(f"     validation set: 70 -> {len(set(byg) & names)}, a "
        f"{len(set(byg) & names) / 70:.1f}x expansion")
    say()
    say(f"     AND A CORRECTION TO LOOP 127. It put the experimental noise floor at 2.85x from")
    say(f"     leave-one-out WITHIN AN EC -- but an EC class holds many enzymes and substrates, so")
    say(f"     that is enzyme-to-enzyme variation, not reproducibility. Same protein AND same")
    say(f"     substrate gives true replicates:")
    say(f"       {len(rep)} protein-substrate pairs measured more than once, {len(folds)} values")
    say(f"       TRUE experimental floor: median {np.median(folds):.2f}x  "
        f"(75th {np.percentile(folds, 75):.2f}x)")
    say(f"     So 4x is roughly {4.0 / np.median(folds):.0f}x LOOSER than what can actually be")
    say(f"     measured. Loop 127's threshold survives but for the opposite reason to the one")
    say(f"     given, and a tighter filter is available.")
    res["gap3"] = {"human_records": len(dl), "mapped": sum(len(v) for v in byg.values()),
                   "genes": len(byg), "in_model": len(set(byg) & names),
                   "true_floor_median": float(np.median(folds)),
                   "true_floor_75th": float(np.percentile(folds, 75)),
                   "n_replicate_pairs": len(rep), "n_replicate_values": int(len(folds))}
    say()

    say("=" * 100)
    say("  TWO OF THREE FETCHES DELIVERED")
    say("=" * 100)
    say(f"  gap 1 dynamics   +{len(prot_new - cur):,} genes on the protein side, and "
        f"{len(same_cell):,} same-cell-line pairs")
    say(f"  gap 2 signing    NO GAIN -- {len(tot):,} against {len(cursig):,} existing "
        f"({len(tot) / len(cursig):.2f}x)")
    say(f"  gap 3 kinetics   validation 70 -> {len(set(byg) & names)}, and the true noise floor "
        f"is {np.median(folds):.2f}x not 2.85x")
    say("=" * 100)

    man = RM.manifest(inputs=[MATH, ITZ, KDEG, DLK] + OPS,
                      available=len(names), used=len(prot_new), selection="filtered", seed=None,
                      controls=["the existing signed network as the baseline gap 2 had to beat",
                                "both ends required to be model genes, stated as the filter",
                                "exact sequence match rather than name matching for DLKcat",
                                "same protein AND same substrate for the true replicate floor",
                                "the current state vector as the baseline gap 1 had to beat"],
                      note="recorded as a module because a fetch that failed is worth as much as "
                           "one that worked, and gap 2's negative is written down nowhere else")
    RM.report(man, emit=say)
    json.dump({"test": "fetch_gaps", "manifest": man, **res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "fetch_gaps.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'fetch_gaps.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
