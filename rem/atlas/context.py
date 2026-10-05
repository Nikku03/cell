"""Adding context to the dynamic cell, so contradictions can resolve -- or be shown not to.

WHERE DYNCELL STOPPED. Propagating direction over the signed network left 1,169 genes it could not
explain, and the examples were not noise but real biology: BCL2 comes back UP from TP53/NR3C1 and
DOWN from RELA; CRH UP from NR3C1 and DOWN from RELA. Both directions are correct IN DIFFERENT
CONTEXTS -- p53 represses BCL2 while NF-kB induces it, and the glucocorticoid receptor antagonises
NF-kB at CRH. A context-free network has no variable in which to put that, so genuine context
dependence surfaces as unexplainable contradiction.

THE CONTEXT THAT IS ACTUALLY ON DISK. cell_complete.json carries `emask`: a per-gene bitmask over
the 200 cell types in `ctnames`, for 7,496 of 16,492 genes. A regulator cannot act in a cell type
where it is not expressed, so a contradiction RESOLVES wherever only one side of it is present.
That is the same move biology makes, and it needs no rate constants.

AND THE MASK IS NOT UNIFORMLY TRUSTWORTHY, WHICH IS WHY X1 BLOCKS. A five-cell-type probe before
this module was written recovered the marker TFs of erythroid progenitor (KLF1, GATA1, MCM2, NFE2),
both monocyte types (SPI1) and plasmacytoid dendritic cell (SPIB) -- but returned FALSE for all
three markers of CD8-positive memory T cell (SLA2, TBX21, MYBL1). So the decode is right and the
mask's COVERAGE is patchy, and the rate has to be measured across every annotated cell type before
any resolution number is quoted.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

X1  THE MASK'S OWN VALIDITY, BLOCKING, MEASURED ACROSS ALL 40 ANNOTATED CELL TYPES. For each cell
    type with marker TFs in `celltypes`, what fraction of its markers does the mask call expressed
    in that cell type? A marker is by definition expressed in the cell type it marks.
    PREDECLARED: below 70% recovery the mask is not a usable context and NO resolution number may
    be reported from it. The gate is informed by the five-type probe above, which is disclosed
    rather than hidden, and 70% is set above what that probe suggests (60%) so passing is not
    automatic.

X0  THE CEILING GATE. Price resolution before claiming it: of the contradictions, how many have
    their two sides differing in expression in AT LEAST ONE cell type?
    PREDECLARED: a contradiction whose opposing regulators are co-expressed everywhere cannot be
    resolved by this kind of context at all, and that count is the ceiling. If it is most of them,
    cell-type expression is the wrong context variable and this module says so.

X2  RESOLUTION, SPLIT BY SOURCE. Report expression-based resolution and evidence-tier resolution
    SEPARATELY. PREDECLARED: a combined rate hides which mechanism did the work, and evidence tier
    is a statement about our confidence rather than about the cell, so the two must never be added
    into one number.

X3  THE MATCHED CONTROL. Permute the expression masks across genes, preserving each gene's mask
    exactly but reassigning which gene holds it. PREDECLARED: real context must resolve more
    contradictions than permuted context. If it does not, resolution is an artefact of mask
    sparsity and nothing biological is happening.

X4  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import csv
import gzip
import json
import random
import time

HERE = os.path.dirname(__file__)
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
ENCY = R("colab", "data", "cell_complete.json.gz")
CTRI = R("colab", "data", "networks", "collectri.csv")
BANK = R("memory_bank", "cell")
OUT = os.path.join(HERE, "RESULTS_context.txt")
RULE = "=" * 97

import importlib.util
_s = importlib.util.spec_from_file_location("dyncell", os.path.join(HERE, "dyncell.py"))
dyn = importlib.util.module_from_spec(_s); _s.loader.exec_module(dyn)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("ADDING CONTEXT SO THE CONTRADICTIONS CAN RESOLVE"); P_(RULE)
    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    ct = D["ctnames"]
    em = {int(k): int(v) for k, v in D["emask"].items()}
    P_(f"  {len(ct)} cell types; expression mask for {len(em):,} of {len(names):,} genes"
       f"  ({100*len(em)/len(names):.1f}%)")

    def expr(g, c):
        v = em.get(idx.get(g, -1))
        return None if v is None else bool(v >> c & 1)

    # ---- X1  BLOCKING ----------------------------------------------------------------------
    P_("\n" + RULE); P_("X1  THE MASK'S OWN VALIDITY (BLOCKING)"); P_(RULE)
    hit = tot = nocov = 0
    rows = []
    for cell, marks in D["celltypes"].items():
        if cell not in ct:
            continue
        c = ct.index(cell)
        got = [expr(m, c) for m in marks]
        h = sum(1 for x in got if x is True)
        n = sum(1 for x in got if x is None)
        hit += h; tot += len(marks); nocov += n
        rows.append((cell, h, len(marks)))
    rate = 100 * hit / max(tot, 1)
    for cell, h, n in sorted(rows, key=lambda r: r[1] / max(r[2], 1))[:4]:
        P_(f"    worst: {cell:<44} {h}/{n}")
    for cell, h, n in sorted(rows, key=lambda r: -r[1] / max(r[2], 1))[:3]:
        P_(f"    best:  {cell:<44} {h}/{n}")
    P_(f"\n  marker TFs recovered in their own cell type  {hit}/{tot} = {rate:.1f}%")
    P_(f"  markers with NO mask coverage at all          {nocov}")
    if rate < 70.0:
        P_(f"\n  X1: FAIL -- {rate:.1f}% against a predeclared 70% bar. The mask is not a usable")
        P_( "  context. NO resolution number is reported, and the module stops here as predeclared.")
        P_( "\n  WHAT THIS MEANS, STATED PLAINLY: the contradictions dyncell found are real and the")
        P_( "  context variable that would resolve them is the right one, but the expression data")
        P_( "  on disk cannot carry it. A marker gene is expressed in the cell type it defines --")
        P_( "  that is what makes it a marker -- so a mask that misses them is missing ordinary")
        P_( "  expression, not edge cases. Resolving these contradictions needs a cell-type")
        P_( "  expression matrix this repository does not have.")
        P_(f"\n  runtime {time.time()-t0:.1f}s")
        open(OUT, "w").write("\n".join(out) + "\n")
        return
    P_(f"\n  X1: PASS -- {rate:.1f}% recovery, the mask may be used as context")

    # ---- rebuild the contradictions ----------------------------------------------------------
    protein = {names[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    iso, dbd_dead, tier = {}, {}, {}
    pj = R("outputs", "protein_tf_layer.json")
    if os.path.exists(pj):
        iso = {g: v["isoforms"] for g, v in json.load(open(pj)).items() if v.get("isoforms")}
    ij = R("outputs", "isoform_edge_bounds.json")
    if os.path.exists(ij):
        for g, v in json.load(open(ij)).items():
            dbd_dead[g] = len(v.get("dbd_disrupted", [])) / max(v.get("n_isoforms") or 1, 1)
    E = dyn.build_edges(D, names)
    seeds = {g: -1 for g in ("TP53", "NR3C1", "RELA", "MYC", "STAT3") if g in E}
    C = dyn.Cell(names, protein, iso, dbd_dead, tier)
    dyn.propagate(C, E, seeds)
    P_(f"\n  contradictions to resolve: {len(C.conflict):,}")

    # ---- X0 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("X0  THE CEILING: HOW MANY ARE EVEN RESOLVABLE BY EXPRESSION?"); P_(RULE)
    resolvable, no_cov, coexpressed = 0, 0, 0
    detail = {}
    for g, votes in C.conflict.items():
        up = {s for s, sg in votes if sg > 0}
        dn = {s for s, sg in votes if sg < 0}
        ok_ct = []
        covered = False
        for c in range(len(ct)):
            eu = [expr(s, c) for s in up]
            ed = [expr(s, c) for s in dn]
            if any(x is None for x in eu + ed):
                continue
            covered = True
            if any(eu) != any(ed):
                ok_ct.append(c)
        if not covered:
            no_cov += 1
        elif ok_ct:
            resolvable += 1
            detail[g] = ok_ct
        else:
            coexpressed += 1
    n = len(C.conflict)
    P_(f"  resolvable in >=1 cell type       {resolvable:>6,}  {100*resolvable/n:>5.1f}%")
    P_(f"  opposing sides co-expressed       {coexpressed:>6,}  {100*coexpressed/n:>5.1f}%")
    P_(f"  no mask coverage for both sides   {no_cov:>6,}  {100*no_cov/n:>5.1f}%")
    P_(f"\n  X0: the ceiling on expression-context resolution is {100*resolvable/n:.1f}%")

    # ---- X2 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("X2  RESOLUTION, BY SOURCE, NEVER COMBINED"); P_(RULE)
    ev = {}
    with open(CTRI) as fh:
        for r in csv.DictReader(fh):
            ev[(r["source"].upper(), r["target"].upper())] = r["sign.decision"]
    tiered = 0
    for g, votes in C.conflict.items():
        if g in detail:
            continue
        tiers = {s: ev.get((s.upper(), g.upper()), "absent") for s, _ in votes}
        good = {s for s, t in tiers.items() if t in ("PMID", "regulon")}
        bad = {s for s, t in tiers.items() if t == "default activation"}
        if good and bad and not (good & bad):
            sg = {sgn for s, sgn in votes if s in good}
            if len(sg) == 1:
                tiered += 1
    P_(f"  SOURCE 1  cell-type expression    {resolvable:>6,}  {100*resolvable/n:>5.1f}%"
       f"   (a statement about the CELL)")
    P_(f"  SOURCE 2  evidence tier           {tiered:>6,}  {100*tiered/n:>5.1f}%"
       f"   (a statement about OUR CONFIDENCE)")
    P_( "  Reported separately and never summed: the second resolves a contradiction by trusting")
    P_( "  one curator over another, which is not the cell deciding anything.")
    for g in ("BCL2", "CRH", "CSN2"):
        if g in detail:
            cs = [ct[c] for c in detail[g][:3]]
            P_(f"\n    {g}: resolves in {len(detail[g])} cell types, e.g. {cs}")
        elif g in C.conflict:
            P_(f"\n    {g}: NOT resolvable by expression -- both sides co-expressed or uncovered")

    # ---- X3 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("X3  THE MATCHED CONTROL: PERMUTED MASKS"); P_(RULE)
    rng = random.Random(20261005)
    keys = list(em); vals = [em[k] for k in keys]; rng.shuffle(vals)
    emp = dict(zip(keys, vals))

    def exprp(g, c):
        v = emp.get(idx.get(g, -1))
        return None if v is None else bool(v >> c & 1)

    res_p = 0
    for g, votes in C.conflict.items():
        up = {s for s, sg in votes if sg > 0}; dn = {s for s, sg in votes if sg < 0}
        for c in range(len(ct)):
            eu = [exprp(s, c) for s in up]; ed = [exprp(s, c) for s in dn]
            if any(x is None for x in eu + ed):
                continue
            if any(eu) != any(ed):
                res_p += 1
                break
    P_(f"  resolved, real masks       {resolvable:>6,}")
    P_(f"  resolved, permuted masks   {res_p:>6,}")
    ok = resolvable > res_p
    P_(f"\n  X3: {'PASS' if ok else 'FAIL'} -- real context resolves"
       f" {'more' if ok else 'NO MORE'} than permuted,"
       f" so resolution {'is not an artefact of mask sparsity' if ok else 'IS an artefact of mask sparsity'}")

    json.dump({g: [ct[c] for c in cs] for g, cs in detail.items()},
              open(os.path.join(BANK, "context_resolution.json"), "w"))
    P_(f"\n  written: memory_bank/cell/context_resolution.json ({len(detail):,} genes)")

    P_("\n" + RULE); P_("X4  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. The mask is binary PRESENCE, not activity. A TF can be expressed and not active.")
    P_("  2. Resolving a contradiction by ABSENCE says which edge cannot fire here. It does not")
    P_("     predict the magnitude or even confirm the surviving edge fires.")
    P_("  3. Coverage bounds everything: the mask holds 45.5% of genes, so most contradictions")
    P_("     cannot be adjudicated either way.")
    P_("  4. No measurement enters this loop still. Context narrows the network against itself.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
