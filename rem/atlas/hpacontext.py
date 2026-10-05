"""Closing the loop: a real cell-type expression matrix, and the contradictions it resolves.

WHAT WAS MISSING, AND IT WAS ONE ORDINARY OBJECT. dyncell left 1,169 genes whose incoming signs
disagreed, and the examples were real biology -- BCL2 UP from TP53/NR3C1 and DOWN from RELA, CRH UP
from NR3C1 and DOWN from RELA. Both directions are correct in different cell types, so a regulator
that is not expressed cannot be responsible. context.py tried to adjudicate that with the
encyclopedia's own `emask` and its BLOCKING gate failed: the mask recovered marker TFs in their own
cell type only 45/89 = 50.6% of the time, with whole cell types at 0 of 3 and 0 of 4. The variable
was right and the data could not carry it.

SO FETCH DATA THAT CAN. The Human Protein Atlas publishes consensus single-cell RNA by cell type:
3,087,080 rows over 20,151 genes and 154 cell types, against the mask's 7,496 genes. That is the
object four independent routes in this record said was missing.

LICENCE, BECAUSE THIS REPO HAS A PROBLEM THERE. HPA data is CC BY-SA 4.0. That is free to use with
attribution, but SHARE-ALIKE -- a derivative distributed onward inherits the copyleft. It is
therefore NOT in the same clean class as UniProt's CC BY 4.0, and it is not in the refused class of
SIGNOR's CC-BY-NC either. Flagged here so a commercial decision is made deliberately rather than
discovered later, which is exactly what went wrong with signor_human.tsv.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

H1  THE MATRIX'S OWN VALIDITY, BLOCKING, IN BOTH DIRECTIONS, ON TEXTBOOK CASES. Twelve
    marker/cell-type pairs whose answer is not in dispute (ALB in hepatocytes, INS in pancreatic
    islet cells, MYH7 in cardiomyocytes, RHO in rod photoreceptors, ...) must come back EXPRESSED;
    five deliberately absurd pairs (ALB in cardiomyocytes, INS in astrocytes, RHO in hepatocytes,
    ...) must come back NOT expressed.
    PREDECLARED: >=90% on the positives AND >=90% on the negatives. The bar is higher than the 70%
    set for `emask` because these are textbook facts about a standard resource -- anything less
    means the threshold or the parse is wrong, not that biology is subtle. A matrix that passes
    only the positives is not validated: a matrix calling everything expressed would do that.

H0  THE CEILING GATE. Of the 1,169 contradictions, how many have BOTH opposing sides covered by
    the matrix at all?
    PREDECLARED: resolution may only ever be quoted against the covered subset, and the uncovered
    count is reported beside it. The honest denominator is not 1,169 unless coverage is complete.

H2  RESOLUTION, AND WHICH DIRECTION IT PICKS. For each covered contradiction, the cell types where
    exactly one side is expressed, and therefore which way the gene moves there.
    PREDECLARED: report the DISTRIBUTION over how many cell types resolve each contradiction, not
    a mean. A contradiction resolved in 1 of 154 cell types and one resolved in 140 are different
    claims and a mean would merge them.

H3  THE MATCHED CONTROL. Permute which gene holds which expression profile, preserving every
    profile exactly. PREDECLARED: real expression must resolve more than permuted. If not,
    resolution is an artefact of profile sparsity and nothing biological happened.

H4  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import csv
import gzip
import io
import json
import random
import time
import urllib.request
import zipfile

HERE = os.path.dirname(__file__)
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CACHE = os.environ.get("HPA_SC_ZIP", os.path.join(HERE, "_cache", "hpa_sc.tsv.zip"))
URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip"
ENCY = R("colab", "data", "cell_complete.json.gz")
BANK = R("memory_bank", "cell")
OUT = os.path.join(HERE, "RESULTS_hpacontext.txt")
RULE = "=" * 97
THRESH = 1.0          # nCPM; HPA's own detection cutoff

import importlib.util
_s = importlib.util.spec_from_file_location("dyncell", os.path.join(HERE, "dyncell.py"))
dyn = importlib.util.module_from_spec(_s); _s.loader.exec_module(dyn)

POS = [("ALB", "hepatocytes"), ("INS", "pancreatic islet cells"), ("MYH7", "cardiomyocytes"),
       ("CD3E", "t-cells"), ("SPI1", "monocytes"), ("GATA1", "erythrocyte progenitors"),
       ("KRT14", "basal keratinocytes"), ("GFAP", "astrocytes"),
       ("SFTPC", "alveolar cells type 2"), ("MS4A1", "b-cells"), ("MBP", "oligodendrocytes"),
       ("RHO", "rod photoreceptor cells")]
NEG = [("ALB", "cardiomyocytes"), ("INS", "astrocytes"), ("MYH7", "b-cells"),
       ("RHO", "hepatocytes"), ("SFTPC", "t-cells")]


def fetch():
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 1_000_000:
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "rem-atlas/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r:
        open(CACHE, "wb").write(r.read())
    return CACHE


def load(path):
    """gene -> bitmask over cell types where nCPM >= THRESH."""
    z = zipfile.ZipFile(path)
    name = z.namelist()[0]
    cts, ci = [], {}
    mask = collections.defaultdict(int)
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, "utf8"), delimiter="\t")
        next(rd)
        for row in rd:
            if len(row) < 4:
                continue
            g, c, v = row[1].strip().upper(), row[2], row[3]
            j = ci.get(c)
            if j is None:
                j = ci[c] = len(cts); cts.append(c)
            try:
                if float(v) >= THRESH:
                    mask[g] |= 1 << j
            except ValueError:
                pass
    return cts, dict(mask)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("A REAL CELL-TYPE EXPRESSION MATRIX, AND WHAT IT RESOLVES"); P_(RULE)
    P_("  fetching Human Protein Atlas consensus single-cell RNA (CC BY-SA 4.0)...")
    cts, M = load(fetch())
    P_(f"  {len(M):,} genes x {len(cts)} cell types; detection threshold nCPM >= {THRESH}")
    pres = {g: bin(m).count('1') for g, m in M.items()}
    srt = sorted(pres.values())
    P_(f"  cell types per gene: median {srt[len(srt)//2]}, p25 {srt[len(srt)//4]},"
       f" p75 {srt[3*len(srt)//4]}, max {srt[-1]}")

    def ex(g, c):
        m = M.get(g.upper())
        if m is None or c not in cts:
            return None
        return bool(m >> cts.index(c) & 1)

    # ---- H1  BLOCKING ----------------------------------------------------------------------
    P_("\n" + RULE); P_("H1  VALIDITY ON TEXTBOOK CASES, BOTH DIRECTIONS (BLOCKING)"); P_(RULE)
    ph = [(g, c, ex(g, c)) for g, c in POS]
    nh = [(g, c, ex(g, c)) for g, c in NEG]
    pok = sum(1 for _, _, v in ph if v is True)
    nok = sum(1 for _, _, v in nh if v is False)
    for g, c, v in ph:
        P_(f"    {'PASS' if v is True else 'FAIL'}  expect PRESENT  {g:<7} in {c}")
    for g, c, v in nh:
        P_(f"    {'PASS' if v is False else 'FAIL'}  expect ABSENT   {g:<7} in {c}")
    pr, nr = 100 * pok / len(POS), 100 * nok / len(NEG)
    P_(f"\n  positives {pok}/{len(POS)} = {pr:.1f}%     negatives {nok}/{len(NEG)} = {nr:.1f}%")
    if pr < 90 or nr < 90:
        P_(f"\n  H1: FAIL -- the matrix or the threshold is wrong. Nothing downstream reported.")
        open(OUT, "w").write("\n".join(out) + "\n")
        return
    P_(f"\n  H1: PASS -- textbook cases recovered in both directions")

    # ---- rebuild the contradictions ----------------------------------------------------------
    D = json.load(gzip.open(ENCY))
    names = [r["name"] for r in D["genes"]]
    protein = {names[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    iso, dbd = {}, {}
    pj = R("outputs", "protein_tf_layer.json")
    if os.path.exists(pj):
        iso = {g: v["isoforms"] for g, v in json.load(open(pj)).items() if v.get("isoforms")}
    ij = R("outputs", "isoform_edge_bounds.json")
    if os.path.exists(ij):
        for g, v in json.load(open(ij)).items():
            dbd[g] = len(v.get("dbd_disrupted", [])) / max(v.get("n_isoforms") or 1, 1)
    E = dyn.build_edges(D, names)
    seeds = {g: -1 for g in ("TP53", "NR3C1", "RELA", "MYC", "STAT3") if g in E}
    C = dyn.Cell(names, protein, iso, dbd, {})
    dyn.propagate(C, E, seeds)
    n = len(C.conflict)
    P_(f"\n  contradictions from dyncell: {n:,}")

    # ---- H0 / H2 -----------------------------------------------------------------------------
    P_("\n" + RULE); P_("H0  COVERAGE, THEN H2  RESOLUTION"); P_(RULE)
    nct = len(cts)
    covered, resolved, co_everywhere, uncov = 0, 0, 0, 0
    detail, hist = {}, collections.Counter()
    for g, votes in C.conflict.items():
        up = {s for s, sg in votes if sg > 0}
        dn = {s for s, sg in votes if sg < 0}
        mu = [M.get(s.upper()) for s in up]
        md = [M.get(s.upper()) for s in dn]
        if any(x is None for x in mu + md) or not mu or not md:
            uncov += 1
            continue
        covered += 1
        U = 0
        for x in mu:
            U |= x
        Dn = 0
        for x in md:
            Dn |= x
        only_up = U & ~Dn
        only_dn = Dn & ~U
        k = bin(only_up).count('1') + bin(only_dn).count('1')
        if k:
            resolved += 1
            hist[min(k, 160)] += 1
            detail[g] = {"n_cell_types_resolved": k,
                         "up_only": [cts[i] for i in range(nct) if only_up >> i & 1][:5],
                         "down_only": [cts[i] for i in range(nct) if only_dn >> i & 1][:5]}
        else:
            co_everywhere += 1
    P_(f"  both sides covered by the matrix   {covered:>6,}  {100*covered/n:>5.1f}%")
    P_(f"  one or both sides uncovered        {uncov:>6,}  {100*uncov/n:>5.1f}%")
    P_(f"\n  RESOLVED in >=1 cell type          {resolved:>6,}  {100*resolved/max(covered,1):>5.1f}% of covered")
    P_(f"  co-expressed in every cell type    {co_everywhere:>6,}  {100*co_everywhere/max(covered,1):>5.1f}% of covered")
    P_(f"\n  as a fraction of ALL {n:,} contradictions: {100*resolved/n:.1f}%")
    P_(f"\n  H2 DISTRIBUTION -- cell types in which a contradiction resolves:")
    P_(f"    {'cell types':>12} {'contradictions':>16}")
    for lo, hi in ((1, 1), (2, 5), (6, 20), (21, 50), (51, 100), (101, 160)):
        c = sum(v for k2, v in hist.items() if lo <= k2 <= hi)
        P_(f"    {f'{lo}-{hi}':>12} {c:>16,}")
    for g in ("BCL2", "CRH", "CSN2", "IGF1R", "RB1"):
        if g in detail:
            d = detail[g]
            P_(f"\n    {g}: resolves in {d['n_cell_types_resolved']} cell types")
            if d["up_only"]:
                P_(f"       goes UP   where only its activators are present: {d['up_only'][:3]}")
            if d["down_only"]:
                P_(f"       goes DOWN where only its repressors are present: {d['down_only'][:3]}")
        elif g in C.conflict:
            P_(f"\n    {g}: not resolved (co-expressed everywhere, or uncovered)")

    # ---- H3 ----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("H3  THE MATCHED CONTROL: PERMUTED PROFILES"); P_(RULE)
    rng = random.Random(20261005)
    ks = list(M); vs = [M[k] for k in ks]; rng.shuffle(vs)
    Mp = dict(zip(ks, vs))
    rp = 0
    for g, votes in C.conflict.items():
        up = {s for s, sg in votes if sg > 0}; dn = {s for s, sg in votes if sg < 0}
        mu = [Mp.get(s.upper()) for s in up]; md = [Mp.get(s.upper()) for s in dn]
        if any(x is None for x in mu + md) or not mu or not md:
            continue
        U = 0
        for x in mu:
            U |= x
        Dn = 0
        for x in md:
            Dn |= x
        if (U & ~Dn) or (Dn & ~U):
            rp += 1
    P_(f"  resolved, real profiles      {resolved:>6,}")
    P_(f"  resolved, permuted profiles  {rp:>6,}")
    ok = resolved > rp
    P_(f"\n  H3: {'PASS' if ok else 'FAIL'} -- real expression resolves"
       f" {'more' if ok else 'NO MORE'} than permuted"
       f" ({resolved/max(rp,1):.3f}x). {'Not an artefact of profile sparsity.' if ok else 'Resolution IS an artefact.'}")

    os.makedirs(BANK, exist_ok=True)
    json.dump({"source": "Human Protein Atlas rna_single_cell_type (CC BY-SA 4.0)",
               "n_cell_types": nct, "threshold_nCPM": THRESH,
               "contradictions": n, "covered": covered, "resolved": resolved,
               "resolution": detail},
              open(os.path.join(BANK, "context_resolution.json"), "w"))
    P_(f"\n  written: memory_bank/cell/context_resolution.json ({len(detail):,} resolved genes)")

    P_("\n" + RULE); P_("H4  WHAT THIS DOES NOT SETTLE"); P_(RULE)
    P_("  1. Expression is PRESENCE, not activity. A TF can be transcribed and not active, so a")
    P_("     resolution says which edge CANNOT fire here, not that the surviving one does.")
    P_("  2. HPA consensus is averaged over donors and studies. It cannot speak to a particular")
    P_("     experiment, and the contradictions came from a network with no experiment behind it.")
    P_("  3. nCPM >= 1.0 is HPA's cutoff, adopted here unexamined. A sweep would move the numbers")
    P_("     and has not been run.")
    P_("  4. This closes the loop against a REFERENCE, not against a measurement of the system")
    P_("     being modelled. The remaining gap is the same one four routes already named: a")
    P_("     perturbation experiment in a named cell type. Nothing here substitutes for it.")
    P_("  5. CC BY-SA 4.0 is share-alike. A distributed derivative inherits the copyleft.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
