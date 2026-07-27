"""DOSAGE BANDS: give every node a baseline, a floor and a ceiling, so a knockout becomes a dose rather than a switch.

WHY THIS IS THE RIGHT NEXT PIECE. Everything modelled so far treats a knockout as binary -- the protein is there or
it is not, the reaction runs or it stops. Real perturbation is a DOSE. CRISPR knockout removes most of a protein,
CRISPRi knocks it down partway, and a cell tolerates a 50% loss of one enzyme while dying from a 20% loss of
another. The consequence map says "this reaction stops"; what it should say is "flux through this reaction falls to
X% of baseline, and the cell tolerates that or it does not". Three numbers per node make that possible:

    E0    BASELINE   PaxDb-style protein abundance in PPM. The node's native concentration -- what "normal" means
                     for this protein, and the scale against which any loss is a fraction.
    Emin  FLOOR      how far the dose can fall before the node fails. Sourced three ways because they disagree and
                     the disagreement is informative:
                       gnomAD pLI / LOEUF  -- human population constraint: is losing one copy selected against?
                       ClinGen HI score    -- expert-curated haploinsufficiency, the clinical ground truth
                       DepMap essentiality -- measured fitness cost in cells, which is what this project scores on
    Emax  CEILING    how far the dose can rise before it is toxic. ClinGen triplosensitivity, the dosage direction
                     almost every network model ignores entirely: too much of a subunit is as bad as too little,
                     because unbalanced complex stoichiometry aggregates.

THE THREE FLOOR SOURCES MEASURE DIFFERENT THINGS AND MUST NOT BE AVERAGED. pLI asks whether heterozygous loss is
depleted in healthy humans over evolutionary time. DepMap asks whether a cancer cell line dies in three weeks.
ClinGen asks whether a clinician has seen disease from one bad copy. A ribosomal protein is essential in DepMap and
constrained in gnomAD; a haploinsufficient developmental TF may be strongly constrained and completely dispensable
in K562. Reporting them separately, and measuring how often they agree, is the honest form -- collapsing them to one
"essentiality" number destroys exactly the distinction that makes dosage interesting.

WHAT THIS FILE DOES AND DOES NOT CLAIM. It assembles and audits the bands: coverage per source, agreement between
sources, and whether the bands separate the knockouts this project can already measure. It does NOT yet claim the
bands improve prediction -- that is a separate test, and asserting it here without measuring would repeat the
mistake this session has already had to correct twice.
"""
import collections
import csv
import gzip
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GNOMAD = SP / "gnomad_constraint.txt.bgz"
CLINGEN = SP / "ClinGen_gene_curation_list_GRCh38.tsv"
DEPMAP = SP / "CRISPRGeneEffect.csv"
ESS_CUT = -0.5


def load_gnomad():
    """pLI and LOEUF per gene. The file is bgzip, which gzip reads fine."""
    out = {}
    with gzip.open(GNOMAD, "rt") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            g = row.get("gene")
            if not g:
                continue
            def num(k):
                v = row.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            rec = {"pLI": num("pLI"), "loeuf": num("oe_lof_upper"), "oe_lof": num("oe_lof")}
            # a gene appears once per transcript in some releases; keep the most constrained
            if g not in out or (rec["pLI"] is not None and (out[g]["pLI"] or -1) < rec["pLI"]):
                out[g] = rec
    return out


def load_clingen():
    """ClinGen curated haploinsufficiency and triplosensitivity.

    Scores are ordinal, not probabilities: 3 = sufficient evidence, 2 = emerging, 1 = little, 0 = no evidence,
    30 = autosomal recessive, 40 = dosage sensitivity unlikely. 30 and 40 are NOT weak evidence and are kept as
    their own categories rather than being coerced onto the 0-3 scale.
    """
    hi, ts = {}, {}
    with open(CLINGEN) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            g = p[0].strip()
            def sc(x):
                x = (x or "").strip()
                return int(x) if re.fullmatch(r"-?\d+", x) else None
            # COLUMN INDICES VERIFIED AGAINST THE FILE HEADER, not guessed. Haploinsufficiency Score is column 5
            # (index 4) and Triplosensitivity Score is column 13 (index 12); columns 7-12 are Haploinsufficiency
            # PMIDs. An earlier version read index 6 and silently parsed PubMed IDs as scores -- 679 genes came back
            # with a "value" and not one of them was a valid 0-3/30/40 code, which is what exposed it.
            h = sc(p[4])
            t = sc(p[12]) if len(p) > 12 else None
            if h is not None:
                hi[g] = h
            if t is not None:
                ts[g] = t
    return hi, ts


def load_depmap_mean():
    with open(DEPMAP) as f:
        r = csv.reader(f)
        head = next(r)
        genes = [h.split(" (")[0] for h in head[1:]]
        tot = np.zeros(len(genes))
        cnt = np.zeros(len(genes))
        for row in r:
            v = np.array([np.nan if x == "" else float(x) for x in row[1:]])
            ok = ~np.isnan(v)
            tot[ok] += v[ok]
            cnt[ok] += 1
    eff = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return {g: float(e) for g, e in zip(genes, eff) if not np.isnan(e)}


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: v for k, v in D["ppm"].items() if k.isdigit() and int(k) < len(names)}
    gno = load_gnomad()
    hi, ts = load_clingen()
    dep = load_depmap_mean()
    print(f"sources: PaxDb-style PPM {len(ppm):,} | gnomAD {len(gno):,} | "
          f"ClinGen HI {len(hi):,} / TS {len(ts):,} | DepMap {len(dep):,}")

    bands, cov = {}, collections.Counter()
    for g in names:
        e0 = ppm.get(g)
        gn = gno.get(g, {})
        rec = {
            "E0_ppm": e0,
            "Emin_pLI": gn.get("pLI"),
            "Emin_loeuf": gn.get("loeuf"),
            "Emin_clingen_HI": hi.get(g),
            "Emin_depmap": dep.get(g),
            "Emax_clingen_TS": ts.get(g),
        }
        for k, v in rec.items():
            cov[k] += v is not None
        bands[g] = rec
    n = len(names)
    print(f"\ncoverage over {n:,} network genes")
    for k, c in cov.most_common():
        print(f"   {k:<18} {c:>7,}  ({c/n:.1%})")

    # ---- do the three floor sources agree? they measure different things, so this is the informative check ----
    pli_hi = {g for g, b in bands.items() if (b["Emin_pLI"] or 0) >= 0.9}
    dep_ess = {g for g, b in bands.items() if b["Emin_depmap"] is not None and b["Emin_depmap"] < ESS_CUT}
    cg_hi = {g for g, b in bands.items() if b["Emin_clingen_HI"] == 3}
    print(f"\nFLOOR sources, as gene sets")
    print(f"   gnomAD pLI >= 0.9        {len(pli_hi):>6,}   population constraint")
    print(f"   DepMap mean effect < {ESS_CUT}  {len(dep_ess):>6,}   fitness cost in cells")
    print(f"   ClinGen HI = 3           {len(cg_hi):>6,}   curated haploinsufficient")
    print(f"   pLI AND DepMap           {len(pli_hi & dep_ess):>6,}   "
          f"({len(pli_hi & dep_ess)/max(len(pli_hi | dep_ess),1):.1%} of their union)")
    print(f"   pLI but NOT DepMap       {len(pli_hi - dep_ess):>6,}   constrained in humans, dispensable in K562")
    print(f"   DepMap but NOT pLI       {len(dep_ess - pli_hi):>6,}   needed by the cell, not dosage-constrained")
    print(f"   ClinGen HI=3 AND DepMap  {len(cg_hi & dep_ess):>6,} of {len(cg_hi):,}")
    print("   -> the sources are NOT interchangeable; they are kept as separate fields, never averaged")

    # ---- ceiling: the direction most models ignore ----
    ts3 = {g for g, b in bands.items() if b["Emax_clingen_TS"] == 3}
    ts40 = {g for g, b in bands.items() if b["Emax_clingen_TS"] == 40}
    print(f"\nCEILING (triplosensitivity)")
    print(f"   ClinGen TS = 3  (extra copy is pathogenic)      {len(ts3):>5,}")
    print(f"   ClinGen TS = 40 (dosage sensitivity unlikely)   {len(ts40):>5,}")
    print(f"   genes that are BOTH HI=3 and TS=3               {len(cg_hi & ts3):>5,}  "
          f"-- a narrow tolerated band in both directions")

    # ---- does abundance relate to constraint? a sanity check on E0 ----
    pairs = [(b["E0_ppm"], b["Emin_loeuf"]) for b in bands.values()
             if b["E0_ppm"] and b["Emin_loeuf"] is not None and b["Emin_loeuf"] > 0]
    if len(pairs) > 100:
        x = np.log10([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        print(f"\nSANITY: corr(log10 abundance, LOEUF) = {np.corrcoef(x, y)[0,1]:+.4f} over {len(pairs):,} genes")
        print("   negative is expected: abundant proteins tend to be more constrained (lower LOEUF)")

    json.dump({"n_genes": n, "coverage": dict(cov),
               "floor_sets": {"pLI>=0.9": len(pli_hi), "DepMap_essential": len(dep_ess), "ClinGen_HI3": len(cg_hi),
                              "pLI_and_DepMap": len(pli_hi & dep_ess), "pLI_not_DepMap": len(pli_hi - dep_ess),
                              "DepMap_not_pLI": len(dep_ess - pli_hi)},
               "ceiling_sets": {"ClinGen_TS3": len(ts3), "ClinGen_TS40": len(ts40), "HI3_and_TS3": len(cg_hi & ts3)},
               "bands": bands}, open(OUT / "dosage_bands.json", "w"))
    print(f"\n  -> {OUT/'dosage_bands.json'}")


if __name__ == "__main__":
    main()
