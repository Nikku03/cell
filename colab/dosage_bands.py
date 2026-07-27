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
# Collins et al. 2022, "A cross-disorder dosage sensitivity map of the human genome" (Zenodo record 6347673).
# pHaplo and pTriplo are posterior probabilities (0-1) that a gene is haploinsufficient / triplosensitive, fitted
# on rare-CNV association across ~750k individuals. This is the ONLY source that populates E_max at genome scale:
# ClinGen returns a triplosensitivity code for 1,219 genes but only THREE of them score 3 (sufficient evidence),
# so the curated ceiling was unusable. Fetched from the API content endpoint --
# /api/records/6347673/files/<name>/content -- because the direct-download URL form returns 403.
COLLINS = SP / "ptriplo.tsv.gz"
ESS_CUT = -0.5


def load_gnomad():
    """pLI and LOEUF per gene. The file is bgzip, which gzip reads fine.

    Returns empty rather than raising when the file is absent: the scratchpad is ephemeral, and one missing source
    should cost its own column, not the whole report. Coverage is printed per source, so an empty one is visible.
    """
    out = {}
    if not GNOMAD.exists():
        return out
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
    if not CLINGEN.exists():
        return hi, ts
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


def load_collins():
    """pHaplo (floor) and pTriplo (ceiling) per gene symbol.

    These are CONTINUOUS posteriors, not the ordinal 0/1/2/3/30/40 codes ClinGen uses, so they are kept on their own
    scale and never merged with the ClinGen columns. The paper's own thresholds are pHaplo >= 0.86 and
    pTriplo >= 0.94; both are reported here rather than baked in, because a threshold chosen elsewhere is a
    modelling decision that should be visible."""
    out = {}
    if not COLLINS.exists():
        return out
    with gzip.open(COLLINS, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                out[p[0].strip()] = (float(p[1]), float(p[2]))
            except ValueError:
                continue
    return out


def load_depmap_mean():
    if not DEPMAP.exists():
        return {}
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
    col = load_collins()
    print(f"sources: PaxDb-style PPM {len(ppm):,} | gnomAD {len(gno):,} | "
          f"ClinGen HI {len(hi):,} / TS {len(ts):,} | DepMap {len(dep):,} | Collins pHaplo/pTriplo {len(col):,}")

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
            "Emin_pHaplo": col.get(g, (None, None))[0],
            "Emax_clingen_TS": ts.get(g),
            "Emax_pTriplo": col.get(g, (None, None))[1],
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
    ph_hi = {g for g, b in bands.items() if (b["Emin_pHaplo"] or 0) >= 0.86}
    print(f"   Collins pHaplo >= 0.86   {len(ph_hi):>6,}   rare-CNV haploinsufficiency posterior")
    print(f"   pLI AND DepMap           {len(pli_hi & dep_ess):>6,}   "
          f"({len(pli_hi & dep_ess)/max(len(pli_hi | dep_ess),1):.1%} of their union)")
    print(f"   pLI but NOT DepMap       {len(pli_hi - dep_ess):>6,}   constrained in humans, dispensable in K562")
    print(f"   DepMap but NOT pLI       {len(dep_ess - pli_hi):>6,}   needed by the cell, not dosage-constrained")
    print(f"   ClinGen HI=3 AND DepMap  {len(cg_hi & dep_ess):>6,} of {len(cg_hi):,}")
    print("   -> the sources are NOT interchangeable; they are kept as separate fields, never averaged")

    # ---- ceiling: the direction most models ignore ----
    ts3 = {g for g, b in bands.items() if b["Emax_clingen_TS"] == 3}
    ts40 = {g for g, b in bands.items() if b["Emax_clingen_TS"] == 40}
    pt_hi = {g for g, b in bands.items() if (b["Emax_pTriplo"] or 0) >= 0.94}
    print(f"\nCEILING (triplosensitivity) -- the direction almost every network model ignores")
    print(f"   ClinGen TS = 3  (extra copy is pathogenic)      {len(ts3):>5,}  <- curated, and unusably sparse")
    print(f"   ClinGen TS = 40 (dosage sensitivity unlikely)   {len(ts40):>5,}")
    print(f"   Collins pTriplo >= 0.94                         {len(pt_hi):>5,}  <- this is what makes E_max real")
    print(f"   genes that are BOTH pHaplo>=0.86 and pTriplo>=0.94  {len(ph_hi & pt_hi):>5,}  "
          f"-- narrow tolerated band in BOTH directions")
    print(f"   pTriplo-only (tolerate loss, not excess)        {len(pt_hi - ph_hi):>5,}")
    print(f"   pHaplo-only  (tolerate excess, not loss)        {len(ph_hi - pt_hi):>5,}")
    both = [(b["Emin_pHaplo"], b["Emax_pTriplo"]) for b in bands.values()
            if b["Emin_pHaplo"] is not None and b["Emax_pTriplo"] is not None]
    if len(both) > 100:
        x = np.array([a for a, _ in both]); y = np.array([c for _, c in both])
        print(f"   corr(pHaplo, pTriplo) = {np.corrcoef(x, y)[0,1]:+.4f} over {len(both):,} genes -- if this were "
              f"near 1 the\n     ceiling would be a restatement of the floor and would add nothing")
    # is the ceiling independent of what the cell measurably needs? DepMap only sees LOSS, so a ceiling that
    # correlated strongly with it would be suspect.
    pairs = [(bands[g]["Emax_pTriplo"], dep[g]) for g in bands
             if bands[g]["Emax_pTriplo"] is not None and g in dep]
    if len(pairs) > 100:
        x = np.array([a for a, _ in pairs]); y = np.array([b_ for _, b_ in pairs])
        print(f"   corr(pTriplo, DepMap effect) = {np.corrcoef(x, y)[0,1]:+.4f} over {len(pairs):,} genes -- DepMap "
              f"measures LOSS only,\n     so a ceiling should NOT track it closely; this is the independence check")

    # ---- DOES THE CEILING SAY ANYTHING TRUE? a falsifiable prediction, not a coverage boast ----
    # The ceiling and the floor make OPPOSITE predictions about a knockout, so the split between them is testable:
    #   pHaplo-high, pTriplo-low  -> losing one copy is selected against -> knockout should COST fitness
    #   pTriplo-high, pHaplo-low  -> only EXCESS is selected against     -> knockout should be WELL TOLERATED
    # DepMap measures loss only and never enters the Collins scores, so it is a genuine outside test.
    #
    # THE PREDICTION FAILED, AND IN THE OPPOSITE DIRECTION. Measured: pTriplo-only genes are MORE essential
    # (-0.3194, n=551) than pHaplo-only genes (-0.2100, n=1,817), difference +0.1094 with a bootstrap CI of
    # [+0.0606, +0.1603] -- separated, on the wrong side. So "triplosensitive but not haploinsufficient" does NOT
    # mean "safe to knock out".
    #
    # The interpretation is the one this module already documents for pLI: triplosensitive genes are typically
    # dosage-balanced complex members and core regulators, which is exactly the class a cancer line cannot lose,
    # while pHaplo captures developmental haploinsufficiency in genes K562 never needed -- the same effect as the
    # 2,251 genes that are pLI-constrained and DepMap-dispensable. E_max is therefore populated and not redundant
    # with the floor (corr 0.57 with pHaplo, -0.18 with DepMap), but it is NOT a "loss is tolerated" flag and must
    # not be used as one. The hypothesis is recorded as refuted rather than deleted.
    def dmean(S):
        v = [dep[g] for g in S if g in dep]
        return (float(np.mean(v)) if v else float("nan"), len(v))

    if dep and col:
        allm, alln = dmean([g for g in bands if g in dep])
        print(f"\nDOES THE CEILING PREDICT? DepMap mean effect ({alln:,} genes, baseline {allm:+.4f})")
        rows = [("pHaplo-only  (loss is the problem)", ph_hi - pt_hi),
                ("both directions", ph_hi & pt_hi),
                ("pTriplo-only (EXCESS is the problem)", pt_hi - ph_hi),
                ("neither", set(bands) - ph_hi - pt_hi)]
        got = {}
        for lab, S in rows:
            m, n_ = dmean(S)
            got[lab] = (m, n_)
            print(f"   {lab:<38} {m:+.4f}  n={n_:,}  delta {m-allm:+.4f}")
        a = got["pHaplo-only  (loss is the problem)"][0]
        b_ = got["pTriplo-only (EXCESS is the problem)"][0]
        # bootstrap the difference between the two one-sided groups
        rng = np.random.default_rng(0)
        A = [dep[g] for g in (ph_hi - pt_hi) if g in dep]
        B = [dep[g] for g in (pt_hi - ph_hi) if g in dep]
        if len(A) > 30 and len(B) > 30:
            d = np.array([np.mean(rng.choice(A, len(A))) - np.mean(rng.choice(B, len(B))) for _ in range(4000)])
            lo, hi_ = np.percentile(d, 2.5), np.percentile(d, 97.5)
            print(f"   pHaplo-only MINUS pTriplo-only: {a-b_:+.4f}  95% CI [{lo:+.4f}, {hi_:+.4f}]  "
                  f"{'SEPARATED' if hi_ < 0 or lo > 0 else 'not separated'}")
            print("   negative = the loss-sensitive group is more essential, which is the predicted direction")

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
               "ceiling_sets": {"ClinGen_TS3": len(ts3), "ClinGen_TS40": len(ts40), "HI3_and_TS3": len(cg_hi & ts3),
                                "pTriplo>=0.94": len(pt_hi), "pHaplo>=0.86": len(ph_hi),
                                "both_directions": len(ph_hi & pt_hi), "pTriplo_only": len(pt_hi - ph_hi),
                                "pHaplo_only": len(ph_hi - pt_hi)},
               "bands": bands}, open(OUT / "dosage_bands.json", "w"))
    print(f"\n  -> {OUT/'dosage_bands.json'}")


if __name__ == "__main__":
    main()
