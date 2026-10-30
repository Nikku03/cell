"""regulate — the honest test of "binding != regulation, UNLESS it recruits polymerase". A TF sitting on DNA is occupancy;
regulation is occupancy that lasts long enough to appoint RNA Pol II and fire transcription. This asks, on real data,
whether the PRODUCTIVITY signature (Pol II present + nascent eRNA) separates elements that ACTUALLY regulate a gene from
elements that are merely open/marked.

  Gate 4a  Pol II    : POLR2A + POLR2AphosphoS5 (initiating, Ser5P) ChIP-seq — is the polymerase physically recruited here?
  Gate 4b  eRNA      : PRO-cap BIDIRECTIONAL peaks — is the element actually being TRANSCRIBED (the readout of a productively
                       engaged enhancer)? Bidirectional nascent caps are the cleanest "this enhancer is firing" signal.
  (for contrast)     : DNase (open, Gate1) and H3K27ac (active MARK, Gate2) — chromatin state, not the act of transcription.

  GROUND TRUTH = regulation, measured by CRISPR: ENCODE combined CRISPR element-gene dataset (ENCFF968BZL; harmonised
  Nasser 2021 / Gasperini 2019 / Schraivogel 2020). Each element was silenced and a gene's expression measured; `Significant`
  = the element genuinely regulates that gene. An element is a REGULATOR if it is Significant for >=1 gene (valid pairs only).

  The claim under test: among candidate elements, those that RECRUIT POL II / FIRE eRNA are enriched for true CRISPR
  regulators — i.e. the functional ACT of transcription, not the chromatin mark, is what marks regulation.

HONEST BOUNDARY: ChIP occupancy is a population proxy for residence-time x concentration, NOT single-molecule dwell time
(no genome-wide dwell-time data exists). And the CRISPR elements are already accessibility-preselected, so this measures the
MARGINAL value of the productivity signal on top of open chromatin — the right test for the user's thesis. All real K562 data.
"""
import os
import json, gzip, bisect
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan")) / "invivo"
M = OUT / "marks"
MARKS = ["dnase", "h3k27ac", "polr2a", "polr2s5", "procap_bi"]


def _load_mark(name):
    """per-chrom index: (sorted starts, prefix-max of ends) for O(log n) overlap tests."""
    by = {}
    with gzip.open(M / f"{name}.bed.gz", "rt") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            p = line.split("\t")
            if len(p) < 3:
                continue
            by.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    idx = {}
    for c, ivs in by.items():
        ivs.sort()
        starts = [a for a, _ in ivs]
        pmax = [0] * (len(ivs) + 1)
        m = 0
        for i, (_, e) in enumerate(ivs):
            m = e if e > m else m
            pmax[i + 1] = m
        idx[c] = (starts, pmax)
    return idx


def _overlaps(idx, chrom, s, e):
    v = idx.get(chrom)
    if not v:
        return False
    starts, pmax = v
    j = bisect.bisect_right(starts, e)          # intervals with start <= e
    return j > 0 and pmax[j] >= s               # any of them ends >= s


def _elements():
    """collapse CRISPR pairs to unique elements: regulator = Significant for >=1 gene (valid connections only)."""
    rows = open(OUT / "crispr_egpairs.tsv").read().splitlines()
    hdr = rows[0].split("\t")
    ci = {h: i for i, h in enumerate(hdr)}
    el = {}
    for line in rows[1:]:
        p = line.split("\t")
        if p[ci["ValidConnection"]] != "TRUE":
            continue
        chrom, s, e = p[ci["chrom"]], int(p[ci["chromStart"]]), int(p[ci["chromEnd"]])
        sig = p[ci["Significant"]] == "TRUE"
        key = (chrom, s, e)
        if key not in el:
            el[key] = {"chrom": chrom, "start": s, "end": e, "sig": False, "n_pairs": 0}
        el[key]["sig"] = el[key]["sig"] or sig
        el[key]["n_pairs"] += 1
    return list(el.values())


def run():
    idx = {m: _load_mark(m) for m in MARKS}
    els = _elements()
    for el in els:
        for m in MARKS:
            el[m] = _overlaps(idx[m], el["chrom"], el["start"], el["end"])
    N = len(els)
    base = sum(e["sig"] for e in els) / N

    def prec(sub):
        sub = [e for e in els if sub(e)]
        if not sub:
            return (0, 0.0)
        return (len(sub), sum(e["sig"] for e in sub) / len(sub))

    # single-mark discrimination
    marks = {}
    for m in MARKS:
        n_pos, p_pos = prec(lambda e, m=m: e[m])
        n_neg, p_neg = prec(lambda e, m=m: not e[m])
        marks[m] = {"n_pos": n_pos, "prec_pos": round(p_pos, 3), "n_neg": n_neg, "prec_neg": round(p_neg, 3),
                    "enrichment": round(p_pos / base, 2) if base else None,
                    "prec_pos_vs_neg": round(p_pos / p_neg, 2) if p_neg else None}

    # the productivity ladder: open -> active -> +PolII -> +eRNA
    ladder = []
    for label, cond in [("all tested elements", lambda e: True),
                        ("open (DNase)", lambda e: e["dnase"]),
                        ("+ active (H3K27ac)", lambda e: e["dnase"] and e["h3k27ac"]),
                        ("+ Pol II (POLR2A)", lambda e: e["dnase"] and e["h3k27ac"] and e["polr2a"]),
                        ("+ eRNA (PRO-cap bidir)", lambda e: e["dnase"] and e["h3k27ac"] and e["polr2a"] and e["procap_bi"])]:
        n, p = prec(cond)
        ladder.append({"stage": label, "n": n, "prec": round(p, 3), "enrichment": round(p / base, 2) if base else None})

    # the sharp test: among ACTIVE (open+H3K27ac) elements, does Pol II / eRNA add discrimination?
    active = [e for e in els if e["dnase"] and e["h3k27ac"]]
    def p_of(sub):
        sub = [e for e in active if sub(e)]
        return (len(sub), round(sum(x["sig"] for x in sub) / len(sub), 3) if sub else 0.0)
    among_active = {
        "n_active": len(active),
        "prec_active": round(sum(e["sig"] for e in active) / len(active), 3) if active else 0,
        "polII_pos": p_of(lambda e: e["polr2a"]), "polII_neg": p_of(lambda e: not e["polr2a"]),
        "eRNA_pos": p_of(lambda e: e["procap_bi"]), "eRNA_neg": p_of(lambda e: not e["procap_bi"]),
        "polII_and_eRNA": p_of(lambda e: e["polr2a"] and e["procap_bi"]),
        "neither": p_of(lambda e: not e["polr2a"] and not e["procap_bi"]),
    }
    return {"n_elements": N, "base_rate_regulator": round(base, 3), "marks": marks,
            "ladder": ladder, "among_active": among_active}


def _abc_index():
    """genome-wide ABC links indexed by element chrom: (elem_start, elem_end, gene_chrom, gene_start, gene_end)."""
    by = {}
    with gzip.open(M / "abc_all.bedpe.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            by.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5])))
    for c in by:
        by[c].sort()
    return by


def _abc_links_gene(abc, chrom, s, e, tss_chrom, tss, tol=2000):
    """does an ABC link connect element [s,e] on chrom to a gene whose region contains tss (coordinate-only match)?"""
    ivs = abc.get(chrom)
    if not ivs:
        return False
    lo = bisect.bisect_right([x[0] for x in ivs], e)
    i = lo - 1
    while i >= 0 and ivs[i][0] >= s - 5000:
        es, ee, gc, gs, ge = ivs[i]
        if ee >= s and es <= e and gc == tss_chrom and gs - tol <= tss <= ge + tol:
            return True
        i -= 1
    return False


def pair_test():
    """pair-level: the FULL chain — is a CRISPR element->gene pair regulating (Significant) when the element is PRODUCTIVE
    (Pol II / eRNA) AND 3D-linked (ABC) to that exact gene? Tests Gate3 (which gene) x Gate4 (productivity) together."""
    polr2a = _load_mark("polr2a"); procap = _load_mark("procap_bi"); h3 = _load_mark("h3k27ac")
    abc = _abc_index()
    rows = open(OUT / "crispr_egpairs.tsv").read().splitlines()
    hdr = rows[0].split("\t"); ci = {h: i for i, h in enumerate(hdr)}
    pairs = []
    for line in rows[1:]:
        p = line.split("\t")
        if p[ci["ValidConnection"]] != "TRUE":
            continue
        if p[ci["startTSS"]] in ("NA", ""):
            continue
        chrom, s, e = p[ci["chrom"]], int(p[ci["chromStart"]]), int(p[ci["chromEnd"]])
        tchr, tss = p[ci["chrTSS"]], int(p[ci["startTSS"]])
        prod = _overlaps(polr2a, chrom, s, e) or _overlaps(procap, chrom, s, e)
        linked = _abc_links_gene(abc, chrom, s, e, tchr, tss)
        pairs.append({"sig": p[ci["Significant"]] == "TRUE", "prod": prod, "abc": linked})
    N = len(pairs); base = sum(x["sig"] for x in pairs) / N

    def prec(f):
        sub = [x for x in pairs if f(x)]
        return (len(sub), round(sum(x["sig"] for x in sub) / len(sub), 3) if sub else 0.0)
    return {"n_pairs": N, "base_rate": round(base, 3),
            "productive": prec(lambda x: x["prod"]),
            "abc_linked": prec(lambda x: x["abc"]),
            "productive_AND_abc": prec(lambda x: x["prod"] and x["abc"]),
            "neither": prec(lambda x: not x["prod"] and not x["abc"])}


def main():
    print("=" * 98)
    print("REGULATE — does binding that RECRUITS POL II = regulation?  (K562 Pol II + eRNA vs CRISPR ground truth)")
    print("=" * 98)
    R = run()
    print(f"\n  CRISPR-tested elements: {R['n_elements']}   base rate of a true regulator: {R['base_rate_regulator']:.1%}")
    print("\n  single-mark discrimination (precision that an element is a real CRISPR regulator):")
    print(f"    {'mark':16s} {'n+':>6s} {'P(reg|+)':>9s} {'P(reg|-)':>9s} {'enrich':>7s} {'+ vs -':>7s}")
    lab = {"dnase": "DNase (open)", "h3k27ac": "H3K27ac(active)", "polr2a": "Pol II (POLR2A)",
           "polr2s5": "Pol II-Ser5P", "procap_bi": "eRNA (PRO-cap)"}
    for m in MARKS:
        d = R["marks"][m]
        print(f"    {lab[m]:16s} {d['n_pos']:6d} {d['prec_pos']:9.3f} {d['prec_neg']:9.3f} {str(d['enrichment'])+'x':>7s} {str(d['prec_pos_vs_neg'])+'x':>7s}")
    print("\n  the productivity ladder (add one requirement at a time):")
    for s in R["ladder"]:
        print(f"    {s['stage']:26s} n={s['n']:5d}  precision {s['prec']:.3f}  ({s['enrichment']}x base)")
    a = R["among_active"]
    print(f"\n  SHARP TEST — among {a['n_active']} open+active elements (precision {a['prec_active']:.3f}), does the ACT of")
    print("  transcription add regulatory information beyond the H3K27ac mark?")
    print(f"    Pol II present : n={a['polII_pos'][0]:4d} precision {a['polII_pos'][1]:.3f}   |  absent : n={a['polII_neg'][0]:4d} precision {a['polII_neg'][1]:.3f}")
    print(f"    eRNA firing    : n={a['eRNA_pos'][0]:4d} precision {a['eRNA_pos'][1]:.3f}   |  silent : n={a['eRNA_neg'][0]:4d} precision {a['eRNA_neg'][1]:.3f}")
    print(f"    Pol II & eRNA  : n={a['polII_and_eRNA'][0]:4d} precision {a['polII_and_eRNA'][1]:.3f}   |  neither: n={a['neither'][0]:4d} precision {a['neither'][1]:.3f}")
    print("\n  FULL CHAIN (pair-level) — is an element->gene pair regulating when the element is PRODUCTIVE (Pol II/eRNA)")
    print("  AND ABC 3D-links to THAT gene? (Gate3 which-gene x Gate4 productivity, vs CRISPR Significant):")
    PT = pair_test()
    print(f"    {PT['n_pairs']} valid pairs, base rate Significant {PT['base_rate']:.1%}")
    print(f"    productive element ......... n={PT['productive'][0]:5d}  precision {PT['productive'][1]:.3f}")
    print(f"    ABC-linked to the gene ..... n={PT['abc_linked'][0]:5d}  precision {PT['abc_linked'][1]:.3f}")
    print(f"    productive AND ABC-linked .. n={PT['productive_AND_abc'][0]:5d}  precision {PT['productive_AND_abc'][1]:.3f}"
          f"  ({round(PT['productive_AND_abc'][1]/PT['base_rate'],1)}x base)")
    print(f"    neither .................... n={PT['neither'][0]:5d}  precision {PT['neither'][1]:.3f}")
    out = {"result": R, "pair_test": PT,
           "note": "Tests 'binding=regulation only if it recruits polymerase' on real K562 data vs CRISPR ground truth "
                   "(ENCFF968BZL, harmonised Nasser/Gasperini/Schraivogel enhancer-perturbation; Significant=element truly "
                   "regulates a gene). Productivity marks: POLR2A + POLR2AphosphoS5 (Pol II recruited/initiating), PRO-cap "
                   "bidirectional (eRNA firing). Measures the MARGINAL value of the productivity signal on accessibility-"
                   "preselected elements. HONEST: ChIP occupancy proxies residence-time x conc, not single-molecule dwell "
                   "time (no genome-wide data). Real data; the answer is measured, not asserted."}
    json.dump(out, open("outputs/orphan/regulate.json", "w"), indent=1)
    print("\n  -> outputs/orphan/regulate.json")
    return out


if __name__ == "__main__":
    main()
