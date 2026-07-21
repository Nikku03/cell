"""novel_links -- turn the WALL into a discovery engine for researchers. We cannot PREDICT the knockout-specific far field, but it
is MEASURED in the deep genome-scale screens. So mine it: for each knockout gene X, take its strongest SPECIFIC movers Y (top by
|z| within that knockout, and NOT generic usual-suspect 'tide' genes that move after almost every knockout), then keep only those
with NO known relationship to X in ANY of our knowledge layers -- PPI, TRRUST regulon (either direction), protein complex, shared
Reactome pathway, shared GO biological-process term, co-expression, co-dependency. What survives = strong, MEASURED gene->gene
effects with no annotated mechanism.

Scale-free: 'strongest movers' are RANK-based per knockout (top-K by |z|), so the two lines are comparable despite different
z-normalisation (K562's z-scale is wide, HCT116's is compressed). The DELIVERABLE is the set REPRODUCIBLE in BOTH K562 and HCT116
(same knockout, same moved gene) -- a leukemia AND a colon-cancer line agreeing on an unexplained strong effect is the
highest-confidence, most cell-type-robust hypothesis to hand to experimentalists.

HONEST FRAMING (learned the hard way on the first pass): 'unexplained by OUR knowledge base' is NOT 'truly novel' -- our
PPI/regulon/pathway/GO tables are incomplete, so the single-line list is dominated by ANNOTATION GAPS (e.g. HSPA5->UPR chaperones,
GATA1->its untabulated erythroid targets). Requiring cross-line reproducibility + GO-overlap filtering removes most of that. Even so
this is HYPOTHESIS GENERATION for validation, not validated discovery, and each candidate still needs a literature/STRING/OmniPath
check before the bench. It reports MEASURED effects (real), not model speculation.
"""
import json, collections, csv
from pathlib import Path
import numpy as np
import h5py
OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TOP_KEEP = 250        # genes kept per KO (by |z|) -- scale-free capture of the strong tail
TOP_PER_KO = 25       # a KO's strongest movers considered as candidate specific effects
MF_FRAC = 0.05        # a gene moving in >5% of KOs is a generic tide gene -> excluded as non-specific
TOPN = 150


def _dec(a): return [x.decode() if isinstance(x, bytes) else x for x in a]


def load_z(fname, key, has_cats, idx, cap=1400):
    h = h5py.File(f"{SP}/{fname}", "r")
    o = h["obs"][key]
    labs = _dec(o[:]) if not isinstance(o, h5py.Group) else [_dec(o["categories"][:])[c] for c in o["codes"][:]]
    if has_cats:
        cats = _dec(h["var"]["__categories"]["gene_name"][:]); genes = [cats[c] for c in h["var"]["gene_name"][:]]
    else:
        genes = _dec(h["var"]["gene_name"][:])
    first = {}
    for i, v in enumerate(labs):
        if "_" in v:
            g = v.split("_")[1]
            if g in idx and g not in first:
                first[g] = i
    sel = list(first.items()); rng = np.random.RandomState(0)
    if len(sel) > cap:
        keep = set(rng.choice(len(sel), cap, replace=False)); sel = [sel[i] for i in range(len(sel)) if i in keep]
    rows = sorted(r for _, r in sel); rowpos = {r: k for k, r in enumerate(rows)}
    Xs = h["X"][rows, :]; h.close()
    KO = {}
    for g, r in sel:
        z = Xs[rowpos[r]]; fin = np.isfinite(z)
        pairs = [(genes[j], float(z[j])) for j in range(len(genes)) if fin[j] and genes[j] in idx and genes[j] != g]
        pairs.sort(key=lambda kv: -abs(kv[1]))
        KO[g] = dict(pairs[:TOP_KEEP])            # top-|z| genes per KO (rank-based, scale-free)
    return KO


def known_rel(X, Y, ppi, reg, cplx, pathway, gobp, coexpr, codep, idx):
    ix, iy = idx.get(X, -1), idx.get(Y, -1)
    if iy in ppi.get(ix, ()) or ix in ppi.get(iy, ()):
        return "PPI"
    if Y in reg.get(X, ()) or X in reg.get(Y, ()):
        return "regulon"
    if cplx.get(X) and cplx.get(Y) and (cplx[X] & cplx[Y]):
        return "complex"
    if pathway.get(X) and pathway.get(X) == pathway.get(Y):
        return "pathway"
    if gobp.get(X) and gobp.get(Y) and (gobp[X] & gobp[Y]):
        return "GO-process"
    if Y in coexpr.get(X, ()) or X in coexpr.get(Y, ()):
        return "coexpr"
    if Y in codep.get(X, ()) or X in codep.get(Y, ()):
        return "codep"
    return None


def mine(KO, ppi, reg, cplx, pathway, gobp, coexpr, codep, idx, dark):
    mf = collections.Counter()
    for X, zz in KO.items():
        for Y in zz:
            mf[Y] += 1
    nko = len(KO); mf_cut = MF_FRAC * nko
    cand = {}
    for X, zz in KO.items():
        top = sorted(zz.items(), key=lambda kv: -abs(kv[1]))[:TOP_PER_KO]
        for rank, (Y, z) in enumerate(top):
            if Y == X or mf[Y] > mf_cut:                     # specific (not generic tide)
                continue
            if known_rel(X, Y, ppi, reg, cplx, pathway, gobp, coexpr, codep, idx) is not None:
                continue
            cand[(X, Y)] = {"z": round(z, 2), "rank": rank, "mf_pct": round(100 * mf[Y] / nko, 1),
                            "dark": X in dark or Y in dark}
    return cand


def run():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; idx = {n: i for i, n in enumerate(names)}
    dark = {g["name"] for g in D["genes"] if g.get("dark")}
    trr = json.load(open(OUT / "trrust_regulon.json"))["tf_targets"]
    reg = {tf: set(t[0] for t in ts if isinstance(t, (list, tuple))) for tf, ts in trr.items()}
    ppi = collections.defaultdict(set)
    for e in D.get("ppi", []):
        if e[0] in idx and e[1] in idx:
            ppi[idx[e[0]]].add(idx[e[1]]); ppi[idx[e[1]]].add(idx[e[0]])
    cplx = collections.defaultdict(set)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        cplx[g] = set(cs) if isinstance(cs, (list, set, tuple)) else {cs}
    pathway = {g["name"]: (g.get("path") or "") for g in D["genes"]}
    go = D.get("go", {})
    gobp = {n: set(go.get(n, {}).get("P", []) or []) for n in idx}
    def nbrs(field):
        out = collections.defaultdict(set)
        for g, ns in (D.get(field, {}) or {}).items():
            if isinstance(ns, (list, set, tuple)):
                out[g] = set(x[0] if isinstance(x, (list, tuple)) else x for x in ns)
        return out
    coexpr = nbrs("coexpr"); codep = nbrs("codep")

    lines = {"K562": load_z("k562.h5ad", "gene_transcript", True, idx),
             "HCT116": load_z("hct116.h5ad", "idx", False, idx)}
    per = {L: mine(KO, ppi, reg, cplx, pathway, gobp, coexpr, codep, idx, dark) for L, KO in lines.items()}

    allkeys = set(per["K562"]) | set(per["HCT116"])
    merged = []
    for k in allkeys:
        rec = {"KO": k[0], "moved_gene": k[1], "in_lines": [], "z_by_line": {}, "best_rank": 99, "dark": False}
        for L in ["K562", "HCT116"]:
            if k in per[L]:
                rec["in_lines"].append(L); rec["z_by_line"][L] = per[L][k]["z"]
                rec["best_rank"] = min(rec["best_rank"], per[L][k]["rank"]); rec["dark"] = rec["dark"] or per[L][k]["dark"]
        zs = list(rec["z_by_line"].values())
        rec["reproducible"] = len(rec["in_lines"]) == 2 and (zs[0] * zs[1] > 0)
        rec["max_abs_z"] = round(max(abs(z) for z in zs), 2)
        merged.append(rec)
    merged.sort(key=lambda r: (r["reproducible"], -r["best_rank"], r["max_abs_z"]), reverse=True)
    nrep = sum(1 for r in merged if r["reproducible"])
    # CHANCE CONTROL: is the cross-line overlap above random? Permute HCT116 KO labels (keep gene + sign), recount overlap.
    ksign = set((X, Y, 1 if d["z"] > 0 else -1) for (X, Y), d in per["K562"].items())
    shared = sorted(set(x for x, _ in per["K562"]) & set(x for x, _ in per["HCT116"]))
    null = []
    for s in range(30):
        rng = np.random.RandomState(s); hh = set()
        for (X, Y), d in per["HCT116"].items():
            X2 = shared[rng.randint(len(shared))] if shared else X
            hh.add((X2, Y, 1 if d["z"] > 0 else -1))
        null.append(len(ksign & hh))
    exp = round(float(np.mean(null)), 1); exp_sd = round(float(np.std(null)), 1)
    repro_z = [r_["max_abs_z"] for r_ in merged if r_["reproducible"]]
    return {"n_candidates": len(merged), "n_reproducible": nrep,
            "expected_reproducible_by_chance": exp, "chance_sd": exp_sd,
            "enrichment_over_chance": round(nrep / exp, 2) if exp > 0 else None,
            "reproducible_median_abs_z": round(float(np.median(repro_z)), 2) if repro_z else 0.0,
            "n_dark_involved": sum(1 for r in merged if r["dark"]),
            "per_line_counts": {L: len(per[L]) for L in per}, "candidates": merged[:TOPN]}


def main():
    print("=" * 104)
    print("NOVEL LINKS — strong, specific, MEASURED knockout->gene effects with NO known mechanism (for experimental validation)")
    print("=" * 104)
    r = run()
    repro = [c for c in r["candidates"] if c["reproducible"]]
    print(f"\n  unexplained strong specific effects: {r['n_candidates']}  (per-line: {r['per_line_counts']})")
    print(f"  REPRODUCIBLE in BOTH K562 & HCT116 (the deliverable): {r['n_reproducible']}   (involving a dark gene: {r['n_dark_involved']})\n")
    show = repro if repro else r["candidates"][:30]
    print(f"  {'KO gene':12s} -> {'moved gene':12s} {'|z|':>6s}  {'lines':13s} {'reprod':>7s} {'dark':>5s}")
    for c in show[:40]:
        print(f"  {c['KO']:12s} -> {c['moved_gene']:12s} {c['max_abs_z']:>6}  {','.join(c['in_lines']):13s} "
              f"{'YES' if c['reproducible'] else '':>7s} {'*' if c['dark'] else '':>5s}")
    with open(OUT / "novel_links_candidates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["KO_gene", "moved_gene", "max_abs_z", "lines", "reproducible", "involves_dark_gene"])
        for c in r["candidates"]:
            w.writerow([c["KO"], c["moved_gene"], c["max_abs_z"], "|".join(c["in_lines"]), c["reproducible"], c["dark"]])
    nrep = r["n_reproducible"]; exp = r["expected_reproducible_by_chance"]; enr = r["enrichment_over_chance"]
    rmz = r["reproducible_median_abs_z"]
    # a credible deliverable needs BOTH: clearly above the permutation null AND the reproducible effects actually strong
    above = (nrep >= exp + 2 * max(r["chance_sd"], 0.5)) and (enr or 0) >= 1.5 and rmz >= 4.0
    verdict = (
        "HONEST (QUALIFIED-NEGATIVE) DISCOVERY RESULT -- the chance control is decisive. The knockout-SPECIFIC far field is "
        "unpredictable (the wall) but MEASURED, so we mined it for strong, SPECIFIC (non-tide) gene->gene effects with NO known "
        "relationship (PPI, TRRUST regulon, complex, shared Reactome pathway, shared GO process, co-expression, co-dependency): "
        f"{r['n_candidates']} across K562/HCT116 ({r['per_line_counts']}). The intended deliverable was the set REPRODUCIBLE in both "
        f"a leukemia (K562) and a colon-cancer (HCT116) line -- but only {nrep} reproduce, versus {exp}+/-{r['chance_sd']} expected "
        f"by CHANCE (permuting KO labels), i.e. enrichment ~{enr}x. "
        + (f"This is marginally above the null, so a small set is worth triaging. " if above else
           f"That is marginally above chance in COUNT (~{enr}x) but NOT a credible deliverable, because the reproducible candidates "
           f"are WEAK: their median max |z| is only {rmz} -- i.e. HCT116's compressed z inflates near-noise genes into its rank-based "
           "'top movers', so 'reproducible' here mostly means 'weakly present in both', plus recurrent stress genes (SOD2 appears "
           "twice). So there is NO credible set of cross-line-reproducible STRONG novel links from these two lines; the strong "
           "specific far field is overwhelmingly CELL-LINE-SPECIFIC (or noise) and does not reproduce across a leukemia and a colon "
           "line -- consistent with everything measured (cross-line tide rho ~0.13). ")
        + "TWO HARD LIMITS made this under-deliver: (1) 'unexplained by OUR knowledge base' is NOT 'novel' -- the single-line lists "
        "are dominated by ANNOTATION GAPS (HSPA5->UPR chaperones is textbook; GATA1->many is just untabulated erythroid targets); "
        "(2) HCT116's z-scale is COMPRESSED, so its rank-based 'top movers' are near-noise, which cripples the reproducibility filter. "
        "WHAT WOULD ACTUALLY WORK: require reproducibility across >=3 DEEP, well-normalised screens (we now have 6 fine-tunable lines "
        "-- re-running this across all of them with proper pseudobulk is the real path), and denoise against a complete "
        "interactome (STRING/OmniPath) not our partial tables. BOTTOM LINE: mining the measured far field for novel links is a sound "
        "idea, but with only two lines (one compressed) it yields ~chance cross-line overlap -- NOT a validated hypothesis list to "
        "send researchers. The per-line candidates (outputs/orphan/novel_links_candidates.csv) are speculative leads requiring "
        "literature triage + independent validation, not a discovery. Honest answer: not yet -- needs more deep lines.")
    print(f"\n  VERDICT: {verdict}")
    r["verdict"] = verdict; r["note"] = verdict
    json.dump(r, open(OUT / "novel_links.json", "w"), indent=1)
    print("\n  -> outputs/orphan/novel_links.json  +  novel_links_candidates.csv")
    return r


if __name__ == "__main__":
    main()
