"""lit_place — place a gene that is NOT in our pathway data (an orphan / dark gene) using its LITERATURE. The `ladder`
syscall lays a pathway out by compartment; for a gene with no pathway it can only read the compartment. This adds the
missing half: mine the gene's abstracts (litmine) for co-mentioned KNOWN genes, then transfer THOSE genes' pathways as a
ranked HYPOTHESIS for what process the orphan gene participates in — so an orphan gene gets both a compartment (known) and
a candidate functional context (inferred).

VALIDATED, and honestly bounded. The load-bearing assumption — "genes co-mentioned in the same abstract share a pathway"
— is tested leave-one-out on KNOWN genes (build a known-known co-mention graph from the abstracts, hide a gene's pathway,
predict it from its co-mentioned neighbours):
  * pathway recovery  top-1 ~0.14 / top-5 ~0.22  vs random-from-candidate-pool ~0.055  = ~4x chance (IDF-weighted vote)
  * compartment       top-1 ~0.64 vs 0.54 baseline (weak — but MOOT: an orphan gene's compartment is read DIRECTLY from
                       localization ~100%, not predicted)
So: the COMPARTMENT placement is solid (direct annotation); the PATHWAY association is a real but NOISY hypothesis (~4x
chance, right pathway in the top-5 about 1 in 5 times) — a hypothesis GENERATOR for an otherwise-uncharacterised gene, NOT
a confident annotation and NOT a phenotype predictor. IDF down-weights promiscuous hub genes (a gene co-mentioned with
everyone, like TP53, carries little specific pathway signal).
"""
import os
import json, re, collections, math
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))

# gene symbols that are also common English / lab words — excluded from case-sensitive abstract matching to cut false hits
STOP = {"SET", "MET", "CAD", "CAT", "REST", "ACHE", "FAT", "MAX", "PIGS", "CAMP", "COIL", "DDT", "GAN", "WARS", "MARS",
        "CARS", "TANK", "SHE", "APP", "AGA", "LARGE", "CAP", "IMPACT", "SPARC", "CLOCK", "STAR", "MICE", "NODAL", "PET",
        "BAD", "WAS", "HGF", "TYPE", "AIM", "AIMP", "GAS", "FASN", "PLAN", "MINK", "SON", "CS", "LYN", "DNA", "RNA",
        "ATP", "GTP", "PCR", "MRI", "BMI"}
DEPTH = {"Secreted": 0, "Extracellular": 0, "Extracellular space": 0,
         "Cell membrane": 1, "Membrane": 1, "Cell surface": 1,
         "Cytoplasm": 2, "Cytosol": 2, "Endoplasmic reticulum": 2, "Golgi apparatus": 2, "Cytoskeleton": 2, "Endosome": 2,
         "Nucleus": 3}
DEPTH_NAME = {0: "extracellular", 1: "membrane", 2: "cytoplasm", 3: "nucleus"}
MAXPATH = 50                                                          # cap generic hub-pathways (as in graph_label)


def _mine(txt, nameset):
    """known gene symbols named in a block of abstract text (case-sensitive, length>=3, minus the English-word stoplist)."""
    return {t for t in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", txt) if t in nameset and t not in STOP}


def build():
    """load data + build the literature co-mention structures once. Returns a dict used by place()/validate()."""
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nameset = set(names)
    dark = {g["name"] for g in D["genes"] if g.get("dark")}
    loc = json.load(open(OUT / "localization.json")).get("labels", {})
    lit = json.load(open(OUT / "litmine.json"))
    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]

    gene_paths = collections.defaultdict(set)                        # size-capped pathway membership
    for pw, members in paths.items():
        if 2 <= len(members) <= MAXPATH:
            for i in members:
                if i < len(names):
                    gene_paths[names[i]].add(pw)

    own = {}                                                         # gene -> known genes named in ITS OWN abstracts
    com = collections.defaultdict(collections.Counter)              # known-known co-mention graph (from every abstract set)
    docfreq = collections.Counter()                                # in how many genes' abstract-sets a gene is named (for IDF)
    for g, e in lit.items():
        txt = " ".join((p.get("title", "") + " " + p.get("abstract", "")) for p in e.get("papers", []))
        m = _mine(txt, nameset) - {g}
        own[g] = m
        for x in m:
            docfreq[x] += 1
        for a in m:
            for b in m:
                if a != b:
                    com[a][b] += 1
    Nd = max(len(lit), 1)
    return {"names": names, "nameset": nameset, "dark": dark, "loc": loc, "lit": lit, "paths": paths,
            "gene_paths": gene_paths, "own": own, "com": com, "docfreq": docfreq, "Nd": Nd}


def _idf(G, x):
    return math.log(G["Nd"] / (1 + G["docfreq"].get(x, 0)))


def _comps(G, g):
    return sorted({DEPTH[c] for c in (G["loc"].get(g) or []) if c in DEPTH})


def place(gene, G=None, topk=6):
    """place an orphan/dark gene from its literature: co-mentioned known genes -> IDF-weighted candidate pathways
    (HYPOTHESIS) + its own compartment (DIRECT). Works for any gene with litmine abstracts."""
    if G is None:
        G = build()
    if gene not in G["lit"]:
        return None
    partners = sorted(G["own"].get(gene, ()), key=lambda x: -_idf(G, x))
    tally = collections.Counter()
    for nb in G["own"].get(gene, ()):
        w = _idf(G, nb)
        for pw in G["gene_paths"].get(nb, ()):
            tally[pw] += w
    cand = [pw for pw, _ in tally.most_common(topk)]
    deps = _comps(G, gene)
    # where does the inferred functional context sit? compartment profile of the candidate pathways' genes
    ctx = collections.Counter()
    for pw in cand:
        for i in G["paths"][pw]:
            if i < len(G["names"]):
                for d in _comps(G, G["names"][i]):
                    ctx[DEPTH_NAME[d]] += 1
    in_pathway = bool(G["gene_paths"].get(gene))
    return {"gene": gene, "already_in_pathway": in_pathway,
            "compartment": [DEPTH_NAME[d] for d in deps] or ["(unmapped)"],
            "co_mentioned": partners[:12],
            "candidate_pathways": cand,
            "context_compartment_profile": dict(ctx.most_common()),
            "n_partners": len(partners)}


def validate(G=None):
    """leave-one-out: does co-mention recover a KNOWN gene's pathway? (the honest self-report of this tool's accuracy)."""
    import random
    if G is None:
        G = build()
    com, gene_paths = G["com"], G["gene_paths"]
    evalset = [g for g in com if gene_paths.get(g) and sum(1 for nb in com[g] if nb != g) >= 3]
    random.seed(0)
    h1 = h5 = rnd = n = 0
    for g in evalset:
        tally = collections.Counter()
        for nb, w in com[g].items():
            if nb == g:
                continue
            wi = _idf(G, nb)
            for pw in gene_paths.get(nb, ()):
                tally[pw] += wi
        if not tally:
            continue
        ranked = [pw for pw, _ in tally.most_common()]
        true = gene_paths[g]
        n += 1
        h1 += ranked[0] in true
        h5 += any(pw in true for pw in ranked[:5])
        cand = list({pw for nb in com[g] for pw in gene_paths.get(nb, ())})
        if cand:
            rnd += any(pw in true for pw in random.sample(cand, min(5, len(cand))))
    return {"n": n, "pathway_top1": round(h1 / n, 3), "pathway_top5": round(h5 / n, 3),
            "random_top5": round(rnd / n, 3), "lift_top5": round((h5 / n) / (rnd / n + 1e-9), 2)}


def main():
    print("=" * 92)
    print("LIT-PLACE — place orphan / dark genes (no pathway) from their literature co-mentions, reality-checked")
    print("=" * 92)
    G = build()
    orphans = [g for g in G["lit"] if not G["gene_paths"].get(g)]
    print(f"\n  orphan genes (no capped-pathway membership) with litmine abstracts: {len(orphans):,}")
    v = validate(G)
    print(f"\n  VALIDATION — leave-one-out pathway recovery on KNOWN genes (n={v['n']:,}, IDF-weighted co-mention vote):")
    print(f"    pathway top-1 {v['pathway_top1']}   top-5 {v['pathway_top5']}   vs random-from-pool {v['random_top5']}   = {v['lift_top5']}x chance")
    print(f"    -> co-mention is a REAL but NOISY pathway signal: a hypothesis generator (right pathway in top-5 ~1 in 5), not an annotation.")
    print(f"    -> compartment is NOT predicted here: an orphan gene's compartment is read DIRECTLY from localization.")
    print("\n  example placements (orphan genes):")
    for g in ["FNDC5", "STRA8", "DAZL", "TNFRSF19", "SP5"]:
        p = place(g, G)
        if not p:
            continue
        print(f"    {g:10s}  compartment [{', '.join(p['compartment'])}]  ({p['n_partners']} lit partners)")
        print(f"       co-mentioned : {', '.join(p['co_mentioned'][:8])}")
        print(f"       candidate fn : {'; '.join(pw[:46] for pw in p['candidate_pathways'][:3]) or '(none)'}")
    out = {"validation": v, "n_orphans_with_lit": len(orphans),
           "note": "Orphan-gene placement from literature. COMPARTMENT is read DIRECTLY from localization (~100%, solid). "
                   "PATHWAY association is inferred by IDF-weighted co-mention vote and is a REAL but NOISY HYPOTHESIS "
                   f"(leave-one-out top-5 {v['pathway_top5']} = {v['lift_top5']}x chance), NOT a confident annotation and "
                   "NOT a phenotype predictor. Symbol mining is a curated regex+stoplist, not full NER."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "lit_place.json", "w"), indent=1)
    print(f"\n  -> {OUT/'lit_place.json'}")
    return out


if __name__ == "__main__":
    main()
