"""connect — CONNECT the three orphan-gene signals into one mechanistic hypothesis card, and cross-check them against
each other. lit_place gave literature (candidate pathway) + compartment; this adds the PPI layer and, crucially, uses
AGREEMENT BETWEEN SOURCES as the confidence signal:

  compartment   (localization)      DIRECT annotation, solid
  literature    (litmine co-mention) a NOISY pathway hypothesis (top-1 ~0.13)
  PPI partners  (cell_complete ppi)  measured physical edges -> partners' pathways are a STRONGER vote (top-1 ~0.54)
  CORROBORATED  (lit AND ppi agree)  the HIGH-CONFIDENCE tier: top-1 ~0.71, 5.7x lit alone, at ~18% coverage
  transport     (proc=transport)     a HEURISTIC: when the gene and its inferred reaction sit in DIFFERENT compartments,
                                      flag a transport/trafficking partner that could bridge the gap (a lead, NOT validated)

Why this is the right way to connect (all MEASURED here, held-out on KNOWN genes):
  * PPI-partner pathway vote        top-1 precision 0.540
  * lit-only pathway vote           top-1 precision 0.125
  * CORROBORATED (both agree)       top-1 precision 0.706  (fires for ~18% of genes) = cross-source agreement is the win
  * PPI partners co-localize        share >=1 compartment 0.927 vs 0.507 random = 1.83x  -> the spatial cross-check and
                                      the transport step (which fires on the ~7% that DON'T co-localize) are well-founded
HONEST: still a DESCRIPTIVE near-field triangulation. The corroborated pathway is a strong LEAD (71% precision), not a
proven annotation; the transport carrier is a flagged hypothesis; none of this is a phenotype predictor. It tells you the
most-likely process + physical context + where it happens + what might move it — a starting mechanism to test.
"""
import json, collections, math
from pathlib import Path
import lit_place as lp
OUT = Path("outputs/orphan")
TRANSPORT_PROC = {"transport/uptake", "trafficking/secretion"}


def build():
    G = lp.build()
    D = json.load(open(OUT / "cell_complete.json"))
    names = G["names"]
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if a < len(names) and b < len(names) and a != b:
            ppi[names[a]].add(names[b]); ppi[names[b]].add(names[a])
    proc = {g["name"]: g.get("proc", "") for g in D["genes"]}
    G.update({"ppi": ppi, "proc": proc})
    return G


def _lit_vote(G, gene, known=False):
    """candidate pathways from literature co-mention (IDF-weighted). For an orphan use its OWN abstracts; for a KNOWN
    gene (validation) use the co-mention graph, since known genes aren't in litmine."""
    src = G["com"].get(gene, {}) if known else {n: 1 for n in G["own"].get(gene, ())}
    t = collections.Counter()
    for nb in src:
        if nb == gene:
            continue
        w = lp._idf(G, nb)
        for pw in G["gene_paths"].get(nb, ()):
            t[pw] += w
    return t


def _ppi_vote(G, gene):
    """candidate pathways from PHYSICAL partners' pathway membership, hub-partners down-weighted."""
    t = collections.Counter()
    for p in G["ppi"].get(gene, ()):
        if p == gene:
            continue
        deg = len(G["ppi"].get(p, ()))
        for pw in G["gene_paths"].get(p, ()):
            t[pw] += 1.0 / math.log(2 + deg)
    return t


def _corroborate(lv, pv, k=5):
    """pathways present in BOTH the lit and ppi top-k -> the high-confidence tier, ranked by combined normalised score."""
    lset, pset = {p for p, _ in lv.most_common(k)}, {p for p, _ in pv.most_common(k)}
    both = lset & pset
    lmax, pmax = (max(lv.values()) if lv else 1), (max(pv.values()) if pv else 1)
    return sorted(both, key=lambda pw: lv[pw] / lmax + pv[pw] / pmax, reverse=True)


def _depset(G, g):
    return {lp.DEPTH[c] for c in (G["loc"].get(g) or []) if c in lp.DEPTH}


def connect(gene, G=None):
    """the mechanistic hypothesis card for a gene: compartment (direct) + literature + PPI + their corroboration + a
    transport lead where compartments disagree."""
    if G is None:
        G = build()
    if gene not in G["names"]:
        return None
    known = bool(G["gene_paths"].get(gene))
    lv = _lit_vote(G, gene, known=False)
    pv = _ppi_vote(G, gene)
    cor = _corroborate(lv, pv)
    my = _depset(G, gene)
    partners = sorted(G["ppi"].get(gene, ()), key=lambda p: -len(G["ppi"].get(p, ())))
    # partners that sit in a corroborated (or top-ppi) pathway = the physical context for that process
    ctx_pw = cor[0] if cor else (pv.most_common(1)[0][0] if pv else None)
    ctx_partners = [p for p in partners if ctx_pw and ctx_pw in G["gene_paths"].get(p, ())]
    # where does the inferred reaction happen? compartments of the context partners
    ctx_deps = set().union(*[_depset(G, p) for p in ctx_partners]) if ctx_partners else set()
    span = ctx_deps - my                                               # compartments the reaction needs but the gene isn't in
    # transport lead: a transport/trafficking partner whose compartments bridge my-compartment and the reaction compartment
    carriers = []
    if span:
        for p in partners:
            if G["proc"].get(p) in TRANSPORT_PROC:
                pd = _depset(G, p)
                if pd & my and pd & ctx_deps:                          # the carrier touches both sides of the gap
                    carriers.append(p)
    return {"gene": gene, "already_annotated": known,
            "compartment": sorted(lp.DEPTH_NAME[d] for d in my) or ["(unmapped)"],
            "n_ppi": len(partners), "top_partners": partners[:10],
            "lit_pathways": [p for p, _ in lv.most_common(5)],
            "ppi_pathways": [p for p, _ in pv.most_common(5)],
            "corroborated": cor[:5],
            "context_pathway": ctx_pw,
            "context_partners": ctx_partners[:8],
            "reaction_compartments": sorted(lp.DEPTH_NAME[d] for d in ctx_deps),
            "compartment_gap": sorted(lp.DEPTH_NAME[d] for d in span),
            "transport_leads": carriers[:5]}


def validate(G=None):
    """held-out on KNOWN genes: precision of lit-only vs ppi-only vs corroborated, + PPI spatial consistency."""
    import random
    if G is None:
        G = build()
    gp = G["gene_paths"]
    evalset = [g for g in G["com"] if gp.get(g) and sum(1 for nb in G["com"][g] if nb != g) >= 3 and len(G["ppi"].get(g, ())) >= 1]
    lh = ln = ph = pn = ch = cn = 0
    for g in evalset:
        true = gp[g]
        lv = _lit_vote(G, g, known=True); pv = _ppi_vote(G, g)
        if lv:
            ln += 1; lh += lv.most_common(1)[0][0] in true
        if pv:
            pn += 1; ph += pv.most_common(1)[0][0] in true
        cor = _corroborate(lv, pv)
        if cor:
            cn += 1; ch += cor[0] in true
    # spatial consistency
    random.seed(0)
    D = json.load(open(OUT / "cell_complete.json")); names = G["names"]
    allg = [g for g in names if _depset(G, g)]
    sh = st = 0
    for a, b in D["ppi"][:60000]:
        if a < len(names) and b < len(names):
            ca, cb = _depset(G, names[a]), _depset(G, names[b])
            if ca and cb:
                st += 1; sh += bool(ca & cb)
    rsh = rt = 0
    for _ in range(st):
        ca, cb = _depset(G, random.choice(allg)), _depset(G, random.choice(allg))
        if ca and cb:
            rt += 1; rsh += bool(ca & cb)
    return {"n": len(evalset),
            "lit_only_top1": round(lh / ln, 3), "ppi_only_top1": round(ph / pn, 3),
            "corroborated_top1": round(ch / cn, 3), "corroborated_coverage": round(cn / len(evalset), 3),
            "corroboration_lift": round((ch / cn) / (lh / ln), 2),
            "ppi_colocalize": round(sh / st, 3), "random_colocalize": round(rsh / rt, 3),
            "coloc_enrichment": round((sh / st) / (rsh / rt), 2)}


def render(c):
    if not c:
        return "connect: no such gene."
    L = [f"connect — {c['gene']}  [{', '.join(c['compartment'])}]"
         + ("  (already annotated)" if c["already_annotated"] else "  (orphan — no pathway)"),
         f"  physical partners ({c['n_ppi']}): {', '.join(c['top_partners'][:8])}"]
    if c["corroborated"]:
        L.append(f"  CORROBORATED pathway (lit AND a physical partner agree — ~71% precision):")
        for pw in c["corroborated"][:3]:
            L.append(f"    ✓ {pw}")
    else:
        L.append("  no lit/ppi corroboration; best physical-partner pathway (~54% precision):")
        for pw in c["ppi_pathways"][:2]:
            L.append(f"    · {pw}")
    if c["lit_pathways"]:
        L.append(f"  literature-only leads (hypothesis, noisy): {'; '.join(p[:34] for p in c['lit_pathways'][:2])}")
    if c["context_pathway"]:
        L.append(f"  reaction context '{c['context_pathway'][:44]}' via partners {', '.join(c['context_partners'][:5])}")
        L.append(f"    happens in [{', '.join(c['reaction_compartments'])}]; gene is in [{', '.join(c['compartment'])}]")
    if c["compartment_gap"]:
        L.append(f"  ⚠ compartment gap: reaction needs [{', '.join(c['compartment_gap'])}] the gene isn't in.")
        if c["transport_leads"]:
            L.append(f"    transport LEAD (heuristic): {', '.join(c['transport_leads'])} could bridge it (transport/trafficking partner).")
        else:
            L.append("    no transport-annotated partner bridges the gap (no carrier lead).")
    return "\n".join(L)


def main():
    print("=" * 96)
    print("CONNECT — triangulate literature + PPI + spatial into one mechanistic hypothesis, cross-checked")
    print("=" * 96)
    G = build()
    v = validate(G)
    print(f"\n  VALIDATION (held-out, KNOWN genes, n={v['n']:,}):")
    print(f"    pathway top-1 precision:  lit-only {v['lit_only_top1']}   ppi-only {v['ppi_only_top1']}   "
          f"CORROBORATED {v['corroborated_top1']} (fires {v['corroborated_coverage']:.0%})")
    print(f"    -> corroboration (lit AND ppi agree) is {v['corroboration_lift']}x lit-only precision = cross-source agreement is the signal")
    print(f"    spatial: PPI partners co-localize {v['ppi_colocalize']} vs {v['random_colocalize']} random = {v['coloc_enrichment']}x "
          f"-> location cross-check + transport step are grounded")
    print("\n  example mechanistic cards (orphan genes):")
    for g in ["UNC13C", "WDHD1", "CSE1L", "STRA8"]:      # corroboration; corroboration; transport-carrier; lit-only
        print("\n  " + render(connect(g, G)).replace("\n", "\n  "))
    out = {"validation": v,
           "note": "Triangulation of literature (noisy pathway hypothesis) + PPI (stronger partner-pathway vote) + spatial "
                   "(direct compartment). The WIN is cross-source AGREEMENT: corroborated (lit AND ppi) pathway top-1 "
                   f"{v['corroborated_top1']} vs lit-only {v['lit_only_top1']} = {v['corroboration_lift']}x, at "
                   f"{v['corroborated_coverage']:.0%} coverage. PPI partners co-localize {v['ppi_colocalize']} vs "
                   f"{v['random_colocalize']} random ({v['coloc_enrichment']}x), grounding the location cross-check; the "
                   "transport carrier is a flagged HEURISTIC lead on the ~7% that don't co-localize. Descriptive near-field "
                   "triangulation -- a strong starting mechanism (corroborated ~71% precision), NOT a proven annotation or "
                   "a phenotype predictor."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "connect.json", "w"), indent=1)
    print(f"\n  -> {OUT/'connect.json'}")
    return out


if __name__ == "__main__":
    main()
