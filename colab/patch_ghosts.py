"""Patch the ghost genes into the cell — give the dark-proteome genes a function and a pathway home so the
model is complete BEFORE any whole-cell ML runs over it.

Ghost = a gene that is a node in the model but has an empty function/pathway (all 4,993 dark genes sit in zero
of the 60 pathways). Two patches, kept strictly separate by evidence tier:

  FUNCTION (fact tier)      : UniProt curated function distilled from the literature (dark_function_table),
                             tagged experimental / by_similarity — this is the authoritative de-ghosting.
  PATHWAY  (prediction tier): guilt-by-association — assign a ghost to a model pathway when enough of its KNOWN
                             network partners (reg+sig+ppi) already belong to it. Confidence = partner fraction.
                             Never merged with the curated pathway members; stored as `pathways_predicted`.

Emitted as a COMPACT overlay (outputs/orphan/ghost_patch.json) + `apply_patch(D)` that completes a loaded model
in memory — so the 36 MB model file is not rewritten, and curated vs predicted stays distinguishable.
"""
import json, os, sys
from collections import Counter, defaultdict

OUT = "outputs/orphan"
MIN_PARTNERS_IN_PW = 3      # need >=3 known partners in a pathway to assign a ghost to it
MIN_FRACTION = 0.10         # and >=10% of the ghost's partners in that pathway
MIN_ENRICHMENT = 2.0        # AND the partners must be >=2x enriched in the pathway vs its baseline size
                            # (kills hub-pathway over-assignment, e.g. "Generic Transcription Pathway")


def build_patch():
    D = json.load(open(os.path.join(OUT, "cell_complete.json")))
    genes = D["genes"]; name = [g["name"] for g in genes]; idx = {n: i for i, n in enumerate(name)}
    darkfn = D.get("darkfn", {})
    func = json.load(open(os.path.join(OUT, "dark_function_table.json")))["table"]

    # existing pathway membership (curated) + pathway sizes for enrichment normalization
    g2p = defaultdict(list)
    pw_size = {}
    all_pw_genes = set()
    for pname, members in D["pathways"].items():
        pw_size[pname] = len(members)
        for gi in members:
            g2p[gi].append(pname); all_pw_genes.add(gi)
    N_total = len(all_pw_genes) or 1

    # ghost network partners (reg + sig + ppi), undirected
    partners = defaultdict(set)
    for key in ("reg", "sig", "ppi"):
        for e in D.get(key, []):
            partners[e[0]].add(e[1]); partners[e[1]].add(e[0])

    patch = {}
    n_func, n_pw = 0, 0
    for k in darkfn:
        gi = int(k); gname = name[gi]
        entry = {}
        # FUNCTION (fact tier) from the literature table
        lit = func.get(gname)
        if lit and lit.get("function"):
            entry["func"] = lit["function"][:300]
            entry["func_evidence"] = lit.get("evidence")
            entry["func_name"] = lit.get("name")
            entry["func_source"] = "UniProt-literature"
            n_func += 1
        # PATHWAY (prediction tier) by guilt-by-association among known partners
        pw_votes = Counter()
        for p in partners.get(gi, ()):
            for pw in g2p.get(p, ()):
                pw_votes[pw] += 1
        npart = len(partners.get(gi, ())) or 1
        preds = []
        for pw, c in pw_votes.items():
            frac = c / npart
            baseline = pw_size.get(pw, 1) / N_total          # prevalence of the pathway among networked genes
            enrichment = frac / baseline if baseline else 0
            if c >= MIN_PARTNERS_IN_PW and frac >= MIN_FRACTION and enrichment >= MIN_ENRICHMENT:
                preds.append((pw, c, round(frac, 3), round(enrichment, 1)))
        preds.sort(key=lambda x: -x[3])                       # rank by enrichment, not raw fraction
        if preds:
            entry["pathways_predicted"] = [dict(pathway=pw, partners_in_pathway=c, confidence=frac, enrichment=en)
                                           for pw, c, frac, en in preds[:5]]
            n_pw += 1
        if entry:
            patch[gname] = entry

    meta = dict(n_ghosts=len(darkfn), n_with_function=n_func, n_with_pathway=n_pw,
                function_tier="fact (UniProt curated literature)",
                pathway_tier="prediction (network guilt-by-association, confidence-tagged)")
    json.dump(dict(meta=meta, patch=patch), open(os.path.join(OUT, "ghost_patch.json"), "w"))
    return meta, patch


def apply_patch(D, patch=None):
    """Overlay the ghost patch onto a loaded model IN MEMORY: fill each ghost gene's function, mark it resolved,
    and attach its predicted pathway memberships (kept separate from curated `pathways`). Returns D (mutated)."""
    if patch is None:
        p = json.load(open(os.path.join(OUT, "ghost_patch.json")))
        patch = p["patch"]
    idx = {g["name"]: i for i, g in enumerate(D["genes"])}
    D.setdefault("pathways_predicted", {})
    for gname, e in patch.items():
        i = idx.get(gname)
        if i is None:
            continue
        g = D["genes"][i]
        if e.get("func"):
            g["func"] = e["func"]; g["func_evidence"] = e["func_evidence"]; g["func_source"] = e["func_source"]
            g["ghost_patched"] = True
            if not g.get("proc") or g.get("proc") == "other":
                g["proc"] = (e.get("func_name") or e["func"])[:40]
        for pw in e.get("pathways_predicted", []):
            D["pathways_predicted"].setdefault(pw["pathway"], []).append(i)
    return D


if __name__ == "__main__":
    meta, patch = build_patch()
    print("=" * 74)
    print("GHOST-GENE PATCH")
    print("=" * 74)
    print(f"  ghost genes:            {meta['n_ghosts']}")
    print(f"  patched with FUNCTION:  {meta['n_with_function']}  (fact tier — UniProt curated literature)")
    print(f"  patched with PATHWAY:   {meta['n_with_pathway']}  (prediction tier — network guilt-by-association)")
    print("  -> outputs/orphan/ghost_patch.json (compact overlay; apply_patch(D) completes the model at load)")
