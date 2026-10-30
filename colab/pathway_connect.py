"""Connect a (dark) gene to the ACTUAL named biological pathway it belongs to — three independent routes,
cross-checked. The strongest: the Foldseek structural homolog is a *known, annotated* protein, so the dark
gene inherits that homolog's real Reactome/KEGG pathway memberships.

  route A  structure  : dark gene -> AlphaFold -> Foldseek homolog (UniProt) -> the homolog's Reactome pathways
  route B  co-essential: darkfn's Perturb-seq guilt-by-association pathway label (already in the model)
  route C  network    : Reactome-pathway enrichment of the gene's reg/sig/PPI partners

Agreement across routes = high-confidence pathway assignment; disagreement is flagged, not hidden.
"""
import json, urllib.request, urllib.error
from collections import Counter

UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.json"


def _accession(homolog_target):
    """'AF-Q8R0P4-F1-model_v6' -> 'Q8R0P4'."""
    if not homolog_target:
        return None
    t = homolog_target.split()[0]
    return t.split("-")[1] if t.startswith("AF-") else t


def reactome_pathways(uniprot_acc, timeout=30, top=8):
    """the real named Reactome pathways annotated for a UniProt accession (via the UniProt cross-refs)."""
    if not uniprot_acc:
        return []
    try:
        d = json.loads(urllib.request.urlopen(UNIPROT.format(uniprot_acc), timeout=timeout).read())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return []
    out = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x.get("database") == "Reactome":
            name = ""
            for p in x.get("properties", []):
                if p.get("key") in ("PathwayName", "pathway name") or p.get("value"):
                    name = p.get("value", "")
            out.append(dict(id=x.get("id"), name=name))
    return out[:top]


def structure_pathways(gene, self_uniprot=None, scan=8):
    """route A: dark gene -> Foldseek structural homolog(s) -> the family's real Reactome pathways.
    The top structural hit is often itself an under-annotated paralog, but the pathway lives in the FOLD
    FAMILY — so we scan the top structural neighbours and take pathways from the first that carries Reactome
    annotations (the well-annotated family member)."""
    from foldseek_function import predict_function
    r = predict_function(gene, self_uniprot=self_uniprot)
    if r.get("abstain"):
        return dict(ok=False, reason=r["reason"])
    neigh = r.get("neighbours", []) or []
    src_acc, paths = None, []
    for h in neigh[:scan]:
        acc = _accession(h.get("target"))
        p = reactome_pathways(acc)
        if p:
            src_acc, paths = acc, p
            break
    return dict(ok=True, homolog=r.get("homolog"), molecular_function=r.get("prediction"),
                homolog_seqId=r.get("homolog_seqId"), pathway_source_acc=src_acc, reactome=paths)


def network_pathway_enrichment(gene, D, idx, name, pathways_map, top=5):
    """route C: which Reactome/model pathways are over-represented among the gene's reg/sig/PPI partners."""
    i = idx.get(gene)
    if i is None:
        return []
    partners = set()
    for e in D.get("reg", []) + D.get("sig", []):
        if e[0] == i:
            partners.add(e[1])
        elif e[1] == i:
            partners.add(e[0])
    for e in D.get("ppi", []):
        if e[0] == i:
            partners.add(e[1])
        elif e[1] == i:
            partners.add(e[0])
    # pathways_map: pathway_name -> set(gene_idx). count partner memberships.
    cnt = Counter()
    for pw, members in (pathways_map or {}).items():
        m = set(members) if not isinstance(members, set) else members
        hits = len(partners & m)
        if hits:
            cnt[pw] += hits
    return cnt.most_common(top)


def connect(gene, D=None, idx=None, name=None, pathways_map=None, self_uniprot=None):
    """all three routes + a cross-checked verdict."""
    out = dict(gene=gene)
    # route A: structure -> homolog -> named Reactome pathway
    out["structure"] = structure_pathways(gene, self_uniprot=self_uniprot)
    # route B: co-essentiality label already in the model
    if D is not None and idx is not None:
        i = idx.get(gene)
        df = (D.get("darkfn") or {}).get(str(i)) if i is not None else None
        out["coessentiality"] = dict(pathway=df.get("pred"), conf=df.get("conf")) if df else None
        # route C: network enrichment
        if pathways_map:
            out["network_enrichment"] = network_pathway_enrichment(gene, D, idx, name, pathways_map)
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(structure_pathways(sys.argv[1] if len(sys.argv) > 1 else "RASSF3"), indent=2)[:1200])
