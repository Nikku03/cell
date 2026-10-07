"""Structure-based function prediction for the dark proteome — AlphaFold structure + Foldseek homology.

The idea (the user's): for a protein of unknown function, **structure is more conserved than sequence**, so a
*structural* homolog transfers function even when sequence identity is too low for a sequence method (ESM,
BLAST) to see it. So: fetch the **AlphaFold** structure → search it against the **AlphaFold/Swiss-Prot**
database with **Foldseek** → transfer the function (annotation/GO) of the top structural neighbours, weighted
by structural similarity.

This complements the existing `darkfn` (Perturb-seq co-essentiality guilt-by-association): that finds the
*pathway a gene acts in*; this finds the *molecular activity of the fold* (enzyme class, domain, binding).
They are independent evidence — agreement raises confidence, and structure reaches folds co-essentiality can't.

`colab/foldseek_function.py`. Network: AlphaFold EBI + the public Foldseek web API (search.foldseek.com).
Validated by `colab/validate_foldseek_function.py` (recovers known function for benchmark proteins via
structure, blind to their sequence annotation).
"""
import json, time, urllib.request, io
import requests

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
FOLDSEEK = "https://search.foldseek.com/api"
# Swiss-Prot AlphaFold DB carries curated function/GO; UniProt50 for broader structural coverage.
DEFAULT_DBS = ["afdb-swissprot"]


def uniprot_for(gene, species="human"):
    """resolve a gene symbol -> reviewed UniProt accession (Swiss-Prot) via mygene."""
    import mygene
    hits = mygene.MyGeneInfo().querymany([gene], scopes="symbol", fields="uniprot",
                                         species=species, verbose=False)
    for h in hits:
        up = (h.get("uniprot") or {}).get("Swiss-Prot")
        if up:
            return up if isinstance(up, str) else up[0]
    return None


def alphafold_pdb_bytes(uniprot, timeout=30):
    """fetch the AlphaFold model PDB for a UniProt accession (version resolved via the EBI API)."""
    meta = json.loads(urllib.request.urlopen(AF_API.format(uniprot), timeout=timeout).read())
    pdb_url = meta[0]["pdbUrl"]
    return urllib.request.urlopen(pdb_url, timeout=timeout).read()


def foldseek_search(pdb_bytes, dbs=None, mode="3diaa", poll=6, max_wait=600):
    """submit a structure to the Foldseek web API, poll to completion, return parsed hits (list of dicts).
    Each hit: {target, seqId, prob, eval, score, alntmscore, qtmscore, ttmscore, taxname, description}."""
    dbs = dbs or DEFAULT_DBS
    files = {"q": ("query.pdb", io.BytesIO(pdb_bytes), "application/octet-stream")}
    data = [("mode", mode)] + [("database[]", d) for d in dbs]
    r = requests.post(f"{FOLDSEEK}/ticket", files=files, data=data, timeout=60)
    r.raise_for_status()
    ticket = r.json()
    tid = ticket["id"]
    waited = 0
    while waited < max_wait:
        st = requests.get(f"{FOLDSEEK}/ticket/{tid}", timeout=30).json()
        status = st.get("status")
        if status == "COMPLETE":
            break
        if status == "ERROR":
            raise RuntimeError(f"Foldseek job error for ticket {tid}")
        time.sleep(poll); waited += poll
    else:
        raise TimeoutError(f"Foldseek job {tid} did not finish in {max_wait}s")
    res = requests.get(f"{FOLDSEEK}/result/{tid}/0", timeout=60).json()
    hits = []
    for db in res.get("results", []):
        for a in db.get("alignments", []):
            # the web API nests alignments one level deeper in some versions
            rows = a if isinstance(a, list) else [a]
            for h in rows:
                if not isinstance(h, dict):
                    continue
                tgt = h.get("target") or ""
                # the function description is the text after the AFDB model id in the target header
                desc = tgt.split(" ", 1)[1].strip() if " " in tgt else ""
                hits.append(dict(
                    target=tgt.split(" ", 1)[0], description=desc,
                    seqId=h.get("seqId"), prob=h.get("prob"), eval=h.get("eval"), score=h.get("score"),
                    taxname=h.get("taxName") or h.get("taxname")))
    return hits


def predict_function(gene, dbs=None, top=8, min_prob=0.5, self_uniprot=None):
    """full pipeline: gene -> AlphaFold structure -> Foldseek -> function transferred from structural neighbours.
    Returns {gene, uniprot, tier, confidence, prediction, neighbours[...]} or an abstain dict."""
    up = self_uniprot or uniprot_for(gene)
    if not up:
        return dict(gene=gene, abstain=True, reason="no reviewed UniProt")
    try:
        pdb = alphafold_pdb_bytes(up)
    except Exception as e:
        return dict(gene=gene, uniprot=up, abstain=True, reason=f"no AlphaFold structure: {e}")
    try:
        hits = foldseek_search(pdb, dbs=dbs)
    except Exception as e:
        return dict(gene=gene, uniprot=up, abstain=True, reason=f"Foldseek failed: {e}")
    # drop self-hits (same UniProt), uninformative headers, and weak hits
    def acc(t):
        # AFDB target ids look like "AF-P12345-F1-model_v6"
        return t.split("-")[1] if t and t.startswith("AF-") else t
    UNINFORMATIVE = ("uncharacterized", "putative uncharacterized", "")
    neigh = [h for h in hits if acc(h["target"]) != up and (h.get("prob") or 0) >= min_prob]
    neigh.sort(key=lambda h: -(h.get("prob") or 0))
    if not neigh:
        return dict(gene=gene, uniprot=up, abstain=True,
                    reason="no confident structural homolog (dark fold / novel structure)")
    # the transferred annotation = the best hit whose header is actually informative
    annotated = [h for h in neigh if h.get("description")
                 and h["description"].lower() not in UNINFORMATIVE
                 and "uncharacterized" not in h["description"].lower()]
    best = annotated[0] if annotated else neigh[0]
    top_hits = neigh[:top]
    # low sequence identity + high structural prob = structure sees what sequence methods (ESM/BLAST) cannot.
    # scan ALL confident hits (not just the ortholog-dominated top) for the best twilight-zone homolog.
    twilight = [h for h in neigh if (h.get("prob") or 0) >= 0.9 and (h.get("seqId") or 100) < 30
                and h.get("description") and "uncharacterized" not in h["description"].lower()]
    low_seq = (best.get("seqId") or 100) < 30
    return dict(gene=gene, uniprot=up, tier="predicted",
                confidence=round(float(best.get("prob") or 0), 3),
                prediction=best.get("description") or "(unannotated structural homolog)",
                homolog=best.get("target"), homolog_taxon=best.get("taxname"),
                homolog_seqId=best.get("seqId"), homolog_eval=best.get("eval"),
                structure_beyond_sequence=bool(low_seq),
                twilight_homologs=[dict(target=h["target"], desc=h.get("description"),
                                        seqId=h.get("seqId"), prob=h.get("prob")) for h in twilight[:10]],
                neighbours=[dict(target=h["target"], desc=h.get("description"), prob=h.get("prob"),
                                 seqId=h.get("seqId"), eval=h.get("eval")) for h in top_hits],
                provenance="AlphaFold structure + Foldseek (afdb-swissprot) structural homology")


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "AAMDC"
    r = predict_function(g)
    print(json.dumps(r, indent=2)[:1500])
