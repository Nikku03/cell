"""litmine — extract the INDIVIDUAL literature for a gene from PubMed, to fill the model's dark genes from the
one place high-throughput screens can't reach: focused single-gene studies.

Motivation: the K562 Perturb-seq screens cover ~8,700 genes expressed in that line; the ~7,800 they miss are largely
genes K562 doesn't express — but 2,540 of the model's DARK genes still have >=5 PubMed papers. Those papers hold the
causal/functional knowledge (what the gene does, what it regulates, what its knockout causes) that no screen we can
fetch will provide. This mines it, GROUNDED in real abstracts with DOIs.

Honest scope: this is a DESCRIPTIVE/QUALITATIVE layer (functions, interactions, phenotypes + citations), not the
quantitative measured signatures the debugger/predictor use. It fills the model's annotation gaps from literature;
it does NOT raise the causal-prediction number (that needs measured perturbation data). The fetch (search+abstracts)
is deterministic and reproducible via NCBI E-utilities; turning abstracts into structured facts is an LLM step done
downstream (grounded in the fetched text) — litmine stores the source papers so any extracted fact is auditable.

-> outputs/orphan/litmine.json   ({gene: {n_papers, papers:[{pmid,title,year,doi,abstract}]}})
"""
import os, sys, json, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET

OUT = "outputs/orphan"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# bias the query toward CAUSAL/functional papers, not mere mentions
CAUSAL = "(function OR regulates OR knockout OR knockdown OR mechanism OR pathway OR mutation OR deficiency)"


def search(gene, retmax=6):
    term = f'{gene}[Title/Abstract] AND {CAUSAL}'
    url = f"{EUTILS}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}&retmax={retmax}&retmode=json&sort=relevance"
    d = json.loads(urllib.request.urlopen(url, timeout=30).read())
    return d.get("esearchresult", {}).get("idlist", [])


def fetch(pmids):
    if not pmids:
        return []
    url = f"{EUTILS}/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=45).read())
    out = []
    for art in root.findall(".//PubmedArticle"):
        doi = next((a.text for a in art.findall(".//ArticleId") if a.get("IdType") == "doi"), "")
        out.append({
            "pmid": art.findtext(".//PMID") or "",
            "title": (art.findtext(".//ArticleTitle") or "").strip(),
            "year": art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "",
            "doi": doi,
            "abstract": " ".join((a.text or "") for a in art.findall(".//AbstractText")).strip()[:1500],
        })
    return out


def mine(gene, retmax=6):
    pmids = search(gene, retmax)
    time.sleep(0.4)                                   # NCBI: <=3 req/s without an API key
    papers = fetch(pmids)
    time.sleep(0.4)
    return {"gene": gene, "n_papers": len(papers), "papers": papers}


def mine_genes(genes, path=f"{OUT}/litmine.json"):
    db = json.load(open(path)) if os.path.exists(path) else {}
    for g in genes:
        if g in db:
            continue
        try:
            db[g] = mine(g)
            print(f"  {g:10} {db[g]['n_papers']} papers", flush=True)
        except Exception as e:
            print(f"  {g:10} FAILED: {str(e)[:60]}", flush=True)
    json.dump(db, open(path, "w"), indent=1)
    return db


if __name__ == "__main__":
    # default batch: dark-in-model but literature-rich genes (function known to the field, not to our pipeline)
    genes = sys.argv[1:] or ["FNDC5", "NPPB", "POSTN", "BMAL1", "DISC1", "SERPINB5", "CD82", "SHBG"]
    db = mine_genes(genes)
    print(f"\nlitmine.json now covers {len(db)} genes")
