"""GENE ONTOLOGY lens — the canonical functional annotation (NCBI gene2go).

GO gives each gene curated Molecular Function / Biological Process / Cellular Component terms. Two uses:
(1) many 'dark' genes (no Reactome pathway, no disease) DO have a GO molecular-function term -> assign it
    directly, and transfer GO terms across functional neighbors;
(2) GO is the standard GOLD STANDARD for validating the co-expression network (guilt-by-association AUROC).

Reads gene2go.gz (NCBI) + gene_info for Entrez->symbol. Keeps human, non-IEA-preferred but includes IEA.
-> go_terms.json  {gene: {F:[terms], P:[terms], C:[terms]}}
"""
import json, gzip
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def entrez2sym():
    m={}; p=H/"gene_info.gz"
    if p.exists():
        with gzip.open(p,"rt") as f:
            next(f)
            for l in f:
                c=l.split("\t")
                if len(c)>2 and c[0]=="9606": m[c[1]]=c[2]
    return m

CAT={"Function":"F","Process":"P","Component":"C"}

def main():
    f=H/"gene2go.gz"
    if not f.exists():
        print("no gene2go.gz -> skipping GO lens"); return
    e2s=entrez2sym()
    go=defaultdict(lambda: {"F":[], "P":[], "C":[]})
    with gzip.open(f,"rt") as fh:
        for l in fh:
            if l.startswith("#"): continue
            p=l.rstrip("\n").split("\t")
            if len(p)<8 or p[0]!="9606": continue
            sym=e2s.get(p[1])
            if not sym: continue
            cat=CAT.get(p[7])
            term=p[5]
            if cat and p[4]!="NOT" and term not in go[sym][cat]:
                go[sym][cat].append(term)
    go={g:{k:v[:8] for k,v in d.items() if v} for g,d in go.items()}
    json.dump(go, open(OUT/"go_terms.json","w"))
    nF=sum(1 for d in go.values() if d.get("F"))
    print(f"GO: annotations for {len(go)} genes ({nF} with a Molecular Function term) -> go_terms.json")

if __name__=="__main__":
    main()
