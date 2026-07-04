"""DOMAIN lens — Pfam/InterPro domain-based function for dark genes.

A protein's domains reveal function even with no expression/interaction data. We map each gene to its
Pfam/InterPro domains, learn each domain's typical function from the CHARACTERIZED proteins that carry
it, then transfer that to dark genes sharing the domain (domain-based guilt-by-association) — an
independent signal from structure and expression.

Reads uniprot_domains.tsv (accession, gene, Pfam ids, InterPro ids) — fetched from UniProt on Colab
via: fields=accession,gene_primary,xref_pfam,xref_interpro. Skips gracefully if absent.
-> domain_function.json  {dark_gene: {pred, ev, domains, conf, src}}
"""
import json, csv
from collections import defaultdict, Counter
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def load_domains():
    """gene -> set(domain ids) from uniprot_domains.tsv (defensive on column layout)."""
    f=H/"uniprot_domains.tsv"
    if not f.exists(): return {}
    g2d=defaultdict(set)
    rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
    def col(row,*keys):
        for k in keys:
            for i,h in enumerate(hdr):
                if k in h and i<len(row): return row[i]
        return ""
    for r in rd:
        if not r: continue
        gene=(col(r,"gene_primary","gene")).split()[0] if col(r,"gene_primary","gene") else ""
        doms=col(r,"pfam")+";"+col(r,"interpro")
        ids=[d for d in doms.replace(",",";").split(";") if d.strip()]
        if gene and ids:
            for d in ids: g2d[gene].add(d.strip())
    return g2d

def main():
    g2d=load_domains()
    if not g2d:
        print("no uniprot_domains.tsv -> skipping domain lens (fetch Pfam/InterPro from UniProt on Colab)"); return
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]
    dark=set(g["name"] for g in G if g["dark"])
    func={g["name"]:(g.get("path") or g.get("proc") or "other") for g in G}
    # domain -> function counts from CHARACTERIZED proteins only
    dom_fn=defaultdict(Counter); dom_genes=defaultdict(list)
    for g,doms in g2d.items():
        if g in dark: continue
        fn=func.get(g)
        if not fn or fn=="other": continue
        for d in doms: dom_fn[d][fn]+=1; dom_genes[d].append(g)
    out={}
    for g in dark:
        doms=g2d.get(g)
        if not doms: continue
        votes=Counter(); ev=[]; used=[]
        for d in doms:
            if d in dom_fn:
                for fn,n in dom_fn[d].items(): votes[fn]+=n
                ev+= dom_genes[d][:3]; used.append(d)
        if votes:
            pred,n=votes.most_common(1)[0]
            out[g]=dict(pred=pred, ev=list(dict.fromkeys(ev))[:5], domains=used[:4],
                        conf="high" if n>=4 else "low", src="domain(Pfam/InterPro)")
    json.dump(out,open(OUT/"domain_function.json","w"))
    print(f"domain lens: {len(g2d)} genes with domains -> function assigned to {len(out)} dark genes")

if __name__=="__main__":
    main()
