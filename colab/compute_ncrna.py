"""PHASE 5 — non-coding-RNA -> target-gene regulatory layer.

RNAcentral is only a registry; the INTERACTIONS come from miRTarBase (miRNA->mRNA, experimentally
supported) and LncTarD (lncRNA->target). This turns "we have ncRNA" into an actual regulatory layer:
ncRNA -> gene edges, mapped to HGNC symbols. -> ncrna_targets.json {ncrna:[genes]} + gene2ncrna reverse.

Reads miRTarBase hsa_MTI (.xlsx or .tsv/.csv) and/or LncTarD (flat tsv). Maps Entrez IDs -> symbol via
gene_info. Skips gracefully if neither file is present. Download URLs are China-hosted and change per
release, so the notebook passes them via env vars (MIRTARBASE_URL / LNCTARD_URL).
"""
import json, gzip, csv
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

def _find(*names):
    for n in names:
        p=H/n
        if p.exists() and p.stat().st_size>500: return p
    return None

def read_mirtarbase(e2s, syms):
    """miRNA -> target symbol, strong (functional) evidence preferred."""
    f=_find("mirtarbase_hsa.xlsx","hsa_MTI.xlsx","mirtarbase_hsa.tsv","mirtarbase_hsa.csv")
    if not f: return {}
    rows=[]
    if f.suffix==".xlsx":
        try:
            import openpyxl
            wb=openpyxl.load_workbook(f,read_only=True); ws=wb.active
            it=ws.iter_rows(values_only=True); hdr=[str(x) for x in next(it)]
            for r in it: rows.append(dict(zip(hdr,r)))
        except Exception as ex:
            print("  miRTarBase xlsx read failed:",repr(ex)[:80]); return {}
    else:
        delim="\t" if f.suffix==".tsv" else ","
        rows=list(csv.DictReader(open(f),delimiter=delim))
    def col(r,*ks):
        for k in ks:
            for rk in r:
                if rk and k.lower() in str(rk).lower(): return r[rk]
        return None
    out=defaultdict(set)
    for r in rows:
        mir=col(r,"miRNA"); ent=col(r,"Entrez"); tgt=col(r,"Target Gene")
        sup=(col(r,"Support Type") or "")
        if "weak" in str(sup).lower(): continue          # keep functional/strong evidence
        sym = e2s.get(str(ent).split(".")[0]) if ent else None
        sym = sym or (tgt if tgt in syms else None)
        if mir and sym: out[str(mir)].add(sym)
    return {k:sorted(v) for k,v in out.items()}

def read_lnctard(syms):
    f=_find("lnctard.txt","lnctard.tsv","LncTarD.txt","lnctard_2.0.txt")
    if not f: return {}
    out=defaultdict(set)
    for r in csv.DictReader(open(f,encoding="utf-8",errors="ignore"),delimiter="\t"):
        def g(*ks):
            for k in ks:
                for rk in r:
                    if rk and k.lower() in rk.lower(): return r[rk]
            return None
        lnc=g("Regulator","lncRNA"); tgt=g("Target")
        if lnc and tgt and tgt in syms: out[str(lnc)].add(str(tgt))
    return {k:sorted(v) for k,v in out.items()}

def main():
    syms=set()
    cc=OUT/"cell_complete.json"
    if cc.exists(): syms=set(g["name"] for g in json.load(open(cc))["genes"])
    e2s=entrez2sym()
    mir=read_mirtarbase(e2s,syms); lnc=read_lnctard(syms)
    if not mir and not lnc:
        print("ncRNA: no miRTarBase/LncTarD file present -> skipping (set MIRTARBASE_URL/LNCTARD_URL)"); return
    ncrna={**{k:v for k,v in mir.items()}, **{k:v for k,v in lnc.items()}}
    gene2ncrna=defaultdict(list)
    for nc,tg in ncrna.items():
        for g in tg:
            if g in syms: gene2ncrna[g].append(nc)
    json.dump(dict(ncrna_targets=ncrna, gene2ncrna={k:v[:25] for k,v in gene2ncrna.items()}),
              open(OUT/"ncrna_targets.json","w"))
    print(f"ncRNA: {len(mir)} miRNAs + {len(lnc)} lncRNAs -> targets on {len(gene2ncrna)} genes")

if __name__=="__main__":
    main()
