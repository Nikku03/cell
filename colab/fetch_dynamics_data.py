"""FETCH turnover + cell-cycle data -> the raw inputs for a DYNAMICS layer (Problem 1, temporal).

Static abundances become dynamics once you know how fast each species turns over: a protein's response
time to any perturbation is set by its degradation rate (tau = half-life/ln2). Three commercial-safe sources:
  - Mathieson et al. 2018 (Nat Commun, CC-BY): PROTEIN half-lives, human primary cells (B/NK/monocytes).
  - RNADecayCafe (Vock/Simon, CC-BY, Zenodo): mRNA half-lives across 12 human cell lines.
  - MSigDB WHITFIELD_CELL_CYCLE_* (Whitfield 2002 HeLa): genes with their peak CELL-CYCLE phase.

-> protein_halflife.tsv (gene half_life_h)   mrna_halflife.tsv (gene half_life_h)   cellcycle_phase.tsv (gene phase)
We extract numeric VALUES (facts) at runtime; source files are not committed. Skips gracefully per-source.
"""
import os, io, csv, urllib.request
from statistics import median
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
MATHIESON_URL=os.environ.get("PROT_HALFLIFE_URL",
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03106-1/MediaObjects/41467_2018_3106_MOESM5_ESM.xlsx")
RNADECAY_URL=os.environ.get("MRNA_HALFLIFE_URL",
    "https://zenodo.org/records/15785218/files/AvgKdegs_genes_v1.csv")
MSIGDB=("https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp?geneSetName=%s&fileType=TSV")
PHASES=[("WHITFIELD_CELL_CYCLE_G1_S","G1/S"),("WHITFIELD_CELL_CYCLE_S","S"),
        ("WHITFIELD_CELL_CYCLE_G2","G2"),("WHITFIELD_CELL_CYCLE_G2_M","G2/M"),("WHITFIELD_CELL_CYCLE_M_G1","M/G1")]

def _get(url, dest, binary=True, timeout=90):
    f=H/dest
    if f.exists() and f.stat().st_size>2000: return f.read_bytes() if binary else f.read_text()
    try:
        print(f"  fetching {dest} ..."); data=urllib.request.urlopen(url, timeout=timeout).read()
        try: H.mkdir(parents=True, exist_ok=True); f.write_bytes(data)
        except Exception: pass
        return data if binary else data.decode("utf-8","ignore")
    except Exception as e:
        print(f"  fetch failed ({dest}): {repr(e)[:90]}"); return None

def protein_halflives():
    data=_get(MATHIESON_URL, "mathieson_halflife.xlsx", binary=True)
    if not data: return {}
    try:
        import openpyxl
        wb=openpyxl.load_workbook(io.BytesIO(data), read_only=True); ws=wb["protein half lives high qual"]
    except Exception as e:
        print("  Mathieson parse failed:",repr(e)[:80]); return {}
    out={}
    for i,r in enumerate(ws.iter_rows(values_only=True)):
        if i==0 or not r or not r[0]: continue
        vals=[]
        for j in range(1,len(r),3):                       # half-life is the 1st of each (value,qual,rsq) triplet
            try:
                v=float(r[j])
                if 0.1<=v<=2000: vals.append(v)
            except (TypeError,ValueError): pass
        if vals: out[str(r[0]).strip()]=round(median(vals),3)
    return out

def mrna_halflives():
    data=_get(RNADECAY_URL, "rnadecaycafe.csv", binary=False)
    if not data: return {}
    per={}
    rd=csv.DictReader(io.StringIO(data))
    for row in rd:
        g=(row.get("feature_ID") or "").strip()
        try: hl=float(row.get("avg_halflife"))
        except (TypeError,ValueError): continue
        if g and 0.05<=hl<=1000: per.setdefault(g,[]).append(hl)
    return {g:round(median(v),3) for g,v in per.items()}

def cellcycle_phase():
    out={}
    for setname,phase in PHASES:
        txt=_get(MSIGDB%setname, f"{setname}.tsv", binary=False, timeout=45)
        if not txt: continue
        for line in txt.splitlines():
            if line.startswith("GENE_SYMBOLS"):
                for g in line.split("\t",1)[-1].split(","):
                    g=g.strip()
                    if g and g not in out: out[g]=phase        # first (earliest-phase) assignment wins
                break
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prot=protein_halflives(); mrna=mrna_halflives(); cyc=cellcycle_phase()
    with open(OUT/"protein_halflife.tsv","w") as o:
        o.write("gene\thalf_life_h\tsource\n")
        for g,hl in sorted(prot.items()): o.write(f"{g}\t{hl}\tMathieson2018\n")
    with open(OUT/"mrna_halflife.tsv","w") as o:
        o.write("gene\thalf_life_h\tsource\n")
        for g,hl in sorted(mrna.items()): o.write(f"{g}\t{hl}\tRNADecayCafe\n")
    with open(OUT/"cellcycle_phase.tsv","w") as o:
        o.write("gene\tphase\tsource\n")
        for g,ph in sorted(cyc.items()): o.write(f"{g}\t{ph}\tWhitfield2002\n")
    print(f"dynamics data: {len(prot)} protein half-lives (Mathieson CC-BY), {len(mrna)} mRNA half-lives "
          f"(RNADecayCafe CC-BY), {len(cyc)} cell-cycle-phase genes (Whitfield/MSigDB)")

if __name__=="__main__":
    main()
