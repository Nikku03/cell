"""FETCH a real gene-expression TIME COURSE to validate the time-resolution equation.

GSE6783 (Amit et al.): HeLa stimulated with EGF, sampled at 0, 20, 40, 60, 120, 240, 480 min (7 points,
Affymetrix HG-U133A / GPL96). This is measured mRNA dynamics we can predict from mRNA half-lives.
  - expression: GSE6783_series_matrix.txt.gz   (probe x 7 timepoints)
  - probe -> gene symbol: GSE6783_family.soft.gz platform table (NCBI platform FTP is proxy-blocked; the
    series-path family SOFT embeds the same GPL96 ID->Gene Symbol table and IS reachable).
Probes are collapsed to genes by the most-variable probe (the responsive one). Values linearized if log2.
-> timecourse_egf.tsv (gene t0 t20 t40 t60 t120 t240 t480)   [consumed by compute_timeresolved.py]
"""
import os, gzip, io, urllib.request
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
MATRIX_URL=os.environ.get("TIMECOURSE_MATRIX_URL",
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6783/matrix/GSE6783_series_matrix.txt.gz")
SOFT_URL=os.environ.get("TIMECOURSE_SOFT_URL",
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6783/soft/GSE6783_family.soft.gz")
TIMES=[0,20,40,60,120,240,480]                       # minutes (unstimulated + 6 EGF points)

def _dl(url, dest):
    f=H/dest
    if f.exists() and f.stat().st_size>10000: return f.read_bytes()
    try:
        print(f"  fetching {dest} ..."); data=urllib.request.urlopen(url, timeout=120).read()
        try: H.mkdir(parents=True, exist_ok=True); f.write_bytes(data)
        except Exception: pass
        return data
    except Exception as e:
        print(f"  fetch failed ({dest}): {repr(e)[:90]}"); return None

def probe2gene():
    data=_dl(SOFT_URL,"GSE6783_family.soft.gz")
    if not data: return {}
    m={}; in_tab=False; id_i=sym_i=None
    with gzip.open(io.BytesIO(data),"rt",errors="ignore") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"): in_tab=True; continue
            if line.startswith("!platform_table_end"): break
            if in_tab:
                p=line.rstrip("\n").split("\t")
                if id_i is None:                       # header row
                    low=[c.lower() for c in p]
                    id_i=0; sym_i=next((i for i,c in enumerate(low) if c=="gene symbol"),None)
                    continue
                if sym_i is not None and len(p)>sym_i and p[id_i] and p[sym_i]:
                    m[p[id_i]]=p[sym_i].split("///")[0].strip()   # first symbol
    return m

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mtx=_dl(MATRIX_URL,"GSE6783_series_matrix.txt.gz")
    if not mtx: print("  time course unavailable -> validation of the time equation will skip"); return
    p2g=probe2gene()
    rows=[]; in_tab=False
    with gzip.open(io.BytesIO(mtx),"rt",errors="ignore") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"): in_tab=True; continue
            if line.startswith("!series_matrix_table_end"): break
            if in_tab:
                p=line.rstrip("\n").split("\t")
                if p[0].strip('"')=="ID_REF": continue
                try: vals=[float(x) for x in p[1:8]]
                except ValueError: continue
                if len(vals)==7: rows.append((p[0].strip('"'), vals))
    if not rows: print("  no expression rows parsed -> time course skipped"); return
    # detect log2 (Affy intensities are usually linear ~10^2-10^4; if all small, it's log2 -> linearize)
    mx=max(max(v) for _,v in rows[:2000])
    log2 = mx < 25
    genebest={}                                       # gene -> (variance, values); keep the most-variable probe
    for probe,vals in rows:
        g=p2g.get(probe)
        if not g: continue
        lin=[2**x for x in vals] if log2 else vals
        if min(lin)<=0: continue
        var=max(lin)/min(lin)                          # fold-range = responsiveness
        if g not in genebest or var>genebest[g][0]: genebest[g]=(var, lin)
    with open(OUT/"timecourse_egf.tsv","w") as o:
        o.write("gene\t"+"\t".join(f"t{t}" for t in TIMES)+"\n")
        for g,(_,lin) in sorted(genebest.items()):
            o.write(g+"\t"+"\t".join(f"{x:.4f}" for x in lin)+"\n")
    print(f"time course: GSE6783 EGF (HeLa), {len(genebest)} genes x {len(TIMES)} timepoints "
          f"(0-480 min){' [linearized from log2]' if log2 else ''} -> timecourse_egf.tsv")

if __name__=="__main__":
    main()
