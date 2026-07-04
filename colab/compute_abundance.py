"""ABUNDANCE lens — PaxDb integrated protein abundance (the first QUANTITATIVE layer).

Everything else in the model is presence/relationship; PaxDb gives an actual number: integrated protein
abundance (ppm, ~copies) per gene, aggregated across many human proteomics studies. This is a real
"how much of each molecule" signal — toward the every-molecule goal — and the abundances an
enzyme-constrained (ecModel) kinetics layer needs. -> paxdb_abundance.json {gene: ppm}

Reads a PaxDb integrated file (string id 9606.ENSP... + abundance) + string_aliases for ENSP->symbol.
Skips gracefully if absent. URL is env-overridable (PAXDB_URL) since PaxDb paths are versioned.
"""
import json, gzip, os
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
# where a manually-uploaded PaxDb file might sit (Drive root + subfolders), in addition to the data dir
DRIVE_DIRS=[Path("/content/drive/MyDrive/virtual_cell_data"),
            Path("/content/drive/MyDrive/virtual_cell_data/human_raw"),
            Path("/content/drive/MyDrive/virtual_cell_data/expression_geo")]

def ensp2sym():
    m={}; f=H/"string_aliases.txt.gz"
    if f.exists():
        for l in gzip.open(f,"rt"):
            p=l.rstrip("\n").split("\t")
            if len(p)>=3 and p[2]=="Ensembl_HGNC_symbol": m[p[0]]=p[1]
    return m

def _open(f): return gzip.open(f,"rt",errors="ignore") if str(f).endswith(".gz") else open(f,errors="ignore")

def _find():
    # explicit path wins (PAXDB_FILE), then known names, then broad globs across data dir + Drive folders
    env=os.environ.get("PAXDB_FILE","")
    if env and Path(env).exists(): return Path(env)
    names=["paxdb_human.txt","paxdb_human_integrated.txt","9606-WHOLE_ORGANISM-integrated.txt","paxdb.txt"]
    pats=["*paxdb*.txt","*paxdb*.txt.gz","*PaxDb*.txt*","*WHOLE_ORGANISM*integrated*","9606*integrated*.txt*","9606*.txt"]
    for d in [H]+[x for x in DRIVE_DIRS if x.exists()]:
        for n in names:
            for p in (d/n, d/(n+".gz")):
                if p.exists() and p.stat().st_size>1000: return p
        hits=[]
        for pat in pats: hits+=sorted(d.glob(pat))
        hits=[h for h in hits if h.is_file() and h.stat().st_size>1000]
        if hits: return hits[0]
    return None

def main():
    f=_find()
    if not f:
        print("no PaxDb file found (looked in data dir + Drive virtual_cell_data/{,human_raw,expression_geo}). "
              "Set PAXDB_FILE=/full/path, or place a *paxdb*.txt there -> skipping abundance lens"); return
    print(f"abundance: using PaxDb file {f}")
    e2s=ensp2sym()
    import re
    SYM=re.compile(r"^[A-Za-z][A-Za-z0-9\-\.@_]{0,20}$")   # plausible HGNC symbol (has a leading letter)
    ab={}; via_name=0; via_ensp=0
    for l in _open(f):
        if l.startswith("#") or not l.strip(): continue
        p=[t.strip() for t in l.replace(",","\t").split("\t") if t.strip()]
        # 1) prefer the explicit gene_name column (PaxDb integrated files carry it) -> no alias dependency
        sym=next((t for t in p if "ENSP" not in t and "." not in t and SYM.match(t)
                  and not t.replace("-","").isdigit()), None)
        # 2) fall back to string id (9606.ENSP...) -> symbol via string_aliases
        if not sym:
            sid=next((t for t in p if "ENSP" in t), None)
            if sid:
                ensp=sid.split(".")[-1] if "." in sid else sid
                sym=e2s.get("9606."+ensp) or e2s.get(ensp)
                if sym: via_ensp+=1
        else: via_name+=1
        if not sym: continue
        # abundance = the last positive float on the line (PaxDb 'abundance' is the final column)
        val=None
        for t in reversed(p):
            try:
                v=float(t)
                if v>0: val=v; break
            except: continue
        if val is None: continue
        ab[sym.upper()]=max(ab.get(sym.upper(),0.0), round(val,3))
    json.dump(ab, open(OUT/"paxdb_abundance.json","w"))
    if ab:
        import statistics
        vals=sorted(ab.values())
        print(f"abundance (PaxDb): {len(ab)} genes ({via_name} via gene_name col, {via_ensp} via ENSP map) | "
              f"ppm range {vals[0]:.2g}..{vals[-1]:.2g} | median {statistics.median(vals):.2g}")
    else:
        print("abundance: PaxDb parsed 0 genes (check id mapping / format)")

if __name__=="__main__":
    main()
