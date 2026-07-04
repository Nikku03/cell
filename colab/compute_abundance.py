"""ABUNDANCE lens — PaxDb integrated protein abundance (the first QUANTITATIVE layer).

Everything else in the model is presence/relationship; PaxDb gives an actual number: integrated protein
abundance (ppm, ~copies) per gene, aggregated across many human proteomics studies. This is a real
"how much of each molecule" signal — toward the every-molecule goal — and the abundances an
enzyme-constrained (ecModel) kinetics layer needs. -> paxdb_abundance.json {gene: ppm}

Reads a PaxDb integrated file (string id 9606.ENSP... + abundance) + string_aliases for ENSP->symbol.
Skips gracefully if absent. URL is env-overridable (PAXDB_URL) since PaxDb paths are versioned.
"""
import json, gzip
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def ensp2sym():
    m={}; f=H/"string_aliases.txt.gz"
    if f.exists():
        for l in gzip.open(f,"rt"):
            p=l.rstrip("\n").split("\t")
            if len(p)>=3 and p[2]=="Ensembl_HGNC_symbol": m[p[0]]=p[1]
    return m

def _find():
    for n in ["paxdb_human.txt","paxdb_human_integrated.txt","9606-WHOLE_ORGANISM-integrated.txt","paxdb.txt"]:
        p=H/n
        if p.exists() and p.stat().st_size>1000: return p
    hits=sorted(H.glob("*paxdb*.txt"))+sorted(H.glob("9606*integrated*.txt"))
    return hits[0] if hits else None

def main():
    f=_find()
    if not f:
        print("no PaxDb file -> skipping abundance lens (set PAXDB_URL on Colab)"); return
    e2s=ensp2sym()
    ab={}
    for l in open(f):
        if l.startswith("#") or not l.strip(): continue
        p=l.replace(",","\t").split("\t")
        # find the string id token (9606.ENSP...) and the abundance (first float after it)
        sid=next((t for t in p if "ENSP" in t), None)
        if not sid: continue
        ensp=sid.split(".")[-1] if "." in sid else sid
        sym=e2s.get("9606."+ensp) or e2s.get(ensp)
        if not sym: continue
        val=None
        for t in p:
            try:
                v=float(t)
                if v>0: val=v; break
            except: continue
        if val is None: continue
        ab[sym]=max(ab.get(sym,0.0), round(val,3))
    json.dump(ab, open(OUT/"paxdb_abundance.json","w"))
    if ab:
        import statistics
        vals=sorted(ab.values())
        print(f"abundance (PaxDb): {len(ab)} genes | ppm range {vals[0]:.2g}..{vals[-1]:.2g} | median {statistics.median(vals):.2g}")
    else:
        print("abundance: PaxDb parsed 0 genes (check id mapping / format)")

if __name__=="__main__":
    main()
