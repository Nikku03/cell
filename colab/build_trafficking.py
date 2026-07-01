"""For example genes, fetch UniProt localization/trafficking and build the molecular JOURNEY:
genome -> transcription -> hnRNA -> splicing -> mRNA -> nuclear export -> cytoplasm ->
translation (+/- SRP/translocon) -> folding -> transport -> final compartment, with the
machinery at each step. Output cell_trafficking.json for the spatial cell diagram."""
import json, urllib.request, re, time, warnings, socket
socket.setdefaulttimeout(35); warnings.filterwarnings("ignore")
from pathlib import Path
OUT=Path("outputs/orphan"); SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
# gene -> UniProt ACC (from human.faa)
acc={}
for l in open(SC/"human.faa"):
    if l.startswith(">"):
        m=re.match(r">\w+\|(\w+)\|",l); gn=[t[3:] for t in l.split() if t.startswith("GN=")]
        if m and gn: acc.setdefault(gn[0],m.group(1))
import csv
TFset=set()
for r in csv.DictReader(open("data/external_data/human/collectri.tsv"),delimiter="\t"):
    if r["source_genesymbol"]: TFset.add(r["source_genesymbol"])
GENES=["SLC2A1","INSR","ALB","TP53","GAPDH","SOD2","LDLR","EGFR"]
def hj(url):
    for _ in range(2):
        try: return json.load(urllib.request.urlopen(url,timeout=30))
        except Exception: time.sleep(2)
    return None
def uni(ac):
    d=hj(f"https://rest.uniprot.org/uniprotkb/{ac}.json?fields=cc_subcellular_location,ft_signal,ft_transmem,protein_name,cc_function")
    if not d: return None
    name=(((d.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName") or {}).get("value","")
    sig=any(f["type"]=="Signal" for f in d.get("features",[]))
    ntm=sum(1 for f in d.get("features",[]) if f["type"]=="Transmembrane")
    locs=[]
    for c in d.get("comments",[]):
        if c.get("commentType")=="SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations",[]):
                v=(sl.get("location") or {}).get("value")
                if v: locs.append(v)
    return dict(name=name,signal=sig,n_tm=ntm,locations=locs)
# journey templates
def steps_common(gene,chrom):
    return [
     dict(step="gene",loc="nucleus",species="DNA",machinery="chromosome "+(chrom or "?"),desc=f"{gene} gene in the genome"),
     dict(step="transcription",loc="nucleus",species="pre-mRNA (hnRNA)",machinery="RNA Polymerase II",desc="DNA transcribed to hnRNA; 5' cap added"),
     dict(step="splicing",loc="nucleus",species="mRNA",machinery="spliceosome (snRNPs)",desc="introns removed, exons joined; poly-A tail added -> mature mRNA"),
     dict(step="export",loc="nuclear pore",species="mRNA",machinery="nuclear pore complex + NXF1/TAP",desc="mRNA exported to cytoplasm"),
    ]
def build(gene,u,chrom):
    s=steps_common(gene,chrom)
    secretory = u["signal"] or (u["n_tm"]>0 and any(k in " ".join(u["locations"]).lower() for k in ["membrane","secreted","endoplasmic","golgi","lysosom"]))
    L=" ".join(u["locations"]).lower()
    if secretory:
        dest = "plasma membrane" if (u["n_tm"]>0 and "cell membrane" in L) else ("secreted" if "secreted" in L else ("ER/Golgi" if ("endoplasmic" in L or "golgi" in L) else "plasma membrane"))
        s+=[
         dict(step="translation+targeting",loc="ER",species="nascent chain",machinery="ribosome + SRP + SRP receptor",desc="signal peptide emerges -> SRP pauses translation, docks ribosome on the ER"),
         dict(step="translocation",loc="ER",species=("membrane protein" if u["n_tm"]>0 else "protein"),machinery="Sec61 translocon (+ signal peptidase)",desc="co-translational insertion into ER "+("membrane" if u["n_tm"]>0 else "lumen")),
         dict(step="ER folding",loc="ER",species="folded protein",machinery="BiP/calnexin chaperones + N-glycosylation",desc="folding + glycosylation, quality control"),
         dict(step="ER->Golgi",loc="Golgi",species="cargo",machinery="COPII vesicles",desc="vesicle transport to the Golgi"),
         dict(step="Golgi",loc="Golgi",species="mature glycoprotein",machinery="Golgi glycosyltransferases",desc="glycan maturation + sorting"),
         dict(step="Golgi->surface",loc=("plasma membrane" if dest=="plasma membrane" else "extracellular"),species="cargo vesicle",machinery="secretory vesicle + SNAREs (exocytosis)",desc=f"delivered to {dest}"),
         dict(step="FINAL",loc=("plasma membrane" if dest=="plasma membrane" else ("extracellular" if dest=="secreted" else "ER/Golgi")),species=u["name"] or gene,machinery="",desc=f"final location: {dest}"),
        ]; route="secretory"
    elif (u["locations"] and "nucleus" in u["locations"][0].lower()) or gene in TFset:
        s+=[dict(step="translation",loc="cytoplasm",species="protein",machinery="free ribosome",desc="translated in cytosol"),
            dict(step="nuclear import",loc="nuclear pore",species="protein",machinery="importin-a/b (NLS)",desc="NLS recognized, imported through nuclear pore"),
            dict(step="FINAL",loc="nucleus",species=u["name"] or gene,machinery="",desc="final location: nucleus")]; route="nuclear"
    elif "mitochond" in (u["locations"][0].lower() if u["locations"] else ""):
        s+=[dict(step="translation",loc="cytoplasm",species="precursor",machinery="free ribosome",desc="translated with mitochondrial targeting sequence (MTS)"),
            dict(step="mito import",loc="mitochondrion",species="protein",machinery="TOM/TIM translocase + MPP",desc="imported through outer/inner membranes; MTS cleaved"),
            dict(step="FINAL",loc="mitochondrion",species=u["name"] or gene,machinery="",desc="final location: mitochondrion")]; route="mitochondrial"
    else:
        s+=[dict(step="translation",loc="cytoplasm",species="protein",machinery="free ribosome",desc="translated on free ribosomes in the cytosol"),
            dict(step="FINAL",loc="cytoplasm",species=u["name"] or gene,machinery="",desc="final location: cytosol")]; route="cytosolic"
    return route,s
# gene coordinates
import gzip
chrom={}
for l in gzip.open("data/external_data/human/refGene.txt.gz","rt"):
    p=l.split("\t")
    if len(p)>=13 and "_" not in p[2]: chrom.setdefault(p[12],p[2])
out={}
for g in GENES:
    ac=acc.get(g)
    if not ac: print(f"{g}: no acc"); continue
    u=uni(ac)
    if not u: print(f"{g}: no uniprot"); continue
    route,steps=build(g,u,chrom.get(g))
    out[g]=dict(acc=ac,name=u["name"],locations=u["locations"],signal=u["signal"],n_tm=u["n_tm"],route=route,chrom=chrom.get(g),steps=steps)
    print(f"  {g:8s} {route:12s} loc={u['locations'][:2]} TM={u['n_tm']} sig={u['signal']}",flush=True)
    time.sleep(0.3)
json.dump(out,open(OUT/"cell_trafficking.json","w"),indent=2)
print(f"\nwrote cell_trafficking.json ({len(out)} genes)")
