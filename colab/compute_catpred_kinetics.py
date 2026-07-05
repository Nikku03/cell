"""CatPred kcat/Km inputs + output parsing (the GPU inference itself runs in a dedicated notebook cell).

CatPred needs, per enzyme-reaction: a substrate SMILES, the enzyme protein sequence, and (optionally) a
3D-structure path. We PREPARE that table from what we already have:
  - gene -> primary metabolic reaction (Human-GEM / metabolic_graph, highest-flux if flux present)
  - reaction -> a representative substrate -> SMILES (CatPred-DB metabolite_inchi_smiles map, name-matched)
  - gene -> UniProt accession (uniprot_acc_ptm.tsv) -> sequence (UniProt human proteome FASTA)
  - gene -> AlphaFold model path (af_human/ dir) if present  ->  CatPred uses structure when available
-> catpred_input.csv  (SMILES, sequence, pdbpath, gene, reaction)

Then a notebook cell clones+installs CatPred and runs it on catpred_input.csv, producing predictions. This
script also PARSES that output (catpred_predictions.csv) back into catpred_kinetics.json {gene:{kcat,km,
kcat_unc,km_unc}} for refine_kinetics. Both halves skip gracefully if their inputs are missing.
"""
import json, os, csv, gzip, re, urllib.request
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan"); AF=H/"af_human"

def uniprot_seqs(accs):
    """accession -> sequence from a human proteome FASTA (fetched once; env UNIPROT_FASTA_URL)."""
    f=H/"uniprot_human.fasta"
    if not (f.exists() and f.stat().st_size>1e6):
        url=os.environ.get("UNIPROT_FASTA_URL",
            "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&format=fasta&compressed=true")
        try:
            print("  fetching UniProt human proteome FASTA ..."); urllib.request.urlretrieve(url, str(f)+".gz")
            import shutil, gzip as gz
            with gz.open(str(f)+".gz","rb") as i, open(f,"wb") as o: shutil.copyfileobj(i,o)
        except Exception as e:
            print("  UniProt FASTA fetch failed:",repr(e)[:100]); return {}
    seq={}; ac=None; buf=[]
    for l in open(f, errors="ignore"):
        if l.startswith(">"):
            if ac and buf: seq[ac]="".join(buf)
            m=re.match(r">\w+\|([^|]+)\|", l); ac=m.group(1) if m else None; buf=[]
        else: buf.append(l.strip())
    if ac and buf: seq[ac]="".join(buf)
    return {a:seq[a] for a in accs if a in seq}

def smiles_map():
    """metabolite name (lower) -> SMILES, from CatPred-DB's metabolite_inchi_smiles table if present."""
    m={}
    for fn in ["metabolite_inchi_smiles_brenda_pubchem.tsv","metabolite_smiles.tsv"]:
        f=H/fn
        if f.exists():
            rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
            ni=next((i for i,h in enumerate(hdr) if "name" in h or "metabolite" in h),0)
            si=next((i for i,h in enumerate(hdr) if "smiles" in h),-1)
            if si<0: continue
            for r in rd:
                if len(r)>max(ni,si) and r[ni] and r[si]: m.setdefault(r[ni].strip().lower(), r[si].strip())
    return m

def gene2acc():
    m={}; f=H/"uniprot_acc_ptm.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); next(rd,None)
        for r in rd:
            if len(r)>=2 and r[0] and r[1]: m.setdefault(r[1].split()[0], r[0])   # gene -> accession
    return m

def prepare():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> CatPred prep skipped"); return 0
    D=json.load(open(cc)); G=D["genes"]
    gem=json.load(open(OUT/"metabolic_graph.json")) if (OUT/"metabolic_graph.json").exists() else {}
    if not gem.get("rxns"): print("metabolic_graph.json absent -> no reactions to score; skipping CatPred prep"); return 0
    mets=gem.get("mets",[]); g2r=gem.get("gene2rxn",{})
    flux=(json.load(open(OUT/"flux.json")).get("flux",{}) if (OUT/"flux.json").exists() else {})
    HUB={"H2O","H+","H","CO2","O2","ATP","ADP","AMP","NAD+","NADH","NADP+","NADPH","Pi","PPi","CoA","NH3"}
    sm=smiles_map(); g2a=gene2acc()
    seqs=uniprot_seqs(set(g2a.get(g["name"],"") for g in G if g["name"] in g2a))
    rows=[]
    for g in G:
        nm=g["name"]; ac=g2a.get(nm); seq=seqs.get(ac)
        rxns=g2r.get(nm, [])
        if not seq or not rxns: continue
        # pick the highest-flux reaction for this gene, else the first
        best=None
        for ri in rxns:
            r=gem["rxns"][ri] if ri<len(gem["rxns"]) else None
            if not r: continue
            v=abs(flux.get(r.get("id"),{}).get("v",0))
            if best is None or v>best[1]: best=(r,v)
        if not best: continue
        r=best[0]
        # representative substrate with a SMILES (skip hubs)
        subs=[mets[i] for i in r.get("s",[]) if i<len(mets)]
        smi=next((sm[s.lower()] for s in subs if s not in HUB and s.lower() in sm), None)
        if not smi: continue
        pdb=""
        if AF.exists():
            p=AF/f"AF-{ac}-F1-model_v4.pdb"
            if p.exists(): pdb=str(p)
        rows.append(dict(SMILES=smi, sequence=seq, pdbpath=pdb, gene=nm, reaction=r.get("id")))
    with open(OUT/"catpred_input.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["SMILES","sequence","pdbpath","gene","reaction"]); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"CatPred input prepared: {len(rows)} enzyme-reaction rows -> catpred_input.csv "
          f"({sum(1 for r in rows if r['pdbpath'])} with AlphaFold structure)")
    return len(rows)

def parse_output():
    """parse CatPred per-parameter result CSVs -> catpred_kinetics.json. CatPred output columns are
    'Prediction_(s^(-1))' (kcat, 1/s) or 'Prediction_(...)' (Km) + 'SD_total'; rows keyed by 'sequence'.
    We map sequence -> gene via catpred_input.csv. Km is converted mM -> uM to match our [S] (env CATPRED_KM_UNIT)."""
    import glob
    seq2gene={}
    inp=OUT/"catpred_input.csv"
    if inp.exists():
        for r in csv.DictReader(open(inp)):
            if r.get("sequence") and r.get("gene"): seq2gene[r["sequence"]]=r["gene"]
    km_to_uM=float(os.environ.get("CATPRED_KM_TO_UM","1000"))    # CatPred Km assumed mM -> uM
    def find_result(param, envk):
        p=os.environ.get(envk)
        if p and os.path.exists(p): return p
        cands=set()
        for pat in ["CatPred/results/**/*.csv","CatPred/../results/**/*.csv","results/**/*.csv",f"{OUT}/*.csv"]:
            cands.update(glob.glob(pat, recursive=True))
        # require the parameter to appear in the file PATH (CatPred separates kcat/km by dir or filename),
        # so kcat and km results never collide, and the file must carry a Prediction column
        hits=[]
        for c in cands:
            low=c.lower()
            if param not in low: continue
            if param=="km" and ("kcat" in low and "km" not in low.replace("kcat","")): continue
            try:
                if "prediction_(" in open(c).readline().lower(): hits.append(c)
            except OSError: pass
        return sorted(hits)[-1] if hits else None
    out={}
    for param, envk in [("kcat","CATPRED_KCAT_OUT"), ("km","CATPRED_KM_OUT")]:
        f=find_result(param, envk)
        if not f: continue
        for r in csv.DictReader(open(f)):
            g=seq2gene.get(r.get("sequence"))
            if not g: continue
            pred=next((r[c] for c in r if c.lower().startswith("prediction_(")), r.get("Prediction_log10"))
            try: val=float(pred)
            except (TypeError,ValueError): continue
            rec=out.setdefault(g, {}); sd=r.get("SD_total") or r.get("SD_epistemic")
            if param=="kcat": rec["kcat"]=val
            else: rec["km"]=val*km_to_uM
            if sd:
                try: rec[param+"_unc"]=float(sd)
                except ValueError: pass
    if not out: return False
    json.dump(dict(catpred=out, n=len(out)), open(OUT/"catpred_kinetics.json","w"))
    print(f"CatPred predictions parsed: {len(out)} enzymes -> catpred_kinetics.json "
          f"({sum(1 for v in out.values() if 'kcat' in v)} kcat, {sum(1 for v in out.values() if 'km' in v)} Km)")
    return True

def main():
    if not parse_output():                # if predictions exist, parse them; else prepare inputs
        prepare()
        print("  (run CatPred on catpred_input.csv in the GPU cell -> catpred_predictions.csv, then re-run to parse)")

if __name__=="__main__":
    main()
