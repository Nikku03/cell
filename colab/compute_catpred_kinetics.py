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
    """metabolite name (lower) -> SMILES, from CatPred-DB's metabolite_inchi_smiles table (fetched if absent)."""
    m={}
    prim=H/"metabolite_inchi_smiles_brenda_pubchem.tsv"
    if not (prim.exists() and prim.stat().st_size>1000):
        url=os.environ.get("CATPRED_SMILES_URL",
            "https://raw.githubusercontent.com/maranasgroup/CatPred-DB/main/datasets/metabolite_inchi_smiles_brenda_pubchem.tsv")
        try:
            print("  fetching CatPred-DB metabolite SMILES map ..."); urllib.request.urlretrieve(url, prim)
        except Exception as e:
            print("  SMILES map fetch failed:",repr(e)[:100],"-> CatPred substrates cannot be mapped")
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
    # CatPred's sequence validator accepts ONLY the 20 standard amino acids and ABORTS THE WHOLE RUN on the
    # first violation (e.g. a selenocysteine 'U'). Sanitize every sequence up front: substitute the two
    # translated non-standard residues (U=selenocysteine->Cys, O=pyrrolysine->Lys) and drop any sequence that
    # still contains a non-standard char (B/Z/J/X/* etc.) so no invalid row ever reaches CatPred.
    _VALID=set("ACDEFGHIKLMNPQRSTVWY")
    def clean_seq(s):
        if not s: return None
        s=s.strip().upper().replace("U","C").replace("O","K")
        return s if (s and all(ch in _VALID for ch in s)) else None
    rows=[]; n_dropped=0
    for g in G:
        nm=g["name"]; ac=g2a.get(nm); seq=clean_seq(seqs.get(ac))
        rxns=g2r.get(nm, [])
        if seqs.get(ac) and not seq: n_dropped+=1
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
        # representative substrate with a SMILES (skip hubs; try all substrates, chirality-insensitive)
        subs=[mets[i] for i in r.get("s",[]) if i<len(mets)]
        def find_smi(name):
            n=name.lower(); return sm.get(n) or sm.get(re.sub(r"^[ld]-","",n)) or sm.get("d-"+n) or sm.get("l-"+n)
        smi=next((find_smi(s) for s in subs if s not in HUB and find_smi(s)), None)
        if not smi: continue
        # CatPred uses pdbpath ONLY as a unique per-sequence KEY for its protein-records JSON (name+seq); the
        # inference path does NOT read the file. But it REQUIRES a non-empty pdbpath with a unique basename per
        # sequence and errors on empty cells (which pandas reads as NaN -> os.path.basename(float) crash). Key it
        # on the UniProt accession (1:1 with the sequence): a real AlphaFold model if present, else '{ac}.pdb'.
        pdb=f"{ac}.pdb"
        if AF.exists():
            p=AF/f"AF-{ac}-F1-model_v4.pdb"
            if p.exists(): pdb=str(p)
        rows.append(dict(SMILES=smi, sequence=seq, pdbpath=pdb, gene=nm, reaction=r.get("id")))
    with open(OUT/"catpred_input.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["SMILES","sequence","pdbpath","gene","reaction"]); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"CatPred input prepared: {len(rows)} enzyme-reaction rows -> catpred_input.csv "
          f"({sum(1 for r in rows if r['pdbpath'])} with AlphaFold structure"
          f"{f', {n_dropped} dropped for non-standard residues' if n_dropped else ''})")
    return len(rows)

def parse_output():
    """parse CatPred result CSVs -> catpred_kinetics.json. Handles BOTH CatPred output formats:
      - POSTPROCESSED (results/): 'Prediction_(s^(-1))'/'Prediction_(mM)' (linear) + 'SD_total' (log10 units).
      - RAW (predict.py --preds_path): 'log10kcat_max'/'log10km_mean' (log10) + '<target>_mve_uncal_var'.
    Value = linear kcat (1/s) / Km (mM->uM); uncertainty is the log10-scale SD refine_kinetics blends on.
    Rows keyed by 'sequence' -> gene via catpred_input.csv."""
    import glob, math
    seq2gene={}
    inp=OUT/"catpred_input.csv"
    if inp.exists():
        for r in csv.DictReader(open(inp)):
            if r.get("sequence") and r.get("gene"): seq2gene[r["sequence"]]=r["gene"]
    km_to_uM=float(os.environ.get("CATPRED_KM_TO_UM","1000"))    # CatPred Km is mM -> uM
    TARGET={"kcat":"log10kcat_max","km":"log10km_mean"}
    def _hdr_ok(path, param):
        try: h=open(path).readline().lower()
        except OSError: return False
        return ("prediction_(" in h) or (TARGET[param] in h)     # postprocessed OR raw target column
    def find_result(param, envk):
        p=os.environ.get(envk)
        if p and os.path.exists(p) and _hdr_ok(p, param): return p
        cands=set()
        for pat in ["CatPred/results/**/*.csv","CatPred/../results/**/*.csv","results/**/*.csv",f"{OUT}/*.csv"]:
            cands.update(glob.glob(pat, recursive=True))
        hits=[c for c in cands if param in c.lower()
              and not (param=="km" and "kcat" in c.lower() and "km" not in c.lower().replace("kcat",""))
              and _hdr_ok(c, param)]
        return sorted(hits)[-1] if hits else None
    def row_value(r, param):
        """(linear value, log10-SD) from either format, or (None,None)."""
        tcol=TARGET[param]; var_col=tcol+"_mve_uncal_var"
        lin=next((r[c] for c in r if c.lower().startswith("prediction_(") and r[c] not in (None,"")), None)
        val=None
        if lin not in (None,""):
            try: val=float(lin)
            except ValueError: val=None
        for logc in (r.get("Prediction_log10"), r.get(tcol)):
            if val is None and logc not in (None,""):
                try: val=10**float(logc)
                except ValueError: pass
        sd=r.get("SD_total") or r.get("SD_epistemic")
        if not sd and r.get(var_col) not in (None,""):
            try: sd=math.sqrt(max(float(r[var_col]),0.0))
            except ValueError: sd=None
        return val, sd
    out={}
    for param, envk in [("kcat","CATPRED_KCAT_OUT"), ("km","CATPRED_KM_OUT")]:
        f=find_result(param, envk)
        if not f: continue
        n=0
        for r in csv.DictReader(open(f)):
            g=seq2gene.get(r.get("sequence"))
            if not g: continue
            val, sd = row_value(r, param)
            if val is None or val<=0: continue
            rec=out.setdefault(g, {})
            if param=="kcat": rec["kcat"]=val
            else: rec["km"]=val*km_to_uM
            if sd:
                try: rec[param+"_unc"]=float(sd)
                except (TypeError,ValueError): pass
            n+=1
        print(f"  parsed {n} {param} predictions from {f}")
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
