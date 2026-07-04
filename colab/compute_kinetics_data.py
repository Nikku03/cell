"""Assemble MEASURED human kcat anchors from public sources.

DLKcat's curated dataset has ~2,400 human kcat values keyed by EC number (from BRENDA/SABIO, in-vitro).
We aggregate them per EC -> ec_kcat.tsv (EC -> median human kcat), which seeds the enzyme-family priors
in compute_kinetics with REAL measured values instead of guesses. Gene-level, condition-tagged anchors
(SABIO-RK with pH/temp; eHMN) go in kinetics_measured.tsv (env-overridable, added when available).
-> ec_kcat.tsv
"""
import json, os, statistics, urllib.request
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def main():
    dl=H/"dlkcat.json"
    if not (dl.exists() and dl.stat().st_size>1e6):
        url=os.environ.get("DLKCAT_URL","https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/DeeplearningApproach/Data/database/Kcat_combination_0918.json")
        try:
            print("downloading DLKcat kcat dataset ..."); urllib.request.urlretrieve(url, dl)
        except Exception as e:
            print("DLKcat fetch failed:",repr(e)[:120],"-> skipping kinetics anchors"); return
    try:
        d=json.load(open(dl))
    except Exception as e:
        print("DLKcat unreadable:",e); return
    ec_vals=defaultdict(list); nhum=0
    for e in d:
        if not isinstance(e,dict) or "sapiens" not in str(e.get("Organism","")).lower(): continue
        try: v=float(e["Value"])
        except: continue
        ec=str(e.get("ECNumber","")).strip()
        if v>0 and ec and ec[0].isdigit(): ec_vals[ec].append(v); nhum+=1
    with open(OUT/"ec_kcat.tsv","w") as f:
        f.write("ec\tmedian_kcat\tn\tsource\n")
        for ec,vs in sorted(ec_vals.items()):
            f.write(f"{ec}\t{statistics.median(vs):.4g}\t{len(vs)}\tDLKcat(human,in-vitro)\n")
    print(f"EC-level anchors: {nhum} human kcat values across {len(ec_vals)} EC numbers -> ec_kcat.tsv")
    # --- CatPred-DB: GENE-LEVEL human kcat WITH pH/temperature (the top 'measured' tier) ---
    catpred_gene_anchors()

def uniprot2sym():
    """UniProt accession -> HGNC symbol from uniprot_acc_ptm.tsv (accession, gene_primary, ...)."""
    import csv
    m={}
    for fn in ["uniprot_acc_ptm.tsv","uniprot_ec.tsv"]:
        f=H/fn
        if f.exists():
            rd=csv.reader(open(f),delimiter="\t"); next(rd,None)
            for r in rd:
                if len(r)>=2 and r[0] and r[1]: m.setdefault(r[0], r[1].split()[0])
    return m

def catpred_gene_anchors():
    import csv, os, statistics, urllib.request
    f=H/"catpred_kcat.csv"
    if not (f.exists() and f.stat().st_size>1e6):
        url=os.environ.get("CATPRED_KCAT_URL","https://raw.githubusercontent.com/maranasgroup/CatPred-DB/main/datasets/processed/kcat_max_wt_singleSeqs_wpdbs.csv")
        try: print("downloading CatPred-DB kcat ..."); urllib.request.urlretrieve(url, f)
        except Exception as e: print("  CatPred fetch failed:",repr(e)[:100],"-> gene-level anchors skipped"); return
    u2s=uniprot2sym()
    if not u2s: print("  no uniprot->symbol map (uniprot_acc_ptm.tsv) -> CatPred gene mapping skipped"); return
    per_gene=defaultdict(list)   # gene -> [(kcat, ph, temp, in_vitro)]
    for r in csv.DictReader(open(f)):
        if not str(r.get("taxonomy_id","")).startswith("9606"): continue
        g=u2s.get(r.get("uniprot",""))
        try: kc=float(r["value"])
        except: continue
        if not g or kc<=0: continue
        iv = 1 if r.get("sequence_source","").lower()=="brenda" else 0   # BRENDA = in-vitro (flag), SABIO = keep
        per_gene[g].append((kc, r.get("ph","") or "?", r.get("temperature","") or "?", iv, r.get("sequence_source","?")))
    with open(OUT/"kinetics_measured.tsv","w") as out:
        out.write("gene\tkcat\tkm\tph\ttemp\tsource\tin_vitro\n")
        n=0
        for g,vals in per_gene.items():
            # prefer a SABIO (in_vivo-ish, condition-tagged) record; else the median
            invivo=[v for v in vals if v[3]==0]
            rep=min(invivo or vals, key=lambda v:abs(v[0]-statistics.median([x[0] for x in (invivo or vals)])))
            out.write(f"{g}\t{rep[0]:.4g}\t\t{rep[1]}\t{rep[2]}\t{rep[4]}\t{rep[3]}\n"); n+=1
    print(f"GENE-level anchors: {n} human enzymes with measured kcat + conditions (CatPred) -> kinetics_measured.tsv")

if __name__=="__main__":
    main()
