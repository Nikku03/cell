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
    print(f"kinetics anchors: {nhum} human kcat values across {len(ec_vals)} EC numbers -> ec_kcat.tsv")

if __name__=="__main__":
    main()
