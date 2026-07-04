"""CONDITION AWARENESS — what the cell does under a condition.

Each condition (physical: temperature/pH/pressure/osmotic; and biological: hypoxia, oxidative stress,
inflammation, starvation, ER stress, DNA damage, ...) is sensed by a known SENSOR/master regulator.
We pull that sensor's regulon LIVE from the model's regulatory network -> the predicted response gene
set for the condition. This makes the cell model condition-aware: "under heat, HSF1 activates its
regulon; here are the responding genes." Grounded in textbook stress-response biology; the responding
gene set is data-driven from our network. Dark genes appearing in a condition's response set get a
condition-specific functional hint.

Runs on cell_complete.json (regulatory network). -> conditions.json
"""
import json
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")

# condition -> (category, sensor/master genes, one-line mechanism). Sensors are the entry points; the
# response gene set is computed from their regulon in OUR network.
CONDITIONS = {
 "heat / high temperature":      ("physical","HSF1",           "heat-shock response: HSF1 -> chaperones (HSPA/HSPB/DNAJ)"),
 "cold / low temperature":       ("physical","CIRBP,RBM3",     "cold-shock RNA-binding proteins"),
 "acidosis / low pH":            ("physical","ATF4,GPR68,LDHA","acid sensing (OGR1/GPR68) + metabolic/ATF4 stress"),
 "high pressure / mechanical":   ("physical","YAP1,WWTR1",     "mechanotransduction: Hippo (YAP/TAZ), PIEZO channels"),
 "osmotic stress":               ("physical","NFAT5",          "tonicity response (NFAT5/TonEBP)"),
 "hypoxia / low O2":             ("environmental","HIF1A,EPAS1","HIF stabilization -> VEGFA, SLC2A1, glycolysis"),
 "oxidative stress":             ("environmental","NFE2L2",     "NRF2/KEAP1 antioxidant response"),
 "nutrient starvation":          ("metabolic","ATF4,FOXO3,TFEB","integrated stress + autophagy (mTOR/AMPK axis)"),
 "ER stress / unfolded protein": ("cellular","XBP1,ATF6,ATF4", "unfolded-protein response"),
 "DNA damage":                   ("cellular","TP53",           "p53 checkpoint + repair"),
 "inflammation / immune":        ("environmental","NFKB1,STAT1","NF-kB / interferon response"),
 "heavy metal / xenobiotic":     ("environmental","MTF1,AHR",   "metallothionein + xenobiotic (AhR) response"),
}

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]
    idx={g["name"]:i for i,g in enumerate(G)}; dark=set(i for i,g in enumerate(G) if g["dark"])
    out=defaultdict(list); sgn={}
    for e in D.get("reg",[]):
        a,b=e[0],e[1]; out[a].append(b); sgn[(a,b)]=e[2] if len(e)>2 else 0
    conditions={}
    for cond,(cat,sensors,mech) in CONDITIONS.items():
        resp_up=[]; resp_dn=[]; sensor_present=[]; dark_hits=[]
        for s in sensors.split(","):
            si=idx.get(s.strip())
            if si is None: continue
            sensor_present.append(s.strip())
            for t in out.get(si,[]):
                (resp_up if sgn.get((si,t),0)>=0 else resp_dn).append(G[t]["name"])
                if t in dark: dark_hits.append(G[t]["name"])
        conditions[cond]=dict(category=cat, sensors=sensor_present, mechanism=mech,
                              n_response=len(set(resp_up)|set(resp_dn)),
                              up=sorted(set(resp_up))[:40], down=sorted(set(resp_dn))[:40],
                              dark_responders=sorted(set(dark_hits))[:20])
    json.dump(conditions, open(OUT/"conditions.json","w"))
    print(f"condition awareness: {len(conditions)} conditions wired to their sensor regulons")
    for c,v in conditions.items():
        print(f"  {c:32} sensors={v['sensors']} response={v['n_response']} dark-responders={len(v['dark_responders'])}")

if __name__=="__main__":
    main()
