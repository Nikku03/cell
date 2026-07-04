"""CONCENTRATION layer — absolute protein/TF concentration from genome + condition (human port of the
E. coli beta = intrinsic x master-program method).

The E. coli chain was: beta(gene,cond) = intrinsic_beta(promoter seq) x Sum master-program activities.
Human port: we DON'T predict the intrinsic term from promoter sequence (human promoter grammar is far
harder); we MEASURE it (PaxDb abundance / cell-type expression). We keep the master-program multiplier
(condition sensors -> regulon shift) and dilution. So:

  [protein, condition] (nM) = baseline_abundance(PaxDb ppm -> absolute) x condition_multiplier / dilution

Then the payoff:
  TF occupancy      = [TF] / ([TF] + Kd)            -> quantitative regulation (which targets are ON)
  reaction velocity = kcat x [E] x [S]/(Km+[S])     -> real in-vivo rate (uses the kinetics layer)

Absolute conversion anchors (human cell, order-of-magnitude, flagged): ~2.3e9 protein copies/cell,
~2000 um^3 volume. -> concentration.json  {gene:{baseline_nM, tier, per_condition, occupancy?}}
"""
import json, math
from collections import defaultdict
from pathlib import Path
OUT=Path("outputs/orphan")
TOTAL_PROTEIN=2.3e9          # copies/cell (BioNumbers, human, median) — anchor
CELL_VOL_L=2.0e-12          # ~2000 um^3
AVOGADRO=6.022e23
KD_TF_nM=float(__import__("os").environ.get("TF_KD_NM","20"))   # typical TF-DNA Kd ~ nM
FOLD=float(__import__("os").environ.get("COND_FOLD","3.0"))     # default per-program fold-change magnitude

def ppm_to_nM(ppm):
    copies=ppm*1e-6*TOTAL_PROTEIN
    return copies/(CELL_VOL_L*AVOGADRO)*1e9

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]; N=len(G)
    idx={g["name"]:i for i,g in enumerate(G)}
    ppm={int(k):v for k,v in D.get("ppm",{}).items()}
    # cell-type expression proxy (abund 0-15) as fallback baseline where PaxDb missing
    abund={int(k):v for k,v in D.get("abund",{}).items()}
    tf=set(i for i,g in enumerate(G) if g.get("tf"))
    cond=json.load(open(OUT/"conditions.json")) if (OUT/"conditions.json").exists() else {}
    kin=(json.load(open(OUT/"kinetics.json")).get("kinetics",{}) if (OUT/"kinetics.json").exists() else {})
    # median ppm among knowns (to scale the abund fallback onto the same nM axis)
    med_ppm=sorted(ppm.values())[len(ppm)//2] if ppm else 50.0

    out={}
    for i,g in enumerate(G):
        if i in ppm:
            base_nM=ppm_to_nM(ppm[i]); tier="measured-abundance(PaxDb)"
        elif i in abund and abund[i]>0:
            base_nM=ppm_to_nM(med_ppm*(2**(abund[i]-7)))   # abund is a log2-ish 0-15 scale -> relative ppm
            tier="expression-proxy"
        else:
            continue
        rec=dict(baseline_nM=round(base_nM,3), tier=tier, log10_nM=round(math.log10(base_nM+1e-9),2))
        if i in tf: rec["is_tf"]=True
        out[g["name"]]=rec

    # condition multipliers: [protein,cond] = baseline x (FOLD if in up-set, 1/FOLD if down-set)
    for cname,cv in cond.items():
        upset=set(cv.get("up",[])); dnset=set(cv.get("down",[]))
        for g in (upset|dnset):
            if g in out:
                m=FOLD if g in upset else 1.0/FOLD
                out[g].setdefault("per_condition",{})[cname]=round(out[g]["baseline_nM"]*m,3)

    # TF occupancy at baseline (quantitative regulation: is this TF above its Kd?)
    for g,rec in out.items():
        if rec.get("is_tf"):
            c=rec["baseline_nM"]; rec["occupancy"]=round(c/(c+KD_TF_nM),3)

    # reaction velocity where we have kcat (Vmax-ish; substrate-saturation approximated as 0.5 without [S])
    nvel=0
    for g,rec in out.items():
        k=kin.get(g)
        if k and k.get("kcat_per_s"):
            E_M=rec["baseline_nM"]*1e-9
            rec["velocity_rel_per_s"]=round(k["kcat_per_s"]*E_M*0.5,6)   # kcat*[E]*saturation(~0.5); rel in-vivo rate
            rec["velocity_kcat_tier"]=k.get("tier"); nvel+=1

    payload=dict(concentration=out,
                 anchors=dict(total_protein_copies=TOTAL_PROTEIN, cell_volume_L=CELL_VOL_L, TF_Kd_nM=KD_TF_nM, cond_fold=FOLD),
                 summary=dict(genes=len(out), from_paxdb=sum(1 for r in out.values() if "PaxDb" in r["tier"]),
                              from_expression=sum(1 for r in out.values() if r["tier"]=="expression-proxy"),
                              tfs_with_occupancy=sum(1 for r in out.values() if "occupancy" in r),
                              with_velocity=nvel))
    json.dump(payload, open(OUT/"concentration.json","w"))
    s=payload["summary"]
    print(f"concentration: {s['genes']} genes ({s['from_paxdb']} PaxDb + {s['from_expression']} expression-proxy) "
          f"-> absolute nM | {s['tfs_with_occupancy']} TFs with occupancy | {s['with_velocity']} enzymes with velocity")
    print("  method: measured baseline x master-program condition multiplier (E. coli port); absolute conversion is order-of-magnitude (flagged)")

if __name__=="__main__":
    main()
