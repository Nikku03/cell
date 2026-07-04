"""CONCENTRATION layer — absolute protein/TF concentration from genome + condition (human port of the
E. coli beta = intrinsic x master-program method).

  [protein, cond, cell-type] (nM) = baseline_abundance x condition_multiplier x cell-type_factor
  baseline: PaxDb (measured) where available; else EXPRESSION-CALIBRATED (fit ppm~expr) x degradation (#2)
  condition_multiplier: master-program (sensor regulon) shift; magnitudes from ARCHS4 fit if present (#1)
  cell-type_factor: per-cell-type expression / mean expression (#4)
  TF occupancy = [TF]/([TF]+Kd), Kd per DNA-binding-domain family (#3)
  velocity = kcat x [E] x saturation (kinetics layer)
-> concentration.json
"""
import json, math, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")
TOTAL_PROTEIN=2.3e9; CELL_VOL_L=2.0e-12; AVOGADRO=6.022e23
FOLD=float(__import__("os").environ.get("COND_FOLD","3.0"))
KD_DEFAULT=float(__import__("os").environ.get("TF_KD_NM","20"))
# #3 typical DNA-binding-domain Kd (nM) by Pfam accession (literature order-of-magnitude)
DBD_KD={"PF00010":5,"PF00170":10,"PF00046":50,"PF00096":150,"PF00105":5,"PF00250":20,
        "PF00505":50,"PF00178":10,"PF00870":10,"PF00554":5,"PF02864":10,"PF00320":15,"PF00104":5}

def ppm_to_nM(ppm): return ppm*1e-6*TOTAL_PROTEIN/(CELL_VOL_L*AVOGADRO)*1e9

def load_celltype_expr(idx):
    """returns (per_gene_mean_expr{i:float}, per_gene_ct{i:{ct:expr}}, ctnames) — orientation-robust."""
    f=OUT/"celltype_expression.csv"
    if not f.exists(): return {},{},[]
    rd=csv.reader(open(f)); header=next(rd); rows=[r for r in rd if r]
    hdr_hits=sum(1 for h in header[1:] if h in idx); row_hits=sum(1 for r in rows if r[0] in idx)
    mean={}; perct=defaultdict(dict); ctnames=[]
    if row_hits>hdr_hits:                          # genes x celltypes
        ctnames=header[1:]
        for r in rows:
            i=idx.get(r[0])
            if i is None: continue
            vals=[float(x) if x else 0.0 for x in r[1:1+len(ctnames)]]
            if any(vals):
                mean[i]=float(np.mean(vals))
                for c,v in zip(ctnames,vals): perct[i][c]=v
    else:                                          # celltypes x genes
        gcols=header[1:]; mat=[]
        for r in rows: ctnames.append(r[0]); mat.append([float(x) if x else 0.0 for x in r[1:]])
        for ci,gname in enumerate(gcols):
            i=idx.get(gname)
            if i is None: continue
            vals=[mat[t][ci] for t in range(len(ctnames)) if ci<len(mat[t])]
            if any(vals):
                mean[i]=float(np.mean(vals))
                for t,c in enumerate(ctnames): perct[i][c]=vals[t] if t<len(vals) else 0.0
    return mean, perct, ctnames

def load_pfam():
    m={}; f=H/"uniprot_domains.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); hdr=[h.lower() for h in next(rd,[])]
        gi=next((i for i,h in enumerate(hdr) if "gene" in h),1); pi=next((i for i,h in enumerate(hdr) if "pfam" in h),-1)
        for r in rd:
            if pi>=0 and gi<len(r) and pi<len(r) and r[gi] and r[pi]:
                d=[x for x in r[pi].replace(",",";").split(";") if x.strip()]
                if d: m[r[gi].split()[0]]=[x.strip() for x in d]
    return m

def load_halflife(idx):
    """optional protein half-life (hours) for #2 degradation weighting: [protein]~mRNA x half-life."""
    hl={}; f=H/"protein_halflife.tsv"
    if f.exists():
        rd=csv.reader(open(f),delimiter="\t"); next(rd,None)
        for r in rd:
            if len(r)>=2 and r[0] in idx:
                try: hl[idx[r[0]]]=float(r[1])
                except: pass
    return hl

def main():
    D=json.load(open(OUT/"cell_complete.json")); G=D["genes"]
    idx={g["name"]:i for i,g in enumerate(G)}
    ppm={int(k):v for k,v in D.get("ppm",{}).items()}
    tf=set(i for i,g in enumerate(G) if g.get("tf"))
    cond=json.load(open(OUT/"conditions.json")) if (OUT/"conditions.json").exists() else {}
    kin=(json.load(open(OUT/"kinetics.json")).get("kinetics",{}) if (OUT/"kinetics.json").exists() else {})
    mult=json.load(open(OUT/"condition_multipliers.json")) if (OUT/"condition_multipliers.json").exists() else {}  # #1 (Colab)
    pfam=load_pfam()
    mean_expr, perct, ctnames = load_celltype_expr(idx)
    hl=load_halflife(idx)

    # #2 CALIBRATE expression -> ppm on genes with BOTH; apply to PaxDb-missing genes
    both=[i for i in ppm if i in mean_expr and mean_expr[i]>0 and ppm[i]>0]   # ppm/expr>0 (log10 needs it)
    calib=None
    if len(both)>=50:
        X=np.log10([mean_expr[i] for i in both]); Y=np.log10([ppm[i] for i in both])
        m=np.isfinite(X)&np.isfinite(Y)
        if m.sum()>=50:
            a,b=np.polyfit(X[m],Y[m],1)
            if np.isfinite(a) and np.isfinite(b): calib=(a,b)
    med_hl=float(np.median(list(hl.values()))) if hl else 1.0

    out={}
    for i,g in enumerate(G):
        if i in ppm:
            base=ppm_to_nM(ppm[i]); tier="measured-abundance(PaxDb)"
        elif calib and i in mean_expr and mean_expr[i]>0:
            pred_ppm=10**(calib[0]*math.log10(mean_expr[i])+calib[1])
            if i in hl: pred_ppm*= (hl[i]/med_hl)              # #2 degradation: longer-lived -> accumulates
            base=ppm_to_nM(pred_ppm); tier="expression-calibrated"
        else:
            continue
        rec=dict(baseline_nM=round(base,3), tier=tier, log10_nM=round(math.log10(base+1e-9),2))
        # #4 per-cell-type concentration = baseline x (expr_in_ct / mean_expr)
        if i in perct and mean_expr.get(i,0)>0:
            pc={c:round(base*(perct[i][c]/mean_expr[i]),2) for c in list(perct[i])[:20] if perct[i][c]>0}
            if pc: rec["per_celltype"]=pc
        if i in tf:
            rec["is_tf"]=True
            kd=next((DBD_KD[d] for d in pfam.get(g["name"],[]) if d in DBD_KD), KD_DEFAULT)   # #3
            rec["Kd_nM"]=kd; rec["occupancy"]=round(base/(base+kd),3)
        out[g["name"]]=rec

    # condition multipliers (#1 magnitudes if fit from ARCHS4, else default direction x FOLD).
    # Apply to the UNION of the sensor-regulon responders AND any gene with a data-fit multiplier for
    # this condition (the ARCHS4 fit measures real shifts beyond the regulon).
    for cname,cv in cond.items():
        up=set(cv.get("up",[])); dn=set(cv.get("down",[]))
        cm={g:v for g,v in mult.get(cname,{}).items() if not g.startswith("_")}
        for gname in (up|dn|set(cm)):
            if gname in out:
                m=cm.get(gname) if gname in cm else (FOLD if gname in up else 1.0/FOLD)
                out[gname].setdefault("per_condition",{})[cname]=round(out[gname]["baseline_nM"]*m,3)

    nvel=0
    for g,rec in out.items():
        k=kin.get(g)
        if k and k.get("kcat_per_s"):
            rec["velocity_rel_per_s"]=round(k["kcat_per_s"]*rec["baseline_nM"]*1e-9*0.5,6)
            rec["velocity_kcat_tier"]=k.get("tier"); nvel+=1

    s=dict(genes=len(out), from_paxdb=sum(1 for r in out.values() if "PaxDb" in r["tier"]),
           from_expr_calibrated=sum(1 for r in out.values() if r["tier"]=="expression-calibrated"),
           with_celltype=sum(1 for r in out.values() if "per_celltype" in r),
           tfs_with_occupancy=sum(1 for r in out.values() if "occupancy" in r),
           tf_specific_kd=sum(1 for r in out.values() if r.get("Kd_nM",KD_DEFAULT)!=KD_DEFAULT),
           with_velocity=nvel, condition_multipliers_fit=any(not k.startswith("_") for k in mult))
    json.dump(dict(concentration=out, summary=s,
                   anchors=dict(total_protein_copies=TOTAL_PROTEIN, cell_volume_L=CELL_VOL_L, cond_fold=FOLD),
                   calibration=("log10ppm=%.2f*log10expr%+.2f"%calib if calib else None)), open(OUT/"concentration.json","w"))
    print(f"concentration: {s['genes']} genes ({s['from_paxdb']} PaxDb + {s['from_expr_calibrated']} expr-calibrated) -> nM")
    print(f"  #2 calibration: {'fit '+('log10ppm=%.2f*log10expr%+.2f'%calib) if calib else 'no expression data (Colab)'} | degradation genes: {len(hl)}")
    print(f"  #3 TF Kd: {s['tfs_with_occupancy']} TFs with occupancy ({s['tf_specific_kd']} with family-specific Kd)")
    print(f"  #4 cell-type: {s['with_celltype']} genes with per-cell-type concentration ({len(ctnames)} cell types)")
    print(f"  #1 condition multipliers: {'FIT from ARCHS4' if mult else 'default direction x%g (run fit_condition_multipliers on Colab)'%FOLD} | velocity: {nvel}")

if __name__=="__main__":
    main()
