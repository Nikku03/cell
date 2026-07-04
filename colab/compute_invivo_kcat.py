"""IN-VIVO kcat (a data-generating play: X + Y -> Z where Z barely exists).

We now have flux v (absolute, mmol/gDW/hr) AND enzyme abundance [E] (PaxDb nM). Their ratio is the
IN-VIVO APPARENT turnover number:
        kcat_app [1/s] = v / ([E] * VMAX_CONV)          (Davidi et al. 2016, ported to human)
Human in-vivo kcat essentially does not exist in any database -- it's normally only obtainable by pairing
fluxomics with proteomics. We already have both layers, so we generate it for free, and compare it to the
in-vitro kcat we imputed:
    saturation ratio = kcat_app / kcat_in_vitro
      ~1   -> enzyme runs near capacity (a real bottleneck)
      <<1  -> large unused reserve (enzyme is regulated / inhibited / over-expressed)
      >1   -> our in-vitro kcat is an underestimate for this enzyme (flags bad imputed values)
This is also a self-consistency check on the kinetics + flux + concentration layers at once.

Reads flux.json, concentration.json, kinetics.json, metabolic_graph.json. Skips gracefully.
-> invivo_kcat.json {gene:{kcat_app_per_s, kcat_invitro_per_s, saturation_ratio, tier}}
"""
import json, math
from pathlib import Path
OUT=Path("outputs/orphan")
# same physical conversion as compute_flux: kcat[1/s]*[E][nM] * VMAX_CONV = flux[mmol/gDW/hr]
import os
CELL_VOL_L=float(os.environ.get("CELL_VOL_L","2.0e-12")); GDW_PER_CELL=float(os.environ.get("GDW_PER_CELL","3.0e-10"))
VMAX_CONV=1e-9*CELL_VOL_L/GDW_PER_CELL*1000.0*3600.0

def main():
    ff=OUT/"flux.json"; cf=OUT/"concentration.json"; kf=OUT/"kinetics.json"; mg=OUT/"metabolic_graph.json"
    if not (ff.exists() and cf.exists()):
        print("in-vivo kcat needs flux.json + concentration.json -> skipping"); return
    flux=json.load(open(ff)); fu=flux.get("summary",{}).get("flux_units","")
    fx=flux.get("flux",{})
    conc=(json.load(open(cf)).get("concentration",{}))
    kin=(json.load(open(kf)).get("kinetics",{}) if kf.exists() else {})
    if "absolute" not in fu:
        print(f"in-vivo kcat: flux units are '{fu}' (not absolute) -> kcat_app would be relative; "
              "skipping until the flux map is absolute (needs measured exchange anchor)"); return
    out={}
    for rid,r in fx.items():
        v=abs(r.get("v",0.0))
        if v<1e-9: continue
        for g in r.get("genes",[]):
            E=conc.get(g,{}).get("baseline_nM")
            if not E or E<=0: continue
            kcat_app = v/(E*VMAX_CONV)                     # 1/s
            rec=dict(kcat_app_per_s=round(kcat_app,4), reaction=rid, flux=round(v,4), E_nM=round(E,2))
            kv=kin.get(g,{}).get("kcat_per_s")
            if kv and kv>0:
                rec["kcat_invitro_per_s"]=round(kv,3); rec["kcat_invitro_tier"]=kin[g].get("tier")
                rec["saturation_ratio"]=round(kcat_app/kv,4)
            # keep the highest-flux reaction per gene as the representative
            if g not in out or v>out[g].get("flux",0): out[g]=rec
    # honest tiers on the apparent value
    for g,r in out.items():
        sr=r.get("saturation_ratio")
        if sr is None: r["interpretation"]="in-vivo kcat (no in-vitro to compare)"
        elif sr>2: r["interpretation"]="in-vitro kcat likely underestimate (or flux/abundance error)"
        elif sr>0.3: r["interpretation"]="near capacity — real bottleneck"
        else: r["interpretation"]="large unused reserve — regulated/inhibited"
    both=[r for r in out.values() if "saturation_ratio" in r]
    import statistics
    med_sat=statistics.median([r["saturation_ratio"] for r in both]) if both else None
    payload=dict(invivo_kcat=out, summary=dict(enzymes=len(out), with_invitro_comparison=len(both),
                 median_saturation_ratio=(round(med_sat,3) if med_sat else None),
                 near_capacity=sum(1 for r in both if r["saturation_ratio"]>0.3),
                 large_reserve=sum(1 for r in both if r["saturation_ratio"]<=0.3),
                 note="kcat_app = v/([E]*conv); apparent in-vivo turnover generated from flux x abundance"))
    json.dump(payload, open(OUT/"invivo_kcat.json","w"))
    s=payload["summary"]
    print(f"in-vivo kcat: generated apparent turnover for {s['enzymes']} enzymes ({s['with_invitro_comparison']} "
          f"vs in-vitro) | median saturation {s['median_saturation_ratio']} | {s['near_capacity']} near-capacity, "
          f"{s['large_reserve']} with reserve  [novel human in-vivo kcat dataset]")

if __name__=="__main__":
    main()
