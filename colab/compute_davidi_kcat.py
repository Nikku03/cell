"""DAVIDI in-vivo kcat (MAX-over-conditions) — the legitimate way to get kcat from flux data.

kcat_app,i,c = v_i,c / [E]_i is enzyme UTILIZATION in condition c, and it is <= kcat: most enzymes run
below capacity, so a SINGLE condition underestimates kcat (this is why single-condition flux-deconvolution
scored ~48x in refine_kinetics). Davidi et al. 2016 (PNAS) recover kcat by taking the MAXIMUM over many
conditions -- the condition where an enzyme is most utilized reveals its true turnover:
        kcat_max,i = max_c ( |v_i,c| / ([E]_i * VMAX_CONV) )                    [1/s]

FluxProfilingREGP measures exchange fluxes for ~13 conditions (MCF10A CORE/REGP + 11 NCI-60 lines). We run
ecFBA per condition (each condition's measured exchanges as the boundary), read internal fluxes via pFBA,
and take the per-enzyme max apparent turnover. Then we VALIDATE against our measured kcat (held-out is not
needed -- these are independent measurements) and report median fold-error vs the 9.6x imputed baseline.

Honesty tiers / limitations:
  - internal fluxes are FBA-INFERRED (only the exchanges are measured), so kcat_max is model-anchored, not
    a direct measurement; the max-over-conditions still removes most of the single-condition underestimation.
  - enzyme abundance [E]_i is a SINGLE PaxDb profile (we lack per-condition proteomics), so cross-condition
    variation comes only from flux, not from E. Real Davidi varies both. Noted, not hidden.

Reuses compute_flux (parse + solver). Reads concentration.json ([E]) + kinetics.json (validation).
-> davidi_kcat.json {gene:{kcat_max_per_s, n_active_conditions, best_condition, reaction, ...}}
"""
import json, os, math, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
OUT=Path("outputs/orphan")
CELL_VOL_L=float(os.environ.get("CELL_VOL_L","2.0e-12")); GDW_PER_CELL=float(os.environ.get("GDW_PER_CELL","3.0e-10"))
VMAX_CONV=1e-9*CELL_VOL_L/GDW_PER_CELL*1000.0*3600.0
TOL=float(os.environ.get("FLUX_TOL","0.3"))
RAW=OUT/"_fluxprofile_raw.csv"
REGP_URL=os.environ.get("FLUXPROFILE_URL",
    "https://raw.githubusercontent.com/Radboud-YuLab/FluxProfilingREGP/main/data/exchange_fluxes/output/exch_fluxes_MCF10A_NCI60.csv")

def load_conditions():
    """the FluxProfilingREGP CSV -> {condition_name: {metabolite: flux}}. Columns other than 'metabolite'
    are flux conditions (CORE, REGP = MCF10A methods; the rest are NCI-60 cell lines)."""
    if not (RAW.exists() and RAW.stat().st_size>200):
        try:
            import urllib.request; print("  fetching FluxProfilingREGP conditions ..."); urllib.request.urlretrieve(REGP_URL, RAW)
        except Exception as e:
            print("  FluxProfilingREGP fetch failed:",repr(e)[:100]); return {}
    rows=list(csv.DictReader(open(RAW)))
    if not rows: return {}
    conds=[c for c in rows[0] if c and c.lower()!="metabolite"]
    out={c:{} for c in conds}
    for r in rows:
        m=(r.get("metabolite") or "").strip()
        if not m: continue
        for c in conds:
            try: out[c][m]=float(r[c])
            except (TypeError,ValueError): pass
    return out

def main():
    try:
        import scipy  # noqa
        from compute_flux import (parse_gem, build_S, solve_fba, pfba, met_to_exchange, mx_get,
                                   vmax_bounds, find_biomass, BIG)
    except Exception as e:
        print("Davidi kcat needs scipy + compute_flux:",repr(e)[:90]); return
    cf=OUT/"concentration.json"
    if not cf.exists(): print("Davidi kcat needs concentration.json ([E]) -> skipping"); return
    conc=json.load(open(cf)).get("concentration",{})
    kin=(json.load(open(OUT/"kinetics.json")).get("kinetics",{}) if (OUT/"kinetics.json").exists() else {})
    conds=load_conditions()
    if len(conds)<3: print(f"Davidi kcat needs >=3 flux conditions (got {len(conds)}) -> skipping"); return

    rxns, met_idx = parse_gem()
    if not rxns: print("Human-GEM absent -> Davidi kcat skipped (runs on Colab)"); return
    n=len(rxns); S=build_S(rxns,len(met_idx)); met2ex=met_to_exchange(rxns, met_idx)
    bio=find_biomass(rxns)
    if bio is None: print("no biomass reaction -> Davidi kcat skipped"); return
    vmax,_,_=vmax_bounds(rxns)
    gene2rxn=defaultdict(list)
    for j,r in enumerate(rxns):
        for gsym in r["genes"]: gene2rxn[gsym].append(j)
    c=np.zeros(n); c[bio]=1.0
    base_lb=np.array([-BIG if r["rev"] else 0.0 for r in rxns]); base_ub=np.array([BIG]*n)

    def with_vmax(scale):
        lb=base_lb.copy(); ub=base_ub.copy()
        for ri,cap in vmax.items():
            ub[ri]=min(ub[ri], cap*scale)
            if rxns[ri]["rev"]: lb[ri]=max(lb[ri], -cap*scale)
        return lb,ub

    # per-condition ecFBA -> internal flux distribution (pFBA). Same open-medium + Vmax anchor as compute_flux.
    kcat_app=defaultdict(dict)     # gene -> {condition: kcat_app}
    grew=[]
    for cname, mets in conds.items():
        lb0,ub0=with_vmax(1)       # start at Vmax x1 (physiological); relax only if infeasible
        # overlay this condition's measured exchange bounds
        for m,f in mets.items():
            mx=mx_get(met2ex, m)
            if not mx: continue
            j,sign=mx
            if f<0:                 # uptake availability ceiling (like compute_flux, no forced band)
                lo,hi=(round(f*(1+TOL),6),0.0) if sign<0 else (0.0, round(-f*(1+TOL),6))
            else:                   # secretion allowance
                lo,hi=(0.0, round(f*(1+TOL),6)) if sign<0 else (round(-f*(1+TOL),6), 0.0)
            lb0[j]=lo; ub0[j]=hi
        z=None; lb=ub=None
        for scale in (1,10,100,1000):
            lb,ub=with_vmax(scale)
            for m,f in mets.items():   # re-apply exchange bounds at this scale
                mx=mx_get(met2ex,m)
                if not mx: continue
                j,sign=mx
                if f<0: lo,hi=(round(f*(1+TOL),6),0.0) if sign<0 else (0.0, round(-f*(1+TOL),6))
                else:   lo,hi=(0.0, round(f*(1+TOL),6)) if sign<0 else (round(-f*(1+TOL),6), 0.0)
                lb[j]=lo; ub[j]=hi
            z=solve_fba(S,lb,ub,c,True)
            if z is not None and float(z[bio])>1e-4: break
            z=None
        if z is None: continue
        zstar=float(z[bio]); grew.append((cname,zstar))
        v=pfba(S,lb,ub,c,zstar) if zstar>1e-9 else z
        if v is None: v=z
        for g,ris in gene2rxn.items():
            E=conc.get(g,{}).get("baseline_nM")
            if not E or E<=0: continue
            vg=max((abs(float(v[ri])) for ri in ris), default=0.0)     # gene's most-active reaction here
            if vg<1e-9: continue
            kcat_app[g][cname]=vg/(E*VMAX_CONV)

    if not grew:
        print("Davidi kcat: no condition produced growth -> skipping"); return

    # per-enzyme MAX over conditions (the Davidi estimate). Also keep the median for a saturation read.
    out={}
    for g, byc in kcat_app.items():
        vals=sorted(byc.values())
        kmax=vals[-1]; best=max(byc, key=byc.get)
        rec=dict(kcat_max_per_s=round(kmax,4), kcat_median_per_s=round(float(np.median(vals)),4),
                 n_active_conditions=len(vals), best_condition=best)
        kv=kin.get(g,{}).get("kcat_per_s")
        if kv and kv>0:
            rec["kcat_measured_per_s"]=round(kv,3); rec["measured_tier"]=kin[g].get("tier")
            rec["max_vs_measured"]=round(kmax/kv,3)
        out[g]=rec

    # validation: compare kcat_max to MEASURED kcat (independent data -> no hold-out needed).
    pairs=[(r["kcat_max_per_s"], r["kcat_measured_per_s"]) for r in out.values()
           if r.get("measured_tier")=="measured" and r.get("kcat_measured_per_s")]
    val=None
    if len(pairs)>=5:
        errs=[abs(math.log10(p/m)) for p,m in pairs if p>0 and m>0]
        # single-condition baseline: median apparent (utilization) vs measured, same enzymes
        base=[abs(math.log10(np.median(list(kcat_app[g].values()))/out[g]["kcat_measured_per_s"]))
              for g in out if out[g].get("measured_tier")=="measured" and kcat_app.get(g)]
        val=dict(n=len(errs), median_fold_error=round(10**float(np.median(errs)),2),
                 within_2x=round(sum(1 for e in errs if e<=math.log10(2))/len(errs),2),
                 within_3x=round(sum(1 for e in errs if e<=math.log10(3))/len(errs),2),
                 single_condition_fold_error=round(10**float(np.median(base)),2) if base else None,
                 imputed_baseline_fold_error=9.61, under_over=round(float(np.median(
                     [math.log10(p/m) for p,m in pairs if p>0 and m>0])),3))
    payload=dict(davidi_kcat=out,
                 conditions=[{"name":n_,"biomass_per_hr":round(z_,4)} for n_,z_ in grew],
                 validation=val,
                 summary=dict(enzymes=len(out), n_conditions=len(grew),
                              with_measured=sum(1 for r in out.values() if r.get("kcat_measured_per_s")),
                              note="kcat_max = max_condition |v|/([E]*conv); Davidi 2016 max-over-conditions in-vivo kcat"))
    json.dump(payload, open(OUT/"davidi_kcat.json","w"))
    s=payload["summary"]
    print(f"Davidi kcat: {s['enzymes']} enzymes over {s['n_conditions']} conditions "
          f"({', '.join(nm for nm,_ in grew[:6])}{'...' if len(grew)>6 else ''}) | {s['with_measured']} vs measured")
    if val:
        print(f"  >>> max-over-conditions vs measured: median fold-error {val['median_fold_error']}x "
              f"(single-condition {val['single_condition_fold_error']}x, imputed baseline {val['imputed_baseline_fold_error']}x) "
              f"| within-2x {val['within_2x']:.0%}, within-3x {val['within_3x']:.0%} (n={val['n']})")
        bias="under-estimate" if val["under_over"]<-0.05 else ("over-estimate" if val["under_over"]>0.05 else "unbiased")
        print(f"  median log10(kcat_max/measured) = {val['under_over']:+.2f} ({bias}; Davidi expects ~0 to slightly negative)")

if __name__=="__main__":
    main()
