"""REFINE kcat — fuse CatPred predictions with our own data to beat raw CatPred (first-principles).

CatPred predicts in-vitro, standard-condition kcat/Km from sequence+chemistry (~3-4x error, R2~0.5) with a
per-prediction uncertainty. We hold independent signals CatPred cannot see, and each fixes one of its error
sources. The rate law v = kcat*[E]*[S]/(Km+[S]) ties CatPred's outputs to quantities we already have
([E]=PaxDb nM, [S]=metabolite uM, v=absolute FBA flux). Per enzyme we pick the best-grounded kcat:

  TIER 1  measured        -> our 391 BRENDA/SABIO/DLKcat values (fact). Also used to CALIBRATE CatPred bias.
  TIER 2  flux-deconvolved -> for flux-carrying enzymes: kcat = (v/[E]) * (Km+[S])/[S], using CatPred's Km
          (easy to predict) + our [S], with v/[E] the observed in-vivo apparent turnover. This pins kcat to
          a physical measurement (flux) instead of a pure prediction. Cross-checked vs CatPred.
  TIER 3  CatPred(calibrated) -> raw CatPred * global bias-correction fit on the measured overlap, kept when
          its uncertainty is low; else blended with the EC-class prior we already compute.
  Every enzyme also gets Km (CatPred) -> saturation -> a real velocity, upgrading flux from capacity to rate.

Inputs (all optional, degrades gracefully): kinetics.json (measured + EC tiers), catpred_kinetics.json
(kcat/Km/uncertainty from compute_catpred_kinetics), flux.json (absolute v), concentration.json ([E] nM),
metabolites.json ([S] uM). -> kinetics_refined.json  and a self-report of the calibration + tier counts.
"""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")
CELL_VOL_L=2.0e-12; GDW_PER_CELL=3.0e-10
VMAX_CONV=1e-9*CELL_VOL_L/GDW_PER_CELL*1000.0*3600.0     # nM*(1/s) -> mmol/gDW/hr (matches compute_flux)

def load(name, key=None):
    f=OUT/name
    if not f.exists(): return {}
    d=json.load(open(f)); return d.get(key,{}) if key else d

def substrate_conc_for(rxn_id, flux_rec, metab, gem_rxn):
    """representative substrate concentration [S] (uM) for a reaction: min over its non-hub substrates that
    have a concentration (the limiting substrate). Falls back to a typical 100 uM if none mapped."""
    subs=[]
    # gem_rxn maps rxn_id -> list of substrate base-names (built in main from metabolic_graph if present)
    for m in gem_rxn.get(rxn_id, []):
        r=metab.get(m)
        if r and r.get("concentration_uM") and not r.get("hub"): subs.append(r["concentration_uM"])
    return min(subs) if subs else None

def main():
    kin=load("kinetics.json","kinetics")
    cat=load("catpred_kinetics.json","catpred")            # {gene:{kcat,km,kcat_unc,km_unc}}
    flux=load("flux.json"); fx=flux.get("flux",{}); absolute=("absolute" in flux.get("summary",{}).get("flux_units",""))
    conc=load("concentration.json","concentration")
    metab=load("metabolites.json","metabolites")
    # reaction -> substrate base-names, from metabolic_graph if available (for [S] lookup)
    gem=load("metabolic_graph.json"); gem_rxn={}
    if gem.get("rxns"):
        mets=gem.get("mets",[])
        for r in gem["rxns"]:
            gem_rxn[r.get("id")]=[mets[i] for i in r.get("s",[]) if i<len(mets)]
    # gene -> (best flux reaction, v) for the deconvolution
    g2flux={}
    for rid,r in fx.items():
        for g in r.get("genes",[]):
            if g not in g2flux or abs(r.get("v",0))>abs(g2flux[g][1]): g2flux[g]=(rid, r.get("v",0.0))

    # ---- CALIBRATE CatPred bias on the measured overlap (log10 space) ----
    bias=0.0; n_cal=0
    if cat and kin:
        diffs=[]
        for g,k in kin.items():
            if k.get("tier")=="measured" and g in cat and cat[g].get("kcat"):
                mk=k.get("kcat_per_s"); ck=cat[g]["kcat"]
                if mk and mk>0 and ck>0: diffs.append(math.log10(mk)-math.log10(ck))
        if len(diffs)>=8:
            bias=float(np.median(diffs)); n_cal=len(diffs)      # add to log10(CatPred) to match measured
    def cat_cal(g):
        c=cat.get(g,{});
        if not c.get("kcat"): return None
        return 10**(math.log10(c["kcat"])+bias)

    out={}; tiers=Counter(); xchecks=[]
    genes=set(kin)|set(cat)|set(g2flux)
    for g in genes:
        k=kin.get(g,{}); c=cat.get(g,{})
        km = c.get("km")                                    # CatPred Km (uM) if available
        rec=dict(km_uM=km)
        # TIER 1: measured
        if k.get("tier")=="measured" and k.get("kcat_per_s"):
            rec.update(kcat_per_s=round(k["kcat_per_s"],4), tier="measured", source="BRENDA/SABIO/DLKcat")
        else:
            # TIER 2: flux-deconvolved kcat = (v/[E]) * (Km+[S])/[S]
            deconv=None
            if absolute and g in g2flux:
                rid,v=g2flux[g]; E=conc.get(g,{}).get("baseline_nM")
                if E and E>0 and abs(v)>1e-9:
                    app = abs(v)/(E*VMAX_CONV)               # in-vivo apparent kcat (1/s)
                    S = substrate_conc_for(rid, fx.get(rid,{}), metab, gem_rxn)
                    if km and S: deconv = app*(km+S)/S       # remove saturation -> true kcat
                    else: deconv = app                       # no Km/[S]: apparent is the best estimate
            if deconv is not None:
                rec.update(kcat_per_s=round(deconv,4), tier="flux-deconvolved",
                           source="v/[E] * (Km+[S])/[S]" if (km and S) else "v/[E] (apparent)")
                cc=cat_cal(g)
                if cc and cc>0:
                    rec["catpred_kcat"]=round(cc,4); rec["deconv_vs_catpred"]=round(deconv/cc,3)
                    xchecks.append(abs(math.log10(deconv/cc)))
            else:
                # TIER 3: calibrated CatPred, blended with EC-class prior by uncertainty
                cc=cat_cal(g)
                if cc:
                    unc=c.get("kcat_unc")
                    prior=k.get("kcat_per_s") if k.get("tier") in ("EC-measured","EC-class") else None
                    if prior and unc and unc>0.5:            # high uncertainty -> lean on the EC prior
                        kc=10**(0.5*math.log10(cc)+0.5*math.log10(prior))
                        rec.update(kcat_per_s=round(kc,4), tier="catpred+ECprior")
                    else:
                        rec.update(kcat_per_s=round(cc,4), tier="catpred(calibrated)", kcat_unc=unc)
                elif k.get("kcat_per_s"):                    # no CatPred -> keep whatever imputation we had
                    rec.update(kcat_per_s=round(k["kcat_per_s"],4), tier=k.get("tier","imputed"))
                else:
                    continue
        # velocity upgrade: saturation-aware if we have Km + [S]
        if g in g2flux and km:
            rid,_=g2flux[g]; S=substrate_conc_for(rid, fx.get(rid,{}), metab, gem_rxn)
            if S: rec["saturation"]=round(S/(km+S),3)
        tiers[rec["tier"]]+=1; out[g]=rec
    xmed=10**float(np.median(xchecks)) if xchecks else None
    payload=dict(kinetics_refined=out, calibration=dict(catpred_log10_bias=round(bias,3), n_calibration=n_cal,
                 deconv_vs_catpred_median_fold=(round(xmed,2) if xmed else None)),
                 summary=dict(enzymes=len(out), by_tier=dict(tiers), with_km=sum(1 for r in out.values() if r.get("km_uM")),
                              flux_absolute=absolute, catpred_available=bool(cat)))
    json.dump(payload, open(OUT/"kinetics_refined.json","w"))
    s=payload["summary"]
    print(f"refined kinetics: {s['enzymes']} enzymes | tiers {dict(tiers)}")
    print(f"  CatPred bias calibration: {bias:+.3f} log10 (fit on {n_cal} measured) | Km available for {s['with_km']}")
    if xmed: print(f"  flux-deconvolved vs CatPred cross-check: {xmed:.2f}x median (agreement where both exist)")
    if not cat: print("  (catpred_kinetics.json absent -> ran flux-deconvolution + measured only; add CatPred for full ensemble)")

if __name__=="__main__":
    main()
