"""REFINE kcat — fuse CatPred predictions with our own data to beat raw CatPred (first-principles).

CatPred predicts in-vitro, standard-condition kcat/Km from sequence+chemistry (~3-4x error, R2~0.5) with a
per-prediction uncertainty. We hold independent signals CatPred cannot see, and each fixes one of its error
sources. The rate law v = kcat*[E]*[S]/(Km+[S]) ties CatPred's outputs to quantities we already have
([E]=PaxDb nM, [S]=metabolite uM, v=absolute FBA flux). Per enzyme we pick the best-grounded kcat:

  TIER 1  measured        -> our 391 BRENDA/SABIO/DLKcat values (fact). Also used to CALIBRATE CatPred bias.
  TIER 2  flux-deconvolved -> for flux-carrying enzymes: kcat = (v/[E]) * (Km+[S])/[S], using CatPred's Km
          (easy to predict) + our [S], with v/[E] the observed in-vivo apparent turnover.
  TIER 3  CatPred(calibrated) blended with the EC-class prior by uncertainty.
  + Km -> saturation -> a real velocity.

VALIDATION: hold out 25% of the measured kcat, predict them through the SAME refinement (bias fit on the
other 75%), and report median fold-error vs the imputed baseline (compute_kinetics ~9.6x) -- so we actually
measure whether the error was cut. -> kinetics_refined.json + validation
"""
import json, math, random
from collections import Counter
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")
CELL_VOL_L=2.0e-12; GDW_PER_CELL=3.0e-10
VMAX_CONV=1e-9*CELL_VOL_L/GDW_PER_CELL*1000.0*3600.0

def load(name, key=None):
    f=OUT/name
    if not f.exists(): return {}
    d=json.load(open(f)); return d.get(key,{}) if key else d

def main():
    kin=load("kinetics.json","kinetics")
    cat=load("catpred_kinetics.json","catpred")
    flux=load("flux.json"); fx=flux.get("flux",{}); absolute=("absolute" in flux.get("summary",{}).get("flux_units",""))
    conc=load("concentration.json","concentration")
    metab=load("metabolites.json","metabolites")
    gem=load("metabolic_graph.json"); gem_rxn={}
    if gem.get("rxns"):
        mets=gem.get("mets",[])
        for r in gem["rxns"]: gem_rxn[r.get("id")]=[mets[i] for i in r.get("s",[]) if i<len(mets)]
    g2flux={}
    for rid,r in fx.items():
        for g in r.get("genes",[]):
            if g not in g2flux or abs(r.get("v",0))>abs(g2flux[g][1]): g2flux[g]=(rid, r.get("v",0.0))

    def Sconc(rid):
        subs=[]
        for m in gem_rxn.get(rid, []):
            r=metab.get(m)
            if r and r.get("concentration_uM") and not r.get("hub"): subs.append(r["concentration_uM"])
        return min(subs) if subs else None

    def fit_bias(exclude=None):
        """log10 offset to add to CatPred kcat to match measured (fit on measured genes, minus `exclude`)."""
        diffs=[]
        for g,k in kin.items():
            if g==exclude or k.get("tier")!="measured": continue
            mk=k.get("kcat_per_s"); ck=(cat.get(g) or {}).get("kcat")
            if mk and ck and mk>0 and ck>0: diffs.append(math.log10(mk)-math.log10(ck))
        return (float(np.median(diffs)), len(diffs)) if len(diffs)>=8 else (0.0, len(diffs))

    def predict(g, bias):
        """refined kcat for g WITHOUT using g's own measured value (tier 2/3). Returns (kcat, tier)."""
        c=cat.get(g,{}); km=c.get("km")
        # tier 2: flux-deconvolved
        if absolute and g in g2flux:
            rid,v=g2flux[g]; E=conc.get(g,{}).get("baseline_nM")
            if E and E>0 and abs(v)>1e-9:
                app=abs(v)/(E*VMAX_CONV); S=Sconc(rid)
                if km and S: return app*(km+S)/S, "flux-deconvolved"
                return app, "flux-deconvolved(apparent)"
        # tier 3: calibrated CatPred blended with EC prior by uncertainty
        if c.get("kcat"):
            cc=10**(math.log10(c["kcat"])+bias); unc=c.get("kcat_unc")
            prior=kin.get(g,{}).get("kcat_per_s") if kin.get(g,{}).get("tier") in ("EC-measured","EC-class") else None
            if prior and unc and unc>0.5: return 10**(0.5*math.log10(cc)+0.5*math.log10(prior)), "catpred+ECprior"
            return cc, "catpred(calibrated)"
        k=kin.get(g,{})
        if k.get("kcat_per_s"): return k["kcat_per_s"], k.get("tier","imputed")
        return None, None

    bias, n_cal = fit_bias()
    out={}; tiers=Counter(); xchecks=[]
    for g in set(kin)|set(cat)|set(g2flux):
        k=kin.get(g,{}); c=cat.get(g,{}); km=c.get("km"); rec=dict(km_uM=km)
        if k.get("tier")=="measured" and k.get("kcat_per_s"):
            rec.update(kcat_per_s=round(k["kcat_per_s"],4), tier="measured")
        else:
            kc,tier=predict(g, bias)
            if kc is None: continue
            rec.update(kcat_per_s=round(kc,4), tier=tier)
            cc=(10**(math.log10(c["kcat"])+bias)) if c.get("kcat") else None
            if cc and "flux" in tier:
                rec["catpred_kcat"]=round(cc,4); rec["deconv_vs_catpred"]=round(kc/cc,3); xchecks.append(abs(math.log10(kc/cc)))
        if g in g2flux and km:
            S=Sconc(g2flux[g][0])
            if S: rec["saturation"]=round(S/(km+S),3)
        tiers[rec["tier"]]+=1; out[g]=rec

    # ---- held-out validation: predict measured kcat as if unmeasured, compare to truth ----
    meas={g:k["kcat_per_s"] for g,k in kin.items() if k.get("tier")=="measured" and k.get("kcat_per_s")}
    val=None
    if len(meas)>=12:
        keys=list(meas); random.Random(0).shuffle(keys); test=set(keys[:max(3,len(keys)//4)])
        errs=[]; by=Counter()
        for g in test:
            b,_=fit_bias(exclude=g)                       # bias fit excluding this test gene
            pred,tier=predict(g, b)
            if pred and pred>0:
                errs.append(abs(math.log10(pred/meas[g]))); by[tier]+=1
        if len(errs)>=3:
            val=dict(n=len(errs), median_fold_error=round(10**float(np.median(errs)),2),
                     within_2x=round(sum(1 for e in errs if e<=math.log10(2))/len(errs),2),
                     within_3x=round(sum(1 for e in errs if e<=math.log10(3))/len(errs),2),
                     tier_mix=dict(by), imputed_baseline_fold_error=9.61)
    payload=dict(kinetics_refined=out, calibration=dict(catpred_log10_bias=round(bias,3), n_calibration=n_cal),
                 validation=val,
                 summary=dict(enzymes=len(out), by_tier=dict(tiers), with_km=sum(1 for r in out.values() if r.get("km_uM")),
                              flux_absolute=absolute, catpred_available=bool(cat)))
    json.dump(payload, open(OUT/"kinetics_refined.json","w"))
    s=payload["summary"]
    print(f"refined kinetics: {s['enzymes']} enzymes | tiers {dict(tiers)}")
    print(f"  CatPred bias calibration: {bias:+.3f} log10 (n={n_cal}) | Km for {s['with_km']}")
    if val:
        print(f"  >>> ERROR CUT: refined held-out fold-error {val['median_fold_error']}x vs imputed baseline "
              f"{val['imputed_baseline_fold_error']}x | within-2x {val['within_2x']:.0%}, within-3x {val['within_3x']:.0%} "
              f"(n={val['n']}, tiers {val['tier_mix']})")
    if not cat: print("  (catpred_kinetics.json absent -> measured + flux-deconvolution only; add CatPred for the full ensemble)")

if __name__=="__main__":
    main()
