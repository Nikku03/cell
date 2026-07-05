"""TIME RESOLUTION (Problem 1, solved to the level the data supports) — the relaxation equation.

Two-timescale (Born-Oppenheimer) cell. Every species relaxes to its moving steady-state target with a time
constant set by its OWN turnover, all of which we measure:
    dx/dt = k*(x*(t) - x),   tau = 1/k
  - metabolite: tau = pool/flux = [S]/v            (seconds; [S] from metabolites, v from ecFBA)
  - mRNA:       tau_R = t_half(mRNA)/ln2            (RNADecayCafe)
  - protein:    tau_P = t_half(protein)/ln2         (Mathieson), driven through its mRNA

For a step perturbation (transcription x fold f at t=0) this integrates to a PARAMETER-FREE closed form:
    mRNA:    R(t)/R0 = f + (1-f) e^(-t/tau_R)
    protein: P(t)/P0 = f + (1-f)[c e^(-a t) - a e^(-c t)]/(c-a),   a=1/tau_R, c=1/tau_P
Key falsifiable prediction: a gene reaches HALF of its total change in exactly one mRNA half-life.

VALIDATION (held-out, real data): GSE6783 EGF time course (HeLa, 0-480 min). For sustained-monotonic
responders we (1) predict the fraction-of-change-complete at each intermediate timepoint from the measured
gene-specific half-life and compare to naive baselines + an oracle floor, and (2) correlate each gene's
MEASURED mRNA half-life with the time constant that best fits its observed trajectory. -> timeresolved.json
"""
import json, math, csv
from statistics import median
from pathlib import Path
OUT=Path("outputs/orphan")
LN2=math.log(2)
TIMES=[0,20,40,60,120,240,480]; MID=[20,40,60,120,240]      # minutes

# ---------- the equation ----------
def mrna_response(f, tau_R_min, t): return f + (1.0-f)*math.exp(-t/tau_R_min)
def protein_response(f, tau_R_min, tau_P_min, t):
    a, c = 1.0/tau_R_min, 1.0/tau_P_min
    if abs(c-a) < 1e-9: return f + (1.0-f)*(1.0+a*t)*math.exp(-a*t)      # equal-rate limit
    return f + (1.0-f)*(c*math.exp(-a*t) - a*math.exp(-c*t))/(c-a)
def frac_complete(tau_R_min, t): return 1.0 - math.exp(-t/tau_R_min)     # fraction of total change done by t

def _load_hl(name):
    f=OUT/name; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f), delimiter="\t"):
        try: m[r["gene"]]=float(r["half_life_h"])
        except (KeyError, ValueError, TypeError): pass
    return m

def _load_timecourse():
    f=OUT/"timecourse_egf.tsv"; tc={}
    if not f.exists(): return tc
    for r in csv.DictReader(open(f), delimiter="\t"):
        try: tc[r["gene"]]=[float(r[f"t{t}"]) for t in TIMES]
        except (KeyError, ValueError): pass
    return tc

def _best_tau(fr_obs):
    """grid-search the time constant (min) that best fits observed fraction-complete points -> oracle floor."""
    best=None
    for tau in [10,20,30,45,60,90,120,180,240,360,480,720,1000,1500,2500,4000]:
        e=sum((frac_complete(tau,t)-fo)**2 for t,fo in fr_obs)
        if best is None or e<best[1]: best=(tau,e)
    return best[0]

def main():
    mrna=_load_hl("mrna_halflife.tsv"); prot=_load_hl("protein_halflife.tsv")
    tc=_load_timecourse()
    if not (mrna and tc):
        print("time-resolution: need mrna_halflife.tsv + timecourse_egf.tsv (run fetch_dynamics_data + "
              "fetch_timecourse first) -> validation skipped");
        if not mrna and not tc: return
    tau_med_h = median(list(mrna.values())) if mrna else 4.0
    tau_med_min = tau_med_h*60/LN2

    # ---- held-out validation on the EGF time course ----
    err_gene=[]; err_glob=[]; err_step=[]; err_lin=[]; err_oracle=[]; pairs=[]
    n_test=0
    for g,vals in tc.items():
        hl=mrna.get(g)
        if not hl: continue
        R0, Rf = vals[0], vals[-1]
        if R0<=0 or Rf<=0 or abs(math.log2(Rf/R0))<1.0: continue     # need a >=2-fold SUSTAINED change
        fr_obs=[]
        for t,R in zip(TIMES,vals):
            if t in MID: fr_obs.append((t, (R-R0)/(Rf-R0)))
        # sustained-monotonic filter: intermediate fractions stay in [-0.25,1.25] (drop transient overshoots)
        if any(fo<-0.25 or fo>1.25 for _,fo in fr_obs): continue
        n_test+=1
        tau_g = hl*60/LN2                                            # measured gene-specific tau (min)
        tau_o = _best_tau(fr_obs)                                    # oracle best-fit tau
        pairs.append((hl, tau_o*LN2/60))                            # (measured half-life h, fitted half-life h)
        for t,fo in fr_obs:
            err_gene.append(abs(frac_complete(tau_g,t)-fo))
            err_glob.append(abs(frac_complete(tau_med_min,t)-fo))
            err_step.append(abs(1.0-fo))                            # instant-step (no kinetics)
            err_lin.append(abs(t/480.0-fo))                        # linear ramp
            err_oracle.append(abs(frac_complete(tau_o,t)-fo))
    def med(x): return round(median(x),4) if x else None
    # correlation of measured half-life vs oracle-fitted half-life (log space) -> does t_half predict kinetics?
    corr=None
    if len(pairs)>=8:
        import statistics as st
        xs=[math.log10(a) for a,b in pairs if a>0 and b>0]; ys=[math.log10(b) for a,b in pairs if a>0 and b>0]
        mx,my=st.mean(xs),st.mean(ys); sx,sy=st.pstdev(xs),st.pstdev(ys)
        if sx>0 and sy>0:
            corr=round(sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(len(xs)*sx*sy),3)
    val=dict(n_genes=n_test, timepoints=MID,
             median_abs_fraction_error=dict(
                 measured_gene_specific=med(err_gene), global_single_halflife=med(err_glob),
                 instant_step_no_kinetics=med(err_step), linear_ramp=med(err_lin), oracle_bestfit=med(err_oracle)),
             halflife_vs_fitted_log_pearson=corr,
             improvement_vs_step=(round((med(err_step)-med(err_gene))/med(err_step),3)
                                  if err_step and med(err_step) else None))

    # ---- metabolite fast-layer time constants (tau = pool/flux); illustrative, not separately validated ----
    metab=(json.load(open(OUT/"metabolites.json")).get("metabolites",{}) if (OUT/"metabolites.json").exists() else {})
    flux=(json.load(open(OUT/"flux.json")).get("flux",{}) if (OUT/"flux.json").exists() else {})
    met_tau=[]
    if metab and flux:
        # total consumption flux per metabolite (mmol/gDW/hr) vs pool (uM -> mmol/gDW via cell vol); tau in seconds
        VCELL_L=2e-12; GDW=3e-10
        for m,rec in metab.items():
            cuM=rec.get("concentration_uM")
            if not cuM: continue
            # pool (mmol/gDW) = uM*1e-3 mmol/L * VCELL_L / GDW
            pool = cuM*1e-3*VCELL_L/GDW
            v = sum(abs(flux.get(rid,{}).get("v",0)) for rid in rec.get("consumed_by",[]))  # mmol/gDW/hr
            if v>1e-9:
                tau_s = pool/v*3600.0
                met_tau.append((m, round(tau_s,3)))
        met_tau.sort(key=lambda x:x[1])

    payload=dict(equation=dict(
                    mrna="R(t)/R0 = f + (1-f) exp(-t/tau_R)",
                    protein="P(t)/P0 = f + (1-f)[c exp(-a t) - a exp(-c t)]/(c-a), a=1/tau_R c=1/tau_P",
                    metabolite="tau = pool/flux = [S]/v",
                    prediction="half of the total change is reached in one mRNA half-life"),
                 validation=val,
                 metabolite_time_constants_s=dict(met_tau[:40]),
                 median_metabolite_tau_s=(round(median([t for _,t in met_tau]),3) if met_tau else None),
                 note="turnover/linear-response time resolution; metabolite tau is analytic (pool/flux), "
                      "expression tau validated vs GSE6783 EGF time course")
    json.dump(payload, open(OUT/"timeresolved.json","w"))
    v=val
    print(f"time resolution: EGF held-out test on {v['n_genes']} sustained responders")
    e=v["median_abs_fraction_error"]
    print(f"  median |fraction error| — MEASURED half-life {e['measured_gene_specific']} vs "
          f"global {e['global_single_halflife']}, instant-step {e['instant_step_no_kinetics']}, "
          f"linear {e['linear_ramp']}, oracle-floor {e['oracle_bestfit']}")
    print(f"  measured half-life vs fitted time-constant: log-Pearson r={v['halflife_vs_fitted_log_pearson']} "
          f"| improvement over no-kinetics: {round(100*v['improvement_vs_step'])}% lower error" if v['improvement_vs_step'] else "")
    if met_tau:
        print(f"  fast layer: {len(met_tau)} metabolite time constants (pool/flux), median {payload['median_metabolite_tau_s']} s "
              f"(metabolism equilibrates ~instantly vs the hour-scale expression layer)")

if __name__=="__main__":
    main()
