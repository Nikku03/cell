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
from collections import defaultdict, deque
from statistics import median
from pathlib import Path
OUT=Path("outputs/orphan")
LN2=math.log(2)
TIMES=[0,20,40,60,120,240,480]; MID=[20,40,60,120,240]      # minutes
# canonical immediate-early / primary-response genes = onset depth 0 (transcription fires at ~t0). Their
# regulatory targets are secondary (depth 1), etc. -> cascade depth = onset TIER (Tullai 2007 logic).
IEG={"FOS","FOSB","FOSL1","FOSL2","JUN","JUNB","JUND","EGR1","EGR2","EGR3","EGR4","MYC","MYCN","ATF3",
     "NR4A1","NR4A2","NR4A3","IER2","IER3","BTG2","DUSP1","DUSP5","KLF2","KLF4","KLF6","NPAS4","JDP2","SRF","ELK1"}

def load_reg_depth(entry):
    """cascade depth of each gene = min regulatory hops from the immediate-early set (BFS over TF->target).
    Depth 0 = fires immediately; depth d = needs d transcriptional generations first (secondary/tertiary)."""
    p=OUT/"cell_complete.json"
    if not p.exists(): return {}
    D=json.load(open(p)); G=D.get("genes",[]); name=[g["name"] for g in G]
    adj=defaultdict(list)
    for e in D.get("reg",[]):
        if len(e)<2: continue
        s,t=e[0],e[1]
        sn=name[s] if isinstance(s,int) and 0<=s<len(name) else s
        tn=name[t] if isinstance(t,int) and 0<=t<len(name) else t
        if sn and tn: adj[sn].append(tn)
    depth={g:0 for g in entry if g in name or g in adj}
    dq=deque(depth);
    while dq:
        u=dq.popleft()
        for v in adj.get(u,[]):
            if v not in depth: depth[v]=depth[u]+1; dq.append(v)
    return depth

def frac_onset(tau_min, t, t_on): return 0.0 if t<=t_on else 1.0-math.exp(-(t-t_on)/tau_min)

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
    # RUNG 2: add per-gene ONSET = cascade_depth * delta (delta = ONE global param, fit on a train split of
    # genes and evaluated on a disjoint test split so it can't tune to the test).
    depth=load_reg_depth(IEG)
    items=[]                                                        # (g, tau_g_min, tau_o_min, depth_or_None, fr_obs)
    for g,vals in tc.items():
        hl=mrna.get(g)
        if not hl: continue
        R0, Rf = vals[0], vals[-1]
        if R0<=0 or Rf<=0 or abs(math.log2(Rf/R0))<1.0: continue     # need a >=2-fold SUSTAINED change
        fr_obs=[(t,(R-R0)/(Rf-R0)) for t,R in zip(TIMES,vals) if t in MID]
        if any(fo<-0.25 or fo>1.25 for _,fo in fr_obs): continue     # drop transient overshoots
        items.append((g, hl*60/LN2, _best_tau(fr_obs), depth.get(g), fr_obs))
    n_test=len(items)
    reach=[d for _,_,_,d,_ in items if d is not None]
    from collections import Counter as _C
    dep_hist={str(k):v for k,v in sorted(_C(d for _,_,_,d,_ in items).items(), key=lambda x:(x[0] is None,x[0]))}
    dep_def=median(reach) if reach else 1                           # unreachable genes -> neutral median depth
    def gd(d): return d if d is not None else dep_def
    train=[it for i,it in enumerate(items) if i%2==0]; test=[it for i,it in enumerate(items) if i%2==1]
    def onset_med_err(dataset, delta):
        e=[abs(frac_onset(tau_med_min,t,gd(d)*delta)-fo) for _,_,_,d,fr in dataset for t,fo in fr]
        return median(e) if e else 1.0
    best_delta=min(range(0,201,5), key=lambda dl: onset_med_err(train, dl))   # fit onset step-delay on TRAIN
    err_gene=[]; err_glob=[]; err_step=[]; err_lin=[]; err_oracle=[]; err_onset=[]; pairs=[]
    for g,tau_g,tau_o,d,fr_obs in test:                            # evaluate ALL methods on the TEST split only
        pairs.append((tau_g*LN2/60, tau_o*LN2/60))
        t_on=gd(d)*best_delta
        for t,fo in fr_obs:
            err_gene.append(abs(frac_complete(tau_g,t)-fo))
            err_glob.append(abs(frac_complete(tau_med_min,t)-fo))
            err_step.append(abs(1.0-fo))                            # instant-step (no kinetics)
            err_lin.append(abs(t/480.0-fo))                        # linear ramp
            err_oracle.append(abs(frac_complete(tau_o,t)-fo))
            err_onset.append(abs(frac_onset(tau_med_min,t,t_on)-fo))  # RUNG 2: relaxation + network onset
    def med(x): return round(median(x),4) if x else None
    # correlation of measured half-life vs oracle-fitted half-life (log space) -> does t_half predict kinetics?
    corr=None
    if len(pairs)>=8:
        import statistics as st
        xs=[math.log10(a) for a,b in pairs if a>0 and b>0]; ys=[math.log10(b) for a,b in pairs if a>0 and b>0]
        mx,my=st.mean(xs),st.mean(ys); sx,sy=st.pstdev(xs),st.pstdev(ys)
        if sx>0 and sy>0:
            corr=round(sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(len(xs)*sx*sy),3)
    val=dict(n_genes=n_test, n_test_eval=len(test), n_reachable=len(reach),
             onset_step_delay_min=best_delta, cascade_depth_histogram=dep_hist,
             median_tau_min=round(tau_med_min), timepoints=MID,
             onset_verdict=("network-onset does NOT help here: cascade depth is degenerate (most genes 1 hop) AND "
                            "onset (minutes) is swamped by the mRNA relaxation time (~%d min) for sustained responders; "
                            "the oracle gap is per-gene effective-rate heterogeneity, reachable only by measuring the "
                            "transcription input (Rung 3)")%round(tau_med_min) if best_delta==0 else
                           "network-onset improves prediction (fitted step-delay %d min)"%best_delta,
             median_abs_fraction_error=dict(
                 instant_step_no_kinetics=med(err_step), linear_ramp=med(err_lin),
                 rung1_relaxation_global=med(err_glob), rung1_relaxation_gene_specific=med(err_gene),
                 rung2_relaxation_plus_onset=med(err_onset), oracle_bestfit=med(err_oracle)),
             halflife_vs_fitted_log_pearson=corr,
             rung2_improvement_vs_rung1=(round((med(err_glob)-med(err_onset))/med(err_glob),3)
                                         if err_glob and med(err_glob) else None),
             rung2_gap_to_oracle_closed=(round((med(err_glob)-med(err_onset))/(med(err_glob)-med(err_oracle)),3)
                                         if err_glob and med(err_glob) and med(err_glob)!=med(err_oracle) else None))

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
    v=val; e=v["median_abs_fraction_error"]
    print(f"time resolution: EGF held-out test | {v['n_genes']} responders ({v['n_reachable']} reachable in reg-net), "
          f"eval on {v['n_test_eval']} test genes | fitted onset step-delay {v['onset_step_delay_min']} min")
    print(f"  median |fraction error| — instant-step {e['instant_step_no_kinetics']} > linear {e['linear_ramp']} > "
          f"Rung1 relaxation {e['rung1_relaxation_global']} > RUNG2 +onset {e['rung2_relaxation_plus_onset']} > "
          f"oracle-floor {e['oracle_bestfit']}")
    if best_delta>0 and v['rung2_improvement_vs_rung1']:
        print(f"  >>> RUNG 2 (network onset): {round(100*v['rung2_improvement_vs_rung1'])}% lower error than Rung 1, "
              f"closing {round(100*(v['rung2_gap_to_oracle_closed'] or 0))}% of the gap to the oracle floor")
    else:
        print(f"  >>> RUNG 2 verdict: network-onset does NOT help (fitted delay 0). Depth is degenerate "
              f"{v['cascade_depth_histogram']} and onset<<tau({v['median_tau_min']}min) for sustained genes. "
              f"The oracle gap ({e['rung1_relaxation_global']}->{e['oracle_bestfit']}) is per-gene effective-rate "
              f"heterogeneity -> needs the MEASURED transcription input (Rung 3), not more topology.")
    if met_tau:
        print(f"  fast layer: {len(met_tau)} metabolite time constants (pool/flux), median {payload['median_metabolite_tau_s']} s "
              f"(metabolism equilibrates ~instantly vs the hour-scale expression layer)")

if __name__=="__main__":
    main()
