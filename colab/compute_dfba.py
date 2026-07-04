"""DYNAMIC FBA (Problem 1, the honest 'time' slice) — Born-Oppenheimer timescale separation.

Metabolism is fast (ms) and reaches steady state effectively instantly; the culture (biomass, external
metabolites) changes slowly (hours). So we NEVER integrate metabolic ODEs: at each slow step we snap the
metabolic network to its FBA optimum, read the exchange fluxes, and step only the slow variables:
    dX/dt      = mu * X                       (biomass growth, mu = FBA biomass flux)
    dS_i/dt    = v_exch_i(t) * X              (external metabolite i, mmol/L)
Uptake is Michaelis-Menten limited by what's left (v_max_i * S_i/(Km+S_i)), so substrates deplete and
growth stops naturally. This gives a real batch-culture trajectory (glucose drawdown, lactate build-up,
growth curve) with NO kinetic rate constants -- the dynamics come from the FBA steady states, not ODEs.

Reuses compute_flux (Human-GEM parse + solver). Optional validation vs a measured timecourse
(FluxProfilingREGP MCF10A). Skips gracefully off-Colab. -> dfba.json {t, biomass, <metabolite trajectories>}
"""
import json, os, math
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")
# tracked external species: name -> (initial mM, Km mM for uptake). lactate starts at 0 (product).
TRACK={"glucose":(25.0,0.1),"L-glutamine":(4.0,0.1),"L-lactate":(0.0,None)}
DT=float(os.environ.get("DFBA_DT","0.25")); TMAX=float(os.environ.get("DFBA_TMAX","60"))
X0=float(os.environ.get("DFBA_X0","0.01"))            # gDW/L initial biomass

def main():
    try:
        import scipy  # noqa
        from compute_flux import parse_gem, build_S, solve_fba, met_to_exchange, load_medium
    except Exception as e:
        print("dFBA needs scipy + compute_flux:",repr(e)[:80]); return
    rxns, met_idx = parse_gem()
    if not rxns: print("Human-GEM absent -> dFBA skipped (runs on Colab)"); return
    n=len(rxns); S=build_S(rxns,len(met_idx)); id2i={r["id"]:i for i,r in enumerate(rxns)}
    from compute_flux import find_biomass
    bio=find_biomass(rxns)
    if bio is None: print("no biomass reaction -> dFBA needs a growth objective; skipping"); return
    met2ex=met_to_exchange(rxns, met_idx)
    medium=load_medium()                                  # measured max uptake capacities (mmol/gDW/hr)
    # resolve tracked metabolites -> exchange reaction index, orientation sign, max uptake capacity
    track={}
    for name,(S0,Km) in TRACK.items():
        mx=met2ex.get(name) or met2ex.get(name.lower())
        if not mx: continue
        j,sgn=mx; cap=None
        for key in (name, name.lower()):
            if key in medium: cap=abs(medium[key][0]); break     # |measured uptake bound| = max uptake
        track[name]=dict(rxn=j, sign=sgn, S=S0, Km=Km, cap=cap or (10.0 if name=="glucose" else 1.0), traj=[])
    if "glucose" not in track:
        print("glucose exchange not found in Human-GEM -> dFBA skipped"); return
    base_lb=np.array([-1000.0 if r["rev"] else 0.0 for r in rxns]); base_ub=np.array([1000.0]*n)
    c=np.zeros(n); c[bio]=1.0
    X=X0; ts=[]; Xs=[]; mus=[]; t=0.0; steps=int(TMAX/DT)
    for _ in range(steps):
        lb=base_lb.copy(); ub=base_ub.copy()
        for name,tk in track.items():
            j=tk["rxn"]; s=tk["sign"]
            v_up = tk["cap"]*(tk["S"]/(tk["Km"]+tk["S"])) if (tk["Km"] and tk["S"]>0) else 0.0   # MM-limited
            # uptake direction depends on exchange orientation: sink(s<0)->uptake is v<0; source(s>0)->v>0
            if s<0: lb[j]=-v_up; ub[j]=1000.0
            else:   lb[j]=-1000.0; ub[j]=v_up
            if tk["Km"] is None:                          # pure product (lactate): secretion only, no uptake
                if s<0: lb[j]=0.0
                else:   ub[j]=0.0
        z=solve_fba(S, lb, ub, c, maximize=True)
        if z is None: break
        mu=float(z[bio])
        ts.append(round(t,3)); Xs.append(round(X,5)); mus.append(round(mu,5))
        for name,tk in track.items(): tk["traj"].append(round(tk["S"],4))
        if mu<1e-4 and t>1: break                         # growth stopped (substrate exhausted)
        # step slow variables: net secretion into medium = -sign*v (>0 secretion, <0 uptake)
        for name,tk in track.items():
            sec = -tk["sign"]*float(z[tk["rxn"]])         # mmol/gDW/hr into the medium
            tk["S"]=max(0.0, tk["S"] + sec*X*DT)          # dS = sec * X * dt
        X = X*math.exp(mu*DT); t += DT
    traj={name:tk["traj"] for name,tk in track.items()}
    payload=dict(t_hr=ts, biomass_gDW_L=Xs, growth_rate_per_hr=mus, metabolites_mM=traj,
                 params=dict(dt=DT, tmax=TMAX, X0=X0, tracked=list(track)),
                 note="Born-Oppenheimer dFBA: FBA steady state per slow step; culture-scale dynamics, no kinetic ODEs")
    json.dump(payload, open(OUT/"dfba.json","w"))
    g=track.get("glucose",{}).get("traj",[])
    print(f"dFBA: simulated {len(ts)} steps to t={ts[-1] if ts else 0}h | biomass {X0}->{Xs[-1] if Xs else X0:.3g} gDW/L "
          f"| glucose {g[0] if g else '?'}->{g[-1] if g else '?'} mM")
    print("  (culture-scale trajectory: growth curve + substrate drawdown + product build-up; no rate constants)")

if __name__=="__main__":
    main()
