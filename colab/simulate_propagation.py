"""PROPAGATION-THERAPY simulation — a self-propagating cure vs a traditional drug.

The idea: instead of a drug that must physically reach EVERY diseased cell (hard — solid tumors are only
~30-70% penetrable), engineer a therapeutic where a treated cell SECRETES a factor that induces the same
therapeutic state (e.g., apoptosis, or a corrective program) in its neighbours, which then re-secrete -- a
self-propagating wave from a small seed. This is the computational analogue of the suicide-gene 'bystander
effect' (HSV-TK/GCV via gap junctions), generalized to a designed secreted signal.

We simulate a 2-D tissue (cellular automaton) and compare, as a function of dose/penetration:
  - TRADITIONAL: each cell is cured only if the drug physically reaches it (prob = penetration).
  - PROPAGATION: seed a small fraction; a diseased+sensitive cell is cured if enough neighbours are actively
    secreting; cured cells secrete for a limited window; the wave percolates if secretion x sensitivity is
    above a threshold.
Outputs the final cured fraction for both, the propagation percolation threshold, and the efficiency edge
(cure coverage per unit of drug actually delivered). No model data needed -> self-contained. -> propagation_sim.json
"""
import json, os, math
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def simulate_propagation(size=120, seed_frac=0.02, sensitivity=0.9, secrete_steps=3, neigh_thresh=1,
                         steps=200, rng=None):
    """CA on a size x size tissue. Returns cured-fraction trajectory. neigh_thresh = # secreting neighbours
    needed to cure a cell (lower = easier spread)."""
    rng=rng or np.random.RandomState(0)
    N=size*size
    state=np.zeros((size,size),dtype=np.int8)              # 0 diseased, 1 cured
    secreting=np.zeros((size,size),dtype=np.int16)          # remaining secretion steps
    sens=(rng.random((size,size))<sensitivity)             # which cells CAN be cured by the signal
    # seed
    seed=rng.random((size,size))<seed_frac
    state[seed]=1; secreting[seed]=secrete_steps
    traj=[state.mean()]
    for _ in range(steps):
        # count secreting neighbours (4-neighbourhood)
        s=(secreting>0).astype(np.int16)
        nb=np.zeros_like(s)
        nb[1:,:]+=s[:-1,:]; nb[:-1,:]+=s[1:,:]; nb[:,1:]+=s[:,:-1]; nb[:,:-1]+=s[:,1:]
        newly=(state==0)&sens&(nb>=neigh_thresh)
        state[newly]=1; secreting[newly]=secrete_steps
        secreting[secreting>0]-=1
        traj.append(state.mean())
        if not newly.any() and (secreting>0).sum()==0: break
    return np.array(traj), float(state.mean())

def main():
    rng=np.random.RandomState(0); size=120
    # 1) TRADITIONAL drug: cured fraction == penetration (each cell independently reached)
    trad={round(p,2): round(p,3) for p in [0.1,0.3,0.5,0.7]}
    # 2) PROPAGATION: sweep dose (seed) and signal strength (neigh_thresh: 1 easy, 2 harder)
    prop={}
    for seedf in [0.005,0.02,0.05]:
        for thr in [1,2]:
            _,fin=simulate_propagation(size, seed_frac=seedf, neigh_thresh=thr, rng=np.random.RandomState(1))
            prop[f"seed{seedf}_thresh{thr}"]=round(fin,3)
    # 3) percolation threshold in sensitivity (fraction of cells that CAN receive the signal)
    perc=[]
    for sensv in [0.5,0.6,0.7,0.8,0.9,1.0]:
        _,fin=simulate_propagation(size, seed_frac=0.02, sensitivity=sensv, neigh_thresh=1, rng=np.random.RandomState(2))
        perc.append((sensv, round(fin,3)))
    # efficiency: cure-coverage per unit drug DELIVERED. Traditional delivers to `pen` fraction to cure `pen`.
    # Propagation delivers to `seed` fraction and (if it percolates) cures ~all -> coverage/dose >> 1.
    seed_used=0.02; _,prop_cov=simulate_propagation(size, seed_frac=seed_used, neigh_thresh=1, rng=np.random.RandomState(3))
    eff=dict(traditional_coverage_per_dose=1.0,                       # cure = drug delivered
             propagation_coverage_per_dose=round(prop_cov/seed_used,1))# cure >> drug delivered
    res=dict(traditional_final_cured=trad, propagation_final_cured=prop,
             percolation_sensitivity=perc, efficiency=eff,
             note="2-D tissue CA. Traditional drug cures only what it reaches (=penetration). A propagating "
                  "therapy cures ~all cells from a ~2% seed IF the secretion signal percolates (sensitivity above "
                  "~0.6-0.7). Coverage-per-dose is the efficiency edge: propagation cures many cells per cell dosed.")
    json.dump(res, open(OUT/"propagation_sim.json","w"))
    print("PROPAGATION-THERAPY SIMULATION (120x120 tissue)")
    print("  TRADITIONAL drug (cured = penetration):", trad)
    print("  PROPAGATION (final cured fraction):")
    for k,v in prop.items(): print(f"    {k}: {v}")
    print("  percolation vs sensitivity (fraction of cells that can receive the signal):")
    print("   ", ", ".join(f"{s}->{f}" for s,f in perc))
    print(f"  >>> EFFICIENCY: propagation cures ~{eff['propagation_coverage_per_dose']}x more cells per cell dosed "
          f"than a traditional drug (2% seed -> {prop_cov:.0%} cured when it percolates)")
    print("  (this is why a propagating cure beats a drug that must reach every cell -- a percolating wave "
          "from a small seed vs a penetration-limited plateau)")

if __name__=="__main__":
    main()
