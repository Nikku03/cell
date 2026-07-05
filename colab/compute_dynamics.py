"""DYNAMICS (Problem 1, turnover / linear-response level) — real per-gene temporal constants.

We don't need full mechanistic ODEs to add genuine time information. Every species' response time to ANY
perturbation is set by how fast it turns over: for dX/dt = production - kdeg*X, the time constant is
    tau = 1/kdeg = half-life / ln2
So with measured protein + mRNA half-lives we can state, per gene:
  - response time (tau) and settling time (~5*tau) — how long a perturbation takes to play out;
  - degradation rate kdeg;
  - implied steady-state SYNTHESIS rate = kdeg * abundance (production needed to hold [E]);
  - turnover class (fast / medium / stable);
  - cell-cycle peak phase (Whitfield), i.e. WHEN in the cycle the gene is highest.
Cross-layer: ESSENTIAL + FAST-turnover proteins are 'synthesis-vulnerable' — they need constant resynthesis,
so blocking translation/transcription depletes them fastest (acute drug-response targets). ESSENTIAL +
STABLE proteins take days to deplete after knockdown (why some KDs show no quick phenotype).

Honest scope: this is the LINEAR-RESPONSE / turnover layer, not a full spatial-kinetic cell simulation.
It complements compute_dfba (culture-scale) and compute_attractors (state switching). -> dynamics.json
"""
import json, math, csv
from statistics import median
from pathlib import Path
OUT=Path("outputs/orphan")
LN2=math.log(2)

def _num(name, col):
    f=OUT/name; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f), delimiter="\t"):
        try: m[r["gene"]]=float(r[col])
        except (KeyError, ValueError, TypeError): pass
    return m

def _str(name, col):
    f=OUT/name; m={}
    if not f.exists(): return m
    for r in csv.DictReader(open(f), delimiter="\t"):
        if r.get("gene") and r.get(col): m[r["gene"]]=r[col]
    return m

def main():
    prot=_num("protein_halflife.tsv","half_life_h")
    mrna=_num("mrna_halflife.tsv","half_life_h")
    phase=_str("cellcycle_phase.tsv","phase")
    if not (prot or mrna or phase):
        print("dynamics: no turnover/cell-cycle inputs -> run fetch_dynamics_data first (skipped)"); return
    conc=(json.load(open(OUT/"concentration.json")).get("concentration",{}) if (OUT/"concentration.json").exists() else {})
    ess={}
    if (OUT/"cell_complete.json").exists():
        ess={g["name"]:g.get("ess",0) for g in json.load(open(OUT/"cell_complete.json")).get("genes",[])}
    # turnover classes are DATA-DRIVEN quartiles of the actual proteome (absolute cutoffs vary by cell type;
    # here median ~56h, so a fixed '<5h=fast' catches nothing). fast = shortest-half-life quartile, etc.
    pv=sorted(prot.values())
    q1=pv[len(pv)//4] if pv else 0; q3=pv[(3*len(pv))//4] if pv else 0
    def tclass(pt): return "fast" if pt<=q1 else ("stable" if pt>=q3 else "medium")
    genes=set(prot)|set(mrna)|set(phase)
    out={}
    for g in genes:
        rec={}; pt=prot.get(g); mt=mrna.get(g)
        if pt and pt>0:
            rec.update(protein_half_life_h=pt, protein_kdeg_per_h=round(LN2/pt,5),
                       response_time_h=round(pt/LN2,3), settling_time_h=round(5*pt/LN2,2),
                       turnover_class=tclass(pt))
            E=conc.get(g,{}).get("baseline_nM")
            if E and E>0: rec["synthesis_rate_nM_per_h"]=round((LN2/pt)*E,4)   # production to hold steady state
        if mt and mt>0:
            rec.update(mrna_half_life_h=mt, mrna_kdeg_per_h=round(LN2/mt,5))
        if g in phase: rec["cell_cycle_phase"]=phase[g]
        if ess.get(g): rec["essential"]=True
        if rec: out[g]=rec
    fast=sorted((dict(gene=g, half_life_h=r["protein_half_life_h"], response_time_h=r["response_time_h"],
                      essential=r.get("essential",False)) for g,r in out.items() if r.get("turnover_class")=="fast"),
                key=lambda x:x["half_life_h"])
    stable=sorted((dict(gene=g, half_life_h=r["protein_half_life_h"]) for g,r in out.items()
                   if r.get("turnover_class")=="stable"), key=lambda x:-x["half_life_h"])
    synth_vuln=sorted((dict(gene=g, half_life_h=r["protein_half_life_h"], response_time_h=r["response_time_h"])
                       for g,r in out.items() if r.get("essential") and r.get("turnover_class")=="fast"),
                      key=lambda x:x["half_life_h"])
    cyc={}
    for g,r in out.items():
        if r.get("cell_cycle_phase"): cyc.setdefault(r["cell_cycle_phase"],[]).append(g)
    payload=dict(dynamics=out, fast_responders=fast[:200], stable_proteins=stable[:200],
                 synthesis_vulnerable=synth_vuln[:200],
                 cell_cycle_genes_by_phase={k:sorted(v) for k,v in cyc.items()},
                 summary=dict(genes=len(out), with_protein_halflife=len(prot), with_mrna_halflife=len(mrna),
                              cell_cycle_regulated=len(phase), fast=len(fast), stable=len(stable),
                              synthesis_vulnerable=len(synth_vuln),
                              median_protein_half_life_h=round(median(list(prot.values())),2) if prot else None,
                              median_mrna_half_life_h=round(median(list(mrna.values())),2) if mrna else None,
                              fast_cutoff_h=round(q1,2), stable_cutoff_h=round(q3,2)),
                 note="turnover/linear-response dynamics: tau=half-life/ln2 per gene; complements dFBA + attractors; "
                      "not a full mechanistic ODE simulation")
    json.dump(payload, open(OUT/"dynamics.json","w"))
    s=payload["summary"]
    print(f"dynamics: {s['genes']} genes with temporal constants | protein t1/2 for {s['with_protein_halflife']} "
          f"(median {s['median_protein_half_life_h']}h), mRNA t1/2 for {s['with_mrna_halflife']} "
          f"(median {s['median_mrna_half_life_h']}h) | {s['cell_cycle_regulated']} cell-cycle-phased")
    print(f"  turnover quartiles: {s['fast']} fast (t1/2<={s['fast_cutoff_h']}h, rapid responders), "
          f"{s['stable']} stable (t1/2>={s['stable_cutoff_h']}h, slow to deplete) | "
          f"{s['synthesis_vulnerable']} ESSENTIAL+fast = synthesis-vulnerable (translation-block hits fastest)")

if __name__=="__main__":
    main()
