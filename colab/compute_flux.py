"""FLUX (enzyme-constrained FBA on Human-GEM) — compute a genome-wide metabolic flux distribution and
validate it against measured 13C-MFA fluxes.

Kinetics gives CAPACITY, not flux: v <= Vmax = kcat*[E]. The network (mass balance S*v=0) + boundary
anchors (measured uptake/secretion) turn those ceilings into an actual flux distribution, propagating
even to reactions with no kinetics. We therefore:
  1. parse Human-GEM into a proper stoichiometric matrix S (coefficients + compartments preserved);
  2. bound each enzymatic reaction by Vmax = kcat (kinetics.json) x [E] (concentration.json) -- ecFBA;
  3. constrain exchange/uptake from a medium file (Jain CORE-style) if present, else a default medium;
  4. solve FBA (maximize biomass, then pFBA to minimise total flux) with scipy HiGHS;
  5. VALIDATE: hold out measured central-carbon fluxes (CeCaFDB/13C-MFA tsv), predict them, report
     median fold-error / within-2x / log-Pearson -- honest, since 13C ground truth is itself ~10-30%.

Every flux carries a tier: exchange-anchored > vmax(measured-kcat) > vmax(imputed-kcat) > network-only.
Skips gracefully if Human-GEM or scipy is absent (heavy solve -> Colab). -> flux.json
"""
import json, re, math
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
BIG=1000.0                                   # default flux bound (mmol/gDW/h convention)
COEF=re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(.+)$")

def parse_gem():
    """Human-GEM txt (Rxn\tformula\tGPR) -> (reactions, met_index). Coefficients + compartments kept.
    reaction = dict(id, rev, genes, stoich{met_id: coef(- reactant/+ product)}, is_exchange)."""
    f=H/"human_gem.txt"
    if not f.exists(): return None,None
    ENSG=re.compile(r"ENSG\d+"); ensg2sym=_ensg2sym()
    met_idx={}
    def mid(n): return met_idx.setdefault(n,len(met_idx))
    def side(s):
        out={}
        for tok in s.split(" + "):
            tok=tok.strip()
            if not tok: continue
            m=COEF.match(tok); c=float(m.group(1)) if m else 1.0
            name=m.group(2).strip() if m else tok
            if name: out[name]=out.get(name,0.0)+c        # full 'metabolite[compartment]' id kept
        return out
    rxns=[]
    for l in open(f):
        p=l.rstrip("\n").split("\t")
        if len(p)<3 or p[0]=="Rxn name": continue
        rid,formula,gpr=p[0],p[1],p[2]
        rev="<=>" in formula
        sep=" <=> " if rev else (" => " if " => " in formula else None)
        if not sep: continue
        L,R=formula.split(sep,1)
        subs=side(L); prods=side(R)
        stoich={}
        for n,c in subs.items(): stoich[mid(n)]=stoich.get(mid(n),0.0)-c
        for n,c in prods.items(): stoich[mid(n)]=stoich.get(mid(n),0.0)+c
        if not stoich: continue
        is_exch=(not subs) or (not prods)                 # boundary exchange/sink/demand
        genes=sorted({ensg2sym[e] for e in set(ENSG.findall(gpr)) if e in ensg2sym})
        rxns.append(dict(id=rid, rev=rev, genes=genes, stoich=stoich, is_exchange=is_exch,
                         eq=formula[:120]))
    return rxns, met_idx

def _ensg2sym():
    import gzip
    ensp2sym={}; ensp2ensg={}
    af=H/"string_aliases.txt.gz"
    if not af.exists(): return {}
    for l in gzip.open(af,"rt"):
        p=l.rstrip("\n").split("\t")
        if len(p)<3: continue
        if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
        elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
    return {ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}

def vmax_bounds(rxns):
    """Vmax = kcat x [E] per reaction (max over isozymes). Returns (vmax{ri:val}, tier{ri:str}).
    Relative units: kcat_per_s (kinetics.json) x baseline_nM (concentration.json) -> normalised so the
    median maps to a generous cap; the ecFBA point is that LOW-capacity enzymes bind, not the scale."""
    kin=(json.load(open(OUT/"kinetics.json")).get("kinetics",{}) if (OUT/"kinetics.json").exists() else {})
    conc=(json.load(open(OUT/"concentration.json")).get("concentration",{}) if (OUT/"concentration.json").exists() else {})
    raw={}; meas={}
    for ri,r in enumerate(rxns):
        best=None; best_meas=False
        for g in r["genes"]:
            k=kin.get(g)
            if not k or not k.get("kcat_per_s"): continue
            E=conc.get(g,{}).get("baseline_nM")
            if E is None or E<=0: continue
            cap=k["kcat_per_s"]*E
            if best is None or cap>best:
                best=cap; best_meas=(k.get("tier")=="measured")
        if best is not None: raw[ri]=best; meas[ri]=best_meas
    if not raw: return {},{}
    med=float(np.median(list(raw.values())))
    vmax={ri: (v/med)*BIG for ri,v in raw.items()}        # median enzyme -> BIG; slower/scarcer bind first
    tier={ri: ("vmax(measured-kcat)" if meas[ri] else "vmax(imputed-kcat)") for ri in raw}
    return vmax, tier

def load_medium():
    """optional measured exchange bounds: tsv 'rxn_id\tlb\tub' (uptake negative). env FLUX_MEDIUM."""
    import os
    m={}; f=Path(os.environ.get("FLUX_MEDIUM", str(OUT/"flux_medium.tsv")))
    if f.exists():
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=3 and not p[0].lower().startswith("rxn"):
                try: m[p[0]]=(float(p[1]),float(p[2]))
                except: pass
    return m

def load_measured_flux():
    """optional 13C-MFA validation set: tsv 'rxn_id\tflux' (relative, e.g. glucose uptake=100).
    env FLUX_MEASURED -> {rxn_id: flux}."""
    import os
    m={}; f=Path(os.environ.get("FLUX_MEASURED", str(OUT/"flux_measured_13c.tsv")))
    if f.exists():
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=2 and not p[0].lower().startswith("rxn"):
                try: m[p[0]]=float(p[1])
                except: pass
    return m

def build_S(rxns, nmet):
    from scipy import sparse
    rows=[]; cols=[]; vals=[]
    for j,r in enumerate(rxns):
        for mi,c in r["stoich"].items():
            rows.append(mi); cols.append(j); vals.append(c)
    return sparse.csr_matrix((vals,(rows,cols)), shape=(nmet,len(rxns)))

def solve_fba(S, lb, ub, c, maximize=True):
    """max/min c.v s.t. S v = 0, lb<=v<=ub. Returns v or None."""
    from scipy.optimize import linprog
    obj=(-np.array(c)) if maximize else np.array(c)
    b=np.zeros(S.shape[0])
    res=linprog(obj, A_eq=S, b_eq=b, bounds=list(zip(lb,ub)), method="highs")
    return res.x if res.success else None

def pfba(S, lb, ub, c, zstar, frac=0.95):
    """minimise total flux L1 (sum|v|) s.t. c.v >= frac*zstar. Split v=vf-vr>=0 over all reactions."""
    from scipy.optimize import linprog
    from scipy import sparse
    n=S.shape[1]
    # variables [vf(n), vr(n)]; v = vf - vr
    Aeq=sparse.hstack([S, -S]).tocsr()
    beq=np.zeros(S.shape[0])
    # objective: min sum(vf+vr)
    cost=np.ones(2*n)
    # bounds: vf in [max(0,lb..)], but easier: vf,vr in [0, ub_abs]; enforce v-range via A_ub
    ubabs=[max(abs(l),abs(u)) for l,u in zip(lb,ub)]
    bnds=[(0,ubabs[i]) for i in range(n)]+[(0,ubabs[i]) for i in range(n)]
    # v>=lb and v<=ub : (vf-vr)>=lb -> -vf+vr<=-lb ; (vf-vr)<=ub -> vf-vr<=ub
    I=sparse.identity(n,format="csr")
    Aub=sparse.vstack([sparse.hstack([-I, I]), sparse.hstack([I, -I])]).tocsr()
    bub=np.concatenate([-np.array(lb), np.array(ub)])
    # biomass floor: c.v >= frac*zstar -> -c.(vf-vr) <= -frac*zstar
    crow=sparse.hstack([-sparse.csr_matrix(c), sparse.csr_matrix(c)]).tocsr()
    Aub=sparse.vstack([Aub, crow]).tocsr(); bub=np.concatenate([bub,[-frac*zstar]])
    res=linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnds, method="highs")
    if not res.success: return None
    vf=res.x[:n]; vr=res.x[n:]; return vf-vr

def find_biomass(rxns):
    # Human-GEM generic biomass reactions (ids), then any reaction mentioning 'biomass'
    KNOWN={"MAR13082","MAR10024","MAR10063","BIOMASS_HUMAN","BIOMASS_MAINTENANCE","BIOMASS_REACTION"}
    for j,r in enumerate(rxns):
        if r["id"].upper() in KNOWN: return j
    for j,r in enumerate(rxns):
        if "biomass" in (r["id"]+" "+r["eq"]).lower(): return j
    return None

def validate(rxns, v, measured, ref_id):
    """compare predicted vs measured relative flux (both normalised to |ref|). hold-out fold-error."""
    id2i={r["id"]:i for i,r in enumerate(rxns)}
    ri=id2i.get(ref_id)
    if ri is None or abs(v[ri])<1e-9: return None
    scale=abs(v[ri])
    errs=[]; pv=[]; mv=[]
    for rid,mf in measured.items():
        i=id2i.get(rid)
        if i is None: continue
        pred=v[i]/scale*abs(measured.get(ref_id,100.0))
        if abs(mf)<1e-9 or abs(pred)<1e-9: continue
        errs.append(abs(math.log10(abs(pred)/abs(mf)))); pv.append(pred); mv.append(mf)
    if len(errs)<3: return None
    return dict(n=len(errs), median_fold_error=round(10**float(np.median(errs)),2),
                within_2x=round(sum(1 for e in errs if e<=math.log10(2))/len(errs),2),
                within_3x=round(sum(1 for e in errs if e<=math.log10(3))/len(errs),2),
                log_pearson_r=round(float(np.corrcoef(np.log10(np.abs(pv)),np.log10(np.abs(mv)))[0,1]),3)
                            if np.std(pv)>0 else 0.0)

def main():
    try:
        import scipy  # noqa
    except Exception:
        print("scipy not installed -> flux solve skipped (pip install scipy)"); return
    rxns, met_idx = parse_gem()
    if not rxns:
        print("Human-GEM (human_gem.txt) absent -> flux skipped (runs on Colab)"); return
    nmet=len(met_idx); n=len(rxns)
    S=build_S(rxns, nmet)
    # bounds
    lb=np.array([-BIG if r["rev"] else 0.0 for r in rxns]); ub=np.array([BIG]*n)
    vmax, vtier = vmax_bounds(rxns)
    for ri,cap in vmax.items():
        ub[ri]=min(ub[ri],cap)
        if rxns[ri]["rev"]: lb[ri]=max(lb[ri],-cap)
    # medium / exchange constraints
    medium=load_medium(); id2i={r["id"]:i for i,r in enumerate(rxns)}
    for rid,(l,u) in medium.items():
        if rid in id2i: lb[id2i[rid]]=l; ub[id2i[rid]]=u
    # objective
    bio=find_biomass(rxns)
    c=np.zeros(n)
    driver="biomass" if bio is not None else None
    if bio is not None: c[bio]=1.0
    if driver is None:
        print("no biomass reaction found and no exchange objective -> emitting capacity map only "
              "(Vmax bounds), no flux solve");
        payload=dict(flux={}, summary=dict(reactions=n, metabolites=nmet, vmax_bounded=len(vmax),
                     note="no objective/driver; ran capacity analysis only"), validation=None)
        json.dump(payload, open(OUT/"flux.json","w")); return
    z=solve_fba(S, lb, ub, c, maximize=True)
    if z is None:
        print("FBA infeasible with current bounds -> relaxing exchange to default medium and retrying")
        z=solve_fba(S, np.array([-BIG if r["rev"] else 0.0 for r in rxns]), np.array([BIG]*n), c, True)
    if z is None:
        print("FBA infeasible even relaxed -> skipping (check Human-GEM parse / biomass)"); return
    zstar=float(z[bio])
    v=pfba(S, lb, ub, c, zstar) if zstar>1e-9 else z
    if v is None: v=z
    # tiers
    ntier={}
    for ri,r in enumerate(rxns):
        if r["id"] in medium: ntier[ri]="exchange-anchored"
        elif ri in vtier: ntier[ri]=vtier[ri]
        else: ntier[ri]="network-only"
    active=int(np.sum(np.abs(v)>1e-6))
    out={}
    for ri,r in enumerate(rxns):
        if abs(v[ri])<1e-6: continue
        out[r["id"]]=dict(v=round(float(v[ri]),4), tier=ntier[ri], genes=r["genes"][:6],
                          rev=r["rev"])
    # validation vs measured 13C flux
    measured=load_measured_flux()
    val=None
    if measured:
        ref=next((k for k in ("MAR09034","EX_glc__D_e","glucose") if k in id2i), None) or next(iter(measured))
        val=validate(rxns, v, measured, ref)
    from collections import Counter
    tc=Counter(ntier.values())
    payload=dict(flux=out, biomass=round(zstar,4),
        summary=dict(reactions=n, metabolites=nmet, active_reactions=active,
                     vmax_bounded=len(vmax), measured_kcat_bounds=tc.get("vmax(measured-kcat)",0),
                     imputed_kcat_bounds=tc.get("vmax(imputed-kcat)",0),
                     exchange_anchored=tc.get("exchange-anchored",0), objective=driver),
        validation=val)
    json.dump(payload, open(OUT/"flux.json","w"))
    print(f"flux: {n} reactions / {nmet} metabolites | biomass={zstar:.3g} | {active} active fluxes")
    print(f"  ecFBA bounds: {len(vmax)} Vmax-bounded ({tc.get('vmax(measured-kcat)',0)} measured-kcat, "
          f"{tc.get('vmax(imputed-kcat)',0)} imputed), {tc.get('exchange-anchored',0)} exchange-anchored")
    if val: print(f"  13C validation: median fold-error {val['median_fold_error']}x, within-2x {val['within_2x']:.0%}, "
                  f"within-3x {val['within_3x']:.0%}, log-R {val['log_pearson_r']} (n={val['n']})")
    else: print("  (no 13C-MFA measured-flux file present -> validation skipped; add flux_measured_13c.tsv)")

if __name__=="__main__":
    main()
