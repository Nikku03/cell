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
        # split on the arrow with flexible whitespace; ONE side may be empty (exchange/sink/demand rxns)
        if "<=>" in formula: parts=re.split(r"\s*<=>\s*", formula, maxsplit=1)
        elif "=>" in formula: parts=re.split(r"\s*=>\s*", formula, maxsplit=1)
        elif "-->" in formula: parts=re.split(r"\s*-->\s*", formula, maxsplit=1)
        else: continue
        if len(parts)!=2: continue
        L,R=parts
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

# physical constants for the ABSOLUTE Vmax units pass (env-overridable). Vmax=kcat*[E] is a volumetric
# rate (per L cell); convert to mmol/gDW/hr to match the measured exchange fluxes.
CELL_VOL_L   =float(__import__("os").environ.get("CELL_VOL_L","2.0e-12"))     # L per cell (matches concentration.py)
GDW_PER_CELL =float(__import__("os").environ.get("GDW_PER_CELL","3.0e-10"))   # gDW per cell (~0.3 ng, mammalian)
IMPUTED_SLACK=float(__import__("os").environ.get("VMAX_IMPUTED_SLACK","5.0")) # headroom for 6x-noisy imputed kcat
#   nM(=1e-9 mol/L) * (1/s) * V_cell[L/cell] / mass[gDW/cell] * 1000(mmol/mol) * 3600(s/hr)
VMAX_CONV = 1e-9 * CELL_VOL_L / GDW_PER_CELL * 1000.0 * 3600.0

def vmax_bounds(rxns):
    """ABSOLUTE enzyme capacity Vmax = SUM_isozymes kcat*[E], in mmol/gDW/hr (commensurable with the
    measured exchange fluxes). kcat from kinetics.json, [E] (nM) from concentration.json (real once PaxDb
    is present). Falls back to a relative normalisation only if no absolute [E] is available.
    Returns (vmax{ri:mmol/gDW/hr}, tier{ri}, units_str)."""
    kin=(json.load(open(OUT/"kinetics.json")).get("kinetics",{}) if (OUT/"kinetics.json").exists() else {})
    conc=(json.load(open(OUT/"concentration.json")).get("concentration",{}) if (OUT/"concentration.json").exists() else {})
    cap={}; meas={}
    for ri,r in enumerate(rxns):
        tot=0.0; got=False; is_meas=False
        for g in r["genes"]:                              # SUM over isozymes = total reaction capacity
            k=kin.get(g)
            if not k or not k.get("kcat_per_s"): continue
            E=conc.get(g,{}).get("baseline_nM")
            if E is None or E<=0: continue
            tot += k["kcat_per_s"]*E; got=True
            if k.get("tier")=="measured": is_meas=True
        if got: cap[ri]=tot; meas[ri]=is_meas
    if not cap: return {},{},"none"
    have_E = bool(conc)
    if have_E:                                            # ABSOLUTE: kcat*[E] -> mmol/gDW/hr
        vmax={ri: v*VMAX_CONV*(1.0 if meas[ri] else IMPUTED_SLACK) for ri,v in cap.items()}
        units="mmol/gDW/hr(absolute)"
    else:                                                 # fallback: relative normalisation (no abundance)
        med=float(np.median(list(cap.values())))
        vmax={ri:(v/med)*BIG for ri,v in cap.items()}; units="relative(normalised)"
    tier={ri: ("vmax(measured-kcat)" if meas[ri] else "vmax(imputed-kcat)") for ri in cap}
    return vmax, tier, units

def load_medium():
    """exchange bounds: tsv 'key<TAB>lb<TAB>ub' (uptake negative). key = Human-GEM rxn id OR metabolite
    name (resolved to its exchange reaction in main). env FLUX_MEDIUM."""
    import os
    m={}; f=Path(os.environ.get("FLUX_MEDIUM", str(OUT/"flux_medium.tsv")))
    if f.exists():
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=3 and p[0].lower() not in ("rxn","rxn_id","metabolite"):
                try: m[p[0]]=(float(p[1]),float(p[2]))
                except: pass
    return m

def load_measured_flux():
    """13C-MFA validation set: tsv 'key<TAB>flux[...]' (relative, glucose uptake=100). key = rxn id OR
    gene symbol (resolved via GPR in validate). env FLUX_MEASURED -> {key: flux}."""
    import os
    m={}; f=Path(os.environ.get("FLUX_MEASURED", str(OUT/"flux_measured_13c.tsv")))
    if f.exists():
        for l in open(f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=2 and p[0].lower() not in ("rxn","rxn_id","gene"):
                try: m[p[0]]=float(p[1])
                except: pass
    return m

def base_met(name):
    import re
    return re.sub(r"\[[^\]]*\]\s*$","",name).strip()

def met_to_exchange(rxns, met_idx):
    """metabolite base-name -> (rxn_index, coef_sign) for single-metabolite boundary reactions.
    Prefers extracellular/system compartments. coef_sign tells uptake direction. Keys are registered
    both exact and lower-cased so a medium/validation name resolves regardless of Human-GEM casing."""
    inv={i:n for n,i in met_idx.items()}
    out={}
    for j,r in enumerate(rxns):
        if not r["is_exchange"] or len(r["stoich"])!=1: continue
        mi,coef=next(iter(r["stoich"].items())); nm=inv[mi]
        bn=base_met(nm); ext = any(t in nm for t in ["[e]","[s]","[x]"])
        for key in (bn, bn.lower()):
            if key not in out or (ext and "[" in nm): out[key]=(j, 1.0 if coef>0 else -1.0)
    return out

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

def _pred_flux(key, v, id2i, gene2rxn, met2ex=None):
    """predicted flux for a measured key: Human-GEM rxn id -> its flux; gene -> total |flux| its reactions
    carry; metabolite name -> |flux| of its exchange reaction (for measured exchange fluxes)."""
    if key in id2i: return abs(v[id2i[key]])
    if met2ex:
        mx=met2ex.get(key) or met2ex.get(key.lower())
        if mx: return abs(v[mx[0]])
    if key in gene2rxn: return float(sum(abs(v[j]) for j in gene2rxn[key]))
    return None

def validate(rxns, v, measured, ref_key, gene2rxn, met2ex=None):
    """compare predicted vs measured flux, both normalised to the reference key. A measured key resolves
    by rxn-id, metabolite (exchange), or gene (sum of |flux| over its reactions)."""
    id2i={r["id"]:i for i,r in enumerate(rxns)}
    pref=_pred_flux(ref_key, v, id2i, gene2rxn, met2ex)
    if not pref or pref<1e-9: return None
    mref=abs(measured.get(ref_key,100.0)) or 100.0
    errs=[]; pv=[]; mv=[]; used=[]
    for key,mf in measured.items():
        if key==ref_key: continue
        pf=_pred_flux(key, v, id2i, gene2rxn, met2ex)
        if pf is None: continue
        pred=pf/pref*mref
        if abs(mf)<1e-9 or pred<1e-9: continue
        errs.append(abs(math.log10(pred/abs(mf)))); pv.append(pred); mv.append(abs(mf)); used.append(key)
    if len(errs)<3: return None
    return dict(n=len(errs), median_fold_error=round(10**float(np.median(errs)),2),
                within_2x=round(sum(1 for e in errs if e<=math.log10(2))/len(errs),2),
                within_3x=round(sum(1 for e in errs if e<=math.log10(3))/len(errs),2),
                log_pearson_r=round(float(np.corrcoef(np.log10(pv),np.log10(mv))[0,1]),3) if np.std(pv)>0 else 0.0,
                ref=ref_key, compared=used[:20])

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
    id2i={r["id"]:i for i,r in enumerate(rxns)}
    # BASE bounds: reversibility + medium/exchange anchors (NO Vmax yet) — key may be a rxn id or metabolite
    blb=np.array([-BIG if r["rev"] else 0.0 for r in rxns]); bub=np.array([BIG]*n)
    medium=load_medium(); met2ex=met_to_exchange(rxns, met_idx); n_med=0
    for key,(l,u) in medium.items():
        mx=met2ex.get(key) or met2ex.get(key.lower())
        if key in id2i: blb[id2i[key]]=l; bub[id2i[key]]=u; n_med+=1
        elif mx:
            j,sign=mx
            lo,hi=(l,u) if sign<0 else (-u,-l)      # medium is uptake-negative; flip if exchange is source-oriented
            blb[j]=lo; bub[j]=hi; n_med+=1
    # enzyme Vmax overlay — ABSOLUTE mmol/gDW/hr (kcat x [E]); commensurable with the exchange anchor
    vmax, vtier, vunits = vmax_bounds(rxns)
    def with_vmax(scale):
        lb=blb.copy(); ub=bub.copy()
        for ri,cap in vmax.items():
            ub[ri]=min(ub[ri], cap*scale)
            if rxns[ri]["rev"]: lb[ri]=max(lb[ri], -cap*scale)
        return lb,ub
    gene2rxn=defaultdict(list)
    for j,r in enumerate(rxns):
        for gsym in r["genes"]: gene2rxn[gsym].append(j)
    bio=find_biomass(rxns); c=np.zeros(n)
    if bio is None:
        print("no biomass reaction found -> emitting Vmax capacity map only, no flux solve")
        json.dump(dict(flux={}, summary=dict(reactions=n, metabolites=nmet, vmax_bounded=len(vmax),
                  vmax_units=vunits, note="no biomass objective"), validation=None), open(OUT/"flux.json","w")); return
    c[bio]=1.0
    # feasibility ladder: KEEP the exchange anchor (absolute scale); relax Vmax before abandoning it.
    z=None; vmax_scale=None; used_vmax=True; anchored=(n_med>0)
    for scale in (1,10,100,1000):
        lb,ub=with_vmax(scale); z=solve_fba(S,lb,ub,c,True)
        if z is not None: vmax_scale=scale; break
    if z is None:                                   # drop Vmax bounds, keep exchange anchor
        lb,ub=blb.copy(),bub.copy(); z=solve_fba(S,lb,ub,c,True); used_vmax=False
    if z is None:                                   # last resort: open exchange too (loses absolute anchor)
        lb=np.array([-BIG if r["rev"] else 0.0 for r in rxns]); ub=np.array([BIG]*n)
        z=solve_fba(S,lb,ub,c,True); anchored=False
    if z is None:
        print("FBA infeasible even fully relaxed -> skipping (check Human-GEM parse / biomass)"); return
    if vmax_scale and vmax_scale>1:
        print(f"  note: Vmax bounds relaxed x{vmax_scale} for feasibility (cell-mass/kcat scale); enzyme layer still applied")
    zstar=float(z[bio])
    v=pfba(S, lb, ub, c, zstar) if zstar>1e-9 else z
    if v is None: v=z
    # tiers
    ntier={}
    anchored_ids=set(id2i[k] for k in medium if k in id2i)|set(
        (met2ex.get(k) or met2ex.get(k.lower()))[0] for k in medium if (met2ex.get(k) or met2ex.get(k.lower())))
    for ri,r in enumerate(rxns):
        if ri in anchored_ids: ntier[ri]="exchange-anchored"
        elif ri in vtier: ntier[ri]=vtier[ri]
        else: ntier[ri]="network-only"
    active=int(np.sum(np.abs(v)>1e-6))
    out={}
    for ri,r in enumerate(rxns):
        if abs(v[ri])<1e-6: continue
        out[r["id"]]=dict(v=round(float(v[ri]),4), tier=ntier[ri], genes=r["genes"][:6],
                          rev=r["rev"])
    # validation vs measured 13C flux
    # validation — prefer the REAL measured exchange set (FluxProfilingREGP, absolute) if present,
    # else the internal 13C set. Reference for normalisation = glucose.
    import os as _os
    exch_f=OUT/"flux_measured_exch.tsv"; measured={}; val_kind=None
    if exch_f.exists():
        for l in open(exch_f):
            p=l.rstrip("\n").split("\t")
            if len(p)>=2 and p[0].lower() not in ("metabolite","gene","rxn"):
                try: measured[p[0]]=float(p[1])
                except: pass
        val_kind="measured-exchange(FluxProfilingREGP)"
    if not measured:
        measured=load_measured_flux(); val_kind="internal-13C(curated)"
    val=None
    if measured:
        ref=next((k for k in ("glucose","D-glucose","MAR09034","EX_glc__D_e","HK1","HK2")
                  if _pred_flux(k,v,id2i,gene2rxn,met2ex) not in (None,0)), None) or next(iter(measured))
        val=validate(rxns, v, measured, ref, gene2rxn, met2ex)
        if val: val["kind"]=val_kind
    from collections import Counter
    tc=Counter(ntier.values())
    flux_units = "mmol/gDW/hr(absolute)" if anchored else "relative"
    payload=dict(flux=out, biomass=round(zstar,4),
        summary=dict(reactions=n, metabolites=nmet, active_reactions=active,
                     flux_units=flux_units, absolute=bool(anchored), vmax_units=vunits,
                     vmax_bounded=len(vmax), vmax_applied=used_vmax, vmax_relaxation=vmax_scale,
                     measured_kcat_bounds=tc.get("vmax(measured-kcat)",0),
                     imputed_kcat_bounds=tc.get("vmax(imputed-kcat)",0),
                     exchange_anchored=tc.get("exchange-anchored",0), exchange_constraints=n_med,
                     objective="biomass"),
        validation=val)
    json.dump(payload, open(OUT/"flux.json","w"))
    print(f"flux: {n} reactions / {nmet} metabolites | biomass={zstar:.3g} | {active} active fluxes | units={flux_units}")
    print(f"  ecFBA bounds: {len(vmax)} Vmax-bounded [{vunits}] ({tc.get('vmax(measured-kcat)',0)} measured-kcat, "
          f"{tc.get('vmax(imputed-kcat)',0)} imputed), {n_med} exchange constraints applied")
    if val: print(f"  {val.get('kind','')} validation: median fold-error {val['median_fold_error']}x, within-2x {val['within_2x']:.0%}, "
                  f"within-3x {val['within_3x']:.0%}, log-R {val['log_pearson_r']} (n={val['n']})")
    else: print("  (no measured-flux file present -> validation skipped)")

if __name__=="__main__":
    main()
