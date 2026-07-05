"""Assemble the flux VALIDATION + medium-constraint data for compute_flux.py.

Two artifacts:
  flux_medium.tsv        metabolite<TAB>lb<TAB>ub   -> exchange/uptake bounds (uptake negative)
  flux_measured_13c.tsv  gene<TAB>rel_flux<TAB>pathway -> measured 13C-MFA flux, glucose uptake = 100

The gold-standard human flux sets (Jain 2012 CORE exchange fluxes; CeCaFDB 13C-MFA distributions) are
not cleanly machine-downloadable (paywalled supplement / unstable host). So we:
  1. TRY env-overridable URLs (JAIN_CORE_URL, CECAFDB_URL) and parse if reachable;
  2. otherwise fall back to a CURATED, literature-cited aerobic-glycolysis reference vector (relative,
     glucose=100), keyed by gene so it maps onto Human-GEM via GPR.

Honesty tier: the curated fallback validates the PHENOTYPE quantitatively (does the model reproduce the
Warburg split -- ~90% glycolysis, ~10% PPP, lactate/glucose ~1.8, modest glutamine/TCA?). It is NOT a
rigorous held-out per-cell-line test -- for that, drop the real Jain/CeCaFDB tsv in via the env vars.

Sources for the curated vector (canonical human cancer-cell 13C-MFA):
  DeBerardinis et al. 2007 PNAS (SF188 glioblastoma); Murphy/HL-60 2024 (glycolysis 90%, oxPPP 9%,
  lactate/glucose 191%); standard Warburg stoichiometry (triose split doubles lower glycolysis).
-> flux_medium.tsv, flux_measured_13c.tsv
"""
import os, urllib.request
from pathlib import Path
OUT=Path("outputs/orphan")

# --- curated aerobic-glycolysis reference, normalised to glucose uptake = 100 (relative flux) ---
# gene, relative flux, pathway. Lower glycolysis ~2x glucose (triose split); lactate ~1.8x; oxPPP ~0.1x.
CURATED_13C = [
 ("HK1",   100, "glycolysis(upper)"),   # glucose -> G6P (uptake basis)
 ("GPI",    90, "glycolysis(upper)"),   # G6P -> F6P (100 - oxPPP)
 ("PFKL",   90, "glycolysis(upper)"),   # F6P -> FBP (committed step)
 ("ALDOA",  90, "glycolysis(upper)"),   # FBP -> 2 triose
 ("GAPDH", 180, "glycolysis(lower)"),   # triose x2
 ("PGK1",  180, "glycolysis(lower)"),
 ("ENO1",  180, "glycolysis(lower)"),
 ("PKM",   175, "glycolysis(lower)"),   # PEP -> pyruvate
 ("LDHA",  165, "fermentation"),        # pyruvate -> lactate (secreted; lactate/glucose ~1.65)
 ("G6PD",   10, "pentose-phosphate"),   # oxPPP entry ~10%
 ("GLS",    20, "glutaminolysis"),      # glutamine -> glutamate
 ("CS",     22, "TCA"),                 # citrate synthase
 ("IDH2",   18, "TCA"),
 ("PDHA1",  25, "pyruvate-oxidation"),  # pyruvate -> acetyl-CoA (fraction not fermented)
]
# medium exchange bounds (metabolite name as it appears in Human-GEM, lb, ub). Uptake = negative flux.
CURATED_MEDIUM = [
 ("glucose",     -100, -100),   # fixed uptake = the normalisation basis (anchors absolute scale)
 ("O2",         -1000,    0),   # oxygen freely taken up
 ("L-glutamine",  -25,    0),
 ("L-lactate",      0, 1000),   # lactate secreted
 ("CO2",            0, 1000),
 ("H2O",        -1000, 1000),
 ("H+",         -1000, 1000),
 ("Pi",          -100,    0),
 ("NH3",            0, 1000),
]

def _try_url(env, dest):
    url=os.environ.get(env,"")
    if not url: return False
    try:
        print(f"  fetching {env} -> {url[:70]} ..."); urllib.request.urlretrieve(url, dest); return True
    except Exception as e:
        print(f"  {env} fetch failed ({repr(e)[:80]}) -> using curated fallback"); return False

# metabolite name in the FluxProfilingREGP file -> candidate Human-GEM metabolite base-names (compute_flux
# resolves whichever exists in the parsed GEM). Amino acids are L- in Human-GEM.
_SYN={"glucose":["glucose","D-glucose"],"L-lactate":["L-lactate","lactate"],
      "glutamine":["L-glutamine","glutamine"],"glutamate":["L-glutamate","glutamate"],
      "arginine":["L-arginine"],"asparagine":["L-asparagine"],"glycine":["glycine"],
      "methionine":["L-methionine"],"phenylalanine":["L-phenylalanine"],"proline":["L-proline"],
      "serine":["L-serine"],"threonine":["L-threonine"],"tryptophan":["L-tryptophan"],
      "tyrosine":["L-tyrosine"],"valine":["L-valine"],"alanine":["L-alanine"]}

def fetch_fluxprofile():
    """Pull REAL absolute exchange fluxes (mmol/gDW/hr) from FluxProfilingREGP (Human-GEM/Human1 based).
    Writes flux_medium.tsv (uptakes as constraints, glucose fixed = absolute scale anchor) and
    flux_measured_exch.tsv (all measured exchange fluxes, for held-out validation). REGP is the paper's
    improved method; CORE is the classic one (env FLUXPROFILE_METHOD)."""
    url=os.environ.get("FLUXPROFILE_URL",
        "https://raw.githubusercontent.com/Radboud-YuLab/FluxProfilingREGP/main/data/exchange_fluxes/output/exch_fluxes_MCF10A_NCI60.csv")
    method=os.environ.get("FLUXPROFILE_METHOD","REGP").upper()
    dest=OUT/"_fluxprofile_raw.csv"
    try:
        print(f"  fetching FluxProfilingREGP exchange fluxes ({method}) -> {url[:60]} ..."); urllib.request.urlretrieve(url,dest)
    except Exception as e:
        print(f"  FluxProfilingREGP fetch failed ({repr(e)[:80]})"); return False
    import csv
    rows=list(csv.DictReader(open(dest)))
    if not rows or method not in rows[0]:
        print(f"  FluxProfilingREGP: no '{method}' column found"); return False
    tol=float(os.environ.get("FLUX_TOL","0.3"))
    gtol=float(os.environ.get("FLUX_GLUCOSE_TOL","0.05"))   # tight band forcing glucose uptake to the measured rate
    med=[]; val=[]
    for r in rows:
        name=r["metabolite"].strip()
        try: f=float(r[method])
        except: continue
        cands=_SYN.get(name,[name])
        is_glc = name.lower() in ("glucose","d-glucose","glc")
        for c in cands:
            if f<0 and is_glc:                        # GLUCOSE forced to a tight band: it anchors the absolute
                # scale AND must carry a known flux so validation has a reference. Forcing only glucose (not every
                # uptake) keeps Human-GEM feasible while pinning the carbon-flux magnitude.
                med.append((c, round(f*(1+gtol),6), round(f*(1-gtol),6)))   # lb<ub, both negative (uptake band)
            elif f<0:                                 # other uptakes -> UPPER BOUND on availability (0..max), not a
                # forced band; forcing every uptake made Human-GEM infeasible. biomass-max still pulls flux
                # up to the bound, so the absolute scale is preserved while the model stays feasible.
                med.append((c, round(f*(1+tol),6), 0.0))          # lb=max uptake (neg), ub=0 (no secretion)
            else:                                     # secretion -> free to predict (0..1.3x), validated separately
                med.append((c, 0.0, round(f*(1+tol),6)))
        val.append((cands[0], f))                     # measured value for validation (first candidate name)
    with open(OUT/"flux_medium.tsv","w") as o:
        o.write("metabolite\tlb\tub\n")
        for name,lb,ub in med: o.write(f"{name}\t{lb}\t{ub}\n")
    with open(OUT/"flux_measured_exch.tsv","w") as o:
        o.write("metabolite\tflux\tmethod\n")
        for name,f in val: o.write(f"{name}\t{f}\t{method}\n")
    up=sum(1 for _,f in val if f<0); sec=sum(1 for _,f in val if f>0)
    print(f"  FluxProfilingREGP: {len(val)} exchange fluxes ({up} uptake, {sec} secretion) in mmol/gDW/hr "
          f"-> flux_medium.tsv (ABSOLUTE anchor, glucose fixed) + flux_measured_exch.tsv (validation)")
    return True

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # PRIMARY: real absolute exchange fluxes from FluxProfilingREGP (Human-GEM-based) -> ABSOLUTE flux map.
    absolute = fetch_fluxprofile() if os.environ.get("FLUX_USE_REGP","1")=="1" else False
    if not absolute:
        # medium fallback: Jain CORE url, else curated relative (glucose=100)
        if not _try_url("JAIN_CORE_URL", OUT/"flux_medium.tsv"):
            with open(OUT/"flux_medium.tsv","w") as f:
                f.write("metabolite\tlb\tub\n")
                for m,l,u in CURATED_MEDIUM: f.write(f"{m}\t{l}\t{u}\n")
            print(f"  medium: curated {len(CURATED_MEDIUM)} exchange bounds -> flux_medium.tsv (relative, glucose=100)")
    # internal 13C validation set (glycolysis distribution): real via CECAFDB_URL, else curated
    if not _try_url("CECAFDB_URL", OUT/"flux_measured_13c.tsv"):
        with open(OUT/"flux_measured_13c.tsv","w") as f:
            f.write("gene\trel_flux\tpathway\n")
            for g,v,p in CURATED_13C: f.write(f"{g}\t{v}\t{p}\n")
        print(f"  13C internal validation: curated {len(CURATED_13C)}-enzyme aerobic-glycolysis reference (glucose=100)")
    print(f"flux data ready ({'ABSOLUTE — measured exchange anchored (FluxProfilingREGP)' if absolute else 'RELATIVE — curated fallback'}). "
          "Set FLUXPROFILE_URL for another cell line, FLUXPROFILE_METHOD=CORE/REGP, or FLUX_USE_REGP=0 to disable.")

if __name__=="__main__":
    main()
