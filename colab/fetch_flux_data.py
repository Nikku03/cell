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

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # medium: try Jain CORE, else curated. (If a Jain file is provided it is expected pre-mapped to
    # metabolite\tlb\tub; we don't guess its schema — presence just suppresses the fallback.)
    got_medium=_try_url("JAIN_CORE_URL", OUT/"flux_medium.tsv")
    if not got_medium:
        with open(OUT/"flux_medium.tsv","w") as f:
            f.write("metabolite\tlb\tub\n")
            for m,l,u in CURATED_MEDIUM: f.write(f"{m}\t{l}\t{u}\n")
        print(f"  medium: curated {len(CURATED_MEDIUM)} exchange bounds -> flux_medium.tsv (glucose uptake fixed=100)")
    got_13c=_try_url("CECAFDB_URL", OUT/"flux_measured_13c.tsv")
    if not got_13c:
        with open(OUT/"flux_measured_13c.tsv","w") as f:
            f.write("gene\trel_flux\tpathway\n")
            for g,v,p in CURATED_13C: f.write(f"{g}\t{v}\t{p}\n")
        print(f"  13C validation: curated {len(CURATED_13C)}-enzyme aerobic-glycolysis reference "
              f"(glucose=100) -> flux_measured_13c.tsv")
    print("flux data ready. (Set JAIN_CORE_URL / CECAFDB_URL to use real per-cell-line data for a "
          "rigorous held-out test instead of the phenotype-level curated set.)")

if __name__=="__main__":
    main()
