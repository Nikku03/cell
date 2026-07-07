"""Enzyme-constrained flux — the quantitative rung of the mutation→phenotype chain.

Turns "flux drops" into "flux drops by X%". On the curated Human-GEM (FBA), each enzyme is given a finite
capacity and its activity can be dialed down (a mutation), re-solving for the flux/growth response. This
closes the coupling the model was missing: kinetics/stability → quantitative flux.

Three things it provides:
  1. flux_control_curve  — sweep an enzyme's activity 100%→0%, get the biomass/flux response (the shape that
     reproduces metabolic dominance: buffered near WT, collapses only near-complete loss → why IEMs are recessive).
  2. essentiality        — FBA single-gene deletion → which enzymes the cell cannot lose (validated vs known).
  3. mutation_to_flux    — ΔΔG → folded/active fraction (folding equilibrium) → capacity multiplier → Δflux,
     wiring the ΔΔG node (colab/ddg_predictor.py) into a quantitative flux consequence.

Honest scope: without absolute proteomics the per-enzyme capacity is set from WT flux and a saturation
assumption σ (enzyme runs at σ of capacity → 1/σ excess), so absolute %s carry that assumption; the essentiality
recovery and the *shape* of the dominance law are assumption-light.
"""
import os, urllib.request, numpy as np
from cobra.flux_analysis import pfba, single_gene_deletion

HUMANGEM_URL = "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.xml"
GLC, O2 = "MAR09034", "MAR09048"
RT_310 = 0.616  # kcal/mol at 310 K


def load_humangem(path="HumanGEM.xml"):
    import cobra
    if not os.path.exists(path):
        urllib.request.urlretrieve(HUMANGEM_URL, path)
    m = cobra.io.read_sbml_model(path); m.solver = 'glpk'
    if GLC in [r.id for r in m.reactions]:
        m.reactions.get_by_id(GLC).lower_bound = -1.0     # glucose-limited
        m.reactions.get_by_id(O2).lower_bound = -1000.0   # aerobic
    return m


def biomass_id(m):
    return [r.id for r in m.reactions if r.objective_coefficient != 0][0]


def flux_control_curve(m, enzyme_rxn_ids, sigma=0.5,
                       activities=(1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.05, 0.0)):
    """For each enzyme reaction, cap v ≤ activity·(flux_WT/σ) and record biomass fraction retained.
    Returns array [enzyme, activity]. σ = WT saturation (excess capacity = 1/σ)."""
    wt = pfba(m); wt_bio = m.slim_optimize()
    curves = []
    for rid in enzyme_rxn_ids:
        r = m.reactions.get_by_id(rid)
        C = abs(wt.fluxes[rid]) / sigma
        lo0, hi0 = r.bounds; row = []
        for a in activities:
            cap = a * C
            r.bounds = (max(lo0, -cap), min(hi0, cap))
            b = m.slim_optimize()
            row.append((b if b is not None else 0.0) / wt_bio)
        r.bounds = (lo0, hi0); curves.append(row)
    return np.array(curves), list(activities), wt_bio


def folded_fraction(ddg, dG_unfold_wt=7.0):
    """Active-enzyme multiplier from a folding-stability change ΔΔG (kcal/mol, >0 destabilizing).
    Two-state folding equilibrium: fraction folded = 1/(1+exp(−ΔG_unfold/RT)). A destabilizing mutation
    lowers ΔG_unfold by ΔΔG, shrinking the folded (active) fraction. Returns f_mut / f_wt in [0,1]."""
    f_wt = 1.0 / (1.0 + np.exp(-dG_unfold_wt / RT_310))
    f_mut = 1.0 / (1.0 + np.exp(-(dG_unfold_wt - ddg) / RT_310))
    return float(f_mut / f_wt)


def mutation_to_flux(m, enzyme_rxn_id, ddg, sigma=0.5, dG_unfold_wt=7.0):
    """Full quantitative step: ΔΔG → active fraction → capacity → biomass fraction retained."""
    mult = folded_fraction(ddg, dG_unfold_wt)
    wt = pfba(m); wt_bio = m.slim_optimize()
    r = m.reactions.get_by_id(enzyme_rxn_id)
    C = abs(wt.fluxes[enzyme_rxn_id]) / sigma
    lo0, hi0 = r.bounds
    r.bounds = (max(lo0, -mult*C), min(hi0, mult*C))
    bio = m.slim_optimize(); r.bounds = (lo0, hi0)
    return {"ddg": ddg, "active_fraction": round(mult, 3),
            "biomass_retained": round((bio or 0.0)/wt_bio, 3)}


def essentiality(m, biomass_frac=0.01):
    """FBA single-gene deletion → set of genes whose loss collapses biomass below `biomass_frac` of WT."""
    wt = m.slim_optimize()
    res = single_gene_deletion(m, m.genes)
    ess = {}
    for _, row in res.iterrows():
        ids = row['ids']; gid = list(ids)[0] if ids else None
        if gid is None: continue
        gr = row['growth']
        ess[gid] = (gr is None) or np.isnan(gr) or (gr < biomass_frac * wt)
    return ess
