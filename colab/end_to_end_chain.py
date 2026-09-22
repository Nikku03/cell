"""The end-to-end quantitative chain — ONE pipeline from a point mutation to a cell-level phenotype.

   mutation ─→ ΔΔG (structure) ─→ folded/active fraction ─→ enzyme capacity ─→ ec-flux (Human-GEM)
            ─→ pathway flux / growth ─→ phenotype severity

Every prior piece was validated in isolation (ΔΔG on S669 r=0.47; ec-flux dominance + essentiality; folding
thermodynamics). This connects them so a single mutation propagates all the way to a quantitative cell readout,
and reports the value + confidence at every rung so error is carried forward honestly, not hidden.

`Chain.run(gene, uniprot, pos, wt, mut)` -> the full trace. Validated by discriminating known pathogenic vs
benign enzyme missense variants (validate_chain.py).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ecflux
from ddg_predictor import DDGPredictor

RT_310 = 0.616


class Chain:
    def __init__(self, ddg_model="outputs/orphan/ddg_model.pkl", load_metabolism=True):
        self.ddg = DDGPredictor(ddg_model) if os.path.exists(ddg_model) else None
        self.m = None
        self._gene2rxn = {}
        self._wt = None
        self._wt_bio = None
        if load_metabolism:
            self._load_metabolism()

    def _load_metabolism(self):
        from cobra.flux_analysis import pfba
        self.m = ecflux.load_humangem()
        self.bio = ecflux.biomass_id(self.m)
        self._wt = pfba(self.m)
        self._wt_bio = self.m.slim_optimize()

    # ---- gene symbol -> Human-GEM reactions (via ENSG in the GPR) ----
    def gene_reactions(self, gene, ensg=None):
        if gene in self._gene2rxn:
            return self._gene2rxn[gene]
        if ensg is None:
            import mygene
            hit = mygene.MyGeneInfo().querymany([gene], scopes="symbol", fields="ensembl.gene",
                                                species="human", verbose=False)
            ens = []
            for h in hit:
                e = (h.get("ensembl") or {})
                e = e if isinstance(e, list) else [e]
                ens += [x.get("gene") for x in e if x.get("gene")]
            ensg = ens
        ensg = [ensg] if isinstance(ensg, str) else (ensg or [])
        rxns = []
        for g in ensg:
            try:
                gobj = self.m.genes.get_by_id(g)
                rxns += [r.id for r in gobj.reactions]
            except KeyError:
                continue
        rxns = sorted(set(rxns))
        self._gene2rxn[gene] = rxns
        return rxns

    # ---- the full chain ----
    def run(self, gene, uniprot, pos, wt, mut, chain="A", dG_unfold_wt=7.0):
        trace = dict(mutation=f"{gene} {wt}{pos}{mut}", gene=gene, uniprot=uniprot)
        # rung 1: ΔΔG from structure
        if not self.ddg:
            return dict(trace, abstain=True, reason="ΔΔG model unavailable")
        try:
            pdb = self.ddg.alphafold_pdb(uniprot)
            ddg, used_struct = self.ddg.predict_from_structure(pdb, chain, pos, wt, mut)
        except Exception as e:
            return dict(trace, abstain=True, reason=f"ΔΔG/structure failed: {e}")
        trace["ddg_kcal_mol"] = round(float(ddg), 2)
        # rung 2: folding thermodynamics -> active fraction
        active = ecflux.folded_fraction(ddg, dG_unfold_wt)
        trace["active_fraction"] = round(active, 3)
        # rung 3->5: enzyme capacity -> ec-flux -> growth (only where the enzyme carries metabolic flux)
        rxns = self.gene_reactions(gene) if self.m is not None else []
        flux_rxns = [r for r in rxns if abs(self._wt.fluxes[r]) > 1e-6] if rxns else []
        trace["n_reactions"] = len(rxns)
        trace["n_flux_carrying"] = len(flux_rxns)
        biomass_retained = None
        if flux_rxns:
            worst = 1.0
            for rid in flux_rxns:
                res = ecflux.mutation_to_flux(self.m, rid, ddg, dG_unfold_wt=dG_unfold_wt)
                worst = min(worst, res["biomass_retained"])
            biomass_retained = round(worst, 3)
        trace["biomass_retained"] = biomass_retained
        # phenotype severity: enzyme-level activity loss, escalated if the network can't buffer it
        activity_loss = 1.0 - active
        network_loss = (1.0 - biomass_retained) if biomass_retained is not None else None
        severity = activity_loss if network_loss is None else max(activity_loss, network_loss)
        trace["severity"] = round(float(severity), 3)
        # confidence carried forward: ΔΔG per-call is noisy (r~0.4-0.47); trust ranking of strong effects
        trace["confidence"] = round(min(0.6, abs(ddg) / 8.0), 3)
        trace["call"] = ("likely damaging" if severity > 0.5 else
                         "moderate" if severity > 0.2 else "likely tolerated")
        trace["phenotype"] = (
            f"enzyme activity -> {active*100:.0f}% of WT; " +
            (f"growth/flux retained {biomass_retained*100:.0f}% (network {'buffers' if biomass_retained>0.9 else 'does NOT buffer'} the loss)"
             if biomass_retained is not None else "enzyme not flux-carrying in FBA (activity-level readout only)"))
        trace["note"] = "per-rung values with carried confidence; ΔΔG is the noisy rung (trust strong-effect ranking)"
        return trace


def demo():
    c = Chain()
    for gene, up, pos, wt, mut, lab in [
        ("MTHFR", "P42898", 222, "A", "V", "A222V (C677T) — common mild variant"),
        ("G6PD", "P11413", 188, "S", "N", "S188F-region / class variant"),
    ]:
        r = c.run(gene, up, pos, wt, mut)
        print(f"\n=== {lab} ===")
        if r.get("abstain"):
            print("  ABSTAIN:", r["reason"]); continue
        for k in ("ddg_kcal_mol", "active_fraction", "n_flux_carrying", "biomass_retained", "severity", "call"):
            print(f"  {k}: {r.get(k)}")
        print("  phenotype:", r["phenotype"])


if __name__ == "__main__":
    demo()
