"""incell_rates — turn the in-vitro kcat into IN-CELL quantities, two honest ways, neither needing measured flux.

kcat (CatPred/BRENDA) is a constant measured in buffer. Two cell-relevant quantities derive from it:

  A) CELL-TYPE CAPACITY   Vmax(enzyme, cell-type) = kcat . abundance,  gated by whether the enzyme is expressed
     in that cell type (the 200-cell-type emask). This is the max flux a cell type could push through the enzyme.
     Honest scope: abundance is the single reference proteome (ppm); cell-type variation is the on/off expression
     gate (emask is presence/absence, not graded levels), so magnitudes are reference-scaled, presence is real.

  B) SATURATION OPERATING RATE   v = kcat . [S]/([S]+Km).  The enzyme's substrate concentration [S] (its "solvent")
     pulls the in-vitro kcat down to the rate it actually runs at in the cell. [S] comes from the enzyme's
     Human-GEM reaction substrate matched to a curated set of measured mammalian intracellular concentrations
     (Park 2016 / Bennett 2009 order-of-magnitude). Honest scope: one Km per enzyme paired to its primary
     substrate; concentrations are literature-typical, not cell-type-specific (no such human data exists); so the
     saturation sigma is a coarse operating estimate, validated by whether its distribution matches the measured
     in-vivo median (~0.5, Davidi 2016).

Nothing measured is overwritten; outputs are additive overlays.
  -> outputs/orphan/celltype_capacity.json    {gene_idx: {vmax_ref, n_celltypes, log10_kcat_ppm}}
  -> outputs/orphan/saturation_kapp.json       {gene_idx: {substrate, conc_uM, km_uM, sigma, v_incell_per_s}}
  -> outputs/orphan/incell_rates_validation.json
"""
import os, sys, json, urllib.request, ssl
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
_CA = "/root/.ccr/ca-bundle.crt"
HUMANGEM_URL = "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.xml"
GENES_TSV_URL = "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/genes.tsv"

# measured mammalian intracellular concentrations (uM), order-of-magnitude from Park 2016 (Nat Chem Biol, iBMK
# cells) and Bennett 2009 / standard references. Keyed by normalized Human-GEM metabolite name. Cofactors last.
INTRACELL_UM = {
    "glucose": 1000, "glucose-6-phosphate": 200, "fructose-6-phosphate": 80,
    "fructose-1,6-bisphosphate": 200, "dihydroxyacetone phosphate": 100,
    "glyceraldehyde-3-phosphate": 20, "3-phospho-d-glycerate": 60, "2-phospho-d-glycerate": 15,
    "phosphoenolpyruvate": 20, "pyruvate": 100, "lactate": 2000,
    "acetyl-coa": 30, "citrate": 400, "isocitrate": 20, "2-oxoglutarate": 100,
    "succinate": 500, "fumarate": 100, "malate": 500, "oxaloacetate": 5,
    "6-phospho-d-gluconate": 50, "ribose-5-phosphate": 50, "ribulose-5-phosphate": 40,
    "sedoheptulose-7-phosphate": 30, "erythrose-4-phosphate": 20,
    "glutamate": 10000, "glutamine": 5000, "aspartate": 2000, "asparagine": 300,
    "alanine": 3000, "glycine": 2000, "serine": 1000, "threonine": 400, "proline": 400,
    "valine": 400, "leucine": 300, "isoleucine": 200, "methionine": 100, "phenylalanine": 100,
    "tyrosine": 100, "tryptophan": 40, "histidine": 100, "lysine": 400, "arginine": 300,
    "glutathione": 5000, "phosphocreatine": 5000, "creatine": 5000, "glycerol-3-phosphate": 200,
    "udp-glucose": 300, "s-adenosylmethionine": 15, "s-adenosylhomocysteine": 5,
    # cofactors (ubiquitous; used only if no primary substrate has a concentration)
    "atp": 3000, "adp": 500, "amp": 100, "gtp": 500, "gdp": 150, "utp": 500, "ctp": 200,
    "nad+": 500, "nadh": 100, "nadp+": 20, "nadph": 100, "coa": 100, "fad": 20,
}
_COFACTORS = {"atp", "adp", "amp", "gtp", "gdp", "utp", "ctp", "nad+", "nadh", "nadp+",
              "nadph", "coa", "fad", "h2o", "h+", "phosphate", "pi", "co2", "o2", "nh3"}


def _fetch(url, path):
    if os.path.exists(path):
        return path
    ctx = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else None
    print(f"  downloading {url.split('/')[-1]} ...")
    open(path, "wb").write(urllib.request.urlopen(url, timeout=300, context=ctx).read())
    return path


def _norm(name):
    """normalize a metabolite name for concentration lookup: lowercase, drop stereo prefixes/compartment."""
    n = (name or "").strip().lower()
    for pre in ("l-", "d-", "(s)-", "(r)-"):
        if n.startswith(pre):
            n = n[len(pre):]
    return n


# ---------------- Part A: cell-type capacity ----------------
def celltype_capacity(C=None, save=True):
    """Vmax_ref = kcat . ppm per enzyme; the 200-cell-type emask says where it is present. Emits reference Vmax +
    the # of cell types the enzyme is active in. A tissue-specific enzyme is active in few cell types, a
    housekeeping one in most — the cell-type capacity map falls out of Vmax_ref gated by emask."""
    C = C or CompleteCell()
    idx = C.idx
    kin = C.kin
    ppm = {int(k): float(v) for k, v in C.D.get("ppm", {}).items()}
    em = {int(k): int(v) for k, v in C.D.get("emask", {}).items()}
    out = {}
    for sym, v in kin.items():
        kc = v.get("kcat_per_s", 0)
        gi = idx.get(sym)
        if gi is None or gi not in ppm or kc <= 0:
            continue
        if ppm[gi] <= 0:
            continue
        vmax = kc * ppm[gi]
        nct = bin(em[gi]).count("1") if gi in em else None
        out[str(gi)] = {"vmax_ref": float(f"{vmax:.4g}"), "log10_kcat_ppm": round(float(np.log10(vmax)), 3),
                        "n_celltypes": nct}
    with_ct = [o for o in out.values() if o["n_celltypes"] is not None]
    summary = {
        "n_enzymes": len(out),
        "n_with_celltype_gate": len(with_ct),
        "median_celltypes_active": int(np.median([o["n_celltypes"] for o in with_ct])) if with_ct else None,
        "n_housekeeping_gt150ct": sum(1 for o in with_ct if o["n_celltypes"] > 150),
        "n_restricted_lt20ct": sum(1 for o in with_ct if o["n_celltypes"] < 20),
    }
    if save:
        json.dump({"capacity": out, "meta": summary}, open(f"{OUT}/celltype_capacity.json", "w"))
    return summary, out


# ---------------- Part B: saturation operating rate ----------------
def saturation_rates(C=None, model_path="HumanGEM.xml", save=True):
    """v = kcat . [S]/([S]+Km). Map each enzyme (with kcat+Km) to its Human-GEM reaction substrates, take the
    primary (non-cofactor) substrate that has a measured intracellular concentration, compute the saturation
    sigma and the operating rate. Validate the sigma distribution against the measured in-vivo median (~0.5)."""
    import cobra
    _fetch(HUMANGEM_URL, model_path)
    genes_path = _fetch(GENES_TSV_URL, "HumanGEM_genes.tsv")
    C = C or CompleteCell()
    idx = C.idx
    kin = C.kin
    # ENSG -> symbol
    ens2sym = {}
    with open(genes_path) as fh:
        hdr = [h.strip().strip('"').lower() for h in fh.readline().split("\t")]
        gi_ = next((i for i, h in enumerate(hdr) if h == "genes"), 0)
        si_ = next((i for i, h in enumerate(hdr) if "symbol" in h), 4)
        for ln in fh:
            p = [x.strip().strip('"') for x in ln.split("\t")]
            if len(p) > max(gi_, si_) and p[gi_] and p[si_]:
                ens2sym[p[gi_]] = p[si_].split(";")[0]
    print("  loading Human-GEM (cobra) ...")
    model = cobra.io.read_sbml_model(model_path)

    out = {}
    for g in model.genes:
        sym = ens2sym.get(g.id)
        if not sym or sym not in idx or sym not in kin:
            continue
        rec = kin[sym]
        kc, km = rec.get("kcat_per_s", 0), rec.get("km_uM", 0)
        if not (kc > 0 and km and km > 0):
            continue
        gi = idx[sym]
        # a database Km refers to the PRIMARY substrate, not a ubiquitous cofactor, so pair Km only with primary
        # metabolites that have a measured concentration (cofactors are excluded — using [NADPH] etc. is meaningless)
        primary = []
        for r in g.reactions:
            for mt in r.reactants:
                nm = _norm(mt.name)
                if nm in INTRACELL_UM and nm not in _COFACTORS:
                    primary.append((nm, INTRACELL_UM[nm]))
        if not primary:
            continue
        n_sub = len(set(s for s, _ in primary))
        subname, conc = max(primary, key=lambda x: x[1])   # if >1 primary, the more abundant (flagged via n_sub)
        sigma = conc / (conc + km)
        out[str(gi)] = {"symbol": sym, "substrate": subname, "conc_uM": conc, "km_uM": round(float(km), 2),
                        "kcat_per_s": round(float(kc), 3), "sigma": round(float(sigma), 3),
                        "v_incell_per_s": round(float(kc * sigma), 3), "n_primary_substrates": n_sub}
    sigmas = [o["sigma"] for o in out.values()]
    summary = {
        "n_enzymes_with_operating_rate": len(out),
        "median_sigma": round(float(np.median(sigmas)), 3) if sigmas else None,
        "davidi_invivo_median_sigma": 0.5,
        "frac_below_half_saturated": round(float(np.mean([s < 0.5 for s in sigmas])), 3) if sigmas else None,
        "n_ambiguous_multi_substrate": sum(1 for o in out.values() if o["n_primary_substrates"] > 1),
        "sigma_p10_p50_p90": [round(float(np.percentile(sigmas, p)), 3) for p in (10, 50, 90)] if sigmas else None,
    }
    if save:
        json.dump({"kapp": out, "meta": summary}, open(f"{OUT}/saturation_kapp.json", "w"))
    return summary, out


def run():
    C = CompleteCell()
    capA, _ = celltype_capacity(C)
    capB, _ = saturation_rates(C)
    val = {"celltype_capacity": capA, "saturation_operating_rate": capB}
    json.dump(val, open(f"{OUT}/incell_rates_validation.json", "w"))
    print("=" * 74)
    print("IN-CELL RATES FROM kcat  (two overlays, no measured flux needed)")
    print("=" * 74)
    print("  A) CELL-TYPE CAPACITY  (Vmax = kcat . ppm, gated by 200-cell-type emask):")
    print(f"       enzymes                {capA['n_enzymes']:,}  (with cell-type gate {capA['n_with_celltype_gate']:,})")
    print(f"       median cell types active {capA['median_celltypes_active']}/200   "
          f"housekeeping(>150) {capA['n_housekeeping_gt150ct']:,}  restricted(<20) {capA['n_restricted_lt20ct']:,}")
    print("  B) SATURATION OPERATING RATE  (v = kcat . [S]/([S]+Km), primary substrate only):")
    print(f"       enzymes with an in-cell rate  {capB['n_enzymes_with_operating_rate']:,}  "
          f"(multi-substrate/ambiguous {capB['n_ambiguous_multi_substrate']:,})")
    print(f"       median saturation sigma        {capB['median_sigma']}   "
          f"(Davidi measured in-vivo median {capB['davidi_invivo_median_sigma']})")
    print(f"       sigma p10/p50/p90              {capB['sigma_p10_p50_p90']}   "
          f"frac sub-half-saturated {capB['frac_below_half_saturated']}")
    print("=" * 74)
    return val


if __name__ == "__main__":
    run()
