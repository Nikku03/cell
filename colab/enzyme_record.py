"""enzyme_record — one joined record per enzyme: the full context around its kinetics, assembled from the data
we actually have, with the in-vitro/in-vivo values and the HONEST error between them attached.

Joins, per enzyme:
  - uniprot accession (sequence retrievable via molecular_engine.uniprot_seq)
  - PPI partners           (CompleteCell.ppi_adj)
  - reaction metabolites   (metabolite_enzyme.json: enz2met -> substrates/products)
  - in-vitro kcat + tier   (kinetics_refined_corrected)
  - in-vivo kcat           (davidi_* -- FLAGGED cross-species proxy, see kinetic_confidence)
  - in-vitro/in-vivo error (ratio + log10 fold + the kinetic_confidence sigma)
  - pathway membership     (Reactome pathways)
  - in-cell operating rate (saturation_kapp: v_incell) -- the closest thing we have to flux for the enzyme
Honest gaps stated inline: true PATHWAY flux is not measured (only per-enzyme in-cell rate for 270); the in-vivo
kcat is a proxy; most kcat is predicted.
-> outputs/orphan/enzyme_records.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import kinetic_confidence as kc

OUT = "outputs/orphan"


def _load(f):
    p = os.path.join(OUT, f)
    return json.load(open(p)) if os.path.exists(p) else None


def build(C=None, max_partners=15, max_reactions=12):
    if C is None:
        from complete_cell import CompleteCell
        C = CompleteCell()
    kin = (_load("kinetics_refined_corrected.json") or {}).get("kinetics_refined", {})
    me = _load("metabolite_enzyme.json") or {}
    enz2met = me.get("enz2met", {})
    kapp = (_load("saturation_kapp.json") or {}).get("kapp", {})
    kapp_by_idx = {int(k): v for k, v in kapp.items()}
    # invert pathways: gene index -> [pathway names]
    import cancer_cell_map as ccm
    paths = ccm._pathways(C)
    gene2path = {}
    for pw, members in (paths or {}).items():
        for gi in members:
            gene2path.setdefault(gi, []).append(pw)

    records = {}
    for g, kd in kin.items():
        i = C.idx.get(g)
        if i is None:
            continue
        sig, caveats = kc.rate_uncertainty(kd)
        invitro = kd.get("kcat_original", kd.get("kcat_per_s"))
        invivo = kd.get("davidi_kcat_max_per_s") or kd.get("invivo_apparent_kcat_per_s")
        ratio = round(invivo / invitro, 3) if (invitro and invivo and invitro > 0) else None
        rec = {
            "gene": g,
            "uniprot": C.genes[i].get("uniprot"),   # None for many core genes -> resolve via uniprot_seq(g)
            "ppi_partners": [C.name[j] for j in list(C.ppi_adj.get(i, []))[:max_partners]],
            "n_ppi": len(C.ppi_adj.get(i, [])),
            "reaction_metabolites": (enz2met.get(str(i)) or enz2met.get(i) or [])[:max_reactions],
            "kcat_invitro_per_s": invitro,
            "kcat_tier": kd.get("tier"),
            "kcat_invivo_per_s": invivo,
            "invivo_over_invitro": ratio,
            "kcat_log10_sigma": sig,
            "kcat_fold_uncertainty": round(10 ** sig, 1),
            "km_uM": kd.get("km_uM"),
            "pathways": gene2path.get(i, [])[:8],
            "n_pathways": len(gene2path.get(i, [])),
            "incell_rate_per_s": kapp_by_idx.get(i, {}).get("v_incell_per_s"),
            "incell_saturation_sigma": kapp_by_idx.get(i, {}).get("sigma"),
            "caveats": caveats + (["pathway FLUX not measured; incell_rate is per-enzyme, available for ~270 only"]
                                  if kapp_by_idx.get(i) is None else []),
        }
        records[g] = rec
    json.dump(records, open(f"{OUT}/enzyme_records.json", "w"), indent=1, default=str)
    return records


def _coverage(recs):
    n = len(recs)
    def frac(k, pred=lambda v: v not in (None, [], 0)):
        return sum(1 for r in recs.values() if pred(r.get(k))) / n
    print(f"enzyme records built: {n}")
    for k in ["kcat_invitro_per_s", "km_uM", "kcat_invivo_per_s", "ppi_partners",
              "reaction_metabolites", "pathways", "incell_rate_per_s"]:
        print(f"   {k:24} present for {frac(k):.0%}")
    meas = sum(1 for r in recs.values() if r.get("kcat_tier") in ("measured", "EC-measured"))
    print(f"   kcat MEASURED (not predicted): {meas} ({meas/n:.0%})")


if __name__ == "__main__":
    recs = build()
    _coverage(recs)
    import random
    ex = recs.get("HNF4A") or recs[random.Random(0).choice(list(recs))]
    print("\nexample record:")
    print(json.dumps(ex, indent=1, default=str)[:900])
