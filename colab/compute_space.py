"""SPACE (Problem 2) — subcellular spatial structure, honestly at COMPARTMENT resolution.

We can't do continuous geometry/diffusion (no data constrains it), but we CAN give real spatial structure:
  1. MULTI-compartment localization per protein from HPA immunofluorescence (main + additional locations,
     with a reliability tier) -- upgrades the model's single coarse `comp` tag to the true set of places a
     protein is, including multi-localized proteins (moonlighting/trafficking).
  2. COMPARTMENT TRANSPORT topology from Human-GEM: which metabolites move between compartments and via
     which reactions -> the cell's internal logistics network.
  3. CO-LOCALIZATION consistency: a PPI is spatially plausible only if the partners share a compartment;
     cross-compartment interactions flag membrane-contact-site / trafficking dependence.

Reads HPA subcellular (fetched) + Human-GEM (via compute_flux.parse_gem) + cell_complete (PPI). Skips
gracefully if inputs absent. -> space.json {locations, compartment_transport, colocalization}
"""
import json, os, io, zipfile, urllib.request
from collections import defaultdict, Counter
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
HPA_URL=os.environ.get("HPA_SUBCELL_URL","https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip")

def load_hpa():
    """symbol -> {main:[...], additional:[...], all:[...], reliability}. From HPA immunofluorescence."""
    f=H/"subcellular_location.tsv"
    if not f.exists():
        z=H/"subcellular_location.tsv.zip"
        if not z.exists():
            try:
                print("  fetching HPA subcellular localization ..."); urllib.request.urlretrieve(HPA_URL, z)
            except Exception as e:
                print("  HPA fetch failed:",repr(e)[:100],"-> spatial localization skipped"); return {}
        try:
            zf=zipfile.ZipFile(z); name=next(n for n in zf.namelist() if n.endswith(".tsv"))
            open(f,"wb").write(zf.read(name))
        except Exception as e:
            print("  HPA unzip failed:",repr(e)[:80]); return {}
    import csv
    m={}; rd=csv.DictReader(open(f), delimiter="\t")
    for r in rd:
        sym=(r.get("Gene name") or "").strip()
        if not sym: continue
        main=[x.strip() for x in (r.get("Main location") or "").split(";") if x.strip()]
        add =[x.strip() for x in (r.get("Additional location") or "").split(";") if x.strip()]
        ext =[x.strip() for x in (r.get("Extracellular location") or "").split(";") if x.strip()]
        allloc=sorted(set(main+add+ext))
        if allloc:
            m[sym]=dict(main=main, additional=add, all=allloc, reliability=(r.get("Reliability") or "").strip())
    return m

def compartment_transport(rxns, met_idx):
    """metabolite base-name moving between compartments -> transport reactions. Human-GEM logistics."""
    import re
    inv={i:n for n,i in met_idx.items()}
    def comp(n):
        m=re.search(r"\[([^\]]*)\]\s*$", n); return m.group(1) if m else "?"
    def base(n): return re.sub(r"\[[^\]]*\]\s*$","",n).strip()
    edges=Counter(); rxn_ex={}
    for r in rxns:
        comps=defaultdict(set)      # base metabolite -> set of compartments it appears in (this reaction)
        for mi in r["stoich"]:
            nm=inv[mi]; comps[base(nm)].add(comp(nm))
        for bn,cs in comps.items():
            if len(cs)>=2:          # same metabolite in >=2 compartments = a transport step
                cl=sorted(cs)
                for a in range(len(cl)):
                    for b in range(a+1,len(cl)):
                        edges[(cl[a],cl[b])]+=1
                        rxn_ex.setdefault((cl[a],cl[b]), r["id"])
    return edges, rxn_ex

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> space skipped"); return
    D=json.load(open(cc)); G=D["genes"]; names=set(g["name"] for g in G)
    idx={g["name"]:i for i,g in enumerate(G)}
    hpa=load_hpa()
    loc={}
    for g in G:
        h=hpa.get(g["name"])
        if h:
            loc[g["name"]]=dict(locations=h["all"], main=h["main"], n_loc=len(h["all"]),
                                multi=len(h["all"])>1, reliability=h["reliability"], model_comp=g.get("comp"))
    # compartment transport from Human-GEM
    transport=[]
    try:
        from compute_flux import parse_gem
        rxns, met_idx = parse_gem()
        if rxns:
            edges, rxn_ex = compartment_transport(rxns, met_idx)
            transport=[dict(**{"from":a}, to=b, n_reactions=n, example=rxn_ex.get((a,b)))
                       for (a,b),n in edges.most_common(60)]
    except Exception as e:
        print("  compartment transport skipped:",repr(e)[:80])
    # co-localization consistency of PPIs (share a compartment vs cross-compartment)
    same=0; cross=0; checked=0
    for a,b in D.get("ppi",[])[:200000]:
        na,nb=G[a]["name"],G[b]["name"]
        la=set(loc.get(na,{}).get("locations",[])); lb=set(loc.get(nb,{}).get("locations",[]))
        if la and lb:
            checked+=1
            if la&lb: same+=1
            else: cross+=1
    multi=sum(1 for v in loc.values() if v["multi"])
    comp_occ=Counter(l for v in loc.values() for l in v["locations"])
    payload=dict(locations=loc, compartment_transport=transport,
                 colocalization=dict(ppi_checked=checked, same_compartment=same, cross_compartment=cross,
                                     same_frac=round(same/checked,3) if checked else None),
                 summary=dict(proteins_localized=len(loc), multi_localized=multi,
                              n_transport_edges=len(transport),
                              top_compartments=comp_occ.most_common(12)))
    json.dump(payload, open(OUT/"space.json","w"))
    s=payload["summary"]
    print(f"space: {s['proteins_localized']} proteins with HPA localization ({s['multi_localized']} multi-localized) | "
          f"{s['n_transport_edges']} compartment-transport edges | PPI co-localization: "
          f"{payload['colocalization']['same_frac']} share a compartment (n={checked})")

if __name__=="__main__":
    main()
