"""metabolite_enzyme — connect the isolated metabolite layer to the enzyme/kinetics layer.

The metabolite layer (Human-GEM species, HMDB) sits unconnected to genes. But generxn already carries each
enzyme's reaction EQUATIONS — the metabolites are right there in the strings. This parses them into a bipartite
index:  metabolite <-> the enzymes that produce/consume it.  That opens the traversal
    metabolite -> enzymes -> reaction -> pathway/disease
which did not exist before. Hub cofactors (ATP, H2O, NADH...) touch thousands of enzymes and are down-weighted
so the index surfaces specific substrate relationships, not the ubiquitous ones.

-> outputs/orphan/metabolite_enzyme.json   {met2enz:{name:[gene_idx]}, enz2met:{gene_idx:[names]}, meta:{...}}
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
_COEF = re.compile(r"^\s*\d+(?:\.\d+)?\s+")          # leading stoichiometric coefficient
_COMP = re.compile(r"\[[a-z]\]")                     # compartment tag [c]/[e]/[m]... (anywhere)
_ARROW = re.compile(r"<=>|=>|->|→|<->|⇌")
# ubiquitous cofactors / currency metabolites — real but non-specific (touch ~everything)
_HUBS = {"h2o", "h+", "atp", "adp", "amp", "pi", "ppi", "nad+", "nadh", "nadp+", "nadph", "co2", "o2",
         "coa", "nh3", "nh4+", "h2o2", "gtp", "gdp", "utp", "udp", "fad", "fadh2", "hco3-", "na+", "k+"}


def _mets(eq):
    """metabolite names from one reaction equation string (both sides), coefficient + compartment stripped."""
    out = set()
    for side in _ARROW.split(eq):
        for term in side.split(" + "):
            t = _COMP.sub("", _COEF.sub("", term)).strip()
            if t:
                out.add(t)
    return out


def run():
    C = CompleteCell()
    generxn = C.D.get("generxn", {})
    enz2met, met2enz = {}, {}
    for g, rxns in generxn.items():
        try:
            gi = int(g)
        except (TypeError, ValueError):
            continue
        names = set()
        for eq in rxns:
            names |= _mets(eq)
        if not names:
            continue
        enz2met[gi] = sorted(names)
        for nm in names:
            met2enz.setdefault(nm, []).append(gi)

    # separate specific substrates from hub cofactors by degree + name
    deg = {m: len(e) for m, e in met2enz.items()}
    hubs = {m for m in met2enz if m.lower() in _HUBS or deg[m] > 400}
    specific = {m: v for m, v in met2enz.items() if m not in hubs}
    top = sorted(specific.items(), key=lambda kv: -len(kv[1]))[:10]

    meta = {
        "n_enzymes_linked": len(enz2met),
        "n_metabolites": len(met2enz),
        "n_specific_metabolites": len(specific),
        "n_hub_cofactors": len(hubs),
        "top_specific_metabolites": [{"metabolite": m, "n_enzymes": len(e)} for m, e in top],
    }
    json.dump({"met2enz": met2enz, "enz2met": {str(k): v for k, v in enz2met.items()},
               "hubs": sorted(hubs), "meta": meta}, open(f"{OUT}/metabolite_enzyme.json", "w"))
    print("=" * 68)
    print("METABOLITE <-> ENZYME  (parsed from Human-GEM reactions in generxn)")
    print("=" * 68)
    print(f"  enzymes linked to metabolites   {meta['n_enzymes_linked']:,}")
    print(f"  distinct metabolites            {meta['n_metabolites']:,}  "
          f"({meta['n_specific_metabolites']:,} specific, {meta['n_hub_cofactors']} hub cofactors)")
    print(f"  most-connected specific metabolites:")
    for m, e in top[:6]:
        print(f"     {m[:40]:40} {len(e):>4} enzymes")
    print("=" * 68)
    return meta


if __name__ == "__main__":
    run()
