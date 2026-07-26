"""THE CELL AS TYPED REACTIONS, NOT AS A PROTEIN-PROTEIN GRAPH.

WHAT WAS WRONG WITH EVERYTHING BEFORE THIS FILE. Every direction module in this project scored itself on PPI edges,
and that restriction quietly threw away most of what a reaction contains. A reaction is

    inputs (proteins, complexes, metabolites, cofactors)  --[catalyst]-->  outputs (products, energy, by-products)

and only a small minority of those participants are proteins. Asking "which way does this protein-protein edge
point" ignores the metabolite that was consumed, the ATP that was spent, and the enzyme that did the work unless it
happens to also be a PPI partner. Measured here: of the 191,447 PPI edges, only 18.23% have both endpoints in any
curated reaction at all. The interactome was never the right object.

WHAT THIS BUILDS. One typed reaction table over the two halves of human biology that are curated separately and
barely overlap:

    Reactome   signalling, transcription, trafficking, complex assembly -- protein-centric, 15,597 reactions
    Human-GEM  genome-scale metabolism -- metabolite-centric, 12,931 reactions, 8,461 metabolites, 2,848 enzymes

Both are already on disk. Each reaction becomes {inputs, outputs, catalysts} with EVERY participant typed
(PROTEIN / COMPLEX / SET / METABOLITE / DRUG / RNA / OTHER) rather than filtered down to proteins.

ENERGY AND BY-PRODUCTS ARE KEPT AND LABELLED, NOT DROPPED. Earlier modules excluded ATP, ADP, H2O and phosphate
because joining reactions on them produces a dense meaningless graph -- true when the goal is inferring which
reaction feeds which. But they are real outputs, and a model that cannot say a step SPENDS ATP is missing the point
of the step. So promiscuous species are tagged `currency` and left in place; any consumer that needs to avoid the
join-on-ATP trap filters on the tag rather than on a hard-coded blacklist.

WHAT IS MEASURED, since a build with no measurement is just a data dump: how much of the cell this covers that the
PPI view could not -- participants by type, genes reachable, protein-metabolite pairs gained, and the overlap
between the two sources, which is small enough that the union is close to a sum.
"""
import collections
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
TREE = SP / "entity_tree.json"
UP2G = SP / "reactome/up2gene.tsv"
P1 = SP / "car_pass1_reactions.json"
P2 = SP / "car_pass2_catalysts.json"
PECOMP = SP / "pe_compartment.json"
GEM = SP / "HumanGEM.xml"
CURRENCY_MIN = 200          # a species touching more reactions than this is tagged currency (swept, not assumed)

SC2TYPE = {"EntityWithAccessionedSequence": "PROTEIN", "GenomeEncodedEntity": "PROTEIN",
           "Complex": "COMPLEX", "DefinedSet": "SET", "CandidateSet": "SET", "Polymer": "COMPLEX",
           "SimpleEntity": "METABOLITE", "ChemicalDrug": "DRUG", "ProteinDrug": "DRUG", "RNADrug": "DRUG",
           "OtherEntity": "OTHER", "Cell": "OTHER"}


SUPP = SP / "entity_supp.json"


def supplement(missing):
    """Fetch entities absent from entity_tree.json and return them in the same shape.

    WHY THIS IS NEEDED AND WHY IT IS CHEAP. Reactome refers to an entity sometimes by stId and sometimes by bare
    integer dbId; entity_tree.json is keyed by stId only. Just 1,319 distinct ids are affected -- but they fill
    46,450 of 100,686 participant slots, because the ones referenced by dbId are the high-frequency currency species
    (ATP, ADP, H2O). Left unresolved they are typed OTHER, which is how a census can report 46% "OTHER" and look
    like a modelling choice rather than a join failure.
    """
    import time
    import urllib.error
    import urllib.request
    have = json.loads(SUPP.read_text()) if SUPP.exists() else {}
    todo = sorted(set(missing) - set(have))
    if not todo:
        return have
    ua = {"User-Agent": "Mozilla/5.0 (compatible; cell-network-research/1.0)",
          "Content-Type": "text/plain"}
    print(f"  resolving {len(todo):,} entities missing from the tree", flush=True)
    failed = 0
    for i in range(0, len(todo), 20):
        ok = False
        for attempt in range(6):
            try:
                req = urllib.request.Request("https://reactome.org/ContentService/data/query/ids",
                                             data=",".join(todo[i:i + 20]).encode(), headers=ua)
                with urllib.request.urlopen(req, timeout=120) as r:
                    for o in json.loads(r.read().decode()):
                        key = str(o.get("dbId")) if o.get("dbId") else o.get("stId")
                        if not key:
                            continue
                        re_ = o.get("referenceEntity")
                        have[key] = {"sc": o.get("schemaClass"), "nm": o.get("displayName"),
                                     "kids": [c.get("stId") or str(c.get("dbId")) for c in
                                              (o.get("hasComponent") or []) + (o.get("hasMember") or [])
                                              if isinstance(c, dict)],
                                     "up": (re_ or {}).get("identifier") if isinstance(re_, dict) else None}
                ok = True
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(float(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1))
                    continue
                break
            except Exception:
                time.sleep(2 ** attempt)
        failed += not ok
        time.sleep(1.2)
    SUPP.write_text(json.dumps(have))
    if failed:
        # LOUD, NOT SILENT. An earlier version of this function swallowed every failure and wrote an empty file,
        # so the census reported 46% of participants as type OTHER -- which reads as a modelling decision rather
        # than as "the join never happened". Reactome's origin returns 521 when it is down, which no amount of
        # retrying fixes; the run continues on what is on disk and says exactly what is missing.
        print(f"    !! {failed:,}/{-(-len(todo)//20):,} batches FAILED (Reactome unreachable). "
              f"{len(have):,} resolved; the rest stay UNRESOLVED and are reported as such, not as a type.",
              flush=True)
    return have


def load_reactome():
    T = json.loads(TREE.read_text())
    _rx = json.loads(P1.read_text())
    _need = {p for v in _rx.values() for p in v.get("in", []) + v.get("out", [])} - set(T)
    T.update(supplement(_need))
    up2g = {}
    for line in UP2G.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] and p[1]:
            up2g[p[0]] = p[1]
    rx = _rx
    ca = json.loads(P2.read_text()) if P2.exists() else {}
    pecomp = json.loads(PECOMP.read_text())["comp"]

    gmemo = {}

    def genes(pe, stack=()):
        if pe in gmemo:
            return gmemo[pe]
        if pe in stack:
            return set()
        d = T.get(pe)
        if d is None:
            return set()
        g = set()
        if d.get("up") and up2g.get(d["up"]):
            g.add(up2g[d["up"]])
        for k in d.get("kids") or []:
            g |= genes(k, stack + (pe,))
        gmemo[pe] = g
        return g

    def describe(pe):
        d = T.get(pe) or {}
        nm = d.get("nm") or pe
        m = re.match(r"^(.*) \[([^\]]+)\]$", nm)
        return {"id": pe, "type": (SC2TYPE.get(d.get("sc"), "OTHER") if d else "UNRESOLVED"),
                "name": m.group(1) if m else nm,
                "compartment": (m.group(2) if m else None) or pecomp.get(pe),
                "genes": sorted(genes(pe))}

    reactions = []
    for r, v in rx.items():
        cats = set()
        for c in v.get("ca") or []:
            d = ca.get(str(c))
            if not d:
                continue
            for x in (d.get("au") or ([d["pe"]] if d.get("pe") else [])):
                cats |= genes(x)
        reactions.append({"id": r, "source": "reactome", "category": v.get("cat"),
                          "in": [describe(p) for p in v.get("in", [])],
                          "out": [describe(p) for p in v.get("out", [])],
                          "catalysts": sorted(cats), "reversible": False})
    return reactions


def load_gem():
    """Parse Human-GEM SBML. iterparse with element clearing, because the file is 43 MB and a full DOM is wasteful.

    The FBC extension is what makes this worth parsing: `fbc:geneProduct` maps an Ensembl id to a gene SYMBOL, and
    `fbc:geneProductAssociation` attaches the enzymes to each reaction. That is the catalyst annotation for
    metabolism, which Reactome covers thinly.
    """
    ns = {"s": "http://www.sbml.org/sbml/level3/version1/core",
          "f": "http://www.sbml.org/sbml/level3/version1/fbc/version2"}
    species, gp, reactions = {}, {}, []
    cur = None
    for ev, el in ET.iterparse(str(GEM), events=("start", "end")):
        tag = el.tag.split("}")[-1]
        if ev == "end" and tag == "species":
            species[el.get("id")] = {"name": el.get("name"), "compartment": el.get("compartment"),
                                     "formula": el.get(f"{{{ns['f']}}}chemicalFormula")}
            el.clear()
        elif ev == "end" and tag == "geneProduct":
            gp[el.get(f"{{{ns['f']}}}id")] = el.get(f"{{{ns['f']}}}label")
            el.clear()
        elif ev == "start" and tag == "reaction":
            cur = {"id": el.get("id"), "reversible": el.get("reversible") == "true",
                   "in": [], "out": [], "cat": set()}
        elif ev == "end" and tag == "reaction" and cur is not None:
            reactions.append(cur)
            cur = None
            el.clear()
        elif ev == "end" and tag == "geneProductRef" and cur is not None:
            # STORE THE RAW REF AND RESOLVE IT LATER. fbc:listOfGeneProducts appears AFTER </listOfReactions> in
            # this file (line 557,588 vs 557,580), so `gp` is still empty while reactions are being read. Resolving
            # inline silently produced ZERO enzymes for all 12,931 metabolic reactions -- a whole half of the cell
            # with no catalysts and no error.
            ref = el.get(f"{{{ns['f']}}}geneProduct")
            if ref:
                cur["cat"].add(ref)
        elif ev == "end" and tag == "listOfReactants" and cur is not None:
            cur["in"] = [c.get("species") for c in el if c.get("species")]
        elif ev == "end" and tag == "listOfProducts" and cur is not None:
            cur["out"] = [c.get("species") for c in el if c.get("species")]

    def describe(sid):
        d = species.get(sid) or {}
        return {"id": sid, "type": "METABOLITE", "name": d.get("name") or sid,
                "compartment": d.get("compartment"), "formula": d.get("formula"), "genes": []}

    out = []
    for r in reactions:
        out.append({"id": r["id"], "source": "humangem", "category": "metabolic",
                    "in": [describe(s) for s in r["in"]], "out": [describe(s) for s in r["out"]],
                    "catalysts": sorted({gp[c] for c in r["cat"] if c in gp}), "reversible": r["reversible"]})
    return out, species, gp


def main():
    rea = load_reactome()
    gem, species, gp = load_gem()
    allrx = rea + gem
    print(f"reactions: {len(rea):,} Reactome + {len(gem):,} Human-GEM = {len(allrx):,}")
    print(f"  Human-GEM: {len(species):,} metabolite species, {len(gp):,} gene products")

    # ---- currency tagging: measured, not blacklisted ----
    touch = collections.Counter()
    for r in allrx:
        for p in r["in"] + r["out"]:
            touch[p["id"]] += 1
    currency = {e for e, c in touch.items() if c > CURRENCY_MIN}
    names = {}
    for r in allrx:
        for p in r["in"] + r["out"]:
            names.setdefault(p["id"], p["name"])
    print(f"\n  species touching >{CURRENCY_MIN} reactions, tagged `currency` (kept, not dropped): {len(currency):,}")
    print("    " + ", ".join(f"{names.get(e,e)}({touch[e]})" for e, _ in touch.most_common(10)))

    # ---- the census that says what the PPI view was missing ----
    ptype = collections.Counter()
    for r in allrx:
        for p in r["in"] + r["out"]:
            ptype[p["type"]] += 1
    print(f"\n  participant slots by type: {dict(ptype.most_common())}")
    known = sum(v for k, v in ptype.items() if k != "UNRESOLVED")
    nonprot = sum(v for k, v in ptype.items() if k not in ("PROTEIN", "COMPLEX", "SET", "UNRESOLVED"))
    print(f"    non-protein share of TYPED slots: {nonprot/max(known,1):.1%}  "
          f"<- invisible to a protein-protein graph")
    if ptype.get("UNRESOLVED"):
        print(f"    UNRESOLVED {ptype['UNRESOLVED']:,} slots ({ptype['UNRESOLVED']/sum(ptype.values()):.1%}) -- "
              f"entities Reactome refers to by bare dbId that could not be fetched; a join gap, not a type")

    genes_rea = {g for r in rea for p in r["in"] + r["out"] for g in p["genes"]} | \
                {g for r in rea for g in r["catalysts"]}
    genes_gem = {g for r in gem for g in r["catalysts"]}
    D = json.load(open(OUT / "cell_complete.json"))
    ppi_genes = set()
    names_all = [g["name"] for g in D["genes"]]
    N = len(names_all)
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            ppi_genes.add(names_all[a])
            ppi_genes.add(names_all[b])
    print(f"\n  genes with at least one reaction: Reactome {len(genes_rea):,}, Human-GEM {len(genes_gem):,}, "
          f"union {len(genes_rea | genes_gem):,}")
    print(f"    overlap between the two sources: {len(genes_rea & genes_gem):,} "
          f"({len(genes_rea & genes_gem)/max(len(genes_rea | genes_gem),1):.1%}) -- "
          f"they cover different halves, so the union is close to a sum")
    print(f"    genes in the PPI network: {len(ppi_genes):,}; "
          f"reaction genes NOT in it: {len(( genes_rea | genes_gem) - ppi_genes):,}")

    # ---- protein <-> non-protein pairs: the relations the interactome cannot express ----
    pm = set()
    pp = set()
    for r in allrx:
        prot = {g for p in r["in"] + r["out"] for g in p["genes"]} | set(r["catalysts"])
        mets = {p["id"] for p in r["in"] + r["out"] if p["type"] in ("METABOLITE", "DRUG") and p["id"] not in currency}
        for g in prot:
            for m in mets:
                pm.add((g, m))
        pl = sorted(prot)
        for i, a in enumerate(pl):
            for b in pl[i + 1:]:
                pp.add((a, b))
    print(f"\n  distinct protein-metabolite pairs (non-currency): {len(pm):,}")
    print(f"  distinct protein-protein co-participation pairs   : {len(pp):,}")
    print(f"    ratio: {len(pm)/max(len(pp),1):.2f} protein-metabolite relations per protein-protein one")

    steps = [{"id": r["id"], "source": r["source"], "category": r["category"], "reversible": r["reversible"],
              "in": [{"id": p["id"], "type": p["type"], "name": p["name"], "compartment": p["compartment"],
                      "genes": p["genes"], "currency": p["id"] in currency} for p in r["in"]],
              "out": [{"id": p["id"], "type": p["type"], "name": p["name"], "compartment": p["compartment"],
                       "genes": p["genes"], "currency": p["id"] in currency} for p in r["out"]],
              "catalysts": r["catalysts"]} for r in allrx]
    payload = {"n_reactions": len(allrx), "n_reactome": len(rea), "n_humangem": len(gem),
               "participant_types": dict(ptype), "nonprotein_share": nonprot / sum(ptype.values()),
               "n_currency": len(currency), "currency": sorted(currency),
               "genes_reactome": len(genes_rea), "genes_humangem": len(genes_gem),
               "genes_union": len(genes_rea | genes_gem), "genes_overlap": len(genes_rea & genes_gem),
               "genes_not_in_ppi": len((genes_rea | genes_gem) - ppi_genes),
               "protein_metabolite_pairs": len(pm), "protein_protein_pairs": len(pp),
               "steps": steps}
    json.dump(payload, open(OUT / "reaction_network.json", "w"))
    sz = (OUT / "reaction_network.json").stat().st_size / 1e6
    print(f"\n  -> {OUT/'reaction_network.json'} ({sz:.1f} MB, {len(steps):,} typed reaction steps)")


if __name__ == "__main__":
    main()
