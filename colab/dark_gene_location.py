"""SPATIAL DATA FOR THE DARK GENES, AND USING IT TO FALSIFY THEIR PATHWAY PLACEMENTS.

WHAT IS AVAILABLE. 4,806 of the 4,993 dark genes (96.3%) carry a curated UniProt SUBCELLULAR LOCATION string, and
514 of the 520 that received a predicted pathway do. That is a FACT tier -- curated literature -- exactly like the
function annotation, and unlike the pathway prediction it sits on.

WHY THAT COMBINATION IS WORTH MORE THAN EITHER PIECE. The pathway placement came from PPI guilt-by-association over
Reactome membership. The location came from experimental literature. They are independent, so location can FALSIFY a
placement: a protein documented as secreted cannot be doing its work inside a mitochondrial matrix pathway. This is
a cheap, strong consistency test that costs no new data, and it is the kind of test a prediction should have to pass
before being written into a model.

    compartment(gene)  from UniProt SUBCELLULAR LOCATION
    compartment(P)     from where P's reactions actually happen, per compartments.json
    agreement          do they overlap at all?
    control            the same gene against SIZE-MATCHED random pathways, because a placement into a pathway that
                       spans six compartments agrees with everything and proves nothing

A UniProt location string is not one location. "Golgi apparatus, Golgi stack membrane; Single-pass type II membrane
protein. Note=Also found in nucleus" names three. All of them are extracted, because taking the first would score a
multi-located protein as disagreeing whenever the curator happened to list a secondary site first.

WHAT A FAILURE HERE WOULD MEAN. Low agreement would not prove the placements wrong -- many pathways span
compartments, and a peripheral regulator need not sit inside the compartment it regulates. But agreement no better
than the size-matched control would mean location is adding no support, and the placements would rest entirely on
the co-essentiality evidence measured separately.
"""
import collections
import json
from pathlib import Path

import numpy as np

from compartmentalize import canonical

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GMT = SP / "ReactomePathways.gmt"
NRAND = 40


def parse_locations(s, unmapped):
    """Every canonical compartment named anywhere in a UniProt SUBCELLULAR LOCATION string.

    Taking only the first named site would misscore multi-located proteins, which are common: a protein listed as
    "Golgi ...; Note=Also found in nucleus" genuinely occupies both, and scoring it against the Golgi alone would
    count a correct nuclear placement as a contradiction.
    """
    if not s:
        return set()
    body = s.replace("SUBCELLULAR LOCATION:", " ")
    out = set()
    for frag in body.replace(";", ".").replace(",", ".").split("."):
        f = frag.strip()
        if not f or f.lower().startswith(("note=", "isoform")):
            continue
        c = canonical(f, unmapped)
        if c:
            out.add(c)
    return out


def main():
    tbl = json.loads((OUT / "dark_function_table.json").read_text())["table"]
    patch = json.loads((OUT / "ghost_patch.json").read_text())["patch"]
    comp = json.loads((OUT / "compartments.json").read_text())
    net = json.loads((OUT / "reaction_network.json").read_text())
    paths = {}
    for line in GMT.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[0]] = set(p[2:])

    unmapped = collections.Counter()
    loc = {g: parse_locations(v.get("location"), unmapped) for g, v in tbl.items()}
    loc = {g: c for g, c in loc.items() if c}
    print(f"dark genes: {len(tbl):,}; with a LOCATION string: "
          f"{sum(1 for v in tbl.values() if v.get('location')):,}; mapped to a canonical compartment: {len(loc):,}")
    dist = collections.Counter(c for cs in loc.values() for c in cs)
    print(f"  where the dark proteome lives: {dict(dist.most_common(10))}")
    print(f"  genes with >1 location: {sum(1 for c in loc.values() if len(c) > 1):,}")

    # where each pathway actually happens, from the located reaction network
    step_comp = comp["steps"]
    gene_steps = collections.defaultdict(set)
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        for g in gs:
            gene_steps[g].add(s["id"])
    pw_comp = {}
    for name, genes in paths.items():
        cs = collections.Counter()
        for g in genes:
            for sid in gene_steps.get(g, ()):
                for c in step_comp.get(sid, {}).get("compartments", []):
                    cs[c] += 1
        if cs:
            pw_comp[name] = cs
    print(f"  pathways with a resolvable location: {len(pw_comp):,}/{len(paths):,}")

    # main + size-matched control
    sized = sorted(pw_comp, key=lambda n: len(paths[n]))
    sizes = np.array([len(paths[n]) for n in sized])
    rows = []
    for gene, v in patch.items():
        preds = v.get("pathways_predicted")
        if not preds or gene not in loc:
            continue
        pw = preds[0]["pathway"]
        if pw not in pw_comp:
            continue
        top = {c for c, _ in pw_comp[pw].most_common(3)}
        hit = bool(loc[gene] & top)
        want = len(paths[pw])
        lo = max(0, int(np.searchsorted(sizes, want)) - NRAND // 2)
        pool = [n for n in sized[lo:lo + NRAND] if n != pw]
        if len(pool) < 5:
            continue
        ctrl = np.mean([bool(loc[gene] & {c for c, _ in pw_comp[n].most_common(3)}) for n in pool])
        rows.append({"gene": gene, "pathway": pw, "gene_loc": sorted(loc[gene]),
                     "pathway_loc": sorted(top), "agree": hit, "ctrl": float(ctrl),
                     "confidence": preds[0].get("confidence")})

    if not rows:
        raise SystemExit("no dark gene had both a mapped location and a locatable predicted pathway")
    ag = np.array([r["agree"] for r in rows], float)
    ct = np.array([r["ctrl"] for r in rows])
    d = ag - ct
    se = float(d.std() / np.sqrt(len(d)))
    print(f"\n  LOCATION CONSISTENCY of the pathway placements  (n={len(rows):,})")
    print(f"    gene location overlaps its predicted pathway's top-3 compartments : {ag.mean():.3f}")
    print(f"    same, against SIZE-MATCHED random pathways                        : {ct.mean():.3f}")
    print(f"    delta                                                             : {d.mean():+.4f} +/- {se:.4f}")

    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["pathway"]].append(r["agree"] - r["ctrl"])
    print(f"\n  by target pathway")
    for pw, ds in sorted(byp.items(), key=lambda kv: -len(kv[1]))[:8]:
        a = np.array(ds)
        print(f"    {pw[:42]:<44} n={len(ds):<5,} delta {a.mean():+.4f}")

    contra = [r for r in rows if not r["agree"] and r["ctrl"] < 0.2]
    print(f"\n  placements CONTRADICTED by location ({len(contra):,}): gene sits nowhere near the pathway, and the "
          f"control says that is not merely a promiscuous pathway")
    for r in contra[:6]:
        print(f"    {r['gene']:<10} at {','.join(r['gene_loc']):<28} placed in {r['pathway'][:34]:<36} "
              f"which runs in {','.join(r['pathway_loc'])}")

    if unmapped:
        print(f"\n  unmapped location fragments: {len(unmapped):,} distinct; "
              f"top {[s for s, _ in unmapped.most_common(6)]}")

    json.dump({"n_dark": len(tbl), "n_with_location": len(loc),
               "location_distribution": dict(dist), "n_tested": len(rows),
               "agreement": float(ag.mean()), "control": float(ct.mean()),
               "delta": float(d.mean()), "se": se,
               "by_pathway": {k: float(np.mean(v)) for k, v in byp.items()},
               "contradicted": contra, "rows": rows},
              open(OUT / "dark_gene_location.json", "w"))
    print(f"\n  -> {OUT/'dark_gene_location.json'}")


if __name__ == "__main__":
    main()
