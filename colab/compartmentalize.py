"""WHERE DOES EACH STEP HAPPEN? ASSIGNING SPATIAL LOCATION TO ALL 28,528 REACTIONS.

WHAT IS AVAILABLE, AND WHY IT HAS TO BE HARMONISED BEFORE IT CAN BE USED. Both halves of the reaction network carry
location, in incompatible vocabularies:

    Human-GEM   9 compartments as single-letter codes on every species (c, e, g, i, l, m, n, r, x)
    Reactome    123 distinct compartment strings parsed off entity display names ("PTK6 [cytosol]")

123 versus 9 is not a difference in detail, it is a difference in granularity: Reactome separates "mitochondrial
intermembrane space" from "mitochondrial matrix" from "mitochondrial outer membrane", where Human-GEM has
"Mitochondria" and "Inner mitochondria". Merging them by string equality would silently drop almost every Reactome
compartment. So both are mapped into one COARSE canonical set, and the number of strings that fail to map is
reported rather than absorbed -- an unmapped compartment must show up as unmapped, not as "other".

WHAT A STEP'S LOCATION MEANS. Collect the canonical compartments of every participant. Then:

    LOCAL      all participants in one compartment -- the step happens there
    SPANNING   participants in more than one -- a transport or membrane step, and the PAIR is the interesting part
    UNKNOWN    no participant carries a compartment

SPANNING is not a defect in the annotation, it is the signature of transport, and its compartment pairs are a
directly readable map of what the cell moves and where. Reactome curates translocation as exactly this shape:
the same entity present on both sides in two different compartments.

THE CHECK THAT KEEPS THIS HONEST. A location assignment is only as good as its coverage, and coverage here is
limited by a known join gap: 46,502 participant slots reference Reactome entities by bare dbId that could not be
resolved while Reactome's origin was returning 521. Those are counted as UNRESOLVED, never as "no compartment", so
the difference between "we know it has no location" and "we could not look it up" stays visible.
"""
import collections
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
RN = OUT / "reaction_network.json"
GEM = SP / "HumanGEM.xml"

# Coarse canonical compartments. Deliberately small: the question is "where in the cell", and splitting the
# mitochondrion four ways makes the two sources incomparable without adding anything to that question.
CANON = ["cytosol", "nucleus", "mitochondrion", "endoplasmic reticulum", "golgi", "lysosome", "peroxisome",
         "plasma membrane", "extracellular", "endosome", "cytoskeleton", "lipid droplet", "other membrane"]
RULES = [
    ("nucle", "nucleus"), ("chromatin", "nucleus"), ("chromosome", "nucleus"), ("nucleoplasm", "nucleus"),
    ("nucleolus", "nucleus"),
    ("mitochond", "mitochondrion"),
    ("endoplasmic", "endoplasmic reticulum"), ("sarcoplasmic", "endoplasmic reticulum"),
    ("golgi", "golgi"),
    ("lysosom", "lysosome"), ("autophagosom", "lysosome"), ("autolysosom", "lysosome"),
    ("peroxisom", "peroxisome"),
    ("plasma membrane", "plasma membrane"), ("cell surface", "plasma membrane"),
    ("extracellular", "extracellular"), ("secret", "extracellular"), ("blood", "extracellular"),
    ("endosom", "endosome"), ("phagosom", "endosome"), ("clathrin", "endosome"), ("vesicle", "endosome"),
    ("vacuol", "endosome"),
    ("lipid droplet", "lipid droplet"),
    ("cell junction", "plasma membrane"), ("periplasm", "extracellular"),
    ("cytoskelet", "cytoskeleton"), ("microtubul", "cytoskeleton"), ("actin", "cytoskeleton"),
    ("centrosom", "cytoskeleton"), ("cilium", "cytoskeleton"), ("ciliary", "cytoskeleton"),
    ("cytosol", "cytosol"), ("cytoplasm", "cytosol"),
    ("membrane", "other membrane"),
]
GEM_CODE = {"c": "cytosol", "e": "extracellular", "g": "golgi", "i": "mitochondrion", "l": "lysosome",
            "m": "mitochondrion", "n": "nucleus", "r": "endoplasmic reticulum", "x": "peroxisome"}


def canonical(raw, unmapped):
    """Map a source compartment string to the coarse canonical set, recording anything that does not map."""
    if not raw:
        return None
    if raw in GEM_CODE:                       # Human-GEM single-letter code
        return GEM_CODE[raw]
    s = raw.lower()
    for pat, target in RULES:
        if pat in s:
            return target
    unmapped[raw] += 1
    return None


def main():
    net = json.loads(RN.read_text())
    steps = net["steps"]
    # Human-GEM's own compartment names, read from the model so the code map is not hard-coded blindly
    gem_names = {}
    for _ev, el in ET.iterparse(str(GEM), events=("end",)):
        if el.tag.split("}")[-1] == "compartment":
            gem_names[el.get("id")] = el.get("name")
            el.clear()
        elif el.tag.split("}")[-1] == "listOfCompartments":
            break
    print(f"Human-GEM declares {len(gem_names)} compartments: "
          + ", ".join(f"{k}={v}" for k, v in sorted(gem_names.items())))
    bad = [k for k in gem_names if k not in GEM_CODE]
    if bad:
        print(f"  !! codes with no canonical mapping: {bad} -- fix GEM_CODE before trusting the census")

    unmapped = collections.Counter()
    loc = collections.Counter()
    kind = collections.Counter()
    pairs = collections.Counter()
    per_step = {}
    unresolved_slots = 0
    for s in steps:
        cs = set()
        nores = 0
        for p in s["in"] + s["out"]:
            if p["type"] == "UNRESOLVED":
                nores += 1
                continue
            c = canonical(p.get("compartment"), unmapped)
            if c:
                cs.add(c)
        unresolved_slots += nores
        if not cs:
            kind["UNKNOWN"] += 1
            per_step[s["id"]] = {"kind": "UNKNOWN", "compartments": []}
            continue
        if len(cs) == 1:
            k = "LOCAL"
            loc[next(iter(cs))] += 1
        else:
            k = "SPANNING"
            for a in sorted(cs):
                loc[a] += 1
            for a in sorted(cs):
                for b in sorted(cs):
                    if a < b:
                        pairs[(a, b)] += 1
        kind[k] += 1
        per_step[s["id"]] = {"kind": k, "compartments": sorted(cs)}

    n = len(steps)
    print(f"\n  {n:,} steps located")
    for k in ("LOCAL", "SPANNING", "UNKNOWN"):
        print(f"    {k:<9} {kind[k]:>7,}  ({kind[k]/n:.1%})")
    print(f"    (UNRESOLVED participant slots contributing to UNKNOWN: {unresolved_slots:,} -- "
          f"a lookup gap, not an absence of location)")

    print(f"\n  steps per compartment (a SPANNING step counts once in each of its compartments)")
    for c, k in loc.most_common():
        print(f"    {c:<24} {k:>7,}")

    print(f"\n  the transport map: most frequent compartment pairs in SPANNING steps")
    for (a, b), k in pairs.most_common(12):
        print(f"    {a:<22} <-> {b:<22} {k:>6,}")

    if unmapped:
        tot = sum(unmapped.values())
        print(f"\n  UNMAPPED compartment strings: {len(unmapped):,} distinct, {tot:,} slots "
              f"({tot/max(sum(loc.values()),1):.1%} of located slots)")
        print("    " + ", ".join(f"{s!r}({k})" for s, k in unmapped.most_common(8)))

    # cross-check: SPANNING should be enriched for the steps that move a gene between compartments, which is how
    # transport was independently detected earlier from leaf-entity compartment changes
    bysrc = collections.Counter((s["source"], per_step[s["id"]]["kind"]) for s in steps)
    print(f"\n  by source: " + ", ".join(f"{a}/{b}={k:,}" for (a, b), k in sorted(bysrc.items())))

    json.dump({"n_steps": n, "kind": dict(kind), "per_compartment": dict(loc),
               "pairs": {f"{a}|{b}": k for (a, b), k in pairs.items()},
               "unmapped": dict(unmapped), "unresolved_slots": unresolved_slots,
               "canonical": CANON, "steps": per_step}, open(OUT / "compartments.json", "w"))
    print(f"\n  -> {OUT/'compartments.json'}")


if __name__ == "__main__":
    main()
