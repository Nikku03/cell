"""Resolve Human-GEM metabolites to molecular structures. A cache, not a test -- no gates.

WHY THIS IS NEEDED AT ALL. Human-GEM stores elemental FORMULAS, because that is what a
stoichiometric model consumes. Formulas cannot separate isomers, and loop 169 measured that isomer
ambiguity is where the ranker's remaining errors live: of the single-new-product errors, 85.5% have
a median of 4 chemically DISTINCT molecules all balancing the residual exactly. C18H22O3 is C18H22O3
whether the oxygens sit on a ring or a side chain, so no amount of formula arithmetic can choose.

THREE SOURCES, IN PRIORITY ORDER, because they disagree and the most specific should win:
  1. InChI carried in the Human-GEM SBML itself      -- authoritative for this model, 12.6%
  2. PubChem, by the CID the SBML annotates          -- 1,390 distinct CIDs
  3. MetaNetX chem_prop SMILES                       -- 1,082 rows recovered by streaming filter

Union coverage measured before fetching: 66.8% of all 8,461 metabolites, and on the isomer-ambiguity
cases specifically, 30.1% have EVERY competitor resolvable against 2.7% from InChI alone.

WHAT IS NOT DONE HERE, and it matters for reading anything downstream: coverage is not random.
Common, well-studied metabolites are annotated; unusual ones are not. In the worked example of
loop 169, MAR02104, the molecules with no identifier at all were precisely the estradiol quinones --
including the right answer. So structure features will be present exactly where the problem is
easiest, and that bias is a property of the annotation, not of the method.

-> colab/data/ml/metabolite_structures.json
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = Path("colab/data/ml/metabolite_structures.json")
BATCH, PAUSE = 100, 0.25


def main():
    ids = json.load(open(SP / "met_ids.json"))
    struct = {}

    # 1. InChI straight from the SBML
    n_inchi = 0
    for sp, d in ids.items():
        if "inchi" in d:
            struct[sp] = {"src": "sbml_inchi", "inchi": d["inchi"]}
            n_inchi += 1
    print(f"  SBML InChI      : {n_inchi:,}", flush=True)

    # 2. MetaNetX SMILES
    mnx = {}
    f = SP / "mnx_hits.tsv"
    if f.exists():
        for ln in open(f):
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) > 8 and p[8].strip():
                mnx[p[0]] = p[8].strip()
    n_mnx = 0
    for sp, d in ids.items():
        if sp in struct:
            continue
        m = d.get("metanetx.chemical")
        if m and m in mnx:
            struct[sp] = {"src": "metanetx", "smiles": mnx[m]}
            n_mnx += 1
    print(f"  MetaNetX SMILES : {n_mnx:,}", flush=True)

    # 3. PubChem, batched
    need = sorted({d["pubchem.compound"] for sp, d in ids.items()
                   if sp not in struct and "pubchem.compound" in d})
    print(f"  PubChem CIDs to fetch: {len(need):,} in {-(-len(need)//BATCH)} batches", flush=True)
    got = {}
    for i in range(0, len(need), BATCH):
        chunk = need[i:i + BATCH]
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               + ",".join(chunk) + "/property/SMILES/CSV")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    txt = r.read().decode()
                for ln in txt.splitlines()[1:]:
                    parts = ln.split(",", 1)
                    if len(parts) == 2:
                        got[parts[0].strip('"')] = parts[1].strip().strip('"')
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    batch {i//BATCH} failed: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        time.sleep(PAUSE)
        if (i // BATCH) % 5 == 0:
            print(f"    {i+len(chunk)}/{len(need)} cids, {len(got):,} smiles", flush=True)
    n_pc = 0
    for sp, d in ids.items():
        if sp in struct:
            continue
        c = d.get("pubchem.compound")
        if c and c in got and got[c]:
            struct[sp] = {"src": "pubchem", "smiles": got[c]}
            n_pc += 1
    print(f"  PubChem SMILES  : {n_pc:,}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(struct, open(OUT, "w"))
    print(f"  TOTAL {len(struct):,} of {len(ids):,} metabolites ({len(struct)/len(ids):.1%}) "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
