"""PER-KNOCKOUT OUTPUT SPACE: predict a knockout's movers from the MEASURED profiles of its network neighbours.

THE DIAGNOSIS THIS IS BUILT FROM. output_ceiling.py measured why the multi-layer reasoner fails, and the answer was
not its reasoning. It emits from a fixed menu of nine sensor programmes, 89 genes, against an answer space of
1,120-2,007 distinct genes of which 50-57% move for exactly ONE knockout. Firing every sensor perfectly caps at
4.4% recall. No improvement to the firing logic can exceed that, because the ceiling is the vocabulary.

So the fix has to change what the model is ALLOWED TO SAY, not how it decides. Every knockout needs its own output
distribution over all 8,246 measured genes, not a selection from a shared menu.

THE MECHANISM, AND WHY IT IS NOT CHEATING. For a knockout k, look up k's neighbours in the network -- proteins it
binds, complexes it belongs to, reactions it participates in -- and transfer THEIR measured response profiles onto
k. Genes that move when k's neighbours are removed are the prediction for k.

    output space   all 8,246 genes, weighted per knockout          (vs 89 shared)
    specificity    every knockout has different neighbours, so every prediction differs
    no peeking     k's OWN profile is never read. Neighbours are restricted to the training pool, which excludes
                   BOTH sealed cohorts entirely -- not just k -- so a cohort member can never be a donor for
                   another cohort member.

This is the same construction the wall_* modules call PHYS, where it scored 0.421 recall@50 against a 0.241 tide
null. It has never been run through the sealed knockout-prediction protocol, which is what this file does.

WHAT WOULD MAKE IT A FAILURE. It has to beat the frequency baseline -- the "usual suspects" predictor that uses no
biology and that every mechanistic method in this project has lost to. And it has to beat its own shuffled control,
where a knockout is given a DIFFERENT knockout's neighbours: if that matches, the gain is from the shape of the
output rather than from whose neighbours were used, which is exactly the failure the multi-layer model turned out
to have.
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TAU = 1.0
NPRED = int(os.environ.get("NT_NPRED", "20"))     # comparable to the other methods' median prediction size
MAXNBR = 25                                       # donors per knockout, by graph degree order


def build_graph(names):
    """PPI + complex + pathway + reaction co-participation, keyed by gene NAME.

    The index-space defect fixed elsewhere in this project applies here too: cell_complete's ppi endpoints are
    INTEGER INDICES into genes[], so they are mapped through names[] before use rather than keyed raw.
    """
    D = json.load(open(OUT / "cell_complete.json"))
    nb = collections.defaultdict(set)
    allnames = [g["name"] for g in D["genes"]]
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < len(allnames) and 0 <= b < len(allnames) and a != b:
            nb[allnames[a]].add(allnames[b])
            nb[allnames[b]].add(allnames[a])
    grp = collections.defaultdict(set)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        gn = allnames[int(g)] if g.isdigit() and int(g) < len(allnames) else g
        for c in (cs if isinstance(cs, (list, set, tuple)) else [cs]) if cs else []:
            grp[("c", c)].add(gn)
    for g in D["genes"]:
        if g.get("path"):
            grp[("p", g["path"])].add(g["name"])
    for k, mem in grp.items():
        if 2 <= len(mem) <= 200:
            ml = sorted(mem)
            for a in ml:
                nb[a].update(x for x in ml if x != a)
    net = json.load(open(OUT / "reaction_network.json"))
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if 2 <= len(gs) <= 40:
            gl = sorted(gs)
            for a in gl:
                nb[a].update(x for x in gl if x != a)
    return nb


def main():
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    M = np.abs(z["M"])
    ki = {k: i for i, k in enumerate(kos)}
    tide = (M >= TAU).mean(0) >= 0.05

    # LEAKAGE CONTROL: both sealed cohorts are removed from the donor pool wholesale. Excluding only the knockout
    # being predicted would still let cohort members donate to each other, and they are scored against the same
    # answers.
    sealed = set()
    for t in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{t}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    pool = [k for k in kos if k not in sealed]
    print(f"readout {len(kos):,} perturbations x {len(genes):,} genes; tide {int(tide.sum())}")
    print(f"donor pool {len(pool):,} (both sealed cohorts of {len(sealed)} removed wholesale)")

    nb = build_graph(genes)
    poolset = set(pool)
    rng = np.random.default_rng(0)

    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        doss = json.loads(dp.read_text())
        for variant in ("nbr", "nbrshuf"):
            preds, ndon = {}, []
            shufkeys = sorted(doss)
            perm = rng.permutation(len(shufkeys))
            for j, k in enumerate(sorted(doss)):
                src = k if variant == "nbr" else shufkeys[perm[j]]
                donors = sorted(nb.get(src, set()) & poolset)[:MAXNBR]
                ndon.append(len(donors))
                if not donors:
                    preds[k] = {"variant": variant, "n_donors": 0, "predicted_genes": [],
                                "reasoning": "no network neighbour with a measured profile"}
                    continue
                idx = [ki[d] for d in donors if d in ki]
                if not idx:
                    preds[k] = {"variant": variant, "n_donors": 0, "predicted_genes": []}
                    continue
                # vote: how many donors move each gene past threshold, tide genes excluded to match the scorer
                votes = (M[idx] >= TAU).sum(0).astype(np.float32)
                votes[tide] = 0
                if k in genes:
                    votes[genes.index(k)] = 0
                order = np.argsort(-votes)
                top = [genes[i] for i in order[:NPRED] if votes[i] > 0]
                preds[k] = {"variant": variant, "n_donors": len(idx),
                            "reasoning": f"profiles of {len(idx)} network neighbours in the donor pool, "
                                         f"gene ranked by how many of them move it",
                            "predicted_genes": top}
            p = OUT / f"ko_predictions_{variant}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            sz = sorted(len(v["predicted_genes"]) for v in preds.values())
            print(f"  {tag or 'cohort1':<9} {variant:<8} donors median {int(np.median(ndon))}; "
                  f"prediction median {sz[len(sz)//2]}, empty {sum(1 for s in sz if s == 0)}/{len(sz)}"
                  f"  -> {p.name}")


if __name__ == "__main__":
    main()
