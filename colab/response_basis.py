"""SOLVING THE 89-GENE PROBLEM: learn the response programmes from data instead of hand-writing them.

THE PROBLEM, MEASURED. The multi-layer reasoner emits genes from a fixed menu -- nine hand-written stress
programmes, 89 genes total -- against an answer space of 1,120-2,007 distinct genes of which 50-57% move for
exactly ONE knockout. Firing every sensor perfectly caps at 4.4% recall. Only 34 of the 89 ever move at all, and
the best possible 89-gene list chosen with the answers in hand reaches 39.8%, so the menu is roughly 9x worse than
a menu of that size could be. It is small AND badly chosen.

Neighbour transfer got around this by giving each knockout its own output distribution, and it beat the frequency
baseline -- but it works by BORROWING the measured profiles of a gene's network neighbours. It explains nothing and
it needs its neighbours to have been measured. The mechanistic path still had no way to say anything
knockout-specific.

THE FIX: DERIVE THE PROGRAMMES, DO NOT WRITE THEM. Factorise the measured response matrix into k components. Each
component is a weight over all 8,246 genes -- a programme the cell actually uses, rather than one a human guessed.
Then learn which components each LESION CLASS drives, from training knockouts only. A held-out knockout is
predicted by: its annotation -> its lesion classes -> the component mixture those classes drive -> a ranked list
over all 8,246 genes.

    vocabulary     89 hand-picked genes            ->  k components x 8,246 genes
    source         a human's stress-biology guess  ->  factorisation of 4,720 measured perturbations
    specificity    a shared menu                   ->  a per-knockout mixture
    mechanism      retained: the route is annotation -> lesion class -> programme, never the gene's own profile

WHAT MAKES THIS NOT CHEATING, AND THE TWO WAYS IT COULD BE.

  LEAKAGE 1: the components must come from the TRAINING POOL ONLY. Both sealed cohorts (400 knockouts) are removed
  wholesale before factorising -- not just the knockout being predicted -- because a cohort member contributing to
  a component would leak into every other cohort member's prediction.

  LEAKAGE 2: the lesion-class -> component map must also be fitted on training knockouts only. It is, and the
  held-out knockout's own row is never read at any point.

THE CONTROLS THAT DECIDE IT. Against the frequency baseline, which no mechanistic method here has ever beaten. And
against a SHUFFLED variant where lesion classes are permuted across knockouts: if that matches, the components are
carrying the result and the lesion classes are decorative -- which is exactly how the hand-written sensor menu
failed.
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
K = int(os.environ.get("RB_K", "60"))          # components
NPRED = int(os.environ.get("RB_NPRED", "20"))  # genes predicted per knockout, comparable to the other methods
TAU = 1.0


def lesion_classes():
    """Each gene's lesion classes, from ANNOTATION ONLY -- the same classifier the sensor model used.

    Reused deliberately: the point of this file is to change the OUTPUT VOCABULARY while holding the input
    reasoning fixed, so that any difference is attributable to the vocabulary rather than to a better classifier.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from multilayer_reason import classify
    cmap = json.loads((OUT / "consequence_map.json").read_text())
    paths = collections.defaultdict(list)
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]:
                paths[g].append(p[0])
    dark = {}
    try:
        for g, v in json.loads((OUT / "dark_function_table.json").read_text())["table"].items():
            dark[g] = v.get("function")
    except Exception:
        pass
    out = {}
    for g in set(list(paths) + list(cmap)):
        m = cmap.get(g, {})
        pw = paths.get(g, []) + (m.get("pathways_member") or [])
        out[g] = classify(g, pw, m.get("compartments") or [], dark.get(g))
    return out


def main():
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    M = z["M"]
    ki = {k: i for i, k in enumerate(kos)}
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= 0.05

    sealed = set()
    for t in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{t}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
    train = [k for k in kos if k not in sealed]
    tidx = [ki[k] for k in train]
    print(f"readout {len(kos):,} x {len(genes):,}; training pool {len(train):,} "
          f"(both sealed cohorts of {len(sealed)} removed BEFORE factorising)")

    # ---------------- the learned vocabulary ----------------
    from sklearn.decomposition import NMF
    X = A[tidx].copy()
    X[:, tide] = 0.0                      # tide genes are excluded from the truth set, so exclude them here too
    print(f"factorising {X.shape[0]:,} x {X.shape[1]:,} into {K} components ...")
    nmf = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(X)              # knockout x component
    H = nmf.components_                   # component x gene
    print(f"  reconstruction error {nmf.reconstruction_err_:.1f}; "
          f"component gene support (weight>0.01 of max): "
          f"{int(np.median((H > 0.01 * H.max(1, keepdims=True)).sum(1))):,} genes median")

    # ---------------- lesion class -> component mixture, fitted on TRAINING knockouts only ----------------
    lc = lesion_classes()
    cls_of = {k: sorted(lc.get(k, set())) for k in kos}
    allcls = sorted({c for v in cls_of.values() for c in v})
    prof = {}
    for c in allcls:
        rows = [i for i, k in enumerate(train) if c in cls_of.get(k, ())]
        if len(rows) >= 5:
            prof[c] = W[rows].mean(0)
    globalmix = W.mean(0)
    print(f"  lesion classes with >=5 training knockouts: {len(prof)}/{len(allcls)}  {sorted(prof)}")

    rng = np.random.default_rng(0)
    for tag in ("", "_holdout"):
        dp = OUT / f"ko_pred_dossiers{tag}.json"
        if not dp.exists():
            continue
        doss = sorted(json.loads(dp.read_text()))
        for variant in ("basis", "basisshuf"):
            perm = rng.permutation(len(doss))
            preds = {}
            for j, k in enumerate(doss):
                src = k if variant == "basis" else doss[perm[j]]
                cs = [c for c in cls_of.get(src, ()) if c in prof]
                mix = np.mean([prof[c] for c in cs], 0) if cs else globalmix
                score = mix @ H                       # a weight over ALL genes, specific to this knockout
                score[tide] = 0.0
                if k in genes:
                    score[genes.index(k)] = 0.0
                order = np.argsort(-score)[:NPRED]
                preds[k] = {"variant": variant, "lesion_classes": cs,
                            "reasoning": f"annotation -> lesion classes {cs or '[none, global mixture]'} -> "
                                         f"mixture of {K} data-derived response components -> ranked over "
                                         f"{len(genes):,} genes",
                            "predicted_genes": [genes[i] for i in order if score[i] > 0]}
            p = OUT / f"ko_predictions_{variant}{tag}.json"
            json.dump(preds, open(p, "w"), indent=1)
            sz = sorted(len(v["predicted_genes"]) for v in preds.values())
            nc = sorted(len(v["lesion_classes"]) for v in preds.values())
            print(f"  {tag or 'cohort1':<9} {variant:<10} median {sz[len(sz)//2]} genes predicted, "
                  f"empty {sum(1 for s in sz if s == 0)}/{len(sz)}; "
                  f"lesion classes median {nc[len(nc)//2]}  -> {p.name}")

    json.dump({"k": K, "n_train": len(train), "classes_fitted": sorted(prof),
               "vocabulary_genes": int((H.sum(0) > 0).sum())},
              open(OUT / "response_basis.json", "w"))
    print(f"\n  vocabulary now spans {int((H.sum(0) > 0).sum()):,} genes (was 89)")
    print(f"  -> {OUT/'response_basis.json'}")


if __name__ == "__main__":
    main()
