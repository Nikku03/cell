"""MARKOV SPREAD x MULTI-LAYER REASONING: let the walk say HOW MUCH each protein is hit, then reason over each one.

THE PROBLEM THIS SOLVES, WHICH THE PREVIOUS ROUND DIAGNOSED PRECISELY. cascade_layers.py added six damage modes and
scored WORSE than the one-step model it extended (0.0474 vs 0.0786). The reason was not the layers -- it was that
every layer voted with equal weight. `proteotoxic` fired on 28 of 50 knockouts because almost every protein is in
some complex, and a programme that fires on more than half the knockouts cannot discriminate between them. What was
missing is a quantity: HOW MUCH of the cell is actually hit in each way.

A random walk from the knockout node supplies exactly that quantity. It is the one thing the walk is genuinely good
at -- earlier in this project it was shown that a walk CANNOT infer edge direction (its "opinion" is identically the
degree ratio) and that feeding measured direction into it does not help. But nobody claimed it cannot measure
spread, and spread is what a firing threshold needs.

    walk from the knockout   ->  mass m(p) on every protein p
    each protein has LESION CLASSES from its own annotation (mitochondrial, ER, translation, ...)
    sensor score S(c) = sum of m(p) over proteins p in class c
    a sensor fires only when S(c) clears a threshold, so a class that is merely COMMON does not fire --
    it must be where the knockout's mass actually landed

So the walk is used as a weighting function over the reasoning, not as a predictor. Each affected protein still gets
reasoned about individually: its roles, its lesion classes, and -- if it is a transcription factor -- its regulon,
weighted by how hard the walk hit it.

THREE CONTROLS, because a weighting scheme is easy to fool:
    UNIFORM   every neighbour within the same radius weighted equally. If this matches, the MASS is doing nothing
              and only the identity of the neighbourhood matters.
    SHUFFLED  the mass vector permuted across proteins. If this matches, the weighting is not tracking biology.
    NOWALK    the one-step model, i.e. only the knockout's own lesion classes -- the incumbent at 0.0786.
"""
import collections
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilayer_reason import SENSORS, classify

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NSTEP, DAMP = 3, 0.5
TOPMASS = 300         # proteins retained per knockout; the tail is numerically irrelevant and slows nothing down
FIRE_Q = 0.75         # a sensor fires when its mass share is in the top quartile ACROSS knockouts -- a relative
                      # threshold, so "fires for everything" is impossible by construction


def main():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}
    N = len(names)

    # ---------- graph: PPI + reaction co-participation (the pair that scored best in chain_test) ----------
    E = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            E.add((a, b))
            E.add((b, a))
    net = json.load(open(OUT / "reaction_network.json"))
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        ids = [nidx[g] for g in gs if g in nidx]
        if 2 <= len(ids) <= 60:
            for a in ids:
                for b in ids:
                    if a != b:
                        E.add((a, b))
    a = np.fromiter((x for x, _ in E), int, len(E))
    b = np.fromiter((y for _, y in E), int, len(E))
    G = sparse.coo_matrix((np.ones(len(a), np.float32), (a, b)), shape=(N, N)).tocsr()
    deg = np.asarray(G.sum(1)).ravel()
    inv = np.zeros(N, np.float32)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    PT = (sparse.diags(inv) @ G).T.tocsr()
    print(f"graph {len(E):,} arcs over {N:,} nodes")

    # ---------- per-gene lesion classes, computed ONCE for the whole network ----------
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
    unmapped = collections.Counter()
    gene_lesions = {}
    for g in names:
        m = cmap.get(g, {})
        pw = paths.get(g, []) + (m.get("pathways_member") or [])
        gene_lesions[g] = classify(g, pw, m.get("compartments") or [], dark.get(g))
    cls_count = collections.Counter(c for v in gene_lesions.values() for c in v)
    print(f"lesion classes over the whole network: {dict(cls_count.most_common())}")
    print(f"  genes with no class: {sum(1 for v in gene_lesions.values() if not v):,}/{N:,}")

    # signed regulons, for the TF arm
    tf_reg = collections.defaultdict(set)
    for e in D.get("reg", []):
        try:
            s_, t_, w = int(e[0]), int(e[1]), (int(e[2]) if len(e) > 2 else 0)
        except (TypeError, ValueError, IndexError):
            continue
        if w and 0 <= s_ < N and 0 <= t_ < N and s_ != t_:
            tf_reg[names[s_]].add(names[t_])

    doss = json.loads((OUT / "ko_pred_dossiers.json").read_text())
    seeds = [k for k in doss if k in nidx]
    print(f"{len(seeds)}/{len(doss)} knockouts are in the network")

    # ---------- walk ----------
    L = np.zeros((N, len(seeds)), np.float32)
    for j, k in enumerate(seeds):
        L[nidx[k], j] = 1.0
    cur, acc = L, np.zeros_like(L)
    for s_ in range(NSTEP):
        cur = PT @ cur
        acc += (DAMP ** (s_ + 1)) * cur

    rng = np.random.default_rng(0)
    variants = {}
    for name in ("WALK", "UNIFORM", "SHUFFLED"):
        variants[name] = {}
        for j, k in enumerate(seeds):
            m = acc[:, j].copy()
            idx = np.argsort(-m)[:TOPMASS]
            idx = idx[m[idx] > 0]
            if name == "UNIFORM":
                w = np.ones(len(idx), np.float32)
            elif name == "SHUFFLED":
                w = m[idx][rng.permutation(len(idx))]
            else:
                w = m[idx]
            variants[name][k] = (idx, w)

    # ---------- sensor scores: mass landing on each lesion class ----------
    def sensor_scores(idx, w):
        sc = collections.Counter()
        tot = float(w.sum()) or 1.0
        for i, wi in zip(idx.tolist(), w.tolist()):
            for c in gene_lesions[names[i]]:
                sc[c] += wi
        return {c: v / tot for c, v in sc.items()}

    preds_by_variant = {}
    for vname, V in variants.items():
        allsc = {k: sensor_scores(*V[k]) for k in seeds}
        # RELATIVE threshold per class: a sensor fires for a knockout only if that knockout is in the top quartile
        # for that class. A class present everywhere therefore cannot fire everywhere.
        cut = {}
        for c in SENSORS:
            vals = [allsc[k].get(c, 0.0) for k in seeds]
            cut[c] = float(np.quantile(vals, FIRE_Q)) if any(vals) else 1e9
        preds = {}
        for k in seeds:
            fired = sorted(c for c, v in allsc[k].items() if v > max(cut[c], 1e-9))
            idx, w = V[k]
            genes, seen = [], {k}

            def add(xs):
                for x in xs:
                    if x and x not in seen and "; " not in x:
                        seen.add(x)
                        genes.append(x)

            add(sorted(tf_reg.get(k, ())))                       # the knockout's own regulon
            n_own = len(genes)
            # TFs the walk hit hardest contribute their regulons, in mass order
            tfhit = [(names[i], wi) for i, wi in zip(idx.tolist(), w.tolist())
                     if names[i] != k and names[i] in tf_reg]
            for g, _ in sorted(tfhit, key=lambda t: -t[1])[:3]:
                add(sorted(tf_reg[g])[:10])
            n_tf = len(genes) - n_own
            for c in fired:
                add(SENSORS[c]["programme"])
            preds[k] = {"variant": vname, "sensors_fired": fired,
                        "sensor_scores": {c: round(v, 4) for c, v in sorted(allsc[k].items(),
                                                                            key=lambda t: -t[1])[:5]},
                        "n_own_regulon": n_own, "n_walk_tf_regulon": n_tf,
                        "reasoning": f"walk mass over {len(idx)} proteins; sensors {fired}",
                        "predicted_genes": genes[:60]}
        preds_by_variant[vname] = preds
        out = OUT / f"ko_predictions_mmw_{vname.lower()}.json"
        json.dump(preds, open(out, "w"), indent=1)
        nf = collections.Counter(c for p in preds.values() for c in p["sensors_fired"])
        sz = sorted(len(p["predicted_genes"]) for p in preds.values())
        print(f"  {vname:<9} sensors fired {dict(nf.most_common())}; "
              f"median prediction {sz[len(sz)//2]}, empty {sum(1 for s in sz if s == 0)}")
    print(f"\n  -> ko_predictions_mmw_{{walk,uniform,shuffled}}.json")


if __name__ == "__main__":
    main()
