"""SEALED KNOCKOUT PREDICTION: reason from biology to a prediction, THEN look at the answer.

WHY THIS EXISTS SEPARATELY FROM ko_dossier.py. That file takes a knockout's MEASURED movers and annotates them --
useful for understanding, useless as a test, because the answer is in hand while the story is being told. Any
reasoner shown the answer will produce a story that fits it. This file enforces the opposite order.

    STAGE 1  build a dossier per knockout from A PRIORI information only: function, compartment, pathways, signed
             TF targets, reactions catalysed and what they produce, reaction partners, upstream TFs.
             The measured response is written to a SEPARATE file and not read.
    STAGE 2  predictions are written down and committed.
    STAGE 3  score against the sealed answers.

The reasoning chain the dossier is built to support is the biological one, not a graph walk: knocked-out gene ->
what it transcriptionally regulates, and in which direction -> what those gene products DO -> what the cell
therefore lacks -> what fails as a consequence.

WHAT IS IN A DOSSIER AND WHY

    function / location   UniProt curated text: what the protein does and where
    pathways              Reactome sets it belongs to
    tf_targets_up/down    SIGNED regulation only, restricted to genes that are actually measured. The 91% of the
                          regulation layer that is unsigned ChIP binding is reported as a COUNT and flagged, never
                          mixed in: binding was measured against Perturb-seq regulation in this same cell line and
                          overlapped at chance (fold 0.96, p=0.68, permutation-verified).
    catalyses             steps the protein drives, with substrates and products -- what the cell stops making
    reaction_partners     what fails alongside it
    upstream_tfs          how a perturbation elsewhere could reach this gene

SELECTION IS BY FUNCTIONAL CLASS, NOT BY EFFECT SIZE. Picking the largest responders would select ribosomal and
proteasomal knockouts whose answer is "everything moves", which tests nothing about mechanism and flatters any
predictor. Classes are assigned from a priori annotation only, and the middle member of each class is taken rather
than the extreme, so the sample is not chosen to suit the prediction.

SCORING. For each knockout the prediction is a ranked gene list. Reported: hits against the measured specific
movers, precision at the predicted-set size, and recall. Two controls, because a knockout's movers are not a
uniform random draw from the transcriptome:
    RANDOM        genes drawn at random from the measured non-tide readout
    FREQUENCY     the globally most-often-moved non-tide genes -- the "usual suspects" predictor, which needs no
                  biology at all and is the real bar to clear
"""
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TAU, TIDE_FRAC, MIN_SPEC = 1.0, 0.05, 5
# Cohort size. 50 was set by the old readout, which had only 378 scorable knockouts and could not support more
# without exhausting the smaller functional classes. The gwps readout has 1,110, so KO_NPICK lets the cohort grow
# to match -- which is the entire reason for rebuilding: every comparison this project ran came back
# "indistinguishable" on n<=39, and that is a power problem rather than a modelling one.
NPICK = int(os.environ.get("KO_NPICK", "50"))


def load():
    """The readout matrix. Two possible sources, and which one is used is an EXPLICIT choice, never automatic.

    The default stays nlz_K562.pkl: 1,400 knockouts, each stored as its TOP 250 GENES ONLY -- 3.0% of the
    transcriptome, everything else recorded as exactly zero. Every number this project has published was measured
    against that file, so silently preferring a richer source the moment it appears on disk would make old and new
    results look comparable when they are not.

    Set CELL_READOUT=gwps to use the genome-wide rebuild instead (full 8,248-gene profiles over ~5,200 perturbations
    that clear Replogle's energy test). That switch INVALIDATES rather than extends every published comparison
    number and must be paired with re-scoring all methods -- which is the point of using it.
    """
    if os.environ.get("CELL_READOUT", "").lower() == "gwps":
        z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
        kos, genes, M = list(z["kos"]), list(z["genes"]), z["M"]
        ki = {k: i for i, k in enumerate(kos)}
        gi = {g: i for i, g in enumerate(genes)}
        A = np.abs(M)
        tide = (A >= TAU).mean(0) >= TIDE_FRAC
        print(f"  [readout=gwps] {len(kos):,} perturbations x {len(genes):,} genes; "
              f"tide {int(tide.sum()):,}")
        return kos, genes, A, tide, ki, gi
    KO = pickle.load(open(SP / "nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    genes = sorted({g for v in KO.values() for g in v})
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(kos), len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    return kos, genes, A, tide, ki, gi


def build(skip=0, tag=""):
    """`skip` discards the first N round-robin picks and takes the NEXT NPICK, yielding a cohort that is DISJOINT
    from the default one but selected by the identical rule -- same classes, same round-robin, same middle-member
    choice, still never by effect size. That is what makes it a usable holdout: a rule tuned on the first 50 can
    be tested on the next 50 without the selection itself changing underneath it."""
    kos, genes, A, tide, ki, gi = load()
    gset = set(genes)
    spec = {k: sorted(genes[t] for t in np.where(A[ki[k]] >= TAU)[0] if not tide[t]) for k in kos}

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    up, down, upstream = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    unsigned = collections.Counter()
    for e in D.get("reg", []):
        try:
            s, t = int(e[0]), int(e[1])
            w = int(e[2]) if len(e) > 2 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= s < N and 0 <= t < N) or s == t:
            continue
        sn, tn = names[s], names[t]
        if w == 0:
            unsigned[sn] += 1
            continue
        if tn in gset:
            (up if w > 0 else down)[sn].append(tn)
        upstream[tn].append(sn)

    func = {}
    try:
        for g, v in json.loads((OUT / "dark_function_table.json").read_text())["table"].items():
            func[g] = v
    except Exception:
        pass

    net = json.load(open(OUT / "reaction_network.json"))
    comp = json.loads((OUT / "compartments.json").read_text())["steps"]
    catal = collections.defaultdict(list)
    partic = collections.defaultdict(list)
    partners = collections.defaultdict(collections.Counter)
    gcomp = collections.defaultdict(collections.Counter)
    for s in net["steps"]:
        gs = set(s["catalysts"])
        for p in s["in"] + s["out"]:
            gs |= set(p["genes"])
        if not (1 <= len(gs) <= 60):
            continue
        where = comp.get(s["id"], {}).get("compartments", [])
        desc = {"id": s["id"], "src": s["source"], "where": where,
                "in": [p["name"] for p in s["in"] if not p.get("currency")][:5],
                "out": [p["name"] for p in s["out"] if not p.get("currency")][:5]}
        for g in s["catalysts"]:
            catal[g].append(desc)
        for g in gs:
            for c in where:
                gcomp[g][c] += 1
            for h in gs:
                if h != g:
                    partners[g][h] += 1
        for g in gs - set(s["catalysts"]):
            partic[g].append(desc)

    paths = collections.defaultdict(list)
    for line in (SP / "ReactomePathways.gmt").read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            for g in p[2:]:
                paths[g].append(p[0])

    scorable = [k for k in kos if len(spec[k]) >= MIN_SPEC]
    cls = {}
    for k in scorable:
        if up.get(k) or down.get(k):
            cls[k] = "TF (signed regulon)"
        elif catal.get(k):
            cls[k] = "enzyme/catalyst"
        elif partic.get(k):
            cls[k] = "reaction participant"
        elif paths.get(k):
            cls[k] = "pathway member only"
        else:
            cls[k] = "no annotation context"
    byc = collections.defaultdict(list)
    for k, c in cls.items():
        byc[c].append(k)
    order = sorted(byc, key=lambda c: -len(byc[c]))
    pick, i = [], 0
    while len(pick) < NPICK + skip and any(byc.values()):
        c = order[i % len(order)]
        i += 1
        if byc[c]:
            pick.append(byc[c].pop(len(byc[c]) // 2))
    pick = sorted(pick[skip:])

    doss = {}
    for k in pick:
        f = func.get(k, {})
        doss[k] = {
            "gene": k, "class": cls[k], "protein_name": f.get("name"),
            "function": (f.get("function") or "")[:700] or None,
            "uniprot_location": (f.get("location") or "")[:250] or None,
            "compartments": [c for c, _ in gcomp.get(k, collections.Counter()).most_common(3)],
            "pathways": paths.get(k, [])[:8], "n_pathways": len(paths.get(k, [])),
            "n_up": len(set(up.get(k, []))), "n_down": len(set(down.get(k, []))),
            "tf_targets_up": sorted(set(up.get(k, [])))[:40],
            "tf_targets_down": sorted(set(down.get(k, [])))[:40],
            "unsigned_chip_targets_FLAGGED": unsigned.get(k, 0),
            "n_catalyses": len(catal.get(k, [])), "catalyses": catal.get(k, [])[:5],
            "n_participates": len(partic.get(k, [])), "participates_in": partic.get(k, [])[:5],
            "reaction_partners": [g for g, _ in partners.get(k, collections.Counter()).most_common(12)],
            "upstream_tfs": sorted(set(upstream.get(k, [])))[:12],
        }
    ans = {k: {"n": len(spec[k]), "movers": spec[k]} for k in pick}
    json.dump(doss, open(OUT / f"ko_pred_dossiers{tag}.json", "w"), indent=1)
    json.dump(ans, open(OUT / f"ko_pred_answers{tag}.json", "w"), indent=1)
    print(f"{len(pick)} knockouts selected by functional class (never by effect size):")
    for c, n in collections.Counter(cls[k] for k in pick).most_common():
        print(f"   {c:<28} {n}")
    sz = sorted(len(spec[k]) for k in pick)
    print(f"  measured movers: median {sz[len(sz)//2]}, range {sz[0]}-{sz[-1]}")
    print(f"  -> {OUT/('ko_pred_dossiers'+tag+'.json')} (a priori only)")
    print(f"  -> {OUT/('ko_pred_answers'+tag+'.json')} (sealed)")


def score(pred_file="outputs/orphan/ko_predictions.json", tag=""):
    kos, genes, A, tide, ki, gi = load()
    ans = json.loads((OUT / f"ko_pred_answers{tag}.json").read_text())
    preds = json.loads(Path(pred_file).read_text())
    nontide = [genes[i] for i in range(len(genes)) if not tide[i]]
    freq = collections.Counter()
    for k in kos:
        for t in np.where(A[ki[k]] >= TAU)[0]:
            if not tide[t]:
                freq[genes[t]] += 1
    rng = np.random.default_rng(0)
    rows = []
    for k, p in preds.items():
        if k not in ans:
            continue
        truth = set(ans[k]["movers"])
        pl = [g for g in p.get("predicted_genes", []) if g in gi]
        n = len(pl)
        if not n:
            continue
        hit = len(set(pl) & truth)
        # controls sized to the SAME prediction length
        rnd = np.mean([len(set(rng.choice(nontide, n, replace=False)) & truth) for _ in range(200)])
        frq = len({g for g, _ in freq.most_common(n)} & truth)
        rows.append({"gene": k, "n_pred": n, "n_truth": len(truth), "hits": hit,
                     "precision": hit / n, "recall": hit / max(len(truth), 1),
                     "random_hits": float(rnd), "frequency_hits": frq,
                     "lift_vs_random": hit / max(rnd, 1e-9), "beats_frequency": hit > frq})
    if not rows:
        raise SystemExit("no scorable predictions")
    H = sum(r["hits"] for r in rows)
    R = sum(r["random_hits"] for r in rows)
    F = sum(r["frequency_hits"] for r in rows)
    P = sum(r["n_pred"] for r in rows)
    print(f"\n  {len(rows)} knockouts scored, {P:,} predicted genes total")
    print(f"    hits: reasoned {H}   frequency-baseline {F}   random {R:.1f}")
    print(f"    precision: reasoned {H/P:.4f}   frequency {F/P:.4f}   random {R/P:.4f}")
    print(f"    lift over random {H/max(R,1e-9):.2f}x   over frequency {H/max(F,1e-9):.2f}x")
    print(f"    knockouts where reasoning beat the frequency baseline: "
          f"{sum(1 for r in rows if r['beats_frequency'])}/{len(rows)}")
    print(f"\n  per knockout (hits/predicted, truth size, vs frequency)")
    for r in sorted(rows, key=lambda r: -r["hits"]):
        flag = "  <-- beats freq" if r["beats_frequency"] else ""
        print(f"    {r['gene']:<10} {r['hits']:>3}/{r['n_pred']:<3} of {r['n_truth']:<4} truth   "
              f"freq {r['frequency_hits']:<3} rand {r['random_hits']:.1f}{flag}")
    # ONE SCORE FILE PER PREDICTION FILE. A single fixed output path meant each scoring run silently clobbered
    # the previous one, so the four methods could not be compared after the fact and an earlier analysis was run
    # against a stale file without noticing.
    tag = Path(pred_file).stem.replace("ko_predictions", "").strip("_") or "base"
    outp = OUT / f"ko_score_{tag}.json"
    json.dump({"method": tag, "rows": rows, "hits": H, "random": R, "frequency": F, "n_pred": P,
               "precision": H / P, "beat_frequency": sum(1 for r in rows if r["beats_frequency"]),
               "n_scored": len(rows)}, open(outp, "w"), indent=1)
    print(f"\n  -> {outp}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "holdout":
        build(skip=NPICK, tag="_holdout")
    else:
        build()
