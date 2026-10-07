"""SPATIAL AND NETWORK CONTEXT AS A RANKING SIGNAL -- constraints rather than similarity.

WHY THIS IS DIFFERENT FROM THE SIX THAT FAILED.  Every re-ranker so far scored a candidate by RESEMBLANCE:
embedding similarity, sequence pairing, pocket shape, docking energy, learned preference. All of them lost
to the language model's own ordering. Localisation is not a resemblance score, it is a physical
CONSTRAINT -- a cytosolic enzyme cannot catalyse a reaction in the mitochondrial matrix, whatever it looks
like. Constraints and preferences fail differently, and a constraint has never been tested here.

THREE SIGNALS, TWO SPATIAL AND ONE FROM THE NETWORK

    compartment   the candidate's annotated localisation against the compartments the reaction's
                  participants sit in. Both sides are free text with different vocabularies -- "nucleus"
                  against "nucleoplasm", "mitochondrion" against "mitochondrial matrix" -- so both are
                  reduced to a controlled vocabulary first, the same reduction the fill's audit needed.
    process       the candidate's coarse functional class against the reaction's shape. A step whose only
                  change is a compartment label is transport, and transporters are a named class.
    pathway       THE NETWORK TERM. The query reaction has chemically nearest SOLVED neighbours, and those
                  neighbours have catalysts with known pathways. A candidate sharing a pathway with the
                  neighbourhood's catalysts is in the right part of the network. This uses the reaction's
                  POSITION rather than any property of the protein, which nothing here has tried.

THE CONTROL THAT DECIDES IT, AND IT IS NOT THE USUAL ONE.  The prompts already render every participant
with its compartment, so the model has seen this information. The question is therefore not "does
localisation predict the catalyst" -- it plainly does -- but "does stating it explicitly add anything the
model did not already use". So the run first measures whether the model's ERRORS are
compartment-inconsistent. If its wrong answers respect compartment just as well as its right ones, the
constraint is already satisfied by everything it proposes and cannot separate them, and no weighting of it
will help.

ARMS
    tournament        the current ordering, and the bar
    spatial           re-ranked by compartment + process, LLM rank as prior
    network           re-ranked by pathway agreement with the neighbourhood's catalysts
    spatial_network   both
    shuffled_comp     the same pipeline with gene compartments randomly permuted across genes. Separates
                      "real localisation informs" from "any per-gene score reshuffles usefully".
    random            matched-magnitude reordering

PREDECLARED, before any number:
    a spatial or network arm beats the tournament order AND beats its shuffled control
        -> constraints succeed where preferences failed, and the reason six re-rankers lost was that they
           were all scoring resemblance.
    the model's errors already respect compartment as well as its correct answers do
        -> the constraint is saturated. It is real, the model already uses it, and explicit weighting is
           redundant rather than wrong.
    the arms lose like the others
        -> seventh failure, and the finding is that re-ranking this shortlist does not work by any means
           tried: similarity, structure, learning, or constraint.

-> outputs/orphan/cell_orphan_spatial.json
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_fill import build_pool, examples_for, _compvocab, K  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEED = 271
MEMBRANE = ("membrane", "transporter", "channel")


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("SPATIAL AND NETWORK CONTEXT AS A RANKING SIGNAL")
    report("=" * 100)
    report("  Six re-rankers scored RESEMBLANCE -- embeddings, sequence pairing, pocket shape, docking,")
    report("  learned preference -- and all lost to the model's own ordering. Localisation is not a")
    report("  resemblance score, it is a physical CONSTRAINT, and constraints have never been tested here.")
    report("  But the prompts already show every compartment, so the real question is whether stating it")
    report("  explicitly adds anything the model did not already use.")

    D = json.load(open(OUT / "cell_complete.json"))
    gmeta = {g["name"].upper(): g for g in D["genes"]}
    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    ans = json.load(open(OUT / "rankv2_answers.json"))
    B, orphans, cat_of = build_pool()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    rng = np.random.default_rng(SEED)

    # shuffled-compartment control: permute the comp labels across genes, keeping the distribution
    names = sorted(gmeta)
    perm = list(names)
    rng.shuffle(perm)
    shuf_comp = {a: (gmeta[b].get("comp") or "") for a, b in zip(names, perm)}

    rows = []
    for pid, m in keep.items():
        a = ans.get(pid)
        if not a or not a.get("tournament"):
            continue
        o = meta[m["orig"]]
        s = steps[o["step"]]
        rcomp = set()
        for q in (s.get("in") or []) + (s.get("out") or []):
            rcomp |= _compvocab(q.get("compartment"))
        # transport shape: the same species named on both sides in different compartments
        ins = {q.get("name") for q in (s.get("in") or [])}
        outs = {q.get("name") for q in (s.get("out") or [])}
        is_transport = len(ins & outs) > 0
        # the network term: pathways of the catalysts of this reaction's chemical neighbourhood
        nb, _ = examples_for(o["step"], P, cats, cat_of)
        npath = Counter()
        nproc = Counter()
        for j in nb:
            for g in cats[j]:
                gm = gmeta.get(g.upper())
                if not gm:
                    continue
                if gm.get("path"):
                    npath[gm["path"].strip()] += 1
                if gm.get("proc"):
                    nproc[gm["proc"].strip()] += 1
        rows.append({"order": a["tournament"][:10], "truth": {g.upper() for g in o["truth"]},
                     "obscure": o["obscure"], "rcomp": rcomp, "transport": is_transport,
                     "npath": npath, "nproc": nproc})
    report(f"\n  {len(rows)} held-out reactions | "
           f"{sum(1 for r in rows if r['rcomp'])} carry a compartment | "
           f"{sum(1 for r in rows if r['transport'])} look like transport | "
           f"{sum(1 for r in rows if r['npath'])} have a pathway neighbourhood")

    def comp_score(g, r, cmap):
        gm = gmeta.get(g.upper())
        if not gm or not r["rcomp"]:
            return 0.0
        gc = (cmap.get(g.upper(), gm.get("comp") or "")).lower()
        if not gc:
            return 0.0
        if any(w in gc for w in MEMBRANE):
            return 0.5 if r["transport"] else 0.0     # a membrane protein spans compartments by design
        gv = _compvocab(gc)
        if not gv:
            return 0.0
        return 1.0 if (gv & r["rcomp"]) else -1.0

    def proc_score(g, r):
        gm = gmeta.get(g.upper())
        if not gm:
            return 0.0
        p = (gm.get("proc") or "").lower()
        if r["transport"] and "transport" in p:
            return 1.0
        if r["nproc"] and p and p == (r["nproc"].most_common(1)[0][0] or "").lower():
            return 0.5
        return 0.0

    def path_score(g, r):
        gm = gmeta.get(g.upper())
        if not gm or not r["npath"] or not gm.get("path"):
            return 0.0
        tot = sum(r["npath"].values()) or 1
        return 2.0 * r["npath"].get(gm["path"].strip(), 0) / tot

    def rr(r, fn):
        base = {g: -i for i, g in enumerate(r["order"])}
        return sorted(r["order"], key=lambda g: -(0.35 * base[g] + fn(g, r)))

    ARMS = {
        "tournament": lambda r: r["order"],
        "spatial": lambda r: rr(r, lambda g, x: comp_score(g, x, {}) + proc_score(g, x)),
        "network": lambda r: rr(r, lambda g, x: path_score(g, x)),
        "spatial_network": lambda r: rr(r, lambda g, x: comp_score(g, x, {}) + proc_score(g, x)
                                        + path_score(g, x)),
        "shuffled_comp": lambda r: rr(r, lambda g, x: comp_score(g, x, shuf_comp) + proc_score(g, x)),
        "random": lambda r: rr(r, lambda g, x: rng.normal()),
    }
    ob = np.array([r["obscure"] for r in rows])
    report(f"\n    {'arm':<20}{'@1':>8}{'@3':>8}{'@5':>8}{'@1 obs':>9}")
    res = {}
    for a, fn in ARMS.items():
        h1 = np.array([bool(r["truth"] & set(fn(r)[:1])) for r in rows])
        h3 = np.array([bool(r["truth"] & set(fn(r)[:3])) for r in rows])
        h5 = np.array([bool(r["truth"] & set(fn(r)[:5])) for r in rows])
        res[a] = {"@1": float(h1.mean()), "@3": float(h3.mean()), "@5": float(h5.mean()),
                  "@1_obscure": float(h1[ob].mean()) if ob.sum() else None}
        report(f"    {a:<20}{h1.mean():>8.3f}{h3.mean():>8.3f}{h5.mean():>8.3f}"
               f"{(h1[ob].mean() if ob.sum() else float('nan')):>9.3f}")

    # ---- IS THE CONSTRAINT ALREADY SATISFIED? --------------------------------------------------------
    report("\n  IS THE CONSTRAINT ALREADY SATISFIED? -- the question that decides whether this can help")
    report("  If the model's WRONG answers respect compartment as well as its right ones, then everything")
    report("  it proposes already satisfies the constraint and it cannot separate them.")
    corr, wrong = [], []
    for r in rows:
        if not r["rcomp"]:
            continue
        top = r["order"][0]
        (corr if (r["truth"] & {top}) else wrong).append(comp_score(top, r, {}))
    report(f"    compartment score of CORRECT top-1 answers   {np.mean(corr):+.3f}  (n={len(corr)})")
    report(f"    compartment score of WRONG   top-1 answers   {np.mean(wrong):+.3f}  (n={len(wrong)})")
    gap = float(np.mean(corr) - np.mean(wrong)) if corr and wrong else float("nan")
    report(f"    gap {gap:+.3f}")

    report("\n  READING")
    best = max((res[a]["@1"], a) for a in ARMS if a != "tournament")
    t_ = res["tournament"]["@1"]
    s_ = res["shuffled_comp"]["@1"]
    if best[0] > t_ + 0.01 and best[0] > s_ + 0.01:
        report(f"  {best[1]} beats the ordering ({best[0]:.3f} vs {t_:.3f}) and its shuffled control")
        report(f"  ({s_:.3f}). Constraints succeed where six resemblance scores failed.")
    elif abs(gap) < 0.15:
        report(f"  THE CONSTRAINT IS ALREADY SATISFIED. Correct and wrong top-1 answers score {np.mean(corr):+.3f}")
        report(f"  and {np.mean(wrong):+.3f} on compartment -- a gap of {gap:+.3f}. Everything the model")
        report("  proposes is already in the right compartment, because the prompt shows it the")
        report("  compartments and it uses them. The signal is real, it is not NEW, and weighting it")
        report("  explicitly can only add noise. That is a different failure from the other six: not")
        report("  'too weak' but 'already spent'.")
    else:
        report(f"  Seventh failure: best arm {best[0]:.3f} against {t_:.3f} for the ordering as given.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_orphan_spatial", "n": len(rows), "arms": res,
               "comp_correct": float(np.mean(corr)) if corr else None,
               "comp_wrong": float(np.mean(wrong)) if wrong else None, "log": log},
              open(OUT / "cell_orphan_spatial.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_spatial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
