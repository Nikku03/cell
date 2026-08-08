"""AN INDEPENDENT CHECK: score the fill pipeline on hidden answers that MATCH the orphans.

WHY THE HAND CHECK WAS NOT ENOUGH.  I verified 30 filled rows by reading the chemistry and judging the gene,
and got ~87%. That number cannot be used: I am the same model that produced the predictions, so wherever it
was confidently wrong I would be confidently wrong in the same way. Same-model verification catches
structural faults -- it found the giveaways and the association steps -- but it cannot estimate accuracy.

An independent check needs an answer key the predictor never sees. There is one: the 10,977 reactions that
DO have annotated catalysts. Hide the answer, run the identical pipeline, score against the annotation.

BUT A HELD-OUT SET IS ONLY A CHECK ON THE FILL IF IT LOOKS LIKE THE FILL.  The transfer analysis found the
existing 400-reaction benchmark does not: real orphans have far thinner retrieval (mean 5.3 examples against
7.8, and 35% with NONE against 11.8%), and worse, the benchmark contains NO obscure reaction with zero
examples, so the cell holding a third of the deliverable had never been measured at all. Reporting 0.688
against the fill was therefore quoting a number earned on an easier population.

So this set is STRATIFIED TO MATCH, cell by cell, on the two axes that were shown to matter:

    literature bin  x  number of retrieved examples

Every cell is filled to the orphans' own proportion where the solved pool can supply it, and where it
cannot, the shortfall is REPORTED rather than quietly redistributed into easier cells.

FOUR LEAKS CLOSED, because a held-out solved reaction is not automatically a fair proxy for an orphan:
    self          a reaction never appears among its own examples. Already enforced upstream, asserted here.
    sibling       held-out reactions are removed from EVERY example pool. Orphans have no catalyst, so no
                  orphan can ever appear as an example for another orphan -- but two held-out solved
                  reactions could show each other's answers, which is a leak the fill setting does not have.
    giveaway      rows whose catalyst symbol appears verbatim in the reaction text are dropped, because
                  6.7% of the fill rows were shown to be label-reading rather than prediction and including
                  them here would score that same freebie as if it were skill.
    association   complex-assembly steps are dropped, matching the fill's own filter.

WHAT IS REPORTED
    top-1 overall, and per cell, so the projection onto the fill is a weighted sum of MEASURED cells
    the same triage fields the fill carries (starved, copied), so accuracy can be read against them
    the cells the solved pool cannot supply, named, since those remain unmeasured for the fill too

PREDECLARED, before any number:
    matched-set accuracy within ~0.05 of 0.688
        -> the headline transfers, the fill can be quoted at it, and the transfer worry is closed.
    matched-set accuracy materially below 0.688
        -> the fill is worse than the benchmark implies and must be quoted at THIS number instead. The
           earlier headline was measured on a population that does not resemble the deliverable.
    zero-example cells still unfillable from the solved pool
        -> that share of the fill stays unmeasured no matter what, and must be labelled as such rather
           than covered by an overall average.

usage:  python cell_orphan_holdout.py dump [N]    write matched held-out prompts
        python cell_orphan_holdout.py score       score worker answers against the hidden truth

-> outputs/orphan/cell_orphan_holdout.json
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
import cell_orphan_obscurity as O  # noqa: E402
from cell_orphan_incontext import _render  # noqa: E402
from cell_orphan_fill import build_pool, K, BATCH, NONCAT, ABSTRACT  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/holdout")
NTARGET = 700
SEED = 149
ECELLS = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-11", 8, 11), ("12", 12, 12)]


def ecell(k):
    for lab, lo, hi in ECELLS:
        if lo <= k <= hi:
            return lab
    return "12"


def examples_excluding(i, P, cats, cat_of, banned, k=K):
    """Nearest solved reactions, with the whole held-out set banned from the pool.

    The ban is what makes this a fair proxy. An orphan can never see another orphan's catalyst because
    orphans have none; two held-out solved reactions could show each other's answers, and that leak would
    inflate the score in exactly the cells this set exists to measure.
    """
    pi = P[i]
    sc = {}
    for p in pi:
        for j in cat_of.get(p, ()):
            if j == i or j in banned:
                continue
            u = len(pi | P[j])
            sc[j] = max(sc.get(j, 0), len(pi & P[j]) / u if u else 0)
    top = sorted(sc, key=lambda j: -sc[j])[:k]
    return top, [sc[j] for j in top]


def cmd_dump(ntarget=NTARGET):
    B, orph_real, cat_of = build_pool()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    papers = O._lit()
    rng = np.random.default_rng(SEED)

    # ---- the ORPHAN profile we must match -----------------------------------------------------------
    prof = defaultdict(int)
    for i in orph_real[:4000]:
        top, _ = examples_excluding(i, P, cats, cat_of, set())
        prof[ecell(len(top))] += 1
    tot = sum(prof.values()) or 1
    prof = {k: v / tot for k, v in prof.items()}
    print("  orphan retrieval profile: " + ", ".join(f"{k} {v:.1%}" for k, v in sorted(prof.items())))

    # ---- candidate solved reactions, with their literature bin --------------------------------------
    cand = [i for i in cats if P[i] and steps[i].get("category") not in NONCAT + ABSTRACT]
    lit = {}
    for i in cand:
        pv = [papers(g) for g in cats[i]]
        if any(v is None for v in pv):
            continue
        lit[i] = max(pv)
    print(f"  {len(lit)} solved reactions with a literature count")

    # cell = (obscure?, example cell). Examples computed with NOTHING banned yet, only to bucket them.
    buckets = defaultdict(list)
    for i in lit:
        top, _ = examples_excluding(i, P, cats, cat_of, set())
        buckets[(O._binof(lit[i]) == 0, ecell(len(top)))].append(i)

    want, sel, short = {}, [], []
    for lab, _, _ in ECELLS:
        n_cell = int(round(prof.get(lab, 0.0) * ntarget))
        # hold the obscure share of the fill population: the gaps are obscure by construction
        for ob, frac in ((True, 0.60), (False, 0.40)):
            k = int(round(n_cell * frac))
            pool = buckets.get((ob, lab), [])
            take = min(k, len(pool))
            if take < k:
                short.append((lab, "obscure" if ob else "known", k, take))
            if take:
                sel += [int(x) for x in rng.choice(pool, take, replace=False)]
            want[(ob, lab)] = (k, take)
    sel = sorted(set(sel))
    banned = set(sel)
    print(f"  selected {len(sel)} held-out reactions")
    if short:
        print("  CELLS THE SOLVED POOL CANNOT SUPPLY (reported, not redistributed):")
        for lab, ob, k, take in short:
            print(f"    examples {lab:<5} {ob:<8} wanted {k:>3}, available {take:>3}")

    # ---- build prompts with the whole held-out set banned from every pool ---------------------------
    SP.mkdir(parents=True, exist_ok=True)
    meta, txt, kept = {}, [], []
    for i in sel:
        top, sims = examples_excluding(i, P, cats, cat_of, banned)
        s = steps[i]
        body = _render(s, steps)
        truth = {g.upper() for g in cats[i]}
        # GIVEAWAY and ASSOCIATION filters, identical in spirit to the fill's, so the two populations match
        if any(re.search(r"\b" + re.escape(g) + r"\b", body.upper()) for g in truth):
            continue
        outl = [l for l in body.split("\n") if l.strip().startswith("OUT")]
        inl = [l for l in body.split("\n") if l.strip().startswith("IN")]
        if outl and outl[0].count("[COMPLEX") == 1 and outl[0].count("+") == 0 and inl and inl[0].count("+"):
            continue
        pid = len(kept)
        kept.append(i)
        ex = [f"  SOLVED: {_render(steps[j], steps)}\n     catalyst: {', '.join(sorted(cats[j]))}"
              for j in top]
        head = (f"#{pid}  ({s.get('source','?')}, {s.get('category','?')}, "
                f"{'reversible' if s.get('reversible') else 'irreversible'})")
        txt.append(head + "\n" + body + "\n  --- SOLVED REACTIONS FOR REFERENCE ---\n" + "\n".join(ex)
                   if ex else head + "\n" + body + "\n  --- NO SIMILAR SOLVED REACTION FOUND ---")
        meta[str(pid)] = {"step": int(i), "truth": sorted(truth), "n_examples": len(top),
                          "lit": int(lit[i]), "obscure": bool(O._binof(lit[i]) == 0),
                          "cell": ecell(len(top)),
                          "shown": bool(truth & {g.upper() for j in top for g in cats[j]})}
    for b in range(0, len(txt), BATCH):
        (SP / f"hold_b{b // BATCH}.txt").write_text(("\n\n" + "=" * 90 + "\n\n").join(txt[b:b + BATCH]))
    # the answer must never be in the prompt
    for pid, t_ in enumerate(txt):
        for g in meta[str(pid)]["truth"]:
            assert not re.search(r"\b" + re.escape(g) + r"\b", t_.split("--- SOLVED")[0].upper()), \
                f"holdout {pid} names its own answer"
    json.dump(meta, open(OUT / "holdout_meta.json", "w"), indent=1)
    print(f"  wrote {len(txt)} prompts in {(len(txt)+BATCH-1)//BATCH} batches -> {SP}")
    print(f"  obscure {np.mean([m['obscure'] for m in meta.values()]):.1%} | "
          f"mean examples {np.mean([m['n_examples'] for m in meta.values()]):.1f} | "
          f"answer shown {np.mean([m['shown'] for m in meta.values()]):.1%}")
    return 0


def cmd_score():
    meta = json.load(open(OUT / "holdout_meta.json"))
    f = OUT / "holdout_answers.json"
    if not f.exists():
        print("  no answers yet")
        return 1
    ans = json.load(open(f))
    pids = [p for p in meta if p in ans]
    hit = np.array([str(ans[p].get("gene", "")).upper() in set(meta[p]["truth"]) for p in pids])
    ob = np.array([meta[p]["obscure"] for p in pids])
    sh = np.array([meta[p]["shown"] for p in pids])
    cell = np.array([meta[p]["cell"] for p in pids])
    print("=" * 100)
    print("INDEPENDENT CHECK -- hidden answers, matched to the orphan retrieval profile")
    print("=" * 100)
    print(f"  {len(pids)} of {len(meta)} answered")
    print(f"\n  top-1 overall            {hit.mean():.3f}   (n={len(hit)})")
    print(f"  top-1 obscure            {hit[ob].mean():.3f}   (n={ob.sum()})")
    print(f"  top-1 obscure, NOT shown {hit[ob & ~sh].mean():.3f}   (n={(ob & ~sh).sum()})")
    print(f"\n  BY RETRIEVAL CELL -- the axis the transfer analysis found decisive")
    print(f"    {'examples':<10}{'n':>6}{'all':>9}{'obscure':>10}{'n obs':>7}")
    for lab, _, _ in ECELLS:
        m = cell == lab
        if not m.sum():
            continue
        mo = m & ob
        print(f"    {lab:<10}{m.sum():>6}{hit[m].mean():>9.3f}"
              f"{(hit[mo].mean() if mo.sum() else float('nan')):>10.3f}{mo.sum():>7}")
    res = {"test": "cell_orphan_holdout", "n": len(pids), "top1": float(hit.mean()),
           "top1_obscure": float(hit[ob].mean()) if ob.sum() else None,
           "by_cell": {lab: {"n": int((cell == lab).sum()),
                             "acc": float(hit[cell == lab].mean()) if (cell == lab).sum() else None}
                       for lab, _, _ in ECELLS}}
    print(f"\n  READING")
    print(f"  This is the number the filled table should carry, not 0.688: it was measured on hidden")
    print(f"  answers, with the held-out set banned from its own example pools, giveaways and association")
    print(f"  steps removed, and the retrieval profile matched to the orphans cell by cell.")
    json.dump(res, open(OUT / "cell_orphan_holdout.json", "w"), indent=2)
    return 0


def cmd_rank():
    """Score RANKED answers: recall@k for k = 1, 5, 10, 20.

    Top-1 is the wrong metric for a curation aid. A curator handed one wrong gene has nothing; a curator
    handed a ranked twenty containing the right gene has a two-minute job. The two numbers can differ
    enormously and only one of them describes how the artifact would actually be used.
    """
    meta = json.load(open(OUT / "holdout_meta.json"))
    f = OUT / "holdout_ranked.json"
    if not f.exists():
        print("  no ranked answers yet")
        return 1
    ans = json.load(open(f))
    pids = [p for p in meta if p in ans]
    ob = np.array([meta[p]["obscure"] for p in pids])
    sh = np.array([meta[p]["shown"] for p in pids])
    cell = np.array([meta[p]["cell"] for p in pids])
    KS = (1, 5, 10, 20)
    hits = {k: [] for k in KS}
    nlist = []
    for p in pids:
        lst = [str(g).upper() for g in (ans[p].get("genes") or [])]
        nlist.append(len(lst))
        tr = set(meta[p]["truth"])
        for k in KS:
            hits[k].append(bool(tr & set(lst[:k])))
    for k in KS:
        hits[k] = np.array(hits[k])
    print("=" * 100)
    print("RANKED RECALL on the independent held-out set")
    print("=" * 100)
    print(f"  {len(pids)} reactions | mean list length {np.mean(nlist):.1f}")
    print(f"\n  {'population':<26}" + "".join(f"{'@'+str(k):>9}" for k in KS) + f"{'n':>7}")
    for lab, m in (("ALL", np.ones(len(pids), bool)), ("obscure", ob),
                   ("obscure, answer NOT shown", ob & ~sh)):
        print(f"  {lab:<26}" + "".join(f"{hits[k][m].mean():>9.3f}" for k in KS) + f"{int(m.sum()):>7}")
    print(f"\n  BY RETRIEVAL CELL (all reactions)")
    print(f"    {'examples':<10}" + "".join(f"{'@'+str(k):>9}" for k in KS) + f"{'n':>7}")
    for lab, _, _ in ECELLS:
        m = cell == lab
        if not m.sum():
            continue
        print(f"    {lab:<10}" + "".join(f"{hits[k][m].mean():>9.3f}" for k in KS) + f"{int(m.sum()):>7}")
    res = {"test": "cell_orphan_holdout_ranked", "n": len(pids),
           "recall": {str(k): {"all": float(hits[k].mean()),
                               "obscure": float(hits[k][ob].mean()) if ob.sum() else None,
                               "obscure_not_shown": float(hits[k][ob & ~sh].mean())} for k in KS}}
    print(f"\n  READING")
    o1, o10, o20 = (res["recall"][str(k)]["obscure"] for k in (1, 10, 20))
    print(f"  On obscure enzymes the list is worth far more than its head: {o1:.3f} at rank 1 against")
    print(f"  {o10:.3f} in a top-10 and {o20:.3f} in a top-20. A curator does not need the model to be")
    print(f"  right, only to put the right gene somewhere they will look.")
    json.dump(res, open(OUT / "cell_orphan_holdout_ranked.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "score"
    if c == "rank":
        raise SystemExit(cmd_rank())
    if c == "dump":
        raise SystemExit(cmd_dump(int(sys.argv[2]) if len(sys.argv) > 2 else NTARGET))
    raise SystemExit(cmd_score())
