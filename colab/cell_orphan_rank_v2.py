"""A BETTER RANKER: refine the model's own ordering instead of overwriting it with a foreign signal.

WHAT THE EVIDENCE SAYS TO DO.  Four things are now measured and together they point at one design.

    the ceiling is high        recall@50 on obscure reactions is 0.808 while rank-1 is 0.474, so the answer
                               is usually present and badly placed. Ranking, not generation, is the gap.
    the prior is strong        every foreign re-ranker lost to leaving the list alone: ESM 0.122 vs 0.675,
                               fusion 0.643, pocket-hinge 0.483 vs 0.763, docking 0.600 vs 0.825.
    but structure is real      docking the TRUE substrate beat docking a random one, 0.600 vs 0.475 and
                               +0.208 on obscure. The signal exists; it is simply weaker than the prior.
    the ordering is one pass   all fifty candidates were ranked in a single listwise sweep, which is the
                               weakest way to use a model for ranking and the one part never revisited.

So the move is not another external signal. It is to spend more of the MODEL's own judgement on the
ordering, in forms known to beat a single listwise pass, and to change the ranking only where it is
uncertain rather than everywhere.

THREE ARMS, EACH ATTACKING A DIFFERENT WEAKNESS OF THE ONE-PASS ORDER

    tournament   pairwise comparison of the top candidates. Ranking many items at once forces a model to
                 hold every candidate in view simultaneously; comparing two at a time asks a far easier
                 question and is well established to beat listwise ordering. The reaction and its worked
                 examples stay in view, so this refines the same judgement rather than importing another.
    fusion       three INDEPENDENT rankings, combined by reciprocal rank fusion. Not self-consistency on a
                 single answer, which was predicted to fail here because every error names a MORE published
                 gene and voting concentrates on the mode of a popularity-biased distribution. Rank fusion
                 keeps a candidate that one sample ranks first and another ranks tenth, which majority
                 voting discards.
    confidence   the same ranking plus a declared confidence and a stated reason for doubt. This is not a
                 ranker; it is what makes SELECTIVE re-ranking possible. Overwriting a confident correct
                 answer is exactly how the previous four re-rankers lost, so knowing where the model is
                 unsure is worth more than another global reordering.

WHAT IS SCORED, and against what.  The same held-out reactions with hidden answers, so every number is
comparable to the 0.474/0.808 already on the ledger. The bar is the EXISTING top-50 order, because that is
what would ship today.

PREDECLARED, before any number:
    tournament or fusion beats the existing order on obscure top-1
        -> ranking was under-served rather than saturated, and the gain is free of any new data.
    neither beats it, but confidence separates right from wrong answers
        -> the ordering is as good as this model makes it, and the win is abstention or selective docking
           on the uncertain subset instead. That is a smaller claim and still a useful one.
    neither beats it and confidence does not separate
        -> 0.474 is this method's ceiling at rank 1, the shortlist is the deliverable, and effort should
           move to retrieval, which is the one axis measured to matter (0.25 at 4-7 examples against 0.64
           at 12).

usage:  python cell_orphan_rank_v2.py dump     write re-ranking prompts for the held-out set
        python cell_orphan_rank_v2.py score    score the arms against hidden truth

-> outputs/orphan/cell_orphan_rank_v2.json
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
HOLD = SP / "holdout"
DEST = SP / "rankv2"
TOPK = 10
BATCH = 50
SEED = 211


def cmd_dump():
    meta = json.load(open(OUT / "holdout_meta.json"))
    rk = json.load(open(OUT / "holdout_top50_partial.json"))
    sep = "\n\n" + "=" * 90 + "\n\n"
    cache, txt, keep = {}, [], {}
    for p in sorted(rk, key=lambda x: int(x)):
        if p not in meta:
            continue
        b, k = int(p) // BATCH, int(p) % BATCH
        if b not in cache:
            f = HOLD / f"hold_b{b}.txt"
            cache[b] = f.read_text().split(sep) if f.exists() else []
        if k >= len(cache[b]):
            continue
        body = cache[b][k]
        cands = rk[p]["genes"][:TOPK]
        pid = len(txt)
        keep[str(pid)] = {"orig": p, "cands": cands}
        head = body.split("--- SOLVED")[0].strip().split("\n")
        head[0] = f"#{pid}"
        block = "\n".join(head)
        ex = body.split("--- SOLVED REACTIONS FOR REFERENCE ---")
        exs = ("\n  --- SOLVED REACTIONS FOR REFERENCE ---" + ex[1]) if len(ex) > 1 else ""
        cl = "\n  --- CANDIDATES TO RANK (current order) ---\n" + "\n".join(
            f"    {i+1}. {g}" for i, g in enumerate(cands))
        txt.append(block + cl + exs)
    DEST.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(txt), BATCH):
        (DEST / f"rank_b{b // BATCH}.txt").write_text(sep.join(txt[b:b + BATCH]))
    json.dump(keep, open(OUT / "rankv2_meta.json", "w"), indent=1)
    print(f"  wrote {len(txt)} re-ranking prompts in {(len(txt)+BATCH-1)//BATCH} batches -> {DEST}")
    print(f"  each carries the reaction, its worked examples, and the current top-{TOPK}")
    return 0


def rrf(rankings, k=60.0):
    """Reciprocal rank fusion: score = sum 1/(k + rank). Keeps a candidate that one ranking puts first and
    another puts tenth, which majority voting over single answers discards -- and that is the case this
    arm exists for, since every error here names a more-published gene and voting would concentrate on it."""
    s = defaultdict(float)
    for r in rankings:
        for i, g in enumerate(r):
            s[g] += 1.0 / (k + i + 1)
    return [g for g, _ in sorted(s.items(), key=lambda kv: -kv[1])]


def cmd_score():
    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    f = OUT / "rankv2_answers.json"
    if not f.exists():
        print("  no answers yet")
        return 1
    ans = json.load(open(f))
    KS = (1, 3, 5)
    arms = {}
    ob, base_rows = [], []
    for pid, m in keep.items():
        o = meta[m["orig"]]
        base_rows.append((pid, m, set(o["truth"]), o["obscure"]))
    rows = [r for r in base_rows if r[0] in ans]
    if len(rows) < 30:
        print(f"  only {len(rows)} answered; not scoring")
        return 1
    ob = np.array([r[3] for r in rows])

    def add(name, get):
        h = {k: [] for k in KS}
        for pid, m, tr, _ in rows:
            order = get(pid, m) or m["cands"]
            for k in KS:
                h[k].append(bool(tr & set(order[:k])))
        arms[name] = {k: np.array(v) for k, v in h.items()}

    add("existing_order", lambda pid, m: m["cands"])
    add("tournament", lambda pid, m: ans[pid].get("tournament"))
    add("fusion", lambda pid, m: rrf([r for r in ans[pid].get("samples", []) if r] or [m["cands"]]))
    add("confidence_order", lambda pid, m: ans[pid].get("confident"))

    print("=" * 100)
    print("A BETTER RANKER -- refining the model's own ordering")
    print("=" * 100)
    print(f"  {len(rows)} held-out reactions ({int(ob.sum())} obscure)")
    print(f"    {'arm':<20}{'@1':>8}{'@3':>8}{'@5':>8}{'@1 obs':>9}")
    res = {}
    for a, h in arms.items():
        res[a] = {f"@{k}": float(h[k].mean()) for k in KS}
        res[a]["@1_obscure"] = float(h[1][ob].mean()) if ob.sum() else None
        print(f"    {a:<20}{h[1].mean():>8.3f}{h[3].mean():>8.3f}{h[5].mean():>8.3f}"
              f"{(h[1][ob].mean() if ob.sum() else float('nan')):>9.3f}")

    # does declared confidence separate right from wrong? this licenses SELECTIVE re-ranking
    conf = np.array([float(ans[pid].get("confidence", 0.5)) for pid, _, _, _ in rows])
    right = arms["existing_order"][1]
    print(f"\n  DOES CONFIDENCE KNOW? -- separating right from wrong at rank 1")
    if len(set(conf.tolist())) > 1:
        hi, lo = conf >= np.median(conf), conf < np.median(conf)
        print(f"    declared HIGH confidence  n={int(hi.sum()):>4}  correct {right[hi].mean():.3f}")
        print(f"    declared LOW  confidence  n={int(lo.sum()):>4}  correct {right[lo].mean():.3f}")
        gap = float(right[hi].mean() - right[lo].mean())
        res["confidence_gap"] = gap
        print(f"    gap {gap:+.3f} -- a positive gap means selective re-ranking is possible: apply the")
        print(f"    docking score ONLY to the low-confidence half, where overwriting costs least.")
    print(f"\n  READING")
    b = max((res[a]["@1"], a) for a in arms)
    e = res["existing_order"]["@1"]
    if b[0] > e + 0.02:
        print(f"  {b[1]} improves rank-1 to {b[0]:.3f} from {e:.3f} using no new data -- the ordering was")
        print("  under-served by a single listwise pass, which is the cheapest kind of gain available.")
    else:
        print(f"  NO ARM BEATS THE EXISTING ORDER ({e:.3f}). Ranking is saturated for this model, and the")
        print("  remaining lever is retrieval, which is the one axis measured to matter: 0.25 at 4-7")
        print("  examples against 0.64 at twelve.")
    json.dump({"test": "cell_orphan_rank_v2", "n": len(rows), "arms": res},
              open(OUT / "cell_orphan_rank_v2.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "score"
    raise SystemExit(cmd_dump() if c == "dump" else cmd_score())
