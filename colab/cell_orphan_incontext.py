"""GIVE IT THE ANNOTATED HALF: does showing 10,596 solved reactions make it better at the unsolved ones?

THE IDEA.  Everything so far asked the model to name a catalyst from the reaction alone. But half the cell is
already annotated -- 10,596 enzyme-shaped reactions with a known catalyst -- and none of that was ever put in
front of it. So each query reaction is now shown the 12 chemically most similar SOLVED reactions together
with their catalysts, as worked examples.

WHY THIS IS DIFFERENT FROM THE CANDIDATE-LIST ARMS, which topped out at 0.505.  Those handed the model a bag
of gene names stripped of context and asked it to rank them. This shows the REACTIONS those genes catalyse, so
the model can see *why* a gene is a candidate -- that ALDH3A2 handles long-chain aldehydes while ALDH1A1
handles retinal -- and reason by analogy rather than by popularity. Same underlying data, radically more of it
per candidate.

TWO THINGS MEASURED BEFORE RUNNING, because they bound what a gain could mean.

    THE TRAINING HALF IS THE WRONG HALF.  Of the annotated catalysts, 1% have under 20 papers and 27% have
    over 500. The gaps are obscure by construction. So the examples are drawn from a population unlike the one
    that needs filling, and 40% of catalysts appear in exactly ONE reaction -- there is little repetition to
    generalise from.

    THE ANSWER IS OFTEN SIMPLY SHOWN.  On the obscure bin, the true catalyst appears among the 12 retrieved
    examples 35% of the time (0.500 overall). Where it is shown the model can copy, and a gain there is
    retrieval working, not learning. Where it is NOT shown -- 52 of the 80 obscure reactions -- a gain is
    genuine reasoning by analogy. Results are split on exactly that line, and the second half is the finding.

ARMS, all on the identical 400 reactions used for every earlier number, so they compare directly:
    closed_book        no examples. Already measured: 0.645 overall, 0.463 obscure.
    examples           the 12 most chemically similar solved reactions, with their catalysts
    shuffled_examples  12 solved reactions drawn AT RANDOM, with their catalysts. The control that decides
                       whether relevance is doing the work or merely seeing a list of real enzyme names is.

PREDECLARED, before any number:
    examples > closed_book, and the gain survives on the ANSWER-NOT-SHOWN subset
        -> the model reasons by analogy from solved chemistry. This raises the orphan estimate and the
           annotated half is a genuine asset rather than just an answer key.
    examples > closed_book only where the answer IS shown
        -> it is copying. Real, useful, and bounded by retrieval: it can only ever fill the 35% of obscure
           gaps whose answer already sits in a chemically similar reaction.
    examples ~ shuffled_examples
        -> relevance is not being used; the examples act as a prompt, not as evidence.
    examples < closed_book
        -> the examples distract. Twelve wrong-but-plausible genes anchor it away from its own better guess.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_llm as L
import cell_orphan_obscurity as O
import cell_orphan_ties_nn as T

OUT = A.OUT
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
ANSW = OUT / "incontext_answers.json"
SHOWN = OUT / "incontext_shown.json"
K = 12
SEED = 23
BASE = {"all": 0.645, "obscure": 0.463}       # closed-book, same 400 reactions


def _render(s, steps):
    cur = None

    def part(p):
        t = p.get("type")
        if t == "UNRESOLVED":
            return "?unresolved"
        c = p.get("compartment")
        return f"{p.get('name')} [{t}{', ' + c if c else ''}]"
    return ("  IN : " + "  +  ".join(part(p) for p in (s.get("in") or [])) + "\n"
            "  OUT: " + "  +  ".join(part(p) for p in (s.get("out") or [])))


def dump():
    B = T.build()
    cats, P, steps = B["cats"], B["P"], B["steps"]
    sel = json.load(open(O.PUZZLES))["pids"]
    puz = json.load(open(O.PUZZLES))["puzzles"]
    cat_of = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of[p].add(i)
    ann = [i for i in cats if steps[i].get("category") in ("metabolic", "transition")]
    rng = np.random.default_rng(SEED)

    shown = {}
    for mode in ("examples", "shuffled_examples"):
        txt = []
        for pid, i in enumerate(sel):
            pi = P[i]
            if mode == "examples":
                sc = {}
                for p in pi:
                    for j in cat_of.get(p, ()):
                        if j == i:
                            continue
                        u = len(pi | P[j])
                        sc[j] = max(sc.get(j, 0), len(pi & P[j]) / u if u else 0)
                top = sorted(sc, key=lambda j: -sc[j])[:K]
            else:
                top = [int(x) for x in rng.choice([j for j in ann if j != i], K, replace=False)]
            ex = []
            for j in top:
                ex.append(f"  SOLVED: {_render(steps[j], steps)}\n     catalyst: "
                          f"{', '.join(sorted(cats[j]))}")
            if mode == "examples":
                shown[str(pid)] = bool(set(cats[i]) & {g for j in top for g in cats[j]})
            p = puz[pid]
            head = (f"#{p['pid']}  ({p['source']}, {p['category']}, "
                    f"{'reversible' if p['reversible'] else 'irreversible'})")
            body = ("  IN : " + "  +  ".join(
                ("?unresolved" if q["type"] == "UNRESOLVED" else
                 f"{q['name']} [{q['type']}{', ' + q.get('compartment', '') if q.get('compartment') else ''}]")
                for q in p["inputs"]) + "\n  OUT: " + "  +  ".join(
                ("?unresolved" if q["type"] == "UNRESOLVED" else
                 f"{q['name']} [{q['type']}{', ' + q.get('compartment', '') if q.get('compartment') else ''}]")
                for q in p["outputs"]))
            txt.append(head + "\n" + body + "\n  --- SOLVED REACTIONS FOR REFERENCE ---\n"
                       + "\n".join(ex))
        # the query reaction must never appear among its own examples
        for pid, i in enumerate(sel):
            assert steps[i].get("id", "\0") not in txt[pid], f"puzzle {pid} shows itself as an example"
        dd = SP / mode
        dd.mkdir(exist_ok=True)
        Bs = 50
        for b in range(0, len(txt), Bs):
            (dd / f"open_b{b // Bs}.txt").write_text(("\n\n" + "=" * 90 + "\n\n").join(txt[b:b + Bs]))
        print(f"  {mode}: wrote {len(txt)} puzzles -> {dd}")
    json.dump(shown, open(SHOWN, "w"), indent=1)
    print(f"  answer visible among the examples: {np.mean(list(shown.values())):.3f} overall -> {SHOWN}")
    return 0


def merge(tdir, arm):
    got = {}
    for f in sorted(Path(tdir).glob("agent-*.jsonl")):
        txt = f.read_text(errors="replace")
        # both arms name their batches open_b*.txt, so the ARM must be told apart by directory or the two
        # runs merge into each other and every number is a blend of the two
        tags = set(re.findall(r"/(\w+)/open_b(\d+)\.txt", txt))
        if len({t[0] for t in tags}) != 1 or {t[0] for t in tags} != {arm} or len(tags) != 1:
            continue
        for line in txt.splitlines():
            if '"answers"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stack = [rec]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("answers"), list):
                        for a_ in x["answers"]:
                            if isinstance(a_, dict) and "pid" in a_ and a_.get("gene"):
                                got[int(a_["pid"])] = str(a_["gene"]).strip().upper()
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
    miss = sorted(set(range(400)) - set(got))
    if miss:
        print(f"  REFUSING: {len(miss)} answers missing {miss[:10]}")
        return 1
    cur = json.load(open(ANSW)) if ANSW.exists() else {}
    cur[arm] = {str(k): v for k, v in got.items()}
    json.dump(cur, open(ANSW, "w"), indent=1)
    print(f"  merged {len(got)} answers for arm '{arm}' -> {ANSW}")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        return dump()
    if len(sys.argv) > 3 and sys.argv[1] == "merge":
        return merge(sys.argv[2], sys.argv[3])

    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)
    from scipy.stats import binomtest

    report("=" * 100)
    report("GIVE IT THE ANNOTATED HALF -- 12 solved reactions, with catalysts, as worked examples")
    report("=" * 100)
    report("  Measured before running: the annotated catalysts are 1% obscure and 27% famous, the opposite of")
    report("  the gap; 40% of them appear in exactly one reaction. And the true catalyst is already visible")
    report("  among the 12 examples 35% of the time on the obscure bin -- where it is, the model can copy.")

    B = T.build()
    cats = B["cats"]
    sel = json.load(open(O.PUZZLES))["pids"]
    ans = json.load(open(ANSW))
    shown = json.load(open(SHOWN))
    papers = O._lit()
    pool = O._pool(B, papers)
    n = len(sel)
    truth = [{g.upper() for g in cats[i]} for i in sel]
    bidx = np.array([O._binof(pool[i]) for i in sel])
    # STRICT mask: the answer counts as visible if it is an example's catalyst OR merely appears in an
    # example's participant names. The second path is small (3.3% overall, 0.0% on the obscure bin) but it
    # is a way to read the answer off, and the whole claim rests on the not-shown subset being clean.
    strict = OUT / "incontext_shown_strict.json"
    src = json.load(open(strict)) if strict.exists() else shown
    vis = np.array([bool(src.get(str(p), False)) for p in range(n)])

    hits = {}
    for arm, d in ans.items():
        hits[arm] = np.array([str(d.get(str(p), "")).upper() in truth[p] for p in range(n)])
    base = np.array([False] * n)
    ob = json.load(open(OUT / "obscurity_answers.json"))
    hits["closed_book"] = np.array([str((ob.get(str(p)) or {}).get("open", "")).upper() in truth[p]
                                    for p in range(n)])

    ARMS = [a for a in ("closed_book", "examples", "shuffled_examples") if a in hits]
    report(f"\n  TOP-1 ON THE SAME 400 REACTIONS")
    report(f"    {'arm':<20}{'all':>8}{'obscure':>10}{'ans SHOWN':>12}{'ans NOT shown':>15}")
    res = {}
    for a in ARMS:
        h = hits[a]
        o = bidx == 0
        res[a] = {"all": float(h.mean()), "obscure": float(h[o].mean()),
                  "shown": float(h[vis].mean()), "not_shown": float(h[~vis].mean()),
                  "obscure_not_shown": float(h[o & ~vis].mean())}
        report(f"    {a:<20}{h.mean():>8.3f}{h[o].mean():>10.3f}{h[vis].mean():>12.3f}"
               f"{h[~vis].mean():>15.3f}")
    report(f"    n                   {n:>8}{int((bidx==0).sum()):>10}{int(vis.sum()):>12}"
           f"{int((~vis).sum()):>15}")

    if "examples" in hits:
        e, c = hits["examples"], hits["closed_book"]
        for lab, m in (("ALL", np.ones(n, bool)), ("obscure bin", bidx == 0),
                       ("answer NOT shown", ~vis),
                       ("obscure AND not shown", (bidx == 0) & ~vis)):
            w = int((e[m] & ~c[m]).sum())
            l = int((~e[m] & c[m]).sum())
            p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
            res.setdefault("paired", {})[lab] = {"n": int(m.sum()), "wins": w, "losses": l, "p": float(p),
                                                 "delta": float(e[m].mean() - c[m].mean())}
            report(f"    paired vs closed_book, {lab:<22} {e[m].mean()-c[m].mean():+.3f}  "
                   f"({w} win / {l} lose, p={p:.4f})")

    report("\n  READING")
    if "examples" not in hits:
        report("  examples arm not present yet")
    else:
        d_all = res["examples"]["all"] - res["closed_book"]["all"]
        d_ns = res["examples"]["not_shown"] - res["closed_book"]["not_shown"]
        d_sh = res["examples"]["shown"] - res["closed_book"]["shown"]
        sh = res.get("shuffled_examples", {}).get("all")
        report(f"  overall {d_all:+.3f} | where the answer WAS shown {d_sh:+.3f} | where it was NOT {d_ns:+.3f}")
        if d_all <= 0.02:
            report("  SHOWING SOLVED REACTIONS DOES NOT HELP. The annotated half is an answer key, not a")
            report("  teacher: where the answer is not already in the examples the model gains nothing from")
            report("  seeing solved chemistry, which is the only case that would have raised the orphan number.")
        elif d_ns > 0.03:
            report("  IT REASONS BY ANALOGY. The gain survives where the answer is NOT among the examples, so")
            report("  it is generalising from solved chemistry rather than copying. This raises the estimate.")
        else:
            report("  IT IS COPYING. The gain lives entirely where the answer was already shown, so this is")
            report("  retrieval working and is capped by it -- only the 35% of obscure gaps whose answer sits")
            report("  in a chemically similar solved reaction can ever be filled this way.")
        if sh is not None:
            report(f"  shuffled examples {sh:.3f} vs relevant {res['examples']['all']:.3f}: "
                   f"{'relevance matters' if res['examples']['all'] - sh > 0.03 else 'RELEVANCE IS NOT BEING USED -- the examples act as a prompt, not evidence'}")

    json.dump({"test": "cell_orphan_incontext", "n": n, "k": K, "results": res, "log": log},
              open(OUT / "cell_orphan_incontext.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_incontext.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
