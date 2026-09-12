"""WOULD WE HAVE FOUND IT? Separating "recovers what we already wrote down" from "would have discovered it".

WHAT THE PREVIOUS TEST ACTUALLY SHOWED.  `cell_orphan_candidates` hid a reaction's catalyst and recovered it at
recall@1 0.276 against a 0.019 control. But it hid only the REACTION -- the catalyst itself stayed in the
vocabulary through its OTHER reactions. So what was measured is: *given an enzyme we already know catalyses
things, can we propose it for one more reaction?* That is substrate-scope expansion. It is useful and it is not
discovery, and reporting it as discovery would be the single easiest overstatement available here.

THE STRUCTURAL LIMIT, stated before running because it is arithmetic rather than a finding: the candidate
vocabulary is built from genes already annotated as catalysts. A protein never annotated as catalysing anything
CANNOT be proposed, at any k, by any scoring function over this vocabulary. **1,932 of 5,282 catalysts (37%)
appear in exactly one reaction** -- for those, holding out that reaction removes the gene from the vocabulary
and recall is 0 by construction, not by failure.

So this test asks three sharper questions.

    1  REDUNDANCY   Stratify recall by how many OTHER reactions the true catalyst has. If recall is high only
                    for well-connected enzymes and collapses at 0-1, the method rides on annotation redundancy
                    and fails exactly where discovery would be needed.

    2  TEMPORAL     Reactome stable IDs increase with curation date (range here 68,595 to 9,988,829). Build the
                    index using ONLY reactions below a cutoff and test on those above: given what was curated
                    then, would the method have proposed what was curated later? This is the closest thing to a
                    prospective test available without a wet lab.

    3  OBSCURITY    Stratify by the catalyst's publication count. A method that only works on famous proteins is
                    a method that re-finds textbook biology.

PREDECLARED, before any number:
    recall at 0-other-reactions ~ 0                  -> expected and structural. Report as a LIMIT, never as a
                                                        failure, and never let the headline recall imply
                                                        discovery.
    temporal recall ~ random-holdout recall          -> the method is not merely following curation order; what
                                                        it does for old reactions it does for new ones.
    temporal recall << random-holdout recall         -> the earlier number was inflated by knowing the future.
    recall flat across publication deciles           -> it is not just re-finding famous proteins.
"""
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT, SP = A.OUT, A.SP
KS = (1, 5, 20, 100)
NONCAT = ("binding", "dissociation")
ABSTRACT = ("omitted", "uncertain")
N_EVAL = 2000


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("WOULD WE HAVE FOUND IT?  redundancy, temporal holdout, and obscurity")
    report("=" * 100)
    report("  PREDECLARED: recall at 0-other-reactions ~ 0 is STRUCTURAL, not failure -- the gene leaves the")
    report("  vocabulary. Temporal ~ random => not following curation order. Flat across pubs => not just fame.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    currency = set(str(x) for x in (net.get("currency") or []))
    cell = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    pubs = {g["name"]: (g.get("pubs") or 0) for g in cell["genes"]}

    def parts(s):
        return {str(p.get("id") or p.get("name") or "")
                for side in ("in", "out") for p in (s.get(side) or [])} - currency - {""}

    P = {i: parts(s) for i, s in enumerate(steps)}
    cats = {i: list(s["catalysts"]) for i, s in enumerate(steps) if s.get("catalysts")}
    per_gene = Counter(c for cs in cats.values() for c in cs)
    report(f"\n  {len(cats):,} catalysed reactions | {len(per_gene):,} distinct catalysts")
    report(f"  catalysts appearing in exactly ONE reaction: {sum(1 for v in per_gene.values() if v == 1):,} "
           f"({100*sum(1 for v in per_gene.values() if v==1)/len(per_gene):.0f}%)")

    def build_index(allowed):
        idx = defaultdict(set)
        for i in allowed:
            for p in P[i]:
                idx[p].add(i)
        return idx

    def candidates(i, idx, drop=()):
        sc = defaultdict(float)
        pi = P[i]
        if not pi:
            return []
        seen = set()
        for p in pi:
            for j in idx.get(p, ()):
                if j == i or j in seen:
                    continue
                seen.add(j)
                u = len(pi | P[j])
                jac = len(pi & P[j]) / u if u else 0.0
                for c in cats[j]:
                    if c in drop:
                        continue
                    sc[c] = max(sc[c], jac)
        return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]

    rng = np.random.default_rng(A.SEED)
    full_idx = build_index(list(cats))
    evalable = [i for i in cats if P[i]]
    ev = [evalable[int(x)] for x in rng.choice(len(evalable), min(N_EVAL, len(evalable)), replace=False)]

    # ---------------- 1. REDUNDANCY ----------------
    report("\n  [1] REDUNDANCY -- recall by how many OTHER reactions the true catalyst has")
    buckets = {"0 (gene leaves vocab)": (0, 0), "1": (1, 1), "2-4": (2, 4), "5-19": (5, 19), "20+": (20, 10**9)}
    agg = {b: {k: [0, 0] for k in KS} for b in buckets}
    for i in ev:
        truth = set(cats[i])
        # "other reactions" = the minimum, across the true catalysts, of how many OTHER reactions each appears in
        others = min(per_gene[c] - 1 for c in truth)
        b = next(b for b, (lo, hi) in buckets.items() if lo <= others <= hi)
        ch = candidates(i, full_idx)
        for k in KS:
            agg[b][k][0] += bool(truth & set(ch[:k]))
            agg[b][k][1] += 1
    report(f"    {'bucket':<24}{'n':>7}" + "".join(f"{'r@'+str(k):>10}" for k in KS))
    red = {}
    for b in buckets:
        n = agg[b][KS[0]][1]
        if not n:
            continue
        red[b] = {str(k): agg[b][k][0] / n for k in KS}
        report(f"    {b:<24}{n:>7,}" + "".join(f"{agg[b][k][0]/n:>10.3f}" for k in KS))

    # ---------------- 2. TEMPORAL ----------------
    report("\n  [2] TEMPORAL -- index built only on EARLIER-curated reactions, tested on later ones")
    num = {}
    for i, s in enumerate(steps):
        m = re.match(r"R-HSA-(\d+)$", str(s.get("id") or ""))
        if m:
            num[i] = int(m.group(1))
    dated = sorted([i for i in cats if i in num and P[i]], key=lambda i: num[i])
    if len(dated) < 400:
        report("    too few dated reactions -- skipped")
        temporal = None
    else:
        cut = int(0.75 * len(dated))
        old, new = dated[:cut], dated[cut:]
        report(f"    {len(old):,} earlier reactions form the index; {len(new):,} later ones are the test set")
        report(f"    cutoff R-HSA-{num[dated[cut]]:,}")
        old_idx = build_index(old)
        old_vocab = {c for i in old for c in cats[i]}
        te = [new[int(x)] for x in rng.choice(len(new), min(800, len(new)), replace=False)]
        # matched RANDOM holdout: same number of reactions removed, chosen at random rather than by date
        rnd_hold = set(int(x) for x in rng.choice(len(dated), len(new), replace=False))
        rnd_old = [dated[j] for j in range(len(dated)) if j not in rnd_hold]
        rnd_idx = build_index(rnd_old)
        rnd_te = [dated[j] for j in list(rnd_hold)[:len(te)]]
        th = {k: 0 for k in KS}
        rh = {k: 0 for k in KS}
        inv = 0
        for i in te:
            truth = set(cats[i])
            inv += bool(truth & old_vocab)
            ch = candidates(i, old_idx)
            for k in KS:
                th[k] += bool(truth & set(ch[:k]))
        for i in rnd_te:
            truth = set(cats[i])
            ch = candidates(i, rnd_idx)
            for k in KS:
                rh[k] += bool(truth & set(ch[:k]))
        report(f"    of the later reactions, {inv}/{len(te)} ({100*inv/len(te):.0f}%) have a catalyst that was")
        report(f"    ALREADY in the earlier vocabulary -- the rest cannot be found at any k, by construction")
        report(f"    {'arm':<22}" + "".join(f"{'r@'+str(k):>10}" for k in KS))
        report(f"    {'temporal (later)':<22}" + "".join(f"{th[k]/len(te):>10.3f}" for k in KS))
        report(f"    {'random holdout':<22}" + "".join(f"{rh[k]/len(rnd_te):>10.3f}" for k in KS))
        temporal = {"n_test": len(te), "in_old_vocab": inv,
                    "temporal": {str(k): th[k] / len(te) for k in KS},
                    "random": {str(k): rh[k] / len(rnd_te) for k in KS}}

    # ---------------- 3. OBSCURITY ----------------
    report("\n  [3] OBSCURITY -- recall by the true catalyst's publication count")
    pv = sorted({pubs.get(c, 0) for cs in cats.values() for c in cs})
    qs = np.percentile(pv, [25, 50, 75]) if pv else [0, 0, 0]
    ob = {f"<={int(qs[0])}": [0, 0], f"{int(qs[0])+1}-{int(qs[1])}": [0, 0],
          f"{int(qs[1])+1}-{int(qs[2])}": [0, 0], f">{int(qs[2])}": [0, 0]}
    keys = list(ob)
    for i in ev:
        truth = set(cats[i])
        pmin = min(pubs.get(c, 0) for c in truth)
        b = keys[0] if pmin <= qs[0] else keys[1] if pmin <= qs[1] else keys[2] if pmin <= qs[2] else keys[3]
        ch = candidates(i, full_idx)
        ob[b][0] += bool(truth & set(ch[:20]))
        ob[b][1] += 1
    report(f"    {'publications':<20}{'n':>7}{'r@20':>10}")
    obs = {}
    for b in keys:
        h, n = ob[b]
        if n:
            obs[b] = h / n
            report(f"    {b:<20}{n:>7,}{h/n:>10.3f}")

    report("\n  READING")
    z = red.get("0 (gene leaves vocab)", {}).get("100", None)
    if z is not None:
        report(f"  Recall for catalysts with NO other reaction is {z:.3f} at k=100. This is the structural limit,")
        report("  not a failure: the gene is not in the vocabulary, so no score over that vocabulary can name it.")
        report("  37% of all catalysts sit in that position. The method expands the scope of enzymes we already")
        report("  know; it does not nominate a protein that has never been annotated as catalysing anything.")
    if temporal:
        d = temporal["temporal"]["20"] - temporal["random"]["20"]
        if abs(d) < 0.05:
            report(f"  Temporal and random holdout agree at k=20 ({d:+.3f}): the method is not riding on curation")
            report("  order -- what it does for old reactions it does for later-curated ones.")
        elif d < 0:
            report(f"  Temporal holdout is {-d:.3f} WORSE than random at k=20. The headline number benefited from")
            report("  knowing the future; a prospective run would do worse than the retrospective one suggests.")
        else:
            report(f"  Temporal holdout BEATS random by {d:.3f}, which no hypothesis here predicted and needs")
            report("  explaining before it is used.")

    json.dump({"test": "cell_would_we_have_found_it", "ks": list(KS),
               "n_catalysts": len(per_gene),
               "singleton_catalysts": sum(1 for v in per_gene.values() if v == 1),
               "redundancy": red, "temporal": temporal, "obscurity": obs, "log": log},
              open(OUT / "cell_would_we_have_found_it.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'cell_would_we_have_found_it.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
