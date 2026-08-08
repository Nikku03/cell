"""HOW MANY OF THE 1,400 ARE RIGHT, AND CAN WE NAME WHICH ONES?

THE PROBLEM WITH THE QUESTION.  The 1,400 filled rows are orphan reactions -- they have no annotated
catalyst and never did, so there is no answer key and no row can be marked correct by looking it up. Any
honest answer is therefore an ESTIMATE with an interval, plus a statement of whether the estimate can be
pushed down to individual rows or only holds in aggregate. Those are different deliverables and only the
second one is useful to a curator, who works row by row.

Three things are computed, in increasing order of what they buy.

1. TRANSPORT.  Accuracy was measured on 333 held-out reactions under the SAME single-answer protocol that
   produced the fill (colab/cell_orphan_holdout.py, top-1 0.727 overall / 0.515 obscure). Held-out is not
   the fill: it was stratified to be 29.1% obscure by design, and the fill is whatever it is. So the rate
   is re-weighted onto the fill's own composition by direct standardisation, with a bootstrap interval.

   THE STRATIFIER IS THE TRICK.  Obscurity is defined by the literature count of the TRUE catalyst, which
   the fill does not have. But the PREDICTED gene's literature count is observable on both sides, and on
   held-out it turns out to be a near-perfect one-way indicator:

       model predicted an obscure gene  ->  the truth was obscure in 51 of 51 cases
       model predicted a famous gene    ->  the truth was obscure in 46 of 282

   which makes predicted fame a legitimate stratifier that exists on every fill row.

2. SEPARABILITY -- can we name WHICH rows?  The confidence signal is the largest effect ever measured in
   this line (0.891 correct when the model is confident against 0.456 when not, a spread of +0.435). The
   fill answers DO NOT CARRY IT. They have two fields, gene and why, and no confidence was ever requested.
   So the question becomes whether the flags that DO exist on every row -- example count, copied-from-an-
   example, compartment agreement -- separate right from wrong well enough to substitute. Measured as AUC
   on held-out, against the confidence signal that is missing, so the cost of not having asked is a number.

3. INDEPENDENT AGREEMENT, and this is the part that can actually label rows.  A second solver answers the
   same reaction CLOSED-BOOK -- reaction text only, no retrieved examples, no sight of the first answer.
   Agreement between two solvers on different information is evidence; how much evidence is measured
   rather than assumed, by running the identical procedure on the 333 held-out reactions where truth IS
   known and reading off

       P(correct | the two agree)      and      P(correct | they disagree)

   Then the same solver runs on a stratified sample of the fill and the calibration is applied. This is
   the only route to a per-row label, because it is the only signal here that is generated fresh for each
   row rather than inherited from a benchmark average.

PREDECLARED, before any number:
    P(correct | agree) - P(correct | disagree) > 0.25 on the calibration set
        -> agreement is a usable per-row label, the fill can be split into a trustworthy part and a
           doubtful part, and the count of "rows we are sure about" is the agreeing subset times its
           calibrated rate.
    the gap is small, or agreement is near-universal or near-zero
        -> the second solver is not independent enough (it agrees with everything) or not strong enough
           (it agrees with nothing), no per-row label is available, and the honest answer to "how many are
           right" stays an aggregate estimate with no way to say which.
    separability of the existing flags is near chance (AUC < 0.6)
        -> the fill as it stands cannot be triaged from its own metadata, and the missing confidence field
           is a concrete, cheap defect to fix rather than a philosophical limit.

usage:  python cell_orphan_howmany.py est      transport + separability, no agents needed
        python cell_orphan_howmany.py prep     write the closed-book item lists for the workflow
        python cell_orphan_howmany.py score    calibrate agreement and apply it to the fill

-> outputs/orphan/cell_orphan_howmany.json
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
import cell_orphan_obscurity as O  # noqa: E402
from cell_orphan_incontext import _render  # noqa: E402
from cell_orphan_fill import K  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/howmany")
SEED = 4127
NSAMPLE = int(os.environ.get("HOWMANY_N", 250))
BOOT = 2000
MINCELL = 12          # below this a stratum falls back to its fame-bin marginal rather than fitting noise


def ecell(n):
    return "0" if n == 0 else ("1-3" if n <= 3 else ("4-7" if n <= 7 else ("8-11" if n <= 11 else "12")))


def load_holdout(papers):
    """The 333 held-out reactions with truth, the fill-protocol answer, and every observable a fill row
    also carries. This is the calibration population for everything below."""
    hm = json.load(open(OUT / "holdout_meta.json"))
    ha = json.load(open(OUT / "holdout_answers.json"))
    rows = []
    for p, m in hm.items():
        a = ha.get(p)
        if not a:
            continue
        g = str(a.get("gene", "")).strip().upper()
        pv = papers(g)
        rows.append({"pid": p, "gene": g, "truth": {x.upper() for x in m["truth"]},
                     "hit": g in {x.upper() for x in m["truth"]},
                     "n_examples": m["n_examples"], "cell": ecell(m["n_examples"]),
                     "truth_obscure": bool(m["obscure"]),
                     "pred_papers": pv, "pred_obscure": bool(pv is not None and O._binof(pv) == 0),
                     "step": int(m["step"])})
    return rows


def load_fill(papers):
    res = json.load(open(OUT / "cell_orphan_fill.json"))
    rows = []
    for r in res["rows"]:
        g = str(r["gene"]).strip().upper()
        pv = papers(g)
        rows.append({"pid": r["pid"], "gene": g, "step": r["step"],
                     "n_examples": r["n_examples"], "cell": ecell(r["n_examples"]),
                     "pred_papers": pv, "pred_obscure": bool(pv is not None and O._binof(pv) == 0),
                     "copied": bool(r["copied"]), "compartment_ok": bool(r["compartment_ok"]),
                     "is_gene": bool(r["is_gene"]), "giveaway": bool(r["giveaway"]),
                     "association": bool(r["association"]), "top_sim": float(r["top_sim"])})
    return rows


def auc(score, label):
    """Rank AUC with ties at 0.5, written out because the separability question is entirely about whether
    a flag with three values can do the job of a continuous confidence."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(s), float)
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


def standardise(cal, fill, key, rng, boot=BOOT):
    """Direct standardisation: the held-out rate inside each stratum, re-weighted by the FILL's share of
    that stratum. The bootstrap resamples the calibration reactions (the fill weights are known exactly,
    it is the rates that are uncertain)."""
    wt = Counter(key(r) for r in fill)
    tot = sum(wt.values())
    idx = defaultdict(list)
    for i, r in enumerate(cal):
        idx[key(r)].append(i)
    hit = np.array([r["hit"] for r in cal], bool)
    fam = defaultdict(list)                       # fallback pool: the fame-bin marginal
    for i, r in enumerate(cal):
        fam[r["pred_obscure"]].append(i)

    def estimate(h, ix, fm):
        """One standardised rate. h is the hit vector, ix the stratum->indices map, fm the fallback map.
        A stratum thinner than MINCELL uses its fame-bin marginal instead of fitting a handful of rows --
        the fill has cells the benchmark barely sampled, and an n=3 cell would otherwise swing the total."""
        num, used, fell = 0.0, 0, []
        for k, w in wt.items():
            ii = ix.get(k, [])
            if len(ii) < MINCELL:
                ii = fm.get(k[0] if isinstance(k, tuple) else k, [])
                fell.append(k)
            if not ii:
                continue
            num += w * h[ii].mean()
            used += w
        return (num / used if used else float("nan")), used, fell

    pt, used, fell = estimate(hit, idx, fam)
    bs = []
    for _ in range(boot):
        take = rng.integers(0, len(cal), len(cal))
        sub = [cal[i] for i in take]
        si, sf = defaultdict(list), defaultdict(list)
        for j, r in enumerate(sub):
            si[key(r)].append(j)
            sf[r["pred_obscure"]].append(j)
        v, u, _f = estimate(np.array([r["hit"] for r in sub], bool), si, sf)
        if u:
            bs.append(v)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan")))
    return pt, float(lo), float(hi), tot, used, fell


def cmd_est():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    papers = O._lit()
    cal = load_holdout(papers)
    fill = load_fill(papers)
    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("HOW MANY OF THE 1,400 ARE RIGHT -- and can we say which")
    report("=" * 100)
    report(f"  calibration: {len(cal)} held-out reactions, same single-answer protocol, truth known")
    report(f"  target:      {len(fill)} filled orphan rows, no truth and no answer key")

    # ---- 1. is predicted fame a legitimate stand-in for the missing obscurity label? -----------------
    report("\n  1. THE STRATIFIER. Obscurity is defined on the TRUE catalyst, which the fill lacks. The")
    report("     PREDICTED gene's literature count exists on every row -- does it track the truth's?")
    po = np.array([r["pred_obscure"] for r in cal])
    to = np.array([r["truth_obscure"] for r in cal])
    report(f"    {'model predicted':<26}{'n':>6}{'truth obscure':>15}")
    for lab, sel in (("an obscure gene", po), ("a famous gene", ~po)):
        report(f"    {lab:<26}{sel.sum():>6}{to[sel].mean():>15.1%}")
    report(f"    held-out truth-obscure share {to.mean():.1%} (stratified by design)")
    fpo = np.mean([r["pred_obscure"] for r in fill])
    p_ob_given_pred_ob = to[po].mean() if po.any() else float("nan")
    p_ob_given_pred_fam = to[~po].mean() if (~po).any() else float("nan")
    fill_ob = fpo * p_ob_given_pred_ob + (1 - fpo) * p_ob_given_pred_fam
    report(f"\n    the fill predicts an obscure gene on {fpo:.1%} of rows, so its implied truth-obscure")
    report(f"    share is {fill_ob:.1%} -- against {to.mean():.1%} in the benchmark. The benchmark is")
    report(f"    HARDER than the fill on this axis, which is the opposite of what was assumed when the")
    report(f"    obscure figure was quoted as the governing number for the whole worklist.")

    # ---- 2. transport ---------------------------------------------------------------------------------
    report("\n  2. TRANSPORT. Held-out rate re-weighted onto the fill's own composition.")
    raw = float(np.mean([r["hit"] for r in cal]))
    report(f"    held-out top-1 as measured                       {raw:.3f}")
    for lab, key in (("by predicted fame", lambda r: (r["pred_obscure"],)),
                     ("by predicted fame x example count", lambda r: (r["pred_obscure"], r["cell"]))):
        pt, lo, hi, tot, used, fell = standardise(cal, fill, key, rng)
        report(f"    standardised to the fill, {lab:<34}{pt:.3f}  [{lo:.3f}, {hi:.3f}]")
        if fell:
            report(f"      ({len(set(fell))} strata below n={MINCELL} fell back to the fame-bin marginal)")
    pt, lo, hi, _, _, _ = standardise(cal, fill, lambda r: (r["pred_obscure"], r["cell"]), rng)

    # ---- 3. the rows that are wrong or trivial by construction ---------------------------------------
    nass = sum(r["association"] for r in fill)
    nbad = sum(not r["is_gene"] for r in fill)
    ngive = sum(r["giveaway"] for r in fill)
    net = len(fill) - nass - nbad
    report("\n  3. ROWS THAT DO NOT NEED AN ESTIMATE AT ALL")
    report(f"    association steps, no catalyst is due      {nass:>5}  wrong by construction")
    report(f"    not a real gene symbol                     {nbad:>5}  wrong by construction")
    report(f"    gene NAMED in the reaction text            {ngive:>5}  right, but read off a label")
    report(f"    leaves {net} rows where the estimate applies")

    # ---- 4. separability: can we name WHICH? ----------------------------------------------------------
    report("\n  4. CAN WE NAME WHICH ONES? The confidence signal that separates best (0.891 against 0.456)")
    report("     WAS NEVER COLLECTED for the fill -- its answers carry gene and why, nothing else. So the")
    report("     question is whether the flags that do exist can substitute.")
    hit = np.array([r["hit"] for r in cal], bool)
    report(f"    {'observable on every fill row':<40}{'AUC':>8}")
    for lab, sc in (("example count", [r["n_examples"] for r in cal]),
                    ("predicted-gene literature count", [(r["pred_papers"] or 0) for r in cal]),
                    ("full retrieval (>=12 examples)", [float(r["n_examples"] >= K) for r in cal])):
        report(f"    {lab:<40}{auc(sc, hit):>8.3f}")
    report("    a coin is 0.500. These are what a curator would have to triage on today.")

    # ---- 5. what self-review was worth, now that there is a truth-based number to compare it to -------
    report("\n  5. THE HAND CHECK, RE-READ AGAINST GROUND TRUTH. 30 rows were verified by hand early on --")
    report("     10 full-retrieval, 10 partial, 10 zero-example, which is within 4 points of the fill's own")
    report("     30/37/33 split -- and scored 26/30. It was flagged at the time as NOT independent, because")
    report("     the reviewer was the model that made the predictions. Held-out gives the size of that.")
    tier = lambda n: "zero" if n == 0 else ("full" if n >= K else "partial")
    hand = {"full": (10, 10), "partial": (8, 10), "zero": (8, 10)}
    report(f"    {'tier':<10}{'hand check':>13}{'measured':>11}{'n':>6}{'inflation':>11}")
    infl = {}
    for t in ("full", "partial", "zero"):
        h = [r["hit"] for r in cal if tier(r["n_examples"]) == t]
        a, b = hand[t]
        infl[t] = a / b - float(np.mean(h))
        report(f"    {t:<10}{a}/{b} = {a/b:>5.0%}{np.mean(h):>11.3f}{len(h):>6}{infl[t]:>+11.3f}")
    ha_ = sum(a for a, _ in hand.values()) / 30
    infl["all"] = ha_ - raw
    report(f"    {'ALL':<10}{'26/30 = 87%':>13}{raw:>11.3f}{len(cal):>6}{infl['all']:>+11.3f}")
    report("    Self-review inflates by +0.14 overall, and the inflation is ORDERED: largest where the")
    report("    examples were in front of both the predictor and the reviewer (+0.21 at full retrieval),")
    report("    almost nil where there was nothing to agree with (+0.04 at zero examples). A second look")
    report("    at shared evidence mostly re-endorses the first look; a second look at bare chemistry does")
    report("    not. That is the mechanism, and it is why the closed-book arm withholds the examples.")

    res = {"test": "cell_orphan_howmany", "n_cal": len(cal), "n_fill": len(fill),
           "hand_check": {"score": "26/30", "rate": ha_, "inflation_vs_measured": infl},
           "held_out_raw": raw, "standardised": pt, "ci": [lo, hi],
           "fill_implied_obscure": float(fill_ob), "held_out_obscure": float(to.mean()),
           "p_truth_obscure_given_pred_obscure": float(p_ob_given_pred_ob),
           "p_truth_obscure_given_pred_famous": float(p_ob_given_pred_fam),
           "association": int(nass), "not_gene": int(nbad), "giveaway": int(ngive), "net": int(net),
           "expected_correct": [float(lo * net), float(pt * net), float(hi * net)],
           "auc": {"example_count": auc([r["n_examples"] for r in cal], hit),
                   "pred_lit": auc([(r["pred_papers"] or 0) for r in cal], hit)}, "log": log}
    report("\n  READING")
    report(f"  Point estimate {pt:.3f} on {net} scorable rows = about {pt*net:.0f} correct, interval")
    report(f"  [{lo*net:.0f}, {hi*net:.0f}]. That is an AGGREGATE. Nothing above tells a curator WHICH")
    report(f"  {pt*net:.0f}, because the one signal that separates well was never asked for. Part 3 of this")
    report("  module -- an independent closed-book second opinion, calibrated on the same held-out set --")
    report("  is the route to a per-row label. Run prep, then the agents, then score.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "cell_orphan_howmany.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_howmany.json'}")
    return 0


def cmd_prep():
    """Write the closed-book items. NO retrieved examples and NO sight of the first answer -- if the second
    solver saw either, agreement would measure shared input rather than independent arrival."""
    papers = O._lit()
    B = T.build()
    steps = B["steps"]
    cal = load_holdout(papers)
    fill = load_fill(papers)
    rng = np.random.default_rng(SEED)
    # stratified sample of the fill, proportional to example cell so the agreement rate transports
    by = defaultdict(list)
    for r in fill:
        by[r["cell"]].append(r)
    take = []
    for c, rs in sorted(by.items()):
        k = max(1, round(NSAMPLE * len(rs) / len(fill)))
        idx = rng.choice(len(rs), size=min(k, len(rs)), replace=False)
        take += [rs[i] for i in idx]
    items = ([{"key": f"cal:{r['pid']}", "text": _render(steps[r["step"]], steps)} for r in cal]
             + [{"key": f"fill:{r['pid']}", "text": _render(steps[r["step"]], steps)} for r in take])
    SP.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(SP / "items.json", "w"), indent=1)
    json.dump({"cal": [r["pid"] for r in cal], "fill": [r["pid"] for r in take]},
              open(SP / "split.json", "w"), indent=1)
    print(f"  {len(cal)} calibration + {len(take)} fill items -> {SP/'items.json'}")
    print(f"  fill sample by example cell: {dict(Counter(r['cell'] for r in take))}")
    return 0


def cmd_score():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    f = SP / "second.json"
    if not f.exists():
        print(f"  no second-opinion answers at {f}; run prep and the agents first")
        return 1
    second = json.load(open(f))
    papers = O._lit()
    cal = {r["pid"]: r for r in load_holdout(papers)}
    fill = {str(r["pid"]): r for r in load_fill(papers)}
    report("=" * 100)
    report("INDEPENDENT AGREEMENT -- what a second closed-book solver buys, measured not assumed")
    report("=" * 100)

    ca, fa = [], []
    for k, g in second.items():
        g = str(g).strip().upper()
        if not g:
            continue
        kind, pid = k.split(":", 1)
        if kind == "cal" and pid in cal:
            r = cal[pid]
            ca.append({"agree": g == r["gene"], "hit": r["hit"], "second_hit": g in r["truth"],
                       "pred_obscure": r["pred_obscure"], "cell": r["cell"]})
        elif kind == "fill" and pid in fill:
            r = fill[pid]
            fa.append({"pid": pid, "agree": g == r["gene"], "second": g, "first": r["gene"],
                       "pred_obscure": r["pred_obscure"], "cell": r["cell"],
                       "association": r["association"], "is_gene": r["is_gene"]})
    ag = np.array([r["agree"] for r in ca], bool)
    hi_ = np.array([r["hit"] for r in ca], bool)
    sh = np.array([r["second_hit"] for r in ca], bool)
    report(f"\n  CALIBRATION on {len(ca)} held-out reactions where truth is known")
    report(f"    the two solvers agree on              {ag.mean():.1%}")
    report(f"    first solver (with examples) correct  {hi_.mean():.3f}")
    report(f"    second solver (closed-book) correct   {sh.mean():.3f}")
    pa = float(hi_[ag].mean()) if ag.any() else float("nan")
    pd_ = float(hi_[~ag].mean()) if (~ag).any() else float("nan")
    report(f"\n    P(first answer correct | the two AGREE)      {pa:.3f}   (n={int(ag.sum())})")
    report(f"    P(first answer correct | they DISAGREE)     {pd_:.3f}   (n={int((~ag).sum())})")
    gap = pa - pd_
    report(f"    gap {gap:+.3f}")
    report(f"    for reference, the confidence signal that was never collected: 0.891 against 0.456 (+0.435)")

    agf = np.array([r["agree"] for r in fa], bool)
    report(f"\n  APPLIED to {len(fa)} sampled fill rows")
    report(f"    the two solvers agree on {agf.mean():.1%} of them")
    exp = agf.mean() * pa + (1 - agf.mean()) * pd_
    report(f"    implied accuracy on the sample {exp:.3f}")
    n_fill = len(fill)
    report(f"    scaled to all {n_fill} rows: about {agf.mean()*n_fill:.0f} agreeing rows at {pa:.3f},")
    report(f"    {(1-agf.mean())*n_fill:.0f} disagreeing at {pd_:.3f}")

    report("\n  READING")
    if gap > 0.25 and 0.05 < ag.mean() < 0.95:
        report(f"  AGREEMENT IS A USABLE PER-ROW LABEL. It separates by {gap:+.3f}, which is the same order")
        report(f"  as the confidence signal that was never collected. The {agf.mean()*n_fill:.0f} agreeing")
        report(f"  rows are the defensible part of the worklist at {pa:.3f}; the rest is at {pd_:.3f} and")
        report("  should be treated as unreviewed. This is a label, not an average, so a curator can act on it.")
    elif ag.mean() > 0.95 or ag.mean() < 0.05:
        report(f"  THE SECOND SOLVER IS NOT INDEPENDENT ENOUGH ({ag.mean():.1%} agreement). It either")
        report("  reproduces the first answer or never reaches it, and either way agreement carries no")
        report("  information about correctness.")
    else:
        report(f"  AGREEMENT DOES NOT LABEL ROWS: {pa:.3f} against {pd_:.3f}, a gap of {gap:+.3f}. The")
        report("  estimate stays aggregate, and the practical conclusion is that the fill should be re-run")
        report("  asking for confidence, which is the one signal measured to separate well here.")

    res = json.load(open(OUT / "cell_orphan_howmany.json")) if (OUT / "cell_orphan_howmany.json").exists() else {}
    res["agreement"] = {"n_cal": len(ca), "n_fill": len(fa), "agree_cal": float(ag.mean()),
                        "agree_fill": float(agf.mean()), "p_correct_given_agree": pa,
                        "p_correct_given_disagree": pd_, "gap": float(gap),
                        "second_solver_alone": float(sh.mean()), "implied_fill_accuracy": float(exp),
                        "rows": fa, "log": log}
    json.dump(res, open(OUT / "cell_orphan_howmany.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_howmany.json'}")
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "est"
    raise SystemExit({"est": cmd_est, "prep": cmd_prep, "score": cmd_score}[c]())
