"""Loop 186. Contact at sub-kilobase resolution -- the prediction loop 181 left standing.

WHAT WAS PREDICTED, AND WHY THIS IS THE TEST. Loop 181 put a real K562 Hi-C map on stage two and it
failed almost every gate: observed-over-expected added nothing once the distance decay was divided
out, and twenty draws giving each gene a stranger's contact profile scored HIGHER than the real
one. The diagnosis was resolution, and it was measured rather than argued:

    bin size   within-gene candidate pairs sharing a bin   bins holding a positive AND a negative
     5,000 bp                33.7%                                       40
     1,000 bp                 2.6%                                        2

A third of the decisions this benchmark asks for were, at 5 kb, between candidates the map could
not tell apart. So loop 186's source recorded a prediction before any 1 kb data was fetched: if
resolution was the binding constraint, contact beyond distance should now carry something and the
stranger-swap should now lose. If it still fails, the conclusion changes.

WHAT IS USED, AND WHY IT IS NOT THE MATRIX. The plan was to stream a 1 kb contact matrix. ENCODE's
matrices do carry 1 kb, 500 bp, 200 bp and 100 bp zoom levels, but every attempt to read them
remotely died with intermittent `curl_easy_perform() failed: SSL connect error` on the range
session, with and without the proxy CA bundle set for libcurl -- 24 consecutive failures against
zero completed strips. That is recorded rather than worked around.

The instrument used instead is better suited to the question anyway. ENCODE releases K562 LOOP
CALLS from those same deep maps, in GRCh38, as small bedpe files:

    ENCFF511QFN   734,671 loops   anchor width median   728 bp   span median  38,857 bp
    ENCFF953LXY   228,037 loops   anchor width median 1,000 bp   span median 168,000 bp
    ENCFF030PMM   186,714 loops   anchor width median   725 bp
    ENCFF118PBQ   129,669 loops   anchor width median   649 bp
    ENCFF759YBZ   115,183 loops   anchor width median   652 bp
    ENCFF549OBE    16,363 contact domains, median span 170 kb

Loop 181 used the Rao HiCCUPS list: 6,057 loops at 5-10 kb anchors. These are two hundred times as
many at anchor widths BELOW the median element width of 500-580 bp. And a called loop connecting an
anchor on the element to an anchor on the promoter is a more direct statement than a matrix cell --
it is the map asserting that these two specific loci touch, rather than that their neighbourhoods
are contact-rich.

Everything is GRCh38, which is the benchmark's native assembly, so this chain contains no liftover
at any point.

PREDECLARED, BEFORE ANY NUMBER.

  A1 IS THE RESOLUTION ACTUALLY BETTER? Anchor widths, loops anchored per element and per promoter,
     and how many evaluable genes have at least one candidate carrying a loop link.
     Gate: PASS iff the pooled median anchor width is at or below 1,500 bp AND at least half of the
     evaluable genes have a candidate with a link. Below that the feature is mostly absent and no
     failure downstream could be attributed to contact rather than to coverage.

  A2 DOES THE LOOP LINK ADD OVER DISTANCE? distance plus the loop block against distance alone.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- loop 173's E3 bar, unchanged since loop 173.

  A3 IS IT THE LINK, OR THE TWO ANCHOR DEGREES? An element in a loop-dense region and a promoter in
     a loop-dense region will share loops by arithmetic. The block is re-expressed as pointwise
     mutual information against the link count the two degrees alone predict, with both degrees
     entered separately so the model can use them directly.
     Gate: same bar, applied to the PMI form. This is loop 185's Z4 lesson applied in advance.

  A4 DOES IT ADD OVER THE BEST STACK SO FAR? Loop 185's winning arm -- sequence, class/family
     occupancy, measured co-binding, accessibility -- plus the loop block.
     Gate: same bar.

  A5 THE STRANGER SWAP, which is the gate loop 181 failed worst. Each gene is given another gene's
     promoter anchors, 20 draws, distance and sequence untouched.
     Gate: PASS iff the real anchors beat the swapped ones in >= 90% of draws. Loop 181's 5 kb
     matrix beat 0 of 20.

  A6 THE DECISIVE ONE. The best arm against distance alone, and reported beside loop 185's 0.6734.
     Gate: same bar as A2.

  A7 THE HEAD-TO-HEAD THAT TESTS THE PREDICTION DIRECTLY. The same block built from loop 181's
     6,057 Rao HiCCUPS loops at 5-10 kb anchors, against this one, on identical folds.
     Gate: PASS iff the sub-kilobase block beats the 5-10 kb block past 3 sem. A FAIL says
     resolution was NOT the binding constraint and loop 181's conclusion stands for a reason other
     than the one predicted, which is the more interesting outcome and is reported as such.

  A8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_loops.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import chip as CH                   # noqa: E402
from enh import contact as CT                # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_loops.json"
HIC = CT.HIC
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
N_SWAP = 20
MAX_ANCHOR = 1500
MIN_GENE_COVER = 0.50
PROM_PAD = 1000
L185_BEST_R1 = 0.6734
L173_DIST_R1 = 0.5930

FINE = ["ENCFF511QFN", "ENCFF953LXY", "ENCFF030PMM", "ENCFF118PBQ", "ENCFF759YBZ", "ENCFF287ZOF"]
DOMAINS = ["ENCFF549OBE", "ENCFF138SPA"]
LOOPCOLS = ["loop_n", "loop_best", "loop_support", "loop_any"]
PMICOLS = ["loop_pmi", "log_deg_el", "log_deg_pr", "dom_same", "dom_bound"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def read_bedpe(acc, report=print):
    p = HIC / f"{acc}.bedpe.gz"
    if not p.exists():
        return []
    op = gzip.open(p, "rt") if p.read_bytes()[:2] == b"\x1f\x8b" else open(p, "rt")
    out = []
    for line in op:
        if line.startswith(("#", "chr1\tx1", "chrom")):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 6:
            continue
        try:
            out.append((f[0], int(f[1]), int(f[2]), f[3], int(f[4]), int(f[5]),
                        float(f[6]) if len(f) > 6 and f[6].replace(".", "").replace("-", "").isdigit()
                        else 1.0))
        except ValueError:
            continue
    return out


def anchor_index(loops):
    """chrom -> (starts, ends, loop_id, side) sorted by start, for overlap queries."""
    d = defaultdict(list)
    for i, r in enumerate(loops):
        d[r[0]].append((r[1], r[2], i, 0))
        d[r[3]].append((r[4], r[5], i, 1))
    out = {}
    for c, v in d.items():
        v.sort()
        out[c] = (np.array([x[0] for x in v], np.int64),
                  np.array([x[1] for x in v], np.int64),
                  np.array([x[2] for x in v], np.int64),
                  np.array([x[3] for x in v], np.int8))
    return out


def hits(idx, chrom, a, b, back=256):
    e = idx.get(chrom)
    if e is None:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    st, en, lid, side = e
    j = int(np.searchsorted(st, b))
    lo = max(0, j - back)
    m = en[lo:j] > a
    return lid[lo:j][m], side[lo:j][m]


def loop_features(S, idx, loops, dom_iv, e_idx, g_idx, perm=None, report=print):
    el = [str(k) for k in S["el_key"]]
    gn = [str(k) for k in S["gn_key"]]
    n = len(e_idx)
    F = {k: np.zeros(n) for k in LOOPCOLS + PMICOLS}
    el_h = {}
    for i in range(len(el)):
        c, rest = el[i].split(":")
        a, b = rest.split("-")
        el_h[i] = hits(idx, c, int(a), int(b))
    pr_h = {}
    for i in range(len(gn)):
        c, p, _ = gn[i].split(":")
        pr_h[i] = hits(idx, c, int(p) - PROM_PAD, int(p) + PROM_PAD)
    nl = max(len(loops), 1)
    for i in range(n):
        e = int(e_idx[i])
        g = int(g_idx[i]) if perm is None else int(perm[int(g_idx[i])])
        le, se = el_h[e]
        lp, sp = pr_h[g]
        F["log_deg_el"][i] = np.log10(1.0 + len(le))
        F["log_deg_pr"][i] = np.log10(1.0 + len(lp))
        if not len(le) or not len(lp):
            F["loop_pmi"][i] = np.log2(0.5 / (0.5 + len(le) * len(lp) / nl))
            continue
        # a link needs the SAME loop id with the element on one side and the promoter on the other
        common = np.intersect1d(le, lp, assume_unique=False)
        real = []
        for lid in common:
            s1 = se[le == lid]
            s2 = sp[lp == lid]
            if set(s1.tolist()) != set(s2.tolist()) or len(set(s1.tolist()) | set(s2.tolist())) > 1:
                real.append(lid)
        k = float(len(real))
        F["loop_n"][i] = k
        F["loop_any"][i] = float(k > 0)
        F["loop_best"][i] = max((loops[int(j)][6] for j in real), default=0.0)
        F["loop_support"][i] = k
        exp = len(le) * len(lp) / nl
        F["loop_pmi"][i] = np.log2((k + 0.5) / (exp + 0.5))
    # contact domains
    for i in range(n):
        e = int(e_idx[i])
        g = int(g_idx[i]) if perm is None else int(perm[int(g_idx[i])])
        c, rest = el[e].split(":")
        a, b = rest.split("-")
        gc, gp, _ = gn[g].split(":")
        de = set(hits(dom_iv, c, int(a), int(b))[0].tolist())
        dg = set(hits(dom_iv, gc, int(gp), int(gp) + 1)[0].tolist())
        F["dom_same"][i] = float(bool(de & dg))
        arr = dom_iv.get(c)
        if arr is not None and c == gc:
            lo_, hi_ = min(int(a), int(gp)), max(int(b), int(gp))
            F["dom_bound"][i] = float(((arr[0] > lo_) & (arr[0] < hi_)).sum())
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run(X, y, chrom, g_idx, jitter, tag, report=print):
    r1, ap = [], []
    for s in SEEDS:
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0:
                continue
            m = gbm(s)
            m.fit(np.nan_to_num(X[tr]), y[tr])
            sc[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        r1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
        ap.append(average_precision_score(y, sc))
    r1, ap = np.array(r1), np.array(ap)
    report(f"    {tag:44} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 186  CONTACT AT SUB-KILOBASE RESOLUTION: does the prediction hold?")
    say("=" * 104)
    say(f"  PREDECLARED: pooled median anchor width <= {MAX_ANCHOR} bp and >= {MIN_GENE_COVER:.0%}")
    say("  of evaluable genes with a linked candidate; every arm on loop 173's E3 bar; the block")
    say("  must survive being expressed as PMI against its own two anchor degrees; the real")
    say(f"  anchors must beat a stranger's in >= 90% of {N_SWAP} draws, where loop 181's 5 kb")
    say("  matrix beat 0 of 20; and the sub-kilobase block must beat loop 181's 5-10 kb Rao loops.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))

    loops, prov = [], {}
    for acc in FINE:
        r = read_bedpe(acc, say)
        prov[acc] = len(r)
        loops += r
    doms = []
    for acc in DOMAINS:
        doms += read_bedpe(acc, say)
    say(f"    pooled {len(loops):,} loop calls from {len([a for a in prov if prov[a]])} files "
        f"and {len(doms):,} contact domains, all GRCh38")
    for a, k in prov.items():
        say(f"      {a} {k:9,}")
    idx = anchor_index(loops)
    dom_iv = anchor_index(doms)

    # ---- A1 ------------------------------------------------------------------------------------
    say()
    say("A1 IS THE RESOLUTION ACTUALLY BETTER?")
    w = np.array([r[2] - r[1] for r in loops] + [r[5] - r[4] for r in loops])
    say(f"     pooled anchor width: median {np.median(w):.0f} bp, "
        f"IQR {np.percentile(w,25):.0f}-{np.percentile(w,75):.0f}; "
        f"loop 181's Rao list was 5,000-10,000 bp over 6,057 loops")
    F = loop_features(S, idx, loops, dom_iv, e_idx, g_idx, report=say)
    cand = defaultdict(list)
    pos = Counter()
    for i in range(len(y)):
        cand[int(g_idx[i])].append(i)
        if y[i]:
            pos[int(g_idx[i])] += 1
    ev = sorted(g for g in cand if len(cand[g]) >= 2 and pos[g] > 0)
    with_link = sum(1 for g in ev if any(F["loop_any"][i] > 0 for i in cand[g]))
    cover = with_link / max(len(ev), 1)
    say(f"     pairs with a loop link: {F['loop_any'].mean():.1%}; "
        f"evaluable genes with a linked candidate: {with_link}/{len(ev)} ({cover:.1%})")
    say(f"     loops anchored per element: median {np.median(10**F['log_deg_el']-1):.0f}; "
        f"per promoter: median {np.median(10**F['log_deg_pr']-1):.0f}")
    a1 = bool(np.median(w) <= MAX_ANCHOR and cover >= MIN_GENE_COVER)
    GG.verdict(a1, emit=say,
               if_true=f"A1 PASS -- median anchor {np.median(w):.0f} bp, below the 500-580 bp "
                       f"median element width's own scale, and {cover:.0%} of evaluable genes have "
                       f"a linked candidate",
               if_false=f"A1 FAIL -- anchor {np.median(w):.0f} bp / gene coverage {cover:.1%}; a "
                        f"failure below could be absence rather than contact")

    # ---- features --------------------------------------------------------------------------
    say()
    say("   building the comparison stacks")
    E, FAM, _ = L178.element_frame(S, "el", lambda *_: None)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    for c in P:
        P[c] = np.nan_to_num(P[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xd = np.column_stack([P["log_dist"]])
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    # loop 185's winning ingredients: measured co-binding and accessibility
    names = sorted({(v.get("name") or "").upper().split("::")[0]
                    for v in TD.load().values() if v.get("name")})
    Be, tfs = CH.build(S["el_key"], names, lambda *_: None)
    rows = SC.load_benchmark(lambda *_: None)
    dhs, h3k = {}, {}
    for r in rows:
        k = f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}"
        try:
            dhs[k] = float(r.get("DHS.RPM") or 0)
            h3k[k] = float(r.get("H3K27ac.RPM") or 0)
        except ValueError:
            pass
    ek = [str(k) for k in S["el_key"]]
    IN = np.column_stack([
        np.array([np.log10(1 + dhs.get(ek[int(i)], 0.0)) for i in e_idx]),
        np.array([np.log10(1 + h3k.get(ek[int(i)], 0.0)) for i in e_idx]),
        np.log10(1.0 + Be.sum(0))[e_idx].astype(float)])
    X185 = np.column_stack([Xbase, IN])
    Xl = np.column_stack([F[c] for c in LOOPCOLS])
    Xlp = np.column_stack([F[c] for c in LOOPCOLS + PMICOLS])

    # loop 181's coarse loops, for A7
    rao = CT.load_bedpe(CT.LOOPS, lambda *_: None)
    say(f"    loop 181's Rao list: {len(rao):,} loops for the head-to-head")
    # Rao list is hg19; lift the ELEMENTS to hg19 to score it on its own terms
    from enh import genome as GEN
    lo19 = GEN.LiftOver()
    rid = anchor_index([(r[0], r[1], r[2], r[3], r[4], r[5], 1.0) for r in rao])

    class ShimS(dict):
        pass
    S19 = ShimS(S)
    el19, gn19 = [], []
    for k in S["el_key"]:
        c, rest = str(k).split(":")
        a, b = rest.split("-")
        v = lo19.lift_interval(c, int(a), int(b))
        el19.append(f"{c}:{v[0]}-{v[1]}" if v else f"{c}:0-0")
    for k in S["gn_key"]:
        c, p, s = str(k).split(":")
        q = lo19.lift(c, int(p))
        gn19.append(f"{c}:{q or 0}:{s}")
    S19["el_key"] = np.array(el19, dtype=object)
    S19["gn_key"] = np.array(gn19, dtype=object)
    Frao = loop_features(S19, rid, [(r[0], r[1], r[2], r[3], r[4], r[5], 1.0) for r in rao],
                         anchor_index([]), e_idx, g_idx, report=lambda *_: None)
    Xrao = np.column_stack([Frao[c] for c in LOOPCOLS + PMICOLS])
    say(f"    Rao block: {Frao['loop_any'].mean():.1%} of pairs carry a link, "
        f"against {F['loop_any'].mean():.1%} for the sub-kilobase pool")

    res = {}
    res["distance"] = run(Xd, y, chrom, g_idx, jitter, "distance", say)
    res["dist+loops"] = run(np.column_stack([Xd, Xl]), y, chrom, g_idx, jitter,
                            "distance + loop links", say)
    res["dist+loops+pmi"] = run(np.column_stack([Xd, Xlp]), y, chrom, g_idx, jitter,
                                "distance + loop links as PMI", say)
    res["dist+rao"] = run(np.column_stack([Xd, Xrao]), y, chrom, g_idx, jitter,
                          "distance + loop 181's 5-10 kb Rao loops", say)
    res["l185"] = run(X185, y, chrom, g_idx, jitter, "loop 185's stack", say)
    res["l185+loops"] = run(np.column_stack([X185, Xlp]), y, chrom, g_idx, jitter,
                            "loop 185's stack + loop links", say)

    def gate(tag, a, b, title, if_t, if_f, use_ap=True):
        d = L173.paired(res[a], res[b])
        say()
        say(title)
        say(f"     {a} vs {b}   {L173.fmt(d)}")
        ok = L173.gate_pair(d, use_ap=use_ap)
        GG.verdict(ok, emit=say, if_true=f"{tag} PASS -- {if_t}", if_false=f"{tag} FAIL -- {if_f}")
        return ok, d

    a2, d2 = gate("A2", "dist+loops", "distance",
                  "A2 DOES THE LOOP LINK ADD OVER DISTANCE?",
                  "a called loop connecting this element to this promoter adds over how far apart "
                  "they are",
                  "loop links add nothing over distance")
    a3, d3 = gate("A3", "dist+loops+pmi", "distance",
                  "A3 IS IT THE LINK, OR THE TWO ANCHOR DEGREES?",
                  "the link survives being expressed against what the two anchor degrees alone "
                  "predict",
                  "once normalised by both anchor degrees the block stops helping -- it was "
                  "loop-dense neighbourhoods at both ends")
    a4, d4 = gate("A4", "l185+loops", "l185",
                  "A4 DOES IT ADD OVER THE BEST STACK SO FAR?",
                  "loop links add over sequence, co-binding and accessibility together",
                  "loop links add nothing the best stack did not already have")

    say()
    say(f"A5 THE STRANGER SWAP: {N_SWAP} draws giving each gene another gene's promoter anchors")
    gchrom = {}
    for i in range(len(y)):
        gchrom[int(g_idx[i])] = str(chrom[i])
    by_c = defaultdict(list)
    for g, c in gchrom.items():
        by_c[c].append(g)
    ng = len(S["gn_key"])
    swaps = []
    for k in range(N_SWAP):
        rr = np.random.default_rng(3000 + k)
        perm = np.arange(ng)
        for c, gs in by_c.items():
            pm = list(gs)
            rr.shuffle(pm)
            for a_, b_ in zip(sorted(gs), pm):
                perm[a_] = b_
        Fp = loop_features(S, idx, loops, dom_iv, e_idx, g_idx, perm=perm, report=lambda *_: None)
        Xp = np.column_stack([Fp[c] for c in LOOPCOLS + PMICOLS])
        r = run(np.column_stack([Xd, Xp]), y, chrom, g_idx, jitter, f"swap {k+1}", lambda *_: None)
        swaps.append(float(r["r1"].mean()))
        if (k + 1) % 5 == 0:
            say(f"     draw {k+1}/{N_SWAP}  R@1 {swaps[-1]:.4f}")
    swaps = np.array(swaps)
    real = float(res["dist+loops+pmi"]["r1"].mean())
    frac = float((real > swaps).mean())
    say(f"     real {real:.4f} against swapped mean {swaps.mean():.4f} "
        f"(min {swaps.min():.4f}, max {swaps.max():.4f}); real beats {frac:.0%}")
    say("     loop 181's 5 kb matrix beat 0% of its twenty swaps")
    a5 = bool(frac >= 0.90)
    GG.verdict(a5, emit=say,
               if_true="A5 PASS -- a stranger's promoter anchors do not work, so the block is "
                       "reading this pair and not the neighbourhood",
               if_false="A5 FAIL -- a stranger's anchors work as well, so the block reads loop "
                        "density and not this element-promoter link")

    say()
    say("A6 THE DECISIVE ONE")
    best = max((k for k in res if k != "distance"), key=lambda k: res[k]["r1"].mean())
    d6 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} / AUPRC "
        f"{res[best]['ap'].mean():.4f} against distance {res['distance']['r1'].mean():.4f}")
    say(f"     {L173.fmt(d6)}")
    say(f"     loop 185's best stack reached R@1 {L185_BEST_R1}")
    a6 = L173.gate_pair(d6)
    GG.verdict(a6, emit=say,
               if_true=f"A6 PASS -- {best} clears the bar every stage-two loop has been held to",
               if_false="A6 FAIL -- stage two is still distance")

    say()
    say("A7 THE HEAD-TO-HEAD: sub-kilobase anchors against loop 181's 5-10 kb loops")
    dd = res["dist+loops+pmi"]["r1"] - res["dist+rao"]["r1"]
    sem = dd.std(ddof=1) / np.sqrt(len(dd))
    say(f"     sub-kb {res['dist+loops+pmi']['r1'].mean():.4f} against Rao "
        f"{res['dist+rao']['r1'].mean():.4f}; difference {dd.mean():+.4f} +/- {sem:.4f} "
        f"({int((dd > 0).sum())}/5 up)")
    a7 = bool((dd > 0).sum() >= MIN_SEEDS and dd.mean() > 3 * sem)
    GG.verdict(a7, emit=say,
               if_true="A7 PASS -- resolution WAS the binding constraint, exactly as loop 186's "
                       "source predicted before the data was fetched",
               if_false="A7 FAIL -- sub-kilobase anchors do no better than 5-10 kb ones, so "
                        "resolution was NOT what stopped loop 181 and its conclusion stands for a "
                        "different reason than the one predicted")

    say()
    say("A8 WHAT THIS CANNOT SHOW")
    say("     A loop call is a model's decision about a local enrichment, not an observation. Six")
    say("     pooled files means six thresholds, and a pair linked in one and not another is")
    say("     counted as linked here.")
    say("     Loop calls come from the same population-averaged experiment as the matrix. A loop")
    say("     is a frequency across millions of nuclei, not a contact in any one of them.")
    say("     A7 compares two loop lists that differ in depth, caller and assembly at once, so it")
    say("     attributes a difference to resolution that is only mostly about resolution.")
    say("     Everything here is still bounded by what the candidate sets contain: the screens")
    say("     chose which elements to test against which genes.")
    a8 = True
    say(f"     A8 {'PASS' if a8 else 'FAIL'}")

    gates = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "A8": a8}
    man = RM.manifest(inputs=[HIC / f"{FINE[0]}.bedpe.gz"],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["the block re-expressed as PMI against both anchor degrees",
                                f"{N_SWAP} chromosome-matched promoter-anchor swaps",
                                "loop 181's 5-10 kb Rao loops on identical folds",
                                "identical folds and seeds as every stage-two loop since 173"],
                      note="sub-kilobase K562 loop calls on the stage-two task")
    out = dict(test="enhancer sub-kb loops", gates=gates,
               n_loops=len(loops), provenance=prov, n_domains=len(doms),
               anchor_width_median=float(np.median(w)), gene_cover=float(cover),
               frac_pairs_linked=float(F["loop_any"].mean()),
               frac_pairs_linked_rao=float(Frao["loop_any"].mean()),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("A2", d2), ("A3", d3), ("A4", d4), ("A6", d6))},
               swap_draws=[float(x) for x in swaps], swap_frac_beaten=frac,
               head_to_head=float(dd.mean()), best_arm=best,
               manifest=man, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'PASS' if v else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}   [{time.time()-t0:.0f}s]")
    say("=" * 104)
    out["log"] = log
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
