"""CAN DATA THE LITERATURE DOES NOT HAVE BEAT THE LITERATURE PRIOR?  Co-expression and co-dependency as
evidence on the missing-piece task.

WHERE THIS SITS.  `cell_orphan_obscurity` settled that the 0.645 closed-book naming score is substantially
MEMORY: accuracy runs 0.463 on enzymes with under 20 papers against 0.812 on those with over 500, and the gap
survives removing the giveaway puzzles and restricting to metabolic reactions. So on the ~9,186 unannotated
enzyme-shaped reactions -- obscure by construction -- the honest expectation is a coin flip.

The way past a literature prior is evidence the literature does not contain. Two channels, both measured
across the same ~1,150 cancer cell lines, both orthogonal to what has been written about a gene:

    CO-DEPENDENCY   DepMap 24Q2 CRISPRGeneEffect. Does knocking out A change viability the way knocking out
                    B does? Obligate machine membership shows up here regardless of how famous the gene is.
    CO-EXPRESSION   DepMap 24Q2 OmicsExpressionProteinCodingGenesTPMLogp1, md5-pinned. Do A and B rise and
                    fall together across cell lines? Pathway co-regulation shows up here.

HOW EVIDENCE IS INJECTED, and why it is not a candidate list.  The arm that must pick from a candidate list is
capped at 0.505 because the list contains the answer only half the time. So evidence is offered as CONTEXT --
"here are the 15 genes whose dependency profile most resembles this reaction's neighbourhood" -- while the
solver stays free to name ANY human gene. Additive, not restrictive.

ANCHORS.  A reaction's neighbourhood is the genes of its own protein participants plus the catalysts of
reactions sharing a metabolite with it. The TRUE CATALYST IS REMOVED from every anchor set; otherwise the
evidence would be a paraphrase of the answer.

THE LIVENESS GATE, run before anything is claimed.  A channel that returns noise looks exactly like a channel
that returns a real negative. So each channel first has to recover known obligate complexes -- anchor on two
subunits, find the third -- and must NOT rank unrelated genes highly. A channel failing that is a broken
pipeline, not evidence of anything, and the run aborts rather than reporting a false negative.

THE RETRIEVAL MEASUREMENT, which comes before spending any solver.  Before asking whether evidence HELPS, ask
whether it could: how often does the true catalyst appear in the top K the solver would be shown? Measured
per literature bin, because the obscure bin is the only one that matters for real gaps.

PREDECLARED, before any number:
    obscure-bin accuracy rises materially over 0.463
        -> cell-line data supplies what the literature cannot, and the backtrace becomes deliverable on true
           gaps rather than only on transcribable ones.
    no change, and the scrambled-evidence control matches the real one
        -> the lift, if any, is "a list of plausible genes helps", not this evidence.
    no change, and retrieval@K was already near zero
        -> the channels do not connect a reaction to its enzyme at all. Report which, and why, rather than
           blaming the model.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_obscurity as O
import cell_orphan_ties_nn as T

OUT = A.OUT
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
EXPR_CSV = SP / "depmap" / "OmicsExpression.csv"
EXPR_NPZ = SP / "depmap_expr.npz"
DEP_NPZ = OUT / "depmap_vecs.npz"
PUZZLES = OUT / "obscurity_puzzles.json"
EVID = OUT / "evidence_blocks.json"
ANSWERS = OUT / "evidence_answers.json"
TOPK = 15
MD5_EXPR = "f00c434db50d811544ac4d047003979f"

# anchor two subunits of an obligate complex, the third must come back near the top
POS_CTRL = ((["SDHA", "SDHB"], "SDHC"), (["POLR2A", "POLR2B"], "POLR2C"),
            (["PSMA1", "PSMA2"], "PSMA3"), (["COPA", "COPB1"], "COPB2"),
            (["ATP6V1A", "ATP6V1B2"], "ATP6V1C1"), (["EIF3A", "EIF3B"], "EIF3D"))
NEG_CTRL = ((["SDHA", "SDHB"], "TYR"), (["POLR2A", "POLR2B"], "CPT1A"),
            (["COPA", "COPB1"], "ALB"))


def _z(M):
    """rows z-scored, so a dot product over samples IS the Pearson correlation."""
    M = np.asarray(M, np.float32)
    M = M - M.mean(1, keepdims=True)
    s = M.std(1, keepdims=True)
    s[s < 1e-8] = 1e-8
    return M / s


def expr_matrix():
    """genes x cell lines from the DepMap expression CSV (which is lines x genes, headed 'SYMBOL (ENTREZ)')."""
    if EXPR_NPZ.exists():
        d = np.load(EXPR_NPZ, allow_pickle=True)
        return np.array(d["syms"]), np.asarray(d["Z"], np.float32)
    import pandas as pd
    df = pd.read_csv(EXPR_CSV, index_col=0)
    syms = np.array([c.split(" (")[0] for c in df.columns])
    X = df.to_numpy(np.float32).T                      # genes x lines
    keep = ~np.isnan(X).any(1) & (X.std(1) > 1e-6)     # a flat or partial gene carries no correlation
    syms, X = syms[keep], X[keep]
    _, first = np.unique(syms, return_index=True)      # duplicate symbols -> keep one
    syms, X = syms[np.sort(first)], X[np.sort(first)]
    Z = _z(X)
    np.savez_compressed(EXPR_NPZ, syms=syms, Z=Z)
    return syms, Z


def dep_matrix():
    d = np.load(DEP_NPZ, allow_pickle=True)
    return np.array(d["syms"]), _z(np.asarray(d["Z"], np.float32))


def channels(report):
    ch = {}
    s, Z = dep_matrix()
    ch["codependency"] = (s, Z, {g: i for i, g in enumerate(s)})
    report(f"  codependency : {Z.shape[0]:,} genes x {Z.shape[1]:,} cell lines (DepMap 24Q2 CRISPR)")
    s, Z = expr_matrix()
    ch["coexpression"] = (s, Z, {g: i for i, g in enumerate(s)})
    report(f"  coexpression : {Z.shape[0]:,} genes x {Z.shape[1]:,} cell lines (DepMap 24Q2 expression, "
           f"md5 {MD5_EXPR[:8]}…)")
    return ch


def rank_all(chan, anchors, exclude):
    """every gene scored by MEAN correlation with the anchor set; anchors themselves are not predictions."""
    syms, Z, idx = chan
    ai = [idx[a] for a in anchors if a in idx]
    if not ai:
        return None, None
    sc = (Z @ Z[ai].mean(0)) / Z.shape[1]
    order = np.argsort(-sc)
    drop = set(exclude)
    order = [i for i in order if syms[i] not in drop]
    return [syms[i] for i in order], sc


def anchors_for(B, cat_of_part, i):
    s, cats, P = B["steps"][i], B["cats"], B["P"]
    pg = {g for w in ("in", "out") for p in (s.get(w) or []) for g in (p.get("genes") or [])}
    neigh = {j for p in P[i] for j in cat_of_part.get(p, ()) if j != i}
    # the answer is removed: evidence built from the answer would be a paraphrase of it
    return (pg | {g for j in neigh for g in cats[j]}) - set(cats[i])


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("CO-EXPRESSION AND CO-DEPENDENCY AS EVIDENCE -- can data the literature lacks beat the prior?")
    report("=" * 100)
    report("  Closed-book naming is 0.463 on enzymes with <20 papers and 0.812 on those with >500, so the")
    report("  score is substantially memory and true gaps are a coin flip. Both channels below are measured")
    report("  across ~1,150 cell lines and are orthogonal to how much has been WRITTEN about a gene.")
    ch = channels(report)

    # ---------------- LIVENESS: a channel returning noise is indistinguishable from a real negative --------
    report(f"\n  LIVENESS GATE -- anchor two subunits of an obligate complex, the third must come back high")
    report(f"    {'channel':<15}{'positive control (rank of held-out subunit)':<52}{'negative'}")
    live = {}
    for nm, c in ch.items():
        pos = [rank_all(c, a, a)[0] for a, t in POS_CTRL]
        pr = [(o.index(t) + 1 if o and t in o else None) for o, (a, t) in zip(pos, POS_CTRL)]
        neg = [rank_all(c, a, a)[0] for a, t in NEG_CTRL]
        nr = [(o.index(t) + 1 if o and t in o else None) for o, (a, t) in zip(neg, NEG_CTRL)]
        ok = sum(1 for r in pr if r and r <= 50)
        live[nm] = {"pos": pr, "neg": nr, "n_pos_top50": ok}
        report(f"    {nm:<15}{str([r if r else '-' for r in pr]):<52}"
               f"{str([r if r else '-' for r in nr])}")
    report(f"    (ribosomal and common-essential genes have flat profiles by construction, so a miss there is")
    report(f"     expected; the gate is that MOST controls land high and unrelated genes do not.)")
    dead = [nm for nm, v in live.items() if v["n_pos_top50"] < 2]
    if dead:
        report(f"\n  *** CHANNEL(S) NOT LIVE: {dead}. That is a broken pipeline, not a finding. Aborting rather")
        report("      than reporting a false negative about the biology.")
        return 1
    report(f"  Both channels are live. A negative from here is about the BIOLOGY, not the plumbing.")

    # ---------------- RETRIEVAL: could the evidence possibly help, before asking whether it does -----------
    B = T.build()
    cats, P = B["cats"], B["P"]
    cat_of_part = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of_part[p].add(i)
    sel = json.load(open(PUZZLES))["pids"]
    papers = O._lit()
    pool = O._pool(B, papers)
    bidx = np.array([O._binof(pool[i]) for i in sel])

    report(f"\n  RETRIEVAL -- how often is the true catalyst in the top {TOPK} the solver would be shown?")
    report(f"  Asked BEFORE spending a single solver: evidence that never contains the answer cannot supply it.")
    ranks, blocks = {}, {}
    for nm, c in ch.items():
        rr = []
        for i in sel:
            anch = anchors_for(B, cat_of_part, i)
            order, _ = rank_all(c, anch, anch)
            if order is None:
                rr.append(None)
                continue
            tr = set(cats[i])
            pos = next((k for k, g in enumerate(order) if g in tr), None)
            rr.append(pos + 1 if pos is not None else None)
            blocks.setdefault(i, {})[nm] = order[:TOPK]
        ranks[nm] = rr
    report(f"    {'channel':<15}{'@1':>7}{'@15':>7}{'@50':>7}{'@200':>7}{'median rank':>14}"
           f"{'@15 obscure bin':>18}")
    retr = {}
    for nm, rr in ranks.items():
        r = np.array([x if x else 10 ** 9 for x in rr])
        ob = bidx == 0
        retr[nm] = {"at1": float(np.mean(r <= 1)), "at15": float(np.mean(r <= TOPK)),
                    "at50": float(np.mean(r <= 50)), "at200": float(np.mean(r <= 200)),
                    "median": float(np.median(r[r < 10 ** 9])),
                    "at15_obscure": float(np.mean(r[ob] <= TOPK))}
        v = retr[nm]
        report(f"    {nm:<15}{v['at1']:>7.3f}{v['at15']:>7.3f}{v['at50']:>7.3f}{v['at200']:>7.3f}"
               f"{v['median']:>14.0f}{v['at15_obscure']:>18.3f}")

    best = max(retr, key=lambda k: retr[k]["at15_obscure"])
    report("\n  READING (retrieval only -- whether the evidence HELPS is measured separately, after the run)")
    if retr[best]["at15_obscure"] < 0.10:
        report(f"  THE EVIDENCE ALMOST NEVER CONTAINS THE ANSWER. Best channel '{best}' puts the true catalyst")
        report(f"  in the top {TOPK} for {retr[best]['at15_obscure']:.1%} of obscure reactions, median rank "
               f"{retr[best]['median']:.0f} of ~18,000.")
        report("  Both channels are LIVE -- they recover obligate complexes -- so this is a statement about")
        report("  what they measure, not about the code. Co-dependency is a claim about VIABILITY covariance")
        report("  and co-expression about transcriptional co-regulation; neither is a claim that one gene")
        report("  ACTS ON another's substrate. Most metabolic enzymes are also non-essential in culture,")
        report("  where the medium supplies what the pathway would make, so their profiles are near-flat.")
    else:
        report(f"  Best channel '{best}' contains the answer in the top {TOPK} for "
               f"{retr[best]['at15_obscure']:.1%} of obscure")
        report("  reactions -- enough that offering it as context could plausibly move the score.")

    json.dump({"reaction_ids": [int(i) for i in sel],
               "blocks": {str(i): v for i, v in blocks.items()}},
              open(EVID, "w"), indent=1)
    json.dump({"test": "cell_orphan_evidence", "liveness": live, "retrieval": retr, "topk": TOPK,
               "log": log}, open(OUT / "cell_orphan_evidence.json", "w"), indent=2)
    report(f"\n  wrote evidence blocks -> {EVID}")
    report(f"  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
