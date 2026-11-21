"""Loop 180. Which genes should an enhancer predictor be scored on at all?

THE GAP THIS CLOSES, AND IT IS ONE I LEFT OPEN. The original request was to remove the
self-regulating factors, on the reasoning that a gene held in an autoregulatory loop does not need
a distal element to influence its transcription. Loop 175 tested that on the MOTIF side -- it
removed self-regulating factors from the set of matrices contributing to the sequence features --
and found nothing, with a size-matched control showing that dropping any 366 matrices helped just
as much. What loop 175 never tested is the GENE side: dropping the genes that are themselves
autoregulating factors from the evaluation. Loop 175's own A8 said so in as many words, and the
gate was designed and then not implemented. This loop implements it, and then goes past it, because
the literal version cannot be tested.

WHY THE LITERAL VERSION CANNOT BE TESTED, measured before anything is claimed. TRRUST curates 19
factor symbols with a self-loop that also carry a JASPAR matrix here. Of the 199 evaluable genes in
this benchmark, exactly TWO are among them: GATA1 and MYB. Two genes out of 199 move within-gene
R@1 by at most 0.010, which is smaller than the seed-to-seed spread of every arm measured so far.
R1 states that as a predeclared power floor rather than reporting a number from it.

THE GENERALISATION, WHICH IS WELL POWERED AND IS THE SAME IDEA. Autoregulation is one route to a
gene whose expression does not route through a distal enhancer. There are others, and the benchmark
already carries a flag for the biggest one: `measuredGeneUbiquitousExpressed`. 55 of the 199
evaluable genes (27.6%) are flagged ubiquitously expressed -- housekeeping genes, driven from their
own promoters, with far less distal-enhancer dependence than a tissue-restricted gene. If
"enhancer-independent genes are noise in an enhancer benchmark" is right, those 55 are where the
noise is, and the effect should be visible at n=55 where it cannot be at n=2.

AND A LABEL-FREE VERSION OF THE SAME AXIS. Ubiquitous expression is an annotation. The
sequence-computable proxy for it is the promoter's CpG-island character: CpG-island promoters
belong overwhelmingly to broadly expressed genes and are comparatively enhancer-independent, while
CpG-poor promoters belong to tissue-restricted genes that depend on distal elements (Deaton & Bird,
Genes Dev 2011). Observed-over-expected CpG in the 1 kb promoter window is computed here from the
same sequence already extracted, so R5's stratification uses no labels and no annotation at all.

THE COMPARISON THAT IS ACTUALLY VALID, and it is not the obvious one. R@1 measured on a subset of
genes is not comparable to R@1 measured on all of them -- the gene sets differ, so the number moves
for reasons that have nothing to do with the model. What IS comparable across strata is the
INCREMENT the sequence stack buys over distance alone on the same genes. Every gate below is
written on that increment, not on a raw R@1.

PREDECLARED, BEFORE ANY NUMBER.

  R1 IS THE LITERAL VERSION POWERED? The stratum of evaluable genes that are curated autoregulating
     factors.
     Gate: PASS iff it holds at least 10 genes. On FAIL, the stratum is reported and NO claim is
     made from it, in either direction. This gate is written to fail on the data as it stands and
     is here so the failure is on the record rather than discovered later.

  R2 DOES DROPPING ENHANCER-INDEPENDENT GENES HELP? The sequence stack's increment over distance on
     the tissue-restricted stratum against the same increment on the ubiquitously expressed one.
     Gate: PASS iff the increment on the tissue-restricted stratum exceeds the increment on the
     ubiquitous stratum by more than 3 sem of the paired per-seed difference.

  R3 THE SIZE-MATCHED CONTROL. Dropping 55 of 199 genes changes which genes are averaged over, and
     any subset moves the number. 20 random 55-gene drops, same folds, same seeds.
     Gate: PASS iff the real split's increment exceeds the matched draws' in >= 90% of them.

  R4 ARE THE STRATA DIFFERENT FOR A REASON THAT IS NOT ABOUT ENHANCERS? Candidate counts and
     distance-only R@1 per stratum.
     Gate: PASS iff the median candidate count of the two strata is within a factor of two. A FAIL
     means the strata differ in how hard the ranking is, and R2's difference cannot be attributed
     to enhancer dependence.

  R5 THE LABEL-FREE AXIS. Evaluable genes split into quartiles by promoter CpG observed/expected,
     computed from sequence alone.
     Gate: PASS iff the sequence increment over distance is monotone decreasing across the
     quartiles from CpG-poor to CpG-rich, or at least has a negative Spearman correlation with the
     quartile index. This is the direction the hypothesis predicts and the opposite direction would
     refute it rather than being reported as an interesting nuance.

  R6 THE SHUFFLE STILL DECIDES, on whichever stratum looks best.
     Gate: real beats dinucleotide-shuffled in >= 4/5 seeds past 3 sem.

  R7 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_dependence.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import genome as GEN                # noqa: E402
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_dependence.json"
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
MIN_AUTOREG_GENES = 10
N_MATCHED = 20
CAND_RATIO_MAX = 2.0

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def promoter_cpg(S, report=print):
    """Observed/expected CpG in each gene's 1 kb promoter window, from the same sequence loop 173
    extracted. Label-free and annotation-free."""
    lo = GEN.LiftOver()
    keys = [str(k).split(":") for k in S["gn_key"]]
    reg = []
    for k in keys:
        p = lo.lift(k[0], int(k[1]))
        reg.append((k[0], (p or 0) - SC.PROMOTER_PAD, (p or 0) + SC.PROMOTER_PAD))
    seqs = GEN.Genome().extract_cached(reg, "prom", report)
    oe = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        ok = s <= 3
        n = int(ok.sum())
        if n < 100:
            continue
        c = int((s == 1).sum())
        g = int((s == 2).sum())
        cg = int(((s[:-1] == 1) & (s[1:] == 2)).sum())
        oe[i] = cg * n / max(c * g, 1)
    report(f"    promoter CpG observed/expected: median {np.median(oe):.3f}, "
           f"IQR [{np.percentile(oe, 25):.3f}, {np.percentile(oe, 75):.3f}]")
    return oe


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def oof(X, y, chrom, seed):
    fold = L173.folds_for(chrom, seed)
    sc = np.zeros(len(y))
    for f in range(NFOLD):
        te = fold == f
        tr = ~te
        if te.sum() == 0 or y[tr].sum() == 0:
            continue
        m = gbm(seed)
        m.fit(np.nan_to_num(X[tr]), y[tr])
        sc[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
    return sc


def r1_on(sc, y, g_idx, jitter, genes):
    """within-gene R@1 restricted to a set of genes."""
    keep = np.isin(g_idx, list(genes))
    if keep.sum() == 0:
        return 0.0
    return L173.within_gene(sc[keep], y[keep], g_idx[keep], jitter[keep])[0]


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 180  WHICH GENES SHOULD AN ENHANCER PREDICTOR BE SCORED ON AT ALL?")
    say("=" * 104)
    say(f"  PREDECLARED: the curated autoregulating-factor stratum needs >= {MIN_AUTOREG_GENES}")
    say("  genes for any claim; every comparison is on the sequence stack's INCREMENT over")
    say("  distance, never on a raw R@1 across different gene sets; the split must beat 90% of")
    say(f"  {N_MATCHED} size-matched random gene drops; the strata's median candidate counts must")
    say(f"  be within a factor of {CAND_RATIO_MAX}; and the CpG increment must decrease from")
    say("  CpG-poor to CpG-rich, which is the direction the hypothesis predicts.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))
    sym = [str(k).split(":")[-1] for k in S["gn_key"]]

    cand = defaultdict(list)
    pos = Counter()
    for i in range(len(y)):
        cand[int(g_idx[i])].append(i)
        if y[i]:
            pos[int(g_idx[i])] += 1
    ev = sorted(gg for gg in cand if len(cand[gg]) >= 2 and pos[gg] > 0)
    say(f"    {len(ev)} evaluable genes")

    # ---- strata ---------------------------------------------------------------------------------
    ar = json.load(open(Path("colab/data/tf_autoregulation.json")))["matrices"]
    selfsym = {v["name"].upper() for v in ar.values() if v["cls"] == "SELF" and v.get("name")}
    autoreg = [gg for gg in ev if sym[gg].upper() in selfsym]
    rows = SC.load_benchmark(lambda *_: None)
    ub = {}
    for r in rows:
        ub[r["measuredGeneSymbol"]] = str(r.get("measuredGeneUbiquitousExpressed", "")).upper()
    ubiq = [gg for gg in ev if ub.get(sym[gg], "") == "TRUE"]
    spec = [gg for gg in ev if gg not in set(ubiq)]
    say(f"    curated autoregulating factors among them: {len(autoreg)} "
        f"({', '.join(sym[x] for x in autoreg) or 'none'})")
    say(f"    ubiquitously expressed: {len(ubiq)} ({len(ubiq)/len(ev):.1%});  "
        f"tissue-restricted: {len(spec)}")

    # ---- R1 -------------------------------------------------------------------------------------
    say()
    say("R1 IS THE LITERAL VERSION POWERED?")
    say(f"     TRRUST curates {len(selfsym)} self-looping factor symbols that also carry a matrix "
        f"here; {len(autoreg)} of the {len(ev)} evaluable genes are among them")
    say(f"     {len(autoreg)} genes move within-gene R@1 by at most "
        f"{len(autoreg)/len(ev):.4f} -- smaller than the seed spread of every arm measured")
    r1_gate = bool(len(autoreg) >= MIN_AUTOREG_GENES)
    GG.verdict(r1_gate, emit=say,
               if_true=f"R1 PASS -- {len(autoreg)} genes is enough to test the literal version",
               if_false=f"R1 FAIL -- {len(autoreg)} genes cannot support a claim in either "
                        f"direction, so the literal version is UNTESTABLE on this benchmark and "
                        f"nothing below is reported as testing it")

    # ---- features and the two arms ---------------------------------------------------------------
    E, FAM, _ = L178.element_frame(S, "el", say)
    Es, FAMs, _ = L178.element_frame(S, "sh", say)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    Ps, _, _ = L173.build_features(S, "sh", report=lambda *_: None)
    for fr in (P, Ps):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xstack = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xstack_s = np.column_stack([Ps[c] for c in base_cols] + [FAMs[c][e_idx] for c in fam_cols])
    Xdist = np.column_stack([P["log_dist"]])
    say(f"    stack: {Xstack.shape[1]} columns (loop 173's 34 plus loop 177's class/family)")

    SC_stack, SC_dist, SC_shuf = [], [], []
    for s in SEEDS:
        SC_stack.append(oof(Xstack, y, chrom, s))
        SC_dist.append(oof(Xdist, y, chrom, s))
        SC_shuf.append(oof(Xstack_s, y, chrom, s))

    def increment(genes):
        d = np.array([r1_on(SC_stack[k], y, g_idx, jitter, genes)
                      - r1_on(SC_dist[k], y, g_idx, jitter, genes) for k in range(len(SEEDS))])
        return d

    inc_all = increment(ev)
    inc_ub = increment(ubiq)
    inc_sp = increment(spec)
    for nm, gs, inc in (("all genes", ev, inc_all), ("ubiquitous", ubiq, inc_ub),
                        ("tissue-restricted", spec, inc_sp)):
        r_s = np.mean([r1_on(SC_stack[k], y, g_idx, jitter, gs) for k in range(len(SEEDS))])
        r_d = np.mean([r1_on(SC_dist[k], y, g_idx, jitter, gs) for k in range(len(SEEDS))])
        say(f"    {nm:20} n={len(gs):3d}   stack R@1 {r_s:.4f}   distance R@1 {r_d:.4f}   "
            f"increment {inc.mean():+.4f}")

    # ---- R2 -------------------------------------------------------------------------------------
    say()
    say("R2 DOES DROPPING ENHANCER-INDEPENDENT GENES HELP?")
    d = inc_sp - inc_ub
    sem = d.std(ddof=1) / np.sqrt(len(d))
    say(f"     tissue-restricted increment {inc_sp.mean():+.4f} against ubiquitous "
        f"{inc_ub.mean():+.4f}; difference {d.mean():+.4f} +/- {sem:.4f} "
        f"({int((d > 0).sum())}/5 up)")
    r2 = bool((d > 0).sum() >= MIN_SEEDS and d.mean() > 3 * sem)
    GG.verdict(r2, emit=say,
               if_true="R2 PASS -- sequence buys more over distance on genes that are not "
                       "ubiquitously expressed, which is what enhancer dependence predicts",
               if_false="R2 FAIL -- the sequence stack buys no more on tissue-restricted genes than "
                        "on housekeeping ones, so enhancer-independent targets are not what was "
                        "holding stage two back")

    # ---- R3 -------------------------------------------------------------------------------------
    say()
    say(f"R3 THE SIZE-MATCHED CONTROL: {N_MATCHED} random {len(ubiq)}-gene drops")
    draws = []
    for k in range(N_MATCHED):
        rr = np.random.default_rng(9000 + k)
        drop = set(rr.choice(ev, len(ubiq), replace=False).tolist())
        kept = [gg for gg in ev if gg not in drop]
        draws.append(float(increment(kept).mean()))
    draws = np.array(draws)
    frac = float((inc_sp.mean() > draws).mean())
    say(f"     real split {inc_sp.mean():+.4f} against matched draws mean {draws.mean():+.4f} "
        f"(min {draws.min():+.4f}, max {draws.max():+.4f}); real beats {frac:.0%}")
    r3 = bool(frac >= 0.90)
    GG.verdict(r3, emit=say,
               if_true="R3 PASS -- the ubiquitous split beats dropping the same number of genes at "
                       "random, so it is the genes and not the subsetting",
               if_false="R3 FAIL -- a random gene drop of the same size does as well, so whatever "
                        "R2 showed is subset noise")

    # ---- R4 -------------------------------------------------------------------------------------
    say()
    say("R4 ARE THE STRATA DIFFERENT FOR A REASON THAT IS NOT ABOUT ENHANCERS?")
    cu = np.median([len(cand[gg]) for gg in ubiq]) if ubiq else 0
    cs = np.median([len(cand[gg]) for gg in spec]) if spec else 0
    du = np.mean([r1_on(SC_dist[k], y, g_idx, jitter, ubiq) for k in range(len(SEEDS))])
    ds = np.mean([r1_on(SC_dist[k], y, g_idx, jitter, spec) for k in range(len(SEEDS))])
    say(f"     median candidates per gene: ubiquitous {cu:.0f}, tissue-restricted {cs:.0f}")
    say(f"     distance-only R@1: ubiquitous {du:.4f}, tissue-restricted {ds:.4f}")
    ratio = max(cu, cs) / max(min(cu, cs), 1)
    r4 = bool(ratio <= CAND_RATIO_MAX)
    GG.verdict(r4, emit=say,
               if_true=f"R4 PASS -- the strata carry comparable candidate sets (ratio {ratio:.2f}), "
                       f"so R2's difference is not a difference in how hard the ranking is",
               if_false=f"R4 FAIL -- candidate counts differ by {ratio:.2f}x, so the strata differ "
                        f"in ranking difficulty and R2 cannot be attributed to enhancer dependence")

    # ---- R5 -------------------------------------------------------------------------------------
    say()
    say("R5 THE LABEL-FREE AXIS: promoter CpG observed/expected")
    oe = promoter_cpg(S, say)
    vals = np.array([oe[gg] for gg in ev])
    qs = np.quantile(vals, [0.25, 0.5, 0.75])
    quart = [[] for _ in range(4)]
    for gg in ev:
        quart[int(np.searchsorted(qs, oe[gg]))].append(gg)
    incs = []
    for qi, gs in enumerate(quart):
        i_ = increment(gs)
        incs.append(float(i_.mean()))
        say(f"     Q{qi+1} CpG o/e <= {qs[qi] if qi < 3 else vals.max():.3f}   n={len(gs):3d}   "
            f"increment {i_.mean():+.4f}")
    rho = stats.spearmanr(range(4), incs).correlation
    say(f"     Spearman(quartile, increment) = {rho:+.3f}   "
        f"(the hypothesis predicts NEGATIVE: CpG-poor promoters gain more)")
    r5 = bool(rho < 0)
    GG.verdict(r5, emit=say,
               if_true="R5 PASS -- the sequence increment falls as promoters become more "
                       "CpG-island-like, which is the direction enhancer dependence predicts and "
                       "which uses no labels and no annotation",
               if_false="R5 FAIL -- the increment does not fall with promoter CpG content, so the "
                        "label-free version of the axis does not reproduce the annotation-based one")

    # ---- R6 -------------------------------------------------------------------------------------
    say()
    say("R6 THE SHUFFLE STILL DECIDES")
    best_genes = spec if inc_sp.mean() >= inc_ub.mean() else ubiq
    rs = np.array([r1_on(SC_stack[k], y, g_idx, jitter, best_genes) for k in range(len(SEEDS))])
    rh = np.array([r1_on(SC_shuf[k], y, g_idx, jitter, best_genes) for k in range(len(SEEDS))])
    dd = rs - rh
    sem6 = dd.std(ddof=1) / np.sqrt(len(dd))
    say(f"     on the {'tissue-restricted' if best_genes is spec else 'ubiquitous'} stratum: "
        f"real {rs.mean():.4f}, shuffled {rh.mean():.4f}, "
        f"difference {dd.mean():+.4f} +/- {sem6:.4f} ({int((dd > 0).sum())}/5 up)")
    r6 = bool((dd > 0).sum() >= MIN_SEEDS and dd.mean() > 3 * sem6)
    GG.verdict(r6, emit=say,
               if_true="R6 PASS -- on this stratum the real sequence beats a composition-matched "
                       "shuffle, so the stack is reading sites here",
               if_false="R6 FAIL -- the shuffle matches it on this stratum too, so restricting the "
                        "gene set did not turn composition into sites")

    say()
    say("R7 WHAT THIS CANNOT SHOW")
    say("     The literal request -- drop the autoregulating factors' own genes -- is untestable")
    say("     here at n=2 and stays untested. R2 through R5 test a GENERALISATION of it, not it.")
    say("     `measuredGeneUbiquitousExpressed` is the benchmark's annotation, and a gene being")
    say("     broadly expressed is evidence about, not proof of, enhancer independence.")
    say("     Every stratum is a subset of 199 genes, so even the well-powered arms here are")
    say("     working with tens of genes and the seed spread is correspondingly wide.")
    say("     None of this touches the real missing variable for stage two, which is which promoter")
    say("     the element is physically looped to.")
    r7 = True
    say(f"     R7 {'PASS' if r7 else 'FAIL'}")

    gates = {"R1": r1_gate, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6, "R7": r7}
    man = RM.manifest(inputs=[Path("colab/data/tf_autoregulation.json")],
                      available=len(ev), used=len(ev), selection="evaluable genes, stratified",
                      seed=L173.TIE_SEED,
                      controls=[f"{N_MATCHED} size-matched random gene drops",
                                "candidate-count and distance-only comparison across strata",
                                "a label-free promoter-CpG axis beside the annotation one",
                                "dinucleotide shuffle on the winning stratum"],
                      note="does removing enhancer-independent genes rescue stage two")
    out = dict(test="enhancer dependence by gene", gates=gates,
               n_evaluable=len(ev), n_autoreg=len(autoreg),
               autoreg_genes=[sym[x] for x in autoreg],
               n_ubiquitous=len(ubiq), n_specific=len(spec),
               increments=dict(all=[float(x) for x in inc_all],
                               ubiquitous=[float(x) for x in inc_ub],
                               specific=[float(x) for x in inc_sp]),
               matched_draws=[float(x) for x in draws], matched_frac_beaten=frac,
               candidates=dict(ubiquitous=float(cu), specific=float(cs)),
               distance_r1=dict(ubiquitous=float(du), specific=float(ds)),
               cpg_quartile_increment=incs, cpg_spearman=float(rho),
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
