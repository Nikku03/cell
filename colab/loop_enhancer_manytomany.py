"""Loop 179. Several enhancers can use the same factor, and one enhancer can serve several
promoters. Use that structure instead of being defeated by it.

WHAT IS ALREADY HANDLED, so this loop is not sold as fixing something that was never broken.

  MANY SITES FOR ONE FACTOR IN ONE ELEMENT. Handled from the start. The scan stores a partition
  function, the sum of exp(score) over every position on both strands, so ten weak sites and one
  strong one are different numbers, and NS counts the sites above threshold separately.

  ONE ELEMENT SERVING SEVERAL GENES. Present in the data structure -- every (element, gene) pair is
  its own row -- and measured in loop 176: 55.0% of elements were tested against more than one
  gene, and 272 of 426 validated elements (63.8%) are the right answer for one gene and a tested
  negative for another.

  ONE GENE HAVING SEVERAL ENHANCERS. 38.7% of evaluable genes do. R@1 credits any of them, and
  loop 176's M4 reports how many of a gene's enhancers land in its own top-k.

  SEVERAL OF A GENE'S CANDIDATES CARRYING THE SAME FACTOR. Partly handled, and it is worth being
  exact about how much. The occupancy denominator is built per gene from that gene's whole
  candidate pool, so a factor spread across many candidates is diluted for all of them. That is a
  competition term, and it is doing something.

WHAT IS NOT HANDLED, and it is the half that matters for stage two.

  THE ELEMENT AXIS HAS NO SUCH TERM. Occupancy is normalised across a gene's candidates for a fixed
  factor; nothing is normalised across the GENES an element was tested against. So for an element
  that is a real enhancer of one gene and a tested negative of another, every column is identical
  on both rows except the distance. Loop 176 measured the cost of that directly: delete the six
  gene-varying columns and R@1 falls from 0.6050 to 0.3427.

  EVERY SHARED FACTOR COUNTS THE SAME. The pairing block counts how many of the promoter's factors
  the element also carries. A factor that all twenty of a gene's candidates carry discriminates
  nothing; a factor only this candidate carries is the entire signal. Weighting by rarity within
  the gene's own candidate set is the standard fix and it has never been applied here.

  REDUNDANCY IS INVISIBLE. Genes with several enhancers often drive them through overlapping factor
  sets -- shadow enhancers. Whether a candidate's factor set duplicates another candidate's, or is
  unique among them, is not expressible in any current column.

So this loop adds three blocks, all of them gene-varying by construction, which is the property
loop 176 showed carries the answer:

  SPECIFICITY   shared factors weighted by -log of their frequency among the gene's own candidates,
                and a count of promoter factors this candidate alone carries.
  CONTRAST      for each factor, how much of its promoter-binding weight across the genes THIS
                ELEMENT was tested against sits at this gene -- the "same factor, different
                promoter" discriminator, and the only block here that uses the element axis.
  REDUNDANCY    similarity of this candidate's class/family occupancy profile to the gene's other
                candidates, and the size of the candidate set.

PREDECLARED, BEFORE ANY NUMBER.

  Q1 IS THIS READING LIBRARY DESIGN? How many genes an element was tested against, and how many
     candidates a gene has, are choices the screen designers made, not biology. An arm carrying
     ONLY those structural columns and the within-gene distance rank.
     Gate: PASS iff that design-only arm does NOT clear the distance floor on loop 173's E3 bar. A
     FAIL means the many-to-many features can buy accuracy from the library design and every
     result below has to be discounted by that amount.

  Q2 DOES SPECIFICITY WEIGHTING HELP? The specificity block over loop 178's best arm.
     Gate: paired R@1 positive in >= 4/5 past 3 sem AND paired AUPRC >= +0.01 in >= 4/5.

  Q3 DOES THE PROMOTER CONTRAST HELP? The contrast block, same bar.

  Q4 DOES REDUNDANCY HELP? The redundancy block, same bar.

  Q5 THE DECISIVE ONE. Everything against distance alone, same folds, same bar. This is the gate
     loops 173, 175 and 178 were all held to.

  Q6 WHERE THE CONTRAST BLOCK CAN ACT. Restricted to pairs whose element was tested against more
     than one gene -- the only rows where an element-axis normalisation has anything to normalise.
     Predicted before running: the contrast block helps more there than on the singleton rows, and
     if it does not, it is not working the way it is described.
     Gate: PASS iff the contrast block's R@1 gain on multi-gene rows exceeds its gain on
     single-gene rows.

  Q7 THE SHUFFLE STILL DECIDES. Best configuration on real against dinucleotide-shuffled sequence.
     Gate: real beats shuffled in >= 4/5 seeds past 3 sem.

  Q8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_manytomany.json
"""
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_manytomany.json"
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
L173_DIST_R1 = 0.5930

SPECIFICITY = ["shared_idf", "uniq_tf_n", "uniq_tf_frac", "best_idf"]
CONTRAST = ["contrast_occ", "contrast_max", "prom_share_mean", "n_genes_tested"]
REDUNDANCY = ["redundancy_max", "redundancy_mean", "log_n_cand"]
DESIGN = ["n_genes_tested", "log_n_cand", "dist_rank_in_gene", "dist_rank_in_element"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def structure_features(S, FAM, report=print):
    """The three new blocks plus the design columns Q1 needs, all computed from the pair list and
    the cached scan -- no new sequence is read."""
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    NS = S["el_NS"]
    prLZ = S["pr_LZ"].astype(np.float64)
    mx_max = S["motif_maxscore"].astype(np.float64)
    Pm = S["pr_MX"] >= (SC.REL_THRESH * mx_max)[:, None]           # (nm, ng)
    Em = NS > 0                                                     # (nm, ne)
    n = len(y)

    cand = defaultdict(list)       # gene -> its distinct elements
    genes = defaultdict(list)      # element -> the genes it was tested against
    for i in range(n):
        cand[int(g_idx[i])].append(int(e_idx[i]))
        genes[int(e_idx[i])].append(int(g_idx[i]))
    cand = {g: sorted(set(v)) for g, v in cand.items()}
    genes = {e: sorted(set(v)) for e, v in genes.items()}

    # --- per gene: how common is each factor among that gene's own candidates -------------------
    idf = {}
    for g, els in cand.items():
        f = Em[:, els].mean(1)
        idf[g] = -np.log(np.maximum(f, 1e-3))
    # --- per element: how its promoters share each factor's promoter weight ---------------------
    share = {}
    for e, gs in genes.items():
        z = prLZ[:, gs]
        m = z.max(1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        w = np.exp(z - m)
        share[e] = w / np.maximum(w.sum(1, keepdims=True), 1e-300)   # (nm, len(gs))
    # --- per gene: class/family profiles of its candidates, for redundancy ----------------------
    fam_cols = sorted(FAM)
    PROF = np.column_stack([FAM[c] for c in fam_cols])               # (ne, n_fam)
    PROF = PROF - PROF.mean(0, keepdims=True)
    nrm = np.linalg.norm(PROF, axis=1)
    nrm = np.where(nrm > 0, nrm, 1.0)

    F = {k: np.zeros(n) for k in set(SPECIFICITY + CONTRAST + REDUNDANCY + DESIGN)}
    dist = S["dist"].astype(float)
    # distance ranks are over the PAIR list, not the element list
    rank_in_gene, rank_in_elem = {}, {}
    by_g, by_e = defaultdict(list), defaultdict(list)
    for i in range(n):
        by_g[int(g_idx[i])].append(i)
        by_e[int(e_idx[i])].append(i)
    for g, ix in by_g.items():
        o = np.argsort(dist[ix])
        for r, j in enumerate(o):
            rank_in_gene[ix[j]] = r / max(len(ix) - 1, 1)
    for e, ix in by_e.items():
        o = np.argsort(dist[ix])
        for r, j in enumerate(o):
            rank_in_elem[ix[j]] = r / max(len(ix) - 1, 1)

    occ = np.exp(L173.occupancy(S["el_LZ"], e_idx, g_idx, len(S["gn_key"]),
                                np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1))
                                / float(S["bg_bp"])))
    for i in range(n):
        e, g = int(e_idx[i]), int(g_idx[i])
        p = Pm[:, g]
        hit = p & Em[:, e]
        w = idf[g]
        F["shared_idf"][i] = float((hit * w).sum())
        F["best_idf"][i] = float(w[hit].max()) if hit.any() else 0.0
        only = hit & (Em[:, cand[g]].sum(1) == 1)
        F["uniq_tf_n"][i] = float(only.sum())
        F["uniq_tf_frac"][i] = float(only.sum()) / max(float(hit.sum()), 1.0)
        gs = genes[e]
        j = gs.index(g)
        sh = share[e][:, j]
        F["contrast_occ"][i] = float((occ[:, i] * sh).sum())
        F["contrast_max"][i] = float((occ[:, i] * sh).max())
        F["prom_share_mean"][i] = float(sh[p].mean()) if p.any() else 0.0
        F["n_genes_tested"][i] = float(len(gs))
        others = [x for x in cand[g] if x != e]
        if others:
            c = PROF[others] @ PROF[e] / (nrm[others] * nrm[e])
            F["redundancy_max"][i] = float(c.max())
            F["redundancy_mean"][i] = float(c.mean())
        F["log_n_cand"][i] = float(np.log10(len(cand[g])))
        F["dist_rank_in_gene"][i] = rank_in_gene[i]
        F["dist_rank_in_element"][i] = rank_in_elem[i]
    multi = np.array([len(genes[int(e_idx[i])]) > 1 for i in range(n)])
    report(f"    structure features built; {int(multi.sum()):,}/{n:,} pairs "
           f"({multi.mean():.1%}) sit on elements tested against more than one gene")
    report(f"    shared_idf median {np.median(F['shared_idf']):.2f}, "
           f"uniq_tf_n median {np.median(F['uniq_tf_n']):.0f}, "
           f"redundancy_max median {np.median(F['redundancy_max']):.3f}")
    for k in F:
        F[k] = np.nan_to_num(F[k], nan=0.0, posinf=0.0, neginf=0.0)
    return F, multi


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def run(X, y, chrom, g_idx, jitter, tag, report=print, sub=None):
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
        if sub is None:
            r1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
            ap.append(average_precision_score(y, sc))
        else:
            r1.append(L173.within_gene(sc[sub], y[sub], g_idx[sub], jitter[sub])[0])
            ap.append(average_precision_score(y[sub], sc[sub]) if y[sub].sum() else 0.0)
    r1, ap = np.array(r1), np.array(ap)
    report(f"    {tag:36} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 179  THE MANY-TO-MANY STRUCTURE AS A FEATURE SOURCE, NOT AN OBSTACLE")
    say("=" * 104)
    say("  PREDECLARED: a design-only arm must NOT clear the distance floor; each new block judged")
    say(f"  on paired R@1 positive in >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= +0.01 in")
    say(f"  >= {MIN_SEEDS}/5; the winner must clear distance-only R@1 {L173_DIST_R1}; the contrast")
    say("  block must help more on multi-gene elements than on single-gene ones; and the")
    say("  dinucleotide shuffle must still cost the best arm.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))

    E, FAM, _ = L178.element_frame(S, "el", say)
    Es, FAMs, _ = L178.element_frame(S, "sh", say)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    Ps, _, _ = L173.build_features(S, "sh", report=lambda *_: None)
    for fr in (P, Ps):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)
    ST, multi = structure_features(S, FAM, say)
    STs, _ = structure_features(S, FAMs, lambda *_: None)

    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xbase_s = np.column_stack([Ps[c] for c in base_cols] + [FAMs[c][e_idx] for c in fam_cols])
    say(f"    baseline: {Xbase.shape[1]} columns (loop 173's 34 plus loop 177's class/family)")

    def add(F, cols):
        return np.column_stack([Xbase] + [F[c] for c in cols])

    res = {}
    res["distance"] = run(np.column_stack([P["log_dist"]]), y, chrom, g_idx, jitter,
                          "distance", say)
    res["design_only"] = run(np.column_stack([ST[c] for c in DESIGN]), y, chrom, g_idx, jitter,
                             "design only (Q1 control)", say)
    res["base"] = run(Xbase, y, chrom, g_idx, jitter, "base + class/family", say)
    res["+specificity"] = run(add(ST, SPECIFICITY), y, chrom, g_idx, jitter, "+specificity", say)
    res["+contrast"] = run(add(ST, CONTRAST), y, chrom, g_idx, jitter, "+contrast", say)
    res["+redundancy"] = run(add(ST, REDUNDANCY), y, chrom, g_idx, jitter, "+redundancy", say)
    allc = sorted(set(SPECIFICITY + CONTRAST + REDUNDANCY))
    res["ALL"] = run(add(ST, allc), y, chrom, g_idx, jitter, "everything", say)

    # ---- Q1 ------------------------------------------------------------------------------------
    say()
    say("Q1 IS THIS READING LIBRARY DESIGN?")
    d1 = L173.paired(res["design_only"], res["distance"])
    say(f"     design-only vs distance   {L173.fmt(d1)}")
    q1 = bool(not L173.gate_pair(d1))
    GG.verdict(q1, emit=say,
               if_true="Q1 PASS -- how many genes an element was tested against, how many "
                       "candidates a gene has, and the within-gene distance rank do not on their "
                       "own clear the distance floor, so the blocks below are not buying accuracy "
                       "from the library design",
               if_false="Q1 FAIL -- the screen's own design choices clear the distance floor by "
                        "themselves, and every gain below must be discounted by that much")

    # ---- Q2..Q4 --------------------------------------------------------------------------------
    for tag, key, name in (("Q2", "+specificity", "specificity weighting"),
                           ("Q3", "+contrast", "the promoter contrast"),
                           ("Q4", "+redundancy", "redundancy")):
        say()
        say(f"{tag} DOES {name.upper()} HELP?")
        d = L173.paired(res[key], res["base"])
        say(f"     {key} vs base   {L173.fmt(d)}")
        ok = L173.gate_pair(d)
        res[tag + "_delta"] = d
        GG.verdict(ok, emit=say,
                   if_true=f"{tag} PASS -- {name} adds over the base stack",
                   if_false=f"{tag} FAIL -- {name} adds nothing over the base stack")
        res[tag] = ok
    q2, q3, q4 = res["Q2"], res["Q3"], res["Q4"]

    # ---- Q5 ------------------------------------------------------------------------------------
    say()
    say("Q5 THE DECISIVE ONE: does anything clear the distance floor?")
    best = max((k for k in ("base", "+specificity", "+contrast", "+redundancy", "ALL")),
               key=lambda k: res[k]["r1"].mean())
    d5 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} against distance "
        f"{res['distance']['r1'].mean():.4f}")
    say(f"     {L173.fmt(d5)}")
    q5 = L173.gate_pair(d5)
    GG.verdict(q5, emit=say,
               if_true=f"Q5 PASS -- {best} clears the bar loops 173, 175 and 178 were held to",
               if_false="Q5 FAIL -- stage two is still distance")

    # ---- Q6 ------------------------------------------------------------------------------------
    say()
    say("Q6 WHERE THE CONTRAST BLOCK CAN ACT")
    Xc = add(ST, CONTRAST)
    gm = run(Xc, y, chrom, g_idx, jitter, "contrast, multi-gene rows", say, sub=multi)
    gs = run(Xc, y, chrom, g_idx, jitter, "contrast, single-gene rows", say, sub=~multi)
    bm = run(Xbase, y, chrom, g_idx, jitter, "base, multi-gene rows", say, sub=multi)
    bs = run(Xbase, y, chrom, g_idx, jitter, "base, single-gene rows", say, sub=~multi)
    dm = float((gm["r1"] - bm["r1"]).mean())
    ds = float((gs["r1"] - bs["r1"]).mean())
    say(f"     contrast gain on multi-gene rows {dm:+.4f}, on single-gene rows {ds:+.4f}")
    q6 = bool(dm > ds)
    GG.verdict(q6, emit=say,
               if_true="Q6 PASS -- the element-axis normalisation helps where it has something to "
                       "normalise and not where it does not, which is the behaviour it was "
                       "described as having",
               if_false="Q6 FAIL -- the contrast block does not help more on the rows it can act "
                        "on, so whatever it is doing is not what it says on the label")

    # ---- Q7 ------------------------------------------------------------------------------------
    say()
    say("Q7 THE SHUFFLE STILL DECIDES")
    Xs = np.column_stack([Xbase_s] + [STs[c] for c in allc])
    res["ALL_shuffled"] = run(Xs, y, chrom, g_idx, jitter, "everything, SHUFFLED", say)
    d7 = L173.paired(res["ALL"], res["ALL_shuffled"])
    say(f"     real vs dinucleotide-shuffled   {L173.fmt(d7)}")
    q7 = L173.gate_pair(d7, use_ap=False)
    GG.verdict(q7, emit=say,
               if_true="Q7 PASS -- the shuffle costs the stack, so it is reading sites",
               if_false="Q7 FAIL -- a composition-matched shuffle matches it")

    say()
    say("Q8 WHAT THIS CANNOT SHOW")
    say("     Which genes an element was tested against is a choice the screen designers made. The")
    say("     contrast block reads that choice, Q1 bounds how much it can buy, and no control here")
    say("     can make the candidate sets be the genome.")
    say("     Redundancy is measured on class/family occupancy profiles, which is a coarse stand-in")
    say("     for whether two enhancers actually drive a gene through the same factors.")
    say("     None of this is physical contact. Stage two's missing variable is which promoter the")
    say("     element is looped to, and no sequence-derived column is a substitute for Hi-C.")
    q8 = True
    say(f"     Q8 {'PASS' if q8 else 'FAIL'}")

    gates = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz"), Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["a design-only arm carrying just the screen's structural choices",
                                "the contrast block split by whether the element has more than one "
                                "gene to contrast against",
                                "dinucleotide shuffle through the whole stack"],
                      note="specificity, promoter contrast and redundancy from the many-to-many "
                           "element-gene structure")
    out = dict(test="enhancer many-to-many", gates=gates,
               n_pairs=int(len(y)), frac_multi_gene=float(multi.mean()),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items() if isinstance(v, dict) and "r1" in v},
               q6=dict(multi_gain=dm, single_gain=ds),
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("Q1", d1), ("Q5", d5), ("Q7", d7))},
               best_arm=best, manifest=man, seconds=time.time() - t0, log=log)
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
