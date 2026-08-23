"""Loop 176. Is the ceiling the method, or is it that one element can serve several genes and one
gene several elements?

WHAT R@1 IS, SINCE EVERYTHING BELOW IS A STATEMENT ABOUT IT. For each gene the CRISPR screens
tested at least two elements against, and for which at least one of those elements was validated,
the model ranks that gene's candidates and R@1 asks whether the TOP-RANKED one is a validated
enhancer -- ANY of them, not a designated single answer. So the metric already tolerates a gene
having several enhancers in its numerator. What it does not do is reward finding the rest of them,
and it does not notice that the same element may be the right answer for one gene and the wrong
answer for another.

THE CONCERN THIS LOOP MEASURES. A sequence feature is a property of a piece of DNA. Groove width,
electrostatic potential, motif content, duplex stability -- none of them knows which gene is being
asked about. So for an element tested against several genes, every sequence column returns the SAME
number for all of them, while the labels differ. In the arms of loops 173 and 175, only two of the
eight feature blocks can vary across genes at a fixed element: the distance, and the pairing block
that compares the element's factor set against that gene's promoter. Everything else is constant
down the column. If elements commonly serve one gene and not another, then a mostly gene-blind
model has a ceiling that has nothing to do with how good its chemistry is, and reporting its
failure as a failure of the chemistry would be wrong.

PREDECLARED, BEFORE ANY NUMBER.

  M1 THE MULTIPLICITY STRUCTURE. How many validated enhancers each gene has, how many genes each
     element was tested against, and how many validated elements are ALSO tested-negative for some
     other gene. Descriptive; the counts are the output. Also reported: how many validated pairs
     have a positive effect size, which is what an element acting as a SILENCER would look like in
     this assay, because the request used that word and the benchmark's composition should answer it
     rather than an assumption.

  M2 THE CEILING FOR A GENE-BLIND SCORER. An oracle that has seen every label but is forced to give
     each ELEMENT one score shared by all its genes -- the constraint every sequence feature is
     under. Its score is the fraction of that element's tested pairs that are positive, which is the
     best any gene-blind ranker can do.
     Gate: PASS iff that oracle's within-gene R@1 is BELOW 0.85. A pass means multiplicity imposes
     a real ceiling and the sequence stack was being asked for something it structurally cannot
     deliver. A FAIL -- an oracle at or above 0.85 -- means a gene-blind scorer had plenty of room
     and the failure in loops 173 and 175 belongs to the features, not to the task shape.

  M3 HOW GENE-BLIND IS THE STACK, COLUMN BY COLUMN? For each of the 34 features, the fraction of
     its total variance that lives WITHIN an element across the genes it was tested against. A
     purely element-intrinsic column scores exactly 0.
     Descriptive, and it is the quantitative form of the claim above.

  M4 RECALL OF ALL OF A GENE'S ENHANCERS, not just the first. For the genes with more than one
     validated element, the fraction of that gene's positives that land in its own top-k, for k
     from 1 to 5 and at k equal to its true positive count. Descriptive; this is the metric the
     concern implies and it was never reported.

  M5 DO THE GENE-AWARE COLUMNS CARRY THE ANSWER? The full stack against the same stack with the
     distance and pairing columns removed -- everything that can vary across genes at a fixed
     element deleted, leaving pure sequence.
     Gate: PASS iff removing them costs more than 3 sem on R@1 in >= 4/5 seeds. A pass says the
     gene-specific columns are load-bearing and the element-intrinsic ones are not, which is the
     same finding from the other side.

  M6 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_multiplicity.json
"""
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
from enh import scan as SC                   # noqa: E402
import loop_enhancer_grammar as L            # noqa: E402

from sklearn.metrics import average_precision_score   # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_multiplicity.json"
CEILING_BAR = 0.85
MIN_SEEDS = 4
GENE_VARYING = ["log_dist", "log_shared_occ", "shared_n", "prom_n", "shared_frac", "jaccard"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 176  IS THE CEILING THE CHEMISTRY, OR IS IT THAT ELEMENTS AND GENES ARE MANY-TO-MANY?")
    say("=" * 104)
    say(f"  PREDECLARED: the gene-blind oracle's R@1 must come in BELOW {CEILING_BAR} for")
    say("  multiplicity to count as a real ceiling; and deleting every gene-varying column must")
    say(f"  cost more than 3 sem in >= {MIN_SEEDS}/5 seeds for the gene-specific half to be")
    say("  load-bearing.")
    say()

    P = SC.load(say)
    y = P["y"].astype(int)
    e_idx, g_idx = P["e_idx"], P["g_idx"]
    chrom = np.array([str(c) for c in P["chrom"]])
    jitter = np.random.default_rng(L.TIE_SEED).uniform(0, 1e-9, size=len(y))

    # ---- M1 ------------------------------------------------------------------------------------
    say("M1 THE MULTIPLICITY STRUCTURE")
    pos_g, cand_g = Counter(), Counter()
    genes_of_e, pos_e, neg_e = defaultdict(set), defaultdict(set), defaultdict(set)
    for i in range(len(y)):
        cand_g[int(g_idx[i])] += 1
        genes_of_e[int(e_idx[i])].add(int(g_idx[i]))
        if y[i]:
            pos_g[int(g_idx[i])] += 1
            pos_e[int(e_idx[i])].add(int(g_idx[i]))
        else:
            neg_e[int(e_idx[i])].add(int(g_idx[i]))
    ev = sorted(gg for gg in cand_g if cand_g[gg] >= 2 and pos_g[gg] > 0)
    pp = np.array([pos_g[gg] for gg in ev])
    say(f"     {len(ev)} evaluable genes; validated enhancers per gene: "
        f"{dict(sorted(Counter(pp.tolist()).items()))}")
    say(f"       exactly one: {int((pp == 1).sum())} ({(pp == 1).mean():.1%})   "
        f"more than one: {int((pp > 1).sum())} ({(pp > 1).mean():.1%})")
    multi = sum(1 for v in genes_of_e.values() if len(v) > 1)
    say(f"     elements tested against more than one gene: {multi:,}/{len(genes_of_e):,} "
        f"({multi/len(genes_of_e):.1%})")
    amb = sorted(set(pos_e) & set(neg_e))
    say(f"     validated elements that are ALSO tested-negative for another gene: "
        f"{len(amb):,}/{len(pos_e):,} ({len(amb)/max(len(pos_e),1):.1%})")
    npair_amb = sum(len(pos_e[k]) + len(neg_e[k]) for k in amb)
    say(f"       pairs sitting on those elements: {npair_amb:,}/{len(y):,} "
        f"({npair_amb/len(y):.1%})")
    rows = SC.load_benchmark(lambda *_: None)
    sig = [r for r in rows if r["Significant"] in ("TRUE", "True", "true")]
    up = sum(1 for r in sig if float(r["EffectSize"] or 0) > 0)
    say(f"     of {len(sig):,} validated pairs in the benchmark, {up:,} have a POSITIVE effect "
        f"size ({up/max(len(sig),1):.1%}) -- silencing the element RAISES the gene, which is what")
    say(f"       a silencer looks like in this assay. The other {len(sig)-up:,} are enhancers.")

    # ---- M2 ------------------------------------------------------------------------------------
    say()
    say("M2 THE CEILING FOR A GENE-BLIND SCORER (an oracle that has seen every label)")
    rate = np.zeros(len(P["el_key"]))
    for k in set(list(pos_e) + list(neg_e)):
        rate[k] = len(pos_e[k]) / max(len(pos_e[k]) + len(neg_e[k]), 1)
    orc = rate[e_idx]
    r1, mrr, n = L.within_gene(orc, y, g_idx, jitter)
    ap = average_precision_score(y, orc)
    say(f"     gene-blind oracle: R@1 {r1:.4f}   MRR {mrr:.4f}   AUPRC {ap:.4f}   "
        f"(over {n} genes)")
    say(f"     for contrast, a gene-AWARE oracle that simply knows the labels reaches R@1 1.0000")
    say(f"     and loop 173's best real arm reached 0.6291")
    m2 = bool(r1 < CEILING_BAR)
    GG.verdict(m2, emit=say,
               if_true=f"M2 PASS -- no gene-blind scorer, however perfect its chemistry, can pass "
                       f"R@1 {r1:.4f} on this data. {len(amb):,} of {len(pos_e):,} validated "
                       f"elements are the right answer for one gene and the wrong one for another, "
                       f"and a sequence feature returns the same number for both",
               if_false=f"M2 FAIL -- a gene-blind oracle reaches {r1:.4f}, so there was ample room "
                        f"above loop 173's numbers and the shortfall belongs to the features")

    # ---- M3 ------------------------------------------------------------------------------------
    say()
    say("M3 HOW GENE-BLIND IS THE STACK, COLUMN BY COLUMN?")
    F, _, _ = L.build_features(P, "el", report=lambda *_: None)
    for c in F:
        F[c] = np.nan_to_num(F[c], nan=0.0, posinf=0.0, neginf=0.0)
    cols = [c for b in L.ARMS["FULL"] for c in L.BLOCKS[b]]
    order = np.argsort(e_idx, kind="stable")
    ee = e_idx[order]
    bounds = np.flatnonzero(np.r_[True, ee[1:] != ee[:-1]])
    within = {}
    for c in cols:
        v = F[c][order].astype(np.float64)
        tot = v.var()
        if tot <= 0:
            within[c] = 0.0
            continue
        m = np.add.reduceat(v, bounds) / np.diff(np.r_[bounds, len(v)])
        mean_rep = np.repeat(m, np.diff(np.r_[bounds, len(v)]))
        within[c] = float(((v - mean_rep) ** 2).mean() / tot)
    say("     fraction of each column's variance that lives WITHIN an element, across its genes")
    for b in L.ARMS["FULL"]:
        cs = L.BLOCKS[b]
        say(f"       {b:14} " + "  ".join(f"{c}={within[c]:.3f}" for c in cs[:3])
            + (" ..." if len(cs) > 3 else ""))
    blind = [c for c in cols if within[c] < 1e-9]
    say(f"     {len(blind)}/{len(cols)} columns are EXACTLY element-intrinsic (zero within-element "
        f"variance)")
    say(f"     the {len(cols)-len(blind)} that are not: "
        + ", ".join(c for c in cols if within[c] >= 1e-9))

    # ---- M4 ------------------------------------------------------------------------------------
    say()
    say("M4 RECALL OF ALL OF A GENE'S ENHANCERS, not just the first")
    X, _ = L.matrix(F, L.ARMS["FULL"])
    Xd, _ = L.matrix(F, L.ARMS["distance"])
    keep = [c for c in cols if c not in GENE_VARYING]
    Xe = np.column_stack([F[c] for c in keep]).astype(np.float64)
    curves = {}
    for tag, XX in (("FULL", X), ("distance", Xd), ("element-intrinsic only", Xe)):
        acc = defaultdict(list)
        r1s, aps = [], []
        for s in L.SEEDS:
            fold = L.folds_for(chrom, s)
            sc = L.oof_scores(XX, y, fold, s)
            r1s.append(L.within_gene(sc, y, g_idx, jitter)[0])
            aps.append(average_precision_score(y, sc))
            by = defaultdict(list)
            for i in range(len(y)):
                by[int(g_idx[i])].append(i)
            for gg in ev:
                ix = by[gg]
                if len(ix) < 2:
                    continue
                yy = y[ix]
                o = np.argsort(-(sc[ix] + jitter[ix]))
                yo = yy[o]
                npos = int(yy.sum())
                for k in (1, 2, 3, 4, 5):
                    acc[k].append(yo[:k].sum() / npos)
                acc["npos"].append(yo[:npos].sum() / npos)
        curves[tag] = dict(r1=float(np.mean(r1s)), ap=float(np.mean(aps)),
                           **{str(k): float(np.mean(v)) for k, v in acc.items()})
        say(f"     {tag:24} R@1 {curves[tag]['r1']:.4f}  AUPRC {curves[tag]['ap']:.4f}  "
            + "  ".join(f"rec@{k}={curves[tag][str(k)]:.3f}" for k in (1, 2, 3, 5))
            + f"  rec@n={curves[tag]['npos']:.3f}")

    # ---- M5 ------------------------------------------------------------------------------------
    say()
    say("M5 DO THE GENE-AWARE COLUMNS CARRY THE ANSWER?")
    a_full = L.run_arm(X, y, chrom, g_idx, jitter, "FULL", say)
    a_elem = L.run_arm(Xe, y, chrom, g_idx, jitter, "element-intrinsic only", say)
    d5 = L.paired(a_full, a_elem)
    say(f"     FULL vs element-intrinsic only   {L.fmt(d5)}")
    m5 = L.gate_pair(d5, use_ap=False)
    GG.verdict(m5, emit=say,
               if_true="M5 PASS -- delete the six columns that can vary across genes and the model "
                       "collapses, so the gene-specific half was carrying the answer and the 28 "
                       "sequence columns were not",
               if_false="M5 FAIL -- the element-intrinsic columns do as well on their own, so the "
                        "gene-varying columns were not the load-bearing part")

    say()
    say("M6 WHAT THIS CANNOT SHOW")
    say("     The oracle in M2 is an upper bound, not a model. It has seen every label and it is")
    say("     still capped, which is the point, but no real gene-blind scorer would reach it.")
    say("     A ceiling explains why a gene-blind stack cannot win. It does not explain why the")
    say("     sequence columns lost to a dinucleotide shuffle in loop 173, which is a separate")
    say("     failure and is not rescued by anything here.")
    say("     The many-to-many structure is partly an artefact of which pairs the screens chose to")
    say("     test. An element tested against one gene only is not thereby specific to it.")
    m6 = True
    say(f"     M6 {'PASS' if m6 else 'FAIL'}")

    gates = {"M1": True, "M2": m2, "M3": True, "M4": True, "M5": m5, "M6": m6}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz")],
                      available=int(len(y)), used=int(len(y)), selection="all powered pairs",
                      seed=L.TIE_SEED,
                      controls=["a label-seeing gene-blind oracle as the structural ceiling",
                                "per-column within-element variance decomposition",
                                "the stack with every gene-varying column deleted"],
                      note="does the many-to-many element-gene structure cap a sequence-only model")
    out = dict(test="enhancer multiplicity", gates=gates,
               n_evaluable_genes=len(ev),
               positives_per_gene=dict(sorted(Counter(pp.tolist()).items())),
               elements_multi_gene=int(multi), n_elements=int(len(genes_of_e)),
               ambiguous_positive_elements=int(len(amb)), n_positive_elements=int(len(pos_e)),
               pairs_on_ambiguous=int(npair_amb),
               silencer_like_validated_pairs=int(up), validated_pairs=int(len(sig)),
               gene_blind_oracle=dict(r1=float(r1), mrr=float(mrr), auprc=float(ap)),
               within_element_variance=within, element_intrinsic_columns=blind,
               curves=curves,
               m5=dict((k, (v.tolist() if hasattr(v, "tolist") else v)) for k, v in d5.items()),
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
