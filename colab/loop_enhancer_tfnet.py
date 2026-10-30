"""Loop 183. Use the project's own TF regulatory network: does an element carry sites for the
factors already known to regulate this gene?

WHY THIS IS DIFFERENT FROM EVERY PAIRING FEATURE SO FAR. Loop 173's pairing block asked whether an
element carries sites for the factors whose motifs appear in the gene's PROMOTER SEQUENCE. That is
a sequence-derived guess at which factors run the gene, and it failed every gate it faced. The
network answers the same question from evidence instead of from a 1 kb window: CollecTRI's curated
causal edges, DoRothEA's A-D binding tier, and a third block of ChIP-derived edges, 612,133 in all
over 16,492 gene symbols, already assembled and tiered in this project.

The feature is then the natural one and it is gene-varying, which is the property loop 176 showed
carries everything on this task: for the pair (element, gene), how much of the element's occupancy
belongs to factors that are KNOWN regulators of that gene.

WHAT THE JOIN ACTUALLY SUPPORTS, measured before the gates so no arm is read as stronger than its
coverage. 1,922 of 2,205 benchmark genes (87.2%) are on the network roster and all 736 JASPAR
matrices map to it. But the tiers are not interchangeable:

    curated causal (CollecTRI)   median 1 regulator per gene, 53.8% of genes have any, and only
                                 46.7% have one that ALSO carries a matrix here
    binding (DoRothEA A-D)       median 10 regulators, 99.1% of genes have any
    third block (ChIP roster)    median 2, 57.4%

So the curated tier is the better evidence and the thinner column, and the binding tier is the
reverse. Both are built, W4 puts them head to head, and the curated arm is labelled
coverage-limited wherever it appears.

THE STRUCTURAL POINT THAT SHAPES W2. A feature computed from the gene alone -- how many regulators
it has, how well studied it is -- is CONSTANT across that gene's candidate elements, so it cannot
change within-gene R@1 at all. It can move pooled AUPRC, and that is exactly the channel through
which "well-studied genes have more validated enhancers" would enter. W2 therefore gates on AUPRC,
not on R@1, because on R@1 the control is inert by construction and a pass would mean nothing.

PREDECLARED, BEFORE ANY NUMBER.

  W1 DOES THE JOIN SUPPORT THE FEATURE? Coverage of the benchmark genes by each tier, counting only
     regulators that also carry a matrix here.
     Gate: PASS iff the binding tier gives at least 80% of evaluable genes a regulator with a
     matrix. The curated tier's coverage is reported, not gated, and its arm is labelled by it.

  W2 IS THE NETWORK ENCODING HOW WELL STUDIED A GENE IS? An arm carrying only gene-level network
     columns -- regulator counts per tier -- plus distance.
     Gate: PASS iff that arm does NOT clear the AUPRC bar over distance alone. On R@1 it is inert
     by construction and that is stated rather than counted as evidence.

  W3 DOES THE REGULATOR-MATCH BLOCK ADD? The base stack plus the network match columns against the
     base stack.
     Gate: paired per-seed R@1 positive in >= 4/5 and past 3 sem, AND paired AUPRC >= +0.01 in
     >= 4/5 -- loop 173's E3 bar, unchanged since loop 173.

  W4 CURATED CAUSAL EDGES, OR BINDING EDGES? The two tiers entered separately, each over the base
     stack.
     Gate: PASS iff the curated tier's increment exceeds the binding tier's by more than 3 sem of
     the paired difference. A pass says the feature is about regulation; a fail says it is about
     occupancy, which the sequence columns already measure.

  W5 WHICH FACTORS, OR HOW MANY? The regulator sets permuted across genes, so every gene keeps its
     regulator COUNT exactly and loses the identity of its regulators.
     Gate: PASS iff real beats permuted on R@1 in >= 4/5 seeds past 3 sem.

  W6 THE DECISIVE ONE. The best arm against distance alone, identical folds.
     Gate: same bar as W3. This is the gate loops 173, 175, 178, 179, 181 and 182 were all held to.

  W7 THE SHUFFLE, on the arm selected by measured R@1, with the selection printed before the
     comparison.
     Gate: real beats dinucleotide-shuffled in >= 4/5 seeds past 3 sem.

  W8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_tfnet.json
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
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_grammar as L173         # noqa: E402
import loop_enhancer_potency as L178         # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_tfnet.json"
BUNDLE = Path("colab/data/net_bundle.json.gz")
SEEDS = L173.SEEDS
NFOLD = 5
MIN_SEEDS = 4
MIN_COVER = 0.80
L173_DIST_R1 = 0.5930
# row ranges from outputs/orphan/cell_tfnet.json, which partitioned `reg` by row order
TIERS = {"cur": (0, 55716), "bind": (55716, 278405), "unk": (278405, None)}

CUR_COLS = ["cur_occ", "cur_n", "cur_frac", "cur_best", "cur_idf", "cur_act", "cur_rep"]
BIND_COLS = ["bind_occ", "bind_n", "bind_frac", "bind_best", "bind_idf"]
UNK_COLS = ["unk_occ", "unk_n", "unk_frac"]
GENE_COLS = ["n_reg_cur", "n_reg_bind", "n_reg_unk"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def network(S, report=print, perm_seed=None):
    """Regulator sets per benchmark gene, restricted to factors that carry a matrix here.

    `perm_seed` permutes which gene gets which regulator set, preserving every gene's regulator
    COUNT exactly -- W5's control."""
    nb = json.load(gzip.open(BUNDLE))
    names, reg = nb["names"], nb["reg"]
    idx = {n.upper(): i for i, n in enumerate(names)}
    dom = TD.load()
    mid = [str(m) for m in S["motif_ids"]]
    mname = [(dom.get(m, {}).get("name") or "").upper().split("::")[0] for m in mid]
    gene_of_matrix = {}
    for k, nm in enumerate(mname):
        if nm in idx:
            gene_of_matrix.setdefault(idx[nm], []).append(k)
    gsym = [str(k).split(":")[-1].upper() for k in S["gn_key"]]
    grow = [idx.get(g, -1) for g in gsym]

    n_tgt = Counter()
    sets = {t: defaultdict(dict) for t in TIERS}
    for t, (lo, hi) in TIERS.items():
        for r in reg[lo:(hi if hi is not None else len(reg))]:
            tf, tg = int(r[0]), int(r[1])
            n_tgt[tf] += 1
            for k in gene_of_matrix.get(tf, ()):
                sets[t][tg][k] = int(r[2]) if len(r) > 2 else 0
    ntot = max(len(names), 1)
    idf = np.array([np.log(ntot / max(n_tgt.get(g, 0) + 1, 1)) for g in range(len(names))])
    matrix_idf = np.zeros(len(mid))
    for g, ks in gene_of_matrix.items():
        for k in ks:
            matrix_idf[k] = idf[g]
    if perm_seed is not None:
        rng = np.random.default_rng(perm_seed)
        for t in TIERS:
            rows = sorted(sets[t])
            vals = [sets[t][r] for r in rows]
            order = rng.permutation(len(rows))
            sets[t] = {rows[i]: vals[order[i]] for i in range(len(rows))}
    return sets, grow, matrix_idf, gene_of_matrix


def net_features(S, occ, sets, grow, matrix_idf, report=print):
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    NS = S["el_NS"]
    n = len(e_idx)
    F = {k: np.zeros(n) for k in CUR_COLS + BIND_COLS + UNK_COLS + GENE_COLS}
    Em = NS > 0
    for i in range(n):
        e, g = int(e_idx[i]), int(g_idx[i])
        gr = grow[g]
        for t, pre in (("cur", "cur"), ("bind", "bind"), ("unk", "unk")):
            d = sets[t].get(gr, {}) if gr >= 0 else {}
            F[f"n_reg_{t}"][i] = len(d)
            if not d:
                continue
            ks = np.fromiter(d.keys(), int, len(d))
            o = occ[ks, i]
            hit = Em[ks, e]
            F[f"{pre}_occ"][i] = float(o.sum())
            F[f"{pre}_n"][i] = float(hit.sum())
            F[f"{pre}_frac"][i] = float(hit.sum()) / len(ks)
            if pre in ("cur", "bind"):
                F[f"{pre}_best"][i] = float(o.max())
                F[f"{pre}_idf"][i] = float((o * matrix_idf[ks]).sum())
            if pre == "cur":
                sg = np.fromiter(d.values(), int, len(d))
                F["cur_act"][i] = float(o[sg > 0].sum()) if (sg > 0).any() else 0.0
                F["cur_rep"][i] = float(o[sg < 0].sum()) if (sg < 0).any() else 0.0
    for k in ("cur_occ", "cur_best", "cur_idf", "cur_act", "cur_rep",
              "bind_occ", "bind_best", "bind_idf", "unk_occ"):
        F[k] = np.log10(np.maximum(F[k], 1e-300))
    for k in GENE_COLS:
        F[k] = np.log10(1.0 + F[k])
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
    report(f"    {tag:40} R@1 {r1.mean():.4f} +/- {r1.std(ddof=1)/np.sqrt(len(SEEDS)):.4f}   "
           f"AUPRC {ap.mean():.4f}")
    return dict(r1=r1, ap=ap, mrr=np.zeros(len(SEEDS)))


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 183  THE TF REGULATORY NETWORK: does the element carry sites for THIS gene's factors?")
    say("=" * 104)
    say(f"  PREDECLARED: the binding tier must cover >= {MIN_COVER:.0%} of evaluable genes with a")
    say("  regulator that carries a matrix here; the gene-only control is gated on AUPRC because on")
    say("  R@1 it is inert by construction; every arm on loop 173's E3 bar -- paired R@1 positive")
    say(f"  in >= {MIN_SEEDS}/5 past 3 sem AND paired AUPRC >= +0.01 in >= {MIN_SEEDS}/5; the")
    say("  curated tier must beat the binding tier for the feature to be about regulation; the")
    say("  regulator sets must not be permutable at no cost; the winner must clear distance-only")
    say(f"  R@1 {L173_DIST_R1}; and the shuffle runs on the arm selected by measured R@1.")
    say()

    S = SC.load(say)
    y = S["y"].astype(int)
    e_idx, g_idx = S["e_idx"], S["g_idx"]
    chrom = np.array([str(c) for c in S["chrom"]])
    jitter = np.random.default_rng(L173.TIE_SEED).uniform(0, 1e-9, size=len(y))

    bg = np.exp(L173._logsumexp(S["bg_LZ"].astype(np.float64), axis=1)) / float(S["bg_bp"])
    occ = np.exp(L173.occupancy(S["el_LZ"], e_idx, g_idx, len(S["gn_key"]), bg))
    occ_s = np.exp(L173.occupancy(S["sh_LZ"], e_idx, g_idx, len(S["gn_key"]), bg))

    sets, grow, midf, gom = network(S, say)
    say(f"    {len(gom)} network genes carry one of the {len(S['motif_ids'])} matrices")

    # ---- W1 ------------------------------------------------------------------------------------
    say()
    say("W1 DOES THE JOIN SUPPORT THE FEATURE?")
    cand = defaultdict(list)
    pos = Counter()
    for i in range(len(y)):
        cand[int(g_idx[i])].append(i)
        if y[i]:
            pos[int(g_idx[i])] += 1
    ev = sorted(g for g in cand if len(cand[g]) >= 2 and pos[g] > 0)
    cov = {}
    for t in TIERS:
        c = sum(1 for g in ev if grow[g] >= 0 and len(sets[t].get(grow[g], {})) > 0)
        cov[t] = c / max(len(ev), 1)
        med = np.median([len(sets[t].get(grow[g], {})) for g in ev if grow[g] >= 0])
        say(f"     {t:5} tier: {c}/{len(ev)} evaluable genes ({cov[t]:.1%}) have a regulator with "
            f"a matrix here; median {med:.0f}")
    w1 = bool(cov["bind"] >= MIN_COVER)
    GG.verdict(w1, emit=say,
               if_true=f"W1 PASS -- the binding tier reaches {cov['bind']:.1%}; the curated tier at "
                       f"{cov['cur']:.1%} is the better evidence and the thinner column, and its "
                       f"arm is labelled coverage-limited",
               if_false=f"W1 FAIL -- the binding tier covers only {cov['bind']:.1%}, so the "
                        f"regulator-match columns would be mostly zeros")

    # ---- features --------------------------------------------------------------------------
    say()
    say("   building features")
    N = net_features(S, occ, sets, grow, midf, say)
    Ns = net_features(S, occ_s, sets, grow, midf, lambda *_: None)
    E, FAM, _ = L178.element_frame(S, "el", say)
    Es, FAMs, _ = L178.element_frame(S, "sh", say)
    P, _, _ = L173.build_features(S, "el", report=lambda *_: None)
    Ps, _, _ = L173.build_features(S, "sh", report=lambda *_: None)
    for fr in (P, Ps):
        for c in fr:
            fr[c] = np.nan_to_num(fr[c], nan=0.0, posinf=0.0, neginf=0.0)
    base_cols = [c for b in L173.ARMS["FULL"] for c in L173.BLOCKS[b]]
    fam_cols = sorted(FAM)
    Xbase = np.column_stack([P[c] for c in base_cols] + [FAM[c][e_idx] for c in fam_cols])
    Xbase_s = np.column_stack([Ps[c] for c in base_cols] + [FAMs[c][e_idx] for c in fam_cols])
    Xd = np.column_stack([P["log_dist"]])
    ALL = CUR_COLS + BIND_COLS + UNK_COLS
    say(f"    base {Xbase.shape[1]} columns; network block {len(ALL)}")

    res = {}
    res["distance"] = run(Xd, y, chrom, g_idx, jitter, "distance", say)
    res["gene_only"] = run(np.column_stack([Xd] + [N[c] for c in GENE_COLS]), y, chrom, g_idx,
                           jitter, "distance + gene-level counts (W2 control)", say)
    res["base"] = run(Xbase, y, chrom, g_idx, jitter, "base stack", say)
    res["+curated"] = run(np.column_stack([Xbase] + [N[c] for c in CUR_COLS]), y, chrom, g_idx,
                          jitter, "+ curated causal regulators", say)
    res["+binding"] = run(np.column_stack([Xbase] + [N[c] for c in BIND_COLS]), y, chrom, g_idx,
                          jitter, "+ binding-tier regulators", say)
    res["+network"] = run(np.column_stack([Xbase] + [N[c] for c in ALL]), y, chrom, g_idx,
                          jitter, "+ the whole network block", say)

    # ---- W2 ------------------------------------------------------------------------------------
    say()
    say("W2 IS THE NETWORK ENCODING HOW WELL STUDIED A GENE IS?")
    d2 = L173.paired(res["gene_only"], res["distance"])
    say(f"     gene-level counts + distance vs distance   {L173.fmt(d2)}")
    say("     (R@1 is inert here by construction: a per-gene constant cannot reorder that gene's "
        "own candidates)")
    w2 = bool(d2["n_ap_pass"] < MIN_SEEDS)
    GG.verdict(w2, emit=say,
               if_true="W2 PASS -- regulator counts alone do not clear the AUPRC bar, so the block "
                       "below is not buying accuracy from how well studied a gene is",
               if_false="W2 FAIL -- gene-level regulator counts clear the AUPRC bar on their own, "
                        "so part of any gain below is study depth and not regulation")

    # ---- W3, W4 --------------------------------------------------------------------------------
    say()
    say("W3 DOES THE REGULATOR-MATCH BLOCK ADD?")
    d3 = L173.paired(res["+network"], res["base"])
    say(f"     +network vs base   {L173.fmt(d3)}")
    w3 = L173.gate_pair(d3)
    GG.verdict(w3, emit=say,
               if_true="W3 PASS -- matching the element's sites against this gene's KNOWN "
                       "regulators adds over the sequence stack",
               if_false="W3 FAIL -- knowing which factors run the gene does not help pick which "
                        "element runs it")

    say()
    say("W4 CURATED CAUSAL EDGES, OR BINDING EDGES?")
    dc = L173.paired(res["+curated"], res["base"])
    db = L173.paired(res["+binding"], res["base"])
    say(f"     curated increment  {L173.fmt(dc)}   (coverage {cov['cur']:.1%})")
    say(f"     binding increment  {L173.fmt(db)}   (coverage {cov['bind']:.1%})")
    dd = res["+curated"]["r1"] - res["+binding"]["r1"]
    sem = dd.std(ddof=1) / np.sqrt(len(dd))
    say(f"     curated minus binding   {dd.mean():+.4f} +/- {sem:.4f} ({int((dd > 0).sum())}/5 up)")
    w4 = bool((dd > 0).sum() >= MIN_SEEDS and dd.mean() > 3 * sem)
    GG.verdict(w4, emit=say,
               if_true="W4 PASS -- curated causal edges beat binding edges, so the feature is "
                       "about regulation rather than occupancy",
               if_false="W4 FAIL -- the curated tier does not beat the binding tier, so whatever "
                        "the block reads is occupancy, which the sequence columns already measure")

    # ---- W5 ------------------------------------------------------------------------------------
    say()
    say("W5 WHICH FACTORS, OR HOW MANY?")
    pr1, pap = [], []
    for s in SEEDS:
        ps, _, _, _ = network(S, lambda *_: None, perm_seed=6000 + s)
        Np = net_features(S, occ, ps, grow, midf, lambda *_: None)
        Xp = np.column_stack([Xbase] + [Np[c] for c in ALL])
        fold = L173.folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0:
                continue
            m = gbm(s)
            m.fit(np.nan_to_num(Xp[tr]), y[tr])
            sc[te] = m.predict_proba(np.nan_to_num(Xp[te]))[:, 1]
        pr1.append(L173.within_gene(sc, y, g_idx, jitter)[0])
        pap.append(average_precision_score(y, sc))
    res["+network_perm"] = dict(r1=np.array(pr1), ap=np.array(pap), mrr=np.zeros(len(SEEDS)))
    say(f"    {'+network, regulator sets permuted':40} R@1 {np.mean(pr1):.4f}   "
        f"AUPRC {np.mean(pap):.4f}")
    d5 = L173.paired(res["+network"], res["+network_perm"])
    say(f"     real vs permuted   {L173.fmt(d5)}")
    w5 = L173.gate_pair(d5, use_ap=False)
    GG.verdict(w5, emit=say,
               if_true="W5 PASS -- WHICH factors regulate the gene matters, not merely how many",
               if_false="W5 FAIL -- permuting the regulator sets across genes costs nothing, so "
                        "the block reads regulator COUNT and not regulator identity")

    # ---- W6, W7 --------------------------------------------------------------------------------
    say()
    say("W6 THE DECISIVE ONE")
    best = max((k for k in res if k not in ("distance", "gene_only", "+network_perm")),
               key=lambda k: res[k]["r1"].mean())
    d6 = L173.paired(res[best], res["distance"])
    say(f"     best arm {best} at R@1 {res[best]['r1'].mean():.4f} / AUPRC "
        f"{res[best]['ap'].mean():.4f} against distance {res['distance']['r1'].mean():.4f} / "
        f"{res['distance']['ap'].mean():.4f}")
    say(f"     {L173.fmt(d6)}")
    w6 = L173.gate_pair(d6)
    GG.verdict(w6, emit=say,
               if_true=f"W6 PASS -- {best} clears the bar every stage-two loop has been held to",
               if_false="W6 FAIL -- stage two is still distance")

    say()
    say("W7 THE SHUFFLE, ON THE ARM SELECTED BY MEASURED R@1")
    say(f"     selected arm: {best}")
    cols = {"base": [], "+curated": CUR_COLS, "+binding": BIND_COLS, "+network": ALL}[best]
    Xsh = np.column_stack([Xbase_s] + [Ns[c] for c in cols]) if cols else Xbase_s
    res["best_shuffled"] = run(Xsh, y, chrom, g_idx, jitter, f"{best}, SHUFFLED elements", say)
    d7 = L173.paired(res[best], res["best_shuffled"])
    say(f"     real vs dinucleotide-shuffled   {L173.fmt(d7)}")
    w7 = L173.gate_pair(d7, use_ap=False)
    GG.verdict(w7, emit=say,
               if_true="W7 PASS -- on the arm that won, real sequence beats a composition-matched "
                       "shuffle",
               if_false="W7 FAIL -- the shuffle matches the winning arm")

    say()
    say("W8 WHAT THIS CANNOT SHOW")
    say("     A TF-gene edge says nothing about WHICH element the factor acts through, so this")
    say("     block can only reweight the element's own motif content. It cannot supply the")
    say("     element-to-promoter correspondence that stage two actually lacks.")
    say("     CollecTRI and DoRothEA are literature-derived. A gene studied through its enhancers")
    say("     may have edges that exist because of the very experiments this benchmark scores, and")
    say("     W2 bounds only the gene-level channel of that, not the per-factor one.")
    say("     The third tier's provenance is recorded as unidentified in cell_tfnet.json, best")
    say("     guess ENCODE ChIP by roster match, and it is carried but never leaned on.")
    say("     Loop 178's element-intrinsic ceiling of R@1 0.4422 and loop 181's 5 kb resolution")
    say("     limit still bound everything here.")
    w8 = True
    say(f"     W8 {'PASS' if w8 else 'FAIL'}")

    gates = {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5, "W6": w6, "W7": w7, "W8": w8}
    man = RM.manifest(inputs=[BUNDLE, Path("colab/data/tf_domains.json")],
                      available=int(len(y)), used=int(len(y)), selection="loop 173's pairs",
                      seed=L173.TIE_SEED,
                      controls=["a gene-level-only arm gated on AUPRC, where it is not inert",
                                "the curated tier put head to head with the binding tier",
                                "regulator sets permuted across genes, counts held exactly",
                                "the shuffle on the arm selected by measured R@1"],
                      note="TF regulatory network as the source of which factors run each gene")
    out = dict(test="enhancer tf network", gates=gates,
               coverage={k: float(v) for k, v in cov.items()},
               n_matrices_on_roster=len(gom),
               arms={k: {m: [float(x) for x in v[m]] for m in ("r1", "ap")}
                     for k, v in res.items()},
               deltas={k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                           for kk, vv in d.items()}
                       for k, d in (("W2", d2), ("W3", d3), ("W4_curated", dc),
                                    ("W4_binding", db), ("W5", d5), ("W6", d6), ("W7", d7))},
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
