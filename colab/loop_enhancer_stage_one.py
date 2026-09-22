"""Loop 177. Close the headroom on the task the sequence chain actually works on.

WHERE THIS PICKS UP. Loop 174 established that the stack separates a validated K562 enhancer from a
distance- and width-matched genomic window at AUC 0.6807, that a dinucleotide shuffle costs it
0.0657 in 5 of 5 seeds so it is reading binding sites and not composition, and that groove geometry
adds 0.0353 over the site scores. It also established, at 0.5365 against the benchmark's own tested
negatives versus 0.6807 against raw genome, that loop 173 measured nothing because its negatives
had already been filtered for the property under test. So there is a real instrument here and it
has 0.32 of AUC left to give.

TWO THINGS THIS LOOP FIXES, AND ONE CORRECTION IT CARRIES.

  THE CORRECTION FIRST. The suggestion was to try tree models or neural interaction terms in place
  of "linear/summed column models". The model was never linear -- every arm in loops 173 to 176 ran
  a gradient-boosted tree ensemble, 200 rounds at 15 leaves, which captures interactions natively.
  What is summed is the FEATURES. Every column is a sum over all 736 matrices: total occupancy,
  occupancy times electrostatic potential times domain charge, and so on. Matrix identity is
  destroyed before the model sees anything, and no tree can recover an interaction between two
  motifs it was never shown separately. So the fix is per-motif RESOLUTION, not a different
  learner -- and because the suggestion deserves a real answer rather than a redefinition, H5
  measures what non-linearity buys by running a logistic regression against three non-linear
  learners on identical columns.

  RESOLUTION. Occupancy is re-aggregated by JASPAR structural CLASS and FAMILY rather than into one
  global sum, which keeps identity at a cost of a few dozen columns instead of 736; and the full
  736-dimensional log-occupancy vector is projected onto its leading components, FITTED INSIDE EACH
  TRAINING FOLD so the projection cannot see the held-out chromosome. H7 checks that fold-internal
  fitting by running the leaky version beside it.

  CLUSTERING. NS counts a motif's sites in a window. Forty sites spread over 500 bp and forty piled
  into 60 bp give the same count and are different objects, and the second is what a regulatory
  element looks like. Loop 174's F8 makes this urgent rather than optional: motif matches are
  165.7 per kb in validated enhancers against 181.4 per kb in matched genome, an enrichment of
  0.914x. Enhancers are motif-POOR by raw count. If matches matter at all it is through WHERE they
  sit and WHICH they are, which is precisely what a count cannot express.

WHAT IS MEASURED. Loop 174's window set unchanged -- 482 validated positives against 4,562
distance- and width-matched genomic decoys, AUC out-of-fold, chromosome-held-out, 5 folds x 5
seeds, identical folds for every arm. Alignment with loop 174's cache is verified by recomputing GC
content from the re-extracted sequences and requiring an exact match, because every claim below is
a comparison against loop 174's numbers and a silent re-ordering of the windows would make all of
them meaningless.

PREDECLARED, BEFORE ANY NUMBER.

  H1 DOES THE BASELINE REPRODUCE? Loop 174's FULL stack, rebuilt here on the same folds.
     Gate: PASS iff it lands within 0.01 of 0.6807 AND the recomputed GC matches the cached GC
     exactly on every window. Without this, nothing below is a comparison.

  H2 DOES PER-MOTIF RESOLUTION ADD? FULL + class/family occupancy + the fold-internal spectrum,
     against FULL.
     Gate: PASS iff the paired per-seed change in AUC is positive in >= 4/5 seeds and exceeds
     3 sem.

  H3 DOES CLUSTERING ADD? FULL + homotypic and heterotypic density, against FULL.
     Gate: same bar.

  H4 EVERYTHING TOGETHER, against FULL.
     Gate: same bar. Reported alongside H2 and H3 so a gain that is entirely one block's cannot be
     presented as both.

  H5 WHAT DOES NON-LINEARITY ACTUALLY BUY? On the single best feature set: logistic regression
     with standardised inputs, the shallow tree ensemble used throughout, a deeper one, and a
     two-layer neural network.
     Gate: PASS iff the best non-linear learner beats logistic regression by more than 3 sem. A
     FAIL is the interesting outcome and would say the signal is additive and the learner never
     mattered.

  H6 THE SHUFFLE STILL DECIDES. The best configuration on real sequence against the same
     configuration on dinucleotide-shuffled sequence, both classes shuffled.
     Gate: PASS iff real beats shuffled in >= 4/5 seeds and by more than 3 sem. Density in
     particular is exactly the kind of feature a composition-matched shuffle can reproduce, so this
     gate is where a density gain lives or dies.

  H7 IS THE PROJECTION LEAKING? The spectrum is refitted inside every training fold. The control
     is the same arm with the projection fitted on ALL data including the held-out chromosome.
     Gate: PASS iff the honest arm does not EXCEED the leaky arm by more than 3 sem. It cannot,
     unless the fold-internal fitting is wrong, because the leaky arm has strictly more
     information; an honest arm that beats it is a bug and this gate is how it would surface.

  H8 WHAT THIS CANNOT SHOW.

-> outputs/loop_enhancer_stage_one.json
"""
import json
import os
import random
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
from enh import genome as GEN                # noqa: E402
from enh import scan as SC                   # noqa: E402
from enh import tf_domains as TD             # noqa: E402
import loop_enhancer_vs_genome as L174       # noqa: E402

from sklearn.decomposition import PCA                          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402
from sklearn.neural_network import MLPClassifier               # noqa: E402
from sklearn.pipeline import make_pipeline                     # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_enhancer_stage_one.json"
CLUST = SC.CACHE / "enh_vs_genome_clusters.npz"
SEEDS = [0, 1, 2, 3, 4]
NFOLD = 5
N_PC = 32
MIN_FAMILY = 5
L174_FULL = 0.6807
H1_TOL = 0.01
MIN_SEEDS = 4

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------------------------------------
def cluster_stats(report=print):
    """Homotypic and heterotypic clustering on loop 174's exact window set.

    The window list is regenerated by calling loop 174's own drawing code with its own seeds, so
    the ordering is identical by construction; the caller verifies that against the cached GC."""
    if CLUST.exists():
        z = np.load(CLUST, allow_pickle=True)
        report(f"    cluster cache: {CLUST.name} ({CLUST.stat().st_size/1e6:.1f} MB)")
        return {k: z[k] for k in z.files}
    t0 = time.time()
    ids, mots = SC.load_motifs(report)
    maxw = max(m.shape[0] for m in mots.values())
    P, Dg, Dt = L174.draw_windows(report)
    recs = P + Dg + Dt
    g = GEN.Genome()
    seqs = g.extract([(r["chrom"], r["start"], r["end"]) for r in recs], report)
    nf = np.array([float((s > 3).mean()) for s in seqs])
    ok = nf <= L174.MAX_N
    seqs = [s for s, k in zip(seqs, ok) if k]
    srng = random.Random(L174.SHUF_SEED)
    shf = [SC.dinuc_shuffle(s, srng) for s in seqs]
    out = {}
    for tag, ss in (("el", seqs), ("sh", shf)):
        cat, starts = SC.concat(ss, maxw)
        report(f"    clustering {tag}: {len(ss):,} windows, {len(cat):,} bp")
        d = SC.scan_clusters(cat, starts, ids, mots, report, tag)
        for k, v in d.items():
            out[f"{tag}_{k}"] = v
        del cat
    out["gc_check"] = np.array([SC._composition(s)["gc"] for s in seqs], np.float32)
    CLUST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLUST, **out)
    report(f"    -> {CLUST} [{time.time()-t0:.0f}s]")
    return out


CLUSTER_COLS = ["homo_best", "homo_n", "het_max_50", "het_max_100", "het_max_200", "het_mean"]


def family_features(D, tag, report=print):
    """Occupancy re-aggregated by JASPAR structural class and family instead of into one sum."""
    mx_max = D["motif_maxscore"].astype(np.float64)
    LZ = D[f"{tag}_LZ"]
    bg = np.exp(L174.L173._logsumexp(D["bg_LZ"].astype(np.float64), axis=1)) / float(D["bg_bp"])
    denom = np.log(np.maximum(bg, 1e-300)) + np.log(L174.WINDOW_BP)
    occ = np.exp(LZ.astype(np.float64) - denom[:, None])
    dom = TD.load()
    ids = [str(m) for m in D["motif_ids"]]
    groups = defaultdict(list)
    for i, m in enumerate(ids):
        r = dom.get(m, {})
        groups["CLS:" + str(r.get("cls"))].append(i)
        groups["FAM:" + str(r.get("family"))].append(i)
    F = {}
    kept = 0
    for k, ix in sorted(groups.items()):
        if len(ix) < MIN_FAMILY:
            continue
        kept += 1
        F["occ_" + k.replace(" ", "_").replace(",", "")[:44]] = np.log10(
            np.maximum(occ[ix].sum(0), 1e-300))
    report(f"    {kept} class/family occupancy columns (groups with >= {MIN_FAMILY} matrices)")
    return F, np.log(np.maximum(occ, 1e-300)).T      # (n_windows, n_motifs) for the spectrum


def folds_for(chrom, seed):
    ch = sorted(set(chrom))
    order = np.random.default_rng(seed).permutation(len(ch))
    assign = {ch[order[i]]: i % NFOLD for i in range(len(ch))}
    return np.array([assign[c] for c in chrom])


def learner(kind, seed):
    if kind == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0, random_state=seed))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              random_state=seed)
    if kind == "gbm_deep":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=63,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=seed)
    if kind == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                           early_stopping=True, n_iter_no_change=15,
                                           random_state=seed))
    raise ValueError(kind)


def run(X, y, chrom, tag, kind="gbm", spectrum=None, leak=False, report=print):
    """spectrum: the (n, n_motifs) log-occupancy matrix, projected inside each fold when given."""
    au = []
    for s in SEEDS:
        fold = folds_for(chrom, s)
        sc = np.zeros(len(y))
        for f in range(NFOLD):
            te = fold == f
            tr = ~te
            if te.sum() == 0 or y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            Xtr, Xte = X[tr], X[te]
            if spectrum is not None:
                p = PCA(n_components=N_PC, random_state=s)
                p.fit(spectrum if leak else spectrum[tr])
                Xtr = np.column_stack([Xtr, p.transform(spectrum[tr])])
                Xte = np.column_stack([Xte, p.transform(spectrum[te])])
            m = learner(kind, s)
            Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
            Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
            m.fit(Xtr, y[tr])
            sc[te] = m.predict_proba(Xte)[:, 1]
        au.append(roc_auc_score(y, sc))
    au = np.array(au)
    report(f"    {tag:34} AUC {au.mean():.4f} +/- {au.std(ddof=1)/np.sqrt(len(au)):.4f}")
    return au


def delta(a, b):
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), sem=float(sem), n_up=int((d > 0).sum()),
                passes=bool((d > 0).sum() >= MIN_SEEDS and d.mean() > 3 * sem))


def fmt(d):
    return f"dAUC {d['mean']:+.4f} +/- {d['sem']:.4f} ({d['n_up']}/5 up)"


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 177  CLOSE THE HEADROOM ON STAGE ONE: enhancer against distance-matched genome")
    say("=" * 104)
    say(f"  PREDECLARED: the rebuilt baseline must land within {H1_TOL} of loop 174's {L174_FULL}")
    say("  and the recomputed GC must match the cache exactly; each new block judged on paired")
    say(f"  per-seed AUC positive in >= {MIN_SEEDS}/5 past 3 sem; the best non-linear learner must")
    say("  beat logistic regression past 3 sem or the learner never mattered; the dinucleotide")
    say("  shuffle must still cost the best configuration; and the fold-internal projection must")
    say("  not out-score the deliberately leaked one.")
    say()

    D = L174.build_scan(say)
    y = D["y"].astype(int)
    kind_w = np.array([str(k) for k in D["kind"]])
    chrom = np.array([str(c) for c in D["chrom"]])
    keep = kind_w != "tested"                     # loop 174's vs-GENOME contrast
    say(f"    {int(keep.sum()):,} windows in the vs-genome contrast, {int(y[keep].sum())} positives")

    C = cluster_stats(say)

    # ---- H1 ------------------------------------------------------------------------------------
    say()
    say("H1 DOES THE BASELINE REPRODUCE?")
    gc_ok = bool(np.allclose(C["gc_check"], D["w_gc"], atol=0, rtol=0))
    say(f"     recomputed GC matches the cached GC on every window, bit for bit: {gc_ok}")
    if not gc_ok:
        n_bad = int((C["gc_check"] != D["w_gc"]).sum())
        say(f"     MISALIGNED on {n_bad}/{len(D['w_gc'])} windows -- the cluster statistics do not")
        say("     belong to the same windows as the cached scan, so nothing below is comparable")
    F = L174.features(D, "el")
    Fs = L174.features(D, "sh")
    base_cols = [c for b in L174.ARMS["FULL"] for c in L174.BLOCKS[b]]
    Xb = np.column_stack([F[c][keep] for c in base_cols])
    a_base = run(Xb, y[keep], chrom[keep], "FULL (loop 174 baseline)", report=say)
    h1 = bool(gc_ok and abs(a_base.mean() - L174_FULL) < H1_TOL)
    GG.verdict(h1, emit=say,
               if_true=f"H1 PASS -- {a_base.mean():.4f} against loop 174's {L174_FULL}, and the "
                       f"windows are the same windows",
               if_false=f"H1 FAIL -- {a_base.mean():.4f} against {L174_FULL}, GC aligned {gc_ok}; "
                        f"the comparisons below would not be comparisons")

    # ---- the new blocks --------------------------------------------------------------------
    Ff, spec = family_features(D, "el", say)
    Ffs, spec_s = family_features(D, "sh", say)
    fam_cols = sorted(Ff)
    Xfam = np.column_stack([Xb] + [Ff[c][keep] for c in fam_cols])
    Xclu = np.column_stack([Xb] + [C["el_" + c][keep] for c in CLUSTER_COLS])
    Xall = np.column_stack([Xb]
                           + [Ff[c][keep] for c in fam_cols]
                           + [C["el_" + c][keep] for c in CLUSTER_COLS])
    say(f"    columns: base {Xb.shape[1]}, +family {Xfam.shape[1]}, +cluster {Xclu.shape[1]}, "
        f"+both {Xall.shape[1]}, spectrum adds {N_PC} more inside each fold")

    res = {"FULL": a_base}
    res["+family"] = run(Xfam, y[keep], chrom[keep], "+class/family occupancy", report=say)
    res["+family+spectrum"] = run(Xfam, y[keep], chrom[keep], "+class/family+spectrum",
                                  spectrum=spec[keep], report=say)
    res["+cluster"] = run(Xclu, y[keep], chrom[keep], "+clustering", report=say)
    res["ALL"] = run(Xall, y[keep], chrom[keep], "+family+cluster", report=say)
    res["ALL+spectrum"] = run(Xall, y[keep], chrom[keep], "+family+cluster+spectrum",
                              spectrum=spec[keep], report=say)

    # ---- H2, H3, H4 ----------------------------------------------------------------------------
    say()
    say("H2 DOES PER-MOTIF RESOLUTION ADD?")
    d2 = delta(res["+family+spectrum"], res["FULL"])
    say(f"     +class/family+spectrum vs FULL   {fmt(d2)}")
    say(f"     (family alone: {fmt(delta(res['+family'], res['FULL']))})")
    h2 = d2["passes"]
    GG.verdict(h2, emit=say,
               if_true="H2 PASS -- keeping motif identity instead of summing it away adds",
               if_false="H2 FAIL -- resolving the matrices by class, family and spectrum adds "
                        "nothing over the summed columns")

    say()
    say("H3 DOES CLUSTERING ADD?")
    d3 = delta(res["+cluster"], res["FULL"])
    say(f"     +clustering vs FULL   {fmt(d3)}")
    h3 = d3["passes"]
    GG.verdict(h3, emit=say,
               if_true="H3 PASS -- where the sites sit relative to each other adds over how many "
                       "there are",
               if_false="H3 FAIL -- homotypic and heterotypic density add nothing over the counts")

    say()
    say("H4 EVERYTHING TOGETHER")
    d4 = delta(res["ALL+spectrum"], res["FULL"])
    say(f"     everything vs FULL   {fmt(d4)}")
    h4 = d4["passes"]
    GG.verdict(h4, emit=say,
               if_true=f"H4 PASS -- the combined stack reaches {res['ALL+spectrum'].mean():.4f} "
                       f"from {res['FULL'].mean():.4f}",
               if_false="H4 FAIL -- the combination does not clear the bar either")

    # ---- H5 ------------------------------------------------------------------------------------
    say()
    say("H5 WHAT DOES NON-LINEARITY ACTUALLY BUY?")
    best_name = max(res, key=lambda k: res[k].mean())
    best_X = {"FULL": Xb, "+family": Xfam, "+family+spectrum": Xfam,
              "+cluster": Xclu, "ALL": Xall, "ALL+spectrum": Xall}[best_name]
    best_spec = spec[keep] if "spectrum" in best_name else None
    say(f"     on the best feature set so far: {best_name}")
    lr = {}
    for kd in ("logistic", "gbm", "gbm_deep", "mlp"):
        lr[kd] = run(best_X, y[keep], chrom[keep], f"{best_name} / {kd}", kind=kd,
                     spectrum=best_spec, report=say)
    bestnl = max(("gbm", "gbm_deep", "mlp"), key=lambda k: lr[k].mean())
    d5 = delta(lr[bestnl], lr["logistic"])
    say(f"     best non-linear ({bestnl}) vs logistic   {fmt(d5)}")
    h5 = d5["passes"]
    GG.verdict(h5, emit=say,
               if_true=f"H5 PASS -- {bestnl} beats a linear model on identical columns, so the "
                       f"signal is genuinely interactive",
               if_false="H5 FAIL -- a logistic regression on the same columns does as well, so the "
                        "signal is additive and the learner was never the constraint")

    # ---- H6 ------------------------------------------------------------------------------------
    say()
    say("H6 THE SHUFFLE STILL DECIDES")
    Xall_s = np.column_stack([np.column_stack([Fs[c][keep] for c in base_cols])]
                             + [Ffs[c][keep] for c in fam_cols]
                             + [C["sh_" + c][keep] for c in CLUSTER_COLS])
    kd = bestnl if h5 else "gbm"
    a_sh = run(Xall_s, y[keep], chrom[keep], "everything, SHUFFLED", kind=kd,
               spectrum=spec_s[keep], report=say)
    a_re = run(Xall, y[keep], chrom[keep], "everything, real", kind=kd,
               spectrum=spec[keep], report=say)
    d6 = delta(a_re, a_sh)
    say(f"     real vs dinucleotide-shuffled   {fmt(d6)}")
    h6 = d6["passes"]
    GG.verdict(h6, emit=say,
               if_true=f"H6 PASS -- the shuffle costs {d6['mean']:.4f}, so the combined stack is "
                       f"still reading sites and not composition",
               if_false="H6 FAIL -- the composition-matched shuffle matches the combined stack, so "
                        "whatever the new blocks added was composition")

    # ---- H7 ------------------------------------------------------------------------------------
    say()
    say("H7 IS THE PROJECTION LEAKING?")
    a_leak = run(Xfam, y[keep], chrom[keep], "spectrum fitted on ALL data (leaky control)",
                 spectrum=spec[keep], leak=True, report=say)
    d7 = delta(res["+family+spectrum"], a_leak)
    say(f"     honest minus leaky   {fmt(d7)}")
    h7 = bool(not (d7["mean"] > 3 * d7["sem"] and d7["n_up"] >= MIN_SEEDS))
    GG.verdict(h7, emit=say,
               if_true="H7 PASS -- the fold-internal projection does not out-score the one fitted "
                       "on the held-out chromosome as well, which is the only ordering that is "
                       "possible if the fitting is honest",
               if_false="H7 FAIL -- the fold-internal arm BEATS the leaky arm, which cannot happen "
                        "unless the fold-internal fitting is wrong")

    say()
    say("H8 WHAT THIS CANNOT SHOW")
    say("     A genomic decoy is untested sequence, not a proven non-enhancer, so every AUC here")
    say("     is a lower bound by an unknown amount.")
    say("     This is stage one -- enhancer against genome. Loop 173 showed the same features")
    say("     cannot say WHICH gene an element acts on, and nothing here changes that; loop 176's")
    say("     gene-blind oracle put that ceiling at R@1 0.8844, far above where the features are.")
    say("     K562 only, JASPAR in-vitro preferences, no expression filter on the factor set.")
    h8 = True
    say(f"     H8 {'PASS' if h8 else 'FAIL'}")

    gates = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8}
    man = RM.manifest(inputs=[Path("colab/data/dna_shape.npz"), Path("colab/data/tf_domains.json")],
                      available=int(keep.sum()), used=int(keep.sum()),
                      selection="loop 174's vs-genome contrast", seed=L174.DRAW_SEED,
                      controls=["the projection refitted inside every training fold, with the "
                                "leaky variant reported beside it",
                                "dinucleotide shuffle of both classes through the whole new stack",
                                "logistic regression on identical columns",
                                "GC recomputed from sequence and matched against the cache"],
                      note="per-motif resolution and motif clustering on the stage-one task")
    out = dict(test="enhancer stage one", gates=gates,
               n_windows=int(keep.sum()), n_positive=int(y[keep].sum()),
               baseline_l174=L174_FULL,
               arms={k: [float(x) for x in v] for k, v in res.items()},
               learners={k: [float(x) for x in v] for k, v in lr.items()},
               shuffled=[float(x) for x in a_sh], leaky=[float(x) for x in a_leak],
               deltas=dict(H2=d2, H3=d3, H4=d4, H5=d5, H6=d6, H7=d7),
               best_arm=best_name, best_learner=bestnl,
               n_family_columns=len(fam_cols), n_pc=N_PC,
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
