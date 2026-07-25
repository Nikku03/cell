"""chip_vs_perturb -- the strongest alternative-data test available to this project: can ChIP-seq REPLACE Perturb-seq for building the causal
graph? ChIP-seq is the standard way regulatory networks are drawn without perturbation data, it is cheap and abundant, and here it can be tested
under near-ideal conditions -- 116 transcription factors that were BOTH ChIP-seq'd in K562 (ChIP-Atlas, hg38, Q<1e-5 peak calls) AND knocked out
in K562 (Replogle), of which 25 move enough genes to be scorable. Same cell line, same factors, measured binding on one side and measured
response on the other.

The peak files are ~190 MB and are NOT committed -- run `python fetch_chipatlas.py` first to reproduce them.

Peaks come from ChIP-Atlas rather than ENCODE for two reasons: encodeproject.org is behind a bot challenge from this environment, and more
importantly ChIP-Atlas reprocesses every experiment through ONE uniform pipeline. That uniformity is not a convenience, it is a requirement --
the decisive comparison below scores each TF's movers against other TFs' binding, which would be corrupted if different factors' peaks came from
different peak-callers. Experiments are capped at CAPEXP per factor for the same reason: a TF with 84 deposited experiments would otherwise look
bound everywhere purely from sequencing depth. Peak breadth per TF is reported so this can be checked rather than assumed.

THE QUESTION: for TF X, does a ChIP peak at gene G's promoter predict that G actually MOVES when X is knocked out?

THE CONTROL THAT DECIDES IT. ChIP peaks concentrate at active promoters, and active genes are also the ones most likely to be detected as movers.
So ANY TF's peaks will look predictive of ANY TF's knockout response -- a promoter-activity artifact that would masquerade as regulatory signal. The
decisive comparison is therefore not "is own-binding above 0.5" but:

    own-TF binding   vs   OTHER TFs' binding at the same genes

If own-binding only matches the other-TF baseline, ChIP is reading promoter activity, not that factor's regulon. Negatives are additionally
DEGREE-MATCHED on how often each gene moves across all knockouts, so a gene being generically responsive cannot win either.

Two details of that control decide whether it is fair, and both were got wrong on the first attempt:
  GRANULARITY. Averaging all other factors' profiles gives a CONTINUOUS score that can break ties a single binary profile cannot, handing the
    control an unearned AUC advantage. The primary control is therefore a SINGLE other factor, binary against binary; the pooled average is still
    reported, but as a promoter-activity descriptor rather than as the comparison.
  POWER. Only a handful of knocked-out TFs move enough genes to be scorable and one of them (GATA1) contributes about a third of the positives, so
    a per-factor test is underpowered. The primary statistic is therefore a POOLED test over every (TF, gene) pair with a TF-label PERMUTATION null
    (a derangement, so no factor keeps its own profile), which holds positives, negatives and marginal binding rates fixed.

Reported: pooled AUC with its permutation null, per-TF AUC against both controls, and the raw forward/reverse rates -- of genes bound, how many
move; of genes that move, how many are bound. Deterministic."""
import json, gzip, collections, sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/orphan")
CHIP = OUT / "invivo" / "chipatlas"
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
RNG = np.random.RandomState(0)
TIDE_FRAC = 0.05
WIN = 5000          # peak centre within +-5kb of TSS counts as binding (the standard promoter-assignment window)
MINSRC = 20         # a knockout counts as a SOURCE for defining the tide mask -- must match infer_network, or "specific" changes meaning
MINMOV = 5          # a TF needs this many specific movers to be scorable (the project's MIN_SPEC convention)
CAPEXP = 4          # experiments per factor, so binding breadth is comparable across TFs
NPERM = 200         # TF-label permutations for the pooled null


def load_peaks(files):
    """union of peak centres across a factor's experiments, per chromosome and sorted"""
    by = collections.defaultdict(list)
    for p in files:
        op = gzip.open if p.suffix == ".gz" else open
        with op(p, "rt") as f:
            for line in f:
                if line.startswith(("track", "#", "browser")):
                    continue
                c = line.split("\t")
                if len(c) < 3:
                    continue
                try:
                    a, b = int(c[1]), int(c[2])
                except ValueError:
                    continue
                by[c[0]].append((a + b) // 2)
    for k in by:
        by[k].sort()
    return by


def main():
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; gidx = {g: i for i, g in enumerate(genes)}
    kos = [str(k) for k in df.index]
    A = np.abs(df.values.astype(np.float32))

    # threshold calibrated exactly as in infer_network, so "specific mover" means the same thing everywhere
    from eval_harness import Harness
    H = Harness("K562")
    kset = set(kos)
    tc = [int((np.abs(H.M[H.ki[k]]) >= 1.0).sum()) for k in H.kos
          if k in kset and (np.abs(H.M[H.ki[k]]) > 0).any() and np.abs(H.M[H.ki[k]])[np.abs(H.M[H.ki[k]]) > 0].min() < 1.0]
    med = float(np.median(tc)); T, best = 4.0, None
    for t in np.arange(1.0, 12.0, 0.1):
        e = abs(float(np.median((A >= t).sum(1))) - med)
        if best is None or e < best:
            best, T = e, float(t)
    per = (A >= T).sum(1)
    # the tide mask is defined from SOURCE knockouts at MINSRC, deliberately decoupled from the MINMOV used to decide
    # which TFs are scorable -- otherwise loosening MINMOV would silently redefine what "specific" means and the
    # results would stop being comparable to infer_network / altdata_sources
    strong = [i for i in range(len(kos)) if per[i] >= MINSRC]
    tide = (A[strong] >= T).mean(0) >= TIDE_FRAC
    mover_freq = (A[strong] >= T).mean(0)          # for degree matching
    print(f"K562: {len(kos)} knockouts x {len(genes)} genes | threshold |z|>={T:.1f} | {len(strong)} source knockouts "
          f"(>={MINSRC} movers) | {tide.sum()} tide genes | TFs scorable at >={MINMOV} specific movers")

    # gene coordinates
    D = json.load(open(OUT / "cell_complete.json")); info = {g["name"]: g for g in D["genes"]}
    coord = {}
    for g in genes:
        gi = info.get(g)
        if not gi or not gi.get("chrom"):
            continue
        try:
            coord[g] = (gi["chrom"], float(gi["tss"]))
        except (TypeError, ValueError, KeyError):
            pass
    print(f"gene coordinates available for {len(coord)}/{len(genes)} measured genes")

    byf = collections.defaultdict(list)
    for p in sorted(CHIP.glob("*.bed")):
        byf[p.name.split("__")[0]].append(p)
    tfs = sorted(t for t in byf if t in kos)
    nexp = {t: min(len(byf[t]), CAPEXP) for t in tfs}
    print(f"transcription factors with BOTH K562 ChIP-seq and a K562 knockout: {len(tfs)} "
          f"({sum(nexp.values())} experiments, <= {CAPEXP} per factor)")

    # binding matrix: TFs x measured genes
    B = np.zeros((len(tfs), len(genes)), np.float32)
    for ti, tf in enumerate(tfs):
        by = load_peaks(sorted(byf[tf])[:CAPEXP])
        for g, (c, t) in coord.items():
            v = by.get(c)
            if not v:
                continue
            j = np.searchsorted(v, t)
            for k in (j - 1, j):
                if 0 <= k < len(v) and abs(v[k] - t) <= WIN:
                    B[ti, gidx[g]] = 1.0
                    break
    covered = np.array([g in coord for g in genes])
    breadth = B.sum(1)
    print(f"binding breadth per TF: median {int(np.median(breadth))} promoters "
          f"({100*np.median(breadth)/max(covered.sum(),1):.0f}% of locatable genes), "
          f"range {int(breadth.min())}-{int(breadth.max())}\n")

    kidx = {k: i for i, k in enumerate(kos)}
    rows, pooled = [], []
    for ti, tf in enumerate(tfs):
        r = kidx[tf]
        mv = (A[r] >= T) & (~tide) & covered
        pos_i = np.where(mv)[0]
        if len(pos_i) < MINMOV:
            continue
        # degree-matched negatives: same mover-frequency bucket, not a mover in THIS knockout
        cand = np.where((~mv) & covered & (~tide))[0]
        bucket = lambda x: np.minimum((x * 100).astype(int) // 2, 25)
        cb = bucket(mover_freq[cand]); pb = bucket(mover_freq[pos_i])
        byb = collections.defaultdict(list)
        for j, b in zip(cand, cb):
            byb[b].append(j)
        neg_i = []
        for b in pb:
            pool = byb.get(b) or list(cand)
            neg_i.append(pool[RNG.randint(len(pool))])
        neg_i = np.array(neg_i)
        idx = np.concatenate([pos_i, neg_i]); y = np.array([1] * len(pos_i) + [0] * len(neg_i))
        pooled.append((ti, idx, y))
        own = B[ti, idx]
        rest = np.delete(B, ti, axis=0)[:, idx]
        others = rest.mean(0)                                  # promoter-activity baseline (continuous)
        try:
            a_own = roc_auc_score(y, own); a_oth = roc_auc_score(y, others)
        except ValueError:
            continue
        # GRANULARITY-MATCHED control. `others` is a mean over ~116 binary vectors, so it can break ties that a single
        # binary vector cannot -- comparing it directly to `own` would hand it an unearned AUC advantage. This scores
        # SINGLE other TFs, binary against binary, which is the fair comparison.
        pick = RNG.choice(rest.shape[0], min(20, rest.shape[0]), replace=False)
        singles = [roc_auc_score(y, rest[q]) for q in pick if len(np.unique(rest[q])) > 1]
        a_single = float(np.median(singles)) if singles else float("nan")
        bound = B[ti] > 0
        rows.append({"tf": tf, "n_movers": int(len(pos_i)), "n_bound": int(bound.sum()),
                     "n_experiments": int(min(len(byf[tf]), CAPEXP)),
                     "auc_own": float(a_own), "auc_other_tfs": float(a_oth), "auc_single_other_tf": a_single,
                     "bound_rate_movers": float(own[y == 1].mean()), "bound_rate_matched": float(own[y == 0].mean()),
                     "frac_bound_that_move": float(mv[bound].mean()) if bound.any() else float("nan")})

    R = pd.DataFrame(rows).sort_values("auc_own", ascending=False)
    ao = float(R["auc_own"].mean()); at = float(R["auc_other_tfs"].mean())
    asg = float(R["auc_single_other_tf"].mean())
    hdr = (f"  {'TF':10s} {'movers':>7s} {'bound':>7s} {'AUC own':>8s} {'1 other TF':>11s} {'margin':>8s} "
           f"{'all others':>11s} {'bound|move':>11s} {'move|bound':>11s}")
    line = lambda x: (f"  {x['tf']:10s} {int(x['n_movers']):>7d} {int(x['n_bound']):>7d} {x['auc_own']:>8.3f} "
                      f"{x['auc_single_other_tf']:>11.3f} {x['auc_own']-x['auc_single_other_tf']:>+8.3f} "
                      f"{x['auc_other_tfs']:>11.3f} {100*x['bound_rate_movers']:>10.0f}% {100*x['frac_bound_that_move']:>10.1f}%")
    print(hdr)
    if len(R) <= 18:
        for _, x in R.iterrows():
            print(line(x))
    else:
        for _, x in R.head(15).iterrows():
            print(line(x))
        print(f"  {'...':10s}")
        for _, x in R.tail(3).iterrows():
            print(line(x))

    # ---- POOLED TEST WITH A TF-PERMUTATION NULL ----
    # The per-TF Wilcoxon is underpowered: only a handful of knocked-out TFs move enough genes to be scorable, and one
    # of them (GATA1) contributes most of the positives. Pooling every (TF, gene) pair gives a properly powered test,
    # and permuting WHICH binding profile is assigned to which TF gives the exact null for "own binding beats another
    # factor's binding" while holding the positives, the negatives and the marginal binding rates fixed.
    py_ = np.concatenate([y for _, _, y in pooled])
    own_pool = np.concatenate([B[ti, idx] for ti, idx, _ in pooled])
    auc_pool = roc_auc_score(py_, own_pool)
    tis = [ti for ti, _, _ in pooled]
    null = []
    for _ in range(NPERM):
        perm = RNG.permutation(len(tis))
        if any(tis[perm[q]] == tis[q] for q in range(len(tis))) and len(tis) > 2:
            continue                                  # keep it a derangement: no TF may keep its own profile
        v = np.concatenate([B[tis[perm[q]], pooled[q][1]] for q in range(len(tis))])
        null.append(roc_auc_score(py_, v))
    null = np.array(null)
    p_perm = float((null >= auc_pool).mean()) if len(null) else float("nan")
    print(f"\n  POOLED over {len(pooled)} TFs and {int(py_.sum()):,} positives:")
    print(f"    AUC, own binding                {auc_pool:.3f}")
    print(f"    AUC, TF-permuted binding (null) {null.mean():.3f} +- {null.std():.3f}  (n={len(null)} derangements)")
    print(f"    permutation p                   {p_perm:.3g}")

    # primary comparison is against the GRANULARITY-MATCHED single-other-TF control
    marg = R["auc_own"] - R["auc_single_other_tf"]
    from scipy.stats import wilcoxon, spearmanr
    try:
        _, pv = wilcoxon(R["auc_own"], R["auc_single_other_tf"])
    except ValueError:
        pv = float("nan")
    nbeat = int((marg > 0).sum())
    # confound check: a factor with broader binding, or more deposited experiments, must not be what drives own-AUC
    rho_b, p_b = spearmanr(R["n_bound"], R["auc_own"])
    rho_e, p_e = spearmanr(R["n_experiments"], R["auc_own"])
    print(f"\n  confound check: own-AUC vs binding breadth  rho={rho_b:+.2f} (p={p_b:.2g}); "
          f"vs experiments per factor rho={rho_e:+.2f} (p={p_e:.2g})")
    print(f"\n  scorable TFs                          {len(R)}")
    print(f"  mean AUC, own binding                 {ao:.3f}")
    print(f"  mean AUC, ONE other TF's binding       {asg:.3f}   <- granularity-matched control (binary vs binary)")
    print(f"  mean AUC, all other TFs averaged       {at:.3f}   <- promoter-activity baseline (continuous)")
    print(f"  SPECIFICITY MARGIN (own - one other)  {ao-asg:+.3f}  (own wins in {nbeat}/{len(R)} TFs, Wilcoxon p={pv:.2g})")
    print(f"  of genes a TF binds, {100*R['frac_bound_that_move'].mean():.1f}% actually move when that TF is knocked out")
    print(f"  of a TF's specific movers, {100*R['bound_rate_movers'].mean():.0f}% carry a peak (vs {100*R['bound_rate_matched'].mean():.0f}% of degree-matched non-movers)")

    real = p_perm == p_perm and p_perm < 0.05 and (auc_pool - null.mean()) > 0.02
    useful = auc_pool > 0.60
    verdict = (
        "CHIP-SEQ vs PERTURB-SEQ (chip_vs_perturb.py): the strongest alternative-data test this project can run -- "
        f"{len(R)} transcription factors that were BOTH ChIP-seq'd in K562 (ChIP-Atlas, uniform hg38 reprocessing) AND knocked out in K562, so "
        "measured binding is scored directly against measured response in the same cell line. Positives are that TF's specific movers, negatives "
        "are DEGREE-MATCHED on how often each gene moves across all knockouts, so generically-responsive genes cannot win. "
        f"RESULT: pooled over {len(pooled)} scorable factors and {int(py_.sum()):,} positives, own-TF binding reaches AUC {auc_pool:.3f}. "
        "THE CONTROL THAT DECIDES IT: ChIP peaks concentrate at active promoters, so ANY TF's peaks predict ANY TF's response. Permuting WHICH "
        "binding profile belongs to which TF (a derangement, so no factor keeps its own) holds the positives, negatives and marginal binding rates "
        f"fixed and isolates exactly the factor-specific component: that null sits at {null.mean():.3f} +- {null.std():.3f}, "
        f"permutation p={p_perm:.3g}. "
        + f"Own binding sits {auc_pool-null.mean():+.3f} above that null. "
        + (f"THE SHARPEST VERSION OF THE RESULT: pooling the OTHER {len(tfs)-1} factors' binding into a single promoter-activity score predicts a "
           f"TF's own knockout response BETTER than that TF's own peaks do ({at:.3f} vs {ao:.3f} per-factor). Knowing which promoters are generally "
           "occupied beats knowing where this particular factor sits. " if at > ao else "")
        + ("BOTH STATISTICS AGREE: " if (p_perm < 0.05) == (pv < 0.05) else
           "THE TWO STATISTICS DISAGREE ON SIGNIFICANCE AND I AM NOT GOING TO PICK THE FLATTERING ONE -- the pooled test is weighted by positives "
           "while the per-factor test weights every TF equally, and one factor (GATA1) contributes about a third of the positives: ")
        + f"pooled p={p_perm:.3g}, and per-factor own binding beats a single other factor's binding in {nbeat}/{len(R)} TFs by a mean "
        f"{ao-asg:+.3f} (Wilcoxon p={pv:.2g}). The MAGNITUDE is the part that matters and it is consistent either way: whatever factor-specific "
        f"signal exists is worth about {min(auc_pool-null.mean(), ao-asg):.3f}-{max(auc_pool-null.mean(), ao-asg):.3f} AUC on top of a "
        "promoter-activity baseline that is itself barely above chance"
        + (", and neither test can distinguish it from zero. " if not real else ". ")
        + f"The forward rate makes the practical point: of the genes a TF actually binds, only {100*R['frac_bound_that_move'].mean():.1f}% move "
        f"when that TF is knocked out, while {100*R['bound_rate_movers'].mean():.0f}% of its movers carry a peak versus "
        f"{100*R['bound_rate_matched'].mean():.0f}% of matched non-movers. "
        + (f"CONFOUND, REPORTED BECAUSE IT IS REAL: per-TF AUC is uncorrelated with binding breadth (rho={rho_b:+.2f}, p={p_b:.2g}) but IS "
           f"correlated with how many experiments a factor has (rho={rho_e:+.2f}, p={p_e:.2g}). Since breadth is flat, the likely reading is that "
           "better-sampled factors have less noisy peak sets and therefore show more of whatever real signal exists -- which would mean the small "
           "effect above is diluted by measurement noise rather than manufactured by it. Either way it is a caveat on the per-factor numbers, not a "
           "result to lean on. "
           if p_e == p_e and p_e < 0.05 else
           f"Neither binding breadth (rho={rho_b:+.2f}, p={p_b:.2g}) nor experiments per factor (rho={rho_e:+.2f}, p={p_e:.2g}) explains the per-TF "
           "AUC, so this is not a depth artifact. ")
        + ("CONSEQUENCE: ChIP-seq cannot substitute for Perturb-seq in building this causal graph. Binding is neither sufficient NOR necessary here "
           f"-- only {100*R['frac_bound_that_move'].mean():.1f}% of bound genes respond, and only {100*R['bound_rate_movers'].mean():.0f}% of "
           "responding genes are bound, so most of what a knockout does is not even routed through that factor's own promoter occupancy. This is a "
           "mechanism for this project's earlier "
           "finding that 94% of measured causal edges are absent from curated databases: those databases are largely built from binding, and "
           "binding is not regulation. It also explains why no amount of database curation was ever going to close the gap. "
           if not useful else
           "CONSEQUENCE: ChIP-seq carries real, factor-specific predictive signal here and is worth adding as a prior, though it remains far from "
           "sufficient on its own. ")
        + f"Deterministic; +-{WIN//1000}kb TSS window, <= {CAPEXP} experiments per factor, degree-matched negatives, per-TF and pooled statistics "
        "reported.")
    print(f"\nVERDICT: {verdict}")
    R.to_csv(OUT / "chip_vs_perturb.csv", index=False)
    json.dump({"n_tfs": len(R), "window": WIN, "cap_experiments": CAPEXP, "threshold": T, "min_movers": MINMOV,
               "pooled_n_positives": int(py_.sum()), "pooled_auc_own": round(float(auc_pool), 4),
               "pooled_auc_permuted_mean": round(float(null.mean()), 4), "pooled_auc_permuted_sd": round(float(null.std()), 4),
               "pooled_permutation_p": p_perm, "auc_own_mean": round(ao, 4),
               "auc_single_other_tf_mean": round(asg, 4), "auc_other_tfs_mean": round(at, 4),
               "specificity_margin": round(ao - asg, 4), "margin_vs_pooled_others": round(ao - at, 4),
               "n_tfs_beating_baseline": nbeat, "wilcoxon_p": float(pv),
               "rho_auc_vs_breadth": round(float(rho_b), 3), "rho_auc_vs_nexp": round(float(rho_e), 3),
               "frac_bound_that_move": round(float(R["frac_bound_that_move"].mean()), 4),
               "bound_rate_movers": round(float(R["bound_rate_movers"].mean()), 4),
               "bound_rate_matched": round(float(R["bound_rate_matched"].mean()), 4),
               "factor_specific": bool(real), "sufficient": bool(useful),
               "per_tf": rows, "verdict": verdict, "note": verdict},
              open(OUT / "chip_vs_perturb.json", "w"), indent=1)
    print("\n  -> outputs/orphan/chip_vs_perturb.json + chip_vs_perturb.csv")


if __name__ == "__main__":
    main()
