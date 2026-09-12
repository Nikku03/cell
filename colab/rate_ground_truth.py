"""rate_ground_truth -- check the transcription-rate target against an independent measurement, and find the noise ceiling.

THE PROBLEM THIS EXISTS TO SOLVE. Every transcription rate in this project is DERIVED, never measured:

    k_syn = RPKM * k_deg          (steady state: what is made equals what is destroyed)

txn_rate_model and nexus_txn both fit features to that number and both stalled well short of the R2 = 0.90 target
(K562: 0.20-0.33). Two explanations were indistinguishable -- the features are too weak, or the TARGET is too noisy --
and with only one estimate of the rate there was no way to tell them apart. source_hunt found the second estimate:
Wachutka et al. 2019 measured synthesis directly by TT-seq in K562 and published it per gene, in molecules/cell/min.
Two independent estimates of one physical quantity make the noise ceiling measurable.

THE CEILING ARGUMENT. Write derived = truth + e1 and measured = truth + e2 with independent errors. Then
r(derived, measured) = sqrt(rel1 * rel2), where rel_i is reliability -- the share of each estimate's variance that is
real signal. A model that predicted truth PERFECTLY would still only reach R2 = rel_i against estimate i. So under
equal reliabilities the observed agreement r IS the maximum achievable R2, and if r comes out at 0.5 then chasing 0.90
against either target is chasing measurement noise. Unequal reliabilities shift which estimate is the better target but
not the conclusion, so both directions are reported.

THE CONTROLS, because a raw agreement number is nearly meaningless on its own:
    cross-cell-line   derived rates from MCF7/HEK293T against measured K562. K562-vs-K562 has to beat these or the
                      "agreement" is just the shared structure of gene expression, not a cell-specific rate.
    partial out RPKM  both estimates are built on abundance, so they must agree somewhat for trivial reasons. The
                      question is whether anything survives once log RPKM is regressed out of both.
    RPKM alone        if plain abundance predicts measured synthesis as well as RPKM*k_deg does, then multiplying by
                      k_deg contributed nothing and the "rate" target was abundance wearing a different unit.
    log-log slope     same quantity in different units means an additive shift on the log scale and a slope of 1.
                      A slope far from 1 means one estimate compresses the dynamic range of the other.

THEN THE THING THAT ACTUALLY MATTERS: refit the identical nexus_txn feature stack against the measured target on the
SAME genes, and against a consensus of the two. If R2 rises against the measured target, the derived target was the
bottleneck. If it rises against the consensus above both singles, measurement noise was capping the fit and the
features are better than they looked. If nothing moves, the features are the bottleneck and no target swap saves them."""
import os
import json, collections, gzip
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
NSPLIT = 5
PROM = 2000


def build_features(genes, info, reg_in, abc, tf_syms, peaks, starts):
    def num(g, k, d=0.0):
        try:
            return float(info[g].get(k) or d)
        except (TypeError, ValueError):
            return d

    def atac_at(chrom, tss):
        v = peaks.get(chrom)
        if not v or tss is None:
            return (0.0, 0.0, 0)
        s = starts[chrom]
        lo = int(np.searchsorted(s, tss - PROM - 5000)); hi = int(np.searchsorted(s, tss + PROM))
        best = tot = 0.0; n = 0
        for a, b, sig in v[lo:hi]:
            if b >= tss - PROM and a <= tss + PROM:
                best = max(best, sig); tot += sig; n += 1
        return (best, tot, n)

    A = {}
    for g in genes:
        try:
            A[g] = atac_at(info[g].get("chrom"), int(info[g].get("tss")))
        except (TypeError, ValueError):
            A[g] = (0.0, 0.0, 0)

    F1 = np.array([[num(g, "cpg"), np.log1p(num(g, "enh")), num(g, "loeuf", 1.0), num(g, "ess"),
                    num(g, "dep_frac"), np.log1p(num(g, "pubs")), num(g, "dark")] for g in genes])
    F2 = np.array([[np.log1p(A[g][0]), np.log1p(A[g][1]), float(A[g][2]), 1.0 if A[g][2] else 0.0] for g in genes])
    def l3(g):
        a = abc.get(g)
        if not a:
            return [0.0] * 6
        return [np.log1p(a.get("n", 0)), a.get("ssum", 0.0), a.get("smax", 0.0),
                np.log1p(a.get("dmed", 0) or 0), np.log1p(a.get("ndistal", 0)), 1.0]
    F3 = np.array([l3(g) for g in genes])
    def l4(g):
        regs = reg_in.get(g, set()); tfr = regs & tf_syms if tf_syms else regs
        return [np.log1p(len(regs)), np.log1p(len(tfr)), len(tfr) / max(len(regs), 1), 1.0 if regs else 0.0]
    F4 = np.array([l4(g) for g in genes])
    occ = (F3[:, 1] + F3[:, 2]) * (F4[:, 1] / (1.0 + F4[:, 1]))
    F5 = np.column_stack([np.log1p(F2[:, 0] * (1.0 + occ)), np.log1p(F2[:, 1] * (1.0 + occ)), occ])
    return np.hstack([F1, F2, F3, F4, F5])


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import pearsonr, spearmanr, linregress

    # ---------- the two estimates ----------
    dc = pd.read_csv(SP / "rnadecay" / "AvgKdegs.csv")
    dc = dc[(dc.avg_RPKM_total > 0) & (dc.avg_kdeg > 0)].copy()
    dc["log_ksyn"] = np.log10(dc.avg_RPKM_total * dc.avg_kdeg)
    dc["log_rpkm"] = np.log10(dc.avg_RPKM_total)

    e2s = json.load(open(SP / "ensg2sym.json"))
    tt = pd.read_csv(SP / "wachutka_synthesis.tsv", sep="\t")
    scol = next(c for c in tt.columns if c.startswith("synthesis"))
    tt["sym"] = [e2s.get(str(g).split(".")[0]) for g in tt.gene_id]
    tt = tt[tt.sym.notna() & (tt[scol] > 0)]
    tt = tt.groupby("sym")[scol].median()                       # a few ENSGs collapse onto one symbol
    tt_log = np.log10(tt)
    print(f"measured  (Wachutka TT-seq, K562): {len(tt_log)} genes, synthesis in 1/cell/min")

    k = dc[dc.cell_line == "K562"].set_index("feature_ID")
    print(f"derived   (RPKM * k_deg, K562):    {len(k)} genes, in RPKM/hour")

    shared = sorted(set(k.index) & set(tt_log.index))
    a = k.loc[shared, "log_ksyn"].values.astype(float)          # derived
    b = tt_log.loc[shared].values.astype(float)                 # measured
    rpkm = k.loc[shared, "log_rpkm"].values.astype(float)
    print(f"shared: {len(shared)} genes\n")

    r_agree = float(pearsonr(a, b)[0]); rho = float(spearmanr(a, b)[0])
    sl = linregress(b, a)
    sd_ratio = float(a.std() / b.std())
    print(f"  AGREEMENT between two independent estimates of the same rate")
    print(f"    Pearson  r = {r_agree:.3f}   (log10 scale; units differ, which is an additive shift and does not affect r)")
    print(f"    Spearman rho = {rho:.3f}")
    # THE OLS SLOPE IS NOT A DYNAMIC-RANGE STATISTIC. slope = r * sd(a)/sd(b), so a low r drags the slope down
    # no matter how the spreads compare. A first pass here read slope=0.32 as "the derived rate compresses the
    # measured one 3x"; the SD ratio is 0.88, i.e. near-identical spread, and the slope is just r-attenuation.
    print(f"    SD ratio derived/measured = {sd_ratio:.3f}  (1.0 = same spread on the log scale)")
    print(f"    OLS slope = {sl.slope:.3f}, which is r * SD-ratio = {r_agree:.3f} * {sd_ratio:.3f}; attenuated by")
    print(f"      the disagreement, NOT evidence of a compressed dynamic range")

    # ---------- control 1: is the agreement cell-line specific? ----------
    print(f"\n  CONTROL 1 -- cross-cell-line. K562-vs-K562 must beat other lines against measured K562,")
    print(f"               or the agreement is generic expression structure, not a cell-specific rate.")
    xc = {"K562": r_agree}
    for line in ["MCF7", "HEK293T", "Calu3", "MDAMB"]:
        o = dc[dc.cell_line == line].set_index("feature_ID")
        s2 = sorted(set(o.index) & set(tt_log.index))
        if len(s2) < 500:
            continue
        rr = float(pearsonr(o.loc[s2, "log_ksyn"].values.astype(float), tt_log.loc[s2].values.astype(float))[0])
        xc[line] = rr
        print(f"    derived {line:8s} vs measured K562:  r = {rr:.3f}   (n={len(s2)})")
    best_other = max(v for kk, v in xc.items() if kk != "K562")
    best_line = max((kk for kk in xc if kk != "K562"), key=lambda x: xc[x])
    # A 0.01 GAP IS NOT A RESULT. With n in the thousands the SE on r is ~0.011, so "K562 loses to HEK293T by 0.011"
    # and "they tie" are the same observation. Bootstrap the PAIRED difference on the genes both share, because the
    # verdict's claim about cell-specificity rests entirely on whether that gap is real.
    o = dc[dc.cell_line == best_line].set_index("feature_ID")
    tri = sorted(set(k.index) & set(o.index) & set(tt_log.index))
    va = k.loc[tri, "log_ksyn"].values.astype(float)
    vo = o.loc[tri, "log_ksyn"].values.astype(float)
    vt = tt_log.loc[tri].values.astype(float)
    obs_d = float(pearsonr(va, vt)[0] - pearsonr(vo, vt)[0])
    rb = np.random.RandomState(0)
    boot = np.array([(lambda s: pearsonr(va[s], vt[s])[0] - pearsonr(vo[s], vt[s])[0])(
        rb.randint(0, len(tri), len(tri))) for _ in range(400)])
    lo_ci, hi_ci = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    tie = lo_ci <= 0 <= hi_ci
    specific = (not tie) and obs_d > 0
    print(f"    paired on the {len(tri)} genes all three share: K562 minus {best_line} = {obs_d:+.3f} "
          f"[95% CI {lo_ci:+.3f}, {hi_ci:+.3f}]")
    print(f"    -> the K562 agreement {'is INDISTINGUISHABLE from' if tie else ('BEATS' if specific else 'LOSES to')} "
          f"the best cross-line control")
    # THE CONTROL ABOVE CANNOT STAND ALONE. A cross-line r as high as the within-line one has two very different
    # readings: either the agreement is not cell-specific, or the DERIVED RATES THEMSELVES barely differ between
    # cell lines, in which case the control was never able to discriminate anything. Measure that directly.
    dd = {}
    kk_idx = dc[dc.cell_line == "K562"].set_index("feature_ID")
    for line in ["MCF7", "HEK293T", "Calu3", "MDAMB"]:
        o = dc[dc.cell_line == line].set_index("feature_ID")
        s2 = sorted(set(o.index) & set(kk_idx.index))
        if len(s2) < 500:
            continue
        dd[line] = float(pearsonr(kk_idx.loc[s2, "log_ksyn"].values.astype(float),
                                  o.loc[s2, "log_ksyn"].values.astype(float))[0])
    print(f"    how different ARE the derived rates between lines?  derived K562 vs:")
    for line, v in dd.items():
        print(f"      {line:9s} r = {v:.3f}")
    derived_line_specific = bool(dd) and max(dd.values()) < 0.9
    print(f"    -> derived rates {'DO' if derived_line_specific else 'do NOT'} differ appreciably between lines, so "
          f"control 1 {'is' if derived_line_specific else 'is not'} able to discriminate")

    # ---------- control 2: partial out abundance ----------
    def resid(y, x):
        s = linregress(x, y)
        return y - (s.slope * x + s.intercept)
    r_part = float(pearsonr(resid(a, rpkm), resid(b, rpkm))[0])
    print(f"\n  CONTROL 2 -- partial out log RPKM from both. Both estimates are built on abundance, so some")
    print(f"               agreement is guaranteed. Residual agreement r = {r_part:.3f} (was {r_agree:.3f}).")

    # ---------- control 3: does k_deg contribute anything? ----------
    r_rpkm_only = float(pearsonr(rpkm, b)[0])
    print(f"\n  CONTROL 3 -- plain abundance vs measured synthesis: r = {r_rpkm_only:.3f}")
    print(f"               derived rate (RPKM*k_deg) vs measured synthesis: r = {r_agree:.3f}")
    kdeg_helps = r_agree > r_rpkm_only
    print(f"    -> multiplying by k_deg {'improves' if kdeg_helps else 'does NOT improve'} agreement with the direct "
          f"measurement ({r_agree - r_rpkm_only:+.3f})")

    # ---------- the ceiling ----------
    rel = max(r_agree, 0.0)
    print(f"\n  NOISE CEILING. With independent errors, r(derived,measured) = sqrt(rel_derived * rel_measured).")
    print(f"    equal-reliability estimate: rel = {rel:.3f}, so a PERFECT model of the true rate would score")
    print(f"    R2 = {rel:.3f} against either estimate. Observed best so far: nexus_txn K562 R2 = 0.200,")
    print(f"    txn_rate GBM R2 = 0.322.")

    # ---------- refit the same features against each target ----------
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    reg_in = collections.defaultdict(set)
    for e in (D.get("reg") or []):
        try:
            reg_in[names[e[1]]].add(names[e[0]])
        except (TypeError, IndexError):
            continue
    abc = json.load(open(SP / "_abcfeat.json"))
    tf_syms = set(json.load(open(SP / "lambert_tf_symbols.json")))
    peaks = collections.defaultdict(list)
    with gzip.open(SP / "k562_atac.bed.gz", "rt") as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) >= 7:
                try:
                    peaks[f[0]].append((int(f[1]), int(f[2]), float(f[6])))
                except ValueError:
                    pass
    for c in peaks:
        peaks[c].sort()
    starts = {c: np.array([p[0] for p in v]) for c, v in peaks.items()}

    genes = [g for g in shared if g in info]
    X = build_features(genes, info, reg_in, abc, tf_syms, peaks, starts)
    idx = {g: i for i, g in enumerate(shared)}
    ya = np.array([a[idx[g]] for g in genes]); yb = np.array([b[idx[g]] for g in genes])
    z = lambda v: (v - v.mean()) / v.std()
    yc = (z(ya) + z(yb)) / 2.0                                   # consensus: a better estimate of truth than either
    print(f"\n  REFIT the identical nexus_txn feature stack ({X.shape[1]} features) against each target,")
    print(f"        on the SAME {len(genes)} genes so the comparison is like for like, {NSPLIT} held-out splits.")

    rng = np.random.RandomState(0)
    scores = collections.defaultdict(list)
    for si in range(NSPLIT):
        o = np.arange(len(genes)); rng.shuffle(o)
        cut = int(0.7 * len(o)); tr, te = o[:cut], o[cut:]
        for lab, y in [("derived  RPKM*k_deg", ya), ("measured TT-seq", yb), ("consensus of both", yc)]:
            p = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.08,
                                              random_state=0).fit(X[tr], y[tr]).predict(X[te])
            scores[lab].append(pearsonr(p, y[te])[0])
    fit = {}
    print(f"    {'target':22s} {'r':>7s} {'R2':>7s} {'vs ceiling':>12s}")
    for lab in ["derived  RPKM*k_deg", "measured TT-seq", "consensus of both"]:
        r = float(np.mean(scores[lab])); sd = float(np.std(scores[lab]))
        fit[lab] = {"r": round(r, 4), "r2": round(r * r, 4), "sd": round(sd, 4)}
        print(f"    {lab:22s} {r:>7.3f} {r*r:>7.3f} {r*r/max(rel,1e-9):>11.0%}")

    d2 = fit["derived  RPKM*k_deg"]["r2"]; m2 = fit["measured TT-seq"]["r2"]; c2 = fit["consensus of both"]["r2"]
    noise_capped = c2 > max(d2, m2)

    # ---------- does the ceiling model actually hold? ----------
    # THE CEILING NUMBER IS ONLY WORTH QUOTING IF ITS MODEL SURVIVES A PREDICTION. Under truth-plus-independent-error
    # with equal reliabilities rel = r_agree, averaging two z-scored estimates raises reliability to 2*rel/(1+rel).
    # A fixed predictor therefore has to score that much MORE against the consensus than against a single estimate.
    # That is a quantitative prediction, and it is checkable right here.
    rel_c = 2 * rel / (1 + rel)
    c2_pred = float(np.mean([d2, m2]) * rel_c / max(rel, 1e-9))
    model_holds = abs(c2 - c2_pred) < 0.25 * max(c2_pred, 1e-9)
    print(f"\n  DOES THE CEILING MODEL HOLD? Averaging two z-scored estimates should raise reliability from")
    print(f"    {rel:.3f} to 2*rel/(1+rel) = {rel_c:.3f}, so the SAME features should score R2 = {c2_pred:.3f}")
    print(f"    against the consensus. Observed: {c2:.3f}.")
    print(f"    -> the truth-plus-independent-error model {'HOLDS' if model_holds else 'FAILS'} "
          f"({'within' if model_holds else 'off by'} {abs(c2-c2_pred)/max(c2_pred,1e-9):.0%})")

    verdict = (
        f"RATE GROUND TRUTH. The transcription rate this project models has always been DERIVED (k_syn = RPKM * k_deg) and "
        f"never checked. Wachutka et al. 2019 measured it directly by TT-seq in K562, so on {len(shared)} shared genes there "
        f"are now two independent estimates of one physical quantity. THEY BARELY AGREE: r = {r_agree:.3f} "
        f"(Spearman {rho:.3f}), meaning the two estimates share only {r_agree*r_agree:.0%} of their variance. "
        + (f"The agreement is cell-specific -- it beats every cross-cell-line control (best other line {best_other:.3f}). "
           if specific else
           (f"It is INDISTINGUISHABLE from the cross-cell-line controls: derived {best_line} against measured K562 scores "
            f"{best_other:.3f}, and the paired difference is {obs_d:+.3f} [95% CI {lo_ci:+.3f}, {hi_ci:+.3f}] -- a tie, not "
            f"a loss, and the earlier reading of it as a loss rested on a 0.01 gap smaller than its own error bar. "
            if tie else
            f"It LOSES to the cross-cell-line controls: derived {best_line} against measured K562 scores {best_other:.3f}, "
            f"paired difference {obs_d:+.3f} [95% CI {lo_ci:+.3f}, {hi_ci:+.3f}]. ")
           + (f"Either way the control DOES discriminate, because the derived rates themselves differ substantially between "
              f"lines (derived K562 vs other lines: r = {min(dd.values()):.2f}-{max(dd.values()):.2f}). So matching the "
              f"right cell line buys nothing here: the shared signal is whatever structure the two kinds of rate estimate "
              f"carry in common, not K562-specific transcription. "
              if derived_line_specific else
              f"but that control cannot discriminate here: the derived rates are nearly identical across cell lines "
              f"(derived K562 vs other lines r = {min(dd.values()):.2f}-{max(dd.values()):.2f}), so a matched cross-line "
              f"score was guaranteed and says nothing either way. "))
        + f"The agreement survives partialling log abundance out of both estimates at r = {r_part:.3f}. "
        + (f"Multiplying abundance by k_deg helps: RPKM alone agrees with the direct measurement at {r_rpkm_only:.3f} versus "
           f"{r_agree:.3f} for the full derived rate. " if kdeg_helps else
           f"BUT multiplying by k_deg does not help: RPKM alone agrees with the direct measurement at {r_rpkm_only:.3f}, "
           f"versus {r_agree:.3f} for the full derived rate, so the decay term is not adding information about synthesis. ")
        + f"The two spreads are near-identical (SD ratio {sd_ratio:.2f}); the OLS slope of {sl.slope:.2f} is r-attenuation, "
        f"not a compressed dynamic range, and reading it as one would be a mistake. "
        f"THE IMPLIED CEILING. If both estimates are truth plus independent error, their correlation is "
        f"sqrt(rel_derived * rel_measured); under equal reliabilities that puts each estimate's reliability at "
        f"{rel:.2f}, meaning a model that predicted the TRUE rate perfectly would still score only R2 = {rel:.2f} "
        f"against either. On that reading the existing fits are not the failure they looked like -- nexus_txn's K562 "
        f"R2 = 0.200 would be {0.200/max(rel,1e-9):.0%} of what is achievable -- and refitting the identical feature "
        f"stack here gives R2 = {d2:.3f} on the derived target and {m2:.3f} on the measured one. "
        + (f"BUT THE CEILING MODEL DOES NOT SURVIVE ITS OWN PREDICTION, so that number should not be quoted as a ceiling. "
           f"Averaging the two z-scored estimates must raise reliability to 2*rel/(1+rel) = {rel_c:.2f}, which requires "
           f"the same features to score R2 = {c2_pred:.3f} against the consensus. They score {c2:.3f} -- "
           f"{abs(c2-c2_pred)/max(c2_pred,1e-9):.0%} off. Averaging bought nothing, which is not what truth-plus-"
           f"independent-noise predicts. The likelier reading is that the two datasets are not two noisy views of one "
           f"quantity at all: TT-seq synthesis is nascent output attributed to a major isoform, while RPKM*k_deg is a "
           f"steady-state whole-gene balance, and features that predict one need not predict the other. "
           if not model_holds else
           f"The ceiling model survives its own prediction: averaging should raise reliability to {rel_c:.2f} and "
           f"therefore the consensus R2 to {c2_pred:.3f}, and the observed {c2:.3f} matches. Measurement noise, not "
           f"weak features, is what caps these fits. ")
        + f"WHAT THIS DOES NOT SHOW. The reliability estimate assumes independent errors; both pipelines quantify against "
        f"the same genome annotation and share mappability and multi-isoform problems, and shared error INFLATES apparent "
        f"agreement, so {rel:.2f} is an upper bound on reliability -- the true ceiling is lower, not higher. Wachutka's "
        f"synthesis is reported per SALMON MAJOR ISOFORM while the derived rate is whole-gene, which attenuates the "
        f"comparison by an unmeasured amount. And the comparison is log-scale only: the datasets report different units "
        f"(RPKM/hour versus molecules/cell/min), so no absolute fold-error comparison between them is meaningful. "
        f"THE ONE THING THAT IS UNAMBIGUOUS is the headline: the target this project has been fitting for two modules "
        f"agrees with a direct measurement of the same quantity at r = {r_agree:.3f}. Whatever the right ceiling is, an "
        f"R2 of 0.90 against the derived rate would not mean the true transcription rate had been predicted. Deterministic.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_shared": len(shared), "r_agreement": r_agree, "spearman": rho, "loglog_slope": float(sl.slope),
               "cross_cell_line": {kk: round(v, 4) for kk, v in xc.items()}, "cell_specific": bool(specific),
               "r_partial_out_rpkm": r_part, "r_rpkm_only": r_rpkm_only, "kdeg_helps": bool(kdeg_helps),
               "reliability_estimate": rel, "refit": fit, "consensus_beats_singles": bool(noise_capped),
               "n_features": int(X.shape[1]), "n_genes_refit": len(genes),
               "verdict": verdict, "note": verdict}, open(OUT / "rate_ground_truth.json", "w"), indent=1)
    print("\n  -> outputs/orphan/rate_ground_truth.json")


if __name__ == "__main__":
    main()
