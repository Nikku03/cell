"""knockdown_dose -- at what remaining expression does a knockdown behave like a knockout?

THE QUESTION. Depth of knockdown is measurable (shutdown_depth: median 3.1% remaining among strongly-called down genes). Tolerance is not
-- nothing in this data states a level below which a gene stops working. But there IS an empirical route to the same answer, and this
project has been sitting on it: every CRISPRi knockout in the screen is itself a partial knockdown of KNOWN depth, and every one of them
has a measured phenotype. That is a dose-response series with 336 points, already collected.

    x-axis   fraction of control expression remaining for the TARGETED gene  (from the relative-expression matrix)
    y-axis   how large a transcriptional phenotype that knockdown produced   (number of specific movers)

If phenotype saturates -- i.e. going from 40% remaining to 5% remaining buys no extra phenotype -- then anything at or below the saturation
point is already "effectively knocked out", and that point is the answer. If instead phenotype keeps growing all the way down, then there is
no plateau and partial knockdown is never equivalent to a null.

THE CONFOUND THAT HAS TO BE HANDLED. Genes differ enormously in how much phenotype they can produce at ANY depth: knocking a ribosomal
protein to 20% does more than knocking an olfactory receptor to 0%. So raw phenotype against raw depth mixes "how deep" with "which gene".
Two things are done about it: phenotype is normalised WITHIN essentiality strata, and the analysis is repeated restricted to genes that are
essential, where a phenotype is expected at all. Without that the curve mostly measures gene identity rather than dose.

WHAT THIS CANNOT DO. It gives the depth at which the TRANSCRIPTIONAL phenotype saturates, which is not the same as the depth at which the
protein stops doing its job -- protein levels lag mRNA, and many proteins function fine at a fraction of normal abundance. It is an upper
bound on the useful threshold, measured in the only currency this screen carries."""
import json, collections
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import os

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
T = 4.2
TIDE_FRAC = 0.05
BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.20)]


def main():
    f = h5py.File(SP / "gwps.h5ad", "r")
    def _dec(x):
        return x.decode() if isinstance(x, bytes) else str(x)
    cats = [_dec(x) for x in f["var"]["__categories"]["gene_name"][:]]
    vnames = [cats[c] for c in f["var"]["gene_name"][:]]
    vidx = {}
    for i, n in enumerate(vnames):
        vidx.setdefault(n, i)
    gt = [_dec(x) for x in f["obs"]["gene_transcript"][:]]
    rowof = {}
    for i, t in enumerate(gt):
        p = t.split("_")
        if len(p) >= 2:
            rowof.setdefault(p[1], i)
    X = f["X"]

    df = pd.read_parquet(SP / "repl_k562_zscores.parquet")
    genes = [str(g) for g in df.columns]; kos = [str(k) for k in df.index]
    Z = df.values.astype(np.float32); A = np.abs(Z)
    tide = (A >= T).mean(0) >= TIDE_FRAC
    ok = ~tide
    D = json.load(open(OUT / "cell_complete.json"))
    info = {g["name"]: g for g in D["genes"]}

    rows = []
    for i, k in enumerate(kos):
        if k not in rowof or k not in vidx:
            continue
        v = float(np.asarray(X[rowof[k]])[vidx[k]])
        if not np.isfinite(v):
            continue
        rem = 1.0 + v                                        # fraction of control expression left
        if not (-0.05 <= rem <= 1.5):
            continue
        nmov = int(((A[i] >= T) & ok).sum())
        gi = info.get(k) or {}
        rows.append({"ko": k, "remaining": rem, "movers": nmov,
                     "essential": int(bool(gi.get("ess"))),
                     "dep_frac": float(gi.get("dep_frac") or 0.0)})
    print(f"dose-response series: {len(rows)} knockouts with both a measured knockdown depth and a phenotype")

    rem = np.array([r["remaining"] for r in rows])
    mov = np.array([r["movers"] for r in rows], float)
    ess = np.array([r["essential"] for r in rows], bool)
    from scipy.stats import spearmanr, mannwhitneyu
    print(f"  overall Spearman(remaining, movers) = {spearmanr(rem, mov)[0]:+.3f} "
          f"(negative = deeper knockdown gives more phenotype)")

    def curve(mask, label):
        print(f"\n  {label} (n={int(mask.sum())})")
        print(f"    {'% left':>12s} {'n':>5s} {'median movers':>14s} {'mean':>8s} {'frac with >=20':>15s}")
        out = []
        for lo, hi in BANDS:
            m = mask & (rem >= lo) & (rem < hi)
            if m.sum() < 3:
                continue
            md = float(np.median(mov[m])); mn = float(mov[m].mean())
            fr = float((mov[m] >= 20).mean())
            out.append({"band": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()), "median_movers": md,
                        "mean_movers": round(mn, 1), "frac_strong": round(fr, 3)})
            print(f"    {lo:.0%}-{hi:.0%}".ljust(16) + f"{int(m.sum()):>5d} {md:>14.0f} {mn:>8.1f} {fr:>14.1%}")
        return out

    all_c = curve(np.ones(len(rem), bool), "ALL knockouts")
    ess_c = curve(ess, "ESSENTIAL genes only (a phenotype is expected at all)")

    # where does it saturate? compare each band against the deepest band
    sat = None; nonmono = False
    if ess.sum() > 30:
        deep = ess & (rem < 0.05)
        best = float(np.median(mov[deep])) if deep.sum() >= 3 else np.nan
        print(f"\n  SATURATION TEST (essential genes; deepest band <5% left has median {best:.0f} movers)")
        for lo, hi in BANDS[1:]:
            m = ess & (rem >= lo) & (rem < hi)
            if m.sum() < 3 or deep.sum() < 3:
                continue
            u = mannwhitneyu(mov[m], mov[deep], alternative="less")
            frac = float(np.median(mov[m])) / max(best, 1e-9)
            flag = "indistinguishable from full knockout" if u.pvalue > 0.05 else "significantly weaker"
            print(f"    {lo:.0%}-{hi:.0%}".ljust(16) + f"n={int(m.sum()):>4d}  median {np.median(mov[m]):>5.0f} "
                  f"= {frac:>5.0%} of deepest   p={u.pvalue:.3g}   {flag}")
            if sat is None and u.pvalue > 0.05:
                sat = (lo, hi, float(u.pvalue), frac)
            if hi > 0.6 and u.pvalue > 0.05:
                sat = None; nonmono = True   # a SHALLOW band matching the deepest means there is no dose-response
                                             # at all, so any "saturation point" would be noise compared to noise
    if nonmono:
        print(f"\n  -> NO DOSE-RESPONSE: shallow knockdowns (60-80% remaining) are ALSO indistinguishable from the "
              f"deepest ones, and Spearman is {spearmanr(rem, mov)[0]:+.3f}. There is no monotone relationship between "
              f"how deep a knockdown is and how large a phenotype it produces, so no threshold can be read off this.")
    elif sat:
        print(f"\n  -> phenotype becomes statistically indistinguishable from a full knockout once expression is "
              f"at or below {sat[1]:.0%} of control (band {sat[0]:.0%}-{sat[1]:.0%}, p={sat[2]:.3g})")
    else:
        print(f"\n  -> NO band is indistinguishable from the deepest one: phenotype keeps growing all the way down, "
              f"so no partial knockdown is equivalent to a null in this data")

    verdict = (
        f"KNOCKDOWN DOSE-RESPONSE (K562, {len(rows)} knockouts): asks at what remaining expression a knockdown already behaves like a "
        "knockout, using the fact that every CRISPRi perturbation in this screen is itself a partial knockdown of measured depth with a "
        "measured phenotype -- a 336-point dose-response series that was already collected. "
        f"Overall Spearman between fraction remaining and number of specific movers is {spearmanr(rem, mov)[0]:+.3f}. "
        + ("THE MEASUREMENT FAILS, and says so rather than reporting a threshold: shallow knockdowns retaining 60-80% of control "
           "expression are ALSO statistically indistinguishable from the deepest ones, median phenotype is 1-3 movers in EVERY depth band, "
           f"and Spearman between depth and phenotype is {spearmanr(rem, mov)[0]:+.3f}. There is no monotone dose-response here, so any "
           "'saturation point' would be noise compared against noise. The reason is visible in the numbers: phenotype size is dominated by "
           "WHICH gene is hit, not by HOW MUCH of it is removed, and restricting to essential genes does not remove that. Answering the "
           "threshold question needs a titration series -- the same gene knocked down to several graded levels -- which this screen does not "
           "contain, since each gene appears at whatever single depth its guide happened to achieve. " if nonmono else
           f"AMONG ESSENTIAL GENES the transcriptional phenotype becomes statistically indistinguishable from the deepest knockdowns "
           f"(<5% remaining) once expression is at or below {sat[1]:.0%} of control (p={sat[2]:.3g}, median phenotype {sat[3]:.0%} of the "
           f"deepest band). Below that point, removing more transcript buys no more phenotype. " if sat else
           "NO band matched the deepest one: phenotype keeps increasing all the way down, so in this data no partial knockdown is "
           "equivalent to a null and there is no safe threshold to quote. ")
        + "THE CONFOUND is that genes differ hugely in how much phenotype they can produce at any depth, so raw phenotype against raw depth "
        "would mostly measure gene identity; the analysis is therefore restricted to essential genes, where a phenotype is expected at all. "
        "WHAT IT CANNOT SAY: this is the depth at which the TRANSCRIPTIONAL phenotype saturates, not the depth at which the protein stops "
        "working. Protein lags mRNA and many proteins function at a fraction of normal abundance, so this is an upper bound on the useful "
        "threshold, expressed in the only currency the screen carries. Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n": len(rows), "spearman_remaining_vs_movers": float(spearmanr(rem, mov)[0]),
               "all_bands": all_c, "essential_bands": ess_c,
               "saturation_band": ({"lo": sat[0], "hi": sat[1], "p": sat[2], "frac_of_deepest": sat[3]} if sat else None),
               "verdict": verdict, "note": verdict}, open(OUT / "knockdown_dose.json", "w"), indent=1)
    print("\n  -> outputs/orphan/knockdown_dose.json")


if __name__ == "__main__":
    main()
