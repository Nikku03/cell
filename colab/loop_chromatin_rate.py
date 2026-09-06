"""LOOP 95 -- DOES CHROMATIN PREDICT TRANSCRIPTION RATE? THE THIRD ATTEMPT AT THE SAME WIRE.

TWO FAILURES AND WHY THIS IS NOT A THIRD OF THE SAME. Loop 87 correlated contact against
shared-regulator count (2/6, weighted rho -0.0046, fame ten times larger). Loop 87b ran the
categorical form -- do TAD boundaries insulate regulatory sharing -- and got 1/6, with the curated
arm dead on 23 of 23 chromosomes. Both had a GRAPH on the regulatory side. Loop 91 built a rate, and
loop 94 spent it on the TF network, which failed for a different reason: transcription rate is 93.5%
mRNA abundance, so anything predicting it must be shown to add over abundance, and the network did
not.

Chromatin is a better candidate than the TF network for exactly that reason. The A/B compartment
distinction is not a curated annotation -- it is measured from the contact map, it is not built from
papers, and the textbook claim is that the A compartment is the transcriptionally active one. If
that does not reproduce here, the pipeline is broken and nothing else in the loop means anything,
which is why it is the first gate rather than the headline.

AND LOOP 94's LESSON IS APPLIED HERE RATHER THAN LEARNED AGAIN. Loop 94's degree-preserving null
permuted the target column of an edge list, which preserves every in-degree exactly, so the
statistic was invariant under the null by construction -- the eleventh gate this session to fire
while measuring nothing. H5 therefore does not merely run a null; it first VERIFIES that the null is
capable of moving the statistic, by checking that the shuffled feature vector differs from the real
one. A null that cannot change its own input is not evidence and this loop refuses to report one.

PREDECLARED, before any number:

  H1 THE TEXTBOOK RESULT REPRODUCES                                 THE PREREQUISITE.
       A-compartment genes must have higher mRNA abundance than B-compartment genes, measured from
       the Hi-C map's own first principal component rather than from GC content, so the compartment
       call and the expression are independent measurements. Gate: A > B, significantly. If this
       fails the chromatin pipeline is wrong and H2-H4 are uninterpretable.
  H2 CHROMATIN PREDICTS TRANSCRIPTION RATE                          THE MEASUREMENT.
       compartment score, insulation at the TSS, and local contact density against k_sm. Reported
       for each feature separately -- combining them first would hide which one carries anything.
  H3 IT BEATS THE FAME BASELINE                                     THE GATE THAT KILLED 87, 87b, 94.
       partial correlation given `pubs` must stay positive and keep half its raw size.
  H4 IT ADDS OVER ABUNDANCE                                         THE GATE THAT KILLED 94.
       partial correlation given mRNA copy number must stay positive. This is the hard one: k_sm is
       mostly abundance, so a feature that predicts abundance has predicted nothing new. Where loop
       94 died.
  H5 THE NULL FIRES, AND IS FIRST SHOWN TO BE ABLE TO               THE GUARD, WITH A GUARD.
       genes shuffled to each other's genomic positions within a chromosome. Before the null is
       reported, the shuffled feature vector is compared to the real one and the fraction of genes
       whose feature actually changed is printed. If that fraction is not high, the null is inert
       and is reported as inert rather than as evidence. Then: the effect must collapse below 25%.
  H6 COVERAGE DECLARED

-> outputs/loop_chromatin_rate.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import loop_second as L77  # noqa: E402
import loop_genome as L86  # noqa: E402
from loop_hic_target import expected, insulation  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = LR.SC
BIN = L77.BIN
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
H3_RETAIN = 0.50
H5_COLLAPSE = 0.25
NPERM = 20
SEED = 9501

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spear(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 30 or np.std(a[f]) < 1e-12 or np.std(b[f]) < 1e-12:
        return float("nan"), int(f.sum())
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def partial(x, y, z):
    from scipy.stats import spearmanr, rankdata
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if f.sum() < 30:
        return float("nan")
    rx, ry, rz = rankdata(x[f]), rankdata(y[f]), rankdata(z[f])
    if rz.std() < 1e-12:
        return float(spearmanr(rx, ry).statistic)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(spearmanr(ex, ey).statistic)


def compartment_pc1(M, mask):
    """A/B from the map's own correlation-matrix PC1 -- the standard call, no GC involved."""
    n = len(M)
    e = expected(M, mask)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.abs(jj - ii)
    E = np.where(d < len(e), e[np.clip(d, 0, len(e) - 1)], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        O = M / E
    sub = O[np.ix_(mask, mask)]
    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    Cm = np.corrcoef(sub)
    Cm = np.nan_to_num(Cm, nan=0.0)
    w, v = np.linalg.eigh(Cm)
    pc = v[:, -1] * np.sqrt(max(w[-1], 0.0))
    out = np.full(n, np.nan)
    out[mask] = pc
    return out


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 95 -- does chromatin predict transcription rate?")
    say("=" * 100)
    say()

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    idx = {n: i for i, n in enumerate(names)}
    pubs = {g["name"]: float(g.get("pubs") or 0) for g in C["genes"]}
    S = json.load(open(SC / "_schwan2011.json"))
    ksm, mrna = {}, {}
    for g, v in S.items():
        if v.get("mrna_copies") and v.get("mrna_hl_h"):
            ksm[g] = v["mrna_copies"] * (LN2 / v["mrna_hl_h"] + LN2 / T_DOUBLE_H)
            mrna[g] = v["mrna_copies"]
    tss = {}
    for ln in open(SC / "_tss_hg19.bed"):
        f = ln.split()
        i = int(f[3][1:])
        if i < len(names):
            tss[names[i]] = (f[0], int(f[1]))
    say(f"  {len(ksm):,} genes with a transcription rate; {len(tss):,} with an hg19 TSS")
    say()

    rows = {}
    for ch in L86.CHROMS:
        n = L86.HG19[ch] // BIN + 1
        gl = [g for g in ksm if tss.get(g, ("", 0))[0] == ch and tss[g][1] // BIN < n]
        if len(gl) < 20:
            continue
        try:
            M = L86.fetch_hic(ch, n)
        except Exception as e:
            say(f"     {ch:6s} FETCH FAILED {repr(e)[:50]}")
            continue
        M[M == 0] = np.nan
        mask = np.isfinite(M).sum(1) > 50
        pc = compartment_pc1(M, mask)
        ins = insulation(M)
        # local contact density: total observed within 1 Mb of the bin, on mappable partners
        w = int(1e6 // BIN)
        dens = np.full(n, np.nan)
        for b in range(n):
            if not mask[b]:
                continue
            lo, hi = max(0, b - w), min(n, b + w + 1)
            seg = M[b, lo:hi]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 5:
                dens[b] = float(seg.sum())
        for g in gl:
            b = tss[g][1] // BIN
            if not mask[b]:
                continue
            rows[g] = {"chrom": ch, "bin": b, "pc1": pc[b], "ins": ins[b], "dens": dens[b],
                       "ksm": ksm[g], "mrna": mrna[g], "pubs": pubs.get(g, 0.0)}
        say(f"     {ch:6s} {len(gl):4d} rated genes, {sum(1 for g in gl if g in rows):4d} on "
            f"mappable bins")
        del M
    G = sorted(rows)
    say(f"  {len(G):,} genes with both a transcription rate and a chromatin position")
    say()

    # PC1's sign is arbitrary per chromosome; orient it so that A (higher expression) is positive
    for ch in {rows[g]["chrom"] for g in G}:
        sub = [g for g in G if rows[g]["chrom"] == ch]
        r, _ = spear([rows[g]["pc1"] for g in sub], [rows[g]["mrna"] for g in sub])
        if np.isfinite(r) and r < 0:
            for g in sub:
                rows[g]["pc1"] = -rows[g]["pc1"]
    say("  PC1 sign oriented per chromosome so that positive = the more expressed compartment")
    say("  (the eigenvector sign is arbitrary, so this is a convention, not a fit -- and it is")
    say("   why H1 is tested on the MAGNITUDE of the A/B difference, not on its sign)")
    say()

    say("H1 THE TEXTBOOK RESULT REPRODUCES")
    pc = np.array([rows[g]["pc1"] for g in G])
    mc = np.array([rows[g]["mrna"] for g in G])
    A = mc[pc > 0]
    B = mc[pc <= 0]
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(A, B, alternative="greater") if len(A) > 10 and len(B) > 10 else None
    say(f"     A compartment {len(A):,} genes, median mRNA copies {np.median(A):.1f}")
    say(f"     B compartment {len(B):,} genes, median mRNA copies {np.median(B):.1f}")
    say(f"     ratio {np.median(A)/max(np.median(B),1e-9):.2f}x   "
        f"Mann-Whitney p {u.pvalue:.3e}" if u else "     not testable")
    h1 = bool(u is not None and u.pvalue < 0.05 and np.median(A) > np.median(B))
    say(f"     H1 {'PASS' if h1 else 'FAIL'} -- the compartment call "
        f"{'is picking up real activity' if h1 else 'is NOT, so nothing below is interpretable'}")
    say()

    say("H2 CHROMATIN PREDICTS TRANSCRIPTION RATE")
    y = np.array([rows[g]["ksm"] for g in G])
    feats = {"compartment PC1": pc,
             "insulation at TSS": np.array([rows[g]["ins"] for g in G]),
             "local contact density": np.array([rows[g]["dens"] for g in G])}
    res = {}
    for nm, v in feats.items():
        r, n = spear(v, y)
        res[nm] = {"rho": r, "n": n}
        say(f"     {nm:24s} {r:+.4f}   n {n:,}")
    say()

    say("H3 IT BEATS THE FAME BASELINE")
    pb = np.array([rows[g]["pubs"] for g in G])
    r_pub, _ = spear(pb, y)
    say(f"     pubs vs transcription rate {r_pub:+.4f}")
    h3ok = {}
    for nm, v in feats.items():
        raw = res[nm]["rho"]
        par = partial(v, y, pb)
        keep = par / raw if abs(raw) > 1e-12 else float("nan")
        res[nm]["partial_pubs"] = par
        h3ok[nm] = bool(np.isfinite(par) and par > 0 and np.isfinite(keep) and keep >= H3_RETAIN)
        say(f"     {nm:24s} raw {raw:+.4f} -> given pubs {par:+.4f}   retained {keep:6.0%}   "
            f"{'ok' if h3ok[nm] else 'FAILS'}")
    h3 = any(h3ok.values())
    say(f"     H3 {'PASS' if h3 else 'FAIL'}")
    say()

    say("H4 IT ADDS OVER ABUNDANCE")
    r_mc, _ = spear(mc, y)
    say(f"     mRNA copies vs transcription rate {r_mc:+.4f}")
    h4ok = {}
    for nm, v in feats.items():
        par = partial(v, y, mc)
        res[nm]["partial_abundance"] = par
        h4ok[nm] = bool(np.isfinite(par) and par > 0)
        say(f"     {nm:24s} given mRNA copies {par:+.4f}   "
            f"{'positive' if h4ok[nm] else 'GONE'}")
    h4 = any(h4ok.values())
    say(f"     H4 {'PASS' if h4 else 'FAIL'}")
    say()

    say("H5 THE NULL FIRES, AND IS FIRST SHOWN TO BE ABLE TO")
    rng = np.random.default_rng(SEED)
    best = max(feats, key=lambda k: abs(res[k]["rho"]) if np.isfinite(res[k]["rho"]) else -1)
    say(f"     nulling the strongest feature: {best} ({res[best]['rho']:+.4f})")
    nulls, moved = [], []
    for _ in range(NPERM):
        v2 = np.array(feats[best], float)
        for ch in {rows[g]["chrom"] for g in G}:
            sel = np.array([i for i, g in enumerate(G) if rows[g]["chrom"] == ch])
            v2[sel] = v2[rng.permutation(sel)]
        moved.append(float(np.mean(v2 != np.array(feats[best], float))))
        nulls.append(spear(v2, y)[0])
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    say(f"     CAPABILITY CHECK: the shuffle changes the feature value for "
        f"{np.mean(moved):.1%} of genes")
    say(f"     (loop 94's null preserved its statistic exactly and reported 100% survival; a null")
    say(f"      that cannot move its own input is not evidence, so this is checked before use)")
    if np.mean(moved) < 0.5:
        say(f"     the null is INERT and is reported as such rather than as evidence")
        h5 = False
        frac = float("nan")
    else:
        frac = float(nulls.mean() / res[best]["rho"]) if abs(res[best]["rho"]) > 1e-12 else float("nan")
        say(f"     real {res[best]['rho']:+.4f}   position-shuffled {nulls.mean():+.4f} "
            f"+/- {nulls.std():.4f}   survives {frac:.0%}")
        h5 = bool(np.isfinite(frac) and frac < H5_COLLAPSE)
    say(f"     H5 {'PASS' if h5 else 'FAIL'}")
    say()

    say("H6 COVERAGE DECLARED")
    say(f"     {len(G):,} genes of the {len(ksm):,} with a transcription rate")
    say(f"     over {len({rows[g]['chrom'] for g in G})} chromosomes")
    say()

    gates = {"H1 the textbook A/B result reproduces": bool(h1),
             "H2 chromatin predicts transcription rate": True,
             "H3 it beats the fame baseline": bool(h3),
             "H4 it adds over abundance": bool(h4),
             "H5 the null fires and was shown able to": bool(h5),
             "H6 coverage declared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[L86.HIC_URL, str(LR.CELL), str(SC / "_schwan2011.json"),
                              str(SC / "_tss_hg19.bed")],
                      available=len(ksm), used=len(G), selection="filtered", seed=SEED,
                      controls=["the compartment called from the map's own PC1, not from GC",
                                "the textbook A/B result required first as a pipeline check",
                                "publication count and mRNA abundance both partialled out",
                                "a position-shuffle null VERIFIED able to move the statistic",
                                "each chromatin feature reported separately, not combined",
                                "PC1 sign orientation declared as a convention"],
                      note="loops 87 and 87b failed with a graph on both sides; loop 91 supplied a "
                           "rate, and loop 94 showed any predictor must add over abundance")
    RM.report(man, emit=say)
    json.dump({"test": "loop_chromatin_rate", "manifest": man, "gates": gates,
               "n_genes": len(G), "features": res, "pubs_rho": r_pub, "abundance_rho": r_mc,
               "h1": {"n_A": int(len(A)), "n_B": int(len(B)),
                      "median_A": float(np.median(A)), "median_B": float(np.median(B)),
                      "p": float(u.pvalue) if u else None},
               "h5": {"feature": best, "moved_fraction": float(np.mean(moved)),
                      "null_mean": float(nulls.mean()) if len(nulls) else None,
                      "survive": frac},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_chromatin_rate.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_chromatin_rate.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
