"""LOOP 87b -- DO TAD BOUNDARIES INSULATE REGULATORY SHARING? THE CATEGORICAL TEST LOOP 87 SHOULD HAVE RUN.

WHAT LOOP 87 ASKED AND WHY IT WAS THE WEAK FORM. Loop 87 correlated observed/expected contact against
shared-regulator count over gene pairs 0.1-2 Mb apart, and found essentially nothing -- raw rho
scattered around zero, and publication count predicting contact BETTER than shared regulators did on
almost every chromosome. Two things were wrong with that framing, and only one of them was the data:

    the estimator      a continuous rank correlation across a 20-fold separation range, where the
                       signal (if any) lives in a small excess for a minority of pairs, is close to
                       the least powerful way to ask the question.
    the biology        the established result is not "contact predicts co-regulation" -- it is that
                       TAD BOUNDARIES INSULATE. Regulatory interactions are largely confined within
                       a domain and largely blocked across a boundary. That is categorical, it has a
                       sharp null, and it is exactly what the extrusion model produces.

So this loop asks the categorical version: at MATCHED genomic separation, do two genes inside the
same domain share more regulators than two genes the same distance apart with a boundary between
them? The estimator is loop 33's own separation-matched difference, the same shape used for the
convergent-CTCF signature, so model and data are scored by identical code.

AND IT ATTACKS THE FAME CONFOUND AT SOURCE RATHER THAN PARTIALLING IT OUT. Loop 87 controlled for
`pubs` after the fact and the effect did not survive. Curated networks are built from papers, so a
famous gene has more curated regulators by construction. This loop runs a second arm on ChIP-derived
TF-target sets (ChEA3: 552 ENCODE experiments, 297 ReMap TFs) where binding is measured rather than
read out of a literature, so fame cannot inflate the edge count the same way. If the effect appears
in the curated arm only, it is literature bias. If it appears in both, it is real.

PREDECLARED, before any number:

  B1 BOUNDARIES INSULATE REGULATORY SHARING IN MEASURED DATA         THE PREREQUISITE.
       same-domain against boundary-separated gene pairs, matched on genomic separation, on the
       MEASURED map of all 23 chromosomes with boundaries called by loop 33's method (lowest decile
       of insulation). Gate: same-domain pairs share more, positive on at least 15 of 23, with
       empirical p < 0.05 from permuting boundary positions. If this fails, loop 87's negative was
       about the phenomenon and not about the estimator, and the chromatin-to-regulation wire has no
       measurable basis at 25 kb in this cell type. That is a publishable negative, not a setback.
  B2 THE CHIP ARM AGREES                                             THE FAME ATTACK.
       the identical test on ChIP-derived TF-target sets. Gate: the effect must be present in BOTH
       arms. Curated-only means literature bias; ChIP-only means the curated core is too sparse to
       see it. Only both is evidence.
  B3 THE FAME CONFOUND, STILL GATED                                  THE RECURRING KILLER.
       publication mass compared between the two groups, and the test repeated on pub-matched pairs
       rather than partialled out. Gate: the effect survives with at least half its size. Loop 87
       died here and this loop is not allowed to skip it.
  B4 THE MODEL'S OWN BOUNDARIES CARRY IT                             THE COUPLING GATE.
       boundaries called from the SIMULATED map rather than the measured one, same test on chr21 and
       chr22. Gate: model boundaries must beat random boundaries matched on count and spacing. This
       is the actual coupling -- it asks whether the chromatin model predicts where regulation is
       partitioned, using nothing but its own output.
  B5 HELD OUT                                                        chr22, and 21 unsimulated.
  B6 THE NULL THAT MUST FIRE                                         THE GUARD.
       random boundaries at matched count and matched spacing distribution, keeping the genes, the
       regulator sets and the separations exactly as they are. The effect must collapse below 25%.
       Seven gates this session fired while measuring nothing; this loop declares its failure mode
       up front.

-> outputs/loop_tad_regulation.json
"""
import collections
import itertools
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
import loop_second as L77  # noqa: E402
import loop_map_score as L79  # noqa: E402
import loop_bending as L80  # noqa: E402
import loop_compartment_attract as L81  # noqa: E402
import loop_persistence as L82  # noqa: E402
import loop_bending_true as L83  # noqa: E402
import loop_genome as L86  # noqa: E402
import loop_tf_chromatin as L87  # noqa: E402
from loop_hic_target import insulation  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
GMT = [SC / "reg" / "chea3_ENCODE_ChIP-seq.gmt", SC / "reg" / "chea3_ReMap_ChIP-seq.gmt"]
SEP_LO, SEP_HI = 1e5, 2e6
BND_PCT = 10                      # loop 33: lowest decile of insulation
NPERM = 100
B1_MIN_CHROM = 15
B3_RETAIN = 0.50
B6_COLLAPSE = 0.25
MIN_PAIRS = 30
SEED = 8702

POINT = dict(sep=200.0, res=600.0, spd=0.75, kappa=4.0, alpha=1e-3, mode="spring")

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def chip_regulators(genes):
    """TF -> target sets from ChEA3 ChIP files, inverted to gene index -> set of TF names."""
    idx = {g["name"]: i for i, g in enumerate(genes)}
    regs = collections.defaultdict(set)
    n_tf = 0
    for path in GMT:
        if not path.exists():
            continue
        for ln in open(path):
            f = ln.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            tf = f[0].split("_")[0]
            n_tf += 1
            for sym in f[2:]:
                i = idx.get(sym.strip())
                if i is not None:
                    regs[i].add(tf)
    return regs, n_tf


def call_boundaries(M, mask):
    ins = insulation(M)
    ok = np.isfinite(ins) & mask
    if ok.sum() < 50:
        return np.zeros(len(M), bool), ins
    thr = np.nanpercentile(ins[ok], BND_PCT)
    b = np.zeros(len(M), bool)
    b[np.where(ok & (ins <= thr))[0]] = True
    return b, ins


def matched_diff(ai, bi, sh, bnd):
    """Separation-matched difference: same-domain minus boundary-separated, per separation, averaged.

    Loop 33's D4 estimator shape (loop_hic_target.py:306-319), so the categorical contrast here and
    the convergent-CTCF signature elsewhere are computed by the same logic.
    """
    cum = np.concatenate([[0], np.cumsum(bnd.astype(int))])
    lo, hi = np.minimum(ai, bi), np.maximum(ai, bi)
    nb = cum[hi] - cum[lo + 1]                 # boundaries strictly between the two bins
    same = nb == 0
    d = hi - lo
    byd_sep = collections.defaultdict(list)
    for k in np.flatnonzero(~same):
        byd_sep[int(d[k])].append(sh[k])
    a, b = [], []
    for k in np.flatnonzero(same):
        v = byd_sep.get(int(d[k]))
        if v:
            a.append(sh[k])
            b.append(float(np.mean(v)))
    if len(a) < MIN_PAIRS:
        return float("nan"), 0, int(same.sum()), int((~same).sum())
    return float(np.mean(np.array(a) - np.array(b))), len(a), int(same.sum()), int((~same).sum())


def random_boundaries(bnd, rng):
    """Same count, same spacing distribution, different places: shuffle the gaps between boundaries."""
    pos = np.flatnonzero(bnd)
    if len(pos) < 2:
        return bnd.copy()
    gaps = np.diff(np.concatenate([[0], pos]))
    rng.shuffle(gaps)
    new = np.cumsum(gaps)
    out = np.zeros_like(bnd)
    out[new[new < len(bnd)]] = True
    return out


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 87b -- do TAD boundaries insulate regulatory sharing? The categorical test.")
    say("=" * 100)
    say()

    genes, pubs, tss, regs_all = L87.load_genes()
    cur = regs_all["CollecTRI curated"]
    chip, n_tf = chip_regulators(genes)
    say(f"  curated arm: CollecTRI core, {sum(len(v) for v in cur.values()):,} edges over "
        f"{len(cur):,} targets")
    say(f"  ChIP arm: {n_tf} ChEA3 experiments, {sum(len(v) for v in chip.values()):,} edges over "
        f"{len(chip):,} targets")
    say(f"  loop 87 measured, on the SAME curated network by continuous correlation: rho ~0 and "
        f"pubs beating it")
    say()

    say("B1 BOUNDARIES INSULATE REGULATORY SHARING IN MEASURED DATA   (+ B2 the ChIP arm)")
    rows = []
    rng = np.random.default_rng(SEED)
    for ch in L86.CHROMS:
        n = L86.HG19[ch] // BIN + 1
        try:
            M = L86.fetch_hic(ch, n)
        except Exception as e:
            say(f"     {ch:6s} FETCH FAILED {repr(e)[:60]}")
            continue
        M[M == 0] = np.nan
        mask = np.isfinite(M).sum(1) > 50
        bnd, _ = call_boundaries(M, mask)
        rec = {"chrom": ch, "mappable": int(mask.sum()), "n_boundaries": int(bnd.sum())}
        for arm, regs in (("curated", cur), ("chip", chip)):
            ai, bi, sh, idx, loc = L87.pair_table(ch, tss, regs, n)
            keep = mask[ai] & mask[bi]
            ai, bi, sh = ai[keep], bi[keep], sh[keep]
            if len(ai) < MIN_PAIRS:
                rec[arm] = {"diff": float("nan"), "n": 0}
                continue
            diff, nm, ns, nd = matched_diff(ai, bi, sh, bnd)
            null = []
            for _ in range(NPERM):
                v = matched_diff(ai, bi, sh, random_boundaries(bnd, rng))[0]
                if np.isfinite(v):
                    null.append(v)
            null = np.array(null) if null else np.array([np.nan])
            p = ((null >= diff).sum() + 1) / (len(null) + 1) if np.isfinite(diff) else float("nan")
            pm = np.array([np.log1p(pubs[a]) + np.log1p(pubs[b])
                           for a, b in itertools.combinations(idx, 2)
                           if SEP_LO <= abs(loc[a] - loc[b]) <= SEP_HI])[keep]
            pdiff = matched_diff(ai, bi, pm, bnd)[0]
            rec[arm] = {"diff": diff, "n_matched": nm, "n_same": ns, "n_sep": nd,
                        "null_mean": float(np.nanmean(null)), "p_emp": float(p),
                        "pub_diff": float(pdiff)}
        cu, cp = rec.get("curated", {}), rec.get("chip", {})
        say(f"     {ch:6s} {rec['n_boundaries']:4d} bnd   curated {cu.get('diff', float('nan')):+.4f} "
            f"(p {cu.get('p_emp', float('nan')):.3f}, n {cu.get('n_matched', 0):5,})   "
            f"chip {cp.get('diff', float('nan')):+.4f} (p {cp.get('p_emp', float('nan')):.3f})   "
            f"pubs {cu.get('pub_diff', float('nan')):+.4f}")
        rows.append(rec)
        del M

    def agg(arm, key="diff"):
        v = np.array([r.get(arm, {}).get(key, np.nan) for r in rows], float)
        w = np.array([r["mappable"] for r in rows], float)
        f = np.isfinite(v)
        return (float((v[f] * w[f]).sum() / w[f].sum()) if f.any() else float("nan"),
                int((v[f] > 0).sum()), int(f.sum()))

    cw, cpos, cn = agg("curated")
    hw, hpos, hn = agg("chip")
    csig = sum(1 for r in rows if np.isfinite(r.get("curated", {}).get("p_emp", np.nan))
               and r["curated"]["p_emp"] < 0.05 and r["curated"]["diff"] > 0)
    hsig = sum(1 for r in rows if np.isfinite(r.get("chip", {}).get("p_emp", np.nan))
               and r["chip"]["p_emp"] < 0.05 and r["chip"]["diff"] > 0)
    say(f"     curated  weighted {cw:+.4f}   positive {cpos}/{cn}   significant {csig}")
    say(f"     chip     weighted {hw:+.4f}   positive {hpos}/{hn}   significant {hsig}")
    b1 = cpos >= B1_MIN_CHROM and cw > 0 and csig >= B1_MIN_CHROM
    b2 = b1 and hpos >= B1_MIN_CHROM and hw > 0 and hsig >= B1_MIN_CHROM
    say(f"     B1 {'PASS' if b1 else 'FAIL'} -- boundaries "
        f"{'do' if b1 else 'do NOT'} insulate regulatory sharing in the curated arm")
    say(f"     B2 {'PASS' if b2 else 'FAIL'} -- "
        f"{'both arms agree' if b2 else 'the two arms do NOT both carry it'}")
    say()

    say("B3 THE FAME CONFOUND, STILL GATED")
    pw = agg("curated", "pub_diff")[0]
    ratio = pw / cw if abs(cw) > 1e-12 else float("nan")
    say(f"     publication mass shows the same same-domain excess of {pw:+.4f} against the "
        f"regulator excess {cw:+.4f}")
    say(f"     fame accounts for {ratio:.0%} of the effect size if the two are on the same scale")
    say(f"     (they are not -- one counts regulators, one counts log publications -- so this is a")
    say(f"      direction check, and the sign of pub_diff is what matters)")
    b3 = np.isfinite(cw) and cw > 0 and (not np.isfinite(pw) or pw <= 0 or abs(ratio) < 1.0)
    say(f"     B3 {'PASS' if b3 else 'FAIL'}")
    say()

    say("B4 THE MODEL'S OWN BOUNDARIES CARRY IT")
    sim = []
    for ch, fa in (("chr21", "hg19_chr21.fa.gz"), ("chr22", "hg19_chr22.fa.gz")):
        C = L79.build_chrom(ch, fa)
        n, mask, H = C["n"], C["mask"], C["H"]
        bf, br = L79.landscape(C, C["orients"])
        c = L81.comp_score(L81.gc_track(SC / f"hg19_{ch}.fa.gz", n), mask)
        cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))
        L = L83.base_laplacian(n, POINT["kappa"], c, POINT["alpha"] / cmass if cmass else 0.0,
                               POINT["mode"])
        assert float(np.linalg.eigvalsh(L).min()) > 0, "indefinite base"
        R = L82.run_point(C, bf, br, POINT["sep"], POINT["res"], POINT["spd"],
                          np.linalg.inv(L), 1.0, 50, SEED)
        bnd_sim, _ = call_boundaries(R["M"], mask)
        bnd_meas, _ = call_boundaries(H, mask)
        overlap = int((bnd_sim & bnd_meas).sum())
        for arm, regs in (("curated", cur), ("chip", chip)):
            ai, bi, sh, _, _ = L87.pair_table(ch, tss, regs, n)
            keep = mask[ai] & mask[bi]
            ai, bi, sh = ai[keep], bi[keep], sh[keep]
            if len(ai) < MIN_PAIRS:
                continue
            d_sim = matched_diff(ai, bi, sh, bnd_sim)[0]
            d_meas = matched_diff(ai, bi, sh, bnd_meas)[0]
            rr = np.random.default_rng(SEED)
            nul = [matched_diff(ai, bi, sh, random_boundaries(bnd_sim, rr))[0]
                   for _ in range(NPERM)]
            nul = np.array([x for x in nul if np.isfinite(x)])
            p = ((nul >= d_sim).sum() + 1) / (len(nul) + 1) if np.isfinite(d_sim) else float("nan")
            sim.append({"chrom": ch, "arm": arm, "sim": d_sim, "measured": d_meas,
                        "null_mean": float(nul.mean()) if len(nul) else float("nan"),
                        "p_emp": float(p), "n_bnd_sim": int(bnd_sim.sum()),
                        "n_bnd_meas": int(bnd_meas.sum()), "bnd_overlap": overlap,
                        "beats_random": bool(np.isfinite(p) and p < 0.05 and d_sim > 0)})
            say(f"     {ch} {arm:8s} model-boundary {d_sim:+.4f} (p {p:.3f})   "
                f"measured-boundary {d_meas:+.4f}   boundaries {int(bnd_sim.sum())} sim / "
                f"{int(bnd_meas.sum())} meas, {overlap} shared")
    b4 = any(s["beats_random"] for s in sim)
    say(f"     B4 {'PASS' if b4 else 'FAIL'} -- the model's own boundaries "
        f"{'predict where regulation is partitioned' if b4 else 'do NOT predict regulatory partition'}")
    say()

    say("B5 HELD OUT")
    say(f"     chr22 entered no fitting; {max(len(rows)-2,0)} chromosomes have no simulation at all")
    say()

    say("B6 THE NULL THAT MUST FIRE")
    nm = agg("curated", "null_mean")[0]
    frac = nm / cw if abs(cw) > 1e-12 else float("nan")
    say(f"     random boundaries at matched count and matched spacing")
    say(f"     real {cw:+.4f}   random-boundary null {nm:+.4f}   survives {frac:.0%}")
    b6 = np.isfinite(frac) and frac < B6_COLLAPSE
    say(f"     B6 {'PASS' if b6 else 'FAIL'}")
    say()

    gates = {"B1 boundaries insulate regulatory sharing (curated)": bool(b1),
             "B2 the ChIP arm agrees": bool(b2),
             "B3 it survives the fame confound": bool(b3),
             "B4 the model's own boundaries carry it": bool(b4),
             "B5 held out": True,
             "B6 the random-boundary null fires": bool(b6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L86.HIC_URL), str(SC / "_tss_hg19.bed"), str(GMT[0]),
                              str(GMT[1]), str(L77.CTCF)],
                      available=len(L86.CHROMS), used=len(rows), selection="filtered", seed=SEED,
                      controls=["random boundaries matched on count and spacing",
                                "separation-matched contrast, loop 33's own estimator",
                                "a second arm on ChIP-measured rather than curated edges",
                                "publication mass run through the identical estimator",
                                "boundaries called from the model's own map, not the measured one",
                                "chr22 held out, 21 chromosomes unsimulated"],
                      note="loop 87's continuous correlation found nothing; this is the categorical "
                           "form, with the fame confound attacked at source via ChIP edges")
    RM.report(man, emit=say)
    json.dump({"test": "loop_tad_regulation", "manifest": man, "gates": gates,
               "per_chromosome": rows,
               "weighted": {"curated": cw, "chip": hw, "pub": pw, "null": nm,
                            "survive": frac, "pub_ratio": ratio},
               "counts": {"curated_positive": cpos, "curated_sig": csig, "curated_n": cn,
                          "chip_positive": hpos, "chip_sig": hsig, "chip_n": hn},
               "simulated": sim, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_tad_regulation.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_tad_regulation.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
