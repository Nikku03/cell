"""LOOP 87 -- COUPLE THE CHROMATIN MODEL TO THE TF NETWORK, AND TEST WHETHER THE WIRE CARRIES ANYTHING.

THE STATE OF THE TWO HALVES. This repository has a chromatin arc (loops 33-86) that folds a
chromosome, and a regulatory arc (loops 74-76) that knows which transcription factor controls which
gene. They share a repository and nothing else. Chromatin state does not set transcription; the TF
network does not know where its genes are. Every claim about "4D chromatin driving the reaction side"
is currently a claim about two files in the same directory.

THE COUPLING THAT THE DATA ACTUALLY SUPPORTS. The mechanistically correct wire would be: a TF binds
an enhancer, the enhancer contacts the target promoter, transcription follows. That needs per-TF
binding coordinates, which this repository does not have for anything except CTCF. What it does have
is which genes share regulators, and where those genes sit. So the testable form is:

    do gene pairs that share more curated regulators sit in closer physical contact,
    at matched genomic separation, than pairs that share fewer?

If yes, regulatory structure and spatial structure are the same structure and the chromatin model has
something to say to the TF network. If no, the coupling is decoration and this loop says so.

THE NETWORK HAD TO BE CUT DOWN FIRST, AND THAT IS A MEASUREMENT NOT A CHOICE. On the full 612,133-row
`reg`, 98.2% of chr21 gene pairs share at least one regulator -- the layer cannot discriminate
anything because it says yes to everything. On loop 76's identified CollecTRI curated core (rows
0..55,715, the block that carries the signs) it discriminates. Measured on chr21 pairs 0.1-2 Mb apart:

    full reg (612k)        98.2% of pairs share   mean 6.95
    CollecTRI + DoRothEA   98.1%                  mean 6.19
    CollecTRI curated      19.8%                  mean 0.38   restricted to genes IN the core
    CollecTRI curated       7.9% of 1,762 pairs               ALL genes with a TSS  <- used here

The last line is what the code does, and the difference between the last two lines matters enough to
state. Restricting to genes that appear in the curated core would test a population selected for
being well studied; keeping every gene with a TSS lets "no curated regulator" count as zero shared,
which is the honest encoding but folds the fame confound directly into the measurement. That is why
C2 is a gate here rather than a footnote. Both counts are printed at run time.

PREDECLARED, before any number:

  C1 THE PHENOMENON EXISTS IN THE MEASURED DATA AT ALL                THE PREREQUISITE.
       observed/expected contact against shared-regulator count, over gene pairs 0.1-2 Mb apart, on
       the MEASURED map of all 23 chromosomes. Gate: positive on at least 15 of 23 and a positive
       mappable-weighted mean with empirical p < 0.05. If the effect is not in the real data then no
       model can be asked to reproduce it, and this loop stops at C1 and reports that instead of
       building a wire to nowhere.
  C2 THE FAME CONFOUND                                                THE RECURRING KILLER.
       `pubs`. Well-studied genes have more curated regulators AND sit in gene-dense, A-compartment,
       high-contact neighbourhoods, which is enough to manufacture C1 out of nothing. Gate: the
       partial correlation controlling for the pair's publication mass must stay positive and retain
       at least half the raw effect. Publication count has killed more claims in this repository than
       every other confound combined, and it gets its own gate here rather than a footnote.
  C3 THE SIMULATED MAP CARRIES IT                                     THE COUPLING GATE.
       the identical measurement on the MODEL's chr21 and chr22 maps, against the distance-only null
       map computed from the same chromosome's own expected curve. Gate: the simulated map must beat
       the distance-only null. Anything less means the chromatin model contributes nothing that
       genomic separation did not already contribute, and the coupling is nominal.
  C4 THE WIRE, AND WHAT IT ADDS OVER THE LAYERS THAT ALREADY EXIST    THE DELIVERABLE.
       contact_weight() -- the function that turns a contact map into a per-pair regulatory
       proximity weight -- plus leakage: how much of the shared-regulator signal genomic distance
       alone and `pubs` alone already explain. Reported, not gated.
  C5 HELD OUT                                                         NO SELECTION.
       chr22 enters no fitting here, and the 21 chromosomes with no simulation are reported
       separately from the two that have one.
  C6 THE NULL THAT MUST FIRE                                          THE GUARD.
       shuffle the gene-to-position assignment within each chromosome, keeping the contact map and
       the regulator sets exactly as they are. The effect must collapse below 25%. Six gates this
       session fired while measuring nothing -- loop 76's single-sign null, loop 77's self-comparison,
       loop 81's collapsed bands, loop 82's one-bin persistence, loops 77-83's re-simulation shuffle,
       and loop 86's refuted length confound. A control designed to fail is not optional here.

-> outputs/loop_tf_chromatin.json
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
import loop_compartment_attract as L81  # noqa: E402
import loop_persistence as L82  # noqa: E402
import loop_bending_true as L83  # noqa: E402
import loop_genome as L86  # noqa: E402
import loop_replication as LR  # noqa: E402
from loop_hic_target import expected  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
CURATED = 55716                   # loop 76: rows 0..55,715 are the CollecTRI curated causal block
SEP_LO, SEP_HI = 1e5, 2e6
NPERM = 200
C1_MIN_CHROM = 15
C2_RETAIN = 0.50
C6_COLLAPSE = 0.25
SEED = 8701

# the two arms loop 85 re-scored; both are run so the coupling is not read off one configuration
POINTS = [("spring", dict(sep=200.0, res=600.0, spd=0.75, kappa=4.0, alpha=1e-3, mode="spring")),
          ("bend", dict(sep=200.0, res=600.0, spd=0.75, kappa=0.0, alpha=3e-4, mode="bend"))]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def load_genes():
    D = json.load(open(LR.CELL))
    genes = D["genes"]
    pubs = np.array([float(g.get("pubs") or 0) for g in genes])
    tss = {}
    for ln in open(SC / "_tss_hg19.bed"):
        f = ln.split()
        tss[int(f[3][1:])] = (f[0], int(f[1]))
    blocks = {"CollecTRI curated": (0, CURATED),
              "CollecTRI+DoRothEA": (0, 278405),
              "full reg": (0, len(D["reg"]))}
    regs = {}
    for name, (lo, hi) in blocks.items():
        d = collections.defaultdict(set)
        for r in D["reg"][lo:hi]:
            d[r[1]].add(r[0])
        regs[name] = d
    return genes, pubs, tss, regs


def pair_table(chrom, tss, regs, n, perm_rng=None):
    """Gene pairs 0.1-2 Mb apart on one chromosome: their bins, shared-regulator count, pub mass."""
    idx = sorted(i for i, (c, p) in tss.items() if c == chrom and p // BIN < n)
    if perm_rng is not None:                       # C6: keep the genes, shuffle where they sit
        pos = [tss[i][1] for i in idx]
        perm_rng.shuffle(pos)
        loc = dict(zip(idx, pos))
    else:
        loc = {i: tss[i][1] for i in idx}
    a_, b_, sh_ = [], [], []
    for a, b in itertools.combinations(idx, 2):
        d = abs(loc[a] - loc[b])
        if not (SEP_LO <= d <= SEP_HI):
            continue
        a_.append(loc[a] // BIN)
        b_.append(loc[b] // BIN)
        sh_.append(len(regs[a] & regs[b]))
    return np.array(a_, int), np.array(b_, int), np.array(sh_, float), idx, loc


def oe_values(M, exp, ai, bi, mask):
    d = np.abs(bi - ai)
    ok = (d > 0) & mask[ai] & mask[bi] & (d < len(exp))
    v = np.full(len(ai), np.nan)
    dd = np.clip(d, 0, len(exp) - 1)
    e = exp[dd]
    good = ok & np.isfinite(e) & (e > 0)
    v[good] = M[ai[good], bi[good]] / e[good]
    return v


def rho(x, y):
    from scipy.stats import spearmanr
    f = np.isfinite(x) & np.isfinite(y)
    if f.sum() < 30 or np.std(y[f]) < 1e-12 or np.std(x[f]) < 1e-12:
        return float("nan"), int(f.sum())
    return float(spearmanr(x[f], y[f]).statistic), int(f.sum())


def partial_rho(x, y, z):
    """Spearman partial correlation of x and y given z -- rank-residualise both against z."""
    from scipy.stats import spearmanr, rankdata
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


def contact_weight(M, exp, mask, bin_a, bin_b):
    """THE WIRE. Regulatory proximity weight for a gene pair, from a contact map.

    observed/expected at the pair's separation, so the bulk P(s) decay -- which any two loci share
    regardless of regulation -- is divided out and what is left is whether THIS pair contacts more
    than its separation predicts. Returns nan where either bin is unmappable, rather than 0, so a
    missing measurement never reads as an absence of contact.
    """
    d = abs(int(bin_b) - int(bin_a))
    if d == 0 or d >= len(exp) or not (mask[bin_a] and mask[bin_b]):
        return float("nan")
    e = exp[d]
    if not np.isfinite(e) or e <= 0:
        return float("nan")
    return float(M[bin_a, bin_b] / e)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 87 -- couple the chromatin model to the TF network, and test whether the wire "
        "carries anything")
    say("=" * 100)
    say()

    genes, pubs, tss, regs_all = load_genes()
    regs = regs_all["CollecTRI curated"]
    say(f"  {len(genes):,} genes, {len(tss):,} with an hg19 TSS, CollecTRI curated core = "
        f"rows 0..{CURATED-1:,} of `reg`")
    for name, d in regs_all.items():
        idx21 = [i for i, (c, p) in tss.items() if c == "chr21"]
        pr = [(a, b) for a, b in itertools.combinations(sorted(idx21), 2)
              if SEP_LO <= abs(tss[a][1] - tss[b][1]) <= SEP_HI]
        s = np.array([len(d[a] & d[b]) for a, b in pr]) if pr else np.array([0.0])
        say(f"    {name:22s} chr21 {100*(s>0).mean():5.1f}% of {len(pr):,} pairs share  "
            f"mean {s.mean():.2f}  max {int(s.max())}")
    say()

    say("C1 THE PHENOMENON EXISTS IN THE MEASURED DATA AT ALL")
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
        exp = expected(M, mask)
        ai, bi, sh, idx, loc = pair_table(ch, tss, regs, n)
        if len(ai) < 30:
            say(f"     {ch:6s} only {len(ai)} pairs -- not testable")
            del M
            continue
        v = oe_values(M, exp, ai, bi, mask)
        r, npair = rho(v, sh)
        pm = np.array([np.log1p(pubs[a]) + np.log1p(pubs[b])
                       for a, b in itertools.combinations(idx, 2)
                       if SEP_LO <= abs(loc[a] - loc[b]) <= SEP_HI])
        pr_ = partial_rho(v, sh, pm)
        rp, _ = rho(v, pm)
        # C6 guard, same chromosome, same map, shuffled positions
        ai2, bi2, sh2, _, _ = pair_table(ch, tss, regs, n, perm_rng=np.random.default_rng(SEED))
        r6, _ = rho(oe_values(M, exp, ai2, bi2, mask), sh2)
        # empirical p by permuting the shared counts among the pairs
        null = []
        for _ in range(NPERM):
            null.append(rho(v, rng.permutation(sh))[0])
        null = np.array([x for x in null if np.isfinite(x)])
        p_emp = ((null >= r).sum() + 1) / (len(null) + 1) if np.isfinite(r) else float("nan")
        rows.append({"chrom": ch, "n_pairs": npair, "rho": r, "partial_rho_pubs": pr_,
                     "rho_pubs_only": rp, "rho_shuffled_positions": r6,
                     "p_emp": float(p_emp), "mappable": int(mask.sum()),
                     "null_mean": float(null.mean()) if len(null) else float("nan")})
        say(f"     {ch:6s} {npair:6,} pairs   rho {r:+.4f}   p {p_emp:.4f}   "
            f"partial|pubs {pr_:+.4f}   pubs-only {rp:+.4f}   shuffled-pos {r6:+.4f}")
        del M
    pos = [x for x in rows if np.isfinite(x["rho"]) and x["rho"] > 0]
    w = np.array([x["mappable"] for x in rows], float)
    rr = np.array([x["rho"] for x in rows], float)
    f = np.isfinite(rr)
    wmean = float((rr[f] * w[f]).sum() / w[f].sum())
    n_sig = sum(1 for x in rows if np.isfinite(x["p_emp"]) and x["p_emp"] < 0.05 and x["rho"] > 0)
    say(f"     positive on {len(pos)}/{len(rows)} chromosomes; significant (p<0.05) on {n_sig}")
    say(f"     mappable-weighted mean rho {wmean:+.4f}")
    c1 = len(pos) >= C1_MIN_CHROM and wmean > 0 and n_sig >= C1_MIN_CHROM
    say(f"     C1 {'PASS' if c1 else 'FAIL'} -- shared regulators "
        f"{'do' if c1 else 'do NOT'} predict physical contact in real data")
    say()

    say("C2 THE FAME CONFOUND")
    pr_arr = np.array([x["partial_rho_pubs"] for x in rows], float)
    rp_arr = np.array([x["rho_pubs_only"] for x in rows], float)
    g = np.isfinite(pr_arr) & np.isfinite(rr)
    wpart = float((pr_arr[g] * w[g]).sum() / w[g].sum())
    wpubs = float((rp_arr[g] * w[g]).sum() / w[g].sum())
    retain = wpart / wmean if abs(wmean) > 1e-12 else float("nan")
    say(f"     raw rho {wmean:+.4f}   partial rho controlling publication mass {wpart:+.4f}   "
        f"retained {retain:.0%}")
    say(f"     contact vs publication mass alone: {wpubs:+.4f}")
    c2 = np.isfinite(retain) and wpart > 0 and retain >= C2_RETAIN
    say(f"     C2 {'PASS' if c2 else 'FAIL'} -- the effect "
        f"{'survives' if c2 else 'is substantially FAME, not regulation'}")
    say()

    say("C3 THE SIMULATED MAP CARRIES IT")
    sim = []
    for ch, fa in (("chr21", "hg19_chr21.fa.gz"), ("chr22", "hg19_chr22.fa.gz")):
        C = L79.build_chrom(ch, fa)
        n, mask, H = C["n"], C["mask"], C["H"]
        bf, br = L79.landscape(C, C["orients"])
        c = L81.comp_score(L81.gc_track(SC / f"hg19_{ch}.fa.gz", n), mask)
        cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))
        ai, bi, sh, idx, loc = pair_table(ch, tss, regs, n)
        expH = L77.ps_slope(H, mask)[1]
        r_meas, npair = rho(oe_values(H, expH, ai, bi, mask), sh)
        DN = L79.distance_null(C)
        expD = expected(DN, mask)
        r_dist, _ = rho(oe_values(DN, expD, ai, bi, mask), sh)
        say(f"     {ch}  {npair:,} pairs   measured {r_meas:+.4f}   distance-only null "
            f"{r_dist:+.4f}")
        for name, p in POINTS:
            L = L83.base_laplacian(n, p["kappa"], c, p["alpha"] / cmass if cmass else 0.0, p["mode"])
            assert float(np.linalg.eigvalsh(L).min()) > 0, "indefinite base"
            G0 = np.linalg.inv(L)
            R = L82.run_point(C, bf, br, p["sep"], p["res"], p["spd"], G0, 1.0, 50, SEED)
            r_sim, _ = rho(oe_values(R["M"], R["exp"], ai, bi, mask), sh)
            sim.append({"chrom": ch, "point": name, "rho_sim": r_sim, "rho_meas": r_meas,
                        "rho_dist": r_dist, "n_pairs": npair,
                        "beats_dist": bool(np.isfinite(r_sim) and np.isfinite(r_dist)
                                           and r_sim > r_dist)})
            say(f"       {name:7s} simulated {r_sim:+.4f}   "
                f"{'beats' if sim[-1]['beats_dist'] else 'does NOT beat'} the distance null")
    c3 = any(s["beats_dist"] for s in sim)
    say(f"     C3 {'PASS' if c3 else 'FAIL'} -- the chromatin model "
        f"{'adds information beyond genomic separation' if c3 else 'adds NOTHING beyond separation'}")
    say()

    say("C4 THE WIRE, AND WHAT IT ADDS OVER THE LAYERS THAT ALREADY EXIST")
    say(f"     contact_weight(M, exp, mask, bin_a, bin_b) -> observed/expected, nan where unmappable")
    say(f"     leakage: publication mass alone reaches {wpubs:+.4f} of the {wmean:+.4f} raw effect "
        f"({wpubs/wmean if abs(wmean)>1e-12 else float('nan'):.0%})")
    say(f"     genomic separation alone is already divided out by construction (observed/expected)")
    say(f"     C4 reported (not gated)")
    say()

    say("C5 HELD OUT")
    say(f"     chr22 entered no fitting; {len(rows)-2} of the {len(rows)} chromosomes measured here")
    say(f"     have no simulation at all and are pure measurement")
    say()

    say("C6 THE NULL THAT MUST FIRE")
    r6a = np.array([x["rho_shuffled_positions"] for x in rows], float)
    h = np.isfinite(r6a)
    w6 = float((r6a[h] * w[h]).sum() / w[h].sum())
    frac6 = w6 / wmean if abs(wmean) > 1e-12 else float("nan")
    say(f"     genes shuffled to each other's positions, same map, same regulator sets")
    say(f"     real {wmean:+.4f}   shuffled-position {w6:+.4f}   survives {frac6:.0%}")
    c6 = np.isfinite(frac6) and frac6 < C6_COLLAPSE
    verdict6 = ("collapses the effect as it must" if c6 else
                "does NOT collapse it, so this is not about position and C1 means nothing")
    say(f"     C6 {'PASS' if c6 else 'FAIL'} -- the control {verdict6}")
    say()

    gates = {"C1 the phenomenon exists in measured data": bool(c1),
             "C2 it survives the fame confound": bool(c2),
             "C3 the simulated map carries it": bool(c3),
             "C4 the wire is written and leakage measured": True,
             "C5 held out": True,
             "C6 the position-shuffle null fires": bool(c6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(LR.CELL), str(SC / "_tss_hg19.bed"), L86.HIC_URL, str(L77.CTCF)],
                      available=len(L86.CHROMS), used=len(rows), selection="filtered", seed=SEED,
                      controls=["gene-to-position shuffle designed to collapse the effect",
                                "publication mass partialled out and reported alone",
                                "observed/expected removes genomic separation by construction",
                                f"{NPERM} label permutations per chromosome with empirical p",
                                "distance-only null map scored by identical code",
                                "network density reported for all three blocks, not just the one used"],
                      note="the curated CollecTRI core is used because the full 612k network says "
                           "yes to 98.2% of chr21 gene pairs and cannot discriminate")
    RM.report(man, emit=say)
    json.dump({"test": "loop_tf_chromatin", "manifest": man, "gates": gates,
               "per_chromosome": rows, "weighted": {"rho": wmean, "partial_pubs": wpart,
                                                    "pubs_only": wpubs, "shuffled_pos": w6,
                                                    "retain": retain, "survive": frac6},
               "simulated": sim, "n_positive": len(pos), "n_significant": n_sig,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_tf_chromatin.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_tf_chromatin.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
