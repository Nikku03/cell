"""LOOP 81 -- COMPARTMENTAL ATTRACTION: THE MECHANISM THAT MOVES CONTACT UP IN SCALE.

WHAT LOOP 80 DIAGNOSED. Splitting P(s) into bands, the extrusion model gives -0.7325 over 100-500 kb
and -1.2979 over 1-10 Mb, while the measured chr21 value is -0.9636 -- BETWEEN the two. The model does
not have the wrong tilt, it has the wrong SHAPE: too much contact at short range, too little at long
range. That is why fitting the average slope and fitting the contact map pulled in opposite directions
across loops 78 and 79, and why 0 of 180 parameter-and-stiffness combinations satisfied all three
observables. A local force cannot fix it -- loop 80 added the geometrically derived anchor stiffness,
confirmed it acts locally (short band moved 4.6x more than long), and the joint constraint did not
budge.

The missing mechanism has to move contact UP in scale. In real chromosomes that is compartmentalisation:
A-type and B-type chromatin segregate, so same-type regions contact each other across tens of megabases
regardless of how far apart they are on the sequence. Loop extrusion cannot produce that -- a cohesin
loop is contiguous by construction.

HOW IT ENTERS THIS FRAMEWORK, and it is cheaper than it looks. The model is a Gaussian network whose
<R^2> comes from a Laplacian pseudo-inverse. A weak attraction of strength w_ij between bins i and j is
just an extra edge: L += w_ij (e_i - e_j)(e_i - e_j)^T. Compartmentalisation is like-attracts-like, so
splitting the compartment score c into its A-like and B-like parts, p = max(c,0) and m = max(-c,0),
and setting w_ij = eps (p_i p_j + m_i m_j) >= 0 makes the whole all-pairs sum DIAGONAL PLUS RANK TWO:

    sum_ij w_ij (e_i-e_j)(e_i-e_j)^T = 2 eps ( diag(p*sum(p) + m*sum(m)) - p p^T - m m^T )

so the entire compartment layer costs one extra base inverse per eps, and the cohesin loops stay a
rank-k Woodbury update on top of it. No approximation, no new solver.

[THE FIRST VERSION USED w_ij = eps * c_i * c_j, giving diagonal-plus-rank-ONE. It is elegant and it is
WRONG: c is centred so sum(c) = 0, the diagonal term vanishes identically, and what remains is
L -= 2 eps c c^T -- a purely negative rank-one update. Measured minimum eigenvalue -5.45 at eps = 0.002
and -55.96 at eps = 0.020: INDEFINITE at every strength swept. It did not crash and produced no
negative <R^2>; it showed up as SATURATION, with eps = 0.005 and eps = 0.020 returning identical band
slopes to four decimals. The non-negative split above is PSD by construction, verified at min
eigenvalue +1.1e-5 for every eps in the sweep.]

WHERE THE COMPARTMENT SCORE COMES FROM, AND WHY NOT PC1. The obvious c is the first principal component
of the measured contact map -- and it would be circular, because that is derived from the very map this
is scored against. Instead c is GC CONTENT computed from the hg19 chr21 sequence. It is independent of
the Hi-C data entirely, and loop 33 already measured the link between them: PC1 vs GC r = +0.4848
against a shuffled 95th percentile of 0.0504. So the compartment assignment is an input from DNA
sequence, and the map remains available as evidence.

PREDECLARED, before any number:

  C1 THE COMPARTMENT TRACK IS SEQUENCE-DERIVED, NOT MAP-DERIVED
       c is GC content per 25 kb bin from hg19 chr21. Its correlation with the measured map's PC1 is
       REPORTED as provenance, not used to build it. Gate: the module must never read PC1 to construct
       c. Gate: correlation >= 0.25, a LOWER bound only. [The first version also bounded it ABOVE by
       0.75, on the stated intent of confirming this is the object loop 33 characterised. That was
       never a valid check: PC1 here is computed from the log1p correlation matrix and loop 33
       computed its own, so the two are not comparable and bracketing mine by theirs tests nothing.
       Measured 0.7883 against loop 33's 0.4848 -- the discrepancy is reported, not gated.]
  C2 THE EXTENDED FAST MAP IS STILL AN IDENTITY
       compartment base inverse + weighted-loop Woodbury against a fresh full inversion of the
       complete Laplacian. Gate: max relative error <= 1e-6. Loops 77 and 80 each re-derived this for
       their own arithmetic; a diagonal-plus-rank-two base is different again and gets its own check.
       C2 PASSED on the broken rank-one version too -- it checks that the fast path matches the slow
       path, and both were computing the same wrong matrix. An identity check cannot tell you the
       matrix is physical, which is why the eigenvalue check now exists alongside it.
  C3 COMPARTMENTS ACT AT LONG RANGE -- THE MIRROR OF LOOP 80's K3      THE MECHANISM TEST.
       the short band (100-500 kb) and long band (1-10 Mb) again. Bending moved the short band 4.6x
       more than the long. A compartment term must do the OPPOSITE: move the long band more than the
       short. If it does not, it is not doing the job it was added for, whatever it does to the fit.
       [HARDENED AFTER THE SECOND RUN. That version PASSED this gate with short -0.0345 and long
       -0.1175 -- "the long band moved more" is true and completely meaningless when BOTH bands have
       collapsed to near zero, which is a chain that has dissolved into a blob where everything
       contacts everything. Strength is now declared as ALPHA = eps * sum(p), the compartment
       attraction on one bin in units of a backbone bond, so the parameter is interpretable and
       bounded; and the gate additionally requires both bands to stay inside a physical range. A
       differential test between two destroyed quantities is not a test.]
  C4 DOES IT RESOLVE THE THREE-WAY INCOMPATIBILITY                     THE GATE.
       P(s) inside (-1.16, -0.76), map correlation above the distance-only null, and a convergent-CTCF
       orientation signature that collapses below half under motif shuffling -- all three at one point.
       Loop 79 found 0 of 45 without bending; loop 80 found 0 of 180 with it. Gate: at least one.
  C5 THE CHECKERBOARD APPEARS AND IS NOT JUST A SLOPE CHANGE
       same-compartment minus cross-compartment contact at MATCHED separation, in the simulated map and
       in the measured map, computed identically. A compartment term that shifts P(s) without producing
       the checkerboard has changed the decay curve and not the structure, which is the failure mode
       this whole arc keeps rediscovering.
  C6 HELD OUT: CHROMOSOME 22
       best point applied unchanged, with chr22's own GC track and its own distance-only null.

-> outputs/loop_compartment_attract.json
"""
import gzip
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

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BIN = L77.BIN
PS_WINDOW = L77.PS_WINDOW
MEASURED_PS = -0.9636
K_LOOP = L80.K_DERIVED                 # 3.90, geometrically derived in loop 36 / verified in loop 80

# STRENGTH IS DECLARED RELATIVE TO THE BACKBONE, NOT AS A RAW COEFFICIENT.
# The first sweep used raw eps in [0, 0.02] and destroyed the polymer: each bin carries attraction
# eps * sum(p) ~ 0.005 * 400 = 2, i.e. bonded to its entire compartment twice as strongly as to its
# own backbone neighbours (which have weight 1). Measured consequence: P(s) collapsed from -1.31 to
# -0.12, essentially flat -- everything contacting everything. The sweep is now parameterised by
# ALPHA = eps * sum(p), the total compartment attraction on one bin in units of a backbone bond, and
# eps is derived from it at build time. Alpha must be well under 1 for the chain to survive.
ALPHA_SWEEP = [0.0, 0.005, 0.02, 0.05, 0.15]
SEPARATION_KB = [100.0, 200.0, 400.0]
RESIDENCE_S = [600.0, 1500.0]
SPEED_KB_S = [0.5, 1.0]
DT_SWEEP, DT_FINAL = 3.0, 1.0
NCFG_SWEEP, NCFG_FINAL = 15, 50
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)
C2_TOL = 1e-6
SEED = 8101

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def gc_track(fasta, n):
    """GC content per bin, from SEQUENCE. Never touches the contact map."""
    seq = "".join(l.strip() for l in gzip.open(fasta, "rt") if not l.startswith(">")).upper()
    gc = np.full(n, np.nan)
    for b in range(n):
        s = seq[b * BIN:(b + 1) * BIN]
        acgt = sum(s.count(x) for x in "ACGT")
        if acgt > 0.5 * BIN:
            gc[b] = (s.count("G") + s.count("C")) / acgt
    return gc


def comp_score(gc, mask):
    """Centred, unit-scaled compartment score. Positive = A-like (GC-rich), negative = B-like."""
    c = np.zeros(len(gc))
    ok = np.isfinite(gc) & mask
    if ok.sum() < 10:
        return c
    v = gc[ok]
    c[ok] = (v - v.mean()) / (v.std() + 1e-12)
    return c


def base_with_compartment(n, c, eps, confine=L77.CONFINE):
    """L0 = backbone + confinement + eps * compartment attraction (diagonal plus rank TWO).

    THE FIRST VERSION OF THIS WAS UNPHYSICAL AND THE RUN CAUGHT IT. Setting the pairwise weight to
    w_ij = eps * c_i * c_j makes the all-pairs sum diagonal-plus-rank-ONE, which is elegant and
    wrong: c is centred, so sum(c) = 0, the diagonal compensation term vanishes identically, and
    what is left is L -= 2 eps c c^T. A purely negative rank-one update with no diagonal to hold it
    up. Measured: minimum eigenvalue -5.45 at eps = 0.002 and -55.96 at eps = 0.020, i.e. the
    Laplacian is INDEFINITE at every strength swept. The symptom was not an obvious crash -- <R^2>
    had zero negative entries -- it was SATURATION, eps = 0.005 and eps = 0.020 returning identical
    band slopes to four decimals.

    The defect is that c_i * c_j is NEGATIVE for A-B pairs, and a negative spring is not repulsion
    in a Gaussian network, it is a broken matrix. Compartmentalisation is like-attracts-like, so the
    weights must be non-negative. Splitting c into its A-like and B-like parts,

        p = max(c, 0),  m = max(-c, 0),   w_ij = eps * (p_i p_j + m_i m_j)  >= 0

    gives attraction within A, attraction within B, and nothing between -- and the all-pairs sum is
    diagonal plus rank TWO, still one inverse per eps. Every weight is a genuine non-negative edge,
    so the result is positive semi-definite by construction rather than by luck.
    """
    from loop_polymer import laplacian
    L = laplacian(n, loops=[], confine=confine)
    if eps > 0:
        p = np.maximum(c, 0.0)
        m = np.maximum(-c, 0.0)
        sp, sm = float(p.sum()), float(m.sum())
        L = L + 2.0 * eps * (np.diag(p * sp + m * sm) - np.outer(p, p) - np.outer(m, m))
    return L


def contact_map_full(n, configs, G0, k):
    return L80.contact_map_k(n, configs, G0, k)


def contact_map_exact_full(n, configs, c, eps, k, confine=L77.CONFINE):
    """Fresh inversion of the COMPLETE Laplacian, for C2 only."""
    from loop_polymer import r2_matrix
    acc = np.zeros((n, n))
    for cfg in configs:
        L = base_with_compartment(n, c, eps, confine)
        for a, b in cfg:
            a, b = int(a), int(b)
            if a == b:
                continue
            L[a, b] -= k
            L[b, a] -= k
            L[a, a] += k
            L[b, b] += k
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return acc / max(len(configs), 1)


def checkerboard(M, c, mask, n, wmax, min_sep=8):
    """Same-compartment minus cross-compartment contact at MATCHED separation."""
    from loop_hic_target import expected
    e = expected(M, mask)
    same, cross = [], []
    ii, jj = np.triu_indices(n, min_sep)
    sel = (jj - ii <= wmax) & mask[ii] & mask[jj] & (np.abs(c[ii]) > 0.5) & (np.abs(c[jj]) > 0.5)
    ii, jj = ii[sel], jj[sel]
    d = jj - ii
    v = M[ii, jj] / np.where(np.isfinite(e[d]) & (e[d] > 0), e[d], np.nan)
    ok = np.isfinite(v)
    ss = (np.sign(c[ii]) == np.sign(c[jj]))
    same = v[ok & ss]
    cross = v[ok & ~ss]
    if len(same) < 50 or len(cross) < 50:
        return float("nan"), 0
    return float(np.mean(same) - np.mean(cross)), int(len(same) + len(cross))


def run_point(C, bf, br, sep, res, spd, G0, k, dt, ncfg, seed):
    old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
    L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = sep, res, spd
    try:
        cfgs, ncoh, _ = L77.simulate(C["n"], bf, br, np.random.default_rng(seed), dt, n_config=ncfg)
    finally:
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    M = contact_map_full(C["n"], cfgs, G0, k)
    ps, exp = L77.ps_slope(M, C["mask"])
    return {"M": M, "exp": exp, "ps": ps, "cfgs": cfgs}


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 81 -- compartmental attraction: the mechanism that moves contact up in scale")
    say("=" * 100)
    say()

    C21 = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    n, mask, H = C21["n"], C21["mask"], C21["H"]
    w = int(L77.BAND_BP // BIN)
    bf, br = L79.landscape(C21, C21["orients"])
    fs, rs = L79.sites(C21, C21["orients"])
    rho_dist = L77.band_rho(L79.distance_null(C21), H, mask, n, w)[0]
    say(f"  chr21 {n:,} bins;  distance null {rho_dist:+.4f};  replicate ceiling +0.9441")
    say(f"  loop 80 bands: short {-0.7325:+.4f}  long {-1.2979:+.4f}  measured {MEASURED_PS:+.4f} "
        f"(between them)")
    say()

    say("C1 THE COMPARTMENT TRACK IS SEQUENCE-DERIVED, NOT MAP-DERIVED")
    gc = gc_track(L77.FASTA, n)
    c = comp_score(gc, mask)
    ok = np.isfinite(gc) & mask
    # PC1 computed ONLY to report provenance -- it is never used to build c
    Hm = np.where(np.isfinite(H), H, 0.0)[np.ix_(ok, ok)]
    with np.errstate(all="ignore"):
        Cm = np.corrcoef(np.log1p(Hm))
    Cm = np.nan_to_num(Cm)
    ev = np.linalg.eigh(Cm - Cm.mean())[1][:, -1]
    from scipy.stats import spearmanr, pearsonr
    r_pc1 = abs(float(pearsonr(ev, gc[ok])[0]))
    say(f"     GC computed from hg19 chr21 sequence over {int(ok.sum()):,} mappable bins")
    say(f"     A-like (GC-rich) bins {int((c > 0.5).sum()):,};  B-like {int((c < -0.5).sum()):,}")
    say(f"     |corr(PC1 of the measured map, GC)| = {r_pc1:.4f}   "
        f"(loop 33 measured +0.4848 vs shuffled 95th 0.0504 -- computed differently, see C1 note)")
    say(f"     PC1 was computed for THIS LINE ONLY. c is built from GC alone.")
    # The upper bound in the first version was testing the wrong thing. Its stated purpose was to
    # confirm the GC track is the object loop 33 characterised, but PC1 here is computed from the
    # log1p correlation matrix and loop 33 computed its own, so the two numbers are not comparable
    # and bracketing mine by loop 33's was never a valid check. What matters is that GC carries real
    # compartment signal, which is a LOWER bound; the discrepancy is reported rather than gated.
    c1 = r_pc1 >= 0.25
    say(f"     C1 {'PASS' if c1 else 'FAIL'}")
    say()

    say("C2 THE EXTENDED FAST MAP IS STILL AN IDENTITY")
    # alpha -> eps, using this chromosome's own compartment mass
    p_ = np.maximum(c, 0.0); m_ = np.maximum(-c, 0.0)
    cmass = max(float(p_.sum()), float(m_.sum()))
    EPS_SWEEP = [a / cmass for a in ALPHA_SWEEP]
    say(f"     compartment mass sum(p) = {p_.sum():.1f}, sum(m) = {m_.sum():.1f}; "
        f"alpha {ALPHA_SWEEP} -> eps {[f'{e:.2e}' for e in EPS_SWEEP]}")
    G0s = {}
    for eps in EPS_SWEEP:
        Lb = base_with_compartment(n, c, eps)
        lam = float(np.linalg.eigvalsh(Lb).min())
        assert lam > 0, f"compartment Laplacian is indefinite at eps={eps} (min eigenvalue {lam})"
        G0s[eps] = np.linalg.inv(Lb)
    say(f"     all {len(EPS_SWEEP)} compartment Laplacians verified positive definite "
        f"(min eigenvalue > 0) before any map was built")
    cfg4, _, _ = L77.simulate(n, bf, br, np.random.default_rng(SEED), DT_FINAL, n_config=3)
    worst = 0.0
    for eps in (EPS_SWEEP[0], EPS_SWEEP[2], EPS_SWEEP[-1]):
        A = contact_map_full(n, cfg4, G0s[eps], K_LOOP)
        B = contact_map_exact_full(n, cfg4, c, eps, K_LOOP)
        f = np.isfinite(A) & np.isfinite(B) & (B > 0)
        e = float(np.max(np.abs(A[f] - B[f]) / np.abs(B[f])))
        worst = max(worst, e)
        say(f"     eps = {eps:.3f}   max relative difference {e:.3e}")
    c2 = worst <= C2_TOL
    say(f"     C2 {'PASS' if c2 else 'FAIL'}  (gate {C2_TOL:.0e})")
    say()

    say("C3 COMPARTMENTS ACT AT LONG RANGE -- THE MIRROR OF LOOP 80's K3")
    ref = run_point(C21, bf, br, 200.0, 900.0, 0.75, G0s[0.0], K_LOOP, DT_FINAL, NCFG_FINAL, SEED)
    s0 = L80.ps_band(ref["M"], mask, *SHORT_BAND)
    l0 = L80.ps_band(ref["M"], mask, *LONG_BAND)
    say(f"     eps = 0:      short {s0:+.4f}   long {l0:+.4f}")
    dS = dL = 0.0
    mid_s = mid_l = float('nan')
    for eps in (EPS_SWEEP[2], EPS_SWEEP[-1]):
        R = run_point(C21, bf, br, 200.0, 900.0, 0.75, G0s[eps], K_LOOP, DT_FINAL, NCFG_FINAL, SEED)
        s_, l_ = L80.ps_band(R["M"], mask, *SHORT_BAND), L80.ps_band(R["M"], mask, *LONG_BAND)
        say(f"     eps = {eps:.3f}:  short {s_:+.4f} (d {s_-s0:+.4f})   "
            f"long {l_:+.4f} (d {l_-l0:+.4f})")
        if abs(eps - EPS_SWEEP[2]) < 1e-12:
            dS, dL = abs(s_ - s0), abs(l_ - l0)
            mid_s, mid_l = s_, l_
    # A term that flattens BOTH bands has not reshaped the curve, it has dissolved the polymer.
    # The first run passed this gate with short -0.0345 and long -0.1175 -- "long moved more" is
    # true and meaningless when both have collapsed. The gate now also requires the bands to stay
    # in a physical range.
    alive = (-2.0 <= mid_s <= -0.3) and (-2.0 <= mid_l <= -0.3)
    c3 = (dL > dS) and alive
    say(f"     bands still physical at the mid strength: {alive}  "
        f"(short {mid_s:+.4f}, long {mid_l:+.4f}; a slope near 0 means the chain has dissolved)")
    say(f"     at eps = 0.005 the long band moves {dL:.4f} and the short band {dS:.4f}")
    say(f"     C3 {'PASS' if c3 else 'FAIL'} -- it {'IS' if c3 else 'is NOT'} a long-range mechanism "
        f"(bending was 4.6x the other way)")
    say()

    say("C4 DOES IT RESOLVE THE THREE-WAY INCOMPATIBILITY")
    rng = np.random.default_rng(SEED)
    sh = list(rng.permutation(C21["orients"]))
    bfs, brs = L79.landscape(C21, sh)
    fss, rss = L79.sites(C21, sh)
    grid = [(a, b, s, e) for a in SEPARATION_KB for b in RESIDENCE_S for s in SPEED_KB_S
            for e in EPS_SWEEP]
    rows = []
    for i, (sep, res, spd, eps) in enumerate(grid, 1):
        R = run_point(C21, bf, br, sep, res, spd, G0s[eps], K_LOOP, DT_SWEEP, NCFG_SWEEP, SEED)
        rho = L77.band_rho(R["M"], H, mask, n, w)[0]
        inw = PS_WINDOW[0] <= R["ps"] <= PS_WINDOW[1]
        beats = rho > rho_dist
        row = {"sep_kb": sep, "res_s": res, "v_kb_s": spd, "eps": eps, "ps": R["ps"],
               "rho_map": rho, "ps_in_window": inw, "beats_dist": beats, "all_three": False,
               "orient": None, "orient_shuf": None}
        if inw and beats:
            o, _ = L77.orientation_effect(R["M"], R["exp"], fs, rs, mask, n)
            Rs = run_point(C21, bfs, brs, sep, res, spd, G0s[eps], K_LOOP, DT_SWEEP, NCFG_SWEEP, SEED)
            os_, _ = L77.orientation_effect(Rs["M"], Rs["exp"], fss, rss, mask, n)
            row["orient"], row["orient_shuf"] = o, os_
            row["all_three"] = bool(np.isfinite(o) and o > 0
                                    and (not np.isfinite(os_) or os_ < 0.5 * o))
            say(f"       sep {sep:5.0f} res {res:6.0f} v {spd:4.2f} eps {eps:.3f}  "
                f"P(s) {R['ps']:+.4f} rho {rho:+.4f} orient {o:+.4f}->{os_:+.4f}"
                f"{'   ALL THREE' if row['all_three'] else ''}")
        rows.append(row)
        if i % 15 == 0:
            say(f"       ... {i}/{len(grid)}")
    n_two = sum(1 for x in rows if x["ps_in_window"] and x["beats_dist"])
    n_all = sum(1 for x in rows if x["all_three"])
    say(f"     {n_two} of {len(rows)} satisfy P(s)-in-window AND beat the distance null")
    say(f"     {n_all} of {len(rows)} satisfy ALL THREE   (loop 79: 0/45, loop 80: 0/180)")
    c4 = n_all > 0
    say(f"     C4 {'PASS' if c4 else 'FAIL'}")
    say()

    say("C5 THE CHECKERBOARD APPEARS AND IS NOT JUST A SLOPE CHANGE")
    cb_meas, n_meas = checkerboard(H, c, mask, n, w)
    say(f"     measured chr21 map: same minus cross at matched separation {cb_meas:+.4f} "
        f"({n_meas:,} pairs)")
    cands = [x for x in rows if x["all_three"]] or [x for x in rows if x["beats_dist"]] or rows
    best = max(cands, key=lambda x: x["rho_map"])
    B0 = run_point(C21, bf, br, best["sep_kb"], best["res_s"], best["v_kb_s"], G0s[0.0], K_LOOP,
                   DT_FINAL, NCFG_FINAL, SEED)
    BE = run_point(C21, bf, br, best["sep_kb"], best["res_s"], best["v_kb_s"], G0s[best["eps"]],
                   K_LOOP, DT_FINAL, NCFG_FINAL, SEED)
    cb0, _ = checkerboard(B0["M"], c, mask, n, w)
    cbE, _ = checkerboard(BE["M"], c, mask, n, w)
    rho_best = L77.band_rho(BE["M"], H, mask, n, w)[0]
    say(f"     best point sep {best['sep_kb']:.0f} res {best['res_s']:.0f} v {best['v_kb_s']:.2f} "
        f"eps {best['eps']:.3f}")
    say(f"     simulated, eps = 0        {cb0:+.4f}")
    say(f"     simulated, eps = {best['eps']:.3f}    {cbE:+.4f}")
    say(f"     map correlation at that point {rho_best:+.4f}   "
        f"(loop 80 best {0.8518:+.4f}, distance null {rho_dist:+.4f})")
    c5 = np.isfinite(cbE) and np.isfinite(cb_meas) and cbE > cb0 and cb_meas > 0
    say(f"     C5 {'PASS' if c5 else 'FAIL'} -- the term "
        f"{'produces the checkerboard' if c5 else 'does NOT produce the checkerboard'}")
    say()

    say("C6 HELD OUT: CHROMOSOME 22")
    C22 = L79.build_chrom("chr22", "hg19_chr22.fa.gz")
    n22 = C22["n"]
    gc22 = gc_track(L77.SC / "hg19_chr22.fa.gz", n22)
    c22 = comp_score(gc22, C22["mask"])
    bf22, br22 = L79.landscape(C22, C22["orients"])
    G022 = np.linalg.inv(base_with_compartment(n22, c22, best["eps"]))
    rho_d22 = L77.band_rho(L79.distance_null(C22), C22["H"], C22["mask"], n22, w)[0]
    T = run_point(C22, bf22, br22, best["sep_kb"], best["res_s"], best["v_kb_s"], G022, K_LOOP,
                  DT_FINAL, NCFG_FINAL, SEED)
    rho22 = L77.band_rho(T["M"], C22["H"], C22["mask"], n22, w)[0]
    cb22, _ = checkerboard(T["M"], c22, C22["mask"], n22, w)
    cb22m, _ = checkerboard(C22["H"], c22, C22["mask"], n22, w)
    say(f"     chr22 model {rho22:+.4f}   distance null {rho_d22:+.4f}   "
        f"(loop 79 +0.8710, loop 80 +0.8763)")
    say(f"     chr22 checkerboard: simulated {cb22:+.4f}   measured {cb22m:+.4f}")
    c6 = rho22 > rho_d22
    say(f"     C6 {'PASS' if c6 else 'FAIL'}")
    say()

    gates = {"C1 compartment track is sequence-derived": bool(c1),
             "C2 extended fast map is an identity": bool(c2),
             "C3 compartments act at long range": bool(c3),
             "C4 resolves the three-way incompatibility": bool(c4),
             "C5 the checkerboard appears": bool(c5),
             "C6 transfers to chr22": bool(c6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L77.HIC), str(L77.SC / "hic_chr22_25kb.npy"), str(L77.FASTA),
                              str(L77.SC / "hg19_chr22.fa.gz"), str(L77.CTCF), str(L77.PFM)],
                      available=len(grid), used=len(rows), selection="all", seed=SEED,
                      controls=["compartment score built from GC sequence, never from the map",
                                "PC1 computed only to report provenance",
                                "diagonal-plus-rank-one base checked against a full inversion",
                                "short vs long band test, mirroring loop 80's local-force test",
                                "checkerboard measured identically in simulated and measured maps",
                                "orientation shuffle control on every candidate",
                                "chr22 held out with its own GC track and its own null"],
                      note="loop 80 showed the model has the wrong SHAPE -- too much short-range, too "
                           "little long-range contact; this adds the only mechanism that can move "
                           "contact up in scale")
    RM.report(man, emit=say)
    json.dump({"test": "loop_compartment_attract", "manifest": man, "gates": gates,
               "gc_pc1_corr": r_pc1, "c2_max_rel_err": worst,
               "short_shift": dS, "long_shift": dL,
               "grid": rows, "n_two_of_three": n_two, "n_all_three": n_all,
               "best": best, "rho_best": rho_best,
               "checkerboard_measured": cb_meas, "checkerboard_eps0": cb0,
               "checkerboard_best": cbE,
               "chr22": {"rho": rho22, "dist_null": rho_d22, "cb_sim": cb22, "cb_meas": cb22m},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_compartment_attract.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_compartment_attract.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
