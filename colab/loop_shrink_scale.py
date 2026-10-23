"""Loop 233. Shrinkage at scale: all 1.9M cells, per-gene, and does it actually reduce the error?

WHAT LOOP 232 ESTABLISHED ON TEN KNOCKOUTS, and why it needs redoing at scale. Fitting subsample
error against a held-out half over n = 1 to 400 cells gave

    err(n) = 0.9707 * n^(-0.4617)        R^2 0.9980

with no irreducible floor -- the floor model predicted unseen cell counts WORSE (held-out 0.00919
against the power law's 0.00685) and the shuffled floor sat BELOW the real one, which is what a
fitted-noise parameter does. It also found that shrinking the estimate toward zero cut error from
0.08712 to 0.04477, very nearly in half, at an optimal multiplier of 0.0500.

Three things about that are unfinished, and loop 232's own N7 named the first:

  THE TEN WERE NOT A RANDOM SAMPLE. They were chosen for having the MOST cells, 1,006 to 1,963
  against a dataset median of 178. Deeply sequenced perturbations may be deep for reasons -- guide
  efficiency, growth rate -- that also make them less noisy. The law needs checking on all 9,522.

  THE SHRINKAGE WAS ONE NUMBER FOR EVERY GENE. But genes differ enormously in how much real signal
  they carry: loop 224 X2 measured per-gene reliability with quartiles 0.1448 and 0.3668, a
  2.5-fold spread. A single k must be a compromise, and the errors-in-variables prescription is
  per-quantity, not global.

  IT WAS NEVER TESTED ON ANYTHING DOWNSTREAM. Halving the error of predicting one half of the
  cells from the other is not obviously worth anything. Loop 224 X6 measured a real task --
  K562-to-RPE1 transfer -- at 0.22862 unweighted, and that is where a noise correction either pays
  or does not.

HOW k IS ESTIMATED HERE, and why this construction avoids the fault that killed loop 228's H2a.
That loop built a switching fraction whose two halves divided by the SAME pooled variance, and the
control -- shuffle the numerator, keep the denominator -- showed the statistic surviving at +0.9626
with all biology destroyed. The lesson is to never let the two sides of a comparison share an
estimated quantity. So k is not derived from a variance decomposition at all. For gene g,
minimising E[(k*A_g - B_g)^2] over perturbations gives

    k_g = cov(A_g, B_g) / var(A_g)

which is simply the regression slope of one half on the other. It is the errors-in-variables
shrinkage read straight off the data, with no shared denominator anywhere in it.

AND IT IS FITTED ON PERTURBATIONS THAT ARE NEVER SCORED. Per-gene k means 8,248 free parameters. If
they were fitted and tested on the same perturbations, P4 would pass by construction. The
perturbations are split in half: k is fitted on one set and every number below is measured on the
other.

PREDECLARED, BEFORE ANY NUMBER.

  P1 DOES THE STREAM REPRODUCE THE PUBLISHED VALUES?  -- everything requires it
     Half A and half B recombined, against the published bulk row, on perturbations with a single
     promoter row.
     Gate: PASS iff median Pearson exceeds 0.99 with median absolute difference below 0.01.

  P2 DOES THE NOISE LAW HOLD GENOME-WIDE?
     Bin all 9,522 perturbations by cell count and measure half-A-against-half-B error per bin,
     then fit err(n) = a * n^(-b) across bins.
     Gate: PASS iff the fit explains more than 95% of the variance in log error. The exponent is
     REPORTED and compared against loop 232's 0.4617 on ten deep knockouts, not gated -- if the
     ten were unrepresentative, the honest outcome is a different b, not a failure.

  P3 DOES GLOBAL SHRINKAGE REDUCE HELD-OUT ERROR?
     One k fitted on the training perturbations, applied to the held-out ones.
     Gate: PASS iff root-mean-square error falls by at least 10% against k = 1.

  P4 DOES PER-GENE SHRINKAGE BEAT ONE GLOBAL NUMBER?
     8,248 separate k_g, all fitted on training perturbations only.
     Gate: PASS iff held-out error falls by at least a further 5% against the global k. A FAIL
     means the 2.5-fold spread in per-gene reliability does not translate into a better estimator,
     and the simpler single number should be used.

  P5 DOES SHRINKAGE HELP A REAL DOWNSTREAM TASK?
     K562-to-RPE1 transfer on shared perturbations and shared genes, shrunk against unshrunk.
     Loop 224 X6 measured 0.22862 unweighted and 0.30025 reliability-weighted.
     Gate: PASS iff the shrunk agreement exceeds the unshrunk. Note that a uniform rescaling
     cannot change a Pearson correlation, so this tests PER-GENE shrinkage specifically -- the
     global arm is reported and is expected to be identical by construction, which is a check on
     the arithmetic rather than a result.

  P6 CONTROL: DOES k SURVIVE SHUFFLED PERTURBATION LABELS?
     Half B's perturbation labels permuted before k is fitted, so no real pairing remains.
     Gate: PASS iff the fitted global k falls below 0.02 under shuffling while the real one
     exceeds it. A k that stays high when the pairing is destroyed would be measuring something
     about the estimator rather than about reproducibility.

  P7 WHAT THIS CANNOT SHOW -- written before the run.
     Both halves of every perturbation share library preparation, day and gem group, so k measures
     reproducibility of SAMPLING within one experiment. A shrinkage tuned to that will be too
     gentle for any use that crosses experiments, where batch effects add variance this design
     cannot see.
     Shrinkage reduces squared error by trading variance for bias. It makes every estimate smaller,
     so any downstream use that cares about the SIZE of an effect rather than its rank is being
     handed a deliberately understated number, and no gate here penalises that.
     The matrix is the NORMALIZED one, already z-scored against non-targeting controls within gem
     group, so some variance was removed before we saw it and the optimal k measured here is
     larger than it would be on raw counts.
"""
import os, sys, json, time, io, warnings
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp")
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_shrink_scale.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
BULK = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
RPE1 = SCR / "perturbseq" / "rpe1_normalized_bulk_01.h5ad"
URL = "https://ndownloader.figshare.com/files/35774440"
FSIZE, NCELL, NGENE, XOFF = 65830941948, 1989578, 8248, 2048
BLOCK, WORKERS, SEED, MINCELL = 2048, 16, 233233, 20
REF_B, REF_K, REF_X6 = 0.4617, 0.0500, 0.22862
P3_BAR, P4_BAR, SHUF_BAR = 0.10, 0.05, 0.02

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def fetch_cells(lo, hi, tries=8):
    b0, b1 = XOFF + lo * NGENE * 4, XOFF + hi * NGENE * 4 - 1
    for t in range(tries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={b0}-{b1}"}, timeout=300,
                             allow_redirects=True)
            if r.status_code in (200, 206) and len(r.content) == (b1 - b0 + 1):
                return np.frombuffer(r.content, "<f4").reshape(hi - lo, NGENE)
        except Exception:
            pass
        time.sleep(min(2 * (t + 1), 15))
    raise IOError(f"cells {lo}-{hi} failed")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "shrinkage at scale, per gene"}
    say("=" * 104)
    say("LOOP 233 -- SHRINKAGE AT SCALE: ALL 1.9M CELLS, PER GENE, ON A REAL TASK")
    say("=" * 104)
    say("     Loop 232 fitted err(n) = 0.9707 * n^(-0.4617) on TEN knockouts chosen for having")
    say("     the most cells, and found shrinkage cutting error from 0.08712 to 0.04477 at a")
    say("     single k = 0.05. This checks the law on all 9,522, gives every gene its own k, and")
    say("     asks whether any of it helps a task that was not built for it.")
    say("     k_g = cov(A_g, B_g) / var(A_g) -- a regression slope, with no shared denominator")
    say("     anywhere in it. That is the fault that killed loop 228's H2a and it cannot recur here.")

    import h5py
    from rangefile import RangeFile
    rf = RangeFile(URL, size=FSIZE, block=8 << 20)
    hf = h5py.File(io.BufferedReader(rf, buffer_size=1 << 20), "r")
    cats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in hf["obs/__categories/gene"][:]])
    code = hf["obs/gene"][:].astype(np.int32)
    gid = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in hf["var/gene_id"][:]])
    NT = int(np.where(cats == "non-targeting")[0][0])
    npert = len(cats)
    keep = code != NT
    say(f"     {NCELL:,} cells, {npert:,} labels; {int(keep.sum()):,} perturbed cells to stream")

    SA = np.zeros((npert, NGENE), np.float32)
    SB = np.zeros((npert, NGENE), np.float32)
    nA = np.zeros(npert, np.int64); nB = np.zeros(npert, np.int64)

    def accum(A, cds, dest, cnt):
        o = np.argsort(cds, kind="stable")
        cs = cds[o]; Ao = A[o]
        b = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        dest[cs[b]] += np.add.reduceat(Ao, b, axis=0)
        cnt += np.bincount(cs, minlength=len(cnt))

    blocks = [(i, min(i + BLOCK, NCELL)) for i in range(0, NCELL, BLOCK)]
    tstart, done = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as ex:
        pend, it = deque(), iter(blocks)
        for _ in range(2 * WORKERS):
            b0 = next(it, None)
            if b0 is None: break
            pend.append((b0, ex.submit(fetch_cells, *b0)))
        while pend:
            (lo, hi), fut = pend.popleft()
            A = fut.result()
            nxt = next(it, None)
            if nxt is not None:
                pend.append((nxt, ex.submit(fetch_cells, *nxt)))
            m = keep[lo:hi]
            if not m.any(): continue
            AA = A[m]; cc = code[lo:hi][m]
            par = (np.arange(lo, hi)[m] % 2) == 0
            fin = np.isfinite(AA).all(1)
            if (par & fin).any(): accum(AA[par & fin], cc[par & fin], SA, nA)
            if (~par & fin).any(): accum(AA[~par & fin], cc[~par & fin], SB, nB)
            done += 1
            if done % 200 == 0:
                el = time.time() - tstart
                say(f"       {done:,}/{len(blocks):,} blocks   {el/60:.1f} min   "
                    f"{done*BLOCK*NGENE*4/1e6/el:.0f} MB/s")
    say(f"     stream complete in {(time.time()-tstart)/60:.1f} min; "
        f"{int(nA.sum()+nB.sum()):,} finite cells accumulated")

    ok = (nA >= MINCELL) & (nB >= MINCELL) & (np.arange(npert) != NT)
    idx = np.where(ok)[0]
    MA = SA[idx] / nA[idx, None].astype(np.float64)
    MB = SB[idx] / nB[idx, None].astype(np.float64)
    ncell = (nA[idx] + nB[idx]).astype(float)
    say(f"     {len(idx):,} perturbations with >={MINCELL} cells in BOTH halves")

    # ---------------------------------------------------------------- P1
    say("P1 DOES THE STREAM REPRODUCE THE PUBLISHED VALUES?")
    bh = h5py.File(BULK, "r")
    bidx = np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in bh["obs"][bh["obs"].attrs.get("_index", "_index")][:]])
    bsym = np.array([s.split("_")[1] if s.count("_") >= 3 else s for s in bidx])
    bg = np.array([x.decode() if isinstance(x, bytes) else str(x)
                   for x in bh["var"][bh["var"].attrs.get("_index", "_index")][:]])
    gm = {g: i for i, g in enumerate(bg)}
    gj = np.array([j for j in range(NGENE) if gid[j] in gm])
    gb = np.array([gm[gid[j]] for j in gj])
    uq, uc = np.unique(bsym, return_counts=True)
    single = set(uq[uc == 1])
    pos = {s: i for i, s in enumerate(bsym)}
    rs, ds = [], []
    for k, p in enumerate(idx[:: max(1, len(idx) // 200)][:200]):
        nm = cats[p]
        if nm not in single: continue
        comb = (SA[p] + SB[p]) / float(nA[p] + nB[p])
        mine = comb[gj].astype(np.float64)
        theirs = np.asarray(bh["X"][pos[nm], :])[gb].astype(np.float64)
        m = np.isfinite(mine) & np.isfinite(theirs)
        rs.append(pear(mine[m], theirs[m])); ds.append(float(np.median(np.abs(mine[m]-theirs[m]))))
    r1, d1 = float(np.median(rs)), float(np.median(ds))
    say(f"     {len(rs)} single-promoter perturbations compared: median Pearson {r1:+.5f}, "
        f"median |difference| {d1:.5f}")
    G.add("P1", bool(r1 > 0.99 and d1 < 0.01), stat=float(r1),
          if_true=lambda: f"P1 PASS -- the halves recombine to the published values at r {r1:+.4f}",
          if_false=lambda: f"P1 FAIL -- r {r1:+.4f}, median |diff| {d1:.4f}")
    res["integrity"] = {"r": r1, "med_absdiff": d1, "n": len(rs)}

    # ---------------------------------------------------------------- P2
    say("P2 DOES THE NOISE LAW HOLD GENOME-WIDE?")
    qs = np.quantile(ncell, np.linspace(0, 1, 9)[1:-1])
    bins = np.searchsorted(qs, ncell)
    ns_, er_ = [], []
    for b in range(8):
        m = bins == b
        if m.sum() < 20: continue
        ns_.append(float(np.median(ncell[m] / 2)))
        er_.append(float(np.sqrt(np.mean((MA[m] - MB[m]) ** 2))))
    ns_, er_ = np.array(ns_), np.array(er_)
    lx, ly = np.log(ns_), np.log(er_)
    Am = np.column_stack([np.ones_like(lx), -lx])
    cf, *_ = np.linalg.lstsq(Am, ly, rcond=None)
    pr = Am @ cf
    r2 = 1 - np.sum((ly - pr) ** 2) / np.sum((ly - ly.mean()) ** 2)
    a_g, b_g = float(np.exp(cf[0])), float(cf[1])
    say(f"     {len(ns_)} cell-count bins spanning {ns_.min():.0f} to {ns_.max():.0f} cells per half")
    for n_, e_ in zip(ns_, er_):
        say(f"       n={n_:>6.0f}   half-A vs half-B RMSE {e_:.5f}")
    say(f"     fitted   err(n) = {a_g:.4f} * n^(-{b_g:.4f})     R^2 {r2:.4f}")
    say(f"     loop 232 on ten deep knockouts measured b = {REF_B:.4f}")
    G.add("P2", bool(r2 > 0.95), stat=float(r2), requires=("P1",),
          if_true=lambda: f"P2 PASS -- the law holds genome-wide at R^2 {r2:.3f}; b = {b_g:.3f} "
                          f"against loop 232's {REF_B:.3f} on ten deep knockouts",
          if_false=lambda: f"P2 FAIL -- R^2 {r2:.3f}; a single exponent does not describe all "
                           f"9,522 perturbations even though it described ten")
    res["law"] = {"a": a_g, "b": b_g, "r2": float(r2), "loop232_b": REF_B,
                  "ns": [float(x) for x in ns_], "err": [float(x) for x in er_]}

    # ---------------------------------------------------------------- P3
    say("P3 DOES GLOBAL SHRINKAGE REDUCE HELD-OUT ERROR?")
    perm = rng.permutation(len(idx))
    tr, te = perm[: len(idx) // 2], perm[len(idx) // 2:]
    say(f"     k fitted on {len(tr):,} perturbations, scored on {len(te):,} never used for fitting")
    kg = float(np.sum(MA[tr] * MB[tr]) / np.sum(MA[tr] * MA[tr]))
    e1 = float(np.sqrt(np.mean((MA[te] - MB[te]) ** 2)))
    ek = float(np.sqrt(np.mean((kg * MA[te] - MB[te]) ** 2)))
    red = (e1 - ek) / e1
    say(f"     fitted global k = {kg:.4f}   (loop 232 found {REF_K:.4f} on ten knockouts)")
    say(f"     held-out RMSE: k=1 {e1:.5f}   k={kg:.3f} {ek:.5f}   reduction {100*red:.1f}%")
    G.add("P3", bool(red >= P3_BAR), stat=float(red), requires=("P1",),
          if_true=lambda: f"P3 PASS -- shrinking by {kg:.3f} cuts held-out error {100*red:.0f}%",
          if_false=lambda: f"P3 FAIL -- only {100*red:.1f}% reduction against a {100*P3_BAR:.0f}% bar")
    res["global"] = {"k": kg, "err_1": e1, "err_k": ek, "reduction": red}

    # ---------------------------------------------------------------- P4
    say("P4 DOES PER-GENE SHRINKAGE BEAT ONE GLOBAL NUMBER?")
    num = np.sum(MA[tr] * MB[tr], axis=0)
    den = np.sum(MA[tr] * MA[tr], axis=0)
    kgene = np.where(den > 0, num / np.maximum(den, 1e-12), kg)
    kgene = np.clip(kgene, 0.0, 2.0)
    eg = float(np.sqrt(np.mean((kgene[None, :] * MA[te] - MB[te]) ** 2)))
    red2 = (ek - eg) / ek
    say(f"     8,248 separate k, all fitted on the training perturbations only")
    say(f"     per-gene k: median {np.median(kgene):.4f}, quartiles "
        f"{np.percentile(kgene,25):.4f} / {np.percentile(kgene,75):.4f}, "
        f"range {kgene.min():.4f} to {kgene.max():.4f}")
    say(f"     held-out RMSE: global {ek:.5f}   per-gene {eg:.5f}   "
        f"further reduction {100*red2:.1f}%")
    G.add("P4", bool(red2 >= P4_BAR), stat=float(red2), requires=("P1",),
          if_true=lambda: f"P4 PASS -- per-gene shrinkage cuts a further {100*red2:.1f}%",
          if_false=lambda: f"P4 FAIL -- {100*red2:.1f}% further against a {100*P4_BAR:.0f}% bar; "
                           f"the spread in per-gene reliability does not buy a better estimator")
    res["pergene"] = {"err": eg, "reduction_vs_global": red2,
                      "k_median": float(np.median(kgene)),
                      "k_q25": float(np.percentile(kgene, 25)),
                      "k_q75": float(np.percentile(kgene, 75))}

    # ---------------------------------------------------------------- P5
    say("P5 DOES SHRINKAGE HELP A REAL DOWNSTREAM TASK?")
    uw = wg = wp = float("nan")
    try:
        rh = h5py.File(RPE1, "r")
        ridx = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in rh["obs"][rh["obs"].attrs.get("_index", "_index")][:]])
        rsym = np.array([s.split("_")[1] if s.count("_") >= 3 else s for s in ridx])
        rg = np.array([x.decode() if isinstance(x, bytes) else str(x)
                       for x in rh["var"][rh["var"].attrs.get("_index", "_index")][:]])
        rgm = {g: i for i, g in enumerate(rg)}
        rpos = {s: i for i, s in enumerate(rsym)}
        sg = [(j, rgm[gid[j]]) for j in range(NGENE) if gid[j] in rgm]
        gjr = np.array([x[0] for x in sg]); gbr = np.array([x[1] for x in sg])
        sp = [(k, rpos[cats[p]]) for k, p in enumerate(idx) if cats[p] in rpos]
        sp = sp[:: max(1, len(sp) // 400)][:400]
        K = np.array([(MA[k] + MB[k]) / 2 for k, _ in sp])[:, gjr]
        R = np.array([np.asarray(rh["X"][b, :])[gbr] for _, b in sp])
        uw = pear(K, R)
        wg = pear(kg * K, R)
        wp = pear(K * kgene[gjr][None, :], R)
        say(f"     {len(sp)} shared perturbations x {len(gjr):,} shared genes")
        say(f"       unweighted              {uw:+.5f}   (loop 224 X6 measured {REF_X6:+.5f})")
        say(f"       global k                {wg:+.5f}   identical by construction -- a uniform")
        say(f"                                          rescaling cannot move a Pearson")
        say(f"       per-gene k              {wp:+.5f}   delta {wp-uw:+.5f}")
        rh.close()
    except Exception as e:
        say(f"     RPE1 comparison did not run: {type(e).__name__}: {e}")
    G.add("P5", bool(np.isfinite(wp) and wp > uw), stat=float(wp),
          requires=("P1",), void_if=(not np.isfinite(wp)),
          void_reason="the RPE1 file did not yield a comparable matrix",
          if_true=lambda: f"P5 PASS -- per-gene shrinkage raises cross-line agreement from "
                          f"{uw:+.4f} to {wp:+.4f}",
          if_false=lambda: f"P5 FAIL -- per-gene {wp:+.4f} against unweighted {uw:+.4f}")
    res["crossline"] = {"unweighted": uw, "global": wg, "pergene": wp}

    # ---------------------------------------------------------------- P6
    say("P6 CONTROL: DOES k SURVIVE SHUFFLED PERTURBATION LABELS?")
    sh = rng.permutation(len(tr))
    ksh = float(np.sum(MA[tr] * MB[tr][sh]) / np.sum(MA[tr] * MA[tr]))
    say(f"     half B's perturbation labels permuted before fitting")
    say(f"     real k {kg:.4f}   shuffled k {ksh:.4f}")
    G.add("P6", bool(abs(ksh) < SHUF_BAR < kg), stat=float(abs(ksh)), requires=("P1",),
          if_true=lambda: f"P6 PASS -- destroying the pairing collapses k to {ksh:.4f} against "
                          f"the real {kg:.4f}",
          if_false=lambda: f"P6 FAIL -- shuffled k {ksh:.4f} against real {kg:.4f}; k is not "
                           f"measuring reproducibility")
    res["shuffle"] = {"k_real": kg, "k_shuffled": ksh}

    # ---------------------------------------------------------------- P7
    say("P7 WHAT THIS CANNOT SHOW")
    say("     Both halves share library preparation, day and gem group, so k measures")
    say("     reproducibility of SAMPLING within one experiment. A shrinkage tuned to that is too")
    say("     gentle for anything crossing experiments, where batch variance this design cannot")
    say("     see would add to the noise and call for a smaller k.")
    say("     Shrinkage trades variance for bias: every estimate comes out SMALLER. Any use that")
    say("     cares about the size of an effect rather than its rank is handed a deliberately")
    say("     understated number, and no gate here penalises that.")
    say("     The matrix is the NORMALIZED one, already z-scored against non-targeting controls")
    say("     within gem group, so variance was removed before we saw it and the k measured here")
    say("     is LARGER than it would be on raw counts.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed("outputs/loop233_kgene.npz", gene_id=gid, k=kgene.astype(np.float32))
    say(f"     written {OUT} and outputs/loop233_kgene.npz")


if __name__ == "__main__":
    main()
