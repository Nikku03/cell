"""Loop 232. The noise law: how does error fall with cell count, and where does it stop falling?

THE QUESTION, MADE PRECISE. Take one gene knockout. Read 100 individual cells. Compare against the
value you get from combining every cell of that knockout. How wrong are you, and how does that
wrongness shrink as you add cells? If the answer is the textbook one, error falls as 1/sqrt(n) for
ever and more cells always help. If it is not, there is a FLOOR that averaging cannot cross, and
the practically important number is where the floor is -- because past that point every additional
cell is wasted.

WHAT THIS PROJECT ALREADY KNOWS, and why the textbook answer is not the safe bet:
    loop 224 X2  a 183-cell pseudobulk value has reliability 0.2299. Within-perturbation variance
                 1.0617 against between-perturbation variance 0.00174 -- cell-to-cell noise is
                 610x the biological differences it is meant to resolve.
    loop 224 X3  the independence model predicts real split-half reproducibility at r +0.9893,
                 median error 0.0389, but predicts 0.1305 against an observed 0.0920 -- optimistic
                 in exactly the direction that non-independent cell noise would produce.
    loop 224 X4  84.5% of the 200 strongest perturbation-by-gene effects are BIMODAL at
                 single-cell level, against a matched Gaussian control with a 0.0% false-positive
                 rate. If the cells are two populations, the sample mean is not the estimator with
                 the lowest error, and a robust estimator can beat it.

THE TWO EQUATIONS BEING FITTED, both stated before any data is read:

    CLASSICAL      err(n) = a * n^(-b)                  b = 0.5 is independent sampling noise
    WITH A FLOOR   err(n) = sqrt( a^2 / n + c^2 )       c > 0 is variance no averaging removes

They are nested only in the limit c -> 0, so N3 does not compare them by fit quality on the points
they were fitted to -- that would favour the model with more parameters by construction. It fits
both on a subset of cell counts and scores them on cell counts they never saw.

A THIRD EQUATION, from the reliability rather than the curve. Errors-in-variables says the
minimum-MSE rescaling of a noisy measurement is to MULTIPLY IT BY ITS OWN RELIABILITY. Loop 224
measured that reliability at 0.2299 for a 183-cell mean. That predicts a specific number -- the
optimal shrinkage multiplier at n = 183 should be about 0.23 -- and N5 tests it by finding the
empirically optimal multiplier and comparing. This is the one place in the loop where theory makes
a point prediction and the data can refuse it.

THE GROUND TRUTH IS HELD OUT, WHICH MATTERS MORE THAN IT SOUNDS. Each perturbation's cells are
split in half. One half is the target. The other half is the only pool the subsamples are drawn
from, so no cell ever appears in both the estimate and the thing it is scored against. Comparing a
subsample against the mean of ALL cells -- including that subsample -- would make error fall faster
than it truly does, purely by overlap, and would produce a floor that is an artefact of the design.

TEN KNOCKOUTS, chosen by cell count so the curve can be traced far enough to see whether it flattens:
RPL3 1963, PINK1 1745, DUSP9 1351, RAP1GAP 1270, HBA2 1199, SIMC1 1170, STK38L 1169, ANAPC15 1091,
PPP2R1A 1061, ESPN 1006. Median across the whole dataset is 178, so these are the deep ones.

PREDECLARED, BEFORE ANY NUMBER.

  N1 DO THE STREAMED CELLS AGGREGATE TO THE PUBLISHED VALUE?  -- everything requires it
     The mean over all of a perturbation's cells against that perturbation's row in the published
     bulk file.
     Gate: PASS iff Pearson exceeds 0.99 with median absolute difference below 0.01, on all ten.
     Loop 224 X1 verified this genome-wide at r +1.00000; this re-verifies it on the ten.

  N2 DOES ERROR FALL AS A POWER OF CELL COUNT?
     Fit log err = log a - b log n over n = 1, 2, 5, 10, 20, 50, 100, 200, 400.
     Gate: PASS iff the power law explains more than 95% of the variance in log err, so that b is
     a meaningful number rather than a slope through scatter. The VALUE of b is reported, not
     gated -- b = 0.5 is the independent-sampling prediction and gating on it would be assuming
     the answer.

  N3 IS THERE AN IRREDUCIBLE FLOOR?
     Fit both equations on n in {1, 2, 5, 10, 20} and score both on n in {50, 100, 200, 400},
     which neither model saw.
     Gate: PASS iff the floor model's held-out error is at least 20% below the power law's. A FAIL
     means the floor is not needed to describe the data and c is a free parameter fitting noise.

  N4 IS THE SAMPLE MEAN THE BEST ESTIMATOR AT n = 100?
     Mean against median, 10% trimmed mean and a Huber M-estimator, all scored against the same
     held-out target.
     Gate: PASS iff the best alternative beats the mean by at least 5% in error. Given X4's 84.5%
     bimodality the mean is not the maximum-likelihood estimator, but whether that costs anything
     measurable is what this asks.

  N5 IS THE OPTIMAL SHRINKAGE THE MEASURED RELIABILITY?  -- the point prediction
     Sweep a multiplier k over the n = 183 estimate, find the k minimising held-out error.
     Gate: PASS iff the empirically optimal k is within 0.10 of 0.2299. This gate can fail in
     either direction and neither is assumed.

  N6 CONTROL: DOES THE LAW SURVIVE SHUFFLED CELL ASSIGNMENT?
     Cells reassigned to perturbations at random, everything else identical.
     Gate: PASS iff the fitted floor c rises by at least 50%. Under shuffling every perturbation
     has the same true value, so a floor that does not move would mean the real floor is a
     property of the estimator rather than of the biology.

  N7 WHAT THIS CANNOT SHOW -- written before the run.
     Ten knockouts chosen for having the MOST cells are not a random sample of knockouts. Deeply
     sequenced perturbations may be deep for reasons -- guide efficiency, growth rate -- that
     correlate with how noisy they are.
     The matrix is the NORMALIZED one, already z-scored against non-targeting controls within gem
     group, so some cell-to-cell variance was removed by the depositors before we saw it. That
     biases the floor DOWNWARD: the real floor is at least this high and probably higher.
     Both halves of every split share library preparation, day and gem group, so a floor measured
     here is a floor on sampling, not on the experiment. Batch effects between preparations are
     invisible to this design and would add to it.
"""
import os, sys, json, time, io, warnings
from pathlib import Path
import numpy as np
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp")
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_noise_law.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
CACHE = SCR / "loop232_cells.npz"
BULK = SCR / "perturbseq" / "K562_gwps_normalized_bulk_01.h5ad"
URL = "https://ndownloader.figshare.com/files/35774440"
FSIZE, NCELL, NGENE, XOFF = 65830941948, 1989578, 8248, 2048
NPERT, NDRAW, SEED = 10, 400, 232232
NS = [1, 2, 5, 10, 20, 50, 100, 200, 400]
FIT_N, TEST_N = [1, 2, 5, 10, 20], [50, 100, 200, 400]
REL_183, GAP = 0.2299, 64

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


def huber(x, k=1.345, iters=25):
    m = np.median(x, axis=0)
    for _ in range(iters):
        s = 1.4826 * np.median(np.abs(x - m), axis=0) + 1e-9
        r = np.clip((x - m) / s, -k, k)
        m = m + s * r.mean(axis=0)
    return m


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "the noise law from individual cells"}
    say("=" * 104)
    say("LOOP 232 -- THE NOISE LAW: HOW FAR DOES AVERAGING CELLS ACTUALLY GET YOU?")
    say("=" * 104)
    say("     Two equations, both written before any data is read:")
    say("       CLASSICAL      err(n) = a * n^(-b)              b = 0.5 is independent sampling")
    say("       WITH A FLOOR   err(n) = sqrt(a^2/n + c^2)       c > 0 is what averaging cannot fix")
    say("     And one point prediction: errors-in-variables says the optimal rescaling of a noisy")
    say("     measurement is its own reliability, which loop 224 measured at 0.2299 for 183 cells.")

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
    cnt = np.bincount(code.astype(np.int64), minlength=len(cats))
    picks = [i for i in np.argsort(-cnt) if i != NT][:NPERT]
    say("     ten knockouts, chosen by cell count so the curve can be traced far enough:")
    say("       " + ", ".join(f"{cats[i]} {cnt[i]}" for i in picks))

    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        CELLS = {str(k): c[k] for k in c.files if k != "names"}
        say(f"     cell cache found; {len(CELLS)} perturbations, no streaming repeated")
    else:
        # ONE SEQUENTIAL PASS, not one request per cell. The first attempt coalesced each
        # perturbation's cells with a 64-cell gap tolerance, but 1,963 cells scattered through
        # 1,989,578 sit about 1,000 apart, so almost every cell became its own range request:
        # ~1,900 requests at ~1 s each for RPL3 alone, and 15 minutes in it had not finished one
        # perturbation. A contiguous scan of the whole matrix reads 66 GB at ~36 MB/s in about
        # 30 minutes and collects all ten at once.
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        want = {int(p): cats[p] for p in picks}
        wanted_mask = np.isin(code, list(want.keys()))
        say(f"     {int(wanted_mask.sum()):,} cells wanted across {len(want)} perturbations; "
            f"streaming the full matrix once rather than one request per cell")
        BLOCK, WORKERS = 2048, 16
        blocks = [(i, min(i + BLOCK, NCELL)) for i in range(0, NCELL, BLOCK)]
        acc = {p: [] for p in want}
        tstart, done = time.time(), 0
        with ThreadPoolExecutor(WORKERS) as ex:
            pend, it = deque(), iter(blocks)
            for _ in range(2 * WORKERS):
                b0 = next(it, None)
                if b0 is None:
                    break
                pend.append((b0, ex.submit(fetch_cells, *b0)))
            while pend:
                (lo, hi), fut = pend.popleft()
                A = fut.result()
                nxt = next(it, None)
                if nxt is not None:
                    pend.append((nxt, ex.submit(fetch_cells, *nxt)))
                m = wanted_mask[lo:hi]
                if m.any():
                    cc = code[lo:hi][m]
                    AA = A[m]
                    for p in np.unique(cc):
                        acc[int(p)].append(AA[cc == p])
                done += 1
                if done % 120 == 0:
                    el = time.time() - tstart
                    got = sum(sum(len(x) for x in v) for v in acc.values())
                    say(f"       {done:,}/{len(blocks):,} blocks   {el/60:.1f} min   "
                        f"{done*BLOCK*NGENE*4/1e6/el:.0f} MB/s   {got:,} cells kept")
        CELLS = {want[p]: np.vstack(v).astype(np.float32) for p, v in acc.items() if v}
        for k, v in CELLS.items():
            say(f"       {k:<10} {v.shape[0]:>5} cells")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **CELLS)
        say(f"     {(time.time()-tstart)/60:.1f} min; cached to {CACHE.name}")

    # ---------------------------------------------------------------- N1
    say("N1 DO THE STREAMED CELLS AGGREGATE TO THE PUBLISHED VALUE?")
    bh = h5py.File(BULK, "r")
    bkey_raw = bh["obs"][bh["obs"].attrs.get("_index", "_index")][:]
    bidx = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in bkey_raw])
    bsym = np.array([s.split("_")[1] if s.count("_") >= 3 else s for s in bidx])
    bg = np.array([x.decode() if isinstance(x, bytes) else str(x)
                   for x in bh["var"][bh["var"].attrs.get("_index", "_index")][:]])
    gmap = {g: i for i, g in enumerate(bg)}
    gj = np.array([j for j in range(NGENE) if gid[j] in gmap])
    gb = np.array([gmap[gid[j]] for j in gj])
    uq, uc = np.unique(bsym, return_counts=True)
    single = set(uq[uc == 1])
    rs, ds = [], []
    for name, X in CELLS.items():
        if name not in single:
            continue
        row = int(np.where(bsym == name)[0][0])
        mine = X[:, gj].mean(0).astype(np.float64)
        theirs = np.asarray(bh["X"][row, :])[gb].astype(np.float64)
        m = np.isfinite(mine) & np.isfinite(theirs)
        rs.append(pear(mine[m], theirs[m])); ds.append(float(np.median(np.abs(mine[m] - theirs[m]))))
    r1, d1 = float(np.median(rs)), float(np.median(ds))
    say(f"     {len(rs)} of {len(CELLS)} perturbations have a single promoter row and are comparable")
    say(f"     mean-of-all-cells vs published bulk: median Pearson {r1:+.5f}, "
        f"median |difference| {d1:.5f}")
    G.add("N1", bool(r1 > 0.99 and d1 < 0.01), stat=float(r1),
          if_true=lambda: f"N1 PASS -- the streamed cells reproduce the published values at "
                          f"r {r1:+.4f}",
          if_false=lambda: f"N1 FAIL -- r {r1:+.4f}, median |diff| {d1:.4f}")
    res["integrity"] = {"r": r1, "med_absdiff": d1, "n_compared": len(rs)}

    # ---------------------------------------------------------------- curves
    fin = None
    for X in CELLS.values():
        f = np.isfinite(X).all(0)
        fin = f if fin is None else (fin & f)
    nbad = int((~fin).sum())
    say(f"     gene screen: {int(fin.sum()):,} of {NGENE:,} genes are finite in EVERY cell of")
    say(f"     every perturbation; {nbad} dropped. Only 39 of 13,025 cells carry a non-finite")
    say(f"     value, but averaging over all genes let those few poison every RMSE -- the first")
    say(f"     run returned nan for the whole curve. Loop 208 A4 already measured ~17.8% of")
    say(f"     Perturb-seq ROWS as carrying non-finite values and this loop did not screen.")
    CELLS = {k: X[:, fin] for k, X in CELLS.items()}
    say("     building error curves: each perturbation's cells split in half, one half the TARGET,")
    say("     subsamples drawn only from the other half so no cell is in both")
    EST = {"mean": lambda x: x.mean(0), "median": lambda x: np.median(x, axis=0),
           "trim10": lambda x: np.sort(x, axis=0)[max(1, int(.1 * len(x))):
                                                   len(x) - max(1, int(.1 * len(x)))].mean(0)
           if len(x) >= 10 else x.mean(0),
           "huber": lambda x: huber(x)}
    curves = {k: {n: [] for n in NS} for k in EST}
    shrink_pool = []
    for name, X in CELLS.items():
        X = X.astype(np.float64)
        nc = len(X)
        pm = rng.permutation(nc)
        half = nc // 2
        target = X[pm[:half]].mean(0)
        pool = X[pm[half:]]
        for n in NS:
            if n > len(pool):
                continue
            for _ in range(NDRAW // max(1, n // 8 + 1)):
                s = pool[rng.choice(len(pool), n, replace=False)]
                for k, f in EST.items():
                    curves[k][n].append(float(np.sqrt(np.mean((f(s) - target) ** 2))))
        if len(pool) >= 183:
            for _ in range(60):
                s = pool[rng.choice(len(pool), 183, replace=False)].mean(0)
                shrink_pool.append((s, target))
    ERR = {k: np.array([np.mean(curves[k][n]) if curves[k][n] else np.nan for n in NS])
           for k in EST}
    ns = np.array(NS, float)
    ok = np.isfinite(ERR["mean"])
    say("     root-mean-square error against the held-out target, by cell count:")
    for i, n in enumerate(NS):
        if ok[i]:
            say(f"       n={n:<4} mean {ERR['mean'][i]:.5f}   median {ERR['median'][i]:.5f}   "
                f"trim10 {ERR['trim10'][i]:.5f}   huber {ERR['huber'][i]:.5f}")

    # ---------------------------------------------------------------- N2
    say("N2 DOES ERROR FALL AS A POWER OF CELL COUNT?")
    lx, ly = np.log(ns[ok]), np.log(ERR["mean"][ok])
    A = np.column_stack([np.ones_like(lx), -lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    a_pl, b_pl = float(np.exp(coef[0])), float(coef[1])
    say(f"     fitted   err(n) = {a_pl:.4f} * n^(-{b_pl:.4f})     R^2 {r2:.4f}")
    say(f"     independent sampling predicts b = 0.5; measured b = {b_pl:.4f}")
    say("     b is REPORTED, not gated -- gating on 0.5 would be assuming the answer")
    G.add("N2", bool(r2 > 0.95), stat=float(r2), requires=("N1",),
          if_true=lambda: f"N2 PASS -- the power law explains {r2:.1%} of the variance, so "
                          f"b = {b_pl:.3f} is a meaningful number",
          if_false=lambda: f"N2 FAIL -- R^2 {r2:.3f}; a single exponent does not describe this")
    res["power_law"] = {"a": a_pl, "b": b_pl, "r2": float(r2),
                        "ns": NS, "err_mean": [float(x) for x in ERR["mean"]]}

    # ---------------------------------------------------------------- N3
    say("N3 IS THERE AN IRREDUCIBLE FLOOR?")
    fi = [NS.index(n) for n in FIT_N if NS.index(n) < len(ok) and ok[NS.index(n)]]
    ti = [NS.index(n) for n in TEST_N if NS.index(n) < len(ok) and ok[NS.index(n)]]
    lxf, lyf = np.log(ns[fi]), np.log(ERR["mean"][fi])
    Af = np.column_stack([np.ones_like(lxf), -lxf])
    cf, *_ = np.linalg.lstsq(Af, lyf, rcond=None)
    pl_test = np.exp(cf[0]) * ns[ti] ** (-cf[1])
    best, ba, bc = np.inf, None, None
    for a in np.linspace(0.01, 20, 400):
        for c in np.linspace(0.0, 5.0, 400):
            p = np.sqrt(a ** 2 / ns[fi] + c ** 2)
            e = np.sum((p - ERR["mean"][fi]) ** 2)
            if e < best:
                best, ba, bc = e, a, c
    fl_test = np.sqrt(ba ** 2 / ns[ti] + bc ** 2)
    e_pl = float(np.mean(np.abs(pl_test - ERR["mean"][ti])))
    e_fl = float(np.mean(np.abs(fl_test - ERR["mean"][ti])))
    say(f"     both fitted on n in {FIT_N}, scored on n in {TEST_N} which neither model saw")
    say(f"       power law  err(n) = {np.exp(cf[0]):.4f} * n^(-{cf[1]:.4f})   "
        f"held-out mean |error| {e_pl:.5f}")
    say(f"       with floor err(n) = sqrt({ba:.4f}^2/n + {bc:.4f}^2)   "
        f"held-out mean |error| {e_fl:.5f}")
    say(f"     fitted floor c = {bc:.5f}; the two terms are equal at n = "
        f"{(ba/max(bc,1e-9))**2:.0f} cells")
    G.add("N3", bool(e_fl <= 0.8 * e_pl), stat=float(e_fl / max(e_pl, 1e-12)), requires=("N1",),
          if_true=lambda: f"N3 PASS -- the floor model predicts unseen cell counts "
                          f"{100*(1-e_fl/max(e_pl,1e-12)):.0f}% better; c = {bc:.4f} is real and "
                          f"averaging past n = {(ba/max(bc,1e-9))**2:.0f} buys little",
          if_false=lambda: f"N3 FAIL -- floor {e_fl:.5f} against power law {e_pl:.5f}; a floor is "
                           f"not needed and c is fitting noise")
    res["floor"] = {"a": float(ba), "c": float(bc), "heldout_powerlaw": e_pl,
                    "heldout_floor": e_fl, "n_equal": float((ba / max(bc, 1e-9)) ** 2)}

    # ---------------------------------------------------------------- N4
    say("N4 IS THE SAMPLE MEAN THE BEST ESTIMATOR AT n = 100?")
    i100 = NS.index(100)
    base = ERR["mean"][i100]
    alt = {k: ERR[k][i100] for k in EST if k != "mean"}
    bk = min(alt, key=alt.get)
    gain = (base - alt[bk]) / base
    for k in EST:
        say(f"       {k:<8} {ERR[k][i100]:.5f}" + ("   <- sample mean" if k == "mean" else ""))
    say(f"     best alternative {bk}: {100*gain:+.1f}% error against the mean")
    G.add("N4", bool(gain >= 0.05), stat=float(gain), requires=("N1",),
          if_true=lambda: f"N4 PASS -- {bk} beats the sample mean by {100*gain:.1f}%; with 84.5% "
                          f"of strong effects bimodal the mean is not the right estimator",
          if_false=lambda: f"N4 FAIL -- best alternative {bk} gains only {100*gain:.1f}%; the "
                           f"sample mean is not measurably improvable here")
    res["estimators"] = {k: float(ERR[k][i100]) for k in EST}
    res["estimators"]["best_gain"] = float(gain)

    # ---------------------------------------------------------------- N5
    say("N5 IS THE OPTIMAL SHRINKAGE THE MEASURED RELIABILITY?")
    if shrink_pool:
        ks = np.linspace(0.0, 1.5, 151)
        errs = []
        Sarr = np.array([s for s, _ in shrink_pool]); Tarr = np.array([t for _, t in shrink_pool])
        for k in ks:
            errs.append(float(np.sqrt(np.mean((k * Sarr - Tarr) ** 2))))
        errs = np.array(errs)
        kbest = float(ks[np.argmin(errs)])
        say(f"     {len(shrink_pool)} draws of 183 cells; optimal multiplier k = {kbest:.4f}")
        say(f"     errors-in-variables predicts k = reliability = {REL_183:.4f} (loop 224 X2)")
        say(f"     error at k=1 {errs[np.argmin(np.abs(ks-1))]:.5f}, at k={kbest:.2f} "
            f"{errs.min():.5f}")
        ok5 = bool(abs(kbest - REL_183) <= 0.10)
    else:
        kbest, ok5 = float("nan"), False
    G.add("N5", ok5, stat=float(kbest), requires=("N1",),
          if_true=lambda: f"N5 PASS -- the optimal multiplier {kbest:.3f} lands within 0.10 of "
                          f"the independently measured reliability {REL_183:.3f}; the "
                          f"errors-in-variables prediction holds",
          if_false=lambda: f"N5 FAIL -- optimal k {kbest:.3f} against a predicted "
                           f"{REL_183:.3f}; the point prediction is refused")
    res["shrinkage"] = {"k_optimal": kbest, "k_predicted": REL_183}

    # ---------------------------------------------------------------- N6
    say("N6 CONTROL: DOES THE LAW SURVIVE SHUFFLED CELL ASSIGNMENT?")
    allc = np.vstack([X for X in CELLS.values()]).astype(np.float64)
    sizes = [len(X) for X in CELLS.values()]
    sp = rng.permutation(len(allc))
    off, errs_s = 0, {n: [] for n in NS}
    for sz in sizes:
        Xs = allc[sp[off:off + sz]]; off += sz
        pm = rng.permutation(sz); half = sz // 2
        tgt = Xs[pm[:half]].mean(0); pool = Xs[pm[half:]]
        for n in NS:
            if n > len(pool): continue
            for _ in range(40):
                s = pool[rng.choice(len(pool), n, replace=False)].mean(0)
                errs_s[n].append(float(np.sqrt(np.mean((s - tgt) ** 2))))
    Es = np.array([np.mean(errs_s[n]) if errs_s[n] else np.nan for n in NS])
    oks = np.isfinite(Es)
    bs, bcs = None, None
    best = np.inf
    for a in np.linspace(0.01, 20, 300):
        for c in np.linspace(0.0, 5.0, 300):
            p = np.sqrt(a ** 2 / ns[oks] + c ** 2)
            e = np.sum((p - Es[oks]) ** 2)
            if e < best: best, bs, bcs = e, a, c
    say(f"     shuffled floor c = {bcs:.5f} against the real floor c = {bc:.5f}")
    G.add("N6", bool(bc > 1e-6 and bcs >= 1.5 * bc),
          stat=float(bcs / bc) if bc > 1e-6 else None, requires=("N1",),
          void_if=(bc <= 1e-6),
          void_reason=f"the real floor c = {bc:.6f} is indistinguishable from zero, so a ratio "
                      f"against it has no denominator -- the first run passed this gate on "
                      f"0.0000 >= 1.5 x 0.0000, which is Family One",
          if_true=lambda: f"N6 PASS -- shuffling raises the floor from {bc:.4f} to {bcs:.4f}, so "
                          f"the real floor is a property of the biology, not the estimator",
          if_false=lambda: f"N6 FAIL -- shuffled floor {bcs:.4f} against real {bc:.4f}; the floor "
                           f"does not move when the biology is destroyed and is therefore a "
                           f"property of the estimator")
    res["shuffle"] = {"c_shuffled": float(bcs), "c_real": float(bc)}

    # ---------------------------------------------------------------- N7
    say("N7 WHAT THIS CANNOT SHOW")
    say("     Ten knockouts chosen for having the MOST cells are not a random sample. Deeply")
    say("     sequenced perturbations may be deep for reasons -- guide efficiency, growth rate --")
    say("     that correlate with how noisy they are.")
    say("     The matrix is the NORMALIZED one, already z-scored against non-targeting controls")
    say("     within gem group, so variance was removed before we saw it. The floor measured here")
    say("     is a LOWER bound on the real floor.")
    say("     Both halves of every split share library preparation, day and gem group, so this is")
    say("     a floor on SAMPLING, not on the experiment. Batch effects are invisible here and")
    say("     would add to it.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
